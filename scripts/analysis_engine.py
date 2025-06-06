import os
import sys
import json
import optuna
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
from src.trading_system import StrategyConfig, BitgetAPI, DualNNFXSystem

# --- Optimization Parameter Ranges ---
optimization_ranges = {
    "tema_period": {"min": 10, "max": 50, "step": 5},
    "cmf_window": {"min": 10, "max": 30, "step": 2},
    "risk_per_trade": {"min": 0.01, "max": 0.03, "step": 0.005}
}

# --- API Config Path ---
API_CONFIG_PATH = Path("config/api_config.json")

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("analysis_engine")

# --- Load API Config ---
def load_api_config():
    if API_CONFIG_PATH.exists():
        with open(API_CONFIG_PATH, "r") as f:
            return json.load(f)
    logger.warning(f"API config not found at {API_CONFIG_PATH}. Using empty config.")
    return {}

# --- Get Top 8 Pairs by Volume ---
def get_top_pairs(api_config):
    api = BitgetAPI(**api_config)
    symbols = api.get_symbols()
    logger.info(f"Selected top {len(symbols)} pairs: {symbols[:8]}")
    return symbols[:8]

# --- Fetch klines for a symbol ---
def fetch_klines(api_config, symbol, limit=1000):
    api = BitgetAPI(**api_config)
    df = api.get_klines(symbol, "4H", limit=limit)
    return df

# --- Run Backtest for a Symbol with Given Params and Date Range ---
def run_backtest(api_config, params_dict, symbol, start_date=None, end_date=None):
    api = BitgetAPI(**api_config)
    strat_cfg = StrategyConfig(params_dict=params_dict)
    system = DualNNFXSystem(api, strat_cfg)
    result = system.backtest_pair(symbol, start_date=start_date, end_date=end_date)
    return result

# --- Plotly: Parameter Importance ---
def plot_param_importance(study, symbol, out_path):
    try:
        fig = optuna.visualization.plot_param_importances(study)
        pio.write_html(fig, file=out_path, auto_open=False)
        logger.info(f"Parameter importance plot saved: {out_path}")
    except Exception as e:
        logger.warning(f"Could not generate parameter importance plot for {symbol}: {e}")

# --- Plotly: Equity Curve ---
def plot_equity_curve(equity_curve, symbol, out_path):
    try:
        df = pd.DataFrame(equity_curve)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['equity'], mode='lines', name='Equity'))
        fig.update_layout(title=f"Equity Curve: {symbol}", xaxis_title="Time", yaxis_title="Equity ($)")
        pio.write_html(fig, file=out_path, auto_open=False)
        logger.info(f"Equity curve plot saved: {out_path}")
    except Exception as e:
        logger.warning(f"Could not generate equity curve plot for {symbol}: {e}")

# --- Plotly: Performance Comparison ---
def plot_performance_comparison(results, out_path):
    try:
        df = pd.DataFrame([
            {"symbol": s, "Sharpe": r["best_value"], "OOS_Sharpe": r["forward_test"].get("sharpe_ratio", 0),
             "Win Rate": r["forward_test"].get("win_rate", 0), "Profit Factor": r["forward_test"].get("profit_factor", 0)}
            for s, r in results.items()
        ])
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df['symbol'], y=df['Sharpe'], name='IS Sharpe'))
        fig.add_trace(go.Bar(x=df['symbol'], y=df['OOS_Sharpe'], name='OOS Sharpe'))
        fig.add_trace(go.Bar(x=df['symbol'], y=df['Win Rate'], name='OOS Win Rate'))
        fig.add_trace(go.Bar(x=df['symbol'], y=df['Profit Factor'], name='OOS Profit Factor'))
        fig.update_layout(barmode='group', title="Performance Comparison Across Pairs")
        pio.write_html(fig, file=out_path, auto_open=False)
        logger.info(f"Performance comparison plot saved: {out_path}")
    except Exception as e:
        logger.warning(f"Could not generate performance comparison plot: {e}")

# --- Main Analysis Engine ---
def main():
    logger.info("Starting comprehensive analysis engine with walk-forward and HTML output...")
    api_config = load_api_config()
    pairs = get_top_pairs(api_config)
    all_results = {}
    n_trials = 20  # Can be increased for more thorough optimization
    results_dir = Path("results"); results_dir.mkdir(exist_ok=True)

    for symbol in pairs:
        logger.info(f"\n=== Fetching klines for {symbol} ===")
        df_klines = fetch_klines(api_config, symbol, limit=1000)
        if df_klines.empty or len(df_klines) < 200:
            logger.warning(f"Not enough data for {symbol}, skipping.")
            continue
        # --- Split into IS/OOS ---
        n = len(df_klines)
        is_end_idx = int(n * 0.8)
        is_start_date = df_klines.index[0].strftime('%Y-%m-%d')
        is_end_date = df_klines.index[is_end_idx-1].strftime('%Y-%m-%d')
        oos_start_date = df_klines.index[is_end_idx].strftime('%Y-%m-%d')
        oos_end_date = df_klines.index[-1].strftime('%Y-%m-%d')
        logger.info(f"{symbol} IS: {is_start_date} to {is_end_date} | OOS: {oos_start_date} to {oos_end_date}")

        # --- Optimize on IS ---
        def objective(trial):
            tema_period = trial.suggest_int("tema_period", optimization_ranges["tema_period"]["min"], optimization_ranges["tema_period"]["max"], step=optimization_ranges["tema_period"]["step"])
            cmf_window = trial.suggest_int("cmf_window", optimization_ranges["cmf_window"]["min"], optimization_ranges["cmf_window"]["max"], step=optimization_ranges["cmf_window"]["step"])
            risk_per_trade = trial.suggest_float("risk_per_trade", optimization_ranges["risk_per_trade"]["min"], optimization_ranges["risk_per_trade"]["max"], step=optimization_ranges["risk_per_trade"]["step"])
            params_dict = {
                "indicators": {
                    "tema_period": tema_period,
                    "cmf_window": cmf_window
                },
                "risk_per_trade": risk_per_trade
            }
            result = run_backtest(api_config, params_dict, symbol, start_date=is_start_date, end_date=is_end_date)
            sharpe = result.get("sharpe_ratio", 0)
            logger.info(f"Trial params: {params_dict} | IS Sharpe: {sharpe}")
            return sharpe

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        best_params = study.best_trial.params
        best_value = study.best_value
        all_trials = [(t.params, t.value) for t in study.trials]
        all_results[symbol] = {
            "best_params": best_params,
            "best_value": best_value,
            "all_trials": all_trials
        }

        # --- Forward Test on OOS ---
        logger.info(f"Running forward test for {symbol} with best params...")
        forward_result = run_backtest(api_config, {
            "indicators": {
                "tema_period": best_params["tema_period"],
                "cmf_window": best_params["cmf_window"]
            },
            "risk_per_trade": best_params["risk_per_trade"]
        }, symbol, start_date=oos_start_date, end_date=oos_end_date)
        all_results[symbol]["forward_test"] = forward_result
        logger.info(f"Forward test result for {symbol}: {forward_result}")

        # --- Save equity curve plots ---
        if "equity_curve" in forward_result and forward_result["equity_curve"]:
            eq_path = results_dir / f"{symbol}_oos_equity_curve.html"
            plot_equity_curve(forward_result["equity_curve"], symbol, eq_path)

        # --- Save parameter importance plot ---
        param_imp_path = results_dir / f"{symbol}_param_importance.html"
        plot_param_importance(study, symbol, param_imp_path)

    # --- Save Results ---
    results_path = results_dir / "analysis_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"All results saved to {results_path}")

    # --- Performance Comparison Plot ---
    perf_path = results_dir / "performance_comparison.html"
    plot_performance_comparison(all_results, perf_path)

    # --- Print Summary ---
    logger.info("\n=== Summary of Best Configs and Forward Test Results ===")
    for symbol, res in all_results.items():
        logger.info(f"{symbol}: Best IS Sharpe={res['best_value']}, Best Params={res['best_params']}")
        logger.info(f"{symbol}: OOS Sharpe={res['forward_test'].get('sharpe_ratio', 'N/A')}, OOS Win Rate={res['forward_test'].get('win_rate', 'N/A')}, OOS PF={res['forward_test'].get('profit_factor', 'N/A')}")

if __name__ == "__main__":
    main() 