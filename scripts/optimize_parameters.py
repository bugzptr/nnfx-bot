import os
import sys
import json
import optuna
import logging
from pathlib import Path
from src.trading_system import StrategyConfig, _backtest_worker_process

# --- Optimization Parameter Ranges ---
optimization_ranges = {
    "tema_period": {"min": 10, "max": 50, "step": 5},
    "cmf_window": {"min": 10, "max": 30, "step": 2},
    "risk_per_trade": {"min": 0.01, "max": 0.03, "step": 0.005}
}

# --- Symbol to Optimize (can be changed) ---
SYMBOL_TO_OPTIMIZE = "BTCUSDT"

# --- API Config Path ---
API_CONFIG_PATH = Path("config/api_config.json")

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("optuna_optimization")

# --- Load API Config ---
def load_api_config():
    if API_CONFIG_PATH.exists():
        with open(API_CONFIG_PATH, "r") as f:
            return json.load(f)
    logger.warning(f"API config not found at {API_CONFIG_PATH}. Using empty config.")
    return {}

# --- Objective Function for Optuna ---
def objective(trial):
    # Suggest parameters
    tema_period = trial.suggest_int("tema_period", optimization_ranges["tema_period"]["min"], optimization_ranges["tema_period"]["max"], step=optimization_ranges["tema_period"]["step"])
    cmf_window = trial.suggest_int("cmf_window", optimization_ranges["cmf_window"]["min"], optimization_ranges["cmf_window"]["max"], step=optimization_ranges["cmf_window"]["step"])
    risk_per_trade = trial.suggest_float("risk_per_trade", optimization_ranges["risk_per_trade"]["min"], optimization_ranges["risk_per_trade"]["max"], step=optimization_ranges["risk_per_trade"]["step"])

    # Build parameter dict (based on your config structure)
    params_dict = {
        "indicators": {
            "tema_period": tema_period,
            "cmf_window": cmf_window
        },
        "risk_per_trade": risk_per_trade
    }

    # Load API config
    api_config = load_api_config()

    # Run backtest for the symbol using the worker process (single process for now)
    result = _backtest_worker_process(api_config, params_dict, SYMBOL_TO_OPTIMIZE)

    # Use Sharpe ratio as the objective (or another metric)
    sharpe = result.get("sharpe_ratio", 0)
    logger.info(f"Trial params: {params_dict} | Sharpe: {sharpe}")
    return sharpe

if __name__ == "__main__":
    logger.info("Starting Optuna parameter optimization...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)
    logger.info(f"Best trial: {study.best_trial.params}, Value: {study.best_value}") 