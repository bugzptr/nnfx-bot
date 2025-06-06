import pandas as pd
import numpy as np
import requests
import hmac
import hashlib
import base64
import time
import json
import os
from datetime import datetime, timedelta
import ta
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
import logging
from pathlib import Path
import sys
import io
from concurrent.futures import ProcessPoolExecutor, as_completed

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='ta')
warnings.filterwarnings('ignore', category=RuntimeWarning) 

# Configure logging
if not logging.getLogger().handlers: 
    logging.basicConfig(
        level=logging.DEBUG, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)] 
    )
logger = logging.getLogger(__name__) 

# --- Default Configuration (Fallback) ---
default_config_for_fallback = {
    "major_bases_for_filtering": ["BTC", "ETH", "SOL", "AVAX", "LINK", "ADA", "DOT", "MATIC", "BNB", "XRP", "DOGE", "TRX", "LTC"],
    "filter_by_major_bases_for_top_volume": True,
    "top_n_pairs_by_volume_to_scan": 8,
    "backtest_kline_limit": 1000,
    "backtest_min_data_after_get_klines": 100,
    "backtest_min_data_after_indicators": 50,
    "backtest_min_trades_for_ranking": 3, 
    "risk_per_trade": 0.015,
    "stop_loss_atr_multiplier": 2.0,
    "take_profit_atr_multiplier": 3.0,
    "indicators": { "tema_period": 21, "cci_period": 14, "elder_fi_period": 13,
                    "chandelier_period": 22, "chandelier_multiplier": 3.0,
                    "kijun_sen_period": 26, "williams_r_period": 14, "williams_r_threshold": -50,
                    "klinger_fast_ema": 34, "klinger_slow_ema": 55, "klinger_signal_ema": 13,
                    "psar_step": 0.02, "psar_max_step": 0.2,
                    "atr_period_risk": 14, "atr_period_chandelier": 22},
    "scoring": { "win_rate_weight": 0.25, "profit_factor_weight": 0.25, "return_pct_weight": 0.20,
                 "trade_frequency_weight": 0.10, "sharpe_ratio_weight": 0.20,
                 "drawdown_penalty_multiplier": 1.5, "drawdown_penalty_weight": 0.05,
                 "consecutive_loss_penalty_multiplier": 3.0, "consecutive_loss_penalty_weight": 0.03,
                 "var_95_penalty_multiplier": 5.0, "var_95_penalty_weight": 0.02,
                 "profit_factor_inf_score": 50.0, "sharpe_ratio_inf_score": 50.0}, 
    "signal_confidence": { "min_recent_data_for_confidence": 5, "default_confidence": 0.5, 
                           "signal_support_weight": 0.4, "trend_consistency_weight": 0.25,
                           "momentum_consistency_weight": 0.2, "volume_consistency_weight": 0.15},
    "cleanup": {"cache_days_old": 7, "max_results_to_keep": 20, "cache_klines_freshness_hours": 4}
}

# --- Configuration Loader ---
class StrategyConfig:
    def __init__(self, config_path_str="config/strategy_config.json"):
        self.config_path = Path(config_path_str)
        self.params = self._load_config()
        logger.info(f"Strategy configuration loaded from: {self.config_path if self.config_path.exists() else 'Fallback Defaults'}")

    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            logger.warning(f"Strategy config file not found: {self.config_path}. Using hardcoded fallback defaults.")
            try:
                self.config_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.config_path, "w") as f_cfg:
                    json.dump(default_config_for_fallback, f_cfg, indent=4)
                logger.info(f"Created dummy strategy config at {self.config_path}. Please review and customize.")
            except Exception as e_create_cfg: logger.error(f"Could not create dummy strategy config: {e_create_cfg}")
            return default_config_for_fallback.copy() 
        try:
            with open(self.config_path, "r") as f: return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {self.config_path}: {e}. Using fallback defaults.")
            return default_config_for_fallback.copy()
        except Exception as e:
            logger.error(f"Error loading strategy config {self.config_path}: {e}. Using fallback defaults.")
            return default_config_for_fallback.copy()

    def get(self, key_path: str, default_val_override: Any = None) -> Any:
        keys = key_path.split('.')
        current_level = self.params
        try:
            for key in keys: current_level = current_level[key]
            return current_level
        except (KeyError, TypeError):
            current_level_fallback = default_config_for_fallback 
            try:
                for key_fb in keys: current_level_fallback = current_level_fallback[key_fb]
                return current_level_fallback
            except (KeyError, TypeError):
                return default_val_override

strategy_config_global = StrategyConfig()

class BitgetAPI:
    def __init__(self, api_key: str = "", secret_key: str = "", passphrase: str = "", sandbox: bool = True):
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase
        self.base_url = "https://api.bitget.com"
        self.session = requests.Session()
        self.rate_limit_delay = 0.25 

        for directory in ["data", "results", "logs", "config"]:
            Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Bitget API initialized (API Key present: {bool(api_key)})")

    def _generate_signature(self, timestamp: str, method: str, request_path: str, query_string: str = "", body_string: str = "") -> str:
        if not self.secret_key: return ""
        message_to_sign = timestamp + method.upper() + request_path
        if body_string: 
            message_to_sign += body_string
        # Bitget v1 does not typically include query_string in signature for GET requests
        mac = hmac.new(self.secret_key.encode('utf-8'), message_to_sign.encode('utf-8'), hashlib.sha256)
        return base64.b64encode(mac.digest()).decode('utf-8')

    def _get_headers(self, method: str, request_path: str, query_string: str = "", body_string: str = "") -> Dict[str, str]:
        timestamp = str(int(time.time() * 1000))
        # Pass query_string and body_string to _generate_signature if your API requires them in the sig base string
        # For Bitget v1 spot, usually only timestamp + method + requestPath + body (if POST)
        # For GET, body is empty string for signature.
        sig_body_string = body_string if method.upper() != "GET" else ""
        headers = {
            "ACCESS-KEY": self.api_key,
            "ACCESS-SIGN": self._generate_signature(timestamp, method, request_path, query_string="", body_string=sig_body_string), # query_string often not in v1 sig
            "ACCESS-TIMESTAMP": timestamp,
            "ACCESS-PASSPHRASE": self.passphrase,
            "Content-Type": "application/json;charset=UTF-8", 
            "locale": "en-US"
        }
        return {k: v for k, v in headers.items() if v} 

    def get_klines(self, symbol: str, granularity: str = "4H", limit: int = 500) -> pd.DataFrame:
        api_symbol_for_request = symbol if symbol.endswith('_SPBL') else symbol + '_SPBL'
        safe_symbol_fname = api_symbol_for_request.replace('/', '_') 
        cache_file = Path(f"data/{safe_symbol_fname}_{granularity.lower()}_klines.csv")
        expected_cols = ['open', 'high', 'low', 'close', 'volume']
        cache_freshness_h = strategy_config_global.get("cleanup.cache_klines_freshness_hours", 4)

        if cache_file.exists():
            if (time.time() - cache_file.stat().st_mtime) < cache_freshness_h * 3600:
                try:
                    df_cache = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                    if not df_cache.empty and all(c in df_cache.columns for c in expected_cols) and \
                       not (df_cache[expected_cols].isnull().values.any() or np.isinf(df_cache[expected_cols].values).any()):
                        logger.debug(f"[{symbol}] Using valid cached klines.")
                        return df_cache
                except Exception as e_cache: logger.warning(f"[{symbol}] Kline cache read error: {e_cache}. Refetching.")
        
        time.sleep(self.rate_limit_delay)
        request_path = "/api/spot/v1/market/candles"
        params = {"symbol": api_symbol_for_request, "period": granularity.lower(), "limit": str(limit)}
        
        max_r, df_out = 3, pd.DataFrame()
        for attempt in range(max_r):
            headers = self._get_headers("GET", request_path) # Body is empty for GET signature
            try:
                resp = self.session.get(f"{self.base_url}{request_path}", params=params, headers=headers, timeout=20)
                logger.debug(f"[{symbol}] Kline API (att {attempt+1}): {resp.status_code}, Resp: {resp.text[:250]}")
                resp.raise_for_status()
                api_data = resp.json()
                if str(api_data.get('code')) == '00000' and isinstance(api_data.get('data'), list):
                    if not api_data['data']: logger.warning(f"[{symbol}] API success but no candle data."); return df_out
                    
                    df_temp = pd.DataFrame(api_data['data'], columns=['ts', 'open', 'high', 'low', 'close', 'baseVol', 'quoteVol'])
                    df_temp = df_temp.rename(columns={'ts': 'timestamp', 'baseVol': 'volume'})
                    df_temp['timestamp'] = pd.to_datetime(df_temp['timestamp'].astype(np.int64), unit='ms')
                    df_temp.set_index('timestamp', inplace=True)
                    
                    for col_name in expected_cols:
                        if col_name in df_temp.columns: df_temp[col_name] = pd.to_numeric(df_temp[col_name], errors='coerce')
                        else: logger.error(f"[{symbol}] Critical kline col '{col_name}' missing!"); return df_out
                    
                    df_out = df_temp[expected_cols].copy() 
                    df_out.sort_index(inplace=True)
                    df_out.replace([np.inf, -np.inf], np.nan, inplace=True); df_out.dropna(inplace=True)
                    if df_out.empty: logger.warning(f"[{symbol}] No valid klines after processing."); return df_out
                    try: df_out.to_csv(cache_file); logger.debug(f"[{symbol}] Fetched & cached klines.")
                    except Exception as e_csv: logger.error(f"[{symbol}] Error caching klines: {e_csv}")
                    return df_out 
                else:
                    err_msg, err_code = api_data.get('msg', 'Unknown API error'), api_data.get('code', 'N/A')
                    logger.warning(f"[{symbol}] API error klines (Code {err_code}): {msg}")
                    if str(err_code) == '40309': return df_out 
                    if attempt < max_r - 1: time.sleep(1 + 2**attempt + np.random.rand()) 
            except requests.exceptions.RequestException as e_req:
                logger.warning(f"[{symbol}] Request failed klines (att {attempt+1}): {e_req}")
                if attempt < max_r - 1: time.sleep(1 + 2**attempt + np.random.rand())
            except json.JSONDecodeError as e_json:
                logger.error(f"[{symbol}] JSON decode error klines (att {attempt+1}): {e_json}. Resp: {resp.text[:200] if 'resp' in locals() else 'N/A'}")
                if attempt < max_r - 1: time.sleep(1 + 2**attempt + np.random.rand())
        logger.error(f"[{symbol}] Failed to fetch klines after {max_r} attempts.")
        return df_out

    def get_all_tickers_data(self) -> List[Dict[str, Any]]: 
        request_path = "/api/spot/v1/market/tickers"
        try:
            time.sleep(self.rate_limit_delay)
            headers = self._get_headers("GET", request_path)
            response = self.session.get(f"{self.base_url}{request_path}", headers=headers, timeout=15)
            logger.debug(f"All tickers API response: {response.status_code} {response.text[:100]}")
            response.raise_for_status()
            data = response.json()
            if str(data.get('code')) == '00000' and isinstance(data.get('data'), list):
                logger.info(f"Successfully fetched data for {len(data['data'])} tickers.")
                return data['data']
            else:
                logger.error(f"Failed to fetch all tickers: {data.get('msg','Err')} (Code {data.get('code','N/A')})")
        except requests.exceptions.RequestException as e: logger.error(f"Request exception for all tickers: {e}")
        except json.JSONDecodeError as e: logger.error(f"JSON decode error for all tickers: {e}. Resp: {response.text[:200] if 'response' in locals() else 'N/A'}")
        return []

class NNFXIndicators: 
    def __init__(self, config: StrategyConfig):
        self.config_params = config.get("indicators", {})
    def _get_param(self, key: str, default: Any) -> Any: return self.config_params.get(key, default)
    def tema(self, data: pd.Series) -> pd.Series:
        p = self._get_param("tema_period", 21)
        try:
            e1=data.ewm(span=p,adjust=False).mean(); e2=e1.ewm(span=p,adjust=False).mean(); e3=e2.ewm(span=p,adjust=False).mean()
            return 3*e1 - 3*e2 + e3
        except Exception as e: logger.error(f"TEMA err(p={p}): {e}"); return pd.Series(np.nan,index=data.index)
    def kijun_sen(self, high: pd.Series, low: pd.Series) -> pd.Series:
        p = self._get_param("kijun_sen_period", 26); min_p = max(1,min(p, p//2 if p>1 else 1))
        try: return (high.rolling(p,min_p).max() + low.rolling(p,min_p).min())/2
        except Exception as e: logger.error(f"KijunSen err(p={p}): {e}"); return pd.Series(np.nan,index=high.index)
    def cci(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        p = self._get_param("cci_period", 14)
        try: return ta.trend.CCIIndicator(high,low,close,p,fillna=False).cci()
        except Exception as e: logger.error(f"CCI err(p={p}): {e}"); return pd.Series(np.nan,index=high.index)
    def williams_r(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        p = self._get_param("williams_r_period", 14)
        try: return ta.momentum.WilliamsRIndicator(high,low,close,p,fillna=False).williams_r()
        except Exception as e: logger.error(f"W%R err(p={p}): {e}"); return pd.Series(np.nan,index=high.index)
    def elder_force_index(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        p = self._get_param("elder_fi_period", 13)
        try: return ta.volume.ForceIndexIndicator(close,volume,p,fillna=False).force_index()
        except Exception as e: logger.error(f"ElderFI err(p={p}): {e}"); return pd.Series(np.nan,index=close.index)
    def klinger_oscillator(self, high: pd.Series, low: pd.Series, close: pd.Series, vol: pd.Series) -> Tuple[pd.Series, pd.Series]:
        f,s,sg = self._get_param("klinger_fast_ema",34), self._get_param("klinger_slow_ema",55), self._get_param("klinger_signal_ema",13)
        try:
            ki=ta.volume.KlingerOscillator(high,low,close,vol,f,s,sg,fillna=False); return ki.klinger(), ki.klinger_signal()
        except Exception as e: logger.error(f"Klinger err(f{f}s{s}g{sg}): {e}"); nan_s=pd.Series(np.nan,index=high.index); return nan_s,nan_s
    def chandelier_exit(self, high: pd.Series, low: pd.Series, close: pd.Series) -> Tuple[pd.Series, pd.Series]:
        p,m,ap = self._get_param("chandelier_period",22), self._get_param("chandelier_multiplier",3.0), self._get_param("atr_period_chandelier",22)
        try:
            atr_s = self.atr(high,low,close,ap); min_p=max(1,min(p,p//2 if p>1 else 1))
            h_h = high.rolling(p,min_p).max(); l_l = low.rolling(p,min_p).min()
            return h_h-(m*atr_s), l_l+(m*atr_s)
        except Exception as e: logger.error(f"Chandelier err(p{p}m{m}): {e}"); nan_s=pd.Series(np.nan,index=high.index); return nan_s,nan_s
    def parabolic_sar(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        st,mst = self._get_param("psar_step",0.02), self._get_param("psar_max_step",0.2)
        try: return ta.trend.PSARIndicator(high,low,close,st,mst,fillna=False).psar()
        except Exception as e: logger.error(f"PSAR err(s{st}m{mst}): {e}"); return pd.Series(np.nan,index=high.index)
    def atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: Optional[int] = None) -> pd.Series:
        atr_p = period if period is not None else self._get_param("atr_period_risk",14)
        try: return ta.volatility.AverageTrueRange(high,low,close,atr_p,fillna=False).average_true_range()
        except Exception as e: logger.error(f"ATR err(p={atr_p}): {e}"); return pd.Series(np.nan,index=high.index)

def _backtest_worker_process(api_config_dict: Dict, strategy_config_path_str: str, symbol: str) -> Dict:
    try:
        worker_api = BitgetAPI(**api_config_dict)
        worker_strategy_config = StrategyConfig(strategy_config_path_str) 
        worker_system = DualNNFXSystem(worker_api, worker_strategy_config)
        result = worker_system.backtest_pair(symbol)
        return result
    except Exception as e_w:
        # This print will go to the worker's stdout, not easily captured by main process logger
        print(f"WORKER PID {os.getpid()} UNHANDLED ERROR for {symbol}: {e_w}", file=sys.stderr)
        return {"symbol": symbol, "error": f"Worker process unhandled exception: {str(e_w)}"}

class DualNNFXSystem:
    def __init__(self, bitget_api_instance: BitgetAPI, config_instance: StrategyConfig):
        self.api = bitget_api_instance
        self.config = config_instance 
        self.indicators_instance = NNFXIndicators(config_instance) 
        self.indicator_calc_errors_tracker = {} 

    def _safe_calc_ind(self, idx, log_name_prefix, func, *args): 
        try:
            res = func(*args)
            if isinstance(res, pd.Series) and res.isnull().all() and not res.empty:
                logger.warning(f"Indicator '{log_name_prefix}' resulted in all NaNs (len {len(res)}).")
            elif isinstance(res, tuple) and all(isinstance(s,pd.Series) and s.isnull().all() and not s.empty for s in res):
                logger.warning(f"Indicator '{log_name_prefix}' (tuple) resulted in all NaNs.")
            return res
        except Exception as e:
            logger.error(f"Err in indicator '{log_name_prefix}': {e}", exc_info=False)
            if "Tuple" in str(func.__annotations__.get('return','')): 
                 nan_s = pd.Series(np.nan, index=idx, dtype=float)
                 num_series_expected = 2 
                 return tuple([nan_s.copy() for _ in range(num_series_expected)])
            return pd.Series(np.nan, index=idx, dtype=float)

    def calculate_indicators(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        self.indicator_calc_errors_tracker[symbol] = [] 
        data = df.copy()
        req_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(c in data.columns for c in req_cols):
            miss_c = [c for c in req_cols if c not in data.columns]
            logger.error(f"[{symbol}] Missing required columns for indicators: {miss_c}")
            for ind_col in ['tema','cci','elder_fi','chandelier_long','chandelier_short','kijun_sen','williams_r','klinger','klinger_signal','psar','atr']: data[ind_col] = np.nan
            return data

        idx = data.index
        ind = self.indicators_instance 
        data['tema'] = self._safe_calc_ind(idx, f"[{symbol}] TEMA", ind.tema, data['close'])
        data['cci'] = self._safe_calc_ind(idx, f"[{symbol}] CCI", ind.cci, data['high'], data['low'], data['close'])
        data['elder_fi'] = self._safe_calc_ind(idx, f"[{symbol}] ElderFI", ind.elder_force_index, data['close'], data['volume'])
        cl, cs = self._safe_calc_ind(idx, f"[{symbol}] Chandelier", ind.chandelier_exit, data['high'], data['low'], data['close'])
        data['chandelier_long'], data['chandelier_short'] = cl, cs
        
        data['kijun_sen'] = self._safe_calc_ind(idx, f"[{symbol}] KijunSen", ind.kijun_sen, data['high'], data['low'])
        data['williams_r'] = self._safe_calc_ind(idx, f"[{symbol}] W%R", ind.williams_r, data['high'], data['low'], data['close'])
        k, ks = self._safe_calc_ind(idx, f"[{symbol}] Klinger", ind.klinger_oscillator, data['high'], data['low'], data['close'], data['volume'])
        data['klinger'], data['klinger_signal'] = k, ks
        data['psar'] = self._safe_calc_ind(idx, f"[{symbol}] PSAR", ind.parabolic_sar, data['high'], data['low'], data['close'])
        data['atr'] = self._safe_calc_ind(idx, f"[{symbol}] ATR-Risk", ind.atr, data['high'], data['low'], data['close']) 
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame: 
        df = data.copy()
        expected_cols = ['tema', 'cci', 'elder_fi', 'kijun_sen', 'williams_r', 'klinger', 'klinger_signal', 
                         'chandelier_long', 'chandelier_short', 'psar', 'close']
        for col in expected_cols:
            if col not in df.columns: df[col] = np.nan 

        df['system_a_baseline'] = np.select([df['close'] > df['tema'], df['close'] < df['tema']], [1, -1], default=0)
        df['system_a_confirmation'] = np.select([df['cci'] > 0, df['cci'] < 0], [1, -1], default=0)
        df['system_a_volume'] = np.select([df['elder_fi'] > 0, df['elder_fi'] < 0], [1, -1], default=0)
        df['system_b_baseline'] = np.select([df['close'] > df['kijun_sen'], df['close'] < df['kijun_sen']], [1, -1], default=0)
        wpr_thresh = self.config.get("indicators.williams_r_threshold", -50) 
        df['system_b_confirmation'] = np.select([df['williams_r'] > wpr_thresh, df['williams_r'] < wpr_thresh], [1, -1], default=0)
        df['system_b_volume'] = np.select([df['klinger'] > df['klinger_signal'], df['klinger'] < df['klinger_signal']], [1, -1], default=0)

        df['long_signal'] = ((df['system_a_baseline'] == 1) & (df['system_a_confirmation'] == 1) & (df['system_a_volume'] == 1) &
                             (df['system_b_baseline'] == 1) & (df['system_b_confirmation'] == 1) & (df['system_b_volume'] == 1)).astype(bool)
        df['short_signal'] = ((df['system_a_baseline'] == -1) & (df['system_a_confirmation'] == -1) & (df['system_a_volume'] == -1) &
                              (df['system_b_baseline'] == -1) & (df['system_b_confirmation'] == -1) & (df['system_b_volume'] == -1)).astype(bool)
        df['long_exit'] = ((df['close'] < df['chandelier_long']) | (df['close'] < df['psar'])).astype(bool)
        df['short_exit'] = ((df['close'] > df['chandelier_short']) | (df['close'] > df['psar'])).astype(bool)
        return df

    def backtest_pair(self, symbol: str) -> Dict: 
        # This print is for worker process, will appear on console if worker runs directly
        # print(f"WORKER PID {os.getpid()}: Backtesting {symbol}...")
        logger.debug(f"[{symbol}] Backtest process started for symbol.") # Main process won't see this if in worker.

        cfg = self.config 
        
        df_klines = self.api.get_klines(symbol, "4H", limit=cfg.get("backtest_kline_limit", 1000))
        if df_klines.empty: return {"symbol": symbol, "error": "No kline data"}
        if len(df_klines) < cfg.get("backtest_min_data_after_get_klines", 100):
            return {"symbol": symbol, "error": f"Insufficient klines: {len(df_klines)}"}

        df_indicators = self.calculate_indicators(df_klines, symbol)
        df_indicators.dropna(inplace=True) 
        if len(df_indicators) < cfg.get("backtest_min_data_after_indicators", 50):
            return {"symbol": symbol, "error": f"Insufficient data after indicators/dropna: {len(df_indicators)}"}

        df_signals = self.generate_signals(df_indicators)
        
        trades_list, current_pos, equity_hist = [], None, []
        equity = 10000.0
        risk_val = cfg.get("risk_per_trade", 0.015)
        sl_mult = cfg.get("stop_loss_atr_multiplier", 2.0)
        tp_mult = cfg.get("take_profit_atr_multiplier", 3.0)

        for _, candle_row in df_signals.iterrows(): 
            if current_pos is None:
                atr = candle_row.get('atr', np.nan)
                if pd.notna(atr) and atr > 1e-9:
                    price = candle_row['close']
                    if candle_row.get('long_signal', False):
                        current_pos = {'type': 'long', 'entry_price': price, 'entry_time': candle_row.name, 
                                       'stop_loss': price - sl_mult * atr, 'take_profit': price + tp_mult * atr, 
                                       'atr_at_entry': atr, 'position_size': (equity * risk_val) / (sl_mult * atr)}
                    elif candle_row.get('short_signal', False):
                         current_pos = {'type': 'short', 'entry_price': price, 'entry_time': candle_row.name, 
                                       'stop_loss': price + sl_mult * atr, 'take_profit': price - tp_mult * atr, 
                                       'atr_at_entry': atr, 'position_size': (equity * risk_val) / (sl_mult * atr)}
            elif current_pos is not None:
                exit_now, reason = False, ""
                price = candle_row['close']
                if current_pos['type'] == 'long':
                    if price <= current_pos['stop_loss']: exit_now,reason=True,"SL"
                    elif price >= current_pos['take_profit']: exit_now,reason=True,"TP"
                    elif candle_row.get('long_exit',False): exit_now,reason=True,"Signal"
                elif current_pos['type'] == 'short':
                    if price >= current_pos['stop_loss']: exit_now,reason=True,"SL"
                    elif price <= current_pos['take_profit']: exit_now,reason=True,"TP"
                    elif candle_row.get('short_exit',False): exit_now,reason=True,"Signal"

                if exit_now:
                    pnl_p = (price - current_pos['entry_price']) if current_pos['type'] == 'long' else (current_pos['entry_price'] - price)
                    pnl_r_val = pnl_p / (sl_mult * current_pos['atr_at_entry']) if current_pos['atr_at_entry'] > 1e-9 else 0.0
                    pnl_usd = pnl_p * current_pos['position_size']
                    equity = max(0, equity + pnl_usd) 
                    trades_list.append({'symbol':symbol,'type':current_pos['type'],'entry_time':current_pos['entry_time'],
                                        'exit_time':candle_row.name,'entry_price':current_pos['entry_price'],'exit_price':price,
                                        'pnl_pips':pnl_p,'pnl_r':pnl_r_val,'pnl_dollar':pnl_usd,'exit_reason':reason,
                                        'atr_at_entry':current_pos['atr_at_entry'],'equity_after_trade':equity})
                    current_pos = None
            equity_hist.append({'timestamp': candle_row.name, 'equity': equity, 'in_position': current_pos is not None})
        
        if not trades_list: return {'symbol':symbol,'total_trades':0,'final_equity':equity,'equity_curve':equity_hist,'trades':[],'error':'No trades'}
        df_trades = pd.DataFrame(trades_list)
        if not df_trades.empty: 
             df_trades['entry_time'] = pd.to_datetime(df_trades['entry_time']).dt.tz_localize(None)
             df_trades['exit_time'] = pd.to_datetime(df_trades['exit_time']).dt.tz_localize(None)
        if not equity_hist: equity_hist.append({'timestamp': df_signals.index[0] if not df_signals.empty else pd.Timestamp.now(tz=None).normalize(), 'equity': 10000.0, 'in_position': False})
        
        n_trades = len(df_trades)
        wins_df = df_trades[df_trades['pnl_r']>0]; losses_df = df_trades[df_trades['pnl_r']<0]
        wr = len(wins_df)/n_trades if n_trades>0 else 0.0
        avg_w = wins_df['pnl_r'].mean() if not wins_df.empty else 0.0
        avg_l = losses_df['pnl_r'].mean() if not losses_df.empty else 0.0
        sum_profit_r = wins_df['pnl_r'].sum()
        sum_loss_r_abs = abs(losses_df['pnl_r'].sum())
        pf = sum_profit_r / sum_loss_r_abs if sum_loss_r_abs > 1e-9 else (float('inf') if sum_profit_r > 1e-9 else 0.0)
        
        return {
            'symbol':symbol, 'total_trades':n_trades, 'win_rate':wr, 'avg_win_r':avg_w, 'avg_loss_r':avg_l, 'profit_factor':pf,
            'total_return_r':df_trades['pnl_r'].sum(), 'total_return_pct':(equity/10000.0-1.0)*100.0,
            'max_consecutive_losses':self._calculate_max_consecutive_losses(df_trades),
            'max_drawdown_pct':self._calculate_max_drawdown(equity_hist),
            'sharpe_ratio':self._calculate_sharpe_ratio(df_trades,equity_hist), 
            'sortino_ratio':self._calculate_sortino_ratio(df_trades,equity_hist),
            'var_95_r':np.percentile(df_trades['pnl_r'],5) if n_trades>0 else 0.0,
            'max_loss_r':df_trades['pnl_r'].min() if n_trades>0 else 0.0,
            'final_equity':equity, 'trades':df_trades.to_dict('records'), 'equity_curve':equity_hist
        }

    def _calculate_max_consecutive_losses(self, trades_df: pd.DataFrame) -> int: 
        if trades_df.empty: return 0
        c, mc = 0,0
        for r in trades_df['pnl_r']: c = c+1 if r<0 else 0; mc = max(mc,c)
        return mc

    def _calculate_max_drawdown(self, equity_curve_data: List[Dict]) -> float: 
        if not equity_curve_data: return 0.0
        eq_s = pd.Series([p['equity'] for p in equity_curve_data])
        if eq_s.empty: return 0.0
        pk = eq_s.expanding(min_periods=1).max().replace(0,np.nan)
        dd = (eq_s - pk)/pk; dd_min = dd.min()
        return abs(dd_min*100.0) if pd.notna(dd_min) else 0.0
    
    def _calculate_annualization_factor(self, trades_df: pd.DataFrame, equity_curve_data: List[Dict]) -> float: 
        if trades_df.empty or len(trades_df)<2 or not equity_curve_data or len(equity_curve_data)<2: return 1.0
        st_ec = pd.to_datetime(equity_curve_data[0]['timestamp']).tz_localize(None)
        et_ec = pd.to_datetime(equity_curve_data[-1]['timestamp']).tz_localize(None)
        dur_d = max(1.0, (et_ec-st_ec).total_seconds()/(24*3600.0))
        tpy = (len(trades_df)/dur_d)*252.0
        return np.sqrt(tpy) if tpy>0 else 1.0

    def _calculate_sharpe_ratio(self, trades_df: pd.DataFrame, equity_curve_data: List[Dict], rfr_annual_pct: float = 0.0) -> float: 
        if trades_df.empty or trades_df['pnl_r'].isnull().all() or len(trades_df['pnl_r'].dropna())<2: return 0.0
        ret_r = trades_df['pnl_r'].dropna(); mean_r,std_r = ret_r.mean(),ret_r.std()
        if std_r==0 or pd.isna(std_r): return np.inf if mean_r>0 else (0.0 if mean_r==0 else -np.inf)
        return (mean_r/std_r) * self._calculate_annualization_factor(trades_df,equity_curve_data)

    def _calculate_sortino_ratio(self, trades_df: pd.DataFrame, equity_curve_data: List[Dict], target_r_trade: float = 0.0) -> float: 
        if trades_df.empty or trades_df['pnl_r'].isnull().all() or len(trades_df['pnl_r'].dropna())<2: return 0.0
        ret_r = trades_df['pnl_r'].dropna(); mean_r = ret_r.mean()
        down_dev_sq_r = (target_r_trade - ret_r[ret_r < target_r_trade])**2
        if down_dev_sq_r.empty: return np.inf if mean_r > target_r_trade else 0.0
        exp_down_dev_r = np.sqrt(down_dev_sq_r.mean())
        if exp_down_dev_r==0 or pd.isna(exp_down_dev_r): return np.inf if mean_r > target_r_trade else 0.0
        return ((mean_r-target_r_trade)/exp_down_dev_r) * self._calculate_annualization_factor(trades_df,equity_curve_data)

    def scan_pairs(self, symbols: List[str], save_results: bool = True, max_workers: Optional[int] = None) -> pd.DataFrame: 
        logger.info(f"Starting scan: {len(symbols)} pairs, workers <= {max_workers or os.cpu_count()}.")
        results, failed = [], {}
        api_cfg = {"api_key":self.api.api_key,"secret_key":self.api.secret_key,"passphrase":self.api.passphrase,"sandbox":False}
        strat_cfg_path = str(self.config.config_path.resolve())

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures_map = {executor.submit(_backtest_worker_process, api_cfg, strat_cfg_path, s): s for s in symbols}
            for i, fut in enumerate(as_completed(futures_map)):
                s_name = futures_map[fut]
                try:
                    res = fut.result()
                    logger.info(f"Main: Completed {s_name} ({i+1}/{len(symbols)}). Trades: {res.get('total_trades', 'N/A')}, Err: {res.get('error', 'None')}")
                    min_tr = self.config.get("backtest_min_trades_for_ranking",3)
                    if 'error' not in res or (res.get('error') == 'No trades' and res.get('total_trades',0) == 0) :
                        if res.get('total_trades',0) >= min_tr :
                            res['score'] = self._calculate_score(res) 
                            results.append(res) 
                        elif res.get('error') == 'No trades': 
                            failed[s_name] = f"Trades < {min_tr} ({res.get('total_trades',0)})"
                        else: # Other type of error from result dict 
                            failed[s_name] = res.get('error', 'Unknown processing error')
                    else: 
                        failed[s_name] = res.get('error', 'Unknown worker error')
                except Exception as e_fut:
                    logger.error(f"Exception for {s_name} future: {e_fut}", exc_info=False) # Keep main log cleaner
                    failed[s_name] = f"Future processing error: {str(e_fut)[:100]}" # Store truncated error
        
        logger.info(f"Scan done. Ranked: {len(results)}, Failed/Skipped: {len(failed)}.")
        df_res = pd.DataFrame(results) 
        if not df_res.empty:
            cols_for_final_df = ['symbol', 'score', 'total_trades', 'win_rate', 'profit_factor', 'total_return_r', 
                                 'total_return_pct', 'max_drawdown_pct', 'sharpe_ratio', 'sortino_ratio', 
                                 'max_consecutive_losses', 'var_95_r', 'max_loss_r']
            # Create missing columns with default values if not present in all results
            for col_report in cols_for_final_df:
                if col_report not in df_res.columns:
                    is_numeric_col = any(kw in col_report for kw in ['pct', 'ratio', 'score', 'r', 'trades', 'losses'])
                    df_res[col_report] = np.nan if is_numeric_col else (0 if 'trades' in col_report or 'losses' in col_report else '')
            df_res = df_res[cols_for_final_df].sort_values('score', ascending=False)

        if save_results: self._save_scan_artifacts(df_res, symbols, failed)
        return df_res

    def _save_scan_artifacts(self, df_ranked: pd.DataFrame, symbols_scanned: List[str], failed_pairs_info: Dict[str, str]): 
        ts = datetime.now().strftime('%Y%m%d_%H%M%S'); res_dir = Path("results"); res_dir.mkdir(exist_ok=True)
        if not df_ranked.empty: df_ranked.to_csv(res_dir/f"scan_results_{ts}.csv",index=False); logger.info(f"Results: {res_dir}/scan_results_{ts}.csv")
        with open(res_dir/f"scan_summary_{ts}.txt",'w') as f:
            f.write(f"Scan Summary {ts}\nAttempted: {len(symbols_scanned)}, Ranked: {len(df_ranked)}, Failed: {len(failed_pairs_info)}\n")
            if not df_ranked.empty:
                f.write("\nTop Performers:\n")
                for _,r in df_ranked.head(min(10,len(df_ranked))).iterrows(): f.write(f"  {r.get('symbol','N/A'):<12} Score:{r.get('score',0):.2f} WR:{r.get('win_rate',0):.1%} PF:{r.get('profit_factor',0):.2f}\n")
            if failed_pairs_info: f.write("\nFailed/Skipped:\n"); [f.write(f"  - {s}: {reason}\n") for s,reason in failed_pairs_info.items()]
        logger.info(f"Summary: {res_dir}/scan_summary_{ts}.txt")

    def _calculate_score(self, result_dict: Dict) -> float: 
        cfg_s = self.config.get("scoring",{}); min_tr = self.config.get("backtest_min_trades_for_ranking",3)
        if result_dict.get('total_trades',0) < min_tr: return -999.0
        
        wr,pf,ret,nt,sr,dd,mcl,v95 = (result_dict.get(k,d) for k,d in [
            ('win_rate',0.0),('profit_factor',0.0),('total_return_pct',0.0),('total_trades',0),
            ('sharpe_ratio',0.0),('max_drawdown_pct',100.0),('max_consecutive_losses',20),('var_95_r',-5.0)])

        s_wr = wr*100; s_pf = min(pf*20,100) if np.isfinite(pf) else (cfg_s.get("profit_factor_inf_score",50.0) if pf>0 else 0)
        s_ret=min(max(ret,-100),100); s_nt=min(nt*0.5,100)
        s_sr = min(max(sr*25,-100),100) if np.isfinite(sr) else (cfg_s.get("sharpe_ratio_inf_score",50.0) if sr>0 else -50)
        p_dd = dd*cfg_s.get("drawdown_penalty_multiplier",1.5); p_mcl = mcl*cfg_s.get("consecutive_loss_penalty_multiplier",3.0)
        p_v95= abs(v95)*cfg_s.get("var_95_penalty_multiplier",5.0)
        
        score = (s_wr*cfg_s.get("win_rate_weight",0.25) + s_pf*cfg_s.get("profit_factor_weight",0.25) + 
                 s_ret*cfg_s.get("return_pct_weight",0.20) + s_nt*cfg_s.get("trade_frequency_weight",0.10) + 
                 s_sr*cfg_s.get("sharpe_ratio_weight",0.20) - 
                 (p_dd*cfg_s.get("drawdown_penalty_weight",0.05)) - (p_mcl*cfg_s.get("consecutive_loss_penalty_weight",0.03)) -
                 (p_v95*cfg_s.get("var_95_penalty_weight",0.02)))
        return round(max(score,-1000.0),2)

    def get_current_signals(self, symbols: List[str]) -> pd.DataFrame: 
        logger.info(f"Getting current signals for {len(symbols)} pairs...")
        signals, cfg_sc = [], self.config.get("signal_confidence",{})
        for s in symbols:
            try:
                df_k = self.api.get_klines(s,"4H",200)
                if df_k.empty or len(df_k)<self.config.get("backtest_min_data_after_indicators",50): continue
                df_i = self.calculate_indicators(df_k,s); df_i.dropna(inplace=True)
                if df_i.empty or len(df_i)<2: continue
                df_s = self.generate_signals(df_i)
                last,curr = df_s.iloc[-2],df_s.iloc[-1] 
                sig,conf="NONE",0.0
                # Pass N most recent *closed* candles for confidence. df_s.iloc[-X:-1]
                hist_for_conf = df_s.iloc[-(cfg_sc.get("min_recent_data_for_confidence",5)+1) : -1] 
                
                if last.get('long_signal',False): sig,conf = "LONG", self._calculate_signal_confidence(hist_for_conf,'long',cfg_sc)
                elif last.get('short_signal',False): sig,conf = "SHORT", self._calculate_signal_confidence(hist_for_conf,'short',cfg_sc)
                
                price,atr_v = last.get('close',np.nan), last.get('atr',np.nan)
                slm,tpm = self.config.get("stop_loss_atr_multiplier",2.0), self.config.get("take_profit_atr_multiplier",3.0)
                sl_d,tp_d = (m*atr_v if pd.notna(atr_v) else np.nan for m in [slm,tpm])
                sl_p,tp_p = np.nan,np.nan
                if sig=="LONG" and pd.notna(price) and pd.notna(sl_d): sl_p,tp_p = price-sl_d, price+tp_d
                elif sig=="SHORT" and pd.notna(price) and pd.notna(sl_d): sl_p,tp_p = price+sl_d, price-tp_d
                
                signals.append({'symbol':s,'signal':sig,'confidence':round(conf,3),'price_at_signal':price,'atr_at_signal':atr_v,
                                'stop_loss_price':sl_p,'take_profit_price':tp_p,
                                'risk_reward_ratio':tp_d/sl_d if pd.notna(sl_d) and sl_d>1e-9 else 0.0,
                                'signal_timestamp':last.name,'current_candle_forming_time':curr.name})
                time.sleep(0.05) 
            except Exception as e_sig: logger.error(f"Err get_signals for {s}: {e_sig}",exc_info=False)
        df_o = pd.DataFrame(signals)
        return df_o.sort_values(['signal','confidence'],ascending=[True,False]) if not df_o.empty else pd.DataFrame()

    def _calculate_signal_confidence(self, recent_df: pd.DataFrame, sig_type: str, cfg_c: Dict) -> float: 
        min_h = cfg_c.get("min_recent_data_for_confidence",5)
        if len(recent_df)<min_h: return cfg_c.get("default_confidence",0.5)
        sw,tw,mw,vw = cfg_c.get("signal_support_weight",.4),cfg_c.get("trend_consistency_weight",.25),cfg_c.get("momentum_consistency_weight",.2),cfg_c.get("volume_consistency_weight",.15)
        direction = 1 if sig_type=='long' else -1
        sig_col = 'long_signal' if sig_type=='long' else 'short_signal'
        s_sup = recent_df.get(sig_col,pd.Series(False, index=recent_df.index)).mean() # Ensure index match if col missing
        s_trend = (recent_df.get('system_a_baseline',pd.Series(0, index=recent_df.index)) == direction).mean()
        s_mom = (recent_df.get('system_a_confirmation',pd.Series(0, index=recent_df.index)) == direction).mean()
        s_vol = (recent_df.get('system_a_volume',pd.Series(0, index=recent_df.index)) == direction).mean()
        conf = s_sup*sw + s_trend*tw + s_mom*mw + s_vol*vw
        return min(max(conf,0.0),1.0) 

    def export_detailed_results(self, rankings_df: pd.DataFrame, current_signals_df: pd.DataFrame) -> str: 
        ts=datetime.now().strftime('%Y%m%d_%H%M%S'); res_dir=Path("results"); res_dir.mkdir(exist_ok=True)
        fn=res_dir/f"nnfx_analysis_{ts}.xlsx"
        try:
            with pd.ExcelWriter(fn,engine='openpyxl') as w:
                rankings_df.head(min(len(rankings_df),200)).to_excel(w,sheet_name='Backtest Rankings',index=False)
                current_signals_df.to_excel(w,sheet_name='Current Signals',index=False)
                if not rankings_df.empty:
                    # Use .get for safety in case a metric column was all NaN and dropped from df_ranked
                    avg_wr_val = rankings_df['win_rate'].mean() if 'win_rate' in rankings_df else np.nan
                    pf_finite = rankings_df['profit_factor'][np.isfinite(rankings_df['profit_factor'])] if 'profit_factor' in rankings_df else pd.Series(dtype=float)
                    avg_pf_val = pf_finite.mean() if not pf_finite.empty else np.nan
                    pd.DataFrame({"Metric":["Total Ranked","Avg WR","Avg PF"], 
                                  "Value":[len(rankings_df),f"{avg_wr_val:.1%}",f"{avg_pf_val:.2f}"]}).to_excel(w,sheet_name="Summary",index=False)
            logger.info(f"Results exported: {fn}"); return str(fn)
        except Exception as e_ex: logger.error(f"Excel export err: {e_ex}",exc_info=True); return ""

def run_comprehensive_analysis(api_cfg_dict: Dict, main_strat_cfg: StrategyConfig, 
                               symbols_override: Optional[List[str]] = None,
                               max_w: Optional[int] = None) -> Dict: 
    logger.info("="*60 + "\nNNFX Bot - Comprehensive Analysis\n" + "="*60)
    t_start = time.time()
    
    api_inst = BitgetAPI(**api_cfg_dict)
    system_inst = DualNNFXSystem(api_inst, main_strat_cfg) 
    
    symbols_to_scan = []
    if symbols_override:
        symbols_to_scan = symbols_override
        logger.info(f"Using provided override list of {len(symbols_to_scan)} symbols.")
    else:
        logger.info("Fetching all tickers for dynamic selection by volume...")
        all_t = api_inst.get_all_tickers_data() # List of dicts
        
        if all_t:
            logger.info(f"CRITICAL DEBUG: Total tickers fetched: {len(all_t)}")
            if all_t: 
                logger.info(f"CRITICAL DEBUG: Sample raw ticker [0]: {all_t[0] if all_t else 'N/A'}")
                # Find a BTCUSDT like pair for detailed sample
                for ticker_sample_debug in all_t:
                    # Check multiple possible keys for symbol identifier
                    sym_id_debug = ticker_sample_debug.get('symbolId', ticker_sample_debug.get('symbol', ticker_sample_debug.get('s', '')))
                    if "BTC" in sym_id_debug.upper() and "USDT" in sym_id_debug.upper():
                        logger.info(f"CRITICAL DEBUG: Potential BTCUSDT like sample: {ticker_sample_debug}")
                        break


            # -------- SYMBOL IDENTIFIER KEY - THIS IS THE PART TO ADJUST --------
            # Based on your CRITICAL DEBUG output, change 'symbolId' to the correct key from Bitget API
            SYMBOL_IDENTIFIER_KEY_FROM_API = 'symbolId' # <--- CHANGE THIS if CRITICAL DEBUG shows a different key
            # --------------------------------------------------------------------
            logger.info(f"DEBUG: Using '{SYMBOL_IDENTIFIER_KEY_FROM_API}' as the key for symbol identification.")

            usdt_t_init = [tk for tk in all_t if tk.get(SYMBOL_IDENTIFIER_KEY_FROM_API,'').upper().endswith('USDT')]
            logger.info(f"DEBUG: Found {len(usdt_t_init)} USDT tickers initially (using key '{SYMBOL_IDENTIFIER_KEY_FROM_API}').")
            if usdt_t_init: logger.debug(f"DEBUG: USDT ticker sample [0] (pre-base filter): {usdt_t_init[0]}")
            
            usdt_t_base_filtered = usdt_t_init
            if main_strat_cfg.get("filter_by_major_bases_for_top_volume", True):
                maj_b_list = main_strat_cfg.get("major_bases_for_filtering", [])
                maj_b_set = set(b.upper() for b in maj_b_list)
                logger.info(f"DEBUG: Base filtering active. {len(maj_b_set)} major bases. Sample: {list(maj_b_set)[:5]}")
                
                temp_filtered_list = []
                for tk in usdt_t_init: 
                    symbol_id_val = tk.get(SYMBOL_IDENTIFIER_KEY_FROM_API, '').upper()
                    if symbol_id_val.endswith('USDT'):
                        base_curr = symbol_id_val[:-4] 
                        if base_curr in maj_b_set:
                            temp_filtered_list.append(tk)
                usdt_t_base_filtered = temp_filtered_list
                logger.info(f"DEBUG: {len(usdt_t_base_filtered)} USDT tickers after base filter.")
                if usdt_t_base_filtered: logger.debug(f"DEBUG: USDT ticker sample [0] (post-base filter): {usdt_t_base_filtered[0]}")
                elif usdt_t_init: logger.warning("DEBUG: Base filtering resulted in empty list. Check 'major_bases_for_filtering' in config.")
            
            def get_vol(ticker_d_item): 
                # -------- VOLUME KEY - THIS IS THE PART TO ADJUST --------
                # Based on your CRITICAL DEBUG output for a ticker, identify the USDT volume key
                VOLUME_KEY_FROM_API = 'usdtVol' # <--- CHANGE THIS if CRITICAL DEBUG shows a different key like 'quoteVol', 'quoteVolume', 'vol24hQuote', etc.
                # Fallback keys if the primary one is not found or is invalid
                FALLBACK_VOLUME_KEYS = ['quoteVol', 'quoteVolume', 'vol'] 
                # -------------------------------------------------------
                logger.debug(f"DEBUG get_vol: For {ticker_d_item.get(SYMBOL_IDENTIFIER_KEY_FROM_API)}, trying primary volume key '{VOLUME_KEY_FROM_API}'.")

                vol_str = ticker_d_item.get(VOLUME_KEY_FROM_API)
                if vol_str is not None:
                    try: return float(str(vol_str))
                    except (ValueError,TypeError): 
                        logger.debug(f"DEBUG get_vol: Primary key '{VOLUME_KEY_FROM_API}' value '{vol_str}' invalid for {ticker_d_item.get(SYMBOL_IDENTIFIER_KEY_FROM_API)}. Trying fallbacks.")
                
                for fb_key in FALLBACK_VOLUME_KEYS:
                    vol_str_fb = ticker_d_item.get(fb_key)
                    if vol_str_fb is not None:
                        logger.debug(f"DEBUG get_vol: For {ticker_d_item.get(SYMBOL_IDENTIFIER_KEY_FROM_API)}, trying fallback volume key '{fb_key}'.")
                        try: return float(str(vol_str_fb))
                        except (ValueError,TypeError): continue
                
                logger.debug(f"DEBUG get_vol: Volume not found or invalid for {ticker_d_item.get(SYMBOL_IDENTIFIER_KEY_FROM_API)} after all attempts, returning 0.")
                return 0.0

            t_with_vol = [{'vol': get_vol(tk), 'ticker_obj': tk} for tk in usdt_t_base_filtered]
            t_with_vol.sort(key=lambda x: x['vol'], reverse=True)
            sorted_usdt_t = [item['ticker_obj'] for item in t_with_vol]

            log_top_5_vols = [{'s':t.get(SYMBOL_IDENTIFIER_KEY_FROM_API), 'v':get_vol(t)} for t in sorted_usdt_t[:5]]
            logger.info(f"DEBUG: Top 5 USDT tickers post-sort by volume: {log_top_5_vols}")
            if sorted_usdt_t and not any(item['v']>0 for item in log_top_5_vols): 
                logger.warning("DEBUG: Top sorted tickers all show 0 volume. Check volume key config and API response.")

            top_n_cfg = main_strat_cfg.get("top_n_pairs_by_volume_to_scan", 8)
            final_t_list = sorted_usdt_t[:top_n_cfg]
            
            symbols_to_scan = [tk.get(SYMBOL_IDENTIFIER_KEY_FROM_API,'').replace('_SPBL','') for tk in final_t_list if tk.get(SYMBOL_IDENTIFIER_KEY_FROM_API)]
            symbols_to_scan = [s for s in symbols_to_scan if s] 
            logger.info(f"Selected Top {len(symbols_to_scan)} symbols for scan: {symbols_to_scan}")
        else: 
            logger.error("Ticker fetch failed. Using fallback: ['BTCUSDT', 'ETHUSDT']")
            symbols_to_scan = ['BTCUSDT', 'ETHUSDT']

    if not symbols_to_scan:
        logger.error("No symbols selected. Aborting analysis.")
        return {'success':False,'error':'No symbols to scan','scan_duration_seconds':0,'total_duration_seconds':time.time()-t_start}

    logger.info(f"Starting scan for {len(symbols_to_scan)} symbols...")
    t_scan_start = time.time()
    df_ranks = system_inst.scan_pairs(symbols_to_scan, save_results=True, max_workers=max_w)
    scan_dur = time.time() - t_scan_start
    logger.info(f"Scan completed in {scan_dur:.2f}s.")

    payload = {'success':False, 'scan_duration_seconds':round(scan_dur,2)}
    if not df_ranks.empty:
        payload['success'] = True
        payload['rankings_top_5'] = df_ranks[['symbol','score','win_rate','total_return_pct']].head().to_dict('records')
        top_sigs_symbols = df_ranks['symbol'].head(5).tolist() # Use 'symbol' from df_ranks
        if top_sigs_symbols:
            logger.info(f"Getting current signals for top ranked: {top_sigs_symbols}")
            df_curr_sigs = system_inst.get_current_signals(top_sigs_symbols)
            payload['current_active_signals'] = df_curr_sigs[df_curr_sigs['signal']!='NONE'].to_dict('records')
            payload['export_file'] = system_inst.export_detailed_results(df_ranks, df_curr_sigs)
        else: 
            payload['current_active_signals'] = []
            payload['export_file'] = system_inst.export_detailed_results(df_ranks, pd.DataFrame())
    else:
        payload['error'] = "No pairs ranked successfully."
        system_inst.export_detailed_results(pd.DataFrame(),pd.DataFrame()) 

    payload['total_duration_seconds'] = round(time.time()-t_start,2)
    logger.info(f"Analysis finished in {payload['total_duration_seconds']:.2f}s.")
    return payload

def cleanup_old_files(): 
    cfg_cl = strategy_config_global.get("cleanup",{}); d_old=cfg_cl.get("cache_days_old",7); max_res=cfg_cl.get("max_results_to_keep",20)
    cut_ts=(datetime.now()-timedelta(days=d_old)).timestamp(); clean_n=0
    for d_name,time_sort_cl in [("data",False),("results",True),("logs",True)]:
        p_dir=Path(d_name); p_dir.mkdir(exist_ok=True)
        files=[f for ft in ["*.csv","*.json","*.txt","*.xlsx","*.log"] for f in p_dir.glob(ft)]
        del_list=[]
        if time_sort_cl: files.sort(key=lambda x:x.stat().st_mtime); del_list=files[:-max_res] if len(files)>max_res else []
        else: del_list=[f for f in files if f.stat().st_mtime < cut_ts]
        for old_f in del_list:
            try: old_f.unlink(); logger.debug(f"Cleaned: {old_f}"); clean_n+=1
            except Exception as e_cl: logger.warning(f"Cleanup err {old_f}: {e_cl}")
    logger.info(f"Cleanup done. Removed {clean_n} files.")

if __name__ == "__main__":
    # This check ensures multiprocessing works correctly when script is frozen/compiled
    # For normal Python script execution, it's not strictly necessary but good practice.
    if sys.platform.startswith('win'): # For Windows compatibility with multiprocessing
        # You might need to do this if you're packaging with PyInstaller on Windows
        # from multiprocessing import freeze_support
        # freeze_support()
        pass


    ts_run = datetime.now().strftime('%Y%m%d_%H%M%S')
    h_file_log = None
    try:
        logs_dir_main = Path("logs"); logs_dir_main.mkdir(parents=True, exist_ok=True)
        path_log_file_main = logs_dir_main / f"system_run_{ts_run}.log"
        h_file_log = logging.FileHandler(path_log_file_main)
        h_file_log.setLevel(logging.INFO) 
        h_file_log.setFormatter(logging.Formatter('%(asctime)s - %(processName)s - %(name)s - %(levelname)s - %(message)s')) # Added processName
        logging.getLogger().addHandler(h_file_log)
        
        for h_console_main in logging.getLogger().handlers:
            if isinstance(h_console_main, logging.StreamHandler) and not isinstance(h_console_main, logging.FileHandler):
                h_console_main.setLevel(logging.INFO) 
                h_console_main.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')) # Simpler console format
        
        logger.info(f"NNFX System Run ID: {ts_run}")
        logger.info(f"Strategy Config: {strategy_config_global.config_path}")
        logger.info(f"File Log (INFO+): {path_log_file_main}")
        logger.info(f"Console Log (INFO+) active.")

        cfg_api_main = {}
        path_api_cfg_main = Path("config/api_config.json")
        if path_api_cfg_main.exists():
            try: 
                with open(path_api_cfg_main,"r") as f: cfg_api_main=json.load(f)
                logger.info("API credentials loaded.")
            except Exception as e_api_main: logger.error(f"API cfg load err {path_api_cfg_main}: {e_api_main}. No API keys.")
        else:
            logger.warning(f"API cfg {path_api_cfg_main} not found. Creating dummy. Public API only.")
            try:
                path_api_cfg_main.parent.mkdir(exist_ok=True, parents=True)
                with open(path_api_cfg_main,"w") as f_dum_main: json.dump({"api_key":"", "secret_key":"", "passphrase":"", "sandbox":True},f_dum_main,indent=4)
                logger.info(f"Dummy API cfg created: {path_api_cfg_main}. Please update.")
            except Exception as e_dum_api_main: logger.error(f"Dummy API cfg creation err: {e_dum_api_main}")
        
        symbols_for_run_main = None 
        # symbols_for_run_main = ['BTCUSDT', 'ETHUSDT'] # Example manual override
        
        cpus_main = os.cpu_count()
        num_proc_workers_main = max(1, cpus_main - 1 if cpus_main and cpus_main > 1 else 1)
        # num_proc_workers_main = 1 # Force single worker for easier debugging if needed
        logger.info(f"Using up to {num_proc_workers_main} worker processes for scan.")

        output_analysis_main = run_comprehensive_analysis(
            cfg_api_main, strategy_config_global, 
            symbols_override=symbols_for_run_main, 
            max_w=num_proc_workers_main)
        
        if output_analysis_main.get('success'): logger.info("Main: Analysis OK.")
        else: logger.error(f"Main: Analysis FAILED: {output_analysis_main.get('error','Unknown')}")

    except Exception as e_main_run_block: logger.critical("MAIN EXECUTION BLOCK ERROR:", exc_info=True)
    finally:
        logger.info("Final cleanup...")
        cleanup_old_files()
        logger.info(f"Session {ts_run} ended.")
        if h_file_log: logging.getLogger().removeHandler(h_file_log); h_file_log.close()