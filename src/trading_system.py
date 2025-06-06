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
# from functools import partial # Not strictly needed with the worker function approach

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='ta')
warnings.filterwarnings('ignore', category=RuntimeWarning) # For potential mean of empty slice in pandas/numpy

# Configure logging
# Ensure basicConfig is called only once
if not logging.getLogger().handlers: # Check if handlers are already configured
    logging.basicConfig(
        level=logging.DEBUG, # Set root logger level to DEBUG for console
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)] # Default to stdout
    )
logger = logging.getLogger(__name__) # Get logger for this module

# --- Configuration Loader ---
class StrategyConfig:
    def __init__(self, config_path_str="config/strategy_config.json"):
        self.config_path = Path(config_path_str)
        self.params = self._load_config()
        logger.info(f"Strategy configuration loaded from: {self.config_path}")

    def _load_config(self) -> Dict[str, Any]:
        default_config_for_fallback = {
            "major_bases_for_filtering": ["BTC", "ETH", "SOL", "AVAX", "LINK", "ADA", "DOT", "MATIC"],
            "filter_by_major_bases_for_top_volume": True,
            "top_n_pairs_by_volume_to_scan": 8,
            "backtest_kline_limit": 1000,
            "backtest_min_data_after_get_klines": 100,
            "backtest_min_data_after_indicators": 50,
            "backtest_min_trades_for_ranking": 5,
            "risk_per_trade": 0.015,
            "stop_loss_atr_multiplier": 2.0,
            "take_profit_atr_multiplier": 3.0,
            "indicators": { "tema_period": 21, "cci_period": 14, "elder_fi_period": 13,
                            "chandelier_period": 22, "chandelier_multiplier": 3.0,
                            "kijun_sen_period": 26, "williams_r_period": 14,
                            "klinger_fast_ema": 34, "klinger_slow_ema": 55, "klinger_signal_ema": 13,
                            "psar_step": 0.02, "psar_max_step": 0.2,
                            "atr_period_risk": 14, "atr_period_chandelier": 22},
            "scoring": { "win_rate_weight": 0.25, "profit_factor_weight": 0.25, "return_pct_weight": 0.20,
                         "trade_frequency_weight": 0.10, "sharpe_ratio_weight": 0.20,
                         "drawdown_penalty_multiplier": 1.5, "drawdown_penalty_weight": 0.05,
                         "consecutive_loss_penalty_multiplier": 3.0, "consecutive_loss_penalty_weight": 0.03,
                         "var_95_penalty_multiplier": 5.0, "var_95_penalty_weight": 0.02,
                         "profit_factor_inf_score": 50, "sharpe_ratio_inf_score": 50},
            "signal_confidence": { "min_recent_data_for_confidence": 3, "default_confidence": 0.5,
                                   "signal_support_weight": 0.4, "trend_consistency_weight": 0.25,
                                   "momentum_consistency_weight": 0.2, "volume_consistency_weight": 0.15},
            "cleanup": {"cache_days_old": 7, "max_results_to_keep": 20, "cache_klines_freshness_hours": 4}
        }
        if not self.config_path.exists():
            logger.warning(f"Strategy config file not found: {self.config_path}. Using hardcoded fallback defaults.")
            # Create dummy config file
            try:
                self.config_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.config_path, "w") as f_cfg:
                    json.dump(default_config_for_fallback, f_cfg, indent=4)
                logger.info(f"Created dummy strategy config at {self.config_path}. Please review and customize.")
            except Exception as e_create_cfg:
                logger.error(f"Could not create dummy strategy config: {e_create_cfg}")
            return default_config_for_fallback
        try:
            with open(self.config_path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {self.config_path}: {e}. Using fallback defaults.")
            return default_config_for_fallback
        except Exception as e:
            logger.error(f"Error loading strategy config {self.config_path}: {e}. Using fallback defaults.")
            return default_config_for_fallback

    def get(self, key_path: str, default: Any = None) -> Any:
        keys = key_path.split('.')
        value = self.params
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            # logger.warning(f"Config key '{key_path}' not found or path invalid. Using provided default: {default}")
            # Fallback to default config for this key if provided one does not exist
            temp_value = default_config_for_fallback # Use the global fallback
            try:
                for key_fb in keys:
                    temp_value = temp_value[key_fb]
                logger.warning(f"Config key '{key_path}' not found. Using fallback default: {temp_value}")
                return temp_value
            except (KeyError, TypeError):
                logger.warning(f"Config key '{key_path}' not found in primary or fallback. Using provided default: {default}")
                return default


strategy_config_global = StrategyConfig()

class BitgetAPI:
    def __init__(self, api_key: str = "", secret_key: str = "", 
                 passphrase: str = "", sandbox: bool = True):
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase
        self.base_url = "https://api.bitget.com" 
        self.session = requests.Session()
        self.rate_limit_delay = 0.2  

        for directory in ["data", "results", "logs", "config"]:
            Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Bitget API initialized (API Key present: {bool(api_key)})")

    def _generate_signature(self, timestamp: str, method: str, request_path: str, body: str = "") -> str:
        if not self.secret_key: return "" 
        message = timestamp + method.upper() + request_path + body
        mac = hmac.new(self.secret_key.encode('utf-8'), message.encode('utf-8'), hashlib.sha256)
        return base64.b64encode(mac.digest()).decode('utf-8')

    def _get_headers(self, method: str, request_path: str, body: str = "") -> Dict[str, str]:
        timestamp = str(int(time.time() * 1000))
        headers = {
            "ACCESS-KEY": self.api_key,
            "ACCESS-SIGN": self._generate_signature(timestamp, method, request_path, body),
            "ACCESS-TIMESTAMP": timestamp,
            "ACCESS-PASSPHRASE": self.passphrase,
            "Content-Type": "application/json",
            "locale": "en-US" 
        }
        return {k: v for k, v in headers.items() if v} 

    def get_klines(self, symbol: str, granularity: str = "4H", 
                   limit: int = 500) -> pd.DataFrame:
        api_symbol = symbol if symbol.endswith('_SPBL') else symbol + '_SPBL'
        # Ensure granularity is part of the filename to avoid clashes if different granularities are fetched
        cache_file = Path(f"data/{api_symbol}_{granularity.lower()}_klines.csv")
        expected_cols_for_cache = ['open', 'high', 'low', 'close', 'volume']
        
        cache_freshness_hours = strategy_config_global.get("cleanup.cache_klines_freshness_hours", 4)

        if cache_file.exists():
            cache_age_seconds = time.time() - cache_file.stat().st_mtime
            if cache_age_seconds < cache_freshness_hours * 3600:
                try:
                    cached_df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                    if not cached_df.empty and all(col in cached_df.columns for col in expected_cols_for_cache):
                        if not (cached_df[expected_cols_for_cache].isnull().values.any() or \
                                np.isinf(cached_df[expected_cols_for_cache].values).any()):
                            logger.debug(f"[{symbol}] Using valid cached kline data.")
                            return cached_df
                        else: logger.warning(f"[{symbol}] Cached kline data contains NaN/inf. Refetching.")
                    else: logger.warning(f"[{symbol}] Cached kline data is empty/missing cols. Refetching.")
                except Exception as e:
                    logger.warning(f"[{symbol}] Failed to read/validate kline cache: {e}. Refetching.")
        
        time.sleep(self.rate_limit_delay)
        # Corrected request path for Bitget API v1 market candles
        request_path_for_signature = "/api/spot/v1/market/candles" 
        full_url = f"{self.base_url}{request_path_for_signature}"
        
        # Parameters should be part of the query string for GET, not in signature body for this endpoint
        params_str = f"symbol={api_symbol}&period={granularity.lower()}&limit={limit}"
        request_path_with_params = f"{request_path_for_signature}?{params_str}"


        max_retries = 3
        for attempt in range(max_retries):
            # Headers must be generated per request due to timestamp
            # For public GET endpoints, some APIs don't need full auth headers, but Bitget might.
            # If Bitget /market/candles is truly public and doesn't need auth, headers can be simplified.
            # Assuming it might need auth for higher rate limits or consistency:
            current_headers = self._get_headers("GET", request_path_for_signature, params_str if self.api_key else "") # Pass params to sig if auth'd

            try:
                # Pass params directly to requests.get, not in the URL if headers are complex/signed
                response = self.session.get(full_url, params={"symbol": api_symbol, "period": granularity.lower(), "limit": str(limit)}, timeout=15, headers=current_headers)
                logger.debug(f"[{symbol}] Raw API klines response (attempt {attempt+1}): {response.status_code} {response.text[:200]}")
                response.raise_for_status()
                data = response.json()
                
                if str(data.get('code')) == '00000' and isinstance(data.get('data'), list):
                    if not data['data']: # Empty list of candles
                        logger.warning(f"[{symbol}] API returned success but no candle data.")
                        return pd.DataFrame()

                    df = pd.DataFrame(data['data'], columns=['ts', 'open', 'high', 'low', 'close', 'baseVol', 'quoteVol'])
                    df = df.rename(columns={'ts': 'timestamp', 'baseVol': 'volume'})
                    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(np.int64), unit='ms')
                    df = df.set_index('timestamp')
                    
                    for col in expected_cols_for_cache:
                        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
                        else:
                            logger.error(f"[{symbol}] Critical kline column '{col}' missing. Data sample: {data['data'][0] if data['data'] else 'empty'}")
                            return pd.DataFrame()
                    
                    df = df[expected_cols_for_cache].copy() # Use .copy() to avoid SettingWithCopyWarning
                    df.sort_index(inplace=True)
                    df.replace([np.inf, -np.inf], np.nan, inplace=True)
                    df.dropna(inplace=True)
                    
                    if df.empty:
                        logger.warning(f"[{symbol}] No valid kline data after processing.")
                        return pd.DataFrame()
                    try:    
                        df.to_csv(cache_file)
                        logger.debug(f"[{symbol}] Fetched and cached kline data.")
                    except Exception as e_csv: logger.error(f"[{symbol}] Error caching klines: {e_csv}")
                    return df
                else:
                    msg = data.get('msg', 'Unknown API error')
                    code = data.get('code', 'N/A')
                    logger.warning(f"[{symbol}] API error for klines (Code: {code}): {msg}")
                    if str(code) == '40309': # Symbol delisted/unavailable
                        logger.error(f"[{symbol}] Symbol appears delisted/unavailable per API. Not retrying.")
                        return pd.DataFrame() 
                    if attempt < max_retries - 1: time.sleep(2 ** attempt + np.random.rand()) # Add jitter
            except requests.exceptions.RequestException as e:
                logger.warning(f"[{symbol}] Request failed for klines, attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1: time.sleep(2 ** attempt + np.random.rand())
            except json.JSONDecodeError as e:
                logger.error(f"[{symbol}] JSON decode error for klines (attempt {attempt+1}): {e}. Response: {response.text[:200] if 'response' in locals() else 'N/A'}")
                if attempt < max_retries - 1: time.sleep(2 ** attempt + np.random.rand())

        logger.error(f"[{symbol}] Failed to fetch kline data after {max_retries} attempts.")
        return pd.DataFrame()

    def get_all_tickers_data(self) -> List[Dict[str, Any]]:
        request_path_for_signature = "/api/spot/v1/market/tickers"
        full_url = f"{self.base_url}{request_path_for_signature}"
        try:
            time.sleep(self.rate_limit_delay)
            current_headers = self._get_headers("GET", request_path_for_signature, "")
            response = self.session.get(full_url, timeout=15, headers=current_headers)
            logger.debug(f"All tickers API response: {response.status_code} {response.text[:100]}")
            response.raise_for_status()
            data = response.json()
            if str(data.get('code')) == '00000' and isinstance(data.get('data'), list):
                logger.info(f"Successfully fetched data for {len(data['data'])} tickers.")
                return data['data']
            else:
                logger.error(f"Failed to fetch all tickers data: {data.get('msg', 'Unknown error')} (Code: {data.get('code')})")
                return []
        except requests.exceptions.RequestException as e:
            logger.error(f"Request exception while fetching all tickers: {e}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error while fetching all tickers: {e}. Response: {response.text[:200] if 'response' in locals() else 'N/A'}")
            return []

class NNFXIndicators:
    def __init__(self, config: StrategyConfig):
        self.config_params = config.get("indicators", {})

    def _get_param(self, key: str, default: Any) -> Any:
        return self.config_params.get(key, default)

    def tema(self, data: pd.Series) -> pd.Series:
        period = self._get_param("tema_period", 21)
        try:
            ema1 = data.ewm(span=period, adjust=False).mean()
            ema2 = ema1.ewm(span=period, adjust=False).mean()
            ema3 = ema2.ewm(span=period, adjust=False).mean()
            return 3 * ema1 - 3 * ema2 + ema3
        except Exception as e:
            logger.error(f"Error calculating TEMA(p={period}): {e}")
            return pd.Series(np.nan, index=data.index, dtype=float)
    
    def kijun_sen(self, high: pd.Series, low: pd.Series) -> pd.Series:
        period = self._get_param("kijun_sen_period", 26)
        try:
            # Ensure min_periods is at least 1 and not greater than window
            min_p = max(1, min(period, period // 2 if period > 1 else 1))
            highest_high = high.rolling(window=period, min_periods=min_p).max()
            lowest_low = low.rolling(window=period, min_periods=min_p).min()
            return (highest_high + lowest_low) / 2
        except Exception as e:
            logger.error(f"Error calculating Kijun-Sen(p={period}): {e}")
            return pd.Series(np.nan, index=high.index, dtype=float)

    def cci(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        period = self._get_param("cci_period", 14)
        try:
            return ta.trend.CCIIndicator(high=high, low=low, close=close, window=period, fillna=False).cci()
        except Exception as e:
            logger.error(f"Error calculating CCI(p={period}): {e}")
            return pd.Series(np.nan, index=high.index, dtype=float)

    def williams_r(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        period = self._get_param("williams_r_period", 14)
        try:
            return ta.momentum.WilliamsRIndicator(high=high, low=low, close=close, lbp=period, fillna=False).williams_r()
        except Exception as e:
            logger.error(f"Error calculating Williams %R(p={period}): {e}")
            return pd.Series(np.nan, index=high.index, dtype=float)

    def elder_force_index(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        period = self._get_param("elder_fi_period", 13)
        try:
            return ta.volume.ForceIndexIndicator(close=close, volume=volume, window=period, fillna=False).force_index()
        except Exception as e:
            logger.error(f"Error calculating Elder FI(p={period}): {e}")
            return pd.Series(np.nan, index=close.index, dtype=float)

    def klinger_oscillator(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> Tuple[pd.Series, pd.Series]:
        fast = self._get_param("klinger_fast_ema", 34)
        slow = self._get_param("klinger_slow_ema", 55)
        signal = self._get_param("klinger_signal_ema", 13)
        try:
            k_ind = ta.volume.KlingerOscillator(high=high, low=low, close=close, volume=volume, window_fast=fast, window_slow=slow, window_sign=signal, fillna=False)
            return k_ind.klinger(), k_ind.klinger_signal()
        except Exception as e:
            logger.error(f"Error calculating Klinger(f={fast},s={slow},sig={signal}): {e}")
            nan_s = pd.Series(np.nan, index=high.index, dtype=float)
            return nan_s, nan_s
    
    def chandelier_exit(self, high: pd.Series, low: pd.Series, close: pd.Series) -> Tuple[pd.Series, pd.Series]:
        period = self._get_param("chandelier_period", 22)
        multiplier = self._get_param("chandelier_multiplier", 3.0)
        atr_period = self._get_param("atr_period_chandelier", period)
        try:
            atr_s = self.atr(high, low, close, period=atr_period)
            min_p = max(1, min(period, period // 2 if period > 1 else 1))
            highest_h = high.rolling(window=period, min_periods=min_p).max()
            lowest_l = low.rolling(window=period, min_periods=min_p).min()
            long_exit = highest_h - (multiplier * atr_s)
            short_exit = lowest_l + (multiplier * atr_s)
            return long_exit, short_exit
        except Exception as e:
            logger.error(f"Error calculating Chandelier Exit(p={period},m={multiplier}): {e}")
            nan_s = pd.Series(np.nan, index=high.index, dtype=float)
            return nan_s, nan_s

    def parabolic_sar(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        step = self._get_param("psar_step", 0.02)
        max_step = self._get_param("psar_max_step", 0.2)
        try:
            return ta.trend.PSARIndicator(high=high, low=low, close=close, step=step, max_step=max_step, fillna=False).psar()
        except Exception as e:
            logger.error(f"Error calculating PSAR(s={step},m={max_step}): {e}")
            return pd.Series(np.nan, index=high.index, dtype=float)
    
    def atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: Optional[int] = None) -> pd.Series:
        atr_p = period if period is not None else self._get_param("atr_period_risk", 14)
        try:
            return ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=atr_p, fillna=False).average_true_range()
        except Exception as e:
            logger.error(f"Error calculating ATR(p={atr_p}): {e}")
            return pd.Series(np.nan, index=high.index, dtype=float)

# --- Worker function for parallel backtesting ---
# Has to be at the top level or a static method of a picklable class
def _backtest_worker_process(api_config_dict: Dict, strategy_config_path_str: str, symbol: str) -> Dict:
    # Each worker needs its own instances, especially for non-picklable objects like sessions
    # or to avoid shared state issues.
    try:
        local_api = BitgetAPI(**api_config_dict)
        local_strategy_config = StrategyConfig(strategy_config_path_str) # Load config in worker
        local_system = DualNNFXSystem(local_api, local_strategy_config)
        # logger.debug(f"Worker for {symbol} starting backtest.") # This logger won't go to main process easily
        return local_system.backtest_pair(symbol)
    except Exception as e_worker:
        # logger.error(f"Worker for {symbol} failed: {e_worker}") # Again, logging here is tricky
        # print(f"WORKER ERROR for {symbol}: {e_worker}", file=sys.stderr) # Print to stderr for visibility
        return {"symbol": symbol, "error": f"Worker process failed: {e_worker}"}


class DualNNFXSystem:
    def __init__(self, bitget_api_instance: BitgetAPI, config_instance: StrategyConfig):
        self.api = bitget_api_instance
        self.config = config_instance 
        self.indicators = NNFXIndicators(config_instance)
        self.indicator_calc_errors = {} 

    def _safe_calculate_indicator(self, data_df_index, indicator_name_log, calculation_func, *args):
        # This helper might be too generic if output structures vary widely.
        # Direct try-except in calculate_indicators is often clearer.
        try:
            result = calculation_func(*args)
            # Check if result is all NaNs, which might indicate an issue from underlying library
            if isinstance(result, pd.Series) and result.isnull().all():
                logger.warning(f"Indicator '{indicator_name_log}' resulted in all NaNs.")
            elif isinstance(result, tuple) and all(isinstance(s, pd.Series) and s.isnull().all() for s in result):
                logger.warning(f"Indicator '{indicator_name_log}' (tuple) resulted in all NaNs.")
            return result
        except Exception as e:
            logger.error(f"Error in indicator '{indicator_name_log}' calculation: {e}", exc_info=False)
            if "Tuple" in str(calculation_func.__annotations__.get('return', '')):
                nan_s = pd.Series(np.nan, index=data_df_index, dtype=float)
                if indicator_name_log in ["klinger", "chandelier_exit"]: return nan_s, nan_s
            return pd.Series(np.nan, index=data_df_index, dtype=float)

    def calculate_indicators(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        self.indicator_calc_errors[symbol] = [] 
        data = df.copy()
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_cols):
            missing = [col for col in required_cols if col not in data.columns]
            logger.error(f"[{symbol}] Missing required columns for indicators: {missing}")
            self.indicator_calc_errors[symbol].append(f"Missing columns: {missing}")
            # Add NaN columns for all expected indicators so subsequent steps don't fail on missing columns
            expected_indicator_cols = ['tema', 'cci', 'elder_fi', 'chandelier_long', 'chandelier_short',
                                       'kijun_sen', 'williams_r', 'klinger', 'klinger_signal', 'psar', 'atr']
            for col_ind in expected_indicator_cols: data[col_ind] = np.nan
            return data

        idx = data.index # For creating NaN series with correct index

        # System A
        data['tema'] = self._safe_calculate_indicator(idx, f"[{symbol}] tema", self.indicators.tema, data['close'])
        data['cci'] = self._safe_calculate_indicator(idx, f"[{symbol}] cci", self.indicators.cci, data['high'], data['low'], data['close'])
        data['elder_fi'] = self._safe_calculate_indicator(idx, f"[{symbol}] elder_fi", self.indicators.elder_force_index, data['close'], data['volume'])
        data['chandelier_long'], data['chandelier_short'] = self._safe_calculate_indicator(
            idx, f"[{symbol}] chandelier_exit", self.indicators.chandelier_exit, data['high'], data['low'], data['close']
        )
        
        # System B
        data['kijun_sen'] = self._safe_calculate_indicator(idx, f"[{symbol}] kijun_sen", self.indicators.kijun_sen, data['high'], data['low'])
        data['williams_r'] = self._safe_calculate_indicator(idx, f"[{symbol}] williams_r", self.indicators.williams_r, data['high'], data['low'], data['close'])
        data['klinger'], data['klinger_signal'] = self._safe_calculate_indicator(
            idx, f"[{symbol}] klinger", self.indicators.klinger_oscillator, data['high'], data['low'], data['close'], data['volume']
        )
        data['psar'] = self._safe_calculate_indicator(idx, f"[{symbol}] psar", self.indicators.parabolic_sar, data['high'], data['low'], data['close'])
        
        data['atr'] = self._safe_calculate_indicator(idx, f"[{symbol}] atr_risk", self.indicators.atr, data['high'], data['low'], data['close'])
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        indicator_cols_for_signal = ['tema', 'cci', 'elder_fi', 'kijun_sen', 'williams_r', 'klinger', 'klinger_signal',
                                     'chandelier_long', 'chandelier_short', 'psar']
        for col in indicator_cols_for_signal:
            if col not in df.columns: df[col] = np.nan # Ensure columns exist, fill with NaN if created by error handling

        df['system_a_baseline'] = np.where(df['close'] > df['tema'], 1, np.where(df['close'] < df['tema'], -1, 0))
        df['system_a_confirmation'] = np.where(df['cci'] > 0, 1, np.where(df['cci'] < 0, -1, 0))
        df['system_a_volume'] = np.where(df['elder_fi'] > 0, 1, np.where(df['elder_fi'] < 0, -1, 0))
        
        df['system_b_baseline'] = np.where(df['close'] > df['kijun_sen'], 1, np.where(df['close'] < df['kijun_sen'], -1, 0))
        
        # Williams %R logic: confirm actual NNFX rule.
        # A common rule: Long confirmation if W%R > -50, Short confirmation if W%R < -50.
        # Let's use -50 as a common example, this should be configurable or verified.
        wpr_threshold = -50 # Example, make this configurable if desired
        df['system_b_confirmation'] = np.select(
            [df['williams_r'] > wpr_threshold, df['williams_r'] < wpr_threshold],
            [1, -1], default=0
        )
        
        df['system_b_volume'] = np.where(df['klinger'] > df['klinger_signal'], 1, np.where(df['klinger'] < df['klinger_signal'], -1, 0))

        df['long_signal'] = (
            (df['system_a_baseline'] == 1) & (df['system_a_confirmation'] == 1) & (df['system_a_volume'] == 1) &
            (df['system_b_baseline'] == 1) & (df['system_b_confirmation'] == 1) & (df['system_b_volume'] == 1)
        ).astype(bool)
        df['short_signal'] = (
            (df['system_a_baseline'] == -1) & (df['system_a_confirmation'] == -1) & (df['system_a_volume'] == -1) &
            (df['system_b_baseline'] == -1) & (df['system_b_confirmation'] == -1) & (df['system_b_volume'] == -1)
        ).astype(bool)
        
        df['long_exit'] = ((df['close'] < df['chandelier_long']) | (df['close'] < df['psar'])).astype(bool)
        df['short_exit'] = ((df['close'] > df['chandelier_short']) | (df['close'] > df['psar'])).astype(bool)
        return df

    def backtest_pair(self, symbol: str) -> Dict:
        #logger.info(f"[{symbol}] Backtesting started in worker/process...") # Use print for worker logs or setup queue handler
        print(f"WORKER [{os.getpid()}] Backtesting {symbol}...") # Simple print for worker visibility

        cfg_bt = self.config # StrategyConfig instance
        
        df = self.api.get_klines(symbol, "4H", limit=cfg_bt.get("backtest_kline_limit", 1000))
        if df.empty: return {"symbol": symbol, "error": "No data available from API"}
        if len(df) < cfg_bt.get("backtest_min_data_after_get_klines", 100):
            return {"symbol": symbol, "error": f"Insufficient initial data: {len(df)} candles"}

        df = self.calculate_indicators(df, symbol)
        # Critical: dropna *after* all indicators are calculated and assigned
        df.dropna(inplace=True) 
        
        if len(df) < cfg_bt.get("backtest_min_data_after_indicators", 50):
            return {"symbol": symbol, "error": f"Insufficient data after indicators & dropna: {len(df)} rows"}

        df = self.generate_signals(df)
        
        trades, position, equity_curve = [], None, []
        current_equity = 10000.0 # Initial equity
        risk_per_trade = cfg_bt.get("risk_per_trade", 0.015)
        sl_atr_mult = cfg_bt.get("stop_loss_atr_multiplier", 2.0)
        tp_atr_mult = cfg_bt.get("take_profit_atr_multiplier", 3.0)

        for i in range(len(df)): # Main backtesting loop
            row = df.iloc[i]
            # Entry logic
            if position is None:
                atr_val = row.get('atr') # Already calculated
                if pd.notna(atr_val) and atr_val > 1e-9: # Ensure ATR is valid
                    entry_price = row['close'] # Assuming entry at close of signal candle
                    if row.get('long_signal', False):
                        sl = entry_price - (sl_atr_mult * atr_val)
                        tp = entry_price + (tp_atr_mult * atr_val)
                        pos_size = (current_equity * risk_per_trade) / (sl_atr_mult * atr_val)
                        position = {'type': 'long', 'entry_price': entry_price, 'entry_time': row.name, 
                                    'stop_loss': sl, 'take_profit': tp, 'atr_at_entry': atr_val, 'position_size': pos_size}
                    elif row.get('short_signal', False):
                        sl = entry_price + (sl_atr_mult * atr_val)
                        tp = entry_price - (tp_atr_mult * atr_val)
                        pos_size = (current_equity * risk_per_trade) / (sl_atr_mult * atr_val)
                        position = {'type': 'short', 'entry_price': entry_price, 'entry_time': row.name,
                                    'stop_loss': sl, 'take_profit': tp, 'atr_at_entry': atr_val, 'position_size': pos_size}
            # Exit logic
            elif position is not None:
                exit_trade, exit_reason = False, ""
                current_price = row['close'] # Exit check at close of current candle
                if position['type'] == 'long':
                    if current_price <= position['stop_loss']: exit_trade, exit_reason = True, "Stop Loss"
                    elif current_price >= position['take_profit']: exit_trade, exit_reason = True, "Take Profit"
                    elif row.get('long_exit', False): exit_trade, exit_reason = True, "Exit Signal"
                elif position['type'] == 'short':
                    if current_price >= position['stop_loss']: exit_trade, exit_reason = True, "Stop Loss"
                    elif current_price <= position['take_profit']: exit_trade, exit_reason = True, "Take Profit"
                    elif row.get('short_exit', False): exit_trade, exit_reason = True, "Exit Signal"
                
                if exit_trade:
                    pnl_pips = (current_price - position['entry_price']) if position['type'] == 'long' else (position['entry_price'] - current_price)
                    pnl_r = pnl_pips / (sl_atr_mult * position['atr_at_entry']) if position['atr_at_entry'] > 1e-9 else 0.0
                    pnl_dollar = pnl_pips * position['position_size']
                    current_equity += pnl_dollar
                    current_equity = max(0, current_equity) # Prevent negative equity

                    trades.append({'symbol': symbol, 'type': position['type'], 'entry_time': position['entry_time'], 
                                   'exit_time': row.name, 'entry_price': position['entry_price'], 'exit_price': current_price,
                                   'pnl_pips': pnl_pips, 'pnl_r': pnl_r, 'pnl_dollar': pnl_dollar, 
                                   'exit_reason': exit_reason, 'atr_at_entry': position['atr_at_entry'], 'equity_after_trade': current_equity})
                    position = None
            equity_curve.append({'timestamp': row.name, 'equity': current_equity, 'in_position': position is not None})
        
        # Package results
        if not trades:
            # print(f"WORKER [{os.getpid()}] No trades for {symbol}.") # Worker print
            return {'symbol': symbol, 'total_trades': 0, 'final_equity': current_equity, 
                    'equity_curve': equity_curve, 'trades': [], 'error': 'No trades generated'} # Add empty trades list

        trades_df = pd.DataFrame(trades)
        # Ensure timestamps are timezone-naive for calculations if they come from different sources or are mixed
        if not trades_df.empty:
            trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time']).dt.tz_localize(None)
            trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time']).dt.tz_localize(None)
        
        if not equity_curve: # Should not happen if loop runs, but defensive
             first_timestamp = df.index[0] if not df.empty else pd.Timestamp.now().normalize()
             equity_curve.append({'timestamp': first_timestamp, 'equity': 10000.0, 'in_position': False})


        total_trades = len(trades_df)
        wins = trades_df[trades_df['pnl_r'] > 0]
        losses = trades_df[trades_df['pnl_r'] < 0]
        win_rate = len(wins) / total_trades if total_trades > 0 else 0.0
        avg_win_r = wins['pnl_r'].mean() if not wins.empty else 0.0
        avg_loss_r = losses['pnl_r'].mean() if not losses.empty else 0.0
        
        sum_profit_r = wins['pnl_r'].sum()
        sum_loss_r_abs = abs(losses['pnl_r'].sum())
        profit_factor = sum_profit_r / sum_loss_r_abs if sum_loss_r_abs > 1e-9 else (float('inf') if sum_profit_r > 1e-9 else 0.0)

        total_return_r = trades_df['pnl_r'].sum()
        total_return_pct = (current_equity / 10000.0 - 1.0) * 100.0

        # print(f"WORKER [{os.getpid()}] Finished {symbol} with {total_trades} trades.") # Worker print
        return {
            'symbol': symbol, 'total_trades': total_trades, 'win_rate': win_rate,
            'avg_win_r': avg_win_r, 'avg_loss_r': avg_loss_r, 'profit_factor': profit_factor,
            'total_return_r': total_return_r, 'total_return_pct': total_return_pct,
            'max_consecutive_losses': self._calculate_max_consecutive_losses(trades_df),
            'max_drawdown_pct': self._calculate_max_drawdown(equity_curve),
            'sharpe_ratio': self._calculate_sharpe_ratio(trades_df, equity_curve),
            'sortino_ratio': self._calculate_sortino_ratio(trades_df, equity_curve),
            'var_95_r': np.percentile(trades_df['pnl_r'], 5) if total_trades > 0 else 0.0,
            'max_loss_r': trades_df['pnl_r'].min() if total_trades > 0 else 0.0,
            'final_equity': current_equity, 
            'trades': trades_df.to_dict('records'), # Convert DF to list of dicts for pickling/JSON
            'equity_curve': equity_curve # List of dicts, already picklable
        }

    def _calculate_max_consecutive_losses(self, trades_df: pd.DataFrame) -> int: # Same
        if trades_df.empty: return 0
        consecutive, max_c = 0, 0
        for r_val in trades_df['pnl_r']:
            consecutive = consecutive + 1 if r_val < 0 else 0
            max_c = max(max_c, consecutive)
        return max_c

    def _calculate_max_drawdown(self, equity_curve_data: List[Dict]) -> float: # Same
        if not equity_curve_data: return 0.0
        equity_values = [p['equity'] for p in equity_curve_data]
        if not equity_values: return 0.0
        equity = pd.Series(equity_values)
        peak = equity.expanding(min_periods=1).max()
        # Ensure peak is not zero to avoid division by zero if equity curve starts at 0 or hits 0
        # This case should be rare with initial_equity > 0.
        safe_peak = peak.replace(0, np.nan) # Replace 0s with NaN to avoid division by zero if equity hits 0
        drawdown = (equity - safe_peak) / safe_peak
        max_dd_val = drawdown.min()
        return abs(max_dd_val * 100.0) if pd.notna(max_dd_val) else 0.0
    
    def _calculate_annualization_factor(self, trades_df: pd.DataFrame, equity_curve_data: List[Dict]) -> float: # Same
        if trades_df.empty or len(trades_df) < 2 or not equity_curve_data or len(equity_curve_data) < 2:
            return 1.0 

        # Timestamps from equity_curve should be more reliable for overall duration
        start_time_ec = pd.to_datetime(equity_curve_data[0]['timestamp']).tz_localize(None)
        end_time_ec = pd.to_datetime(equity_curve_data[-1]['timestamp']).tz_localize(None)
        
        duration_days = (end_time_ec - start_time_ec).total_seconds() / (24 * 3600.0)
        if duration_days < 1.0: duration_days = 1.0 
        
        num_trades = len(trades_df)
        trades_per_year = (num_trades / duration_days) * 252.0 # Trading days in a year
        
        return np.sqrt(trades_per_year) if trades_per_year > 0 else 1.0

    def _calculate_sharpe_ratio(self, trades_df: pd.DataFrame, equity_curve_data: List[Dict], risk_free_rate_annual_pct: float = 0.0) -> float: # Updated
        if trades_df.empty or trades_df['pnl_r'].isnull().all() or len(trades_df['pnl_r'].dropna()) < 2: return 0.0
        
        trade_returns_r = trades_df['pnl_r'].dropna() # Use R-multiples
        mean_return_r = trade_returns_r.mean()
        std_return_r = trade_returns_r.std()

        if std_return_r == 0 or pd.isna(std_return_r):
            return float('inf') if mean_return_r > 0 else (0.0 if mean_return_r == 0 else float('-inf'))
            
        # Risk-free rate for R-multiples is typically 0 unless R is defined against a benchmark
        # The annual risk_free_rate_annual_pct is not directly comparable to R-multiples per trade.
        # For simplicity, assume risk-free R-multiple per trade = 0.
        
        sharpe_per_trade = mean_return_r / std_return_r
        annual_factor = self._calculate_annualization_factor(trades_df, equity_curve_data)
        return sharpe_per_trade * annual_factor

    def _calculate_sortino_ratio(self, trades_df: pd.DataFrame, equity_curve_data: List[Dict], target_return_r_per_trade: float = 0.0) -> float: # Updated
        if trades_df.empty or trades_df['pnl_r'].isnull().all() or len(trades_df['pnl_r'].dropna()) < 2: return 0.0
        
        trade_returns_r = trades_df['pnl_r'].dropna()
        mean_return_r = trade_returns_r.mean()
        
        downside_diff_r = target_return_r_per_trade - trade_returns_r
        downside_returns_sq_r = downside_diff_r[downside_diff_r > 0]**2 # Only consider actual downside deviations squared
        
        if downside_returns_sq_r.empty: # No returns below target
            return float('inf') if mean_return_r > target_return_r_per_trade else 0.0
            
        expected_downside_deviation_r = np.sqrt(downside_returns_sq_r.mean()) # sqrt of mean of squared downside deviations
        
        if expected_downside_deviation_r == 0 or pd.isna(expected_downside_deviation_r):
            return float('inf') if mean_return_r > target_return_r_per_trade else 0.0
            
        sortino_per_trade = (mean_return_r - target_return_r_per_trade) / expected_downside_deviation_r
        annual_factor = self._calculate_annualization_factor(trades_df, equity_curve_data)
        return sortino_per_trade * annual_factor

    def scan_pairs(self, symbols: List[str], save_results: bool = True, max_workers: Optional[int] = None) -> pd.DataFrame:
        logger.info(f"Starting backtest scan on {len(symbols)} pairs using up to {max_workers or os.cpu_count()} workers...")
        results_list, failed_pairs_map = [], {} # Store errors for failed pairs

        api_cfg_dict = {"api_key": self.api.api_key, "secret_key": self.api.secret_key, 
                        "passphrase": self.api.passphrase, "sandbox": False} 
        
        # Path to strategy_config.json must be absolute or relative to where worker runs
        # Assuming it's in the same dir or accessible via relative path from worker's CWD
        strategy_cfg_path = str(self.config.config_path.resolve())


        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_backtest_worker_process, api_cfg_dict, strategy_cfg_path, symbol): symbol for symbol in symbols}
            
            for i, future in enumerate(as_completed(futures)):
                symbol = futures[future]
                try:
                    result = future.result()
                    # Main process logger
                    logger.info(f"Main: Completed processing for {symbol} ({i+1}/{len(symbols)})")
                    
                    min_trades = self.config.get("backtest_min_trades_for_ranking", 3)
                    if 'error' not in result or (result.get('error') == 'No trades generated' and result.get('total_trades',0) == 0) : # Treat "No trades" as a valid outcome, not an error for appending
                        if result.get('total_trades', 0) >= min_trades:
                             # Result already contains metrics, and 'trades' is list of dicts
                            score_input_result = result.copy() # Use the full result for scoring
                            # The trades are already dicts, no need to create 'trades_df_for_scoring' here
                            # just ensure _calculate_score can handle this format if it needs the DataFrame.
                            # For now, _calculate_score uses the metrics directly.
                            results_list.append({
                                'symbol': result['symbol'], 'total_trades': result.get('total_trades',0),
                                'win_rate': result.get('win_rate',0), 'profit_factor': result.get('profit_factor',0),
                                'total_return_r': result.get('total_return_r',0), 'total_return_pct': result.get('total_return_pct',0),
                                'max_drawdown_pct': result.get('max_drawdown_pct',0), 
                                'sharpe_ratio': result.get('sharpe_ratio',0), 'sortino_ratio': result.get('sortino_ratio',0),
                                'max_consecutive_losses': result.get('max_consecutive_losses',0),
                                'var_95_r': result.get('var_95_r',0), 'max_loss_r': result.get('max_loss_r',0),
                                'score': self._calculate_score(score_input_result) 
                            })
                        elif result.get('error') == 'No trades generated': # Not enough trades but no other error
                            failed_pairs_map[symbol] = f"Not enough trades: {result.get('total_trades',0)}"
                            logger.info(f"[{symbol}] Skipped ranking due to insufficient trades: {result.get('total_trades',0)}.")
                        else: # Other type of error from result dict (e.g. insufficient data)
                            failed_pairs_map[symbol] = result.get('error', 'Unknown processing error')
                            logger.warning(f"[{symbol}] Failed/Skipped: {failed_pairs_map[symbol]}")
                    else: # Hard error from worker or explicit error field
                        failed_pairs_map[symbol] = result.get('error', 'Unknown worker error')
                        logger.warning(f"[{symbol}] Failed/Skipped (explicit error): {failed_pairs_map[symbol]}")

                except Exception as e_future: # Exception from future.result() itself
                    logger.error(f"Exception retrieving result for {symbol}: {e_future}", exc_info=True)
                    failed_pairs_map[symbol] = f"Future processing error: {e_future}"
        
        logger.info(f"Scan complete. Successfully considered for ranking: {len(results_list)}, Failed/Skipped: {len(failed_pairs_map)}")
        if results_list:
            df_final_results = pd.DataFrame(results_list).sort_values('score', ascending=False)
            if save_results: self._save_scan_artifacts(df_final_results, symbols, failed_pairs_map)
            return df_final_results
        else:
            logger.warning("No pairs met ranking criteria or all failed.")
            if save_results and symbols: self._save_scan_artifacts(pd.DataFrame(), symbols, failed_pairs_map)
            return pd.DataFrame()

    def _save_scan_artifacts(self, df_ranked: pd.DataFrame, symbols_scanned: List[str], failed_pairs_info: Dict[str, str]):
        # ... (artifact saving, use df_ranked and failed_pairs_info) ...
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        Path("results").mkdir(parents=True, exist_ok=True)
        if not df_ranked.empty:
            results_file = Path(f"results/scan_results_{timestamp}.csv")
            df_ranked.to_csv(results_file, index=False)
            logger.info(f"Ranked scan results saved to {results_file}")
        
        summary_file = Path(f"results/scan_summary_{timestamp}.txt")
        with open(summary_file, 'w') as f:
            f.write(f"NNFX Bot - Scan Summary - {timestamp}\n" + "="*60 + "\n")
            f.write(f"Total pairs attempted: {len(symbols_scanned)}\n")
            f.write(f"Successfully ranked pairs: {len(df_ranked)}\n")
            f.write(f"Failed or skipped pairs: {len(failed_pairs_info)}\n\n")
            if not df_ranked.empty:
                f.write("TOP PERFORMING PAIRS (max 10 or all if fewer)\n" + "-"*40 + "\n")
                for _, row in df_ranked.head(min(10, len(df_ranked))).iterrows():
                    f.write(f"{row['symbol']:<12} Score: {row.get('score',0):.2f} | WR: {row.get('win_rate',0):.1%} | PF: {row.get('profit_factor',0):.2f} | Ret%: {row.get('total_return_pct',0):.1f}% | Trades: {row.get('total_trades',0)}\n")
            if failed_pairs_info:
                f.write(f"\nFAILED/SKIPPED PAIRS ({len(failed_pairs_info)}):\n")
                for sym, reason in failed_pairs_info.items():
                    f.write(f"  - {sym}: {reason}\n")
        logger.info(f"Scan summary saved to {summary_file}")


    def _calculate_score(self, result_dict: Dict) -> float: # Same
        cfg_score = self.config.get("scoring", {})
        min_trades_for_score = self.config.get("backtest_min_trades_for_ranking", 3)
        if result_dict.get('total_trades', 0) < min_trades_for_score : return -999.0 

        win_rate = result_dict.get('win_rate', 0.0)
        profit_factor = result_dict.get('profit_factor', 0.0)
        return_pct = result_dict.get('total_return_pct', 0.0)
        total_trades = result_dict.get('total_trades', 0)
        sharpe = result_dict.get('sharpe_ratio', 0.0)
        max_dd = result_dict.get('max_drawdown_pct', 100.0) # Default to high DD if missing
        max_con_loss = result_dict.get('max_consecutive_losses', 20) # Default high
        var_95 = result_dict.get('var_95_r', -5.0) 

        win_rate_s = win_rate * 100.0
        pf_s = min(profit_factor * 20.0, 100.0) if np.isfinite(profit_factor) else (cfg_score.get("profit_factor_inf_score", 50) if profit_factor > 0 else 0.0)
        ret_s = min(max(return_pct, -100.0), 100.0) 
        trades_s = min(total_trades * 0.5, 100.0) 
        sharpe_s = min(max(sharpe * 25.0, -100.0), 100.0) if np.isfinite(sharpe) else (cfg_score.get("sharpe_ratio_inf_score", 50) if sharpe > 0 else -50.0)

        dd_p = max_dd * cfg_score.get("drawdown_penalty_multiplier", 1.5)
        con_loss_p = max_con_loss * cfg_score.get("consecutive_loss_penalty_multiplier", 3.0)
        var_p = abs(var_95) * cfg_score.get("var_95_penalty_multiplier", 5.0)

        score = (
            win_rate_s * cfg_score.get("win_rate_weight", 0.25) +
            pf_s       * cfg_score.get("profit_factor_weight", 0.25) +
            ret_s      * cfg_score.get("return_pct_weight", 0.20) +
            trades_s   * cfg_score.get("trade_frequency_weight", 0.10) +
            sharpe_s   * cfg_score.get("sharpe_ratio_weight", 0.20) -
            (dd_p       * cfg_score.get("drawdown_penalty_weight", 0.05)) - # Ensure penalties are subtracted
            (con_loss_p * cfg_score.get("consecutive_loss_penalty_weight", 0.03)) -
            (var_p      * cfg_score.get("var_95_penalty_weight", 0.02))
        )
        return round(max(score, -1000.0), 2)


    def get_current_signals(self, symbols: List[str]) -> pd.DataFrame: # Mostly same, use config
        logger.info(f"Getting current signals for {len(symbols)} pairs...")
        signals_data = []
        cfg_sig_conf = self.config.get("signal_confidence", {})
        kline_limit_for_signals = 200 # Or from config

        for symbol in symbols: # Can be parallelized too if performance becomes an issue
            try:
                df = self.api.get_klines(symbol, "4H", kline_limit_for_signals) 
                if df.empty or len(df) < self.config.get("backtest_min_data_after_indicators", 50):
                    logger.warning(f"[{symbol}] Insufficient data for current signals ({len(df) if not df.empty else 0} rows).")
                    continue

                df = self.calculate_indicators(df, symbol)
                df.dropna(inplace=True)
                if df.empty or len(df) < 2: 
                    logger.warning(f"[{symbol}] Data became empty after indicators/dropna for current signals.")
                    continue
                df = self.generate_signals(df)
                
                latest, current = df.iloc[-2], df.iloc[-1] # Latest closed, current forming
                signal_type, confidence = "NONE", 0.0
                
                if latest.get('long_signal', False):
                    signal_type = "LONG"
                    confidence = self._calculate_signal_confidence(df.iloc[-cfg_sig_conf.get("min_recent_data_for_confidence",3)-1:-1], 'long', cfg_sig_conf) # Use -1 to exclude current forming candle from confidence history
                elif latest.get('short_signal', False):
                    signal_type = "SHORT"
                    confidence = self._calculate_signal_confidence(df.iloc[-cfg_sig_conf.get("min_recent_data_for_confidence",3)-1:-1], 'short', cfg_sig_conf)
                
                current_price = latest.get('close', np.nan)
                atr_val = latest.get('atr', np.nan)
                sl_mult = self.config.get("stop_loss_atr_multiplier", 2.0)
                tp_mult = self.config.get("take_profit_atr_multiplier", 3.0)

                stop_dist = sl_mult * atr_val if pd.notna(atr_val) else np.nan
                tp_dist = tp_mult * atr_val if pd.notna(atr_val) else np.nan
                
                sl_price, tp_price = np.nan, np.nan
                if signal_type == "LONG" and pd.notna(current_price) and pd.notna(stop_dist):
                    sl_price = current_price - stop_dist
                    tp_price = current_price + tp_dist
                elif signal_type == "SHORT" and pd.notna(current_price) and pd.notna(stop_dist):
                    sl_price = current_price + stop_dist
                    tp_price = current_price - tp_dist

                signals_data.append({
                    'symbol': symbol, 'signal': signal_type, 'confidence': round(confidence, 3),
                    'price_at_signal': current_price, 'atr_at_signal': atr_val,
                    'stop_loss_price': sl_price, 'take_profit_price': tp_price,
                    'risk_reward_ratio': tp_dist / stop_dist if pd.notna(stop_dist) and stop_dist > 1e-9 else 0.0,
                    'signal_timestamp': latest.name, 'current_candle_forming_time': current.name
                })
                time.sleep(0.1) # Small delay
            except Exception as e:
                logger.error(f"Error getting signals for {symbol}: {e}", exc_info=False) # Keep log concise
        
        df_out = pd.DataFrame(signals_data)
        return df_out.sort_values(['signal', 'confidence'], ascending=[True, False]) if not df_out.empty else pd.DataFrame()


    def _calculate_signal_confidence(self, recent_df_closed_candles: pd.DataFrame, signal_type: str, cfg_confidence: Dict) -> float: # Same
        min_hist = cfg_confidence.get("min_recent_data_for_confidence",3)
        if len(recent_df_closed_candles) < min_hist: return cfg_confidence.get("default_confidence", 0.5)
        
        # Weights from config
        sup_w = cfg_confidence.get("signal_support_weight",0.4)
        trend_w = cfg_confidence.get("trend_consistency_weight",0.25)
        mom_w = cfg_confidence.get("momentum_consistency_weight",0.20)
        vol_w = cfg_confidence.get("volume_consistency_weight",0.15)

        # Determine positive/negative condition based on signal_type
        cond_map = {'long': 1, 'short': -1}
        direction = cond_map[signal_type]
            
        signal_col = 'long_signal' if signal_type == 'long' else 'short_signal'
        
        signal_support = recent_df_closed_candles.get(signal_col, pd.Series(False, index=recent_df_closed_candles.index)).mean()
        trend_consistency = (recent_df_closed_candles.get('system_a_baseline', pd.Series(0)) == direction).mean()
        momentum_consistency = (recent_df_closed_candles.get('system_a_confirmation', pd.Series(0)) == direction).mean()
        volume_consistency = (recent_df_closed_candles.get('system_a_volume', pd.Series(0)) == direction).mean()
        
        confidence = (signal_support * sup_w + trend_consistency * trend_w + 
                      momentum_consistency * mom_w + volume_consistency * vol_w)
        return min(max(confidence, 0.0), 1.0) # Clamp to [0,1]

    def export_detailed_results(self, rankings_df: pd.DataFrame, current_signals_df: pd.DataFrame) -> str: # Same
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = Path("results")
        results_dir.mkdir(parents=True, exist_ok=True)
        filename = results_dir / f"nnfx_analysis_{timestamp}.xlsx"
        
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Limit rows for Excel to prevent huge files if many pairs are scanned
                rankings_df.head(min(len(rankings_df), 200)).to_excel(writer, sheet_name='Backtest Rankings', index=False)
                current_signals_df.to_excel(writer, sheet_name='Current Signals', index=False)
                
                if not rankings_df.empty:
                    # Add a summary sheet if desired
                    summary_data = {
                        "Metric": ["Total Pairs Ranked", "Avg. Win Rate", "Avg. Profit Factor"],
                        "Value": [len(rankings_df), f"{rankings_df['win_rate'].mean():.1%}", f"{rankings_df[np.isfinite(rankings_df['profit_factor'])]['profit_factor'].mean():.2f}"]
                    }
                    pd.DataFrame(summary_data).to_excel(writer, sheet_name="Quick Summary", index=False)

            logger.info(f"Detailed results exported to {filename}")
            return str(filename)
        except Exception as e:
            logger.error(f"Error exporting results to Excel: {e}", exc_info=True)
            return ""

def run_comprehensive_analysis(api_config_dict: Dict, main_strategy_config: StrategyConfig, 
                               test_symbols_override: Optional[List[str]] = None,
                               max_workers: Optional[int] = None) -> Dict:
    logger.info("=" * 60 + "\nNNFX Bot - Comprehensive Analysis Started\n" + "=" * 60)
    start_time_total = time.time()
    
    api = BitgetAPI(**api_config_dict)
    system = DualNNFXSystem(api, main_strategy_config) 
    
    test_symbols = []
    if test_symbols_override:
        # Limit override list by config if it's longer than top_n, or just use it as is
        # top_n_config = main_strategy_config.get("top_n_pairs_by_volume_to_scan", 8)
        # test_symbols = test_symbols_override[:top_n_config]
        test_symbols = test_symbols_override # Use the full override list
        logger.info(f"Using provided override list of {len(test_symbols)} symbols.")
    else:
        logger.info("Fetching all tickers to determine top N by volume...")
        all_tickers = api.get_all_tickers_data()
        if all_tickers:
            usdt_tickers = [t for t in all_tickers if t.get('symbolId','').upper().endswith('USDT')]
            logger.debug(f"Found {len(usdt_tickers)} USDT tickers initially.")

            if main_strategy_config.get("filter_by_major_bases_for_top_volume", True):
                major_bases = set(main_strategy_config.get("major_bases_for_filtering", []))
                logger.debug(f"Filtering by {len(major_bases)} major bases: {sorted(list(major_bases))[:10]}...")
                
                filtered_usdt_tickers = []
                for t in usdt_tickers:
                    symbol_id_upper = t.get('symbolId', '').upper()
                    if symbol_id_upper.endswith('USDT'):
                        base_currency = symbol_id_upper[:-4] # Remove 'USDT'
                        if base_currency in major_bases:
                            filtered_usdt_tickers.append(t)
                usdt_tickers = filtered_usdt_tickers
                logger.debug(f"Found {len(usdt_tickers)} USDT tickers after major base filtering.")

            def get_usdt_vol(ticker_dict_item): # Renamed to avoid conflict
                vol_str = ticker_dict_item.get('usdtVol', '0') # Bitget uses 'usdtVol' in tickers
                try: return float(vol_str)
                except (ValueError, TypeError): return 0.0
            
            usdt_tickers.sort(key=get_usdt_vol, reverse=True)
            logger.debug(f"Top 5 USDT tickers by volume (before slicing): "
                         f"{[{'s': t.get('symbolId'), 'v': t.get('usdtVol')} for t in usdt_tickers[:5]]}")
            
            top_n = main_strategy_config.get("top_n_pairs_by_volume_to_scan", 8)
            test_symbols = [t['symbolId'].replace('_SPBL', '') for t in usdt_tickers[:top_n]] 
            logger.info(f"Selected Top {len(test_symbols)} symbols by USDT volume for scan: {test_symbols}")
        else:
            logger.error("Could not fetch tickers for dynamic selection. Using hardcoded fallback.")
            test_symbols = ['BTCUSDT', 'ETHUSDT'] 

    if not test_symbols:
        logger.error("No symbols selected for scanning. Aborting analysis.")
        return {'success': False, 'error': 'No symbols to scan', 'scan_duration_seconds': 0, 'total_duration_seconds': time.time()-start_time_total}

    logger.info(f"Starting scan for {len(test_symbols)} symbols: {test_symbols}")
    scan_start_time = time.time()
    rankings_df = system.scan_pairs(test_symbols, save_results=True, max_workers=max_workers)
    scan_duration = time.time() - scan_start_time
    logger.info(f"Pair scan completed in {scan_duration:.2f} seconds.")

    analysis_result_payload = {'success': False, 'scan_duration_seconds': round(scan_duration,2)}

    if not rankings_df.empty:
        analysis_result_payload['success'] = True
        # Provide a brief summary of rankings in the return payload
        analysis_result_payload['rankings_top_5_summary'] = rankings_df[['symbol', 'score', 'win_rate', 'total_return_pct', 'total_trades']].head().to_dict('records')
        
        top_ranked_symbols_for_signals = rankings_df['symbol'].head(5).tolist()
        if top_ranked_symbols_for_signals:
            logger.info(f"Getting current signals for top ranked: {top_ranked_symbols_for_signals}")
            current_signals_df = system.get_current_signals(top_ranked_symbols_for_signals)
            analysis_result_payload['current_signals_active'] = current_signals_df[current_signals_df['signal'] != 'NONE'].to_dict('records')
            export_file = system.export_detailed_results(rankings_df, current_signals_df)
            analysis_result_payload['export_file'] = export_file
        else:
            logger.info("No pairs from ranking to get current signals for (e.g. all failed ranking).")
            analysis_result_payload['current_signals_active'] = []
            export_file = system.export_detailed_results(rankings_df, pd.DataFrame()) # Export rankings only
            analysis_result_payload['export_file'] = export_file

        logger.info(f"Top performers from scan ({len(rankings_df)} ranked):")
        for _, row in rankings_df.head().iterrows(): logger.info(f"  {row['symbol']}: Score {row.get('score',0):.2f}, WR {row.get('win_rate',0):.1%}")
    else:
        logger.warning("No pairs were successfully ranked. Check logs from workers for individual pair errors.")
        analysis_result_payload['error'] = "No pairs successfully ranked or met criteria."
        # Save empty/minimal report structure
        system.export_detailed_results(pd.DataFrame(), pd.DataFrame()) 

    total_duration = time.time() - start_time_total
    logger.info(f"Comprehensive analysis finished in {total_duration:.2f} seconds.")
    analysis_result_payload['total_duration_seconds'] = round(total_duration,2)
    return analysis_result_payload


def cleanup_old_files(): # Same
    cfg_cleanup = strategy_config_global.get("cleanup", {}) # Use global config instance
    days_old = cfg_cleanup.get("cache_days_old", 7)
    max_results_logs = cfg_cleanup.get("max_results_to_keep", 20)

    cutoff_ts = (datetime.now() - timedelta(days=days_old)).timestamp()
    cleaned_count = 0
    
    for dir_name, is_time_sorted_cleanup in [("data", False), ("results", True), ("logs", True)]:
        d_path = Path(dir_name)
        d_path.mkdir(parents=True, exist_ok=True)
        
        # Glob all relevant file types, could be more specific if needed
        files_in_dir = [f for f_type in ["*.csv", "*.json", "*.txt", "*.xlsx", "*.log"] for f in d_path.glob(f_type)]

        to_delete_list = []
        if is_time_sorted_cleanup: # For results and logs, keep the newest N
            files_in_dir.sort(key=lambda x: x.stat().st_mtime) # Sort by modification time, oldest first
            if len(files_in_dir) > max_results_logs:
                to_delete_list = files_in_dir[:-max_results_logs] # Keep the last N
        else: # For 'data' (cache), delete if older than cutoff_ts
            to_delete_list = [f for f in files_in_dir if f.stat().st_mtime < cutoff_ts]

        for old_f in to_delete_list:
            try:
                old_f.unlink()
                logger.debug(f"Cleaned old file: {old_f}")
                cleaned_count += 1
            except Exception as e_clean:
                logger.warning(f"Could not remove old file {old_f}: {e_clean}")
    logger.info(f"Cleanup completed. Removed {cleaned_count} old files.")

if __name__ == "__main__":
    run_ts_main = datetime.now().strftime('%Y%m%d_%H%M%S')
    main_log_file_handler = None # Initialize to ensure it's defined for finally block
    
    # --- Global strategy_config_global is already initialized ---
    # No need to re-initialize it here unless you want a different config for a specific run.

    try:
        # --- Setup File Logging for this specific run ---
        logs_dir_main = Path("logs")
        logs_dir_main.mkdir(parents=True, exist_ok=True)
        log_file_path_main = logs_dir_main / f"system_run_{run_ts_main}.log"
        
        main_log_file_handler = logging.FileHandler(log_file_path_main)
        main_log_file_handler.setLevel(logging.INFO) 
        main_log_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(main_log_file_handler)
        
        # Adjust console handler level if desired (basicConfig sets root level)
        # For example, to make console less verbose than file:
        for h in logging.getLogger().handlers:
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler): # Target console StreamHandler
                h.setLevel(logging.INFO) # Or logging.WARNING

        logger.info(f"NNFX Bot System Run ID: {run_ts_main}")
        logger.info(f"Using Strategy Config: {strategy_config_global.config_path}")
        logger.info(f"File logging (INFO+) to: {log_file_path_main}")
        logger.info(f"Console logging (INFO+) active.") # Or (DEBUG+) if basicConfig level is kept for console

        # --- Load API Config ---
        api_credentials = {}
        api_credentials_path = Path("config/api_config.json")
        if api_credentials_path.exists():
            try:
                with open(api_credentials_path, "r") as f_api: api_credentials = json.load(f_api)
                logger.info("API credentials loaded.")
            except json.JSONDecodeError as e_json_api:
                logger.error(f"Error decoding API credentials from {api_credentials_path}: {e_json_api}. Proceeding without API keys.")
            except Exception as e_load_api:
                logger.error(f"Error loading API credentials {api_credentials_path}: {e_load_api}. Proceeding without API keys.")
        else:
            logger.warning(f"API credentials file {api_credentials_path} not found. Creating dummy. Bot will use public API access only.")
            try:
                api_credentials_path.parent.mkdir(parents=True, exist_ok=True)
                with open(api_credentials_path, "w") as f_dummy_api:
                    json.dump({"api_key": "YOUR_API_KEY_HERE", 
                               "secret_key": "YOUR_SECRET_KEY_HERE", 
                               "passphrase": "YOUR_PASSPHRASE_IF_NEEDED", # Specific to some exchanges like KuCoin, Bitget might not need
                               "sandbox": True}, f_dummy_api, indent=4)
                logger.info(f"Created dummy API credentials at {api_credentials_path}. Please update it.")
            except Exception as e_create_dummy_api:
                logger.error(f"Could not create dummy api_config.json: {e_create_dummy_api}")
        
        # --- Run Analysis ---
        # To use a specific list of symbols for testing, pass it to `test_symbols_override`.
        # e.g., test_symbols_override=['BTCUSDT', 'ETHUSDT']
        # Setting to None will trigger dynamic fetching of top N by volume based on strategy_config.
        override_symbols = None 
        # override_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT'] # Example override

        # Determine number of workers for parallel processing
        # Leave at least one core for the OS and main process if possible
        cpu_cores = os.cpu_count()
        workers_to_use = max(1, cpu_cores - 1) if cpu_cores and cpu_cores > 1 else 1
        logger.info(f"Will use up to {workers_to_use} worker processes for scanning.")

        analysis_run_output = run_comprehensive_analysis(
            api_credentials, 
            strategy_config_global, # Pass the global StrategyConfig instance
            test_symbols_override=override_symbols,
            max_workers=workers_to_use 
        )
        
        if analysis_run_output.get('success'):
            logger.info("Main: Comprehensive analysis completed successfully.")
            if analysis_run_output.get('export_file'): logger.info(f"Main: Report available at: {analysis_run_output['export_file']}")
        else:
            logger.error(f"Main: Comprehensive analysis FAILED: {analysis_run_output.get('error', 'Unknown error')}")

    except Exception as e_main_block:
        logger.critical("Critical unhandled error in __main__ execution block:", exc_info=True)
    finally:
        logger.info("Performing final cleanup of old files...")
        cleanup_old_files()
        logger.info(f"Session {run_ts_main} complete. Review logs for details.")
        if main_log_file_handler: # Ensure handler exists before trying to remove/close
            logging.getLogger().removeHandler(main_log_file_handler)
            main_log_file_handler.close()