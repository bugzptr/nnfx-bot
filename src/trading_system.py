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
from functools import partial

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='ta')
warnings.filterwarnings('ignore', category=RuntimeWarning) # For potential mean of empty slice

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)] # Ensure logs go to stdout
)
logger = logging.getLogger(__name__)

# --- Configuration Loader ---
class StrategyConfig:
    def __init__(self, config_path="config/strategy_config.json"):
        self.config_path = Path(config_path)
        self.params = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            logger.error(f"Strategy config file not found: {self.config_path}")
            # Fallback to some very basic defaults if config is missing, though it's better to ensure file exists
            # This is primarily to prevent outright crashes if the file is missing during development.
            # In a production scenario, you might want to raise an error.
            return { # Provide minimal defaults to avoid AttributeError everywhere
                "major_bases_for_filtering": ["BTC", "ETH"],
                "filter_by_major_bases_for_top_volume": False,
                "top_n_pairs_by_volume_to_scan": 2,
                "backtest_kline_limit": 500,
                "backtest_min_data_after_get_klines": 50,
                "backtest_min_data_after_indicators": 30,
                "backtest_min_trades_for_ranking": 3,
                "risk_per_trade": 0.01,
                "stop_loss_atr_multiplier": 2.0,
                "take_profit_atr_multiplier": 3.0,
                "indicators": { "tema_period": 20, "cci_period": 20, "elder_fi_period": 13,
                                "chandelier_period": 20, "chandelier_multiplier": 3.0,
                                "kijun_sen_period": 20, "williams_r_period": 14,
                                "klinger_fast_ema": 30, "klinger_slow_ema": 50, "klinger_signal_ema": 13,
                                "psar_step": 0.02, "psar_max_step": 0.2,
                                "atr_period_risk": 14, "atr_period_chandelier": 20},
                "scoring": { "win_rate_weight": 0.2, "profit_factor_weight": 0.2, "return_pct_weight": 0.2,
                             "trade_frequency_weight": 0.1, "sharpe_ratio_weight": 0.2,
                             "drawdown_penalty_multiplier": 1.0, "drawdown_penalty_weight": 0.05,
                             "consecutive_loss_penalty_multiplier": 1.0, "consecutive_loss_penalty_weight": 0.03,
                             "var_95_penalty_multiplier": 1.0, "var_95_penalty_weight": 0.02,
                             "profit_factor_inf_score": 50, "sharpe_ratio_inf_score": 50},
                "signal_confidence": { "min_recent_data_for_confidence": 3, "default_confidence": 0.5,
                                       "signal_support_weight": 0.4, "trend_consistency_weight": 0.25,
                                       "momentum_consistency_weight": 0.2, "volume_consistency_weight": 0.15},
                "cleanup": {"cache_days_old": 7, "max_results_to_keep": 10}
            }
        try:
            with open(self.config_path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {self.config_path}: {e}")
            raise  # Reraise as this is a critical configuration error
        except Exception as e:
            logger.error(f"Error loading strategy config {self.config_path}: {e}")
            raise

    def get(self, key_path: str, default: Any = None) -> Any:
        keys = key_path.split('.')
        value = self.params
        try:
            for key in keys:
                value = value[key]
            return value
        except KeyError:
            logger.warning(f"Config key '{key_path}' not found. Using default: {default}")
            return default
        except TypeError: # Handle case where an intermediate key is not a dict
             logger.warning(f"Config path '{key_path}' implies nesting that doesn't exist. Using default: {default}")
             return default

# Global config instance (or pass it around)
# For simplicity in this refactor, it's global. In larger apps, dependency injection is preferred.
strategy_config = StrategyConfig()

class BitgetAPI:
    def __init__(self, api_key: str = "", secret_key: str = "", 
                 passphrase: str = "", sandbox: bool = True):
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase
        self.base_url = "https://api.bitget.com" # Consider sandbox URL toggle if Bitget supports it easily
        self.session = requests.Session()
        self.rate_limit_delay = 0.2  # 200ms between requests, per instance

        for directory in ["data", "results", "logs", "config"]:
            Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Bitget API initialized ({'Sandbox' if sandbox else 'Live'} mode - Note: URL is currently hardcoded to live)")

    def _generate_signature(self, timestamp: str, method: str, request_path: str, body: str = "") -> str:
        if not self.secret_key: return "" # No signature if no secret key
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
            "locale": "en-US" # Often good to specify locale
        }
        return {k: v for k, v in headers.items() if v} # Remove empty headers if keys are missing

    def get_klines(self, symbol: str, granularity: str = "4H", 
                   limit: int = 500) -> pd.DataFrame:
        api_symbol = symbol if symbol.endswith('_SPBL') else symbol + '_SPBL'
        cache_file = Path(f"data/{api_symbol}_{granularity.lower()}.csv")
        expected_cols_for_cache = ['open', 'high', 'low', 'close', 'volume']

        if cache_file.exists():
            cache_age = time.time() - cache_file.stat().st_mtime
            if cache_age < strategy_config.get("cleanup.cache_days_old", 7) * 86400 / 6: # Fresher cache for klines (e.g., 1/6th of general cleanup)
                try:
                    cached_df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                    if not cached_df.empty and all(col in cached_df.columns for col in expected_cols_for_cache):
                        if not (cached_df[expected_cols_for_cache].isnull().values.any() or \
                                np.isinf(cached_df[expected_cols_for_cache].values).any()):
                            logger.debug(f"Using valid cached kline data for {symbol}")
                            return cached_df
                        else:
                            logger.warning(f"Cached kline data for {symbol} contains NaN/inf. Refetching.")
                    else:
                        logger.warning(f"Cached kline data for {symbol} is empty or missing columns. Refetching.")
                except Exception as e:
                    logger.warning(f"Failed to read/validate kline cache for {symbol}: {e}. Refetching.")
        
        time.sleep(self.rate_limit_delay)
        url = f"{self.base_url}/api/spot/v1/market/candles"
        params = {"symbol": api_symbol, "period": granularity.lower(), "limit": str(limit)}
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, params=params, timeout=15, headers=self._get_headers("GET", "/api/spot/v1/market/candles", "")) # Add headers for public endpoints too if required by API
                logger.debug(f"Raw API response for {symbol} klines: {response.text[:200]}") # Log snippet
                response.raise_for_status()
                data = response.json()
                
                if str(data.get('code')) == '00000' and data.get('data'):
                    df = pd.DataFrame(data['data'], columns=['ts', 'open', 'high', 'low', 'close', 'baseVol', 'quoteVol']) # Explicit columns
                    df = df.rename(columns={'ts': 'timestamp', 'baseVol': 'volume'})
                    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(np.int64), unit='ms')
                    df = df.set_index('timestamp')
                    
                    for col in expected_cols_for_cache:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        else:
                            logger.error(f"Critical kline column '{col}' missing for {symbol}. API data: {data['data'][:1] if data['data'] else 'empty'}")
                            return pd.DataFrame()
                    
                    df = df[expected_cols_for_cache]
                    df = df.sort_index()
                    df = df.replace([np.inf, -np.inf], np.nan).dropna()
                    
                    if df.empty:
                        logger.warning(f"No valid kline data for {symbol} after processing/cleaning.")
                        return pd.DataFrame()
                        
                    df.to_csv(cache_file)
                    logger.debug(f"Fetched and cached kline data for {symbol}")
                    return df
                else:
                    logger.warning(f"API error for {symbol} klines: {data.get('msg', 'Unknown error code')} (Code: {data.get('code')})")
                    # Specific handling for common errors like delisted symbols
                    if str(data.get('code')) == '40309': # Symbol delisted
                        logger.error(f"Symbol {symbol} appears to be delisted. Skipping.")
                        return pd.DataFrame() # Return empty, don't retry for delisted.
                    if attempt < max_retries - 1: time.sleep(2 ** attempt)
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed for {symbol} klines, attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1: time.sleep(2 ** attempt)
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error for {symbol} klines response: {e}. Response text: {response.text[:200] if 'response' in locals() else 'N/A'}")
                if attempt < max_retries - 1: time.sleep(2 ** attempt)

        logger.error(f"Failed to fetch kline data for {symbol} after {max_retries} attempts")
        return pd.DataFrame()

    def get_all_tickers_data(self) -> List[Dict[str, Any]]:
        """Fetches data for all tickers, including volume."""
        url = f"{self.base_url}/api/spot/v1/market/tickers"
        try:
            time.sleep(self.rate_limit_delay) # Rate limit before call
            response = self.session.get(url, timeout=15, headers=self._get_headers("GET", "/api/spot/v1/market/tickers", ""))
            response.raise_for_status()
            data = response.json()
            if str(data.get('code')) == '00000' and isinstance(data.get('data'), list):
                logger.info(f"Successfully fetched data for {len(data['data'])} tickers.")
                return data['data']
            else:
                logger.error(f"Failed to fetch all tickers data: {data.get('msg', 'Unknown error')}")
                return []
        except requests.exceptions.RequestException as e:
            logger.error(f"Request exception while fetching all tickers: {e}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error while fetching all tickers: {e}. Response: {response.text[:200] if 'response' in locals() else 'N/A'}")
            return []


class NNFXIndicators:
    def __init__(self, config: StrategyConfig):
        self.config = config.get("indicators", {}) # Get the 'indicators' sub-dictionary

    def tema(self, data: pd.Series) -> pd.Series:
        period = self.config.get("tema_period", 21)
        try:
            ema1 = data.ewm(span=period, adjust=False).mean()
            ema2 = ema1.ewm(span=period, adjust=False).mean()
            ema3 = ema2.ewm(span=period, adjust=False).mean()
            return 3 * ema1 - 3 * ema2 + ema3
        except Exception as e:
            logger.error(f"Error calculating TEMA(period={period}): {e}")
            return pd.Series(np.nan, index=data.index, dtype=float)
    
    def kijun_sen(self, high: pd.Series, low: pd.Series) -> pd.Series:
        period = self.config.get("kijun_sen_period", 26)
        try:
            highest_high = high.rolling(window=period, min_periods=max(1, period // 2)).max() # Allow fewer min_periods
            lowest_low = low.rolling(window=period, min_periods=max(1, period // 2)).min()
            return (highest_high + lowest_low) / 2
        except Exception as e:
            logger.error(f"Error calculating Kijun-Sen(period={period}): {e}")
            return pd.Series(np.nan, index=high.index, dtype=float)
    
    def cci(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        period = self.config.get("cci_period", 14)
        try:
            return ta.trend.CCIIndicator(high=high, low=low, close=close, window=period, fillna=False).cci()
        except Exception as e:
            logger.error(f"Error calculating CCI(period={period}): {e}")
            return pd.Series(np.nan, index=high.index, dtype=float)

    def williams_r(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        period = self.config.get("williams_r_period", 14)
        try:
            return ta.momentum.WilliamsRIndicator(high=high, low=low, close=close, lbp=period, fillna=False).williams_r()
        except Exception as e:
            logger.error(f"Error calculating Williams %R(period={period}): {e}")
            return pd.Series(np.nan, index=high.index, dtype=float)

    def elder_force_index(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        period = self.config.get("elder_fi_period", 13)
        try:
            return ta.volume.ForceIndexIndicator(close=close, volume=volume, window=period, fillna=False).force_index()
        except Exception as e: # Catch specific ta errors if known, else general
            logger.error(f"Error calculating Elder's Force Index(period={period}): {e}")
            return pd.Series(np.nan, index=close.index, dtype=float)

    def klinger_oscillator(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> Tuple[pd.Series, pd.Series]:
        fast = self.config.get("klinger_fast_ema", 34)
        slow = self.config.get("klinger_slow_ema", 55)
        signal = self.config.get("klinger_signal_ema", 13)
        try:
            klinger_indicator = ta.volume.KlingerOscillator(
                high=high, low=low, close=close, volume=volume, 
                window_fast=fast, window_slow=slow, window_sign=signal, fillna=False
            )
            return klinger_indicator.klinger(), klinger_indicator.klinger_signal()
        except Exception as e:
            logger.error(f"Error calculating Klinger Oscillator(f={fast},s={slow},sig={signal}): {e}")
            empty_series = pd.Series(np.nan, index=high.index, dtype=float)
            return empty_series, empty_series
    
    def chandelier_exit(self, high: pd.Series, low: pd.Series, close: pd.Series) -> Tuple[pd.Series, pd.Series]:
        period = self.config.get("chandelier_period", 22)
        multiplier = self.config.get("chandelier_multiplier", 3.0)
        atr_period = self.config.get("atr_period_chandelier", period) # Use chandelier period for ATR by default
        try:
            atr = self.atr(high, low, close, period=atr_period) # Use own ATR for consistency
            
            highest_high = high.rolling(window=period, min_periods=max(1,period//2)).max()
            lowest_low = low.rolling(window=period, min_periods=max(1,period//2)).min()
            
            long_exit = highest_high - (multiplier * atr)
            short_exit = lowest_low + (multiplier * atr)
            return long_exit, short_exit
        except Exception as e:
            logger.error(f"Error calculating Chandelier Exit(p={period},m={multiplier}): {e}")
            empty_series = pd.Series(np.nan, index=high.index, dtype=float)
            return empty_series, empty_series

    def parabolic_sar(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        step = self.config.get("psar_step", 0.02)
        max_step = self.config.get("psar_max_step", 0.2)
        try:
            # PSARIndicator from 'ta' lib needs 'close'
            return ta.trend.PSARIndicator(high=high, low=low, close=close, step=step, max_step=max_step, fillna=False).psar()
        except Exception as e:
            logger.error(f"Error calculating Parabolic SAR(step={step},max_step={max_step}): {e}")
            return pd.Series(np.nan, index=high.index, dtype=float)
    
    def atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: Optional[int] = None) -> pd.Series:
        # Allow period override for specific uses like Chandelier, otherwise use risk ATR period
        atr_period = period if period is not None else self.config.get("atr_period_risk", 14)
        try:
            return ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=atr_period, fillna=False).average_true_range()
        except Exception as e:
            logger.error(f"Error calculating ATR(period={atr_period}): {e}")
            return pd.Series(np.nan, index=high.index, dtype=float)

class DualNNFXSystem:
    def __init__(self, bitget_api_instance: BitgetAPI, config: StrategyConfig):
        self.api = bitget_api_instance
        self.config = config # Store the main config
        self.indicators = NNFXIndicators(config) # Pass config to indicators
        self.indicator_calc_errors = {} # To track errors per symbol

    def _safe_calculate_indicator(self, data_df, indicator_name, calculation_func, *args):
        try:
            return calculation_func(*args)
        except Exception as e:
            logger.error(f"Error in indicator '{indicator_name}' calculation: {e}", exc_info=False) # Keep log concise
            # Store error for this symbol if running in a symbol-specific context (not done here yet)
            # Return NaN series or tuple of NaN series matching expected output structure
            if isinstance(data_df.index, pd.DatetimeIndex): # Ensure index compatibility
                idx = data_df.index
                if "Tuple" in str(calculation_func.__annotations__.get('return', '')): # Check if it's supposed to return a tuple
                    # This is a bit hacky, better to know expected output structure
                    # For Klinger & Chandelier that return 2 series:
                    if indicator_name in ["klinger", "chandelier_exit"]:
                         return pd.Series(np.nan, index=idx), pd.Series(np.nan, index=idx)
                return pd.Series(np.nan, index=idx)
            return pd.Series(dtype=float) # Fallback if no index

    def calculate_indicators(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        self.indicator_calc_errors[symbol] = [] # Reset errors for this symbol
        data = df.copy()
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_cols):
            missing = [col for col in required_cols if col not in data.columns]
            logger.error(f"[{symbol}] Missing required columns for indicator calculation: {missing}")
            self.indicator_calc_errors[symbol].append(f"Missing columns: {missing}")
            return data # Return original data, it will likely be dropped by dropna later

        # System A
        data['tema'] = self._safe_calculate_indicator(data, 'tema', self.indicators.tema, data['close'])
        data['cci'] = self._safe_calculate_indicator(data, 'cci', self.indicators.cci, data['high'], data['low'], data['close'])
        data['elder_fi'] = self._safe_calculate_indicator(data, 'elder_fi', self.indicators.elder_force_index, data['close'], data['volume'])
        data['chandelier_long'], data['chandelier_short'] = self._safe_calculate_indicator(
            data, 'chandelier_exit', self.indicators.chandelier_exit, data['high'], data['low'], data['close']
        )
        
        # System B
        data['kijun_sen'] = self._safe_calculate_indicator(data, 'kijun_sen', self.indicators.kijun_sen, data['high'], data['low'])
        data['williams_r'] = self._safe_calculate_indicator(data, 'williams_r', self.indicators.williams_r, data['high'], data['low'], data['close'])
        data['klinger'], data['klinger_signal'] = self._safe_calculate_indicator(
            data, 'klinger', self.indicators.klinger_oscillator, data['high'], data['low'], data['close'], data['volume']
        )
        data['psar'] = self._safe_calculate_indicator(data, 'psar', self.indicators.parabolic_sar, data['high'], data['low'], data['close'])
        
        # Risk Management
        data['atr'] = self._safe_calculate_indicator(data, 'atr_risk', self.indicators.atr, data['high'], data['low'], data['close']) # Uses atr_period_risk
        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        # Ensure indicator columns exist, default to 0 or False if they resulted in all NaNs and were not dropped
        # This is important if _safe_calculate_indicator returns all NaNs and they weren't dropped yet
        indicator_cols_for_signal = [
            'tema', 'cci', 'elder_fi', 'kijun_sen', 'williams_r', 'klinger',
            'chandelier_long', 'chandelier_short', 'psar'
        ]
        for col in indicator_cols_for_signal:
            if col not in df.columns: df[col] = 0 # Or np.nan, but 0 simplifies np.where if NaNs are already handled

        # System A Signals
        df['system_a_baseline'] = np.where(df['close'] > df['tema'], 1, np.where(df['close'] < df['tema'], -1, 0))
        df['system_a_confirmation'] = np.where(df['cci'] > 0, 1, np.where(df['cci'] < 0, -1, 0))
        df['system_a_volume'] = np.where(df['elder_fi'] > 0, 1, np.where(df['elder_fi'] < 0, -1, 0))
        
        # System B Signals
        df['system_b_baseline'] = np.where(df['close'] > df['kijun_sen'], 1, np.where(df['close'] < df['kijun_sen'], -1, 0))
        # Williams %R: Standard is -20 (overbought), -80 (oversold).
        # NNFX Confirmation might be: Long if NOT oversold (e.g., W%R > -80 or > -50). Short if NOT overbought (e.g., W%R < -20 or < -50).
        # The exact rule for NNFX confirmation using W%R can vary. The user should verify this logic.
        # Current: 1 if > -80 (not deeply oversold). -1 if < -20 (not deeply overbought). 0 otherwise. This makes the -1 condition hard to meet if already > -80.
        # A more typical NNFX might use a central threshold like -50.
        # Example: 1 if df['williams_r'] > -50, -1 if df['williams_r'] < -50, else 0
        # For now, keeping original structure but with a strong comment.
        # USER VERIFY: Williams %R confirmation logic. Current logic:
        # If W%R = -70, system_b_confirmation = 1.
        # If W%R = -10, system_b_confirmation = 0 (because it's not > -80, and not < -20).
        # If W%R = -90, system_b_confirmation might be -1 if not > -80 is false.
        # This needs careful thought on the np.where nesting.
        # A clearer, non-nested approach:
        cond_gt_80 = df['williams_r'] > -80
        cond_lt_20 = df['williams_r'] < -20
        df['system_b_confirmation'] = np.select(
            [cond_gt_80, cond_lt_20], # if > -80, then 1. Else if < -20, then -1.
            [1, -1],
            default=0 # If between -80 and -20 (inclusive of -80, exclusive of -20 if using this logic)
        )
        # This means: >-80 is bullish (1). <=-80 AND <-20 is bearish (-1). Between -20 and -80 (inclusive) is neutral (0).
        # This is one interpretation. Another: Bullish if >-50, Bearish if <-50.
        # logger.info("Williams %R confirmation logic needs verification against specific NNFX rules.")

        df['system_b_volume'] = np.where(df['klinger'] > df.get('klinger_signal', 0), 1, np.where(df['klinger'] < df.get('klinger_signal',0), -1, 0)) # Compare Klinger to its signal line

        # Combined signals
        df['long_signal'] = (
            (df['system_a_baseline'] == 1) & (df['system_a_confirmation'] == 1) & (df['system_a_volume'] == 1) &
            (df['system_b_baseline'] == 1) & (df['system_b_confirmation'] == 1) & (df['system_b_volume'] == 1)
        ).astype(bool)
        df['short_signal'] = (
            (df['system_a_baseline'] == -1) & (df['system_a_confirmation'] == -1) & (df['system_a_volume'] == -1) &
            (df['system_b_baseline'] == -1) & (df['system_b_confirmation'] == -1) & (df['system_b_volume'] == -1)
        ).astype(bool)
        
        # Exit signals
        df['long_exit'] = ((df['close'] < df['chandelier_long']) | (df['close'] < df['psar'])).astype(bool)
        df['short_exit'] = ((df['close'] > df['chandelier_short']) | (df['close'] > df['psar'])).astype(bool)
        return df

    def backtest_pair(self, symbol: str) -> Dict:
        logger.info(f"Backtesting {symbol}...")
        cfg_bt = self.config # Main config object
        
        df = self.api.get_klines(symbol, "4H", limit=cfg_bt.get("backtest_kline_limit", 1000))
        if df.empty: return {"symbol": symbol, "error": "No data available"}
        if len(df) < cfg_bt.get("backtest_min_data_after_get_klines", 100):
            return {"symbol": symbol, "error": f"Insufficient data: {len(df)} candles"}

        df = self.calculate_indicators(df, symbol)
        df = df.dropna() # Drop rows with NaNs from indicator calculations
        
        if len(df) < cfg_bt.get("backtest_min_data_after_indicators", 50):
            return {"symbol": symbol, "error": f"Insufficient data after indicators: {len(df)} rows"}

        df = self.generate_signals(df) # Generate signals on cleaned data
        
        trades, position, equity_curve = [], None, []
        current_equity = 10000.0
        risk_per_trade = cfg_bt.get("risk_per_trade", 0.015)
        sl_atr_mult = cfg_bt.get("stop_loss_atr_multiplier", 2.0)
        tp_atr_mult = cfg_bt.get("take_profit_atr_multiplier", 3.0)

        for i in range(len(df)):
            row = df.iloc[i]
            if position is None:
                atr_val = row.get('atr', np.nan)
                if pd.notna(atr_val) and atr_val > 1e-9: # Ensure ATR is valid and not effectively zero
                    if row.get('long_signal', False):
                        stop_loss = row['close'] - (sl_atr_mult * atr_val)
                        take_profit = row['close'] + (tp_atr_mult * atr_val)
                        position_size = (current_equity * risk_per_trade) / (sl_atr_mult * atr_val)
                        position = {'type': 'long', 'entry_price': row['close'], 'entry_time': row.name, 
                                    'stop_loss': stop_loss, 'take_profit': take_profit, 'atr': atr_val, 'position_size': position_size}
                    elif row.get('short_signal', False):
                        stop_loss = row['close'] + (sl_atr_mult * atr_val)
                        take_profit = row['close'] - (tp_atr_mult * atr_val)
                        position_size = (current_equity * risk_per_trade) / (sl_atr_mult * atr_val)
                        position = {'type': 'short', 'entry_price': row['close'], 'entry_time': row.name,
                                    'stop_loss': stop_loss, 'take_profit': take_profit, 'atr': atr_val, 'position_size': position_size}
            elif position is not None:
                exit_trade, exit_reason = False, ""
                if position['type'] == 'long':
                    if row['close'] <= position['stop_loss']: exit_trade, exit_reason = True, "Stop Loss"
                    elif row['close'] >= position['take_profit']: exit_trade, exit_reason = True, "Take Profit"
                    elif row.get('long_exit', False): exit_trade, exit_reason = True, "Exit Signal"
                elif position['type'] == 'short':
                    if row['close'] >= position['stop_loss']: exit_trade, exit_reason = True, "Stop Loss"
                    elif row['close'] <= position['take_profit']: exit_trade, exit_reason = True, "Take Profit"
                    elif row.get('short_exit', False): exit_trade, exit_reason = True, "Exit Signal"
                
                if exit_trade:
                    pnl_pips = (row['close'] - position['entry_price']) if position['type'] == 'long' else (position['entry_price'] - row['close'])
                    # Ensure position['atr'] is not zero for pnl_r calculation
                    pnl_r = pnl_pips / (sl_atr_mult * position['atr']) if position['atr'] > 1e-9 else 0.0
                    pnl_dollar = pnl_pips * position['position_size']
                    current_equity += pnl_dollar
                    trades.append({'symbol': symbol, 'type': position['type'], 'entry_time': position['entry_time'], 
                                   'exit_time': row.name, 'entry_price': position['entry_price'], 'exit_price': row['close'],
                                   'pnl_pips': pnl_pips, 'pnl_r': pnl_r, 'pnl_dollar': pnl_dollar, 
                                   'exit_reason': exit_reason, 'atr_at_entry': position['atr'], 'equity_after_trade': current_equity})
                    position = None
            equity_curve.append({'timestamp': row.name, 'equity': current_equity, 'in_position': position is not None})
        
        if not trades:
            logger.info(f"[{symbol}] No trades generated.")
            return {'symbol': symbol, 'total_trades': 0, 'final_equity': current_equity, 'equity_curve': equity_curve, 'trades': pd.DataFrame()}

        trades_df = pd.DataFrame(trades)
        total_trades = len(trades_df)
        wins = trades_df[trades_df['pnl_r'] > 0]
        losses = trades_df[trades_df['pnl_r'] < 0]
        win_rate = len(wins) / total_trades if total_trades > 0 else 0
        avg_win_r = wins['pnl_r'].mean() if not wins.empty else 0
        avg_loss_r = losses['pnl_r'].mean() if not losses.empty else 0 # Will be negative
        
        sum_profit_r = wins['pnl_r'].sum()
        sum_loss_r = abs(losses['pnl_r'].sum()) # Sum of absolute losses
        profit_factor = sum_profit_r / sum_loss_r if sum_loss_r > 0 else (float('inf') if sum_profit_r > 0 else 0)

        total_return_r = trades_df['pnl_r'].sum()
        total_return_pct = (current_equity / 10000.0 - 1) * 100

        return {
            'symbol': symbol, 'total_trades': total_trades, 'win_rate': win_rate,
            'avg_win_r': avg_win_r, 'avg_loss_r': avg_loss_r, 'profit_factor': profit_factor,
            'total_return_r': total_return_r, 'total_return_pct': total_return_pct,
            'max_consecutive_losses': self._calculate_max_consecutive_losses(trades_df),
            'max_drawdown_pct': self._calculate_max_drawdown(equity_curve),
            'sharpe_ratio': self._calculate_sharpe_ratio(trades_df, equity_curve),
            'sortino_ratio': self._calculate_sortino_ratio(trades_df, equity_curve),
            'var_95_r': np.percentile(trades_df['pnl_r'], 5) if total_trades > 0 else 0,
            'max_loss_r': trades_df['pnl_r'].min() if total_trades > 0 else 0,
            'final_equity': current_equity, 'trades': trades_df.to_dict('records'), # Store trades as dicts for easier JSON if needed
            'equity_curve': equity_curve
        }

    def _calculate_max_consecutive_losses(self, trades_df: pd.DataFrame) -> int:
        if trades_df.empty: return 0
        consecutive, max_c = 0, 0
        for r_val in trades_df['pnl_r']:
            consecutive = consecutive + 1 if r_val < 0 else 0
            max_c = max(max_c, consecutive)
        return max_c

    def _calculate_max_drawdown(self, equity_curve_data: List[Dict]) -> float:
        if not equity_curve_data: return 0.0
        equity = pd.Series([p['equity'] for p in equity_curve_data])
        if equity.empty: return 0.0
        peak = equity.expanding(min_periods=1).max()
        drawdown = (equity - peak) / peak
        return abs(drawdown.min() * 100) if pd.notna(drawdown.min()) else 0.0
    
    def _calculate_annualization_factor(self, trades_df: pd.DataFrame, equity_curve_data: List[Dict]) -> float:
        if trades_df.empty or len(trades_df) < 2 or not equity_curve_data:
            return 1.0 # No basis for annualization

        # Try to infer duration from equity curve timestamps
        start_time = equity_curve_data[0]['timestamp']
        end_time = equity_curve_data[-1]['timestamp']
        
        if not isinstance(start_time, pd.Timestamp) or not isinstance(end_time, pd.Timestamp):
             # Fallback if timestamps are not proper pandas Timestamps (e.g. from parallel processing raw dicts)
            try:
                start_time = pd.to_datetime(start_time)
                end_time = pd.to_datetime(end_time)
            except: # If conversion fails, cannot annualize well
                return 1.0

        duration_days = (end_time - start_time).total_seconds() / (24 * 3600)
        if duration_days < 1: duration_days = 1 # Avoid division by zero for very short backtests
        
        num_trades = len(trades_df)
        trades_per_year = (num_trades / duration_days) * 252 # Assuming 252 trading days/year
        
        return np.sqrt(trades_per_year) if trades_per_year > 0 else 1.0


    def _calculate_sharpe_ratio(self, trades_df: pd.DataFrame, equity_curve_data: List[Dict], risk_free_rate_annual: float = 0.0) -> float:
        if trades_df.empty or len(trades_df['pnl_r']) < 2: return 0.0
        
        # Use P&L from trades for returns, not R-multiples for Sharpe if using equity-based risk-free rate
        # Or, assume R-multiples are fine and risk_free_rate is in terms of R. Let's use R-multiples for consistency.
        trade_returns_r = trades_df['pnl_r']
        
        if trade_returns_r.std() == 0:
            return float('inf') if trade_returns_r.mean() > 0 else 0.0 # Simplified risk_free_rate for R-multiples = 0
            
        annual_factor = self._calculate_annualization_factor(trades_df, equity_curve_data)
        
        # Sharpe on R-multiples per trade: (Mean R - RiskFree R_per_trade) / StdDev R_per_trade
        # Assuming RiskFree R_per_trade is 0 for simplicity.
        sharpe_per_trade = trade_returns_r.mean() / trade_returns_r.std()
        return sharpe_per_trade * annual_factor

    def _calculate_sortino_ratio(self, trades_df: pd.DataFrame, equity_curve_data: List[Dict], target_return_r_per_trade: float = 0.0) -> float:
        if trades_df.empty or len(trades_df['pnl_r']) < 2: return 0.0
        
        trade_returns_r = trades_df['pnl_r']
        downside_returns_r = trade_returns_r[trade_returns_r < target_return_r_per_trade]
        
        if downside_returns_r.empty:
            return float('inf') if trade_returns_r.mean() > target_return_r_per_trade else 0.0
        
        downside_std_r = downside_returns_r.std()
        if downside_std_r == 0 or pd.isna(downside_std_r): # Handle std of single point or all same
            return float('inf') if trade_returns_r.mean() > target_return_r_per_trade else 0.0
            
        annual_factor = self._calculate_annualization_factor(trades_df, equity_curve_data)
        sortino_per_trade = (trade_returns_r.mean() - target_return_r_per_trade) / downside_std_r
        return sortino_per_trade * annual_factor

    def scan_pairs(self, symbols: List[str], save_results: bool = True, max_workers: Optional[int] = None) -> pd.DataFrame:
        logger.info(f"Starting backtest scan on {len(symbols)} pairs using {max_workers or os.cpu_count()} workers...")
        results, failed_pairs = [], []

        # partial_backtest_pair = partial(self.backtest_pair) # If backtest_pair was static or top-level
        # For instance method, we pass 'self' and 'symbol'
        
        # To make self.backtest_pair work with ProcessPoolExecutor,
        # ensure self.api and self.indicators (and their configs) are picklable.
        # requests.Session might not be, so self.api might need to be re-init in worker.
        # For simplicity now, assuming it might work or fall back to serial.
        # A robust way is to make a static/top-level worker function.
        
        # Top-level worker function for pickling safety
        def _backtest_worker(api_config_dict: Dict, strategy_config_dict: Dict, symbol: str) -> Dict:
            # Re-initialize non-picklable or process-specific resources here
            local_api = BitgetAPI(**api_config_dict)
            local_strategy_config = StrategyConfig() # Assumes strategy_config.json is accessible
            # If StrategyConfig instance itself is complex, pass its dict representation strategy_config.params
            local_system = DualNNFXSystem(local_api, local_strategy_config) # Pass StrategyConfig instance
            return local_system.backtest_pair(symbol)

        api_cfg_dict = {"api_key": self.api.api_key, "secret_key": self.api.secret_key, 
                        "passphrase": self.api.passphrase, "sandbox": False} # Assuming live for workers
        strategy_cfg_dict = self.config.params # Pass the loaded dict

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Prepare futures
            futures = {executor.submit(_backtest_worker, api_cfg_dict, strategy_cfg_dict, symbol): symbol for symbol in symbols}
            
            for i, future in enumerate(as_completed(futures)):
                symbol = futures[future]
                try:
                    result = future.result()
                    logger.info(f"Completed processing for {symbol} ({i+1}/{len(symbols)})")
                    min_trades = self.config.get("backtest_min_trades_for_ranking", 3)
                    if 'error' not in result and result.get('total_trades', 0) >= min_trades :
                        # Reconstruct trades_df if it was converted to dict of records
                        if isinstance(result.get('trades'), list):
                             result['trades_df_for_scoring'] = pd.DataFrame(result['trades'])
                        else: # Should already be a DataFrame if not modified, or empty
                             result['trades_df_for_scoring'] = result.get('trades', pd.DataFrame())

                        results.append({
                            'symbol': result['symbol'], 'total_trades': result['total_trades'],
                            'win_rate': result['win_rate'], 'profit_factor': result['profit_factor'],
                            'total_return_r': result['total_return_r'], 'total_return_pct': result['total_return_pct'],
                            'max_drawdown_pct': result['max_drawdown_pct'], 
                            'sharpe_ratio': result['sharpe_ratio'], 'sortino_ratio': result['sortino_ratio'],
                            'max_consecutive_losses': result['max_consecutive_losses'],
                            'var_95_r': result['var_95_r'], 'max_loss_r': result['max_loss_r'],
                            'score': self._calculate_score(result) # Pass full result dict
                        })
                    else:
                        failed_pairs.append(symbol)
                        error_msg = result.get('error', f"Insufficient trades: {result.get('total_trades',0)}")
                        logger.warning(f"Failed/Skipped {symbol}: {error_msg}")
                except Exception as e:
                    logger.error(f"Error processing future for {symbol}: {e}", exc_info=True)
                    failed_pairs.append(symbol)
        
        logger.info(f"Scan complete. Successfully ranked: {len(results)}, Failed/Skipped: {len(failed_pairs)}")
        if results:
            df = pd.DataFrame(results).sort_values('score', ascending=False)
            if save_results: self._save_scan_artifacts(df, symbols, failed_pairs)
            return df
        else:
            logger.warning("No successful backtests met ranking criteria.")
            if save_results and symbols: self._save_scan_artifacts(pd.DataFrame(), symbols, failed_pairs)
            return pd.DataFrame()

    def _save_scan_artifacts(self, df: pd.DataFrame, symbols_scanned: List[str], failed_pairs: List[str]):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        Path("results").mkdir(parents=True, exist_ok=True)
        results_file = Path(f"results/scan_results_{timestamp}.csv")
        summary_file = Path(f"results/scan_summary_{timestamp}.txt")

        if not df.empty:
            df.to_csv(results_file, index=False)
            logger.info(f"Scan results saved to {results_file}")
        
        with open(summary_file, 'w') as f:
            f.write(f"NNFX Bot - Dual System Backtest Summary - {timestamp}\n" + "="*60 + "\n")
            f.write(f"Total pairs attempted: {len(symbols_scanned)}\n")
            f.write(f"Successfully ranked pairs (met criteria): {len(df)}\n")
            f.write(f"Failed or skipped pairs: {len(failed_pairs)}\n\n")
            if not df.empty:
                f.write("TOP PERFORMING PAIRS (max 10 or all if fewer)\n" + "-"*40 + "\n")
                for _, row in df.head(min(10, len(df))).iterrows():
                    f.write(f"{row['symbol']:<12} Score: {row['score']:.2f} | WR: {row['win_rate']:.1%} | PF: {row['profit_factor']:.2f} | Ret: {row['total_return_pct']:.1f}% | DD: {row['max_drawdown_pct']:.1f}% | Trades: {row['total_trades']}\n")
                f.write("\nAGGREGATE STATISTICS (for ranked pairs)\n" + "-"*40 + "\n")
                f.write(f"Avg Win Rate: {df['win_rate'].mean():.1%}\n")
                f.write(f"Avg Profit Factor: {df['profit_factor'][np.isfinite(df['profit_factor'])].mean():.2f}\n")
                f.write(f"Avg Return %: {df['total_return_pct'].mean():.1f}%\n")
            if failed_pairs:
                f.write(f"\nFAILED/SKIPPED PAIRS ({len(failed_pairs)}):\n" + ", ".join(failed_pairs) + "\n")
        logger.info(f"Scan summary saved to {summary_file}")

    def _calculate_score(self, result: Dict) -> float:
        # result now contains 'trades_df_for_scoring' if trades occurred
        cfg_score = self.config.get("scoring", {})
        if result.get('total_trades', 0) < self.config.get("backtest_min_trades_for_ranking", 3): return -999.0 # Very low score if not enough trades

        win_rate = result.get('win_rate', 0)
        profit_factor = result.get('profit_factor', 0)
        return_pct = result.get('total_return_pct', 0)
        total_trades = result.get('total_trades', 0)
        sharpe = result.get('sharpe_ratio', 0)
        max_dd = result.get('max_drawdown_pct', 100)
        max_con_loss = result.get('max_consecutive_losses', 20)
        var_95 = result.get('var_95_r', -5) # Typically negative

        # Normalize metrics (0-100 scale, roughly)
        win_rate_s = win_rate * 100
        pf_s = min(profit_factor * 20, 100) if np.isfinite(profit_factor) else (cfg_score.get("profit_factor_inf_score", 50) if profit_factor > 0 else 0)
        ret_s = min(max(return_pct, -100), 100) # Cap return score
        trades_s = min(total_trades * 0.5, 100) # Max 100 for 200 trades
        sharpe_s = min(max(sharpe * 25, -100), 100) if np.isfinite(sharpe) else (cfg_score.get("sharpe_ratio_inf_score", 50) if sharpe > 0 else -50)

        # Penalties (positive values, subtracted)
        dd_p = max_dd * cfg_score.get("drawdown_penalty_multiplier", 1.5)
        con_loss_p = max_con_loss * cfg_score.get("consecutive_loss_penalty_multiplier", 3.0)
        var_p = abs(var_95) * cfg_score.get("var_95_penalty_multiplier", 5.0)

        score = (
            win_rate_s * cfg_score.get("win_rate_weight", 0.25) +
            pf_s       * cfg_score.get("profit_factor_weight", 0.25) +
            ret_s      * cfg_score.get("return_pct_weight", 0.20) +
            trades_s   * cfg_score.get("trade_frequency_weight", 0.10) +
            sharpe_s   * cfg_score.get("sharpe_ratio_weight", 0.20) -
            dd_p       * cfg_score.get("drawdown_penalty_weight", 0.05) -
            con_loss_p * cfg_score.get("consecutive_loss_penalty_weight", 0.03) -
            var_p      * cfg_score.get("var_95_penalty_weight", 0.02)
        )
        return round(max(score, -1000.0), 2) # Allow significant negative scores for bad results

    def get_current_signals(self, symbols: List[str]) -> pd.DataFrame:
        # This can also be parallelized if symbols list is long
        logger.info(f"Getting current signals for {len(symbols)} pairs...")
        signals_data = []
        cfg_sig_conf = self.config.get("signal_confidence", {})

        for symbol in symbols:
            try:
                df = self.api.get_klines(symbol, "4H", 200) # Enough for indicators
                if df.empty or len(df) < self.config.get("backtest_min_data_after_indicators", 50):
                    logger.warning(f"[{symbol}] Insufficient data for current signals.")
                    continue

                df = self.calculate_indicators(df, symbol)
                df = df.dropna()
                if df.empty or len(df) < 2: continue # Need at least 2 rows
                df = self.generate_signals(df)
                
                latest, current = df.iloc[-2], df.iloc[-1]
                signal_type, confidence = "NONE", 0.0
                
                if latest.get('long_signal', False):
                    signal_type, confidence = "LONG", self._calculate_signal_confidence(df.iloc[-5:], 'long', cfg_sig_conf)
                elif latest.get('short_signal', False):
                    signal_type, confidence = "SHORT", self._calculate_signal_confidence(df.iloc[-5:], 'short', cfg_sig_conf)
                
                # ... (rest of signal calculation logic as before, using .get for safety) ...
                atr_val = latest.get('atr', np.nan)
                stop_dist = self.config.get("stop_loss_atr_multiplier", 2.0) * atr_val if pd.notna(atr_val) else 0
                tp_dist = self.config.get("take_profit_atr_multiplier", 3.0) * atr_val if pd.notna(atr_val) else 0
                
                signals_data.append({
                    'symbol': symbol, 'signal': signal_type, 'confidence': round(confidence, 3),
                    'price': latest.get('close', np.nan), 'atr': atr_val,
                    'stop_loss_price': latest.get('close', np.nan) - stop_dist if signal_type == "LONG" else (latest.get('close', np.nan) + stop_dist if signal_type == "SHORT" else np.nan),
                    'take_profit_price': latest.get('close', np.nan) + tp_dist if signal_type == "LONG" else (latest.get('close', np.nan) - tp_dist if signal_type == "SHORT" else np.nan),
                    'risk_reward_ratio': tp_dist / stop_dist if stop_dist > 1e-9 else 0,
                    'timestamp': latest.name, 
                })
                time.sleep(0.1) # Small delay even in serial loop
            except Exception as e:
                logger.error(f"Error getting signals for {symbol}: {e}", exc_info=True)
        
        return pd.DataFrame(signals_data).sort_values(['signal', 'confidence'], ascending=[True, False]) if signals_data else pd.DataFrame()

    def _calculate_signal_confidence(self, recent_data: pd.DataFrame, signal_type: str, cfg: Dict) -> float:
        if len(recent_data) < cfg.get("min_recent_data_for_confidence", 3): return cfg.get("default_confidence", 0.5)
        
        sup_w = cfg.get("signal_support_weight",0.4)
        trend_w = cfg.get("trend_consistency_weight",0.25)
        mom_w = cfg.get("momentum_consistency_weight",0.20)
        vol_w = cfg.get("volume_consistency_weight",0.15)

        if signal_type == 'long':
            sig_col, base_gt, conf_gt, vol_gt = 'long_signal', True, True, True
        else: # short
            sig_col, base_gt, conf_gt, vol_gt = 'short_signal', False, False, False
            
        # .get(col, pd.Series(False, index=recent_data.index)) handles missing columns safely
        signal_support = recent_data.get(sig_col, pd.Series(False, index=recent_data.index)).mean()
        trend_consistency = (recent_data.get('system_a_baseline', pd.Series(0, index=recent_data.index)) > 0 if base_gt else recent_data.get('system_a_baseline', pd.Series(0, index=recent_data.index)) < 0).mean()
        momentum_consistency = (recent_data.get('system_a_confirmation', pd.Series(0, index=recent_data.index)) > 0 if conf_gt else recent_data.get('system_a_confirmation', pd.Series(0, index=recent_data.index)) < 0).mean()
        volume_consistency = (recent_data.get('system_a_volume', pd.Series(0, index=recent_data.index)) > 0 if vol_gt else recent_data.get('system_a_volume', pd.Series(0, index=recent_data.index)) < 0).mean()
        
        confidence = (signal_support*sup_w + trend_consistency*trend_w + momentum_consistency*mom_w + volume_consistency*vol_w)
        return min(max(confidence, 0.0), 1.0)

    def export_detailed_results(self, rankings: pd.DataFrame, current_signals: pd.DataFrame) -> str:
        # ... (export logic largely same, ensure it handles empty DataFrames gracefully) ...
        # (Make sure to use Path objects for file paths)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = Path("results")
        results_dir.mkdir(parents=True, exist_ok=True)
        filename = results_dir / f"nnfx_analysis_{timestamp}.xlsx"
        
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                rankings.head(100).to_excel(writer, sheet_name='Backtest Rankings', index=False) # Limit rows for Excel
                current_signals.to_excel(writer, sheet_name='Current Signals', index=False)
                # Add summary sheets as before
            logger.info(f"Detailed results exported to {filename}")
            return str(filename)
        except Exception as e:
            logger.error(f"Error exporting results to Excel: {e}", exc_info=True)
            return ""

# --- Main Execution ---
def run_comprehensive_analysis(api_config_dict: Dict, main_strategy_config: StrategyConfig, 
                               test_symbols_override: Optional[List[str]] = None,
                               max_workers: Optional[int] = None) -> Dict:
    logger.info("=" * 60 + "\nNNFX Bot - Comprehensive Analysis Started\n" + "=" * 60)
    start_time_total = time.time()
    
    api = BitgetAPI(**api_config_dict)
    system = DualNNFXSystem(api, main_strategy_config) # Pass StrategyConfig instance
    
    test_symbols = []
    if test_symbols_override:
        test_symbols = test_symbols_override[:main_strategy_config.get("top_n_pairs_by_volume_to_scan", 8)]
        logger.info(f"Using provided override list of {len(test_symbols)} symbols.")
    else:
        logger.info("Fetching all tickers to determine top N by volume...")
        all_tickers = api.get_all_tickers_data()
        if all_tickers:
            # Filter for USDT pairs
            usdt_tickers = [t for t in all_tickers if t.get('symbolId','').endswith('USDT')]
            
            # Optionally filter by major bases
            if main_strategy_config.get("filter_by_major_bases_for_top_volume", True):
                major_bases = set(main_strategy_config.get("major_bases_for_filtering", []))
                usdt_tickers = [t for t in usdt_tickers if t.get('symbolId','').replace('USDT','') in major_bases]

            # Sort by USDT volume (descending), handle missing or non-numeric volume
            def get_usdt_vol(ticker):
                vol_str = ticker.get('usdtVol', '0')
                try: return float(vol_str)
                except ValueError: return 0.0
            
            usdt_tickers.sort(key=get_usdt_vol, reverse=True)
            
            top_n = main_strategy_config.get("top_n_pairs_by_volume_to_scan", 8)
            test_symbols = [t['symbolId'].replace('_SPBL', '') for t in usdt_tickers[:top_n]] # Remove _SPBL if present
            logger.info(f"Top {len(test_symbols)} symbols by USDT volume selected for scan: {test_symbols}")
        else:
            logger.error("Could not fetch tickers for dynamic selection. Using fallback.")
            test_symbols = ['BTCUSDT', 'ETHUSDT'] # Minimal fallback

    if not test_symbols:
        logger.error("No symbols selected for scanning. Aborting analysis.")
        return {'success': False, 'error': 'No symbols to scan', 'scan_duration_seconds': 0}

    logger.info(f"Starting scan for {len(test_symbols)} symbols: {test_symbols}")
    scan_start_time = time.time()
    rankings_df = system.scan_pairs(test_symbols, save_results=True, max_workers=max_workers)
    scan_duration = time.time() - scan_start_time
    logger.info(f"Pair scan completed in {scan_duration:.2f} seconds.")

    analysis_result = {'success': False, 'scan_duration_seconds': round(scan_duration,2)}

    if not rankings_df.empty:
        analysis_result['success'] = True
        analysis_result['rankings_summary'] = rankings_df[['symbol', 'score', 'win_rate', 'profit_factor', 'total_return_pct', 'total_trades']].head().to_dict('records')
        
        top_ranked_symbols = rankings_df['symbol'].head(5).tolist() # Get signals for top 5 ranked
        current_signals_df = system.get_current_signals(top_ranked_symbols)
        analysis_result['current_signals_summary'] = current_signals_df[current_signals_df['signal'] != 'NONE'].to_dict('records')
        
        export_file = system.export_detailed_results(rankings_df, current_signals_df)
        analysis_result['export_file'] = export_file
        logger.info(f"Top performers (from {len(rankings_df)} ranked):")
        for _, row in rankings_df.head().iterrows(): logger.info(f"  {row['symbol']}: Score {row['score']:.2f}, WR {row['win_rate']:.1%}")
    else:
        logger.warning("No pairs were successfully ranked. Exporting empty/minimal report.")
        analysis_result['error'] = "No pairs successfully ranked."
        system.export_detailed_results(pd.DataFrame(), pd.DataFrame()) # Save empty structure

    total_duration = time.time() - start_time_total
    logger.info(f"Comprehensive analysis finished in {total_duration:.2f} seconds.")
    analysis_result['total_duration_seconds'] = round(total_duration,2)
    return analysis_result

def cleanup_old_files():
    cfg_cleanup = strategy_config.get("cleanup", {})
    days_old = cfg_cleanup.get("cache_days_old", 7)
    max_results = cfg_cleanup.get("max_results_to_keep", 20)

    cutoff_date = datetime.now() - timedelta(days=days_old)
    cleaned_count = 0
    
    for d_path in [Path("data"), Path("results"), Path("logs")]: # Include logs in cleanup
        d_path.mkdir(parents=True, exist_ok=True)
        files_to_check = list(d_path.glob("*.csv")) + list(d_path.glob("*.json")) + \
                         list(d_path.glob("*.txt")) + list(d_path.glob("*.xlsx")) + \
                         list(d_path.glob("*.log"))

        if d_path.name == "results" or d_path.name == "logs": # Sort by time for these, keep newest
            files_to_check.sort(key=os.path.getmtime)
            to_delete = files_to_check[:-max_results] if len(files_to_check) > max_results else []
        else: # For 'data' (cache), delete if older than cutoff
            to_delete = [f for f in files_to_check if f.stat().st_mtime < cutoff_date.timestamp()]

        for old_file in to_delete:
            try:
                old_file.unlink()
                logger.debug(f"Cleaned old file: {old_file}")
                cleaned_count += 1
            except Exception as e:
                logger.warning(f"Could not remove old file {old_file}: {e}")
    logger.info(f"Cleanup completed. Removed {cleaned_count} old files.")

if __name__ == "__main__":
    run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file_handler = None
    try:
        # --- Setup Directories and File Logging ---
        Path("logs").mkdir(parents=True, exist_ok=True)
        log_file_path = Path(f"logs/system_run_{run_ts}.log")
        log_file_handler = logging.FileHandler(log_file_path)
        log_file_handler.setLevel(logging.INFO) # File logs at INFO
        log_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(log_file_handler) # Add to root logger
        
        # Adjust console handler to INFO, or keep DEBUG as per basicConfig
        for handler in logging.getLogger().handlers:
            if isinstance(handler, logging.StreamHandler) and handler is not sys.stdout: # basicConfig adds one
                handler.setLevel(logging.INFO) # Set console to INFO for less verbosity there

        logger.info(f"NNFX Bot System Run ID: {run_ts}")
        logger.info(f"Strategy Config Path: {strategy_config.config_path}")
        logger.info(f"File logging to: {log_file_path}")

        # --- Load API Config ---
        api_conf = {}
        api_conf_path = Path("config/api_config.json")
        if api_conf_path.exists():
            with open(api_conf_path, "r") as f: api_conf = json.load(f)
            logger.info("API configuration loaded.")
        else:
            logger.warning(f"API config {api_conf_path} not found. Using defaults (public API access only).")
            # Create dummy api_config if it doesn't exist
            try:
                with open(api_conf_path, "w") as f:
                    json.dump({"api_key": "YOUR_API_KEY", "secret_key": "YOUR_SECRET_KEY", "passphrase": "YOUR_PASSPHRASE", "sandbox": True}, f, indent=4)
                logger.info(f"Created dummy API config at {api_conf_path}. Please update.")
            except Exception as e_cfg:
                logger.error(f"Could not create dummy api_config.json: {e_cfg}")
        
        # --- Run Analysis ---
        # `test_symbols_override=None` will trigger dynamic fetching of top N by volume.
        # To test with a specific list, provide it: e.g. ['BTCUSDT', 'ETHUSDT']
        # Set max_workers for parallel processing, e.g., os.cpu_count() or a fixed number.
        num_workers = max(1, os.cpu_count() - 1 if os.cpu_count() else 1) # Leave one core for OS
        analysis_output = run_comprehensive_analysis(api_conf, strategy_config, 
                                                     test_symbols_override=None, 
                                                     max_workers=num_workers)
        
        if analysis_output.get('success'):
            logger.info("Comprehensive analysis completed successfully.")
            if analysis_output.get('export_file'): logger.info(f"Report: {analysis_output['export_file']}")
        else:
            logger.error(f"Comprehensive analysis FAILED: {analysis_output.get('error', 'Unknown error')}")

    except Exception as e_main:
        logger.critical("Critical error in main execution block:", exc_info=True)
    finally:
        logger.info("Performing cleanup of old files...")
        cleanup_old_files()
        logger.info(f"Session {run_ts} complete.")
        if log_file_handler:
            logging.getLogger().removeHandler(log_file_handler)
            log_file_handler.close()

    # Reminder for user:
    # 1. Ensure strategy_config.json is in the config/ directory and has correct values.
    # 2. Ensure api_config.json is in config/ and has your API keys for authenticated actions.
    # 3. Profiling (e.g., with cProfile or SnakeViz) can help identify further bottlenecks
    #    if performance is still an issue after parallelization.
    # 4. Consider adding more comprehensive unit tests for individual components.