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
from typing import Dict, List, Tuple, Optional, Union
import warnings
import logging
from pathlib import Path
import sys
import io

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BitgetAPI:
    """
    Bitget API wrapper for cryptocurrency market data and trading operations.
    
    Features:
    - Rate limiting and retry logic
    - Data caching for improved performance
    - Comprehensive error handling
    - Support for both sandbox and live environments
    """
    
    def __init__(self, api_key: str = "", secret_key: str = "", 
                 passphrase: str = "", sandbox: bool = True):
        """
        Initialize Bitget API client.
        
        Args:
            api_key: Bitget API key
            secret_key: Bitget secret key
            passphrase: Bitget API passphrase
            sandbox: Use sandbox environment for testing
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase
        self.base_url = "https://api.bitget.com"
        self.session = requests.Session()
        self.rate_limit_delay = 0.2  # 200ms between requests
        
        # Create necessary directories
        for directory in ["data", "results", "logs"]:
            Path(directory).mkdir(exist_ok=True)
        
        logger.info(f"Bitget API initialized ({'Sandbox' if sandbox else 'Live'} mode)")
    
    def _generate_signature(self, timestamp: str, method: str, 
                           request_path: str, body: str = "") -> str:
        """Generate API signature for authenticated requests."""
        message = timestamp + method + request_path + body
        signature = base64.b64encode(
            hmac.new(
                self.secret_key.encode(), 
                message.encode(), 
                hashlib.sha256
            ).digest()
        ).decode()
        return signature
    
    def _get_headers(self, method: str, request_path: str, body: str = "") -> Dict[str, str]:
        """Generate request headers for API calls."""
        timestamp = str(int(time.time() * 1000))
        signature = self._generate_signature(timestamp, method, request_path, body)
        
        return {
            "ACCESS-KEY": self.api_key,
            "ACCESS-SIGN": signature,
            "ACCESS-TIMESTAMP": timestamp,
            "ACCESS-PASSPHRASE": self.passphrase,
            "Content-Type": "application/json"
        }
    
    def get_klines(self, symbol: str, granularity: str = "4H", 
                   limit: int = 500) -> pd.DataFrame:
        """
        Fetch OHLCV data from Bitget with caching and retry logic.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            granularity: Timeframe ('1H', '4H', '1D', etc.)
            limit: Number of candles to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Convert granularity to lowercase for API
            period = granularity.lower()
            # Ensure symbol ends with _SPBL for Bitget spot API
            api_symbol = symbol if symbol.endswith('_SPBL') else symbol + '_SPBL'
            # Check cache first
            cache_file = f"data/{api_symbol}_{period}.csv"
            expected_cols_for_cache = ['open', 'high', 'low', 'close', 'volume'] # Define expected columns for cache

            if os.path.exists(cache_file):
                cache_age = time.time() - os.path.getmtime(cache_file)
                if cache_age < 14400:  # 4 hours
                    try:
                        cached_df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                        logger.debug(f"Using cached data for {symbol}")
                        
                        # Check if all expected columns are present in cached data
                        if not all(col in cached_df.columns for col in expected_cols_for_cache):
                            logger.warning(f"Cached data for {symbol} missing expected columns. Refetching.")
                        # Check for NaN/inf only in expected numeric columns
                        elif cached_df[expected_cols_for_cache].isnull().values.any() or \
                             np.isinf(cached_df[expected_cols_for_cache].values).any():
                            logger.warning(f"Cached data for {symbol} contains NaN/inf values in expected columns. Refetching.")
                        elif cached_df.empty:
                             logger.warning(f"Cached data for {symbol} is empty. Refetching.")
                        else:
                            return cached_df # Valid cache found
                    except Exception as e:
                        logger.warning(f"Failed to read or validate cache for {symbol}: {e}. Refetching.")
            
            # Rate limiting
            time.sleep(self.rate_limit_delay)
            
            # API request
            url = f"{self.base_url}/api/spot/v1/market/candles"
            params = {
                "symbol": api_symbol,
                "period": period,
                "limit": str(limit)
            }
            
            # Retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.session.get(url, params=params, timeout=15)
                    logger.debug(f"Raw API response for {symbol}: {response.text}")
                    response.raise_for_status()
                    data = response.json()
                    
                    if data.get('code') == '00000' and data.get('data'):
                        # Process data
                        raw = data['data']
                        df = pd.DataFrame(raw)
                        # Map Bitget fields to expected columns
                        df = df.rename(columns={
                            'ts': 'timestamp',
                            'open': 'open',
                            'high': 'high',
                            'low': 'low',
                            'close': 'close',
                            'baseVol': 'volume'  # Use base asset volume
                        })
                        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
                        df = df.set_index('timestamp')
                        
                        required_cols_for_processing = ['open', 'high', 'low', 'close', 'volume']
                        for col in required_cols_for_processing:
                            if col in df.columns: # Ensure column exists before conversion
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                            else: 
                                logger.error(f"Critical column '{col}' (expected from API map) not found in data for {symbol}. API response structure might have changed or 'baseVol' was missing.")
                                return pd.DataFrame() # Return empty if essential data is missing

                        # Select only the columns needed for further analysis
                        df = df[required_cols_for_processing]

                        df = df.sort_index()
                        logger.debug(f"Parsed DataFrame for {symbol} (pre-clean):\n{df.head()}\nShape: {df.shape}")
                        
                        # Clean NaN/inf values
                        n_before = len(df)
                        df = df.replace([np.inf, -np.inf], np.nan).dropna()
                        n_after = len(df)
                        logger.debug(f"Cleaned DataFrame for {symbol}: dropped {n_before-n_after} rows; shape now {df.shape}")
                        
                        # Log DataFrame info for diagnostics
                        buf = io.StringIO()
                        df.info(buf=buf)
                        logger.debug(f"DataFrame info for {symbol}:\n{buf.getvalue()}")
                        
                        # Defensive checks
                        if df.empty:
                            logger.warning(f"No data returned for {symbol} after cleaning")
                            return pd.DataFrame()
                        
                        # Cache the data
                        try:
                            df.to_csv(cache_file)
                            logger.debug(f"Cached data for {symbol}")
                        except Exception as e:
                            logger.warning(f"Failed to cache data for {symbol}: {e}")
                        return df
                    
                    else:
                        logger.warning(f"API response error for {symbol}: {data.get('msg', 'Unknown error')}")
                        if attempt < max_retries - 1:
                            time.sleep(2 ** attempt)  # Exponential backoff
                        # Continue to next attempt or exit loop if max_retries reached
                        
                except requests.exceptions.RequestException as e:
                    logger.warning(f"Request failed for {symbol}, attempt {attempt + 1}: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                    # Continue to next attempt or exit loop if max_retries reached
            
            logger.error(f"Failed to fetch data for {symbol} after {max_retries} attempts")
            return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Unexpected error fetching {symbol}: {e}")
            return pd.DataFrame()
    
    def get_symbols(self) -> List[str]:
        """
        Get all available trading pairs with caching.
        
        Returns:
            List of trading pair symbols
        """
        try:
            cache_file = "data/symbols_cache.json"
            
            # Check cache (valid for 24 hours)
            if os.path.exists(cache_file):
                cache_age = time.time() - os.path.getmtime(cache_file)
                if cache_age < 86400:  # 24 hours
                    with open(cache_file, 'r') as f:
                        symbols = json.load(f)
                        logger.debug("Using cached symbols list")
                        return symbols
            
            # Fetch from API
            url = f"{self.base_url}/api/spot/v1/market/tickers"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get('code') == '00000' and data.get('data'): # Ensure 'data' key exists
                symbols = [item['symbol'] for item in data['data']]
                
                # Filter for major USDT pairs
                major_bases = [
                    'BTC', 'ETH', 'ADA', 'DOT', 'LINK', 'UNI', 'AAVE', 'SOL', 
                    'MATIC', 'AVAX', 'ATOM', 'NEAR', 'FTM', 'SAND', 'MANA', 
                    'LTC', 'XRP', 'BCH', 'EOS', 'TRX', 'VET', 'XLM', 'IOTA', 
                    'NEO', 'DASH', 'ZEC', 'ETC', 'XMR', 'ALGO', 'THETA'
                ]
                
                major_pairs = [
                    s for s in symbols 
                    if s.endswith('USDT') and any(s.startswith(base + 'USDT') for base in major_bases) # More precise startswith
                ]
                
                # Cache the symbols
                try:
                    with open(cache_file, 'w') as f:
                        json.dump(major_pairs, f)
                    logger.debug("Cached symbols list")
                except Exception as e:
                    logger.warning(f"Failed to cache symbols: {e}")
                
                return major_pairs
            
            logger.error(f"Failed to fetch symbols: {data.get('msg', 'Unknown error')}")
            return []
            
        except Exception as e:
            logger.error(f"Error fetching symbols: {e}")
            return []


class NNFXIndicators:
    """
    Technical indicator calculations for NNFX trading systems.
    
    Implements all required indicators for both System A and System B
    with optimized calculations and proper error handling.
    """
    
    @staticmethod
    def tema(data: pd.Series, period: int = 21) -> pd.Series:
        """
        Triple Exponential Moving Average (TEMA).
        
        Args:
            data: Price series (typically close prices)
            period: Period for calculation
            
        Returns:
            TEMA values
        """
        try:
            ema1 = data.ewm(span=period, adjust=False).mean()
            ema2 = ema1.ewm(span=period, adjust=False).mean()
            ema3 = ema2.ewm(span=period, adjust=False).mean()
            tema = 3 * ema1 - 3 * ema2 + ema3
            return tema
        except Exception as e:
            logger.error(f"Error calculating TEMA: {e}")
            return pd.Series(index=data.index, dtype=float)
    
    @staticmethod
    def kijun_sen(high: pd.Series, low: pd.Series, period: int = 26) -> pd.Series:
        """
        Kijun-Sen (Ichimoku Base Line).
        
        Args:
            high: High price series
            low: Low price series
            period: Period for calculation
            
        Returns:
            Kijun-Sen values
        """
        try:
            highest_high = high.rolling(window=period, min_periods=period).max()
            lowest_low = low.rolling(window=period, min_periods=period).min()
            kijun = (highest_high + lowest_low) / 2
            return kijun
        except Exception as e:
            logger.error(f"Error calculating Kijun-Sen: {e}")
            return pd.Series(index=high.index, dtype=float)
    
    @staticmethod
    def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Commodity Channel Index (CCI).
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: Period for calculation
            
        Returns:
            CCI values
        """
        try:
            return ta.trend.CCIIndicator(
                high=high, 
                low=low, 
                close=close, 
                window=period
            ).cci()
        except Exception as e:
            logger.error(f"Error calculating CCI: {e}")
            return pd.Series(index=high.index, dtype=float)
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, 
                   period: int = 14) -> pd.Series:
        """
        Williams %R oscillator.
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: Period for calculation
            
        Returns:
            Williams %R values
        """
        try:
            return ta.momentum.WilliamsRIndicator(
                high=high, 
                low=low, 
                close=close, 
                lbp=period
            ).williams_r()
        except Exception as e:
            logger.error(f"Error calculating Williams %R: {e}")
            return pd.Series(index=high.index, dtype=float)
    
    @staticmethod
    def elder_force_index(close: pd.Series, volume: pd.Series, 
                         period: int = 13) -> pd.Series:
        """
        Elder's Force Index.
        
        Args:
            close: Close price series
            volume: Volume series
            period: Period for EMA smoothing
            
        Returns:
            Elder's Force Index values
        """
        try:
            price_change = close.diff()
            force_index = price_change * volume
            return force_index.ewm(span=period, adjust=False).mean()
        except Exception as e:
            logger.error(f"Error calculating Elder's Force Index: {e}")
            return pd.Series(index=close.index, dtype=float)
    
    @staticmethod
    def klinger_oscillator(high: pd.Series, low: pd.Series, close: pd.Series,
                          volume: pd.Series, fast: int = 34, slow: int = 55, 
                          signal: int = 13) -> Tuple[pd.Series, pd.Series]:
        """
        Klinger Oscillator.
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            volume: Volume series
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line EMA period
            
        Returns:
            Tuple of (Klinger line, Signal line)
        """
        try:
            # Typical price
            hlc3 = (high + low + close) / 3
            
            # Price direction
            dm = hlc3.diff()
            
            # Volume Force
            vf = volume * np.where(dm > 0, 1, np.where(dm < 0, -1, 0))
            
            # Klinger line
            fast_ema = vf.ewm(span=fast, adjust=False).mean()
            slow_ema = vf.ewm(span=slow, adjust=False).mean()
            klinger = fast_ema - slow_ema
            
            # Signal line
            signal_line = klinger.ewm(span=signal, adjust=False).mean()
            
            return klinger, signal_line
            
        except Exception as e:
            logger.error(f"Error calculating Klinger Oscillator: {e}")
            empty_series = pd.Series(index=high.index, dtype=float)
            return empty_series, empty_series
    
    @staticmethod
    def chandelier_exit(high: pd.Series, low: pd.Series, close: pd.Series,
                       period: int = 22, multiplier: float = 3.0) -> Tuple[pd.Series, pd.Series]:
        """
        Chandelier Exit indicator.
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: Period for calculation
            multiplier: ATR multiplier
            
        Returns:
            Tuple of (Long exit, Short exit)
        """
        try:
            atr = ta.volatility.AverageTrueRange(
                high=high, 
                low=low, 
                close=close, 
                window=period
            ).average_true_range()
            
            highest_high = high.rolling(window=period, min_periods=period).max()
            lowest_low = low.rolling(window=period, min_periods=period).min()
            
            long_exit = highest_high - (multiplier * atr)
            short_exit = lowest_low + (multiplier * atr)
            
            return long_exit, short_exit
            
        except Exception as e:
            logger.error(f"Error calculating Chandelier Exit: {e}")
            empty_series = pd.Series(index=high.index, dtype=float)
            return empty_series, empty_series
    
    @staticmethod
    def parabolic_sar(high: pd.Series, low: pd.Series,
                     step: float = 0.02, max_step: float = 0.2) -> pd.Series:
        """
        Parabolic SAR indicator.
        
        Args:
            high: High price series
            low: Low price series
            step: Initial acceleration factor
            max_step: Maximum acceleration factor
            
        Returns:
            Parabolic SAR values
        """
        try:
            return ta.trend.PSARIndicator(
                high=high, 
                low=low, 
                step=step, 
                max_step=max_step
            ).psar()
        except Exception as e:
            logger.error(f"Error calculating Parabolic SAR: {e}")
            return pd.Series(index=high.index, dtype=float)
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, 
            period: int = 14) -> pd.Series:
        """
        Average True Range (ATR).
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: Period for calculation
            
        Returns:
            ATR values
        """
        try:
            return ta.volatility.AverageTrueRange(
                high=high, 
                low=low, 
                close=close, 
                window=period
            ).average_true_range()
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return pd.Series(index=high.index, dtype=float)


class DualNNFXSystem:
    """
    Main dual NNFX trading system implementation.
    
    Combines two independent NNFX systems for enhanced signal confirmation:
    - System A: Momentum-based approach
    - System B: Trend-following approach
    
    Features:
    - Comprehensive backtesting with advanced metrics
    - Real-time signal detection with confidence scoring
    - Performance ranking and pair analysis
    - Risk management with ATR-based sizing
    """
    
    def __init__(self, bitget_api: BitgetAPI):
        """
        Initialize the dual NNFX trading system.
        
        Args:
            bitget_api: Configured Bitget API instance
        """
        self.api = bitget_api
        self.indicators = NNFXIndicators()
        logger.info("Dual NNFX System initialized")
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all indicators for both trading systems.
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            DataFrame with all indicators calculated
        """
        try:
            data = df.copy()
            
            # Validate input data
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # System A Indicators (Momentum-Based)
            data['tema'] = self.indicators.tema(data['close'], 21)
            data['cci'] = self.indicators.cci(data['high'], data['low'], data['close'], 14)
            data['elder_fi'] = self.indicators.elder_force_index(data['close'], data['volume'], 13)
            data['chandelier_long'], data['chandelier_short'] = self.indicators.chandelier_exit(
                data['high'], data['low'], data['close'], 22, 3.0
            )
            
            # System B Indicators (Trend-Following)
            data['kijun_sen'] = self.indicators.kijun_sen(data['high'], data['low'], 26)
            data['williams_r'] = self.indicators.williams_r(
                data['high'], data['low'], data['close'], 14
            )
            data['klinger'], data['klinger_signal'] = self.indicators.klinger_oscillator(
                data['high'], data['low'], data['close'], data['volume'], 34, 55, 13
            )
            data['psar'] = self.indicators.parabolic_sar(data['high'], data['low'], 0.02, 0.2)
            
            # Risk Management
            data['atr'] = self.indicators.atr(data['high'], data['low'], data['close'], 14)
            
            return data
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return df # Return original df on error to allow further processing if needed, or pd.DataFrame()
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals for both systems.
        
        Args:
            data: DataFrame with calculated indicators
            
        Returns:
            DataFrame with signal columns added
        """
        try:
            df = data.copy()
            
            # System A Signals (Momentum-Based)
            df['system_a_baseline'] = np.where(
                df['close'] > df['tema'], 1,
                np.where(df['close'] < df['tema'], -1, 0)
            )
            df['system_a_confirmation'] = np.where(
                df['cci'] > 0, 1,
                np.where(df['cci'] < 0, -1, 0)
            )
            df['system_a_volume'] = np.where(
                df['elder_fi'] > 0, 1,
                np.where(df['elder_fi'] < 0, -1, 0)
            )
            
            # System B Signals (Trend-Following)
            df['system_b_baseline'] = np.where(
                df['close'] > df['kijun_sen'], 1,
                np.where(df['close'] < df['kijun_sen'], -1, 0)
            )
            df['system_b_confirmation'] = np.where(
                # Corrected Williams %R logic: > -80 for bullish, < -20 for bearish (oversold/overbought)
                # For NNFX confirmation: bullish if not oversold, bearish if not overbought.
                # Standard Williams %R: -20 (overbought), -80 (oversold)
                # Bullish confirmation: W%R > -80 (i.e., not in deep oversold)
                # Bearish confirmation: W%R < -20 (i.e., not in deep overbought)
                df['williams_r'] > -80, 1, # Potentially 1 if above -80 (not oversold)
                np.where(df['williams_r'] < -20, -1, 0) # Potentially -1 if below -20 (not overbought)
            )
            df['system_b_volume'] = np.where(
                df['klinger'] > 0, 1,
                np.where(df['klinger'] < 0, -1, 0)
            )
            
            # Combined signals (both systems must agree)
            df['long_signal'] = (
                (df['system_a_baseline'] == 1) &
                (df['system_a_confirmation'] == 1) &
                (df['system_a_volume'] == 1) &
                (df['system_b_baseline'] == 1) &
                (df['system_b_confirmation'] == 1) & # Check if this logic is as intended
                (df['system_b_volume'] == 1)
            ).astype(bool) # Ensure boolean type
            
            df['short_signal'] = (
                (df['system_a_baseline'] == -1) &
                (df['system_a_confirmation'] == -1) &
                (df['system_a_volume'] == -1) &
                (df['system_b_baseline'] == -1) &
                (df['system_b_confirmation'] == -1) & # Check if this logic is as intended
                (df['system_b_volume'] == -1)
            ).astype(bool) # Ensure boolean type
            
            # Exit signals
            df['long_exit'] = (
                (df['close'] < df['chandelier_long']) |
                (df['close'] < df['psar'])
            ).astype(bool) # Ensure boolean type
            df['short_exit'] = (
                (df['close'] > df['chandelier_short']) |
                (df['close'] > df['psar'])
            ).astype(bool) # Ensure boolean type
            
            return df
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return data # Return original data on error
    
    def backtest_pair(self, symbol: str, start_date: str = None,
                     risk_per_trade: float = 0.015) -> Dict:
        """
        Backtest strategy on a single trading pair.
        
        Args:
            symbol: Trading pair symbol
            start_date: Start date for backtesting (optional)
            risk_per_trade: Risk percentage per trade
            
        Returns:
            Dictionary with backtest results and metrics
        """
        logger.info(f"Backtesting {symbol}...")
        
        try:
            # Fetch data
            df = self.api.get_klines(symbol, "4H", 1000) # Increased default limit for robust backtest
            
            if df.empty:
                logger.warning(f"No data available for {symbol}")
                return {"symbol": symbol, "error": "No data available"}
            
            # The check for NaN/inf on df.values was the source of the original error.
            # After get_klines modification, df should only contain numeric columns.
            # So, this check should now work or be unnecessary if get_klines guarantees cleanliness.
            # We can be more specific by checking only numeric columns if paranoid.
            numeric_cols = df.select_dtypes(include=np.number).columns
            if df[numeric_cols].isnull().values.any() or np.isinf(df[numeric_cols].values).any():
                logger.warning(f"Data for {symbol} (from get_klines) contains NaN or inf values in numeric columns.")
                # This might indicate an issue in get_klines if it's supposed to clean these.
                # However, get_klines does a dropna after processing.
                return {"symbol": symbol, "error": "Invalid data (NaN or inf after get_klines)"}

            if len(df) < 100: # Increased minimum candles for meaningful backtest
                logger.warning(f"Insufficient data for {symbol}: {len(df)} candles")
                return {"symbol": symbol, "error": "Insufficient data"}
            
            df = self.calculate_indicators(df)
            df = self.generate_signals(df)
            
            # Remove rows with NaN values that might have been introduced by indicators
            df = df.dropna()
            
            if len(df) < 50: # Minimum data after indicator calculation
                logger.warning(f"Insufficient clean data for {symbol} after processing indicators: {len(df)} rows.")
                return {"symbol": symbol, "error": "Insufficient clean data after indicators"}
            
            trades = []
            position = None
            equity_curve = []
            current_equity = 10000  # Starting equity
            
            for i in range(len(df)):
                row = df.iloc[i]
                
                if position is None:
                    if row['long_signal'] and not pd.isna(row['atr']) and row['atr'] > 0: # Ensure ATR is valid
                        stop_loss = row['close'] - (2 * row['atr'])
                        take_profit = row['close'] + (3 * row['atr'])
                        risk_amount = current_equity * risk_per_trade
                        position_size = risk_amount / (2 * row['atr']) if row['atr'] > 0 else 0
                        
                        if position_size > 0:
                            position = {
                                'type': 'long', 'entry_price': row['close'], 'entry_time': row.name,
                                'stop_loss': stop_loss, 'take_profit': take_profit,
                                'atr': row['atr'], 'position_size': position_size, 'risk_amount': risk_amount
                            }
                        
                    elif row['short_signal'] and not pd.isna(row['atr']) and row['atr'] > 0: # Ensure ATR is valid
                        stop_loss = row['close'] + (2 * row['atr'])
                        take_profit = row['close'] - (3 * row['atr'])
                        risk_amount = current_equity * risk_per_trade
                        position_size = risk_amount / (2 * row['atr']) if row['atr'] > 0 else 0

                        if position_size > 0:
                            position = {
                                'type': 'short', 'entry_price': row['close'], 'entry_time': row.name,
                                'stop_loss': stop_loss, 'take_profit': take_profit,
                                'atr': row['atr'], 'position_size': position_size, 'risk_amount': risk_amount
                            }
                
                elif position is not None:
                    exit_trade = False
                    exit_reason = ""
                    
                    if position['type'] == 'long':
                        if row['close'] <= position['stop_loss']: exit_trade, exit_reason = True, "Stop Loss"
                        elif row['close'] >= position['take_profit']: exit_trade, exit_reason = True, "Take Profit"
                        elif row['long_exit']: exit_trade, exit_reason = True, "Exit Signal"
                    
                    elif position['type'] == 'short':
                        if row['close'] >= position['stop_loss']: exit_trade, exit_reason = True, "Stop Loss"
                        elif row['close'] <= position['take_profit']: exit_trade, exit_reason = True, "Take Profit"
                        elif row['short_exit']: exit_trade, exit_reason = True, "Exit Signal"
                    
                    if exit_trade:
                        pnl_pips = (row['close'] - position['entry_price']) if position['type'] == 'long' else (position['entry_price'] - row['close'])
                        pnl_r = pnl_pips / (2 * position['atr']) if position['atr'] > 0 else 0 # Prevent division by zero if ATR was 0
                        pnl_dollar = pnl_pips * position['position_size']
                        current_equity += pnl_dollar
                        
                        trades.append({
                            'symbol': symbol, 'type': position['type'], 'entry_time': position['entry_time'],
                            'exit_time': row.name, 'entry_price': position['entry_price'], 'exit_price': row['close'],
                            'pnl_pips': pnl_pips, 'pnl_r': pnl_r, 'pnl_dollar': pnl_dollar,
                            'exit_reason': exit_reason, 'atr': position['atr'], 'equity': current_equity
                        })
                        position = None
                
                equity_curve.append({'timestamp': row.name, 'equity': current_equity, 'in_position': position is not None})
            
            if trades:
                trades_df = pd.DataFrame(trades)
                total_trades = len(trades_df)
                winning_trades = trades_df[trades_df['pnl_r'] > 0]
                win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
                avg_win = winning_trades['pnl_r'].mean() if len(winning_trades) > 0 else 0
                avg_loss = trades_df[trades_df['pnl_r'] < 0]['pnl_r'].mean() if len(trades_df[trades_df['pnl_r'] < 0]) > 0 else 0 # Ensure avg_loss is negative or zero
                profit_factor = abs(winning_trades['pnl_r'].sum() / trades_df[trades_df['pnl_r'] < 0]['pnl_r'].sum()) if trades_df[trades_df['pnl_r'] < 0]['pnl_r'].sum() != 0 else float('inf')
                total_return_r = trades_df['pnl_r'].sum()
                total_return_pct = ((current_equity - 10000) / 10000) * 100
                
                return {
                    'symbol': symbol, 'total_trades': total_trades, 'win_rate': win_rate,
                    'avg_win': avg_win, 'avg_loss': avg_loss, 'profit_factor': profit_factor,
                    'total_return_r': total_return_r, 'total_return_pct': total_return_pct,
                    'max_consecutive_losses': self._calculate_max_consecutive_losses(trades_df),
                    'max_drawdown': self._calculate_max_drawdown(equity_curve),
                    'sharpe_ratio': self._calculate_sharpe_ratio(trades_df),
                    'sortino_ratio': self._calculate_sortino_ratio(trades_df),
                    'var_95': np.percentile(trades_df['pnl_r'], 5) if not trades_df.empty else 0,
                    'max_loss': trades_df['pnl_r'].min() if not trades_df.empty else 0,
                    'final_equity': current_equity, 'trades': trades_df, 'equity_curve': equity_curve
                }
            else:
                logger.info(f"No trades generated for {symbol}")
                return {
                    'symbol': symbol, 'total_trades': 0, 'win_rate': 0, 'avg_win': 0, 'avg_loss': 0,
                    'profit_factor': 0, 'total_return_r': 0, 'total_return_pct': 0,
                    'max_consecutive_losses': 0, 'max_drawdown': self._calculate_max_drawdown(equity_curve), # Max DD can still be calculated
                    'sharpe_ratio': 0, 'sortino_ratio': 0, 'var_95': 0, 'max_loss': 0,
                    'final_equity': current_equity, 'trades': pd.DataFrame(), 'equity_curve': equity_curve
                }
                
        except Exception as e:
            logger.error(f"Error backtesting {symbol}: {e}", exc_info=True) # Add exc_info for traceback
            return {"symbol": symbol, "error": str(e)}
    
    def _calculate_max_consecutive_losses(self, trades_df: pd.DataFrame) -> int:
        if trades_df.empty: return 0
        consecutive_losses, max_consecutive = 0, 0
        for pnl_r in trades_df['pnl_r']:
            if pnl_r < 0:
                consecutive_losses += 1
                max_consecutive = max(max_consecutive, consecutive_losses)
            else:
                consecutive_losses = 0
        return max_consecutive
    
    def _calculate_max_drawdown(self, equity_curve: List[Dict]) -> float:
        if not equity_curve: return 0
        equity_values = [point['equity'] for point in equity_curve]
        if not equity_values: return 0
        peak, max_dd = equity_values[0], 0
        for equity in equity_values:
            if equity > peak: peak = equity
            if peak == 0: continue # Avoid division by zero if equity somehow hits 0
            drawdown = (peak - equity) / peak
            max_dd = max(max_dd, drawdown)
        return max_dd * 100
    
    def _calculate_sharpe_ratio(self, trades_df: pd.DataFrame, risk_free_rate: float = 0.0) -> float:
        if trades_df.empty or len(trades_df['pnl_r']) < 2: return 0
        returns = trades_df['pnl_r'] # Using R-multiple returns
        if returns.std() == 0: return float('inf') if returns.mean() > risk_free_rate else 0 # Handle zero std dev
        # Assuming R-multiples are per trade, and N trades per year
        # For 4H, approx 2190 candles/year. If avg 1 trade/week = 52 trades/year.
        # This annualization factor is highly dependent on trading frequency.
        # Let's assume daily returns for simplicity, or just calculate non-annualized Sharpe for R-multiples
        # For R-multiples, perhaps a simpler (Mean R / Std Dev R) is more direct.
        # If we stick to annualization:
        # Assuming returns are per trade, and we need to estimate trades per year.
        # The provided factor sqrt(252*6) seems too high for 4H trades if not day trading.
        # Let's use a more conservative estimate or simply calculate Sharpe on the R-multiples directly.
        # For now, let's assume 'returns' are "per period" returns related to the strategy's trades.
        # If 'pnl_r' is daily R-multiple return, then sqrt(252) would be appropriate.
        # Since pnl_r is per trade, and trades are not daily, a simpler Sharpe could be used.
        # Or, estimate trades per year. For 4H candles, 6 per day. Max 252*6 candles.
        # If a trade lasts multiple candles, this changes.
        # The original code's annualization_factor = np.sqrt(252 * 6) might be for a different context.
        # Let's use a simpler Sharpe for R-multiples, or a placeholder for annualization.
        # Placeholder for now, as true annualization depends on trade frequency.
        # For simplicity, if we assume trades_df['pnl_r'] are independent samples of trade outcomes:
        sharpe = (returns.mean() - risk_free_rate) / returns.std()
        # To annualize, we'd need average holding period or trades per year.
        # Let's assume an average of N_trades_per_year for annualization.
        # If backtest covers T days, and has M trades, trades_per_day = M/T. trades_per_year = (M/T)*252
        # For now, let's keep it simpler as the current annualization factor seems arbitrary without more context.
        # We can calculate an approximate number of trading periods in a year for 4H.
        # periods_per_year = 365 * (24/4) = 365 * 6 = 2190
        # If returns.mean() and returns.std() are for each 4H period where a trade could happen:
        # annualization_factor = np.sqrt(periods_per_year) # This would be if pnl_r was per-period return.
        # Since pnl_r is per trade, the number of trades per year is key.
        # Let's use a simplified approach for now or the original factor if it's standard for this system.
        # Given the previous context, let's try to infer trades_per_year:
        if len(trades_df) > 1 and trades_df['exit_time'].iloc[-1] != trades_df['entry_time'].iloc[0]:
             total_duration_days = (trades_df['exit_time'].iloc[-1] - trades_df['entry_time'].iloc[0]).total_seconds() / (24*3600)
             if total_duration_days > 0:
                 trades_per_year = (len(trades_df) / total_duration_days) * 252 # Assuming 252 trading days
                 annualization_factor = np.sqrt(trades_per_year) if trades_per_year > 0 else 1.0
             else: # single day of trades or very short period
                 annualization_factor = np.sqrt(len(trades_df)) # Rough estimate
        else: # not enough data for duration based annualization
            annualization_factor = 1.0 # No annualization if not enough data
        
        return sharpe * annualization_factor

    def _calculate_sortino_ratio(self, trades_df: pd.DataFrame, risk_free_rate: float = 0.0) -> float:
        if trades_df.empty or len(trades_df['pnl_r']) < 2: return 0
        returns = trades_df['pnl_r']
        downside_returns = returns[returns < risk_free_rate] # Consider returns below target/risk-free
        if len(downside_returns) == 0: return float('inf') if returns.mean() > risk_free_rate else 0
        downside_std = downside_returns.std()
        if downside_std == 0: return float('inf') if returns.mean() > risk_free_rate else 0

        # Similar annualization logic as Sharpe
        if len(trades_df) > 1 and trades_df['exit_time'].iloc[-1] != trades_df['entry_time'].iloc[0]:
             total_duration_days = (trades_df['exit_time'].iloc[-1] - trades_df['entry_time'].iloc[0]).total_seconds() / (24*3600)
             if total_duration_days > 0:
                 trades_per_year = (len(trades_df) / total_duration_days) * 252
                 annualization_factor = np.sqrt(trades_per_year) if trades_per_year > 0 else 1.0
             else:
                 annualization_factor = np.sqrt(len(trades_df))
        else:
            annualization_factor = 1.0
            
        sortino = (returns.mean() - risk_free_rate) / downside_std
        return sortino * annualization_factor
    
    def scan_pairs(self, symbols: List[str] = None, save_results: bool = True) -> pd.DataFrame:
        if symbols is None:
            symbols = self.api.get_symbols()
            if not symbols:
                logger.warning("No symbols found from API for scanning.")
                return pd.DataFrame()
            logger.info(f"Auto-detected {len(symbols)} symbols for scanning")
        
        logger.info(f"Starting backtest scan on {len(symbols)} pairs...")
        results, failed_pairs = [], []
        
        for i, symbol in enumerate(symbols):
            try:
                logger.info(f"Processing {symbol} ({i+1}/{len(symbols)})")
                result = self.backtest_pair(symbol)
                
                if 'error' not in result and result.get('total_trades', 0) >= 3: # Min 3 trades for meaningful stats
                    results.append({
                        'symbol': result['symbol'], 'total_trades': result['total_trades'],
                        'win_rate': result['win_rate'], 'profit_factor': result['profit_factor'],
                        'total_return_r': result['total_return_r'], 'total_return_pct': result['total_return_pct'],
                        'max_drawdown': result['max_drawdown'], 'sharpe_ratio': result['sharpe_ratio'],
                        'sortino_ratio': result['sortino_ratio'], 'max_consecutive_losses': result['max_consecutive_losses'],
                        'var_95': result['var_95'], 'max_loss': result['max_loss'],
                        'score': self._calculate_score(result)
                    })
                else:
                    failed_pairs.append(symbol)
                    error_msg = result.get('error', f"Insufficient trades: {result.get('total_trades',0)}")
                    logger.warning(f"Failed to process {symbol}: {error_msg}")
                
                time.sleep(0.1) # Brief pause
                
            except Exception as e:
                logger.error(f"Unexpected error processing {symbol} during scan: {e}", exc_info=True)
                failed_pairs.append(symbol)
        
        logger.info(f"Scan complete. Successfully processed: {len(results)}, Failed/Skipped: {len(failed_pairs)}")
        
        if results:
            df = pd.DataFrame(results).sort_values('score', ascending=False)
            if save_results:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                Path("results").mkdir(parents=True, exist_ok=True) # Ensure results dir exists
                results_file = f"results/scan_results_{timestamp}.csv"
                df.to_csv(results_file, index=False)
                logger.info(f"Results saved to {results_file}")
                self._save_scan_summary(df, symbols, failed_pairs, timestamp)
            return df
        else:
            logger.warning("No successful backtests completed in scan or none met criteria.")
            if save_results and symbols: # Save summary even if no results, to log failed pairs
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                self._save_scan_summary(pd.DataFrame(), symbols, failed_pairs, timestamp)
            return pd.DataFrame()

    def _save_scan_summary(self, df: pd.DataFrame, symbols_scanned: List[str], 
                          failed_pairs: List[str], timestamp: str):
        Path("results").mkdir(parents=True, exist_ok=True)
        summary_file = f"results/scan_summary_{timestamp}.txt"
        
        with open(summary_file, 'w') as f:
            f.write("=" * 60 + "\nNNFX Bot - Dual System Backtest Results\n" + "=" * 60 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total pairs attempted: {len(symbols_scanned)}\n")
            f.write(f"Successful backtests (met criteria): {len(df)}\n")
            f.write(f"Failed or skipped pairs: {len(failed_pairs)}\n\n")
            
            if not df.empty:
                f.write("TOP 10 PERFORMING PAIRS (if available)\n" + "-" * 40 + "\n")
                for i, (_, row) in enumerate(df.head(10).iterrows(), 1):
                    f.write(f"{i:2d}. {row['symbol']:<12} Score: {row['score']:6.2f} | WR: {row['win_rate']:5.1%} | PF: {row['profit_factor']:5.2f} | Ret: {row['total_return_pct']:6.1f}% | DD: {row['max_drawdown']:5.1f}% | Trades: {row['total_trades']:3d}\n")
                
                f.write("\nPERFORMANCE STATISTICS (for successful pairs)\n" + "-" * 40 + "\n")
                f.write(f"Average Win Rate: {df['win_rate'].mean():5.1%}\n")
                f.write(f"Average Profit Factor: {df['profit_factor'][np.isfinite(df['profit_factor'])].mean():5.2f}\n") # Handle inf PF
                f.write(f"Average Return: {df['total_return_pct'].mean():6.1f}%\n")
                f.write(f"Average Max Drawdown: {df['max_drawdown'].mean():5.1f}%\n")
                f.write(f"Average Sharpe Ratio: {df['sharpe_ratio'][np.isfinite(df['sharpe_ratio'])].mean():5.2f}\n") # Handle inf Sharpe
                
                f.write("\nFILTERING HIGHLIGHTS (for successful pairs)\n" + "-" * 40 + "\n")
                f.write(f"Pairs with >60% win rate: {len(df[df['win_rate'] > 0.6])}\n")
                f.write(f"Pairs with >2.0 profit factor: {len(df[df['profit_factor'] > 2.0])}\n")
                f.write(f"Pairs with >10% return: {len(df[df['total_return_pct'] > 10])}\n")
                f.write(f"Pairs with <15% max drawdown: {len(df[df['max_drawdown'] < 15])}\n")
                f.write(f"Pairs with positive Sharpe: {len(df[df['sharpe_ratio'] > 0])}\n")
                
                f.write("\nRISK ANALYSIS (for successful pairs)\n" + "-" * 40 + "\n")
                if not df.empty:
                    f.write(f"Worst performing pair (by return %): {df.iloc[-1]['symbol']} ({df.iloc[-1]['total_return_pct']:.1f}%)\n")
                    f.write(f"Highest drawdown: {df['max_drawdown'].max():.1f}%\n")
                    f.write(f"Most consecutive losses: {df['max_consecutive_losses'].max()}\n")
                    f.write(f"Worst single trade (VaR 95% based on PnL R): {df['var_95'].min():.2f}R\n")
            
            if failed_pairs:
                f.write(f"\nFAILED OR SKIPPED PAIRS ({len(failed_pairs)})\n" + "-" * 40 + "\n")
                for pair in failed_pairs: f.write(f"  {pair}\n")
        
        logger.info(f"Detailed summary saved to {summary_file}")

    def _calculate_score(self, result: Dict) -> float:
        if result.get('total_trades', 0) < 3: return 0 # Use .get for safety
        
        win_rate_score = result.get('win_rate', 0) * 100
        profit_factor = result.get('profit_factor', 0)
        profit_factor_score = min(profit_factor * 20, 100) if np.isfinite(profit_factor) else (50 if profit_factor > 0 else 0) # Handle inf PF
        return_score = min(result.get('total_return_pct', 0) * 2, 100)
        trade_frequency_score = min(result.get('total_trades', 0) * 0.5, 100) # Adjusted weight for trade frequency
        sharpe = result.get('sharpe_ratio', 0)
        sharpe_score = min(max(sharpe * 20, -50), 100) if np.isfinite(sharpe) else (50 if sharpe > 0 else -50) # Handle inf Sharpe, allow negative
        
        drawdown_penalty = result.get('max_drawdown', 100) * 1.5 # Increased penalty sensitivity
        consecutive_loss_penalty = result.get('max_consecutive_losses', 20) * 3 # Increased penalty
        var_penalty = abs(result.get('var_95', -5)) * 5 # Increased penalty for large VaR
        
        score = (
            win_rate_score * 0.25 +
            profit_factor_score * 0.25 +
            return_score * 0.20 +
            trade_frequency_score * 0.10 + # Reduced weight
            sharpe_score * 0.20 -          # Increased weight
            drawdown_penalty * 0.05 -      # Adjusted weights for penalties
            consecutive_loss_penalty * 0.03 -
            var_penalty * 0.02
        )
        return max(score, -100) # Allow negative scores to show truly bad performance
    
    def get_current_signals(self, symbols: List[str]) -> pd.DataFrame:
        logger.info(f"Getting current signals for {len(symbols)} pairs...")
        signals = []
        
        for symbol in symbols:
            try:
                df = self.api.get_klines(symbol, "4H", 200) # Fetch enough data for indicators
                if df.empty or len(df) < 50: # Need enough data for indicators
                    logger.warning(f"Insufficient data for {symbol} to get current signals.")
                    continue

                df = self.calculate_indicators(df)
                df = self.generate_signals(df)
                
                if df.empty or len(df) < 2: # Need at least 2 rows for latest and current
                    logger.warning(f"Not enough processed data for {symbol} to get current signals.")
                    continue
                    
                latest, current = df.iloc[-2], df.iloc[-1] # Latest complete, current forming
                signal_type, confidence = "NONE", 0
                
                if latest['long_signal']:
                    signal_type, confidence = "LONG", self._calculate_signal_confidence(df.iloc[-5:], 'long')
                elif latest['short_signal']:
                    signal_type, confidence = "SHORT", self._calculate_signal_confidence(df.iloc[-5:], 'short')
                
                system_a_score = (latest.get('system_a_baseline',0) + latest.get('system_a_confirmation',0) + latest.get('system_a_volume',0)) / 3
                system_b_score = (latest.get('system_b_baseline',0) + latest.get('system_b_confirmation',0) + latest.get('system_b_volume',0)) / 3
                
                atr_val = latest.get('atr', 0)
                stop_distance = 2 * atr_val if not pd.isna(atr_val) else 0
                tp_distance = 3 * atr_val if not pd.isna(atr_val) else 0
                risk_reward = tp_distance / stop_distance if stop_distance > 0 else 0
                
                signals.append({
                    'symbol': symbol, 'signal': signal_type, 'confidence': confidence,
                    'price': latest['close'], 'atr': atr_val,
                    'stop_loss_distance': stop_distance, 'take_profit_distance': tp_distance,
                    'risk_reward_ratio': risk_reward, 'system_a_score': system_a_score,
                    'system_b_score': system_b_score, 'overall_strength': (system_a_score + system_b_score) / 2,
                    'trend_direction': 'Bullish' if latest['close'] > latest.get('tema',0) else ('Bearish' if latest['close'] < latest.get('tema',0) else 'Neutral'),
                    'momentum': 'Positive' if latest.get('cci',0) > 0 else ('Negative' if latest.get('cci',0) < 0 else 'Neutral'),
                    'volume_pressure': 'Buying' if latest.get('elder_fi',0) > 0 else ('Selling' if latest.get('elder_fi',0) < 0 else 'Neutral'),
                    'timestamp': latest.name, 'current_candle_time': current.name
                })
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Error getting signals for {symbol}: {e}", exc_info=True)
        
        df_signals = pd.DataFrame(signals)
        if not df_signals.empty:
            df_signals = df_signals.sort_values(['signal', 'confidence', 'overall_strength'], ascending=[True, False, False])
        return df_signals
    
    def _calculate_signal_confidence(self, recent_data: pd.DataFrame, signal_type: str) -> float:
        if len(recent_data) < 3: return 0.5 # Default confidence for less data
        
        # Ensure columns exist, default to neutral if not
        if signal_type == 'long':
            signal_support = recent_data['long_signal'].sum() / len(recent_data) if 'long_signal' in recent_data else 0
            trend_consistency = (recent_data['system_a_baseline'] > 0).mean() if 'system_a_baseline' in recent_data else 0.5
            momentum_consistency = (recent_data['system_a_confirmation'] > 0).mean() if 'system_a_confirmation' in recent_data else 0.5
            volume_consistency = (recent_data['system_a_volume'] > 0).mean() if 'system_a_volume' in recent_data else 0.5
        else: # short
            signal_support = recent_data['short_signal'].sum() / len(recent_data) if 'short_signal' in recent_data else 0
            trend_consistency = (recent_data['system_a_baseline'] < 0).mean() if 'system_a_baseline' in recent_data else 0.5
            momentum_consistency = (recent_data['system_a_confirmation'] < 0).mean() if 'system_a_confirmation' in recent_data else 0.5
            volume_consistency = (recent_data['system_a_volume'] < 0).mean() if 'system_a_volume' in recent_data else 0.5
        
        confidence = (signal_support*0.4 + trend_consistency*0.25 + momentum_consistency*0.20 + volume_consistency*0.15)
        return min(max(confidence, 0.0), 1.0) # Clamp between 0 and 1
    
    def export_detailed_results(self, rankings: pd.DataFrame, 
                               current_signals: pd.DataFrame) -> str:
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            Path("results").mkdir(parents=True, exist_ok=True)
            filename = f"results/nnfx_analysis_{timestamp}.xlsx"
            
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                if not rankings.empty:
                    rankings.to_excel(writer, sheet_name='Backtest Rankings', index=False)
                else:
                    pd.DataFrame([{"message": "No ranking data available"}]).to_excel(writer, sheet_name='Backtest Rankings', index=False)

                if not current_signals.empty:
                    current_signals.to_excel(writer, sheet_name='Current Signals', index=False)
                else:
                    pd.DataFrame([{"message": "No current signals available"}]).to_excel(writer, sheet_name='Current Signals', index=False)

                if not rankings.empty:
                    summary_stats_data = {
                        'Metric': ['Total Pairs Tested', 'Average Win Rate', 'Average Profit Factor', 
                                   'Average Return (%)', 'Average Max Drawdown (%)', 'Average Sharpe Ratio',
                                   'Pairs with >60% Win Rate', 'Pairs with >2.0 Profit Factor', 'Pairs with Positive Return'],
                        'Value': [
                            len(rankings), f"{rankings['win_rate'].mean():.1%}", 
                            f"{rankings['profit_factor'][np.isfinite(rankings['profit_factor'])].mean():.2f}",
                            f"{rankings['total_return_pct'].mean():.1f}", f"{rankings['max_drawdown'].mean():.1f}",
                            f"{rankings['sharpe_ratio'][np.isfinite(rankings['sharpe_ratio'])].mean():.2f}",
                            len(rankings[rankings['win_rate'] > 0.6]), len(rankings[rankings['profit_factor'] > 2.0]),
                            len(rankings[rankings['total_return_pct'] > 0])
                        ]}
                    pd.DataFrame(summary_stats_data).to_excel(writer, sheet_name='Summary Statistics', index=False)
                
                if not current_signals.empty:
                    active_signals = current_signals[current_signals['signal'] != 'NONE']
                    if not active_signals.empty:
                        signal_summary = active_signals.groupby('signal').agg(
                            count=('symbol', 'count'),
                            avg_confidence=('confidence', 'mean'),
                            avg_strength=('overall_strength', 'mean')
                        )
                        signal_summary.to_excel(writer, sheet_name='Signal Summary')
            
            logger.info(f"Detailed results exported to {filename}")
            return filename
        except Exception as e:
            logger.error(f"Error exporting results: {e}", exc_info=True)
            return ""


def run_comprehensive_analysis(api_config: Dict = None, test_pairs: List[str] = None) -> Dict:
    logger.info("=" * 60 + "\nNNFX Bot - Comprehensive Analysis Started\n" + "=" * 60)
    
    try:
        api = BitgetAPI(**api_config) if api_config else BitgetAPI()
        system = DualNNFXSystem(api)
        max_pairs = 20 # VPS resource limit or practical limit
        
        if test_pairs is None:
            all_symbols = api.get_symbols()
            if not all_symbols:
                logger.warning("No symbols auto-detected. Using default list.")
                test_symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'SOLUSDT', 'LINKUSDT'] # Fallback
            else:
                test_symbols = all_symbols[:max_pairs]
        else:
            test_symbols = test_pairs[:max_pairs]
        
        logger.info(f"Testing {len(test_symbols)} pairs: {test_symbols}")
        start_time = time.time()
        rankings = system.scan_pairs(test_symbols, save_results=True)
        scan_duration = time.time() - start_time
        logger.info(f"Scan completed in {scan_duration:.1f} seconds")
        
        results_payload = {'success': False, 'scan_duration': scan_duration}

        if not rankings.empty:
            logger.info("\n" + "=" * 50 + "\nTOP 5 PERFORMING PAIRS\n" + "=" * 50)
            top_5 = rankings.head()
            for i, (_, row) in enumerate(top_5.iterrows(), 1):
                logger.info(f"{i}. {row['symbol']:<10} | Score: {row.get('score',0):6.2f} | WR: {row.get('win_rate',0):5.1%} | PF: {row.get('profit_factor',0):5.2f} | Ret: {row.get('total_return_pct',0):6.1f}% | DD: {row.get('max_drawdown',0):5.1f}%")
            
            top_pairs_for_signals = top_5['symbol'].tolist()
            logger.info(f"\nAnalyzing current signals for top {len(top_pairs_for_signals)} pairs...")
            current_signals = system.get_current_signals(top_pairs_for_signals)
            active_signals = current_signals[current_signals['signal'] != 'NONE']
            
            if not active_signals.empty:
                logger.info("\n" + "=" * 50 + "\nACTIVE TRADING SIGNALS\n" + "=" * 50)
                for _, sig in active_signals.iterrows():
                    logger.info(f"{sig['symbol']:<10} | {sig['signal']:<5} | Price: {sig.get('price',0):10.6f} | Conf: {sig.get('confidence',0):5.1%} | R:R: {sig.get('risk_reward_ratio',0):4.1f}")
            else:
                logger.info("\nNo active trading signals found for top pairs.")
            
            export_file = system.export_detailed_results(rankings, current_signals)
            logger.info("\n" + "=" * 50 + "\nSYSTEM PERFORMANCE SUMMARY (Ranked Pairs)\n" + "=" * 50)
            logger.info(f"Pairs successfully ranked: {len(rankings)}")
            logger.info(f"Average win rate: {rankings['win_rate'].mean():.1%}")
            logger.info(f"Average profit factor: {rankings['profit_factor'][np.isfinite(rankings['profit_factor'])].mean():.2f}")
            logger.info(f"Average return: {rankings['total_return_pct'].mean():.1f}%")
            
            results_payload.update({
                'success': True, 'rankings': rankings.to_dict('records'), 
                'current_signals': current_signals.to_dict('records'), 'export_file': export_file,
                'summary': {
                    'pairs_tested_successfully': len(rankings),
                    'avg_win_rate': rankings['win_rate'].mean(),
                    'avg_profit_factor': rankings['profit_factor'][np.isfinite(rankings['profit_factor'])].mean(),
                    'avg_return_pct': rankings['total_return_pct'].mean(),
                    'top_performer_symbol': rankings.iloc[0]['symbol'] if not rankings.empty else None,
                    'active_signals_count': len(active_signals)
                }
            })
        else:
            logger.warning("No pairs passed ranking criteria from the scan.")
            results_payload['error'] = 'No successful backtests or none met ranking criteria.'
            # Try to export empty files if configured, or handle appropriately
            system.export_detailed_results(pd.DataFrame(), pd.DataFrame()) # Save empty report structure

        return results_payload
            
    except Exception as e:
        logger.error(f"Critical error in comprehensive analysis: {e}", exc_info=True)
        return {'success': False, 'error': str(e), 'scan_duration': time.time() - start_time if 'start_time' in locals() else 0}


def cleanup_old_files(days_old: int = 7, max_results_to_keep: int = 20):
    import glob
    cutoff_date = datetime.now() - timedelta(days=days_old)
    cleaned_count = 0
    
    Path("data").mkdir(exist_ok=True)
    for cache_file in glob.glob("data/*.csv"):
        if os.path.getmtime(cache_file) < cutoff_date.timestamp():
            try:
                os.remove(cache_file)
                logger.info(f"Cleaned old cache file: {cache_file}")
                cleaned_count += 1
            except Exception as e:
                logger.warning(f"Could not remove old cache file {cache_file}: {e}")

    Path("results").mkdir(exist_ok=True)
    result_files = sorted(glob.glob("results/*"), key=os.path.getmtime)
    if len(result_files) > max_results_to_keep:
        for old_file in result_files[:-max_results_to_keep]:
            try:
                os.remove(old_file)
                logger.info(f"Cleaned old result file: {old_file}")
                cleaned_count += 1
            except Exception as e:
                logger.warning(f"Could not remove old result file {old_file}: {e}")
    
    logger.info(f"Cleanup completed. Removed {cleaned_count} old files.")


def get_system_status():
    try:
        import psutil # Optional dependency
        status = {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': psutil.cpu_percent(interval=0.1), # Non-blocking
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
            'disk_free_gb': psutil.disk_usage('/').free / (1024**3)
        }
        logger.info(f"System Status - CPU: {status['cpu_percent']:.1f}%, Mem: {status['memory_percent']:.1f}%, Disk: {status['disk_percent']:.1f}%")
        return status
    except ImportError:
        logger.warning("psutil not installed, cannot get detailed system status.")
        return {'error': 'psutil not installed'}
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        return {'error': str(e)}


if __name__ == "__main__":
    # Ensure necessary directories exist at the very start
    for_dirs = ["logs", "data", "results", "config"]
    for d in for_dirs: Path(d).mkdir(parents=True, exist_ok=True)

    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file_path = Path(f"logs/system_run_{run_timestamp}.log")
    
    # Setup file logging specifically for this run
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO) # Or DEBUG if more verbosity is needed in file
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add handler to the root logger or your specific logger
    # If adding to root, be mindful if other modules also use root logger
    logging.getLogger().addHandler(file_handler) # Add to root logger
    # Alternatively, if you want to keep console DEBUG and file INFO for your logger only:
    # logger.addHandler(file_handler) # Add to __name__ logger

    logger.info(f"NNFX Bot Trading System Run: {run_timestamp}")
    logger.info(f"Logging to console (DEBUG level) and to file: {log_file_path} (INFO level)")

    api_config = None
    api_config_file = Path("config/api_config.json")
    if api_config_file.exists():
        try:
            with open(api_config_file, "r") as f:
                api_config = json.load(f)
            logger.info(f"Loaded API configuration from {api_config_file}")
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from {api_config_file}. Using default/no API key.")
        except Exception as e:
            logger.error(f"Error loading API config {api_config_file}: {e}. Using default/no API key.")
    else:
        logger.warning(f"API config file {api_config_file} not found. Using default/no API key (sandbox mode or public endpoints only).")
        # Create a dummy config file if it doesn't exist to guide the user
        try:
            with open(api_config_file, "w") as f:
                json.dump({"api_key": "YOUR_API_KEY", 
                           "secret_key": "YOUR_SECRET_KEY", 
                           "passphrase": "YOUR_PASSPHRASE",
                           "sandbox": True}, f, indent=4)
            logger.info(f"Created a dummy API config file at {api_config_file}. Please update it with your credentials.")
        except Exception as e:
            logger.error(f"Could not create dummy config file: {e}")

    # Define example pairs, or could be read from another config file
    example_pairs_to_test = [
        'BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT',
        'SOLUSDT', 'AVAXUSDT', 'MATICUSDT' 
    ]
    
    analysis_results = run_comprehensive_analysis(api_config, example_pairs_to_test)
    
    if analysis_results.get('success'):
        logger.info("Comprehensive analysis completed successfully.")
        if analysis_results.get('export_file'):
            logger.info(f"Detailed report: {analysis_results['export_file']}")
    else:
        logger.error(f"Comprehensive analysis failed: {analysis_results.get('error', 'Unknown error')}")
    
    logger.info("Performing cleanup of old files...")
    cleanup_old_files()

    logger.info("Fetching final system status...")
    get_system_status()
    
    logger.info(f"Session {run_timestamp} complete. Check log file {log_file_path} for details.")
    logging.getLogger().removeHandler(file_handler) # Clean up handler for this run
    file_handler.close()