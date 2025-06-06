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

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
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
            if os.path.exists(cache_file):
                cache_age = time.time() - os.path.getmtime(cache_file)
                if cache_age < 14400:  # 4 hours
                    try:
                        cached_df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                        logger.debug(f"Using cached data for {symbol}")
                        # Defensive checks on cached data
                        if cached_df.empty or cached_df.isnull().values.any() or np.isinf(cached_df.values).any():
                            logger.warning(f"Cached data for {symbol} is empty or contains NaN/inf values.")
                            return pd.DataFrame()
                        return cached_df
                    except Exception as e:
                        logger.warning(f"Failed to read cache for {symbol}: {e}")
            
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
                        df = pd.DataFrame(
                            data['data'], 
                            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                        )
                        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
                        df = df.set_index('timestamp')
                        df = df.astype(float)
                        df = df.sort_index()
                        logger.debug(f"Parsed DataFrame for {symbol}:\n{df.head()}")
                        
                        # Defensive checks
                        if df.empty:
                            logger.warning(f"No data returned for {symbol}")
                            return pd.DataFrame()
                        if df.isnull().values.any() or np.isinf(df.values).any():
                            logger.warning(f"Data for {symbol} contains NaN or inf values.")
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
                        continue
                        
                except requests.exceptions.RequestException as e:
                    logger.warning(f"Request failed for {symbol}, attempt {attempt + 1}: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                    continue
            
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
            
            if data.get('code') == '00000':
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
                    if s.endswith('USDT') and any(s.startswith(base) for base in major_bases)
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
            return df
    
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
                df['williams_r'] > -80, 1,
                np.where(df['williams_r'] < -20, -1, 0)
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
                (df['system_b_confirmation'] == 1) &
                (df['system_b_volume'] == 1)
            )
            
            df['short_signal'] = (
                (df['system_a_baseline'] == -1) &
                (df['system_a_confirmation'] == -1) &
                (df['system_a_volume'] == -1) &
                (df['system_b_baseline'] == -1) &
                (df['system_b_confirmation'] == -1) &
                (df['system_b_volume'] == -1)
            )
            
            # Exit signals
            df['long_exit'] = (
                (df['close'] < df['chandelier_long']) |
                (df['close'] < df['psar'])
            )
            df['short_exit'] = (
                (df['close'] > df['chandelier_short']) |
                (df['close'] > df['psar'])
            )
            
            return df
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return data
    
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
            df = self.api.get_klines(symbol, "4H", 1000)
            # Defensive checks on fetched data
            if df.empty:
                logger.warning(f"No data available for {symbol}")
                return {"symbol": symbol, "error": "No data available"}
            if df.isnull().values.any() or np.isinf(df.values).any():
                logger.warning(f"Data for {symbol} contains NaN or inf values.")
                return {"symbol": symbol, "error": "Invalid data (NaN or inf)"}
            # Validate data quality
            if len(df) < 100:
                logger.warning(f"Insufficient data for {symbol}: {len(df)} candles")
                return {"symbol": symbol, "error": "Insufficient data"}
            
            # Calculate indicators and generate signals
            df = self.calculate_indicators(df)
            df = self.generate_signals(df)
            
            # Remove rows with NaN values
            df = df.dropna()
            
            if len(df) < 50:
                logger.warning(f"Insufficient clean data for {symbol} after processing")
                return {"symbol": symbol, "error": "Insufficient clean data"}
            
            # Initialize backtesting variables
            trades = []
            position = None
            equity_curve = []
            current_equity = 10000  # Starting equity
            
            # Backtesting loop
            for i in range(len(df)):
                row = df.iloc[i]
                
                # Entry logic
                if position is None:
                    if row['long_signal'] and not pd.isna(row['atr']):
                        stop_loss = row['close'] - (2 * row['atr'])
                        take_profit = row['close'] + (3 * row['atr'])
                        
                        risk_amount = current_equity * risk_per_trade
                        position_size = risk_amount / (2 * row['atr'])
                        
                        position = {
                            'type': 'long',
                            'entry_price': row['close'],
                            'entry_time': row.name,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'atr': row['atr'],
                            'position_size': position_size,
                            'risk_amount': risk_amount
                        }
                        
                    elif row['short_signal'] and not pd.isna(row['atr']):
                        stop_loss = row['close'] + (2 * row['atr'])
                        take_profit = row['close'] - (3 * row['atr'])
                        
                        risk_amount = current_equity * risk_per_trade
                        position_size = risk_amount / (2 * row['atr'])
                        
                        position = {
                            'type': 'short',
                            'entry_price': row['close'],
                            'entry_time': row.name,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'atr': row['atr'],
                            'position_size': position_size,
                            'risk_amount': risk_amount
                        }
                
                # Exit logic
                elif position is not None:
                    exit_trade = False
                    exit_reason = ""
                    
                    if position['type'] == 'long':
                        if row['close'] <= position['stop_loss']:
                            exit_trade, exit_reason = True, "Stop Loss"
                        elif row['close'] >= position['take_profit']:
                            exit_trade, exit_reason = True, "Take Profit"
                        elif row['long_exit']:
                            exit_trade, exit_reason = True, "Exit Signal"
                    
                    elif position['type'] == 'short':
                        if row['close'] >= position['stop_loss']:
                            exit_trade, exit_reason = True, "Stop Loss"
                        elif row['close'] <= position['take_profit']:
                            exit_trade, exit_reason = True, "Take Profit"
                        elif row['short_exit']:
                            exit_trade, exit_reason = True, "Exit Signal"
                    
                    if exit_trade:
                        # Calculate P&L
                        if position['type'] == 'long':
                            pnl_pips = row['close'] - position['entry_price']
                        else:
                            pnl_pips = position['entry_price'] - row['close']
                        
                        pnl_r = pnl_pips / (2 * position['atr'])
                        pnl_dollar = pnl_pips * position['position_size']
                        current_equity += pnl_dollar
                        
                        trades.append({
                            'symbol': symbol,
                            'type': position['type'],
                            'entry_time': position['entry_time'],
                            'exit_time': row.name,
                            'entry_price': position['entry_price'],
                            'exit_price': row['close'],
                            'pnl_pips': pnl_pips,
                            'pnl_r': pnl_r,
                            'pnl_dollar': pnl_dollar,
                            'exit_reason': exit_reason,
                            'atr': position['atr'],
                            'equity': current_equity
                        })
                        
                        position = None
                
                # Record equity curve
                equity_curve.append({
                    'timestamp': row.name,
                    'equity': current_equity,
                    'in_position': position is not None
                })
            
            # Calculate comprehensive statistics
            if trades:
                trades_df = pd.DataFrame(trades)
                
                # Basic metrics
                total_trades = len(trades_df)
                winning_trades = trades_df[trades_df['pnl_r'] > 0]
                losing_trades = trades_df[trades_df['pnl_r'] < 0]
                
                win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
                avg_win = winning_trades['pnl_r'].mean() if len(winning_trades) > 0 else 0
                avg_loss = losing_trades['pnl_r'].mean() if len(losing_trades) > 0 else 0
                profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
                total_return_r = trades_df['pnl_r'].sum()
                total_return_pct = ((current_equity - 10000) / 10000) * 100
                
                # Advanced metrics
                max_consecutive_losses = self._calculate_max_consecutive_losses(trades_df)
                max_drawdown = self._calculate_max_drawdown(equity_curve)
                sharpe_ratio = self._calculate_sharpe_ratio(trades_df)
                sortino_ratio = self._calculate_sortino_ratio(trades_df)
                
                # Risk metrics
                var_95 = np.percentile(trades_df['pnl_r'], 5) if len(trades_df) > 0 else 0
                max_loss = trades_df['pnl_r'].min() if len(trades_df) > 0 else 0
                
                return {
                    'symbol': symbol,
                    'total_trades': total_trades,
                    'win_rate': win_rate,
                    'avg_win': avg_win,
                    'avg_loss': avg_loss,
                    'profit_factor': profit_factor,
                    'total_return_r': total_return_r,
                    'total_return_pct': total_return_pct,
                    'max_consecutive_losses': max_consecutive_losses,
                    'max_drawdown': max_drawdown,
                    'sharpe_ratio': sharpe_ratio,
                    'sortino_ratio': sortino_ratio,
                    'var_95': var_95,
                    'max_loss': max_loss,
                    'final_equity': current_equity,
                    'trades': trades_df,
                    'equity_curve': equity_curve
                }
            else:
                logger.info(f"No trades generated for {symbol}")
                return {
                    'symbol': symbol,
                    'total_trades': 0,
                    'win_rate': 0,
                    'avg_win': 0,
                    'avg_loss': 0,
                    'profit_factor': 0,
                    'total_return_r': 0,
                    'total_return_pct': 0,
                    'max_consecutive_losses': 0,
                    'max_drawdown': 0,
                    'sharpe_ratio': 0,
                    'sortino_ratio': 0,
                    'var_95': 0,
                    'max_loss': 0,
                    'final_equity': 10000,
                    'trades': pd.DataFrame(),
                    'equity_curve': equity_curve
                }
                
        except Exception as e:
            logger.error(f"Error backtesting {symbol}: {e}")
            return {"symbol": symbol, "error": str(e)}
    
    def _calculate_max_consecutive_losses(self, trades_df: pd.DataFrame) -> int:
        """Calculate maximum consecutive losses."""
        if trades_df.empty:
            return 0
        
        consecutive_losses = 0
        max_consecutive = 0
        
        for _, trade in trades_df.iterrows():
            if trade['pnl_r'] < 0:
                consecutive_losses += 1
                max_consecutive = max(max_consecutive, consecutive_losses)
            else:
                consecutive_losses = 0
        
        return max_consecutive
    
    def _calculate_max_drawdown(self, equity_curve: List[Dict]) -> float:
        """Calculate maximum drawdown percentage."""
        if not equity_curve:
            return 0
        
        equity_values = [point['equity'] for point in equity_curve]
        peak = equity_values[0]
        max_dd = 0
        
        for equity in equity_values:
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak
            max_dd = max(max_dd, drawdown)
        
        return max_dd * 100  # Return as percentage
    
    def _calculate_sharpe_ratio(self, trades_df: pd.DataFrame) -> float:
        """Calculate Sharpe ratio (annualized)."""
        if trades_df.empty or len(trades_df) < 2:
            return 0
        
        returns = trades_df['pnl_r']
        if returns.std() == 0:
            return 0
        
        # Approximate annualization factor for 4H trading
        annualization_factor = np.sqrt(252 * 6)  # ~6 trades per day possible
        return (returns.mean() / returns.std()) * annualization_factor
    
    def _calculate_sortino_ratio(self, trades_df: pd.DataFrame) -> float:
        """Calculate Sortino ratio (downside deviation)."""
        if trades_df.empty or len(trades_df) < 2:
            return 0
        
        returns = trades_df['pnl_r']
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return float('inf') if returns.mean() > 0 else 0
        
        annualization_factor = np.sqrt(252 * 6)
        return (returns.mean() / downside_returns.std()) * annualization_factor
    
    def scan_pairs(self, symbols: List[str] = None, save_results: bool = True) -> pd.DataFrame:
        """
        Scan multiple pairs and rank by performance.
        
        Args:
            symbols: List of symbols to scan (None for auto-detection)
            save_results: Whether to save results to files
            
        Returns:
            DataFrame with ranked pairs
        """
        if symbols is None:
            symbols = self.api.get_symbols()
            logger.info(f"Auto-detected {len(symbols)} symbols for scanning")
        
        logger.info(f"Starting backtest scan on {len(symbols)} pairs...")
        
        results = []
        failed_pairs = []
        
        for i, symbol in enumerate(symbols):
            try:
                logger.info(f"Processing {symbol} ({i+1}/{len(symbols)})")
                result = self.backtest_pair(symbol)
                
                if 'error' not in result and result['total_trades'] >= 3:
                    results.append({
                        'symbol': result['symbol'],
                        'total_trades': result['total_trades'],
                        'win_rate': result['win_rate'],
                        'profit_factor': result['profit_factor'],
                        'total_return_r': result['total_return_r'],
                        'total_return_pct': result['total_return_pct'],
                        'max_drawdown': result['max_drawdown'],
                        'sharpe_ratio': result['sharpe_ratio'],
                        'sortino_ratio': result['sortino_ratio'],
                        'max_consecutive_losses': result['max_consecutive_losses'],
                        'var_95': result['var_95'],
                        'max_loss': result['max_loss'],
                        'score': self._calculate_score(result)
                    })
                else:
                    failed_pairs.append(symbol)
                    if 'error' in result:
                        logger.warning(f"Failed to process {symbol}: {result['error']}")
                    else:
                        logger.warning(f"Insufficient trades for {symbol}: {result['total_trades']}")
                
                # Brief pause to avoid overwhelming the API
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Unexpected error processing {symbol}: {e}")
                failed_pairs.append(symbol)
        
        logger.info(f"Scan complete. Processed: {len(results)}, Failed: {len(failed_pairs)}")
        
        if results:
            df = pd.DataFrame(results)
            df = df.sort_values('score', ascending=False)
            
            if save_results:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                results_file = f"results/scan_results_{timestamp}.csv"
                df.to_csv(results_file, index=False)
                logger.info(f"Results saved to {results_file}")
                
                # Save detailed summary
                self._save_scan_summary(df, symbols, failed_pairs, timestamp)
            
            return df
        else:
            logger.warning("No successful backtests completed")
            return pd.DataFrame()
    
    def _save_scan_summary(self, df: pd.DataFrame, symbols: List[str], 
                          failed_pairs: List[str], timestamp: str):
        """Save detailed scan summary."""
        summary_file = f"results/scan_summary_{timestamp}.txt"
        
        with open(summary_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("NNFX Bot - Dual System Backtest Results\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total pairs scanned: {len(symbols)}\n")
            f.write(f"Successful backtests: {len(df)}\n")
            f.write(f"Failed pairs: {len(failed_pairs)}\n\n")
            
            if len(df) > 0:
                f.write("TOP 10 PERFORMING PAIRS\n")
                f.write("-" * 40 + "\n")
                for i, (_, row) in enumerate(df.head(10).iterrows(), 1):
                    f.write(f"{i:2d}. {row['symbol']:<12} ")
                    f.write(f"Score: {row['score']:6.2f} | ")
                    f.write(f"Win Rate: {row['win_rate']:5.1%} | ")
                    f.write(f"PF: {row['profit_factor']:5.2f} | ")
                    f.write(f"Return: {row['total_return_pct']:6.1f}% | ")
                    f.write(f"DD: {row['max_drawdown']:5.1f}% | ")
                    f.write(f"Trades: {row['total_trades']:3d}\n")
                
                f.write("\n" + "PERFORMANCE STATISTICS\n")
                f.write("-" * 40 + "\n")
                f.write(f"Average Win Rate: {df['win_rate'].mean():5.1%}\n")
                f.write(f"Average Profit Factor: {df['profit_factor'].mean():5.2f}\n")
                f.write(f"Average Return: {df['total_return_pct'].mean():6.1f}%\n")
                f.write(f"Average Max Drawdown: {df['max_drawdown'].mean():5.1f}%\n")
                f.write(f"Average Sharpe Ratio: {df['sharpe_ratio'].mean():5.2f}\n")
                
                f.write("\n" + "FILTERING RESULTS\n")
                f.write("-" * 40 + "\n")
                f.write(f"Pairs with >60% win rate: {len(df[df['win_rate'] > 0.6])}\n")
                f.write(f"Pairs with >2.0 profit factor: {len(df[df['profit_factor'] > 2.0])}\n")
                f.write(f"Pairs with >10% return: {len(df[df['total_return_pct'] > 10])}\n")
                f.write(f"Pairs with <15% max drawdown: {len(df[df['max_drawdown'] < 15])}\n")
                f.write(f"Pairs with positive Sharpe: {len(df[df['sharpe_ratio'] > 0])}\n")
                
                # Risk analysis
                f.write("\n" + "RISK ANALYSIS\n")
                f.write("-" * 40 + "\n")
                f.write(f"Worst performing pair: {df.iloc[-1]['symbol']} ")
                f.write(f"({df.iloc[-1]['total_return_pct']:.1f}%)\n")
                f.write(f"Highest drawdown: {df['max_drawdown'].max():.1f}%\n")
                f.write(f"Most consecutive losses: {df['max_consecutive_losses'].max()}\n")
                f.write(f"Worst single trade (VaR 95%): {df['var_95'].min():.2f}R\n")
            
            if failed_pairs:
                f.write(f"\n" + "FAILED PAIRS\n")
                f.write("-" * 40 + "\n")
                for pair in failed_pairs:
                    f.write(f"  {pair}\n")
        
        logger.info(f"Detailed summary saved to {summary_file}")
    
    def _calculate_score(self, result: Dict) -> float:
        """Calculate overall score for pair ranking."""
        if result['total_trades'] < 3:
            return 0
        
        # Normalize metrics (0-100 scale)
        win_rate_score = result['win_rate'] * 100
        profit_factor_score = min(result['profit_factor'] * 20, 100)
        return_score = min(result['total_return_pct'] * 2, 100)
        trade_frequency_score = min(result['total_trades'] * 2, 100)
        sharpe_score = min(max(result['sharpe_ratio'] * 20, 0), 100)
        
        # Penalty factors
        drawdown_penalty = result['max_drawdown'] * 2
        consecutive_loss_penalty = result['max_consecutive_losses'] * 5
        var_penalty = abs(result['var_95']) * 10
        
        # Weighted score calculation
        score = (
            win_rate_score * 0.20 +
            profit_factor_score * 0.20 +
            return_score * 0.20 +
            trade_frequency_score * 0.15 +
            sharpe_score * 0.15 -
            drawdown_penalty * 0.04 -
            consecutive_loss_penalty * 0.03 -
            var_penalty * 0.03
        )
        
        return max(score, 0)
    
    def get_current_signals(self, symbols: List[str]) -> pd.DataFrame:
        """
        Get current trading signals for specified pairs.
        
        Args:
            symbols: List of symbols to analyze
            
        Returns:
            DataFrame with current signals and analysis
        """
        logger.info(f"Getting current signals for {len(symbols)} pairs...")
        
        signals = []
        
        for symbol in symbols:
            try:
                df = self.api.get_klines(symbol, "4H", 100)
                if not df.empty:
                    df = self.calculate_indicators(df)
                    df = self.generate_signals(df)
                    
                    # Get the latest complete candle (not current forming candle)
                    if len(df) >= 2:
                        latest = df.iloc[-2]
                        current = df.iloc[-1]
                    else:
                        latest = df.iloc[-1]
                        current = latest
                    
                    signal_type = "NONE"
                    confidence = 0
                    
                    if latest['long_signal']:
                        signal_type = "LONG"
                        confidence = self._calculate_signal_confidence(df.iloc[-5:], 'long')
                    elif latest['short_signal']:
                        signal_type = "SHORT"
                        confidence = self._calculate_signal_confidence(df.iloc[-5:], 'short')
                    
                    # Calculate system alignment scores
                    system_a_score = (
                        latest['system_a_baseline'] +
                        latest['system_a_confirmation'] +
                        latest['system_a_volume']
                    ) / 3
                    
                    system_b_score = (
                        latest['system_b_baseline'] +
                        latest['system_b_confirmation'] +
                        latest['system_b_volume']
                    ) / 3
                    
                    # Risk calculations
                    if not pd.isna(latest['atr']):
                        stop_distance = 2 * latest['atr']
                        tp_distance = 3 * latest['atr']
                        risk_reward = tp_distance / stop_distance if stop_distance > 0 else 0
                    else:
                        stop_distance = tp_distance = risk_reward = 0
                    
                    signals.append({
                        'symbol': symbol,
                        'signal': signal_type,
                        'confidence': confidence,
                        'price': latest['close'],
                        'atr': latest['atr'],
                        'stop_loss_distance': stop_distance,
                        'take_profit_distance': tp_distance,
                        'risk_reward_ratio': risk_reward,
                        'system_a_score': system_a_score,
                        'system_b_score': system_b_score,
                        'overall_strength': (system_a_score + system_b_score) / 2,
                        'trend_direction': 'Bullish' if latest['tema'] < latest['close'] else 'Bearish',
                        'momentum': 'Positive' if latest['cci'] > 0 else 'Negative',
                        'volume_pressure': 'Buying' if latest['elder_fi'] > 0 else 'Selling',
                        'timestamp': latest.name,
                        'current_candle_time': current.name
                    })
                    
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error getting signals for {symbol}: {e}")
        
        df = pd.DataFrame(signals)
        
        # Sort by signal strength and confidence
        if not df.empty:
            df = df.sort_values(
                ['signal', 'confidence', 'overall_strength'],
                ascending=[True, False, False]
            )
        
        return df
    
    def _calculate_signal_confidence(self, recent_data: pd.DataFrame, signal_type: str) -> float:
        """Calculate confidence level for a signal based on recent data."""
        if len(recent_data) < 3:
            return 0.5
        
        # Count supporting signals in recent candles
        if signal_type == 'long':
            signal_support = recent_data['long_signal'].sum() / len(recent_data)
            trend_consistency = (recent_data['system_a_baseline'] > 0).mean()
            momentum_consistency = (recent_data['system_a_confirmation'] > 0).mean()
        else:
            signal_support = recent_data['short_signal'].sum() / len(recent_data)
            trend_consistency = (recent_data['system_a_baseline'] < 0).mean()
            momentum_consistency = (recent_data['system_a_confirmation'] < 0).mean()
        
        # Volume consistency
        if signal_type == 'long':
            volume_consistency = (recent_data['system_a_volume'] > 0).mean()
        else:
            volume_consistency = (recent_data['system_a_volume'] < 0).mean()
        
        # Weighted confidence calculation
        confidence = (
            signal_support * 0.4 +
            trend_consistency * 0.25 +
            momentum_consistency * 0.20 +
            volume_consistency * 0.15
        )
        
        return min(confidence, 1.0)
    
    def export_detailed_results(self, rankings: pd.DataFrame, 
                               current_signals: pd.DataFrame) -> str:
        """
        Export detailed results to Excel file.
        
        Args:
            rankings: Backtest rankings DataFrame
            current_signals: Current signals DataFrame
            
        Returns:
            Filename of exported file
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"results/nnfx_analysis_{timestamp}.xlsx"
            
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Rankings sheet
                rankings.to_excel(writer, sheet_name='Backtest Rankings', index=False)
                
                # Current signals sheet
                current_signals.to_excel(writer, sheet_name='Current Signals', index=False)
                
                # Summary statistics
                if not rankings.empty:
                    summary_stats = pd.DataFrame({
                        'Metric': [
                            'Total Pairs Tested',
                            'Average Win Rate',
                            'Average Profit Factor',
                            'Average Return (%)',
                            'Average Max Drawdown (%)',
                            'Average Sharpe Ratio',
                            'Pairs with >60% Win Rate',
                            'Pairs with >2.0 Profit Factor',
                            'Pairs with Positive Return'
                        ],
                        'Value': [
                            len(rankings),
                            f"{rankings['win_rate'].mean():.1%}",
                            f"{rankings['profit_factor'].mean():.2f}",
                            f"{rankings['total_return_pct'].mean():.1f}",
                            f"{rankings['max_drawdown'].mean():.1f}",
                            f"{rankings['sharpe_ratio'].mean():.2f}",
                            len(rankings[rankings['win_rate'] > 0.6]),
                            len(rankings[rankings['profit_factor'] > 2.0]),
                            len(rankings[rankings['total_return_pct'] > 0])
                        ]
                    })
                    summary_stats.to_excel(writer, sheet_name='Summary Statistics', index=False)
                
                # Active signals summary
                if not current_signals.empty:
                    active_signals = current_signals[current_signals['signal'] != 'NONE']
                    if not active_signals.empty:
                        signal_summary = active_signals.groupby('signal').agg({
                            'symbol': 'count',
                            'confidence': 'mean',
                            'overall_strength': 'mean'
                        }).rename(columns={'symbol': 'count'})
                        signal_summary.to_excel(writer, sheet_name='Signal Summary')
            
            logger.info(f"Detailed results exported to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error exporting results: {e}")
            return ""


def run_comprehensive_analysis(api_config: Dict = None, test_pairs: List[str] = None) -> Dict:
    """
    Run comprehensive NNFX analysis suitable for VPS deployment.
    
    Args:
        api_config: Bitget API configuration dictionary
        test_pairs: List of pairs to test
        
    Returns:
        Dictionary with analysis results
    """
    logger.info("=" * 60)
    logger.info("NNFX Bot - Comprehensive Analysis Started")
    logger.info("=" * 60)
    
    try:
        # Initialize system
        if api_config:
            api = BitgetAPI(**api_config)
        else:
            api = BitgetAPI()
        
        system = DualNNFXSystem(api)
        
        # Configuration
        max_pairs = 20  # VPS resource limit
        
        # Get test pairs
        if test_pairs is None:
            symbols = api.get_symbols()
            test_symbols = symbols[:max_pairs] if symbols else [
                'BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT',
                'UNIUSDT', 'AAVEUSDT', 'SOLUSDT', 'MATICUSDT', 'AVAXUSDT'
            ]
        else:
            test_symbols = test_pairs[:max_pairs]
        
        logger.info(f"Testing {len(test_symbols)} pairs")
        
        # Run comprehensive backtest scan
        start_time = time.time()
        rankings = system.scan_pairs(test_symbols, save_results=True)
        scan_duration = time.time() - start_time
        
        logger.info(f"Scan completed in {scan_duration:.1f} seconds")
        
        if not rankings.empty:
            # Display top performers
            logger.info("\n" + "=" * 50)
            logger.info("TOP 5 PERFORMING PAIRS")
            logger.info("=" * 50)
            
            top_5 = rankings.head()
            for i, (_, row) in enumerate(top_5.iterrows(), 1):
                logger.info(f"{i}. {row['symbol']:<10} | Score: {row['score']:6.2f} | "
                           f"Win Rate: {row['win_rate']:5.1%} | PF: {row['profit_factor']:5.2f} | "
                           f"Return: {row['total_return_pct']:6.1f}% | DD: {row['max_drawdown']:5.1f}%")
            
            # Get current signals
            top_pairs = top_5['symbol'].tolist()
            logger.info(f"\nAnalyzing current signals for top {len(top_pairs)} pairs...")
            
            current_signals = system.get_current_signals(top_pairs)
            active_signals = current_signals[current_signals['signal'] != 'NONE']
            
            if not active_signals.empty:
                logger.info("\n" + "=" * 50)
                logger.info("ACTIVE TRADING SIGNALS")
                logger.info("=" * 50)
                
                for _, signal in active_signals.iterrows():
                    logger.info(f"{signal['symbol']:<10} | {signal['signal']:<5} | "
                               f"Price: {signal['price']:10.6f} | "
                               f"Confidence: {signal['confidence']:5.1%} | "
                               f"R:R: {signal['risk_reward_ratio']:4.1f}")
            else:
                logger.info("\nNo active trading signals found")
            
            # Export detailed results
            export_file = system.export_detailed_results(rankings, current_signals)
            
            # Summary statistics
            logger.info("\n" + "=" * 50)
            logger.info("SYSTEM PERFORMANCE SUMMARY")
            logger.info("=" * 50)
            logger.info(f"Pairs tested: {len(rankings)}")
            logger.info(f"Average win rate: {rankings['win_rate'].mean():.1%}")
            logger.info(f"Average profit factor: {rankings['profit_factor'].mean():.2f}")
            logger.info(f"Average return: {rankings['total_return_pct'].mean():.1f}%")
            logger.info(f"Pairs with >60% win rate: {len(rankings[rankings['win_rate'] > 0.6])}")
            logger.info(f"Pairs with >2.0 profit factor: {len(rankings[rankings['profit_factor'] > 2.0])}")
            
            return {
                'success': True,
                'rankings': rankings,
                'current_signals': current_signals,
                'export_file': export_file,
                'scan_duration': scan_duration,
                'summary': {
                    'pairs_tested': len(rankings),
                    'avg_win_rate': rankings['win_rate'].mean(),
                    'avg_profit_factor': rankings['profit_factor'].mean(),
                    'avg_return': rankings['total_return_pct'].mean(),
                    'top_performer': rankings.iloc[0]['symbol'] if len(rankings) > 0 else None,
                    'active_signals': len(active_signals)
                }
            }
        
        else:
            logger.warning("No successful backtests completed")
            return {'success': False, 'error': 'No successful backtests'}
            
    except Exception as e:
        logger.error(f"Critical error in analysis: {e}")
        return {'success': False, 'error': str(e)}


# Example usage and utility functions
if __name__ == "__main__":
    # Ensure logs directory exists
    if not os.path.exists("logs"):
        os.makedirs("logs")
    # Set up logging with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f"logs/system_run_{timestamp}.log"
    # Add file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info("NNFX Bot Trading System Started")
    logger.info(f"Log file: {log_file}")
    # Load API config from JSON file
    api_config_path = "config/api_config.json"
    if not os.path.exists(api_config_path):
        logger.error(f"API config file not found: {api_config_path}")
        print(f"ERROR: API config file not found: {api_config_path}")
        sys.exit(1)
    with open(api_config_path, "r") as f:
        api_config = json.load(f)
    
    example_pairs = [
        'BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT'
    ]
    
    # Run analysis
    results = run_comprehensive_analysis(api_config, example_pairs)
    
    if results['success']:
        logger.info("Analysis completed successfully")
        logger.info("System ready for live trading implementation")
    else:
        logger.error(f"Analysis failed: {results.get('error', 'Unknown error')}")
    
    logger.info("Session complete")


def cleanup_old_files(days_old: int = 7):
    """Clean up old cache and result files."""
    import glob
    from datetime import timedelta
    
    cutoff_date = datetime.now() - timedelta(days=days_old)
    cleaned_count = 0
    
    # Clean old cache files
    for cache_file in glob.glob("data/*.csv"):
        if os.path.getmtime(cache_file) < cutoff_date.timestamp():
            os.remove(cache_file)
            logger.info(f"Cleaned old cache file: {cache_file}")
            cleaned_count += 1
    
    # Clean old result files (keep recent ones)
    result_files = glob.glob("results/*")
    if len(result_files) > 20:
        result_files.sort(key=os.path.getmtime)
        for old_file in result_files[:-20]:
            os.remove(old_file)
            logger.info(f"Cleaned old result file: {old_file}")
            cleaned_count += 1
    
    logger.info(f"Cleanup completed. Removed {cleaned_count} files.")


def get_system_status():
    """Get current system status for monitoring."""
    try:
        import psutil
        
        status = {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
            'disk_free_gb': psutil.disk_usage('/').free / (1024**3)
        }
        
        logger.info(f"System Status - CPU: {status['cpu_percent']:.1f}%, "
                   f"Memory: {status['memory_percent']:.1f}%, "
                   f"Disk: {status['disk_percent']:.1f}%")
        
        return status
        
    except ImportError:
        logger.warning("psutil not available for system monitoring")
        return {'error': 'psutil not installed'}
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        