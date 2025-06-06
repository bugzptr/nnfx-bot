NNFX Bot - Dual System Algorithmic Trading Strategy
This project implements a sophisticated algorithmic trading bot for cryptocurrency spot markets, utilizing the Bitget exchange API. It is built upon the core principles of the No Nonsense Forex (NNFX) methodology, employing a dual-system confirmation approach to enhance signal accuracy and filter out market noise.
Core Strategy & Intention:
The primary intention of this bot is to systematically identify and execute high-probability trading opportunities by combining two distinct NNFX-inspired trading systems. A trade signal is only generated when both systems are in agreement, aiming for improved win rates and more robust performance across different market conditions.
System A (Momentum-Based):
Baseline: Triple Exponential Moving Average (TEMA) to gauge short-term trend/momentum.
Confirmation: Commodity Channel Index (CCI) to confirm momentum strength.
Volume: Elder's Force Index to assess the power behind price moves.
System B (Trend-Following):
Baseline: Kijun-Sen (from Ichimoku Cloud) to identify the medium-term trend.
Confirmation: Williams %R to identify overbought/oversold conditions relative to the trend.
Volume: Chaikin Money Flow (CMF) to measure buying and selling pressure (Note: This was adapted from an initial intent to use Klinger/PVO due to library compatibility, with CMF > 0 indicating bullish volume and CMF < 0 indicating bearish volume).
Trade exits are managed by a combination of Chandelier Exit and Parabolic SAR, providing dynamic stop-loss mechanisms.
Key Features & Design Goals:
Dual-System Confirmation: The cornerstone of the strategy, requiring consensus from both System A and System B for trade entries, designed to reduce false signals.
Comprehensive Backtesting Engine:
Allows for iterative testing of the strategy against historical k-line data.
Calculates a wide array of performance metrics, including: Win Rate, Profit Factor, Total Return (R-multiple and percentage), Maximum Drawdown, Sharpe Ratio, Sortino Ratio, Maximum Consecutive Losses, and VaR (Value at Risk via PnL R-multiples).
Automated Symbol Scanning & Ranking:
Dynamically fetches and filters a list of tradable symbols (e.g., top N by USDT volume, optionally filtered by major base currencies).
Backtests each selected symbol in parallel (leveraging concurrent.futures.ProcessPoolExecutor) for efficiency.
Ranks symbols based on a configurable, multi-factor scoring system derived from their backtest performance.
Real-time Signal Identification:
Capable of fetching the latest market data to identify current trading signals based on the dual-system logic.
Includes a basic confidence scoring mechanism for live signals.
Risk Management Framework:
Implements ATR (Average True Range)-based stop-loss placement.
Calculates position size based on a fixed percentage risk of account equity per trade during backtesting.
Data Handling & Persistence:
Integrates with the Bitget API for fetching market data (symbols, k-lines).
Features intelligent caching for k-line data and symbol lists to minimize API calls and speed up subsequent runs.
Includes basic API rate limiting.
Configurability:
Strategy parameters (indicator periods, risk settings, scoring weights, etc.) are managed through an external JSON configuration file (config/strategy_config.json), allowing for easy tuning without code changes.
API keys are also managed via a separate configuration file (config/api_config.json).
Reporting & Analysis:
Generates detailed CSV and text summaries of backtest scan results.
Exports comprehensive analysis (rankings, current signals, summary statistics) to an Excel file.
Modular Design:
Separates concerns into classes for API interaction (BitgetAPI), indicator calculations (NNFXIndicators), and the core trading system logic (DualNNFXSystem).
VPS & Automation Ready: Designed with considerations for running on a Virtual Private Server (VPS), including logging, file-based data persistence, and attempts at resource-efficient processing (e.g., parallel backtesting).
Intended Use:
This bot is intended for traders and developers interested in:
Rigorously backtesting NNFX-style trading strategies in the cryptocurrency markets.
Identifying potentially profitable trading pairs and parameter sets through automated scanning and ranking.
Serving as a foundation for developing a semi-automated or fully-automated live trading bot, once the strategy is thoroughly validated and refined.
Learning about the implementation details of algorithmic trading systems, including API integration, indicator calculation, backtesting mechanics, and basic risk management.
The project emphasizes a data-driven approach to strategy development and aims to provide a robust framework for iterative improvement and testing.
