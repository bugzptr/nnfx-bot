# NNFX Bot - Dual System Trading Strategy

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Bitget](https://img.shields.io/badge/Exchange-Bitget-orange.svg)](https://www.bitget.com/)
[![Status](https://img.shields.io/badge/Status-Beta-yellow.svg)](https://github.com/bugzptr/nnfx-bot)

A sophisticated cryptocurrency trading bot implementing the **No Nonsense Forex (NNFX)** methodology with a dual-system approach for enhanced signal confirmation and improved win rates.

## 🎯 Strategy Overview

This bot implements two independent NNFX systems that must both agree before generating trading signals:

### System A (Momentum-Based)
- **Baseline:** TEMA (Triple Exponential Moving Average)
- **Confirmation:** CCI (Commodity Channel Index)
- **Volume:** Elder's Force Index
- **Exit:** Chandelier Exit

### System B (Trend-Following)
- **Baseline:** Kijun-Sen (Ichimoku)
- **Confirmation:** Williams %R
- **Volume:** Klinger Oscillator
- **Exit:** Parabolic SAR

## 🚀 Key Features

- **Dual System Confirmation:** Higher probability setups with reduced false signals
- **Comprehensive Backtesting:** Advanced metrics including Sharpe ratio, max drawdown, consecutive losses
- **Real-time Signal Detection:** Live market scanning with confidence scoring
- **Risk Management:** ATR-based position sizing and stop-loss placement
- **VPS Optimized:** Resource-efficient design for cloud deployment
- **Data Caching:** Intelligent API rate limiting and data persistence
- **Performance Ranking:** Automated pair scoring and ranking system

## 📊 Performance Metrics

The system tracks comprehensive performance metrics:
- Win Rate & Profit Factor
- Total Return (R-multiple and percentage)
- Maximum Drawdown
- Sharpe Ratio
- Maximum Consecutive Losses
- Trade Frequency Analysis

## 🛠 Installation

### Prerequisites
- Python 3.8 or higher
- Ubuntu 18.04+ (recommended for VPS deployment)
- Bitget account with API access

### Quick Setup

1. **Clone the repository:**
```bash
git clone https://github.com/bugzptr/nnfx-bot.git
cd nnfx-bot
```

2. **Create virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Configure API credentials:**
```bash
cp src/config.py.example src/config.py
nano src/config.py  # Add your Bitget API credentials
```

5. **Run initial test:**
```bash
python run_analysis.py --mode=test
```

## ⚙️ Configuration

Edit `src/config.py` to customize:

```python
# API Configuration
BITGET_CONFIG = {
    'api_key': 'your_api_key_here',
    'secret_key': 'your_secret_key_here', 
    'passphrase': 'your_passphrase_here',
    'sandbox': True  # Set to False for live trading
}

# Trading Parameters
RISK_PER_TRADE = 0.015  # 1.5% risk per trade
MAX_PAIRS_TO_TEST = 15  # Limit for VPS resources

# Crypto pairs to analyze
TEST_PAIRS = [
    'BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT',
    # Add more pairs as needed
]
```

## 🔧 Usage

### Command Line Interface

```bash
# Quick BTC/USDT test
python run_analysis.py --mode=test

# Full pair scanning and backtesting
python run_analysis.py --mode=scan

# Get current trading signals
python run_analysis.py --mode=signals

# Complete analysis (test + scan + signals)
python run_analysis.py --mode=all
```

### System Monitoring

```bash
# Check system status and resources
python monitor.py

# Clean old cache and result files
python cleanup.py
```

### Python API Usage

```python
from src.trading_system import DualNNFXSystem, BitgetAPI

# Initialize system
api = BitgetAPI(api_key="...", secret_key="...", passphrase="...")
system = DualNNFXSystem(api)

# Backtest a single pair
result = system.backtest_pair('BTCUSDT')
print(f"Win Rate: {result['win_rate']:.1%}")
print(f"Profit Factor: {result['profit_factor']:.2f}")

# Scan multiple pairs
rankings = system.scan_pairs(['BTCUSDT', 'ETHUSDT', 'ADAUSDT'])
print(rankings.head())

# Get current signals
signals = system.get_current_signals(['BTCUSDT', 'ETHUSDT'])
active_signals = signals[signals['signal'] != 'NONE']
print(active_signals)
```

## 📁 Project Structure

```
nnfx-bot/
├── src/
│   ├── trading_system.py      # Main trading system implementation
│   ├── config.py              # Configuration settings
│   └── indicators.py          # Technical indicator calculations
├── data/                      # Cached market data
├── results/                   # Backtest results and reports
├── logs/                      # System logs
├── requirements.txt           # Python dependencies
├── run_analysis.py           # Main execution script
├── monitor.py                # System monitoring
├── cleanup.py                # File maintenance
├── SETUP.md                  # Detailed setup instructions
└── README.md                 # This file
```

## 🔄 Automation

### Cron Job (Daily Analysis)
```bash
# Add to crontab (crontab -e)
0 9 * * * cd /path/to/nnfx-bot && ./venv/bin/python run_analysis.py --mode=all
```

### Systemd Service
```bash
# Copy service file and enable
sudo cp scripts/nnfx-bot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable nnfx-bot.service
```

## 📈 Sample Output

```
=== TOP 5 PERFORMING PAIRS ===
1. ADAUSDT
   Score: 78.45
   Win Rate: 68.2%
   Profit Factor: 2.34
   Total Return: 23.7%
   Max Drawdown: 8.1%
   Trades: 22

=== ACTIVE TRADING SIGNALS ===
BTCUSDT: LONG
  Price: 43250.50
  Confidence: 87%
  Stop Loss Distance: 650.25
  Take Profit Distance: 975.38
  System Strength: 91%
```

## 🚨 Risk Management

- **Position Sizing:** ATR-based with configurable risk per trade
- **Stop Loss:** 2x ATR from entry point
- **Take Profit:** 3x ATR (1:1.5 risk/reward ratio)
- **Maximum Risk:** 1.5% of account per trade
- **Dual Confirmation:** Both systems must agree before entry

## ⚠️ Important Notes

1. **Paper Trading First:** Always test thoroughly before live trading
2. **API Rate Limits:** System includes intelligent rate limiting
3. **Data Requirements:** Minimum 100 candles needed for backtesting
4. **Resource Usage:** Monitor VPS resources with `monitor.py`
5. **Security:** Never commit API credentials to version control

## 🐛 Troubleshooting

### Common Issues

**Import Errors:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate
```

**API Connection Issues:**
```bash
# Check API credentials and internet connection
python -c "from src.trading_system import BitgetAPI; api = BitgetAPI(); print(api.get_symbols()[:5])"
```

**Memory Issues:**
```bash
# Reduce MAX_PAIRS_TO_TEST in config.py
# Monitor with: python monitor.py
```

**Insufficient Data:**
```bash
# Some pairs may not have enough historical data
# Check logs for specific error messages
```

## 📊 Backtesting Results

The system provides detailed backtesting reports including:
- Individual trade analysis
- Equity curve visualization
- Performance metrics comparison
- Signal confidence scoring
- System alignment analysis

## 🔮 Future Enhancements

- [ ] Automated order execution
- [ ] Multi-timeframe analysis
- [ ] Machine learning signal filtering
- [ ] Portfolio management
- [ ] Real-time notifications
- [ ] Web-based dashboard
- [ ] Additional exchanges support

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚡ Quick Start for VPS

```bash
# One-liner VPS setup
curl -sSL https://raw.githubusercontent.com/bugzptr/nnfx-bot/main/scripts/vps_setup.sh | bash
```

## 📞 Support

- **Issues:** [GitHub Issues](https://github.com/bugzptr/nnfx-bot/issues)
- **Discussions:** [GitHub Discussions](https://github.com/bugzptr/nnfx-bot/discussions)
- **Documentation:** [Wiki](https://github.com/bugzptr/nnfx-bot/wiki)

## ⭐ Star History

If this project helps you, please consider giving it a star! ⭐

---

**Disclaimer:** This software is for educational purposes only. Cryptocurrency trading involves substantial risk of loss. Never trade with money you cannot afford to lose. Past performance does not guarantee future results.
