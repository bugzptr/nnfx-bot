# NNFX Bot - Detailed Setup Guide

This guide provides step-by-step instructions for setting up the NNFX Bot on various environments.

## 📋 Table of Contents

- [System Requirements](#system-requirements)
- [Ubuntu VPS Setup](#ubuntu-vps-setup)
- [Local Development Setup](#local-development-setup)
- [Bitget API Configuration](#bitget-api-configuration)
- [Initial Testing](#initial-testing)
- [Production Deployment](#production-deployment)
- [Troubleshooting](#troubleshooting)

## 🖥️ System Requirements

### Minimum Requirements
- **OS:** Ubuntu 18.04+ / Windows 10+ / macOS 10.14+
- **Python:** 3.8 or higher
- **RAM:** 2GB (4GB recommended)
- **Storage:** 5GB free space
- **Network:** Stable internet connection

### Recommended VPS Specifications
- **Provider:** DigitalOcean, AWS, Vultr, or similar
- **Instance:** 2 vCPU, 4GB RAM, 40GB SSD
- **OS:** Ubuntu 20.04 LTS
- **Location:** Close to your timezone for optimal monitoring

## 🐧 Ubuntu VPS Setup

### Step 1: Initial Server Setup

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y python3 python3-pip python3-venv git curl htop unzip

# Install additional monitoring tools
sudo apt install -y iotop nethogs
```

### Step 2: Create Non-Root User (if needed)

```bash
# Create new user
sudo adduser trader

# Add to sudo group
sudo usermod -aG sudo trader

# Switch to new user
su - trader
```

### Step 3: Clone and Setup Repository

```bash
# Clone the repository
git clone https://github.com/bugzptr/nnfx-bot.git
cd nnfx-bot

# Create project directory structure
mkdir -p data results logs

# Create Python virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### Step 4: Environment Configuration

```bash
# Copy configuration template
cp src/config.py.example src/config.py

# Edit configuration (add your API credentials)
nano src/config.py
```

### Step 5: Initial Test

```bash
# Test system installation
python run_analysis.py --mode=test

# Check system status
python monitor.py
```

## 💻 Local Development Setup

### Windows Setup

1. **Install Python 3.8+** from [python.org](https://www.python.org/downloads/)

2. **Install Git** from [git-scm.com](https://git-scm.com/download/win)

3. **Clone repository:**
```cmd
git clone https://github.com/bugzptr/nnfx-bot.git
cd nnfx-bot
```

4. **Create virtual environment:**
```cmd
python -m venv venv
venv\Scripts\activate
```

5. **Install dependencies:**
```cmd
pip install -r requirements.txt
```

### macOS Setup

1. **Install Homebrew** (if not installed):
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

2. **Install Python and Git:**
```bash
brew install python git
```

3. **Clone and setup:**
```bash
git clone https://github.com/bugzptr/nnfx-bot.git
cd nnfx-bot
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Linux (Ubuntu/Debian) Setup

Follow the same steps as VPS setup above.

## 🔐 Bitget API Configuration

### Step 1: Create Bitget Account

1. Visit [Bitget](https://www.bitget.com/)
2. Register for an account
3. Complete identity verification
4. Enable 2FA for security

### Step 2: Generate API Keys

1. **Navigate to API Management:**
   - Login to Bitget
   - Go to Account → API Management
   - Create New API Key

2. **API Permissions Required:**
   - ✅ Read Permission (for market data)
   - ✅ Trade Permission (for live trading - optional initially)
   - ❌ Withdraw Permission (not needed)

3. **Security Settings:**
   - Set IP whitelist to your VPS IP
   - Enable passphrase
   - Store credentials securely

### Step 3: Configure API in Bot

Edit `src/config.py`:

```python
# Bitget API Configuration
BITGET_CONFIG = {
    'api_key': 'your_api_key_here',
    'secret_key': 'your_secret_key_here',
    'passphrase': 'your_passphrase_here',
    'sandbox': True  # Set to False for live trading
}
```

### Step 4: Test API Connection

```bash
python -c "
from src.trading_system import BitgetAPI
api = BitgetAPI(**{
    'api_key': 'your_key',
    'secret_key': 'your_secret', 
    'passphrase': 'your_passphrase'
})
print('Available symbols:', api.get_symbols()[:5])
"
```

## 🧪 Initial Testing

### Step 1: Quick System Test

```bash
# Activate virtual environment
source venv/bin/activate

# Run quick test on BTC/USDT
python run_analysis.py --mode=test
```

**Expected Output:**
```
=== Dual NNFX System Started ===
Mode: test
Quick Test Results:
  Total Trades: 15
  Win Rate: 66.7%
  Profit Factor: 2.14
  Total Return: 18.3%
  Max Drawdown: 7.2%
```

### Step 2: Single Pair Backtest

```bash
python -c "
from src.trading_system import *
api = BitgetAPI()
system = DualNNFXSystem(api)
result = system.backtest_pair('ETHUSDT')
print(f'ETH Results: {result[\"win_rate\"]:.1%} win rate, {result[\"profit_factor\"]:.2f} PF')
"
```

### Step 3: System Health Check

```bash
# Check system resources
python monitor.py

# Verify all directories exist
ls -la data/ results/ logs/

# Check Python packages
pip list | grep -E "(pandas|numpy|requests|ta)"
```

## 🚀 Production Deployment

### Step 1: Security Hardening

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Configure firewall
sudo ufw enable
sudo ufw allow ssh
sudo ufw allow 22

# Secure SSH (edit /etc/ssh/sshd_config)
sudo nano /etc/ssh/sshd_config
# Set: PermitRootLogin no
# Set: PasswordAuthentication no (if using SSH keys)
sudo systemctl restart ssh
```

### Step 2: Service Setup

Create systemd service:

```bash
# Create service file
sudo nano /etc/systemd/system/nnfx-bot.service
```

Service content:
```ini
[Unit]
Description=NNFX Trading Bot
After=network.target

[Service]
Type=oneshot
User=trader
WorkingDirectory=/home/trader/nnfx-bot
Environment=PATH=/home/trader/nnfx-bot/venv/bin
ExecStart=/home/trader/nnfx-bot/venv/bin/python /home/trader/nnfx-bot/run_analysis.py --mode=all
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

Enable service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable nnfx-bot.service
```

### Step 3: Automated Scheduling

Setup cron job for regular analysis:

```bash
# Edit crontab
crontab -e

# Add daily analysis at 9 AM UTC
0 9 * * * cd /home/trader/nnfx-bot && ./venv/bin/python run_analysis.py --mode=all >> logs/cron_$(date +\%Y\%m\%d).log 2>&1

# Add weekly cleanup on Sundays at 2 AM
0 2 * * 0 cd /home/trader/nnfx-bot && ./venv/bin/python cleanup.py

# Add hourly signal checks during trading hours
0 8-20 * * 1-5 cd /home/trader/nnfx-bot && ./venv/bin/python run_analysis.py --mode=signals >> logs/signals_$(date +\%Y\%m\%d).log 2>&1
```

### Step 4: Monitoring Setup

Install monitoring tools:

```bash
# Install htop for process monitoring
sudo apt install htop

# Create monitoring script
cat > ~/monitor_bot.sh << 'EOF'
#!/bin/bash
echo "=== NNFX Bot System Status ==="
echo "Date: $(date)"
echo "Uptime: $(uptime)"
echo ""
echo "=== Disk Usage ==="
df -h /
echo ""
echo "=== Memory Usage ==="
free -h
echo ""
echo "=== Recent Logs ==="
tail -5 /home/trader/nnfx-bot/logs/analysis_*.log 2>/dev/null || echo "No logs found"
echo ""
echo "=== Latest Results ==="
ls -la /home/trader/nnfx-bot/results/ | tail -3
EOF

chmod +x ~/monitor_bot.sh
```

### Step 5: Backup Strategy

```bash
# Create backup script
cat > ~/backup_bot.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/home/trader/backups"
BOT_DIR="/home/trader/nnfx-bot"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR

# Backup configuration and results
tar -czf $BACKUP_DIR/nnfx_backup_$DATE.tar.gz \
    $BOT_DIR/src/config.py \
    $BOT_DIR/results/ \
    $BOT_DIR/logs/

# Keep only last 7 backups
ls -t $BACKUP_DIR/nnfx_backup_*.tar.gz | tail -n +8 | xargs rm -f

echo "Backup completed: nnfx_backup_$DATE.tar.gz"
EOF

chmod +x ~/backup_bot.sh

# Add to crontab for daily backups
echo "0 3 * * * /home/trader/backup_bot.sh" | crontab -
```

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. Python Import Errors

**Problem:** `ModuleNotFoundError` when running scripts

**Solution:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt

# Check Python path
python -c "import sys; print(sys.path)"
```

#### 2. API Connection Issues

**Problem:** "Error fetching data" or connection timeouts

**Solution:**
```bash
# Test internet connectivity
ping api.bitget.com

# Check API credentials
python -c "
from src.config import BITGET_CONFIG
print('API Key length:', len(BITGET_CONFIG['api_key']))
print('Secret Key set:', bool(BITGET_CONFIG['secret_key']))
"

# Test with minimal example
python -c "
import requests
response = requests.get('https://api.bitget.com/api/spot/v1/market/tickers')
print('Status:', response.status_code)
"
```

#### 3. Memory Issues

**Problem:** System runs out of memory

**Solution:**
```bash
# Check memory usage
free -h

# Reduce pairs in config.py
nano src/config.py
# Set MAX_PAIRS_TO_TEST = 10

# Add swap file if needed
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### 4. Disk Space Issues

**Problem:** No space left on device

**Solution:**
```bash
# Check disk usage
df -h

# Clean old files
python cleanup.py

# Remove old logs manually
find logs/ -name "*.log" -mtime +7 -delete

# Clean system packages
sudo apt autoremove
sudo apt autoclean
```

#### 5. Permission Issues

**Problem:** Permission denied errors

**Solution:**
```bash
# Fix file permissions
chmod +x run_analysis.py monitor.py cleanup.py

# Fix directory permissions
chmod 755 data results logs

# Check file ownership
ls -la src/config.py
```

### Getting Help

1. **Check Logs:**
```bash
tail -f logs/analysis_*.log
```

2. **Enable Debug Mode:**
```bash
python run_analysis.py --mode=test --quiet=false
```

3. **System Information:**
```bash
python -c "
import sys, platform
print(f'Python: {sys.version}')
print(f'Platform: {platform.platform()}')
print(f'Architecture: {platform.architecture()}')
"
```

4. **Create Issue:**
   - Visit: https://github.com/bugzptr/nnfx-bot/issues
   - Include: Error messages, system info, steps to reproduce

## 📚 Additional Resources

- [NNFX Methodology](https://www.youtube.com/c/NoNonsenseForex)
- [Bitget API Documentation](https://bitgetlimited.github.io/apidoc/en/spot/)
- [Python Virtual Environments Guide](https://docs.python.org/3/tutorial/venv.html)
- [Ubuntu Server Guide](https://ubuntu.com/server/docs)

---

**Next Steps:** After completing setup, proceed to the main [README.md](README.md) for usage instructions and trading strategies.
