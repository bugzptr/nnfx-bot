#!/bin/bash
# NNFX Bot - Quick VPS Setup Script
# Usage: curl -sSL https://raw.githubusercontent.com/bugzptr/nnfx-bot/main/scripts/vps_setup.sh | bash

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print functions
print_header() {
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================${NC}"
}

print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root for security reasons"
   print_warning "Please run as a regular user with sudo privileges"
   exit 1
fi

print_header "NNFX Bot VPS Setup"
echo "This script will set up the NNFX trading bot on your Ubuntu VPS"
echo ""

# Confirm installation
read -p "Do you want to proceed with the installation? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_warning "Installation cancelled"
    exit 0
fi

# Update system
print_status "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install required packages
print_status "Installing required system packages..."
sudo apt install -y python3 python3-pip python3-venv git curl htop unzip tree

# Install additional monitoring tools
print_status "Installing monitoring tools..."
sudo apt install -y iotop nethogs ncdu

# Set up project directory
PROJECT_DIR="$HOME/nnfx-bot"
print_status "Setting up project directory: $PROJECT_DIR"

# Check if directory already exists
if [ -d "$PROJECT_DIR" ]; then
    print_warning "Directory $PROJECT_DIR already exists"
    read -p "Do you want to remove it and start fresh? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$PROJECT_DIR"
        print_status "Removed existing directory"
    else
        print_error "Installation cancelled"
        exit 1
    fi
fi

# Clone repository
print_status "Cloning NNFX Bot repository..."
git clone https://github.com/bugzptr/nnfx-bot.git "$PROJECT_DIR"
cd "$PROJECT_DIR"

# Create directory structure
print_status "Creating directory structure..."
mkdir -p data results logs scripts

# Create Python virtual environment
print_status "Creating Python virtual environment..."
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies
print_status "Installing Python dependencies..."
pip install -r requirements.txt

# Copy configuration template
print_status "Setting up configuration..."
if [ -f "src/config.py.example" ]; then
    cp src/config.py.example src/config.py
    print_status "Configuration template copied to src/config.py"
else
    print_warning "Configuration template not found, creating basic config..."
    cat > src/config.py << 'EOF'
# Basic NNFX Bot Configuration
BITGET_CONFIG = {
    'api_key': '',
    'secret_key': '',
    'passphrase': '',
    'sandbox': True
}

RISK_PER_TRADE = 0.015
TIMEFRAME = "4H"
MAX_PAIRS_TO_TEST = 10

TEST_PAIRS = [
    'BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT'
]

SYSTEM_A_PARAMS = {
    'tema_period': 21,
    'cci_period': 14,
    'elder_fi_period': 13,
    'chandelier_period': 22,
    'chandelier_multiplier': 3.0
}

SYSTEM_B_PARAMS = {
    'kijun_period': 26,
    'williams_period': 14,
    'klinger_fast': 34,
    'klinger_slow': 55,
    'klinger_signal': 13,
    'psar_start': 0.02,
    'psar_increment': 0.02,
    'psar_maximum': 0.2
}
EOF
fi

# Make scripts executable
print_status "Setting up executable scripts..."
chmod +x run_analysis.py monitor.py cleanup.py 2>/dev/null || true

# Create helper scripts
print_status "Creating helper scripts..."

# Create activation script
cat > activate_env.sh << 'EOF'
#!/bin/bash
cd ~/nnfx-bot
source venv/bin/activate
echo "NNFX Bot environment activated"
echo "Available commands:"
echo "  python run_analysis.py --mode=test"
echo "  python run_analysis.py --mode=all"
echo "  python monitor.py"
echo "  python cleanup.py"
EOF
chmod +x activate_env.sh

# Create quick test script
cat > quick_test.sh << 'EOF'
#!/bin/bash
cd ~/nnfx-bot
source venv/bin/activate
echo "Running quick system test..."
python run_analysis.py --mode=test
EOF
chmod +x quick_test.sh

# Create monitoring script
cat > scripts/system_monitor.sh << 'EOF'
#!/bin/bash
echo "=== NNFX Bot System Monitor ==="
echo "Date: $(date)"
echo ""
echo "=== System Resources ==="
echo "CPU Usage: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)%"
echo "Memory Usage: $(free | grep Mem | awk '{printf "%.1f%%", $3/$2 * 100.0}')"
echo "Disk Usage: $(df -h / | awk 'NR==2{printf "%s", $5}')"
echo ""
echo "=== NNFX Bot Status ==="
if [ -d "~/nnfx-bot/results" ]; then
    echo "Latest Results: $(ls -t ~/nnfx-bot/results/ | head -1)"
fi
if [ -d "~/nnfx-bot/logs" ]; then
    echo "Latest Log: $(ls -t ~/nnfx-bot/logs/ | head -1)"
fi
echo ""
echo "=== Recent Bot Activity ==="
tail -5 ~/nnfx-bot/logs/*.log 2>/dev/null || echo "No logs found"
EOF
chmod +x scripts/system_monitor.sh

# Create daily run script
cat > scripts/daily_run.sh << 'EOF'
#!/bin/bash
SCRIPT_DIR="$HOME/nnfx-bot"
cd "$SCRIPT_DIR"
source venv/bin/activate

# Log start
echo "$(date): Starting daily NNFX analysis" >> logs/daily_run.log

# Run analysis
python run_analysis.py --mode=all >> logs/daily_run.log 2>&1

# Cleanup old files
python cleanup.py >> logs/daily_run.log 2>&1

# Log completion
echo "$(date): Daily analysis completed" >> logs/daily_run.log
EOF
chmod +x scripts/daily_run.sh

# Create systemd service template
print_status "Creating systemd service template..."
cat > scripts/nnfx-bot.service << EOF
[Unit]
Description=NNFX Trading Bot Analysis
After=network.target

[Service]
Type=oneshot
User=$USER
WorkingDirectory=$PROJECT_DIR
Environment=PATH=$PROJECT_DIR/venv/bin
ExecStart=$PROJECT_DIR/venv/bin/python $PROJECT_DIR/run_analysis.py --mode=all
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Set up basic firewall
print_status "Configuring basic firewall..."
sudo ufw --force enable
sudo ufw allow ssh
sudo ufw allow 22

# Create initial test
print_status "Running initial system test..."
if python -c "import pandas, numpy, requests, ta; print('All dependencies installed successfully')" 2>/dev/null; then
    print_status "Python dependencies verified successfully"
else
    print_warning "Some Python dependencies may not be installed correctly"
fi

# Create README for user
cat > QUICKSTART.md << 'EOF'
# NNFX Bot - Quick Start Guide

## 🚀 Your bot is now installed!

### Next Steps:

1. **Configure API credentials:**
   ```bash
   nano src/config.py
   # Add your Bitget API credentials
   ```

2. **Run a quick test:**
   ```bash
   ./quick_test.sh
   ```

3. **Activate environment for manual use:**
   ```bash
   ./activate_env.sh
   ```

### Available Scripts:

- `./quick_test.sh` - Run quick system test
- `./activate_env.sh` - Activate Python environment
- `scripts/system_monitor.sh` - Check system status
- `scripts/daily_run.sh` - Run daily analysis

### Manual Commands:

```bash
# Activate environment
source venv/bin/activate

# Run different analysis modes
python run_analysis.py --mode=test     # Quick test
python run_analysis.py --mode=scan     # Full scan
python run_analysis.py --mode=signals  # Get signals
python run_analysis.py --mode=all      # Complete analysis

# System maintenance
python monitor.py    # Check system status
python cleanup.py    # Clean old files
```

### Automation:

1. **Set up daily cron job:**
   ```bash
   crontab -e
   # Add: 0 9 * * * /home/$(whoami)/nnfx-bot/scripts/daily_run.sh
   ```

2. **Install systemd service:**
   ```bash
   sudo cp scripts/nnfx-bot.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable nnfx-bot.service
   ```

### Support:
- GitHub Issues: https://github.com/bugzptr/nnfx-bot/issues
- Documentation: https://github.com/bugzptr/nnfx-bot/wiki
EOF

print_header "Installation Complete!"
echo ""
print_status "NNFX Bot has been successfully installed to: $PROJECT_DIR"
echo ""
print_warning "IMPORTANT NEXT STEPS:"
echo "1. Configure your Bitget API credentials in src/config.py"
echo "2. Run a test: ./quick_test.sh"
echo "3. Read QUICKSTART.md for usage instructions"
echo ""
print_status "Quick commands:"
echo "  cd $PROJECT_DIR"
echo "  ./activate_env.sh    # Activate environment"
echo "  ./quick_test.sh      # Run quick test"
echo ""
print_warning "Remember to:"
echo "- Keep your API credentials secure"
echo "- Start with paper trading (sandbox=True)"
echo "- Monitor system resources regularly"
echo "- Read the full documentation in README.md"
echo ""
echo -e "${GREEN}Happy trading! 🚀${NC}"

# Final file permissions check
chmod -R u+rwx "$PROJECT_DIR"

print_status "Setup completed successfully!"
print_status "Check QUICKSTART.md for next steps"