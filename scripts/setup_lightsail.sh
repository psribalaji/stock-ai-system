#!/bin/bash
# setup_lightsail.sh — One-shot setup for StockAI on AWS Lightsail (Ubuntu 22.04)
#
# Run this ONCE after SSHing into a fresh Lightsail instance:
#   bash setup_lightsail.sh
#
# What it does:
#   1. Installs Python 3.11, git, pip
#   2. Clones the repo (prompts for GitHub token)
#   3. Creates .venv and installs dependencies
#   4. Prompts you to paste your .env API keys
#   5. Installs and enables the stockai systemd service
#   6. Starts the scheduler

set -e

REPO_URL="https://github.com/psribalaji/stock-ai-system.git"
APP_DIR="/home/ubuntu/stock-ai-system"
SERVICE_FILE="/etc/systemd/system/stockai.service"

echo ""
echo "========================================"
echo "  StockAI — Lightsail Setup"
echo "========================================"
echo ""

# ── 1. System packages ────────────────────────────────────────────────────────
echo "[1/6] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3.12 python3.12-venv python3.12-dev git build-essential

# ── 2. Clone repo ─────────────────────────────────────────────────────────────
echo ""
echo "[2/6] Cloning repo..."
echo "  You need a GitHub Personal Access Token (PAT) with 'repo' scope."
echo "  Create one at: https://github.com/settings/tokens"
echo ""
read -rp "  GitHub username: " GH_USER
read -rsp "  GitHub PAT (hidden): " GH_TOKEN
echo ""

CLONE_URL="https://${GH_USER}:${GH_TOKEN}@github.com/psribalaji/stock-ai-system.git"

if [ -d "$APP_DIR" ]; then
    echo "  Directory exists — pulling latest..."
    cd "$APP_DIR" && git pull
else
    git clone "$CLONE_URL" "$APP_DIR"
fi

cd "$APP_DIR"

# Store credentials so 'git pull' works later
git config credential.helper store
echo "https://${GH_USER}:${GH_TOKEN}@github.com" > ~/.git-credentials
chmod 600 ~/.git-credentials

# ── 3. Python venv + dependencies ────────────────────────────────────────────
echo ""
echo "[3/6] Creating virtual environment and installing dependencies..."
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "  Done."

# ── 4. .env file ─────────────────────────────────────────────────────────────
echo ""
echo "[4/6] Setting up .env file..."
if [ -f "$APP_DIR/.env" ]; then
    echo "  .env already exists — skipping. Edit manually if needed: nano $APP_DIR/.env"
else
    cp "$APP_DIR/.env.example" "$APP_DIR/.env"
    echo ""
    echo "  Opening .env for editing. Paste in all your API keys, then save (Ctrl+X → Y → Enter)."
    echo "  Press Enter to continue..."
    read -r
    nano "$APP_DIR/.env"
fi

# ── 5. Systemd service ────────────────────────────────────────────────────────
echo ""
echo "[5/6] Installing systemd service..."
sudo tee "$SERVICE_FILE" > /dev/null <<EOF
[Unit]
Description=StockAI Paper Trading Scheduler
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=${APP_DIR}
Environment="PATH=${APP_DIR}/.venv/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=${APP_DIR}/.venv/bin/python scripts/run_paper_trading.py
Restart=always
RestartSec=30
StandardOutput=append:/var/log/stockai.log
StandardError=append:/var/log/stockai.log

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable stockai

# ── 6. Log rotation ───────────────────────────────────────────────────────────
sudo tee /etc/logrotate.d/stockai > /dev/null <<EOF
/var/log/stockai.log {
    daily
    rotate 14
    compress
    missingok
    notifempty
}
EOF

# ── 7. Start ──────────────────────────────────────────────────────────────────
echo ""
echo "[6/6] Starting scheduler..."
sudo systemctl start stockai
sleep 3
sudo systemctl status stockai --no-pager

echo ""
echo "========================================"
echo "  Setup complete!"
echo ""
echo "  Useful commands:"
echo "    View logs:      sudo tail -f /var/log/stockai.log"
echo "    Check status:   sudo systemctl status stockai"
echo "    Restart:        sudo systemctl restart stockai"
echo "    Stop:           sudo systemctl stop stockai"
echo "    Pull + restart: cd $APP_DIR && git pull && sudo systemctl restart stockai"
echo "========================================"
