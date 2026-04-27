#!/bin/bash
# scripts/auto_deploy.sh — Auto-pull and restart if new commits detected.
# Add to crontab: */5 * * * * /home/ubuntu/stock-ai-system/scripts/auto_deploy.sh >> /home/ubuntu/deploy.log 2>&1

set -e

REPO_DIR="/home/ubuntu/stock-ai-system"
VENV="$REPO_DIR/.venv/bin"
SERVICE="stockai"  # systemd service name

cd "$REPO_DIR"

# Fetch latest without merging
git fetch origin main --quiet

# Check if there are new commits
LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse origin/main)

if [ "$LOCAL" = "$REMOTE" ]; then
    exit 0  # No changes
fi

echo "$(date): New commits detected ($LOCAL → $REMOTE)"

# Pull changes
git pull origin main --quiet

# Install any new dependencies
$VENV/pip install -r requirements.txt --quiet 2>/dev/null

# Restart the service
sudo systemctl restart $SERVICE

echo "$(date): Deployed and restarted successfully"
