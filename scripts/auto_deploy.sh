#!/bin/bash
# scripts/auto_deploy.sh — Auto-pull and restart if new commits detected.
# Add to crontab: */5 * * * * /home/ubuntu/stock-ai-system/scripts/auto_deploy.sh >> /home/ubuntu/deploy.log 2>&1

REPO_DIR="/home/ubuntu/stock-ai-system"
VENV="$REPO_DIR/.venv/bin"
SERVICE="stockai"

cd "$REPO_DIR"

# Fetch latest without merging
git fetch origin main --quiet 2>&1 || { echo "$(date): git fetch failed"; exit 1; }

LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse origin/main)

if [ "$LOCAL" = "$REMOTE" ]; then
    exit 0  # Already up to date
fi

echo "$(date): New commits detected ($LOCAL → $REMOTE)"

# Hard-reset to remote — remote (GitHub) is always the source of truth.
# Any local changes on the server are intentional only if pushed to GitHub first.
git reset --hard origin/main --quiet
if [ $? -ne 0 ]; then
    echo "$(date): git reset failed — aborting"
    exit 1
fi

# Install any new dependencies quietly
$VENV/pip install -r requirements.txt --quiet 2>/dev/null

# Restart the service
sudo systemctl restart $SERVICE
if [ $? -eq 0 ]; then
    echo "$(date): Deployed $(git rev-parse --short HEAD) and restarted $SERVICE"
else
    echo "$(date): systemctl restart failed"
    exit 1
fi
