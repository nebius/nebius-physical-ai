#!/bin/bash
# Deploy npa-lerobot-server to the VM.
# Run this script ON the VM (via SSH or scp + exec).
set -euo pipefail

DEPLOY_ROOT="/opt/lerobot"
VENV="$DEPLOY_ROOT/venv"
NPA_SRC="${1:-.}"  # path to the npa/ package directory

echo "=== Installing npa package into LeRobot venv ==="
"$VENV/bin/pip" install --quiet "$NPA_SRC[server]"

echo "=== Setting up systemd service ==="
sudo mkdir -p /etc/npa-lerobot-server
sudo mkdir -p /var/log/npa-lerobot
sudo chown ubuntu:ubuntu /var/log/npa-lerobot

# Create env file from template if it doesn't exist, merging with existing .env
if [ ! -f /etc/npa-lerobot-server/env ]; then
    # Start with the template
    sudo cp "$NPA_SRC/deploy/env.template" /etc/npa-lerobot-server/env

    # Pull credentials from the existing lerobot .env
    if [ -f "$DEPLOY_ROOT/.env" ]; then
        while IFS='=' read -r key value; do
            case "$key" in
                AWS_ACCESS_KEY_ID|AWS_SECRET_ACCESS_KEY)
                    sudo sed -i "s|^${key}=.*|${key}=${value}|" /etc/npa-lerobot-server/env
                    ;;
                NEBIUS_S3_ENDPOINT)
                    sudo sed -i "s|^AWS_ENDPOINT_URL=.*|AWS_ENDPOINT_URL=${value}|" /etc/npa-lerobot-server/env
                    ;;
                HF_TOKEN)
                    [ -n "$value" ] && sudo sed -i "s|^HF_TOKEN=.*|HF_TOKEN=${value}|" /etc/npa-lerobot-server/env
                    ;;
            esac
        done < "$DEPLOY_ROOT/.env"
    fi
    sudo chmod 600 /etc/npa-lerobot-server/env
    echo "  Created /etc/npa-lerobot-server/env (credentials merged from .env)"
else
    echo "  /etc/npa-lerobot-server/env already exists, skipping"
fi

# Install systemd unit
sudo cp "$NPA_SRC/deploy/npa-lerobot-server.service" /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable npa-lerobot-server
sudo systemctl restart npa-lerobot-server

echo "=== Waiting for server to start ==="
for i in $(seq 1 15); do
    if curl -sf http://localhost:8080/health >/dev/null 2>&1; then
        echo "  Server is up on port 8080"
        exit 0
    fi
    sleep 2
done

echo "  WARNING: Server did not respond within 30s"
echo "  Check: sudo journalctl -u npa-lerobot-server -n 50"
exit 1
