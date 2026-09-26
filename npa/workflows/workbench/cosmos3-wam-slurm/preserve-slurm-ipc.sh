#!/usr/bin/env bash
# Keep batch-job shared memory alive when the operator's SSH session ends.
set -euo pipefail
sudo install -d -m 755 /etc/systemd/logind.conf.d
sudo tee /etc/systemd/logind.conf.d/90-wam-slurm-ipc.conf >/dev/null <<'CONFIG'
[Login]
RemoveIPC=no
CONFIG
sudo chmod 644 /etc/systemd/logind.conf.d/90-wam-slurm-ipc.conf
sudo systemctl restart systemd-logind
