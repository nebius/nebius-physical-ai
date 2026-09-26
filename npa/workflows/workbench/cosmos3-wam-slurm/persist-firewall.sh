#!/usr/bin/env bash
# Preserve the dedicated cluster's current IPv4 rules across worker reboots.
set -euo pipefail
sudo install -d -m 700 /etc/wam-slurm
sudo sh -c 'umask 077; iptables-save > /etc/wam-slurm/iptables.rules'
sudo tee /etc/systemd/system/wam-slurm-firewall.service >/dev/null <<'UNIT'
[Unit]
Description=Restore dedicated Slurm network rules
DefaultDependencies=no
After=local-fs.target
Before=network-pre.target slurmctld.service slurmd.service nfs-server.service
Wants=network-pre.target

[Service]
Type=oneshot
ExecStart=/usr/sbin/iptables-restore /etc/wam-slurm/iptables.rules
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
UNIT
sudo systemctl daemon-reload
sudo systemctl enable wam-slurm-firewall.service
