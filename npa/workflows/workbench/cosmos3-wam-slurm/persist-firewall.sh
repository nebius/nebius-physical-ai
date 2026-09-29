#!/usr/bin/env bash
# Restrict unused IPv6 services and preserve both firewall families across reboots.
set -euo pipefail
# Cluster peers use IPv4; preserve IPv6 SSH and network discovery only.
sudo ip6tables -N WAM_INPUT 2>/dev/null || true
sudo ip6tables -F WAM_INPUT
sudo ip6tables -A WAM_INPUT -i lo -j ACCEPT
sudo ip6tables -A WAM_INPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
sudo ip6tables -A WAM_INPUT -p tcp --dport 22 -j ACCEPT
sudo ip6tables -A WAM_INPUT -p ipv6-icmp -j ACCEPT
sudo ip6tables -A WAM_INPUT -j DROP
sudo ip6tables -C INPUT -j WAM_INPUT 2>/dev/null || sudo ip6tables -I INPUT 1 -j WAM_INPUT
sudo install -d -m 700 /etc/wam-slurm
sudo sh -c 'umask 077; iptables-save > /etc/wam-slurm/iptables.rules'
sudo sh -c 'umask 077; ip6tables-save > /etc/wam-slurm/ip6tables.rules'
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
ExecStart=/usr/sbin/ip6tables-restore /etc/wam-slurm/ip6tables.rules
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
UNIT
sudo systemctl daemon-reload
sudo systemctl enable wam-slurm-firewall.service
