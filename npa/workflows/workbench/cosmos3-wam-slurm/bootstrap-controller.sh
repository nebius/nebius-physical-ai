#!/usr/bin/env bash
# Bootstrap a fresh, dedicated Ubuntu 24.04 Slurm controller and B200 worker.
# Do not run against an existing Slurm installation.
set -euo pipefail
umask 077
: "${NPA_SLURM_CLUSTER_NAME:?Choose a private Slurm cluster name}"
: "${NPA_SLURM_ACCOUNT:?Choose a Slurm accounting account}"
if test -e /etc/slurm/slurm.conf; then
    echo "Refusing to replace an existing Slurm configuration" >&2
    exit 1
fi
sudo env DEBIAN_FRONTEND=noninteractive apt-get update
sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y \
    slurm-wlm=23.11.4-1.2ubuntu5 slurmdbd=23.11.4-1.2ubuntu5 \
    munge mariadb-server nfs-kernel-server nfs-common libosmesa6 \
    libgl1 libglib2.0-0 cmake build-essential sysstat rdma-core \
    infiniband-diags perftest libibverbs-dev
bash "$(dirname "$0")/preserve-slurm-ipc.sh"
node_address=$(hostname -I | awk '{print $1}')
sudo iptables -N WAM_INPUT 2>/dev/null || true
sudo iptables -F WAM_INPUT
sudo iptables -A WAM_INPUT -i lo -j ACCEPT
sudo iptables -A WAM_INPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
sudo iptables -A WAM_INPUT -p tcp --dport 22 -j ACCEPT
sudo iptables -A WAM_INPUT -s "$node_address" -j ACCEPT
sudo iptables -A WAM_INPUT -p icmp -j ACCEPT
sudo iptables -A WAM_INPUT -j DROP
sudo iptables -C INPUT -j WAM_INPUT 2>/dev/null || sudo iptables -I INPUT 1 -j WAM_INPUT
bash "$(dirname "$0")/persist-firewall.sh"
sudo install -d -o slurm -g slurm /var/spool/slurmctld /var/spool/slurmd /var/log/slurm
sudo systemctl enable --now munge mariadb
sudo --preserve-env=NPA_SLURM_CLUSTER_NAME /usr/bin/python3 "$(dirname "$0")/slurm_controller.py"
sudo systemctl enable --now slurmdbd
sudo systemctl enable --now slurmctld slurmd
sudo sacctmgr -i add cluster "$NPA_SLURM_CLUSTER_NAME"
sudo sacctmgr -i add account "$NPA_SLURM_ACCOUNT" Description=WAM Organization=research
sudo sacctmgr -i add user "$USER" "Account=$NPA_SLURM_ACCOUNT"
sinfo -o '%a %l %D %T %G'
srun --nodes=1 --ntasks=1 --gpus=8 nvidia-smi --query-gpu=index,name --format=csv
