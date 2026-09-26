#!/usr/bin/env bash
# Configure a fresh second worker from its controller's private bundle.
set -euo pipefail
: "${WAM_WORKER_BUNDLE:?Path to the private bundle transferred over verified SSH}"
if test -e /etc/slurm/slurm.conf; then
    echo "Refusing to replace an existing Slurm configuration" >&2
    exit 1
fi
uv python install 3.13.15 3.10.21
sudo /usr/bin/python3 "$(dirname "$0")/slurm_worker.py" --bundle "$WAM_WORKER_BUNDLE" --prepare-user
sudo env DEBIAN_FRONTEND=noninteractive apt-get update
sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y \
    slurm-wlm=23.11.4-1.2ubuntu5 munge nfs-common libosmesa6 libgl1 \
    libglib2.0-0 cmake build-essential sysstat rdma-core infiniband-diags \
    perftest libibverbs-dev
sudo /usr/bin/python3 "$(dirname "$0")/slurm_worker.py" --bundle "$WAM_WORKER_BUNDLE"
