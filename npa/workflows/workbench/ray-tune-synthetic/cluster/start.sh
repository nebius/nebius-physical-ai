#!/usr/bin/env bash
# SkyPilot owns this exact service task; native Ray Jobs owns applications.
set -euo pipefail
umask 077
runtime_root="$HOME/.npa-ray-tune"
test -d "$runtime_root" && test ! -L "$runtime_root"
test "$(stat -c %a "$runtime_root")" = 700
export PATH="$runtime_root/env/bin:$PATH"
export RAY_TMPDIR="$runtime_root/runtime"
export RAY_USAGE_STATS_ENABLED=0
unset RAY_ADDRESS RAY_API_SERVER_ADDRESS
read -r ephemeral_start _ < /proc/sys/net/ipv4/ip_local_port_range
if (( ephemeral_start <= 10999 )); then
    echo 'Application ports overlap OS ephemeral range' >&2
    exit 1
fi
exec "$runtime_root/env/bin/ray" start --block --head --node-ip-address=127.0.0.1 \
  --num-cpus=4 --port=6381 --dashboard-host=127.0.0.1 --dashboard-port=8265 \
  --object-manager-port=8077 --node-manager-port=8078 \
  --min-worker-port=10010 --max-worker-port=10999 \
  --ray-client-server-port=10002 --dashboard-agent-listen-port=8267 \
  --dashboard-agent-grpc-port=8268 --runtime-env-agent-port=8269 \
  --metrics-export-port=8270 --disable-usage-stats --temp-dir="$RAY_TMPDIR"
