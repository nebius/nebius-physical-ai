#!/usr/bin/env bash
# One application Ray service per SkyPilot worker. `sky cancel` owns this task;
# `sky down` removes its pods. Never stop SkyPilot's management Ray processes.
set -euo pipefail
umask 077
runtime_root=/opt/npa-ray-train
test -d "$runtime_root" && test ! -L "$runtime_root"
test "$(stat -c %u "$runtime_root")" -eq "$(id -u)"
test "$(stat -c %a "$runtime_root")" = 700
export PATH="$runtime_root/env/bin:$PATH"
export RAY_TMPDIR="$runtime_root"
unset RAY_ADDRESS
export RAY_JOB_ALLOW_DRIVER_ON_WORKER_NODES=0
export RAY_USAGE_STATS_ENABLED=0
export RAY_TRAIN_V2_ENABLED=1
: "${AWS_ACCESS_KEY_ID:?Verified workload storage credentials required}"
: "${AWS_SECRET_ACCESS_KEY:?Verified workload storage credentials required}"
: "${AWS_ENDPOINT_URL_S3:?Verified workload storage endpoint required}"
read -r ephemeral_start _ < /proc/sys/net/ipv4/ip_local_port_range
if (( ephemeral_start <= 10999 )); then
    echo 'Application ports overlap OS ephemeral range' >&2
    exit 1
fi
readarray -t node_ips <<< "$SKYPILOT_NODE_IPS"
rank="${SKYPILOT_NODE_RANK:?SkyPilot node rank required}"
args=(--block --node-ip-address="${node_ips[$rank]}" --num-cpus=8
      --num-gpus="${SKYPILOT_NUM_GPUS_PER_NODE:?GPU count required}"
      --object-manager-port=8077 --node-manager-port=8078
      --min-worker-port=10010 --max-worker-port=10999
      --ray-client-server-port=10002 --dashboard-agent-listen-port=8267
      --dashboard-agent-grpc-port=8268 --runtime-env-agent-port=8269
      --metrics-export-port=8270 --disable-usage-stats)
if (( rank == 0 )); then
    args+=(--head --port=6381 --dashboard-host=127.0.0.1 --dashboard-port=8265
           --include-dashboard=true --temp-dir="$runtime_root/runtime")
else
    args+=(--address="${node_ips[0]}:6381")
fi
exec "$runtime_root/env/bin/ray" start "${args[@]}"
