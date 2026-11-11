#!/bin/bash
# Launch 8 single-GPU Cosmos3-Nano services with the cosmos3-super harness's server
# command (bin/cosmos3-super@951dc11), run through nerdctl with host networking because
# this node's operating-system image provides nerdctl without Docker or a CNI bridge plugin.
# Usage: bash launch-8x1.sh <cell> <off|on>
# Requires bash 4.4 or later. Any second argument other than "on" launches with
# guardrails off. If a later container fails to start, set -e exits and leaves the
# earlier ones running; remove them with: nerdctl rm -f $(nerdctl ps -aq --filter label=io.cosmos3-super.run=<cell>)
set -euo pipefail
CELL=$1; GUARD=$2
BASE="${BASE:?set BASE to the run directory}"
IMG=vllm/vllm-omni@sha256:86fb713bf73e118e537a55fa013dcdbdf2468ed61b98f4f2e7934a9482181dc1
REV=e59a53c25979a090fa8706c9acc0c254a6e89b92
OUT=$BASE/$(echo "$CELL" | tr 'A-Z' 'a-z'); mkdir -p "$OUT"
if [ "$GUARD" = on ]; then
  DEPLOY=/root/.cache/huggingface/cosmos3_omni_guardrails_on.yaml
  [ -f "$HF_CACHE/cosmos3_omni_guardrails_on.yaml" ] || { echo "missing guardrails-on deploy config" >&2; exit 2; }
  [ -n "${HF_TOKEN:-}" ] || { echo "HF_TOKEN required for guardrails" >&2; exit 2; }
  EXTRA_ENV=(-e HF_TOKEN -e HF_HUB_DISABLE_XET=1); GFLAG=()
else
  DEPLOY=/app/vllm-omni/vllm_omni/deploy/cosmos3_omni.yaml
  EXTRA_ENV=(); GFLAG=(--no-guardrails)
fi
# Refuse to start unless the GPUs and ports are free.
[ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] || { echo "GPU processes present" >&2; exit 3; }
if ss -ltn | grep -qE ':(810[0-7])\b'; then echo "ports 8100-8107 busy" >&2; exit 3; fi
for i in 0 1 2 3 4 5 6 7; do
  sudo -E nerdctl -n default run -d --name "c3-${CELL,,}-r$i" \
    --label io.cosmos3-super.run="$CELL" --label io.cosmos3-super.role=backend \
    --network host --gpus "device=$i" --ipc=host --shm-size 32g \
    --ulimit nofile=1048576:1048576 \
    -v "${HF_CACHE:?set HF_CACHE to the Hugging Face cache}":/root/.cache/huggingface \
    -e VLLM_OMNI_VIDEO_SYNC_TIMEOUT=5400 "${EXTRA_ENV[@]}" \
    "$IMG" \
    vllm serve nvidia/Cosmos3-Nano --revision "$REV" --omni --host 127.0.0.1 --port $((8100+i)) \
    --init-timeout 1800 --tensor-parallel-size 1 --deploy-config "$DEPLOY" "${GFLAG[@]}" >/dev/null
done
sudo nerdctl -n default ps --filter label=io.cosmos3-super.run="$CELL" --format '{{.ID}} {{.Names}}' > "$OUT/containers.txt"
echo "launched $(wc -l < "$OUT/containers.txt") containers for $CELL (guardrails=$GUARD)"
