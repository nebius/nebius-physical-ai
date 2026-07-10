#!/usr/bin/env bash
# Golden eval for npa-cosmos2-transfer — a REAL capability test.
#
# Runs an actual Cosmos-Transfer2.5 video-to-video world-transfer inference on a
# bundled robot control example and asserts a non-trivial generated video is
# produced. This exercises the container's real job (synthetic-data augmentation
# via world transfer), not just a torch+CUDA probe.
#
# GPU-gated and heavy: the model runs a multi-step diffusion sample and the gated
# Cosmos-Transfer2.5-2B checkpoints auto-download on first use (HF_TOKEN + NVIDIA
# Open Model License acceptance required). Budget ~10-15 min end to end.
set -euo pipefail

REPO="${COSMOS_TRANSFER_REPO:-/opt/cosmos/cosmos-transfer2.5}"
cd "${REPO}"

# The capability image ships a ready py3.10 inference env (torch cu128 +
# flash-attn). Self-heal if it is absent so the eval still exercises the real
# model on an un-baked base image.
if ! .venv/bin/python -c "import torch, flash_attn" >/dev/null 2>&1; then
  echo "[setup] building cosmos-transfer2.5 inference env (py3.10 + cu128)"
  uv python install 3.10
  echo "3.10" > .python-version
  uv sync --extra=cu128 --python 3.10
fi

SPEC="${COSMOS_TRANSFER_SPEC:-assets/robot_example/depth/robot_depth_spec.json}"
OUT="${COSMOS_TRANSFER_OUT:-outputs/golden-eval}"
rm -rf "${OUT}"

echo "[run] cosmos-transfer2.5 inference: ${SPEC} -> ${OUT}"
.venv/bin/python examples/inference.py -i "${SPEC}" -o "${OUT}"

# Assert a real, non-trivial generated video exists (exclude the control-map viz).
.venv/bin/python - "${OUT}" <<'PY'
import glob
import os
import sys

out = sys.argv[1]
videos = [
    f
    for f in glob.glob(os.path.join(out, "**", "*.mp4"), recursive=True)
    if "control" not in os.path.basename(f)
]
big = [f for f in videos if os.path.getsize(f) > 100_000]
if not big:
    sizes = [(f, os.path.getsize(f)) for f in videos]
    print(f"[FAIL] no non-trivial output video in {out}: {sizes}", file=sys.stderr)
    raise SystemExit(1)
result = big[0]
print(f"[PASS] cosmos-transfer2.5 generated {result} ({os.path.getsize(result)} bytes)")
PY
