#!/usr/bin/env bash
# Golden eval for npa-cosmos3 — a REAL capability test.
#
# Runs an actual Cosmos 3 text2image generation through the same
# `npa workbench cosmos3 generate` path the SDK and the cosmos3-generate workflow
# use, and asserts a decodable image artifact. This exercises the container's real
# job (Cosmos 3 generation), not a torch+CUDA probe.
#
# GPU-gated and heavy: no weights ship in this image, so the gated Cosmos3
# checkpoint downloads on first use under the OPERATOR's own Hugging Face license
# acceptance. Budget ~20-40 min end to end on the first run (checkpoint download
# dominates); guardrails stay enabled.
set -euo pipefail

if [ -z "${HF_TOKEN:-}${HUGGING_FACE_HUB_TOKEN:-}" ]; then
  echo "[FAIL] HF_TOKEN is required: this image bakes no weights, so the golden" >&2
  echo "       eval must download the gated Cosmos3 checkpoint with the" >&2
  echo "       operator's own Hugging Face token and license acceptance." >&2
  exit 1
fi

OUT="${NPA_COSMOS3_OUTPUT_DIR:-/tmp/npa-cosmos3-generate}"
rm -rf "${OUT}/golden-eval"
mkdir -p "${OUT}"

echo "[run] cosmos3 text2image generation -> ${OUT}/golden-eval"
exec python -m npa.smoke.test_cosmos3_generate_functional
