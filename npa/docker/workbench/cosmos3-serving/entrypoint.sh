#!/usr/bin/env bash
# npa-cosmos3-serving entrypoint: preflight, then serve.
#
# Everything that can fail before the server binds a port is checked here and
# reported with the fix, because the alternative is a stack trace several
# minutes into a startup that costs 8 GPUs. The three checks are the three ways
# this container has actually failed to start: a missing token against a gated
# guardrail repo, an unwritable Hugging Face cache mount, and a GPU count that
# does not match the pinned parallel config.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL="${NPA_COSMOS3_SERVE_MODEL:-nvidia/Cosmos3-Super}"
MODEL_REVISION="${NPA_COSMOS3_SERVE_MODEL_REVISION:-e0262be9d8f7586bc24c069a2aed2b665bdff266}"
HOST="${NPA_COSMOS3_SERVE_HOST:-0.0.0.0}"
PORT="${NPA_COSMOS3_SERVE_PORT:-8000}"
GUARDRAILS="${NPA_COSMOS3_SERVE_GUARDRAILS:-on}"
INIT_TIMEOUT="${NPA_COSMOS3_SERVE_INIT_TIMEOUT:-1800}"
EXPECTED_GPUS="${NPA_COSMOS3_SERVE_GPUS:-8}"
EXTRA_ARGS="${NPA_COSMOS3_SERVE_EXTRA_ARGS:-}"

fail() {
  echo "[npa-cosmos3-serving] ERROR: $*" >&2
  exit 1
}

note() {
  echo "[npa-cosmos3-serving] $*"
}

case "${GUARDRAILS}" in
  on | off) ;;
  *) fail "NPA_COSMOS3_SERVE_GUARDRAILS must be 'on' or 'off', got '${GUARDRAILS}'" ;;
esac

# --- preflight 1: real gated-repository access -------------------------------
# A credential is not consent and its mere presence proves nothing. Probe the
# selected model and (when enabled) the separately gated guardrail repository
# before allocating distributed workers or downloading checkpoint bytes.
python "${SCRIPT_DIR}/access_preflight.py"

# --- preflight 2: writable Hugging Face cache --------------------------------
#
# The image runs as a non-root user, so a cache mounted from a root-owned host
# directory is readable but not writable, and the failure surfaces as a lock
# error partway through a download rather than at mount time.
CACHE_DIR="${HF_HOME:-/opt/npa-cosmos3-serving/hf-cache}"
mkdir -p "${CACHE_DIR}" 2>/dev/null || true
if [ ! -w "${CACHE_DIR}" ]; then
  fail "Hugging Face cache '${CACHE_DIR}' is not writable by uid $(id -u).
       Point HF_HOME at a writable path, or chown the mounted cache to the
       container's runtime user. Weights and guardrail models are downloaded
       into this directory at run time; none of them ship in the image."
fi

# --- preflight 3: GPU count matches the pinned parallel config ---------------
#
# The pinned config below is CFG-parallel 2 x Ulysses 4 with HSDP shard 8, which
# is an 8-GPU decomposition. Handing it a different GPU count fails inside
# distributed init with a shape error, so check it here where the message can
# name the cause.
if ! command -v nvidia-smi >/dev/null 2>&1; then
  fail "nvidia-smi not found: this service requires the documented 8-GPU topology."
fi
visible="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
if [ "${visible}" != "${EXPECTED_GPUS}" ]; then
  fail "the pinned parallel config needs ${EXPECTED_GPUS} GPUs, found ${visible}."
fi

# --- serve -------------------------------------------------------------------
#
# The parallel config is pinned rather than exposed. It is the fastest of five
# strategies measured on 8x H200 at the model card's own example shape, about 9%
# faster than plain tensor parallelism at both 35 and 50 steps, and it is the
# configuration the model card itself recommends for 8x H200 / H100 / A100.
# Generation parameters (steps, frame count, resolution, per-request guardrail
# posture) are request fields on /v1/videos, not server flags, so they are not
# part of this surface.
argv=(
  vllm serve "${MODEL}"
  --revision "${MODEL_REVISION}"
  --omni
  --host "${HOST}"
  --port "${PORT}"
  --cfg-parallel-size 2
  --ulysses-degree 4
  --use-hsdp
  --hsdp-shard-size 8
  --init-timeout "${INIT_TIMEOUT}"
)

if [ "${GUARDRAILS}" = "off" ]; then
  argv+=(--no-guardrails)
fi

if [ -n "${EXTRA_ARGS}" ]; then
  # Deliberate word splitting: this is an operator-supplied flag string.
  # shellcheck disable=SC2206
  argv+=(${EXTRA_ARGS})
fi

note "model=${MODEL} guardrails=${GUARDRAILS} port=${PORT} access=confirmed"
note "startup to 'Application startup complete' takes minutes, not seconds: HSDP-sharded"
note "configs reached ready in roughly 280-290 s on a warm page cache and about 320 s"
note "longer on a cold one. Readiness probes tighter than that kill healthy boots."
note "exec: ${argv[*]}"

exec "${argv[@]}"
