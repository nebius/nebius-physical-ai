#!/usr/bin/env bash
# Fail a Cosmos Transfer image build if non-redistributable or unnecessary payload
# appears in the source tree. This is deliberately independent of Docker tooling so
# the exact same guard can run during the build and against an exported image rootfs.
set -euo pipefail

ROOT="${1:-/opt/cosmos/cosmos-transfer2.5}"
[[ -d "${ROOT}" ]] || { echo "forbidden-payload guard: missing ${ROOT}" >&2; exit 1; }

fail_find() {
  local label="$1"
  shift
  local match
  match="$(find "${ROOT}" "$@" -print)"
  if [[ -n "${match}" ]]; then
    echo "forbidden-payload guard: ${label}: ${match}" >&2
    exit 1
  fi
}

# Source provenance must not carry Git object databases or any upstream media tree.
fail_find "Git/LFS metadata is baked" -type d \( -name .git -o -path '*/.git/lfs' \)
[[ ! -e "${ROOT}/assets" ]] || {
  echo "forbidden-payload guard: upstream assets directory is baked" >&2
  exit 1
}
[[ ! -e "${ROOT}/.venv/lib/python3.10/site-packages/skimage/data" ]] || {
  echo "forbidden-payload guard: unused scikit-image data/fetch module is baked" >&2
  exit 1
}
[[ ! -e "${ROOT}/.venv/lib/python3.10/site-packages/wandb/bin" ]] || {
  echo "forbidden-payload guard: unused W&B native service binary is baked" >&2
  exit 1
}

# Model material and caches are always operator-fetched at run time. Do not assume
# a small .pt/.bin is harmless: every weight-like suffix is rejected regardless of
# size or location.
fail_find "model/checkpoint file is baked" -type f \( \
  -iname '*.pt' -o -iname '*.pth' -o -iname '*.ckpt' -o -iname '*.safetensors' \
  -o -iname '*.onnx' -o -iname '*.engine' -o -iname '*.weights' -o -iname '*.gguf' \
  -o -iname '*.npz' \
\) \
  ! -path '*/site-packages/_virtualenv.pth' \
  ! -path '*/site-packages/distutils-precedence.pth' \
  ! -path '*/site-packages/scipy/stats/_sobol_direction_numbers.npz'
cache_match="$(find "${ROOT}" -path "${ROOT}/.venv" -prune -o -type d \( \
  -iname 'huggingface' -o -iname 'torch-cache' \
  -o -path '*/.cache/huggingface' -o -path '*/.cache/torch' \
\) -print -quit)"
[[ -z "${cache_match}" ]] || {
  echo "forbidden-payload guard: model cache is baked: ${cache_match}" >&2
  exit 1
}

# No upstream video/image/audio fixtures are needed. Package metadata icons inside
# site-packages are not control inputs, so this source-only check prunes the venv.
media_match="$(find "${ROOT}" -path "${ROOT}/.venv" -prune -o -type f \( \
  -iname '*.mp4' -o -iname '*.mov' -o -iname '*.avi' -o -iname '*.mkv' \
  -o -iname '*.webm' -o -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \
  -o -iname '*.webp' -o -iname '*.wav' -o -iname '*.mp3' \
\) -print -quit)"
[[ -z "${media_match}" ]] || {
  echo "forbidden-payload guard: upstream media is baked: ${media_match}" >&2
  exit 1
}

# A large non-code file outside the locked venv is almost certainly an accidental
# fixture or model. Shared objects and archives belong only in site-packages.
large_match="$(find "${ROOT}" -path "${ROOT}/.venv" -prune -o -type f -size +10M -print -quit)"
[[ -z "${large_match}" ]] || {
  echo "forbidden-payload guard: unexpected >10 MiB source payload: ${large_match}" >&2
  exit 1
}

echo "forbidden-payload guard: clean (${ROOT})"
