#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NPA_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
REPO_ROOT="$(cd "$NPA_ROOT/.." && pwd)"
NPA_PYTHON="${NPA_PYTHON_BIN:-$NPA_ROOT/.venv/bin/python}"
REGISTRY=""
PUSH=0
FLASH_ATTN_MAX_JOBS="${NPA_SEEDVR2_FLASH_ATTN_MAX_JOBS:-2}"
FLASH_ATTN_NVCC_THREADS="${NPA_SEEDVR2_FLASH_ATTN_NVCC_THREADS:-1}"
FLASH_ATTN_CUDA_ARCHS="${NPA_SEEDVR2_FLASH_ATTN_CUDA_ARCHS:-90}"
APEX_MAX_JOBS="${NPA_SEEDVR2_APEX_MAX_JOBS:-2}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --registry) REGISTRY="${2:?}"; shift 2 ;;
    --flash-attn-max-jobs) FLASH_ATTN_MAX_JOBS="${2:?}"; shift 2 ;;
    --flash-attn-nvcc-threads) FLASH_ATTN_NVCC_THREADS="${2:?}"; shift 2 ;;
    --flash-attn-cuda-archs) FLASH_ATTN_CUDA_ARCHS="${2:?}"; shift 2 ;;
    --apex-max-jobs) APEX_MAX_JOBS="${2:?}"; shift 2 ;;
    --push) PUSH=1; shift ;;
    -h|--help)
      echo "Usage: $0 [--registry HOST/PATH] [--flash-attn-max-jobs N] [--flash-attn-nvcc-threads N] [--flash-attn-cuda-archs LIST] [--apex-max-jobs N] [--push]"
      exit 0
      ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

for setting in \
  "flash-attn-max-jobs=$FLASH_ATTN_MAX_JOBS" \
  "flash-attn-nvcc-threads=$FLASH_ATTN_NVCC_THREADS" \
  "apex-max-jobs=$APEX_MAX_JOBS"; do
  value="${setting#*=}"
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
    echo "${setting%%=*} must be a positive integer" >&2
    exit 2
  }
done

[[ "$FLASH_ATTN_CUDA_ARCHS" =~ ^[0-9]+(\;[0-9]+)*$ ]] || {
  echo "flash-attn-cuda-archs must be a semicolon-separated numeric list" >&2
  exit 2
}

SOURCE_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"
[[ "$SOURCE_SHA" =~ ^[0-9a-f]{40}$ ]] || {
  echo "Full source SHA is required" >&2
  exit 1
}
if [[ "$PUSH" == 1 ]]; then
  echo "Push requires the trusted publication workflow and exact-image gates." >&2
  exit 1
fi
SOURCE_EPOCH="$(git -C "$REPO_ROOT" show -s --format=%ct "$SOURCE_SHA")"
[[ "$SOURCE_EPOCH" =~ ^[1-9][0-9]*$ ]] || {
  echo "Positive source commit epoch is required" >&2
  exit 1
}

BUILD_INPUTS=(
  npa/src/npa
  npa/pyproject.toml
  npa/README.md
  npa/.dockerignore
  npa/docker/workbench/seedvr2
  workflows/main
  workflows/testing
)
git -C "$REPO_ROOT" cat-file -e \
  "$SOURCE_SHA:npa/docker/workbench/seedvr2/Dockerfile" || {
  echo "Commit the SeedVR2 Dockerfile before building a dev-SHA image" >&2
  exit 1
}
if ! git -C "$REPO_ROOT" diff --quiet -- "${BUILD_INPUTS[@]}" || \
   ! git -C "$REPO_ROOT" diff --cached --quiet "$SOURCE_SHA" -- "${BUILD_INPUTS[@]}"; then
  echo "Commit SeedVR2 build inputs before building a dev-SHA image" >&2
  exit 1
fi
UNTRACKED="$(
  git -C "$REPO_ROOT" ls-files --others --exclude-standard -- "${BUILD_INPUTS[@]}"
)"
if [[ -n "$UNTRACKED" ]]; then
  echo "Untracked SeedVR2 build inputs must be committed or removed" >&2
  exit 1
fi

SNAPSHOT="$(mktemp -d "${TMPDIR:-/tmp}/npa-seedvr2-build.XXXXXXXX")"
trap 'rm -rf -- "$SNAPSHOT"' EXIT
git -C "$REPO_ROOT" archive "$SOURCE_SHA" -- "${BUILD_INPUTS[@]}" | tar -x -C "$SNAPSHOT"
BUILD_CONTEXT="$SNAPSHOT/npa"
[[ -x "$NPA_PYTHON" ]] || {
  echo "Repository Python is required for catalog staging" >&2
  exit 1
}
"$NPA_PYTHON" "$BUILD_CONTEXT/src/npa/workflow_build.py" \
  --stage-catalog --package-root "$BUILD_CONTEXT"

IMAGE="${REGISTRY:+${REGISTRY%/}/}npa-seedvr2:dev-${SOURCE_SHA}"
env -u HF_TOKEN -u NGC_API_KEY -u NVIDIA_API_KEY -u NEBIUS_IAM_TOKEN \
  -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY \
  docker buildx build --platform linux/amd64 \
    --file "$BUILD_CONTEXT/docker/workbench/seedvr2/Dockerfile" \
    --tag "$IMAGE" \
    --build-arg "NPA_SOURCE_SHA=$SOURCE_SHA" \
    --build-arg "SOURCE_DATE_EPOCH=$SOURCE_EPOCH" \
    --build-arg "FLASH_ATTN_MAX_JOBS=$FLASH_ATTN_MAX_JOBS" \
    --build-arg "FLASH_ATTN_NVCC_THREADS=$FLASH_ATTN_NVCC_THREADS" \
    --build-arg "FLASH_ATTN_CUDA_ARCHS=$FLASH_ATTN_CUDA_ARCHS" \
    --build-arg "APEX_MAX_JOBS=$APEX_MAX_JOBS" \
    --label "org.opencontainers.image.source=https://github.com/nebius/nebius-physical-ai" \
    --load --provenance=false "$BUILD_CONTEXT"

printf '%s\n' "$IMAGE"
