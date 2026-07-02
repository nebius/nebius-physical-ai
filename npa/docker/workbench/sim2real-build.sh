#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NPA_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

REGISTRY=""
PUSH=0
BASE_IMAGE="${BASE_IMAGE:-npa-base:cuda13-b300-sm80-sm90-sm120-latest}"
GENESIS_IMAGE="${GENESIS_IMAGE:-npa-genesis:0.4.6-sm80-sm90-sm120-latest}"
VLM_TAG="${VLM_TAG:-3.0.1-genuine-sm120}"
ENVGEN_TAG="${ENVGEN_TAG:-0.1.2}"
EVAL_TAG="${EVAL_TAG:-0.1.2-genuine-sm120}"
VLM_RL_TAG="${VLM_RL_TAG:-0.1.1}"

usage() {
  cat <<EOF
Usage: sim2real-build.sh [--registry REGISTRY] [--push]

Builds the Sim2Real reference images one at a time:
  npa-cosmos3-reason:${VLM_TAG} (skipped when SKIP_COSMOS3_REASON=1)
  npa-envgen:${ENVGEN_TAG}
  npa-reference-policy:${ENVGEN_TAG}
  npa-lerobot-vlm-rl:${VLM_RL_TAG}
  npa-loop-eval:${EVAL_TAG}
  npa-rerun-viewer:${RERUN_VIEWER_TAG:-0.31.4}

Set BASE_IMAGE and GENESIS_IMAGE to the pushed CUDA 13 / sm80-sm90-sm120 base
and Genesis image tags before building. Set ENVGEN_TAG when the reference
policy image should build from a non-default envgen tag. Set VLM_TAG and
EVAL_TAG for additive component rebuilds.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --registry)
      if [ "$#" -lt 2 ]; then
        echo "ERROR: --registry requires a value" >&2
        exit 2
      fi
      REGISTRY="${2%/}"
      shift 2
      ;;
    --push)
      PUSH=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [ "$PUSH" -eq 1 ] && [ -z "$REGISTRY" ]; then
  echo "ERROR: --push requires --registry" >&2
  exit 2
fi

build_one() {
  local name="$1"
  local tag="$2"
  local dockerfile="$3"
  local base_arg="$4"
  local local_image="${name}:${tag}"
  local registry_image=""
  local args=(-f "${dockerfile}" --build-arg "${base_arg}" -t "${local_image}")
  if [ -n "$REGISTRY" ]; then
    registry_image="${REGISTRY}/${name}:${tag}"
    args+=(-t "${registry_image}")
  fi
  docker build "${args[@]}" "${NPA_ROOT}"
  echo "Built: ${local_image}"
  if [ -n "$registry_image" ]; then
    echo "Tagged: ${registry_image}"
  fi
  if [ "$PUSH" -eq 1 ]; then
    docker push "${registry_image}"
  fi
}

if [ -n "${SKIP_COSMOS3_REASON:-}" ]; then
  echo "Skipping npa-cosmos3-reason (SKIP_COSMOS3_REASON=${SKIP_COSMOS3_REASON})"
else
  build_one "npa-cosmos3-reason" "${VLM_TAG}" "${SCRIPT_DIR}/cosmos3-reason/Dockerfile" "BASE_IMAGE=${BASE_IMAGE}"
fi
build_one "npa-envgen" "${ENVGEN_TAG}" "${SCRIPT_DIR}/sim2real-envgen/Dockerfile" "BASE_IMAGE=${GENESIS_IMAGE}"
build_one "npa-reference-policy" "${ENVGEN_TAG}" "${SCRIPT_DIR}/sim2real-reference-policy/Dockerfile" "BASE_IMAGE=npa-envgen:${ENVGEN_TAG}"
build_one "npa-lerobot-vlm-rl" "${VLM_RL_TAG}" "${SCRIPT_DIR}/lerobot-vlm-rl/Dockerfile" "BASE_IMAGE=${GENESIS_IMAGE}"
build_one "npa-loop-eval" "${EVAL_TAG}" "${SCRIPT_DIR}/sim2real-eval/Dockerfile" "BASE_IMAGE=${GENESIS_IMAGE}"
build_one "npa-rerun-viewer" "${RERUN_VIEWER_TAG:-0.31.4}" "${SCRIPT_DIR}/rerun-viewer/Dockerfile" "RERUN_SDK_VERSION=${RERUN_VIEWER_TAG:-0.31.4}"
