#!/usr/bin/env bash
# Build (and optionally push) npa-cosmos3-serving: Cosmos3-Super as a served
# endpoint on one 8-GPU node.
#
# The image wraps NVIDIA's vLLM-Omni runtime at a pinned digest and adds a
# preflight entrypoint. Model weights are never baked; they download at runtime
# with the operator's own Hugging Face credentials. No GPU is needed to build.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NPA_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
REGISTRY="${REGISTRY:-cr.eu-north1.nebius.cloud/e00cm0vc6t09m0z5gw}"
BASE_IMAGE="${COSMOS3_SERVING_BASE_IMAGE:-}"
TAG="${COSMOS3_SERVING_TAG:-0.1.0}"
PUSH=0

usage() {
  sed -n '2,6p' "$0"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --registry) REGISTRY="${2:?}"; shift 2 ;;
    --base-image) BASE_IMAGE="${2:?}"; shift 2 ;;
    --tag) TAG="${2:?}"; shift 2 ;;
    --push) PUSH=1; shift ;;
    -h | --help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

LOCAL_REF="npa-cosmos3-serving:${TAG}"
REMOTE_REF="${REGISTRY}/npa-cosmos3-serving:${TAG}"

BUILD_ARGS=()
if [[ -n "${BASE_IMAGE}" ]]; then
  BUILD_ARGS+=(--build-arg "BASE_IMAGE=${BASE_IMAGE}")
fi

echo "=== build ${LOCAL_REF} ==="
docker build --platform linux/amd64 \
  -f "${SCRIPT_DIR}/Dockerfile" \
  "${BUILD_ARGS[@]+"${BUILD_ARGS[@]}"}" \
  -t "${LOCAL_REF}" \
  -t "${REMOTE_REF}" \
  "${NPA_ROOT}"

if [[ "${PUSH}" == "1" ]]; then
  echo "=== push ${REMOTE_REF} ==="
  docker push "${REMOTE_REF}"
fi

echo "Done: ${REMOTE_REF} push=${PUSH}"
