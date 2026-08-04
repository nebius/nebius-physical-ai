#!/usr/bin/env bash
# Build (and optionally push) npa-cosmos3: the Cosmos 3 generation runtime.
#
# The image carries framework SOURCE + its inference env only. Model weights are
# never baked; they download at runtime with the operator's own HF/NGC credentials.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NPA_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
REGISTRY="${REGISTRY:-cr.eu-north1.nebius.cloud/e00cm0vc6t09m0z5gw}"
BASE_IMAGE="${COSMOS3_BASE_IMAGE:-nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04}"
COSMOS3_REF="${COSMOS3_REF:-}"
PUSH=0
TAG=""

usage() {
  sed -n '2,5p' "$0"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --registry) REGISTRY="${2:?}"; shift 2 ;;
    --base-image) BASE_IMAGE="${2:?}"; shift 2 ;;
    --ref) COSMOS3_REF="${2:?}"; shift 2 ;;
    --tag) TAG="${2:?}"; shift 2 ;;
    --push) PUSH=1; shift ;;
    -h | --help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${TAG}" ]]; then
  TAG="$("${NPA_ROOT}/.venv/bin/python" - <<'PY'
from npa.deploy.images import supported_tool_version
print(supported_tool_version("cosmos3"))
PY
)"
fi

LOCAL_REF="npa-cosmos3:${TAG}"
REMOTE_REF="${REGISTRY}/npa-cosmos3:${TAG}"

BUILD_ARGS=(--build-arg "BASE_IMAGE=${BASE_IMAGE}")
if [[ -n "${COSMOS3_REF}" ]]; then
  BUILD_ARGS+=(--build-arg "COSMOS3_REF=${COSMOS3_REF}")
fi

echo "=== build ${LOCAL_REF} (base=${BASE_IMAGE}) ==="
docker build --platform linux/amd64 \
  -f "${SCRIPT_DIR}/Dockerfile" \
  "${BUILD_ARGS[@]}" \
  -t "${LOCAL_REF}" \
  -t "${REMOTE_REF}" \
  "${NPA_ROOT}"

if [[ "${PUSH}" == "1" ]]; then
  echo "=== push ${REMOTE_REF} ==="
  docker push "${REMOTE_REF}"
fi

echo "Done: ${REMOTE_REF} push=${PUSH}"
