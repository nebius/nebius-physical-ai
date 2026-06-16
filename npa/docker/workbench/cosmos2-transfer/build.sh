#!/usr/bin/env bash
# Build (and optionally push) the golden-eval wrapper for npa-cosmos2-transfer.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NPA_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
REGISTRY="${REGISTRY:-cr.eu-north1.nebius.cloud/e00cm0vc6t09m0z5gw}"
BASE_IMAGE="${COSMOS2_TRANSFER_BASE_IMAGE:-${REGISTRY}/npa-cosmos2-transfer:2.5.0}"
PUSH=0
TAG=""

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

if [[ -z "${TAG}" ]]; then
  TAG="$("${NPA_ROOT}/.venv/bin/python" - <<'PY'
from npa.deploy.images import supported_tool_version
print(supported_tool_version("cosmos2-transfer"))
PY
)"
fi

LOCAL_REF="npa-cosmos2-transfer:${TAG}"
REMOTE_REF="${REGISTRY}/npa-cosmos2-transfer:${TAG}"

echo "=== build cosmos2-transfer wrapper ${LOCAL_REF} (base=${BASE_IMAGE}) ==="
docker build --platform linux/amd64 \
  -f "${SCRIPT_DIR}/Dockerfile" \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  -t "${LOCAL_REF}" \
  -t "${REMOTE_REF}" \
  "${NPA_ROOT}"

if [[ "${PUSH}" == "1" ]]; then
  echo "=== push ${REMOTE_REF} ==="
  docker push "${REMOTE_REF}"
fi

echo "Done: ${REMOTE_REF} push=${PUSH}"
