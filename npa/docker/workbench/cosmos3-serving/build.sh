#!/usr/bin/env bash
# Build (and optionally push) npa-cosmos3-serving: Cosmos3-Super as a served
# endpoint on one 8-GPU node.
#
# The public image is a zero-payload bootstrap. The pinned serving closure,
# source, models, and guardrails are operator-entitled runtime fetches.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NPA_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
REGISTRY="${REGISTRY:-}"
BASE_IMAGE="${COSMOS3_SERVING_BASE_IMAGE:-}"
TAG="${COSMOS3_SERVING_TAG:-0.2.0}"
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

if [[ "${PUSH}" == "1" && -z "${REGISTRY}" ]]; then
  echo "ERROR: --push requires --registry" >&2
  exit 2
fi

LOCAL_REF="npa-cosmos3-serving:${TAG}"
REMOTE_REF=""
if [[ -n "${REGISTRY}" ]]; then
  REMOTE_REF="${REGISTRY%/}/npa-cosmos3-serving:${TAG}"
fi

BUILD_ARGS=()
if [[ -n "${BASE_IMAGE}" ]]; then
  BUILD_ARGS+=(--build-arg "BASE_IMAGE=${BASE_IMAGE}")
fi

echo "=== build ${LOCAL_REF} ==="
docker build --platform linux/amd64 \
  -f "${SCRIPT_DIR}/Dockerfile" \
  "${BUILD_ARGS[@]+"${BUILD_ARGS[@]}"}" \
  -t "${LOCAL_REF}" \
  "${NPA_ROOT}"

if [[ -n "${REMOTE_REF}" ]]; then
  docker tag "${LOCAL_REF}" "${REMOTE_REF}"
fi

if [[ "${PUSH}" == "1" ]]; then
  echo "=== push ${REMOTE_REF} ==="
  docker push "${REMOTE_REF}"
fi

echo "Done: ${REMOTE_REF:-$LOCAL_REF} push=${PUSH}"
