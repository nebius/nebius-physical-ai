#!/usr/bin/env bash
# Build (and optionally push) npa-robocasa: the RoboCasa kitchen-task simulation
# runtime. RoboCasa and robosuite are Apache-2.0; kitchen assets are never baked
# and download at runtime under the operator's own network access.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NPA_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
REGISTRY="${REGISTRY:-}"
BASE_IMAGE="${ROBOCASA_BASE_IMAGE:-nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04}"
ROBOCASA_VERSION="${ROBOCASA_VERSION:-0.1.0}"
NPA_SOURCE_SHA="${NPA_SOURCE_SHA:-$(git -C "${NPA_ROOT}" rev-parse HEAD)}"
PUSH=0
TAG=""

usage() {
  sed -n '2,5p' "$0"
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

if [[ -z "${REGISTRY}" ]]; then
  echo "ERROR: pass --registry or set REGISTRY to an authorized registry" >&2
  exit 2
fi
if [[ ! "${NPA_SOURCE_SHA}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "ERROR: NPA_SOURCE_SHA must be the exact 40-character checkout SHA" >&2
  exit 2
fi

TAG="${TAG:-dev-${NPA_SOURCE_SHA}}"
IMAGE="${REGISTRY}/npa-robocasa:${TAG}"

echo "Building ${IMAGE} from ${BASE_IMAGE} (source ${NPA_SOURCE_SHA})"
docker build \
  --build-arg BASE_IMAGE="${BASE_IMAGE}" \
  --build-arg ROBOCASA_VERSION="${ROBOCASA_VERSION}" \
  --build-arg NPA_SOURCE_SHA="${NPA_SOURCE_SHA}" \
  -t "${IMAGE}" \
  -f "${SCRIPT_DIR}/Dockerfile" \
  "${NPA_ROOT}"

if [[ "${PUSH}" == "1" ]]; then
  echo "Pushing ${IMAGE}"
  docker push "${IMAGE}"
fi

echo "Built ${IMAGE}"
