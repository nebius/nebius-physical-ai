#!/usr/bin/env bash
# Build (and optionally push) npa-open3d: CPU point-cloud registration and
# Poisson surface reconstruction. Open3D is MIT and installs from the official
# PyPI wheel; no model weights, datasets or credentials are baked. The upstream
# open3d.data demo scans download at runtime under the operator's own access.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NPA_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
REGISTRY="${REGISTRY:-}"
OPEN3D_VERSION="${OPEN3D_VERSION:-0.20.0}"
# The tag this Dockerfile produces. CPU-only, so it carries no CUDA family.
IMAGE_TAG="${OPEN3D_IMAGE_TAG:-0.20.0-cpu-20260918}"
NPA_SOURCE_SHA="${NPA_SOURCE_SHA:-$(git -C "${NPA_ROOT}" rev-parse HEAD)}"
PUSH=0
TAG=""

usage() {
  sed -n '2,5p' "$0"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --registry) REGISTRY="${2:?}"; shift 2 ;;
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

TAG="${TAG:-${IMAGE_TAG}}"
IMAGE="${REGISTRY}/npa-open3d:${TAG}"

echo "Building ${IMAGE} (open3d ${OPEN3D_VERSION}, source ${NPA_SOURCE_SHA})"
docker build \
  --build-arg OPEN3D_VERSION="${OPEN3D_VERSION}" \
  --build-arg NPA_SOURCE_SHA="${NPA_SOURCE_SHA}" \
  -t "${IMAGE}" \
  -f "${SCRIPT_DIR}/Dockerfile" \
  "${NPA_ROOT}"

if [[ "${PUSH}" == "1" ]]; then
  echo "Pushing ${IMAGE}"
  docker push "${IMAGE}"
fi

echo "Built ${IMAGE}"
