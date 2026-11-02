#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
NPA_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

REGISTRY=""
TAG=""
PUSH=0
NPA_SOURCE_SHA="${NPA_SOURCE_SHA:-$(git -C "${NPA_DIR}/.." rev-parse HEAD)}"

usage() {
  cat <<'EOF'
Usage: build.sh [--registry REGISTRY] [--tag TAG] [--push]

Build the canonical Python 3.12 Genesis recipe from its digest-pinned CUDA
base. The resulting image must attest the exact checked-out source SHA.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --registry) REGISTRY="${2:?missing --registry value}"; shift 2 ;;
    --tag) TAG="${2:?missing --tag value}"; shift 2 ;;
    --push) PUSH=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ! "${NPA_SOURCE_SHA}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "ERROR: NPA_SOURCE_SHA must be the exact 40-character checkout SHA" >&2
  exit 2
fi
if [[ "$(git -C "${NPA_DIR}/.." rev-parse HEAD)" != "${NPA_SOURCE_SHA}" ]]; then
  echo "ERROR: NPA_SOURCE_SHA must match the build-context checkout HEAD" >&2
  exit 2
fi
if [[ "${PUSH}" == "1" && -z "${REGISTRY}" ]]; then
  echo "ERROR: --push requires --registry" >&2
  exit 2
fi

if [[ -z "${TAG}" ]]; then
  TAG="dev-${NPA_SOURCE_SHA}"
fi
LOCAL_IMAGE="npa-genesis:${TAG}"
BUILD_ARGS=(
  --build-arg "NPA_SOURCE_SHA=${NPA_SOURCE_SHA}"
  -f "${SCRIPT_DIR}/Dockerfile"
  -t "${LOCAL_IMAGE}"
)
if [[ -n "${REGISTRY}" ]]; then
  REGISTRY_IMAGE="${REGISTRY%/}/npa-genesis:${TAG}"
  BUILD_ARGS+=(-t "${REGISTRY_IMAGE}")
fi

docker build "${BUILD_ARGS[@]}" "${NPA_DIR}"
image_source_sha="$(docker image inspect --format '{{ index .Config.Labels "org.opencontainers.image.revision" }}' "${LOCAL_IMAGE}" 2>/dev/null || true)"
if [[ "${image_source_sha}" != "${NPA_SOURCE_SHA}" ]]; then
  echo "ERROR: built Genesis image must carry org.opencontainers.image.revision=${NPA_SOURCE_SHA}; found ${image_source_sha:-missing}" >&2
  exit 4
fi
if [[ "${PUSH}" == "1" ]]; then
  docker push "${REGISTRY_IMAGE}"
  echo "${REGISTRY_IMAGE}"
else
  echo "${LOCAL_IMAGE}"
fi
