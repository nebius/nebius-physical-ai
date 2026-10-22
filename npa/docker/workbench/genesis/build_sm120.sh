#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
NPA_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

REGISTRY=""
BASE_IMAGE=""
TAG=""
PUSH=0
NPA_SOURCE_SHA="${NPA_SOURCE_SHA:-$(git -C "${NPA_DIR}/.." rev-parse HEAD)}"

usage() {
  cat <<'EOF'
Usage: build_sm120.sh --base-image IMAGE [--registry REGISTRY] [--tag TAG] [--push]

Builds an additive Genesis image on the widened CUDA 13 base. The default tag
is npa-genesis:0.4.6-sm80-sm90-sm100-sm103-sm120-<UTC>. Registry pushes use
the same tag under REGISTRY/npa-genesis.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-image)
      BASE_IMAGE="${2:?missing --base-image value}"
      shift 2
      ;;
    --registry)
      REGISTRY="${2:?missing --registry value}"
      shift 2
      ;;
    --tag)
      TAG="${2:?missing --tag value}"
      shift 2
      ;;
    --push)
      PUSH=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${BASE_IMAGE}" ]]; then
  echo "--base-image is required" >&2
  usage >&2
  exit 2
fi

if [[ ! "${NPA_SOURCE_SHA}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "NPA_SOURCE_SHA must be the exact 40-character checkout SHA" >&2
  exit 2
fi

if [[ "$(git -C "${NPA_DIR}/.." rev-parse HEAD)" != "${NPA_SOURCE_SHA}" ]]; then
  echo "NPA_SOURCE_SHA must match the build-context checkout HEAD" >&2
  exit 2
fi

DOCKER=(docker)
if [[ -n "${DOCKER_CONTEXT:-}" ]]; then
  DOCKER+=(--context "${DOCKER_CONTEXT}")
fi
parent_source_sha="$("${DOCKER[@]}" image inspect --format '{{ index .Config.Labels "org.opencontainers.image.revision" }}' "${BASE_IMAGE}" 2>/dev/null || true)"
if [[ "${parent_source_sha}" != "${NPA_SOURCE_SHA}" ]]; then
  echo "BASE_IMAGE must carry org.opencontainers.image.revision=${NPA_SOURCE_SHA}; found ${parent_source_sha:-missing}" >&2
  exit 2
fi

if [[ -z "${TAG}" ]]; then
  TAG="0.4.6-sm80-sm90-sm100-sm103-sm120-$(date -u +%Y%m%dT%H%M%SZ)"
fi

LOCAL_IMAGE="npa-genesis:${TAG}"
BUILD_ARGS=(
  --build-arg "BASE_IMAGE=${BASE_IMAGE}"
  --build-arg "NPA_SOURCE_SHA=${NPA_SOURCE_SHA}"
  -f "${SCRIPT_DIR}/Dockerfile.sm120"
  -t "${LOCAL_IMAGE}"
)

if [[ -n "${REGISTRY}" ]]; then
  REGISTRY_IMAGE="${REGISTRY}/npa-genesis:${TAG}"
  BUILD_ARGS+=(-t "${REGISTRY_IMAGE}")
fi

"${DOCKER[@]}" build "${BUILD_ARGS[@]}" "${NPA_DIR}"

if [[ "${PUSH}" == "1" ]]; then
  if [[ -z "${REGISTRY}" ]]; then
    echo "--push requires --registry" >&2
    exit 2
  fi
  "${DOCKER[@]}" push "${REGISTRY_IMAGE}"
  echo "${REGISTRY_IMAGE}"
else
  echo "${LOCAL_IMAGE}"
fi
