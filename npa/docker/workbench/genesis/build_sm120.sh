#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
NPA_DIR=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)

REGISTRY=""
BASE_IMAGE=""
TAG=""
PUSH=0

usage() {
  cat <<'EOF'
Usage: build_sm120.sh --base-image IMAGE [--registry REGISTRY] [--tag TAG] [--push]

Builds an additive Genesis image for Blackwell sm_120 validation. The default
tag is npa-genesis:0.4.6-sm120-<UTC>. Registry pushes use the same tag under
REGISTRY/npa-genesis.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-image)
      BASE_IMAGE=${2:?missing --base-image value}
      shift 2
      ;;
    --registry)
      REGISTRY=${2:?missing --registry value}
      shift 2
      ;;
    --tag)
      TAG=${2:?missing --tag value}
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

if [[ -z "${TAG}" ]]; then
  TAG="0.4.6-sm120-$(date -u +%Y%m%dT%H%M%SZ)"
fi

LOCAL_IMAGE="npa-genesis:${TAG}"
BUILD_ARGS=(
  --build-arg "BASE_IMAGE=${BASE_IMAGE}"
  -f "${SCRIPT_DIR}/Dockerfile.sm120"
  -t "${LOCAL_IMAGE}"
)

if [[ -n "${REGISTRY}" ]]; then
  REGISTRY_IMAGE="${REGISTRY}/npa-genesis:${TAG}"
  BUILD_ARGS+=(-t "${REGISTRY_IMAGE}")
fi

docker build "${BUILD_ARGS[@]}" "${NPA_DIR}"

if [[ "${PUSH}" == "1" ]]; then
  if [[ -z "${REGISTRY}" ]]; then
    echo "--push requires --registry" >&2
    exit 2
  fi
  docker push "${REGISTRY_IMAGE}"
  echo "${REGISTRY_IMAGE}"
else
  echo "${LOCAL_IMAGE}"
fi
