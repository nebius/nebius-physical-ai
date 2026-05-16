#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REGISTRY=""
TAG="${TS:-$(date -u +%Y%m%dT%H%M%SZ)}"
PUSH=0
DOCKER_CONTEXT="${DOCKER_CONTEXT:-}"
CUDA_BASE_TAG="${CUDA_BASE_TAG:-13.0.1-cudnn-devel-ubuntu22.04}"
FLASH_ATTN_COMMIT="${FLASH_ATTN_COMMIT:-0409f9adcbdebff6cc19eb95f370d40e896980bc}"

usage() {
  cat <<'EOF'
Usage: build.sh [--registry REGISTRY] [--tag TAG] [--push]

Builds npa-base:cuda13-b300-${TAG}. Set DOCKER_CONTEXT to build on a remote
Docker context, for example an SSH-accessible B300 VM.
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
    --tag)
      if [ "$#" -lt 2 ]; then
        echo "ERROR: --tag requires a value" >&2
        exit 2
      fi
      TAG="$2"
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

LOCAL_IMAGE="npa-base:cuda13-b300-${TAG}"
BUILD_ARGS=(
  build
  --build-arg "BUILD_TS=${TAG}"
  --build-arg "CUDA_BASE_TAG=${CUDA_BASE_TAG}"
  --build-arg "FLASH_ATTN_COMMIT=${FLASH_ATTN_COMMIT}"
  -t "$LOCAL_IMAGE"
)

if [ -n "$REGISTRY" ]; then
  REGISTRY_IMAGE="${REGISTRY}/npa-base:cuda13-b300-${TAG}"
  BUILD_ARGS+=(-t "$REGISTRY_IMAGE")
else
  REGISTRY_IMAGE=""
fi

if [ -n "$DOCKER_CONTEXT" ]; then
  docker --context "$DOCKER_CONTEXT" "${BUILD_ARGS[@]}" "$SCRIPT_DIR"
else
  docker "${BUILD_ARGS[@]}" "$SCRIPT_DIR"
fi

echo "Built: $LOCAL_IMAGE"
if [ -n "$REGISTRY_IMAGE" ]; then
  echo "Tagged: $REGISTRY_IMAGE"
fi

if [ "$PUSH" -eq 1 ]; then
  if [ -z "$REGISTRY_IMAGE" ]; then
    echo "ERROR: --push requires --registry" >&2
    exit 2
  fi
  if [ -n "$DOCKER_CONTEXT" ]; then
    docker --context "$DOCKER_CONTEXT" push "$REGISTRY_IMAGE"
  else
    docker push "$REGISTRY_IMAGE"
  fi
fi
