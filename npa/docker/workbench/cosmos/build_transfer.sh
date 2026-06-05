#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NPA_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

REGISTRY=""
PUSH=0
DATE_TAG=""
BASE_IMAGE="nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04"
CUDA_NAME="cu128"
VERSION="2.5.0"
SOURCE_REPO="https://github.com/nvidia-cosmos/cosmos-transfer2.5.git"
MODEL_ID="nvidia/Cosmos-Transfer2.5-2B"

usage() {
  cat <<'EOF'
Usage: build_transfer.sh [--registry REGISTRY] [--push] [--date-tag TAG]

Builds the dedicated Cosmos Transfer 2.5 runner image as
npa-cosmos2-transfer:2.5.0. When --registry is provided, also tags
REGISTRY/npa-cosmos2-transfer:2.5.0. --date-tag adds an extra dated tag.
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
    --push)
      PUSH=1
      shift
      ;;
    --date-tag)
      if [ "$#" -lt 2 ]; then
        echo "ERROR: --date-tag requires a value" >&2
        exit 2
      fi
      DATE_TAG="$2"
      shift 2
      ;;
    --base-image)
      if [ "$#" -lt 2 ]; then
        echo "ERROR: --base-image requires a value" >&2
        exit 2
      fi
      BASE_IMAGE="$2"
      shift 2
      ;;
    --cuda-name)
      if [ "$#" -lt 2 ]; then
        echo "ERROR: --cuda-name requires a value" >&2
        exit 2
      fi
      CUDA_NAME="$2"
      shift 2
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

if [ "$PUSH" -eq 1 ] && [ -z "$REGISTRY" ]; then
  echo "ERROR: --push requires --registry" >&2
  exit 2
fi

LOCAL_IMAGE="npa-cosmos2-transfer:${VERSION}"
BUILD_ARGS=(
  -f "$SCRIPT_DIR/Dockerfile.transfer"
  --build-arg "BASE_IMAGE=${BASE_IMAGE}"
  --build-arg "CUDA_NAME=${CUDA_NAME}"
  --build-arg "COSMOS_TRANSFER_IMAGE_VERSION=${VERSION}"
  --build-arg "COSMOS_TRANSFER_SOURCE_REPO=${SOURCE_REPO}"
  --build-arg "COSMOS_TRANSFER_MODEL_ID=${MODEL_ID}"
  -t "$LOCAL_IMAGE"
)

REGISTRY_IMAGES=()
if [ -n "$REGISTRY" ]; then
  REGISTRY_IMAGE="${REGISTRY}/npa-cosmos2-transfer:${VERSION}"
  BUILD_ARGS+=(-t "$REGISTRY_IMAGE")
  REGISTRY_IMAGES+=("$REGISTRY_IMAGE")
  if [ -n "$DATE_TAG" ]; then
    DATED_IMAGE="${REGISTRY}/npa-cosmos2-transfer:${VERSION}-${DATE_TAG}"
    BUILD_ARGS+=(-t "$DATED_IMAGE")
    REGISTRY_IMAGES+=("$DATED_IMAGE")
  fi
fi

docker build "${BUILD_ARGS[@]}" "$NPA_ROOT"

SIZE_BYTES="$(docker image inspect "$LOCAL_IMAGE" --format '{{.Size}}')"
if command -v numfmt >/dev/null 2>&1; then
  SIZE="$(numfmt --to=iec-i --suffix=B "$SIZE_BYTES")"
else
  SIZE="${SIZE_BYTES} bytes"
fi

echo "Built: $LOCAL_IMAGE"
for image in "${REGISTRY_IMAGES[@]}"; do
  echo "Tagged: $image"
done
echo "Image size: $SIZE"

if [ "$PUSH" -eq 1 ]; then
  for image in "${REGISTRY_IMAGES[@]}"; do
    docker push "$image"
  done
fi
