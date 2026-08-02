#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REGISTRY=""
TAG="${TS:-$(date -u +%Y%m%dT%H%M%SZ)}"
PUSH=0
DOCKER_CONTEXT="${DOCKER_CONTEXT:-}"
CUDA_BASE_TAG="${CUDA_BASE_TAG:-13.0.1-cudnn-devel-ubuntu22.04}"
FLASH_ATTN_COMMIT="${FLASH_ATTN_COMMIT:-0409f9adcbdebff6cc19eb95f370d40e896980bc}"
# Pinned because flash-attn-4's unbounded cutlass/quack ranges resolve to a pair
# that fails to import; see the Dockerfile comment. Bump these together.
CUTLASS_DSL_VERSION="${CUTLASS_DSL_VERSION:-4.5.3}"
QUACK_KERNELS_VERSION="${QUACK_KERNELS_VERSION:-0.5.0}"
# Datacenter Blackwell needs both CUDA majors: 10.0/10.3 (B200/B300) and 12.0
# (RTX PRO 6000). sm_103 is omitted from the assertion because stock cu130
# wheels ship sm_100 SASS and rely on 10.0 -> 10.3 forward compatibility.
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0 9.0 10.0 10.3 12.0}"
REQUIRE_TORCH_ARCHS="${REQUIRE_TORCH_ARCHS:-sm_80 sm_90 sm_100 sm_120}"

usage() {
  cat <<'EOF'
Usage: build.sh [--registry REGISTRY] [--tag TAG] [--push]
                [--arch-list "8.0 9.0 10.0 10.3 12.0"]
                [--require-archs "sm_80 sm_90 sm_100 sm_120"]

Builds npa-base:cuda13-b300-${TAG}. Set DOCKER_CONTEXT to build on a remote
Docker context, for example an SSH-accessible B300 VM.

--arch-list sets TORCH_CUDA_ARCH_LIST for source-compiled CUDA extensions in
this image and every child image that inherits the env. --require-archs fails
the build when the prebuilt torch wheel does not report those architectures in
torch.cuda.get_arch_list(); pass an empty string to skip the assertion.

Equivalent env vars: TORCH_CUDA_ARCH_LIST, REQUIRE_TORCH_ARCHS, CUDA_BASE_TAG,
FLASH_ATTN_COMMIT, CUTLASS_DSL_VERSION, QUACK_KERNELS_VERSION, DOCKER_CONTEXT.
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
    --arch-list)
      if [ "$#" -lt 2 ]; then
        echo "ERROR: --arch-list requires a value" >&2
        exit 2
      fi
      TORCH_CUDA_ARCH_LIST="$2"
      shift 2
      ;;
    --require-archs)
      if [ "$#" -lt 2 ]; then
        echo "ERROR: --require-archs requires a value" >&2
        exit 2
      fi
      REQUIRE_TORCH_ARCHS="$2"
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
  --build-arg "CUTLASS_DSL_VERSION=${CUTLASS_DSL_VERSION}"
  --build-arg "QUACK_KERNELS_VERSION=${QUACK_KERNELS_VERSION}"
  --build-arg "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
  --build-arg "REQUIRE_TORCH_ARCHS=${REQUIRE_TORCH_ARCHS}"
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
