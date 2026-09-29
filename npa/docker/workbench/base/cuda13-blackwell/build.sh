#!/usr/bin/env bash
set -euo pipefail

# Both canonical and legacy entrypoints use this build context.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

REGISTRY=""
TAG="${TS:-$(date -u +%Y%m%dT%H%M%SZ)}"
PUSH=0
ATTENTION_BACKEND=fa4
FA2_CUDA_ARCHS="${FA2_CUDA_ARCHS:-120}"
DOCKER_CONTEXT="${DOCKER_CONTEXT:-}"
CUDA_BASE_TAG="${CUDA_BASE_TAG:-13.0.1-cudnn-devel-ubuntu22.04}"
FLASH_ATTN_COMMIT="${FLASH_ATTN_COMMIT:-eed1971f5132630dc296fe37601e834d4b57a248}"
# Quack pins the DSL exactly; keep these aligned with the Dockerfile.
CUTLASS_DSL_VERSION="${CUTLASS_DSL_VERSION:-4.6.2}"
QUACK_KERNELS_VERSION="${QUACK_KERNELS_VERSION:-0.6.4}"
# Datacenter Blackwell needs both CUDA majors: 10.0/10.3 (B200/B300) and 12.0
# (RTX PRO 6000). sm_103 is omitted from the assertion because stock cu130
# wheels ship sm_100 SASS and rely on 10.0 -> 10.3 forward compatibility.
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0 9.0 10.0 10.3 12.0}"
REQUIRE_TORCH_ARCHS="${REQUIRE_TORCH_ARCHS:-sm_80 sm_90 sm_100 sm_120}"

usage() {
  cat <<'EOF'
Usage: build.sh [--registry REGISTRY] [--tag TAG] [--push]
                [--attention-backend fa4|fa2]
                [--arch-list "8.0 9.0 10.0 10.3 12.0"]
                [--require-archs "sm_80 sm_90 sm_100 sm_120"]

Builds npa-base:cuda13-blackwell-${TAG} for RTX PRO 6000 and B200/B300 targets.
The same image also receives cuda13-b300-${TAG} for existing consumers.
--registry tags both names; --push pushes both. No published tag is changed
unless you explicitly build/push that exact suffix. Set DOCKER_CONTEXT to use
a remote Docker daemon. GPU capability still requires per-image validation.
FA4 is the default. --attention-backend fa2 instead builds the separate
cuda13-blackwell-fa2-${TAG} image (legacy alias cuda13-b300-fa2-${TAG}).
FA2 is source-compiled for SM120 by default; FA2_CUDA_ARCHS overrides that list.

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
    --attention-backend)
      if [ "$#" -lt 2 ]; then
        echo "ERROR: --attention-backend requires fa4 or fa2" >&2
        exit 2
      fi
      ATTENTION_BACKEND="$2"
      shift 2
      ;;
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

case "$ATTENTION_BACKEND" in
  fa4) BACKEND_PREFIX="" ;;
  fa2) BACKEND_PREFIX="fa2-" ;;
  *) echo "ERROR: --attention-backend requires fa4 or fa2" >&2; exit 2 ;;
esac

if [ "$PUSH" -eq 1 ] && [ -z "$REGISTRY" ]; then
  echo "ERROR: --push requires --registry" >&2
  exit 2
fi

LOCAL_IMAGE="npa-base:cuda13-blackwell-${BACKEND_PREFIX}${TAG}"
LEGACY_IMAGE="npa-base:cuda13-b300-${BACKEND_PREFIX}${TAG}"
BUILD_ARGS=(
  build
  --build-arg "BUILD_TS=${TAG}"
  --build-arg "CUDA_BASE_TAG=${CUDA_BASE_TAG}"
  --build-arg "FLASH_ATTN_COMMIT=${FLASH_ATTN_COMMIT}"
  --build-arg "ATTENTION_BACKEND=${ATTENTION_BACKEND}"
  --build-arg "FA2_CUDA_ARCHS=${FA2_CUDA_ARCHS}"
  --build-arg "CUTLASS_DSL_VERSION=${CUTLASS_DSL_VERSION}"
  --build-arg "QUACK_KERNELS_VERSION=${QUACK_KERNELS_VERSION}"
  --build-arg "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
  --build-arg "REQUIRE_TORCH_ARCHS=${REQUIRE_TORCH_ARCHS}"
  -t "$LOCAL_IMAGE"
  -t "$LEGACY_IMAGE"
)

if [ -n "$REGISTRY" ]; then
  REGISTRY_IMAGE="${REGISTRY}/npa-base:cuda13-blackwell-${BACKEND_PREFIX}${TAG}"
  LEGACY_REGISTRY_IMAGE="${REGISTRY}/npa-base:cuda13-b300-${BACKEND_PREFIX}${TAG}"
  BUILD_ARGS+=(-t "$REGISTRY_IMAGE" -t "$LEGACY_REGISTRY_IMAGE")
else
  REGISTRY_IMAGE=""
  LEGACY_REGISTRY_IMAGE=""
fi

if [ -n "$DOCKER_CONTEXT" ]; then
  docker --context "$DOCKER_CONTEXT" "${BUILD_ARGS[@]}" "$SCRIPT_DIR"
else
  docker "${BUILD_ARGS[@]}" "$SCRIPT_DIR"
fi

echo "Built: $LOCAL_IMAGE"
echo "Compatibility alias: $LEGACY_IMAGE"
if [ -n "$REGISTRY_IMAGE" ]; then
  echo "Tagged: $REGISTRY_IMAGE"
  echo "Compatibility alias: $LEGACY_REGISTRY_IMAGE"
fi

if [ "$PUSH" -eq 1 ]; then
  for IMAGE in "$REGISTRY_IMAGE" "$LEGACY_REGISTRY_IMAGE"; do
    if [ -n "$DOCKER_CONTEXT" ]; then
      docker --context "$DOCKER_CONTEXT" push "$IMAGE"
    else
      docker push "$IMAGE"
    fi
  done
fi
