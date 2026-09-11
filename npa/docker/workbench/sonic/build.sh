#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NPA_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

REGISTRY=""
PUSH=0
VARIANT="baked"
IMAGE_TAG_OVERRIDE=""
BASE_IMAGE_OVERRIDE=""

usage() {
  cat <<'EOF'
Usage: build.sh [--registry REGISTRY] [--push] [--variant baked|k8s|mujoco] [--tag TAG] [--base-image IMAGE]

Builds the SONIC runtime image as npa-sonic:<version> for --variant baked, or
npa-sonic:<version>-k8s-runtime for --variant k8s. The mujoco variant independently
builds npa-sonic-mujoco:<tag> on a digest-pinned public Python base.
When --tag is provided, it overrides the final image tag.
When --base-image is provided, it overrides the variant default base image.
When --registry is provided, also tags REGISTRY/<image-name>:<tag>.
Use --registry <your-registry>/<namespace> --push to publish.
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
    --variant)
      if [ "$#" -lt 2 ]; then
        echo "ERROR: --variant requires baked, k8s, or mujoco" >&2
        exit 2
      fi
      VARIANT="$2"
      shift 2
      ;;
    --tag)
      if [ "$#" -lt 2 ]; then
        echo "ERROR: --tag requires a value" >&2
        exit 2
      fi
      IMAGE_TAG_OVERRIDE="$2"
      shift 2
      ;;
    --base-image)
      if [ "$#" -lt 2 ]; then
        echo "ERROR: --base-image requires a value" >&2
        exit 2
      fi
      BASE_IMAGE_OVERRIDE="$2"
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

case "$VARIANT" in
  baked)
    TAG_SUFFIX=""
    # Docker Hub CUDA, digest-pinned and freely redistributable. Was
    # nvcr.io/nvidia/isaac-lab, which baked Omniverse Kit and needed an NGC login.
    BASE_IMAGE_DEFAULT="nvidia/cuda:13.0.2-cudnn-devel-ubuntu22.04@sha256:36c66a3ad4608580cf937f7ec3add9323a610956cfe8c9e2f99ef2ea0c896f01"
    ISAAC_LAB_PYTHON="/isaac-sim/python.sh"
    NPA_ISAAC_VENV="/opt/npa/sim/venv"
    NPA_ISAAC_SKIP_TORCH=0
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cu130"
    REQUIRE_TORCH_SM120=1
    # The container runtime injects the host driver + Vulkan ICD given
    # NVIDIA_DRIVER_CAPABILITIES=all; nothing driver-related is baked any more.
    NPA_DRIVER_PROVISIONING="host-mounted"
    NPA_CUDA_ARCHITECTURES="sm80,sm90,sm100,sm103,sm120"
    NPA_ISAAC_LAB_INSTALL_MODE="runtime-fetch-isaac-sim"
    NPA_RUNTIME_USER="ubuntu"
    ;;
  k8s)
    TAG_SUFFIX="-k8s-runtime"
    if [ -n "$REGISTRY" ]; then
      BASE_IMAGE_DEFAULT="${REGISTRY}/npa-base:cuda13-b300-sm80-sm90-sm100-sm103-sm120-v2-latest"
    else
      BASE_IMAGE_DEFAULT="npa-base:cuda13-b300-sm80-sm90-sm100-sm103-sm120-v2-latest"
    fi
    # Isaac always goes through the bootstrap shim, in both variants, so the k8s task
    # templates need no change. The base image's own python venv is reused for
    # everything else (NPA_ISAAC_SKIP_TORCH=1 keeps its CUDA-matched cu130 torch).
    ISAAC_LAB_PYTHON="/isaac-sim/python.sh"
    NPA_ISAAC_VENV="/opt/npa/venv"
    NPA_ISAAC_SKIP_TORCH=1
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cu130"
    REQUIRE_TORCH_SM120=1
    NPA_DRIVER_PROVISIONING="host-mounted"
    NPA_CUDA_ARCHITECTURES="sm80,sm90,sm100,sm103,sm120"
    NPA_ISAAC_LAB_INSTALL_MODE="runtime-fetch-isaac-sim"
    NPA_RUNTIME_USER="root"
    IMAGE_NAME="npa-sonic"
    DOCKERFILE="$SCRIPT_DIR/Dockerfile"
    DEFAULT_IMAGE_TAG=""
    ;;
  mujoco)
    TAG_SUFFIX=""
    BASE_IMAGE_DEFAULT="python:3.11.14-slim-bookworm@sha256:65a93d69fa75478d554f4ad27c85c1e69fa184956261b4301ebaf6dbb0a3543d"
    ISAAC_LAB_PYTHON="/isaac-sim/python.sh"
    # The mujoco layer must pip-install with the IMAGE's python, never with
    # ISAAC_LAB_PYTHON: that is now a bootstrap shim, so using it here would download
    # 4.5 GB of Isaac Sim during the BUILD and bake it into a layer -- exactly what this
    # whole change exists to prevent.
    NPA_IMAGE_PYTHON_DEFAULT="/opt/npa/venv/bin/python"
    NPA_ISAAC_VENV="/opt/npa/venv"
    NPA_ISAAC_SKIP_TORCH=1
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
    REQUIRE_TORCH_SM120=0
    NPA_DRIVER_PROVISIONING="host-mounted"
    NPA_CUDA_ARCHITECTURES="sm80,sm90,sm100,sm120"
    NPA_ISAAC_LAB_INSTALL_MODE="runtime-fetch-refusal-only"
    NPA_RUNTIME_USER="ubuntu"
    IMAGE_NAME="npa-sonic-mujoco"
    DOCKERFILE="$SCRIPT_DIR/Dockerfile.mujoco"
    DEFAULT_IMAGE_TAG="${NPA_SONIC_MUJOCO_TAG:-0.2.0-runtime}"
    ;;
  *)
    echo "ERROR: --variant must be baked, k8s, or mujoco, got: $VARIANT" >&2
    exit 2
    ;;
esac

PYTHON_BIN="${NPA_PYTHON_BIN:-$NPA_ROOT/.venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="$(command -v python3)"
fi

VERSION="$(
  cd "$NPA_ROOT"
  "$PYTHON_BIN" - <<'PY'
from pathlib import Path
import re

try:
    import tomllib
except ModuleNotFoundError:
    text = Path("pyproject.toml").read_text()
    section = text.split("[tool.npa.package-versions]", 1)[1]
    match = re.search(r'^sonic\s*=\s*"([^"]+)"', section, re.MULTILINE)
    if not match:
        raise SystemExit("Could not find [tool.npa.package-versions].sonic")
    print(match.group(1))
else:
    with Path("pyproject.toml").open("rb") as handle:
        data = tomllib.load(handle)
    print(data["tool"]["npa"]["package-versions"]["sonic"])
PY
)"

ISAAC_LAB_VERSION="$(
  cd "$NPA_ROOT"
  "$PYTHON_BIN" - <<'PY'
from pathlib import Path
import re

try:
    import tomllib
except ModuleNotFoundError:
    text = Path("pyproject.toml").read_text()
    section = text.split("[tool.npa.supported-tools]", 1)[1]
    match = re.search(r'^isaac-lab\s*=\s*"([^"]+)"', section, re.MULTILINE)
    if not match:
        raise SystemExit("Could not find [tool.npa.supported-tools].isaac-lab")
    print(match.group(1))
else:
    with Path("pyproject.toml").open("rb") as handle:
        data = tomllib.load(handle)
    print(data["tool"]["npa"]["supported-tools"]["isaac-lab"])
PY
)"

BOOTSTRAP="$SCRIPT_DIR/../common/isaac_bootstrap.sh"
ISAAC_SIM_VERSION="$(sed -n 's/^ISAAC_SIM_VERSION="${ISAAC_SIM_VERSION:-\(.*\)}"$/\1/p' "$BOOTSTRAP")"
ISAAC_LAB_SRC_COMMIT="$(sed -n 's/^ISAAC_LAB_SRC_COMMIT="${NPA_ISAAC_LAB_SRC_COMMIT:-\(.*\)}"$/\1/p' "$BOOTSTRAP")"
if [ -z "$ISAAC_SIM_VERSION" ] || [ -z "$ISAAC_LAB_SRC_COMMIT" ]; then
  echo "ERROR: could not read the Isaac pins out of $BOOTSTRAP" >&2
  exit 1
fi

if [ -z "${IMAGE_NAME:-}" ]; then
  IMAGE_NAME="npa-sonic"
fi
if [ -z "${DOCKERFILE:-}" ]; then
  DOCKERFILE="$SCRIPT_DIR/Dockerfile"
fi
if [ -z "${DEFAULT_IMAGE_TAG:-}" ]; then
  DEFAULT_IMAGE_TAG="${VERSION}${TAG_SUFFIX}"
fi

if [ "$VARIANT" = "mujoco" ]; then
  BASE_IMAGE="${BASE_IMAGE_OVERRIDE:-${NPA_SONIC_MUJOCO_BASE_IMAGE:-$BASE_IMAGE_DEFAULT}}"
else
  BASE_IMAGE="${BASE_IMAGE_OVERRIDE:-$BASE_IMAGE_DEFAULT}"
fi

IMAGE_TAG="${IMAGE_TAG_OVERRIDE:-${DEFAULT_IMAGE_TAG}}"
LOCAL_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"
BUILD_ARGS=(
  --platform linux/amd64
  -f "$DOCKERFILE"
  --build-arg "BASE_IMAGE=${BASE_IMAGE}"
  --build-arg "SONIC_VERSION=${VERSION}"
  --build-arg "ISAAC_LAB_VERSION=${ISAAC_LAB_VERSION}"
  --build-arg "ISAAC_LAB_PYTHON=${ISAAC_LAB_PYTHON}"
  --build-arg "ISAAC_SIM_VERSION=${ISAAC_SIM_VERSION}"
  --build-arg "ISAAC_LAB_SRC_COMMIT=${ISAAC_LAB_SRC_COMMIT}"
  --build-arg "NPA_ISAAC_VENV=${NPA_ISAAC_VENV}"
  --build-arg "NPA_ISAAC_SKIP_TORCH=${NPA_ISAAC_SKIP_TORCH}"
  --build-arg "TORCH_INDEX_URL=${TORCH_INDEX_URL}"
  --build-arg "REQUIRE_TORCH_SM120=${REQUIRE_TORCH_SM120}"
  --build-arg "NPA_DRIVER_PROVISIONING=${NPA_DRIVER_PROVISIONING}"
  --build-arg "NPA_CUDA_ARCHITECTURES=${NPA_CUDA_ARCHITECTURES}"
  --build-arg "NPA_ISAAC_LAB_INSTALL_MODE=${NPA_ISAAC_LAB_INSTALL_MODE}"
  --build-arg "NPA_RUNTIME_USER=${NPA_RUNTIME_USER}"
)

if [ "$VARIANT" = "mujoco" ]; then
  BUILD_ARGS+=(--build-arg "SONIC_MUJOCO_VERSION=${IMAGE_TAG}")
  BUILD_ARGS+=(--build-arg "NPA_IMAGE_PYTHON=${NPA_IMAGE_PYTHON:-$NPA_IMAGE_PYTHON_DEFAULT}")
fi

if [ -n "$REGISTRY" ]; then
  REGISTRY_IMAGE="${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
else
  REGISTRY_IMAGE=""
fi

if [ "$PUSH" -eq 1 ]; then
  BUILDX_BUILDER="${NPA_BUILDX_BUILDER:-npa-sonic-builder}"
  if ! docker buildx inspect "$BUILDX_BUILDER" >/dev/null 2>&1; then
    docker buildx create --name "$BUILDX_BUILDER" --driver docker-container --bootstrap >/dev/null
  fi
  docker buildx build --builder "$BUILDX_BUILDER" --push "${BUILD_ARGS[@]}" -t "$REGISTRY_IMAGE" "$NPA_ROOT"
  echo "Built and pushed: $REGISTRY_IMAGE"
  exit 0
fi

LOCAL_BUILD_TAGS=(-t "$LOCAL_IMAGE")
if [ -n "$REGISTRY_IMAGE" ]; then
  LOCAL_BUILD_TAGS+=(-t "$REGISTRY_IMAGE")
fi

docker build "${BUILD_ARGS[@]}" "${LOCAL_BUILD_TAGS[@]}" "$NPA_ROOT"

SIZE_BYTES="$(docker image inspect "$LOCAL_IMAGE" --format '{{.Size}}')"
if command -v numfmt >/dev/null 2>&1; then
  SIZE="$(numfmt --to=iec-i --suffix=B "$SIZE_BYTES")"
else
  SIZE="${SIZE_BYTES} bytes"
fi

echo "Built: $LOCAL_IMAGE"
if [ -n "$REGISTRY_IMAGE" ]; then
  echo "Tagged: $REGISTRY_IMAGE"
fi
echo "Image size: $SIZE"
