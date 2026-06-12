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
npa-sonic:<version>-k8s-runtime for --variant k8s. The mujoco variant builds the
additive npa-sonic-mujoco:<tag> image from an existing SONIC base image.
When --tag is provided, it overrides the final image tag.
When --base-image is provided, it overrides the variant default base image.
When --registry is provided, also tags REGISTRY/<image-name>:<tag>.
Use --registry cr.eu-north1.nebius.cloud/<your-registry-id> --push to publish.
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
    BASE_IMAGE_DEFAULT="nvcr.io/nvidia/isaac-lab:2.3.2@sha256:388dbc806f48359a964cb9f807feb226da95d0a107f470fdcad9780ea10fe6f2"
    ISAAC_LAB_PYTHON="/isaac-sim/python.sh"
    INSTALL_NVIDIA_DRIVER_USERSPACE=1
    INSTALL_ISAACSIM_EXTRA=0
    REQUIRE_TORCH_SM120=0
    NPA_DRIVER_PROVISIONING="baked"
    NPA_CUDA_ARCHITECTURES="sm80,sm90"
    NPA_ISAAC_LAB_INSTALL_MODE="preinstalled-isaac-sim"
    NPA_RUNTIME_USER="ubuntu"
    ;;
  k8s)
    TAG_SUFFIX="-k8s-runtime"
    if [ -n "$REGISTRY" ]; then
      BASE_IMAGE_DEFAULT="${REGISTRY}/npa-base:cuda13-b300-sm80-sm90-sm120-latest"
    else
      BASE_IMAGE_DEFAULT="npa-base:cuda13-b300-sm80-sm90-sm120-latest"
    fi
    ISAAC_LAB_PYTHON="/opt/npa/venv/bin/python"
    INSTALL_NVIDIA_DRIVER_USERSPACE=0
    INSTALL_ISAACSIM_EXTRA=1
    REQUIRE_TORCH_SM120=1
    NPA_DRIVER_PROVISIONING="host-mounted"
    NPA_CUDA_ARCHITECTURES="sm80,sm90,sm120"
    NPA_ISAAC_LAB_INSTALL_MODE="pip-isaacsim-on-cuda13-sm120"
    NPA_RUNTIME_USER="root"
    IMAGE_NAME="npa-sonic"
    DOCKERFILE="$SCRIPT_DIR/Dockerfile"
    DEFAULT_IMAGE_TAG=""
    ;;
  mujoco)
    TAG_SUFFIX=""
    BASE_IMAGE_DEFAULT=""
    ISAAC_LAB_PYTHON="/isaac-sim/python.sh"
    INSTALL_NVIDIA_DRIVER_USERSPACE=0
    INSTALL_ISAACSIM_EXTRA=0
    REQUIRE_TORCH_SM120=0
    NPA_DRIVER_PROVISIONING="inherited"
    NPA_CUDA_ARCHITECTURES="inherited"
    NPA_ISAAC_LAB_INSTALL_MODE="inherited"
    NPA_RUNTIME_USER="ubuntu"
    IMAGE_NAME="npa-sonic-mujoco"
    DOCKERFILE="$SCRIPT_DIR/Dockerfile.mujoco"
    DEFAULT_IMAGE_TAG="${NPA_SONIC_MUJOCO_TAG:-0.1.3-mvp}"
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
    section = text.split("[tool.npa.supported-tools]", 1)[1]
    match = re.search(r'^sonic\s*=\s*"([^"]+)"', section, re.MULTILINE)
    if not match:
        raise SystemExit("Could not find [tool.npa.supported-tools].sonic")
    print(match.group(1))
else:
    with Path("pyproject.toml").open("rb") as handle:
        data = tomllib.load(handle)
    print(data["tool"]["npa"]["supported-tools"]["sonic"])
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
  BASE_IMAGE="${BASE_IMAGE_OVERRIDE:-${NPA_SONIC_MUJOCO_BASE_IMAGE:-}}"
  if [ -z "$BASE_IMAGE" ]; then
    if [ -n "$REGISTRY" ]; then
      BASE_IMAGE="${REGISTRY}/npa-sonic:${VERSION}"
    else
      BASE_IMAGE="npa-sonic:${VERSION}"
    fi
  fi
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
  --build-arg "INSTALL_NVIDIA_DRIVER_USERSPACE=${INSTALL_NVIDIA_DRIVER_USERSPACE}"
  --build-arg "INSTALL_ISAACSIM_EXTRA=${INSTALL_ISAACSIM_EXTRA}"
  --build-arg "REQUIRE_TORCH_SM120=${REQUIRE_TORCH_SM120}"
  --build-arg "NPA_DRIVER_PROVISIONING=${NPA_DRIVER_PROVISIONING}"
  --build-arg "NPA_CUDA_ARCHITECTURES=${NPA_CUDA_ARCHITECTURES}"
  --build-arg "NPA_ISAAC_LAB_INSTALL_MODE=${NPA_ISAAC_LAB_INSTALL_MODE}"
  --build-arg "NPA_RUNTIME_USER=${NPA_RUNTIME_USER}"
)

if [ "$VARIANT" = "mujoco" ]; then
  BUILD_ARGS+=(--build-arg "SONIC_MUJOCO_VERSION=${IMAGE_TAG}")
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
