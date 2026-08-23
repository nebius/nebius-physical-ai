#!/usr/bin/env bash
#
# build.sh - build (and optionally push) npa-isaac-lab.
#
# isaac-lab was the only workbench image without a build script, which made the
# build-your-own path least supported for the image most likely to need it. It now
# matches the sonic/groot --registry/--push contract.
#
# The image bakes NO NVIDIA Isaac Sim or Isaac Lab code: Isaac is fetched on first run
# from https://pypi.nvidia.com under the operator's own EULA acceptance. So this build
# needs no NGC credentials and no EULA acceptance of its own - see
# docs/workbench/container-packaging.md and
# npa/docker/workbench/common/isaac_bootstrap.sh.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NPA_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

REGISTRY=""
PUSH=0
IMAGE_TAG_OVERRIDE=""
IMAGE_NAME="npa-isaac-lab"
NPA_SOURCE_SHA="${NPA_SOURCE_SHA:-$(git -C "${NPA_ROOT}/.." rev-parse HEAD)}"
if [[ ! "${NPA_SOURCE_SHA}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "ERROR: NPA_SOURCE_SHA must be the exact 40-character checkout SHA" >&2
  exit 2
fi

usage() {
  cat <<'EOF'
Usage: build.sh [--registry REGISTRY] [--push] [--tag TAG]

Builds npa-isaac-lab:<version>, where <version> is the isaac-lab pin in
npa/pyproject.toml ([tool.npa.supported-tools].isaac-lab).

  --registry REGISTRY  also tag REGISTRY/npa-isaac-lab:<tag>
  --push               push to REGISTRY (requires --registry)
  --tag TAG            override the image tag

The CUDA base is digest-pinned in the Dockerfile rather than passed in, so a build is
reproducible from the repo alone; change it there (and in the CI base-image CVE scan
matrix) per docs/security/image-reproducibility.md.

Publish to your own registry:
  build.sh --registry <your-registry>/<namespace> --push

No NGC credentials are needed: this image contains no NVIDIA Isaac bytes. Isaac Sim and
Isaac Lab are downloaded on first RUN, under your own EULA acceptance
(ACCEPT_EULA defaults to Y). Expect a ~4.5 GB download and
~10 GiB of cache on the first start; pre-warm it per node/PVC with
npa/docker/workbench/common/warm-isaac-cache.yaml.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --registry)
      [ "$#" -ge 2 ] || { echo "ERROR: --registry requires a value" >&2; exit 2; }
      REGISTRY="${2%/}"; shift 2 ;;
    --push) PUSH=1; shift ;;
    --tag)
      [ "$#" -ge 2 ] || { echo "ERROR: --tag requires a value" >&2; exit 2; }
      IMAGE_TAG_OVERRIDE="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [ "$PUSH" -eq 1 ] && [ -z "$REGISTRY" ]; then
  echo "ERROR: --push requires --registry" >&2
  exit 2
fi

PYTHON_BIN="${NPA_PYTHON_BIN:-$NPA_ROOT/.venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="$(command -v python3)"
fi

read_pin() {
  cd "$NPA_ROOT"
  NPA_PIN_KEY="$1" "$PYTHON_BIN" - <<'PY'
import os
import re
from pathlib import Path

key = os.environ["NPA_PIN_KEY"]
try:
    import tomllib
except ModuleNotFoundError:
    text = Path("pyproject.toml").read_text()
    section = text.split("[tool.npa.supported-tools]", 1)[1]
    match = re.search(rf'^{re.escape(key)}\s*=\s*"([^"]+)"', section, re.MULTILINE)
    if not match:
        raise SystemExit(f"Could not find [tool.npa.supported-tools].{key}")
    print(match.group(1))
else:
    with Path("pyproject.toml").open("rb") as handle:
        data = tomllib.load(handle)
    print(data["tool"]["npa"]["supported-tools"][key])
PY
}

VERSION="$(read_pin isaac-lab)"
IMAGE_TAG="${IMAGE_TAG_OVERRIDE:-$VERSION}"
LOCAL_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"

# Keep the Isaac Sim pin and the Isaac Lab source commit in one place: the bootstrap's
# defaults. Reading them here means the build cannot drift from what runs.
BOOTSTRAP="$SCRIPT_DIR/../common/isaac_bootstrap.sh"
ISAAC_SIM_VERSION="$(sed -n 's/^ISAAC_SIM_VERSION="${ISAAC_SIM_VERSION:-\(.*\)}"$/\1/p' "$BOOTSTRAP")"
ISAAC_LAB_SRC_COMMIT="$(sed -n 's/^ISAAC_LAB_SRC_COMMIT="${NPA_ISAAC_LAB_SRC_COMMIT:-\(.*\)}"$/\1/p' "$BOOTSTRAP")"
if [ -z "$ISAAC_SIM_VERSION" ] || [ -z "$ISAAC_LAB_SRC_COMMIT" ]; then
  echo "ERROR: could not read the Isaac pins out of $BOOTSTRAP" >&2
  exit 1
fi

BUILD_ARGS=(
  --platform linux/amd64
  -f "$SCRIPT_DIR/Dockerfile"
  --build-arg "ISAAC_LAB_VERSION=${VERSION}"
  --build-arg "ISAAC_SIM_VERSION=${ISAAC_SIM_VERSION}"
  --build-arg "ISAAC_LAB_SRC_COMMIT=${ISAAC_LAB_SRC_COMMIT}"
  --build-arg "NPA_SOURCE_SHA=${NPA_SOURCE_SHA}"
)
if [ -n "$REGISTRY" ]; then
  REGISTRY_IMAGE="${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
else
  REGISTRY_IMAGE=""
fi

echo "Building ${LOCAL_IMAGE}"
echo "  isaac-lab      ${VERSION}"
echo "  isaacsim pin   ${ISAAC_SIM_VERSION}   (fetched at first RUN, not baked)"
echo "  isaaclab src   ${ISAAC_LAB_SRC_COMMIT}"

if [ "$PUSH" -eq 1 ]; then
  BUILDX_BUILDER="${NPA_BUILDX_BUILDER:-npa-isaac-lab-builder}"
  if ! docker buildx inspect "$BUILDX_BUILDER" >/dev/null 2>&1; then
    docker buildx rm "$BUILDX_BUILDER" >/dev/null 2>&1 || true
    docker buildx create --name "$BUILDX_BUILDER" --driver docker-container --bootstrap >/dev/null
  fi
  docker buildx build --builder "$BUILDX_BUILDER" --push "${BUILD_ARGS[@]}" \
    -t "$REGISTRY_IMAGE" "$NPA_ROOT"
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
