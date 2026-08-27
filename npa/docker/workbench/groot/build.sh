#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NPA_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

REGISTRY=""
PUSH=0
IMAGE_TAG_OVERRIDE=""

usage() {
  cat <<'EOF'
Usage: build.sh [--registry REGISTRY] [--push] [--tag TAG]

Builds the GR00T runtime image as npa-groot:<gr00t-runtime-version>.
When --registry is provided, also tags REGISTRY/npa-groot:<tag>.
When --tag is provided, it overrides the final image tag.
Use --registry <your-registry>/<namespace> --push to publish.

  --tag is what lets you validate a candidate without overwriting a canonical tag that
  running workloads resolve. --push streams straight to the registry via buildx instead
  of materialising the image locally first: a GR00T build is ~30 GB unpacked, and the
  local path spends minutes writing bytes nothing reads.

This image bakes NO NVIDIA Isaac Sim or Isaac Lab, so no NGC credentials are needed.
Isaac is fetched at first use of /isaac-sim/python.sh under the operator's own EULA
acceptance; GR00T inference itself needs neither. See
npa/docker/workbench/common/isaac_bootstrap.sh.
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
    --tag)
      if [ "$#" -lt 2 ]; then
        echo "ERROR: --tag requires a value" >&2
        exit 2
      fi
      IMAGE_TAG_OVERRIDE="$2"
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

VERSION="$(
  cd "$NPA_ROOT"
  python3 - <<'PY'
from pathlib import Path
import re

text = Path("src/npa/cli/groot/__init__.py").read_text()
match = re.search(r'^GROOT_RUNTIME_VERSION\s*=\s*"([^"]+)"', text, re.MULTILINE)
if not match:
    raise SystemExit("Could not find GROOT_RUNTIME_VERSION")
print(match.group(1))
PY
)"

REF="$(
  cd "$NPA_ROOT"
  python3 - <<'PY'
from pathlib import Path
import re

text = Path("src/npa/cli/groot/__init__.py").read_text()
match = re.search(r'^GROOT_REPO_REF\s*=\s*"([^"]+)"', text, re.MULTILINE)
if not match:
    raise SystemExit("Could not find GROOT_REPO_REF")
print(match.group(1))
PY
)"

IMAGE_TAG="${IMAGE_TAG_OVERRIDE:-$VERSION}"
LOCAL_IMAGE="npa-groot:${IMAGE_TAG}"

# Keep the Isaac pins in one place: the bootstrap's defaults. Reading them here means the
# build cannot drift from what actually runs.
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
  --build-arg "GROOT_RUNTIME_VERSION=${VERSION}"
  --build-arg "GROOT_REPO_REF=${REF}"
  --build-arg "ISAAC_SIM_VERSION=${ISAAC_SIM_VERSION}"
  --build-arg "ISAAC_LAB_SRC_COMMIT=${ISAAC_LAB_SRC_COMMIT}"
)

if [ -n "$REGISTRY" ]; then
  REGISTRY_IMAGE="${REGISTRY}/npa-groot:${IMAGE_TAG}"
else
  REGISTRY_IMAGE=""
fi

echo "Building ${LOCAL_IMAGE}"
echo "  gr00t          ${VERSION} @ ${REF}"
echo "  isaacsim pin   ${ISAAC_SIM_VERSION}   (fetched at first RUN, not baked)"
echo "  isaaclab src   ${ISAAC_LAB_SRC_COMMIT}"

if [ "$PUSH" -eq 1 ]; then
  BUILDX_BUILDER="${NPA_BUILDX_BUILDER:-npa-groot-builder}"
  if ! docker buildx inspect "$BUILDX_BUILDER" >/dev/null 2>&1; then
    # A builder whose container has been reaped still shows up in `buildx ls`, so `create`
    # would fail on the name. Remove it first.
    docker buildx rm "$BUILDX_BUILDER" >/dev/null 2>&1 || true
    docker buildx create --name "$BUILDX_BUILDER" --driver docker-container --bootstrap >/dev/null
  fi
  docker buildx build --builder "$BUILDX_BUILDER" --push "${BUILD_ARGS[@]}" \
    -t "$REGISTRY_IMAGE" "$NPA_ROOT"
  echo "Built and pushed: $REGISTRY_IMAGE"
  exit 0
fi

BUILD_ARGS+=(-t "$LOCAL_IMAGE")
if [ -n "$REGISTRY_IMAGE" ]; then
  BUILD_ARGS+=(-t "$REGISTRY_IMAGE")
fi

docker build "${BUILD_ARGS[@]}" "$NPA_ROOT"

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
