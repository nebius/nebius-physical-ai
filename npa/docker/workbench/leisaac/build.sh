#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NPA_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
REGISTRY=""
PUSH=0
TAG=""

usage() {
  echo "Usage: build.sh [--registry REGISTRY] [--tag TAG] [--push]"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --registry) REGISTRY="${2%/}"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --push) PUSH=1; shift ;;
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
  echo "ERROR: $PYTHON_BIN is required; create npa/.venv first" >&2
  exit 2
fi

if [ -z "$TAG" ]; then
  TAG="$(cd "$NPA_ROOT" && "$PYTHON_BIN" - <<'PY'
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
from pathlib import Path
with Path("pyproject.toml").open("rb") as handle:
    print(tomllib.load(handle)["tool"]["npa"]["supported-tools"]["leisaac"])
PY
)"
fi

SOURCE_COMMIT="$(git -C "$NPA_ROOT/.." rev-parse HEAD)"
for path in \
  npa/docker/workbench/leisaac \
  npa/docker/workbench/packaging-contract.yaml \
  npa/pyproject.toml \
  npa/src/npa/deploy/images.py \
  npa/src/npa/smoke/capabilities.py \
  npa/src/npa/smoke/golden_evals.yaml \
  npa/src/npa/agent_backend/leisaac_registry.py \
  npa/src/npa/agent_backend/leisaac_transport.py \
  npa/src/npa/agent_backend/leisaac_bundles.py \
  npa/src/npa/workbench/leisaac \
  npa/src/npa/cli/workbench/leisaac.py; do
  if ! git -C "$NPA_ROOT/.." diff --quiet HEAD -- "$path"; then
    echo "ERROR: image/runtime path is not commit-locked: $path" >&2
    exit 2
  fi
done

LOCAL_IMAGE="npa-leisaac:${TAG}"
BUILD_ARGS=(
  --platform linux/amd64
  --provenance=false
  --build-arg "NPA_SOURCE_COMMIT=${SOURCE_COMMIT}"
  -f "$SCRIPT_DIR/Dockerfile"
  -t "$LOCAL_IMAGE"
)
if [ -n "$REGISTRY" ]; then
  REGISTRY_IMAGE="${REGISTRY}/npa-leisaac:${TAG}"
  BUILD_ARGS+=(-t "$REGISTRY_IMAGE")
else
  REGISTRY_IMAGE=""
fi

docker build "${BUILD_ARGS[@]}" "$NPA_ROOT"

SCAN="$NPA_ROOT/scripts/scan_image_omniverse_payload.py"
SCAN_TARBALL="$(mktemp /tmp/npa-leisaac-scan.XXXXXX.tar)"
trap 'rm -f "$SCAN_TARBALL"' EXIT
docker save --output "$SCAN_TARBALL" "$LOCAL_IMAGE"
"$PYTHON_BIN" "$SCAN" --tarball "$SCAN_TARBALL"
rm -f "$SCAN_TARBALL"
trap - EXIT

if [ "$PUSH" -eq 1 ]; then
  docker push "$REGISTRY_IMAGE"
  echo "Built, scanned, and pushed: $REGISTRY_IMAGE"
else
  echo "Built and scanned: $LOCAL_IMAGE"
fi
