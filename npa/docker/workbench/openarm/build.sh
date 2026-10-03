#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NPA_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
REGISTRY=""
TAG=""
PUSH=0
# Catalog identity used only by release promotion; development builds stay dev-<SHA>.
RELEASE_VERSION="2.2.0-isaac0.1.0-rtfetch"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --registry) REGISTRY="${2%/}"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --push) PUSH=1; shift ;;
    --help|-h) echo "Usage: build.sh [--registry REGISTRY] [--tag TAG] [--push]"; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; exit 2 ;;
  esac
done

PYTHON_BIN="${NPA_PYTHON_BIN:-${NPA_ROOT}/.venv/bin/python}"
[ -x "$PYTHON_BIN" ] || { echo "ERROR: ${PYTHON_BIN} is required" >&2; exit 2; }
SOURCE_SHA="$(git -C "${NPA_ROOT}/.." rev-parse HEAD)"
[ "$(printf %s "$SOURCE_SHA" | wc -c)" -eq 40 ] || { echo "ERROR: invalid source SHA" >&2; exit 2; }
TAG="${TAG:-dev-${SOURCE_SHA}}"

for path in \
  npa/docker/workbench/openarm \
  npa/docker/workbench/packaging-contract.yaml \
  npa/pyproject.toml \
  npa/src/npa/cli/workbench/openarm.py \
  npa/src/npa/workbench/openarm \
  npa/src/npa/sdk/workbench/openarm.py \
  npa/src/npa/deploy/images.py \
  workflows/testing/openarm-simulators.yaml; do
  git -C "${NPA_ROOT}/.." diff --quiet HEAD -- "$path" || {
    echo "ERROR: image/runtime path is not commit-locked: $path" >&2
    exit 2
  }
done

LOCAL_IMAGE="npa-openarm:${TAG}"
BUILD_ARGS=(--platform linux/amd64 --provenance=false --build-arg "NPA_SOURCE_SHA=${SOURCE_SHA}" -f "${SCRIPT_DIR}/Dockerfile" -t "$LOCAL_IMAGE")
if [ -n "$REGISTRY" ]; then
  REGISTRY_IMAGE="${REGISTRY}/npa-openarm:${TAG}"
  BUILD_ARGS+=(-t "$REGISTRY_IMAGE")
else
  REGISTRY_IMAGE=""
fi
docker build "${BUILD_ARGS[@]}" "$NPA_ROOT"

SCAN_TARBALL="$(mktemp /tmp/npa-openarm-scan.XXXXXX.tar)"
trap 'rm -f "$SCAN_TARBALL"' EXIT
docker save --output "$SCAN_TARBALL" "$LOCAL_IMAGE"
"$PYTHON_BIN" "${NPA_ROOT}/scripts/scan_image_omniverse_payload.py" --tarball "$SCAN_TARBALL"
rm -f "$SCAN_TARBALL"
trap - EXIT

if [ "$PUSH" -eq 1 ]; then
  [ -n "$REGISTRY_IMAGE" ] || { echo "ERROR: --push requires --registry" >&2; exit 2; }
  docker push "$REGISTRY_IMAGE"
fi
echo "Built, scanned${REGISTRY_IMAGE:+, and tagged ${REGISTRY_IMAGE}}: ${LOCAL_IMAGE}"
