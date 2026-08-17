#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NPA_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
NPA_PYTHON="${NPA_ROOT}/.venv/bin/python"
REPO_ROOT="$(cd "${NPA_ROOT}/.." && pwd)"
REGISTRY="${REGISTRY:-}"
TAG=""
PUSH=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --registry) REGISTRY="${2:?}"; shift 2 ;;
    --tag) TAG="${2:?}"; shift 2 ;;
    --push) PUSH=1; shift ;;
    -h|--help) echo "Usage: $0 [--registry HOST/PATH] [--tag TAG] [--push]"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

[[ -x "$NPA_PYTHON" ]] || { echo "ERROR: ${NPA_PYTHON} is required" >&2; exit 1; }
[[ -n "$TAG" ]] || TAG="$(cd "$NPA_ROOT" && "$NPA_PYTHON" -c 'from npa.deploy.images import supported_tool_version; print(supported_tool_version("ltx2"))')"
if [[ "$PUSH" == 1 && -z "$REGISTRY" ]]; then
  REGISTRY="$(cd "$NPA_ROOT" && "$NPA_PYTHON" -c 'from npa.clients.config import resolve_container_registry; print(resolve_container_registry())')"
fi
SOURCE_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
if [[ "$PUSH" == 1 ]]; then
  git -C "$REPO_ROOT" diff --quiet
  git -C "$REPO_ROOT" diff --cached --quiet
fi
IMAGE="npa-ltx2:${TAG}"
[[ "$PUSH" == 0 ]] || IMAGE="${REGISTRY%/}/npa-ltx2:${TAG}"
ARGS=(
  --platform linux/amd64
  --file "$SCRIPT_DIR/Dockerfile"
  --tag "$IMAGE"
  --build-arg "NPA_SOURCE_COMMIT=$SOURCE_COMMIT"
)
if [[ "$PUSH" == 1 ]]; then
  ARGS+=(--push --provenance=mode=max --sbom=true)
else
  ARGS+=(--load --provenance=false)
fi
# Entitlement variables are stripped from the build environment on purpose: if a
# build could inherit them, the build could perform the fetch the runtime checks
# exist to prevent, and the image would ship what it claims not to.
env -u HF_TOKEN -u NGC_API_KEY -u NEBIUS_IAM_TOKEN \
    -u NPA_LTX_ACCEPT_NVIDIA_RUNTIME_TERMS \
  docker buildx build "${ARGS[@]}" "$NPA_ROOT"
echo "Built: $IMAGE"
echo "Verify the artifact before publishing:"
echo "  ${NPA_PYTHON} ${NPA_ROOT}/scripts/scan_image_ltx_payload.py ${IMAGE}"
