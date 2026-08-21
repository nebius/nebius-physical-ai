#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NPA_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
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

[[ -x "${NPA_ROOT}/.venv/bin/python" ]] || { echo "ERROR: repository venv is required" >&2; exit 1; }
[[ -n "$TAG" ]] || TAG="$(cd "$NPA_ROOT" && .venv/bin/python -c 'from npa.deploy.images import supported_tool_version; print(supported_tool_version("alpamayo2-super"))')"
if [[ "$PUSH" == 1 && -z "$REGISTRY" ]]; then
  REGISTRY="$(cd "$NPA_ROOT" && .venv/bin/python -c 'from npa.clients.config import resolve_container_registry; print(resolve_container_registry())')"
fi
IMAGE="npa-alpamayo2-super:${TAG}"
[[ "$PUSH" == 0 ]] || IMAGE="${REGISTRY%/}/npa-alpamayo2-super:${TAG}"
ARGS=(--platform linux/amd64 --file "$SCRIPT_DIR/Dockerfile" --tag "$IMAGE" --build-arg "NPA_SOURCE_COMMIT=$(git -C "$REPO_ROOT" rev-parse HEAD)")
if [[ "$PUSH" == 1 ]]; then
  git -C "$REPO_ROOT" diff --quiet
  git -C "$REPO_ROOT" diff --cached --quiet
  ARGS+=(--push --provenance=mode=max --sbom=true)
else
  ARGS+=(--load --provenance=false)
fi
env -u HF_TOKEN -u NGC_API_KEY -u NEBIUS_IAM_TOKEN docker buildx build "${ARGS[@]}" "$NPA_ROOT"
if [[ "$PUSH" == 0 ]]; then
  "${NPA_ROOT}/.venv/bin/python" \
    "${NPA_ROOT}/scripts/scan_image_alpamayo2_payload.py" "$IMAGE"
fi
echo "Built: $IMAGE"
