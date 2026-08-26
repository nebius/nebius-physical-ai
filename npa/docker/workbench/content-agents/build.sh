#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NPA_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
REPO_ROOT="$(cd "${NPA_ROOT}/.." && pwd)"
NPA_PYTHON="${NPA_ROOT}/.venv/bin/python"
REGISTRY="${REGISTRY:-}"
TAG="${CONTENT_AGENTS_TAG:-0.5.2-npa2}"
PUSH=0

usage() {
  echo "Usage: $0 [--registry PRIVATE_HOST/PATH] [--tag TAG] [--push]" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --registry) REGISTRY="${2:?}"; shift 2 ;;
    --tag) TAG="${2:?}"; shift 2 ;;
    --push) PUSH=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

[[ -x "$NPA_PYTHON" ]] || { echo "ERROR: ${NPA_PYTHON} is required" >&2; exit 1; }
if [[ -z "$REGISTRY" ]]; then
  if [[ "$PUSH" == 1 ]]; then
    REGISTRY="$(cd "$NPA_ROOT" && "$NPA_PYTHON" -c 'from npa.clients.config import resolve_container_registry; print(resolve_container_registry())')"
  else
    REGISTRY="local.invalid"
  fi
fi

SOURCE_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"
[[ "$SOURCE_SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "ERROR: invalid NPA source SHA" >&2; exit 1; }
IMAGE="${REGISTRY%/}/npa-content-agents:${TAG}"
ARGS=(
  --platform linux/amd64
  --file "$SCRIPT_DIR/Dockerfile"
  --tag "$IMAGE"
  --build-arg "NPA_SOURCE_SHA=$SOURCE_SHA"
  --build-arg "NPA_IMAGE_VERSION=$TAG"
)
if [[ "$PUSH" == 1 ]]; then
  ARGS+=(--push --provenance=mode=max --sbom=true)
else
  ARGS+=(--load --provenance=false)
fi

env -u HF_TOKEN -u NGC_API_KEY -u NVIDIA_API_KEY -u NEBIUS_TOKEN_FACTORY_KEY \
    -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY -u NEBIUS_IAM_TOKEN \
  docker buildx build "${ARGS[@]}" "$NPA_ROOT"

echo "Built public-eligible zero-vendor-payload image: ${IMAGE}"
