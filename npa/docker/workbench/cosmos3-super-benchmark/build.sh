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
    -h|--help)
      echo "Usage: $0 [--registry HOST/PATH] [--tag FULL_GIT_SHA] [--push]"
      exit 0
      ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

SOURCE_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
[[ -n "$TAG" ]] || TAG="$SOURCE_COMMIT"
[[ "$TAG" =~ ^[0-9a-f]{40}$ ]] || {
  echo "ERROR: the benchmark wrapper tag must be a full 40-character git SHA" >&2
  exit 2
}
if [[ "$PUSH" == 1 ]]; then
  git -C "$REPO_ROOT" diff --quiet
  git -C "$REPO_ROOT" diff --cached --quiet
  [[ -n "$REGISTRY" ]] || {
    REGISTRY="$(cd "$NPA_ROOT" && .venv/bin/python -c 'from npa.clients.config import resolve_container_registry; print(resolve_container_registry())')"
  }
fi

IMAGE="npa-cosmos3-super-benchmark:${TAG}"
[[ "$PUSH" == 0 ]] || IMAGE="${REGISTRY%/}/npa-cosmos3-super-benchmark:${TAG}"
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

env -u HF_TOKEN -u NGC_API_KEY -u NEBIUS_IAM_TOKEN \
  docker buildx build "${ARGS[@]}" "$NPA_ROOT"
echo "Built: $IMAGE"
