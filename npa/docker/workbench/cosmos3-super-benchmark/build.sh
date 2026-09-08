#!/usr/bin/env bash
# Build a local candidate; public publication belongs to the trusted workflow.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NPA_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
REPO_ROOT="$(cd "${NPA_ROOT}/.." && pwd)"
TAG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag) TAG="${2:?}"; shift 2 ;;
    --push|--registry)
      echo "ERROR: publication requires the trusted image workflow and its security gates" >&2
      exit 2
      ;;
    -h|--help)
      echo "Usage: $0 [--tag dev-FULL_GIT_SHA] (local candidate only)"
      exit 0
      ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

SOURCE_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
[[ -n "$TAG" ]] || TAG="dev-$SOURCE_COMMIT"
[[ "$TAG" =~ ^dev-[0-9a-f]{40}$ && "$TAG" == "dev-$SOURCE_COMMIT" ]] || {
  echo "ERROR: the candidate tag must be dev- followed by the current full git SHA" >&2
  exit 2
}
IMAGE="npa-cosmos3-super-benchmark:${TAG}"
ARGS=(
  --platform linux/amd64
  --file "$SCRIPT_DIR/Dockerfile"
  --tag "$IMAGE"
  --build-arg "NPA_SOURCE_SHA=$SOURCE_COMMIT"
)
ARGS+=(--load --provenance=false)

env -u HF_TOKEN -u NGC_API_KEY -u NEBIUS_IAM_TOKEN \
  docker buildx build "${ARGS[@]}" "$NPA_ROOT"
echo "Built: $IMAGE"
