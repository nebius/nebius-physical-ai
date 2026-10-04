#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NPA_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
REPO_ROOT="$(cd "${NPA_ROOT}/.." && pwd)"
NPA_PYTHON="${NPA_ROOT}/.venv/bin/python"
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
SOURCE_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
[[ "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo "ERROR: full source SHA required" >&2; exit 2; }
[[ -n "$TAG" ]] || TAG="dev-${SOURCE_COMMIT}"
if [[ "$PUSH" == 1 ]]; then
  echo "ERROR: direct image pushes are disabled for the unbuilt HY-World candidate." >&2
  echo "Build with --load, complete OCI/SBOM/secret/payload scans and exact-digest GPU evidence," >&2
  echo "then use the governed secure publication workflow. This helper never publishes bytes." >&2
  exit 2
fi
IMAGE="npa-hy-world:${TAG}"
ARGS=(--platform linux/amd64 --file "$SCRIPT_DIR/Dockerfile" --tag "$IMAGE" --build-arg "NPA_SOURCE_COMMIT=$SOURCE_COMMIT")
ARGS+=(--load --provenance=false)
env -u HF_TOKEN -u NGC_API_KEY -u NEBIUS_IAM_TOKEN -u NPA_HY_WORLD_LLM_ADDR \
  docker buildx build "${ARGS[@]}" "$NPA_ROOT"
echo "Built: $IMAGE"
echo "Before any governed publication, scan the exact artifact:"
echo "  ${NPA_PYTHON} ${NPA_ROOT}/scripts/scan_image_hy_world_payload.py ${IMAGE}"
