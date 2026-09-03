#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NPA_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
NPA_PYTHON="${NPA_ROOT}/.venv/bin/python"
REGISTRY=""
TAG="0.1.0-cli0.3.63"
PUSH=0
REVISION="$(git -C "$NPA_ROOT/.." rev-parse HEAD)"

usage() {
  echo "Usage: $0 [--registry HOST/PATH] [--tag TAG] [--push]" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --registry) REGISTRY="${2:?}"; shift 2 ;;
    --tag) TAG="${2:?}"; shift 2 ;;
    --push) PUSH=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

[[ -x "$NPA_PYTHON" ]] || {
  echo "ERROR: ${NPA_PYTHON} is required; create npa/.venv first" >&2
  exit 1
}
if [[ "$PUSH" == 1 && -z "$REGISTRY" ]]; then
  echo "ERROR: --push requires --registry" >&2
  exit 2
fi

LOCAL_IMAGE="npa-antioch:${TAG}"
BUILD_ARGS=(
  --platform linux/amd64
  --file "$SCRIPT_DIR/Dockerfile"
  --tag "$LOCAL_IMAGE"
  --build-arg "NPA_REVISION=${REVISION}"
)
if [[ -n "$REGISTRY" ]]; then
  REGISTRY_IMAGE="${REGISTRY%/}/npa-antioch:${TAG}"
  BUILD_ARGS+=(--tag "$REGISTRY_IMAGE")
else
  REGISTRY_IMAGE=""
fi

# The image must be buildable without inheriting any vendor or cloud credential.
env -u ANTIOCH_WORKBENCH_TOKEN -u ANTIOCH_CONFIG_DIR \
  -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY -u AWS_SESSION_TOKEN \
  -u HF_TOKEN -u NGC_API_KEY -u NEBIUS_IAM_TOKEN -u NPA_NEBIUS_IAM_TOKEN \
  docker build "${BUILD_ARGS[@]}" "$NPA_ROOT"

"$NPA_PYTHON" "$NPA_ROOT/scripts/scan_image_antioch_payload.py" "$LOCAL_IMAGE"

if [[ "$PUSH" == 1 ]]; then
  docker push "$REGISTRY_IMAGE"
  echo "Built, payload-scanned, and pushed: $REGISTRY_IMAGE"
else
  echo "Built and payload-scanned: $LOCAL_IMAGE"
fi
