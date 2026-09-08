#!/usr/bin/env bash
# Build only the committed RoboCasa snapshot; trusted publication owns all pushes.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NPA_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
REPO_ROOT="$(cd "$NPA_ROOT/.." && pwd)"
ROBOCASA_PYTHON="${NPA_PYTHON_BIN:-$NPA_ROOT/.venv/bin/python}"
REGISTRY=""
TAG=""
PUSH=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --registry) REGISTRY="${2:?}"; shift 2 ;;
    --tag) TAG="${2:?}"; shift 2 ;;
    --push) PUSH=1; shift ;;
    -h|--help) echo "Usage: $0 [--registry HOST/PATH] [--tag dev-FULL-SHA]"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
SOURCE_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"
[[ "$SOURCE_SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "Full source SHA is required" >&2; exit 1; }
if [[ "$PUSH" == 1 ]]; then
  echo "Push requires the trusted publication workflow and all exact-image safety gates." >&2
  exit 1
fi
TAG="${TAG:-dev-${SOURCE_SHA}}"
[[ "$TAG" == "dev-${SOURCE_SHA}" ]] || { echo "Tag must match the exact source SHA" >&2; exit 1; }
BUILD_INPUTS=(
  npa/src/npa npa/pyproject.toml npa/README.md npa/.dockerignore
  npa/docker/workbench/robocasa npa/docker/workbench/curobo/filter_cudnn_runtime.py
  workflows/main workflows/testing
)
git -C "$REPO_ROOT" cat-file -e "$SOURCE_SHA:npa/docker/workbench/robocasa/Dockerfile" || {
  echo "The RoboCasa Dockerfile must be checked in at the source SHA" >&2; exit 1;
}
if ! git -C "$REPO_ROOT" diff --quiet -- "${BUILD_INPUTS[@]}" || \
   ! git -C "$REPO_ROOT" diff --cached --quiet "$SOURCE_SHA" -- "${BUILD_INPUTS[@]}"; then
  echo "Commit the RoboCasa build inputs before building a dev-SHA image" >&2
  exit 1
fi
UNTRACKED_INPUTS="$(git -C "$REPO_ROOT" ls-files --others --exclude-standard -- "${BUILD_INPUTS[@]}")"
if [[ -n "$UNTRACKED_INPUTS" ]]; then
  echo "Untracked RoboCasa build inputs must be committed or removed before building" >&2
  exit 1
fi
BUILD_SNAPSHOT="$(mktemp -d "${TMPDIR:-/tmp}/npa-robocasa-build.XXXXXXXX")"
trap 'rm -rf -- "$BUILD_SNAPSHOT"' EXIT
git -C "$REPO_ROOT" archive "$SOURCE_SHA" -- "${BUILD_INPUTS[@]}" | tar -x -C "$BUILD_SNAPSHOT"
BUILD_CONTEXT="$BUILD_SNAPSHOT/npa"
[[ -x "$ROBOCASA_PYTHON" ]] || { echo "Repository Python is required for catalog staging" >&2; exit 1; }
"$ROBOCASA_PYTHON" "$BUILD_CONTEXT/src/npa/workflow_build.py" --stage-catalog --package-root "$BUILD_CONTEXT"
IMAGE="${REGISTRY:+${REGISTRY%/}/}npa-robocasa:${TAG}"
env -u HF_TOKEN -u NGC_API_KEY -u NVIDIA_API_KEY -u NEBIUS_IAM_TOKEN \
  -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY \
  docker buildx build --platform linux/amd64 --file "$BUILD_CONTEXT/docker/workbench/robocasa/Dockerfile" \
    --tag "$IMAGE" --build-arg "NPA_SOURCE_SHA=$SOURCE_SHA" \
    --label "org.opencontainers.image.source=https://github.com/nebius/nebius-physical-ai" \
    --load --provenance=false "$BUILD_CONTEXT"
