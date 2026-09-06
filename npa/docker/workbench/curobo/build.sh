#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NPA_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
REPO_ROOT="$(cd "$NPA_ROOT/.." && pwd)"
REGISTRY=""
PUSH=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --registry) REGISTRY="${2:?}"; shift 2 ;;
    --push) PUSH=1; shift ;;
    -h|--help) echo "Usage: $0 [--registry HOST/PATH] [--push]"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
SOURCE_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"
[[ "$SOURCE_SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "Full source SHA is required" >&2; exit 1; }
if [[ "$PUSH" == 1 ]]; then
  echo "Push requires the repository trusted publication workflow and all exact-image safety gates." >&2
  exit 1
fi
CUROBO_SOURCE_EPOCH="$(git -C "$REPO_ROOT" show -s --format=%ct "$SOURCE_SHA")"
[[ "$CUROBO_SOURCE_EPOCH" =~ ^[1-9][0-9]*$ ]] || { echo "Positive source commit epoch is required" >&2; exit 1; }
BUILD_INPUTS=(
  npa/src/npa npa/pyproject.toml npa/README.md npa/.dockerignore
  npa/docker/workbench/curobo
  npa/workflows/workbench/npa-workflows/sim2real.yaml
  npa/workflows/workbench/npa-workflows/physical-ai-data-factory.yaml
)
# The immutable image identity must cover the actual bytes Docker receives.
# Keep unrelated dirty files outside these inputs in relaxed dirty-tree mode.
git -C "$REPO_ROOT" cat-file -e "$SOURCE_SHA:npa/docker/workbench/curobo/Dockerfile" || {
  echo "The cuRobo Dockerfile must be checked in at the source SHA" >&2; exit 1;
}
if ! git -C "$REPO_ROOT" diff --quiet -- "${BUILD_INPUTS[@]}" || \
   ! git -C "$REPO_ROOT" diff --cached --quiet "$SOURCE_SHA" -- "${BUILD_INPUTS[@]}"; then
  echo "Commit the cuRobo build input changes before building a dev-SHA image" >&2
  exit 1
fi
UNTRACKED_INPUTS="$(git -C "$REPO_ROOT" ls-files --others --exclude-standard -- "${BUILD_INPUTS[@]}")"
if [[ -n "$UNTRACKED_INPUTS" ]]; then
  echo "Untracked cuRobo build inputs must be committed or removed before building" >&2
  exit 1
fi
# Snapshot only the resolved commit, so ignored files and writes racing the
# checks cannot enter the image. No checkout, branch or Git mutation is needed.
BUILD_CONTEXT="$(mktemp -d "${TMPDIR:-/tmp}/npa-curobo-build.XXXXXXXX")"
trap 'rm -rf -- "$BUILD_CONTEXT"' EXIT
git -C "$REPO_ROOT" archive "$SOURCE_SHA" -- "${BUILD_INPUTS[@]}" | \
  tar -x --strip-components=1 -C "$BUILD_CONTEXT"
IMAGE="${REGISTRY:+${REGISTRY%/}/}npa-curobo:dev-${SOURCE_SHA}"
# Trusted publication builds this same Dockerfile after source review, SBOM,
# secret/license/payload scans and bootstrap evidence; this helper only builds.
env -u HF_TOKEN -u NGC_API_KEY -u NVIDIA_API_KEY -u NEBIUS_IAM_TOKEN \
  -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY \
  docker buildx build --platform linux/amd64 --file "$BUILD_CONTEXT/docker/workbench/curobo/Dockerfile" \
    --tag "$IMAGE" --build-arg "NPA_SOURCE_SHA=$SOURCE_SHA" \
    --build-arg "SOURCE_DATE_EPOCH=$CUROBO_SOURCE_EPOCH" \
    --label "org.opencontainers.image.source=https://github.com/nebius/nebius-physical-ai" \
    --load --provenance=false "$BUILD_CONTEXT"
