#!/usr/bin/env bash
# Assemble an exact committed source tree and build locally. Publication is a
# separate operation after the parent's exact-byte/security/functional gates.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NPA_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
REPO_ROOT="$(cd "${NPA_ROOT}/.." && pwd)"
SOURCE_SHA="${SOURCE_SHA:-}"
IMAGE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-sha) SOURCE_SHA="${2:?}"; shift 2 ;;
    --image) IMAGE="${2:?}"; shift 2 ;;
    -h|--help)
      echo 'Usage: build.sh --source-sha FULL_COMMITTED_SHA [--image REGISTRY/npa-ncore:dev-FULL_COMMITTED_SHA]'
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ "$SOURCE_SHA" =~ ^[0-9a-f]{40}$ ]] || { echo 'A full committed SOURCE_SHA is required' >&2; exit 2; }
[[ "$(git -C "$REPO_ROOT" rev-parse "${SOURCE_SHA}^{commit}")" == "$SOURCE_SHA" ]]
IMAGE="${IMAGE:-local.invalid/npa-ncore:dev-${SOURCE_SHA}}"
[[ "$IMAGE" == */npa-ncore:dev-"${SOURCE_SHA}" ]] || { echo 'Use npa-ncore:dev-<full-source-sha>' >&2; exit 2; }
NPA_PYTHON="${NPA_ROOT}/.venv/bin/python"
[[ -x "$NPA_PYTHON" ]] || { echo 'npa/.venv/bin/python is required' >&2; exit 1; }
# Export only committed files; unrelated dirty paths never enter the build.
context="$(mktemp -d "${TMPDIR:-/tmp}/npa-ncore-context.XXXXXXXX")"
trap 'rm -rf -- "$context"' EXIT
git -C "$REPO_ROOT" archive "$SOURCE_SHA" npa workflows | tar -x -C "$context"
"$NPA_PYTHON" "$context/npa/src/npa/workflow_build.py" --stage-catalog --package-root "$context/npa"
docker buildx build --platform linux/amd64 --load --provenance=false \
  --build-arg "SOURCE_SHA=$SOURCE_SHA" --tag "$IMAGE" \
  --file "$context/npa/docker/workbench/ncore/Dockerfile" "$context/npa"
echo "Built locally: $IMAGE (not published; artifact gates and real conversion remain required)"
