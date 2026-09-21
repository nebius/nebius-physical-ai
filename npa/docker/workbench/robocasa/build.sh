#!/usr/bin/env bash
# Build a local npa-robocasa candidate. Publication is intentionally delegated
# to the guarded image workflow so payload, complete-byte, Trivy, SBOM, and GPU
# gates cannot be bypassed by this script.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NPA_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
REPO_ROOT="$(git -C "${NPA_ROOT}" rev-parse --show-toplevel)"
REGISTRY="${REGISTRY:-}"
ROBOCASA_VERSION="${ROBOCASA_VERSION:-0.1.1}"
NPA_SOURCE_SHA="${NPA_SOURCE_SHA:-$(git -C "${NPA_ROOT}" rev-parse HEAD)}"
TAG=""

usage() {
  echo "Usage: $0 --registry <registry/namespace> [--tag <tag>]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --registry) REGISTRY="${2:?}"; shift 2 ;;
    --tag) TAG="${2:?}"; shift 2 ;;
    -h | --help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${REGISTRY}" ]]; then
  echo "ERROR: pass --registry or set REGISTRY to an authorized registry" >&2
  exit 2
fi
if [[ ! "${NPA_SOURCE_SHA}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "ERROR: NPA_SOURCE_SHA must be the exact 40-character checkout SHA" >&2
  exit 2
fi
if [[ "$(git -C "${NPA_ROOT}" rev-parse HEAD)" != "${NPA_SOURCE_SHA}" ]]; then
  echo "ERROR: NPA_SOURCE_SHA must equal the build-context checkout HEAD" >&2
  exit 2
fi
if ! WORKTREE_STATUS="$(
  git -C "${NPA_ROOT}" status --porcelain=v1 --untracked-files=all -- .
)"; then
  echo "ERROR: cannot verify the build-context checkout is clean" >&2
  exit 2
fi
if [[ -n "${WORKTREE_STATUS}" ]]; then
  echo "ERROR: commit all NPA build-context changes before building" >&2
  exit 2
fi

TAG="${TAG:-dev-${NPA_SOURCE_SHA}}"
IMAGE="${REGISTRY}/npa-robocasa:${TAG}"

echo "Building local ${IMAGE} from source ${NPA_SOURCE_SHA}"
git -C "${REPO_ROOT}" archive --format=tar "${NPA_SOURCE_SHA}:npa" \
  | docker build \
      --platform linux/amd64 \
      --provenance=false \
      --build-arg ROBOCASA_VERSION="${ROBOCASA_VERSION}" \
      --build-arg NPA_SOURCE_SHA="${NPA_SOURCE_SHA}" \
      -t "${IMAGE}" \
      -f docker/workbench/robocasa/Dockerfile \
      -

echo "Built ${IMAGE}"
