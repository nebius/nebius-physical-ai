#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NPA_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
REGISTRY="${REGISTRY:-}"
NPA_SOURCE_SHA="${NPA_SOURCE_SHA:-$(git -C "${NPA_ROOT}" rev-parse HEAD)}"
PUSH=0
TAG=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --registry) REGISTRY="${2:?}"; shift 2 ;;
    --tag) TAG="${2:?}"; shift 2 ;;
    --push) PUSH=1; shift ;;
    -h|--help) echo "Usage: build.sh --registry REGISTRY [--tag TAG] [--push]"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

if [ -z "${REGISTRY}" ]; then
  echo "ERROR: pass --registry or set REGISTRY to an authorized registry" >&2
  exit 2
fi
if [[ ! "${NPA_SOURCE_SHA}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "ERROR: NPA_SOURCE_SHA must be the exact 40-character checkout SHA" >&2
  exit 2
fi
if [ "${PUSH}" = 1 ] && [ "${REGISTRY%/}" = "ghcr.io/nebius/nebius-physical-ai" ]; then
  echo "ERROR: official public pushes must use publish-public-images.yml so scans, SBOM, attestations, and anonymous verification run before/after push" >&2
  exit 2
fi
if [ -z "${TAG}" ]; then
  TAG="$("${NPA_ROOT}/.venv/bin/python" - <<'PY'
from npa.deploy.images import supported_tool_version
print(supported_tool_version("cosmos3-ray-serve"))
PY
)"
fi

local_ref="npa-cosmos3-ray-serve:${TAG}"
remote_ref="${REGISTRY}/npa-cosmos3-ray-serve:${TAG}"
docker build --platform linux/amd64 \
  -f "${SCRIPT_DIR}/Dockerfile" \
  --build-arg "NPA_SOURCE_SHA=${NPA_SOURCE_SHA}" \
  -t "${local_ref}" -t "${remote_ref}" "${NPA_ROOT}"
if [ "${PUSH}" = 1 ]; then docker push "${remote_ref}"; fi
echo "Done: ${remote_ref} push=${PUSH}"
