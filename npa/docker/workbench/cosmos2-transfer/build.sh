#!/usr/bin/env bash
# Build the complete Cosmos Transfer 2.5 image from its immutable public sources.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NPA_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
NPA_PYTHON="${NPA_ROOT}/.venv/bin/python"

REGISTRY="${REGISTRY:-}"
PUSH=0
TAG=""
CUDA_BASE_IMAGE=""
SOURCE_REVISION=""
PYTHON_VERSION=""
NPA_SOURCE_SHA="${NPA_SOURCE_SHA:-$(git -C "${NPA_ROOT}/.." rev-parse HEAD)}"
if [[ ! "${NPA_SOURCE_SHA}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "ERROR: NPA_SOURCE_SHA must be the exact 40-character checkout SHA" >&2
  exit 2
fi

usage() {
  echo "Usage: $0 [--registry HOST/PATH] [--tag TAG] [--push]" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --registry) REGISTRY="${2:?}"; shift 2 ;;
    --tag) TAG="${2:?}"; shift 2 ;;
    --base-image) CUDA_BASE_IMAGE="${2:?}"; shift 2 ;;
    --source-revision) SOURCE_REVISION="${2:?}"; shift 2 ;;
    --python-version) PYTHON_VERSION="${2:?}"; shift 2 ;;
    --push) PUSH=1; shift ;;
    -h | --help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

[[ -x "${NPA_PYTHON}" ]] || {
  echo "ERROR: ${NPA_PYTHON} is required; create npa/.venv first" >&2
  exit 1
}

if [[ -z "${TAG}" ]]; then
  TAG="$(cd "${NPA_ROOT}" && "${NPA_PYTHON}" - <<'PY'
from npa.deploy.images import supported_tool_version

print(supported_tool_version("cosmos2-transfer"))
PY
)"
fi

if [[ "${PUSH}" == "1" && -z "${REGISTRY}" ]]; then
  REGISTRY="$(cd "${NPA_ROOT}" && "${NPA_PYTHON}" - <<'PY'
from npa.clients.config import resolve_container_registry

print(resolve_container_registry())
PY
)"
fi

LOCAL_REF="npa-cosmos2-transfer:${TAG}"
if [[ "${PUSH}" == "1" ]]; then
  [[ -n "${REGISTRY}" ]] || { echo "ERROR: no source registry is configured" >&2; exit 1; }
  IMAGE_REF="${REGISTRY%/}/npa-cosmos2-transfer:${TAG}"
else
  IMAGE_REF="${LOCAL_REF}"
fi

run_component="${NPA_RUN_ID:-cosmos2-transfer-$$}"
run_component="$(printf '%s' "${run_component}" | tr -c '[:alnum:]_.-' '-')"
BUILDX_BUILDER="${NPA_BUILDX_BUILDER:-npa-${run_component}}"
CREATED_BUILDER=0
cleanup_builder() {
  if [[ "${CREATED_BUILDER}" == "1" ]]; then
    docker buildx rm "${BUILDX_BUILDER}" >/dev/null 2>&1 || true
  fi
}
trap cleanup_builder EXIT

if ! docker buildx inspect "${BUILDX_BUILDER}" >/dev/null 2>&1; then
  docker buildx create --name "${BUILDX_BUILDER}" \
    --driver docker-container --bootstrap >/dev/null
  CREATED_BUILDER=1
fi

BUILD_ARGS=(
  --builder "${BUILDX_BUILDER}"
  --platform linux/amd64
  --file "${SCRIPT_DIR}/Dockerfile"
  --tag "${IMAGE_REF}"
  --build-arg "NPA_SOURCE_SHA=${NPA_SOURCE_SHA}"
)
[[ -z "${CUDA_BASE_IMAGE}" ]] || BUILD_ARGS+=(--build-arg "CUDA_BASE_IMAGE=${CUDA_BASE_IMAGE}")
[[ -z "${SOURCE_REVISION}" ]] || BUILD_ARGS+=(--build-arg "COSMOS_TRANSFER_REVISION=${SOURCE_REVISION}")
[[ -z "${PYTHON_VERSION}" ]] || BUILD_ARGS+=(--build-arg "COSMOS_PYTHON_VERSION=${PYTHON_VERSION}")

if [[ "${PUSH}" == "1" ]]; then
  BUILD_ARGS+=(--push --provenance=mode=max --sbom=true)
else
  BUILD_ARGS+=(--load --provenance=false)
fi

echo "Building ${LOCAL_REF} from the checked-in immutable inputs (build credentials: none)"
env -u HF_TOKEN -u NGC_API_KEY -u NEBIUS_IAM_TOKEN -u NPA_NEBIUS_IAM_TOKEN \
  docker buildx build "${BUILD_ARGS[@]}" "${NPA_ROOT}"

echo "Built: ${IMAGE_REF}"
