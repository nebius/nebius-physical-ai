#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NPA_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
REPO_ROOT="$(cd "${NPA_ROOT}/.." && pwd)"
TAG="${TAG:-local}"
REGISTRY="${REGISTRY:-}"
PUSH=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag) TAG="${2:?}"; shift 2 ;;
    --registry) REGISTRY="${2%/}"; shift 2 ;;
    --push) PUSH=1; shift ;;
    -h|--help) echo "Usage: $0 [--tag TAG] [--registry PRIVATE_HOST/PATH --push]"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ "${PUSH}" == 1 && -z "${REGISTRY}" ]]; then
  echo "ERROR: --push requires --registry" >&2
  exit 2
fi
if [[ "${REGISTRY}" == "ghcr.io/nebius/nebius-physical-ai" ]]; then
  echo "ERROR: OpenVLA-OFT is unvalidated and cannot use the official public registry" >&2
  exit 2
fi

IMAGE="npa-openvla-oft:${TAG}"
if [[ -n "${REGISTRY}" ]]; then
  IMAGE="${REGISTRY}/npa-openvla-oft:${TAG}"
fi
ARGS=(
  --platform linux/amd64
  --file "${SCRIPT_DIR}/Dockerfile"
  --tag "${IMAGE}"
  --build-arg "NPA_SOURCE_COMMIT=$(git -C "${REPO_ROOT}" rev-parse HEAD)"
)
if [[ "${PUSH}" == 1 ]]; then
  ARGS+=(--push --provenance=mode=max --sbom=true)
else
  ARGS+=(--load --provenance=false)
fi

env -u HF_TOKEN -u NGC_API_KEY -u NVIDIA_API_KEY -u NEBIUS_TOKEN_FACTORY_KEY \
  -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY -u NEBIUS_IAM_TOKEN \
  docker buildx build "${ARGS[@]}" "${NPA_ROOT}"
