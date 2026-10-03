#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NPA_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
REPO_ROOT="$(cd "${NPA_ROOT}/.." && pwd)"
TAG="${TAG:-local}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag) TAG="${2:?}"; shift 2 ;;
    -h|--help) echo "Usage: $0 [--tag TAG]"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

docker buildx build --load --platform linux/amd64 \
  --file "${SCRIPT_DIR}/Dockerfile" \
  --tag "npa-openvla-oft:${TAG}" \
  --build-arg "NPA_SOURCE_COMMIT=$(git -C "${REPO_ROOT}" rev-parse HEAD)" \
  "${NPA_ROOT}"
