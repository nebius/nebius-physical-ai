#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

./scripts/install-compose.sh >&2

REGISTRY="${NPA_REGISTRY:-}"
if [[ -z "$REGISTRY" ]]; then
  echo "ERROR: set NPA_REGISTRY to an operator-controlled registry" >&2
  exit 2
fi
case "${REGISTRY%/}" in
  ghcr.io/nebius/nebius-physical-ai)
    echo "ERROR: this build-your-own helper cannot publish to official NPA GHCR" >&2
    exit 2
    ;;
esac
REGISTRY_HOST="${REGISTRY%%/*}"
SOURCE_SHA="$(git -C ../.. rev-parse HEAD)"
[[ "${SOURCE_SHA}" =~ ^[0-9a-f]{40}$ ]] || { echo "ERROR: checkout SHA is not immutable" >&2; exit 2; }
TAG="${MLFLOW_IMAGE_TAG:-dev-${SOURCE_SHA}}"
IMAGE="${MLFLOW_IMAGE:-${REGISTRY}/npa-mlflow-server:${TAG}}"
POSTGRES_SOURCE_IMAGE="${POSTGRES_SOURCE_IMAGE:-cgr.dev/chainguard/postgres@sha256:0edb7d98cf916a0f00f80c0f4b9257c8737c1ee1848d1e4e0f480b12a932d90b}"
POSTGRES_TARGET_IMAGE="${POSTGRES_IMAGE:-${REGISTRY}/npa-mlflow-postgres:${TAG}}"

mkdir -p evidence
if [[ "${MLFLOW_SKIP_BUILD:-0}" != "1" ]]; then
  docker compose build --pull mlflow >&2
fi

echo "Using existing Docker credentials for ${REGISTRY_HOST}; run 'docker login ${REGISTRY_HOST}' if needed." >&2
docker tag npa-mlflow-server:local "$IMAGE"
docker push "$IMAGE" >&2
docker pull "$POSTGRES_SOURCE_IMAGE" >&2
docker tag "$POSTGRES_SOURCE_IMAGE" "$POSTGRES_TARGET_IMAGE"
docker push "$POSTGRES_TARGET_IMAGE" >&2
printf "%s\n" "$IMAGE" > evidence/pushed-image-ref.txt
printf "%s\n" "$POSTGRES_TARGET_IMAGE" > evidence/pushed-postgres-image-ref.txt
docker image inspect "$IMAGE" --format "{{json .RepoDigests}}" > evidence/pushed-image-digests.json
docker image inspect "$POSTGRES_TARGET_IMAGE" --format "{{json .RepoDigests}}" > evidence/pushed-postgres-image-digests.json
printf "MLFLOW_IMAGE=%s\nPOSTGRES_IMAGE=%s\n" "$IMAGE" "$POSTGRES_TARGET_IMAGE" > evidence/pushed-images.env
printf "MLFLOW_IMAGE=%s\n" "$IMAGE" > evidence/pushed-image.env
printf "%s\n" "$IMAGE"
