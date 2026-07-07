#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

./scripts/install-compose.sh >&2

PROJECT_ID="${NPA_PROJECT_ID:-${NPA_PROJECT_ALIAS:-}}"
NPA_PYTHON="${NPA_PYTHON:-/home/ubuntu/nebius-physical-ai/npa/.venv/bin/python}"
if [[ ! -x "$NPA_PYTHON" && -x "../../npa/.venv/bin/python" ]]; then
  NPA_PYTHON="../../npa/.venv/bin/python"
fi

REGISTRY="${NPA_REGISTRY:-}"
if [[ -z "$REGISTRY" ]]; then
  REGISTRY="$("$NPA_PYTHON" - <<PYREG
from npa.clients.config import resolve_container_registry
project = "$PROJECT_ID" or None
print(resolve_container_registry(project))
PYREG
)"
fi
REGISTRY_HOST="${REGISTRY%%/*}"
TAG="${MLFLOW_IMAGE_TAG:-$(git -C ../.. rev-parse --short=12 HEAD 2>/dev/null || date -u +%Y%m%d%H%M%S)}"
IMAGE="${MLFLOW_IMAGE:-${REGISTRY}/npa-mlflow-server:${TAG}}"
POSTGRES_SOURCE_IMAGE="${POSTGRES_SOURCE_IMAGE:-cgr.dev/chainguard/postgres@sha256:0edb7d98cf916a0f00f80c0f4b9257c8737c1ee1848d1e4e0f480b12a932d90b}"
POSTGRES_TARGET_IMAGE="${POSTGRES_IMAGE:-${REGISTRY}/npa-mlflow-postgres:${TAG}}"

mkdir -p evidence
if [[ "${MLFLOW_SKIP_BUILD:-0}" != "1" ]]; then
  docker compose build --pull mlflow >&2
fi

nebius iam get-access-token | docker login "$REGISTRY_HOST" -u iam --password-stdin >/tmp/npa-mlflow-registry-login.log
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
