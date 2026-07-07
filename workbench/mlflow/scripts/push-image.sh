#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

./scripts/install-compose.sh >&2

PROJECT_ID="${NPA_PROJECT_ID:-project-u00zhx4tpr00xh99b28n52}"
NPA_PYTHON="${NPA_PYTHON:-/home/ubuntu/nebius-physical-ai/npa/.venv/bin/python}"
if [[ ! -x "$NPA_PYTHON" && -x "../../npa/.venv/bin/python" ]]; then
  NPA_PYTHON="../../npa/.venv/bin/python"
fi

REGISTRY="${NPA_REGISTRY:-}"
if [[ -z "$REGISTRY" ]]; then
  REGISTRY="$("$NPA_PYTHON" - <<PYREG
from npa.clients.config import resolve_container_registry
print(resolve_container_registry("$PROJECT_ID"))
PYREG
)"
fi
REGISTRY_HOST="${REGISTRY%%/*}"
TAG="${MLFLOW_IMAGE_TAG:-$(git -C ../.. rev-parse --short=12 HEAD 2>/dev/null || date -u +%Y%m%d%H%M%S)}"
IMAGE="${MLFLOW_IMAGE:-${REGISTRY}/npa-mlflow-server:${TAG}}"

mkdir -p evidence
if [[ "${MLFLOW_SKIP_BUILD:-0}" != "1" ]]; then
  docker compose build --pull mlflow >&2
fi

nebius iam get-access-token | docker login "$REGISTRY_HOST" -u iam --password-stdin >/tmp/npa-mlflow-registry-login.log
docker tag npa-mlflow-server:local "$IMAGE"
docker push "$IMAGE" >&2
printf "%s\n" "$IMAGE" > evidence/pushed-image-ref.txt
docker image inspect "$IMAGE" --format "{{json .RepoDigests}}" > evidence/pushed-image-digests.json
printf "MLFLOW_IMAGE=%s\n" "$IMAGE" > evidence/pushed-image.env
printf "%s\n" "$IMAGE"
