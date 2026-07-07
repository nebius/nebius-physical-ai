#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
./scripts/install-compose.sh
./scripts/bootstrap-nebius.sh

compose=(docker compose)
if [[ "${MLFLOW_USE_PUBLISHED_IMAGE:-0}" == "1" ]]; then
  if [[ -z "${MLFLOW_IMAGE:-}" ]]; then
    echo "MLFLOW_IMAGE must be set when MLFLOW_USE_PUBLISHED_IMAGE=1" >&2
    exit 1
  fi
  compose=(docker compose -f docker-compose.yml -f docker-compose.published.yml)
  "${compose[@]}" pull mlflow
  "${compose[@]}" up -d --no-build --remove-orphans
else
  docker compose build --pull
  docker compose up -d --remove-orphans
fi
./scripts/wait-healthy.sh
