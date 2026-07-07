#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
for _ in $(seq 1 90); do
  ps="$(docker compose ps --format json 2>/dev/null || true)"
  if echo "$ps" | jq -e -s 'map(select(.Health == "healthy" or .State == "running")) | length >= 2' >/dev/null 2>&1; then
    curl -fsS http://127.0.0.1:5000/health >/dev/null
    docker compose ps
    exit 0
  fi
  sleep 2
done
docker compose ps || true
docker compose logs --no-color --tail=200 || true
exit 1
