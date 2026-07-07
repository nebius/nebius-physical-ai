#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
./scripts/install-compose.sh
./scripts/bootstrap-nebius.sh
docker compose build --pull
docker compose up -d --remove-orphans
./scripts/wait-healthy.sh
