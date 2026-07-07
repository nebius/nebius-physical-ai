#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
for pass in 1 2; do
  echo "=== clean verification pass ${pass} ==="
  docker compose down --remove-orphans || true
  ./scripts/deploy.sh
  ./scripts/verify.sh
  cp evidence/verify-summary.json "evidence/verify-summary-pass-${pass}.json"
done
