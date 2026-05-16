#!/usr/bin/env bash
set -euo pipefail

export LANCEDB_STORAGE_PATH="${LANCEDB_STORAGE_PATH:-/data/lancedb}"
export LANCEDB_PORT="${LANCEDB_PORT:-8686}"
export LANCEDB_AUTH_MODE="${LANCEDB_AUTH_MODE:-token}"

exec uvicorn npa_lancedb_server:app --app-dir /app --host 0.0.0.0 --port "${LANCEDB_PORT}"
