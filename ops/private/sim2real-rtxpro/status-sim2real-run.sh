#!/usr/bin/env bash
# Live Sim2Real stage progress — kubectl + S3 only (no npa CLI version required).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_ID="${1:?usage: status-sim2real-run.sh <run-id>}"
exec "${SCRIPT_DIR}/status-run-local.sh" "${RUN_ID}" --watch
