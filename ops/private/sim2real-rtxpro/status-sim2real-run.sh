#!/usr/bin/env bash
# Live Sim2Real stage progress via npa workbench workflow status.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PY="${ROOT}/npa/.venv/bin/python"
NPA="${ROOT}/npa/.venv/bin/npa"
RUN_ID="${1:?usage: status-sim2real-run.sh <run-id>}"
exec "${NPA}" workbench workflow status "${RUN_ID}" --watch
