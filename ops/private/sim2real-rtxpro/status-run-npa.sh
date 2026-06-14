#!/usr/bin/env bash
# Live sim2real status via npa CLI (fallback: status-run-local.sh).
set -euo pipefail

RUN_ID="${1:?usage: status-run-npa.sh <run-id> [--watch]}"
WATCH="${2:-}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=lib/operator-env.sh
source "${SCRIPT_DIR}/lib/operator-env.sh"
ROOT="$(npa_repo_root "${SCRIPT_DIR}")"
export NPA_SIM2REAL_REPO="${ROOT}"
NPA_BIN="${ROOT}/npa/.venv/bin/npa"

if [ ! -x "${NPA_BIN}" ]; then
  bash "${ROOT}/ops/private/sim2real-rtxpro/bootstrap-npa-venv.sh" "${ROOT}"
fi

if [ ! -x "${NPA_BIN}" ]; then
  exec "${SCRIPT_DIR}/status-run-local.sh" "${RUN_ID}" "${WATCH}"
fi

for candidate in "sim2real-staged-${RUN_ID}" "${RUN_ID}"; do
  if [ "${WATCH}" = "--watch" ]; then
    if "${NPA_BIN}" workbench workflow status "${candidate}" --watch 2>/dev/null; then
      exit 0
    fi
  elif "${NPA_BIN}" workbench workflow status "${candidate}" 2>/dev/null; then
    exit 0
  fi
done

exec "${SCRIPT_DIR}/status-run-local.sh" "${RUN_ID}" "${WATCH}"
