#!/usr/bin/env bash
# Live runtime-orchestrator tier for npa.workflow/v0.0.1.
#
# Mirrors scripts/npa-workflow-submit-live-e2e.sh, but drives the specs that need
# the runtime tier (`submit --runtime`):
#
#   * token-factory-parallel-fanout.yaml  (cpu)   real concurrent JobGroup + barrier
#   * token-factory-gate-loop.yaml        (cpu)   real runtime early-exit + branch
#   * isaac-lab-rl-sweep.yaml             (multi) parallel GPU sweep + ranking barrier
#
# Cheapest tier first:
#   NPA_E2E_NPA_WORKFLOW_SUBMIT_TIERS=cpu ./scripts/npa-workflow-runtime-live-e2e.sh
#
# Everything else (registry, project, S3 endpoint, GPU remaps) comes from
# ~/.npa/live-e2e.env, exactly like the submit matrix runner. Nothing is hardcoded.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${NPA_LIVE_E2E_REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
cd "$REPO_ROOT"

if [[ -f /home/ubuntu/bin/npa-cloud-env.sh ]]; then
  # shellcheck source=/dev/null
  source /home/ubuntu/bin/npa-cloud-env.sh
fi
if [[ -f "${HOME}/.npa/live-e2e.env" ]]; then
  set -a
  # shellcheck source=/dev/null
  . "${HOME}/.npa/live-e2e.env"
  set +a
fi

PY="${NPA_LIVE_E2E_PYTHON_BIN:-${REPO_ROOT}/npa/.venv/bin/python}"
export PYTHONPATH="${REPO_ROOT}/npa/src${PYTHONPATH:+:$PYTHONPATH}"
export NPA_SKYPILOT_BIN="${NPA_SKYPILOT_BIN:-${HOME}/.npa/skypilot-venv/bin/sky}"
export NPA_INTEGRATION_E2E=1
export NPA_E2E_NPA_WORKFLOW_SUBMIT=1
export NPA_E2E_NPA_WORKFLOW_RUNTIME=1
export NPA_E2E_NPA_WORKFLOW_SUBMIT_TIERS="${NPA_E2E_NPA_WORKFLOW_SUBMIT_TIERS:-cpu}"
export NPA_E2E_NPA_WORKFLOW_SUBMIT_MAX_WAIT_SECONDS="${NPA_E2E_NPA_WORKFLOW_SUBMIT_MAX_WAIT_SECONDS:-3600}"
export NPA_E2E_NPA_WORKFLOW_SUBMIT_POLL_SECONDS="${NPA_E2E_NPA_WORKFLOW_SUBMIT_POLL_SECONDS:-20}"
export NPA_E2E_NPA_WORKFLOW_SUBMIT_CANCEL_ON_TIMEOUT="${NPA_E2E_NPA_WORKFLOW_SUBMIT_CANCEL_ON_TIMEOUT:-1}"
export PYTHONUNBUFFERED=1

RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="${NPA_LIVE_E2E_LOG_DIR:-${HOME}/npa-live-e2e-logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/npa-workflow-runtime-live-${RUN_STAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

log() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }

log "=== npa.workflow runtime live e2e ==="
log "repo:  ${REPO_ROOT}"
log "log:   ${LOG_FILE}"
log "tiers: ${NPA_E2E_NPA_WORKFLOW_SUBMIT_TIERS}"
log "specs: ${NPA_E2E_NPA_WORKFLOW_SUBMIT_SPECS:-(all runtime cases in selected tiers)}"

if [[ ! -x "$PY" ]]; then
  log "ERROR: python not found at $PY"
  exit 2
fi

log "--- unit: engine + renderer + runtime (no cluster) ---"
"$PY" -m pytest \
  npa/tests/orchestration/npa_workflow/ \
  npa/tests/smoke/test_all_workflow_yamls.py \
  npa/tests/smoke/test_npa_workflow_smoke.py \
  -q --timeout=180

log "--- live runtime tier (submit --runtime, poll, decide, replan) ---"
"$PY" -m pytest \
  npa/tests/e2e/test_npa_workflow_submit_live_e2e.py::test_npa_workflow_runtime_live_reaches_terminal \
  npa/tests/e2e/test_npa_workflow_submit_live_e2e.py::test_npa_workflow_runtime_gate_loop_early_exit_vs_full_budget \
  -q --timeout="${NPA_E2E_NPA_WORKFLOW_SUBMIT_MAX_WAIT_SECONDS}" \
  -s --tb=short

log "=== runtime live e2e complete ==="
log "evidence: ${LOG_DIR}/runtime-*.json"
