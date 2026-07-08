#!/usr/bin/env bash
# Live SkyPilot submit matrix for npa.workflow/v0.0.1 twins.
#
# Run on an operator/dev VM that already has:
#   - npa configured (npa configure --interactive)
#   - SkyPilot bootstrapped (npa skypilot bootstrap)
#   - NPA_REGISTRY (or NPA_E2E_REGISTRY) pointing at a reachable Nebius registry
#   - Optional: NEBIUS_TOKEN_FACTORY_KEY for cpu-tier twins
#   - Optional: HF_TOKEN / NGC_API_KEY for SONIC / Cosmos3 twins
#
# Examples:
#   # Full matrix (cpu + gpu + multi)
#   ./scripts/npa-workflow-submit-live-e2e.sh
#
#   # Cheap first: Token Factory CPU twins only
#   NPA_E2E_NPA_WORKFLOW_SUBMIT_TIERS=cpu ./scripts/npa-workflow-submit-live-e2e.sh
#
#   # One GPU twin
#   NPA_E2E_NPA_WORKFLOW_SUBMIT_TIERS=gpu \
#   NPA_E2E_NPA_WORKFLOW_SUBMIT_SPECS=vlm-eval-single.yaml \
#     ./scripts/npa-workflow-submit-live-e2e.sh
#
#   # Plan-only preflight for every twin (no cluster launch)
#   NPA_E2E_NPA_WORKFLOW_SUBMIT_PLAN_ONLY=1 ./scripts/npa-workflow-submit-live-e2e.sh
#
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
NPA="${REPO_ROOT}/npa/.venv/bin/npa"
export NPA_SKYPILOT_BIN="${NPA_SKYPILOT_BIN:-${HOME}/.npa/skypilot-venv/bin/sky}"
export NPA_INTEGRATION_E2E=1
export NPA_E2E_NPA_WORKFLOW_SUBMIT=1
export NPA_E2E_NPA_WORKFLOW_SUBMIT_TIERS="${NPA_E2E_NPA_WORKFLOW_SUBMIT_TIERS:-cpu,gpu,multi}"
export NPA_E2E_NPA_WORKFLOW_SUBMIT_MAX_WAIT_SECONDS="${NPA_E2E_NPA_WORKFLOW_SUBMIT_MAX_WAIT_SECONDS:-3600}"
export NPA_E2E_NPA_WORKFLOW_SUBMIT_POLL_SECONDS="${NPA_E2E_NPA_WORKFLOW_SUBMIT_POLL_SECONDS:-30}"
export NPA_E2E_NPA_WORKFLOW_SUBMIT_CANCEL_ON_TIMEOUT="${NPA_E2E_NPA_WORKFLOW_SUBMIT_CANCEL_ON_TIMEOUT:-1}"
export PYTHONUNBUFFERED=1

RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="${NPA_LIVE_E2E_LOG_DIR:-${HOME}/npa-live-e2e-logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/npa-workflow-submit-live-${RUN_STAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

log() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }

log "=== npa.workflow live submit e2e ==="
log "repo: ${REPO_ROOT}"
log "log:  ${LOG_FILE}"
log "tiers: ${NPA_E2E_NPA_WORKFLOW_SUBMIT_TIERS}"
log "specs: ${NPA_E2E_NPA_WORKFLOW_SUBMIT_SPECS:-(all in selected tiers)}"
log "registry: ${NPA_E2E_REGISTRY:-${NPA_REGISTRY:-(unset)}}"

if [[ ! -x "$PY" ]]; then
  log "ERROR: python not found at $PY"
  exit 2
fi
if [[ ! -x "$NPA" ]]; then
  log "ERROR: npa not found at $NPA"
  exit 2
fi

log "--- preflight: npa --version ---"
"$NPA" --version

log "--- preflight: skypilot status ---"
if ! "$NPA" skypilot status; then
  log "SkyPilot not ready; attempting bootstrap"
  "$NPA" skypilot bootstrap
  "$NPA" skypilot status
fi

log "--- unit: renderer + smoke (no cluster) ---"
"$PY" -m pytest \
  npa/tests/orchestration/npa_workflow/test_skypilot_render.py \
  npa/tests/smoke/test_all_workflow_yamls.py \
  npa/tests/smoke/test_npa_workflow_smoke.py \
  -q --timeout=120

if [[ "${NPA_E2E_NPA_WORKFLOW_SUBMIT_PLAN_ONLY:-0}" == "1" ]]; then
  log "--- plan-only matrix (no sky jobs launch) ---"
  "$PY" -m pytest \
    npa/tests/e2e/test_npa_workflow_submit_live_e2e.py::test_npa_workflow_submit_plan_only_matrix_no_leak \
    -q --timeout=600 -s
  log "=== plan-only matrix complete ==="
  exit 0
fi

log "--- live submit matrix (sky jobs launch + poll) ---"
"$PY" -m pytest \
  npa/tests/e2e/test_npa_workflow_submit_live_e2e.py \
  -q --timeout="${NPA_E2E_NPA_WORKFLOW_SUBMIT_MAX_WAIT_SECONDS}" \
  -s --tb=short

log "=== live submit matrix complete ==="
log "log: ${LOG_FILE}"
