#!/usr/bin/env bash
# Live infra verification for BYOF solution onboarding (workflow + agent chat + optional GPU smoke).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

export NPA_INTEGRATION_E2E=1
PROJECT="${NPA_E2E_PROJECT:-${NPA_AGENT_PROJECT:-eu-north1}}"
PYTEST_ARGS=(-q)

echo "=== BYOF onboarding live infra (project=${PROJECT}) ==="

npa/.venv/bin/python -m pytest npa/tests/e2e/test_byof_onboarding_live_e2e.py \
  "${PYTEST_ARGS[@]}" \
  -k "not live_byof_runner_submit_smoke and not live_byof_runner_registry_smoke and not live_byof_runner_container_build_push and not live_byof_container_has_leisaac" \
  --timeout=120

if [[ "${NPA_AGENT_LIVE:-0}" == "1" ]]; then
  echo "--- live agent onboard_solution chat ---"
  export NPA_AGENT_PROJECT="${NPA_AGENT_PROJECT:-rtxpro}"
  export NPA_AGENT_NAME="${NPA_AGENT_NAME:-agent}"
  npa/.venv/bin/python -m pytest npa/tests/e2e/test_byof_onboarding_live_e2e.py \
    "${PYTEST_ARGS[@]}" \
    -k "live_agent_onboard_solution_chat or live_agent_byof_workflow_draft_validate" \
    --timeout=120
fi

if [[ "${NPA_BYOF_LIVE_CONTAINER:-0}" == "1" ]]; then
  echo "--- live BYOF container build/push/inspect ---"
  nebius profile activate "${NPA_NEBIUS_PROFILE:-agent-sa}" 2>/dev/null || true
  npa/.venv/bin/python -m pytest npa/tests/e2e/test_byof_onboarding_live_e2e.py \
    "${PYTEST_ARGS[@]}" \
    -k "live_byof_runner_container_build_push or live_byof_container_has_leisaac" \
    --timeout="${NPA_BYOF_CONTAINER_TIMEOUT:-3600}"
fi

if [[ "${NPA_BYOF_LIVE_GPU:-0}" == "1" ]]; then
  echo "--- live BYOF runner registry + submit smoke (GPU path) ---"
  nebius profile activate "${NPA_NEBIUS_PROFILE:-agent-sa}" 2>/dev/null || true
  npa/.venv/bin/python -m pytest npa/tests/e2e/test_byof_onboarding_live_e2e.py \
    "${PYTEST_ARGS[@]}" \
    -k "live_byof_runner_registry_smoke or live_byof_runner_submit_smoke" \
    --timeout="${NPA_BYOF_LIVE_TIMEOUT:-21600}"
fi

echo "verify_byof_onboarding_live: ok"
