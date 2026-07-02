#!/usr/bin/env bash
# Generic BYOF onboarding live pipeline: workflow gates, agent chat, container, GPU smoke.
# Resolves project storage/Kubernetes from ~/.npa/config.yaml (customer/operator project).
# LeIsaac is the default validation repo when building a real container image.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

export NPA_INTEGRATION_E2E=1
PROJECT="${NPA_E2E_PROJECT:-${NPA_AGENT_PROJECT:-}}"
if [[ -z "$PROJECT" ]]; then
  PROJECT="$(npa/.venv/bin/python -c 'from npa.workflows.byof.live import resolve_byof_project; print(resolve_byof_project())')"
fi
if [[ -z "$PROJECT" ]]; then
  echo "Set NPA_E2E_PROJECT or configure a writable project in ~/.npa/config.yaml" >&2
  exit 1
fi
export NPA_E2E_PROJECT="$PROJECT"
export NPA_AGENT_PROJECT="${NPA_AGENT_PROJECT:-$PROJECT}"

PYTEST_ARGS=(-q)
PIPELINE="${NPA_BYOF_LIVE_PIPELINE:-0}"

echo "=== BYOF onboarding live pipeline (project=${PROJECT}) ==="

npa/.venv/bin/python -m pytest npa/tests/e2e/test_byof_onboarding_live_e2e.py \
  "${PYTEST_ARGS[@]}" \
  -k "not live_byof_runner_submit_smoke and not live_byof_runner_registry_smoke and not live_byof_runner_container_build_push and not live_byof_container_has_validation_repo and not live_agent_onboard_solution_chat and not live_agent_byof_workflow_draft_validate and not live_agent_oss_repo_onboard and not live_byof_ubuntu_oss" \
  --timeout=120

_run_agent_tier() {
  echo "--- live agent: onboard_solution + generic BYOF workflow draft ---"
  export NPA_AGENT_LIVE=1
  export NPA_AGENT_NAME="${NPA_AGENT_NAME:-agent}"
  npa/.venv/bin/python -m pytest npa/tests/e2e/test_byof_onboarding_live_e2e.py \
    "${PYTEST_ARGS[@]}" \
    -k "live_agent_onboard_solution_chat or live_agent_byof_workflow_draft_validate" \
    --timeout=120
}

_run_container_tier() {
  echo "--- live BYOF container build/push/inspect (validation repo) ---"
  nebius profile activate "${NPA_NEBIUS_PROFILE:-agent-sa}" 2>/dev/null || true
  npa/.venv/bin/python -m pytest npa/tests/e2e/test_byof_onboarding_live_e2e.py \
    "${PYTEST_ARGS[@]}" \
    -k "live_byof_runner_container_build_push or live_byof_container_has_validation_repo" \
    --timeout="${NPA_BYOF_CONTAINER_TIMEOUT:-3600}"
}

_run_gpu_tier() {
  echo "--- live BYOF runner registry + submit smoke (GPU path) ---"
  nebius profile activate "${NPA_NEBIUS_PROFILE:-agent-sa}" 2>/dev/null || true
  npa/.venv/bin/python -m pytest npa/tests/e2e/test_byof_onboarding_live_e2e.py \
    "${PYTEST_ARGS[@]}" \
    -k "live_byof_runner_registry_smoke or live_byof_runner_submit_smoke" \
    --timeout="${NPA_BYOF_LIVE_TIMEOUT:-21600}"
}

_run_ubuntu_oss_tier() {
  echo "--- live Ubuntu OSS BYOF: agent + container + container-verify ---"
  export NPA_BYOF_REPO_URL="${NPA_BYOF_REPO_URL:-https://github.com/githubtraining/hellogitworld.git}"
  export NPA_BYOF_REPO_REF="${NPA_BYOF_REPO_REF:-master}"
  export NPA_BYOF_BASE_PROFILE=ubuntu
  nebius profile activate "${NPA_NEBIUS_PROFILE:-agent-sa}" 2>/dev/null || true
  npa/.venv/bin/python -m pytest npa/tests/e2e/test_byof_onboarding_live_e2e.py \
    "${PYTEST_ARGS[@]}" \
    -k "live_agent_oss_repo_onboard or live_byof_ubuntu_oss" \
    --timeout="${NPA_BYOF_LIVE_TIMEOUT:-7200}"
}

if [[ "$PIPELINE" == "1" ]]; then
  export NPA_AGENT_LIVE=1
  export NPA_BYOF_LIVE_CONTAINER=1
  export NPA_BYOF_LIVE_GPU=1
  _run_agent_tier
  _run_container_tier
  _run_gpu_tier
else
  if [[ "${NPA_AGENT_LIVE:-0}" == "1" ]]; then
    _run_agent_tier
  fi
  if [[ "${NPA_BYOF_LIVE_CONTAINER:-0}" == "1" ]]; then
    _run_container_tier
  fi
  if [[ "${NPA_BYOF_LIVE_GPU:-0}" == "1" ]]; then
    _run_gpu_tier
  fi
  if [[ "${NPA_BYOF_LIVE_UBUNTU:-0}" == "1" ]]; then
    export NPA_AGENT_LIVE=1
    export NPA_BYOF_LIVE_UBUNTU=1
    export NPA_BYOF_LIVE_GPU=1
    _run_ubuntu_oss_tier
  fi
fi

echo "verify_byof_onboarding_live: ok"
