#!/usr/bin/env bash
# Smoke + live-infra validation for npa.workflow YAML changes.
# tmux new -s npa-workflow-smoke-live ./scripts/npa-workflow-smoke-live-tmux.sh
set -euo pipefail

export HOME="${HOME:-/home/ubuntu}"
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN
[[ -f /home/ubuntu/bin/npa-cloud-env.sh ]] && source /home/ubuntu/bin/npa-cloud-env.sh

REPO="/home/ubuntu/nebius-physical-ai"
cd "$REPO"
PY="${REPO}/npa/.venv/bin/python"
export NPA_INTEGRATION_E2E=1

LOG="/tmp/npa-workflow-smoke-live-$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== npa.workflow smoke + live infra log=${LOG} ==="
echo "branch: $(git branch --show-current) @ $(git rev-parse --short HEAD)"

round=1
while true; do
  echo ""
  echo "========== ROUND ${round} $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
  FAILED=0

  echo "--- unit (spec + catalog validation) ---"
  if ! "${PY}" -m pytest npa/tests/orchestration/npa_workflow/ -q --timeout=120; then
    FAILED=1
  fi

  echo "--- smoke ---"
  if ! "${PY}" -m pytest npa/tests/smoke/test_npa_workflow_smoke.py -q --timeout=120; then
    FAILED=1
  fi

  echo "--- live infra runbook ---"
  if ! "${PY}" -m pytest npa/tests/e2e/test_npa_workflow_live_infra.py -q --timeout=300; then
    FAILED=1
  fi

  if [[ "${FAILED}" -eq 0 ]]; then
    echo "--- ROUND ${round} ALL PASSED ---"
  else
    echo "--- ROUND ${round} FAILED ---"
  fi
  round=$((round + 1))
  sleep 600
done
