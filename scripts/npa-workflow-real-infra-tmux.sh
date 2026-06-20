#!/usr/bin/env bash
# Full npa.workflow real-infra test matrix (guide runbook + unit + live e2e).
# tmux new -s npa-workflow-real-infra ./scripts/npa-workflow-real-infra-tmux.sh
set -euo pipefail

export HOME="${HOME:-/home/ubuntu}"
[[ -f /home/ubuntu/bin/npa-cloud-env.sh ]] && source /home/ubuntu/bin/npa-cloud-env.sh

REPO="/home/ubuntu/nebius-physical-ai"
cd "$REPO"

# Always test workflow branch code.
git checkout feat/npa-workflow-v0.0.1 >/dev/null 2>&1 || true

PY="${REPO}/npa/.venv/bin/python"
NPA="${REPO}/npa/.venv/bin/npa"
export NPA_INTEGRATION_E2E=1
LOG="/tmp/npa-workflow-real-infra-$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== npa.workflow REAL INFRA matrix log=${LOG} ==="
echo "branch: $(git branch --show-current) @ $(git rev-parse --short HEAD)"

round=1
while true; do
  echo ""
  echo "========== ROUND ${round} $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="

  echo "--- [1/4] unit + smoke ---"
  "${PY}" -m pytest \
    npa/tests/orchestration/npa_workflow/ \
    npa/tests/smoke/test_npa_workflow_smoke.py \
    -q --timeout=120

  echo "--- [2/4] live e2e (validate/plan CLI) ---"
  "${PY}" -m pytest npa/tests/e2e/test_npa_workflow_live_e2e.py -q --timeout=180

  echo "--- [3/4] live infra runbook (S3 persist, fail manifest, scheduler, require-inputs) ---"
  "${PY}" -m pytest npa/tests/e2e/test_npa_workflow_live_infra.py -q --timeout=300

  echo "--- [4/4] guide quickstart CLI (golden YAMLs) ---"
  SPECS="${REPO}/npa/workflows/workbench/npa-workflows"
  RUN_ID="tmux-guide-r${round}-$(date -u +%H%M%S)"
  for spec in "${SPECS}"/*.yaml; do
    base=$(basename "$spec")
    echo "spec: ${base}"
    "${NPA}" workbench workflow validate-spec "${spec}"
    "${NPA}" workbench workflow plan-spec "${spec}" \
      --run-id "${RUN_ID}" \
      --assume-decision promote_checkpoint
    if [[ "${base}" == "sim2real-vlm-rl.yaml" ]]; then
      "${NPA}" workbench workflow plan-spec "${spec}" \
        --run-id "${RUN_ID}-loop" \
        --assume-decision loop_back \
        --json | "${PY}" -c "
import json, sys
p = json.load(sys.stdin)
states = [s['state'] for s in p['steps']]
assert states.count('finalize') == 1, states
assert states.count('rollouts') == 6, states
print('sim2real promote OK: finalize=1 rollouts=6')
"
    fi
    "${NPA}" workbench workflow run-spec "${spec}" \
      --run-id "${RUN_ID}" \
      --plan-only \
      --scheduler-plan \
      --json | "${PY}" -c "
import json, sys
r = json.load(sys.stdin)
assert r.get('plan', {}).get('steps'), r
assert r.get('scheduler', {}).get('tasks'), r
print('scheduler tasks', len(r['scheduler']['tasks']))
"
  done

  echo "--- ROUND ${round} ALL PASSED ---"
  round=$((round + 1))
  sleep 600
done
