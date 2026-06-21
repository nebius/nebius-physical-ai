#!/usr/bin/env bash
# Live validation of npa.workflow/v0.0.1 specs on operator machine + Nebius creds.
# Run in tmux: tmux new -s npa-workflow-live ./scripts/npa-workflow-live-e2e.sh
set -euo pipefail

export HOME="${HOME:-/home/ubuntu}"
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN
[[ -f /home/ubuntu/bin/npa-cloud-env.sh ]] && source /home/ubuntu/bin/npa-cloud-env.sh

REPO="/home/ubuntu/nebius-physical-ai"
cd "$REPO"
PY="${REPO}/npa/.venv/bin/python"
NPA="${REPO}/npa/.venv/bin/npa"
export NPA_INTEGRATION_E2E=1
SPECS_DIR="${REPO}/npa/workflows/workbench/npa-workflows"
DYNAMIC="sim2real-vlm-rl.yaml tokenfactory-cosmos-gate.yaml"
RUN_ID="npa-workflow-live-$(date -u +%Y%m%dT%H%M%SZ)"
LOG="/tmp/${RUN_ID}.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== NPA workflow live e2e run_id=${RUN_ID} ==="
echo "log: ${LOG}"

round=1
while true; do
  echo ""
  echo "--- round ${round} $(date -u +%Y-%m-%dT%H:%M:%SZ) ---"
  FAILED=0

  echo "--- unit: npa_workflow + workflow CLI fixes ---"
  if ! "${PY}" -m pytest \
    npa/tests/orchestration/npa_workflow/ \
    npa/tests/smoke/test_npa_workflow_smoke.py \
    npa/tests/cli/test_workflow_cli.py::test_workflow_status_prints_status \
    npa/tests/cli/test_workflow_cli.py::test_workflow_status_maps_distillation_error \
    npa/tests/cli/test_workflow_cli.py::test_durable_workflow_status_logs_and_artifacts_read_s3 \
    npa/tests/workbench/test_cosmos3_access.py::test_cosmos3_fetch_clones_and_downloads_without_token_args \
    -q --timeout=120; then
    FAILED=1
  fi

  echo "--- CLI validate-spec / plan-spec (all golden YAMLs) ---"
  for spec in "${SPECS_DIR}"/*.yaml; do
    base=$(basename "$spec")
    echo "spec: ${spec}"
    if ! "${NPA}" workbench workflow validate-spec "${spec}"; then
      FAILED=1
      continue
    fi
    extra=()
    if [[ " ${DYNAMIC} " == *" ${base} "* ]]; then
      extra=(--assume-decision promote_checkpoint)
    fi
    if ! "${NPA}" workbench workflow plan-spec "${spec}" \
      --run-id "${RUN_ID}-r${round}" \
      "${extra[@]}"; then
      FAILED=1
      continue
    fi
    if ! "${NPA}" workbench workflow run-spec "${spec}" \
      --run-id "${RUN_ID}-r${round}" \
      --plan-only \
      "${extra[@]}" \
      --json | "${PY}" -c "
import json, sys
report = json.load(sys.stdin)
steps = [s['state'] for s in report['plan']['steps']]
assert steps, report
if report['workflow'] == 'sim2real-vlm-rl':
    assert steps.count('finalize') == 1, steps
"; then
      FAILED=1
    fi
  done

  echo "--- pytest live e2e + infra ---"
  if ! "${PY}" -m pytest \
    npa/tests/e2e/test_npa_workflow_live_e2e.py \
    npa/tests/e2e/test_npa_workflow_live_infra.py \
    -q --timeout=600; then
    FAILED=1
  fi

  if [[ "${FAILED}" -eq 0 ]]; then
    echo "--- round ${round} ALL PASSED ---"
  else
    echo "--- round ${round} FAILED ---"
  fi
  round=$((round + 1))
  sleep 300
done
