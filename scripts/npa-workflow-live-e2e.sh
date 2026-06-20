#!/usr/bin/env bash
# Live validation of npa.workflow/v0.0.1 specs on operator machine + Nebius creds.
# Run in tmux: tmux new -s npa-workflow-live ./scripts/npa-workflow-live-e2e.sh
set -euo pipefail

export HOME="${HOME:-/home/ubuntu}"
[[ -f /home/ubuntu/bin/npa-cloud-env.sh ]] && source /home/ubuntu/bin/npa-cloud-env.sh

REPO="/home/ubuntu/nebius-physical-ai"
cd "$REPO"
PY="${REPO}/npa/.venv/bin/python"
NPA="${REPO}/npa/.venv/bin/npa"
SPECS_DIR="${REPO}/npa/workflows/workbench/npa-workflows"
RUN_ID="npa-workflow-live-$(date -u +%Y%m%dT%H%M%SZ)"
LOG="/tmp/${RUN_ID}.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== NPA workflow live e2e run_id=${RUN_ID} ==="
echo "log: ${LOG}"

round=1
while true; do
  echo ""
  echo "--- round ${round} $(date -u +%Y-%m-%dT%H:%M:%SZ) ---"

  echo "--- unit: npa_workflow + workflow CLI fixes ---"
  "${PY}" -m pytest \
    npa/tests/orchestration/npa_workflow/ \
    npa/tests/smoke/test_npa_workflow_smoke.py \
    npa/tests/cli/test_workflow_cli.py::test_workflow_status_prints_status \
    npa/tests/cli/test_workflow_cli.py::test_workflow_status_maps_distillation_error \
    npa/tests/cli/test_workflow_cli.py::test_durable_workflow_status_logs_and_artifacts_read_s3 \
    npa/tests/workbench/test_cosmos3_access.py::test_cosmos3_fetch_clones_and_downloads_without_token_args \
    -q --timeout=120

  echo "--- CLI validate-spec / plan-spec (all golden YAMLs) ---"
  for spec in "${SPECS_DIR}"/*.yaml; do
    [[ "$(basename "$spec")" == "README.md" ]] && continue
    echo "spec: ${spec}"
    "${NPA}" workbench workflow validate-spec "${spec}"
    "${NPA}" workbench workflow plan-spec "${spec}" \
      --run-id "${RUN_ID}-r${round}" \
      --assume-decision promote_checkpoint
    "${NPA}" workbench workflow run-spec "${spec}" \
      --run-id "${RUN_ID}-r${round}" \
      --plan-only \
      --assume-decision promote_checkpoint \
      --json | "${PY}" -c "import json,sys; d=json.load(sys.stdin); assert d.get('steps'), d"
  done

  echo "--- pytest live e2e (NPA_INTEGRATION_E2E=1) ---"
  NPA_INTEGRATION_E2E=1 "${PY}" -m pytest npa/tests/e2e/test_npa_workflow_live_e2e.py -q --timeout=120

  echo "--- round ${round} OK ---"
  round=$((round + 1))
  sleep 300
done
