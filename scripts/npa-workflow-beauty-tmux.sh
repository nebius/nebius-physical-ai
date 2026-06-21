#!/usr/bin/env bash
# Validate beautiful npa.workflow YAMLs + BDD100K tests.
# tmux new -s npa-workflow-beauty ./scripts/npa-workflow-beauty-tmux.sh
set -euo pipefail

export HOME="${HOME:-/home/ubuntu}"
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN
[[ -f /home/ubuntu/bin/npa-cloud-env.sh ]] && source /home/ubuntu/bin/npa-cloud-env.sh

REPO="/home/ubuntu/nebius-physical-ai"
cd "$REPO"
PY="${REPO}/npa/.venv/bin/python"
NPA="${REPO}/npa/.venv/bin/npa"
export NPA_INTEGRATION_E2E=1
SPECS="${REPO}/npa/workflows/workbench/npa-workflows"

LOG="/tmp/npa-workflow-beauty-$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== npa.workflow beauty matrix log=${LOG} ==="
echo "branch: $(git branch --show-current) @ $(git rev-parse --short HEAD)"

round=1
while true; do
  echo ""
  echo "========== ROUND ${round} $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
  FAILED=0

  echo "--- golden YAML validate (all specs) ---"
  for spec in "${SPECS}"/*.yaml; do
    base=$(basename "$spec")
    echo "spec: ${base}"
    if ! "${NPA}" workbench workflow validate-spec "${spec}"; then
      FAILED=1
    fi
  done

  echo "--- unit + smoke ---"
  if ! "${PY}" -m pytest \
    npa/tests/orchestration/npa_workflow/ \
    npa/tests/smoke/test_npa_workflow_smoke.py \
    -q --timeout=120; then
    FAILED=1
  fi

  echo "--- live e2e + infra ---"
  if ! "${PY}" -m pytest \
    npa/tests/e2e/test_npa_workflow_live_e2e.py \
    npa/tests/e2e/test_npa_workflow_live_infra.py \
    -q --timeout=300; then
    FAILED=1
  fi

  echo "--- BDD100K pipeline + LanceDB ---"
  if ! "${PY}" -m pytest \
    npa/tests/workflows/test_bdd100k_pipeline.py \
    npa/tests/test_lancedb_bdd100k_import.py \
    npa/tests/test_lancedb_bdd100k_backfill.py \
    npa/tests/test_lancedb_bdd100k_mv.py \
    -q --timeout=300; then
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
