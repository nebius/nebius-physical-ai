#!/usr/bin/env bash
# Full workflow YAML matrix: npa.workflow + SkyPilot parse + live infra tests.
# tmux new -s npa-all-yaml-infra ./scripts/npa-all-yaml-infra-tmux.sh
set -euo pipefail

export HOME="${HOME:-/home/ubuntu}"
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN
[[ -f /home/ubuntu/bin/npa-cloud-env.sh ]] && source /home/ubuntu/bin/npa-cloud-env.sh

REPO="/home/ubuntu/nebius-physical-ai"
cd "$REPO"
PY="${REPO}/npa/.venv/bin/python"
NPA="${REPO}/npa/.venv/bin/npa"
export NPA_INTEGRATION_E2E=1

NPA_SPECS="${REPO}/npa/workflows/workbench/npa-workflows"
SKY_SPECS="${REPO}/npa/workflows/workbench/skypilot"

LOG="/tmp/npa-all-yaml-infra-$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== ALL workflow YAML infra matrix log=${LOG} ==="
echo "branch: $(git branch --show-current) @ $(git rev-parse --short HEAD)"
echo "npa.workflow specs: $(find "${NPA_SPECS}" -maxdepth 1 -name '*.yaml' | wc -l)"
echo "skypilot specs: $(find "${SKY_SPECS}" -maxdepth 1 -name '*.yaml' | wc -l)"

round=1
while true; do
  echo ""
  echo "========== ROUND ${round} $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
  FAILED=0

  echo "--- [1/6] npa.workflow validate (all golden specs) ---"
  for spec in "${NPA_SPECS}"/*.yaml; do
    base=$(basename "$spec")
    echo "validate: ${base}"
    if ! "${NPA}" workbench workflow validate-spec "${spec}"; then
      FAILED=1
    fi
  done

  echo "--- [2/6] npa.workflow plan + scheduler (all golden specs) ---"
  RUN_ID="tmux-all-r${round}-$(date -u +%H%M%S)"
  for spec in "${NPA_SPECS}"/*.yaml; do
    base=$(basename "$spec")
    echo "plan: ${base}"
    extra=()
    if [[ "${base}" == "sim2real-vlm-rl.yaml" ]]; then
      extra=(--assume-decision loop_back)
    fi
    if ! "${NPA}" workbench workflow plan-spec "${spec}" --run-id "${RUN_ID}" "${extra[@]}"; then
      FAILED=1
      continue
    fi
    if ! "${NPA}" workbench workflow run-spec "${spec}" \
      --run-id "${RUN_ID}" \
      --plan-only \
      --scheduler-plan \
      "${extra[@]}" \
      --json | "${PY}" -c "import json,sys; r=json.load(sys.stdin); assert r.get('scheduler',{}).get('tasks'), r"; then
      FAILED=1
    fi
  done

  echo "--- [3/6] skypilot YAML parse (all files) ---"
  if ! "${PY}" - <<'PY'; then
from pathlib import Path
import yaml

root = Path("npa/workflows/workbench/skypilot")
failed = []
for path in sorted(root.glob("*.yaml")):
    try:
        docs = [d for d in yaml.safe_load_all(path.read_text(encoding="utf-8")) if d is not None]
        if not docs or not docs[0].get("name"):
            failed.append(path.name)
    except Exception as exc:
        failed.append(f"{path.name}: {exc}")
if failed:
    raise SystemExit("parse failures: " + ", ".join(failed))
print(f"skypilot parse OK ({len(list(root.glob('*.yaml')))} files)")
PY
    FAILED=1
  fi

  echo "--- [4/6] smoke: all workflow YAMLs ---"
  if ! "${PY}" -m pytest npa/tests/smoke/test_all_workflow_yamls.py -q --timeout=180; then
    FAILED=1
  fi

  echo "--- [5/6] npa.workflow unit + live infra ---"
  if ! "${PY}" -m pytest \
    npa/tests/orchestration/npa_workflow/ \
    npa/tests/smoke/test_npa_workflow_smoke.py \
    npa/tests/e2e/test_npa_workflow_live_e2e.py \
    npa/tests/e2e/test_npa_workflow_live_infra.py \
    -q --timeout=300; then
    FAILED=1
  fi

  echo "--- [6/6] SkyPilot workflow + BDD100K + guardrails ---"
  if ! "${PY}" -m pytest \
    npa/tests/workflows/test_bdd100k_pipeline.py \
    npa/tests/workflows/test_vlm_eval_workflow.py \
    npa/tests/workflows/test_token_factory_workflow.py \
    npa/tests/test_lancedb_bdd100k_import.py \
    npa/tests/test_lancedb_bdd100k_backfill.py \
    npa/tests/test_lancedb_bdd100k_mv.py \
    npa/tests/guardrails/test_workflow_image_check.py \
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
