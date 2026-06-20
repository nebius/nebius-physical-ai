#!/usr/bin/env bash
# Full npa.workflow real-infra test matrix (guide runbook + unit + live e2e).
# tmux new -s npa-workflow-real-infra ./scripts/npa-workflow-real-infra-tmux.sh
set -euo pipefail

export HOME="${HOME:-/home/ubuntu}"
# Tmux server sessions inherit stale AWS_* from server startup; clear before loading ~/.npa.
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN
[[ -f /home/ubuntu/bin/npa-cloud-env.sh ]] && source /home/ubuntu/bin/npa-cloud-env.sh

REPO="/home/ubuntu/nebius-physical-ai"
cd "$REPO"

# Always test workflow branch code.
git checkout feat/npa-workflow-v0.0.1 >/dev/null 2>&1 || true

PY="${REPO}/npa/.venv/bin/python"
NPA="${REPO}/npa/.venv/bin/npa"
export NPA_INTEGRATION_E2E=1

# Preflight: confirm project S3 is writable with the same credential path as live tests.
"${PY}" - <<'PY' || { echo "S3 preflight failed; fix ~/.npa/credentials.yaml or unset stale AWS_* in tmux server"; exit 1; }
from urllib.parse import urlparse

from npa.clients.config import resolve_project_storage
from npa.clients.project_credentials import s3_client_for_project

storage = resolve_project_storage(None)
raw = storage.checkpoint_bucket or ""
if not raw:
    raise SystemExit("checkpoint_bucket not configured")
bucket = urlparse(raw if "://" in raw else f"s3://{raw}").netloc or raw.split("/")[0]
client = s3_client_for_project(None)
key = "npa-workflow-e2e/tmux-preflight.txt"
client.put_object(Bucket=bucket, Key=key, Body=b"ok\n")
print(f"S3 preflight OK bucket={bucket}")
PY
LOG="/tmp/npa-workflow-real-infra-$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== npa.workflow REAL INFRA matrix log=${LOG} ==="
echo "branch: $(git branch --show-current) @ $(git rev-parse --short HEAD)"

round=1
FAILED=0
while true; do
  echo ""
  echo "========== ROUND ${round} $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
  FAILED=0

  echo "--- [1/4] unit + smoke ---"
  if ! "${PY}" -m pytest \
    npa/tests/orchestration/npa_workflow/ \
    npa/tests/smoke/test_npa_workflow_smoke.py \
    -q --timeout=120; then
    FAILED=1
  fi

  echo "--- [2/4] live e2e (validate/plan CLI) ---"
  if ! "${PY}" -m pytest npa/tests/e2e/test_npa_workflow_live_e2e.py -q --timeout=180; then
    FAILED=1
  fi

  echo "--- [3/4] live infra runbook (S3 persist, fail manifest, scheduler, require-inputs) ---"
  if ! "${PY}" -m pytest npa/tests/e2e/test_npa_workflow_live_infra.py -q --timeout=300; then
    FAILED=1
  fi

  echo "--- [4/4] guide quickstart CLI (golden YAMLs) ---"
  SPECS="${REPO}/npa/workflows/workbench/npa-workflows"
  RUN_ID="tmux-guide-r${round}-$(date -u +%H%M%S)"
  for spec in "${SPECS}"/*.yaml; do
    base=$(basename "$spec")
    echo "spec: ${base}"
    if ! "${NPA}" workbench workflow validate-spec "${spec}"; then
      FAILED=1
      continue
    fi
    if ! "${NPA}" workbench workflow plan-spec "${spec}" \
      --run-id "${RUN_ID}" \
      --assume-decision promote_checkpoint; then
      FAILED=1
      continue
    fi
    if [[ "${base}" == "sim2real-vlm-rl.yaml" ]]; then
      if ! "${NPA}" workbench workflow plan-spec "${spec}" \
        --run-id "${RUN_ID}-loop" \
        --assume-decision loop_back \
        --json | "${PY}" -c "
import json, sys
p = json.load(sys.stdin)
states = [s['state'] for s in p['steps']]
assert states.count('finalize') == 1, states
assert states.count('rollouts') == 6, states
print('sim2real promote OK: finalize=1 rollouts=6')
"; then
        FAILED=1
      fi
    fi
    if ! "${NPA}" workbench workflow run-spec "${spec}" \
      --run-id "${RUN_ID}" \
      --plan-only \
      --scheduler-plan \
      --json | "${PY}" -c "
import json, sys
r = json.load(sys.stdin)
assert r.get('plan', {}).get('steps'), r
assert r.get('scheduler', {}).get('tasks'), r
print('scheduler tasks', len(r['scheduler']['tasks']))
"; then
      FAILED=1
    fi
  done

  if [[ "${FAILED}" -eq 0 ]]; then
    echo "--- ROUND ${round} ALL PASSED ---"
  else
    echo "--- ROUND ${round} FAILED (see log); retrying after sleep ---"
  fi
  round=$((round + 1))
  sleep 600
done
