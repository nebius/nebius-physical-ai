#!/usr/bin/env bash
# Full npa.workflow live-infra matrix: all golden YAMLs, real S3, no credential leakage.
# tmux new -s npa-workflow-real-infra ./scripts/npa-workflow-real-infra-tmux.sh
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
DYNAMIC="sim2real-vlm-rl.yaml tokenfactory-cosmos-gate.yaml"

LOG="/tmp/npa-workflow-real-infra-$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== npa.workflow FULL LIVE INFRA matrix log=${LOG} ==="
echo "branch: $(git branch --show-current) @ $(git rev-parse --short HEAD)"
echo "golden specs: $(find "${SPECS}" -maxdepth 1 -name '*.yaml' | wc -l)"

echo "--- S3 preflight (project credentials, not stale AWS_*) ---"
"${PY}" - <<'PY' || { echo "S3 preflight failed; fix ~/.npa/credentials.yaml or unset stale AWS_* in tmux server"; exit 1; }
import sys
from pathlib import Path

sys.path.insert(0, str(Path("npa/tests/e2e")))
from npa_workflow_live_helpers import live_bucket
from npa.clients.project_credentials import s3_client_for_project

bucket = live_bucket(None)
client = s3_client_for_project(None)
key = "npa-workflow-e2e/tmux-preflight.txt"
client.put_object(Bucket=bucket, Key=key, Body=b"ok\n")
print(f"S3 preflight OK bucket={bucket}")
PY

round=1
while true; do
  echo ""
  echo "========== ROUND ${round} $(date -u +%Y-%m-%dT%H:%M:%SZ) =========="
  FAILED=0

  echo "--- [1/6] unit + smoke (all golden YAMLs) ---"
  if ! "${PY}" -m pytest \
    npa/tests/orchestration/npa_workflow/ \
    npa/tests/smoke/test_npa_workflow_smoke.py \
    npa/tests/smoke/test_all_workflow_yamls.py \
    -q --timeout=180; then
    FAILED=1
  fi

  echo "--- [2/6] live e2e (all golden + leak checks) ---"
  if ! "${PY}" -m pytest npa/tests/e2e/test_npa_workflow_live_e2e.py -q --timeout=300; then
    FAILED=1
  fi

  echo "--- [3/6] live infra (S3 persist, all golden on real bucket, leak checks) ---"
  if ! "${PY}" -m pytest npa/tests/e2e/test_npa_workflow_live_infra.py -q --timeout=600; then
    FAILED=1
  fi

  echo "--- [4/6] CLI golden matrix on live bucket (validate/plan/scheduler) ---"
  RUN_ID="tmux-live-r${round}-$(date -u +%H%M%S)"
  BUCKET="$("${PY}" - <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, str(Path("npa/tests/e2e")))
from npa_workflow_live_helpers import live_bucket

print(live_bucket(None))
PY
)"
  for spec in "${SPECS}"/*.yaml; do
    base=$(basename "$spec")
    stem="${base%.yaml}"
    echo "live CLI: ${base} bucket=${BUCKET}"
    LIVE_SPEC="$("${PY}" - "${spec}" "${BUCKET}" "${RUN_ID}" "${stem}" <<'PY'
import sys
from pathlib import Path
import re

spec_path, bucket, run_id, stem = sys.argv[1:5]
text = Path(spec_path).read_text(encoding="utf-8")
text = text.replace("bucket: example-bucket", f"bucket: {bucket}")
text = re.sub(
    r'(prefix:\s*")([^"]*)(")',
    lambda m: f'{m.group(1)}npa-workflow-e2e/{run_id}/{stem}{m.group(3)}',
    text,
    count=1,
)
out = Path(f"/tmp/npa-live-{stem}-{run_id}.yaml")
out.write_text(text, encoding="utf-8")
print(out)
PY
)"
    if ! "${NPA}" workbench workflow validate-spec "${LIVE_SPEC}"; then
      FAILED=1
      continue
    fi
    extra=()
    if [[ " ${DYNAMIC} " == *" ${base} "* ]]; then
      extra=(--assume-decision promote_checkpoint)
    fi
    if ! "${NPA}" workbench workflow plan-spec "${LIVE_SPEC}" \
      --run-id "${RUN_ID}-${stem}" "${extra[@]}"; then
      FAILED=1
      continue
    fi
    if [[ " ${DYNAMIC} " == *" ${base} "* ]]; then
      if ! "${NPA}" workbench workflow plan-spec "${LIVE_SPEC}" \
        --run-id "${RUN_ID}-${stem}-loop" \
        --assume-decision loop_back \
        --json | "${PY}" -c "
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path('npa/tests/e2e')))
from npa_workflow_live_helpers import assert_no_credential_leakage, live_credential_markers
raw = sys.stdin.read()
assert_no_credential_leakage(raw, extra_forbidden=live_credential_markers())
p = json.loads(raw)
assert p.get('steps'), p
print('loop_back steps', len(p['steps']))
"; then
        FAILED=1
      fi
    fi
    if ! "${NPA}" workbench workflow run-spec "${LIVE_SPEC}" \
      --run-id "${RUN_ID}-${stem}" \
      --plan-only \
      --scheduler-plan \
      "${extra[@]}" \
      --json | "${PY}" -c "
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path('npa/tests/e2e')))
from npa_workflow_live_helpers import assert_no_credential_leakage, live_credential_markers
raw = sys.stdin.read()
assert_no_credential_leakage(raw, extra_forbidden=live_credential_markers())
r = json.loads(raw)
assert r.get('plan', {}).get('steps'), r
assert r.get('scheduler', {}).get('tasks'), r
print('scheduler tasks', len(r['scheduler']['tasks']))
"; then
      FAILED=1
    fi
  done

  echo "--- [5/6] audit fixes + skills guardrail ---"
  if ! "${PY}" -m pytest \
    npa/tests/orchestration/npa_workflow/test_audit_fixes.py \
    npa/tests/guardrails/test_skills_index.py \
    -q --timeout=180; then
    FAILED=1
  fi

  echo "--- [6/6] BDD100K + skypilot parse ---"
  if ! "${PY}" -m pytest npa/tests/workflows/test_bdd100k_pipeline.py -q --timeout=120; then
    FAILED=1
  fi
  if ! "${PY}" - <<'PY'; then
from pathlib import Path
import yaml

root = Path("npa/src/npa/workflows/skypilot")
for path in sorted(root.glob("*.yaml")):
    docs = [d for d in yaml.safe_load_all(path.read_text(encoding="utf-8")) if d is not None]
    assert docs and docs[0].get("name"), path.name
print(f"skypilot parse OK ({len(list(root.glob('*.yaml')))} files)")
PY
    FAILED=1
  fi

  if [[ "${FAILED}" -eq 0 ]]; then
    echo "--- ROUND ${round} ALL PASSED (full live infra, all golden YAMLs, leak-checked) ---"
  else
    echo "--- ROUND ${round} FAILED ---"
  fi
  round=$((round + 1))
  sleep 600
done
