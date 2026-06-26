#!/usr/bin/env bash
# Loop until NPA agent workflow YAML feature passes unit + live checks.
# tmux new -s npa-workflow-yaml-loop ./npa/scripts/npa-workflow-yaml-loop.sh
set -euo pipefail
set -o pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export NPA_SSH_KEY="${NPA_SSH_KEY:-$HOME/.ssh/id_ed25519}"
export NPA_AGENT_PROJECT="${NPA_AGENT_PROJECT:-rtxpro}"
LOG="/tmp/npa-workflow-yaml-loop.log"
EXAMPLE_YAML="${ROOT}/npa/workflows/workbench/npa-workflows/sim2real-two-step-agent.yaml"
ATTEMPT=0

exec > >(tee -a "$LOG") 2>&1
echo "=== npa-workflow-yaml-loop log=${LOG} branch=$(git rev-parse --abbrev-ref HEAD) ==="

run_iteration() {
  ATTEMPT=$((ATTEMPT + 1))
  echo ""
  echo "=== attempt ${ATTEMPT} $(date -Is) ==="

  npa/.venv/bin/pip install -e npa -q

  npa/.venv/bin/python -c "
from npa.cli.agent_workflow import generate_sim2real_two_step_yaml
from pathlib import Path
p = Path('${EXAMPLE_YAML}')
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(generate_sim2real_two_step_yaml(), encoding='utf-8')
print('wrote', p)
"

  echo "--- pytest agent workflow ---"
  npa/.venv/bin/python -m pytest \
    npa/tests/cli/test_agent.py \
    npa/tests/cli/test_agent_workflow.py \
    -q

  echo "--- validate-spec + plan-spec ---"
  npa/.venv/bin/npa workbench workflow validate-spec "$EXAMPLE_YAML"
  npa/.venv/bin/npa workbench workflow plan-spec "$EXAMPLE_YAML" --run-id agent-loop-demo

  echo "--- bootstrap ---"
  NPA_SSH_KEY="$NPA_SSH_KEY" npa/.venv/bin/npa agent bootstrap \
    --project "$NPA_AGENT_PROJECT" --name agent

  echo "--- verify-live ---"
  NPA_AGENT_CHAT_LIVE=1 NPA_SSH_KEY="$NPA_SSH_KEY" npa/.venv/bin/npa agent verify-live \
    --project "$NPA_AGENT_PROJECT" --name agent

  echo "--- workflow yaml live smoke ---"
  # shellcheck disable=SC1090
  source "${NPA_AGENT_AUTH_ENV:-$HOME/.npa/agents/${NPA_AGENT_PROJECT}/agent/auth.env}"
  BASE="$("${ROOT}/npa/.venv/bin/npa" agent status --project "$NPA_AGENT_PROJECT" --name agent --json 2>/dev/null \
    | "${ROOT}/npa/.venv/bin/python" -c "import json,sys; print(json.load(sys.stdin).get('public_url','').rstrip('/'))")"
  export AGENT_BASE="$BASE"
  export AGENT_USER AGENT_PASSWORD

  YAML_TEXT="$(cat "$EXAMPLE_YAML")"
  AGENT_BASE="$BASE" AGENT_USER="$AGENT_USER" AGENT_PASSWORD="$AGENT_PASSWORD" \
    EXAMPLE_YAML="$EXAMPLE_YAML" "${ROOT}/npa/.venv/bin/python" - <<'PY'
import json
import os
import httpx

base = os.environ["AGENT_BASE"]
auth = (os.environ["AGENT_USER"], os.environ["AGENT_PASSWORD"])
yaml_path = os.environ["EXAMPLE_YAML"]
yaml_text = open(yaml_path, encoding="utf-8").read()

chat = httpx.post(
    f"{base}/api/chat",
    auth=auth,
    json={"messages": [{"role": "user", "content": "create 2-step sim2real workflow"}]},
    timeout=60.0,
    verify=False,
)
chat.raise_for_status()
payload = chat.json()
assert payload.get("grounded"), payload
assert payload.get("workflow_yaml"), "chat missing workflow_yaml"
assert "augment" in payload["workflow_yaml"]
assert "envgen" in payload["workflow_yaml"]
print("chat_create_workflow_ok")

draft = httpx.put(
    f"{base}/api/workflows/npa/draft",
    auth=auth,
    json={"yaml": yaml_text},
    timeout=30.0,
    verify=False,
)
draft.raise_for_status()
assert draft.json().get("validation", {}).get("ok"), draft.text
print("draft_upload_ok")

validate = httpx.post(
    f"{base}/api/workflows/npa/validate",
    auth=auth,
    json={"yaml": yaml_text},
    timeout=30.0,
    verify=False,
)
validate.raise_for_status()
assert validate.json().get("ok"), validate.text
print("validate_ok")

plan = httpx.post(
    f"{base}/api/workflows/npa/plan",
    auth=auth,
    json={"yaml": yaml_text, "run_id": "live-smoke"},
    timeout=30.0,
    verify=False,
)
plan.raise_for_status()
steps = plan.json().get("plan", {}).get("steps") or []
assert len(steps) >= 2, plan.text
print("plan_ok")

submit = httpx.post(
    f"{base}/api/workflows/npa/submit",
    auth=auth,
    json={"yaml": yaml_text},
    timeout=30.0,
    verify=False,
)
submit.raise_for_status()
assert submit.json().get("run_id"), submit.text
print("submit_ok")
PY

  COMMIT_HASH="$(git rev-parse HEAD)"
  echo "SUCCESS attempt=${ATTEMPT} commit=${COMMIT_HASH} example=${EXAMPLE_YAML}"
}

while true; do
  if run_iteration; then
    exit 0
  fi
  echo "FAILED attempt ${ATTEMPT}; commit+push if dirty, sleep 60, retry..."
  if ! git diff --quiet || ! git diff --cached --quiet || [ -n "$(git ls-files --others --exclude-standard)" ]; then
    git add -A
    git commit -m "$(cat <<EOF
fix(agent): workflow YAML loop iteration ${ATTEMPT}

Automated npa-workflow-yaml-loop retry.
EOF
)" || true
    git push origin HEAD || true
  fi
  sleep 60
done
