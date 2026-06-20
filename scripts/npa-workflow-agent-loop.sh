#!/usr/bin/env bash
# Iterate on npa.workflow gaps: test → commit → push (for PR #131).
# tmux new -s npa-workflow-agent ./scripts/npa-workflow-agent-loop.sh
set -euo pipefail

export HOME="${HOME:-/home/ubuntu}"
REPO="/home/ubuntu/nebius-physical-ai"
cd "$REPO"
PY="${REPO}/npa/.venv/bin/python"
LOG="/tmp/npa-workflow-agent-$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== npa-workflow agent loop log=${LOG} ==="

while true; do
  echo ""
  echo "--- $(date -u +%Y-%m-%dT%H:%M:%SZ) test ---"
  "${PY}" -m pytest \
    npa/tests/orchestration/npa_workflow/ \
    npa/tests/smoke/test_npa_workflow_smoke.py \
    -q --timeout=120

  if NPA_INTEGRATION_E2E=1 "${PY}" -m pytest npa/tests/e2e/test_npa_workflow_live_e2e.py -q --timeout=120; then
    echo "live e2e OK"
  else
    echo "live e2e failed (continuing)"
  fi

  if git diff --quiet && git diff --cached --quiet; then
    echo "no changes to commit"
  else
    git add \
      npa/src/npa/orchestration/npa_workflow/ \
      npa/src/npa/cli/workbench/workflow/__init__.py \
      npa/tests/orchestration/npa_workflow/ \
      npa/tests/e2e/test_npa_workflow_live_e2e.py \
      npa/workflows/workbench/npa-workflows/ \
      docs/workbench/npa-workflow-guide.md \
      docs/workbench/npa-workflow-tool-catalog.md \
      skills/workflows/author-npa-workflow/ \
      scripts/npa-workflow-agent-loop.sh \
      scripts/npa-workflow-live-e2e.sh 2>/dev/null || true
    if git diff --cached --quiet; then
      echo "nothing staged"
    else
      msg="npa.workflow: $(git diff --cached --stat | tail -1)"
      git commit -m "$(cat <<EOF
${msg}

Automated agent-loop commit on feat/npa-workflow-v0.0.1.
EOF
)"
      git push origin feat/npa-workflow-v0.0.1
      echo "pushed to origin/feat/npa-workflow-v0.0.1"
    fi
  fi

  echo "--- sleeping 600s ---"
  sleep 600
done
