#!/usr/bin/env bash
# Deploy agent fixes and loop until live verification passes.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export NPA_AGENT_PROJECT="${NPA_AGENT_PROJECT:-rtxpro}"
export NPA_SSH_KEY="${NPA_SSH_KEY:-$HOME/.ssh/id_ed25519}"

run_success() {
  npa/.venv/bin/pip install -e npa -q
  NPA_SSH_KEY="$NPA_SSH_KEY" npa/.venv/bin/npa agent bootstrap --project "$NPA_AGENT_PROJECT" --name agent
  NPA_AGENT_CHAT_LIVE=1 NPA_SSH_KEY="$NPA_SSH_KEY" npa/.venv/bin/npa agent verify-live --project "$NPA_AGENT_PROJECT" --name agent
  bash npa/scripts/verify_agent_franka.sh
  # shellcheck disable=SC1090
  source "${NPA_AGENT_AUTH_ENV:-$HOME/.npa/agents/${NPA_AGENT_PROJECT}/agent/auth.env}"
  BASE="$("${ROOT}/npa/.venv/bin/npa" agent status --project "$NPA_AGENT_PROJECT" --name agent --json 2>/dev/null \
  | "${ROOT}/npa/.venv/bin/python" -c "import json,sys; print(json.load(sys.stdin).get('public_url','').rstrip('/'))")"
  curl -sk -u "${AGENT_USER}:${AGENT_PASSWORD}" -X POST "${BASE}/api/chat" \
    -H 'content-type: application/json' \
    -d '{"messages":[{"role":"user","content":"what is the current sim2real status"}]}' \
    | "${ROOT}/npa/.venv/bin/python" -c "
import json,sys
r=json.load(sys.stdin)
t=r.get('reply','')
assert r.get('grounded'), 'expected grounded chat'
assert 'run_id' in t or 'stage' in t, 'missing run_id/stage'
assert not t.strip().startswith('GET /api'), 'raw GET path in reply'
assert 'apis_used' in r, 'missing apis_used'
print('chat_status_ok')"
}

attempt=0
while true; do
  attempt=$((attempt + 1))
  echo "=== attempt ${attempt} $(date -Is) ==="
  if run_success; then
    echo "SUCCESS at attempt ${attempt} $(date -Is)"
    exit 0
  fi
  echo "FAILED attempt ${attempt}; sleeping 60s..."
  sleep 60
done
