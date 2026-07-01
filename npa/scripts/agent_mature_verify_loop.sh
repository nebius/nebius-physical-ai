#!/usr/bin/env bash
# Deploy agent fixes and loop until live verification passes.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export NPA_AGENT_PROJECT="${NPA_AGENT_PROJECT:-us-central1}"
export NPA_AGENT_NAME="${NPA_AGENT_NAME:-agent}"
export NPA_SSH_KEY="${NPA_SSH_KEY:-$HOME/.ssh/id_ed25519}"
SLEEP_SECONDS="${NPA_LOOP_SLEEP_SECONDS:-60}"
MAX_ATTEMPTS="${NPA_MAX_ATTEMPTS:-0}"

agent_public_url() {
  "${ROOT}/npa/.venv/bin/npa" agent status --project "$NPA_AGENT_PROJECT" --name "$NPA_AGENT_NAME" --json 2>/dev/null \
    | "${ROOT}/npa/.venv/bin/python" -c 'import json,sys
try:
    print(json.load(sys.stdin).get("public_url","").rstrip("/"))
except Exception:
    print("")'
}

run_success() {
  npa/.venv/bin/pip install -e npa -q || return 1
  if ! NPA_SSH_KEY="$NPA_SSH_KEY" npa/.venv/bin/npa agent bootstrap --project "$NPA_AGENT_PROJECT" --name "$NPA_AGENT_NAME"; then
    echo "bootstrap failed; attempting deploy for ${NPA_AGENT_PROJECT}/${NPA_AGENT_NAME}..."
    NPA_SSH_KEY="$NPA_SSH_KEY" npa/.venv/bin/npa agent deploy --project "$NPA_AGENT_PROJECT" --name "$NPA_AGENT_NAME" || return 1
  fi
  NPA_AGENT_CHAT_LIVE=1 NPA_SSH_KEY="$NPA_SSH_KEY" npa/.venv/bin/npa agent verify-live --project "$NPA_AGENT_PROJECT" --name "$NPA_AGENT_NAME" || return 1
  bash npa/scripts/verify_agent_franka.sh || return 1

  local auth_env="${NPA_AGENT_AUTH_ENV:-$HOME/.npa/agents/${NPA_AGENT_PROJECT}/${NPA_AGENT_NAME}/auth.env}"
  if [[ ! -f "$auth_env" ]]; then
    echo "missing auth env: $auth_env"
    return 1
  fi
  # shellcheck disable=SC1090
  source "$auth_env" || return 1
  if [[ -z "${AGENT_USER:-}" || -z "${AGENT_PASSWORD:-}" ]]; then
    echo "auth env missing AGENT_USER or AGENT_PASSWORD"
    return 1
  fi

  local base
  base="$(agent_public_url)"
  if [[ -z "$base" ]]; then
    echo "could not resolve agent public_url"
    return 1
  fi

  curl -sk -u "${AGENT_USER}:${AGENT_PASSWORD}" -X POST "${base}/api/chat" \
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
  if [[ "$MAX_ATTEMPTS" -gt 0 && "$attempt" -ge "$MAX_ATTEMPTS" ]]; then
    echo "reached NPA_MAX_ATTEMPTS=${MAX_ATTEMPTS}; exiting with failure"
    exit 1
  fi
  echo "FAILED attempt ${attempt}; sleeping ${SLEEP_SECONDS}s..."
  sleep "$SLEEP_SECONDS"
done
