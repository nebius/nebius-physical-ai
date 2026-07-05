#!/usr/bin/env bash
# npa-driven destroy → fresh-setup → smoke chat loop for agent VMs.
# See skills/workflows/agent-fresh-operate/SKILL.md
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

NPA_BIN="${ROOT}/npa/.venv/bin/npa"
PY_BIN="${ROOT}/npa/.venv/bin/python"

export NPA_AGENT_PROJECT="${NPA_AGENT_PROJECT:-fresh-us-central1}"
export NPA_AGENT_NAME="${NPA_AGENT_NAME:-agent}"
export NPA_AGENT_REGION="${NPA_AGENT_REGION:-us-central1}"
export NPA_SSH_KEY="${NPA_SSH_KEY:-$HOME/.ssh/id_ed25519}"
export NPA_NEBIUS_PROFILE="${NPA_NEBIUS_PROFILE:-npa-mk8s}"

NPA_FRESH_SETUP_SKIP_DESTROY="${NPA_FRESH_SETUP_SKIP_DESTROY:-0}"
NPA_FRESH_SETUP_SKIP_DEPLOY="${NPA_FRESH_SETUP_SKIP_DEPLOY:-0}"
NPA_FRESH_SETUP_RUN_VERIFY_LIVE="${NPA_FRESH_SETUP_RUN_VERIFY_LIVE:-0}"
NPA_FRESH_SETUP_RUN_GROUNDED_CHAT="${NPA_FRESH_SETUP_RUN_GROUNDED_CHAT:-1}"
SLEEP_SECONDS="${NPA_LOOP_SLEEP_SECONDS:-60}"
MAX_ATTEMPTS="${NPA_MAX_ATTEMPTS:-0}"

agent_public_url() {
  "${NPA_BIN}" agent status --project "$NPA_AGENT_PROJECT" --name "$NPA_AGENT_NAME" --json 2>/dev/null \
    | "${PY_BIN}" -c 'import json,sys
try:
    print(json.load(sys.stdin).get("public_url","").rstrip("/"))
except Exception:
    print("")'
}

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "missing required env: $name" >&2
    exit 1
  fi
}

activate_nebius_profile() {
  if command -v nebius >/dev/null 2>&1; then
    nebius profile activate "$NPA_NEBIUS_PROFILE" >/dev/null 2>&1 || true
  fi
}

run_destroy() {
  if [[ "$NPA_FRESH_SETUP_SKIP_DESTROY" == "1" ]]; then
    echo "skip destroy (NPA_FRESH_SETUP_SKIP_DESTROY=1)"
    return 0
  fi
  echo "=== destroy ${NPA_AGENT_PROJECT}/${NPA_AGENT_NAME} $(date -Is) ==="
  activate_nebius_profile
  NPA_NEBIUS_PROFILE="$NPA_NEBIUS_PROFILE" \
    NPA_SSH_KEY="$NPA_SSH_KEY" \
    "${NPA_BIN}" agent destroy --project "$NPA_AGENT_PROJECT" --name "$NPA_AGENT_NAME"
}

run_fresh_setup() {
  if [[ "$NPA_FRESH_SETUP_SKIP_DEPLOY" == "1" ]]; then
    echo "skip deploy (NPA_FRESH_SETUP_SKIP_DEPLOY=1)"
    return 0
  fi
  require_env NPA_AGENT_PROJECT_ID
  require_env NPA_AGENT_TENANT_ID
  echo "=== fresh-setup ${NPA_AGENT_PROJECT}/${NPA_AGENT_NAME} $(date -Is) ==="
  activate_nebius_profile
  NPA_NEBIUS_PROFILE="$NPA_NEBIUS_PROFILE" \
    NPA_SSH_KEY="$NPA_SSH_KEY" \
    "${NPA_BIN}" agent fresh-setup \
      --project "$NPA_AGENT_PROJECT" \
      --name "$NPA_AGENT_NAME" \
      --project-id "$NPA_AGENT_PROJECT_ID" \
      --tenant-id "$NPA_AGENT_TENANT_ID" \
      --region "$NPA_AGENT_REGION"
}

run_smoke_verify() {
  local auth_env="${NPA_AGENT_AUTH_ENV:-$HOME/.npa/agents/${NPA_AGENT_PROJECT}/${NPA_AGENT_NAME}/auth.env}"
  if [[ ! -f "$auth_env" ]]; then
    echo "missing auth env: $auth_env" >&2
    return 1
  fi
  # shellcheck disable=SC1090
  source "$auth_env"

  local base
  base="$(agent_public_url)"
  if [[ -z "$base" ]]; then
    echo "could not resolve agent public_url" >&2
    return 1
  fi

  echo "=== smoke verify ${base} $(date -Is) ==="
  curl -sfk -u "${AGENT_USER}:${AGENT_PASSWORD}" "${base}/api/models" \
    | "${PY_BIN}" -c "import json,sys; d=json.load(sys.stdin); assert d.get('ok'), d; print('models_ok', len(d.get('models',[])))"

  curl -sfk -u "${AGENT_USER}:${AGENT_PASSWORD}" \
    -H 'content-type: application/json' \
    -d '{"messages":[{"role":"user","content":"Say hello in one short sentence."}]}' \
    "${base}/api/chat" \
    | "${PY_BIN}" -c "import json,sys; r=json.load(sys.stdin); assert r.get('ok'), r; print('hello_chat_ok')"

  if [[ "$NPA_FRESH_SETUP_RUN_GROUNDED_CHAT" == "1" ]]; then
    curl -sfk -u "${AGENT_USER}:${AGENT_PASSWORD}" \
      -H 'content-type: application/json' \
      -d '{"messages":[{"role":"user","content":"what is the current sim2real status"}]}' \
      "${base}/api/chat" \
      | "${PY_BIN}" -c "
import json,sys
r=json.load(sys.stdin)
t=r.get('reply','')
assert r.get('grounded'), 'expected grounded chat'
assert 'run_id' in t or 'stage' in t, 'missing run_id/stage'
assert not t.strip().startswith('GET /api'), 'raw GET path in reply'
print('grounded_chat_ok')"
  fi

  if [[ "$NPA_FRESH_SETUP_RUN_VERIFY_LIVE" == "1" ]]; then
    NPA_AGENT_CHAT_LIVE=1 NPA_SSH_KEY="$NPA_SSH_KEY" \
      "${NPA_BIN}" agent verify-live --project "$NPA_AGENT_PROJECT" --name "$NPA_AGENT_NAME"
  fi
}

run_success() {
  "${PY_BIN}" -m pip install -e npa -q >/dev/null 2>&1 || return 1
  run_destroy || return 1
  run_fresh_setup || return 1
  run_smoke_verify || return 1
  echo "=== LOOP SUCCESS $(date -Is) ==="
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
    echo "reached NPA_MAX_ATTEMPTS=${MAX_ATTEMPTS}; exiting with failure" >&2
    exit 1
  fi
  echo "FAILED attempt ${attempt}; sleeping ${SLEEP_SECONDS}s..."
  sleep "$SLEEP_SECONDS"
done
