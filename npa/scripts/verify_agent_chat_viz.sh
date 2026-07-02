#!/usr/bin/env bash
# Verify grounded chat + Franka/Rerun visualization on a live NPA agent.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PROJECT="${NPA_AGENT_PROJECT:-rtxpro}"
NAME="${NPA_AGENT_NAME:-agent}"
AUTH_ENV="${NPA_AGENT_AUTH_ENV:-$HOME/.npa/agents/${PROJECT}/${NAME}/auth.env}"
CHECK_APIS_USED="${NPA_AGENT_CHECK_APIS_USED:-1}"

bash "${ROOT}/npa/scripts/verify_agent_franka.sh"

if [[ ! -f "$AUTH_ENV" ]]; then
  echo "missing auth env: $AUTH_ENV" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$AUTH_ENV"

BASE="$("${ROOT}/npa/.venv/bin/npa" agent status --project "$PROJECT" --name "$NAME" --json 2>/dev/null \
  | "${ROOT}/npa/.venv/bin/python" -c "import json,sys; print(json.load(sys.stdin).get('public_url','').rstrip('/'))")"
if [[ -z "$BASE" ]]; then
  echo "could not resolve public_url from npa agent status" >&2
  exit 1
fi

curl -skf -u "${AGENT_USER}:${AGENT_PASSWORD}" \
  -X POST "${BASE}/api/chat" \
  -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"what is the current sim2real status"}]}' \
  | NPA_AGENT_CHECK_APIS_USED="${CHECK_APIS_USED}" "${ROOT}/npa/.venv/bin/python" -c "
import json, os, sys
payload = json.load(sys.stdin)
reply = str(payload.get('reply') or '')
if not payload.get('ok'):
    raise SystemExit('chat status smoke: ok!=true')
if not payload.get('grounded'):
    raise SystemExit('chat status smoke: expected grounded=true')
if 'run_id' not in reply and 'stage' not in reply:
    raise SystemExit('chat status smoke: reply missing run_id/stage')
if reply.strip().startswith('GET /api'):
    raise SystemExit('chat status smoke: raw GET path in reply')
if os.environ.get('NPA_AGENT_CHECK_APIS_USED', '1') == '1':
    apis = payload.get('apis_used')
    if not isinstance(apis, list) or not apis:
        raise SystemExit('chat status smoke: expected non-empty apis_used list')
print('chat_status_ok')
"

curl -skf -u "${AGENT_USER}:${AGENT_PASSWORD}" \
  -X POST "${BASE}/api/chat" \
  -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"add an open source repo, containerize, push to registry, and run LeIsaac on live infra"}]}' \
  | NPA_AGENT_CHECK_APIS_USED="${CHECK_APIS_USED}" "${ROOT}/npa/.venv/bin/python" -c "
import json, os, sys
payload = json.load(sys.stdin)
reply = str(payload.get('reply') or '')
if not payload.get('ok'):
    raise SystemExit('onboard_solution smoke: ok!=true')
if not payload.get('grounded'):
    raise SystemExit('onboard_solution smoke: expected grounded=true')
if 'run_byof_repo.py' not in reply:
    raise SystemExit('onboard_solution smoke: missing BYOF runner command')
if '--base-profile' not in reply and '--base-image' not in reply:
    raise SystemExit('onboard_solution smoke: missing base image guidance')
if '<repo-url>' not in reply or '<task>' not in reply:
    raise SystemExit('onboard_solution smoke: missing runnable placeholders')
if reply.strip().startswith('GET /api'):
    raise SystemExit('onboard_solution smoke: raw GET path in reply')
if os.environ.get('NPA_AGENT_CHECK_APIS_USED', '1') == '1':
    apis = payload.get('apis_used')
    if not isinstance(apis, list) or 'tools' not in apis:
        raise SystemExit('onboard_solution smoke: expected tools in apis_used')
print('chat_onboard_solution_ok')
"

RERUN_CODE="$(curl -sk -o /dev/null -w '%{http_code}' -u "${AGENT_USER}:${AGENT_PASSWORD}" "${BASE}/rerun/")"
if [[ "${RERUN_CODE}" != "200" ]]; then
  echo "rerun iframe root unhealthy: HTTP ${RERUN_CODE}" >&2
  exit 1
fi

STATIC_OK=0
for path in /rerun/index.js /rerun/re_viewer.js /rerun/favicon.ico /rerun/version; do
  code="$(curl -sk -o /dev/null -w '%{http_code}' -u "${AGENT_USER}:${AGENT_PASSWORD}" "${BASE}${path}")"
  if [[ "${code}" == "200" ]]; then
    STATIC_OK=1
    break
  fi
done
if [[ "${STATIC_OK}" -ne 1 ]]; then
  echo "no rerun static asset responded 200" >&2
  exit 1
fi

echo "verify_agent_chat_viz: ok (${BASE})"
