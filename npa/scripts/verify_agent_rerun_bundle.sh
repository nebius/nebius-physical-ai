#!/usr/bin/env bash
# Live gate: Rerun viewer bundle must load promptly (no deferred/lazy tab stall).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PROJECT="${NPA_AGENT_PROJECT:-rtxpro}"
NAME="${NPA_AGENT_NAME:-agent}"
AUTH_ENV="${NPA_AGENT_AUTH_ENV:-$HOME/.npa/agents/${PROJECT}/${NAME}/auth.env}"

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

exec "${ROOT}/npa/.venv/bin/python" -m npa.agent_rerun_bundle_check \
  --base "${BASE}" \
  --user "${AGENT_USER}" \
  --password "${AGENT_PASSWORD}" \
  --insecure
