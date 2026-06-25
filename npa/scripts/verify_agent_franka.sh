#!/usr/bin/env bash
# Verify stock Franka demo is loadable on a live NPA agent (HTTPS + basic auth).
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

curl -skf -u "${AGENT_USER}:${AGENT_PASSWORD}" \
  -X POST "${BASE}/api/sim-viz/load-franka-demo" \
  -H 'content-type: application/json' \
  -d '{"camera":"workspace"}' | grep -q '"ok"[[:space:]]*:[[:space:]]*true'

curl -skf -u "${AGENT_USER}:${AGENT_PASSWORD}" "${BASE}/api/sim-viz/status" \
  | grep -qE '"rerun_ready"[[:space:]]*:[[:space:]]*true|"rrd_uri"[[:space:]]*:[[:space:]]*"[^"]+"'

BYTES="$(curl -skf -u "${AGENT_USER}:${AGENT_PASSWORD}" "${BASE}/api/sim-viz/rrd" | wc -c | tr -d ' ')"
if [[ "${BYTES}" -lt 64 ]]; then
  echo "rrd too small: ${BYTES} bytes" >&2
  exit 1
fi

echo "verify_agent_franka: ok (${BYTES} byte rrd at ${BASE})"
