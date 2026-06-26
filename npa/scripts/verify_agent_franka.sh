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

READY=0
for _attempt in $(seq 1 20); do
  if curl -skf -u "${AGENT_USER}:${AGENT_PASSWORD}" "${BASE}/api/sim-viz/status" \
    | "${ROOT}/npa/.venv/bin/python" -c 'import json,sys; p=json.load(sys.stdin); sys.exit(0 if p.get("rerun_ready") and str(p.get("rrd_uri","")).strip() else 1)'; then
    READY=1
    break
  fi
  sleep 1
done
if [[ "${READY}" -ne 1 ]]; then
  echo "sim-viz status did not reach rerun_ready=true with non-empty rrd_uri" >&2
  exit 1
fi

BYTES="$(curl -skf -u "${AGENT_USER}:${AGENT_PASSWORD}" "${BASE}/api/sim-viz/rrd-blob" | wc -c | tr -d ' ')"
if [[ "${BYTES}" -lt 64 ]]; then
  echo "rrd too small: ${BYTES} bytes" >&2
  exit 1
fi

REC_BYTES="$(curl -skf "${BASE}/rerun/recordings/sim2real.rrd" | wc -c | tr -d ' ')"
if [[ "${REC_BYTES}" -lt 64 ]]; then
  echo "public recording too small: ${REC_BYTES} bytes" >&2
  exit 1
fi

echo "verify_agent_franka: ok (${BYTES} byte rrd, ${REC_BYTES} byte public recording at ${BASE})"
