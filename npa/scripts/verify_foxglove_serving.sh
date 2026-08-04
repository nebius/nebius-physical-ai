#!/usr/bin/env bash
#
# verify_foxglove_serving.sh - prove the Foxglove serving contract locally.
#
# The embedded Foxglove viewer only works if the *serving* layer is right:
#   * the SDK + glue module are reachable same-origin,
#   * recordings are readable by the cross-origin viewer iframe (no auth, CORS),
#   * byte ranges work (MCAP playback streams with Range requests),
#   * the Range CORS preflight is answered.
#
# This script renders the real agent nginx site (the same string `npa agent
# bootstrap` installs), installs the real pinned SDK, writes a real MCAP with
# `npa workbench foxglove convert-run`, serves it all with nginx in Docker, and
# asserts every one of those properties. No cloud resources are touched.
#
# Usage:
#   bash npa/scripts/verify_foxglove_serving.sh [--port 18088] [--keep]
#
# Requires: docker, python (repo venv), and network access for the SDK download.
set -euo pipefail

PORT="${NPA_FOXGLOVE_VERIFY_PORT:-18088}"
KEEP=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --port) PORT="${2:?--port requires a value}"; shift 2 ;;
    --keep) KEEP=1; shift ;;
    -h|--help) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${NPA_PYTHON:-$ROOT/npa/.venv/bin/python}"
[ -x "$PYTHON" ] || PYTHON="$(command -v python3)"
WORK="$(mktemp -d)"
NAME="npa-foxglove-verify-$$"
FAILURES=0

cleanup() {
  if [ "$KEEP" = "1" ]; then
    echo "keeping container $NAME and workdir $WORK"
    return
  fi
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  rm -rf "$WORK"
}
trap cleanup EXIT

ok()   { echo "  PASS: $*"; }
bad()  { echo "  FAIL: $*" >&2; FAILURES=$((FAILURES + 1)); }
step() { echo; echo "== $*"; }

step "render the agent nginx site from npa.cli.agent"
PYTHONPATH="$ROOT/npa/src" NPA_FOXGLOVE_VERIFY_OUT="$WORK/agent.conf" "$PYTHON" - <<'PY'
import os
from pathlib import Path

from npa.cli.agent import _nginx_agent_site_body

body = _nginx_agent_site_body(backend_port=8787, rerun_port=9090)
# One plain HTTP server block; the HTTPS block only differs by TLS material.
conf = "server {\n  listen 80;\n  server_name _;\n" + body + "\n}\n"
Path(os.environ["NPA_FOXGLOVE_VERIFY_OUT"]).write_text(conf, encoding="utf-8")
print(f"wrote {os.environ['NPA_FOXGLOVE_VERIFY_OUT']}")
PY
grep -q "location /foxglove/data/" "$WORK/agent.conf" || { echo "rendered config lacks the Foxglove data location" >&2; exit 1; }

step "install the pinned @foxglove/embed SDK (sha512 verified)"
bash "$ROOT/npa/docker/workbench/foxglove-embed/install-sdk.sh" --dest "$WORK/foxglove/sdk"
mkdir -p "$WORK/foxglove/app" "$WORK/foxglove/data"
cp "$ROOT/npa/src/npa/cli/assets/foxglove/npa-foxglove-host.js" "$WORK/foxglove/app/"

step "produce a real MCAP recording with npa workbench foxglove convert-run"
mkdir -p "$WORK/run/camera/front" "$WORK/run/reports"
PYTHONPATH="$ROOT/npa/src" "$PYTHON" - <<PY
from pathlib import Path
from PIL import Image

root = Path("$WORK/run")
for index in range(4):
    Image.new("RGB", (48, 32), (40 * index, 80, 160)).save(root / "camera" / "front" / f"{index:04d}.png")
(root / "reports" / "metrics.json").write_text('{"success_rate": 0.75, "episodes": 8}', encoding="utf-8")
(root / "reports" / "run.log").write_text("stage 1 ok\nstage 2 ok\n", encoding="utf-8")
PY
PYTHONPATH="$ROOT/npa/src" "$PYTHON" -m npa.cli.main workbench foxglove convert-run \
  --input-path "$WORK/run" --output-path "$WORK/foxglove/data/verify-run.mcap" --run-id verify-run --fps 4
PYTHONPATH="$ROOT/npa/src" "$PYTHON" -m npa.cli.main workbench foxglove inspect \
  --input-path "$WORK/foxglove/data/verify-run.mcap"
chmod -R a+rX "$WORK/foxglove"

step "serve it with nginx using the rendered agent config"
# htpasswd entry for the authenticated locations (verify-only credentials).
docker run --rm httpd:2.4-alpine htpasswd -nbB npa verify-pass > "$WORK/htpasswd"
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" -p "${PORT}:80" \
  -v "$WORK/agent.conf:/etc/nginx/conf.d/default.conf:ro" \
  -v "$WORK/foxglove:/opt/npa-agent/foxglove:ro" \
  -v "$WORK/htpasswd:/etc/nginx/.npa-agent-htpasswd:ro" \
  nginx:1.27-alpine >/dev/null
sleep 2
docker exec "$NAME" nginx -t >/dev/null 2>&1 && ok "rendered agent nginx config is valid" \
  || { docker exec "$NAME" nginx -t; bad "rendered agent nginx config is invalid"; }

BASE="http://127.0.0.1:${PORT}"
AUTH="npa:verify-pass"

step "SDK + glue module are served same-origin (authenticated)"
if curl -fsS -u "$AUTH" "$BASE/foxglove/sdk/index.js" | grep -q './FoxgloveViewer.js'; then
  ok "/foxglove/sdk/index.js is the real @foxglove/embed entry point"
else
  bad "/foxglove/sdk/index.js did not serve the SDK"
fi
if curl -fsS -u "$AUTH" "$BASE/foxglove/sdk/FoxgloveViewer.js" | grep -q 'foxglove-handshake-request'; then
  ok "FoxgloveViewer.js carries the embed handshake"
else
  bad "FoxgloveViewer.js is missing the embed handshake"
fi
if curl -fsS -u "$AUTH" "$BASE/foxglove/app/npa-foxglove-host.js" | grep -q 'mountFoxgloveViewer'; then
  ok "glue module is served"
else
  bad "glue module is not served"
fi
if [ "$(curl -s -o /dev/null -w '%{http_code}' "$BASE/foxglove/sdk/index.js")" = "401" ]; then
  ok "SDK assets require agent auth"
else
  bad "SDK assets are readable without auth"
fi

step "recordings are readable by the cross-origin viewer iframe"
code="$(curl -s -o /dev/null -w '%{http_code}' "$BASE/foxglove/data/verify-run.mcap")"
[ "$code" = "200" ] && ok "recording is readable without credentials (200)" \
  || bad "recording returned $code without credentials"
headers="$(curl -sD- -o /dev/null "$BASE/foxglove/data/verify-run.mcap")"
echo "$headers" | grep -qi 'access-control-allow-origin: \*' \
  && ok "CORS allows the viewer origin" || bad "missing Access-Control-Allow-Origin"
echo "$headers" | grep -qi 'accept-ranges: bytes' \
  && ok "byte ranges advertised" || bad "missing Accept-Ranges"
echo "$headers" | grep -qi 'cross-origin-resource-policy: cross-origin' \
  && ok "CORP allows cross-origin embedding" || bad "missing Cross-Origin-Resource-Policy"
echo "$headers" | grep -qi 'content-encoding: gzip' \
  && bad "recording is compressed (breaks range playback)" || ok "recording is served uncompressed"

step "byte-range playback"
range="$(curl -sD- -o "$WORK/part.bin" -H 'Range: bytes=0-7' "$BASE/foxglove/data/verify-run.mcap")"
echo "$range" | grep -q '206' && ok "range request returns 206" || bad "range request did not return 206"
echo "$range" | grep -qi 'content-range: bytes 0-7/' && ok "Content-Range is exact" || bad "missing Content-Range"
if [ "$(wc -c < "$WORK/part.bin" | tr -d ' ')" = "8" ]; then
  ok "partial read returned exactly 8 bytes"
else
  bad "partial read returned $(wc -c < "$WORK/part.bin") bytes"
fi
# Those 8 bytes must be the MCAP magic: proves ranges land on the real recording.
if head -c 8 "$WORK/foxglove/data/verify-run.mcap" | cmp -s - "$WORK/part.bin"; then
  ok "partial read matches the MCAP magic prefix"
else
  bad "partial read did not match the file prefix"
fi

step "CORS preflight for the Range header"
pre="$(curl -sD- -o /dev/null -X OPTIONS \
  -H 'Origin: https://embed.foxglove.dev' \
  -H 'Access-Control-Request-Method: GET' \
  -H 'Access-Control-Request-Headers: range' \
  "$BASE/foxglove/data/verify-run.mcap")"
echo "$pre" | grep -q '204' && ok "preflight returns 204" || bad "preflight did not return 204"
echo "$pre" | grep -qi 'access-control-allow-headers:.*[Rr]ange' \
  && ok "preflight allows the Range header" || bad "preflight does not allow Range"

echo
if [ "$FAILURES" -eq 0 ]; then
  echo "verify_foxglove_serving: PASS"
else
  echo "verify_foxglove_serving: FAIL ($FAILURES check(s))" >&2
  exit 1
fi
