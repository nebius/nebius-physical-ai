#!/bin/sh
#
# Golden eval for npa-foxglove-embed: prove the container actually does its job.
#
# Checks, against a live server (started here when the script is not run inside an
# already-serving container):
#   1. /healthz reports this service and the pinned SDK version
#   2. /sdk/index.js is the real @foxglove/embed ESM entry point
#   3. /app/npa-foxglove-host.js is the shared NPA glue module
#   4. / is the standalone host page and it imports that glue module
#   5. /data/* honors HTTP Range (byte-exact partial read) — required for MCAP playback
#   6. /data/* answers the CORS preflight that a Range request triggers
#
# Uses only busybox tools present in the runtime image.
set -eu

PORT="${FOXGLOVE_SMOKE_PORT:-8099}"
BASE="http://127.0.0.1:${PORT}"
DATA_DIR="${FOXGLOVE_DATA_DIR:-/srv/data}"
PROBE="${DATA_DIR}/npa-range-probe.bin"
CADDY_PID=""

log() { echo "npa-foxglove-smoke: $*"; }
fail() { echo "npa-foxglove-smoke: FAIL: $*" >&2; exit 1; }

cleanup() {
  [ -n "$CADDY_PID" ] && kill "$CADDY_PID" 2>/dev/null || true
  rm -f "$PROBE" /tmp/npa-fg-part /tmp/npa-fg-body
}
trap cleanup EXIT

wait_healthy() {
  i=0
  while [ "$i" -lt "${1:-20}" ]; do
    if wget -q -O /dev/null "${BASE}/healthz" 2>/dev/null; then
      return 0
    fi
    i=$((i + 1))
    sleep 0.5
  done
  return 1
}

if ! wait_healthy 2; then
  log "no server on ${PORT} yet; starting caddy"
  caddy run --config /etc/caddy/Caddyfile --adapter caddyfile >/tmp/npa-fg-caddy.log 2>&1 &
  CADDY_PID=$!
  wait_healthy 30 || fail "server did not become healthy: $(cat /tmp/npa-fg-caddy.log 2>/dev/null)"
fi

# 1. health
wget -q -O /tmp/npa-fg-body "${BASE}/healthz" || fail "/healthz request failed"
grep -q 'npa-foxglove-embed' /tmp/npa-fg-body || fail "/healthz did not identify the service"
grep -q '@foxglove/embed@' /tmp/npa-fg-body || fail "/healthz did not report the SDK version"
log "health ok: $(cat /tmp/npa-fg-body)"

# 2. real SDK entry point
wget -q -O /tmp/npa-fg-body "${BASE}/sdk/index.js" || fail "/sdk/index.js request failed"
grep -q './FoxgloveViewer.js' /tmp/npa-fg-body || fail "/sdk/index.js is not the @foxglove/embed entry point"
wget -q -O /tmp/npa-fg-body "${BASE}/sdk/FoxgloveViewer.js" || fail "/sdk/FoxgloveViewer.js request failed"
grep -q 'class FoxgloveViewer' /tmp/npa-fg-body || fail "FoxgloveViewer.js does not define the viewer class"
grep -q 'foxglove-handshake-request' /tmp/npa-fg-body || fail "FoxgloveViewer.js is missing the embed handshake"
log "sdk ok: @foxglove/embed served with the FoxgloveViewer class and handshake"

# 3. shared glue module
wget -q -O /tmp/npa-fg-body "${BASE}/app/npa-foxglove-host.js" || fail "glue module request failed"
grep -q 'mountFoxgloveViewer' /tmp/npa-fg-body || fail "glue module is missing mountFoxgloveViewer"
grep -q '../sdk/index.js' /tmp/npa-fg-body || fail "glue module does not import the served SDK"
log "glue ok"

# 4. standalone host page
wget -q -O /tmp/npa-fg-body "${BASE}/" || fail "host page request failed"
grep -q './app/npa-foxglove-host.js' /tmp/npa-fg-body || fail "host page does not load the glue module"
log "host page ok"

# 5. byte-range read (MCAP playback requirement)
#    busybox wget rejects 206 responses, so the range check speaks raw HTTP.
mkdir -p "$DATA_DIR" 2>/dev/null || true
printf '\211MCAP0\r\nRANGE-PROBE-PAYLOAD' > "$PROBE" 2>/dev/null \
  || fail "cannot write range probe into ${DATA_DIR} (mount it writable for the smoke)"
printf 'GET /data/%s HTTP/1.1\r\nHost: 127.0.0.1\r\nRange: bytes=0-7\r\nConnection: close\r\n\r\n' \
  "$(basename "$PROBE")" | nc 127.0.0.1 "$PORT" > /tmp/npa-fg-part || fail "range request failed"
grep -q '206' /tmp/npa-fg-part || fail "range request did not return 206 Partial Content"
grep -qi 'content-range: bytes 0-7/' /tmp/npa-fg-part || fail "range response is missing Content-Range"
grep -qi 'content-length: 8' /tmp/npa-fg-part || fail "range response did not return exactly 8 bytes"
grep -qi 'accept-ranges: bytes' /tmp/npa-fg-part || fail "server does not advertise byte ranges"
log "range ok: 206 Partial Content with an exact 8-byte Content-Range"

# 6. CORS preflight for the Range header
printf 'OPTIONS /data/%s HTTP/1.1\r\nHost: 127.0.0.1\r\nOrigin: https://embed.foxglove.dev\r\nAccess-Control-Request-Method: GET\r\nAccess-Control-Request-Headers: range\r\nConnection: close\r\n\r\n' \
  "$(basename "$PROBE")" | nc 127.0.0.1 "$PORT" > /tmp/npa-fg-body || fail "preflight request failed"
grep -qi 'access-control-allow-origin' /tmp/npa-fg-body || fail "preflight is missing Access-Control-Allow-Origin"
grep -qi 'access-control-allow-headers:.*[Rr]ange' /tmp/npa-fg-body || fail "preflight does not allow the Range header"
log "cors preflight ok"

log "PASS"
