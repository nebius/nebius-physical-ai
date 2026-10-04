#!/usr/bin/env bash
# Capture the ordinary native Rerun viewer, without the startup screenshot path.
#
# Three things this deliberately does not do, each for a reason that cost time once:
#
#   * No `--screenshot-to`. That path asks for an immediate viewport resize during App::new
#     and is what emits the 40000x40000 surface error. Ordinary viewing is the thing under
#     test, so it has to be ordinary.
#   * No `pkill -f`. A full-command-line pattern matches the launcher running it, which is
#     what silently killed an earlier attempt's own X server. Only the retained PID is
#     signalled, and xvfb-run reaps the server it owns.
#   * No suppression or cropping. The full window is captured, banners and all.
#
# The X screen is deliberately larger than the window so the pointer, which starts at screen
# centre, lands outside the window: hovering the mesh raises a tooltip that covers both panes.
# The capture is still the whole window, not a crop of it.
#
# Usage: capture_ordinary_viewer.sh <recording.rrd> <out.png> <status.json> [WxH]

set -uo pipefail

RECORDING="$1"
FRAME="$2"
STATUS="$3"
GEOMETRY="${4:-1024x768}"
WIDTH="${GEOMETRY%x*}"
HEIGHT="${GEOMETRY#*x}"
RERUN="${RERUN_BIN:-rerun}"
LOG="$(mktemp)"
# The inner shell is a separate process under xvfb-run, so these have to be exported
# rather than left as shell variables.
export RECORDING FRAME RERUN LOG GEOMETRY

# A screen wide enough that the pointer's start position is off the window.
SCREEN_W=$((WIDTH * 2 + 200))
SCREEN_H=$((HEIGHT * 2 + 200))

xvfb-run -a --server-args="-screen 0 ${SCREEN_W}x${SCREEN_H}x24" bash -c '
set -uo pipefail
viewer_pid=""
cleanup() {
  # Graceful first, and only ever the PID this script started.
  if [ -n "$viewer_pid" ] && kill -0 "$viewer_pid" 2>/dev/null; then
    kill -TERM "$viewer_pid" 2>/dev/null
    for _ in $(seq 20); do kill -0 "$viewer_pid" 2>/dev/null || break; sleep 0.25; done
    kill -0 "$viewer_pid" 2>/dev/null && kill -KILL "$viewer_pid" 2>/dev/null
  fi
}
trap cleanup EXIT

"$RERUN" --window-size "$GEOMETRY" "$RECORDING" >"$LOG" 2>&1 &
viewer_pid=$!

# Wait for the window to actually exist rather than sleeping a guessed interval.
window=""
for _ in $(seq 60); do
  window=$(xwininfo -root -children 2>/dev/null | grep -i "Rerun Viewer" | head -1 || true)
  [ -n "$window" ] && break
  sleep 0.5
done
[ -z "$window" ] && { echo "NO_WINDOW"; exit 3; }
echo "window: $window"

# Then give it time to load the recording and settle its startup banners.
sleep 22

# Grab the window at the size it actually is, not the size that was requested. The viewer
# does not always honour --window-size under a bare X server, and grabbing the requested
# size instead produced a crop of a larger window -- which is not a capture of the full UI.
actual=$(xwininfo -root -children 2>/dev/null | grep -i "Rerun Viewer" | head -1 \
  | grep -oE "[0-9]+x[0-9]+\+[0-9]+\+[0-9]+" | head -1)
size="${actual%%+*}"
origin="${actual#*+}"
echo "requested: $GEOMETRY  actual: $size at +${origin}"

ffmpeg -loglevel error -f x11grab -draw_mouse 0 \
  -video_size "$size" -i "$DISPLAY+${origin/+/,}" -frames:v 1 -y "$FRAME"
echo "ffmpeg_rc=$?"
echo "captured_size=$size"

# The viewer is still running here; its exit status is recorded after cleanup, separately,
# because a post-capture shutdown error is not a failure of the capture.
kill -TERM "$viewer_pid" 2>/dev/null
wait "$viewer_pid" 2>/dev/null
echo "viewer_rc=$?"
' 2>&1 | tee /tmp/capture-outer.log

RC=$?

python3 - "$FRAME" "$STATUS" "$LOG" "$RC" <<'PY'
import hashlib, json, pathlib, sys
frame, status, log, rc = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3]), sys.argv[4]
outer = pathlib.Path("/tmp/capture-outer.log").read_text() if pathlib.Path("/tmp/capture-outer.log").exists() else ""
viewer_log = log.read_text() if log.exists() else ""
record = {
    "frame": frame.name,
    "frame_sha256": hashlib.sha256(frame.read_bytes()).hexdigest() if frame.exists() else None,
    "frame_exists": frame.exists(),
    "capture_mechanism": "ffmpeg x11grab of the whole window, no --screenshot-to",
    "outer_status": rc,
    "harness_log": outer.strip().splitlines(),
    # Kept whole and unfiltered: the software-rasterizer notice and any shutdown noise are
    # part of the honest record, and the shutdown lines are not capture failures.
    "viewer_log": viewer_log.strip().splitlines(),
}
status.write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps({"frame_sha256": record["frame_sha256"], "outer_status": rc}, indent=2))
PY
