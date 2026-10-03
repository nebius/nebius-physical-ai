#!/usr/bin/env bash
# Boot the real guarded service, generate a video, and validate the artifact.
set -euo pipefail

ROOT="${NPA_LOCAL_OUTPUT_DIR:-/tmp/npa-cosmos3-serving-output}"
PORT="${NPA_COSMOS3_SERVE_PORT:-8000}"
READY_URL="http://127.0.0.1:${PORT}/v1/models"
VIDEO_URL="http://127.0.0.1:${PORT}/v1/videos/sync"
LOG="${ROOT}/server.log"
VIDEO="${ROOT}/cosmos3_serving_smoke.mp4"
PROBE="${ROOT}/ffprobe.json"
RESULT="${ROOT}/cosmos3_serving_smoke.json"
mkdir -p "${ROOT}"

/opt/npa-cosmos3-serving/entrypoint.sh >"${LOG}" 2>&1 &
server_pid=$!
cleanup() {
  kill "${server_pid}" 2>/dev/null || true
  wait "${server_pid}" 2>/dev/null || true
}
trap cleanup EXIT

while ! curl --fail --silent --show-error "${READY_URL}" >/dev/null; do
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    echo "[npa-cosmos3-serving] server exited before readiness" >&2
    tail -200 "${LOG}" >&2
    exit 1
  fi
  sleep 10
done

curl --fail --silent --show-error -X POST "${VIDEO_URL}" \
  -H "Accept: video/mp4" \
  --form-string "prompt=${NPA_COSMOS3_SMOKE_PROMPT:-A small red robot rolls across a clean studio floor.}" \
  --form-string "size=${NPA_COSMOS3_SMOKE_SIZE:-640x360}" \
  --form-string "num_frames=${NPA_COSMOS3_SMOKE_FRAMES:-93}" \
  --form-string "fps=${NPA_COSMOS3_SMOKE_FPS:-24}" \
  --form-string "num_inference_steps=${NPA_COSMOS3_SMOKE_STEPS:-8}" \
  --form-string "guidance_scale=${NPA_COSMOS3_SMOKE_GUIDANCE:-6.0}" \
  --form-string "flow_shift=${NPA_COSMOS3_SMOKE_FLOW_SHIFT:-10.0}" \
  --form-string "max_sequence_length=${NPA_COSMOS3_SMOKE_MAX_SEQUENCE_LENGTH:-4096}" \
  --form-string "seed=${NPA_COSMOS3_SMOKE_SEED:-17}" \
  -o "${VIDEO}"

ffprobe -v error -select_streams v:0 \
  -show_entries stream=codec_name,width,height,nb_frames \
  -of json "${VIDEO}" >"${PROBE}"

NPA_SMOKE_VIDEO="${VIDEO}" NPA_SMOKE_PROBE="${PROBE}" NPA_SMOKE_RESULT="${RESULT}" \
  python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

video = Path(os.environ["NPA_SMOKE_VIDEO"])
probe = json.loads(Path(os.environ["NPA_SMOKE_PROBE"]).read_text())
streams = probe.get("streams") or []
if not streams or streams[0].get("codec_name") != "h264":
    raise SystemExit(f"expected a decoded H.264 video stream, got {streams!r}")
if video.stat().st_size <= 1024:
    raise SystemExit(f"generated video is implausibly small: {video.stat().st_size}")
result = {
    "format": "npa_cosmos3_serving_smoke_v1",
    "status": "completed",
    "guardrails": os.environ.get("NPA_COSMOS3_SERVE_GUARDRAILS", "on"),
    "gpu_count": int(os.environ.get("NPA_COSMOS3_SERVE_GPUS", "8")),
    "image_digest": os.environ.get("NPA_IMAGE_DIGEST", ""),
    "video": {
        "path": str(video),
        "bytes": video.stat().st_size,
        "sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
        "stream": streams[0],
    },
}
if result["guardrails"] != "on" or result["gpu_count"] != 8:
    raise SystemExit("release smoke requires guardrails=on and exactly 8 GPUs")
Path(os.environ["NPA_SMOKE_RESULT"]).write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n"
)
print("NPA_COSMOS3_SERVING_SMOKE_OK", os.environ["NPA_SMOKE_RESULT"])
PY
