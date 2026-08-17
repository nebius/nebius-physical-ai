#!/usr/bin/env bash
# Real Cosmos Transfer 2.5 golden evaluation using only a procedural input.
set -euo pipefail

REPO="${COSMOS_TRANSFER_REPO:-/opt/cosmos/cosmos-transfer2.5}"
PY="${REPO}/.venv/bin/python"
GENERATOR="/opt/cosmos2-transfer/generate_fixture.py"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN is required at run time for the gated Cosmos Transfer weights; no download was attempted." >&2
  exit 78
fi
[[ -x "${PY}" ]] || { echo "ERROR: baked Cosmos Transfer venv is unavailable" >&2; exit 1; }
"${PY}" -c 'import flash_attn, torch; assert torch.cuda.is_available()'

if [[ -n "${COSMOS_TRANSFER_WORKDIR:-}" ]]; then
  WORKDIR="${COSMOS_TRANSFER_WORKDIR}"
  mkdir -p "${WORKDIR}"
else
  WORKDIR="$(mktemp -d /tmp/npa-cosmos2-golden.XXXXXX)"
fi
FIXTURE_DIR="${WORKDIR}/fixture"
OUT="${COSMOS_TRANSFER_OUT:-${WORKDIR}/output}"
mkdir -p "${OUT}"

fixture_json="$("${PY}" "${GENERATOR}" "${FIXTURE_DIR}" --num-steps 4)"
SPEC="${FIXTURE_DIR}/npa-procedural-edge-spec.json"
[[ -s "${SPEC}" ]] || { echo "ERROR: procedural spec was not generated" >&2; exit 1; }

GPU_SAMPLES="${WORKDIR}/gpu-memory-mib.txt"
monitor_pid=""
if command -v nvidia-smi >/dev/null 2>&1; then
  (
    while true; do
      nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits \
        2>/dev/null || true
      sleep 1
    done
  ) >"${GPU_SAMPLES}" &
  monitor_pid="$!"
fi
stop_monitor() {
  if [[ -n "${monitor_pid}" ]]; then
    kill "${monitor_pid}" >/dev/null 2>&1 || true
    wait "${monitor_pid}" >/dev/null 2>&1 || true
  fi
}
trap stop_monitor EXIT

started_ns="$(date +%s%N)"
echo "[run] real Cosmos Transfer 2.5 inference (4 diffusion steps)"
cd "${REPO}"
"${PY}" examples/inference.py -i "${SPEC}" -o "${OUT}"
finished_ns="$(date +%s%N)"
stop_monitor
monitor_pid=""

gpu_name=""
if command -v nvidia-smi >/dev/null 2>&1; then
  gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
fi

"${PY}" - "${OUT}" "${started_ns}" "${finished_ns}" "${GPU_SAMPLES}" \
  "${gpu_name}" "${fixture_json}" <<'PY'
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from npa.workbench.cosmos.transfer import _classify_output_videos

out = Path(sys.argv[1])
started_ns, finished_ns = int(sys.argv[2]), int(sys.argv[3])
gpu_samples = Path(sys.argv[4])
gpu_name = sys.argv[5]
fixture = json.loads(sys.argv[6])
generated, _controls, _masks = _classify_output_videos(out)
videos = sorted(
    (Path(p) for p in generated if Path(p).stat().st_size > 100_000),
    key=lambda p: p.stat().st_size,
    reverse=True,
)
if not videos:
    raise SystemExit(f"no classified generated, non-trivial output MP4 under {out}")
video = videos[0]
probe = json.loads(
    subprocess.run(
        [
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,nb_read_frames,duration:format=duration",
            "-of", "json", str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
)
stream = probe["streams"][0]
width, height = int(stream["width"]), int(stream["height"])
frames = int(stream.get("nb_read_frames") or 0)
duration = float(stream.get("duration") or probe.get("format", {}).get("duration") or 0)
if (width, height) != (fixture["width"], fixture["height"]):
    raise SystemExit(f"unexpected output dimensions: {width}x{height}")
if frames <= 0 or duration <= 0:
    raise SystemExit(f"output is not decodable: frames={frames} duration={duration}")

peak_mib = None
if gpu_samples.is_file():
    values = []
    for line in gpu_samples.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            values.append(int(line.strip()))
        except ValueError:
            pass
    if values:
        peak_mib = max(values)

metrics = {
    "schema": "npa.cosmos2.golden_eval.v1",
    "inference": "examples/inference.py",
    "diffusion_steps": fixture["num_steps"],
    "fixture_provenance": fixture["provenance"],
    "gpu_type": gpu_name,
    "peak_gpu_memory_mib": peak_mib,
    "wall_seconds": round((finished_ns - started_ns) / 1_000_000_000, 3),
    "output_path": str(video),
    "output_bytes": video.stat().st_size,
    "output_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
    "width": width,
    "height": height,
    "frame_count": frames,
    "duration_seconds": duration,
}
metrics_path = out / "npa-golden-eval-metrics.json"
metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(metrics, sort_keys=True))
PY

echo "[PASS] real multi-step Cosmos Transfer output validated under ${OUT}"
