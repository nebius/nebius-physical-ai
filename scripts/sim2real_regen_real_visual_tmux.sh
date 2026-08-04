#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-s2r-real-0725t222636z}"
S3_BUCKET="${S3_BUCKET:-}"
S3_PREFIX="${S3_PREFIX:-sim2real-b}"
S3_ENDPOINT="${S3_ENDPOINT:-https://storage.us-central1.nebius.cloud}"
LOCAL_DIR="${LOCAL_DIR:-/tmp/sim2real-regen/${RUN_ID}-real-visual}"
LOG_DIR="${LOG_DIR:-/tmp/sim2real-visual-audit/${RUN_ID}}"
AGENT_URL="${AGENT_URL:-}"
AGENT_USER="${AGENT_USER:-npa}"
AGENT_PASSWORD="${AGENT_PASSWORD:-}"

mkdir -p "$LOG_DIR"

if [[ -f "$HOME/.npa/sim2real-operator.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$HOME/.npa/sim2real-operator.env"
  set +a
fi

export PYTHONPATH="$PWD/npa/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ -z "$S3_BUCKET" ]]; then
  S3_BUCKET="$(
    npa/.venv/bin/python - <<'PY'
from npa.workflows.sim2real.monitor import load_operator_config

print(load_operator_config().bucket)
PY
  )"
fi
if [[ -z "$S3_BUCKET" ]]; then
  echo "S3_BUCKET is required or must be configured in ~/.npa/config.yaml" >&2
  exit 2
fi

echo "started_at=$(date -Iseconds)"
echo "run_id=$RUN_ID"
echo "local_dir=$LOCAL_DIR"
echo "log_dir=$LOG_DIR"

echo "== targeted tests =="
npa/.venv/bin/python -m pytest \
  npa/tests/workflows/test_sim2real_viz.py \
  npa/tests/workflows/test_sim2real_stages.py \
  npa/tests/workflows/test_sim2real_rerun_regen.py \
  npa/tests/workflows/test_sim2real_rerun_serve.py \
  -q

echo "== regenerate and upload =="
npa/.venv/bin/npa workbench sim2real rerun regen \
  --run-id "$RUN_ID" \
  --s3-bucket "$S3_BUCKET" \
  --s3-prefix "$S3_PREFIX" \
  --s3-endpoint "$S3_ENDPOINT" \
  --local-dir "$LOCAL_DIR" \
  --upload \
  --output json | tee "$LOG_DIR/regen-real-visual.json"

RRD_URI="$(
  npa/.venv/bin/python - "$LOG_DIR/regen-real-visual.json" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload.get("upload_uri") or "")
PY
)"

if [[ -z "$RRD_URI" ]]; then
  echo "ERROR: regen did not return upload_uri" >&2
  exit 1
fi

echo "rrd_uri=$RRD_URI"
echo "== inspect regenerated evidence =="
npa/.venv/bin/python - "$LOG_DIR/regen-real-visual.json" "$LOCAL_DIR" <<'PY' | tee "$LOG_DIR/regen-real-visual-summary.txt"
import json
import sys
from pathlib import Path

regen = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
local_dir = Path(sys.argv[2])
report = json.loads((local_dir / "eval/heldout/report.json").read_text(encoding="utf-8"))
evidence_paths = sorted((local_dir / "inner_loop").glob("outer-*/evidence.json"))
latest = evidence_paths[-1] if evidence_paths else None
evidence = json.loads(latest.read_text(encoding="utf-8")) if latest else {}
print("local_rrd_path", regen.get("local_rrd_path"))
print("heldout_frame_count", regen.get("heldout_frame_count"))
print("synthetic_frame_count", regen.get("synthetic_frame_count"))
print("rollout_count", regen.get("rollout_count"))
print("frame_count", regen.get("frame_count"))
print("rrd_size_bytes", Path(regen["local_rrd_path"]).stat().st_size)
print("success_rate", report.get("success_rate"))
print("render_episodes", len((report.get("render_manifest") or {}).get("episodes") or []))
print("latest_inner_evidence", latest)
visual_index_path = local_dir / "reports/sim2real-visual-index.json"
print("visual_index_path", visual_index_path)
if visual_index_path.is_file():
    visual_index = json.loads(visual_index_path.read_text(encoding="utf-8"))
    print("visual_success_decision", (visual_index.get("success") or {}).get("decision"))
    print("visual_augmentation_frames", (visual_index.get("augmentation") or {}).get("frame_count"))
    print("visual_train_envs", (visual_index.get("dataset") or {}).get("train_count"))
    print("visual_heldout_envs", (visual_index.get("dataset") or {}).get("heldout_count"))
    synthetic = visual_index.get("synthetic") or {}
    print("visual_synthetic_dataset_samples", synthetic.get("dataset_sample_count"))
    print("visual_synthetic_dataset_camera_pngs", synthetic.get("dataset_camera_image_count"))
    print("visual_synthetic_dataset_descriptor_previews", synthetic.get("dataset_descriptor_preview_count"))
    print("visual_synthetic_augmentation_samples", synthetic.get("augmentation_sample_count"))
    print("visual_synthetic_augmentation_pngs", synthetic.get("augmentation_image_count"))
    print("visual_synthetic_augmentation_descriptor_previews", synthetic.get("augmentation_descriptor_preview_count"))
for record in (evidence.get("iterations") or [])[:3]:
    print("iteration", record.get("iteration"))
    print("actions_dir", record.get("actions_dir"))
    sample = record.get("sample_vlm_eval") or {}
    print("vlm_model", sample.get("model"))
    print("vlm_score", sample.get("score"))
    print("trainer_status", (record.get("update") or {}).get("status"))
PY

RRD_LOCAL="$(
  npa/.venv/bin/python - "$LOG_DIR/regen-real-visual.json" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload["local_rrd_path"])
PY
)"
for needle in \
  "synthetic" \
  "heldout/camera" \
  "summary" \
  "signal"; do
  if ! grep -a -q "$needle" "$RRD_LOCAL"; then
    echo "ERROR: regenerated RRD missing broad visual token $needle" >&2
    exit 1
  fi
done

npa/.venv/bin/python - "$LOCAL_DIR" <<'PY'
import json
import sys
from pathlib import Path

local_dir = Path(sys.argv[1])
index = json.loads((local_dir / "reports/sim2real-visual-index.json").read_text(encoding="utf-8"))
success = index.get("success") or {}
synthetic = index.get("synthetic") or {}
dataset = index.get("dataset") or {}
success_rate = success.get("success_rate")
threshold = success.get("threshold", 0.5)
if success.get("decision") != "promote_checkpoint":
    raise SystemExit(f"visual index decision is not promote_checkpoint: {success!r}")
if success_rate is None or float(success_rate) < float(threshold):
    raise SystemExit(f"visual index success_rate {success_rate!r} below threshold {threshold!r}")
if int(dataset.get("train_count") or 0) <= 0 or int(dataset.get("heldout_count") or 0) <= 0:
    raise SystemExit(f"visual index has invalid train/heldout split: {dataset!r}")
if int(synthetic.get("dataset_sample_count") or 0) <= 0:
    raise SystemExit(f"visual index has no synthetic dataset samples: {synthetic!r}")
if int(synthetic.get("augmentation_sample_count") or 0) <= 0:
    raise SystemExit(f"visual index has no synthetic augmentation samples: {synthetic!r}")
PY

if [[ -n "$AGENT_URL" && -n "$AGENT_PASSWORD" ]]; then
  echo "== reload agent viewer =="
  curl -fsS \
    -u "${AGENT_USER}:${AGENT_PASSWORD}" \
    -H "Content-Type: application/json" \
    -X POST \
    "${AGENT_URL%/}/api/sim-viz/load-run" \
    -d "{\"run_id\":\"${RUN_ID}\",\"rrd_uri\":\"${RRD_URI}\",\"camera\":\"heldout-sim\"}" \
    | tee "$LOG_DIR/agent-load-run.json"
  echo
  curl -fsS \
    -u "${AGENT_USER}:${AGENT_PASSWORD}" \
    "${AGENT_URL%/}/api/sim-viz/status?run_id=${RUN_ID}" \
    | tee "$LOG_DIR/agent-status.json"
  echo
else
  echo "agent_reload=skipped"
fi

echo "completed_at=$(date -Iseconds)"
