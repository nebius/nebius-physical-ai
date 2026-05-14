#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${NPA_PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
  if [ -x /isaac-sim/python.sh ]; then
    PYTHON_BIN=/isaac-sim/python.sh
  elif [ -x /opt/isaac-lab/venv/bin/python ]; then
    PYTHON_BIN=/opt/isaac-lab/venv/bin/python
  else
    PYTHON_BIN=python3
  fi
fi

if [ "$#" -gt 0 ]; then
  case "$1" in
    python|python3)
      shift
      exec "$PYTHON_BIN" "$@"
      ;;
  esac
fi

MODE="${1:-${SONIC_MODE:-smoke}}"
if [ "$#" -gt 0 ]; then
  shift
fi

OUTPUT_DIR="${NPA_LOCAL_OUTPUT_DIR:-/tmp/npa-sonic-output}"
mkdir -p "$OUTPUT_DIR"

upload_outputs() {
  if [ -z "${NPA_OUTPUT_PATH:-}" ]; then
    return 0
  fi
  "$PYTHON_BIN" <<'PYUPLOAD'
import os
import pathlib
from urllib.parse import urlparse

import boto3

output_path = os.environ["NPA_OUTPUT_PATH"]
parsed = urlparse(output_path)
if parsed.scheme != "s3" or not parsed.netloc:
    raise SystemExit(f"invalid NPA_OUTPUT_PATH: {output_path}")
prefix = parsed.path.lstrip("/")
if prefix and not prefix.endswith("/"):
    prefix += "/"

local_dir = pathlib.Path(os.environ.get("NPA_LOCAL_OUTPUT_DIR", "/tmp/npa-sonic-output"))
s3 = boto3.client("s3", endpoint_url=os.environ.get("AWS_ENDPOINT_URL"))
for path in local_dir.rglob("*"):
    if path.is_file():
        key = prefix + str(path.relative_to(local_dir))
        s3.upload_file(str(path), parsed.netloc, key)
        print(f"uploaded s3://{parsed.netloc}/{key}", flush=True)
PYUPLOAD
}

write_smoke_summary() {
  local command="$1"
  "$PYTHON_BIN" <<PY
import importlib
import json
import os
import pathlib
import time

out = pathlib.Path(${OUTPUT_DIR@Q})
out.mkdir(parents=True, exist_ok=True)

def import_state(name):
    try:
        importlib.import_module(name)
        return "available"
    except Exception as exc:
        return f"unavailable: {type(exc).__name__}: {exc}"

summary = {
    "status": "success",
    "tool": "sonic",
    "command": ${command@Q},
    "embodiment": os.environ.get("SONIC_EMBODIMENT", "UNITREE_G1_SONIC"),
    "checkpoint": os.environ.get("SONIC_CHECKPOINT", "nvidia/GEAR-SONIC:sonic_release/last.pt"),
    "data_path": os.environ.get("SONIC_DATA_PATH", ""),
    "sample_data": os.environ.get("SONIC_SAMPLE_DATA", "1") == "1",
    "num_envs": int(os.environ.get("SONIC_NUM_ENVS", "16")),
    "steps": int(os.environ.get("SONIC_STEPS", os.environ.get("SONIC_MAX_ITERATIONS", "5"))),
    "gear_sonic_import": import_state("gear_sonic"),
    "isaaclab_import": import_state("isaaclab"),
    "sonic_import_alias": import_state("sonic"),
    "timestamp": int(time.time()),
}
(out / "sonic_smoke_result.json").write_text(json.dumps(summary, indent=2))
(out / "sonic_train_summary.json").write_text(json.dumps(summary, indent=2))
print("NPA_SONIC_CONTAINER_SMOKE_DONE", out / "sonic_smoke_result.json", flush=True)
PY
}

download_sample_data() {
  if [ "${SONIC_DOWNLOAD_SAMPLE_DATA:-0}" != "1" ]; then
    return 0
  fi
  local token_arg=()
  if [ -n "${HF_TOKEN:-}" ]; then
    token_arg=(--token "$HF_TOKEN")
  fi
  "$PYTHON_BIN" /opt/sonic/download_from_hf.py --sample "${token_arg[@]}"
}

case "$MODE" in
  smoke)
    write_smoke_summary smoke
    upload_outputs
    ;;
  train)
    download_sample_data
    if [ "${SONIC_RUN_REAL_TRAIN:-0}" = "1" ]; then
      cd /opt/sonic
      "$PYTHON_BIN" -m accelerate.commands.launch \
        --num_processes="${SONIC_NUM_PROCESSES:-1}" \
        gear_sonic/train_agent_trl.py \
        +exp=manager/universal_token/all_modes/sonic_release \
        "+checkpoint=${SONIC_CHECKPOINT_PATH:-sonic_release/last.pt}" \
        "num_envs=${SONIC_NUM_ENVS:-16}" \
        "headless=${SONIC_HEADLESS:-True}" \
        "++algo.config.num_learning_iterations=${SONIC_MAX_ITERATIONS:-5}"
    fi
    write_smoke_summary train
    upload_outputs
    ;;
  serve)
    if [ "${SONIC_SMOKE:-1}" = "1" ]; then
      write_smoke_summary serve
      upload_outputs
      exit 0
    fi
    cd /opt/sonic/gear_sonic_deploy
    exec ./deploy.sh "${SONIC_DEPLOY_TARGET:-sim}" "$@"
    ;;
  *)
    exec "$MODE" "$@"
    ;;
esac
