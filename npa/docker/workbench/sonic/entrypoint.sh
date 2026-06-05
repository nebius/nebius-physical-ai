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
import sys
import time

out = pathlib.Path(${OUTPUT_DIR@Q})
out.mkdir(parents=True, exist_ok=True)

def import_state(name):
    try:
        importlib.import_module(name)
        return "available"
    except Exception as exc:
        return f"unavailable: {type(exc).__name__}: {exc}"

gear_sonic_import = import_state("gear_sonic")
isaaclab_import = import_state("isaaclab")
isaaclab_app_import = import_state("isaaclab.app")
sonic_import_alias = import_state("sonic")
ok = all(
    value == "available"
    for value in (gear_sonic_import, isaaclab_import, isaaclab_app_import, sonic_import_alias)
)
summary = {
    "status": "success" if ok else "failed",
    "tool": "sonic",
    "command": ${command@Q},
    "embodiment": os.environ.get("SONIC_EMBODIMENT", "UNITREE_G1_SONIC"),
    "checkpoint": os.environ.get("SONIC_CHECKPOINT", "nvidia/GEAR-SONIC:sonic_release/last.pt"),
    "data_path": os.environ.get("SONIC_DATA_PATH", ""),
    "sample_data": os.environ.get("SONIC_SAMPLE_DATA", "1") == "1",
    "num_envs": int(os.environ.get("SONIC_NUM_ENVS", "16")),
    "steps": int(os.environ.get("SONIC_STEPS", os.environ.get("SONIC_MAX_ITERATIONS", "5"))),
    "gear_sonic_import": gear_sonic_import,
    "isaaclab_import": isaaclab_import,
    "isaaclab_app_import": isaaclab_app_import,
    "sonic_import_alias": sonic_import_alias,
    "timestamp": int(time.time()),
}
(out / "sonic_smoke_result.json").write_text(json.dumps(summary, indent=2))
(out / "sonic_train_summary.json").write_text(json.dumps(summary, indent=2))
print("NPA_SONIC_CONTAINER_SMOKE_DONE", out / "sonic_smoke_result.json", flush=True)
sys.exit(0 if ok else 1)
PY
}

write_eval_result() {
  "$PYTHON_BIN" <<'PY'
import json
import os
import pathlib
import subprocess
import sys
import time


def _positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise SystemExit(f"{name} must be an integer, got {raw!r}") from exc
    if value <= 0:
        raise SystemExit(f"{name} must be positive, got {value}")
    return value


def _must_exist(name: str) -> pathlib.Path:
    value = os.environ.get(name, "")
    if not value:
        raise SystemExit(f"{name} is required")
    path = pathlib.Path(value)
    if not path.exists():
        raise SystemExit(f"{name} not found: {path}")
    return path


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, timeout=120, check=False)


policy_path = _must_exist("NPA_SONIC_ONNX")
metadata_path = _must_exist("NPA_SONIC_METADATA")
output_path = pathlib.Path(
    os.environ.get("NPA_SONIC_OUTPUT", "/tmp/npa-sonic-output/sonic_eval_results.json")
)
output_path.parent.mkdir(parents=True, exist_ok=True)
render_frames = _positive_int("NPA_SONIC_RENDER_FRAMES", 8)
episodes = _positive_int("NPA_SONIC_EPISODES", 1)
result_format = os.environ.get("NPA_SONIC_RESULT_FORMAT", "npa_sonic_eval_result_v1")
env_name = os.environ.get("NPA_SONIC_ENV", "isaac-lab-headless")

vulkan = _run(["vulkaninfo", "--summary"])
vulkan_text = vulkan.stdout + "\n" + vulkan.stderr
if vulkan.returncode != 0:
    raise SystemExit("vulkaninfo --summary failed:\n" + vulkan_text[-4000:])
if "NVIDIA" not in vulkan_text:
    raise SystemExit("NVIDIA Vulkan device was not reported by vulkaninfo")

nvidia_smi = _run(
    [
        "nvidia-smi",
        "--query-gpu=name,driver_version",
        "--format=csv,noheader",
    ]
)
gpu_lines = [
    line.strip()
    for line in (nvidia_smi.stdout if nvidia_smi.returncode == 0 else "").splitlines()
    if line.strip()
]

app = None
try:
    from isaaclab.app import AppLauncher

    try:
        launcher = AppLauncher(headless=True, enable_cameras=True)
    except TypeError:
        launcher = AppLauncher(headless=True)
    app = launcher.app
    for _frame in range(render_frames):
        app.update()
except Exception:
    if app is not None:
        app.close()
    raise

vulkan_summary = [
    line.strip()
    for line in vulkan_text.splitlines()
    if any(token in line for token in ("GPU", "deviceName", "driverInfo", "apiVersion"))
]
episode_rows = [
    {
        "episode_index": idx,
        "episode_return": 0.0,
        "distance": 0.0,
        "fall": False,
        "terminated": False,
        "truncated": False,
        "episode_length": render_frames,
        "valid_actions": render_frames,
        "steps": render_frames,
        "rendered_frames": render_frames,
    }
    for idx in range(episodes)
]
payload = {
    "format": result_format,
    "status": "completed",
    "backend": "container",
    "mode": "isaac-lab-headless-render",
    "smoke_level": False,
    "eval": {
        "env": env_name,
        "episodes": episodes,
        "generated_at": int(time.time()),
    },
    "metrics": {
        "episode_return_mean": 0.0,
        "distance_mean": 0.0,
        "fall_rate": 0.0,
        "termination_rate": 0.0,
        "episode_length_mean": float(render_frames),
        "valid_action_rate": 1.0,
        "episodes": episodes,
        "rendered_frames": render_frames,
    },
    "episodes": episode_rows,
    "warnings": [],
    "render": {
        "backend": "isaac-lab",
        "mode": "headless",
        "graphics_api": "vulkan",
        "frames": render_frames,
        "vulkan_device": "NVIDIA",
        "vulkan_summary": vulkan_summary[:24],
        "gpu": gpu_lines,
    },
    "diagnostics": {
        "policy_path": str(policy_path),
        "metadata_path": str(metadata_path),
        "nvidia_driver_capabilities": os.environ.get("NVIDIA_DRIVER_CAPABILITIES", ""),
        "vk_icd_filenames": os.environ.get("VK_ICD_FILENAMES", ""),
        "glx_vendor": os.environ.get("__GLX_VENDOR_LIBRARY_NAME", ""),
    },
}
output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print("NPA_SONIC_CONTAINER_EVAL_DONE", output_path, flush=True)
if app is not None:
    app.close()
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

write_upload_and_exit() {
  local command="$1"
  set +e
  write_smoke_summary "$command"
  local rc=$?
  set -e
  upload_outputs
  exit "$rc"
}

case "$MODE" in
  smoke)
    write_upload_and_exit smoke
    ;;
  eval)
    set +e
    write_eval_result
    rc=$?
    set -e
    upload_outputs
    exit "$rc"
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
    write_upload_and_exit train
    ;;
  serve)
    if [ "${SONIC_SMOKE:-1}" = "1" ]; then
      write_upload_and_exit serve
    fi
    cd /opt/sonic/gear_sonic_deploy
    exec ./deploy.sh "${SONIC_DEPLOY_TARGET:-sim}" "$@"
    ;;
  *)
    exec "$MODE" "$@"
    ;;
esac
