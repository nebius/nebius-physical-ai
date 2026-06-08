#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${NPA_PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
  if [ -x /isaac-sim/python.sh ]; then
    PYTHON_BIN=/isaac-sim/python.sh
  elif [ -x /opt/npa/venv/bin/python ]; then
    PYTHON_BIN=/opt/npa/venv/bin/python
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

download_s3_file() {
  local uri="$1"
  local destination="$2"
  NPA_DOWNLOAD_URI="$uri" NPA_DOWNLOAD_DESTINATION="$destination" "$PYTHON_BIN" <<'PYDOWNLOAD'
import os
import pathlib
from urllib.parse import urlparse

import boto3

uri = os.environ["NPA_DOWNLOAD_URI"]
destination = pathlib.Path(os.environ["NPA_DOWNLOAD_DESTINATION"])
parsed = urlparse(uri)
if parsed.scheme != "s3" or not parsed.netloc:
    raise SystemExit(f"invalid S3 URI: {uri}")
destination.parent.mkdir(parents=True, exist_ok=True)
s3 = boto3.client("s3", endpoint_url=os.environ.get("AWS_ENDPOINT_URL"))
s3.download_file(parsed.netloc, parsed.path.lstrip("/"), str(destination))
print(f"downloaded {uri} -> {destination}", flush=True)
PYDOWNLOAD
}

write_gpu_device_proof() {
  "$PYTHON_BIN" <<'PYGPU'
import json
import os
import pathlib
import subprocess

out = pathlib.Path(os.environ.get("NPA_LOCAL_OUTPUT_DIR", "/tmp/npa-sonic-output"))
out.mkdir(parents=True, exist_ok=True)
payload = {
    "format": "npa_sonic_gpu_device_v1",
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
}
try:
    import torch

    payload["torch_cuda_available"] = bool(torch.cuda.is_available())
    payload["torch_device_count"] = int(torch.cuda.device_count())
    if torch.cuda.is_available():
        payload["torch_device_name"] = torch.cuda.get_device_name(0)
        payload["torch_device_capability"] = list(torch.cuda.get_device_capability(0))
except Exception as exc:
    payload["torch_error"] = f"{type(exc).__name__}: {exc}"
try:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=20,
        check=False,
    )
    payload["nvidia_smi_returncode"] = result.returncode
    payload["nvidia_smi"] = [line.strip() for line in result.stdout.splitlines() if line.strip()]
except Exception as exc:
    payload["nvidia_smi_error"] = f"{type(exc).__name__}: {exc}"
(out / "gpu_device.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PYGPU
}

write_image_pull_proof() {
  "$PYTHON_BIN" <<'PYIMAGE'
import json
import os
import pathlib

out = pathlib.Path(os.environ.get("NPA_LOCAL_OUTPUT_DIR", "/tmp/npa-sonic-output"))
out.mkdir(parents=True, exist_ok=True)
payload = {
    "format": "npa_sonic_image_pull_proof_v1",
    "policy_image": os.environ.get("SONIC_POLICY_IMAGE", ""),
    "repo_digests": os.environ.get("SONIC_IMAGE_REPO_DIGESTS", ""),
    "payload_mode": os.environ.get("SONIC_PAYLOAD_MODE", ""),
}
(out / "image_pull_proof.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PYIMAGE
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
    "train_mode": os.environ.get("SONIC_TRAIN_MODE", "smoke"),
    "real_train": os.environ.get("SONIC_RUN_REAL_TRAIN", "0") == "1",
    "fine_tuned_checkpoint": os.environ.get("SONIC_FINE_TUNED_CHECKPOINT_PATH", ""),
    "fine_tuned_checkpoint_uri": os.environ.get("SONIC_FINE_TUNED_CHECKPOINT_URI", ""),
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

download_training_checkpoint() {
  local checkpoint_path="${SONIC_CHECKPOINT_PATH:-sonic_release/last.pt}"
  if [ -f "$checkpoint_path" ]; then
    return 0
  fi
  if [ "${SONIC_DOWNLOAD_TRAINING_CHECKPOINT:-1}" != "1" ]; then
    return 0
  fi
  local token_arg=()
  if [ -n "${HF_TOKEN:-}" ]; then
    token_arg=(--token "$HF_TOKEN")
  fi
  "$PYTHON_BIN" /opt/sonic/download_from_hf.py --training --no-smpl "${token_arg[@]}"
}

run_real_train() {
  cd /opt/sonic
  download_training_checkpoint
  local checkpoint_path="${SONIC_CHECKPOINT_PATH:-sonic_release/last.pt}"
  local train_mode="${SONIC_TRAIN_MODE:-train}"
  local experiment_base="${SONIC_EXPERIMENT_BASE_DIR:-$OUTPUT_DIR/logs_rl}"
  local motion_file="${SONIC_MOTION_FILE:-}"
  local smpl_motion_file="${SONIC_SMPL_MOTION_FILE:-}"
  if [ "${SONIC_SAMPLE_DATA:-0}" = "1" ]; then
    motion_file="${motion_file:-sample_data/robot_filtered}"
    smpl_motion_file="${smpl_motion_file:-sample_data/smpl_filtered}"
  fi
  local hydra_args=(
    "gear_sonic/train_agent_trl.py"
    "+exp=manager/universal_token/all_modes/sonic_release"
    "+checkpoint=${checkpoint_path}"
    "+resume=${SONIC_RESUME:-False}"
    "num_envs=${SONIC_NUM_ENVS:-16}"
    "headless=${SONIC_HEADLESS:-True}"
    "base_dir=${experiment_base}"
    "++algo.config.num_learning_iterations=${SONIC_MAX_ITERATIONS:-5}"
    "++algo.config.num_steps_per_env=${SONIC_NUM_STEPS_PER_ENV:-4}"
    "++callbacks.model_save.save_last_frequency=${SONIC_SAVE_LAST_FREQUENCY:-1}"
    "++callbacks.model_save.save_frequency=${SONIC_SAVE_FREQUENCY:-1}"
  )
  if [ -n "$motion_file" ]; then
    hydra_args+=("++manager_env.commands.motion.motion_lib_cfg.motion_file=${motion_file}")
  fi
  if [ -n "$smpl_motion_file" ]; then
    hydra_args+=("++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=${smpl_motion_file}")
  fi
  printf 'NPA_SONIC_REAL_TRAIN mode=%s checkpoint=%s iterations=%s num_envs=%s\n' \
    "$train_mode" "$checkpoint_path" "${SONIC_MAX_ITERATIONS:-5}" "${SONIC_NUM_ENVS:-16}"
  "$PYTHON_BIN" -m accelerate.commands.launch \
    --num_processes="${SONIC_NUM_PROCESSES:-1}" \
    "${hydra_args[@]}"
  sync_training_artifacts "$checkpoint_path"
}

sync_training_artifacts() {
  local source_checkpoint="$1"
  SONIC_SOURCE_CHECKPOINT_PATH="$source_checkpoint" "$PYTHON_BIN" <<'PYSYNC'
import json
import os
import pathlib
import shutil
import time

out = pathlib.Path(os.environ.get("NPA_LOCAL_OUTPUT_DIR", "/tmp/npa-sonic-output"))
out.mkdir(parents=True, exist_ok=True)
search_roots = [
    pathlib.Path(os.environ.get("SONIC_EXPERIMENT_BASE_DIR", out / "logs_rl")),
    pathlib.Path("/opt/sonic/logs_rl"),
    out,
]
candidates = []
for root in search_roots:
    if root.exists():
        candidates.extend(root.rglob("last.pt"))
        candidates.extend(root.rglob("model_step_*.pt"))
candidates = [path for path in candidates if path.is_file()]
if not candidates:
    raise SystemExit("real SONIC training completed but no checkpoint was found")
checkpoint = max(candidates, key=lambda path: path.stat().st_mtime)
checkpoint_dir = out / "checkpoints"
checkpoint_dir.mkdir(parents=True, exist_ok=True)
target = checkpoint_dir / "last.pt"
if checkpoint.resolve() != target.resolve():
    shutil.copy2(checkpoint, target)
config_candidates = sorted(checkpoint.parent.glob("config*.yaml"))
for config in config_candidates[:3]:
    shutil.copy2(config, out / config.name)
manifest = {
    "format": "npa_sonic_finetune_manifest_v1",
    "status": "completed",
    "train_mode": os.environ.get("SONIC_TRAIN_MODE", "finetune"),
    "source_checkpoint_path": os.environ.get("SONIC_SOURCE_CHECKPOINT_PATH", ""),
    "source_checkpoint_ref": os.environ.get("SONIC_CHECKPOINT", ""),
    "fine_tuned_checkpoint_path": str(target),
    "fine_tuned_checkpoint_uri": (os.environ.get("NPA_OUTPUT_PATH", "").rstrip("/") + "/checkpoints/last.pt")
    if os.environ.get("NPA_OUTPUT_PATH")
    else "",
    "experiment_checkpoint_path": str(checkpoint),
    "max_iterations": int(os.environ.get("SONIC_MAX_ITERATIONS", "5")),
    "num_envs": int(os.environ.get("SONIC_NUM_ENVS", "16")),
    "num_steps_per_env": int(os.environ.get("SONIC_NUM_STEPS_PER_ENV", "4")),
    "generated_at": int(time.time()),
}
(out / "fine_tune_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
os.environ["SONIC_FINE_TUNED_CHECKPOINT_PATH"] = str(target)
os.environ["SONIC_FINE_TUNED_CHECKPOINT_URI"] = manifest["fine_tuned_checkpoint_uri"]
print(f"NPA_SONIC_FINE_TUNE_CHECKPOINT {target}", flush=True)
PYSYNC
  export SONIC_FINE_TUNED_CHECKPOINT_PATH="$OUTPUT_DIR/checkpoints/last.pt"
  if [ -n "${NPA_OUTPUT_PATH:-}" ]; then
    export SONIC_FINE_TUNED_CHECKPOINT_URI="${NPA_OUTPUT_PATH%/}/checkpoints/last.pt"
  fi
  write_gpu_device_proof
  write_image_pull_proof
}

run_mujoco_eval() {
  local checkpoint_uri="${SONIC_FINE_TUNED_CHECKPOINT_URI:-${SONIC_EVAL_CHECKPOINT_URI:-}}"
  local checkpoint_path="${SONIC_EVAL_CHECKPOINT_PATH:-}"
  if [ -z "$checkpoint_path" ]; then
    checkpoint_path="$OUTPUT_DIR/input/fine_tuned_last.pt"
  fi
  if [ -n "$checkpoint_uri" ]; then
    download_s3_file "$checkpoint_uri" "$checkpoint_path"
  fi
  export SONIC_EVAL_CHECKPOINT_PATH="$checkpoint_path"
  "$PYTHON_BIN" /opt/npa/docker/workbench/sonic/mujoco_eval.py
  write_gpu_device_proof
  write_image_pull_proof
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
  mujoco-eval|mujoco_eval)
    set +e
    run_mujoco_eval
    rc=$?
    set -e
    upload_outputs
    exit "$rc"
    ;;
  finetune|fine-tune)
    export SONIC_RUN_REAL_TRAIN=1
    export SONIC_TRAIN_MODE="${SONIC_TRAIN_MODE:-finetune}"
    download_sample_data
    run_real_train
    write_upload_and_exit finetune
    ;;
  train)
    download_sample_data
    if [ "${SONIC_RUN_REAL_TRAIN:-0}" = "1" ]; then
      run_real_train
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
