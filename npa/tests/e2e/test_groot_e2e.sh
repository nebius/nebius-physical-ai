#!/usr/bin/env bash

set -uo pipefail

SKIP_DEPLOY=0
ARGS=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --skip-deploy)
      SKIP_DEPLOY=1
      shift
      ;;
    --)
      shift
      while [ "$#" -gt 0 ]; do
        ARGS+=("$1")
        shift
      done
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

if [ "${#ARGS[@]}" -lt 4 ] || [ "${#ARGS[@]}" -gt 6 ]; then
  echo "Usage: $0 [--skip-deploy] <project> <name> <gpu-type> <gpu-preset> [bucket] [model]" >&2
  exit 2
fi

PROJECT="${ARGS[0]}"
NAME="${ARGS[1]}"
GPU_TYPE="${ARGS[2]}"
GPU_PRESET="${ARGS[3]}"
S3_BUCKET="${ARGS[4]:-${NPA_E2E_GROOT_BUCKET:-${NPA_S3_BUCKET:-test-bucket-00000000}}}"
MODEL="${ARGS[5]:-${NPA_E2E_GROOT_MODEL:-nvidia/GR00T-N1.7-3B}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NPA_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
CLI="$NPA_DIR/.venv/bin/npa"
PYTHON_BIN="$NPA_DIR/.venv/bin/python"
TOOL="groot"
REMOTE_ROOT="/tmp/npa-e2e-groot-smoke-$$"
GROOT_REPO="/opt/groot/Isaac-GR00T"
GROOT_VENV="$GROOT_REPO/.venv"
GROOT_MODEL_DIR="/opt/groot/models"
ISAAC_LAB_VENV="/opt/isaac-lab/venv"
REGION="${NPA_E2E_REGION:-eu-north1}"
TENANT_ID="${NPA_E2E_TENANT_ID:-${NPA_TENANT_ID:-${NEBIUS_ACCOUNT_ID:-}}}"
PROJECT_ID="${NPA_E2E_PROJECT_ID:-${NPA_PROJECT_ID:-project-test-00000000}}"
EMBODIMENT_TAG="${NPA_E2E_GROOT_EMBODIMENT:-NEW_EMBODIMENT}"
MAX_STEPS="${NPA_E2E_GROOT_MAX_STEPS:-1}"
GLOBAL_BATCH_SIZE="${NPA_E2E_GROOT_BATCH_SIZE:-1}"
DATALOADER_WORKERS="${NPA_E2E_GROOT_DATALOADER_WORKERS:-0}"

TOTAL=0
FAILED=0
DEPLOY_ATTEMPTED=0
TEARDOWN_DONE=0
LOCAL_TMP=""
SSH_HOST=""
SSH_USER=""
SSH_KEY=""
SSH_TARGET=""
SSH_OPTS=()
S3_ENDPOINT=""
S3_PREFIX=""
S3_DATASET_URI=""
S3_MODEL_URI=""
S3_CHECKPOINT_URI=""
S3_EVAL_URI=""
S3_INFER_URI=""

record_pass() {
  TOTAL=$((TOTAL + 1))
  printf 'PASS: %s\n' "$1"
}

record_fail() {
  TOTAL=$((TOTAL + 1))
  FAILED=$((FAILED + 1))
  printf 'FAIL: %s\n' "$1"
}

print_summary() {
  local passed=$((TOTAL - FAILED))
  printf 'SUMMARY: %d/%d steps passed\n' "$passed" "$TOTAL"
}

print_command() {
  printf '+'
  local arg
  for arg in "$@"; do
    printf ' %q' "$arg"
  done
  printf '\n'
}

run_step() {
  local label="$1"
  shift
  print_command "$@"
  if "$@"; then
    record_pass "$label"
    return 0
  fi
  record_fail "$label"
  return 1
}

run_npa_step() {
  local label="$1"
  shift
  run_step "$label" "$CLI" workbench "$TOOL" -p "$PROJECT" -n "$NAME" "$@"
}

ensure_local_tmp() {
  if [ -z "$LOCAL_TMP" ]; then
    LOCAL_TMP="$(mktemp -d "${TMPDIR:-/tmp}/npa-groot-e2e.XXXXXX")"
  fi
}

run_npa_capture_step() {
  local label="$1"
  local output_file="$2"
  shift 2
  ensure_local_tmp
  print_command "$CLI" workbench "$TOOL" -p "$PROJECT" -n "$NAME" "$@"
  if "$CLI" workbench "$TOOL" -p "$PROJECT" -n "$NAME" "$@" >"$output_file" 2>&1; then
    cat "$output_file"
    record_pass "$label"
    return 0
  fi
  cat "$output_file"
  record_fail "$label"
  return 1
}

run_teardown_step() {
  if [ "$TEARDOWN_DONE" -eq 1 ]; then
    return 0
  fi
  TEARDOWN_DONE=1
  run_npa_step "destroy workbench" deploy --destroy --gpu-type "$GPU_TYPE" --gpu-preset "$GPU_PRESET"
}

cleanup_s3_prefix() {
  if [ -z "$S3_BUCKET" ] || [ -z "$S3_PREFIX" ] || [ -z "$S3_ENDPOINT" ]; then
    return 0
  fi

  printf '+ cleanup s3://%s/%s/\n' "$S3_BUCKET" "$S3_PREFIX"
  "$PYTHON_BIN" - "$S3_BUCKET" "$S3_PREFIX/" <<'PY'
import os
import sys

import boto3

bucket = sys.argv[1]
prefix = sys.argv[2].lstrip("/")
s3 = boto3.client(
    "s3",
    endpoint_url=os.environ.get("NEBIUS_S3_ENDPOINT") or os.environ.get("AWS_ENDPOINT_URL") or None,
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID") or None,
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY") or None,
)
for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
    objects = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
    if objects:
        s3.delete_objects(Bucket=bucket, Delete={"Objects": objects})
PY
}

cleanup() {
  local status=$?
  trap - EXIT
  cleanup_s3_prefix || true
  if [ "$DEPLOY_ATTEMPTED" -eq 1 ] && [ "$TEARDOWN_DONE" -eq 0 ]; then
    run_teardown_step || true
  fi
  if [ -n "$LOCAL_TMP" ]; then
    rm -rf "$LOCAL_TMP"
  fi
  print_summary
  if [ "$FAILED" -ne 0 ] || [ "$status" -ne 0 ]; then
    exit 1
  fi
  exit 0
}
trap cleanup EXIT

slugify() {
  local value
  value="$(printf '%s' "$1" | tr -c 'A-Za-z0-9_-' '-' | sed 's/^-*//;s/-*$//')"
  if [ -z "$value" ]; then
    value="workbench"
  fi
  printf '%s' "$value"
}

model_slug() {
  printf '%s' "$1" | sed 's#^ngc://##; s#[/:]#--#g'
}

validate_rt_core_gpu() {
  local normalized
  normalized="$(printf '%s' "$GPU_TYPE" | tr '[:upper:]_' '[:lower:]-')"
  case "$normalized" in
    *l40s*|*rtx*6000*)
      return 0
      ;;
    *)
      printf 'GR00T e2e requires an RT-core GPU type, such as gpu-l40s-a or gpu-rtx-pro-6000. Got: %s\n' "$GPU_TYPE" >&2
      return 1
      ;;
  esac
}

resolve_ssh() {
  local values
  values="$("$PYTHON_BIN" - "$PROJECT" "$NAME" <<'PY'
import os
import sys

from npa.clients.config import resolve_ssh_config

cfg = resolve_ssh_config(project=sys.argv[1], name=sys.argv[2])
print(cfg.ssh.host)
print(cfg.ssh.user)
print(os.path.expanduser(cfg.ssh.key_path))
PY
)"
  if [ "$?" -ne 0 ]; then
    return 1
  fi

  SSH_HOST="$(printf '%s\n' "$values" | sed -n '1p')"
  SSH_USER="$(printf '%s\n' "$values" | sed -n '2p')"
  SSH_KEY="$(printf '%s\n' "$values" | sed -n '3p')"
  SSH_TARGET="${SSH_USER}@${SSH_HOST}"
  SSH_OPTS=(-i "$SSH_KEY" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30)

  [ -n "$SSH_HOST" ] && [ -n "$SSH_USER" ] && [ -n "$SSH_KEY" ]
}

resolve_storage() {
  local values
  values="$("$PYTHON_BIN" - "$PROJECT" "$NAME" "$S3_BUCKET" "$REGION" <<'PY'
import sys
from urllib.parse import urlparse

from npa.clients.config import resolve_ssh_config, resolve_terraform_state


def bucket_name(value: str) -> str:
    if not value:
        return ""
    if value.startswith("s3://"):
        return urlparse(value).netloc
    return value.split("/", 1)[0]


project, name, explicit_bucket, region = sys.argv[1:5]
cfg = resolve_ssh_config(project=project, name=name)
state = resolve_terraform_state(project)
bucket = explicit_bucket or bucket_name(cfg.storage.checkpoint_bucket) or state.bucket
endpoint = cfg.storage.endpoint_url or state.endpoint or f"https://storage.{region}.nebius.cloud"
access_key = cfg.storage.aws_access_key_id or state.access_key
secret_key = cfg.storage.aws_secret_access_key or state.secret_key
print(bucket)
print(endpoint)
print(access_key)
print(secret_key)
PY
)" || return 1

  S3_BUCKET="$(printf '%s\n' "$values" | sed -n '1p')"
  S3_ENDPOINT="$(printf '%s\n' "$values" | sed -n '2p')"
  export AWS_ENDPOINT_URL="$S3_ENDPOINT"
  export NEBIUS_S3_ENDPOINT="$S3_ENDPOINT"
  export AWS_ACCESS_KEY_ID
  export AWS_SECRET_ACCESS_KEY
  AWS_ACCESS_KEY_ID="$(printf '%s\n' "$values" | sed -n '3p')"
  AWS_SECRET_ACCESS_KEY="$(printf '%s\n' "$values" | sed -n '4p')"

  local run_slug
  run_slug="$(slugify "$NAME")"
  S3_PREFIX="${NPA_E2E_GROOT_S3_PREFIX:-test-data/groot/npa-e2e-${run_slug}-$(date +%Y%m%d%H%M%S)-$$}"
  S3_DATASET_URI="s3://$S3_BUCKET/$S3_PREFIX/dataset/"
  S3_MODEL_URI="s3://$S3_BUCKET/$S3_PREFIX/model/"
  S3_CHECKPOINT_URI="s3://$S3_BUCKET/$S3_PREFIX/checkpoint/"
  S3_EVAL_URI="s3://$S3_BUCKET/$S3_PREFIX/eval/"
  S3_INFER_URI="s3://$S3_BUCKET/$S3_PREFIX/infer/"

  [ -n "$S3_BUCKET" ] && [ -n "$S3_ENDPOINT" ]
}

storage_exports() {
  printf 'export AWS_ENDPOINT_URL=%q\n' "$S3_ENDPOINT"
  printf 'export NEBIUS_S3_ENDPOINT=%q\n' "$S3_ENDPOINT"
  printf 'export AWS_ACCESS_KEY_ID=%q\n' "$AWS_ACCESS_KEY_ID"
  printf 'export AWS_SECRET_ACCESS_KEY=%q\n' "$AWS_SECRET_ACCESS_KEY"
}

stage_smoke_files() {
  resolve_ssh || return 1
  ensure_local_tmp
  mkdir -p "$LOCAL_TMP/npa/smoke" || return 1
  printf '' > "$LOCAL_TMP/npa/__init__.py" || return 1
  cp "$NPA_DIR/pyproject.toml" "$LOCAL_TMP/pyproject.toml" || return 1
  cp "$NPA_DIR/src/npa/smoke/__init__.py" "$LOCAL_TMP/npa/smoke/__init__.py" || return 1
  cp "$NPA_DIR/src/npa/smoke/_versions.py" "$LOCAL_TMP/npa/smoke/_versions.py" || return 1
  cp "$NPA_DIR/src/npa/smoke/test_groot_env.py" "$LOCAL_TMP/npa/smoke/test_groot_env.py" || return 1
  cp "$NPA_DIR/src/npa/smoke/test_groot_functional.py" "$LOCAL_TMP/npa/smoke/test_groot_functional.py" || return 1
  cp "$NPA_DIR/src/npa/smoke/test_isaac_lab_env.py" "$LOCAL_TMP/npa/smoke/test_isaac_lab_env.py" || return 1

  print_command ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "rm -rf '$REMOTE_ROOT' && mkdir -p '$REMOTE_ROOT'"
  ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "rm -rf '$REMOTE_ROOT' && mkdir -p '$REMOTE_ROOT'" || return 1

  print_command scp -r -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new "$LOCAL_TMP/npa" "$LOCAL_TMP/pyproject.toml" "$SSH_TARGET:$REMOTE_ROOT/"
  scp -r -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new "$LOCAL_TMP/npa" "$LOCAL_TMP/pyproject.toml" "$SSH_TARGET:$REMOTE_ROOT/" || return 1
}

run_remote_script_step() {
  local label="$1"
  local script="$2"
  print_command ssh "${SSH_OPTS[@]}" "$SSH_TARGET" bash -s
  if ssh "${SSH_OPTS[@]}" "$SSH_TARGET" bash -s <<< "$script"; then
    record_pass "$label"
    return 0
  fi
  record_fail "$label"
  return 1
}

s3_uri_has_objects() {
  "$PYTHON_BIN" - "$1" <<'PY'
import os
import sys
from urllib.parse import urlparse

import boto3

uri = sys.argv[1]
parsed = urlparse(uri)
bucket = parsed.netloc
prefix = parsed.path.lstrip("/")
s3 = boto3.client(
    "s3",
    endpoint_url=os.environ.get("NEBIUS_S3_ENDPOINT") or os.environ.get("AWS_ENDPOINT_URL") or None,
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID") or None,
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY") or None,
)
resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
if not resp.get("Contents"):
    raise SystemExit(f"no objects found under {uri}")
print("S3_OBJECT_FOUND", resp["Contents"][0]["Key"])
PY
}

s3_eval_json_schema_ok() {
  "$PYTHON_BIN" - "$1" <<'PY'
import json
import os
import sys
from urllib.parse import urlparse

import boto3

uri = sys.argv[1].rstrip("/") + "/npa_groot_eval_results.json"
parsed = urlparse(uri)
s3 = boto3.client(
    "s3",
    endpoint_url=os.environ.get("NEBIUS_S3_ENDPOINT") or os.environ.get("AWS_ENDPOINT_URL") or None,
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID") or None,
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY") or None,
)
obj = s3.get_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))
data = json.loads(obj["Body"].read().decode("utf-8"))
if not isinstance(data.get("metrics"), dict):
    raise SystemExit("missing metrics object")
if int(data.get("episode_count", 0)) < 1:
    raise SystemExit("episode_count must be >= 1")
print(json.dumps({"metrics": data["metrics"], "episode_count": data["episode_count"]}, indent=2))
PY
}

s3_infer_json_schema_ok() {
  "$PYTHON_BIN" - "$1" <<'PY'
import json
import os
import sys
from urllib.parse import urlparse

import boto3

uri = sys.argv[1].rstrip("/") + "/npa_groot_infer_results.json"
parsed = urlparse(uri)
s3 = boto3.client(
    "s3",
    endpoint_url=os.environ.get("NEBIUS_S3_ENDPOINT") or os.environ.get("AWS_ENDPOINT_URL") or None,
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID") or None,
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY") or None,
)
obj = s3.get_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))
data = json.loads(obj["Body"].read().decode("utf-8"))
if data.get("status") != "success":
    raise SystemExit(f"infer status is not success: {data.get('status')}")
if int(data.get("trajectory_count", 0)) < 1:
    raise SystemExit("trajectory_count must be >= 1")
predicted = data.get("predicted_actions")
if not isinstance(predicted, list) or not predicted:
    raise SystemExit("missing predicted_actions list")
if "predicted_actions.npz" not in data.get("artifacts", []):
    raise SystemExit("predicted_actions.npz artifact missing")
print(json.dumps({"trajectory_count": data["trajectory_count"], "first": predicted[0]}, indent=2))
PY
}

run_step "validate RT-core GPU flag" validate_rt_core_gpu || exit 1
run_step "npa CLI exists" test -x "$CLI" || exit 1

if [ "$SKIP_DEPLOY" -eq 1 ]; then
  printf 'SKIP: deploy workbench (--skip-deploy)\n'
else
  DEPLOY_ATTEMPTED=1
  run_npa_step "deploy workbench" deploy \
    --gpu-type "$GPU_TYPE" \
    --gpu-preset "$GPU_PRESET" \
    --no-preemptible \
    --region "$REGION" \
    --project-id "$PROJECT_ID" \
    --tenant-id "$TENANT_ID" || exit 1
fi

ensure_local_tmp
SYSTEM_INFO_OUTPUT="$LOCAL_TMP/system-info.txt"
STATUS_OUTPUT="$LOCAL_TMP/status.json"
LIST_OUTPUT="$LOCAL_TMP/list.txt"

run_npa_capture_step "system-info exits 0" "$SYSTEM_INFO_OUTPUT" system-info || exit 1
run_step "system-info reports GPU" grep -Eiq 'L40S|RTX.*6000|NVIDIA' "$SYSTEM_INFO_OUTPUT" || exit 1
run_step "system-info reports GR00T version" grep -q 'gr00t_version:' "$SYSTEM_INFO_OUTPUT" || exit 1
run_step "system-info reports Isaac Lab version" grep -q 'isaaclab_version:' "$SYSTEM_INFO_OUTPUT" || exit 1

run_npa_capture_step "status returns healthy" "$STATUS_OUTPUT" status --output json || exit 1
run_step "status JSON reports server up" grep -q '"server": "up"' "$STATUS_OUTPUT" || exit 1

run_npa_capture_step "list shows named workbench" "$LIST_OUTPUT" list || exit 1
run_step "named GR00T workbench is listed" grep -q "$NAME" "$LIST_OUTPUT" || exit 1

run_step "resolve S3 storage" resolve_storage || exit 1
run_step "scp smoke tests to VM" stage_smoke_files || exit 1

run_remote_script_step "run GR00T environment smoke test" "set -euo pipefail
source $GROOT_VENV/bin/activate
set -a
if [ -f /etc/npa-groot-server/env ]; then . /etc/npa-groot-server/env; fi
set +a
if ! python - <<'PY'
try:
    import tomllib
except ModuleNotFoundError:
    import tomli  # noqa: F401
PY
then
    UV_BIN=\"\$(command -v uv || true)\"
    if [ -z \"\$UV_BIN\" ] && [ -x \"\$HOME/.local/bin/uv\" ]; then
        UV_BIN=\"\$HOME/.local/bin/uv\"
    fi
    if [ -z \"\$UV_BIN\" ]; then
        echo \"uv not found; cannot install tomli into GR00T venv\" >&2
        exit 1
    fi
    \"\$UV_BIN\" pip install --python \"$GROOT_VENV/bin/python\" tomli
fi

export HF_HUB_DISABLE_TELEMETRY=1
export WANDB_DISABLED=true
export PYTHONPATH=\"$REMOTE_ROOT\"
python -m npa.smoke.test_groot_env" || exit 1

run_remote_script_step "run Isaac Lab environment smoke test" "set -euo pipefail
source $ISAAC_LAB_VENV/bin/activate
export OMNI_KIT_ACCEPT_EULA=\"\${OMNI_KIT_ACCEPT_EULA:-YES}\"
python - <<'PY'
try:
    import tomllib
except ModuleNotFoundError:
    import subprocess
    import sys

    subprocess.check_call([sys.executable, \"-m\", \"pip\", \"install\", \"tomli\"])
PY
export PYTHONPATH=\"$REMOTE_ROOT\"
python -m npa.smoke.test_isaac_lab_env" || exit 1

LOCAL_MODEL_DIR="$GROOT_MODEL_DIR/$(model_slug "$MODEL")"
run_npa_step "download default GR00T model to VM" download --model "$MODEL" || exit 1

run_remote_script_step "assert GR00T model files and weight shape" "set -euo pipefail
source $GROOT_VENV/bin/activate
export LOCAL_MODEL_DIR='$LOCAL_MODEL_DIR'
python - <<'PY'
import os
from pathlib import Path

from safetensors import safe_open

model_dir = Path(os.environ[\"LOCAL_MODEL_DIR\"])
if not model_dir.is_dir():
    raise SystemExit(f\"missing model dir: {model_dir}\")
files = sorted(model_dir.rglob(\"*.safetensors\"))
if not files:
    raise SystemExit(f\"no safetensors files under {model_dir}\")
with safe_open(files[0], framework=\"pt\", device=\"cpu\") as handle:
    keys = list(handle.keys())
    if not keys:
        raise SystemExit(f\"no tensor keys in {files[0]}\")
    shape = tuple(handle.get_slice(keys[0]).get_shape())
print(\"GROOT_WEIGHT_SHAPE\", files[0], keys[0], shape)
PY" || exit 1

run_npa_step "download default GR00T model to S3" download --model "$MODEL" --output-path "$S3_MODEL_URI" || exit 1
run_step "S3 model download wrote objects" s3_uri_has_objects "$S3_MODEL_URI" || exit 1

run_remote_script_step "upload GR00T demo dataset to S3" "$(storage_exports)
set -euo pipefail
source $GROOT_VENV/bin/activate
cd $GROOT_REPO
git lfs pull --include 'demo_data/cube_to_bowl_5/**' || true
python - <<'PY'
import os
from pathlib import Path
from urllib.parse import urlparse

import boto3

src = Path(\"demo_data/cube_to_bowl_5\")
if not (src / \"meta\" / \"modality.json\").exists():
    raise SystemExit(f\"missing GR00T modality config under {src}\")
uri = \"${S3_DATASET_URI}\"
parsed = urlparse(uri)
bucket = parsed.netloc
prefix = parsed.path.lstrip(\"/\").rstrip(\"/\") + \"/\"
s3 = boto3.client(
    \"s3\",
    endpoint_url=os.environ.get(\"NEBIUS_S3_ENDPOINT\") or os.environ.get(\"AWS_ENDPOINT_URL\") or None,
    aws_access_key_id=os.environ.get(\"AWS_ACCESS_KEY_ID\") or None,
    aws_secret_access_key=os.environ.get(\"AWS_SECRET_ACCESS_KEY\") or None,
)
for file_path in src.rglob(\"*\"):
    if file_path.is_file():
        s3.upload_file(str(file_path), bucket, prefix + str(file_path.relative_to(src)))
print(\"GROOT_DATASET_UPLOAD_COMPLETE\", uri)
PY" || exit 1
run_step "S3 demo dataset exists" s3_uri_has_objects "$S3_DATASET_URI" || exit 1

FINETUNE_OUTPUT="$LOCAL_TMP/finetune.txt"
run_npa_capture_step "run minimal GR00T finetune" "$FINETUNE_OUTPUT" finetune \
  --input-path "$S3_DATASET_URI" \
  --output-path "$S3_CHECKPOINT_URI" \
  --base-model "$LOCAL_MODEL_DIR" \
  --robot-embodiment "$EMBODIMENT_TAG" \
  --num-gpus 1 \
  --config "$GROOT_REPO/examples/SO100/so100_config.py" \
  --max-steps "$MAX_STEPS" \
  --global-batch-size "$GLOBAL_BATCH_SIZE" \
  --dataloader-num-workers "$DATALOADER_WORKERS" \
  --save-steps 1 \
  --save-total-limit 1 \
  --save-only-model \
  --output json || exit 1
run_step "finetune output has no CUDA OOM" bash -c "! grep -Eiq 'CUDA out of memory|out of memory' '$FINETUNE_OUTPUT'" || exit 1
run_step "S3 finetune checkpoint exists" s3_uri_has_objects "$S3_CHECKPOINT_URI" || exit 1

EVAL_OUTPUT="$LOCAL_TMP/eval.txt"
run_npa_capture_step "run GR00T offline eval" "$EVAL_OUTPUT" eval \
  --input-path "$S3_CHECKPOINT_URI" \
  --dataset-path "$S3_DATASET_URI" \
  --output-path "$S3_EVAL_URI" \
  --robot-embodiment "$EMBODIMENT_TAG" \
  --output json || exit 1
run_step "S3 eval results JSON has expected schema" s3_eval_json_schema_ok "$S3_EVAL_URI" || exit 1

SERVE_OUTPUT="$LOCAL_TMP/serve-pending.txt"
run_npa_capture_step "serve pending placeholder exits cleanly" "$SERVE_OUTPUT" serve --model "$MODEL" --dry-run || exit 1
run_step "serve placeholder reports pending" grep -q 'status: pending' "$SERVE_OUTPUT" || exit 1

INFER_OUTPUT="$LOCAL_TMP/infer.txt"
run_npa_capture_step "run GR00T inference" "$INFER_OUTPUT" infer \
  --input-path "$S3_CHECKPOINT_URI" \
  --dataset-path "$S3_DATASET_URI" \
  --output-path "$S3_INFER_URI" \
  --embodiment-tag "$EMBODIMENT_TAG" \
  --inference-mode pytorch \
  --steps 32 \
  --action-horizon 16 \
  --output json || exit 1
run_step "S3 infer results JSON has expected schema" s3_infer_json_schema_ok "$S3_INFER_URI" || exit 1

if [ "$SKIP_DEPLOY" -eq 1 ]; then
  printf 'SKIP: destroy workbench (--skip-deploy)\n'
else
  run_teardown_step || exit 1
fi
