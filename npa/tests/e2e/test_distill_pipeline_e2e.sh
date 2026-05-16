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

if [ "${#ARGS[@]}" -ne 8 ]; then
  echo "Usage: $0 [--skip-deploy] <genesis-project> <genesis-name> <genesis-gpu-type> <genesis-gpu-preset> <lerobot-project> <lerobot-name> <lerobot-gpu-type> <lerobot-gpu-preset>" >&2
  exit 2
fi

GENESIS_PROJECT="${ARGS[0]}"
GENESIS_NAME="${ARGS[1]}"
GENESIS_GPU_TYPE="${ARGS[2]}"
GENESIS_GPU_PRESET="${ARGS[3]}"
LEROBOT_PROJECT="${ARGS[4]}"
LEROBOT_NAME="${ARGS[5]}"
LEROBOT_GPU_TYPE="${ARGS[6]}"
LEROBOT_GPU_PRESET="${ARGS[7]}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NPA_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
CLI="$NPA_DIR/.venv/bin/npa"
PYTHON_BIN="$NPA_DIR/.venv/bin/python"

RUN_ID="$(date +%Y%m%d%H%M%S)-$$"
LOCAL_ROOT="${NPA_E2E_OUTPUT_DIR:-$NPA_DIR/runs/e2e}/distill-$RUN_ID"

GENESIS_REMOTE_BASE="/tmp/npa-e2e-distill-genesis-$RUN_ID"
GENESIS_TEACHER_DIR="$GENESIS_REMOTE_BASE/teacher"
GENESIS_LOG_DIR="$GENESIS_REMOTE_BASE/logs"
GENESIS_TEACHER_CHECKPOINT="$GENESIS_TEACHER_DIR/model.pt"
GENESIS_NPA_SRC="$GENESIS_REMOTE_BASE/npa-src"

LEROBOT_REMOTE_BASE="/tmp/npa-e2e-distill-lerobot-$RUN_ID"
LEROBOT_NPA_SRC="$LEROBOT_REMOTE_BASE/npa-src"

S3_BUCKET=""
S3_ENDPOINT=""
S3_PREFIX="${NPA_E2E_DISTILL_S3_PREFIX:-e2e-distill}"
S3_DEMOS_URI=""
S3_DATASET_URI=""
S3_STUDENT_CHECKPOINT_URI=""

TOTAL=0
FAILED=0
GENESIS_CREATED=0
LEROBOT_CREATED=0
GENESIS_DESTROYED=0
LEROBOT_DESTROYED=0

STEP_NAMES=()
STEP_CODES=()
STEP_DURATIONS=()

GENESIS_SSH_TARGET=""
GENESIS_SSH_OPTS=()
LEROBOT_SSH_TARGET=""
LEROBOT_SSH_OPTS=()

record_step_result() {
  local label="$1"
  local code="$2"
  local duration="$3"

  STEP_NAMES+=("$label")
  STEP_CODES+=("$code")
  STEP_DURATIONS+=("$duration")

  TOTAL=$((TOTAL + 1))
  if [ "$code" -eq 0 ]; then
    printf 'PASS: %s\n' "$label"
  else
    FAILED=$((FAILED + 1))
    printf 'FAIL: %s\n' "$label"
  fi
}

print_command() {
  printf '+'
  local arg
  for arg in "$@"; do
    printf ' %q' "$arg"
  done
  printf '\n'
}

shell_join() {
  local out=""
  local quoted
  local arg
  for arg in "$@"; do
    printf -v quoted '%q' "$arg"
    out="${out} ${quoted}"
  done
  printf '%s' "${out# }"
}

run_command_step() {
  local label="$1"
  shift

  print_command "$@"
  local start
  local code
  local duration
  start="$(date +%s)"
  "$@"
  code=$?
  duration=$(( $(date +%s) - start ))
  record_step_result "$label" "$code" "$duration"
  return "$code"
}

run_npa_workbench_step() {
  local label="$1"
  local tool="$2"
  local project="$3"
  local name="$4"
  shift 4

  run_command_step "$label" "$CLI" workbench "$tool" -p "$project" -n "$name" "$@"
}

run_npa_step() {
  local label="$1"
  shift
  run_command_step "$label" "$CLI" "$@"
}

run_function_step() {
  local label="$1"
  local fn="$2"
  shift 2

  print_command "$fn" "$@"
  local start
  local code
  local duration
  start="$(date +%s)"
  "$fn" "$@"
  code=$?
  duration=$(( $(date +%s) - start ))
  record_step_result "$label" "$code" "$duration"
  return "$code"
}

print_summary_table() {
  printf '\n%-48s %9s %10s\n' "Step" "Exit code" "Seconds"
  printf '%-48s %9s %10s\n' "----" "---------" "-------"

  local i
  for i in "${!STEP_NAMES[@]}"; do
    printf '%-48s %9s %10s\n' "${STEP_NAMES[$i]}" "${STEP_CODES[$i]}" "${STEP_DURATIONS[$i]}"
  done

  local passed=$((TOTAL - FAILED))
  printf '\nSUMMARY: %d/%d steps passed\n' "$passed" "$TOTAL"
}

resolve_ssh_values() {
  local project="$1"
  local name="$2"

  "$PYTHON_BIN" - "$project" "$name" <<'PY'
import os
import sys

from npa.clients.config import resolve_ssh_config

cfg = resolve_ssh_config(project=sys.argv[1], name=sys.argv[2])
print(cfg.ssh.host)
print(cfg.ssh.user)
print(os.path.expanduser(cfg.ssh.key_path))
PY
}

resolve_genesis_ssh() {
  local values
  values="$(resolve_ssh_values "$GENESIS_PROJECT" "$GENESIS_NAME")" || return 1

  local host
  local user
  local key
  host="$(printf '%s\n' "$values" | sed -n '1p')"
  user="$(printf '%s\n' "$values" | sed -n '2p')"
  key="$(printf '%s\n' "$values" | sed -n '3p')"

  GENESIS_SSH_TARGET="${user}@${host}"
  GENESIS_SSH_OPTS=(-i "$key" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30)
  [ -n "$host" ] && [ -n "$user" ] && [ -n "$key" ]
}

resolve_lerobot_ssh() {
  local values
  values="$(resolve_ssh_values "$LEROBOT_PROJECT" "$LEROBOT_NAME")" || return 1

  local host
  local user
  local key
  host="$(printf '%s\n' "$values" | sed -n '1p')"
  user="$(printf '%s\n' "$values" | sed -n '2p')"
  key="$(printf '%s\n' "$values" | sed -n '3p')"

  LEROBOT_SSH_TARGET="${user}@${host}"
  LEROBOT_SSH_OPTS=(-i "$key" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30)
  [ -n "$host" ] && [ -n "$user" ] && [ -n "$key" ]
}

resolve_storage_paths() {
  local values
  values="$("$PYTHON_BIN" - "$GENESIS_PROJECT" "$GENESIS_NAME" "$LEROBOT_PROJECT" "$LEROBOT_NAME" <<'PY'
import os
import sys
from urllib.parse import urlparse

from npa.clients.config import resolve_ssh_config, resolve_terraform_state


def bucket_name(uri: str) -> str:
    if not uri:
        return ""
    if uri.startswith("s3://"):
        return urlparse(uri).netloc
    return uri.split("/", 1)[0]


configs = [
    resolve_ssh_config(project=sys.argv[1], name=sys.argv[2]),
    resolve_ssh_config(project=sys.argv[3], name=sys.argv[4]),
]
states = [
    resolve_terraform_state(sys.argv[1]),
    resolve_terraform_state(sys.argv[3]),
]

buckets = [bucket_name(cfg.storage.checkpoint_bucket) for cfg in configs]
buckets = [bucket for bucket in buckets if bucket]
if not buckets:
    raise SystemExit("No storage checkpoint_bucket found in workbench config")
if len(set(buckets)) > 1:
    raise SystemExit(f"Workbench storage buckets differ: {buckets}")

endpoint = next((cfg.storage.endpoint_url for cfg in configs if cfg.storage.endpoint_url), "")
endpoint = endpoint or next((state.endpoint for state in states if state.endpoint), "")
access_key = next((cfg.storage.aws_access_key_id for cfg in configs if cfg.storage.aws_access_key_id), "")
access_key = access_key or next((state.access_key for state in states if state.access_key), "")
secret_key = next((cfg.storage.aws_secret_access_key for cfg in configs if cfg.storage.aws_secret_access_key), "")
secret_key = secret_key or next((state.secret_key for state in states if state.secret_key), "")

print(buckets[0])
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

  S3_DEMOS_URI="s3://$S3_BUCKET/$S3_PREFIX/demos/"
  S3_DATASET_URI="s3://$S3_BUCKET/$S3_PREFIX/dataset/"
  S3_STUDENT_CHECKPOINT_URI="s3://$S3_BUCKET/$S3_PREFIX/student-checkpoint/"

  [ -n "$S3_BUCKET" ] && [ -n "$S3_ENDPOINT" ]
}

cleanup_s3_prefix() {
  if [ -z "$S3_BUCKET" ]; then
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

adapter_convert_s3() {
  mkdir -p "$LOCAL_ROOT" || return 1
  "$CLI" adapter convert \
    --input-path "$S3_DEMOS_URI" \
    --output-path "$S3_DATASET_URI" \
    --fps 20 \
    --robot franka_panda \
    --task "Pick and place cube to target"
}

stage_npa_on_genesis() {
  resolve_genesis_ssh || return 1

  print_command ssh "${GENESIS_SSH_OPTS[@]}" "$GENESIS_SSH_TARGET" "rm -rf '$GENESIS_NPA_SRC' && mkdir -p '$GENESIS_NPA_SRC'"
  ssh "${GENESIS_SSH_OPTS[@]}" "$GENESIS_SSH_TARGET" "rm -rf '$GENESIS_NPA_SRC' && mkdir -p '$GENESIS_NPA_SRC'" || return 1

  print_command scp -r "${GENESIS_SSH_OPTS[@]}" "$NPA_DIR/pyproject.toml" "$NPA_DIR/src" "$GENESIS_SSH_TARGET:$GENESIS_NPA_SRC/"
  scp -r "${GENESIS_SSH_OPTS[@]}" "$NPA_DIR/pyproject.toml" "$NPA_DIR/src" "$GENESIS_SSH_TARGET:$GENESIS_NPA_SRC/" || return 1

  print_command ssh "${GENESIS_SSH_OPTS[@]}" "$GENESIS_SSH_TARGET" "eval \"\$(/opt/conda/bin/conda shell.bash hook)\" && conda activate genesis && python -m pip install -e '$GENESIS_NPA_SRC'"
  ssh "${GENESIS_SSH_OPTS[@]}" "$GENESIS_SSH_TARGET" "eval \"\$(/opt/conda/bin/conda shell.bash hook)\" && conda activate genesis && python -m pip install -e '$GENESIS_NPA_SRC'"
}

stage_npa_on_lerobot() {
  resolve_lerobot_ssh || return 1

  print_command ssh "${LEROBOT_SSH_OPTS[@]}" "$LEROBOT_SSH_TARGET" "rm -rf '$LEROBOT_NPA_SRC' && mkdir -p '$LEROBOT_NPA_SRC'"
  ssh "${LEROBOT_SSH_OPTS[@]}" "$LEROBOT_SSH_TARGET" "rm -rf '$LEROBOT_NPA_SRC' && mkdir -p '$LEROBOT_NPA_SRC'" || return 1

  print_command scp -r "${LEROBOT_SSH_OPTS[@]}" "$NPA_DIR/pyproject.toml" "$NPA_DIR/src" "$LEROBOT_SSH_TARGET:$LEROBOT_NPA_SRC/"
  scp -r "${LEROBOT_SSH_OPTS[@]}" "$NPA_DIR/pyproject.toml" "$NPA_DIR/src" "$LEROBOT_SSH_TARGET:$LEROBOT_NPA_SRC/" || return 1

  print_command ssh "${LEROBOT_SSH_OPTS[@]}" "$LEROBOT_SSH_TARGET" "source /opt/lerobot/venv/bin/activate && python -m pip install -e '$LEROBOT_NPA_SRC'"
  ssh "${LEROBOT_SSH_OPTS[@]}" "$LEROBOT_SSH_TARGET" "source /opt/lerobot/venv/bin/activate && python -m pip install -e '$LEROBOT_NPA_SRC'"
}

run_remote_lerobot_npa_step() {
  local label="$1"
  shift

  resolve_lerobot_ssh || {
    record_step_result "$label" 1 0
    return 1
  }
  stage_npa_on_lerobot || {
    record_step_result "$label" 1 0
    return 1
  }

  local npa_line
  npa_line="$(shell_join npa "$@")"
  printf '+ remote npa: %s\n' "$npa_line"

  local remote_script
  remote_script="set -euo pipefail
source /opt/lerobot/venv/bin/activate
set -a
if [ -f /opt/lerobot/.env ]; then . /opt/lerobot/.env; fi
set +a
$npa_line"

  print_command ssh "${LEROBOT_SSH_OPTS[@]}" "$LEROBOT_SSH_TARGET" bash -s
  local start
  local code
  local duration
  start="$(date +%s)"
  ssh "${LEROBOT_SSH_OPTS[@]}" "$LEROBOT_SSH_TARGET" bash -s <<< "$remote_script"
  code=$?
  duration=$(( $(date +%s) - start ))
  record_step_result "$label" "$code" "$duration"
  return "$code"
}

destroy_genesis() {
  if [ "$GENESIS_CREATED" -eq 0 ] || [ "$GENESIS_DESTROYED" -eq 1 ]; then
    return 0
  fi
  if run_npa_workbench_step "destroy Genesis workbench" genesis "$GENESIS_PROJECT" "$GENESIS_NAME" deploy --destroy --gpu-type "$GENESIS_GPU_TYPE" --gpu-preset "$GENESIS_GPU_PRESET"; then
    GENESIS_DESTROYED=1
    return 0
  fi
  return 1
}

destroy_lerobot() {
  if [ "$LEROBOT_CREATED" -eq 0 ] || [ "$LEROBOT_DESTROYED" -eq 1 ]; then
    return 0
  fi
  if run_npa_workbench_step "destroy LeRobot workbench" lerobot "$LEROBOT_PROJECT" "$LEROBOT_NAME" deploy --destroy --gpu-type "$LEROBOT_GPU_TYPE" --gpu-preset "$LEROBOT_GPU_PRESET"; then
    LEROBOT_DESTROYED=1
    return 0
  fi
  return 1
}

cleanup() {
  local status=$?
  trap - EXIT

  cleanup_s3_prefix || true
  destroy_genesis || true
  destroy_lerobot || true

  print_summary_table

  if [ "$FAILED" -ne 0 ] || [ "$status" -ne 0 ]; then
    exit 1
  fi
  exit 0
}
trap cleanup EXIT

run_command_step "npa CLI exists" test -x "$CLI" || exit 1

if [ "$SKIP_DEPLOY" -eq 1 ]; then
  printf 'SKIP: deploy Genesis workbench (--skip-deploy)\n'
  printf 'SKIP: deploy LeRobot workbench (--skip-deploy)\n'
else
  if run_npa_workbench_step "deploy Genesis workbench" genesis "$GENESIS_PROJECT" "$GENESIS_NAME" deploy --gpu-type "$GENESIS_GPU_TYPE" --gpu-preset "$GENESIS_GPU_PRESET"; then
    GENESIS_CREATED=1
  else
    exit 1
  fi

  if run_npa_workbench_step "deploy LeRobot workbench" lerobot "$LEROBOT_PROJECT" "$LEROBOT_NAME" deploy --gpu-type "$LEROBOT_GPU_TYPE" --gpu-preset "$LEROBOT_GPU_PRESET"; then
    LEROBOT_CREATED=1
  else
    exit 1
  fi
fi

run_npa_workbench_step "Genesis system-info" genesis "$GENESIS_PROJECT" "$GENESIS_NAME" system-info || exit 1
run_npa_workbench_step "LeRobot system-info" lerobot "$LEROBOT_PROJECT" "$LEROBOT_NAME" system-info || exit 1
run_function_step "resolve S3 artifact paths" resolve_storage_paths || exit 1
run_function_step "stage NPA on Genesis VM" stage_npa_on_genesis || exit 1

run_npa_workbench_step "Genesis train-teacher 50 iterations" genesis "$GENESIS_PROJECT" "$GENESIS_NAME" \
  train-teacher \
  --n-envs 64 \
  --max-iterations 50 \
  --output "$GENESIS_TEACHER_DIR" \
  --log-dir "$GENESIS_LOG_DIR" || exit 1

run_npa_workbench_step "Genesis generate demos" genesis "$GENESIS_PROJECT" "$GENESIS_NAME" \
  generate-demos \
  --checkpoint "$GENESIS_TEACHER_CHECKPOINT" \
  --n-envs 1 \
  --n-episodes 0 \
  --output-path "$S3_DEMOS_URI" \
  --no-domain-randomize \
  --allow-failure-demos || exit 1

run_function_step "adapter convert Genesis demos" adapter_convert_s3 || exit 1

run_remote_lerobot_npa_step "LeRobot train-student" \
  workbench lerobot train-student \
  --input-path "$S3_DATASET_URI" \
  --policy act \
  --epochs 1 \
  --batch-size 8 \
  --num-workers 4 \
  --output-path "$S3_STUDENT_CHECKPOINT_URI" || exit 1

run_npa_workbench_step "Genesis eval student policy" genesis "$GENESIS_PROJECT" "$GENESIS_NAME" \
  eval-student \
  --input-path "$S3_STUDENT_CHECKPOINT_URI" \
  --n-envs 1 \
  --n-episodes 1 \
  --output "$GENESIS_REMOTE_BASE/eval" \
  --no-domain-randomize \
  --action-space cartesian || exit 1
