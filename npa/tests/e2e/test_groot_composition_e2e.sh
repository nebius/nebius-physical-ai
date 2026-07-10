#!/usr/bin/env bash

set -uo pipefail

if [ "$#" -lt 4 ] || [ "$#" -gt 6 ]; then
  echo "Usage: $0 <project> <groot-name> <groot-gpu-type> <groot-gpu-preset> [bucket] [genesis-name]" >&2
  exit 2
fi

PROJECT="$1"
GROOT_NAME="$2"
GROOT_GPU_TYPE="$3"
GROOT_GPU_PRESET="$4"
S3_BUCKET="${5:-${NPA_E2E_GROOT_BUCKET:-${NPA_S3_BUCKET:-test-bucket-00000000}}}"
GENESIS_PROJECT="${NPA_E2E_GENESIS_PROJECT:-$PROJECT}"
GENESIS_NAME="${6:-${NPA_E2E_GENESIS_NAME:-ctr-genesis-h200-fallback-20260508}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NPA_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
CLI="$NPA_DIR/.venv/bin/npa"
PYTHON_BIN="$NPA_DIR/.venv/bin/python"
REGION="${NPA_E2E_REGION:-eu-north1}"
TENANT_ID="${NPA_E2E_TENANT_ID:-${NPA_TENANT_ID:-${NEBIUS_ACCOUNT_ID:-}}}"
PROJECT_ID="${NPA_E2E_PROJECT_ID:-${NPA_PROJECT_ID:-project-test-00000000}}"
MODEL="${NPA_E2E_GROOT_MODEL:-nvidia/GR00T-N1.7-3B}"
EMBODIMENT_TAG="${NPA_E2E_GROOT_EMBODIMENT:-NEW_EMBODIMENT}"
GENESIS_CHECKPOINT="${NPA_E2E_GENESIS_CHECKPOINT:-/tmp/e2e-container-genesis-retry/teacher/model.pt}"
GENESIS_N_ENVS="${NPA_E2E_COMPOSITION_GENESIS_N_ENVS:-1}"
GENESIS_N_EPISODES="${NPA_E2E_COMPOSITION_GENESIS_N_EPISODES:-0}"
MAX_STEPS="${NPA_E2E_GROOT_MAX_STEPS:-1}"
GLOBAL_BATCH_SIZE="${NPA_E2E_GROOT_BATCH_SIZE:-1}"
DATALOADER_WORKERS="${NPA_E2E_GROOT_DATALOADER_WORKERS:-0}"

TOTAL=0
FAILED=0
GROOT_DEPLOYED=0
GROOT_DESTROYED=0
S3_ENDPOINT=""
S3_PREFIX=""
S3_SIM_URI=""
S3_LEROBOT_URI=""
S3_GROOT_DATA_URI=""
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

run_capture_step() {
  local label="$1"
  local output_file="$2"
  shift 2
  print_command "$@"
  if "$@" >"$output_file" 2>&1; then
    cat "$output_file"
    record_pass "$label"
    return 0
  fi
  cat "$output_file"
  record_fail "$label"
  return 1
}

slugify() {
  printf '%s' "$1" | tr -c 'A-Za-z0-9_-' '-' | sed 's/^-*//;s/-*$//'
}

model_slug() {
  printf '%s' "$1" | sed 's#^ngc://##; s#@.*$##; s#[/:]#--#g'
}

resolve_storage() {
  local values
  values="$("$PYTHON_BIN" - "$PROJECT" "$S3_BUCKET" "$REGION" <<'PY'
import sys
from urllib.parse import urlparse

from npa.clients.config import resolve_terraform_state


def bucket_name(value: str) -> str:
    if value.startswith("s3://"):
        return urlparse(value).netloc
    return value.split("/", 1)[0]


project, explicit_bucket, region = sys.argv[1:4]
state = resolve_terraform_state(project)
bucket = bucket_name(explicit_bucket) or state.bucket
endpoint = state.endpoint or f"https://storage.{region}.nebius.cloud"
print(bucket)
print(endpoint)
print(state.access_key)
print(state.secret_key)
PY
)" || return 1

  S3_BUCKET="$(printf '%s\n' "$values" | sed -n '1p')"
  S3_ENDPOINT="$(printf '%s\n' "$values" | sed -n '2p')"
  export AWS_ENDPOINT_URL="$S3_ENDPOINT"
  export NEBIUS_S3_ENDPOINT="$S3_ENDPOINT"
  AWS_ACCESS_KEY_ID="$(printf '%s\n' "$values" | sed -n '3p')"
  AWS_SECRET_ACCESS_KEY="$(printf '%s\n' "$values" | sed -n '4p')"
  export AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY

  local run_slug
  run_slug="$(slugify "$GROOT_NAME")"
  S3_PREFIX="${NPA_E2E_GROOT_COMPOSITION_S3_PREFIX:-e2e-composition/groot/${run_slug}-$(date +%Y%m%d%H%M%S)-$$}"
  S3_SIM_URI="s3://$S3_BUCKET/$S3_PREFIX/genesis-sim/"
  S3_LEROBOT_URI="s3://$S3_BUCKET/$S3_PREFIX/lerobot/"
  S3_GROOT_DATA_URI="s3://$S3_BUCKET/$S3_PREFIX/groot-data/"
  S3_MODEL_URI="s3://$S3_BUCKET/$S3_PREFIX/model/"
  S3_CHECKPOINT_URI="s3://$S3_BUCKET/$S3_PREFIX/checkpoint/"
  S3_EVAL_URI="s3://$S3_BUCKET/$S3_PREFIX/eval/"
  S3_INFER_URI="s3://$S3_BUCKET/$S3_PREFIX/infer/"

  [ -n "$S3_BUCKET" ] && [ -n "$S3_ENDPOINT" ]
}

s3_uri_has_nonzero_objects() {
  "$PYTHON_BIN" - "$1" <<'PY'
import os
import sys
from urllib.parse import urlparse

import boto3

uri = sys.argv[1]
parsed = urlparse(uri)
s3 = boto3.client(
    "s3",
    endpoint_url=os.environ.get("NEBIUS_S3_ENDPOINT") or os.environ.get("AWS_ENDPOINT_URL") or None,
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID") or None,
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY") or None,
)
count = 0
size = 0
for page in s3.get_paginator("list_objects_v2").paginate(
    Bucket=parsed.netloc,
    Prefix=parsed.path.lstrip("/"),
):
    for obj in page.get("Contents", []):
        count += 1
        size += obj.get("Size", 0)
if count == 0 or size == 0:
    raise SystemExit(f"no non-zero objects found under {uri}")
print({"uri": uri, "objects": count, "bytes": size})
PY
}

s3_json_has_keys() {
  "$PYTHON_BIN" - "$1" "$2" <<'PY'
import json
import os
import sys
from urllib.parse import urlparse

import boto3

uri, filename = sys.argv[1:3]
parsed = urlparse(uri)
prefix = parsed.path.lstrip("/").rstrip("/") + "/"
s3 = boto3.client(
    "s3",
    endpoint_url=os.environ.get("NEBIUS_S3_ENDPOINT") or os.environ.get("AWS_ENDPOINT_URL") or None,
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID") or None,
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY") or None,
)
obj = s3.get_object(Bucket=parsed.netloc, Key=prefix + filename)
payload = json.loads(obj["Body"].read())
if filename == "npa_groot_eval_results.json":
    assert isinstance(payload.get("metrics"), dict)
    assert "episode_count" in payload
if filename == "npa_groot_infer_results.json":
    assert payload.get("status") == "success"
    assert payload.get("trajectory_count", 0) >= 1
print(json.dumps(payload, indent=2)[:2000])
PY
}

cleanup_s3_prefix() {
  if [ -z "$S3_PREFIX" ]; then
    return 0
  fi
  printf '+ cleanup s3://%s/%s/\n' "$S3_BUCKET" "$S3_PREFIX"
  "$PYTHON_BIN" - "$S3_BUCKET" "$S3_PREFIX/" <<'PY'
import os
import sys

import boto3

bucket, prefix = sys.argv[1:3]
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

destroy_groot() {
  if [ "$GROOT_DEPLOYED" -eq 1 ] && [ "$GROOT_DESTROYED" -eq 0 ]; then
    "$CLI" workbench groot -p "$PROJECT" -n "$GROOT_NAME" deploy \
      --destroy \
      --yes \
      --gpu-type "$GROOT_GPU_TYPE" \
      --gpu-preset "$GROOT_GPU_PRESET"
    GROOT_DESTROYED=1
  fi
}

cleanup() {
  local status=$?
  trap - EXIT
  destroy_groot || true
  cleanup_s3_prefix || true
  print_summary
  if [ "$FAILED" -ne 0 ] || [ "$status" -ne 0 ]; then
    exit 1
  fi
  exit 0
}
trap cleanup EXIT

run_step "resolve S3 storage" resolve_storage || exit 1

SIM_SOURCE="${NPA_E2E_GROOT_COMPOSITION_SIM_URI:-}"
if [ -n "$SIM_SOURCE" ]; then
  S3_SIM_URI="$SIM_SOURCE"
  run_step "existing Genesis sim data exists" s3_uri_has_nonzero_objects "$S3_SIM_URI" || exit 1
else
  run_step "Genesis workbench status" "$CLI" workbench genesis -p "$GENESIS_PROJECT" -n "$GENESIS_NAME" status || exit 1
  run_step "Genesis simulate writes S3 demos" "$CLI" workbench genesis -p "$GENESIS_PROJECT" -n "$GENESIS_NAME" simulate \
    --checkpoint "$GENESIS_CHECKPOINT" \
    --n-envs "$GENESIS_N_ENVS" \
    --n-episodes "$GENESIS_N_EPISODES" \
    --allow-failure-demos \
    --output-path "$S3_SIM_URI" || exit 1
  run_step "Genesis S3 handoff exists" s3_uri_has_nonzero_objects "$S3_SIM_URI" || exit 1
fi

run_step "SimToLeRobot converts Genesis S3 to LeRobot S3" "$CLI" adapter convert \
  --input-path "$S3_SIM_URI" \
  --output-path "$S3_LEROBOT_URI" \
  --fps 20 \
  --robot franka_panda \
  --task "Pick and place cube to target" || exit 1
run_step "LeRobot S3 handoff exists" s3_uri_has_nonzero_objects "$S3_LEROBOT_URI" || exit 1

run_step "GR00T convert writes GR00T data S3" "$CLI" workbench groot -p "$PROJECT" -n "$GROOT_NAME" convert \
  --input-path "$S3_LEROBOT_URI" \
  --output-path "$S3_GROOT_DATA_URI" \
  --direction lerobot-to-groot \
  --robot-embodiment "$EMBODIMENT_TAG" || exit 1
run_step "GR00T data S3 handoff exists" s3_uri_has_nonzero_objects "$S3_GROOT_DATA_URI" || exit 1

run_step "Deploy GR00T workbench" "$CLI" workbench groot -p "$PROJECT" -n "$GROOT_NAME" deploy \
  --gpu-type "$GROOT_GPU_TYPE" \
  --gpu-preset "$GROOT_GPU_PRESET" \
  --data-disk-size 200 \
  --no-preemptible \
  --region "$REGION" \
  --project-id "$PROJECT_ID" \
  --tenant-id "$TENANT_ID" || exit 1
GROOT_DEPLOYED=1

run_step "Download pinned GR00T base model" "$CLI" workbench groot -p "$PROJECT" -n "$GROOT_NAME" download \
  --model "$MODEL" \
  --output-path "$S3_MODEL_URI" || exit 1
run_step "GR00T model S3 handoff exists" s3_uri_has_nonzero_objects "$S3_MODEL_URI" || exit 1

LOCAL_MODEL_DIR="/opt/groot/models/$(model_slug "$MODEL")"
FINETUNE_OUTPUT="$(mktemp "${TMPDIR:-/tmp}/npa-groot-composition-finetune.XXXXXX")"
run_capture_step "GR00T finetune consumes converted data" "$FINETUNE_OUTPUT" "$CLI" workbench groot -p "$PROJECT" -n "$GROOT_NAME" finetune \
  --input-path "$S3_GROOT_DATA_URI" \
  --output-path "$S3_CHECKPOINT_URI" \
  --base-model "$LOCAL_MODEL_DIR" \
  --robot-embodiment "$EMBODIMENT_TAG" \
  --num-gpus 1 \
  --max-steps "$MAX_STEPS" \
  --global-batch-size "$GLOBAL_BATCH_SIZE" \
  --dataloader-num-workers "$DATALOADER_WORKERS" \
  --save-steps 1 \
  --save-total-limit 1 \
  --save-only-model \
  --output json || exit 1
run_step "finetune output has no CUDA OOM" bash -c "! grep -Eiq 'CUDA out of memory|out of memory' '$FINETUNE_OUTPUT'" || exit 1
run_step "GR00T checkpoint S3 handoff exists" s3_uri_has_nonzero_objects "$S3_CHECKPOINT_URI" || exit 1

EVAL_OUTPUT="$(mktemp "${TMPDIR:-/tmp}/npa-groot-composition-eval.XXXXXX")"
run_capture_step "GR00T eval consumes checkpoint and data" "$EVAL_OUTPUT" "$CLI" workbench groot -p "$PROJECT" -n "$GROOT_NAME" eval \
  --input-path "$S3_CHECKPOINT_URI" \
  --dataset-path "$S3_GROOT_DATA_URI" \
  --output-path "$S3_EVAL_URI" \
  --robot-embodiment "$EMBODIMENT_TAG" \
  --output json || exit 1
run_step "GR00T eval JSON schema valid" s3_json_has_keys "$S3_EVAL_URI" npa_groot_eval_results.json || exit 1

INFER_OUTPUT="$(mktemp "${TMPDIR:-/tmp}/npa-groot-composition-infer.XXXXXX")"
run_capture_step "GR00T infer consumes checkpoint and data" "$INFER_OUTPUT" "$CLI" workbench groot -p "$PROJECT" -n "$GROOT_NAME" infer \
  --input-path "$S3_CHECKPOINT_URI" \
  --dataset-path "$S3_GROOT_DATA_URI" \
  --output-path "$S3_INFER_URI" \
  --embodiment-tag "$EMBODIMENT_TAG" \
  --inference-mode pytorch \
  --steps 32 \
  --action-horizon 16 \
  --output json || exit 1
run_step "GR00T infer JSON schema valid" s3_json_has_keys "$S3_INFER_URI" npa_groot_infer_results.json || exit 1

run_step "Destroy GR00T workbench" destroy_groot || exit 1
