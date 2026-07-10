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

if [ "${#ARGS[@]}" -ne 4 ]; then
  echo "Usage: $0 [--skip-deploy] <project> <name> <gpu-type> <gpu-preset>" >&2
  exit 2
fi

PROJECT="${ARGS[0]}"
NAME="${ARGS[1]}"
GPU_TYPE="${ARGS[2]}"
GPU_PRESET="${ARGS[3]}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NPA_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
CLI="$NPA_DIR/.venv/bin/npa"
PYTHON_BIN="$NPA_DIR/.venv/bin/python"
TOOL="lerobot"

TOTAL=0
FAILED=0
DEPLOY_ATTEMPTED=0
TEARDOWN_DONE=0
WORK_DIR=""
CHECKPOINT=""
CHECKPOINTS_JSON=""
INFER_JSON=""
OBSERVATION_JSON=""

SSH_HOST=""
SSH_USER=""
SSH_KEY=""
SSH_TARGET=""
SSH_OPTS=()

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

run_teardown_step() {
  if [ "$TEARDOWN_DONE" -eq 1 ]; then
    return 0
  fi
  TEARDOWN_DONE=1
  run_npa_step "destroy workbench" deploy --destroy --yes --gpu-type "$GPU_TYPE" --gpu-preset "$GPU_PRESET"
}

cleanup() {
  local status=$?
  trap - EXIT

  if [ "$DEPLOY_ATTEMPTED" -eq 1 ] && [ "$TEARDOWN_DONE" -eq 0 ]; then
    run_teardown_step || true
  fi
  if [ -n "$WORK_DIR" ]; then
    rm -rf "$WORK_DIR"
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

list_checkpoints() {
  print_command "$CLI" workbench "$TOOL" -p "$PROJECT" -n "$NAME" list-checkpoints --output json
  "$CLI" workbench "$TOOL" -p "$PROJECT" -n "$NAME" list-checkpoints --output json | tee "$CHECKPOINTS_JSON"
}

vm_checkpoint_count() {
  "$PYTHON_BIN" - "$CHECKPOINTS_JSON" <<'PY'
import json
import sys

data = json.loads(open(sys.argv[1]).read())
print(len(data.get("vm_checkpoints", [])))
PY
}

select_latest_checkpoint() {
  resolve_ssh || return 1
  CHECKPOINT="$(
    ssh "${SSH_OPTS[@]}" "$SSH_TARGET" \
      "find /opt/lerobot/checkpoints -maxdepth 4 -name pretrained_model -type d -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-"
  )"
  if [ -z "$CHECKPOINT" ]; then
    return 1
  fi
  printf 'Selected checkpoint: %s\n' "$CHECKPOINT"
}

train_if_needed() {
  local count
  count="$(vm_checkpoint_count)" || return 1

  if [ "$count" -gt 0 ]; then
    select_latest_checkpoint
    return "$?"
  fi

  local run_slug
  local job_name
  run_slug="$(slugify "$NAME")"
  job_name="npa-serve-infer-${run_slug}-$(date +%Y%m%d%H%M%S)"
  local train_output="/opt/lerobot/checkpoints/${job_name}"
  CHECKPOINT="${train_output}/checkpoints/last/pretrained_model"
  run_npa_step "train ACT on lerobot/pusht for 50 steps" train \
    --policy-type act \
    --dataset lerobot/pusht \
    --job-name "$job_name" \
    --steps 50 \
    --batch-size 8 \
    --num-workers 4 \
    --output-path "$train_output"
}

wait_for_policy_server() {
  local deadline=$(( $(date +%s) + 180 ))
  local status_json="$WORK_DIR/status.json"

  while [ "$(date +%s)" -lt "$deadline" ]; do
    print_command "$CLI" workbench "$TOOL" -p "$PROJECT" -n "$NAME" status --output json
    if "$CLI" workbench "$TOOL" -p "$PROJECT" -n "$NAME" status --output json > "$status_json"; then
      cat "$status_json"
      if "$PYTHON_BIN" - "$status_json" <<'PY'
import json
import sys

data = json.loads(open(sys.argv[1]).read())
raise SystemExit(0 if data.get("policy_server", {}).get("running") else 1)
PY
      then
        return 0
      fi
    fi
    sleep 5
  done

  return 1
}

create_observation() {
  resolve_ssh || return 1

  print_command ssh "${SSH_OPTS[@]}" "$SSH_TARGET" python3 - "$CHECKPOINT"
  ssh "${SSH_OPTS[@]}" "$SSH_TARGET" python3 - "$CHECKPOINT" > "$OBSERVATION_JSON" <<'PY'
import json
from pathlib import Path
import sys

checkpoint = Path(sys.argv[1])
config_path = checkpoint / "config.json"
if not config_path.exists():
    raise SystemExit(f"missing checkpoint config: {config_path}")

config = json.loads(config_path.read_text())
features = config.get("input_features") or {}
observation = {}

for key, meta in features.items():
    shape = list(meta.get("shape") or [])
    feature_type = str(meta.get("type") or "").upper()

    if feature_type == "VISUAL" or "image" in key:
        if len(shape) == 3:
            channels, height, width = shape
            observation[key] = [
                [[0 for _ in range(channels)] for _ in range(width)]
                for _ in range(height)
            ]
        else:
            observation[key] = [[[0, 0, 0]]]
    else:
        size = int(shape[0]) if shape else 1
        observation[key] = [0.0 for _ in range(size)]

if not observation:
    observation["observation.state"] = [0.0, 0.0]

print(json.dumps(observation))
PY
}

run_infer() {
  print_command "$CLI" workbench "$TOOL" -p "$PROJECT" -n "$NAME" infer --observation "$OBSERVATION_JSON" --output json
  "$CLI" workbench "$TOOL" -p "$PROJECT" -n "$NAME" infer --observation "$OBSERVATION_JSON" --output json | tee "$INFER_JSON"
}

verify_infer_response() {
  "$PYTHON_BIN" - "$INFER_JSON" <<'PY'
import json
import sys

data = json.loads(open(sys.argv[1]).read())
if "actions" not in data or not isinstance(data["actions"], list):
    raise SystemExit(f"missing actions list in response: {data}")
if "checkpoint" not in data:
    raise SystemExit(f"missing checkpoint in response: {data}")
if "inference_ms" not in data:
    raise SystemExit(f"missing inference_ms in response: {data}")
PY
}

run_step "npa CLI exists" test -x "$CLI" || exit 1
run_step "python exists" test -x "$PYTHON_BIN" || exit 1

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/npa-serve-infer-e2e.XXXXXX")"
CHECKPOINTS_JSON="$WORK_DIR/checkpoints.json"
INFER_JSON="$WORK_DIR/infer.json"
OBSERVATION_JSON="$WORK_DIR/observation.json"

if [ "$SKIP_DEPLOY" -eq 1 ]; then
  printf 'SKIP: deploy workbench (--skip-deploy)\n'
else
  DEPLOY_ATTEMPTED=1
  run_npa_step "deploy workbench" deploy --gpu-type "$GPU_TYPE" --gpu-preset "$GPU_PRESET" || exit 1
fi

run_npa_step "status exits 0" status || exit 1
run_step "list checkpoints on VM" list_checkpoints || exit 1
run_step "select or create checkpoint" train_if_needed || exit 1
run_npa_step "serve checkpoint" serve --checkpoint "$CHECKPOINT" --env-type pusht || exit 1
run_step "wait for policy endpoint healthy" wait_for_policy_server || exit 1
run_step "create dummy observation JSON" create_observation || exit 1
run_step "run inference" run_infer || exit 1
run_step "verify inference response" verify_infer_response || exit 1

if [ "$SKIP_DEPLOY" -eq 1 ]; then
  printf 'SKIP: destroy workbench (--skip-deploy)\n'
else
  run_teardown_step || exit 1
fi
