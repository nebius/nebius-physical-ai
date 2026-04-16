#!/bin/bash
set -euo pipefail

# Thin wrapper around lerobot-train.
# Handles venv activation, env loading, and S3 checkpoint upload on exit.
#
# Usage:
#   ./train.sh --policy.type=act --dataset.repo_id=lerobot/pusht
#
# LeRobot creates the output_dir itself and aborts if it already exists
# (unless --resume=true), so we generate a unique run directory each time.
#
# IMPORTANT: Do NOT export LEROBOT_HOME — LeRobot raises a hard error if
# that env var is set. Use HF_LEROBOT_HOME instead for cache paths.

# Shell-only variable — never exported, so LeRobot never sees LEROBOT_HOME.
_DEPLOY_ROOT="${_DEPLOY_ROOT:-/opt/lerobot}"
VENV="${_DEPLOY_ROOT}/venv/bin/activate"
RUNS_DIR="${_DEPLOY_ROOT}/runs"
RUN_DIR="${RUNS_DIR}/run-$(date +%Y%m%d-%H%M%S)"
DEFAULT_TRAIN_ARGS=(--policy.push_to_hub=false)

if [ -f "${VENV}" ]; then
  source "${VENV}"
fi

if [ -f "${_DEPLOY_ROOT}/.env" ]; then
  set -a
  source "${_DEPLOY_ROOT}/.env"
  set +a
fi

# ── Upload checkpoint on ANY exit (success, Ctrl-C, OOM, error) ────────

_upload_checkpoint() {
  # Don't let upload failures propagate.
  set +e
  if [ -z "${AWS_ACCESS_KEY_ID:-}" ] || ! command -v python3 >/dev/null 2>&1; then
    return
  fi
  if [ ! -d "${RUN_DIR:-}/checkpoints" ]; then
    return
  fi

  local CKPT_BASE="${RUN_DIR}/checkpoints"
  local latest_ckpt

  # LeRobot layout: output_dir/checkpoints/<step>/{pretrained_model/,training_state/}
  # A symlink output_dir/checkpoints/last points to the latest step.
  if [ -L "${CKPT_BASE}/last" ]; then
    latest_ckpt="$(readlink -f "${CKPT_BASE}/last")"
  else
    latest_ckpt="$(find "${CKPT_BASE}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -1)"
  fi

  if [ -z "${latest_ckpt}" ] || [ ! -d "${latest_ckpt}" ]; then
    return
  fi

  local run_name step_name
  run_name="$(basename "${RUN_DIR}")"
  step_name="$(basename "${latest_ckpt}")"
  echo "Uploading checkpoint: ${latest_ckpt}"
  find "${latest_ckpt}" -type f | while read -r fpath; do
    local relpath="${fpath#"${latest_ckpt}/"}"
    python3 "${_DEPLOY_ROOT}/s3_sync.py" upload "${fpath}" \
      --key "checkpoints/${run_name}/${step_name}/${relpath}"
  done
  echo "Checkpoint uploaded to S3: checkpoints/${run_name}/${step_name}/"
}

trap _upload_checkpoint EXIT

echo "Starting lerobot-train at $(date)"
echo "Output directory: ${RUN_DIR}"
echo "Arguments: $*"

# Let LeRobot create the directory — do NOT mkdir it ahead of time.
lerobot-train \
  --output_dir="${RUN_DIR}" \
  "${DEFAULT_TRAIN_ARGS[@]}" \
  "$@"

echo "Training finished at $(date)"
