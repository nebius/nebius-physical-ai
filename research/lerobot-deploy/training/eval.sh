#!/bin/bash
set -euo pipefail

# Thin wrapper around lerobot-eval.
#
# Usage:
#   ./eval.sh --policy.path=/opt/lerobot/runs/run-XXX/checkpoints/050000/pretrained_model
#
# Note: env-specific extras (e.g. lerobot[pusht]) must be installed.
# The default cloud-init installs lerobot[pusht] alongside the base package.

_DEPLOY_ROOT="${_DEPLOY_ROOT:-/opt/lerobot}"
VENV="${_DEPLOY_ROOT}/venv/bin/activate"

if [ -f "${VENV}" ]; then
  source "${VENV}"
fi

if [ -f "${_DEPLOY_ROOT}/.env" ]; then
  set -a
  source "${_DEPLOY_ROOT}/.env"
  set +a
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

echo "Starting lerobot-eval at $(date)"
echo "Arguments: $*"

lerobot-eval "$@"
