#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "check-import" || "${1:-}" == "train" || "${1:-}" == "eval" || "${1:-}" == "validate-checkpoint" || "${1:-}" == "feedback-step" ]]; then
  command="${1:-}"
  shift
  exec python -m npa.workbench.lerobot.policy_container "${command}" "$@"
fi

if [[ "${1:-}" == "serve" || $# -eq 0 ]]; then
  if [[ "${1:-}" == "serve" ]]; then
    shift
  fi
  exec python -m npa.workbench.lerobot.policy_container serve "$@"
fi

exec "$@"
