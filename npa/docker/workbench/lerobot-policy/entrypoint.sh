#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "feedback-step" ]]; then
  shift
  exec python -m npa.workbench.lerobot.policy_container feedback-step "$@"
fi

if [[ "${1:-}" == "serve" || $# -eq 0 ]]; then
  if [[ "${1:-}" == "serve" ]]; then
    shift
  fi
  exec python -m npa.workbench.lerobot.policy_container serve "$@"
fi

exec "$@"
