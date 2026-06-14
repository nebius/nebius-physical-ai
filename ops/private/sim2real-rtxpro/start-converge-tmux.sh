#!/usr/bin/env bash
# Launch converge-until-success in a detached tmux session (full operator access).
#
# Usage:
#   start-converge-tmux.sh              # (re)start converge session
#   start-converge-tmux.sh --if-dead    # start only when session missing
#   start-converge-tmux.sh --watchdog   # also start overnight watchdog session
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=lib/operator-env.sh
source "${SCRIPT_DIR}/lib/operator-env.sh"
# shellcheck source=lib/operator-config.sh
source "${SCRIPT_DIR}/lib/operator-config.sh"
ROOT="$(npa_repo_root "${SCRIPT_DIR}")"
SESSION="${CONVERGE_TMUX_SESSION:-sim2real-converge}"
WATCHDOG_SESSION="${CONVERGE_WATCHDOG_SESSION:-sim2real-watchdog}"
LOG="/tmp/sim2real-cluster/converge/tmux-launch.log"
STATE_DIR="/tmp/sim2real-cluster/converge"
IF_DEAD=0
WITH_WATCHDOG=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --if-dead) IF_DEAD=1; shift ;;
    --watchdog) WITH_WATCHDOG=1; shift ;;
    -h | --help)
      sed -n '2,8p' "$0"
      exit 0
      ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "${STATE_DIR}"

if ! command -v tmux >/dev/null; then
  echo "tmux required" >&2
  exit 1
fi

if [ "${IF_DEAD}" = "1" ] && tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "TMUX_SESSION=${SESSION} (already running)"
else
  tmux kill-session -t "${SESSION}" 2>/dev/null || true
  tmux new-session -d -s "${SESSION}" \
    "bash -lc 'cd \"${ROOT}\" && export KUBECONFIG=\"\${KUBECONFIG:-}\" && while [ ! -f \"${STATE_DIR}/overnight-complete\" ]; do bash \"${SCRIPT_DIR}/converge-until-success.sh\" 2>&1 | tee -a ${LOG}; ec=\$?; echo converge_exit=\$ec | tee -a ${LOG}; [ -f \"${STATE_DIR}/overnight-complete\" ] && break; echo restart_in_120s | tee -a ${LOG}; sleep 120; done; echo done; exec bash'"
  echo "TMUX_SESSION=${SESSION}"
fi

if [ "${WITH_WATCHDOG}" = "1" ]; then
  chmod +x "${SCRIPT_DIR}/watchdog-converge.sh" "${SCRIPT_DIR}/converge-autofix.sh" 2>/dev/null || true
  if tmux has-session -t "${WATCHDOG_SESSION}" 2>/dev/null; then
    echo "WATCHDOG_SESSION=${WATCHDOG_SESSION} (already running)"
  else
    tmux new-session -d -s "${WATCHDOG_SESSION}" \
      "bash -lc 'bash \"${SCRIPT_DIR}/watchdog-converge.sh\" 2>&1 | tee -a ${STATE_DIR}/watchdog-tmux.log; exec bash'"
    echo "WATCHDOG_SESSION=${WATCHDOG_SESSION}"
  fi
fi

echo "LOG=${LOG}"
echo "attach: tmux attach -t ${SESSION}"
