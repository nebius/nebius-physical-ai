#!/usr/bin/env bash
# Launch converge-until-success in tmux (dashboard + converge + cursor-agent).
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
WITH_AGENT=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --if-dead) IF_DEAD=1; shift ;;
    --watchdog) WITH_WATCHDOG=1; shift ;;
    --no-agent) WITH_AGENT=0; shift ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "${STATE_DIR}"
chmod +x "${SCRIPT_DIR}/watchdog-converge.sh" \
  "${SCRIPT_DIR}/converge-autofix.sh" \
  "${SCRIPT_DIR}/converge-cursor-patch.sh" \
  "${SCRIPT_DIR}/converge-until-success.sh" 2>/dev/null || true

if [ -f "${SCRIPT_DIR}/env.local" ]; then
  set -a
  # shellcheck source=/dev/null
  source "${SCRIPT_DIR}/env.local"
  set +a
fi

TMUX_ENV="cd \"${ROOT}\" && export KUBECONFIG=\"\${KUBECONFIG:-}\" && export CONVERGE_STATE_DIR=\"${STATE_DIR}\""

_ensure_agent_window() {
  [[ "${WITH_AGENT}" == "1" ]] || return 0
  if tmux list-windows -t "${SESSION}" -F '#{window_name}' 2>/dev/null | grep -qx agent; then
    return 0
  fi
  tmux new-window -t "${SESSION}" -n agent \
    "bash -lc '${TMUX_ENV} && if [ -f \"${SCRIPT_DIR}/env.local\" ]; then set -a; source \"${SCRIPT_DIR}/env.local\"; set +a; fi; bash \"${SCRIPT_DIR}/converge-cursor-patch.sh\" 2>&1 | tee -a \"${STATE_DIR}/cursor-patch-tmux.log\"; exec bash'"
}

if [ "${IF_DEAD}" = "1" ] && tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "TMUX_SESSION=${SESSION} (already running)"
  _ensure_agent_window
else
  tmux kill-session -t "${SESSION}" 2>/dev/null || true
  tmux new-session -d -s "${SESSION}" -n dashboard \
    "bash -lc '${TMUX_ENV} && while [ ! -f \"${STATE_DIR}/overnight-complete\" ]; do clear; echo \"=== sim2real dashboard ===\"; date -u; head -30 \"${STATE_DIR}/patch-request.md\" 2>/dev/null || echo \"(no patch request)\"; echo; tail -20 \"${STATE_DIR}/converge.log\" 2>/dev/null; sleep 15; done; exec bash'"
  tmux new-window -t "${SESSION}" -n converge \
    "bash -lc '${TMUX_ENV} && if [ -f \"${SCRIPT_DIR}/env.local\" ]; then set -a; source \"${SCRIPT_DIR}/env.local\"; set +a; fi; while [ ! -f \"${STATE_DIR}/overnight-complete\" ]; do bash \"${SCRIPT_DIR}/converge-until-success.sh\" 2>&1 | tee -a ${LOG}; ec=\$?; [ -f \"${STATE_DIR}/overnight-complete\" ] && break; sleep 120; done; exec bash'"
  if [[ "${WITH_AGENT}" == "1" ]]; then
    tmux new-window -t "${SESSION}" -n agent \
      "bash -lc '${TMUX_ENV} && if [ -f \"${SCRIPT_DIR}/env.local\" ]; then set -a; source \"${SCRIPT_DIR}/env.local\"; set +a; fi; bash \"${SCRIPT_DIR}/converge-cursor-patch.sh\" 2>&1 | tee -a \"${STATE_DIR}/cursor-patch-tmux.log\"; exec bash'"
  fi
  echo "TMUX_SESSION=${SESSION}"
fi

if [ "${WITH_WATCHDOG}" = "1" ]; then
  tmux has-session -t "${WATCHDOG_SESSION}" 2>/dev/null || \
    tmux new-session -d -s "${WATCHDOG_SESSION}" \
      "bash -lc 'bash \"${SCRIPT_DIR}/watchdog-converge.sh\" 2>&1 | tee -a ${STATE_DIR}/watchdog-tmux.log; exec bash'"
fi

echo "attach: tmux attach -t ${SESSION}"
