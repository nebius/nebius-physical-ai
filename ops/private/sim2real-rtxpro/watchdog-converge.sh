#!/usr/bin/env bash
# Keep sim2real converge loop alive overnight: restart if tmux/converge stalls.
#
# Usage:
#   watchdog-converge.sh              # loop forever (or until overnight-complete)
#   start-converge-tmux.sh --watchdog # launch this in sim2real-watchdog tmux
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=lib/operator-env.sh
source "${SCRIPT_DIR}/lib/operator-env.sh"
# shellcheck source=lib/operator-config.sh
source "${SCRIPT_DIR}/lib/operator-config.sh"
ROOT="$(npa_repo_root "${SCRIPT_DIR}")"

CONVERGE_SESSION="${CONVERGE_TMUX_SESSION:-sim2real-converge}"
WATCHDOG_SESSION="${CONVERGE_WATCHDOG_SESSION:-sim2real-watchdog}"
POLL_S="${CONVERGE_WATCHDOG_POLL_S:-120}"
STATE_DIR="/tmp/sim2real-cluster/converge"
COMPLETE_FILE="${STATE_DIR}/overnight-complete"
LOG="${STATE_DIR}/watchdog.log"
mkdir -p "${STATE_DIR}"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${LOG}"
}

converge_running() {
  pgrep -f "${SCRIPT_DIR}/converge-until-success.sh" >/dev/null 2>&1 \
    || pgrep -f "${SCRIPT_DIR}/monitor-k8s-job.sh sim2real-converge" >/dev/null 2>&1
}

kick_converge_in_tmux() {
  if ! command -v tmux >/dev/null; then
    return 1
  fi
  local target="${CONVERGE_SESSION}:converge"
  if ! tmux list-windows -t "${CONVERGE_SESSION}" -F '#{window_name}' 2>/dev/null | grep -qx converge; then
    target="${CONVERGE_SESSION}"
  fi
  tmux send-keys -t "${target}" C-c 2>/dev/null || true
  sleep 2
  tmux send-keys -t "${target}" \
    "cd \"${ROOT}\" && bash \"${SCRIPT_DIR}/converge-until-success.sh\" 2>&1 | tee -a \"${STATE_DIR}/tmux-launch.log\"" Enter
}

log "watchdog start poll=${POLL_S}s converge_session=${CONVERGE_SESSION}"

while [ ! -f "${COMPLETE_FILE}" ]; do
  if converge_running; then
    log "ok: converge or monitor active"
  elif tmux has-session -t "${CONVERGE_SESSION}" 2>/dev/null; then
    log "WARN: tmux alive but converge idle — restarting converge in session"
    kick_converge_in_tmux || bash "${SCRIPT_DIR}/start-converge-tmux.sh" --if-dead
  else
    log "WARN: tmux session missing — starting converge"
    bash "${SCRIPT_DIR}/start-converge-tmux.sh" --if-dead
  fi
  sleep "${POLL_S}"
done

log "overnight-complete — watchdog exiting"
