#!/usr/bin/env bash
# Keep golden-eval converge alive: restart if tmux/converge stalls.
#
# Usage:
#   golden_eval_watchdog.sh
#   start_golden_evals_converge_tmux.sh --watchdog
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONVERGE_SESSION="${GOLDEN_EVAL_CONVERGE_SESSION:-golden-evals-converge}"
POLL_S="${GOLDEN_EVAL_WATCHDOG_POLL_S:-120}"
STATE_DIR="${GOLDEN_EVAL_STATE_DIR:-/tmp/golden-evals/converge}"
COMPLETE_FILE="${STATE_DIR}/golden-evals-complete"
LOG="${STATE_DIR}/watchdog.log"

mkdir -p "${STATE_DIR}"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${LOG}"
}

converge_running() {
  pgrep -f "${SCRIPT_DIR}/golden_eval_converge.sh" >/dev/null 2>&1
}

kick_converge_in_tmux() {
  tmux send-keys -t "${CONVERGE_SESSION}" C-c 2>/dev/null || true
  sleep 2
  tmux send-keys -t "${CONVERGE_SESSION}" \
    "cd \"${ROOT}\" && bash \"${SCRIPT_DIR}/golden_eval_converge.sh\" 2>&1 | tee -a \"${STATE_DIR}/converge-tmux.log\"" Enter
}

log "watchdog start poll=${POLL_S}s session=${CONVERGE_SESSION}"

while [[ ! -f "${COMPLETE_FILE}" ]]; do
  if converge_running; then
    log "ok: converge active"
  elif tmux has-session -t "${CONVERGE_SESSION}" 2>/dev/null; then
    log "WARN: tmux alive but converge idle — restarting"
    kick_converge_in_tmux || bash "${SCRIPT_DIR}/start_golden_evals_converge_tmux.sh" --if-dead
  else
    log "WARN: converge session missing — starting"
    bash "${SCRIPT_DIR}/start_golden_evals_converge_tmux.sh" --if-dead
  fi
  sleep "${POLL_S}"
done

log "golden-evals-complete — watchdog exiting"
