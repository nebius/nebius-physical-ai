#!/usr/bin/env bash
# Launch continuous golden-eval converge loop in tmux (patch → push → pull → test).
#
# Windows:
#   dashboard  — tails converge.log + latest fleet summary
#   converge   — golden_eval_converge.sh loop (unit tests + serverless fleet)
#   watchdog   — optional; restarts converge if it stalls (--watchdog)
#
# Usage:
#   start_golden_evals_converge_tmux.sh
#   start_golden_evals_converge_tmux.sh --if-dead
#   start_golden_evals_converge_tmux.sh --watchdog
#   start_golden_evals_converge_tmux.sh --with-build --with-cursor-agent --watchdog
#   start_golden_evals_converge_tmux.sh --once --watchdog
#
# Attach:  tmux attach -t golden-evals-converge
# State:   /tmp/golden-evals/converge/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SESSION="${GOLDEN_EVAL_CONVERGE_SESSION:-golden-evals-converge}"
WATCHDOG_SESSION="${GOLDEN_EVAL_WATCHDOG_SESSION:-golden-evals-watchdog}"
STATE_DIR="${GOLDEN_EVAL_STATE_DIR:-/tmp/golden-evals/converge}"
LOG="${STATE_DIR}/converge-tmux.log"
export GOLDEN_EVAL_SOURCE_REF="${GOLDEN_EVAL_SOURCE_REF:-feat/golden-eval-capability-chart}"

IF_DEAD=0
WITH_WATCHDOG=0
WITH_BUILD=0
WITH_CURSOR_AGENT=0
ONCE=0
CONVERGE_EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --if-dead) IF_DEAD=1; shift ;;
    --watchdog) WITH_WATCHDOG=1; shift ;;
    --with-build) WITH_BUILD=1; shift ;;
    --with-cursor-agent) WITH_CURSOR_AGENT=1; shift ;;
    --once) ONCE=1; CONVERGE_EXTRA+=(--once); shift ;;
    --unit-only) CONVERGE_EXTRA+=(--unit-only); shift ;;
    -h | --help)
      sed -n '2,16p' "$0"
      exit 0
      ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "${STATE_DIR}"
chmod +x "${SCRIPT_DIR}/golden_eval_converge.sh" \
  "${SCRIPT_DIR}/golden_eval_autofix.sh" \
  "${SCRIPT_DIR}/golden_eval_watchdog.sh" \
  "${SCRIPT_DIR}/golden_eval_build_loop.sh" \
  "${SCRIPT_DIR}/build_golden_eval_images.sh" \
  "${SCRIPT_DIR}/golden_eval_cursor_patch.sh" \
  "${SCRIPT_DIR}/start_golden_evals_tmux.sh" 2>/dev/null || true

TMUX_ENV="cd \"${ROOT}\" && export GOLDEN_EVAL_STATE_DIR=\"${STATE_DIR}\" && export GOLDEN_EVAL_SOURCE_REF=\"${GOLDEN_EVAL_SOURCE_REF}\" && export GOLDEN_EVAL_AUTO_COMMIT=\"\${GOLDEN_EVAL_AUTO_COMMIT:-1}\" && export GOLDEN_EVAL_AUTO_PUSH=\"\${GOLDEN_EVAL_AUTO_PUSH:-1}\" && export GOLDEN_EVAL_AUTOFIX_SKIP_GIT=\"\${GOLDEN_EVAL_AUTOFIX_SKIP_GIT:-0}\" && export REGISTRY=\"\${REGISTRY:-cr.eu-north1.nebius.cloud/e00cm0vc6t09m0z5gw}\""

if ! command -v tmux >/dev/null; then
  echo "tmux required" >&2
  exit 1
fi

if [[ "${IF_DEAD}" == "1" ]] && tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "TMUX_SESSION=${SESSION} (already running)"
else
  tmux kill-session -t "${SESSION}" 2>/dev/null || true
  tmux new-session -d -s "${SESSION}" -n dashboard \
    "bash -lc '${TMUX_ENV} && echo STATE_DIR=${STATE_DIR} && echo COMPLETE=${STATE_DIR}/golden-evals-complete && while [ ! -f \"${STATE_DIR}/golden-evals-complete\" ]; do clear; echo \"=== golden-eval converge dashboard ===\"; date -u; echo; echo \"--- converge.log (tail) ---\"; tail -n 30 \"${STATE_DIR}/converge.log\" 2>/dev/null || echo \"(no log yet)\"; echo; echo \"--- build-loop.log (tail) ---\"; tail -n 15 \"${STATE_DIR}/build-loop.log\" 2>/dev/null || echo \"(no build log)\"; echo; echo \"--- latest fleet summary ---\"; if [ -f \"${STATE_DIR}/latest-summary.json\" ]; then cat \"${STATE_DIR}/latest-summary.json\"; else echo \"(pending)\"; fi; sleep 15; done; echo DONE; exec bash'"

  if [[ "${WITH_BUILD}" == "1" ]]; then
    tmux new-window -t "${SESSION}" -n build \
      "bash -lc '${TMUX_ENV} && bash \"${SCRIPT_DIR}/golden_eval_build_loop.sh\" 2>&1 | tee -a \"${STATE_DIR}/build-tmux.log\"; exec bash'"
  fi

  if [[ "${WITH_CURSOR_AGENT}" == "1" ]]; then
    tmux new-window -t "${SESSION}" -n agent \
      "bash -lc '${TMUX_ENV} && bash \"${SCRIPT_DIR}/golden_eval_cursor_patch.sh\" 2>&1 | tee -a \"${STATE_DIR}/cursor-patch-tmux.log\"; exec bash'"
  fi

  if [[ "${ONCE}" == "1" ]]; then
    converge_cmd="bash \"${SCRIPT_DIR}/golden_eval_converge.sh\" ${CONVERGE_EXTRA[*]} 2>&1 | tee -a \"${LOG}\"; echo converge_exit=\$? | tee -a \"${LOG}\"; exec bash"
  else
    converge_cmd="while [ ! -f \"${STATE_DIR}/golden-evals-complete\" ]; do bash \"${SCRIPT_DIR}/golden_eval_converge.sh\" ${CONVERGE_EXTRA[*]:-} 2>&1 | tee -a \"${LOG}\"; ec=\$?; echo converge_exit=\$ec | tee -a \"${LOG}\"; [ -f \"${STATE_DIR}/golden-evals-complete\" ] && break; echo restart_in_120s | tee -a \"${LOG}\"; sleep 120; done; echo done; exec bash"
  fi
  tmux new-window -t "${SESSION}" -n converge \
    "bash -lc '${TMUX_ENV} && ${converge_cmd}'"

  echo "TMUX_SESSION=${SESSION}"
fi

if [[ "${WITH_WATCHDOG}" == "1" ]]; then
  if tmux has-session -t "${WATCHDOG_SESSION}" 2>/dev/null; then
    echo "WATCHDOG_SESSION=${WATCHDOG_SESSION} (already running)"
  else
    tmux new-session -d -s "${WATCHDOG_SESSION}" \
      "bash -lc 'bash \"${SCRIPT_DIR}/golden_eval_watchdog.sh\" 2>&1 | tee -a \"${STATE_DIR}/watchdog-tmux.log\"; exec bash'"
    echo "WATCHDOG_SESSION=${WATCHDOG_SESSION}"
  fi
fi

echo "STATE_DIR=${STATE_DIR}"
echo "LOG=${LOG}"
echo "attach: tmux attach -t ${SESSION}"
