#!/usr/bin/env bash
# Run NPA agent unit smoke + live e2e checks in tmux.
#
# Usage:
#   ./npa/scripts/start_agent_live_tmux.sh --project rtxpro --name agent --verify
#   ./npa/scripts/start_agent_live_tmux.sh --bootstrap --project rtxpro --name agent --verify
#   ./npa/scripts/start_agent_live_tmux.sh --project rtxpro --name agent --verify --browser-e2e
#   ./npa/scripts/start_agent_live_tmux.sh --dry-run
#
# Attach:  tmux attach -t npa-agent-live
# Logs:    /tmp/npa-agent-live/<session>/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ -n "${NPA_AGENT_PYTHON:-}" && -x "${NPA_AGENT_PYTHON}" ]]; then
  PYTHON="${NPA_AGENT_PYTHON}"
elif [[ -x "${ROOT}/npa/.venv/bin/python" ]]; then
  PYTHON="${ROOT}/npa/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="$(command -v python3)"
else
  echo "Missing python; set NPA_AGENT_PYTHON or create ${ROOT}/npa/.venv" >&2
  exit 1
fi

NPA_BIN="${ROOT}/npa/.venv/bin/npa"
SESSION="npa-agent-live"
PROJECT="${NPA_AGENT_PROJECT:-rtxpro}"
NAME="${NPA_AGENT_NAME:-agent}"
BOOTSTRAP=0
VERIFY=0
DRY_RUN=0
CHAT_LIVE="${NPA_AGENT_CHAT_LIVE:-0}"
BROWSER_E2E="${NPA_AGENT_BROWSER_E2E:-0}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --project NAME     NPA project alias (default: ${PROJECT})
  --name NAME        Agent deployment name (default: ${NAME})
  --session NAME     tmux session name (default: ${SESSION})
  --bootstrap        Run npa agent bootstrap before live checks
  --verify           Run npa agent verify-live after pytest live suite
  --chat-live        Set NPA_AGENT_CHAT_LIVE=1 for Token Factory chat smoke
  --browser-e2e      Run Cypress live browser checks after HTTP live tests
  --dry-run          Print planned commands without launching tmux
  --help             Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project)
      PROJECT="$2"
      shift 2
      ;;
    --name)
      NAME="$2"
      shift 2
      ;;
    --session)
      SESSION="$2"
      shift 2
      ;;
    --bootstrap)
      BOOTSTRAP=1
      shift
      ;;
    --verify)
      VERIFY=1
      shift
      ;;
    --chat-live)
      CHAT_LIVE=1
      shift
      ;;
    --browser-e2e)
      BROWSER_E2E=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

LOG_ROOT="/tmp/npa-agent-live/${SESSION}"
mkdir -p "${LOG_ROOT}"

SMOKE_CMD="cd ${ROOT} && ${PYTHON} -m pytest npa/tests/smoke/test_agent_smoke.py npa/tests/cli/test_agent.py -q 2>&1 | tee ${LOG_ROOT}/smoke.log; ec=\${PIPESTATUS[0]}; echo \${ec} > ${LOG_ROOT}/smoke.exit"
LIVE_CMD="cd ${ROOT} && export NPA_INTEGRATION_E2E=1 NPA_AGENT_LIVE=1 NPA_AGENT_PROJECT=${PROJECT} NPA_AGENT_NAME=${NAME} NPA_AGENT_CHAT_LIVE=${CHAT_LIVE} && ${PYTHON} -m pytest npa/tests/e2e/test_agent_live.py -q 2>&1 | tee ${LOG_ROOT}/live.log; ec=\${PIPESTATUS[0]}; echo \${ec} > ${LOG_ROOT}/live.exit"
VERIFY_CMD="cd ${ROOT} && export NPA_INTEGRATION_E2E=1 NPA_AGENT_LIVE=1 NPA_AGENT_PROJECT=${PROJECT} NPA_AGENT_NAME=${NAME} NPA_AGENT_CHAT_LIVE=${CHAT_LIVE} && ${NPA_BIN} agent verify-live --project ${PROJECT} --name ${NAME} 2>&1 | tee ${LOG_ROOT}/verify.log; ec=\${PIPESTATUS[0]}; echo \${ec} > ${LOG_ROOT}/verify.exit"
BOOTSTRAP_CMD="cd ${ROOT} && NPA_SSH_KEY=\${NPA_SSH_KEY:-\$HOME/.ssh/id_ed25519} ${NPA_BIN} agent bootstrap --project ${PROJECT} --name ${NAME} 2>&1 | tee ${LOG_ROOT}/bootstrap.log; ec=\${PIPESTATUS[0]}; echo \${ec} > ${LOG_ROOT}/bootstrap.exit"
BROWSER_CMD="cd ${ROOT} && export NPA_AGENT_PROJECT=${PROJECT} NPA_AGENT_NAME=${NAME} && bash npa/scripts/run_agent_cypress.sh --live --project ${PROJECT} --name ${NAME} 2>&1 | tee ${LOG_ROOT}/browser.log; ec=\${PIPESTATUS[0]}; echo \${ec} > ${LOG_ROOT}/browser.exit"
DASH_CMD="watch -n 2 'echo session=${SESSION}; ls -1 ${LOG_ROOT}/*.exit 2>/dev/null | while read f; do printf \"%s: \" \"\$(basename \"\${f}\")\"; cat \"\${f}\"; echo; done; echo; tail -n 3 ${LOG_ROOT}/live.log 2>/dev/null || true'"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "TMUX_SESSION=${SESSION}"
  echo "LOG_ROOT=${LOG_ROOT}"
  echo "SMOKE: ${SMOKE_CMD}"
  if [[ "${BOOTSTRAP}" -eq 1 ]]; then
    echo "BOOTSTRAP: ${BOOTSTRAP_CMD}"
  fi
  echo "LIVE: ${LIVE_CMD}"
  if [[ "${VERIFY}" -eq 1 ]]; then
    echo "VERIFY: ${VERIFY_CMD}"
  fi
  if [[ "${BROWSER_E2E}" -eq 1 ]]; then
    echo "BROWSER: ${BROWSER_CMD}"
  fi
  echo "DASHBOARD: ${DASH_CMD}"
  exit 0
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required" >&2
  exit 1
fi

tmux kill-session -t "${SESSION}" 2>/dev/null || true
tmux new-session -d -s "${SESSION}" -n smoke "bash -lc '${SMOKE_CMD}; exec bash'"
if [[ "${BOOTSTRAP}" -eq 1 ]]; then
  tmux new-window -t "${SESSION}" -n bootstrap "bash -lc '${BOOTSTRAP_CMD}; exec bash'"
fi
tmux new-window -t "${SESSION}" -n live "bash -lc '${LIVE_CMD}; if [[ ${VERIFY} -eq 1 ]]; then ${VERIFY_CMD}; fi; if [[ ${BROWSER_E2E} -eq 1 ]]; then ${BROWSER_CMD}; fi; exec bash'"
tmux new-window -t "${SESSION}" -n dashboard "bash -lc '${DASH_CMD}'"

echo "TMUX_SESSION=${SESSION}"
echo "LOG_ROOT=${LOG_ROOT}"
echo "Attach with: tmux attach -t ${SESSION}"
