#!/usr/bin/env bash
# Headless cursor-agent patch loop for sim2real converge (tmux `agent` window).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=lib/operator-config.sh
source "${SCRIPT_DIR}/lib/operator-config.sh"
ROOT="$(npa_repo_root "${SCRIPT_DIR}")"
STATE_DIR="${CONVERGE_STATE_DIR:-/tmp/sim2real-cluster/converge}"
PATCH_REQUEST="${STATE_DIR}/patch-request.md"
LAST_PATCHED="${STATE_DIR}/.last-patched-run-id"
LOG="${STATE_DIR}/cursor-patch.log"
BRANCH="${NPA_SOURCE_REF:-feat/sim2real-mandatory-stages}"
PYTHON="${ROOT}/npa/.venv/bin/python"
POLL_S="${CONVERGE_CURSOR_POLL_S:-30}"

if [ -f "${SCRIPT_DIR}/env.local" ]; then
  set -a
  # shellcheck source=/dev/null
  source "${SCRIPT_DIR}/env.local"
  set +a
fi

mkdir -p "${STATE_DIR}"

log() {
  printf '[%s] cursor-patch %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${LOG}"
}

agent_bin() {
  local explicit="${CURSOR_AGENT_BIN:-}"
  if [ -n "${explicit}" ] && [ -x "${explicit}" ]; then
    echo "${explicit}"
    return 0
  fi
  command -v cursor-agent
}

_patch_run_id() {
  [ -f "${PATCH_REQUEST}" ] || return 1
  sed -n 's/^\*\*Run ID:\*\* `\([^`]*\)`/\1/p' "${PATCH_REQUEST}" | head -1
}

_run_agent() {
  local run_id="$1"
  local bin prompt ec=0
  bin="$(agent_bin)" || {
    log "ERROR: cursor-agent not found"
    return 1
  }
  prompt="$(cat <<EOF
CONTINUE sim2real converge patch loop. Fix failed run ${run_id} on branch ${BRANCH}.
Read ${PATCH_REQUEST} and ${STATE_DIR}/failures/${run_id}/. Patch npa/, run pytest, commit + push to origin/${BRANCH}.
EOF
)"
  log "cursor-agent start run_id=${run_id}"
  "${bin}" --print --trust --force --approve-mcps --workspace "${ROOT}" "${prompt}" \
    2>&1 | tee -a "${LOG}" || ec=$?
  log "cursor-agent exit=${ec} run_id=${run_id}"
  return "${ec}"
}

_signal_retry() {
  bash "${SCRIPT_DIR}/converge-autofix.sh" sync 2>&1 | tee -a "${LOG}" || true
  touch "${STATE_DIR}/patch-applied"
  echo "${1}" > "${LAST_PATCHED}"
  log "patch-applied signaled for run_id=${1}"
}

log "start branch=${BRANCH} poll=${POLL_S}s"

if [[ "${CONVERGE_CURSOR_AGENT:-1}" == "0" ]]; then
  exec bash -lc 'while [ ! -f "'"${STATE_DIR}"'/overnight-complete" ]; do sleep 3600; done'
fi

while [ ! -f "${STATE_DIR}/overnight-complete" ]; do
  if [ -f "${PATCH_REQUEST}" ]; then
    run_id="$(_patch_run_id || true)"
    if [ -n "${run_id}" ]; then
      last=""
      [ -f "${LAST_PATCHED}" ] && last="$(cat "${LAST_PATCHED}")"
      if [ "${run_id}" != "${last}" ] && _run_agent "${run_id}"; then
        _signal_retry "${run_id}"
      fi
    fi
  fi
  sleep "${POLL_S}"
done

log "overnight-complete — cursor-patch exiting"
exec bash
