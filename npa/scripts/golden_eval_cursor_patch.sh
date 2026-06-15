#!/usr/bin/env bash
# Headless cursor-agent patch loop for golden-eval converge (tmux `agent` window).
#
# Watches patch-request.md written on fleet failure, invokes cursor-agent to fix
# npa/, commits/pushes to feat/golden-eval, and signals converge to retry.
#
# Usage:
#   golden_eval_cursor_patch.sh
#
# Environment:
#   GOLDEN_EVAL_CURSOR_AGENT=0       disable agent loop (idle)
#   GOLDEN_EVAL_AUTO_COMMIT=1        commit agent changes (default in tmux launcher)
#   GOLDEN_EVAL_AUTO_PUSH=1          push after commit
#   CURSOR_AGENT_BIN                 path to cursor-agent
#   GOLDEN_EVAL_CURSOR_POLL_S=30     poll interval when idle
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
STATE_DIR="${GOLDEN_EVAL_STATE_DIR:-/tmp/golden-evals/converge}"
PATCH_REQUEST="${STATE_DIR}/patch-request.md"
LAST_PATCHED="${STATE_DIR}/.last-patched-run-id"
LOG="${STATE_DIR}/cursor-patch.log"
BRANCH="${GOLDEN_EVAL_SOURCE_REF:-feat/golden-eval}"
PYTHON="${ROOT}/npa/.venv/bin/python"
POLL_S="${GOLDEN_EVAL_CURSOR_POLL_S:-30}"
COMPLETE_FILE="${STATE_DIR}/golden-evals-complete"
AUTOFIX="${SCRIPT_DIR}/golden_eval_autofix.sh"

mkdir -p "${STATE_DIR}"

log() {
  printf '[%s] cursor-patch %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${LOG}"
}

agent_bin() {
  local explicit="${CURSOR_AGENT_BIN:-}"
  if [[ -n "${explicit}" && -x "${explicit}" ]]; then
    echo "${explicit}"
    return 0
  fi
  command -v cursor-agent
}

_patch_run_id() {
  [[ -f "${PATCH_REQUEST}" ]] || return 1
  sed -n 's/^\*\*Run ID:\*\* `\([^`]*\)`/\1/p' "${PATCH_REQUEST}" | head -1
}

_build_prompt() {
  local run_id="$1"
  local fail_dir="${STATE_DIR}/failures/${run_id}"
  cat <<EOF
CONTINUE golden-eval converge patch loop (do NOT restart from scratch).

Goal: fix failed container golden evals so the next serverless fleet is 16/16 PASS.

Context:
- Run ID: ${run_id}
- Branch: ${BRANCH}
- Repo: ${ROOT}
- Failure logs: ${fail_dir}/
- Converge log: ${STATE_DIR}/converge.log
- Rebuild queue: ${STATE_DIR}/rebuild-request
- Patch request: ${PATCH_REQUEST}

Instructions:
1. Read ${PATCH_REQUEST} and logs under ${fail_dir}/.
2. Fix npa/ (manifest, smokes, Dockerfiles, registry versions in pyproject.toml + images.py).
3. Run: ${PYTHON} -m pytest npa/tests/smoke/test_golden_eval_*.py -q
4. Commit and push to origin/${BRANCH}.
5. Do not stop until push succeeds or you are blocked by auth.
EOF
}

_run_agent() {
  local run_id="$1"
  local bin
  bin="$(agent_bin)" || {
    log "ERROR: cursor-agent not found (~/.local/bin/cursor-agent)"
    return 1
  }
  local prompt
  prompt="$(_build_prompt "${run_id}")"
  log "cursor-agent start run_id=${run_id} bin=${bin}"
  local ec=0
  "${bin}" \
    --print \
    --trust \
    --force \
    --approve-mcps \
    --workspace "${ROOT}" \
    "${prompt}" 2>&1 | tee -a "${LOG}" || ec=$?
  log "cursor-agent exit=${ec} run_id=${run_id}"
  return "${ec}"
}

_signal_retry() {
  GOLDEN_EVAL_AUTO_COMMIT="${GOLDEN_EVAL_AUTO_COMMIT:-1}" \
  GOLDEN_EVAL_AUTO_PUSH="${GOLDEN_EVAL_AUTO_PUSH:-1}" \
    bash "${AUTOFIX}" "${1}" 2>&1 | tee -a "${LOG}" || true
  touch "${STATE_DIR}/patch-applied"
  echo "${1}" > "${LAST_PATCHED}"
  log "patch-applied signaled for run_id=${1}"
}

log "start branch=${BRANCH} poll=${POLL_S}s complete=${COMPLETE_FILE}"

if [[ "${GOLDEN_EVAL_CURSOR_AGENT:-1}" == "0" ]]; then
  log "GOLDEN_EVAL_CURSOR_AGENT=0 — idle"
  exec bash -lc "while [[ ! -f \"${COMPLETE_FILE}\" ]]; do sleep 3600; done; echo done; exec bash"
fi

while [[ ! -f "${COMPLETE_FILE}" ]]; do
  if [[ -f "${PATCH_REQUEST}" ]]; then
    run_id="$(_patch_run_id || true)"
    if [[ -n "${run_id}" ]]; then
      last=""
      [[ -f "${LAST_PATCHED}" ]] && last="$(cat "${LAST_PATCHED}")"
      if [[ "${run_id}" != "${last}" ]]; then
        if _run_agent "${run_id}"; then
          _signal_retry "${run_id}"
        else
          log "WARN: cursor-agent failed for ${run_id}"
        fi
      fi
    fi
  fi
  sleep "${POLL_S}"
done

log "golden-evals-complete — cursor-patch exiting"
exec bash
