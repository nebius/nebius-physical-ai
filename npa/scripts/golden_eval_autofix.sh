#!/usr/bin/env bash
# Pull the latest branch tip after a failed golden-eval fleet run (picks up pushed patches).
# Optionally commit/push local golden-eval changes when GOLDEN_EVAL_AUTO_COMMIT/PUSH=1.
#
# Usage:
#   golden_eval_autofix.sh [run_id]
set -euo pipefail

RUN_ID="${1:-unknown}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${ROOT}/npa/.venv/bin/python"
BRANCH="${GOLDEN_EVAL_SOURCE_REF:-feat/golden-eval}"
STATE_DIR="${GOLDEN_EVAL_STATE_DIR:-/tmp/golden-evals/converge}"
LOG="${STATE_DIR}/autofix.log"

mkdir -p "${STATE_DIR}"

log() {
  printf '[%s] autofix %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${LOG}"
}

_maybe_commit_push() {
  if [[ "${GOLDEN_EVAL_AUTO_COMMIT:-0}" != "1" && "${GOLDEN_EVAL_AUTO_PUSH:-0}" != "1" ]]; then
    return 0
  fi
  if [[ -n "$(git -C "${ROOT}" status --porcelain -- npa docs)" ]]; then
    if [[ "${GOLDEN_EVAL_AUTO_COMMIT:-0}" == "1" ]]; then
      log "commit local golden-eval changes run_id=${RUN_ID}"
      git -C "${ROOT}" add npa docs 2>&1 | tee -a "${LOG}"
      git -C "${ROOT}" commit -m "golden-eval: autofix ${RUN_ID}" 2>&1 | tee -a "${LOG}" || true
    else
      log "WARN dirty npa/docs tree; set GOLDEN_EVAL_AUTO_COMMIT=1 to commit before pull"
    fi
  fi
  if [[ "${GOLDEN_EVAL_AUTO_PUSH:-0}" == "1" ]]; then
    local ahead
    ahead="$(git -C "${ROOT}" rev-list --count "origin/${BRANCH}..HEAD" 2>/dev/null || echo 0)"
    if [[ "${ahead}" -gt 0 ]]; then
      log "push ${ahead} local commit(s) to origin/${BRANCH}"
      git -C "${ROOT}" push origin "HEAD:${BRANCH}" 2>&1 | tee -a "${LOG}" || true
    fi
  fi
}

log "run_id=${RUN_ID} branch=${BRANCH}"
_maybe_commit_push
if [[ "${GOLDEN_EVAL_AUTOFIX_SKIP_GIT:-0}" == "1" ]]; then
  log "skip git sync (GOLDEN_EVAL_AUTOFIX_SKIP_GIT=1)"
else
  log "fetch origin/${BRANCH}"
  git -C "${ROOT}" fetch origin "${BRANCH}" 2>&1 | tee -a "${LOG}"
  git -C "${ROOT}" checkout "${BRANCH}" 2>&1 | tee -a "${LOG}" || true
  git -C "${ROOT}" reset --hard "origin/${BRANCH}" 2>&1 | tee -a "${LOG}"
  log "HEAD $(git -C "${ROOT}" log -1 --oneline)"
fi

if [[ -x "${PYTHON}" ]]; then
  log "pip install -e npa"
  "${PYTHON}" -m pip install -e "${ROOT}/npa" -q 2>&1 | tee -a "${LOG}" || true
  log "validate manifest"
  "${PYTHON}" "${ROOT}/npa/scripts/run_golden_evals.py" validate 2>&1 | tee -a "${LOG}"
fi
