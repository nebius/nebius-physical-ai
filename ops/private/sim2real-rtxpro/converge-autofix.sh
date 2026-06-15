#!/usr/bin/env bash
# Post-failure / pre-attempt sync for Cursor patch → push → retry loop.
set -euo pipefail

RUN_ID="${1:-sync}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=lib/operator-config.sh
source "${SCRIPT_DIR}/lib/operator-config.sh"
ROOT="$(npa_repo_root "${SCRIPT_DIR}")"
PYTHON="${ROOT}/npa/.venv/bin/python"
BRANCH="${NPA_SOURCE_REF:-feat/sim2real-mandatory-stages}"
STATE_DIR="${CONVERGE_STATE_DIR:-/tmp/sim2real-cluster/converge}"
LOG="${STATE_DIR}/autofix.log"
PATCH_REQUEST="${STATE_DIR}/patch-request.md"

mkdir -p "${STATE_DIR}"

log() {
  printf '[%s] autofix %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${LOG}"
}

_write_patch_request() {
  local run_id="$1"
  local fail_dir="${STATE_DIR}/failures/${run_id}"
  local orch="${fail_dir}/orchestrator.log"
  local tail=""
  if [ -f "${orch}" ]; then
    tail="$(tail -40 "${orch}" 2>/dev/null || true)"
  fi
  cat > "${PATCH_REQUEST}" <<EOF
# Sim2Real converge — patch request (for Cursor)

**Run ID:** \`${run_id}\`
**Branch:** \`${BRANCH}\`
**Repo:** \`${ROOT}\`
**When:** $(date -u +%Y-%m-%dT%H:%M:%SZ)

## Cursor workflow

1. Read failure logs under \`${fail_dir}/\`
2. Patch \`npa/\` (workflow code) and/or \`ops/private/sim2real-rtxpro/\`
3. **Commit + push** to \`origin/${BRANCH}\`
4. Tmux retries automatically; or run: \`touch ${STATE_DIR}/patch-applied\`

## Orchestrator tail

\`\`\`
${tail}
\`\`\`
EOF
  log "patch-request -> ${PATCH_REQUEST}"
}

_maybe_commit_push() {
  if [[ "${CONVERGE_AUTO_COMMIT:-0}" != "1" && "${CONVERGE_AUTO_PUSH:-0}" != "1" ]]; then
    return 0
  fi
  if [[ -n "$(git -C "${ROOT}" status --porcelain -- npa ops/private/sim2real-rtxpro docs 2>/dev/null)" ]]; then
    if [[ "${CONVERGE_AUTO_COMMIT:-0}" == "1" ]]; then
      log "commit local sim2real changes run_id=${RUN_ID}"
      git -C "${ROOT}" add -A npa ops/private/sim2real-rtxpro docs 2>&1 | tee -a "${LOG}" || true
      git -C "${ROOT}" commit -m "sim2real: converge autofix ${RUN_ID}" 2>&1 | tee -a "${LOG}" || true
    fi
  fi
  if [[ "${CONVERGE_AUTO_PUSH:-0}" == "1" ]]; then
    local ahead
    ahead="$(git -C "${ROOT}" rev-list --count "origin/${BRANCH}..HEAD" 2>/dev/null || echo 0)"
    if [[ "${ahead}" -gt 0 ]]; then
      log "push ${ahead} local commit(s) to origin/${BRANCH}"
      git -C "${ROOT}" push origin "HEAD:${BRANCH}" 2>&1 | tee -a "${LOG}" || true
    fi
  fi
}

_git_sync() {
  if [[ "${CONVERGE_AUTOFIX_SKIP_GIT:-0}" == "1" ]]; then
    log "skip git sync (CONVERGE_AUTOFIX_SKIP_GIT=1) HEAD $(git -C "${ROOT}" log -1 --oneline 2>/dev/null || echo unknown)"
    return 0
  fi
  log "fetch origin/${BRANCH}"
  git -C "${ROOT}" fetch origin "${BRANCH}" 2>&1 | tee -a "${LOG}"
  git -C "${ROOT}" checkout "${BRANCH}" 2>&1 | tee -a "${LOG}" || true
  local ahead
  ahead="$(git -C "${ROOT}" rev-list --count "origin/${BRANCH}..HEAD" 2>/dev/null || echo 0)"
  if [[ "${ahead}" -gt 0 ]]; then
    log "local is ${ahead} commit(s) ahead of origin/${BRANCH}; skip hard reset"
  else
    git -C "${ROOT}" reset --hard "origin/${BRANCH}" 2>&1 | tee -a "${LOG}"
  fi
  log "HEAD $(git -C "${ROOT}" log -1 --oneline)"
}

_pip_sync() {
  if [[ -x "${PYTHON}" ]]; then
    log "pip install -e npa"
    "${PYTHON}" -m pip install -e "${ROOT}/npa" -q 2>&1 | tee -a "${LOG}" || true
  fi
}

_unit_tests() {
  if [[ "${CONVERGE_UNIT_TEST:-0}" != "1" ]] || [[ ! -x "${PYTHON}" ]]; then
    return 0
  fi
  log "unit tests: sim2real workflow smoke"
  (
    cd "${ROOT}"
    "${PYTHON}" -m pytest \
      npa/tests/workflows/test_sim2real_assets.py \
      npa/tests/workflows/test_sim2real_policy_swap.py \
      npa/tests/test_robot_assets.py \
      -q --tb=no
  ) 2>&1 | tee -a "${LOG}" || true
}

log "run_id=${RUN_ID} branch=${BRANCH}"
if [[ "${RUN_ID}" != "sync" ]]; then
  _write_patch_request "${RUN_ID}"
fi
_maybe_commit_push
_git_sync
_pip_sync
_unit_tests
rm -f "${STATE_DIR}/patch-applied" 2>/dev/null || true
