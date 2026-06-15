#!/usr/bin/env bash
# Continuous image build/push loop for golden-eval tmux (runs in build window).
#
# Usage: golden_eval_build_loop.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
STATE_DIR="${GOLDEN_EVAL_STATE_DIR:-/tmp/golden-evals/converge}"
COMPLETE_FILE="${STATE_DIR}/golden-evals-complete"
LOG="${STATE_DIR}/build-loop.log"
BUILD="${SCRIPT_DIR}/build_golden_eval_images.sh"
REGISTRY="${REGISTRY:-cr.eu-north1.nebius.cloud/e00cm0vc6t09m0z5gw}"
POLL_S="${GOLDEN_EVAL_BUILD_POLL_S:-600}"

mkdir -p "${STATE_DIR}"

log() {
  printf '[%s] build %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${LOG}"
}

run_build() {
  local reason="$1"
  log "start reason=${reason} registry=${REGISTRY}"
  set +e
  REGISTRY="${REGISTRY}" bash "${BUILD}" --all --push 2>&1 | tee -a "${LOG}"
  local ec=$?
  set -e
  log "finished ec=${ec}"
  return "${ec}"
}

log "build loop start poll=${POLL_S}s"
run_build "initial" || true

while [[ ! -f "${COMPLETE_FILE}" ]]; do
  if [[ -f "${STATE_DIR}/rebuild-request" ]]; then
    tools="$(tr '\n' ' ' < "${STATE_DIR}/rebuild-request" | xargs || true)"
    rm -f "${STATE_DIR}/rebuild-request"
    if [[ -n "${tools}" ]]; then
      log "rebuild-request tools=${tools}"
      set +e
      REGISTRY="${REGISTRY}" bash "${BUILD}" ${tools} --push 2>&1 | tee -a "${LOG}"
      set -e
    else
      run_build "rebuild-request-empty" || true
    fi
  fi
  sleep "${POLL_S}"
  if [[ ! -f "${COMPLETE_FILE}" ]]; then
    run_build "periodic" || true
  fi
done

log "golden-evals-complete — build loop exiting"
