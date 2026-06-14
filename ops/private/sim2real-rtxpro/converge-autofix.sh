#!/usr/bin/env bash
# Optional post-failure hook: pull latest branch tip (picks up pushed patches).
# Called by converge-until-success.sh after a failed attempt.
set -euo pipefail

RUN_ID="${1:?run_id required}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=lib/operator-config.sh
source "${SCRIPT_DIR}/lib/operator-config.sh"
ROOT="$(npa_repo_root "${SCRIPT_DIR}")"
BRANCH="${NPA_SOURCE_REF:-feat/sim2real-mandatory-stages}"
LOG="/tmp/sim2real-cluster/converge/autofix.log"

log() {
  printf '[%s] autofix %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${LOG}"
}

log "run_id=${RUN_ID} fetch origin/${BRANCH}"
git -C "${ROOT}" fetch origin "${BRANCH}" 2>&1 | tee -a "${LOG}"
git -C "${ROOT}" reset --hard "origin/${BRANCH}" 2>&1 | tee -a "${LOG}"
log "HEAD $(git -C "${ROOT}" log -1 --oneline)"
