#!/usr/bin/env bash
# Self-heal sim2real ops scripts when the git checkout branch drifts.
#
# Tmux converge survives branch switches (e.g. feat/golden-eval) that drop
# ops/private/sim2real-rtxpro from the working tree. Restore from NPA_SOURCE_REF
# before every converge/watchdog cycle.
#
# Usage:
#   ensure-converge-ops.sh          # restore if anything required is missing
#   ensure-converge-ops.sh --check  # exit 1 if missing (no restore)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=lib/operator-env.sh
source "${SCRIPT_DIR}/lib/operator-env.sh"
# shellcheck source=lib/operator-config.sh
source "${SCRIPT_DIR}/lib/operator-config.sh"
ROOT="$(npa_repo_root "${SCRIPT_DIR}")"
BRANCH="${NPA_SOURCE_REF:-feat/sim2real-mandatory-stages}"
STATE_DIR="${CONVERGE_STATE_DIR:-/tmp/sim2real-cluster/converge}"
LOG="${STATE_DIR}/ensure-ops.log"
CHECK_ONLY=0

if [[ "${1:-}" == "--check" ]]; then
  CHECK_ONLY=1
fi

mkdir -p "${STATE_DIR}"

log() {
  printf '[%s] ensure-ops %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${LOG}"
}

REQUIRED=(
  converge-until-success.sh
  converge-autofix.sh
  converge-cursor-patch.sh
  start-converge-tmux.sh
  watchdog-converge.sh
  submit-k8s-staged-job.sh
  monitor-k8s-job.sh
  cleanup-operator.sh
  lib/operator-config.sh
  lib/customer-asset-profile.sh
)

missing=()
for rel in "${REQUIRED[@]}"; do
  if [[ ! -f "${SCRIPT_DIR}/${rel}" ]]; then
    missing+=("${rel}")
  fi
done

if [[ ${#missing[@]} -eq 0 ]]; then
  exit 0
fi

if [[ "${CHECK_ONLY}" == "1" ]]; then
  log "MISSING ${#missing[@]} file(s): ${missing[*]}"
  exit 1
fi

log "restoring ops/private/sim2real-rtxpro from origin/${BRANCH} (missing: ${missing[*]})"
git -C "${ROOT}" fetch origin "${BRANCH}" 2>&1 | tee -a "${LOG}"
git -C "${ROOT}" checkout "origin/${BRANCH}" -- ops/private/sim2real-rtxpro/ 2>&1 | tee -a "${LOG}"
chmod +x "${SCRIPT_DIR}"/*.sh "${SCRIPT_DIR}"/lib/*.sh 2>/dev/null || true

still_missing=()
for rel in "${REQUIRED[@]}"; do
  if [[ ! -f "${SCRIPT_DIR}/${rel}" ]]; then
    still_missing+=("${rel}")
  fi
done

if [[ ${#still_missing[@]} -gt 0 ]]; then
  log "ERROR still missing after restore: ${still_missing[*]}"
  exit 1
fi

log "restore OK HEAD=$(git -C "${ROOT}" rev-parse --short origin/${BRANCH} 2>/dev/null || echo unknown)"
