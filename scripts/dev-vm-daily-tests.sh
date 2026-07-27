#!/usr/bin/env bash
#
# dev-vm-daily-tests.sh - run the daily NPA test tiers on the dev/operator VM.
#
# WHY
#   GitHub-hosted runners live outside the operator's Nebius environment, so they
#   cannot reach real infrastructure to run e2e tests (see
#   docs/testing/live-e2e.md). This script inverts the direction: the
#   .github/workflows/dev-vm-daily-tests.yml workflow SSHes INTO the dev VM
#   (which already has ~/.npa credentials, config, and SkyPilot) and runs the
#   tests there. It is also runnable by hand on the dev VM.
#
# ISOLATION
#   The script uses a DEDICATED CI checkout (NPA_CI_REPO_DIR, default
#   ~/npa-ci-daily) with its own venv so it never disturbs the shared dev clone
#   or other agents' worktrees. See npa/scripts/dev_vm_isolated_session.sh for
#   the interactive equivalent.
#
# GUARDRAIL
#   The GPU-spending live path (scripts/live-e2e.sh, `-m "gpu and e2e"`) must
#   never run unattended on a schedule: it provisions real GPU clusters and can
#   leak spend overnight (docs/testing/live-e2e.md). The `live-gpu` tier here is
#   opt-in only and additionally requires NPA_DAILY_ALLOW_LIVE_GPU=1; the GitHub
#   workflow refuses to set that flag for scheduled runs.
#
# USAGE
#   dev-vm-daily-tests.sh [tier] [git-ref]
#     tier     one of: unit | e2e | e2e-serverless | live-gpu   (default: unit)
#     git-ref  branch, tag, or sha to test                       (default: main)
#
# ENV
#   NPA_CI_REPO_DIR        dedicated CI checkout dir  (default: $HOME/npa-ci-daily)
#   NPA_CI_REPO_URL        git remote to clone        (default: derived from the
#                          shared dev clone or an existing CI checkout)
#   NPA_REPO               shared dev clone used only to derive the remote URL
#                          (default: $HOME/nebius-physical-ai)
#   NPA_DAILY_LOG_DIR      log directory              (default: $HOME/npa-daily-test-logs)
#   NPA_DAILY_ALLOW_LIVE_GPU=1  required to run the live-gpu tier
#   NPA_DAILY_PIP_EXTRAS   pip extras to install      (default: dev,adapter)
#
set -Eeuo pipefail

TEST_TIER="${1:-${NPA_DAILY_TEST_TIER:-unit}}"
GIT_REF="${2:-${NPA_DAILY_GIT_REF:-main}}"

CI_REPO_DIR="${NPA_CI_REPO_DIR:-${HOME}/npa-ci-daily}"
SHARED_REPO_DIR="${NPA_REPO:-${HOME}/nebius-physical-ai}"
PIP_EXTRAS="${NPA_DAILY_PIP_EXTRAS:-dev,adapter}"

RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="${NPA_DAILY_LOG_DIR:-${HOME}/npa-daily-test-logs}"
LOG_FILE="${LOG_DIR}/daily-${TEST_TIER}-${RUN_STAMP}.log"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

log() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
die() { log "ERROR: $*"; exit 2; }

resolve_repo_url() {
  if [[ -n "${NPA_CI_REPO_URL:-}" ]]; then
    printf '%s\n' "$NPA_CI_REPO_URL"
    return 0
  fi
  local dir
  for dir in "$SHARED_REPO_DIR" "$CI_REPO_DIR"; do
    if [[ -d "${dir}/.git" ]]; then
      if git -C "$dir" remote get-url origin >/dev/null 2>&1; then
        git -C "$dir" remote get-url origin
        return 0
      fi
    fi
  done
  return 1
}

sync_checkout() {
  local url
  if [[ ! -d "${CI_REPO_DIR}/.git" ]]; then
    url="$(resolve_repo_url)" || die "cannot resolve a git remote; set NPA_CI_REPO_URL"
    log "Cloning ${url} -> ${CI_REPO_DIR}"
    git clone "$url" "$CI_REPO_DIR"
  fi

  log "Fetching origin in ${CI_REPO_DIR}"
  git -C "$CI_REPO_DIR" fetch --prune --tags origin

  log "Checking out ${GIT_REF}"
  if git -C "$CI_REPO_DIR" rev-parse --verify --quiet "origin/${GIT_REF}" >/dev/null; then
    # Branch: pin the local branch to the fetched tip.
    git -C "$CI_REPO_DIR" checkout -f -B "$GIT_REF" "origin/${GIT_REF}"
  else
    # Tag or explicit sha.
    git -C "$CI_REPO_DIR" checkout -f "$GIT_REF"
  fi
  log "HEAD is now $(git -C "$CI_REPO_DIR" rev-parse HEAD)"
}

ensure_venv() {
  local venv="${CI_REPO_DIR}/.venv"
  if [[ ! -x "${venv}/bin/python" ]]; then
    log "Creating venv at ${venv}"
    python3 -m venv "$venv"
  fi
  "${venv}/bin/python" -m pip install -q --upgrade pip
  log "Installing npa[${PIP_EXTRAS}] (editable)"
  "${venv}/bin/python" -m pip install -q -e "${CI_REPO_DIR}/npa[${PIP_EXTRAS}]"
  printf '%s\n' "${venv}/bin/python"
}

run_unit() {
  local py="$1"
  log "Tier=unit: fast unit suite (no live infra) + ruff lint"
  make -C "$CI_REPO_DIR" test PYTHON="$py"
  # Mirror the CI lint gate scope (.github/workflows/lint.yml: `ruff check
  # src tests`) rather than `make lint`, which scans the whole tree including
  # scripts/, examples/, and docker/ that CI does not gate.
  (cd "${CI_REPO_DIR}/npa" && "$py" -m ruff check src tests)
}

run_e2e() {
  local py="$1"
  run_unit "$py"
  log "Tier=e2e: real-infra S3 e2e suite (self-cleaning, budget-bounded)"
  (
    cd "${CI_REPO_DIR}/npa"
    NPA_INTEGRATION_E2E=1 "$py" -m pytest tests/e2e -m e2e --tb=short
  )
}

run_e2e_serverless() {
  local py="$1"
  [[ -n "${NPA_E2E_SERVERLESS_PROJECT:-}" ]] \
    || die "e2e-serverless tier requires NPA_E2E_SERVERLESS_PROJECT (a sandbox project id)"
  log "Tier=e2e-serverless: serverless endpoint/job e2e against project ${NPA_E2E_SERVERLESS_PROJECT}"
  (
    cd "${CI_REPO_DIR}/npa"
    NPA_INTEGRATION_E2E=1 "$py" -m pytest tests/e2e -m e2e_serverless --tb=short
  )
}

run_live_gpu() {
  [[ "${NPA_DAILY_ALLOW_LIVE_GPU:-0}" == "1" ]] \
    || die "live-gpu tier requires NPA_DAILY_ALLOW_LIVE_GPU=1; it must never run on a schedule (docs/testing/live-e2e.md)"
  log "Tier=live-gpu: delegating to scripts/live-e2e.sh (its own teardown guarantees apply)"
  NPA_LIVE_E2E_REPO_ROOT="$CI_REPO_DIR" bash "${CI_REPO_DIR}/scripts/live-e2e.sh"
}

main() {
  log "Starting daily dev-VM tests: tier=${TEST_TIER} ref=${GIT_REF}"
  log "CI checkout: ${CI_REPO_DIR}"
  log "Log file: ${LOG_FILE}"

  command -v git >/dev/null 2>&1 || die "git not found on dev VM"
  command -v python3 >/dev/null 2>&1 || die "python3 not found on dev VM"
  command -v make >/dev/null 2>&1 || die "make not found on dev VM"

  sync_checkout
  local py
  py="$(ensure_venv)"

  case "$TEST_TIER" in
    unit) run_unit "$py" ;;
    e2e) run_e2e "$py" ;;
    e2e-serverless) run_e2e_serverless "$py" ;;
    live-gpu) run_live_gpu ;;
    *) die "unknown tier '${TEST_TIER}' (expected: unit | e2e | e2e-serverless | live-gpu)" ;;
  esac

  log "Daily dev-VM tests completed: tier=${TEST_TIER} ref=${GIT_REF}"
}

main "$@"
