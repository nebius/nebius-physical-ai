#!/usr/bin/env bash
# Installed as ~/npa-sim2real-demo/run.sh — daily entrypoint (demo|status|sync|seed-trigger).
set -euo pipefail

export PATH="/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/opt/homebrew/bin:/opt/homebrew/opt/python@3.12/libexec/bin:${HOME}/.nebius/bin:${PATH}"

DEMO_ROOT="$(cd "$(dirname "$0")" && pwd)"
export NPA_SIM2REAL_DEMO="${DEMO_ROOT}"
REPO="${DEMO_ROOT}/nebius-physical-ai"
BRANCH="${NPA_BRANCH:-feat/sim2real-mandatory-stages}"

if [ ! -f "${REPO}/npa/pyproject.toml" ]; then
  echo "ERROR: expected ${REPO}/npa/pyproject.toml" >&2
  echo "Run first-time-setup.sh once (see QUICKSTART.md)." >&2
  exit 1
fi

OPS="${REPO}/ops/private/sim2real-rtxpro"
CMD="${1:-help}"
if [ "${CMD}" = "status" ] || [ "${CMD}" = "sync" ]; then
  export NPA_SESSION_QUIET=1
fi

# Session bootstrap: sync operator scripts + virtualenv (idempotent).
if [ -d "${REPO}/.git" ] && [ -f "${OPS}/lib/sync-operator-repo.sh" ]; then
  # shellcheck source=lib/sync-operator-repo.sh
  source "${OPS}/lib/sync-operator-repo.sh"
  OPS="${REPO}/ops/private/sim2real-rtxpro"
  if [ "${NPA_SESSION_QUIET:-0}" = "1" ]; then
    sync_operator_repo "${REPO}" "${BRANCH}" >/dev/null 2>&1 || sync_operator_repo "${REPO}" "${BRANCH}"
  else
    sync_operator_repo "${REPO}" "${BRANCH}"
  fi
fi
OPS="${REPO}/ops/private/sim2real-rtxpro"
if [ -f "${OPS}/bootstrap-npa-venv.sh" ]; then
  if [ "${NPA_SESSION_QUIET:-0}" = "1" ]; then
    bash "${OPS}/bootstrap-npa-venv.sh" "${REPO}" >/dev/null 2>&1 || bash "${OPS}/bootstrap-npa-venv.sh" "${REPO}"
  else
    bash "${OPS}/bootstrap-npa-venv.sh" "${REPO}"
  fi
fi
RUN_SRC="${OPS}/operator-run.sh"
if [ ! -f "${RUN_SRC}" ]; then
  RUN_SRC="${OPS}/mac-run.sh"
fi
if [ -f "${RUN_SRC}" ]; then
  cp "${RUN_SRC}" "${DEMO_ROOT}/run.sh"
  chmod +x "${DEMO_ROOT}/run.sh"
fi

export NPA_SIM2REAL_REPO="${REPO}"
# shellcheck source=lib/operator-env.sh
source "${OPS}/lib/operator-env.sh"
# shellcheck source=lib/operator-shell.sh
source "${OPS}/lib/operator-shell.sh"
operator_bootstrap_shell "${REPO}"

exec "${OPS}/run.sh" "$@"
