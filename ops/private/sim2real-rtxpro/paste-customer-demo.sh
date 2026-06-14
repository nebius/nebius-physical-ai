#!/usr/bin/env bash
# Self-contained Mac operator paste — sync repo, install run.sh, run demo.
set -euo pipefail

export PATH="/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/opt/homebrew/bin:${HOME}/.nebius/bin:${PATH}"
export KUBECONFIG="${KUBECONFIG:-${HOME}/.npa/clusters/npa-rtxpro-mk8s/kubeconfig.resolved}"
export KUBECONTEXT="${KUBECONTEXT:-npa-rtxpro-mk8s}"
if [[ -f "${HOME}/.npa/sim2real-operator.env" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/.npa/sim2real-operator.env"
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEMO="${NPA_SIM2REAL_DEMO:-${DEMO:-${HOME}/npa-sim2real-demo}}"
REPO="${NPA_SIM2REAL_REPO:-${DEMO}/nebius-physical-ai}"
BRANCH="${NPA_SIM2REAL_BRANCH:-feat/sim2real-mandatory-stages}"
SYNC_LIB="${SCRIPT_DIR}/lib/sync-operator-repo.sh"

_sync_repo() {
  if [[ -f "${SYNC_LIB}" ]]; then
    # shellcheck source=lib/sync-operator-repo.sh
    source "${SYNC_LIB}"
    sync_operator_repo "${REPO}" "${BRANCH}"
    return 0
  fi

  # Fallback when checkout predates sync-operator-repo.sh (should not happen after pull).
  local git_bin="${GIT:-$(command -v git || echo /usr/bin/git)}"
  [[ -x "${git_bin}" ]] || {
    echo "ERROR: git not found — xcode-select --install" >&2
    exit 1
  }
  if [[ ! -d "${REPO}/.git" ]]; then
    mkdir -p "$(dirname "${REPO}")"
    "${git_bin}" clone --branch "${BRANCH}" -- https://github.com/nebius/nebius-physical-ai.git "${REPO}"
  fi
  (
    cd "${REPO}"
    "${git_bin}" fetch origin "${BRANCH}"
    "${git_bin}" checkout "${BRANCH}" 2>/dev/null || "${git_bin}" checkout -b "${BRANCH}" "origin/${BRANCH}"
    "${git_bin}" pull --ff-only origin "${BRANCH}" || "${git_bin}" reset --hard "origin/${BRANCH}"
  )
}

_install_run_sh() {
  local mac_run="${REPO}/ops/private/sim2real-rtxpro/mac-run.sh"
  [[ -f "${mac_run}" ]] || {
    echo "ERROR: missing ${mac_run} after sync" >&2
    exit 1
  }
  mkdir -p "${DEMO}"
  cp "${mac_run}" "${DEMO}/run.sh"
  chmod +x "${DEMO}/run.sh"
  echo "Installed ${DEMO}/run.sh"
}

_main() {
  echo "=== Sim2Real operator: sync + customer demo ==="
  _sync_repo
  _install_run_sh
  if [[ "${SIM2REAL_PASTE_SKIP_DEMO:-0}" == "1" ]]; then
    echo "SIM2REAL_PASTE_SKIP_DEMO=1 — stop before ./run.sh demo"
    echo "Next: cd ${DEMO} && ./run.sh demo"
    return 0
  fi
  cd "${DEMO}"
  exec ./run.sh demo
}

_main "$@"
