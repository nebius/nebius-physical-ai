#!/usr/bin/env bash
# Self-contained Mac operator paste — pull if needed, install run.sh, run demo.
# Usage:
#   bash ~/npa-sim2real-demo/nebius-physical-ai/ops/private/sim2real-rtxpro/paste-customer-demo.sh
# Or paste the heredoc block from FRANKA-STOCK-GUIDE.md into a new terminal.
set -euo pipefail

export PATH="/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/opt/homebrew/bin:${HOME}/.nebius/bin:${PATH}"
export KUBECONFIG="${KUBECONFIG:-${HOME}/.npa/clusters/npa-rtxpro-mk8s/kubeconfig.resolved}"
export KUBECONTEXT="${KUBECONTEXT:-npa-rtxpro-mk8s}"
if [[ -f "${HOME}/.npa/sim2real-operator.env" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/.npa/sim2real-operator.env"
fi

DEMO="${NPA_SIM2REAL_DEMO:-${DEMO:-${HOME}/npa-sim2real-demo}}"
REPO="${NPA_SIM2REAL_REPO:-${DEMO}/nebius-physical-ai}"
BRANCH="${NPA_SIM2REAL_BRANCH:-feat/sim2real-mandatory-stages}"
GIT="${GIT:-$(command -v git || true)}"
MAC_RUN="${REPO}/ops/private/sim2real-rtxpro/mac-run.sh"

_die() {
  echo "ERROR: $*" >&2
  exit 1
}

_ensure_git() {
  if [[ -n "${GIT}" && -x "${GIT}" ]]; then
    return 0
  fi
  for candidate in /usr/bin/git /opt/homebrew/bin/git; do
    if [[ -x "${candidate}" ]]; then
      GIT="${candidate}"
      export GIT
      return 0
    fi
  done
  _die "git not found — install Xcode CLI tools: xcode-select --install"
}

_sync_repo() {
  if [[ ! -d "${REPO}/.git" ]]; then
    _die "missing ${REPO} — clone nebius-physical-ai into ~/npa-sim2real-demo/ first"
  fi
  _ensure_git
  echo "=== git pull ${BRANCH} in ${REPO} ==="
  (
    cd "${REPO}"
    "${GIT}" fetch origin "${BRANCH}"
    if "${GIT}" show-ref --verify --quiet "refs/heads/${BRANCH}"; then
      "${GIT}" checkout "${BRANCH}"
    else
      "${GIT}" checkout -b "${BRANCH}" "origin/${BRANCH}"
    fi
    "${GIT}" pull --ff-only origin "${BRANCH}"
  )
}

_install_run_sh() {
  [[ -f "${MAC_RUN}" ]] || _die "missing ${MAC_RUN} after pull"
  cp "${MAC_RUN}" "${DEMO}/run.sh"
  chmod +x "${DEMO}/run.sh"
  echo "Installed ${DEMO}/run.sh"
}

_main() {
  mkdir -p "${DEMO}"
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
