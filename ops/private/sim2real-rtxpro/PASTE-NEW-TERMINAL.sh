#!/usr/bin/env bash
# =============================================================================
# FULL MAC PASTE BLOCK — new terminal, all git pull cases handled.
#
# Paste everything from "bash <<'NPA_SIM2REAL_DEMO'" through "NPA_SIM2REAL_DEMO"
# into a new Mac terminal.
# =============================================================================
bash <<'NPA_SIM2REAL_DEMO'
set -euo pipefail

export PATH="/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/opt/homebrew/bin:${HOME}/.nebius/bin:${PATH}"
export KUBECONFIG="${KUBECONFIG:-${HOME}/.npa/clusters/npa-rtxpro-mk8s/kubeconfig.resolved}"
export KUBECONTEXT="${KUBECONTEXT:-npa-rtxpro-mk8s}"
[[ -f "${HOME}/.npa/sim2real-operator.env" ]] && source "${HOME}/.npa/sim2real-operator.env"

DEMO="${NPA_SIM2REAL_DEMO:-${HOME}/npa-sim2real-demo}"
REPO="${NPA_SIM2REAL_REPO:-${DEMO}/nebius-physical-ai}"
BRANCH="feat/sim2real-mandatory-stages"
PASTE="${REPO}/ops/private/sim2real-rtxpro/paste-customer-demo.sh"

# Bootstrap: clone repo if missing (covers first-time Mac with no checkout).
if [[ ! -d "${REPO}/.git" ]]; then
  GIT=""
  for GIT in "$(command -v git 2>/dev/null || true)" /usr/bin/git /opt/homebrew/bin/git; do
    [[ -n "${GIT}" && -x "${GIT}" ]] || continue
    break
  done
  [[ -n "${GIT}" && -x "${GIT}" ]] || {
    echo "ERROR: git not found. Run: xcode-select --install" >&2
    exit 1
  }
  if [[ -e "${REPO}" && ! -d "${REPO}/.git" ]]; then
    echo "ERROR: ${REPO} exists but is not a git repo — move it aside and re-run" >&2
    exit 1
  fi
  mkdir -p "${DEMO}"
  echo "=== first-time clone ${BRANCH} -> ${REPO} ==="
  "${GIT}" clone --branch "${BRANCH}" -- https://github.com/nebius/nebius-physical-ai.git "${REPO}"
fi

[[ -f "${PASTE}" ]] || {
  echo "ERROR: missing ${PASTE} — clone may have failed or branch is too old" >&2
  exit 1
}

exec bash "${PASTE}"
NPA_SIM2REAL_DEMO
