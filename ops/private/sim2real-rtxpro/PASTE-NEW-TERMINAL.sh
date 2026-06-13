#!/usr/bin/env bash
# =============================================================================
# FULL MAC PASTE BLOCK — new terminal, all git pull cases handled.
# Copy from the line "bash <<'NPA_SIM2REAL_DEMO'" through "NPA_SIM2REAL_DEMO"
# =============================================================================
bash <<'NPA_SIM2REAL_DEMO'
set -euo pipefail

export PATH="/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/opt/homebrew/bin:${HOME}/.nebius/bin:${PATH}"
export KUBECONFIG="${KUBECONFIG:-${HOME}/.npa/clusters/npa-rtxpro-mk8s/kubeconfig.resolved}"
export KUBECONTEXT="${KUBECONTEXT:-npa-rtxpro-mk8s}"
[[ -f "${HOME}/.npa/sim2real-operator.env" ]] && source "${HOME}/.npa/sim2real-operator.env"

DEMO="${HOME}/npa-sim2real-demo"
REPO="${DEMO}/nebius-physical-ai"
BRANCH="feat/sim2real-mandatory-stages"
REMOTE="origin"
GIT=""

find_git() {
  for GIT in "$(command -v git 2>/dev/null || true)" /usr/bin/git /opt/homebrew/bin/git; do
    [[ -n "${GIT}" && -x "${GIT}" ]] && return 0
  done
  echo "ERROR: git not found. Run: xcode-select --install" >&2
  return 1
}

clone_if_missing() {
  if [[ -d "${REPO}/.git" ]]; then
    return 0
  fi
  find_git || return 1
  echo "=== clone ${BRANCH} -> ${REPO} ==="
  mkdir -p "${DEMO}"
  if [[ -d "${REPO}" && ! -d "${REPO}/.git" ]]; then
    echo "ERROR: ${REPO} exists but is not a git repo — move it aside and re-run" >&2
    return 1
  fi
  "${GIT}" clone --branch "${BRANCH}" -- https://github.com/nebius/nebius-physical-ai.git "${REPO}"
}

sync_repo() {
  find_git || return 1
  clone_if_missing || return 1
  cd "${REPO}"
  echo "=== git fetch ${REMOTE} ${BRANCH} ==="
  "${GIT}" fetch "${REMOTE}" "${BRANCH}"
  if ! "${GIT}" show-ref --verify --quiet "refs/remotes/${REMOTE}/${BRANCH}"; then
    echo "ERROR: remote branch ${REMOTE}/${BRANCH} not found" >&2
    return 1
  fi
  cur="$("${GIT}" symbolic-ref -q --short HEAD 2>/dev/null || true)"
  if [[ "${cur}" != "${BRANCH}" ]]; then
    if "${GIT}" show-ref --verify --quiet "refs/heads/${BRANCH}"; then
      echo "=== git checkout ${BRANCH} ==="
      "${GIT}" checkout "${BRANCH}"
    else
      echo "=== git checkout -b ${BRANCH} ${REMOTE}/${BRANCH} ==="
      "${GIT}" checkout -b "${BRANCH}" "${REMOTE}/${BRANCH}"
    fi
  fi
  if ! "${GIT}" diff-index --quiet HEAD -- 2>/dev/null; then
    echo "WARN: dirty tree — auto-stashing before pull"
    "${GIT}" stash push -u -m "sim2real-operator-auto-stash $(date -u +%Y%m%dT%H%M%SZ)" || true
  fi
  echo "=== git pull --ff-only ${REMOTE} ${BRANCH} ==="
  if ! "${GIT}" pull --ff-only "${REMOTE}" "${BRANCH}"; then
    echo "WARN: ff-only failed — reset to ${REMOTE}/${BRANCH}"
    "${GIT}" reset --hard "${REMOTE}/${BRANCH}"
  fi
  echo "=== repo at ==="
  "${GIT}" log -1 --oneline
}

install_run_sh() {
  mac_run="${REPO}/ops/private/sim2real-rtxpro/mac-run.sh"
  [[ -f "${mac_run}" ]] || { echo "ERROR: missing ${mac_run}" >&2; return 1; }
  cp "${mac_run}" "${DEMO}/run.sh"
  chmod +x "${DEMO}/run.sh"
  echo "Installed ${DEMO}/run.sh"
}

echo "=== Sim2Real operator: sync + customer demo ==="
sync_repo
install_run_sh
echo "=== cleanup + submit ==="
cd "${DEMO}"
exec ./run.sh demo
NPA_SIM2REAL_DEMO
