#!/usr/bin/env bash
# Robust git sync for Mac operator checkouts (source only).
#
# Usage:
#   source .../sync-operator-repo.sh
#   sync_operator_repo /path/to/nebius-physical-ai feat/sim2real-mandatory-stages

sync_operator_repo() {
  local repo="${1:?repo path required}"
  local branch="${2:?branch required}"
  local remote="${3:-origin}"
  local git_bin=""

  sync_operator_find_git() {
    if [[ -n "${GIT:-}" && -x "${GIT}" ]]; then
      git_bin="${GIT}"
      return 0
    fi
    local candidate
    for candidate in "$(command -v git 2>/dev/null || true)" /usr/bin/git /opt/homebrew/bin/git; do
      [[ -n "${candidate}" && -x "${candidate}" ]] || continue
      git_bin="${candidate}"
      export GIT="${git_bin}"
      return 0
    done
    cat >&2 <<'EOF'
ERROR: git not found.

Install Xcode command-line tools on Mac:
  xcode-select --install

Then open a new terminal and re-run.
EOF
    return 1
  }

  sync_operator_die() {
    echo "ERROR: $*" >&2
    return 1
  }

  sync_operator_find_git || return 1

  if [[ ! -d "${repo}" ]]; then
    sync_operator_die "missing repo directory: ${repo}

Clone once:
  mkdir -p $(dirname "${repo}")
  git clone --branch ${branch} https://github.com/nebius/nebius-physical-ai.git "${repo}""
    return 1
  fi

  if [[ ! -d "${repo}/.git" ]]; then
    sync_operator_die "${repo} is not a git checkout (.git missing)"
    return 1
  fi

  (
    cd "${repo}" || exit 1
    if ! "${git_bin}" remote get-url "${remote}" >/dev/null 2>&1; then
      sync_operator_die "remote ${remote} not configured in ${repo}"
      exit 1
    fi

    echo "=== git fetch ${remote} ${branch} ==="
    if ! "${git_bin}" fetch "${remote}" "${branch}"; then
      sync_operator_die "git fetch failed — check network and GitHub access"
      exit 1
    fi

    if ! "${git_bin}" show-ref --verify --quiet "refs/remotes/${remote}/${branch}"; then
      sync_operator_die "remote branch ${remote}/${branch} not found after fetch"
      exit 1
    fi

    local head_branch=""
    head_branch="$("${git_bin}" symbolic-ref -q --short HEAD 2>/dev/null || true)"
    if [[ "${head_branch}" != "${branch}" ]]; then
      if "${git_bin}" show-ref --verify --quiet "refs/heads/${branch}"; then
        echo "=== git checkout ${branch} ==="
        "${git_bin}" checkout "${branch}"
      else
        echo "=== git checkout -b ${branch} ${remote}/${branch} ==="
        "${git_bin}" checkout -b "${branch}" "${remote}/${branch}"
      fi
    fi

    if ! "${git_bin}" diff-index --quiet HEAD -- 2>/dev/null; then
      echo "WARN: dirty working tree — stashing local changes before sync"
      "${git_bin}" stash push -u -m "sim2real-operator-auto-stash $(date -u +%Y%m%dT%H%M%SZ)" || true
    fi

    echo "=== git pull --ff-only ${remote} ${branch} ==="
    if ! "${git_bin}" pull --ff-only "${remote}" "${branch}"; then
      echo "WARN: ff-only pull failed — resetting to ${remote}/${branch} for operator scripts"
      "${git_bin}" reset --hard "${remote}/${branch}"
    fi

    echo "=== repo sync OK ==="
    "${git_bin}" log -1 --oneline
  )
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  sync_operator_repo "${1:?repo}" "${2:?branch}"
fi
