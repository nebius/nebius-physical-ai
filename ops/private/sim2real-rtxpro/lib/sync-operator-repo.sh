#!/usr/bin/env bash
# Robust git sync for Mac operator checkouts (source only).
#
# Usage:
#   source .../sync-operator-repo.sh
#   sync_operator_repo /path/to/nebius-physical-ai main

sync_operator_find_git() {
  local candidate
  if [[ -n "${GIT:-}" && -x "${GIT}" ]]; then
    printf '%s\n' "${GIT}"
    return 0
  fi
  for candidate in "$(command -v git 2>/dev/null || true)" /usr/bin/git /opt/homebrew/bin/git; do
    [[ -n "${candidate}" && -x "${candidate}" ]] || continue
    export GIT="${candidate}"
    printf '%s\n' "${candidate}"
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

sync_operator_clone_if_missing() {
  local repo="${1:?repo path required}"
  local branch="${2:?branch required}"
  local remote_url="${3:-https://github.com/nebius/nebius-physical-ai.git}"
  local git_bin="${4:?git required}"

  if [[ -d "${repo}/.git" ]]; then
    return 0
  fi
  if [[ -e "${repo}" && ! -d "${repo}/.git" ]]; then
    echo "ERROR: ${repo} exists but is not a git repo — move it aside and re-run" >&2
    return 1
  fi
  mkdir -p "$(dirname "${repo}")"
  echo "=== git clone --branch ${branch} ${remote_url} -> ${repo} ==="
  "${git_bin}" clone --branch "${branch}" -- "${remote_url}" "${repo}"
}

sync_operator_repo() {
  local repo="${1:?repo path required}"
  local branch="${2:?branch required}"
  local remote="${3:-origin}"
  local git_bin=""

  git_bin="$(sync_operator_find_git)" || return 1
  sync_operator_clone_if_missing "${repo}" "${branch}" "https://github.com/nebius/nebius-physical-ai.git" "${git_bin}" || return 1

  if [[ ! -d "${repo}/.git" ]]; then
    echo "ERROR: ${repo} is not a git checkout (.git missing)" >&2
    return 1
  fi

  (
    cd "${repo}" || exit 1
    if ! "${git_bin}" remote get-url "${remote}" >/dev/null 2>&1; then
      echo "ERROR: remote ${remote} not configured in ${repo}" >&2
      exit 1
    fi

    echo "=== git fetch ${remote} ${branch} ==="
    if ! "${git_bin}" fetch "${remote}" "${branch}"; then
      echo "ERROR: git fetch failed — check network and GitHub access" >&2
      exit 1
    fi

    if ! "${git_bin}" show-ref --verify --quiet "refs/remotes/${remote}/${branch}"; then
      echo "ERROR: remote branch ${remote}/${branch} not found after fetch" >&2
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
      "${git_bin}" stash push -u -m "sim2real-operator-auto-stash $(/bin/date -u +%Y%m%dT%H%M%SZ)" || true
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
