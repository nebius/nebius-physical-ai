#!/usr/bin/env bash
#
# dev_vm_isolated_session.sh - run isolated npa work on a SHARED dev VM.
#
# WHY
#   The dev/operator VM is shared by many concurrent agents/processes. They all
#   used to `git checkout` in one clone (`~/nebius-physical-ai`), so one agent's
#   checkout silently changed another agent's working tree and editable `npa`
#   install mid-run (branch churn). This script gives every process its own
#   isolated workspace so runs never collide.
#
# ISOLATION MODEL (one command per run)
#   * git worktree  - a per-run checkout of a branch under $NPA_WORKTREE_ROOT/<run-id>,
#                     sharing the shared clone's object store but with its own HEAD.
#                     One agent's branch checkout can never disturb another's.
#   * venv          - a per-run virtualenv with `pip install -e <worktree>/npa`, so
#                     the `npa` CLI resolves to THIS run's code, not a global install.
#                     (Fast mode reuses the shared venv's deps via PYTHONPATH instead.)
#   * tmux          - a per-run detached tmux session (`npa-<run-id>`) so long-running
#                     work survives disconnects and is inspectable without attaching.
#
# USAGE
#   dev_vm_isolated_session.sh start <branch> [run-id]     # create worktree+venv+tmux
#   dev_vm_isolated_session.sh exec  <run-id> <command...> # run a command in the session
#   dev_vm_isolated_session.sh path  <run-id>              # print worktree + npa bin paths
#   dev_vm_isolated_session.sh list                        # list isolated worktrees/sessions
#   dev_vm_isolated_session.sh stop  <run-id>              # kill session + remove worktree/venv
#
# ENV
#   NPA_REPO           shared clone (default: $HOME/nebius-physical-ai)
#   NPA_WORKTREE_ROOT  worktree parent dir (default: $HOME/npa-worktrees)
#   NPA_ISOLATED_FAST  =1 => skip per-run venv; reuse shared venv deps + PYTHONPATH
#                          (faster start; use when branch dependencies are unchanged)
#
# EXAMPLE (isolated GPU submit via npa, never touching the shared checkout)
#   S=$(dev_vm_isolated_session.sh start cursor/my-branch-02d7 gpu-run-1)
#   dev_vm_isolated_session.sh exec gpu-run-1 \
#     'npa workbench workflow submit npa/workflows/workbench/npa-workflows/adversarial-scenario-hardening.yaml \
#        --run-id gpu-run-1 --infra k8s/npa-rtxpro-mk8s --deploy-if-absent \
#        --secret-env AWS_ACCESS_KEY_ID --secret-env AWS_SECRET_ACCESS_KEY'
#   dev_vm_isolated_session.sh stop gpu-run-1
#
set -euo pipefail

NPA_REPO="${NPA_REPO:-$HOME/nebius-physical-ai}"
NPA_WORKTREE_ROOT="${NPA_WORKTREE_ROOT:-$HOME/npa-worktrees}"
NPA_ISOLATED_FAST="${NPA_ISOLATED_FAST:-0}"

_tmux() {
  if [ -f /exec-daemon/tmux.portal.conf ]; then
    tmux -f /exec-daemon/tmux.portal.conf "$@"
  else
    tmux "$@"
  fi
}

_worktree_dir() { printf '%s/%s' "$NPA_WORKTREE_ROOT" "$1"; }
_session_name() { printf 'npa-%s' "$1"; }

cmd_start() {
  local branch="${1:?usage: start <branch> [run-id]}"
  local run_id="${2:-run-$(date +%Y%m%d%H%M%S)-$$}"
  local wt sess py npa_bin
  wt="$(_worktree_dir "$run_id")"
  sess="$(_session_name "$run_id")"

  mkdir -p "$NPA_WORKTREE_ROOT"
  git -C "$NPA_REPO" fetch --quiet origin "$branch"
  if [ ! -d "$wt" ]; then
    # Detached worktree at the fetched tip keeps the shared clone's HEAD untouched.
    git -C "$NPA_REPO" worktree add --detach --force "$wt" "origin/$branch" >/dev/null
  fi

  if [ "$NPA_ISOLATED_FAST" = "1" ]; then
    # Reuse the shared venv's installed deps; override the code via PYTHONPATH.
    py="$NPA_REPO/npa/.venv/bin/python"
    npa_bin="$py -m npa.cli.main"
    export PYTHONPATH="$wt/npa/src${PYTHONPATH:+:$PYTHONPATH}"
  else
    if [ ! -x "$wt/.venv/bin/npa" ]; then
      python3 -m venv "$wt/.venv"
      "$wt/.venv/bin/pip" install -q --upgrade pip >/dev/null 2>&1 || true
      "$wt/.venv/bin/pip" install -q -e "$wt/npa"
    fi
    py="$wt/.venv/bin/python"
    npa_bin="$wt/.venv/bin/npa"
  fi

  if ! _tmux has-session -t "=$sess" 2>/dev/null; then
    _tmux new-session -d -s "$sess" -c "$wt" -- bash -l
    if [ "$NPA_ISOLATED_FAST" = "1" ]; then
      _tmux send-keys -t "$sess" "export PATH=$NPA_REPO/npa/.venv/bin:\$PATH PYTHONPATH=$wt/npa/src NPA_RUN_ID=$run_id; cd $wt" C-m
    else
      _tmux send-keys -t "$sess" "export PATH=$wt/.venv/bin:\$PATH NPA_RUN_ID=$run_id; cd $wt" C-m
    fi
  fi

  echo "run_id=$run_id"
  echo "worktree=$wt"
  echo "venv_python=$py"
  echo "npa=$npa_bin"
  echo "tmux=$sess"
}

cmd_exec() {
  local run_id="${1:?usage: exec <run-id> <command...>}"; shift
  local sess wt
  sess="$(_session_name "$run_id")"; wt="$(_worktree_dir "$run_id")"
  _tmux has-session -t "=$sess" 2>/dev/null || { echo "no session for run-id=$run_id (start it first)" >&2; exit 1; }
  # Fire-and-forget into the isolated session; inspect with `tmux capture-pane`.
  _tmux send-keys -t "$sess" "cd $wt && $*" C-m
  echo "sent to $sess: $*"
}

cmd_path() {
  local run_id="${1:?usage: path <run-id>}"
  local wt; wt="$(_worktree_dir "$run_id")"
  echo "worktree=$wt"
  if [ -x "$wt/.venv/bin/npa" ]; then echo "npa=$wt/.venv/bin/npa"; else echo "npa=$NPA_REPO/npa/.venv/bin/python -m npa.cli.main (fast mode; PYTHONPATH=$wt/npa/src)"; fi
}

cmd_list() {
  echo "=== worktrees ==="
  git -C "$NPA_REPO" worktree list 2>/dev/null | grep -F "$NPA_WORKTREE_ROOT" || echo "(none)"
  echo "=== tmux sessions ==="
  _tmux ls 2>/dev/null | grep -E '^npa-' || echo "(none)"
}

cmd_stop() {
  local run_id="${1:?usage: stop <run-id>}"
  local sess wt; sess="$(_session_name "$run_id")"; wt="$(_worktree_dir "$run_id")"
  _tmux kill-session -t "=$sess" 2>/dev/null || true
  git -C "$NPA_REPO" worktree remove --force "$wt" 2>/dev/null || rm -rf "$wt"
  git -C "$NPA_REPO" worktree prune 2>/dev/null || true
  echo "stopped run_id=$run_id (session + worktree removed)"
}

sub="${1:-}"; shift || true
case "$sub" in
  start) cmd_start "$@" ;;
  exec)  cmd_exec "$@" ;;
  path)  cmd_path "$@" ;;
  list)  cmd_list "$@" ;;
  stop)  cmd_stop "$@" ;;
  *) grep -E '^#( |$)' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
