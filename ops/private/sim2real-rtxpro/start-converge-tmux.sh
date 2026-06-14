#!/usr/bin/env bash
# Launch converge-until-success in a detached tmux session (full operator access).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=lib/operator-env.sh
source "${SCRIPT_DIR}/lib/operator-env.sh"
ROOT="$(npa_repo_root "${SCRIPT_DIR}")"
SESSION="${CONVERGE_TMUX_SESSION:-sim2real-converge}"
LOG="/tmp/sim2real-cluster/converge/tmux-launch.log"
mkdir -p /tmp/sim2real-cluster/converge

if ! command -v tmux >/dev/null; then
  echo "tmux required" >&2
  exit 1
fi

tmux kill-session -t "${SESSION}" 2>/dev/null || true
tmux new-session -d -s "${SESSION}" \
  "bash -lc 'cd \"${ROOT}\" && export KUBECONFIG=\"\${KUBECONFIG:-}\" && exec bash \"${SCRIPT_DIR}/converge-until-success.sh\" 2>&1 | tee -a ${LOG}; echo exit=\$?; exec bash'"

echo "TMUX_SESSION=${SESSION}"
echo "LOG=${LOG}"
echo "attach: tmux attach -t ${SESSION}"
