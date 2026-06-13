#!/usr/bin/env bash
# Copy this file to ~/npa-sim2real-demo/run.sh (one-time Mac handoff).
set -euo pipefail
DEMO_ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO="${DEMO_ROOT}/nebius-physical-ai"
OPS="${REPO}/ops/private/sim2real-rtxpro"

if [ ! -f "${REPO}/npa/pyproject.toml" ]; then
  echo "ERROR: expected ${REPO}/npa/pyproject.toml" >&2
  echo "Clone nebius-physical-ai beside this script, then re-run." >&2
  exit 1
fi

export NPA_SIM2REAL_REPO="${REPO}"
# shellcheck source=lib/operator-env.sh
source "${OPS}/lib/operator-env.sh"
# shellcheck source=lib/operator-shell.sh
source "${OPS}/lib/operator-shell.sh"
operator_bootstrap_shell "${REPO}"

exec "${OPS}/run.sh" "$@"
