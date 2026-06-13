#!/usr/bin/env bash
# =============================================================================
# Paste this entire block into a new Mac terminal.
# Installs/updates ~/npa-sim2real-demo/run.sh then runs customer replication.
# =============================================================================
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/opt/homebrew/bin:${HOME}/.nebius/bin:${PATH}"
export KUBECONFIG="${KUBECONFIG:-${HOME}/.npa/clusters/npa-rtxpro-mk8s/kubeconfig.resolved}"
export KUBECONTEXT="${KUBECONTEXT:-npa-rtxpro-mk8s}"
if [[ -f "${HOME}/.npa/sim2real-operator.env" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/.npa/sim2real-operator.env"
fi

DEMO="${HOME}/npa-sim2real-demo"
REPO="${DEMO}/nebius-physical-ai"
MAC_RUN="${REPO}/ops/private/sim2real-rtxpro/mac-run.sh"

if [[ ! -f "${MAC_RUN}" ]]; then
  echo "ERROR: missing ${MAC_RUN}" >&2
  echo "Pull nebius-physical-ai (branch feat/sim2real-mandatory-stages) first." >&2
  exit 1
fi

cp "${MAC_RUN}" "${DEMO}/run.sh"
chmod +x "${DEMO}/run.sh"
echo "Installed ${DEMO}/run.sh (from mac-run.sh)"

cd "${DEMO}" || exit 1
exec ./run.sh demo
