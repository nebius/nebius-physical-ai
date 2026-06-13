#!/usr/bin/env bash
# =============================================================================
# COPY-PASTE into a new Mac terminal (full customer replication from scratch).
#
# One-time setup (if run.sh is not installed yet):
#   cp ~/npa-sim2real-demo/nebius-physical-ai/ops/private/sim2real-rtxpro/mac-run.sh \
#      ~/npa-sim2real-demo/run.sh && chmod +x ~/npa-sim2real-demo/run.sh
#
# Then paste this entire block:
# =============================================================================
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/opt/homebrew/bin:${HOME}/.nebius/bin:${PATH}"
export KUBECONFIG="${KUBECONFIG:-${HOME}/.npa/clusters/npa-rtxpro-mk8s/kubeconfig.resolved}"
export KUBECONTEXT="${KUBECONTEXT:-npa-rtxpro-mk8s}"
if [[ -f "${HOME}/.npa/sim2real-operator.env" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/.npa/sim2real-operator.env"
fi

DEMO="${HOME}/npa-sim2real-demo"
cd "${DEMO}" || { echo "ERROR: missing ${DEMO}" >&2; exit 1; }

echo "=== 1/3 cleanup (reset for customer replay) ==="
./run.sh cleanup

echo ""
echo "=== 2/3 trigger (submit to cluster) ==="
./run.sh trigger

echo ""
echo "=== 3/3 next steps (use RUN_ID from trigger output above) ==="
echo "  ./run.sh status <RUN_ID>"
echo "  ./run.sh sync <RUN_ID>"
