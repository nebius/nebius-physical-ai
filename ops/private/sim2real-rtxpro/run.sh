#!/usr/bin/env bash
# Mac operator interface for sim2real staged runs (kubectl + S3; minimal npa CLI coupling).
#
# Layout A (demo pack on laptop):
#   ~/npa-sim2real-demo/run.sh              -> copy from mac-run.sh
#   ~/npa-sim2real-demo/nebius-physical-ai/
#
# Layout B (repo checkout):
#   ./ops/private/sim2real-rtxpro/run.sh trigger
#
# Commands:
#   trigger   Submit stock/custom trigger run to cluster (WAIT=0 default)
#   status    Live status for RUN_ID (kubectl + S3)
#   sync      Sync completed run from S3 + optional Rerun viz
#   cleanup   Reset local tmp + terminal K8s jobs (replay customer from scratch)
#   demo      cleanup + trigger (stock Franka customer path)
#   submit    Submit only (alias for WAIT=0 trigger)
#   help
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=lib/operator-env.sh
source "${SCRIPT_DIR}/lib/operator-env.sh"
# shellcheck source=lib/operator-config.sh
source "${SCRIPT_DIR}/lib/operator-config.sh"

_REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
if [ -f "${_REPO_ROOT}/npa/pyproject.toml" ]; then
  export NPA_SIM2REAL_REPO="${_REPO_ROOT}"
elif [ -n "${NPA_SIM2REAL_REPO:-}" ] && [ -f "${NPA_SIM2REAL_REPO}/npa/pyproject.toml" ]; then
  :
else
  export NPA_SIM2REAL_REPO="$(npa_repo_root "${SCRIPT_DIR}")"
fi

OPS="${SCRIPT_DIR}"
CMD="${1:-help}"
shift || true

_usage() {
  cat <<EOF
Usage: $(basename "$0") <command> [args]

  trigger [env overrides]     Submit pipeline (default WAIT=0, prints monitor cmd)
  submit                      Same as trigger
  status <run-id>             Live kubectl + S3 stage checklist (--watch)
  sync <run-id>               Sync artifacts from S3; VISUALIZE=1 opens Rerun
  cleanup [options]           Reset tmp + K8s jobs (--run-id, --s3, --dry-run)
  demo                        cleanup + trigger (customer replication from scratch)
  help

Cleanup options (passed through): --run-id ID, --s3, --local-only, --cluster-only, --dry-run

Env (trigger): TRIGGER_DATASET_URI, TRIGGER_DATASET_ID, RUN_ID, WAIT, INNER_ITERATIONS, OUTER_ITERATIONS
Config: ~/.npa/config.yaml + credentials.yaml; kubeconfig under ~/.npa/clusters/<context>/
EOF
}

case "${CMD}" in
  trigger | submit)
    if [ -f "${HOME}/.npa/sim2real-operator.env" ]; then
      # shellcheck disable=SC1091
      source "${HOME}/.npa/sim2real-operator.env"
    fi
    export WAIT="${WAIT:-0}"
    exec "${OPS}/trigger-pipeline.sh"
    ;;
  status)
    RUN_ID="${1:?usage: $(basename "$0") status <run-id>}"
    exec "${OPS}/status-run-local.sh" "${RUN_ID}" --watch
    ;;
  sync)
    RUN_ID="${1:?usage: $(basename "$0") sync <run-id>}"
    export RUN_ID SUBMIT=0 VISUALIZE="${VISUALIZE:-1}"
    exec "${OPS}/run-demo.sh"
    ;;
  cleanup)
    exec "${OPS}/cleanup-operator.sh" "$@"
    ;;
  demo)
    echo "=== Customer replication: cleanup → trigger ==="
    "${OPS}/cleanup-operator.sh" "$@"
    echo ""
    if [ -f "${HOME}/.npa/sim2real-operator.env" ]; then
      # shellcheck disable=SC1091
      source "${HOME}/.npa/sim2real-operator.env"
    fi
    export WAIT="${WAIT:-0}"
    exec "${OPS}/trigger-pipeline.sh"
    ;;
  help | -h | --help)
    _usage
    ;;
  *)
    echo "Unknown command: ${CMD}" >&2
    _usage >&2
    exit 1
    ;;
esac
