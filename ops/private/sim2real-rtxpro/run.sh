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
# shellcheck source=lib/operator-shell.sh
source "${SCRIPT_DIR}/lib/operator-shell.sh"
# shellcheck source=lib/customer-preflight.sh
source "${SCRIPT_DIR}/lib/customer-preflight.sh"
operator_bootstrap_shell "${NPA_SIM2REAL_REPO}"

CMD="${1:-help}"
shift || true

_usage() {
  cat <<EOF
Usage: $(basename "$0") <command> [args]

  trigger [env overrides]     Submit pipeline (default WAIT=0, prints monitor cmd)
  submit                      Same as trigger
  status <run-id>             Live stages via npa (fallback: kubectl + S3)
  sync <run-id>               Sync artifacts from S3; VISUALIZE=1 opens Rerun
  setup                       Scaffold ~/npa-sim2real-demo/private/ from templates
  seed-trigger                Upload stock lerobot/pusht trigger to YOUR bucket
  cleanup [options]           Reset tmp + K8s jobs (--run-id, --s3, --dry-run)
  demo                        cleanup + trigger (customer replication from scratch)
  rehearsal                   Sync golden run from S3 + Rerun (no cluster)
  full                        Submit + wait + sync + Rerun (cluster end-to-end)
  help

Cleanup options (passed through): --run-id ID, --s3, --local-only, --cluster-only, --dry-run

Env (trigger): TRIGGER_DATASET_URI, TRIGGER_DATASET_ID, RUN_ID, WAIT, INNER_ITERATIONS, OUTER_ITERATIONS
Config: ~/npa-sim2real-demo/private/ → ~/.npa/  (or ~/.npa/config.yaml directly)
EOF
}

case "${CMD}" in
  trigger | submit)
    customer_preflight "${NPA_SIM2REAL_DEMO:-${HOME}/npa-sim2real-demo}" || exit 1
    if [ -f "${HOME}/.npa/sim2real-operator.env" ]; then
      # shellcheck disable=SC1091
      source "${HOME}/.npa/sim2real-operator.env"
    fi
    export WAIT="${WAIT:-0}"
    exec "${OPS}/trigger-pipeline.sh"
    ;;
  status)
    RUN_ID="${1:?usage: $(basename "$0") status <run-id>}"
    exec "${OPS}/status-run-npa.sh" "${RUN_ID}" --watch
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
    customer_preflight "${NPA_SIM2REAL_DEMO:-${HOME}/npa-sim2real-demo}" || exit 1
    "${OPS}/cleanup-operator.sh" "$@"
    echo ""
    if [ -f "${HOME}/.npa/sim2real-operator.env" ]; then
      # shellcheck disable=SC1091
      source "${HOME}/.npa/sim2real-operator.env"
    fi
    export WAIT="${WAIT:-0}"
    exec "${OPS}/trigger-pipeline.sh"
    ;;
  setup)
    exec "${OPS}/setup-customer-demo.sh" "${NPA_SIM2REAL_DEMO:-${HOME}/npa-sim2real-demo}"
    ;;
  seed-trigger)
    exec "${OPS}/seed-stock-trigger.sh"
    ;;
  rehearsal)
    if [ -z "${RUN_ID:-}" ]; then
      echo "ERROR: set RUN_ID=<completed-run-id> for rehearsal" >&2
      exit 1
    fi
    export SUBMIT=0 VISUALIZE="${VISUALIZE:-1}"
    exec "${OPS}/run-demo.sh"
    ;;
  full)
    # Legacy private-repo command: submit, wait on cluster, sync + viz.
    if [ -f "${HOME}/.npa/sim2real-operator.env" ]; then
      # shellcheck disable=SC1091
      source "${HOME}/.npa/sim2real-operator.env"
    fi
    export WAIT=1 VISUALIZE="${VISUALIZE:-1}" SUBMIT=1
    exec "${OPS}/run-demo.sh"
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
