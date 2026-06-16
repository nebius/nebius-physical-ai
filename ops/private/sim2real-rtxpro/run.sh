#!/usr/bin/env bash
# Mac operator interface for sim2real staged runs (kubectl + S3; minimal npa CLI coupling).
#
#   ./ops/private/sim2real-rtxpro/run.sh trigger
#   ./ops/private/sim2real-rtxpro/run.sh sync <run-id>
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
operator_bootstrap_shell "${NPA_SIM2REAL_REPO}"

CMD="${1:-help}"
shift || true

_usage() {
  cat <<EOF
Usage: $(basename "$0") <command> [args]

  trigger [env overrides]     Submit pipeline (default WAIT=0, prints monitor cmd)
  submit                      Same as trigger (ignores stale RUN_ID in shell)
  status <run-id>             Live kubectl + S3 stage checklist (--watch)
  sync <run-id>               Sync artifacts from S3; VISUALIZE=1 opens Rerun
  demo                        cleanup + trigger (stock Franka customer path)
  rehearsal                   Sync golden run from S3 + Rerun (no cluster)
  full                        Submit + wait + sync + Rerun (cluster end-to-end)
  rerun-host <run-id>         Deploy shared cluster Rerun viewer (stable public_url)
  help

Env (trigger): TRIGGER_DATASET_URI, TRIGGER_DATASET_ID, WAIT, INNER_ITERATIONS, OUTER_ITERATIONS
Before trigger: unset RUN_ID if you exported it for a prior status/sync session.
Config: ~/.npa/config.yaml + credentials.yaml; kubeconfig under ~/.npa/clusters/<context>/
EOF
}

case "${CMD}" in
  trigger | submit)
    if [ -f "${HOME}/.npa/sim2real-operator.env" ]; then
      # shellcheck disable=SC1091
      source "${HOME}/.npa/sim2real-operator.env"
    fi
    unset RUN_ID
    export WAIT="${WAIT:-0}"
    exec "${OPS}/trigger-pipeline.sh"
    ;;
  status)
    RUN_ID="$(operator_normalize_staged_run_id "${1:?usage: $(basename "$0") status <run-id>}")"
    if [ -x "${OPS}/status-run-npa.sh" ]; then
      exec "${OPS}/status-run-npa.sh" "${RUN_ID}" --watch
    fi
    exec "${OPS}/monitor-k8s-job.sh" "$(operator_orchestrator_job_name "${RUN_ID}")"
    ;;
  sync)
    RUN_ID="$(operator_normalize_staged_run_id "${1:?usage: $(basename "$0") sync <run-id>}")"
    export RUN_ID SUBMIT=0 VISUALIZE="${VISUALIZE:-1}"
    exec "${OPS}/run-demo.sh"
    ;;
  demo)
    echo "=== Customer replication: cleanup → trigger ==="
    if [ -x "${OPS}/cleanup-operator.sh" ]; then
      "${OPS}/cleanup-operator.sh" "$@"
    fi
    echo ""
    if [ -f "${HOME}/.npa/sim2real-operator.env" ]; then
      # shellcheck disable=SC1091
      source "${HOME}/.npa/sim2real-operator.env"
    fi
    unset RUN_ID
    export WAIT="${WAIT:-0}"
    exec "${OPS}/trigger-pipeline.sh"
    ;;
  rehearsal)
    export SUBMIT=0 VISUALIZE="${VISUALIZE:-1}"
    export RUN_ID="${RUN_ID:-rtxpro-isaac-2x2-20260613t043658z}"
    exec "${OPS}/run-demo.sh"
    ;;
  full)
    if [ -f "${HOME}/.npa/sim2real-operator.env" ]; then
      # shellcheck disable=SC1091
      source "${HOME}/.npa/sim2real-operator.env"
    fi
    unset RUN_ID
    export WAIT=1 VISUALIZE="${VISUALIZE:-1}" SUBMIT=1
    exec "${OPS}/run-demo.sh"
    ;;
  rerun-host)
    RUN_ID="$(operator_normalize_staged_run_id "${1:?usage: $(basename "$0") rerun-host <run-id>}")"
    ROOT="${NPA_SIM2REAL_REPO}"
    NPA="${ROOT}/npa/.venv/bin/npa"
    CTX="${KUBECONTEXT:-}"
    if [ -z "${CTX}" ]; then
      npa_read_lines _cfg operator_read_config "${ROOT}"
      CTX="${_cfg[3]:-}"
    fi
    if [ -n "${CTX}" ]; then
      operator_export_kubeconfig "${CTX}" "${ROOT}" || true
    fi
    exec "${NPA}" workbench sim2real rerun serve \
      --run-id "${RUN_ID}" \
      ${CTX:+--cluster-name "${CTX}"}
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
