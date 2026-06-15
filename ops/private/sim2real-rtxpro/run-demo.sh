#!/usr/bin/env bash
# Customer sim2real demo — laptop is interface only; compute on Nebius RTX cluster.
#
# Workflow (same as production):
#   1. Submit via npa workbench workflow submit (sim2real/runbook.yaml)
#   2. Wait for completion (optional)
#   3. Sync reports/sim2real.rrd + stage JSON from S3
#   4. Open local Rerun web viewer
#
# Usage:
#   ./ops/private/sim2real-rtxpro/run-demo.sh              # submit + wait + viz
#   RUN_ID=<completed-run> ./ops/private/sim2real-rtxpro/run-demo.sh   # sync + viz only
#   WAIT=0 ./ops/private/sim2real-rtxpro/run-demo.sh       # submit, print monitor cmd
#
# Prerequisites: ~/.npa/config.yaml, ~/.npa/credentials.yaml, kubeconfig, kubectl
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=lib/operator-env.sh
source "${SCRIPT_DIR}/lib/operator-env.sh"
# shellcheck source=lib/demo-common.sh
source "${SCRIPT_DIR}/lib/demo-common.sh"
# shellcheck source=lib/operator-config.sh
source "${SCRIPT_DIR}/lib/operator-config.sh"
# shellcheck source=lib/asset-profile-guard.sh
source "${SCRIPT_DIR}/lib/asset-profile-guard.sh"

ROOT="$(demo_common_root)"
OPS="${SCRIPT_DIR}"
PY="${ROOT}/npa/.venv/bin/python"
NPA="${ROOT}/npa/.venv/bin/npa"
LOG_DIR="/tmp/sim2real-demo"
mkdir -p "${LOG_DIR}"

RUN_ID="${RUN_ID:-}"
if [ -n "${RUN_ID}" ]; then
  RUN_ID="$(operator_normalize_staged_run_id "${RUN_ID}")"
  export RUN_ID
fi
WAIT="${WAIT:-1}"
VISUALIZE="${VISUALIZE:-1}"
SYNC_DIR="${SYNC_DIR:-/tmp/sim2real-demo/${RUN_ID:-pending}}"
SUBMIT="${SUBMIT:-1}"

if [ -n "${RUN_ID}" ] && [ "${SUBMIT}" != "0" ]; then
  # RUN_ID set — reuse completed cluster run (sync + viz only)
  SUBMIT=0
fi

SYNC_ONLY=0
if [ "${SUBMIT}" = "0" ] && [ -n "${RUN_ID}" ]; then
  SYNC_ONLY=1
fi

demo_bootstrap_venv "${ROOT}"
if [ "${SYNC_ONLY}" = "1" ]; then
  demo_preflight "${ROOT}" 0
else
  demo_preflight "${ROOT}" 1
fi

_cfg=()
while IFS= read -r _line; do
  _cfg+=("${_line}")
done < <(demo_read_storage_config "${ROOT}")
BUCKET="${S3_BUCKET:-${_cfg[0]:-}}"
ENDPOINT="${S3_ENDPOINT:-${_cfg[1]:-}}"
REGISTRY="${REGISTRY:-${_cfg[2]:-}}"
DEFAULT_CTX="${_cfg[3]:-}"
if [ "${SYNC_ONLY}" != "1" ]; then
  if [ -z "${DEFAULT_CTX}" ]; then
    echo "ERROR: k8s_context not set in ~/.npa/config.yaml" >&2
    exit 1
  fi
  export KUBECONFIG="${KUBECONFIG:-$(operator_kubeconfig_path "${KUBECONTEXT:-${DEFAULT_CTX}}")}"
  operator_export_kubeconfig "${KUBECONTEXT:-${DEFAULT_CTX}}" "${ROOT}" || exit 1
  export KUBECONTEXT="${KUBECONTEXT:-${DEFAULT_CTX}}"
fi

if [ -z "${BUCKET}" ]; then
  echo "ERROR: storage.bucket required in ~/.npa/config.yaml" >&2
  exit 1
fi
if [ "${SYNC_ONLY}" != "1" ] && [ -z "${REGISTRY}" ]; then
  echo "ERROR: storage.registry required in ~/.npa/config.yaml for cluster submit" >&2
  exit 1
fi

_submit_and_wait() {
  local submit_log="${LOG_DIR}/submit.log"
  echo "=== Submit sim2real job to cluster ${KUBECONTEXT} ===" | tee "${submit_log}"
  echo "bucket=${BUCKET} registry=${REGISTRY}" | tee -a "${submit_log}"

  export S3_BUCKET="${BUCKET}"
  export S3_ENDPOINT="${ENDPOINT}"
  export REGISTRY="${REGISTRY}"
  export INNER_ITERATIONS="${INNER_ITERATIONS:-1}"
  export OUTER_ITERATIONS="${OUTER_ITERATIONS:-2}"
  export LAUNCH_MONITOR="${LAUNCH_MONITOR:-0}"
  if [ -n "${RUN_ID:-}" ]; then
    export RUN_ID
  fi

  local submit_script
  submit_script="$(operator_resolve_submit_script "${OPS}")"
  if operator_use_workbench_submit; then
    echo "submit_path=workbench (npa workbench workflow submit + runbook.yaml)" | tee -a "${submit_log}"
  else
    echo "submit_path=kubectl (NPA_USE_KUBECTL_SUBMIT=1 — direct script, bypasses npa CLI)" | tee -a "${submit_log}"
  fi
  "${submit_script}" 2>&1 | tee -a "${submit_log}"

  if ! RUN_ID="$(operator_parse_submit_run_id "${submit_log}")"; then
    echo "ERROR: could not parse run_id from submit output" >&2
    exit 1
  fi
  export RUN_ID
  JOB="$(operator_parse_submit_job "${submit_log}" "${RUN_ID}" || true)"
  JOB="${JOB:-$(operator_orchestrator_job_name "${RUN_ID}")}"
  echo "Submitted run_id=${RUN_ID}"
  echo "Submitted job=${JOB}"

  if [ "${WAIT}" != "1" ]; then
    echo ""
    echo "WAIT=0 — job running on cluster. Monitor:"
    echo "  ${OPS}/run.sh status ${RUN_ID}"
    echo "When complete:"
    echo "  ${OPS}/run.sh sync ${RUN_ID}"
    exit 0
  fi

  if operator_use_workbench_submit; then
    echo "=== Waiting for workbench workflow (GPU stages on Nebius) ==="
    "${OPS}/status-run-npa.sh" "${RUN_ID}" --watch
  else
    echo "=== Waiting for kubectl Job ${JOB} (GPU stages on Nebius) ==="
    "${OPS}/monitor-k8s-job.sh" "${JOB}"
  fi
}

_sync_from_s3() {
  SYNC_DIR="/tmp/sim2real-demo/${RUN_ID}"
  echo "=== Sync artifacts from S3 ==="
  echo "run_id=${RUN_ID}"
  echo "s3://${BUCKET}/${S3_PREFIX:-sim2real-b}/${RUN_ID}/"
  echo "local_dir=${SYNC_DIR}/"
  S3_BUCKET="${BUCKET}" S3_ENDPOINT="${ENDPOINT}" \
    "${OPS}/prestage-offline-run.sh" "${RUN_ID}" "${SYNC_DIR}"
}

echo "=== Sim2Real customer demo (cluster compute, local Rerun interface) ==="
if [ "${SYNC_ONLY}" = "1" ]; then
  echo "mode=sync-only run_id=${RUN_ID} bucket=${BUCKET}"
else
  echo "cluster=${KUBECONTEXT} bucket=${BUCKET}"
fi

if [ "${SUBMIT}" = "1" ]; then
  _submit_and_wait
else
  if [ -z "${RUN_ID}" ]; then
    echo "ERROR: set RUN_ID=<completed-run> for sync-only, or SUBMIT=1 to launch new job" >&2
    exit 1
  fi
  echo "Using existing Nebius run: ${RUN_ID}"
fi

_sync_from_s3

REPORT="${SYNC_DIR}/reports/sim2real-report.json"
RRD="${SYNC_DIR}/reports/sim2real.rrd"

echo ""
echo "=== Run summary (from Nebius cluster) ==="
"${PY}" - <<PY
import json, sys
from pathlib import Path
report = json.loads(Path("${REPORT}").read_text())
comps = {c["name"]: c for c in report.get("components", [])}
s14 = comps.get("stage_14_rerun_viz", {})
print(json.dumps({
    "run_id": report.get("run_id"),
    "s3_root": report.get("s3_artifacts", {}).get("root"),
    "decision": report.get("outer_loop", {}).get("latest_decision", {}).get("decision"),
    "reward_trend": report.get("inner_loop", {}).get("reward_trend"),
    "stage_14_rerun_viz_tier": s14.get("tier"),
}, indent=2))
if s14.get("tier") != "WORKS":
    sys.exit(f"stage_14_rerun_viz tier={s14.get('tier')!r}")
PY

demo_visualize_rrd "${RRD}" "${LOG_DIR}" "${RUN_ID}" "${ROOT}"

echo ""
echo "S3 artifact root: s3://${BUCKET}/${S3_PREFIX:-sim2real-b}/${RUN_ID}/"
echo "Local sync: ${SYNC_DIR}"
echo "Submit log: ${LOG_DIR}/submit.log"
