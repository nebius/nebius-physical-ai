#!/usr/bin/env bash
# Customer sim2real demo — laptop is interface only; compute on Nebius RTX cluster.
#
# Workflow (same as production):
#   1. Submit staged job to Kubernetes (GPU sibling Jobs + S3 upload)
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
# shellcheck source=lib/demo-common.sh
source "${SCRIPT_DIR}/lib/demo-common.sh"
# shellcheck source=lib/operator-config.sh
source "${SCRIPT_DIR}/lib/operator-config.sh"

ROOT="$(demo_common_root)"
OPS="${SCRIPT_DIR}"
PY="${ROOT}/npa/.venv/bin/python"
LOG_DIR="/tmp/sim2real-demo"
mkdir -p "${LOG_DIR}"

demo_bootstrap_venv "${ROOT}"
demo_preflight "${ROOT}"

readarray -t _cfg < <(demo_read_storage_config "${ROOT}")
BUCKET="${S3_BUCKET:-${_cfg[0]:-}}"
ENDPOINT="${S3_ENDPOINT:-${_cfg[1]:-}}"
REGISTRY="${REGISTRY:-${_cfg[2]:-}}"
DEFAULT_CTX="${_cfg[3]:-}"
if [ -z "${DEFAULT_CTX}" ]; then
  echo "ERROR: k8s_context not set in ~/.npa/config.yaml" >&2
  exit 1
fi
export KUBECONFIG="${KUBECONFIG:-$(operator_kubeconfig_path "${KUBECONTEXT:-${DEFAULT_CTX}}")}"
export KUBECONTEXT="${KUBECONTEXT:-${DEFAULT_CTX}}"

if [ -z "${BUCKET}" ] || [ -z "${REGISTRY}" ]; then
  echo "ERROR: storage.bucket and storage.registry required in ~/.npa/config.yaml" >&2
  exit 1
fi

RUN_ID="${RUN_ID:-}"
WAIT="${WAIT:-1}"
VISUALIZE="${VISUALIZE:-1}"
SYNC_DIR="${SYNC_DIR:-/tmp/sim2real-demo/${RUN_ID:-pending}}"
SUBMIT="${SUBMIT:-1}"

if [ -n "${RUN_ID}" ] && [ "${SUBMIT}" = "0" ]; then
  : # sync-only mode
elif [ -n "${RUN_ID}" ] && [ "${SUBMIT}" != "0" ]; then
  # RUN_ID set but SUBMIT not disabled — treat as sync-only (reuse completed run)
  SUBMIT=0
fi

_submit_and_wait() {
  local submit_log="${LOG_DIR}/submit.log"
  echo "=== Submit sim2real job to cluster ${KUBECONTEXT} ===" | tee "${submit_log}"
  echo "bucket=${BUCKET} registry=${REGISTRY}" | tee -a "${submit_log}"

  LAUNCH_MONITOR=0 \
    INNER_ITERATIONS="${INNER_ITERATIONS:-1}" \
    OUTER_ITERATIONS="${OUTER_ITERATIONS:-2}" \
    S3_BUCKET="${BUCKET}" \
    REGISTRY="${REGISTRY}" \
    S3_ENDPOINT="${ENDPOINT}" \
    KUBECONTEXT="${KUBECONTEXT}" \
    "${OPS}/submit-k8s-staged-job.sh" 2>&1 | tee -a "${submit_log}"

  RUN_ID="$(grep -oE 'run_id=[^ ]+' "${submit_log}" | tail -1 | cut -d= -f2)"
  JOB="$(grep -oE 'job=[^ ]+' "${submit_log}" | tail -1 | cut -d= -f2)"
  if [ -z "${RUN_ID}" ] || [ -z "${JOB}" ]; then
    echo "ERROR: could not parse run_id/job from submit output" >&2
    exit 1
  fi
  echo "Submitted run_id=${RUN_ID} job=${JOB}"

  if [ "${WAIT}" != "1" ]; then
    echo ""
    echo "WAIT=0 — job running on cluster. When complete:"
    echo "  RUN_ID=${RUN_ID} SUBMIT=0 ${OPS}/run-demo.sh"
    echo "  ${OPS}/monitor-k8s-job.sh ${JOB}"
    exit 0
  fi

  echo "=== Waiting for cluster job (GPU stages on Nebius) ==="
  "${OPS}/monitor-k8s-job.sh" "${JOB}"
}

_sync_from_s3() {
  SYNC_DIR="/tmp/sim2real-demo/${RUN_ID}"
  echo "=== Sync artifacts from S3 ==="
  echo "s3://${BUCKET}/${S3_PREFIX:-sim2real-b}/${RUN_ID}/ -> ${SYNC_DIR}/"
  S3_BUCKET="${BUCKET}" S3_ENDPOINT="${ENDPOINT}" \
    "${OPS}/prestage-offline-run.sh" "${RUN_ID}" "${SYNC_DIR}"
}

echo "=== Sim2Real customer demo (cluster compute, local Rerun interface) ==="
echo "cluster=${KUBECONTEXT} bucket=${BUCKET}"

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
