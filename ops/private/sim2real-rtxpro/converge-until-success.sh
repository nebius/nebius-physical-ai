#!/usr/bin/env bash
# Submit sim2real staged runs until report.json lands; then scale to 10k envs.
# Designed for tmux: survives disconnect; pulls latest branch each attempt.
#
# Usage:
#   converge-until-success.sh              # 800 envs until success, then 10k
#   converge-until-success.sh --once       # single 800-env attempt (no 10k follow-up)
#   NPA_ENV_COUNT=800 converge-until-success.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=lib/operator-env.sh
source "${SCRIPT_DIR}/lib/operator-env.sh"
# shellcheck source=lib/operator-config.sh
source "${SCRIPT_DIR}/lib/operator-config.sh"
ROOT="$(npa_repo_root "${SCRIPT_DIR}")"
export NPA_SIM2REAL_REPO="${ROOT}"

ONCE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --once) ONCE=1; shift ;;
    -h | --help)
      sed -n '2,8p' "$0"
      exit 0
      ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

if [ -f "${SCRIPT_DIR}/env.local" ]; then
  set -a
  # shellcheck source=/dev/null
  source "${SCRIPT_DIR}/env.local"
  set +a
fi

_npa_cfg=()
while IFS= read -r _line; do
  _npa_cfg+=("${_line}")
done < <(operator_read_config "${ROOT}" 2>/dev/null || true)
BUCKET="${S3_BUCKET:-${_npa_cfg[0]:-}}"
ENDPOINT="${S3_ENDPOINT:-${S3_ENDPOINT_URL:-${_npa_cfg[1]:-https://storage.eu-north1.nebius.cloud}}}"
CTX="${KUBECONTEXT:-${_npa_cfg[3]:-}}"
export KUBECONFIG="${KUBECONFIG:-$(operator_kubeconfig_path "${CTX}")}"
operator_export_kubeconfig "${CTX}" "${ROOT}" || true
export S3_BUCKET="${BUCKET}"
export NPA_SIM2REAL_BUCKET="${BUCKET}"
if [[ "${NPA_SIM2REAL_TRIGGER_DATASET_URI:-}" == *YOUR-BUCKET* ]] && [ -n "${BUCKET}" ]; then
  export NPA_SIM2REAL_TRIGGER_DATASET_URI="${NPA_SIM2REAL_TRIGGER_DATASET_URI/YOUR-BUCKET/${BUCKET}}"
fi
BRANCH="${NPA_SOURCE_REF:-feat/sim2real-mandatory-stages}"
PHASE="${CONVERGE_PHASE:-800}"
TARGET_ENVS="${NPA_ENV_COUNT:-800}"
if [ "${PHASE}" = "10k" ]; then
  TARGET_ENVS="${NPA_ENV_COUNT:-10000}"
fi

STATE_DIR="/tmp/sim2real-cluster/converge"
mkdir -p "${STATE_DIR}"
LOG="${STATE_DIR}/converge.log"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${LOG}"
}

sync_repo() {
  log "git fetch origin ${BRANCH}"
  git -C "${ROOT}" fetch origin "${BRANCH}" 2>&1 | tee -a "${LOG}"
  git -C "${ROOT}" checkout "${BRANCH}" 2>&1 | tee -a "${LOG}" || true
  git -C "${ROOT}" reset --hard "origin/${BRANCH}" 2>&1 | tee -a "${LOG}"
  log "HEAD $(git -C "${ROOT}" log -1 --oneline)"
}

s3_report_ok() {
  local run_id="$1"
  RUN_ID_CHECK="${run_id}" BUCKET_CHECK="${BUCKET}" "${ROOT}/npa/.venv/bin/python" - <<'PY'
import boto3, os, sys, yaml
from pathlib import Path
run_id = os.environ["RUN_ID_CHECK"]
bucket = os.environ["BUCKET_CHECK"]
prefix = f"sim2real-b/{run_id}/reports/sim2real-report.json"
creds = yaml.safe_load(Path.home().joinpath(".npa/credentials.yaml").read_text())["storage"]
client = boto3.client(
    "s3",
    endpoint_url=creds["endpoint_url"],
    aws_access_key_id=creds["aws_access_key_id"],
    aws_secret_access_key=creds["aws_secret_access_key"],
)
try:
    client.head_object(Bucket=bucket, Key=prefix)
except Exception:
    sys.exit(1)
sys.exit(0)
PY
}

cleanup_failed_run() {
  local run_id="$1"
  log "storage cleanup for failed run ${run_id}"
  bash "${SCRIPT_DIR}/cleanup-operator.sh" --run-id "${run_id}" --cluster-only 2>&1 | tee -a "${LOG}" || true
  if [ "${CONVERGE_S3_CLEANUP:-1}" = "1" ]; then
    bash "${SCRIPT_DIR}/cleanup-operator.sh" --run-id "${run_id}" --s3 2>&1 | tee -a "${LOG}" || true
  fi
}

run_attempt() {
  local env_count="$1"
  local attempt="$2"
  sync_repo
  export LAUNCH_MONITOR=0
  export NPA_ENV_COUNT="${env_count}"
  export NPA_TRAIN_FRACTION="${NPA_TRAIN_FRACTION:-0.8}"
  export NPA_SIM2REAL_HELDOUT_EVAL_LIMIT="${NPA_SIM2REAL_HELDOUT_EVAL_LIMIT:-8}"
  export INNER_ITERATIONS="${INNER_ITERATIONS:-2}"
  export OUTER_ITERATIONS="${OUTER_ITERATIONS:-2}"
  export RUN_ID="converge-${PHASE}-a${attempt}-$(date -u +%Y%m%dT%H%M%Sz | tr '[:upper:]' '[:lower:]')"

  log "=== attempt ${attempt} phase=${PHASE} envs=${env_count} run_id=${RUN_ID} ==="
  bash "${SCRIPT_DIR}/submit-k8s-staged-job.sh" 2>&1 | tee -a "${LOG}"
  local job="sim2real-${RUN_ID}"
  bash "${SCRIPT_DIR}/monitor-k8s-job.sh" "${job}" 2>&1 | tee -a "${LOG}" || true

  if s3_report_ok "${RUN_ID}"; then
    log "SUCCESS run_id=${RUN_ID} report=sim2real-b/${RUN_ID}/reports/sim2real-report.json"
    echo "${RUN_ID}" > "${STATE_DIR}/last-success-${PHASE}.txt"
    return 0
  fi

  log "FAIL run_id=${RUN_ID} (no sim2real-report.json on S3)"
  kubectl --context "${KUBECONTEXT:-${_npa_cfg[3]:-}}" logs "job/${job}" --tail=80 2>&1 | tee -a "${LOG}" || true
  cleanup_failed_run "${RUN_ID}"
  echo "${RUN_ID}" >> "${STATE_DIR}/failures-${PHASE}.txt"
  return 1
}

converge_phase() {
  local env_count="$1"
  local attempt=1
  local max_attempts="${CONVERGE_MAX_ATTEMPTS:-999}"
  local sleep_s="${CONVERGE_RETRY_SLEEP_S:-90}"

  while [ "${attempt}" -le "${max_attempts}" ]; do
    if run_attempt "${env_count}" "${attempt}"; then
      return 0
    fi
    log "retry in ${sleep_s}s (attempt $((attempt + 1)))"
    sleep "${sleep_s}"
    attempt=$((attempt + 1))
  done
  log "gave up after ${max_attempts} attempts phase=${PHASE}"
  return 1
}

log "converge loop start phase=${PHASE} target_envs=${TARGET_ENVS} bucket=${BUCKET}"
trap 'log "converge loop interrupted"' INT TERM

if converge_phase "${TARGET_ENVS}"; then
  if [ "${ONCE}" = "1" ]; then
    log "done (--once)"
    exit 0
  fi
  if [ "${PHASE}" = "10k" ]; then
    log "10k phase success — converge complete"
    exit 0
  fi
  log "800-env success — starting 10k validation phase (loop until report lands)"
  export CONVERGE_PHASE=10k
  export PHASE=10k
  exec env NPA_ENV_COUNT=10000 CONVERGE_PHASE=10k "$0"
fi

exit 1
