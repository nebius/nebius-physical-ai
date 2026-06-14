#!/usr/bin/env bash
# Reset local + cluster state so operators can replay the customer flow from scratch.
#
# Usage:
#   cleanup-operator.sh                    # local tmp + all terminal sim2real K8s jobs
#   cleanup-operator.sh --run-id <id>      # also delete that orchestrator job + matching s2r jobs
#   cleanup-operator.sh --run-id <id> --s3 # also delete s3://<bucket>/sim2real-b/<id>/
#   cleanup-operator.sh --dry-run
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=lib/operator-env.sh
source "${SCRIPT_DIR}/lib/operator-env.sh"
# shellcheck source=lib/operator-config.sh
source "${SCRIPT_DIR}/lib/operator-config.sh"

ROOT="$(npa_repo_root "${SCRIPT_DIR}")"
export NPA_SIM2REAL_REPO="${ROOT}"

DRY_RUN=0
INCLUDE_S3=0
RUN_ID=""
LOCAL_ONLY=0
CLUSTER_ONLY=0

usage() {
  cat <<'EOF'
Usage: cleanup-operator.sh [options]

Reset operator state to replay the customer journey (cleanup → trigger → status → sync).

Options:
  --run-id ID       Delete orchestrator job sim2real-<ID> and s2r-* jobs for that run
  --s3              With --run-id, also remove S3 artifact prefix sim2real-b/<ID>/
  --local-only      Remove /tmp/sim2real-* only (no kubectl)
  --cluster-only    Remove K8s jobs only (no local tmp)
  --dry-run         Print actions without deleting
  -h, --help

Default (no --run-id): clean local tmp dirs + delete all sim2real-* orchestrator jobs
that are Complete/Failed, plus stale s2r-* siblings (via delete-stale-s2r-jobs.sh).
Active/running jobs are kept unless --run-id targets them (then force delete).
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id)
      RUN_ID="${2:?--run-id requires value}"
      shift 2
      ;;
    --s3) INCLUDE_S3=1; shift ;;
    --local-only) LOCAL_ONLY=1; shift ;;
    --cluster-only) CLUSTER_ONLY=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h | --help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

_npa_cfg=()
while IFS= read -r _line; do
  _npa_cfg+=("${_line}")
done < <(operator_read_config "${ROOT}" 2>/dev/null || true)
BUCKET="${S3_BUCKET:-${_npa_cfg[0]:-}}"
ENDPOINT="${S3_ENDPOINT:-${_npa_cfg[1]:-https://storage.eu-north1.nebius.cloud}}"
CTX="${KUBECONTEXT:-${_npa_cfg[3]:-}}"
PREFIX="${S3_PREFIX:-sim2real-b}"
NS="${KUBENS:-default}"

_run() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY-RUN $*"
  else
    echo "$*"
    "$@"
  fi
}

_cleanup_local() {
  echo "=== Local cleanup ==="
  for dir in /tmp/sim2real-demo /tmp/sim2real-cluster; do
    if [[ -d "${dir}" ]]; then
      if [[ -n "${RUN_ID}" ]]; then
        _run rm -rf "${dir}/${RUN_ID}" "${dir}/${RUN_ID}-metrics.json" 2>/dev/null || true
        _run rm -f "${dir}/sim2real-${RUN_ID}"*.log "${dir}/sim2real-sim2real-${RUN_ID}"*.log 2>/dev/null || true
        _run rm -f "${dir}/sim2real-sim2real-${RUN_ID}"*.yaml 2>/dev/null || true
      else
        _run rm -rf "${dir}"
        _run mkdir -p "${dir}"
      fi
    fi
  done
  if [[ "${DRY_RUN}" != "1" ]]; then
    mkdir -p /tmp/sim2real-demo /tmp/sim2real-cluster
  fi
}

_cleanup_tmux() {
  if command -v tmux >/dev/null 2>&1; then
    for session in sim2real-cluster-live sim2real-demo; do
      if tmux has-session -t "${session}" 2>/dev/null; then
        _run tmux kill-session -t "${session}"
      fi
    done
  fi
}

_cleanup_cluster() {
  if ! command -v kubectl >/dev/null 2>&1; then
    echo "WARN: kubectl not found — skip cluster cleanup" >&2
    return 0
  fi
  if [[ -z "${CTX}" ]]; then
    echo "WARN: k8s_context not set — skip cluster cleanup" >&2
    return 0
  fi
  export KUBECONFIG="${KUBECONFIG:-$(operator_kubeconfig_path "${CTX}")}"
  operator_export_kubeconfig "${CTX}" "${ROOT}" 2>/dev/null || true

  echo "=== Cluster cleanup (context=${CTX} namespace=${NS}) ==="

  if [[ -n "${RUN_ID}" ]]; then
    ORCH="sim2real-${RUN_ID}"
    if kubectl --context "${CTX}" get "job/${ORCH}" -n "${NS}" >/dev/null 2>&1; then
      _run kubectl --context "${CTX}" delete job -n "${NS}" "${ORCH}" --wait=false
    else
      echo "orchestrator job/${ORCH}: not found"
    fi
    while IFS= read -r job; do
      [[ -z "${job}" ]] && continue
      _run kubectl --context "${CTX}" delete job -n "${NS}" "${job}" --wait=false
    done < <(
      kubectl --context "${CTX}" get jobs -n "${NS}" -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' 2>/dev/null \
        | grep -i "${RUN_ID}" || true
    )
  else
    while IFS= read -r job; do
      [[ -z "${job}" ]] && continue
      phase="$(kubectl --context "${CTX}" get "job/${job}" -n "${NS}" \
        -o jsonpath='{.status.active}{" "}{.status.succeeded}{" "}{.status.failed}' 2>/dev/null || echo "0 0 0")"
      read -r active succeeded failed <<<"${phase}"
      if [[ "${active:-0}" =~ ^[1-9] ]]; then
        echo "KEEP active job/${job}"
        continue
      fi
      _run kubectl --context "${CTX}" delete job -n "${NS}" "${job}" --wait=false
    done < <(
      kubectl --context "${CTX}" get jobs -n "${NS}" -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' 2>/dev/null \
        | grep -E '^sim2real-' || true
    )
    if [[ -x "${SCRIPT_DIR}/delete-stale-s2r-jobs.sh" ]]; then
      if [[ "${DRY_RUN}" == "1" ]]; then
        "${SCRIPT_DIR}/delete-stale-s2r-jobs.sh" --dry-run
      else
        "${SCRIPT_DIR}/delete-stale-s2r-jobs.sh"
      fi
    fi
  fi
}

_cleanup_s3() {
  if [[ "${INCLUDE_S3}" != "1" ]]; then
    return 0
  fi
  if [[ -z "${RUN_ID}" ]]; then
    echo "ERROR: --s3 requires --run-id" >&2
    exit 2
  fi
  if [[ -z "${BUCKET}" ]]; then
    echo "ERROR: storage.bucket required for --s3" >&2
    exit 2
  fi
  if ! command -v aws >/dev/null 2>&1; then
    echo "ERROR: aws CLI required for --s3" >&2
    exit 2
  fi
  local uri="s3://${BUCKET}/${PREFIX}/${RUN_ID}/"
  echo "=== S3 cleanup ${uri} ==="
  operator_export_storage_env "${ROOT}" || true
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY-RUN aws s3 rm ${uri} --recursive --endpoint-url ${ENDPOINT}"
  else
    aws s3 rm "${uri}" --recursive --endpoint-url "${ENDPOINT}"
  fi
}

echo "=== Sim2Real operator cleanup ==="
[[ -n "${RUN_ID}" ]] && echo "run_id=${RUN_ID}"
echo "dry_run=${DRY_RUN} include_s3=${INCLUDE_S3}"

if [[ "${CLUSTER_ONLY}" != "1" ]]; then
  _cleanup_local
  _cleanup_tmux
fi
if [[ "${LOCAL_ONLY}" != "1" ]]; then
  _cleanup_cluster
fi
_cleanup_s3

echo ""
echo "=== Next: customer flow ==="
cat <<'EOF'
  ./run.sh trigger              # submit (stock trigger from ~/.npa/sim2real-operator.env)
  ./run.sh status <RUN_ID>      # monitor
  ./run.sh sync <RUN_ID>        # sync S3 + Rerun

Or full reset + submit in one step:
  ./run.sh demo
EOF
