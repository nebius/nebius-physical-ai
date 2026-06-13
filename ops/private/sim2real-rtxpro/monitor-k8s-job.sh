#!/usr/bin/env bash
# Poll a sim2real staged Job until complete/failed, then dump logs + metrics.
# Survives terminal hangup (SIGHUP) and IDE disconnect; ignore SIGINT in the
# polling loop so accidental Ctrl+C in an attached tmux pane does not stop monitoring.
set -euo pipefail

JOB="${1:?usage: monitor-k8s-job.sh <job-name>}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=lib/operator-config.sh
source "${SCRIPT_DIR}/lib/operator-config.sh"
ROOT="$(npa_repo_root "${SCRIPT_DIR}")"
_npa_cfg=()
while IFS= read -r _line; do
  _npa_cfg+=("${_line}")
done < <(operator_read_config "${ROOT}" 2>/dev/null || true)
CTX="${KUBECONTEXT:-${_npa_cfg[3]:-}}"
if [ -z "${CTX}" ]; then
  echo "Set k8s_context in ~/.npa/config.yaml" >&2
  exit 1
fi
export KUBECONFIG="${KUBECONFIG:-$(operator_kubeconfig_path "${CTX}")}"
NS="${KUBENS:-default}"
TIMEOUT_S="${MONITOR_TIMEOUT_S:-7200}"
POLL_S="${MONITOR_POLL_S:-30}"
LOG="/tmp/sim2real-cluster/${JOB}-monitor.log"
mkdir -p /tmp/sim2real-cluster

log() {
  printf '%s\n' "$*" | tee -a "${LOG}"
}

# Ignore hangups from tmux detach / SSH drop; defer SIGINT until we are idle.
trap 'INTERRUPTED=1' INT
trap '' HUP

log "Monitoring ${JOB} on ${CTX} (timeout=${TIMEOUT_S}s poll=${POLL_S}s)..."

start_epoch="$(date +%s)"
status="unknown"
while true; do
  if [[ "${INTERRUPTED:-0}" == "1" ]]; then
    log "SIGINT received — continuing monitor (detach tmux instead of Ctrl+C to leave running)"
    INTERRUPTED=0
  fi

  if ! kubectl --context "${CTX}" get "job/${JOB}" -n "${NS}" >/dev/null 2>&1; then
    log "ERROR: job/${JOB} not found in namespace ${NS}"
    status="missing"
    break
  fi

  succeeded="$(kubectl --context "${CTX}" get "job/${JOB}" -n "${NS}" \
    -o jsonpath='{.status.succeeded}' 2>/dev/null || true)"
  failed="$(kubectl --context "${CTX}" get "job/${JOB}" -n "${NS}" \
    -o jsonpath='{.status.failed}' 2>/dev/null || true)"
  active="$(kubectl --context "${CTX}" get "job/${JOB}" -n "${NS}" \
    -o jsonpath='{.status.active}' 2>/dev/null || true)"

  now="$(date +%s)"
  elapsed=$((now - start_epoch))
  log "t=${elapsed}s active=${active:-0} succeeded=${succeeded:-0} failed=${failed:-0}"

  if [[ "${succeeded:-0}" =~ ^[1-9] ]]; then
    status="complete"
    break
  fi
  if [[ "${failed:-0}" =~ ^[1-9] ]]; then
    status="failed"
    break
  fi
  if (( elapsed >= TIMEOUT_S )); then
    status="timeout"
    log "ERROR: timed out after ${TIMEOUT_S}s"
    break
  fi
  sleep "${POLL_S}"
done

POD="$(kubectl --context "${CTX}" get pods -n "${NS}" -l "job-name=${JOB}" \
  -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
if [[ -n "${POD}" ]]; then
  log "Pod: ${POD}"
  kubectl --context "${CTX}" logs -n "${NS}" "${POD}" --all-containers=true 2>&1 \
    | tee -a "${LOG}" | tail -80
else
  log "WARN: no pod found for job/${JOB}"
fi

kubectl --context "${CTX}" get "job/${JOB}" -n "${NS}" -o wide 2>&1 | tee -a "${LOG}" || true

case "${status}" in
  complete) log "DONE status=complete" ;;
  failed) log "DONE status=failed" ;;
  timeout) log "DONE status=timeout" ;;
  missing) log "DONE status=missing" ;;
  *) log "DONE status=${status}" ;;
esac

if [[ "${status}" == "complete" ]]; then
  exit 0
fi
exit 1
