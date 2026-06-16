#!/usr/bin/env bash
# Run the full golden-eval fleet in tmux until every container PASSes.
# Each attempt: sync branch → unit tests → serverless fleet (tmux) → harvest failures → retry.
#
# Usage:
#   golden_eval_converge.sh                 # loop until all PASS
#   golden_eval_converge.sh --once          # single fleet attempt
#   golden_eval_converge.sh --unit-only     # pytest gate only (no fleet)
#
# Environment:
#   GOLDEN_EVAL_SOURCE_REF          git branch (default feat/golden-eval)
#   GOLDEN_EVAL_MAX_IN_FLIGHT       tmux fleet parallelism (default 4)
#   GOLDEN_EVAL_FLEET_TIMEOUT_S     wait for summary (default 7200)
#   GOLDEN_EVAL_MAX_ATTEMPTS        retry cap (default 999)
#   GOLDEN_EVAL_RETRY_SLEEP_S       backoff between attempts (default 90)
#   GOLDEN_EVAL_AUTO_COMMIT=1       commit npa/docs changes before pull
#   GOLDEN_EVAL_AUTO_PUSH=1         push local commits before pull
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${ROOT}/npa/.venv/bin/python"
DRIVER="${ROOT}/npa/scripts/run_golden_evals.py"
TMUX_LAUNCHER="${SCRIPT_DIR}/start_golden_evals_tmux.sh"
AUTOFIX="${SCRIPT_DIR}/golden_eval_autofix.sh"

BRANCH="${GOLDEN_EVAL_SOURCE_REF:-feat/golden-eval}"
STATE_DIR="${GOLDEN_EVAL_STATE_DIR:-/tmp/golden-evals/converge}"
LOG="${STATE_DIR}/converge.log"
COMPLETE_FILE="${STATE_DIR}/golden-evals-complete"
MAX_IN_FLIGHT="${GOLDEN_EVAL_MAX_IN_FLIGHT:-2}"
FLEET_TIMEOUT_S="${GOLDEN_EVAL_FLEET_TIMEOUT_S:-7200}"
MAX_ATTEMPTS="${GOLDEN_EVAL_MAX_ATTEMPTS:-999}"
RETRY_SLEEP_S="${GOLDEN_EVAL_RETRY_SLEEP_S:-90}"
FLEET_SESSION_PREFIX="${GOLDEN_EVAL_FLEET_SESSION:-golden-evals-fleet}"

ONCE=0
UNIT_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --once) ONCE=1; shift ;;
    --unit-only) UNIT_ONLY=1; shift ;;
    -h | --help)
      sed -n '2,18p' "$0"
      exit 0
      ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "${STATE_DIR}" "${STATE_DIR}/failures"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${LOG}"
}

require_tools() {
  if ! command -v tmux >/dev/null; then
    echo "tmux required" >&2
    exit 1
  fi
  if [[ ! -x "${PYTHON}" ]]; then
    echo "Missing venv at ${PYTHON}" >&2
    exit 1
  fi
}

sync_repo() {
  bash "${AUTOFIX}" "sync" 2>&1 | tee -a "${LOG}"
}

run_unit_tests() {
  log "unit tests: npa/tests/smoke/test_golden_eval_*.py"
  (
    cd "${ROOT}"
    "${PYTHON}" -m pytest npa/tests/smoke/test_golden_eval_*.py -q
  ) 2>&1 | tee -a "${LOG}"
}

_summary_all_pass() {
  local summary="$1"
  SUMMARY_PATH="${summary}" "${PYTHON}" - <<'PY'
import json, os, sys
path = os.environ["SUMMARY_PATH"]
data = json.loads(open(path, encoding="utf-8").read())
results = data.get("results") or []
if not results:
    sys.exit(1)
for row in results:
    if row.get("status") != "PASS":
        sys.exit(1)
sys.exit(0)
PY
}

_summary_done() {
  local summary="$1"
  SUMMARY_PATH="${summary}" "${PYTHON}" - <<'PY'
import json, os, sys
path = os.environ["SUMMARY_PATH"]
data = json.loads(open(path, encoding="utf-8").read())
total = int(data.get("total") or 0)
completed = int(data.get("completed") or 0)
sys.exit(0 if total > 0 and completed >= total else 1)
PY
}

wait_for_fleet() {
  local summary="$1"
  local deadline=$((SECONDS + FLEET_TIMEOUT_S))
  log "waiting for fleet summary=${summary} timeout=${FLEET_TIMEOUT_S}s"
  while [[ "${SECONDS}" -lt "${deadline}" ]]; do
    if [[ -f "${summary}" ]] && _summary_done "${summary}"; then
      log "fleet complete summary=${summary}"
      return 0
    fi
    sleep 15
  done
  log "fleet wait timed out summary=${summary}"
  return 1
}

harvest_failures() {
  local run_id="$1"
  local log_root="$2"
  local out="${STATE_DIR}/failures/${run_id}"
  mkdir -p "${out}"
  if [[ -f "${log_root}/summary.json" ]]; then
    cp "${log_root}/summary.json" "${out}/summary.json"
  fi
  for f in "${log_root}"/*.log; do
    [[ -f "${f}" ]] || continue
    local base
    base="$(basename "${f}" .log)"
    if [[ -f "${log_root}/${base}.status" ]] && [[ "$(cat "${log_root}/${base}.status")" != "PASS" ]]; then
      cp "${f}" "${out}/${base}.log"
    fi
  done
  log "failure artifacts -> ${out}/"
}

run_fleet_tmux() {
  local attempt="$1"
  local session="${FLEET_SESSION_PREFIX}-a${attempt}"
  local launch_log="${STATE_DIR}/fleet-launch-a${attempt}.log"

  log "=== fleet attempt ${attempt} session=${session} max_in_flight=${MAX_IN_FLIGHT} ==="
  tmux kill-session -t "${session}" 2>/dev/null || true

  set +e
  bash "${TMUX_LAUNCHER}" \
    --serverless \
    --max-in-flight "${MAX_IN_FLIGHT}" \
    --session "${session}" \
    >"${launch_log}" 2>&1
  local ec=$?
  set -e
  cat "${launch_log}" | tee -a "${LOG}"
  if [[ "${ec}" -ne 0 ]]; then
    log "fleet launcher failed ec=${ec}"
    return 1
  fi

  local log_root
  log_root="$(sed -n 's/^LOG_ROOT=//p' "${launch_log}" | tail -1)"
  local run_id
  run_id="$(sed -n 's/^GOLDEN_EVALS_RUN=//p' "${launch_log}" | tail -1)"
  if [[ -z "${log_root}" || ! -d "${log_root}" ]]; then
    log "missing LOG_ROOT from launcher output"
    return 1
  fi

  echo "${log_root}" > "${STATE_DIR}/last-log-root.txt"
  echo "${run_id}" > "${STATE_DIR}/last-run-id.txt"
  ln -sf "${log_root}/summary.json" "${STATE_DIR}/latest-summary.json"

  if ! wait_for_fleet "${log_root}/summary.json"; then
    harvest_failures "${run_id:-attempt-${attempt}}" "${log_root}"
    return 1
  fi

  if _summary_all_pass "${log_root}/summary.json"; then
    log "SUCCESS fleet run_id=${run_id} summary=${log_root}/summary.json"
    cp "${log_root}/summary.json" "${STATE_DIR}/last-success-summary.json"
    echo "${run_id}" > "${STATE_DIR}/last-success-run-id.txt"
    return 0
  fi

  log "FAIL fleet run_id=${run_id} (one or more containers did not PASS)"
  harvest_failures "${run_id:-attempt-${attempt}}" "${log_root}"
  return 1
}

run_attempt() {
  local attempt="$1"
  sync_repo
  if ! run_unit_tests; then
    log "unit tests failed attempt=${attempt}"
    return 1
  fi
  if [[ "${UNIT_ONLY}" == "1" ]]; then
    log "unit-only success"
    return 0
  fi
  run_fleet_tmux "${attempt}"
}

log "golden-eval converge start branch=${BRANCH} max_attempts=${MAX_ATTEMPTS}"
require_tools
trap 'log "converge interrupted"' INT TERM

attempt=1
while [[ "${attempt}" -le "${MAX_ATTEMPTS}" ]]; do
  if run_attempt "${attempt}"; then
    touch "${COMPLETE_FILE}"
    log "golden-eval converge complete"
    exit 0
  fi
  if [[ "${ONCE}" == "1" ]]; then
    log "giving up (--once)"
    exit 1
  fi
  log "retry in ${RETRY_SLEEP_S}s (attempt $((attempt + 1)))"
  sleep "${RETRY_SLEEP_S}"
  attempt=$((attempt + 1))
done

log "gave up after ${MAX_ATTEMPTS} attempts"
exit 1
