#!/usr/bin/env bash
# Launch per-container golden evals in tmux (one window per container).
#
# Each window runs the container's golden eval and writes logs under
# /tmp/golden-evals/<run_id>/. A dashboard window refreshes summary.json.
#
# Usage:
#   ./npa/scripts/start_golden_evals_tmux.sh --serverless
#   ./npa/scripts/start_golden_evals_tmux.sh --serverless --max-in-flight 4
#   ./npa/scripts/start_golden_evals_tmux.sh --execute --tools-only
#   ./npa/scripts/start_golden_evals_tmux.sh --dry-run
#
# Attach:  tmux attach -t golden-evals
# Summary: cat /tmp/golden-evals/<run_id>/summary.json
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DRIVER="${ROOT}/npa/scripts/run_golden_evals.py"

if [[ -n "${GOLDEN_EVAL_PYTHON:-}" && -x "${GOLDEN_EVAL_PYTHON}" ]]; then
  PYTHON="${GOLDEN_EVAL_PYTHON}"
elif [[ -x "${ROOT}/npa/.venv/bin/python" ]]; then
  PYTHON="${ROOT}/npa/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="$(command -v python3)"
else
  echo "Missing python; set GOLDEN_EVAL_PYTHON or create ${ROOT}/npa/.venv" >&2
  exit 1
fi

_acquire_slot() {
  local state_dir="$1"
  local max_in_flight="$2"
  local pid=$$
  while true; do
    local running
    running="$(find "${state_dir}" -maxdepth 1 -name 'running.*' 2>/dev/null | wc -l | tr -d ' ')"
    if [[ "${running}" -lt "${max_in_flight}" ]]; then
      touch "${state_dir}/running.${pid}"
      return 0
    fi
    sleep 5
  done
}

_release_slot() {
  local state_dir="$1"
  rm -f "${state_dir}/running.${pid:-$$}"
}

_run_one_container() {
  local name="$1"
  local mode="$2"
  local log_root="$3"
  local max_in_flight="$4"
  local state_dir="$5"
  local timeout="$6"
  local gpu="$7"

  local log="${log_root}/${name}.log"
  local exit_file="${log_root}/${name}.exit"
  local status_file="${log_root}/${name}.status"
  local run_args=(run "${name}")

  if [[ -n "${gpu}" ]]; then
    run_args+=(--gpu "${gpu}")
  fi

  case "${mode}" in
    serverless)
      run_args+=(--serverless --timeout "${timeout}")
      ;;
    execute)
      run_args+=(--execute)
      ;;
    dry-run)
      ;;
    *)
      echo "Unknown mode: ${mode}" >&2
      exit 2
      ;;
  esac

  {
    echo "container=${name}"
    echo "mode=${mode}"
    echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "command=${PYTHON} ${DRIVER} ${run_args[*]}"
    echo "---"
  } >"${log}"

  set +e
  if [[ "${mode}" == "dry-run" ]]; then
    "${PYTHON}" "${DRIVER}" run "${name}" >>"${log}" 2>&1
    ec=$?
  else
    _acquire_slot "${state_dir}" "${max_in_flight}"
    trap '_release_slot "${state_dir}"' EXIT
    "${PYTHON}" "${DRIVER}" "${run_args[@]}" >>"${log}" 2>&1
    ec=$?
    _release_slot "${state_dir}"
    trap - EXIT
  fi
  set -e

  echo "${ec}" >"${exit_file}"
  if [[ "${ec}" -eq 0 ]]; then
    echo "PASS" >"${status_file}"
  else
    echo "FAIL" >"${status_file}"
  fi
  echo "finished ${name} exit=${ec}"
  return "${ec}"
}

_collect_summary_loop() {
  local run_id="$1"
  local mode="$2"
  local log_root="$3"
  shift 3
  local containers=("$@")
  local summary="${log_root}/summary.json"

  while true; do
    local done=0
    local total=${#containers[@]}
    for name in "${containers[@]}"; do
      if [[ -f "${log_root}/${name}.status" ]]; then
        done=$((done + 1))
      fi
    done
    {
      echo "{"
      echo "  \"run_id\": \"${run_id}\","
      echo "  \"mode\": \"${mode}\","
      echo "  \"total\": ${total},"
      echo "  \"completed\": ${done},"
      echo "  \"results\": ["
      local first=1
      for name in "${containers[@]}"; do
        local status="pending"
        local exit_code="null"
        if [[ -f "${log_root}/${name}.status" ]]; then
          status="$(cat "${log_root}/${name}.status")"
        fi
        if [[ -f "${log_root}/${name}.exit" ]]; then
          exit_code="$(cat "${log_root}/${name}.exit")"
        fi
        [[ "${first}" = "1" ]] || echo ","
        first=0
        printf '    {"name": "%s", "status": "%s", "exit_code": %s, "log": "%s/%s.log"}' \
          "${name}" "${status}" "${exit_code}" "${log_root}" "${name}"
      done
      echo
      echo "  ]"
      echo "}"
    } >"${summary}.tmp"
    mv "${summary}.tmp" "${summary}"
    if [[ "${done}" -ge "${total}" && "${mode}" != "dry-run" ]]; then
      echo "summary_ready=${summary}"
      break
    fi
    sleep 10
  done
}

if [[ "${1:-}" == "--run-one" ]]; then
  shift
  _run_one_container "$@"
  exit $?
fi

if [[ "${1:-}" == "--collect-only" ]]; then
  shift
  _collect_summary_loop "$@"
  exit 0
fi

SESSION="${GOLDEN_EVALS_TMUX_SESSION:-golden-evals}"
RUN_ID="run-$(date -u +%Y%m%dT%H%M%SZ)"
LOG_ROOT="/tmp/golden-evals/${RUN_ID}"
STATE_DIR="${LOG_ROOT}/state"

MODE="dry-run"
MAX_IN_FLIGHT=4
INCLUDE_BLOCKED=0
TOOLS_ONLY=0
GPU=""
TIMEOUT="40m"
EXTRA_CONTAINERS=()

usage() {
  sed -n '2,14p' "$0"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --serverless) MODE="serverless"; shift ;;
    --execute) MODE="execute"; shift ;;
    --dry-run) MODE="dry-run"; shift ;;
    --max-in-flight) MAX_IN_FLIGHT="${2:?}"; shift 2 ;;
    --include-blocked) INCLUDE_BLOCKED=1; shift ;;
    --tools-only) TOOLS_ONLY=1; shift ;;
    --gpu) GPU="${2:?}"; shift 2 ;;
    --timeout) TIMEOUT="${2:?}"; shift 2 ;;
    --session) SESSION="${2:?}"; shift 2 ;;
    -h | --help) usage; exit 0 ;;
    --) shift; EXTRA_CONTAINERS+=("$@"); break ;;
    -*) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    *) EXTRA_CONTAINERS+=("$1"); shift ;;
  esac
done

if ! command -v tmux >/dev/null; then
  echo "tmux required" >&2
  exit 1
fi

mkdir -p "${STATE_DIR}"
"${PYTHON}" "${DRIVER}" validate >/dev/null

mapfile -t ALL_CONTAINERS < <(
  "${PYTHON}" - <<PY
from npa.smoke.batch import iter_containers

for name in iter_containers(
    include_blocked=${INCLUDE_BLOCKED},
    include_foundation=${TOOLS_ONLY} is False,
    tools_only=${TOOLS_ONLY},
):
    print(name)
PY
)

if [[ ${#EXTRA_CONTAINERS[@]} -gt 0 ]]; then
  WANTED=()
  for name in "${EXTRA_CONTAINERS[@]}"; do
    found=0
    for c in "${ALL_CONTAINERS[@]}"; do
      if [[ "${c}" == "${name}" ]]; then
        WANTED+=("${name}")
        found=1
        break
      fi
    done
    if [[ "${found}" = "0" ]]; then
      echo "Unknown or filtered container: ${name}" >&2
      exit 2
    fi
  done
  ALL_CONTAINERS=("${WANTED[@]}")
fi

if [[ ${#ALL_CONTAINERS[@]} -eq 0 ]]; then
  echo "No containers selected" >&2
  exit 2
fi

tmux kill-session -t "${SESSION}" 2>/dev/null || true

CONTAINER_ARGS=()
for name in "${ALL_CONTAINERS[@]}"; do
  CONTAINER_ARGS+=("${name}")
done

tmux new-session -d -s "${SESSION}" -n dashboard \
  "bash -lc 'cd \"${ROOT}\" && echo GOLDEN_EVALS_RUN=${RUN_ID} && echo MODE=${MODE} && echo LOG_ROOT=${LOG_ROOT} && echo containers=${#ALL_CONTAINERS[@]} && echo max_in_flight=${MAX_IN_FLIGHT} && bash \"${SCRIPT_DIR}/start_golden_evals_tmux.sh\" --collect-only \"${RUN_ID}\" \"${MODE}\" \"${LOG_ROOT}\" ${CONTAINER_ARGS[*]}; exec bash'"

for name in "${ALL_CONTAINERS[@]}"; do
  safe_name="${name//\//-}"
  tmux new-window -t "${SESSION}" -n "${safe_name}" \
    "bash -lc 'cd \"${ROOT}\" && bash \"${SCRIPT_DIR}/start_golden_evals_tmux.sh\" --run-one \"${name}\" \"${MODE}\" \"${LOG_ROOT}\" \"${MAX_IN_FLIGHT}\" \"${STATE_DIR}\" \"${TIMEOUT}\" \"${GPU}\"; exec bash'"
done

echo "TMUX_SESSION=${SESSION}"
echo "GOLDEN_EVALS_RUN=${RUN_ID}"
echo "LOG_ROOT=${LOG_ROOT}"
echo "attach: tmux attach -t ${SESSION}"
echo "containers: ${ALL_CONTAINERS[*]}"
