#!/usr/bin/env bash
set -euo pipefail

readonly runtime_root="${NPA_ROBOMIMIC_RUNTIME_ROOT:-/opt/npa-runtime/robomimic}"
readonly verifier="/opt/npa/robomimic/verify_image.py"
readonly runtime_lock="/opt/npa/robomimic/runtime-requirements.lock"
readonly expected_inventory_sha256="${NPA_ROBOMIMIC_RUNTIME_INVENTORY_SHA256:-}"

verify_runtime() {
  if [[ ! "${expected_inventory_sha256}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "NPA_ROBOMIMIC_RUNTIME_REFUSED: manager-approved runtime inventory digest is required" >&2
    return 78
  fi
  /usr/local/bin/python3 "${verifier}" runtime \
    --runtime-root "${runtime_root}" \
    --expected-inventory-sha256 "${expected_inventory_sha256}"
}

case "${1:-}" in
  verify)
    verify_runtime
    ;;
  exec)
    shift
    snapshot_parent=""
    child_pid=""
    pending_signal=""
    pending_status=""
    cleanup_snapshot() {
      [[ -n "${snapshot_parent}" ]] || return
      find "${snapshot_parent}" -type d -exec chmod u+w -- {} + 2>/dev/null || true
      rm -rf -- "${snapshot_parent}"
    }
    trap cleanup_snapshot EXIT
    snapshot_parent="$(mktemp -d)"
    snapshot_root="${snapshot_parent}/runtime"
    process_group_running() {
      local group state states
      if ! states="$(ps -e -o pgid=,stat= 2>/dev/null)"; then
        return 0
      fi
      while read -r group state; do
        if [[ "${group}" == "$1" ]]; then
          case "${state}" in
            Z*|X*) ;;
            *) return 0 ;;
          esac
        fi
      done <<<"${states}"
      return 1
    }
    stop_child() {
      local signal_name="$1"
      local signal_status="$2"
      local target_pid="${child_pid}"
      local attempt
      trap '' HUP INT TERM
      child_pid=""
      kill -s "${signal_name}" -- "-${target_pid}" 2>/dev/null || true
      for ((attempt = 0; attempt < 50; attempt += 1)); do
        process_group_running "${target_pid}" || break
        sleep 0.05
      done
      process_group_running "${target_pid}" \
        && kill -s KILL -- "-${target_pid}" 2>/dev/null || true
      wait "${target_pid}" 2>/dev/null || true
      exit "${signal_status}"
    }
    request_stop() {
      if [[ -z "${child_pid}" ]]; then
        pending_signal="$1"
        pending_status="$2"
        return
      fi
      stop_child "$1" "$2"
    }
    run_child() {
      local child_status
      if [[ -n "${pending_signal}" ]]; then
        exit "${pending_status}"
      fi
      set -m
      (
        trap - HUP INT QUIT TERM
        exec "$@"
      ) &
      child_pid="$!"
      set +m
      if [[ -n "${pending_signal}" ]]; then
        stop_child "${pending_signal}" "${pending_status}"
      fi
      if wait "${child_pid}"; then
        child_status=0
      else
        child_status="$?"
      fi
      child_pid=""
      return "${child_status}"
    }
    trap 'request_stop HUP 129' HUP
    trap 'request_stop INT 130' INT
    trap 'request_stop TERM 143' TERM
    if run_child /usr/local/bin/python3 "${verifier}" snapshot \
      --runtime-root "${runtime_root}" \
      --expected-inventory-sha256 "${expected_inventory_sha256}" \
      --destination "${snapshot_root}" >/dev/null; then
      :
    else
      child_status="$?"
      exit "${child_status}"
    fi
    export NPA_ROBOMIMIC_ACTIVE_RUNTIME_ROOT="${snapshot_root}"
    if run_child "${snapshot_root}/payload/bin/python" -c \
      'from robomimic.config import config_factory; from robomimic.algo import algo_factory; from robomimic.utils.file_utils import policy_from_checkpoint; from diffusers.schedulers.scheduling_ddim import DDIMScheduler; from diffusers.schedulers.scheduling_ddpm import DDPMScheduler; from diffusers.training_utils import EMAModel; assert all((config_factory, algo_factory, policy_from_checkpoint, DDIMScheduler, DDPMScheduler, EMAModel))'; then
      :
    else
      child_status="$?"
      exit "${child_status}"
    fi
    # Every child runs in its own process group. A successful payload remains
    # attached to this supervisor until it exits; only then is the private
    # snapshot removed. Exec failure follows the same cleanup path.
    if run_child "${snapshot_root}/payload/bin/python" "$@"; then
      exit 0
    else
      child_status="$?"
      exit "${child_status}"
    fi
    ;;
  assert-refusal)
    empty_root="$(mktemp -d)"
    trap 'rm -rf -- "${empty_root}"' EXIT
    proof="$(/usr/local/bin/python3 "${verifier}" assert-missing-runtime \
      --runtime-root "${empty_root}" --runtime-lock "${runtime_lock}")"
    if find "${empty_root}" -mindepth 1 -print -quit | grep -q .; then
      echo "runtime verifier mutated the missing runtime root" >&2
      exit 1
    fi
    grep -Fq '"refusal_reason": "missing-ready-marker"' <<<"${proof}"
    grep -Fq '"runtime_root_unchanged": true' <<<"${proof}"
    echo "NPA_ROBOMIMIC_RUNTIME_REFUSAL_OK"
    ;;
  *)
    echo "usage: robomimic-runtime {verify|exec|assert-refusal}" >&2
    exit 64
    ;;
esac
