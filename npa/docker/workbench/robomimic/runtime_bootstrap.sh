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
      local candidate group group_seen key state status_file value
      # Avoid an external process-table command in the cleanup path: a stuck
      # command would defeat both signal escalation and snapshot cleanup. The
      # kernel-generated status files are read with Bash builtins. NSpgid's last
      # value is the process-group id in this container's PID namespace.
      kill -0 -- "-$1" 2>/dev/null || return 1
      group_seen=0
      for status_file in /proc/[0-9]*/status; do
        [[ -r "${status_file}" ]] || continue
        group=""
        state=""
        while IFS=$'\t' read -r key value; do
          case "${key}" in
            NSpgid:)
              for candidate in ${value}; do
                group="${candidate}"
              done
              ;;
            State:) state="${value%% *}" ;;
          esac
        done <"${status_file}" || continue
        if [[ "${group}" == "$1" ]]; then
          group_seen=1
          [[ "${state}" == "Z" || "${state}" == "X" ]] || return 0
        fi
      done
      # If kill(0) observed the group but /proc raced with process exit, keep
      # the snapshot until a later poll can prove either a member or absence.
      ((group_seen == 1)) && return 1
      return 0
    }
    monotonic_microseconds() {
      local fraction ignored seconds
      IFS='. ' read -r seconds fraction ignored </proc/uptime || return 1
      [[ "${seconds}" =~ ^[0-9]+$ && "${fraction}" =~ ^[0-9]+$ ]] || return 1
      fraction="${fraction}000000"
      printf '%s\n' "$((10#${seconds} * 1000000 + 10#${fraction:0:6}))"
    }
    stop_child() {
      local signal_name="$1"
      local signal_status="$2"
      local target_pid="${child_pid}"
      local grace_started grace_now
      trap '' HUP INT TERM
      child_pid=""
      kill -s "${signal_name}" -- "-${target_pid}" 2>/dev/null || true
      grace_started="$(monotonic_microseconds)" || grace_started=0
      while process_group_running "${target_pid}"; do
        grace_now="$(monotonic_microseconds)" || grace_now=0
        # /proc/uptime is monotonic. If it becomes unreadable, fail closed by
        # ending the graceful wait and escalating instead of extending cleanup.
        [[ "${grace_started}" == 0 || "${grace_now}" == 0 ]] && break
        ((10#${grace_now} - 10#${grace_started} >= 5000000)) && break
        sleep 0.1
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
      # The process-group leader can exit after starting a background worker.
      # Keep both the group signal target and the verified snapshot alive until
      # every non-zombie member has finished.
      while process_group_running "${child_pid}"; do
        sleep 0.1
      done
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
