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
    snapshot_parent="$(mktemp -d)"
    snapshot_root="${snapshot_parent}/runtime"
    payload_pid=""
    stop_payload() {
      local signal_name="$1"
      local signal_status="$2"
      trap - HUP INT TERM
      if [[ -n "${payload_pid}" ]]; then
        kill -s "${signal_name}" "${payload_pid}" 2>/dev/null || true
        wait "${payload_pid}" 2>/dev/null || true
      fi
      exit "${signal_status}"
    }
    trap 'rm -rf -- "${snapshot_parent}"' EXIT
    trap 'stop_payload HUP 129' HUP
    trap 'stop_payload INT 130' INT
    trap 'stop_payload TERM 143' TERM
    /usr/local/bin/python3 "${verifier}" snapshot \
      --runtime-root "${runtime_root}" \
      --expected-inventory-sha256 "${expected_inventory_sha256}" \
      --destination "${snapshot_root}" >/dev/null
    export NPA_ROBOMIMIC_ACTIVE_RUNTIME_ROOT="${snapshot_root}"
    "${snapshot_root}/payload/bin/python" -c \
      'from robomimic.config import config_factory; from robomimic.algo import algo_factory; from robomimic.utils.file_utils import policy_from_checkpoint; from diffusers.schedulers.scheduling_ddim import DDIMScheduler; from diffusers.schedulers.scheduling_ddpm import DDPMScheduler; from diffusers.training_utils import EMAModel; assert all((config_factory, algo_factory, policy_from_checkpoint, DDIMScheduler, DDPMScheduler, EMAModel))'
    # Supervise the exec in a child so an exec failure remains cleanup-bound.
    # Signals stop and reap a running payload before the EXIT trap removes its
    # snapshot; a normal payload exit is reaped before that cleanup as well.
    (
      exec "${snapshot_root}/payload/bin/python" "$@"
    ) &
    payload_pid="$!"
    set +e
    wait "${payload_pid}"
    payload_status="$?"
    set -e
    payload_pid=""
    exit "${payload_status}"
    ;;
  assert-refusal)
    empty_root="$(mktemp -d)"
    trap 'rm -rf -- "${empty_root}"' EXIT
    /usr/local/bin/python3 "${verifier}" assert-missing-runtime \
      --runtime-root "${empty_root}" --runtime-lock "${runtime_lock}" \
      >"${empty_root}.proof"
    if find "${empty_root}" -mindepth 1 -print -quit | grep -q .; then
      echo "runtime verifier mutated the missing runtime root" >&2
      exit 1
    fi
    grep -Fq '"refusal_reason": "missing-ready-marker"' "${empty_root}.proof"
    grep -Fq '"runtime_root_unchanged": true' "${empty_root}.proof"
    rm -f -- "${empty_root}.proof"
    echo "NPA_ROBOMIMIC_RUNTIME_REFUSAL_OK"
    ;;
  *)
    echo "usage: robomimic-runtime {verify|exec|assert-refusal}" >&2
    exit 64
    ;;
esac
