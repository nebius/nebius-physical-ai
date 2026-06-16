#!/usr/bin/env bash
# Block placeholder asset URIs on stock runs; strip stale custom-asset exports before submit.

customer_task_id_is_valid() {
  local tid="${CUSTOMER_TASK_ID:-}"
  [ -n "${tid}" ] || return 1
  case "${tid}" in
    YOUR-TASK-ID|your-task-id|YOUR_TASK_ID|*YOUR*TASK*ID*)
      return 1
      ;;
  esac
  return 0
}

customer_asset_uri_has_placeholder() {
  local val="$1"
  [[ "${val}" == *YOUR-TASK-ID* || "${val}" == *YOUR-BUCKET* ]]
}

customer_asset_guard_placeholders() {
  local var val
  for var in \
    ROBOT_SPEC_URI ASSETS_URI SCENE_SPEC_URI CAMERAS_URI \
    NPA_SIM2REAL_ROBOT_SPEC_URI NPA_SIM2REAL_CAMERAS_URI; do
    val="${!var:-}"
    if [ -n "${val}" ] && customer_asset_uri_has_placeholder "${val}"; then
      if ! customer_task_id_is_valid; then
        cat >&2 <<EOF
ERROR: ${var} still contains a template placeholder:
  ${val}
Stock Franka demo: unset CUSTOMER_ASSET_PROFILE and custom asset exports, then re-run ./run.sh trigger.
Custom assets: export CUSTOMER_TASK_ID=<your-slug>, upload to S3, run apply-customer-asset-profile.sh --export.
EOF
        return 1
      fi
    fi
  done
  return 0
}

customer_asset_prepare_for_submit() {
  if customer_task_id_is_valid; then
    return 0
  fi
  if [ -n "${CUSTOMER_ASSET_PROFILE:-}" ]; then
    cat >&2 <<EOF
ERROR: CUSTOMER_ASSET_PROFILE=${CUSTOMER_ASSET_PROFILE} requires a real CUSTOMER_TASK_ID.
Stock Franka demo: unset CUSTOMER_ASSET_PROFILE, then re-run ./run.sh trigger.
Custom assets: export CUSTOMER_TASK_ID=<your-slug>, upload to S3, run apply-customer-asset-profile.sh --export.
EOF
    return 1
  fi
  unset CUSTOMER_ASSET_PROFILE_APPLIED CUSTOMER_ASSET_PROFILE_PATH
  unset CUSTOMER_TASK_ID
  unset ASSETS_URI SCENE_SPEC_URI CAMERAS_URI ROBOT_SPEC_URI ROBOT_SOURCE ROBOT_PRESET
  unset NPA_SIM2REAL_ROBOT_SPEC_URI NPA_SIM2REAL_CAMERAS_URI
  unset NPA_SIM2REAL_ROBOT_PRESET NPA_SIM2REAL_ROBOT_SOURCE
  unset ROBOT_MODE OBJECT_MODE SCENE_MODE CAMERA_MODE
  return 0
}

operator_skypilot_cli_ready() {
  if [ -n "${NPA_SKYPILOT_BIN:-}" ] && [ -x "${NPA_SKYPILOT_BIN}" ]; then
    return 0
  fi
  if command -v sky >/dev/null 2>&1; then
    export NPA_SKYPILOT_BIN="$(command -v sky)"
    return 0
  fi
  return 1
}

operator_skypilot_available() {
  operator_skypilot_cli_ready && return 0
  local py="${1:-}"
  if [ -n "${py}" ] && [ -x "${py}" ]; then
    local resolved=""
    resolved="$("${py}" -c "from npa.orchestration.skypilot._bin import resolve_sky_bin; print(resolve_sky_bin())" 2>/dev/null || true)"
    if [ -n "${resolved}" ] && [ -x "${resolved}" ]; then
      export NPA_SKYPILOT_BIN="${resolved}"
      return 0
    fi
  fi
  return 1
}

operator_use_workbench_submit() {
  [[ "${NPA_USE_KUBECTL_SUBMIT:-0}" != "1" ]]
}

operator_resolve_submit_script() {
  local script_dir="${1:?script_dir required}"
  if operator_use_workbench_submit; then
    printf '%s/submit-workbench-job.sh\n' "${script_dir}"
  else
    printf '%s/submit-k8s-staged-job.sh\n' "${script_dir}"
  fi
}
