#!/usr/bin/env bash
# Resolve customer sim asset choices (robot / scene / object / camera) from a named profile.
#
# Usage:
#   export CUSTOMER_ASSET_PROFILE=stock-smoke          # built-in *.profile.example
#   export CUSTOMER_ASSET_PROFILE=industrial           # full four-axis production template
#   export CUSTOMER_ASSET_PROFILE=/path/to/my.profile  # operator copy (chmod 600)
#   customer_asset_profile_apply "$script_dir" "$bucket" "$task_id"
#
# Profiles set modes and URIs; Stage 2 materializes consumed_* specs — customers do not
# author those JSON envelopes by hand.

customer_asset_profiles_dir() {
  local script_dir="${1:?script_dir required}"
  printf '%s/customer-asset-profiles\n' "${script_dir}"
}

customer_asset_profile_resolve_path() {
  local script_dir="$1"
  local profile="$2"
  local dir
  dir="$(customer_asset_profiles_dir "${script_dir}")"
  if [ -z "${profile}" ]; then
    return 1
  fi
  if [ -f "${profile}" ]; then
    printf '%s\n' "${profile}"
    return 0
  fi
  for suffix in ".profile" ".profile.example"; do
    if [ -f "${dir}/${profile}${suffix}" ]; then
      printf '%s\n' "${dir}/${profile}${suffix}"
      return 0
    fi
  done
  echo "ERROR: unknown CUSTOMER_ASSET_PROFILE=${profile} (looked in ${dir}/)" >&2
  return 1
}

customer_asset_profile_expand() {
  local value="$1"
  local bucket="$2"
  local task_id="$3"
  value="${value//YOUR-BUCKET/${bucket}}"
  value="${value//YOUR-TASK-ID/${task_id}}"
  printf '%s' "${value}"
}

customer_asset_profile_apply() {
  local script_dir="$1"
  local bucket="${2:-}"
  local task_id="${3:-}"

  if [ -z "${CUSTOMER_ASSET_PROFILE:-}" ]; then
    return 0
  fi

  local path
  path="$(customer_asset_profile_resolve_path "${script_dir}" "${CUSTOMER_ASSET_PROFILE}")" || return 1

  task_id="${CUSTOMER_TASK_ID:-${task_id:-${RUN_ID:-pilot-task}}}"

  # shellcheck source=/dev/null
  set -a
  source "${path}"
  set +a

  # Operator env overrides (without editing the profile copy).
  if [ -n "${CUSTOMER_ROBOT_PRESET:-}" ]; then
    export ROBOT_PRESET="${CUSTOMER_ROBOT_PRESET}"
  fi

  local robot_mode="${ROBOT_MODE:-preset}"
  case "${robot_mode}" in
    stock_franka)
      export ROBOT_PRESET="franka"
      export NPA_SIM2REAL_ROBOT_PRESET="franka"
      unset ROBOT_SPEC_URI NPA_SIM2REAL_ROBOT_SPEC_URI ROBOT_SOURCE NPA_SIM2REAL_ROBOT_SOURCE
      ;;
    preset)
      export ROBOT_PRESET="${ROBOT_PRESET:-ur5e}"
      export NPA_SIM2REAL_ROBOT_PRESET="${ROBOT_PRESET}"
      ;;
    byo)
      : "${ROBOT_SPEC_URI:?ROBOT_MODE=byo requires ROBOT_SPEC_URI in profile}"
      export ROBOT_PRESET="${ROBOT_PRESET:-}"
      export NPA_SIM2REAL_ROBOT_PRESET="${ROBOT_PRESET}"
      ;;
    *)
      echo "ERROR: ROBOT_MODE must be stock_franka|preset|byo, got ${robot_mode}" >&2
      return 2
      ;;
  esac

  if [ -n "${ROBOT_SPEC_URI:-}" ]; then
    export NPA_SIM2REAL_ROBOT_SPEC_URI="${ROBOT_SPEC_URI}"
  fi

  local object_mode="${OBJECT_MODE:-none}"
  local scene_mode="${SCENE_MODE:-stock}"
  local camera_mode="${CAMERA_MODE:-stock}"

  if [ -n "${SCENE_SPEC_URI:-}" ]; then
    object_mode="${object_mode}"
    if [ "${object_mode}" = "none" ]; then
      object_mode="scene_spec"
    fi
    scene_mode="custom"
  elif [ -n "${ASSETS_URI:-}" ]; then
    object_mode="${object_mode:-mesh}"
    if [ "${scene_mode}" = "stock" ] && [ "${object_mode}" = "mesh" ]; then
      scene_mode="custom"
    fi
  fi

  case "${object_mode}" in
    none) ;;
    mesh)
      : "${ASSETS_URI:?OBJECT_MODE=mesh requires ASSETS_URI in profile}"
      ;;
    scene_spec)
      : "${SCENE_SPEC_URI:?OBJECT_MODE=scene_spec requires SCENE_SPEC_URI in profile}"
      ;;
    *)
      echo "ERROR: OBJECT_MODE must be none|mesh|scene_spec, got ${object_mode}" >&2
      return 2
      ;;
  esac

  case "${scene_mode}" in
    stock)
      if [ "${object_mode}" = "none" ] && [ -z "${ASSETS_URI:-}" ] && [ -z "${SCENE_SPEC_URI:-}" ]; then
        unset ASSETS_URI SCENE_SPEC_URI CAMERAS_URI
      fi
      ;;
    custom)
      if [ "${object_mode}" = "none" ] && [ -z "${ASSETS_URI:-}" ] && [ -z "${SCENE_SPEC_URI:-}" ]; then
        echo "ERROR: SCENE_MODE=custom requires ASSETS_URI or SCENE_SPEC_URI" >&2
        return 2
      fi
      ;;
    *)
      echo "ERROR: SCENE_MODE must be stock|custom, got ${scene_mode}" >&2
      return 2
      ;;
  esac

  case "${camera_mode}" in
    stock)
      if [ -n "${SCENE_SPEC_URI:-}" ] || [ -n "${CAMERAS_URI:-}" ]; then
        camera_mode="custom"
      fi
      ;;
    custom)
      if [ -z "${SCENE_SPEC_URI:-}" ] && [ -z "${CAMERAS_URI:-}" ]; then
        echo "ERROR: CAMERA_MODE=custom requires SCENE_SPEC_URI (cameras block) or CAMERAS_URI" >&2
        return 2
      fi
      ;;
    *)
      echo "ERROR: CAMERA_MODE must be stock|custom, got ${camera_mode}" >&2
      return 2
      ;;
  esac

  export OBJECT_MODE="${object_mode}"
  export SCENE_MODE="${scene_mode}"
  export CAMERA_MODE="${camera_mode}"

  for var in ROBOT_SPEC_URI ASSETS_URI SCENE_SPEC_URI CAMERAS_URI ROBOT_SOURCE; do
    if [ -n "${!var:-}" ]; then
      # shellcheck disable=SC2154
      printf -v "${var}" '%s' "$(customer_asset_profile_expand "${!var}" "${bucket}" "${task_id}")"
      export "${var}"
    fi
  done

  if [ -n "${ROBOT_SPEC_URI:-}" ]; then
    export NPA_SIM2REAL_ROBOT_SPEC_URI="${ROBOT_SPEC_URI}"
  fi
  if [ -n "${ROBOT_PRESET:-}" ]; then
    export NPA_SIM2REAL_ROBOT_PRESET="${ROBOT_PRESET}"
  fi
  if [ -n "${ROBOT_SOURCE:-}" ]; then
    export NPA_SIM2REAL_ROBOT_SOURCE="${ROBOT_SOURCE}"
  fi
  if [ -n "${CAMERAS_URI:-}" ]; then
    export NPA_SIM2REAL_CAMERAS_URI="${CAMERAS_URI}"
  fi

  export CUSTOMER_ASSET_PROFILE_APPLIED="${CUSTOMER_ASSET_PROFILE}"
  export CUSTOMER_ASSET_PROFILE_PATH="${path}"
}

customer_asset_profile_print() {
  cat <<EOF
CUSTOMER_ASSET_PROFILE=${CUSTOMER_ASSET_PROFILE_APPLIED:-}
CUSTOMER_ASSET_PROFILE_PATH=${CUSTOMER_ASSET_PROFILE_PATH:-}
ROBOT_MODE=${ROBOT_MODE:-} ROBOT_PRESET=${ROBOT_PRESET:-} ROBOT_SPEC_URI=${ROBOT_SPEC_URI:-}
SCENE_MODE=${SCENE_MODE:-} OBJECT_MODE=${OBJECT_MODE:-} CAMERA_MODE=${CAMERA_MODE:-}
ASSETS_URI=${ASSETS_URI:-} SCENE_SPEC_URI=${SCENE_SPEC_URI:-} CAMERAS_URI=${CAMERAS_URI:-}
EOF
}
