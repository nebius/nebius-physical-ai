#!/bin/bash

# LeRobot on Nebius — Environment Bootstrap
#
# Automates all Nebius resource creation and credential wiring so that
# `terraform apply` can run with zero manual configuration.
#
# Usage:
#   export NEBIUS_TENANT_ID='tenant-...'
#   export NEBIUS_PROJECT_ID='project-...'
#   export NEBIUS_REGION='eu-north1'
#   source environment.sh
#
# Prerequisites:
#   - nebius CLI installed and configured (nebius config init)
#   - jq installed

_lerobot_env_fail() {
  printf '  Error: %s\n' "$1" >&2
  return 1
}

_lerobot_require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    _lerobot_env_fail "Required command not found: $1"
    return 1
  fi
}

_lerobot_require_real_id() {
  local var_name="$1"
  local value="$2"
  local example="$3"

  if [[ "${value}" == *"..."* ]]; then
    _lerobot_env_fail "${var_name} still contains a placeholder value: ${value}"
    printf "   Replace it with the real ID, e.g. export %s='%s'\n" "${var_name}" "${example}" >&2
    return 1
  fi
}

_lerobot_repo_root() {
  local candidate script_dir
  if command -v git >/dev/null 2>&1; then
    candidate="$(git rev-parse --show-toplevel 2>/dev/null || true)"
    if [ -n "${candidate}" ] && [ -d "${candidate}/terraform" ]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  fi
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
  if [ -d "${script_dir}/terraform" ]; then
    printf '%s\n' "${script_dir}"
    return 0
  fi
  pwd -P
}

_lerobot_bucket_suffix() {
  local input="$1"
  if command -v md5sum >/dev/null 2>&1; then
    printf '%s' "${input}" | md5sum | awk '{print $1}' | cut -c1-8
    return 0
  fi
  if command -v md5 >/dev/null 2>&1; then
    printf '%s' "${input}" | md5 -q | cut -c1-8
    return 0
  fi
  _lerobot_env_fail "Neither md5sum nor md5 is available to derive the bucket name"
  return 1
}

_lerobot_access_key_expiration_date() {
  local date_format='+%Y-%m-%dT%H:%M:%SZ'
  if [ "$(uname)" = "Darwin" ]; then
    date -v +365d "${date_format}"
  else
    date -d '+365 day' "${date_format}"
  fi
}

_lerobot_delete_named_access_key() {
  local existing_key_by_name
  existing_key_by_name="$(
    nebius iam v2 access-key list \
      --parent-id "${NEBIUS_PROJECT_ID}" \
      --format json 2>/dev/null \
      | jq -r '.items[]? | select(.metadata.name == "lerobot-access-key") | .metadata.id' \
      | head -1
  )" || true

  if [ -n "${existing_key_by_name}" ]; then
    printf "  Deleting existing key with same name: %s\n" "${existing_key_by_name}"
    nebius iam v2 access-key delete --id "${existing_key_by_name}" >/dev/null 2>&1 || true
  fi
}

_lerobot_create_access_key() {
  local expiration_date create_json
  expiration_date="$(_lerobot_access_key_expiration_date)" || return 1

  _lerobot_delete_named_access_key

  if ! create_json="$(
    nebius iam v2 access-key create \
      --parent-id "${NEBIUS_PROJECT_ID}" \
      --name "lerobot-access-key" \
      --account-service-account-id "${NEBIUS_SA_ID}" \
      --description 'Access key for LeRobot S3 and API access' \
      --expires-at "${expiration_date}" \
      --format json
  )"; then
    _lerobot_env_fail "Failed to create a service-account access key"
    return 1
  fi

  NEBIUS_SA_ACCESS_KEY_ID="$(printf '%s' "${create_json}" | jq -r '.metadata.id // empty')"
  if [ -z "${NEBIUS_SA_ACCESS_KEY_ID}" ]; then
    _lerobot_env_fail "Access key creation did not return an ID"
    return 1
  fi
  printf "  Created access key: %s\n" "${NEBIUS_SA_ACCESS_KEY_ID}"

  if ! AWS_ACCESS_KEY_ID="$(
    nebius iam v2 access-key get \
      --id "${NEBIUS_SA_ACCESS_KEY_ID}" \
      --format json \
      | jq -r '.status.aws_access_key_id // empty'
  )"; then
    _lerobot_env_fail "Failed to fetch the AWS access key ID"
    return 1
  fi

  if ! AWS_SECRET_ACCESS_KEY="$(
    nebius iam v2 access-key get-secret \
      --id "${NEBIUS_SA_ACCESS_KEY_ID}" \
      --format json \
      | jq -r '.secret // empty'
  )"; then
    _lerobot_env_fail "Failed to fetch the secret for the access key"
    return 1
  fi

  export AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY
}

# ── Main ────────────────────────────────────────────────────────────────

_lerobot_environment_setup() {
  local repo_root terraform_dir env_file backend_file bucket_suffix
  local exists

  # Accept common shorthand.
  if [ -z "${NEBIUS_TENANT_ID:-}" ] && [ -n "${TENANT_ID:-}" ]; then
    export NEBIUS_TENANT_ID="${TENANT_ID}"
  fi

  # ── Validate inputs ──────────────────────────────────────────────────

  if [ -z "${NEBIUS_TENANT_ID:-}" ]; then
    _lerobot_env_fail "NEBIUS_TENANT_ID is not set"
    printf "   Set it with: export NEBIUS_TENANT_ID='tenant-...'\n" >&2
    return 1
  fi
  _lerobot_require_real_id "NEBIUS_TENANT_ID" "${NEBIUS_TENANT_ID}" "tenant-<real-id>" || return 1
  if [ -z "${NEBIUS_PROJECT_ID:-}" ]; then
    _lerobot_env_fail "NEBIUS_PROJECT_ID is not set"
    printf "   Set it with: export NEBIUS_PROJECT_ID='project-...'\n" >&2
    return 1
  fi
  _lerobot_require_real_id "NEBIUS_PROJECT_ID" "${NEBIUS_PROJECT_ID}" "project-<real-id>" || return 1
  if [ -z "${NEBIUS_REGION:-}" ]; then
    _lerobot_env_fail "NEBIUS_REGION is not set"
    printf "   Set it with: export NEBIUS_REGION='eu-north1'\n" >&2
    return 1
  fi

  _lerobot_require_cmd nebius || return 1
  _lerobot_require_cmd jq    || return 1

  repo_root="$(_lerobot_repo_root)" || {
    _lerobot_env_fail "Unable to determine the repository root"
    return 1
  }
  terraform_dir="${repo_root}/terraform"
  env_file="${repo_root}/.env"
  backend_file="${terraform_dir}/terraform_backend_override.tf"

  if [ ! -d "${terraform_dir}" ]; then
    _lerobot_env_fail "Terraform directory not found at ${terraform_dir}"
    return 1
  fi

  printf "\n"
  printf "Validating Nebius credentials...\n"
  printf "   Tenant:  %s\n" "${NEBIUS_TENANT_ID}"
  printf "   Project: %s\n" "${NEBIUS_PROJECT_ID}"
  printf "   Region:  %s\n" "${NEBIUS_REGION}"

  # ── IAM token ────────────────────────────────────────────────────────

  printf "Getting IAM access token...\n"
  if ! NEBIUS_IAM_TOKEN="$(nebius iam get-access-token)"; then
    _lerobot_env_fail "Failed to get an IAM access token from the nebius CLI"
    return 1
  fi
  if [ -z "${NEBIUS_IAM_TOKEN}" ]; then
    _lerobot_env_fail "IAM token is empty"
    return 1
  fi
  export NEBIUS_IAM_TOKEN
  printf "  IAM token obtained\n"

  # ── VPC subnet (informational only — Terraform creates its own) ─────

  printf "Checking for existing VPC subnets...\n"
  NEBIUS_VPC_SUBNET_ID="$(
    nebius vpc subnet list \
      --parent-id "${NEBIUS_PROJECT_ID}" \
      --format json 2>/dev/null \
      | jq -r '.items[0].metadata.id // empty'
  )" || true

  if [ -n "${NEBIUS_VPC_SUBNET_ID}" ]; then
    printf "  Found existing subnet: %s (Terraform will create its own)\n" "${NEBIUS_VPC_SUBNET_ID}"
  else
    printf "  No existing subnets (Terraform will create one)\n"
  fi

  # ── S3 bucket ────────────────────────────────────────────────────────

  printf "Setting up S3 bucket...\n"
  bucket_suffix="$(_lerobot_bucket_suffix "${NEBIUS_TENANT_ID}-${NEBIUS_PROJECT_ID}")" || return 1
  export NEBIUS_BUCKET_NAME="lerobot-${bucket_suffix}"

  if ! exists="$(
    nebius storage bucket list \
      --parent-id "${NEBIUS_PROJECT_ID}" \
      --format json \
      | jq -r --arg B "${NEBIUS_BUCKET_NAME}" \
          '[.items[]? | select(.metadata.name == $B) | .metadata.name][0] // empty'
  )"; then
    _lerobot_env_fail "Failed to query Nebius object storage buckets"
    return 1
  fi

  if [ -z "${exists}" ]; then
    if ! nebius storage bucket create \
      --name "${NEBIUS_BUCKET_NAME}" \
      --parent-id "${NEBIUS_PROJECT_ID}" \
      --versioning-policy 'enabled' >/dev/null; then
      _lerobot_env_fail "Failed to create bucket ${NEBIUS_BUCKET_NAME}"
      return 1
    fi
    printf "  Created new bucket: %s\n" "${NEBIUS_BUCKET_NAME}"
  else
    printf "  Using existing bucket: %s\n" "${NEBIUS_BUCKET_NAME}"
  fi

  # ── Service account ──────────────────────────────────────────────────

  printf "Setting up service account...\n"
  if ! NEBIUS_SA_ID="$(
    nebius iam service-account get-by-name \
      --parent-id "${NEBIUS_PROJECT_ID}" \
      --name "lerobot-training" \
      --format json 2>/dev/null \
      | jq -r '.metadata.id // empty'
  )"; then
    _lerobot_env_fail "Failed to query the lerobot-training service account"
    return 1
  fi

  if [ -z "${NEBIUS_SA_ID}" ]; then
    if ! NEBIUS_SA_ID="$(
      nebius iam service-account create \
        --parent-id "${NEBIUS_PROJECT_ID}" \
        --name "lerobot-training" \
        --description "Service account for LeRobot training on Nebius" \
        --format json \
        | jq -r '.metadata.id // empty'
    )"; then
      _lerobot_env_fail "Failed to create the lerobot-training service account"
      return 1
    fi
    if [ -z "${NEBIUS_SA_ID}" ]; then
      _lerobot_env_fail "Service account creation did not return an ID"
      return 1
    fi
    printf "  Created service account: %s\n" "${NEBIUS_SA_ID}"
  else
    printf "  Using existing service account: %s\n" "${NEBIUS_SA_ID}"
  fi
  export NEBIUS_SA_ID

  # ── Service account permissions ──────────────────────────────────────

  printf "Configuring service account permissions...\n"
  if ! NEBIUS_GROUP_EDITORS_ID="$(
    nebius iam group get-by-name \
      --parent-id "${NEBIUS_TENANT_ID}" \
      --name 'editors' \
      --format json \
      | jq -r '.metadata.id // empty'
  )"; then
    _lerobot_env_fail "Failed to query the tenant editors group"
    return 1
  fi
  if [ -z "${NEBIUS_GROUP_EDITORS_ID}" ]; then
    _lerobot_env_fail "Could not find the editors group in tenant ${NEBIUS_TENANT_ID}"
    return 1
  fi

  if ! IS_MEMBER="$(
    nebius iam group-membership list-members \
      --parent-id "${NEBIUS_GROUP_EDITORS_ID}" \
      --page-size 1000 \
      --format json 2>/dev/null \
      | jq -r --arg SAID "${NEBIUS_SA_ID}" \
          '[.memberships[]? | select(.spec.member_id == $SAID) | .spec.member_id][0] // empty'
  )"; then
    _lerobot_env_fail "Failed to inspect editors group membership"
    return 1
  fi

  if [ -z "${IS_MEMBER}" ]; then
    if ! nebius iam group-membership create \
      --parent-id "${NEBIUS_GROUP_EDITORS_ID}" \
      --member-id "${NEBIUS_SA_ID}" >/dev/null; then
      _lerobot_env_fail "Failed to add the service account to the editors group"
      return 1
    fi
    printf "  Added service account to editors group\n"
  else
    printf "  Service account already in editors group\n"
  fi

  # ── Access key (reuse unexpired, rotate expired, or create new) ──────

  printf "Setting up access key for S3...\n"

  local _create_new_key=true
  local _existing_key_json

  # Look for an existing access key for this service account.
  _existing_key_json="$(
    nebius iam v2 access-key list \
      --parent-id "${NEBIUS_PROJECT_ID}" \
      --format json 2>/dev/null \
      | jq -r --arg SAID "${NEBIUS_SA_ID}" \
          '[.items[]? | select(((.spec.account.service_account.id // .spec.account.service_account_id // "") == $SAID) and ((.status.state // "") == "ACTIVE"))][0] // empty'
  )" || true

  if [ -n "${_existing_key_json}" ]; then
    NEBIUS_SA_ACCESS_KEY_ID="$(printf '%s' "${_existing_key_json}" | jq -r '.metadata.id // empty')"
    AWS_SECRET_ACCESS_KEY="$(printf '%s' "${_existing_key_json}" | jq -r '.status.secret // empty')"
    local _expires_at
    _expires_at="$(printf '%s' "${_existing_key_json}" | jq -r '.spec.expires_at // empty')"

    if [ -n "${_expires_at}" ]; then
      # Check if key is still valid (compare with current UTC time).
      local _now_epoch _exp_epoch
      _now_epoch="$(date -u +%s)"
      if [ "$(uname)" = "Darwin" ]; then
        _exp_epoch="$(date -j -f '%Y-%m-%dT%H:%M:%SZ' "${_expires_at}" +%s 2>/dev/null)" || _exp_epoch=0
      else
        _exp_epoch="$(date -d "${_expires_at}" +%s 2>/dev/null)" || _exp_epoch=0
      fi

      if [ "${_exp_epoch}" -gt "${_now_epoch}" ]; then
        printf "  Reusing existing access key: %s (expires %s)\n" "${NEBIUS_SA_ACCESS_KEY_ID}" "${_expires_at}"
        _create_new_key=false
      else
        printf "  Existing access key %s is expired, creating replacement...\n" "${NEBIUS_SA_ACCESS_KEY_ID}"
        nebius iam v2 access-key delete --id "${NEBIUS_SA_ACCESS_KEY_ID}" >/dev/null 2>&1 || true
      fi
    else
      # No expiry set — reuse it.
      printf "  Reusing existing access key: %s (no expiry)\n" "${NEBIUS_SA_ACCESS_KEY_ID}"
      _create_new_key=false
    fi
  fi

  if [ "${_create_new_key}" = true ]; then
    printf "Fetching AWS-compatible credentials (new key)...\n"
    _lerobot_create_access_key || return 1
  fi

  # ── AWS-compatible credentials ───────────────────────────────────────

  printf "Fetching AWS access key ID...\n"
  if [ "${_create_new_key}" = false ]; then
    if ! AWS_ACCESS_KEY_ID="$(
      nebius iam v2 access-key get \
        --id "${NEBIUS_SA_ACCESS_KEY_ID}" \
        --format json \
        | jq -r '.status.aws_access_key_id // empty'
    )"; then
      _lerobot_env_fail "Failed to fetch the AWS access key ID"
      return 1
    fi
    export AWS_ACCESS_KEY_ID
  fi

  # For reused keys, the secret must come from the existing .env file.
  if [ "${_create_new_key}" = false ] && [ -z "${AWS_SECRET_ACCESS_KEY:-}" ]; then
    if [ -f "${env_file}" ]; then
      AWS_SECRET_ACCESS_KEY="$(
        grep -m1 '^AWS_SECRET_ACCESS_KEY=' "${env_file}" 2>/dev/null | cut -d= -f2-
      )" || true
    fi
    if [ -n "${AWS_SECRET_ACCESS_KEY}" ]; then
      export AWS_SECRET_ACCESS_KEY
    else
      printf "  Warning: cannot retrieve secret for existing key from .env.\n" >&2
      printf "  Creating a new key to obtain a fresh secret...\n" >&2
      # Delete the old key and recurse once.
      nebius iam v2 access-key delete --id "${NEBIUS_SA_ACCESS_KEY_ID}" >/dev/null 2>&1 || true
      _lerobot_create_access_key || return 1
      printf "  Created replacement key: %s\n" "${NEBIUS_SA_ACCESS_KEY_ID}"
    fi
  fi

  if [ -z "${AWS_ACCESS_KEY_ID}" ] || [ -z "${AWS_SECRET_ACCESS_KEY}" ]; then
    _lerobot_env_fail "Nebius did not return complete AWS-compatible S3 credentials"
    return 1
  fi
  export AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY
  printf "  AWS credentials obtained\n"

  # ── Terraform backend ────────────────────────────────────────────────

  printf "Configuring Terraform backend...\n"
  cat > "${backend_file}" <<EOF
terraform {
  backend "s3" {
    bucket = "${NEBIUS_BUCKET_NAME}"
    key    = "lerobot/terraform.tfstate"

    endpoints = {
      s3 = "https://storage.${NEBIUS_REGION}.nebius.cloud:443"
    }
    region = "${NEBIUS_REGION}"

    skip_region_validation      = true
    skip_credentials_validation = true
    use_path_style              = true
    skip_requesting_account_id  = true
    skip_s3_checksum            = true
  }
}
EOF
  printf "  Terraform backend: s3://%s/lerobot/\n" "${NEBIUS_BUCKET_NAME}"

  # ── Export TF_VAR_* ──────────────────────────────────────────────────

  export TF_VAR_iam_token="${NEBIUS_IAM_TOKEN}"
  export TF_VAR_nebius_project_id="${NEBIUS_PROJECT_ID}"
  export TF_VAR_nebius_region="${NEBIUS_REGION}"
  export TF_VAR_service_account_id="${NEBIUS_SA_ID}"
  export TF_VAR_nebius_api_key="${AWS_ACCESS_KEY_ID}"
  export TF_VAR_nebius_secret_key="${AWS_SECRET_ACCESS_KEY}"
  export TF_VAR_s3_bucket="${NEBIUS_BUCKET_NAME}"
  export TF_VAR_s3_endpoint="https://storage.${NEBIUS_REGION}.nebius.cloud"

  # ── .env file (merge — preserve user-added values like HF_TOKEN) ────

  printf "Updating .env file...\n"

  # Keys that environment.sh manages. Any user-added lines are preserved.
  local managed_keys="NEBIUS_API_KEY NEBIUS_SECRET_KEY NEBIUS_ACCOUNT_ID NEBIUS_PROJECT_ID"
  managed_keys="${managed_keys} NEBIUS_REGION NEBIUS_S3_ENDPOINT NEBIUS_S3_BUCKET"
  managed_keys="${managed_keys} AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY"

  local new_env
  new_env="$(cat <<EOF
# Managed by environment.sh — last updated $(date)
# Do NOT commit this file to git.

NEBIUS_API_KEY=${AWS_ACCESS_KEY_ID}
NEBIUS_SECRET_KEY=${AWS_SECRET_ACCESS_KEY}
NEBIUS_ACCOUNT_ID=${NEBIUS_TENANT_ID}
NEBIUS_PROJECT_ID=${NEBIUS_PROJECT_ID}
NEBIUS_REGION=${NEBIUS_REGION}

NEBIUS_S3_ENDPOINT=https://storage.${NEBIUS_REGION}.nebius.cloud
NEBIUS_S3_BUCKET=${NEBIUS_BUCKET_NAME}

AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
EOF
)"

  # If .env already exists, extract user-added lines (not managed, not blank, not comments).
  if [ -f "${env_file}" ]; then
    local user_lines=""
    while IFS= read -r line; do
      # Skip blanks and comments.
      case "${line}" in
        ""|\#*) continue ;;
        NEBIUS_API_KEY=*|NEBIUS_SECRET_KEY=*|NEBIUS_ACCOUNT_ID=*|NEBIUS_PROJECT_ID=*|NEBIUS_REGION=*|NEBIUS_S3_ENDPOINT=*|NEBIUS_S3_BUCKET=*|AWS_ACCESS_KEY_ID=*|AWS_SECRET_ACCESS_KEY=*) continue ;;
      esac
      # Extract key name (everything before first =).
      local key="${line%%=*}"
      # Skip if this key is managed by us.
      local is_managed=false
      for mk in ${managed_keys}; do
        if [ "${key}" = "${mk}" ]; then
          is_managed=true
          break
        fi
      done
      if [ "${is_managed}" = false ]; then
        user_lines="${user_lines}${line}
"
      fi
    done < "${env_file}"

    if [ -n "${user_lines}" ]; then
      new_env="${new_env}

# User-added values (preserved across reruns):
${user_lines}"
    fi
  else
    # First run — add commented-out optional keys as hints.
    new_env="${new_env}

# Optional — uncomment and fill in:
# HF_TOKEN=
# WANDB_API_KEY=
# WANDB_PROJECT=lerobot-nebius"
  fi

  printf '%s\n' "${new_env}" > "${env_file}"
  chmod 600 "${env_file}"
  printf "  .env updated (chmod 600)\n"

  # ── Summary ──────────────────────────────────────────────────────────

  printf "\n"
  printf "===================================================================\n"
  printf "  LeRobot on Nebius — Environment Ready\n"
  printf "===================================================================\n"
  printf "\n"
  printf "  Tenant:         %s\n" "${NEBIUS_TENANT_ID}"
  printf "  Project:        %s\n" "${NEBIUS_PROJECT_ID}"
  printf "  Region:         %s\n" "${NEBIUS_REGION}"
  printf "  S3 Bucket:      %s\n" "${NEBIUS_BUCKET_NAME}"
  printf "  Service Acct:   %s\n" "${NEBIUS_SA_ID}"
  printf "\n"
  printf "  Next:\n"
  printf "    cd %s\n" "${terraform_dir}"
  printf "    terraform init\n"
  printf "    terraform plan\n"
  printf "    terraform apply\n"
  printf "\n"
  printf "===================================================================\n"
}

_lerobot_environment_setup "$@"
_lerobot_environment_status=$?

unset -f _lerobot_environment_setup
unset -f _lerobot_env_fail
unset -f _lerobot_require_cmd
unset -f _lerobot_require_real_id
unset -f _lerobot_repo_root
unset -f _lerobot_bucket_suffix
unset -f _lerobot_access_key_expiration_date
unset -f _lerobot_delete_named_access_key
unset -f _lerobot_create_access_key

return "${_lerobot_environment_status}" 2>/dev/null || exit "${_lerobot_environment_status}"
