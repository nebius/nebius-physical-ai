#!/usr/bin/env bash
# Print or export resolved customer asset profile (dry-run for operators).
#
# Usage:
#   CUSTOMER_ASSET_PROFILE=industrial-ur ./apply-customer-asset-profile.sh
#   CUSTOMER_ASSET_PROFILE=industrial-ur ./apply-customer-asset-profile.sh --export
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=lib/operator-config.sh
source "${SCRIPT_DIR}/lib/operator-config.sh"
# shellcheck source=lib/customer-asset-profile.sh
source "${SCRIPT_DIR}/lib/customer-asset-profile.sh"

ROOT="$(npa_repo_root "${SCRIPT_DIR}")"
_npa_cfg=()
while IFS= read -r _line; do
  _npa_cfg+=("${_line}")
done < <(operator_read_config "${ROOT}" 2>/dev/null || true)
BUCKET="${S3_BUCKET:-${_npa_cfg[0]:-YOUR-BUCKET}}"
TASK_ID="${CUSTOMER_TASK_ID:-${RUN_ID:-pilot-task}}"

if [ -z "${CUSTOMER_ASSET_PROFILE:-}" ]; then
  echo "Set CUSTOMER_ASSET_PROFILE (e.g. stock-smoke, industrial)" >&2
  echo "Profiles: $(ls -1 "${SCRIPT_DIR}/customer-asset-profiles/"*.profile.example 2>/dev/null | xargs -n1 basename | sed 's/.profile.example$//' | tr '\n' ' ')" >&2
  exit 2
fi

customer_asset_profile_apply "${SCRIPT_DIR}" "${BUCKET}" "${TASK_ID}"

if [ "${1:-}" = "--export" ]; then
  customer_asset_profile_print | while IFS= read -r line; do
    key="${line%%=*}"
    val="${line#*=}"
    [ -z "${val}" ] && continue
    printf 'export %s=%q\n' "${key}" "${val}"
  done
else
  customer_asset_profile_print
fi
