#!/usr/bin/env bash
# Customer preflight — validate ~/npa-sim2real-demo/private/ before cluster submit.
#
# shellcheck source=asset-profile-guard.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/asset-profile-guard.sh"

customer_derive_trigger_uri() {
  local demo="$1"
  local py="$2"
  "${py}" - "${demo}/private/config.yaml" <<'PY'
import sys, yaml
from pathlib import Path

cfg = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
storage = cfg.get("storage") or {}
bucket = str(storage.get("bucket", "")).replace("s3://", "").split("/")[0]
if not bucket or "YOUR" in bucket.upper():
    raise SystemExit(1)
batch = "stock-demo"
print(f"s3://{bucket}/sim2real-triggers/{batch}/lerobot-pusht/")
PY
}

customer_preflight() {
  local demo="${1:?demo root required}"
  local priv="${demo}/private"
  local cfg="${priv}/config.yaml"
  local cred="${priv}/credentials.yaml"
  local kube_base="${priv}/clusters"
  local missing=0

  # setup.sh installs into ~/.npa/ — accept that layout when private/ is absent.
  if [ ! -f "${cfg}" ] && [ -f "${HOME}/.npa/config.yaml" ]; then
    cfg="${HOME}/.npa/config.yaml"
    cred="${HOME}/.npa/credentials.yaml"
    kube_base="${HOME}/.npa/clusters"
  fi

  for f in "${cfg}" "${cred}"; do
    if [ ! -f "${f}" ]; then
      echo "ERROR: missing ${f} — run ./setup.sh first" >&2
      missing=1
    elif grep -qE 'YOUR[-_]|example-bucket|<configure' "${f}" 2>/dev/null; then
      echo "ERROR: edit template placeholders in ${f}" >&2
      missing=1
    fi
  done

  local ctx=""
  if [ -f "${cfg}" ]; then
    ctx="$(grep -E 'k8s_context:' "${cfg}" | head -1 | sed 's/.*: *//' | tr -d '"' || true)"
  fi
  if [ -z "${ctx}" ] || echo "${ctx}" | grep -q 'YOUR'; then
    echo "ERROR: set storage.k8s_context in ${cfg}" >&2
    missing=1
  elif [ ! -f "${kube_base}/${ctx}/kubeconfig" ] \
    && [ ! -f "${kube_base}/${ctx}/kubeconfig.resolved" ]; then
    echo "ERROR: missing ${kube_base}/${ctx}/kubeconfig (or .resolved)" >&2
    echo "       Re-run ./setup.sh or see GUIDE.md §1.5." >&2
    missing=1
  fi

  local op_env="${HOME}/.npa/sim2real-operator.env"
  if [ -f "${op_env}" ] && grep -qE 'YOUR-TASK-ID|YOUR-BUCKET' "${op_env}" 2>/dev/null; then
    echo "ERROR: ${op_env} contains YOUR-TASK-ID or YOUR-BUCKET placeholders" >&2
    echo "       Stock Franka: remove CUSTOMER_ASSET_PROFILE lines from that file." >&2
    echo "       Custom assets: set CUSTOMER_TASK_ID to your slug and re-run apply-customer-asset-profile.sh --export" >&2
    missing=1
  fi
  if [ -f "${op_env}" ] \
    && grep -qE '^export CUSTOMER_ASSET_PROFILE=' "${op_env}" 2>/dev/null \
    && ! grep -qE '^export CUSTOMER_TASK_ID=' "${op_env}" 2>/dev/null \
    && ! customer_task_id_is_valid; then
    echo "ERROR: ${op_env} exports CUSTOMER_ASSET_PROFILE without CUSTOMER_TASK_ID" >&2
    echo "       Stock Franka: remove CUSTOMER_ASSET_PROFILE from that file." >&2
    missing=1
  fi
  if [ -n "${CUSTOMER_ASSET_PROFILE:-}" ] && ! customer_task_id_is_valid; then
    echo "ERROR: CUSTOMER_ASSET_PROFILE=${CUSTOMER_ASSET_PROFILE} without a real CUSTOMER_TASK_ID" >&2
    echo "       Stock Franka: unset CUSTOMER_ASSET_PROFILE (and remove from ${op_env})." >&2
    echo "       Custom assets: export CUSTOMER_TASK_ID=<your-slug> before trigger." >&2
    missing=1
  fi
  if ! customer_asset_guard_placeholders; then
    missing=1
  fi

  return "${missing}"
}
