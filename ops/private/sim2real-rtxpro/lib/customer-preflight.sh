#!/usr/bin/env bash
# Customer preflight — validate ~/npa-sim2real-demo/private/ before cluster submit.

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
  local missing=0

  for f in config.yaml credentials.yaml; do
    if [ ! -f "${priv}/${f}" ]; then
      echo "ERROR: missing ${priv}/${f}" >&2
      missing=1
    elif grep -qE 'YOUR[-_]|example-bucket|<configure' "${priv}/${f}" 2>/dev/null; then
      echo "ERROR: edit template placeholders in ${priv}/${f}" >&2
      missing=1
    fi
  done

  local ctx=""
  if [ -f "${priv}/config.yaml" ]; then
    ctx="$(grep -E 'k8s_context:' "${priv}/config.yaml" | head -1 | sed 's/.*: *//' | tr -d '"' || true)"
  fi
  if [ -z "${ctx}" ] || echo "${ctx}" | grep -q 'YOUR'; then
    echo "ERROR: set storage.k8s_context in ${priv}/config.yaml" >&2
    missing=1
  elif [ ! -f "${priv}/clusters/${ctx}/kubeconfig" ]; then
    echo "ERROR: missing ${priv}/clusters/${ctx}/kubeconfig" >&2
    echo "       See QUICKSTART.md for kubeconfig copy steps." >&2
    missing=1
  fi

  return "${missing}"
}
