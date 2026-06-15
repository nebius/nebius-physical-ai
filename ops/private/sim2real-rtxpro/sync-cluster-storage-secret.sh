#!/usr/bin/env bash
# Patch default/npa-storage-credentials from ~/.npa/credentials.yaml + config endpoint.
#
# Cluster pods read AWS_* from this secret via envFrom. The endpoint must match
# storage.endpoint_url in ~/.npa/config.yaml or PutObject fails with AccessDenied.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=lib/operator-config.sh
source "${SCRIPT_DIR}/lib/operator-config.sh"
ROOT="$(npa_repo_root "${SCRIPT_DIR}")"
PY="${ROOT}/npa/.venv/bin/python"

npa_read_lines _cfg operator_read_config "${ROOT}"
ENDPOINT="${_cfg[1]:-}"
if [ -z "${ENDPOINT}" ]; then
  ENDPOINT="$(operator_resolve_storage_endpoint "${ROOT}" || true)"
fi
CTX="${KUBECONTEXT:-${_cfg[3]:-npa-rtxpro-mk8s}}"
NS="${K8S_NAMESPACE:-default}"
SECRET="${STORAGE_SECRET_NAME:-npa-storage-credentials}"

if [ -z "${ENDPOINT}" ]; then
  echo "ERROR: set storage.endpoint_url in ~/.npa/config.yaml" >&2
  exit 1
fi

export KUBECONFIG="${KUBECONFIG:-$(operator_kubeconfig_path "${CTX}")}"
operator_export_kubeconfig "${CTX}" "${ROOT}" || exit 1

eval "$("${PY}" - <<'PY'
import shlex
import yaml
from pathlib import Path

cfg = yaml.safe_load((Path.home() / ".npa" / "config.yaml").read_text()) or {}
creds = yaml.safe_load((Path.home() / ".npa" / "credentials.yaml").read_text()) or {}
storage = creds.get("storage") or {}
endpoint = str(cfg.get("storage", {}).get("endpoint_url") or storage.get("endpoint_url") or "").strip()
ak = str(storage.get("aws_access_key_id") or "").strip()
sk = str(storage.get("aws_secret_access_key") or "").strip()
if not ak or not sk:
    raise SystemExit("ERROR: storage keys missing in ~/.npa/credentials.yaml")
if not endpoint:
    raise SystemExit("ERROR: storage.endpoint_url missing in config/credentials")
for key, val in (
    ("AWS_ACCESS_KEY_ID", ak),
    ("AWS_SECRET_ACCESS_KEY", sk),
    ("AWS_ENDPOINT_URL", endpoint),
    ("S3_ENDPOINT_URL", endpoint),
    ("NEBIUS_S3_ENDPOINT", endpoint),
):
    print("export _S_%s=%s" % (key, shlex.quote(val)))
PY
)"

echo "=== Sync ${SECRET} in ${NS} (context ${CTX}) ==="
echo "  endpoint: ${ENDPOINT}"

kubectl --context "${CTX}" -n "${NS}" create secret generic "${SECRET}" \
  --from-literal=AWS_ACCESS_KEY_ID="${_S_AWS_ACCESS_KEY_ID}" \
  --from-literal=AWS_SECRET_ACCESS_KEY="${_S_AWS_SECRET_ACCESS_KEY}" \
  --from-literal=AWS_ENDPOINT_URL="${_S_AWS_ENDPOINT_URL}" \
  --from-literal=S3_ENDPOINT_URL="${_S_S3_ENDPOINT_URL}" \
  --from-literal=NEBIUS_S3_ENDPOINT="${_S_NEBIUS_S3_ENDPOINT}" \
  --dry-run=client -o yaml \
  | kubectl --context "${CTX}" -n "${NS}" apply -f -

printf 'secret OK: %s\n' "${SECRET}"
