#!/usr/bin/env bash
# Upload stock lerobot/pusht trigger to the customer's S3 bucket.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=lib/operator-env.sh
source "${SCRIPT_DIR}/lib/operator-env.sh"
# shellcheck source=lib/private-install.sh
source "${SCRIPT_DIR}/lib/private-install.sh"
# shellcheck source=lib/customer-preflight.sh
source "${SCRIPT_DIR}/lib/customer-preflight.sh"

DEMO="$(operator_demo_root || echo "${HOME}/npa-sim2real-demo")"
export NPA_SIM2REAL_DEMO="${DEMO}"
operator_install_private_config

if ! customer_preflight "${DEMO}"; then
  exit 1
fi

ROOT="$(npa_repo_root "${SCRIPT_DIR}")"
bash "${ROOT}/ops/private/sim2real-rtxpro/bootstrap-npa-venv.sh" "${ROOT}"
PY="${ROOT}/npa/.venv/bin/python"
PIP="${ROOT}/npa/.venv/bin/pip"
BATCH_ID="${CUSTOMER_BATCH_ID:-stock-demo-$(date -u +%Y%m%dT%H%M%SZ)}"

read -r BUCKET ENDPOINT S3_URI <<EOF
$("${PY}" - "${DEMO}/private/config.yaml" "${BATCH_ID}" <<'PY'
import sys, yaml
from pathlib import Path

cfg = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
storage = cfg.get("storage") or {}
bucket = str(storage.get("bucket", "")).replace("s3://", "").split("/")[0]
endpoint = storage.get("endpoint_url", "https://storage.eu-north1.nebius.cloud")
batch = sys.argv[2]
prefix = f"sim2real-triggers/{batch}/lerobot-pusht/"
print(bucket)
print(endpoint)
print(f"s3://{bucket}/{prefix}")
PY
)
EOF

STAGING="$(mktemp -d)"
trap 'rm -rf "${STAGING}"' EXIT

echo "=== Seed stock trigger (lerobot/pusht) ==="
echo "  destination: ${S3_URI}"

"${PIP}" install -q huggingface_hub 2>/dev/null || "${PIP}" install -q huggingface_hub

"${PY}" - "${STAGING}" <<'PY'
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

dest = Path(sys.argv[1])
path = snapshot_download(
    "lerobot/pusht",
    repo_type="dataset",
    allow_patterns=["meta/*", "data/chunk-000/*"],
)
src = Path(path)
for rel in src.rglob("*"):
    if rel.is_file():
        target = dest / rel.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(rel.read_bytes())
print(f"Downloaded files to {dest}")
PY

"${PY}" - "${STAGING}" "${BUCKET}" "${S3_URI}" "${ENDPOINT}" <<'PY'
import sys
from pathlib import Path
from urllib.parse import urlparse
import boto3
import yaml
from botocore.client import Config

staging, bucket, uri, endpoint = sys.argv[1:5]
parsed = urlparse(uri)
prefix = parsed.path.lstrip("/")
if prefix and not prefix.endswith("/"):
    prefix += "/"

creds = yaml.safe_load((Path.home() / ".npa" / "credentials.yaml").read_text()) or {}
storage = creds.get("storage") or creds.get("aws") or {}
ak = storage.get("aws_access_key_id") or storage.get("access_key_id")
sk = storage.get("aws_secret_access_key") or storage.get("secret_access_key")
if not ak or not sk:
    raise SystemExit("S3 keys missing in credentials.yaml")

client = boto3.client(
    "s3",
    endpoint_url=endpoint,
    aws_access_key_id=ak,
    aws_secret_access_key=sk,
    config=Config(signature_version="s3v4"),
    region_name="eu-north1",
)
root = Path(staging)
for path in root.rglob("*"):
    if not path.is_file():
        continue
    key = prefix + path.relative_to(root).as_posix()
    client.upload_file(str(path), bucket, key)
    print(f"  uploaded s3://{bucket}/{key}")
print("OK")
PY

OP_ENV="${DEMO}/private/operator.env"
if [ -f "${OP_ENV}" ]; then
  if grep -q '^TRIGGER_DATASET_URI=' "${OP_ENV}"; then
    sed -i.bak "s|^TRIGGER_DATASET_URI=.*|TRIGGER_DATASET_URI=${S3_URI}|" "${OP_ENV}"
    rm -f "${OP_ENV}.bak"
  else
    echo "TRIGGER_DATASET_URI=${S3_URI}" >> "${OP_ENV}"
  fi
  chmod 600 "${OP_ENV}"
  operator_install_private_config
fi

echo ""
echo "=== Trigger ready ==="
echo "  TRIGGER_DATASET_URI=${S3_URI}"
echo "  Run: cd ${DEMO} && ./run.sh demo"
