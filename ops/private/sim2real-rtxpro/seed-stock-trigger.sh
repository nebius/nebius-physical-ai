#!/usr/bin/env bash
# Copy the validated stock LeRobot pusht trigger into the operator bucket.
#
# Run once per new bucket/region before the first cluster submit.
#
# Usage:
#   ./ops/private/sim2real-rtxpro/seed-stock-trigger.sh
#   SOURCE_TRIGGER_URI=s3://<golden-bucket>/sim2real-triggers/.../lerobot-pusht/ \
#     ./ops/private/sim2real-rtxpro/seed-stock-trigger.sh
#
# Validates the destination prefix after copy (meta/info.json or data/*.parquet).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=lib/operator-config.sh
source "${SCRIPT_DIR}/lib/operator-config.sh"
# shellcheck source=lib/trigger-preflight.sh
source "${SCRIPT_DIR}/lib/trigger-preflight.sh"
ROOT="$(npa_repo_root "${SCRIPT_DIR}")"
PY="${ROOT}/npa/.venv/bin/python"

if [ ! -x "${PY}" ]; then
  echo "ERROR: bootstrap npa venv first (${ROOT}/npa/.venv/bin/python)" >&2
  exit 1
fi

npa_read_lines _cfg operator_read_config "${ROOT}"
BUCKET="${S3_BUCKET:-${_cfg[0]:-}}"
ENDPOINT="${S3_ENDPOINT:-${_cfg[1]:-https://storage.eu-north1.nebius.cloud}}"
STOCK_BATCH="${STOCK_TRIGGER_BATCH:-trigger-validate-20260611T154016Z}"
DEST_URI="${DEST_TRIGGER_URI:-s3://${BUCKET}/sim2real-triggers/${STOCK_BATCH}/lerobot-pusht/}"
SOURCE_URI="${SOURCE_TRIGGER_URI:-s3://lerobot-d87cf691/sim2real-triggers/${STOCK_BATCH}/lerobot-pusht/}"
SOURCE_ENDPOINT="${SOURCE_ENDPOINT:-https://storage.eu-north1.nebius.cloud}"

if [ -z "${BUCKET}" ]; then
  echo "ERROR: set storage.bucket in ~/.npa/config.yaml" >&2
  exit 1
fi

echo "=== Seed stock trigger ==="
echo "  source: ${SOURCE_URI} (${SOURCE_ENDPOINT})"
echo "  dest:   ${DEST_URI} (${ENDPOINT})"

"${PY}" - "${SOURCE_URI}" "${SOURCE_ENDPOINT}" "${DEST_URI}" "${ENDPOINT}" <<'PY'
import sys
from pathlib import Path
from urllib.parse import urlparse

import boto3
import yaml
from botocore.client import Config

source_uri, source_endpoint, dest_uri, dest_endpoint = sys.argv[1:5]

def parse_s3(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise SystemExit(f"ERROR: expected s3://bucket/prefix/ got {uri!r}")
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return bucket, prefix

def load_client(endpoint: str):
    creds = yaml.safe_load((Path.home() / ".npa" / "credentials.yaml").read_text()) or {}
    storage = creds.get("storage") or {}
    ak = storage.get("aws_access_key_id")
    sk = storage.get("aws_secret_access_key")
    if not ak or not sk:
        raise SystemExit("ERROR: ~/.npa/credentials.yaml storage keys required")
    region = "us-central1" if "us-central1" in endpoint else "eu-north1"
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=ak,
        aws_secret_access_key=sk,
        config=Config(signature_version="s3v4"),
        region_name=region,
    )

src_bucket, src_prefix = parse_s3(source_uri)
dst_bucket, dst_prefix = parse_s3(dest_uri)
src = load_client(source_endpoint)
dst = load_client(dest_endpoint)

copied = 0
paginator = src.get_paginator("list_objects_v2")
for page in paginator.paginate(Bucket=src_bucket, Prefix=src_prefix):
    for item in page.get("Contents") or []:
        key = str(item.get("Key") or "")
        if not key.startswith(src_prefix):
            continue
        rel = key[len(src_prefix) :]
        if not rel:
            continue
        dst_key = f"{dst_prefix}{rel}"
        body = src.get_object(Bucket=src_bucket, Key=key)["Body"].read()
        dst.put_object(Bucket=dst_bucket, Key=dst_key, Body=body)
        copied += 1
        if copied <= 5 or copied % 50 == 0:
            print(f"copied s3://{dst_bucket}/{dst_key}")

if copied == 0:
    raise SystemExit(
        f"ERROR: no objects under {source_uri}\n"
        "       Set SOURCE_TRIGGER_URI to a complete LeRobot prefix."
    )
print(f"seed OK: copied {copied} objects to {dest_uri}")
PY

echo ""
echo "=== Validate destination trigger ==="
trigger_preflight_s3 "${DEST_URI}" "${ENDPOINT}" "${ROOT}"

echo ""
echo "Next: ensure ~/.npa/config.yaml has:"
echo "  storage.sim2real_stock_trigger_uri: ${DEST_URI}"
echo "Then sync cluster storage credentials:"
echo "  ./ops/private/sim2real-rtxpro/sync-cluster-storage-secret.sh"
