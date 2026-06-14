#!/usr/bin/env bash
# Verify a LeRobot trigger prefix on S3 before submitting the sim2real pipeline.
# Usage: trigger_preflight_s3 <s3-uri> [endpoint_url]
trigger_preflight_s3() {
  local uri="${1:?trigger dataset s3 uri required}"
  local endpoint="${2:-https://storage.eu-north1.nebius.cloud}"
  local root="$3"
  local py="${root}/npa/.venv/bin/python"

  if [ ! -x "${py}" ]; then
    echo "ERROR: npa venv required for S3 preflight — bootstrap first" >&2
    return 1
  fi

  "${py}" - "${uri}" "${endpoint}" <<'PY'
import sys
from pathlib import Path
from urllib.parse import urlparse

import boto3
import yaml
from botocore.client import Config

uri, endpoint = sys.argv[1], sys.argv[2]
parsed = urlparse(uri)
if parsed.scheme != "s3" or not parsed.netloc:
    print(f"ERROR: expected s3://bucket/prefix/ got {uri!r}", file=sys.stderr)
    sys.exit(1)
bucket = parsed.netloc
prefix = parsed.path.lstrip("/")
if prefix and not prefix.endswith("/"):
    prefix += "/"

creds_path = Path.home() / ".npa" / "credentials.yaml"
if not creds_path.exists():
    print("ERROR: ~/.npa/credentials.yaml required", file=sys.stderr)
    sys.exit(1)
creds = yaml.safe_load(creds_path.read_text()) or {}
storage = creds.get("storage") or {}
ak = storage.get("aws_access_key_id")
sk = storage.get("aws_secret_access_key")
if not ak or not sk:
    print("ERROR: S3 keys missing in credentials.yaml storage section", file=sys.stderr)
    sys.exit(1)

client = boto3.client(
    "s3",
    endpoint_url=endpoint,
    aws_access_key_id=ak,
    aws_secret_access_key=sk,
    config=Config(signature_version="s3v4"),
    region_name="eu-north1",
)

ready_keys = (
    f"{prefix}meta/info.json",
    f"{prefix}meta/episodes.jsonl",
)
parquet_found = False
for key in ready_keys:
    try:
        client.head_object(Bucket=bucket, Key=key)
        print(f"trigger OK: found s3://{bucket}/{key}")
        sys.exit(0)
    except Exception:
        pass

paginator = client.get_paginator("list_objects_v2")
for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
    for item in page.get("Contents") or []:
        k = str(item.get("Key") or "")
        if k.startswith(f"{prefix}data/") and k.endswith(".parquet"):
            parquet_found = True
            print(f"trigger OK: found s3://{bucket}/{k}")
            sys.exit(0)

print(
    f"ERROR: no LeRobot batch at {uri}\n"
    "       Upload complete dataset first (meta/info.json or data/*.parquet),\n"
    "       then trigger the pipeline.",
    file=sys.stderr,
)
if parquet_found:
    pass
sys.exit(1)
PY
}
