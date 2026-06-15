#!/usr/bin/env bash
# Verify S3 trigger path and write access before submitting the sim2real pipeline.
#
# Usage:
#   trigger_preflight_s3 <s3-uri> [endpoint_url] [repo_root]
#   storage_preflight_write <bucket> [endpoint_url] [repo_root]
#   storage_preflight_cluster_secret <k8s_context> <expected_endpoint> [repo_root]
trigger_preflight_s3() {
  local uri="${1:?trigger dataset s3 uri required}"
  local endpoint="${2:-https://storage.us-central1.nebius.cloud}"
  local root="${3:-}"
  local py
  py="$(_trigger_preflight_python "${root}")" || return 1

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

region = "us-central1" if "us-central1" in endpoint else "eu-north1"
client = boto3.client(
    "s3",
    endpoint_url=endpoint,
    aws_access_key_id=ak,
    aws_secret_access_key=sk,
    config=Config(signature_version="s3v4"),
    region_name=region,
)

ready_keys = (
    f"{prefix}meta/info.json",
    f"{prefix}meta/episodes.jsonl",
)
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
            print(f"trigger OK: found s3://{bucket}/{k}")
            sys.exit(0)

print(
    f"ERROR: no LeRobot batch at {uri}\n"
    "       Run ./ops/private/sim2real-rtxpro/seed-stock-trigger.sh on a new bucket,\n"
    "       or upload a complete dataset (meta/info.json or data/*.parquet) first.",
    file=sys.stderr,
)
sys.exit(1)
PY
}

storage_preflight_write() {
  local bucket="${1:?bucket required}"
  local endpoint="${2:-https://storage.us-central1.nebius.cloud}"
  local root="${3:-}"
  local py
  py="$(_trigger_preflight_python "${root}")" || return 1

  "${py}" - "${bucket}" "${endpoint}" <<'PY'
import sys
from pathlib import Path

import boto3
import yaml
from botocore.client import Config

bucket, endpoint = sys.argv[1], sys.argv[2]
creds = yaml.safe_load((Path.home() / ".npa" / "credentials.yaml").read_text()) or {}
storage = creds.get("storage") or {}
ak = storage.get("aws_access_key_id")
sk = storage.get("aws_secret_access_key")
if not ak or not sk:
    print("ERROR: S3 keys missing in ~/.npa/credentials.yaml", file=sys.stderr)
    sys.exit(1)
region = "us-central1" if "us-central1" in endpoint else "eu-north1"
client = boto3.client(
    "s3",
    endpoint_url=endpoint,
    aws_access_key_id=ak,
    aws_secret_access_key=sk,
    config=Config(signature_version="s3v4"),
    region_name=region,
)
probe = "sim2real-b/_preflight-write-probe/delete-me.txt"
try:
    client.put_object(Bucket=bucket, Key=probe, Body=b"npa-preflight")
    client.delete_object(Bucket=bucket, Key=probe)
    print(f"write OK: s3://{bucket}/sim2real-b/ ({endpoint})")
except Exception as exc:
    print(
        f"ERROR: cannot write to s3://{bucket}/sim2real-b/ at {endpoint}\n"
        f"       {type(exc).__name__}: {exc}\n"
        "       Fix IAM for this bucket/region, then re-run seed-stock-trigger.sh\n"
        "       and sync-cluster-storage-secret.sh.",
        file=sys.stderr,
    )
    sys.exit(1)
PY
}

storage_preflight_cluster_secret() {
  local ctx="${1:?k8s context required}"
  local expected_endpoint="${2:?expected endpoint required}"
  local root="${3:-}"
  local py
  py="$(_trigger_preflight_python "${root}")" || return 1

  "${py}" - "${ctx}" "${expected_endpoint}" <<'PY'
import base64
import json
import subprocess
import sys

ctx, expected = sys.argv[1], sys.argv[2].rstrip("/")
raw = subprocess.check_output(
    [
        "kubectl",
        "--context",
        ctx,
        "-n",
        "default",
        "get",
        "secret",
        "npa-storage-credentials",
        "-o",
        "json",
    ],
    text=True,
)
data = json.loads(raw).get("data") or {}
endpoint_b64 = data.get("AWS_ENDPOINT_URL") or data.get("S3_ENDPOINT_URL") or ""
if not endpoint_b64:
    print(
        "ERROR: npa-storage-credentials missing AWS_ENDPOINT_URL — "
        "run ./ops/private/sim2real-rtxpro/sync-cluster-storage-secret.sh",
        file=sys.stderr,
    )
    sys.exit(1)
actual = base64.b64decode(endpoint_b64).decode("utf-8").rstrip("/")
if actual != expected.rstrip("/"):
    print(
        f"ERROR: cluster secret endpoint mismatch\n"
        f"       secret:  {actual}\n"
        f"       config:  {expected}\n"
        "       Run ./ops/private/sim2real-rtxpro/sync-cluster-storage-secret.sh",
        file=sys.stderr,
    )
    sys.exit(1)
print(f"cluster secret OK: AWS_ENDPOINT_URL={actual}")
PY
}

_trigger_preflight_python() {
  local root="${1:-}"
  local py=""
  if [ -n "${root}" ] && [ -x "${root}/npa/.venv/bin/python" ]; then
    py="${root}/npa/.venv/bin/python"
  elif [ -n "${NPA_SIM2REAL_REPO:-}" ] && [ -x "${NPA_SIM2REAL_REPO}/npa/.venv/bin/python" ]; then
    py="${NPA_SIM2REAL_REPO}/npa/.venv/bin/python"
  elif command -v python3 >/dev/null; then
    py="python3"
  else
    echo "ERROR: python3 required for S3 preflight" >&2
    return 1
  fi
  printf '%s\n' "${py}"
}
