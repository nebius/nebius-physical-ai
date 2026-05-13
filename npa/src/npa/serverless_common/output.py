"""Output helpers for Workbench Serverless Jobs."""

from __future__ import annotations

from urllib.parse import urlparse


def validate_output_path(output_path: str) -> None:
    """Raise ValueError if output_path is not a valid S3 URI."""

    parsed = urlparse(output_path)
    if parsed.scheme != "s3":
        raise ValueError(f"output_path must start with s3://; got {output_path}")
    if not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError(f"output_path missing bucket/prefix: {output_path}")


def build_serverless_output_upload_cmd(local_dir: str, output_path: str) -> str:
    """Return a bash snippet that uploads local_dir contents to NPA_OUTPUT_PATH."""

    return f'''NPA_PYTHON_BIN="${{NPA_PYTHON_BIN:-python3}}"
if ! command -v "$NPA_PYTHON_BIN" >/dev/null 2>&1; then
  NPA_PYTHON_BIN=python
fi
"$NPA_PYTHON_BIN" << 'PYUPLOAD'
import os
import pathlib
from urllib.parse import urlparse

import boto3

output_path = os.environ.get("NPA_OUTPUT_PATH", "{output_path}")
parsed = urlparse(output_path)
if parsed.scheme != "s3" or not parsed.netloc:
    raise SystemExit(f"invalid NPA_OUTPUT_PATH: {{output_path}}")
prefix = parsed.path.lstrip("/")
if prefix and not prefix.endswith("/"):
    prefix += "/"

s3 = boto3.client("s3", endpoint_url=os.environ.get("AWS_ENDPOINT_URL"))
local_dir = pathlib.Path("{local_dir}")

for path in local_dir.rglob("*"):
    if path.is_file():
        key = prefix + str(path.relative_to(local_dir))
        s3.upload_file(str(path), parsed.netloc, key)
        print(f"uploaded s3://{{parsed.netloc}}/{{key}}", flush=True)
PYUPLOAD
'''
