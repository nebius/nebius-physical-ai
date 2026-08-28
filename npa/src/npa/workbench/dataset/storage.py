"""URI storage helpers for dataset-of-record artifacts."""

from __future__ import annotations

import json
import os
from typing import Any

from npa.workbench.storage_scope import authorize_uri


def uri_join(base: str, *parts: str) -> str:
    """Join URI path fragments without losing the scheme."""
    prefix = base.rstrip("/")
    suffix = "/".join(part.strip("/") for part in parts if part.strip("/"))
    return f"{prefix}/{suffix}" if suffix else prefix


def write_bytes_uri(uri: str, payload: bytes) -> None:
    target = authorize_uri(uri, operation="write")
    if target.kind == "s3":
        _s3_client().put_object(Bucket=target.bucket, Key=target.key, Body=payload)
        return
    assert target.local_path is not None
    path = target.local_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def read_bytes_uri(uri: str) -> bytes:
    target = authorize_uri(uri, operation="read")
    if target.kind == "s3":
        response = _s3_client().get_object(Bucket=target.bucket, Key=target.key)
        return response["Body"].read()
    assert target.local_path is not None
    return target.local_path.read_bytes()


def write_json_uri(uri: str, payload: dict[str, Any]) -> None:
    data = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    write_bytes_uri(uri, data)


def read_json_uri(uri: str) -> dict[str, Any]:
    return json.loads(read_bytes_uri(uri).decode("utf-8"))


def _s3_client():
    import boto3
    from botocore.config import Config as BotoConfig

    return boto3.client(
        "s3",
        endpoint_url=os.environ.get("AWS_ENDPOINT_URL") or os.environ.get("NEBIUS_S3_ENDPOINT") or None,
        config=BotoConfig(signature_version="s3v4"),
    )
