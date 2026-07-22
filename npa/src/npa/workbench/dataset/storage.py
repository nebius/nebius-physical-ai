"""URI storage helpers for dataset-of-record artifacts."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


@dataclass(frozen=True)
class S3Uri:
    bucket: str
    key: str


def uri_join(base: str, *parts: str) -> str:
    """Join URI path fragments without losing the scheme."""
    prefix = base.rstrip("/")
    suffix = "/".join(part.strip("/") for part in parts if part.strip("/"))
    return f"{prefix}/{suffix}" if suffix else prefix


def parse_s3_uri(uri: str) -> S3Uri:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"not an S3 URI: {uri}")
    return S3Uri(bucket=parsed.netloc, key=parsed.path.lstrip("/"))


def is_s3_uri(uri: str) -> bool:
    return uri.startswith("s3://")


def write_bytes_uri(uri: str, payload: bytes) -> None:
    if is_s3_uri(uri):
        target = parse_s3_uri(uri)
        _s3_client().put_object(Bucket=target.bucket, Key=target.key, Body=payload)
        return
    path = _local_path(uri)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def read_bytes_uri(uri: str) -> bytes:
    if is_s3_uri(uri):
        target = parse_s3_uri(uri)
        response = _s3_client().get_object(Bucket=target.bucket, Key=target.key)
        return response["Body"].read()
    return _local_path(uri).read_bytes()


def write_json_uri(uri: str, payload: dict[str, Any]) -> None:
    data = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    write_bytes_uri(uri, data)


def read_json_uri(uri: str) -> dict[str, Any]:
    return json.loads(read_bytes_uri(uri).decode("utf-8"))


def _local_path(uri: str) -> Path:
    if uri.startswith("file://"):
        return Path(urlparse(uri).path)
    return Path(uri)


def _s3_client():
    import boto3
    from botocore.config import Config as BotoConfig

    return boto3.client(
        "s3",
        endpoint_url=os.environ.get("AWS_ENDPOINT_URL") or os.environ.get("NEBIUS_S3_ENDPOINT") or None,
        config=BotoConfig(signature_version="s3v4"),
    )
