"""URI storage helpers for detection training artifacts."""

from __future__ import annotations

import json
import os
import hashlib
import tempfile
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

_S3_SETTINGS: ContextVar[dict[str, str]] = ContextVar("detection_s3_settings", default={})


@contextmanager
def storage_settings(settings: Any):
    """Request-local credentials; concurrent runs never mutate process credentials."""
    token = _S3_SETTINGS.set({
        "endpoint_url": settings.endpoint_url,
        "aws_access_key_id": settings.aws_access_key_id,
        "aws_secret_access_key": settings.aws_secret_access_key,
    })
    try:
        yield
    finally:
        _S3_SETTINGS.reset(token)


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
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".artifact-")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_bytes_uri(uri: str) -> bytes:
    if is_s3_uri(uri):
        target = parse_s3_uri(uri)
        response = _s3_client().get_object(Bucket=target.bucket, Key=target.key)
        return response["Body"].read()
    return _local_path(uri).read_bytes()


def write_json_uri(uri: str, payload: dict[str, Any]) -> None:
    data = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    write_bytes_uri(uri, data)


def describe_artifact(uri: str, *, role: str, media_type: str, schema_version: str, epoch: int | None = None):
    """Read the actual object, then record its exact identity and content hash."""
    from .schemas import ArtifactRecord

    data = read_bytes_uri(uri)
    if not data:
        raise ValueError("produced artifact is empty")
    return ArtifactRecord(
        uri=uri, role=role, media_type=media_type, schema_version=schema_version,
        sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data), epoch=epoch,
    )


def _local_path(uri: str):
    from pathlib import Path

    if uri.startswith("file://"):
        return Path(urlparse(uri).path)
    return Path(uri)


def _s3_client():
    import boto3
    from botocore.config import Config as BotoConfig

    return boto3.client(
        "s3",
        **{
            **{"endpoint_url": os.environ.get("AWS_ENDPOINT_URL_S3") or os.environ.get("AWS_ENDPOINT_URL") or os.environ.get("NEBIUS_S3_ENDPOINT") or None},
            **{key: value for key, value in _S3_SETTINGS.get().items() if value},
        },
        config=BotoConfig(signature_version="s3v4"),
    )
