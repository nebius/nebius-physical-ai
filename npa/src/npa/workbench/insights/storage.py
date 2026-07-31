"""URI storage helpers for the insights lineage + metrics store."""

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


def uri_exists(uri: str) -> bool:
    if is_s3_uri(uri):
        target = parse_s3_uri(uri)
        try:
            _s3_client().head_object(Bucket=target.bucket, Key=target.key)
            return True
        except Exception:
            return False
    return _local_path(uri).exists()


def write_json_uri(uri: str, payload: dict[str, Any]) -> None:
    data = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    write_bytes_uri(uri, data)


def read_json_uri(uri: str) -> dict[str, Any]:
    return json.loads(read_bytes_uri(uri).decode("utf-8"))


def write_text_uri(uri: str, text: str) -> None:
    write_bytes_uri(uri, text.encode("utf-8"))


def read_jsonl_uri(uri: str) -> list[dict[str, Any]]:
    """Read a JSONL object into a list of records (empty when absent)."""
    if not uri_exists(uri):
        return []
    text = read_bytes_uri(uri).decode("utf-8")
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        rows.append(json.loads(stripped))
    return rows


def append_jsonl_uri(uri: str, rows: list[dict[str, Any]]) -> int:
    """Append rows to an append-only JSONL object; return the new total count.

    Object storage has no native append, so this reads the existing object and
    rewrites it with the new rows appended. The store stays logically
    append-only (records are never mutated or removed).
    """
    existing = read_jsonl_uri(uri)
    combined = existing + list(rows)
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in combined)
    write_bytes_uri(uri, payload.encode("utf-8"))
    return len(combined)


def list_json_uris(prefix: str) -> list[str]:
    """List all ``*.json`` object URIs under a prefix (S3 or local)."""
    if is_s3_uri(prefix):
        target = parse_s3_uri(prefix)
        client = _s3_client()
        paginator = client.get_paginator("list_objects_v2")
        found: list[str] = []
        for page in paginator.paginate(Bucket=target.bucket, Prefix=target.key):
            for obj in page.get("Contents", []) or []:
                key = obj["Key"]
                if key.endswith(".json"):
                    found.append(f"s3://{target.bucket}/{key}")
        return sorted(found)
    base = _local_path(prefix)
    if base.is_file():
        return [str(base)] if base.suffix == ".json" else []
    if not base.exists():
        return []
    return sorted(str(path) for path in base.rglob("*.json"))


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
