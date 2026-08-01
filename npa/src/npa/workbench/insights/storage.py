"""URI storage helpers for the insights lineage + metrics store."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
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


def shard_prefix_for(uri: str) -> str:
    """Directory-style prefix holding the append shards for a JSONL object.

    ``.../records.jsonl`` -> ``.../records.d``
    """
    base = uri[: -len(".jsonl")] if uri.endswith(".jsonl") else uri
    return f"{base}.d"


def list_jsonl_uris(prefix: str) -> list[str]:
    """List ``*.jsonl`` object URIs under a prefix (S3 or local), sorted."""
    if is_s3_uri(prefix):
        target = parse_s3_uri(prefix)
        client = _s3_client()
        paginator = client.get_paginator("list_objects_v2")
        found: list[str] = []
        for page in paginator.paginate(Bucket=target.bucket, Prefix=target.key.rstrip("/") + "/"):
            for obj in page.get("Contents", []) or []:
                key = obj["Key"]
                if key.endswith(".jsonl"):
                    found.append(f"s3://{target.bucket}/{key}")
        return sorted(found)
    base = _local_path(prefix)
    if not base.exists():
        return []
    return sorted(str(path) for path in base.rglob("*.jsonl"))


def read_jsonl_store(uri: str) -> list[dict[str, Any]]:
    """Read every row of an append-only JSONL store (base object + append shards).

    Reads the legacy single object first so stores written before sharding keep
    working, then every shard under ``<name>.d/`` in sorted (write-time) order.
    """
    rows = read_jsonl_uri(uri)
    for shard_uri in list_jsonl_uris(shard_prefix_for(uri)):
        rows.extend(read_jsonl_uri(shard_uri))
    return rows


def append_jsonl_uri(uri: str, rows: list[dict[str, Any]], *, previous_total: int | None = None) -> int:
    """Append rows to an append-only JSONL store; return the store's row count.

    Object storage has no native append. Rewriting one object read-modify-write
    loses data whenever two writers overlap: both read N rows and both write
    N + their own, so the last write silently drops the other's rows while both
    ingests report success. Instead, every append lands in its own immutable
    shard object under ``<name>.d/``; readers concatenate the base object and all
    shards. That keeps the store genuinely append-only (rows are never mutated or
    removed) and safe for concurrent writers with no database and no locking.

    ``previous_total`` lets a caller that already knows the pre-append count skip
    a full re-read of the store: the total is then arithmetic rather than another
    list + GET of every shard. The returned count is **best effort** under
    concurrent writers — a writer that overlaps this one may land rows this count
    does not include. It is telemetry (surfaced as ``total_records`` /
    ``total_edges``), never an input to a correctness decision.
    """
    new_rows = list(rows)
    if new_rows:
        shard_name = f"{utc_stamp()}-{uuid.uuid4().hex[:12]}.jsonl"
        payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in new_rows)
        write_bytes_uri(uri_join(shard_prefix_for(uri), shard_name), payload.encode("utf-8"))
    if previous_total is not None:
        return previous_total + len(new_rows)
    return len(read_jsonl_store(uri))


def utc_stamp() -> str:
    """Sortable UTC timestamp used to order append shards by write time."""
    return datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%S%f")


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
