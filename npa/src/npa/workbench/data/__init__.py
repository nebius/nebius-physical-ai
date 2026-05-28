"""S3 data bridge helpers for Workbench pipelines."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse


class DataBridgeError(ValueError):
    """Raised when a Workbench data bridge operation is invalid."""


@dataclass(frozen=True)
class DataObject:
    uri: str
    bucket: str
    key: str
    size_bytes: int = 0
    last_modified: str = ""


@dataclass(frozen=True)
class DataSyncResult:
    status: str
    source_uri: str
    output_path: str
    dry_run: bool
    object_count: int
    bytes_total: int
    copied: list[dict[str, Any]]
    generated_at: str


__all__ = [
    "DataObject",
    "DataSyncResult",
    "list_s3_objects",
    "parse_s3_uri",
    "status_s3_prefix",
    "sync_s3_prefix",
]


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Parse an S3 URI into bucket and key/prefix."""
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise DataBridgeError(f"Expected an s3:// URI, got: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def list_s3_objects(
    uri: str,
    *,
    s3_client: Any,
    limit: int = 0,
) -> list[DataObject]:
    """List S3 objects under a URI prefix."""
    if limit < 0:
        raise DataBridgeError(f"limit must be non-negative, got {limit}")
    bucket, prefix = parse_s3_uri(uri)
    objects: list[DataObject] = []
    for item in _iter_s3_objects(s3_client, bucket=bucket, prefix=prefix):
        objects.append(_object_from_item(bucket, item))
        if limit and len(objects) >= limit:
            break
    return objects


def status_s3_prefix(uri: str, *, s3_client: Any, sample_limit: int = 10) -> dict[str, Any]:
    """Return a compact status payload for an S3 prefix."""
    objects = list_s3_objects(uri, s3_client=s3_client)
    sample = [asdict(item) for item in objects[:sample_limit]]
    return {
        "uri": uri,
        "status": "available" if objects else "empty",
        "object_count": len(objects),
        "bytes_total": sum(item.size_bytes for item in objects),
        "sample": sample,
        "generated_at": _now_iso(),
    }


def sync_s3_prefix(
    source_uri: str,
    output_path: str,
    *,
    source_s3_client: Any,
    target_s3_client: Any | None = None,
    dry_run: bool = False,
    limit: int = 0,
) -> DataSyncResult:
    """Copy objects from one S3 prefix to another, preserving relative keys."""
    source_bucket, source_prefix = parse_s3_uri(source_uri)
    target_bucket, target_prefix = parse_s3_uri(output_path)
    target_client = target_s3_client or source_s3_client
    objects = list_s3_objects(source_uri, s3_client=source_s3_client, limit=limit)
    copied: list[dict[str, Any]] = []

    for obj in objects:
        relative_key = _relative_key(obj.key, source_prefix)
        target_key = _join_s3_key(target_prefix, relative_key)
        target_uri = f"s3://{target_bucket}/{target_key}"
        if not dry_run:
            _copy_object(
                source_s3_client=source_s3_client,
                target_s3_client=target_client,
                source_bucket=source_bucket,
                source_key=obj.key,
                target_bucket=target_bucket,
                target_key=target_key,
            )
        copied.append(
            {
                "source_uri": obj.uri,
                "target_uri": target_uri,
                "size_bytes": obj.size_bytes,
            }
        )

    return DataSyncResult(
        status="dry_run" if dry_run else "synced",
        source_uri=source_uri,
        output_path=output_path,
        dry_run=dry_run,
        object_count=len(objects),
        bytes_total=sum(item.size_bytes for item in objects),
        copied=copied,
        generated_at=_now_iso(),
    )


def _iter_s3_objects(s3_client: Any, *, bucket: str, prefix: str):
    if hasattr(s3_client, "get_paginator"):
        paginator = s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            yield from page.get("Contents", [])
        return

    token = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        page = s3_client.list_objects_v2(**kwargs)
        yield from page.get("Contents", [])
        if not page.get("IsTruncated"):
            return
        token = page.get("NextContinuationToken")
        if not token:
            return


def _object_from_item(bucket: str, item: dict[str, Any]) -> DataObject:
    key = str(item.get("Key", ""))
    last_modified = item.get("LastModified", "")
    if isinstance(last_modified, datetime):
        last_modified_text = last_modified.astimezone(timezone.utc).isoformat()
    else:
        last_modified_text = str(last_modified or "")
    return DataObject(
        uri=f"s3://{bucket}/{key}",
        bucket=bucket,
        key=key,
        size_bytes=int(item.get("Size") or 0),
        last_modified=last_modified_text,
    )


def _relative_key(key: str, source_prefix: str) -> str:
    if not source_prefix:
        return key
    if source_prefix.endswith("/") and key.startswith(source_prefix):
        return key[len(source_prefix):]
    source_dir = source_prefix.rstrip("/") + "/"
    if key.startswith(source_dir):
        return key[len(source_dir):]
    if key == source_prefix:
        return key.rsplit("/", 1)[-1]
    return key


def _join_s3_key(prefix: str, relative_key: str) -> str:
    clean_relative = relative_key.lstrip("/")
    if not clean_relative:
        raise DataBridgeError("Cannot sync an object with an empty relative key")
    clean_prefix = prefix.strip("/")
    if not clean_prefix:
        return clean_relative
    return f"{clean_prefix}/{clean_relative}"


def _copy_object(
    *,
    source_s3_client: Any,
    target_s3_client: Any,
    source_bucket: str,
    source_key: str,
    target_bucket: str,
    target_key: str,
) -> None:
    if source_s3_client is target_s3_client and hasattr(target_s3_client, "copy_object"):
        target_s3_client.copy_object(
            Bucket=target_bucket,
            Key=target_key,
            CopySource={"Bucket": source_bucket, "Key": source_key},
        )
        return

    response = source_s3_client.get_object(Bucket=source_bucket, Key=source_key)
    body = response["Body"].read()
    metadata = dict(response.get("Metadata") or {})
    target_s3_client.put_object(
        Bucket=target_bucket,
        Key=target_key,
        Body=body,
        Metadata=metadata,
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
