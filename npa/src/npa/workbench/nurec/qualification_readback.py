"""Stable complete-prefix readback for an NCore qualification run."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

from npa.errors import NpaError
from npa.workbench.nurec.s3_probe import _prefix


READBACK_FORMAT = "npa_ncore_qualification_readback_v1"


class NcoreQualificationReadbackError(NpaError):
    """A complete stable S3 qualification readback could not be proven."""


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def local_inventory(root: Path) -> list[dict[str, Any]]:
    """Hash every regular file under a symlink-free private readback root."""
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise NcoreQualificationReadbackError(
                "qualification readback contains a symbolic link"
            )
        if path.is_dir():
            continue
        if not path.is_file():
            raise NcoreQualificationReadbackError(
                "qualification readback contains a special file"
            )
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _file_sha(path),
            }
        )
    return records


def _snapshot_complete(client: Any, bucket: str, prefix: str) -> list[dict[str, Any]]:
    records = []
    for page in client.s3.get_paginator("list_objects_v2").paginate(
        Bucket=bucket, Prefix=prefix
    ):
        for item in page.get("Contents", []):
            key = item.get("Key")
            size = item.get("Size")
            etag = str(item.get("ETag") or "").strip()
            relative = str(key or "")[len(prefix) :]
            path = PurePosixPath(relative)
            if (
                not isinstance(key, str)
                or not key.startswith(prefix)
                or not relative
                or path.is_absolute()
                or any(part in {"", ".", ".."} for part in path.parts)
                or type(size) is not int
                or size < 0
                or not etag
            ):
                raise NcoreQualificationReadbackError(
                    "qualification prefix enumeration is invalid"
                )
            records.append({"path": path.as_posix(), "bytes": size, "etag": etag})
    return sorted(records, key=lambda item: item["path"])


def _write_private(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def readback_qualification(
    prefix_uri: str,
    destination: Path,
    receipt_path: Path,
    *,
    storage_client: Any = None,
) -> dict[str, Any]:
    """Download one stable complete prefix and bind every local byte."""
    from npa.clients.storage import StorageClient

    if destination.exists() or destination.is_symlink():
        raise NcoreQualificationReadbackError(
            "qualification readback destination must be new"
        )
    client = storage_client or StorageClient.from_environment()
    bucket, prefix = _prefix(prefix_uri)
    before = _snapshot_complete(client, bucket, prefix)
    if not before:
        raise NcoreQualificationReadbackError("qualification evidence prefix is empty")
    destination.mkdir(mode=0o700, parents=True)
    try:
        client.download_directory(prefix_uri, str(destination))
        for path in destination.rglob("*"):
            if path.is_file() and not path.is_symlink():
                path.chmod(0o600)
        after = _snapshot_complete(client, bucket, prefix)
        if before != after:
            raise NcoreQualificationReadbackError(
                "qualification object inventory changed during readback"
            )
        local = local_inventory(destination)
        if (
            len(local) != len(after)
            or {item["path"] for item in local} != {item["path"] for item in after}
            or any(
                local_item["bytes"] != s3_item["bytes"]
                for local_item, s3_item in zip(local, after, strict=True)
            )
        ):
            raise NcoreQualificationReadbackError(
                "local readback differs from the complete object inventory"
            )
        receipt = {
            "format": READBACK_FORMAT,
            "status": "pass",
            "prefix_sha256": hashlib.sha256(prefix_uri.encode()).hexdigest(),
            "stable_listing": True,
            "object_count": len(after),
            "s3_inventory_sha256": _canonical_sha(after),
            "s3_inventory": after,
            "local_inventory_sha256": _canonical_sha(local),
            "local_inventory": local,
        }
        _write_private(receipt_path, receipt)
        return receipt
    except Exception:
        if receipt_path.exists():
            receipt_path.unlink()
        raise
