"""Bounded S3 mutation probe for a future NCore workload prefix."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import secrets
from typing import Any
from urllib.parse import urlparse

from npa.errors import NpaError


PROBE_FORMAT = "npa_ncore_s3_handoff_probe_v1"


class NcoreS3ProbeError(NpaError):
    """The workload handoff prefix failed a required object-store operation."""


def _prefix(uri: str) -> tuple[str, str]:
    parsed = urlparse(str(uri).strip())
    prefix = parsed.path.lstrip("/")
    if (
        parsed.scheme != "s3"
        or not parsed.netloc
        or not prefix
        or not prefix.endswith("/")
        or any(part in {"", ".", ".."} for part in prefix.rstrip("/").split("/"))
    ):
        raise NcoreS3ProbeError("probe prefix must be a non-root s3:// prefix")
    return parsed.netloc, prefix


def _snapshot(client: Any, bucket: str, prefix: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for page in client.s3.get_paginator("list_objects_v2").paginate(
        Bucket=bucket, Prefix=prefix
    ):
        for item in page.get("Contents", []):
            key = item.get("Key")
            size = item.get("Size")
            etag = str(item.get("ETag") or "").strip()
            if (
                not isinstance(key, str)
                or not key.startswith(prefix)
                or type(size) is not int
                or size < 0
                or not etag
            ):
                raise NcoreS3ProbeError("prefix enumeration returned invalid metadata")
            rows.append(
                {
                    "key_sha256": hashlib.sha256(key.encode()).hexdigest(),
                    "bytes": size,
                    "etag_sha256": hashlib.sha256(etag.encode()).hexdigest(),
                }
            )
    return sorted(rows, key=lambda row: row["key_sha256"])


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _delete_owned_probe(
    client: Any,
    *,
    bucket: str,
    key: str,
    uri: str,
    owned_payloads: tuple[bytes, ...],
) -> None:
    readback = client.read_bytes_with_etag(uri)
    if readback is None:
        return
    payload, etag = readback
    if payload not in owned_payloads:
        raise NcoreS3ProbeError("probe cleanup refused an ownership mismatch")
    deleted = client.s3.delete_object(Bucket=bucket, Key=key, IfMatch=etag)
    status = int(deleted.get("ResponseMetadata", {}).get("HTTPStatusCode", 0) or 0)
    if status and status not in {200, 204}:
        raise NcoreS3ProbeError("probe object deletion failed")


def probe_s3_handoff(
    prefix_uri: str,
    output_path: Path,
    *,
    storage_client: Any = None,
) -> dict[str, Any]:
    """Prove create/read/CAS/list/delete against one fresh run-owned object."""
    from npa.clients.storage import (
        StorageClient,
        StoragePreconditionFailed,
    )

    bucket, prefix = _prefix(prefix_uri)
    client = storage_client or StorageClient.from_environment()
    nonce = secrets.token_hex(16)
    key = prefix + ".npa-capability-probe-" + nonce
    uri = f"s3://{bucket}/{key}"
    payload = secrets.token_bytes(257)
    overwrite_control = b"must-not-win"
    payload_sha = hashlib.sha256(payload).hexdigest()
    key_sha = hashlib.sha256(key.encode()).hexdigest()
    may_have_created = False
    try:
        before = _snapshot(client, bucket, prefix)
        may_have_created = True
        try:
            etag = client.put_bytes_conditional(payload, uri, if_none_match=True)
        except StoragePreconditionFailed as exc:
            may_have_created = False
            raise NcoreS3ProbeError("fresh random probe key was not absent") from exc
        readback = client.read_bytes_with_etag(uri)
        if readback is None or readback[0] != payload or readback[1] != etag:
            raise NcoreS3ProbeError("conditional object read-back differs")
        try:
            client.put_bytes_conditional(overwrite_control, uri, if_none_match=True)
        except StoragePreconditionFailed:
            nonoverwrite_rejected = True
        else:
            raise NcoreS3ProbeError("conditional non-overwrite control was accepted")
        during = _snapshot(client, bucket, prefix)
        matches = [row for row in during if row["key_sha256"] == key_sha]
        if (
            len(matches) != 1
            or matches[0]["bytes"] != len(payload)
            or matches[0]["etag_sha256"] != hashlib.sha256(etag.encode()).hexdigest()
        ):
            raise NcoreS3ProbeError("prefix enumeration did not bind the probe object")
        _delete_owned_probe(
            client,
            bucket=bucket,
            key=key,
            uri=uri,
            owned_payloads=(payload, overwrite_control),
        )
        if client.read_bytes_with_etag(uri) is not None:
            raise NcoreS3ProbeError("probe object remained readable after deletion")
        after = _snapshot(client, bucket, prefix)
        if any(row["key_sha256"] == key_sha for row in after):
            raise NcoreS3ProbeError("probe object remained in prefix enumeration")
        may_have_created = False
        receipt = {
            "format": PROBE_FORMAT,
            "status": "ok",
            "scope_sha256": hashlib.sha256(prefix_uri.encode()).hexdigest(),
            "payload_sha256": payload_sha,
            "payload_bytes": len(payload),
            "etag_sha256": hashlib.sha256(etag.encode()).hexdigest(),
            "conditional_create": True,
            "exact_readback": True,
            "conditional_nonoverwrite_rejected": nonoverwrite_rejected,
            "enumerated_exactly_once": True,
            "delete_status": "confirmed",
            "absent_after_delete": True,
            "before_inventory_sha256": _canonical_sha(before),
            "during_inventory_sha256": _canonical_sha(during),
            "after_inventory_sha256": _canonical_sha(after),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(
            output_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(receipt, stream, indent=2, sort_keys=True)
                stream.write("\n")
        except Exception:
            output_path.unlink(missing_ok=True)
            raise
        return receipt
    except (NcoreS3ProbeError, OSError):
        raise
    except Exception as exc:
        raise NcoreS3ProbeError("object-store probe operation failed") from exc
    finally:
        if may_have_created:
            try:
                _delete_owned_probe(
                    client,
                    bucket=bucket,
                    key=key,
                    uri=uri,
                    owned_payloads=(payload, overwrite_control),
                )
            except Exception as exc:
                raise NcoreS3ProbeError("probe object cleanup failed") from exc
