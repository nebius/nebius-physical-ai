"""Verify exact S3 input bytes before scheduling expensive workflow stages."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping
from typing import Any, Protocol
from urllib.parse import urlparse

from botocore.exceptions import ClientError

from npa.clients.storage import StorageClient

_READ_BYTES = 8 * 1024 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MISSING_CODES = {"404", "NoSuchKey", "NotFound"}


class _Storage(Protocol):
    @property
    def s3(self) -> Any: ...


class S3InputPreflightError(RuntimeError):
    """Report every input row that prevented an exact-byte preflight.

    Args:
        rows: Verification rows for the attempted input declarations.

    Returns:
        None.

    Raises:
        None.
    """

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        failed = [row for row in rows if not row.get("verified")]
        summary = ", ".join(
            f"{row.get('uri', '<invalid>')}: {row['status']}" for row in failed
        )
        super().__init__(f"S3 input preflight failed: {summary}")


def _object_location(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    key = parsed.path.lstrip("/")
    invalid = (
        parsed.scheme != "s3"
        or not parsed.netloc
        or not key
        or key.endswith("/")
        or parsed.path != f"/{key}"
        or parsed.query
        or parsed.fragment
        or parsed.username
        or parsed.password
        or ":" in parsed.netloc
        or "\\" in key
        or "\x00" in key
        or any(ord(character) < 32 for character in uri)
        or any(part in {"", ".", ".."} for part in key.split("/"))
    )
    if invalid:
        raise ValueError("uri must be a canonical exact s3://bucket/key object URI")
    return parsed.netloc, key


def _declaration(value: Mapping[str, object]) -> dict[str, Any]:
    uri = value.get("uri")
    digest = value.get("sha256")
    size = value.get("bytes")
    if not isinstance(uri, str):
        raise ValueError("uri must be a string")
    bucket, key = _object_location(uri)
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise ValueError("sha256 must be a complete lowercase digest")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError("bytes must be a nonnegative integer")
    return {
        "uri": uri,
        "bucket": bucket,
        "key": key,
        "expected": {"bytes": size, "sha256": digest},
    }


def _validated_declarations(
    inputs: Iterable[Mapping[str, object]],
) -> list[dict[str, Any]]:
    declarations: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, value in enumerate(inputs):
        try:
            declarations.append(_declaration(value))
        except (AttributeError, TypeError, ValueError) as error:
            uri = value.get("uri") if isinstance(value, Mapping) else None
            failures.append(
                {
                    "index": index,
                    "uri": uri if isinstance(uri, str) else "<invalid>",
                    "status": "invalid",
                    "verified": False,
                    "detail": str(error),
                }
            )
    if failures:
        raise S3InputPreflightError(failures)
    if not declarations:
        raise ValueError("S3 input preflight requires at least one declaration")
    return declarations


def _stream_identity(
    storage: _Storage, declaration: Mapping[str, Any]
) -> dict[str, Any]:
    response = storage.s3.get_object(
        Bucket=declaration["bucket"], Key=declaration["key"]
    )
    body = response["Body"]
    digest = hashlib.sha256()
    size = 0
    try:
        while chunk := body.read(_READ_BYTES):
            digest.update(chunk)
            size += len(chunk)
    finally:
        body.close()
    return {
        "bytes": size,
        "sha256": digest.hexdigest(),
        "content_length": response.get("ContentLength"),
        "etag": str(response.get("ETag") or "").strip(),
    }


def _provider_failure(uri: str, error: Exception) -> dict[str, Any]:
    code = ""
    if isinstance(error, ClientError):
        code = str(error.response.get("Error", {}).get("Code", ""))
    status = "missing" if code in _MISSING_CODES else "provider_error"
    return {
        "uri": uri,
        "status": status,
        "verified": False,
        "provider_code": code or type(error).__name__,
    }


def _verification_row(
    declaration: Mapping[str, Any], actual: Mapping[str, Any]
) -> dict[str, Any]:
    expected = declaration["expected"]
    mismatches = [
        field for field in ("bytes", "sha256") if actual[field] != expected[field]
    ]
    declared = actual["content_length"]
    if declared is not None and int(declared) != actual["bytes"]:
        mismatches.append("provider_content_length")
    return {
        "uri": declaration["uri"],
        "expected": expected,
        "actual": dict(actual),
        "status": "mismatch" if mismatches else "verified",
        "verified": not mismatches,
        "mismatches": mismatches,
    }


def _verify_input(storage: _Storage, declaration: Mapping[str, Any]) -> dict[str, Any]:
    try:
        actual = _stream_identity(storage, declaration)
        return _verification_row(declaration, actual)
    except Exception as error:  # noqa: BLE001 - aggregate every provider failure
        return _provider_failure(str(declaration["uri"]), error)


def preflight_s3_inputs(
    inputs: Iterable[Mapping[str, object]],
    *,
    storage: StorageClient | None = None,
) -> list[dict[str, Any]]:
    """Stream and verify every declared S3 input with one selected client.

    Args:
        inputs: Exact object declarations containing ``uri``, ``sha256``, and
            ``bytes``.
        storage: Selected storage client. The current environment is used when
            omitted.

    Returns:
        Ordered rows with complete streamed byte and SHA-256 evidence.

    Raises:
        ValueError: No declarations were supplied.
        S3InputPreflightError: Any declaration, object, or identity failed.
    """
    declarations = _validated_declarations(inputs)
    selected: _Storage = (
        storage if storage is not None else StorageClient.from_environment()
    )
    rows = [_verify_input(selected, declaration) for declaration in declarations]
    if not all(row["verified"] for row in rows):
        raise S3InputPreflightError(rows)
    return rows
