"""Read one exact local or S3 report document with strict JSON decoding."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from npa.workbench.storage_scope import StorageAuthorizationError, authorize_uri


class ReportDocumentError(ValueError):
    """Raised when a selected report cannot be read or decoded safely.

    Args:
        None.
    Returns:
        None.
    Raises:
        None.
    """


def read_report_document(
    input_path: str, *, storage: Any | None = None
) -> dict[str, Any]:
    """Read exactly one report without listing storage or following its links.

    Args:
        input_path: Exact local JSON path, file URI or S3 object URI.
        storage: Optional scoped client implementing ``download_file``.
    Returns:
        The decoded report object.
    Raises:
        ReportDocumentError: The input is invalid, unreadable or malformed JSON.
        StorageAuthorizationError: An active service scope denies the read.
    """
    value = _exact_input(input_path)
    target = authorize_uri(value, operation="read")
    if target.kind == "local":
        assert target.local_path is not None
        return _read_local_document(target.local_path)
    return _read_s3_document(value, storage=storage)


def _exact_input(input_path: str) -> str:
    if not isinstance(input_path, str) or not input_path.strip():
        raise ReportDocumentError("--input-path must be a local path or S3 object URI")
    value = input_path.strip()
    try:
        parsed = urlparse(value)
    except ValueError as exc:
        raise ReportDocumentError(
            "--input-path must be a local path or S3 object URI"
        ) from exc
    if parsed.scheme not in {"", "file", "s3"}:
        raise ReportDocumentError("--input-path must be a local path or S3 object URI")
    if parsed.scheme == "s3" and (
        not parsed.netloc
        or not parsed.path.lstrip("/")
        or parsed.path.endswith("/")
        or parsed.query
        or parsed.fragment
        or "@" in parsed.netloc
        or ":" in parsed.netloc
    ):
        raise ReportDocumentError("--input-path must be an exact S3 object URI")
    return value


def _read_s3_document(input_path: str, *, storage: Any | None) -> dict[str, Any]:
    if storage is None:
        from npa.clients.storage import LazyStorageClient

        storage = LazyStorageClient()
    with tempfile.TemporaryDirectory(prefix="npa-report-") as temp_dir:
        local_path = Path(temp_dir) / "report.json"
        try:
            storage.download_file(input_path, str(local_path))
        except StorageAuthorizationError:
            raise
        except Exception as exc:  # noqa: BLE001 - sanitize storage provider failures
            raise ReportDocumentError(
                "could not read the report from object storage"
            ) from exc
        return _read_local_document(local_path)


def _read_local_document(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ReportDocumentError("could not read the report") from exc
    try:
        document = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (TypeError, ValueError) as exc:
        raise ReportDocumentError("report must be valid JSON") from exc
    if not isinstance(document, dict):
        raise ReportDocumentError("report must be a JSON object")
    return document


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError("invalid JSON numeric constant")
