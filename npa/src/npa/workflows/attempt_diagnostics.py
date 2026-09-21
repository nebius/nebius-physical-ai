"""Publish immutable, attempt-scoped files from a failed workflow stage."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
import re
from typing import Protocol
from urllib.parse import urlsplit

from npa.clients.storage import StorageClient, StoragePreconditionFailed

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_S3_BUCKET = re.compile(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]")


class _DiagnosticStorage(Protocol):
    def put_bytes_conditional(
        self, payload: bytes, uri: str, *, if_none_match: bool
    ) -> str: ...

    def read_bytes_with_etag(self, uri: str) -> tuple[bytes, str] | None: ...


def _validated_identifier(value: str, *, label: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{label} must contain only letters, digits, '.', '_', or '-'")
    return value


def _validated_name(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("diagnostic name must be a string")
    path = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or path.is_absolute()
        or not path.parts
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
        or any(not _IDENTIFIER.fullmatch(part) for part in path.parts)
    ):
        raise ValueError(
            f"diagnostic name must be a canonical relative path: {value!r}"
        )
    return value


def _validated_root_uri(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError(
            "diagnostic_root_uri must be a canonical s3://bucket/prefix URI"
        )
    parsed = urlsplit(value)
    key = parsed.path.lstrip("/")
    path = PurePosixPath(key)
    canonical_path = f"/{key}" if key else ""
    if (
        parsed.scheme != "s3"
        or not _S3_BUCKET.fullmatch(parsed.netloc)
        or parsed.hostname != parsed.netloc
        or parsed.query
        or parsed.fragment
        or "\\" in value
        or parsed.path not in {canonical_path, f"{canonical_path}/"}
        or (key and (not path.parts or path.as_posix() != key.rstrip("/")))
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(
            "diagnostic_root_uri must be a canonical s3://bucket/prefix URI"
        )
    return value.rstrip("/")


def _read_regular_file(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"diagnostic input must be a regular file: {path}")
    return path.read_bytes()


def _verified_create(storage: _DiagnosticStorage, *, uri: str, payload: bytes) -> None:
    try:
        storage.put_bytes_conditional(payload, uri, if_none_match=True)
    except StoragePreconditionFailed:
        pass
    readback = storage.read_bytes_with_etag(uri)
    if readback is None or readback[0] != payload:
        raise ValueError(f"diagnostic readback differs: {uri}")


def _prepared_files(
    files: Mapping[str, str | Path],
) -> dict[str, bytes]:
    return {
        _validated_name(name): _read_regular_file(Path(source))
        for name, source in files.items()
    }


def _publish_files(
    storage: _DiagnosticStorage,
    *,
    prefix: str,
    files: Mapping[str, bytes],
) -> dict[str, dict[str, object]]:
    published: dict[str, dict[str, object]] = {}
    for name, payload in sorted(files.items()):
        uri = f"{prefix}/originals/{name}"
        _verified_create(storage, uri=uri, payload=payload)
        published[name] = {
            "uri": uri,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }
    return published


def _receipt_bytes(receipt: Mapping[str, object]) -> bytes:
    return (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()


def publish_failed_attempt(
    diagnostic_root_uri: str,
    *,
    run_id: str,
    stage: str,
    attempt_id: str,
    exit_code: int,
    files: Mapping[str, str | Path],
    storage: _DiagnosticStorage | None = None,
) -> dict[str, object]:
    """Publish an explicit file mapping and a final failed-attempt receipt.

    Args:
        diagnostic_root_uri: S3 prefix reserved for diagnostic attempts.
        run_id: Logical workflow run identifier recorded in the receipt.
        stage: Failed workflow stage name.
        attempt_id: Unique immutable identifier for this stage attempt.
        exit_code: Nonzero process exit code.
        files: Mapping of safe relative destination names to local regular files.
        storage: Optional storage client; defaults to the current environment.

    Returns:
        The verified failure receipt, including every published file identity.

    Raises:
        ValueError: An identifier, file, exit code, URI, or readback is invalid.
        ClientError: Object storage rejected a read or write.
    """
    if isinstance(exit_code, bool) or not isinstance(exit_code, int) or exit_code == 0:
        raise ValueError("failed-attempt diagnostics require a nonzero exit code")
    if not files:
        raise ValueError("failed-attempt diagnostics require at least one file")
    resolved_root = _validated_root_uri(diagnostic_root_uri)
    resolved_run = _validated_identifier(run_id, label="run_id")
    resolved_stage = _validated_identifier(stage, label="stage")
    resolved_attempt = _validated_identifier(attempt_id, label="attempt_id")
    prepared = _prepared_files(files)
    client = storage or StorageClient.from_environment()
    prefix = (
        f"{resolved_root}/failed-attempts/{resolved_run}/"
        f"{resolved_stage}/{resolved_attempt}"
    )
    published = _publish_files(client, prefix=prefix, files=prepared)
    receipt: dict[str, object] = {
        "schema": "npa.workflow.failed-attempt-diagnostic.v1",
        "status": "failed",
        "classification": "diagnostic_only_not_success_or_qualification",
        "run_id": resolved_run,
        "stage": resolved_stage,
        "attempt_id": resolved_attempt,
        "exit_code": exit_code,
        "files": published,
    }
    receipt_uri = f"{prefix}/failure-diagnostic.json"
    _verified_create(client, uri=receipt_uri, payload=_receipt_bytes(receipt))
    return {**receipt, "receipt_uri": receipt_uri}
