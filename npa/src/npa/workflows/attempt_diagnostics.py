"""Publish immutable, attempt-scoped files from a failed workflow stage."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Protocol
from urllib.parse import urlsplit

from npa.clients.storage import StorageClient, StoragePreconditionFailed

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_S3_BUCKET = re.compile(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]")
_BUCKET_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?")


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
    if not isinstance(value, str) or any(
        ord(char) <= 32 or ord(char) == 127 for char in value
    ):
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
        or any(not _BUCKET_LABEL.fullmatch(label) for label in parsed.netloc.split("."))
        or re.fullmatch(r"\d+\.\d+\.\d+\.\d+", parsed.netloc)
        or parsed.hostname != parsed.netloc
        or parsed.geturl() != value
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
    # Parent directories belong to the caller; the final component cannot redirect.
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    except OSError as error:
        raise ValueError(f"diagnostic input must be a regular file: {path}") from error
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError(f"diagnostic input must be a regular file: {path}")
        return stream.read()


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


def _file_manifest(prefix: str, files: Mapping[str, bytes]) -> dict:
    published: dict[str, dict[str, object]] = {}
    for name, payload in sorted(files.items()):
        uri = f"{prefix}/originals/{name}"
        published[name] = {
            "uri": uri,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }
    return published


def _receipt_bytes(receipt: Mapping[str, object]) -> bytes:
    return (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()


def _attempt_prefix(root: str, run: str, stage: str, attempt: str) -> str:
    root = _validated_root_uri(root)
    identifiers = {"run_id": run, "stage": stage, "attempt_id": attempt}
    suffix = "/".join(
        _validated_identifier(value, label=key) for key, value in identifiers.items()
    )
    return f"{root}/failed-attempts/{suffix}"


def _failure_receipt(prefix, run, stage, attempt, exit_code, files) -> dict:
    return {
        "schema": "npa.workflow.failed-attempt-diagnostic.v1",
        "status": "failed",
        "classification": "diagnostic_only_not_success_or_qualification",
        "run_id": run,
        "stage": stage,
        "attempt_id": attempt,
        "exit_code": exit_code,
        "files": _file_manifest(prefix, files),
    }


def _publish_attempt(storage, prefix: str, receipt: dict, files: dict) -> dict:
    receipt_uri = f"{prefix}/failure-diagnostic.json"
    payload = _receipt_bytes(receipt)
    existing = storage.read_bytes_with_etag(receipt_uri)
    if existing is not None:
        if existing[0] != payload:
            raise ValueError("completed diagnostic attempt differs")
        for name, identity in receipt["files"].items():
            readback = storage.read_bytes_with_etag(identity["uri"])
            if readback is None or readback[0] != files[name]:
                raise ValueError("completed diagnostic original readback differs")
        return {**receipt, "receipt_uri": receipt_uri}
    # Freeze the entire attempt before originals, including competing writers.
    _verified_create(storage, uri=f"{prefix}/attempt-manifest.json", payload=payload)
    for name, identity in receipt["files"].items():
        _verified_create(storage, uri=identity["uri"], payload=files[name])
    _verified_create(storage, uri=receipt_uri, payload=payload)
    return {**receipt, "receipt_uri": receipt_uri}


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
        files: Safe destination names mapped to closed files in trusted directories.
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
    prefix = _attempt_prefix(diagnostic_root_uri, run_id, stage, attempt_id)
    prepared = _prepared_files(files)
    client = storage or StorageClient.from_environment()
    receipt = _failure_receipt(prefix, run_id, stage, attempt_id, exit_code, prepared)
    return _publish_attempt(client, prefix, receipt, prepared)
