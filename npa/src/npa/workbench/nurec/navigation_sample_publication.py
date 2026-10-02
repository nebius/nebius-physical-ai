"""Recover exact scan-report bytes without replacing native evaluation publications."""

from pathlib import Path

from npa.clients.storage import StorageClient, StoragePreconditionFailed
from npa.workbench.nurec.navigation_publication import (
    _CLAIM,
    _SEAL,
    _inventory,
    _records,
    _s3_location,
)


def _remote_names(storage, destination):
    bucket, prefix, _ = _s3_location(destination)
    pages = storage.s3.get_paginator("list_objects_v2").paginate(
        Bucket=bucket, Prefix=prefix
    )
    return {
        item["Key"].removeprefix(prefix)
        for page in pages
        for item in page.get("Contents", [])
    }


def evaluation_exists(destination: str) -> bool:
    """Detect occupied native output before deciding whether execution is needed.

    Args:
        destination: Local evaluation directory or run-scoped S3 prefix.
    Returns:
        Whether any output exists; callers must verify its completion and identity.
    Raises:
        ValueError: The destination URI is unsupported.
        OSError: Local inspection fails.
        Exception: Provider listing failures propagate without launching native work.
    """
    if destination.startswith("s3://"):
        return bool(_remote_names(StorageClient.from_environment(), destination))
    if not destination or "://" in destination:
        raise ValueError("evaluation output must be a local directory or S3 prefix")
    path = Path(destination)
    return path.exists() or path.is_symlink()


def _same_bytes(existing, expected):
    if existing != expected:
        raise ValueError("existing scan report differs; use a fresh report prefix")


def _put_remote(storage, uri, payload):
    existing = storage.read_bytes_with_etag(uri)
    if existing is not None:
        _same_bytes(existing[0], payload)
        return
    try:
        storage.put_bytes_conditional(payload, uri, if_none_match=True)
    except StoragePreconditionFailed:
        existing = storage.read_bytes_with_etag(uri)
        _same_bytes(None if existing is None else existing[0], payload)


def _put_local(path, payload):
    if path.is_symlink():
        raise ValueError("scan report publication cannot traverse symlinks")
    try:
        with path.open("xb") as stream:
            stream.write(payload)
    except FileExistsError:
        _same_bytes(path.read_bytes(), payload)


def _report_records(directory):
    members = _inventory(directory)
    claim, seal = _records(members)
    return {
        _CLAIM: claim,
        **{name: (directory / name).read_bytes() for name in members},
        _SEAL: seal,
    }


def publish_report(directory: Path, destination: str) -> None:
    """Complete a matching report attempt using conditional, identical-byte writes.

    Args:
        directory: Fresh deterministic report derived from verified native evidence.
        destination: Local report directory or run-scoped S3 prefix.
    Returns:
        None after every artifact and the completion seal have been retained.
    Raises:
        ValueError: Existing bytes, members or paths conflict with this report.
        OSError: Local publication fails.
        Exception: Provider failures propagate, leaving retryable immutable bytes.
    """
    records = _report_records(directory)
    if destination.startswith("s3://"):
        _publish_remote(records, destination)
    else:
        _publish_local(records, destination)


def _check_names(names, records):
    if names - records.keys() or (names and _CLAIM not in names):
        raise ValueError("scan report contains unexpected or unclaimed artifacts")


def _publish_remote(records, destination):
    storage = StorageClient.from_environment()
    names = _remote_names(storage, destination)
    _check_names(names, records)
    base = destination.rstrip("/") + "/"
    for name in names:
        existing = storage.read_bytes_with_etag(base + name)
        _same_bytes(None if existing is None else existing[0], records[name])
    for name, payload in records.items():
        _put_remote(storage, base + name, payload)


def _publish_local(records, destination):
    root = _local_directory(destination)
    names = {path.name for path in root.iterdir()}
    _check_names(names, records)
    for name in names:
        path = root / name
        if path.is_symlink():
            raise ValueError("scan report publication cannot traverse symlinks")
        _same_bytes(path.read_bytes(), records[name])
    for name, payload in records.items():
        _put_local(root / name, payload)


def _local_directory(destination):
    if not destination or "://" in destination:
        raise ValueError("report output must be a local directory or S3 prefix")
    root = Path(destination)
    if any(path.is_symlink() for path in (root, *root.parents)):
        raise ValueError("scan report publication cannot traverse symlinks")
    root.mkdir(parents=True, exist_ok=True)
    return root
