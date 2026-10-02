"""Fail-closed immutable and compare-and-swap S3 publication helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from npa.clients.storage import StoragePreconditionFailed


class PublicationConflict(RuntimeError):
    """Raised when another publisher changed an authority object first."""


@dataclass(frozen=True)
class RemoteObjectSnapshot:
    payload: bytes
    etag: str


def remote_object_snapshot(
    client: Any,
    uri: str,
) -> RemoteObjectSnapshot | None:
    reader = getattr(client, "read_bytes_with_etag", None)
    conditional = getattr(client, "put_bytes_conditional", None)
    if not callable(reader) or not callable(conditional):
        raise RuntimeError(
            "storage client lacks conditional object publication support"
        )
    current = reader(uri)
    if current is None:
        return None
    payload, etag = current
    return RemoteObjectSnapshot(bytes(payload), str(etag))


def upload_immutable_file(client: Any, path: Path, uri: str) -> str:
    """Create one immutable object or prove an existing object is byte-identical."""

    payload = Path(path).read_bytes()
    snapshot = remote_object_snapshot(client, uri)
    if isinstance(snapshot, RemoteObjectSnapshot):
        if snapshot.payload != payload:
            raise PublicationConflict(f"immutable publication object differs: {uri}")
        return uri
    try:
        client.put_bytes_conditional(payload, uri, if_none_match=True)
    except StoragePreconditionFailed as exc:
        current = remote_object_snapshot(client, uri)
        if not isinstance(current, RemoteObjectSnapshot) or current.payload != payload:
            raise PublicationConflict(
                f"immutable publication object was superseded: {uri}"
            ) from exc
    return uri


def upload_immutable_tree(client: Any, local_dir: Path, uri: str) -> str:
    """Publish a non-symlink directory without overwriting different bytes."""

    local_dir = Path(local_dir)
    files = sorted(path for path in local_dir.rglob("*") if path.is_file())
    if not files or any(path.is_symlink() for path in files):
        raise ValueError(f"immutable publication tree is empty or unsafe: {local_dir}")
    prefix = uri.rstrip("/") + "/"
    for path in files:
        relative = path.relative_to(local_dir).as_posix()
        upload_immutable_file(client, path, prefix + relative)
    return prefix


def replace_mutable_file(
    client: Any,
    path: Path,
    uri: str,
    snapshot: RemoteObjectSnapshot | None,
) -> str:
    """Replace one alias only if it still has the version observed pre-publication."""

    payload = Path(path).read_bytes()
    if isinstance(snapshot, RemoteObjectSnapshot) and snapshot.payload == payload:
        return uri
    try:
        if snapshot is None:
            client.put_bytes_conditional(payload, uri, if_none_match=True)
        elif isinstance(snapshot, RemoteObjectSnapshot):
            client.put_bytes_conditional(payload, uri, if_match=snapshot.etag)
        else:  # pragma: no cover - defensive type narrowing
            raise TypeError("invalid remote publication snapshot")
    except StoragePreconditionFailed as exc:
        raise PublicationConflict(f"publication alias was superseded: {uri}") from exc
    return uri


def delete_mutable_file(
    client: Any,
    uri: str,
    snapshot: RemoteObjectSnapshot | None,
) -> None:
    """Remove a stale optional alias after its authoritative report is committed."""

    if snapshot is None:
        return
    delete = getattr(client, "delete_file_conditional", None)
    if not callable(delete):
        raise RuntimeError("storage client lacks conditional object deletion support")
    try:
        delete(uri, if_match=snapshot.etag)
    except StoragePreconditionFailed as exc:
        raise PublicationConflict(f"publication alias was superseded: {uri}") from exc
