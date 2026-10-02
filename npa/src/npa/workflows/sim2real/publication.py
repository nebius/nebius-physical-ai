"""Fail-closed immutable and compare-and-swap S3 publication helpers."""

from __future__ import annotations

import hashlib
import json
import secrets
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from botocore.exceptions import ClientError

from npa.agent_backend.publication_reader import (
    canonical_publication_uri,
    CommittedPublicationSnapshot,
    PublicationConflict,
    parse_publication_journal,
    publication_root_from_canonical_uri,
    publication_root_from_lock_uri,
    resolve_committed_publication,
    valid_publication_id,
)
from npa.clients.storage import StoragePreconditionFailed


_UNSET_SNAPSHOT = object()


@dataclass(frozen=True)
class RemoteObjectSnapshot:
    payload: bytes | None
    etag: str
    size_bytes: int = -1
    sha256: str = ""


@dataclass(frozen=True)
class _PlannedMutation:
    uri: str
    snapshot: RemoteObjectSnapshot | None
    payload_sha256: str = ""
    size_bytes: int = 0
    immutable_uri: str = ""
    path: Path | None = None
    payload: bytes | None = None
    deleted: bool = False

    def journal_object(self) -> dict[str, object]:
        if self.deleted:
            return {"uri": self.uri, "state": "absent"}
        return {
            "uri": self.uri,
            "state": "present",
            "sha256": self.payload_sha256,
            "size_bytes": self.size_bytes,
            "immutable_uri": self.immutable_uri,
        }

    def materialize(self) -> bytes:
        payload = self.payload
        if payload is None:
            if self.path is None:
                raise RuntimeError("publication plan has no payload source")
            digest, size = _file_identity(self.path)
            if size != self.size_bytes or digest != self.payload_sha256:
                raise PublicationConflict(
                    f"planned publication bytes changed before commit: {self.uri}"
                )
            raise RuntimeError("file-backed publication must use the streaming path")
        if (
            len(payload) != self.size_bytes
            or hashlib.sha256(payload).hexdigest() != self.payload_sha256
        ):
            raise PublicationConflict(
                f"planned publication bytes changed before commit: {self.uri}"
            )
        return payload


def _file_identity(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _remote_matches_file(
    client: Any,
    uri: str,
    path: Path,
    *,
    digest: str,
    size: int,
) -> bool:
    version = remote_object_version(client, uri)
    if version is None or version.size_bytes != size:
        return False
    if version.sha256 and version.sha256 != digest:
        return False
    downloader = getattr(client, "download_file", None)
    if callable(downloader):
        with tempfile.TemporaryDirectory(
            prefix="npa-publication-compare-"
        ) as directory:
            downloaded = Path(directory) / "object"
            downloader(uri, str(downloaded))
            return _file_identity(downloaded) == (digest, size)
    snapshot = remote_object_snapshot(client, uri)
    if not isinstance(snapshot, RemoteObjectSnapshot):
        return False
    return snapshot.payload == path.read_bytes()


def remote_object_snapshot(
    client: Any,
    uri: str,
) -> RemoteObjectSnapshot | None:
    reader = getattr(client, "read_small_bytes_with_etag", None)
    if not callable(reader):
        reader = getattr(client, "read_bytes_with_etag", None)
    if not callable(reader):
        raise RuntimeError("storage client lacks versioned object-read support")
    current = reader(uri)
    if current is None:
        return None
    payload, etag = current
    material = bytes(payload)
    return RemoteObjectSnapshot(
        material,
        str(etag),
        len(material),
        hashlib.sha256(material).hexdigest(),
    )


def remote_object_version(
    client: Any,
    uri: str,
) -> RemoteObjectSnapshot | None:
    """Read only CAS and content identity metadata when the client supports it."""

    reader = getattr(client, "read_object_version", None)
    if not callable(reader):
        return remote_object_snapshot(client, uri)
    current = reader(uri)
    if current is None:
        return None
    etag, size, digest = current
    if (
        not isinstance(etag, str)
        or not etag
        or type(size) is not int
        or size < 0
        or not isinstance(digest, str)
    ):
        raise RuntimeError(f"storage client returned invalid object version: {uri}")
    return RemoteObjectSnapshot(None, etag, size, digest)


def upload_immutable_file(client: Any, path: Path, uri: str) -> str:
    """Create one immutable object or prove an existing object is byte-identical."""

    path = Path(path)
    digest, size = _file_identity(path)
    version = remote_object_version(client, uri)
    if isinstance(version, RemoteObjectSnapshot):
        if not _remote_matches_file(
            client,
            uri,
            path,
            digest=digest,
            size=size,
        ):
            raise PublicationConflict(f"immutable publication object differs: {uri}")
        return uri
    file_writer = getattr(client, "put_file_conditional", None)
    try:
        if callable(file_writer):
            file_writer(
                str(path),
                uri,
                if_none_match=True,
                sha256=digest,
            )
            if _file_identity(path) != (digest, size):
                raise PublicationConflict(
                    f"immutable publication source changed during upload: {uri}"
                )
        else:
            client.put_bytes_conditional(
                path.read_bytes(),
                uri,
                if_none_match=True,
            )
    except StoragePreconditionFailed as exc:
        if not _remote_matches_file(
            client,
            uri,
            path,
            digest=digest,
            size=size,
        ):
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

    path = Path(path)
    digest, size = _file_identity(path)
    _replace_mutable_file(
        client,
        path,
        uri,
        snapshot,
        digest=digest,
        size=size,
    )
    return uri


def replace_unjournaled_legacy_file(
    client: Any,
    path: Path,
    uri: str,
    *,
    snapshot: RemoteObjectSnapshot | None | object = _UNSET_SNAPSHOT,
    lock_snapshot: RemoteObjectSnapshot | None | object = _UNSET_SNAPSHOT,
) -> str:
    """CAS one legacy alias only while no generation journal exists.

    A journal racing this compatibility write either changes the alias first,
    causing this CAS to fail, or observes the new alias before publishing its
    complete plan. If the marker appears after the CAS, report the conflict so
    the caller never treats the direct write as authoritative.
    """

    root = publication_root_from_canonical_uri(uri)
    lock_uri = f"{root}/reports/.sim2real-publication.json"
    if lock_snapshot is not _UNSET_SNAPSHOT and lock_snapshot is not None:
        raise PublicationConflict(
            f"reserved publication alias is journal-controlled: {uri}"
        )
    if remote_object_snapshot(client, lock_uri) is not None:
        raise PublicationConflict(
            f"reserved publication alias is journal-controlled: {uri}"
        )
    expected = (
        remote_object_version(client, uri) if snapshot is _UNSET_SNAPSHOT else snapshot
    )
    if expected is not None and not isinstance(expected, RemoteObjectSnapshot):
        raise TypeError("legacy publication snapshot is invalid")
    replace_mutable_file(client, path, uri, expected)
    if remote_object_snapshot(client, lock_uri) is not None:
        raise PublicationConflict(
            f"publication journal raced legacy alias replacement: {uri}"
        )
    return uri


def delete_unjournaled_legacy_file(
    client: Any,
    uri: str,
    *,
    snapshot: RemoteObjectSnapshot | None | object = _UNSET_SNAPSHOT,
    lock_snapshot: RemoteObjectSnapshot | None | object = _UNSET_SNAPSHOT,
) -> None:
    """CAS-delete one legacy alias only while no generation journal exists."""

    root = publication_root_from_canonical_uri(uri)
    lock_uri = f"{root}/reports/.sim2real-publication.json"
    if lock_snapshot is not _UNSET_SNAPSHOT and lock_snapshot is not None:
        raise PublicationConflict(
            f"reserved publication alias is journal-controlled: {uri}"
        )
    if remote_object_snapshot(client, lock_uri) is not None:
        raise PublicationConflict(
            f"reserved publication alias is journal-controlled: {uri}"
        )
    expected = (
        remote_object_version(client, uri) if snapshot is _UNSET_SNAPSHOT else snapshot
    )
    if expected is not None and not isinstance(expected, RemoteObjectSnapshot):
        raise TypeError("legacy publication snapshot is invalid")
    if expected is not None:
        _delete_mutable_object(client, uri, if_match=expected.etag)
    if remote_object_snapshot(client, lock_uri) is not None:
        raise PublicationConflict(
            f"publication journal raced legacy alias deletion: {uri}"
        )


def _replace_mutable_file(
    client: Any,
    path: Path,
    uri: str,
    snapshot: RemoteObjectSnapshot | None,
    *,
    digest: str,
    size: int,
) -> str:
    writer = getattr(client, "put_file_conditional", None)
    if not callable(writer):
        return _replace_mutable_bytes(client, path.read_bytes(), uri, snapshot)
    prior_may_match = bool(
        snapshot is not None
        and (
            (snapshot.payload is not None and snapshot.payload == path.read_bytes())
            or (
                bool(snapshot.sha256)
                and snapshot.sha256 == digest
                and snapshot.size_bytes == size
            )
            or (snapshot.payload is None and not snapshot.sha256)
        )
    )
    try:
        if snapshot is None:
            etag = str(
                writer(
                    str(path),
                    uri,
                    if_none_match=True,
                    sha256=digest,
                )
            )
        else:
            etag = str(
                writer(
                    str(path),
                    uri,
                    if_match=snapshot.etag,
                    sha256=digest,
                )
            )
        if _file_identity(path) != (digest, size):
            raise PublicationConflict(
                f"planned publication bytes changed during commit: {uri}"
            )
        return etag
    except StoragePreconditionFailed as exc:
        raise PublicationConflict(f"publication alias was superseded: {uri}") from exc
    except PublicationConflict:
        raise
    except Exception:
        if prior_may_match or not _remote_matches_file(
            client,
            uri,
            path,
            digest=digest,
            size=size,
        ):
            raise
        current = remote_object_version(client, uri)
        if not isinstance(current, RemoteObjectSnapshot):
            raise
        return current.etag


def _attach_secondary_failure(
    primary: BaseException,
    *,
    operation: str,
    secondary: BaseException,
) -> None:
    """Annotate without allowing note formatting to replace the primary error."""

    try:
        failure_type = f"{type(secondary).__module__}.{type(secondary).__qualname__}"
    except BaseException:
        failure_type = "unknown secondary exception"
    try:
        BaseException.add_note(primary, f"{operation} also failed: {failure_type}")
    except BaseException:
        pass


def _reconcile_object_after_error(
    client: Any,
    uri: str,
    expected_payload: bytes | None,
    primary: BaseException,
) -> tuple[bool, str]:
    """Read back an ambiguously completed write/delete without masking its error."""

    try:
        current = remote_object_snapshot(client, uri)
    except BaseException as secondary:
        _attach_secondary_failure(
            primary,
            operation="publication readback",
            secondary=secondary,
        )
        return False, ""
    if expected_payload is None:
        return current is None, ""
    if (
        isinstance(current, RemoteObjectSnapshot)
        and current.payload == expected_payload
    ):
        return True, current.etag
    return False, ""


def _replace_mutable_bytes(
    client: Any,
    payload: bytes,
    uri: str,
    snapshot: RemoteObjectSnapshot | None,
) -> str:
    """CAS bytes even when content is unchanged, returning the resulting ETag."""

    if snapshot is not None and not isinstance(snapshot, RemoteObjectSnapshot):
        raise TypeError("invalid remote publication snapshot")
    conditional = getattr(client, "put_bytes_conditional", None)
    if not callable(conditional):
        raise RuntimeError(
            "storage client lacks conditional object publication support"
        )
    try:
        if snapshot is None:
            return str(conditional(payload, uri, if_none_match=True))
        return str(conditional(payload, uri, if_match=snapshot.etag))
    except StoragePreconditionFailed as exc:
        raise PublicationConflict(f"publication alias was superseded: {uri}") from exc
    except Exception as exc:
        prior_may_match = bool(
            snapshot is not None
            and (
                snapshot.payload == payload
                or (
                    bool(snapshot.sha256)
                    and snapshot.sha256 == hashlib.sha256(payload).hexdigest()
                    and snapshot.size_bytes == len(payload)
                )
                or (snapshot.payload is None and not snapshot.sha256)
            )
        )
        if prior_may_match:
            raise
        reconciled, etag = _reconcile_object_after_error(client, uri, payload, exc)
        if reconciled:
            return etag
        raise


def _delete_mutable_object(client: Any, uri: str, *, if_match: str) -> None:
    delete = getattr(client, "delete_file_conditional", None)
    if not callable(delete):
        raise RuntimeError("storage client lacks conditional object deletion support")
    try:
        delete(uri, if_match=if_match)
    except StoragePreconditionFailed as exc:
        raise PublicationConflict(f"publication alias was superseded: {uri}") from exc
    except Exception as exc:
        reconciled, _etag = _reconcile_object_after_error(client, uri, None, exc)
        if not reconciled:
            raise


def _journal_bytes(
    *,
    transaction_id: str,
    attempt_id: str,
    state: str,
    objects: list[dict[str, object]],
) -> bytes:
    return (
        json.dumps(
            {
                "schema": "npa.sim2real.mutable_publication.v1",
                "transaction_id": transaction_id,
                "attempt_id": attempt_id,
                "state": state,
                "objects": objects,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )


def _valid_publication_id(value: object) -> bool:
    return valid_publication_id(value)


def _parse_journal(
    snapshot: RemoteObjectSnapshot,
    *,
    lock_uri: str,
) -> dict[str, Any]:
    if snapshot.payload is None:
        raise PublicationConflict(
            f"publication transaction journal is invalid: {lock_uri}"
        )
    return parse_publication_journal(snapshot.payload, lock_uri=lock_uri)


def _publication_root(lock_uri: str) -> str:
    return publication_root_from_lock_uri(lock_uri)


def _assert_journal_bytes(
    client: Any,
    uri: str,
    *,
    expected_payload: bytes,
    expected_etag: str,
) -> None:
    current = remote_object_snapshot(client, uri)
    if (
        not isinstance(current, RemoteObjectSnapshot)
        or current.payload != expected_payload
        or current.etag != expected_etag
    ):
        raise PublicationConflict(f"publication transaction was superseded: {uri}")


def recover_interrupted_publication(
    client: Any,
    *,
    lock_uri: str,
    lock_snapshot: RemoteObjectSnapshot | None,
) -> bool:
    """Roll a fully planned interrupted generation forward, then commit it."""

    if lock_snapshot is None:
        return False
    journal = _parse_journal(lock_snapshot, lock_uri=lock_uri)
    if journal["state"] != "publishing":
        return False
    attempt_id = secrets.token_hex(16)
    objects = list(journal["objects"])
    publishing_payload = _journal_bytes(
        transaction_id=str(journal["transaction_id"]),
        attempt_id=attempt_id,
        state="publishing",
        objects=objects,
    )
    try:
        lock_etag = _replace_mutable_bytes(
            client,
            publishing_payload,
            lock_uri,
            lock_snapshot,
        )
    except PublicationConflict as exc:
        raise PublicationConflict(
            f"publication recovery was superseded: {lock_uri}"
        ) from exc
    downloader = getattr(client, "download_file", None)
    file_writer = getattr(client, "put_file_conditional", None)
    with tempfile.TemporaryDirectory(prefix="npa-publication-recovery-") as directory:
        for index, item in enumerate(objects):
            _assert_journal_bytes(
                client,
                lock_uri,
                expected_payload=publishing_payload,
                expected_etag=lock_etag,
            )
            uri = str(item["uri"])
            current = remote_object_version(client, uri)
            if item["state"] == "absent":
                if isinstance(current, RemoteObjectSnapshot):
                    _delete_mutable_object(client, uri, if_match=current.etag)
            else:
                immutable_uri = str(item["immutable_uri"])
                expected_digest = str(item["sha256"])
                expected_size = int(item["size_bytes"])
                if callable(downloader) and callable(file_writer):
                    immutable_path = Path(directory) / f"object-{index}"
                    downloader(immutable_uri, str(immutable_path))
                    if _file_identity(immutable_path) != (
                        expected_digest,
                        expected_size,
                    ):
                        raise PublicationConflict(
                            f"publication recovery source is invalid: {immutable_uri}"
                        )
                    current_matches = bool(
                        isinstance(current, RemoteObjectSnapshot)
                        and current.payload is not None
                        and len(current.payload) == expected_size
                        and hashlib.sha256(current.payload).hexdigest()
                        == expected_digest
                    )
                    if not current_matches:
                        _replace_mutable_file(
                            client,
                            immutable_path,
                            uri,
                            current,
                            digest=expected_digest,
                            size=expected_size,
                        )
                else:
                    immutable = remote_object_snapshot(client, immutable_uri)
                    if (
                        not isinstance(immutable, RemoteObjectSnapshot)
                        or immutable.payload is None
                        or len(immutable.payload) != expected_size
                        or hashlib.sha256(immutable.payload).hexdigest()
                        != expected_digest
                    ):
                        raise PublicationConflict(
                            f"publication recovery source is invalid: {immutable_uri}"
                        )
                    if not (
                        isinstance(current, RemoteObjectSnapshot)
                        and current.payload == immutable.payload
                    ):
                        _replace_mutable_bytes(
                            client,
                            immutable.payload,
                            uri,
                            current,
                        )
            _assert_journal_bytes(
                client,
                lock_uri,
                expected_payload=publishing_payload,
                expected_etag=lock_etag,
            )
    committed_payload = _journal_bytes(
        transaction_id=str(journal["transaction_id"]),
        attempt_id=attempt_id,
        state="committed",
        objects=objects,
    )
    try:
        _replace_mutable_bytes(
            client,
            committed_payload,
            lock_uri,
            RemoteObjectSnapshot(publishing_payload, lock_etag),
        )
    except PublicationConflict as exc:
        raise PublicationConflict(
            f"publication recovery commit was superseded: {lock_uri}"
        ) from exc
    return True


def resolve_committed_publication_snapshot(
    client: Any,
    canonical_uri: str,
) -> CommittedPublicationSnapshot:
    """Read one journal snapshot that resolves every reserved publication target."""

    def read_journal(uri: str) -> bytes | None:
        typed_reader = getattr(type(client), "read_small_bytes_with_etag", None)
        typed_fallback = getattr(type(client), "read_bytes_with_etag", None)
        if callable(typed_reader) or callable(typed_fallback):
            snapshot = remote_object_snapshot(client, uri)
            return snapshot.payload if snapshot is not None else None
        value = uri.removeprefix("s3://")
        if value == uri or "/" not in value:
            raise ValueError("publication journal is not an S3 object URI")
        bucket, key = value.split("/", 1)
        raw = getattr(client, "_s3", None) or getattr(client, "s3", None)
        if raw is None:
            raise RuntimeError("storage client lacks publication journal reads")
        try:
            raw.head_object(Bucket=bucket, Key=key)
        except KeyError:
            return None
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            if code in {"404", "NoSuchKey", "NotFound"}:
                return None
            raise
        body = None
        try:
            response = raw.get_object(Bucket=bucket, Key=key)
            body = response["Body"]
            payload = body.read(1024 * 1024 + 1)
        finally:
            close = getattr(body, "close", None)
            if callable(close):
                close()
        if len(payload) > 1024 * 1024:
            raise PublicationConflict(
                f"publication transaction journal is invalid: {uri}"
            )
        return bytes(payload)

    return resolve_committed_publication(read_journal, canonical_uri)


def resolve_committed_publication_uri(
    client: Any,
    canonical_uri: str,
) -> str | None:
    """Resolve one reserved alias through a complete committed journal snapshot."""

    return resolve_committed_publication_snapshot(client, canonical_uri).resolve(
        canonical_uri
    )


def _verify_target_identity(
    publication: CommittedPublicationSnapshot,
    canonical_uri: str,
    *,
    digest: str,
    size: int,
) -> str | None:
    canonical_uri = canonical_publication_uri(canonical_uri)
    target = publication.target(canonical_uri)
    if target.immutable_uri is None:
        return None
    if publication.journaled and (digest != target.sha256 or size != target.size_bytes):
        raise PublicationConflict(
            f"committed publication object bytes disagree with its journal: "
            f"{target.immutable_uri}"
        )
    return target.immutable_uri


def verify_committed_publication_file(
    publication: CommittedPublicationSnapshot,
    canonical_uri: str,
    path: Path,
) -> str | None:
    """Verify a downloaded reserved object against its committed byte identity."""

    canonical_uri = canonical_publication_uri(canonical_uri)
    target = publication.target(canonical_uri)
    if target.immutable_uri is None:
        return None
    digest, size = _file_identity(Path(path))
    return _verify_target_identity(
        publication,
        canonical_uri,
        digest=digest,
        size=size,
    )


def read_verified_committed_publication_bytes(
    client: Any,
    publication: CommittedPublicationSnapshot,
    canonical_uri: str,
    *,
    max_bytes: int = 1024 * 1024,
) -> bytes | None:
    """Read one bounded reserved object and enforce its committed identity."""

    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    canonical_uri = canonical_publication_uri(canonical_uri)
    target = publication.target(canonical_uri)
    if target.immutable_uri is None:
        return None
    if publication.journaled and target.size_bytes > max_bytes:
        raise PublicationConflict(
            f"committed publication object exceeds its read bound: "
            f"{target.immutable_uri}"
        )
    reader = getattr(client, "read_small_bytes_with_etag", None)
    if callable(reader):
        current = reader(target.immutable_uri, max_bytes=max_bytes)
    else:
        reader = getattr(client, "read_bytes_with_etag", None)
        if not callable(reader):
            raise RuntimeError("storage client lacks bounded object-read support")
        current = reader(target.immutable_uri)
    if current is None:
        if not publication.journaled:
            return None
        raise PublicationConflict(
            f"committed publication object is missing: {target.immutable_uri}"
        )
    payload = bytes(current[0])
    if len(payload) > max_bytes:
        raise PublicationConflict(
            f"committed publication object exceeds its read bound: "
            f"{target.immutable_uri}"
        )
    _verify_target_identity(
        publication,
        canonical_uri,
        digest=hashlib.sha256(payload).hexdigest(),
        size=len(payload),
    )
    return payload


def _stream_remote_object_identity(
    client: Any,
    uri: str,
    *,
    expected_size: int,
) -> tuple[str, int]:
    value = uri.removeprefix("s3://")
    if value == uri or "/" not in value:
        raise PublicationConflict(f"publication object URI is invalid: {uri}")
    bucket, key = value.split("/", 1)
    raw = getattr(client, "_s3", None) or getattr(client, "s3", None)
    get_object = getattr(raw, "get_object", None)
    if callable(get_object):
        body = None
        try:
            response = get_object(Bucket=bucket, Key=key)
            body = response["Body"]
            digest = hashlib.sha256()
            size = 0
            while size <= expected_size:
                chunk = body.read(min(1024 * 1024, expected_size + 1 - size))
                if not chunk:
                    break
                material = (
                    chunk.encode("utf-8") if isinstance(chunk, str) else bytes(chunk)
                )
                digest.update(material)
                size += len(material)
            return digest.hexdigest(), size
        finally:
            close = getattr(body, "close", None)
            if callable(close):
                close()
    downloader = getattr(client, "download_file", None)
    if not callable(downloader):
        downloader = getattr(client, "download_path", None)
    if callable(downloader):
        with tempfile.TemporaryDirectory(prefix="npa-publication-read-") as directory:
            path = Path(directory) / "object"
            downloader(uri, str(path))
            return _file_identity(path)
    snapshot = remote_object_snapshot(client, uri)
    if snapshot is None or snapshot.payload is None:
        raise PublicationConflict(f"committed publication object is missing: {uri}")
    return snapshot.sha256, snapshot.size_bytes


def verify_committed_publication_object(
    client: Any,
    publication: CommittedPublicationSnapshot,
    canonical_uri: str,
) -> str | None:
    """Stream-hash one selected journal target before exposing its immutable URI."""

    canonical_uri = canonical_publication_uri(canonical_uri)
    target = publication.target(canonical_uri)
    if target.immutable_uri is None:
        return None
    if not publication.journaled:
        return target.immutable_uri
    digest, size = _stream_remote_object_identity(
        client,
        target.immutable_uri,
        expected_size=target.size_bytes,
    )
    return _verify_target_identity(
        publication,
        canonical_uri,
        digest=digest,
        size=size,
    )


class MutablePublicationTransaction:
    """Publish a complete, recoverable alias plan behind one CAS journal.

    Calls inside the context only queue mutations. On clean context exit, the
    complete immutable-backed plan is written to the journal before the first
    alias changes. A later process can therefore roll an interrupted plan
    forward without guessing from partially changed compatibility aliases.
    """

    def __init__(
        self,
        client: Any,
        *,
        lock_uri: str,
        lock_snapshot: RemoteObjectSnapshot | None,
        transaction_id: str,
    ) -> None:
        if not _valid_publication_id(transaction_id):
            raise ValueError("publication transaction identity is invalid")
        self.client = client
        self.lock_uri = lock_uri
        self.lock_snapshot = lock_snapshot
        self.transaction_id = transaction_id
        self.attempt_id = secrets.token_hex(16)
        self._lock_etag = ""
        self._planned: list[_PlannedMutation] = []
        self._entered = False
        self._journal_started = False

    def _journal_payload(self, state: str) -> bytes:
        return _journal_bytes(
            transaction_id=self.transaction_id,
            attempt_id=self.attempt_id,
            state=state,
            objects=[mutation.journal_object() for mutation in self._planned],
        )

    def _assert_recoverable_snapshot(self) -> None:
        if self.lock_snapshot is None:
            return
        payload = _parse_journal(self.lock_snapshot, lock_uri=self.lock_uri)
        if payload["state"] == "publishing":
            raise PublicationConflict(
                f"publication transaction is incomplete: {self.lock_uri}"
            )

    def __enter__(self) -> "MutablePublicationTransaction":
        self._assert_recoverable_snapshot()
        self._entered = True
        return self

    def _queue(self, mutation: _PlannedMutation) -> None:
        if not self._entered:
            raise RuntimeError("publication transaction has not been entered")
        if any(existing.uri == mutation.uri for existing in self._planned):
            raise ValueError(f"publication target is duplicated: {mutation.uri}")
        self._planned.append(mutation)

    def replace_file(
        self,
        path: Path,
        uri: str,
        snapshot: RemoteObjectSnapshot | None,
        *,
        immutable_uri: str = "",
    ) -> str:
        if not immutable_uri or immutable_uri == uri:
            raise ValueError("mutable publication requires a distinct immutable source")
        path = Path(path)
        digest, size = _file_identity(path)
        self._queue(
            _PlannedMutation(
                uri=uri,
                snapshot=snapshot,
                payload_sha256=digest,
                size_bytes=size,
                immutable_uri=immutable_uri,
                path=path,
            )
        )
        return uri

    def replace_bytes(
        self,
        payload: bytes,
        uri: str,
        snapshot: RemoteObjectSnapshot | None,
        *,
        immutable_uri: str,
    ) -> str:
        if not immutable_uri or immutable_uri == uri:
            raise ValueError("mutable publication requires a distinct immutable source")
        payload = bytes(payload)
        self._queue(
            _PlannedMutation(
                uri=uri,
                snapshot=snapshot,
                payload_sha256=hashlib.sha256(payload).hexdigest(),
                size_bytes=len(payload),
                immutable_uri=immutable_uri,
                payload=payload,
            )
        )
        return uri

    def delete_file(
        self,
        uri: str,
        snapshot: RemoteObjectSnapshot | None,
    ) -> None:
        self._queue(
            _PlannedMutation(
                uri=uri,
                snapshot=snapshot,
                deleted=True,
            )
        )

    def _begin_journal(self) -> None:
        if not self._planned:
            raise RuntimeError("publication transaction plan is empty")
        publishing_payload = self._journal_payload("publishing")
        _parse_journal(
            RemoteObjectSnapshot(publishing_payload, "planned"),
            lock_uri=self.lock_uri,
        )
        try:
            self._lock_etag = _replace_mutable_bytes(
                self.client,
                publishing_payload,
                self.lock_uri,
                self.lock_snapshot,
            )
        except PublicationConflict as exc:
            raise PublicationConflict(
                f"publication transaction was superseded: {self.lock_uri}"
            ) from exc
        self._journal_started = True

    def _assert_journal_owned(self) -> None:
        _assert_journal_bytes(
            self.client,
            self.lock_uri,
            expected_payload=self._journal_payload("publishing"),
            expected_etag=self._lock_etag,
        )

    def _apply_mutation(self, mutation: _PlannedMutation) -> None:
        self._assert_journal_owned()
        if mutation.deleted:
            if mutation.snapshot is not None:
                _delete_mutable_object(
                    self.client,
                    mutation.uri,
                    if_match=mutation.snapshot.etag,
                )
        else:
            if mutation.path is not None:
                digest, size = _file_identity(mutation.path)
                if digest != mutation.payload_sha256 or size != mutation.size_bytes:
                    raise PublicationConflict(
                        "planned publication bytes changed before commit: "
                        f"{mutation.uri}"
                    )
                _replace_mutable_file(
                    self.client,
                    mutation.path,
                    mutation.uri,
                    mutation.snapshot,
                    digest=digest,
                    size=size,
                )
            else:
                _replace_mutable_bytes(
                    self.client,
                    mutation.materialize(),
                    mutation.uri,
                    mutation.snapshot,
                )
        self._assert_journal_owned()

    def _finish_journal(self, state: str) -> None:
        self._assert_journal_owned()
        try:
            self._lock_etag = _replace_mutable_bytes(
                self.client,
                self._journal_payload(state),
                self.lock_uri,
                RemoteObjectSnapshot(b"", self._lock_etag),
            )
        except PublicationConflict as exc:
            raise PublicationConflict(
                f"publication transaction journal was superseded: {self.lock_uri}"
            ) from exc

    def __exit__(
        self, exc_type: object, exc: BaseException | None, _tb: object
    ) -> bool:
        try:
            if exc is not None:
                return False

            try:
                self._begin_journal()
                for mutation in self._planned:
                    self._apply_mutation(mutation)
                self._finish_journal("committed")
            except BaseException:
                # Never guess at rollback after a storage error: the remote
                # mutation may have succeeded even when no response arrived.
                # Keep the complete immutable-backed plan in ``publishing`` so
                # the next writer can deterministically roll every target
                # forward while readers refuse the incomplete generation.
                raise
            return False
        finally:
            self._entered = False
