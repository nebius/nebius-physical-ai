"""Fail-closed immutable and compare-and-swap S3 publication helpers."""

from __future__ import annotations

import hashlib
import json
import secrets
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


@dataclass(frozen=True)
class _AppliedMutation:
    uri: str
    snapshot: RemoteObjectSnapshot | None
    published_etag: str
    payload_sha256: str = ""
    size_bytes: int = 0
    immutable_uri: str = ""
    deleted: bool = False


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
    _replace_mutable_bytes(client, payload, uri, snapshot)
    return uri


def _replace_mutable_bytes(
    client: Any,
    payload: bytes,
    uri: str,
    snapshot: RemoteObjectSnapshot | None,
) -> str:
    """CAS bytes even when content is unchanged, returning the resulting ETag."""

    try:
        if snapshot is None:
            return str(client.put_bytes_conditional(payload, uri, if_none_match=True))
        elif isinstance(snapshot, RemoteObjectSnapshot):
            return str(
                client.put_bytes_conditional(payload, uri, if_match=snapshot.etag)
            )
        else:  # pragma: no cover - defensive type narrowing
            raise TypeError("invalid remote publication snapshot")
    except StoragePreconditionFailed as exc:
        raise PublicationConflict(f"publication alias was superseded: {uri}") from exc


class MutablePublicationTransaction:
    """Failure-atomic mutable publication guarded by one run-scoped CAS journal.

    Immutable generation objects are uploaded before entering this transaction.
    The journal serializes every cooperating finalizer. Each mutable write keeps
    the ETag returned by its CAS so a later failure can restore all prior aliases
    while the transaction still owns the journal.

    The committed journal is the one atomic generation pointer. Its object list
    binds each compatibility alias to exact bytes and, where available, an
    immutable generation URI. Readers that need a coherent multi-object view
    must consume that committed list rather than independently sampling aliases.

    A process crash deliberately leaves a ``publishing`` journal. Future
    publishers then fail closed instead of guessing whether a partial generation
    is safe to overwrite.
    """

    def __init__(
        self,
        client: Any,
        *,
        lock_uri: str,
        lock_snapshot: RemoteObjectSnapshot | None,
        transaction_id: str,
    ) -> None:
        if not transaction_id:
            raise ValueError("publication transaction identity is required")
        self.client = client
        self.lock_uri = lock_uri
        self.lock_snapshot = lock_snapshot
        self.transaction_id = transaction_id
        self.attempt_id = secrets.token_hex(16)
        self._lock_etag = ""
        self._applied: list[_AppliedMutation] = []
        self._entered = False

    def _journal_payload(self, state: str) -> bytes:
        objects = []
        if state == "committed":
            objects = [
                (
                    {
                        "uri": mutation.uri,
                        "state": "absent",
                    }
                    if mutation.deleted
                    else {
                        "uri": mutation.uri,
                        "state": "present",
                        "sha256": mutation.payload_sha256,
                        "size_bytes": mutation.size_bytes,
                        **(
                            {"immutable_uri": mutation.immutable_uri}
                            if mutation.immutable_uri
                            else {}
                        ),
                    }
                )
                for mutation in self._applied
            ]
        return (
            json.dumps(
                {
                    "schema": "npa.sim2real.mutable_publication.v1",
                    "transaction_id": self.transaction_id,
                    "attempt_id": self.attempt_id,
                    "state": state,
                    "objects": objects,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            + b"\n"
        )

    def _assert_recoverable_snapshot(self) -> None:
        if self.lock_snapshot is None:
            return
        try:
            payload = json.loads(self.lock_snapshot.payload)
        except (TypeError, ValueError) as exc:
            raise PublicationConflict(
                f"publication transaction journal is invalid: {self.lock_uri}"
            ) from exc
        attempt_id = payload.get("attempt_id") if isinstance(payload, dict) else None
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != "npa.sim2real.mutable_publication.v1"
            or payload.get("state") not in {"committed", "aborted"}
            or not isinstance(payload.get("transaction_id"), str)
            or not payload["transaction_id"]
            or not isinstance(attempt_id, str)
            or len(attempt_id) != 32
            or any(char not in "0123456789abcdef" for char in attempt_id)
            or not isinstance(payload.get("objects"), list)
        ):
            raise PublicationConflict(
                f"publication transaction is incomplete: {self.lock_uri}"
            )

    def __enter__(self) -> "MutablePublicationTransaction":
        self._assert_recoverable_snapshot()
        try:
            self._lock_etag = _replace_mutable_bytes(
                self.client,
                self._journal_payload("publishing"),
                self.lock_uri,
                self.lock_snapshot,
            )
        except PublicationConflict as exc:
            raise PublicationConflict(
                f"publication transaction was superseded: {self.lock_uri}"
            ) from exc
        self._entered = True
        return self

    def replace_file(
        self,
        path: Path,
        uri: str,
        snapshot: RemoteObjectSnapshot | None,
        *,
        immutable_uri: str = "",
    ) -> str:
        if not self._entered:
            raise RuntimeError("publication transaction has not been entered")
        payload = Path(path).read_bytes()
        etag = _replace_mutable_bytes(
            self.client,
            payload,
            uri,
            snapshot,
        )
        self._applied.append(
            _AppliedMutation(
                uri,
                snapshot,
                etag,
                payload_sha256=hashlib.sha256(payload).hexdigest(),
                size_bytes=len(payload),
                immutable_uri=immutable_uri,
            )
        )
        return uri

    def delete_file(
        self,
        uri: str,
        snapshot: RemoteObjectSnapshot | None,
    ) -> None:
        if not self._entered:
            raise RuntimeError("publication transaction has not been entered")
        if snapshot is None:
            self._applied.append(
                _AppliedMutation(
                    uri,
                    snapshot,
                    "",
                    deleted=True,
                )
            )
            return
        delete = getattr(self.client, "delete_file_conditional", None)
        if not callable(delete):
            raise RuntimeError(
                "storage client lacks conditional object deletion support"
            )
        try:
            delete(uri, if_match=snapshot.etag)
        except StoragePreconditionFailed as exc:
            raise PublicationConflict(
                f"publication alias was superseded: {uri}"
            ) from exc
        self._applied.append(
            _AppliedMutation(
                uri,
                snapshot,
                snapshot.etag,
                deleted=True,
            )
        )

    def _restore_mutation(self, mutation: _AppliedMutation) -> None:
        if mutation.deleted:
            if mutation.snapshot is not None:
                _replace_mutable_bytes(
                    self.client,
                    mutation.snapshot.payload,
                    mutation.uri,
                    None,
                )
            return
        if mutation.snapshot is None:
            delete = getattr(self.client, "delete_file_conditional", None)
            if not callable(delete):
                raise RuntimeError(
                    "storage client lacks conditional object deletion support"
                )
            delete(mutation.uri, if_match=mutation.published_etag)
            return
        self.client.put_bytes_conditional(
            mutation.snapshot.payload,
            mutation.uri,
            if_match=mutation.published_etag,
        )

    def _rollback(self) -> list[BaseException]:
        failures: list[BaseException] = []
        for mutation in reversed(self._applied):
            try:
                self._restore_mutation(mutation)
            except BaseException as exc:  # preserve every rollback failure as context
                failures.append(exc)
        return failures

    def _finish_journal(self, state: str) -> None:
        try:
            self._lock_etag = str(
                self.client.put_bytes_conditional(
                    self._journal_payload(state),
                    self.lock_uri,
                    if_match=self._lock_etag,
                )
            )
        except StoragePreconditionFailed as exc:
            raise PublicationConflict(
                f"publication transaction journal was superseded: {self.lock_uri}"
            ) from exc

    @staticmethod
    def _attach_failures(
        error: BaseException,
        failures: list[BaseException],
    ) -> None:
        for failure in failures:
            error.add_note(f"publication rollback also failed: {failure!r}")

    def __exit__(
        self, exc_type: object, exc: BaseException | None, _tb: object
    ) -> bool:
        try:
            if exc is not None:
                failures = self._rollback()
                try:
                    self._finish_journal("aborted")
                except BaseException as journal_error:
                    failures.append(journal_error)
                self._attach_failures(exc, failures)
                return False

            try:
                self._finish_journal("committed")
            except BaseException as commit_error:
                failures = self._rollback()
                try:
                    self._finish_journal("aborted")
                except BaseException as journal_error:
                    failures.append(journal_error)
                self._attach_failures(commit_error, failures)
                raise
            return False
        finally:
            self._entered = False
