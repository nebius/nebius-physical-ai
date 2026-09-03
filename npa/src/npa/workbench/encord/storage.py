"""Conditional artifact persistence and object metadata helpers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import fcntl
import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Protocol
from urllib.parse import unquote, urlparse

from npa.clients.storage import _parse_bucket_uri
from npa.clients.storage import StoragePreconditionFailed
from npa.workbench.encord.identity import canonical_s3_uri
from npa.workbench.encord.integrity import compare_checksums
from npa.workbench.encord.schemas import ChecksumKind, EncordToolError
from npa.workbench.storage_scope import authorize_uri


@dataclass(frozen=True)
class ArtifactVersion:
    kind: Literal["s3_etag", "local_sha256"]
    token: str


@dataclass(frozen=True)
class ObjectMetadata:
    uri: str
    exists: bool
    size: int = 0
    content_type: str = ""
    etag: str = ""
    etag_kind: ChecksumKind = "none"
    checksum: str = ""
    checksum_kind: ChecksumKind = "none"
    version_id: str = ""


class ArtifactNotFound(EncordToolError):
    pass


class ArtifactConflict(EncordToolError):
    pass


class ArtifactInvalid(EncordToolError):
    pass


class ArtifactStore(Protocol):
    def create_json(self, uri: str, payload: Mapping[str, Any]) -> ArtifactVersion: ...

    def replace_json(
        self, uri: str, payload: Mapping[str, Any], version: ArtifactVersion
    ) -> ArtifactVersion: ...

    def read_json(self, uri: str) -> Mapping[str, Any]: ...

    def head(self, uri: str) -> ObjectMetadata: ...


def json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


class ConditionalArtifactStore:
    """S3 compare-and-swap and local atomic JSON artifact storage."""

    def __init__(self, storage_client: Any) -> None:
        self._storage = storage_client

    def create_json(self, uri: str, payload: Mapping[str, Any]) -> ArtifactVersion:
        body = json_bytes(payload)
        if uri.startswith("s3://"):
            canonical = _authorized_s3_uri(uri, operation="create artifact")
            try:
                token = self._storage.put_bytes_conditional(
                    body,
                    canonical,
                    if_none_match=True,
                    content_type="application/json",
                )
            except StoragePreconditionFailed as exc:
                raise ArtifactConflict(f"artifact already exists: {uri}") from exc
            return ArtifactVersion(kind="s3_etag", token=str(token))
        path = _authorized_local_path(uri, operation="create artifact")
        if path.is_symlink() or path.is_dir():
            raise ArtifactInvalid(f"artifact target is not a regular file path: {uri}")
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(body)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError as exc:
                raise ArtifactConflict(f"artifact already exists: {uri}") from exc
            _fsync_directory(path.parent)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return ArtifactVersion(kind="local_sha256", token=_digest(body))

    def replace_json(
        self, uri: str, payload: Mapping[str, Any], version: ArtifactVersion
    ) -> ArtifactVersion:
        body = json_bytes(payload)
        if uri.startswith("s3://"):
            if version.kind != "s3_etag" or not version.token:
                raise ArtifactInvalid("S3 replacement requires an S3 ETag token")
            canonical = _authorized_s3_uri(uri, operation="replace artifact")
            try:
                token = self._storage.put_bytes_conditional(
                    body,
                    canonical,
                    if_match=version.token,
                    content_type="application/json",
                )
            except StoragePreconditionFailed as exc:
                raise ArtifactConflict(f"artifact changed before checkpoint: {uri}") from exc
            return ArtifactVersion(kind="s3_etag", token=str(token))
        if version.kind != "local_sha256" or not version.token:
            raise ArtifactInvalid("local replacement requires a local SHA-256 token")
        path = _authorized_local_path(uri, operation="replace artifact")
        if path.is_symlink() or path.is_dir():
            raise ArtifactInvalid(f"artifact target is not a regular file: {uri}")
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = path.with_name(f".{path.name}.lock")
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                current = path.read_bytes()
            except FileNotFoundError as exc:
                raise ArtifactNotFound(
                    f"artifact disappeared before checkpoint: {uri}"
                ) from exc
            if _digest(current) != version.token:
                raise ArtifactConflict(f"artifact changed before checkpoint: {uri}")
            descriptor, temporary = tempfile.mkstemp(
                prefix=f".{path.name}.", dir=path.parent
            )
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(body)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
                _fsync_directory(path.parent)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        return ArtifactVersion(kind="local_sha256", token=_digest(body))

    def read_json(self, uri: str) -> Mapping[str, Any]:
        if uri.startswith("s3://"):
            canonical = _authorized_s3_uri(uri, operation="read artifact")
            found = self._storage.read_bytes_with_etag(canonical)
            if found is None:
                raise ArtifactNotFound(f"artifact does not exist: {uri}")
            body, _ = found
        else:
            path = _authorized_local_path(uri, operation="read artifact")
            if path.is_symlink() or not path.is_file():
                raise ArtifactNotFound(f"artifact does not exist: {uri}")
            body = path.read_bytes()
        try:
            value = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ArtifactInvalid(f"artifact is not valid UTF-8 JSON: {uri}") from exc
        if not isinstance(value, Mapping):
            raise ArtifactInvalid(f"artifact must contain a JSON object: {uri}")
        return value

    def head(self, uri: str) -> ObjectMetadata:
        if uri.startswith("s3://"):
            return head_object(self._storage, uri)
        path = _authorized_local_path(uri, operation="head artifact")
        if not path.is_file():
            return ObjectMetadata(uri=uri, exists=False)
        return ObjectMetadata(uri=uri, exists=True, size=path.stat().st_size)


def head_object(storage_client: Any, uri: str) -> ObjectMetadata:
    target = authorize_uri(uri, operation="head object")
    if target.kind != "s3" or not target.bucket or not target.key:
        raise ArtifactInvalid("head object requires an exact S3 object URI")
    bucket, key = target.bucket, target.key
    canonical = canonical_s3_uri(bucket, key)
    try:
        response = storage_client.s3.head_object(
            Bucket=bucket, Key=key, ChecksumMode="ENABLED"
        )
    except TypeError:
        response = storage_client.s3.head_object(Bucket=bucket, Key=key)
    except Exception as exc:  # noqa: BLE001 - provider-specific not-found errors
        response_data = getattr(exc, "response", {}) or {}
        code = str(response_data.get("Error", {}).get("Code", ""))
        status = int(
            response_data.get("ResponseMetadata", {}).get("HTTPStatusCode", 0) or 0
        )
        if code in {"404", "NoSuchKey", "NotFound"} or status == 404:
            return ObjectMetadata(uri=canonical, exists=False)
        if code in {"InvalidArgument", "InvalidRequest", "NotImplemented"}:
            response = storage_client.s3.head_object(Bucket=bucket, Key=key)
        else:
            raise
    checksum, kind = checksum_from_head(response)
    return ObjectMetadata(
        uri=canonical,
        exists=True,
        size=int(response.get("ContentLength", 0) or 0),
        content_type=str(response.get("ContentType", "") or ""),
        etag=str(response.get("ETag", "") or "").strip('"'),
        etag_kind="etag_opaque" if response.get("ETag") else "none",
        checksum=checksum,
        checksum_kind=kind,
        version_id=str(response.get("VersionId", "") or ""),
    )


def checksum_from_head(response: Mapping[str, Any]) -> tuple[str, ChecksumKind]:
    sha256 = str(response.get("ChecksumSHA256", "") or "").strip()
    if sha256 and str(response.get("ChecksumType", "FULL_OBJECT")).upper() != "COMPOSITE":
        try:
            normalized = base64.b64decode(sha256, validate=True).hex()
        except ValueError:
            normalized = ""
        if normalized:
            return normalized, "s3_checksum_sha256"
    etag = str(response.get("ETag", "") or "").strip('"')
    if not etag:
        return "", "none"
    return etag, "etag_opaque"


def write_json_object(
    storage_client: Any, uri: str, payload: Mapping[str, Any], *, filename: str
) -> str:
    body = json_bytes(payload)
    if uri.startswith("s3://"):
        canonical = _authorized_s3_uri(uri, operation="write JSON object")
        bucket, key = _parse_bucket_uri(canonical)
        storage_client.s3.put_object(
            Bucket=bucket, Key=key, Body=body, ContentType="application/json"
        )
        return canonical
    path = _authorized_local_path(uri, operation="write JSON object", filename=filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return str(path)


def _json_path(uri: str, *, filename: str = "artifact.json") -> Path:
    path = Path(uri)
    return path if path.suffix == ".json" else path / filename


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _authorized_s3_uri(uri: str, *, operation: str) -> str:
    target = authorize_uri(uri, operation=operation)
    if target.kind != "s3" or not target.bucket or not target.key:
        raise ArtifactInvalid(f"{operation} requires an exact S3 object URI")
    return f"s3://{target.bucket}/{target.key}"


def _authorized_local_path(
    uri: str, *, operation: str, filename: str = "artifact.json"
) -> Path:
    parsed = urlparse(uri)
    raw = Path(unquote(parsed.path) if parsed.scheme == "file" else uri).expanduser()
    if raw.is_symlink():
        raise ArtifactInvalid(f"{operation} rejects symlink targets")
    target = authorize_uri(uri, operation=operation)
    if target.kind != "local" or target.local_path is None:
        raise ArtifactInvalid(f"{operation} requires a local path")
    path = target.local_path
    return path if path.suffix == ".json" else path / filename


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class TransferDigest:
    size: int
    sha256: str


class ObjectStorageGateway(Protocol):
    def list_objects(self, prefix_uri: str) -> Iterable[ObjectMetadata]: ...

    def head(self, uri: str) -> ObjectMetadata: ...

    def copy(self, source_uri: str, destination_uri: str) -> ObjectMetadata: ...

    def download_to_file(self, uri: str, destination: Path) -> TransferDigest: ...

    def upload_file(
        self,
        source: Path,
        destination_uri: str,
        *,
        digest: TransferDigest | None = None,
    ) -> ObjectMetadata: ...


class S3ObjectStorageGateway:
    """Narrow integrity-aware adapter over the repository storage client."""

    def __init__(self, storage_client: Any) -> None:
        self._storage = storage_client

    def list_objects(self, prefix_uri: str) -> Iterable[ObjectMetadata]:
        normalized_prefix = prefix_uri.rstrip("/")
        target = authorize_uri(normalized_prefix, operation="list objects")
        if target.kind != "s3" or not target.bucket:
            raise ArtifactInvalid("object listing requires an S3 prefix")
        prefix = target.key.rstrip("/") + "/" if target.key else ""
        paginator = self._storage.s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=target.bucket, Prefix=prefix):
            for row in page.get("Contents") or []:
                key = str(row["Key"])
                if key.endswith("/"):
                    continue
                yield self.head(canonical_s3_uri(target.bucket, key))

    def head(self, uri: str) -> ObjectMetadata:
        return head_object(self._storage, uri)

    def copy(self, source_uri: str, destination_uri: str) -> ObjectMetadata:
        source = self.head(source_uri)
        if not source.exists:
            raise ArtifactNotFound(f"copy source does not exist: {source_uri}")
        source_target = authorize_uri(source_uri, operation="copy source")
        destination_target = authorize_uri(destination_uri, operation="copy destination")
        if source_target.kind != "s3" or destination_target.kind != "s3":
            raise ArtifactInvalid("server-side copy requires two S3 object URIs")
        self._storage.s3.copy_object(
            Bucket=destination_target.bucket,
            Key=destination_target.key,
            CopySource={"Bucket": source_target.bucket, "Key": source_target.key},
        )
        destination = self.head(destination_uri)
        if not destination.exists or destination.size != source.size:
            raise EncordToolError("server-side copy destination size did not verify")
        checksum_match = compare_checksums(
            source.checksum,
            source.checksum_kind,
            destination.checksum,
            destination.checksum_kind,
        )
        if checksum_match is False:
            raise EncordToolError("server-side copy destination checksum did not verify")
        return destination

    def download_to_file(self, uri: str, destination: Path) -> TransferDigest:
        target = authorize_uri(uri, operation="download object")
        if target.kind != "s3" or not target.bucket or not target.key:
            raise ArtifactInvalid("object download requires an exact S3 URI")
        destination.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        size = 0
        response = self._storage.s3.get_object(Bucket=target.bucket, Key=target.key)
        body = response["Body"]
        temporary = ""
        try:
            descriptor, temporary = tempfile.mkstemp(
                prefix=f".{destination.name}.", dir=destination.parent
            )
            with os.fdopen(descriptor, "wb") as handle:
                for chunk in body.iter_chunks(chunk_size=8 * 1024 * 1024):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
            _fsync_directory(destination.parent)
        except Exception:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)
            raise
        finally:
            body.close()
        return TransferDigest(size=size, sha256=digest.hexdigest())

    def upload_file(
        self,
        source: Path,
        destination_uri: str,
        *,
        digest: TransferDigest | None = None,
    ) -> ObjectMetadata:
        if digest is not None and digest.size != source.stat().st_size:
            raise EncordToolError("local upload digest size does not match the source file")
        self._storage.upload_file(str(source), destination_uri)
        metadata = self.head(destination_uri)
        if not metadata.exists or metadata.size != source.stat().st_size:
            raise EncordToolError("uploaded object destination size did not verify")
        if digest is not None:
            checksum_match = compare_checksums(
                digest.sha256,
                "sha256",
                metadata.checksum,
                metadata.checksum_kind,
            )
            if checksum_match is False:
                raise EncordToolError(
                    "uploaded object destination checksum did not verify"
                )
        return metadata
