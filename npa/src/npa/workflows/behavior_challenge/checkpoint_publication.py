"""Publish large immutable checkpoint members with exact provider readback."""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import uuid
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from io import BufferedReader
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Protocol
from urllib.parse import urlparse

from botocore.exceptions import ClientError

from npa.clients.storage import StorageClient, StoragePreconditionFailed

_READ_BYTES = 8 * 1024 * 1024
_MULTIPART_BYTES = 64 * 1024 * 1024
_MAX_WORKERS = 4


class _Storage(Protocol):
    def put_bytes_conditional(
        self, payload: bytes, uri: str, *, if_none_match: bool
    ) -> str: ...

    def download_file(self, uri: str, local_path: str) -> Any: ...

    @property
    def s3(self) -> Any: ...


def _member_name(value: str) -> str:
    path = PurePosixPath(value)
    invalid = (
        not value
        or value in {".", ".."}
        or path.is_absolute()
        or value != path.as_posix()
        or "\\" in value
        or "\x00" in value
        or "?" in value
        or "#" in value
        or any(ord(character) < 32 for character in value)
        or any(part in {"", ".", ".."} for part in path.parts)
    )
    if invalid:
        raise ValueError(f"Checkpoint member name is not canonical: {value!r}")
    return value


def _member_names(values: Mapping[str, Path]) -> dict[str, Path]:
    normalized = {_member_name(name): Path(path) for name, path in values.items()}
    names = set(normalized)
    for name in names:
        parents = PurePosixPath(name).parents
        if any(
            parent.as_posix() in names
            for parent in parents
            if parent != PurePosixPath(".")
        ):
            raise ValueError("Checkpoint member names overlap as file and directory")
    return normalized


def _destination_prefix(uri: str) -> str:
    parsed = urlparse(uri)
    relative = parsed.path.lstrip("/").rstrip("/")
    path = PurePosixPath(relative)
    invalid = (
        parsed.scheme != "s3"
        or not parsed.netloc
        or not relative
        or relative != path.as_posix()
        or "\\" in relative
        or any(part in {"", ".", ".."} for part in path.parts)
        or parsed.params
        or parsed.query
        or parsed.fragment
        or "@" in parsed.netloc
        or ":" in parsed.netloc
    )
    if invalid:
        raise ValueError("Checkpoint destination must be a scoped s3:// prefix")
    return uri.rstrip("/")


@contextmanager
def _regular_stream(path: Path) -> Iterator[BufferedReader]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"Checkpoint member is not a regular file: {path}") from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"Checkpoint member is not a regular file: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            yield stream
    finally:
        os.close(descriptor)


def _file_identity(path: Path) -> dict[str, int | str]:
    with _regular_stream(path) as stream:
        digest = hashlib.sha256()
        size = 0
        while chunk := stream.read(_READ_BYTES):
            digest.update(chunk)
            size += len(chunk)
        return {"bytes": size, "sha256": digest.hexdigest()}


def _precondition_failed(error: ClientError) -> bool:
    code = str(error.response.get("Error", {}).get("Code", ""))
    status_code = int(
        error.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0) or 0
    )
    return code in {
        "409",
        "412",
        "PreconditionFailed",
        "ConditionalRequestConflict",
    } or status_code in {409, 412}


def _upload_parts(
    storage: _Storage,
    path: Path,
    bucket: str,
    key: str,
    upload_id: str,
    part_bytes: int,
) -> tuple[list[dict[str, Any]], dict[str, int | str]]:
    parts: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    size = 0
    with _regular_stream(path) as stream:
        while payload := stream.read(part_bytes):
            digest.update(payload)
            size += len(payload)
            number = len(parts) + 1
            result = storage.s3.upload_part(
                Bucket=bucket,
                Key=key,
                UploadId=upload_id,
                PartNumber=number,
                Body=payload,
            )
            parts.append({"PartNumber": number, "ETag": result["ETag"]})
    return parts, {"bytes": size, "sha256": digest.hexdigest()}


def _multipart_create(
    storage: _Storage,
    path: Path,
    uri: str,
    expected: Mapping[str, int | str],
    *,
    part_bytes: int,
) -> None:
    parsed = urlparse(uri)
    bucket, key = parsed.netloc, parsed.path.lstrip("/")
    response = storage.s3.create_multipart_upload(Bucket=bucket, Key=key)
    upload_id = response["UploadId"]
    completed = False
    try:
        parts, actual = _upload_parts(storage, path, bucket, key, upload_id, part_bytes)
        if actual != dict(expected):
            raise ValueError("Checkpoint member changed during multipart upload")
        storage.s3.complete_multipart_upload(
            Bucket=bucket,
            Key=key,
            UploadId=upload_id,
            MultipartUpload={"Parts": parts},
            IfNoneMatch="*",
        )
        completed = True
    except ClientError as error:
        if not _precondition_failed(error):
            raise
    finally:
        if not completed:
            storage.s3.abort_multipart_upload(
                Bucket=bucket, Key=key, UploadId=upload_id
            )


def _conditional_create(
    storage: _Storage,
    path: Path,
    uri: str,
    *,
    expected: Mapping[str, int | str],
    part_bytes: int,
) -> None:
    size = int(expected["bytes"])
    if size <= part_bytes:
        with _regular_stream(path) as stream:
            payload = stream.read(part_bytes + 1)
        identity = {
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        if identity != dict(expected):
            raise ValueError("Checkpoint member changed before conditional creation")
        try:
            storage.put_bytes_conditional(payload, uri, if_none_match=True)
        except StoragePreconditionFailed:
            pass
        return
    _multipart_create(storage, path, uri, expected, part_bytes=part_bytes)


def _publish_member(
    name: str,
    path: Path,
    prefix: str,
    scratch: Path,
    storage_factory: Callable[[], _Storage],
    expected: Mapping[str, int | str],
    part_bytes: int,
) -> dict[str, Any]:
    if _file_identity(path) != dict(expected):
        raise ValueError(f"Checkpoint member changed before publication: {name}")
    storage = storage_factory()
    uri = f"{prefix}/{name}"
    _conditional_create(storage, path, uri, expected=expected, part_bytes=part_bytes)
    destination = scratch.joinpath(*PurePosixPath(name).parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    storage.download_file(uri, str(destination))
    if _file_identity(destination) != dict(expected):
        raise ValueError(f"Checkpoint provider readback differs: {name}")
    if _file_identity(path) != dict(expected):
        raise ValueError(f"Checkpoint member changed during publication: {name}")
    destination.unlink()
    return {**expected, "uri": uri, "provider_readback": True}


def _scratch_directory(parent: Path) -> tuple[Path, tuple[int, int]]:
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError("Checkpoint readback parent must be a trusted directory")
    scratch = parent / f".npa-checkpoint-readback-{uuid.uuid4().hex}"
    scratch.mkdir(mode=0o700)
    metadata = scratch.stat(follow_symlinks=False)
    return scratch, (metadata.st_dev, metadata.st_ino)


def _remove_scratch(path: Path, identity: tuple[int, int]) -> None:
    metadata = path.stat(follow_symlinks=False)
    if path.is_symlink() or (metadata.st_dev, metadata.st_ino) != identity:
        raise ValueError("Checkpoint readback directory identity changed")
    shutil.rmtree(path)


def _publication_inputs(
    files: Mapping[str, Path],
    destination_uri: str,
    expected: Mapping[str, Mapping[str, int | str]] | None,
    workers: int,
    part_bytes: int,
) -> tuple[str, dict[str, Path], dict[str, dict[str, int | str]]]:
    if not files or not 1 <= workers <= _MAX_WORKERS:
        raise ValueError("Checkpoint publication requires one to four workers")
    if not 5 * 1024 * 1024 <= part_bytes <= 5 * 1024 * 1024 * 1024:
        raise ValueError(
            "Checkpoint multipart parts must be between five MiB and five GiB"
        )
    prefix = _destination_prefix(destination_uri)
    normalized = _member_names(files)
    identities = {name: _file_identity(path) for name, path in normalized.items()}
    proven = (
        identities
        if expected is None
        else {name: dict(row) for name, row in expected.items()}
    )
    if set(proven) != set(normalized) or proven != identities:
        raise ValueError("Checkpoint expected identities differ from local files")
    if any(int(row["bytes"]) > part_bytes * 10_000 for row in identities.values()):
        raise ValueError("Checkpoint member exceeds the multipart part-count limit")
    return prefix, normalized, proven


def _publish_all(
    files: Mapping[str, Path],
    prefix: str,
    scratch: Path,
    factory: Callable[[], _Storage],
    expected: Mapping[str, Mapping[str, int | str]],
    workers: int,
    part_bytes: int,
) -> dict[str, dict[str, Any]]:
    def transfer(item: tuple[str, Path]) -> tuple[str, dict[str, Any]]:
        name, path = item
        result = _publish_member(
            name, path, prefix, scratch, factory, expected[name], part_bytes
        )
        return name, result

    with ThreadPoolExecutor(max_workers=workers) as pool:
        published = dict(pool.map(transfer, files.items()))
    return dict(sorted(published.items()))


def _require_readback_space(
    scratch: Path, expected: Mapping[str, Mapping[str, int | str]], workers: int
) -> None:
    largest = sorted((int(row["bytes"]) for row in expected.values()), reverse=True)[
        :workers
    ]
    if shutil.disk_usage(scratch).free < sum(largest):
        raise ValueError("Checkpoint provider readback scratch space is insufficient")


def _ready_scratch(
    parent: Path, expected: Mapping[str, Mapping[str, int | str]], workers: int
) -> tuple[Path, tuple[int, int]]:
    scratch, identity = _scratch_directory(parent)
    try:
        _require_readback_space(scratch, expected, workers)
    except Exception:
        _remove_scratch(scratch, identity)
        raise
    return scratch, identity


def publish_immutable_checkpoint_files(
    files: Mapping[str, Path],
    destination_uri: str,
    scratch_parent: Path,
    *,
    expected: Mapping[str, Mapping[str, int | str]] | None = None,
    storage_factory: Callable[[], _Storage] | None = None,
    workers: int = _MAX_WORKERS,
    part_bytes: int = _MULTIPART_BYTES,
) -> dict[str, dict[str, Any]]:
    """Conditionally publish checkpoint files and verify full provider readback.

    Args:
        files: Canonical relative object names mapped to local regular files.
        destination_uri: Scoped S3 prefix that will contain the named members.
        scratch_parent: Trusted local directory for owned temporary readbacks.
        expected: Optional exact byte and SHA-256 identities for every member.
        storage_factory: Optional factory used to create one storage client per worker.
        workers: Number of concurrent upload/readback workers, from one through four.
        part_bytes: Multipart chunk threshold and size, at least five MiB.
    Returns:
        Member identities, destination URIs, and provider-readback evidence.

    Raises:
        ValueError: Inputs, identities, destinations, or provider readbacks differ.
        ClientError: An object-storage operation fails outside an immutable conflict.
        OSError: A local source, scratch, or readback filesystem operation fails.
    """
    prefix, normalized, proven = _publication_inputs(
        files, destination_uri, expected, workers, part_bytes
    )
    factory = storage_factory or StorageClient.from_environment
    scratch, scratch_identity = _ready_scratch(Path(scratch_parent), proven, workers)
    try:
        return _publish_all(
            normalized, prefix, scratch, factory, proven, workers, part_bytes
        )
    finally:
        _remove_scratch(scratch, scratch_identity)
