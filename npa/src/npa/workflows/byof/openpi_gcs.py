"""Anonymous, checksum-verified GCS reads for OpenPI dataset preparation.

This internal helper supports only the three read operations used by the DROID
runner. Its deliberately small argv contract also permits an explicit legacy
``--gsutil`` override. No Google credentials or cloud writes are used.
"""

from __future__ import annotations

import base64
from contextlib import contextmanager
import os
from pathlib import Path
import re
import secrets
import stat
import sys
from typing import BinaryIO, Iterator, Sequence
from urllib.parse import urlsplit


class GCSReadError(RuntimeError):
    """A public download did not meet its path or integrity contract."""


def _object_name(value: str) -> str:
    if (not value or value.startswith("/") or "\\" in value
            or any(ord(c) < 32 or ord(c) == 127 for c in value)
            or any(part in {"", ".", ".."} for part in value.split("/"))):
        raise GCSReadError("invalid GCS object path")
    return value


def _source(uri: str) -> tuple[str, str]:
    parsed = urlsplit(uri)
    if (parsed.scheme != "gs" or parsed.query or parsed.fragment
            or not re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", parsed.netloc)
            or any(c in uri for c in "*?[]\\")
            or any(ord(c) < 32 or ord(c) == 127 for c in uri)):
        raise GCSReadError("expected an exact public GCS object or prefix")
    return parsed.netloc, _object_name(parsed.path.removeprefix("/"))


@contextmanager
def _directory(path: Path) -> Iterator[int]:
    """Keep the destination directory open without following any symlinks."""
    if ".." in path.parts:
        raise GCSReadError("parent traversal in download destination")
    path = path.absolute()
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        for part in path.parts[1:]:
            try:
                os.mkdir(part, mode=0o700, dir_fd=descriptor)
            except FileExistsError:
                pass
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                            | os.O_CLOEXEC, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        yield descriptor
    finally:
        os.close(descriptor)


def _crc32c(handle: BinaryIO) -> tuple[int, str]:
    import google_crc32c

    handle.seek(0)
    digest = google_crc32c.Checksum()
    size = 0
    while chunk := handle.read(1024 * 1024):
        size += len(chunk)
        digest.update(chunk)
    return size, base64.b64encode(digest.digest()).decode("ascii")


def _metadata(blob) -> tuple[int, str, int]:
    try:
        size = int(blob.size)
        generation = int(blob.generation)
        checksum = blob.crc32c
        decoded = base64.b64decode(checksum, validate=True)
    except (TypeError, ValueError) as exc:
        raise GCSReadError("GCS object has invalid integrity metadata") from exc
    if size < 0 or generation < 1 or len(decoded) != 4:
        raise GCSReadError("GCS object has invalid integrity metadata")
    return size, checksum, generation


def _reuse(descriptor: int, name: str, expected: tuple[int, str]) -> bool:
    try:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
                     | os.O_CLOEXEC, dir_fd=descriptor)
    except FileNotFoundError:
        return False
    with os.fdopen(fd, "rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise GCSReadError("download destination must be a regular file with one link")
        if before.st_size != expected[0]:
            return False
        actual = _crc32c(handle)
        after = os.fstat(handle.fileno())
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise GCSReadError("download destination changed during verification")
        return actual == expected


def download(blob, destination: Path) -> bool:
    """Atomically materialize exactly the listed generation; return cache reuse."""
    size, checksum, generation = _metadata(blob)
    if destination.name in {"", ".", ".."}:
        raise GCSReadError("invalid download destination")
    with _directory(destination.parent) as descriptor:
        if _reuse(descriptor, destination.name, (size, checksum)):
            return True
        temporary = ".npa-gcs-" + secrets.token_hex(16)
        fd = os.open(temporary, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
                     | os.O_CLOEXEC, 0o600, dir_fd=descriptor)
        try:
            with os.fdopen(fd, "w+b") as handle:
                blob.download_to_file(handle, raw_download=True, checksum="crc32c",
                                      if_generation_match=generation)
                handle.flush()
                if _crc32c(handle) != (size, checksum):
                    raise GCSReadError("GCS object checksum or byte count differs")
                os.fsync(handle.fileno())
            os.replace(temporary, destination.name,
                       src_dir_fd=descriptor, dst_dir_fd=descriptor)
            os.fsync(descriptor)
        finally:
            try:
                os.unlink(temporary, dir_fd=descriptor)
            except FileNotFoundError:
                pass
    return False


def _objects(client, bucket: str, prefix: str):
    prefix = prefix.rstrip("/") + "/"
    found = False
    for blob in client.list_blobs(bucket, prefix=prefix):
        # GCS directory markers contain no dataset bytes.
        if blob.name.endswith("/") and blob.size == 0:
            continue
        name = _object_name(blob.name)
        if not name.startswith(prefix):
            raise GCSReadError("GCS listing escaped the requested prefix")
        relative = _object_name(name[len(prefix):])
        _metadata(blob)
        found = True
        yield blob, relative
    if not found:
        raise GCSReadError("public GCS prefix contains no objects")


def run(arguments: Sequence[str], *, client=None) -> None:
    """Execute the DROID runner's read-only subset of the gsutil argv contract."""
    args = list(arguments)
    if args[:1] == ["-m"]:
        args.pop(0)
    if args[:3] == ["ls", "-l", "-r"] and len(args) == 4 and args[3].endswith("/**"):
        operation, uri, destination = "list", args[3][:-3], None
    elif args[:3] == ["rsync", "-r", "-c"] and len(args) == 5:
        operation, uri, destination = "sync", args[3], Path(args[4])
    elif args[:1] == ["cp"] and len(args) == 3:
        operation, uri, destination = "copy", args[1], Path(args[2])
    else:
        raise GCSReadError("unsupported public GCS read operation")
    bucket, prefix = _source(uri)
    if destination is not None and "://" in args[-1]:
        raise GCSReadError("GCS download destination must be local")
    owned_client = client is None
    if owned_client:
        from google.cloud import storage

        client = storage.Client.create_anonymous_client()
    try:
        if operation == "copy":
            blob = client.bucket(bucket).get_blob(prefix)
            if blob is None:
                raise GCSReadError("public GCS object was not found")
            download(blob, destination)
            return
        for blob, relative in _objects(client, bucket, prefix):
            if operation == "list":
                if blob.updated is None:
                    raise GCSReadError("GCS object has no update timestamp")
                print(f"{blob.size}  {blob.updated.isoformat()}  gs://{bucket}/{blob.name}")
            else:
                download(blob, destination / relative)
    finally:
        if owned_client:
            client.close()


def main(arguments: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if arguments is None else arguments)
    if args == ["--version"]:
        from importlib.metadata import version

        print(f"npa-openpi-gcs google-cloud-storage={version('google-cloud-storage')}")
        return 0
    try:
        run(args)
    except Exception:
        # Public source names and SDK response URLs stay out of ordinary logs.
        # The exception terminates this read; it cannot establish success.
        print("OpenPI public GCS download failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
