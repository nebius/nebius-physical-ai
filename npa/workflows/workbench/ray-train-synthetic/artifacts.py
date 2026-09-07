"""Native Arrow S3 configuration and verified Train artifact copies."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from urllib.parse import urlsplit


EXPORT_FILES = frozenset({"state.pt", "metrics.json", "metrics.rrd", "result.json", "SHA256SUMS"})


def file_sha256(path: Path) -> str:
    """Hash artifact bytes with bounded memory on supported Python versions.

    Args:
        path: Artifact to read.
    Returns:
        Hexadecimal SHA-256 digest of the complete file.
    Raises:
        OSError: The artifact cannot be opened or read.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def storage(uri: str):
    """Resolve an unsigned destination using each process's AWS environment.

    Args:
        uri: Authorized run-scoped S3 destination.
    Returns:
        Arrow filesystem and bucket-prefixed object path.
    Raises:
        ValueError: Destination, HTTPS endpoint, or credential environment is invalid.
        OSError: Arrow cannot initialize the selected storage backend.
    """
    target = urlsplit(uri)
    if (target.scheme != "s3" or not target.netloc or target.username or target.password
            or target.query or target.fragment or not target.path.strip("/")
            or any(part in ("", ".", "..") for part in target.path.strip("/").split("/"))):
        raise ValueError("storage-path must be a run-scoped unsigned s3:// bucket/prefix")
    endpoint = urlsplit(os.environ.get("AWS_ENDPOINT_URL_S3", ""))
    if (endpoint.scheme != "https" or not endpoint.netloc or endpoint.username or endpoint.password
            or endpoint.query or endpoint.fragment or endpoint.path not in ("", "/")):
        raise ValueError("AWS_ENDPOINT_URL_S3 must be the verified unsigned HTTPS endpoint")
    for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION"):
        if not os.environ.get(name):
            raise ValueError(f"{name} is required for the verified workload bucket")
    from pyarrow import fs

    # Do not pass keys to the constructor: that serializes them into the filesystem.
    filesystem = fs.S3FileSystem(endpoint_override=endpoint.netloc, scheme="https",
                                 region=os.environ["AWS_DEFAULT_REGION"])
    return filesystem, target.netloc + target.path.rstrip("/")


def _publish_file(filesystem, destination, path):
    """Preserve existing remote bytes and verify a newly written object immediately."""
    from pyarrow import fs

    if path.is_symlink() or not path.is_file():
        raise ValueError("Only regular exported artifact files may be published")
    remote = destination + "/" + path.name
    content = path.read_bytes()
    if filesystem.get_file_info(remote).type != fs.FileType.NotFound:
        with filesystem.open_input_file(remote) as stream:
            if stream.read() != content:
                raise ValueError("Existing export differs; preserve it and use a fresh run name")
    else:
        with filesystem.open_output_stream(remote) as stream:
            stream.write(content)
    with filesystem.open_input_file(remote) as stream:
        if hashlib.sha256(stream.read()).digest() != hashlib.sha256(content).digest():
            raise ValueError("S3 export read-after-write verification failed")


def publish(filesystem, destination: str, directory: Path) -> None:
    """Publish verified exports with their checksum manifest last.

    Args:
        filesystem: Authorized Arrow filesystem for the workload.
        destination: Bucket-prefixed export path.
        directory: Local directory containing exactly the exported artifacts.
    Returns:
        None.
    Raises:
        ValueError: Files are invalid, existing bytes differ, or readback differs.
        OSError: Local reads or remote transfers fail.
    """
    if {path.name for path in directory.iterdir()} != EXPORT_FILES:
        raise ValueError("Export contains unexpected or missing files")
    if any(path.is_symlink() or not path.is_file() for path in directory.iterdir()):
        raise ValueError("Only regular exported artifact files may be published")
    paths = sorted(directory.iterdir(), key=lambda path: (path.name == "SHA256SUMS", path.name))
    for path in paths:
        _publish_file(filesystem, destination, path)


def download(filesystem, source: str, directory: Path) -> None:
    """Retrieve manifest-listed export basenames into a fresh private directory.

    Args:
        filesystem: Authorized Arrow filesystem for the workload.
        source: Bucket-prefixed path of the published export.
        directory: Fresh local destination.
    Returns:
        None.
    Raises:
        ValueError: Manifest names or downloaded hashes are invalid.
        OSError: Private directory creation or artifact transfer fails.
    """
    with filesystem.open_input_file(source + "/SHA256SUMS") as stream:
        manifest = stream.read()
    entries = [line.split("  ", 1) for line in manifest.decode().splitlines()]
    expected = EXPORT_FILES - {"SHA256SUMS"}
    if len(entries) != len(expected) or {name for _, name in entries} != expected:
        raise ValueError("Remote checksum manifest does not cover the expected export")
    directory.mkdir(parents=True, exist_ok=False, mode=0o700)
    for digest, name in entries:
        with filesystem.open_input_file(source + "/" + name) as stream:
            content = stream.read()
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError("Downloaded export hash differs")
        (directory / name).write_bytes(content)
        (directory / name).chmod(0o600)
    (directory / "SHA256SUMS").write_bytes(manifest)
    (directory / "SHA256SUMS").chmod(0o600)
