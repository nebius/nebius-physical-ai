"""Native Arrow S3 configuration and verified Train artifact copies."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from urllib.parse import urlsplit


EXPORT_FILES = frozenset({"state.pt", "metrics.json", "metrics.rrd", "result.json", "SHA256SUMS"})


def storage(uri: str):
    """Resolve an explicit unsigned destination using each process's AWS environment."""
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


def publish(filesystem, destination: str, directory: Path) -> None:
    """Write exports with the checksum manifest last and verify every remote byte."""
    from pyarrow import fs

    if {path.name for path in directory.iterdir()} != EXPORT_FILES:
        raise ValueError("Export contains unexpected or missing files")
    if any(path.is_symlink() or not path.is_file() for path in directory.iterdir()):
        raise ValueError("Only regular exported artifact files may be published")
    paths = sorted(directory.iterdir(), key=lambda p: (p.name == "SHA256SUMS", p.name))
    for path in paths:
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


def download(filesystem, source: str, directory: Path) -> None:
    """Retrieve only manifest-listed export basenames into a fresh local directory."""
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
