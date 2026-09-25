"""Exchange navigation bundles through contained local staging and verified S3 files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from urllib.parse import urlparse

from npa.clients.storage import StorageClient


def file_sha256(path: Path) -> str:
    """Hash a file without interpreting checkpoint contents.

    Args:
        path: File to read.
    Returns:
        SHA-256 hex digest.
    Raises:
        OSError: File cannot be read.
    """
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value) -> None:
    """Write deterministic finite evidence.

    Args:
        path: Destination file.
        value: JSON-compatible evidence.
    Returns:
        None.
    Raises:
        ValueError: Evidence contains nonfinite numbers.
        OSError: File cannot be written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def _files(root):
    if root.is_symlink():
        raise ValueError("artifact symlinks are forbidden")
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("artifact symlinks and escaping paths are forbidden")
        if path.is_file() and path != root / "checksums.json":
            files[path.relative_to(root).as_posix()] = file_sha256(path)
    if not files:
        raise ValueError("artifact bundle is empty")
    return files


def _validate_uri(uri):
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError("S3 artifacts require a bucket and run-scoped prefix")
    if parsed.query or parsed.fragment or ".." in parsed.path.split("/"):
        raise ValueError("S3 artifact prefix contains unsupported query or traversal")


def materialize(source: str, target: Path, *, sealed: bool = True) -> Path:
    """Stage a bundle, rejecting symlinks and verifying all sealed bytes.

    Args:
        source: Local directory or S3 prefix.
        target: Fresh private staging directory.
        sealed: Require checksums from a completed upstream stage.
    Returns:
        Validated bundle directory.
    Raises:
        ValueError: URI, paths or checksum manifest are invalid.
        OSError: Required input is absent or unreadable.
    """
    if not source:
        raise ValueError(
            "operator input bundle is required: recipe.json and self-contained scene USDZ"
        )
    if source.startswith("s3:"):
        _validate_uri(source)
        storage = StorageClient.from_environment()
        if sealed:
            from npa.workflows.navigation.publication import materialize_attempt

            materialize_attempt(storage, source, target)
        else:
            storage.download_directory(source, str(target))
    else:
        if "://" in source:
            raise ValueError("only local directories and S3 prefixes are supported")
        _files(Path(source))
        shutil.copytree(source, target)
    hashes = _files(target)
    if sealed and json.loads((target / "checksums.json").read_text()) != hashes:
        raise ValueError("artifact checksum manifest mismatch")
    return target


def publish(root: Path, destination: str) -> None:
    """Seal an immutable stage attempt and verify readback against local evidence.

    Args:
        root: Completed private stage directory.
        destination: New local directory or never-before-claimed run-scoped S3 stage prefix.
    Returns:
        None.
    Raises:
        ValueError: Paths or remote readback fail integrity checks.
        OSError: Publication fails or the local destination already exists.
    """
    expected = _files(root)
    write_json(root / "checksums.json", expected)
    if destination.startswith("s3:"):
        _validate_uri(destination)
        from npa.workflows.navigation.publication import publish_attempt

        publish_attempt(StorageClient.from_environment(), root, destination, expected)
    else:
        if not destination or "://" in destination:
            raise ValueError("output must be a local directory or S3 prefix")
        shutil.copytree(root, destination)
        if _files(Path(destination)) != expected:
            raise ValueError(
                "local artifact publication differs from original manifest"
            )
