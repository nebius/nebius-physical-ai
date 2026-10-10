"""Exchange Physis-Lang experiment artifacts with complete SHA-256 verification."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


def write_json(path: Path, value: object) -> None:
    """Write deterministic finite JSON.

    Args:
        path: Destination file.
        value: JSON-compatible payload.
    Returns:
        None.
    Raises:
        ValueError: Payload contains nonfinite values.
        OSError: Writing fails.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def file_hash(path: Path) -> str:
    """Hash an artifact without retaining its complete bytes in memory.

    Args:
        path: Existing artifact.
    Returns:
        SHA-256 hex digest.
    Raises:
        OSError: Reading fails.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hashes(root: Path) -> dict:
    paths = sorted(root.rglob("*"))
    if root.is_symlink() or any(path.is_symlink() for path in paths):
        raise ValueError("Physis artifact trees cannot contain symbolic links")
    return {
        p.relative_to(root).as_posix(): file_hash(p)
        for p in paths
        if p.is_file() and p != root / "checksums.json"
    }


def materialize(source: str, destination: Path) -> Path:
    """Read and verify every file in a sealed local or S3 stage.

    Args:
        source: Stage directory or S3 prefix.
        destination: Empty download directory for S3 inputs.
    Returns:
        Verified local stage directory.
    Raises:
        ValueError: Missing, additional, or changed artifact bytes.
        OSError: Reading fails.
    """
    root = Path(source)
    if source.startswith("s3://"):
        from npa.clients.storage import StorageClient

        StorageClient.from_environment().download_directory(source, str(destination))
        root = destination
    expected = json.loads((root / "checksums.json").read_text())
    if not expected or expected != _hashes(root):
        raise ValueError("Physis stage checksum manifest does not match its files")
    return root


def _create_object(client, payload: bytes, uri: str) -> None:
    from npa.clients.storage import StoragePreconditionFailed

    try:
        client.put_bytes_conditional(payload, uri, if_none_match=True)
    except StoragePreconditionFailed:
        existing = client.read_bytes_with_etag(uri)
        if existing is None or existing[0] != payload:
            raise ValueError(
                "Physis publication conflicts with existing bytes"
            ) from None


def _compatible_existing(root: Path, existing: Path) -> None:
    expected = _hashes(root)
    observed = _hashes(existing)
    if any(expected.get(name) != digest for name, digest in observed.items()):
        raise ValueError("Physis publication conflicts with existing files")
    manifest = existing / "checksums.json"
    if (
        manifest.exists()
        and manifest.read_bytes() != (root / manifest.name).read_bytes()
    ):
        raise ValueError("Physis publication conflicts with an existing manifest")


def _publish_s3(root: Path, destination: str) -> None:
    from npa.clients.storage import StorageClient

    client = StorageClient.from_environment()
    with tempfile.TemporaryDirectory(prefix="physis-readback-") as temporary:
        existing = Path(temporary) / "existing"
        client.download_directory(destination, str(existing))
        _compatible_existing(root, existing)
        # Reserve the complete expected manifest first. Competing publishers
        # cannot mix different generations, even when they both see an empty prefix.
        names = ["checksums.json", *_hashes(root)]
        for name in names:
            _create_object(
                client, (root / name).read_bytes(), destination.rstrip("/") + "/" + name
            )
        materialize(destination, Path(temporary) / "verified")


def _create_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent) as staging:
        with source.open("rb") as stream:
            shutil.copyfileobj(stream, staging)
        staging.flush()
        try:
            os.link(staging.name, target)
        except FileExistsError:
            if file_hash(target) != file_hash(source):
                raise ValueError(
                    "Physis publication conflicts with existing bytes"
                ) from None


def seal(root: Path) -> None:
    """Record complete artifact hashes before attempting any publication.

    Args:
        root: Finished stage output directory.
    Returns:
        None.
    Raises:
        OSError: Reading artifacts or writing the manifest fails.
    """
    write_json(root / "checksums.json", _hashes(root))


def publish(root: Path, destination: str) -> None:
    """Seal and publish artifacts, safely completing identical partial uploads.

    Args:
        root: Completed stage directory.
        destination: Local directory or run-scoped S3 prefix.
    Returns:
        None.
    Raises:
        ValueError: Existing files conflict or published bytes fail verification.
        OSError: Publication fails.
    """
    seal(root)
    if not destination.startswith("s3://"):
        target = Path(destination)
        _compatible_existing(root, target)
        target.mkdir(parents=True, exist_ok=True)
        for name in ["checksums.json", *_hashes(root)]:
            _create_file(root / name, target / name)
        materialize(destination, target)
        return
    _publish_s3(root, destination)
