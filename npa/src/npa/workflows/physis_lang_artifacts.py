"""Exchange Physis-Lang experiment artifacts with complete SHA-256 verification."""

from __future__ import annotations

import hashlib
import json
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
    return {
        p.relative_to(root).as_posix(): file_hash(p)
        for p in sorted(root.rglob("*"))
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


def publish(root: Path, destination: str) -> None:
    """Seal a completed stage and verify remote publication by full readback.

    Args:
        root: Completed stage directory.
        destination: New local directory or run-scoped S3 prefix.
    Returns:
        None.
    Raises:
        ValueError: Published bytes fail verification.
        OSError: Publication fails or a local destination exists.
    """
    write_json(root / "checksums.json", _hashes(root))
    if not destination.startswith("s3://"):
        shutil.copytree(root, destination)
        return
    from npa.clients.storage import StorageClient

    StorageClient.from_environment().upload_directory(
        str(root), destination, require_empty=True
    )
    with tempfile.TemporaryDirectory(prefix="physis-readback-") as temporary:
        materialize(destination, Path(temporary))
