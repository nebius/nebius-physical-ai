"""Exchange checksum-bound simulation, dataset, checkpoint, and evaluation artifacts."""

import hashlib
import json
import shutil
from pathlib import Path

from npa.clients.storage import StorageClient


def write_json(path: Path, payload: object) -> None:
    """Save deterministic finite JSON.

    Args:
        path: Destination file.
        payload: Serializable evidence.
    Returns:
        None.
    Raises:
        ValueError: A value is nonfinite.
        OSError: Writing fails.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def file_hash(path: Path) -> str:
    """Hash file bytes without loading a checkpoint into memory.

    Args:
        path: File to hash.
    Returns:
        SHA-256 hexadecimal digest.
    Raises:
        OSError: Reading fails.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_hashes(root: Path) -> dict[str, str]:
    """Bind every regular artifact except the manifest itself.

    Args:
        root: Artifact directory.
    Returns:
        Relative paths and SHA-256 digests.
    Raises:
        ValueError: A symlink occurs in the artifact tree.
        OSError: Reading fails.
    """
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Artifact symlinks are not supported")
        if path.is_file() and path != root / "checksums.json":
            result[path.relative_to(root).as_posix()] = file_hash(path)
    return result


def materialize(source: str, target: Path) -> Path:
    """Read a stage and verify its complete checksum manifest.

    Args:
        source: Local directory or S3 prefix.
        target: Scratch directory for downloaded objects.
    Returns:
        Verified directory.
    Raises:
        ValueError: A manifest is empty or bytes differ.
        OSError: Reading fails.
    """
    root = Path(source)
    if source.startswith("s3://"):
        StorageClient.from_environment().download_directory(source, str(target))
        root = target
    expected = json.loads((root / "checksums.json").read_text())
    if not expected or expected != tree_hashes(root):
        raise ValueError("Artifact checksum mismatch or empty manifest")
    return root


def publish(root: Path, destination: str) -> None:
    """Publish artifacts with readback, writing the completion manifest last.

    Args:
        root: Completed stage directory.
        destination: New local directory or fresh run-scoped S3 prefix.
    Returns:
        None.
    Raises:
        ValueError: Published bytes differ.
        OSError: Publication fails or local destination exists.
    """
    hashes = tree_hashes(root)
    if not hashes:
        raise ValueError("Cannot publish an empty stage")
    write_json(root / "checksums.json", hashes)
    if not destination.startswith("s3://"):
        shutil.copytree(root, destination)
        return
    client = StorageClient.from_environment()
    for name, expected in hashes.items():
        uri = destination.rstrip("/") + "/" + name
        client.upload_file(str(root / name), uri)
        actual = client.read_bytes_with_etag(uri)
        if actual is None or hashlib.sha256(actual[0]).hexdigest() != expected:
            raise ValueError("Published artifact failed readback")
    client.upload_file(str(root / "checksums.json"), destination.rstrip("/") + "/checksums.json")
