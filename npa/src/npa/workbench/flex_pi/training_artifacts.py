"""Publish and verify the complete generated Flex-Pi checkpoint byte set."""

import hashlib
import json
from pathlib import Path
import tempfile

from npa.clients.storage import StorageClient, safe_s3_download_target
from npa.workbench.flex_pi.runtime import FlexPiError


def sha256_file(path):
    """Return a streaming SHA-256 digest without loading a checkpoint into memory.

    Args:
        path: Local file to verify.
    Returns:
        Hexadecimal SHA-256 digest.
    Raises:
        OSError: The file cannot be read.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def publish_checkpoint(
    directory: Path, destination: str, storage: StorageClient
) -> dict:
    """Upload every training-state file, retaining hashes for read-after-write proof.

    Args:
        directory: Newly generated complete Accelerate state directory.
        destination: Authorized run-scoped checkpoint prefix.
        storage: Configured owner storage client.
    Returns:
        Manifest listing every state file, size and content hash.
    Raises:
        FlexPiError: The complete training cursor or state is absent.
    """
    _require_complete_state(directory)
    files = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(directory).as_posix()
        row = {
            "path": relative,
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        storage.upload_file(str(path), destination.rstrip("/") + "/" + relative)
        files.append(row)
    return {"schema": "npa.flex_pi.training_checkpoint.v1", "files": files}


def _require_complete_state(directory):
    required = {
        "trainer_state.json",
        "optimizer.bin",
        "scheduler.bin",
        "dataset_stats.json",
        *(f"random_states_{rank}.pkl" for rank in range(4)),
    }
    missing = required.difference(
        path.name for path in directory.iterdir() if path.is_file()
    )
    if missing or not any(
        (directory / name).is_file()
        for name in ("model.safetensors", "pytorch_model.bin")
    ):
        raise FlexPiError(
            "checkpoint lacks complete model, optimizer, scheduler, cursor or four-rank RNG state"
        )


def restore_checkpoint(
    manifest: dict, source: str, directory: Path, storage: StorageClient
) -> None:
    """Read back each expected state file and require exact byte identity.

    Args:
        manifest: The in-memory manifest from this run's checkpoint publication.
        source: Authorized checkpoint prefix.
        directory: Empty private destination for the fresh resume process.
        storage: Configured owner storage client.
    Returns:
        None; writes only verified paths below directory.
    Raises:
        FlexPiError: Any file's size or hash differs from the generated checkpoint.
    """
    for row in manifest["files"]:
        path = safe_s3_download_target(directory, row["path"], "")
        path.parent.mkdir(parents=True, exist_ok=True)
        storage.download_file(source.rstrip("/") + "/" + row["path"], str(path))
        if path.stat().st_size != row["size"] or sha256_file(path) != row["sha256"]:
            raise FlexPiError("checkpoint read-after-write content verification failed")


def publish_json(payload, path: Path, destination: str, storage: StorageClient):
    """Write, publish and verify one finite JSON artifact by exact byte readback.

    Args:
        payload: JSON-serializable evidence.
        path: Private local staging file.
        destination: Authorized storage URI.
        storage: Owner storage client.
    Returns:
        None.
    Raises:
        ValueError: The payload contains a nonfinite number.
        FlexPiError: Published bytes differ from the staged JSON artifact.
    """
    path.write_text(json.dumps(payload, indent=2, allow_nan=False))
    storage.upload_file(str(path), destination)
    with tempfile.TemporaryDirectory(
        prefix="json-readback-", dir=path.parent
    ) as directory:
        readback = Path(directory) / "artifact.json"
        storage.download_file(destination, str(readback))
        if path.read_bytes() != readback.read_bytes():
            raise FlexPiError(
                "JSON artifact read-after-write content verification failed"
            )
