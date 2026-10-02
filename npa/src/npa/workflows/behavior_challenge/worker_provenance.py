"""Publish worker diagnostics without uploading staged model archives."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat


DIAGNOSTIC_SUFFIXES = frozenset(
    {".json", ".log", ".py", ".sh", ".txt", ".yaml", ".yml"}
)
PART_BYTES = 64 * 1024 * 1024


def publish_worker_provenance(storage, workspace: Path, receipt_uri: str, put) -> None:
    """Preserve diagnostics and describe excluded inputs in an immutable index.

    Small diagnostics retain their original object names. Large diagnostics use
    bounded parts whose hashes and order are recorded in the index. Staged
    checkpoint archives, other binary inputs, directories, and links are never
    read as diagnostics; their contents belong to their input admission records.

    Args:
        storage: Campaign object-storage client.
        workspace: Finished worker's local workspace.
        receipt_uri: Worker receipt URI defining its evidence namespace.
        put: Immutable upload callback that verifies complete byte readback.
    Returns:
        None.
    Raises:
        ValueError: A diagnostic changes or stored bytes conflict.
        OSError: The workspace cannot be read.
        Exception: The storage callback fails.
    """
    stem = receipt_uri.removesuffix(".json")
    index = _publish_diagnostics(storage, workspace, stem + "/provenance", put)
    payload = (json.dumps(index, sort_keys=True, indent=2) + "\n").encode()
    put(storage, payload, stem + "/provenance-index.json")


def _publish_diagnostics(storage, workspace, prefix, put):
    files = {}
    excluded = []
    for path in sorted(workspace.iterdir()):
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode):
            continue
        if path.suffix not in DIAGNOSTIC_SUFFIXES:
            excluded.append(
                {"name": path.name, "bytes": before.st_size, "reason": "not_diagnostic"}
            )
            continue
        files[path.name] = _publish_file(storage, path, prefix, before, put)
    return {
        "schema": "npa.behavior.worker-provenance.v1",
        "files": files,
        "excluded_inputs": excluded,
        "excluded_contents_verified": False,
    }


def _publish_file(storage, path, prefix, before, put):
    digest = hashlib.sha256()
    parts = []
    total = 0
    multipart = before.st_size > PART_BYTES
    with path.open("rb") as stream:
        for _ in range(max(1, (before.st_size + PART_BYTES - 1) // PART_BYTES)):
            payload = stream.read(min(PART_BYTES, before.st_size - total))
            uri = f"{prefix}/{path.name}"
            if multipart:
                uri = f"{prefix}/parts/{path.name}/{len(parts):06d}"
            put(storage, payload, uri)
            parts.append(
                {
                    "uri": uri,
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
            digest.update(payload)
            total += len(payload)
            if not multipart:
                break
        after = path.stat()
        if (
            total != before.st_size
            or stream.read(1)
            or (before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_ino, after.st_size, after.st_mtime_ns)
        ):
            raise ValueError("Worker diagnostic changed during publication")
    return {"bytes": total, "sha256": digest.hexdigest(), "parts": parts}
