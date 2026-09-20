"""Stream Arena artifact hashes on every supported NPA Python version."""

from __future__ import annotations

import hashlib
from pathlib import Path


def file_sha256(path: Path) -> str:
    """Hash exact file bytes without loading a replay or video into memory.

    Args:
        path: Artifact to read.
    Returns:
        Lowercase SHA-256 hexadecimal digest.
    Raises:
        OSError: The artifact cannot be read.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
