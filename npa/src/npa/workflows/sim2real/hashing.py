"""Bounded-memory hashing helpers for Sim2Real artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path


DEFAULT_HASH_CHUNK_SIZE = 1024 * 1024


def sha256_file(path: Path, *, chunk_size: int = DEFAULT_HASH_CHUNK_SIZE) -> str:
    """Return a file's SHA-256 digest without loading the file into memory."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
