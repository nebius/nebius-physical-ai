"""Checksum and hashed-stream helpers for transport integrity (from PR #363)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from npa.workbench.encord.schemas import ChecksumKind


@dataclass(frozen=True)
class StreamDigest:
    size: int
    sha256: str


def write_hashed_stream(chunks: Iterable[bytes], destination: Path) -> StreamDigest:
    """Write chunks to destination while computing size + sha256 in one pass."""

    digest = hashlib.sha256()
    size = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        for chunk in chunks:
            if not chunk:
                continue
            handle.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    return StreamDigest(size=size, sha256=digest.hexdigest())


def hash_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> StreamDigest:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while data := handle.read(chunk_size):
            digest.update(data)
            size += len(data)
    return StreamDigest(size=size, sha256=digest.hexdigest())


def etag_checksum(etag: str) -> tuple[str, ChecksumKind]:
    """A content digest from an S3 ETag, when it is one.

    A single-part ETag is the object's MD5; a multipart ETag (contains "-")
    is not a content digest and yields ("", "none").
    """

    value = etag.strip().strip('"').lower()
    if value and "-" not in value and len(value) == 32:
        return value, "md5"
    return "", "none"


def compare_checksums(
    expected: str,
    expected_kind: ChecksumKind,
    observed: str,
    observed_kind: ChecksumKind,
) -> bool | None:
    """True/False when the kinds are comparable, None when no comparison exists."""

    if not expected or not observed or "none" in {expected_kind, observed_kind}:
        return None
    if expected_kind != observed_kind:
        return None
    return expected.strip().strip('"').lower() == observed.strip().strip('"').lower()
