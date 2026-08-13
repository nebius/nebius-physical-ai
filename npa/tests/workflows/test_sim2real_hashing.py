"""Bounded-memory SHA-256 coverage for Sim2Real checkpoints."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from npa.workflows.sim2real.hashing import sha256_file


@pytest.mark.parametrize(
    ("payload", "chunk_size"),
    [
        (b"", 8),
        (b"small checkpoint", 1024),
        (bytes(range(251)) * 19, 37),
    ],
)
def test_sha256_file_matches_reference_for_empty_small_and_multichunk_files(
    tmp_path: Path, payload: bytes, chunk_size: int
) -> None:
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(payload)

    assert (
        sha256_file(path, chunk_size=chunk_size) == hashlib.sha256(payload).hexdigest()
    )


def test_sha256_file_never_calls_path_read_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "checkpoint.pt"
    payload = b"bounded-memory" * 1024
    path.write_bytes(payload)

    def forbidden_read_bytes(_path: Path) -> bytes:
        raise AssertionError(
            "Path.read_bytes() must not be used for checkpoint hashing"
        )

    monkeypatch.setattr(Path, "read_bytes", forbidden_read_bytes)
    assert sha256_file(path, chunk_size=31) == hashlib.sha256(payload).hexdigest()


def test_sha256_file_rejects_nonpositive_chunk_size(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"x")
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        sha256_file(path, chunk_size=0)
