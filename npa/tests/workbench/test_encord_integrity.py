"""Content integrity helpers: ETag-derived checksums and hashed streams."""

from __future__ import annotations

from pathlib import Path

from npa.workbench.encord.integrity import (
    compare_checksums,
    etag_checksum,
    hash_file,
    write_hashed_stream,
)


def test_etag_checksum_and_compare() -> None:
    assert etag_checksum('"a" ') == ("", "none")
    md5 = "d41d8cd98f00b204e9800998ecf8427e"
    assert etag_checksum(f'"{md5}"') == (md5, "md5")
    assert etag_checksum(f'"{md5}-3"') == ("", "none")  # multipart is not a digest
    assert compare_checksums(md5, "md5", md5.upper(), "md5") is True
    assert compare_checksums(md5, "md5", "deadbeef" * 4, "md5") is False
    assert compare_checksums(md5, "md5", "abc", "sha256") is None
    assert compare_checksums("", "none", md5, "md5") is None


def test_write_hashed_stream_digest(tmp_path: Path) -> None:
    import hashlib

    dest = tmp_path / "out.bin"
    digest = write_hashed_stream([b"abc", b"", b"def"], dest)
    assert dest.read_bytes() == b"abcdef"
    assert digest.size == 6
    assert digest.sha256 == hashlib.sha256(b"abcdef").hexdigest()
    assert hash_file(dest) == digest

