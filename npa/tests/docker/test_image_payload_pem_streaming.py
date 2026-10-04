"""Exercise parse-valid PEM whitespace and bounded streaming in real scanners."""

from __future__ import annotations

import io
import subprocess
import sys
import tracemalloc
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from test_image_payload_credentials import (
    SCANNERS_USING_SHARED_RULES,
    _MODULE_PATH,
    _single_layer_archive,
    credentials,
)


@pytest.fixture
def disposable_pem() -> bytes:
    """Generate an untrusted disposable key entirely in memory."""
    return Ed25519PrivateKey.generate().private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )


def _with_gap(pem: bytes, gap: bytes) -> bytes:
    header, body = pem.split(b"\n", 1)
    return header + gap + body


@pytest.mark.parametrize(
    "gap",
    [
        b"\n" * 9,
        b" " * 9 + b"\n",
        b"\v" * 9 + b"\n",
        b"\f" * 9 + b"\n",
        b"\n" * 1048585,
    ],
    ids=["newlines", "spaces", "vertical-tabs", "form-feeds", "over-one-chunk"],
)
def test_parse_valid_pem_after_long_whitespace(
    disposable_pem: bytes, gap: bytes
) -> None:
    payload = _with_gap(disposable_pem, gap)
    original = serialization.load_pem_private_key(disposable_pem, password=None)
    parsed = serialization.load_pem_private_key(payload, password=None)
    assert (
        parsed.public_key().public_bytes_raw()
        == original.public_key().public_bytes_raw()
    )
    assert credentials.content_credential(io.BytesIO(payload)) == "private_key_content"


@pytest.mark.parametrize("scanner", SCANNERS_USING_SHARED_RULES)
@pytest.mark.parametrize("member", ["opt/npa-src/config.bin", "opt/vendor/config.bin"])
@pytest.mark.parametrize("gap_size", [9, 1048585])
def test_native_scanners_reject_pem_after_long_whitespace(
    tmp_path: Path, disposable_pem: bytes, scanner: str, member: str, gap_size: int
) -> None:
    payload = _with_gap(disposable_pem, b"\n" * gap_size)
    serialization.load_pem_private_key(payload, password=None)
    archive = _single_layer_archive(tmp_path / "image.tar", member, payload)
    result = subprocess.run(
        [sys.executable, str(_MODULE_PATH.parent / scanner), "--tarball", str(archive)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1, result.stderr
    assert "private_key_content" in result.stdout


@pytest.mark.parametrize(
    "header",
    [
        b"PRIVATE KEY",
        b"RSA PRIVATE KEY",
        b"DSA PRIVATE KEY",
        b"EC PRIVATE KEY",
        b"OPENSSH PRIVATE KEY",
        b"ENCRYPTED PRIVATE KEY",
    ],
)
def test_every_header_split_with_short_reads(monkeypatch, header: bytes) -> None:
    marker = b"-----BEGIN " + header + b"-----"
    monkeypatch.setattr(credentials, "CONTENT_CHUNK", 41)
    for split in range(1, len(marker) + 1):
        payload = b"\x00" * (41 - split) + marker + b" \t\r\n\f\v" * 30 + b"Y"
        assert (
            credentials.content_credential(io.BytesIO(payload)) == "private_key_content"
        )


@pytest.mark.parametrize("suffix", [b"", b"\x00", b"!", b"\x00Y"])
def test_whitespace_without_body_is_not_a_key(monkeypatch, suffix: bytes) -> None:
    monkeypatch.setattr(credentials, "CONTENT_CHUNK", 7)
    payload = b"-----BEGIN PRIVATE KEY-----" + b"\n" * 100 + suffix
    assert credentials.content_credential(io.BytesIO(payload)) is None


@pytest.mark.parametrize("terminator", [b"", b"\x00", b"!"])
def test_rejected_header_does_not_hide_following_key(
    monkeypatch, terminator: bytes
) -> None:
    monkeypatch.setattr(credentials, "CONTENT_CHUNK", 3)
    header = b"-----BEGIN PRIVATE KEY-----"
    payload = header + b"\n" * 100 + terminator + header + b"\n" * 100 + b"Y"
    assert credentials.content_credential(io.BytesIO(payload)) == "private_key_content"


class _WhitespaceStream:
    """Generate a large whitespace region without allocating the full input."""

    def __init__(self, size: int) -> None:
        self.remaining = size
        self.header = True

    def read(self, size: int) -> bytes:
        if self.header:
            self.header = False
            return b"-----BEGIN PRIVATE KEY-----"
        if self.remaining:
            count = min(size, self.remaining)
            self.remaining -= count
            return b"\n" * count
        return b""


def test_retained_memory_does_not_grow_with_header_whitespace(monkeypatch) -> None:
    monkeypatch.setattr(credentials, "CONTENT_CHUNK", 8192)
    peaks = []
    for size in (1024 * 1024, 16 * 1024 * 1024):
        tracemalloc.start()
        try:
            assert credentials.content_credential(_WhitespaceStream(size)) is None
            peaks.append(tracemalloc.get_traced_memory()[1])
        finally:
            tracemalloc.stop()
    assert max(peaks) < 256 * 1024
    assert max(peaks) < min(peaks) * 8


@pytest.mark.parametrize(
    "tail", [b"Y", b"\x00", b"", b"!-----BEGIN EC PRIVATE KEY-----\nY"]
)
def test_pem_streaming_matches_whole_input_oracle(monkeypatch, tail: bytes) -> None:
    monkeypatch.setattr(credentials, "CONTENT_CHUNK", 13)
    payload = b"-----BEGIN ENCRYPTED PRIVATE KEY-----" + b" \t\r\n\f\v" * 100 + tail
    expected = any(
        pattern.search(payload) for _, pattern in credentials.DECLARED_CONTENT_PATTERNS
    )
    assert (credentials.content_credential(io.BytesIO(payload)) is not None) == expected
