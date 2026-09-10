"""Verify the canonical NCore CPython notice against two official archives."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import os
from pathlib import Path
import stat
import tarfile
import tempfile
from typing import Any, BinaryIO

from npa._public_https import download_public_https


REPOSITORY_PATH = "npa/docker/workbench/ncore/notices/cpython/LICENSE.third-party"
IMAGE_PATH = "usr/share/doc/npa-ncore/cpython/LICENSE.third-party"
ATTRIBUTION_LINES = frozenset({633, 640})
NOTICE_SHA256 = "341832873fd316a37927e79385093fbbfd40a467428480835fe435a80cadf4e5"
NOTICE_SIZE = 54_130

SOURCE_ARCHIVE_URL = "https://www.python.org/ftp/python/3.12.14/Python-3.12.14.tar.xz"
SOURCE_ARCHIVE_SHA256 = (
    "5c8462af5790baf43a321a1559dbe0db06d1be4300fb85fb53c40060668e548a"
)
SOURCE_ARCHIVE_SIZE = 20_820_300
SOURCE_ARCHIVE_MEMBER = "Python-3.12.14/Doc/license.rst"

COMMIT_ARCHIVE_URL = (
    "https://codeload.github.com/python/cpython/tar.gz/"
    "2abcf904b8dac8c999d2b3aac76681abb333798a"
)
COMMIT_ARCHIVE_SHA256 = (
    "72b4d8791b053a808f247e0de247a8b976868e705204e8fcab0812c823873766"
)
COMMIT_ARCHIVE_SIZE = 27_792_672
COMMIT_ARCHIVE_MEMBER = (
    "cpython-2abcf904b8dac8c999d2b3aac76681abb333798a/Doc/license.rst"
)


@dataclass(frozen=True)
class _Archive:
    url: str
    sha256: str
    size: int
    member: str
    cache_name: str
    host: str


_ARCHIVES = (
    _Archive(
        SOURCE_ARCHIVE_URL,
        SOURCE_ARCHIVE_SHA256,
        SOURCE_ARCHIVE_SIZE,
        SOURCE_ARCHIVE_MEMBER,
        "Python-3.12.14.tar.xz",
        "www.python.org",
    ),
    _Archive(
        COMMIT_ARCHIVE_URL,
        COMMIT_ARCHIVE_SHA256,
        COMMIT_ARCHIVE_SIZE,
        COMMIT_ARCHIVE_MEMBER,
        "cpython-2abcf904b8dac8c999d2b3aac76681abb333798a.tar.gz",
        "codeload.github.com",
    ),
)


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_private_directory(proof_directory: Path) -> None:
    details = proof_directory.lstat()
    if not stat.S_ISDIR(details.st_mode) or proof_directory.is_symlink():
        raise ValueError("proof directory must be a real directory")
    if details.st_uid != os.getuid() or stat.S_IMODE(details.st_mode) & 0o077:
        raise ValueError("proof directory must be caller-owned and private")


def _verified_file(path: Path, archive: _Archive) -> bytes:
    details = path.lstat()
    if not stat.S_ISREG(details.st_mode) or path.is_symlink():
        raise ValueError("cached proof archive must be a regular file")
    if (
        details.st_uid != os.getuid()
        or details.st_nlink != 1
        or stat.S_IMODE(details.st_mode) & 0o077
    ):
        raise ValueError("cached proof archive must be caller-owned and private")
    payload = path.read_bytes()
    if len(payload) != archive.size or _digest(payload) != archive.sha256:
        raise ValueError("cached proof archive does not match its public pin")
    return payload


@dataclass
class _PinnedArchiveOutput:
    output: BinaryIO
    expected_size: int
    received: int = 0

    def write(self, payload: bytes) -> int:
        self.received += len(payload)
        if self.received > self.expected_size:
            raise ValueError("proof archive exceeds its public size pin")
        return self.output.write(payload)


def _download_archive(path: Path, archive: _Archive) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as output:
            temporary = Path(output.name)
            download_public_https(
                archive.url, _PinnedArchiveOutput(output, archive.size),
                allowed_hosts=frozenset({archive.host}),
            )
            output.flush()
            os.fsync(output.fileno())
        _verified_file(temporary, archive)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _archive_payload(proof_directory: Path, archive: _Archive) -> bytes:
    path = proof_directory / archive.cache_name
    if not path.exists():
        _download_archive(path, archive)
    return _verified_file(path, archive)


def _notice_member(archive_payload: bytes, archive: _Archive) -> bytes:
    with tarfile.open(fileobj=io.BytesIO(archive_payload), mode="r:*") as bundle:
        matches = [
            member for member in bundle.getmembers() if member.name == archive.member
        ]
        if (
            len(matches) != 1
            or not matches[0].isreg()
            or matches[0].size != NOTICE_SIZE
        ):
            raise ValueError("official archive has no unique regular notice member")
        extracted = bundle.extractfile(matches[0])
        if extracted is None:
            raise ValueError("official archive notice member cannot be read")
        return extracted.read()


def _archive_provenance(archive: _Archive) -> dict[str, Any]:
    return {
        "url": archive.url,
        "sha256": archive.sha256,
        "size": archive.size,
        "member": archive.member,
        "member_sha256": NOTICE_SHA256,
        "member_size": NOTICE_SIZE,
    }


def verify_public_notice(notice: bytes, proof_directory: Path) -> dict[str, Any]:
    """Verify exact notice bytes against two immutable official archives.

    Args:
        notice: Complete candidate notice bytes.
        proof_directory: Existing caller-owned directory with mode ``0700``.

    Returns:
        Public hashes, sizes, member names, and URLs for the verified sources.

    Raises:
        OSError: An archive cannot be downloaded, cached, or read.
        TarError: An archive cannot be parsed.
        ValueError: Any byte, ownership, mode, hash, size, or member check fails.
    """
    if len(notice) != NOTICE_SIZE or _digest(notice) != NOTICE_SHA256:
        raise ValueError("notice does not match the canonical public bytes")
    _require_private_directory(proof_directory)

    provenance = []
    for archive in _ARCHIVES:
        member = _notice_member(_archive_payload(proof_directory, archive), archive)
        if member != notice:
            raise ValueError("official archive member differs from the notice")
        provenance.append(_archive_provenance(archive))

    return {
        "notice": {"sha256": NOTICE_SHA256, "size": NOTICE_SIZE},
        "archives": provenance,
    }
