"""Checksum and streamed-download integrity helpers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Protocol

from npa.workbench.encord.schemas import ChecksumKind


@dataclass(frozen=True)
class StreamDigest:
    size: int
    sha256: str


class HttpDownloader(Protocol):
    def download(self, url: str, destination: Path) -> StreamDigest: ...


class HttpxDownloader:
    def download(self, url: str, destination: Path) -> StreamDigest:
        import httpx

        with httpx.stream("GET", url, timeout=600.0, follow_redirects=True) as response:
            response.raise_for_status()
            return write_hashed_stream(response.iter_bytes(8 * 1024 * 1024), destination)


def write_hashed_stream(chunks: Iterable[bytes], destination: Path) -> StreamDigest:
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
    def chunks() -> Iterable[bytes]:
        with path.open("rb") as handle:
            while data := handle.read(chunk_size):
                yield data

    digest = hashlib.sha256()
    size = 0
    for chunk in chunks():
        digest.update(chunk)
        size += len(chunk)
    return StreamDigest(size=size, sha256=digest.hexdigest())


def compare_checksums(
    expected: str,
    expected_kind: ChecksumKind,
    observed: str,
    observed_kind: ChecksumKind,
) -> bool | None:
    if not expected or not observed or "none" in {expected_kind, observed_kind}:
        return None
    compatible = {
        ("sha256", "sha256"),
        ("sha256", "s3_checksum_sha256"),
        ("s3_checksum_sha256", "sha256"),
        ("s3_checksum_sha256", "s3_checksum_sha256"),
        ("md5", "md5"),
    }
    if (expected_kind, observed_kind) not in compatible:
        return None
    return _normalize_checksum(expected) == _normalize_checksum(observed)


def _normalize_checksum(value: str) -> str:
    return value.strip().strip('"').lower()


def retry_signed_download(
    downloader: HttpDownloader,
    initial_url: str,
    refresh: Callable[[], str],
    destination: Path,
) -> StreamDigest:
    try:
        return downloader.download(initial_url, destination)
    except Exception as first:  # noqa: BLE001 - injected downloader types vary
        try:
            import httpx
        except ModuleNotFoundError:
            raise first
        if not isinstance(first, httpx.HTTPStatusError):
            raise
        refreshed = refresh()
        if not refreshed or refreshed == initial_url:
            raise first
        return downloader.download(refreshed, destination)
