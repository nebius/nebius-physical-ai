"""Run Gitleaks over exact committed history and every candidate-tree blob."""

from __future__ import annotations

import argparse
import base64
import binascii
import bz2
import gzip
import hashlib
import importlib.util
import json
import lzma
import mmap
import os
import re
import secrets
import stat
import subprocess
import sys
import tarfile
import tempfile
import urllib.parse
import zipfile
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Self

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

_MATERIALIZER_PATH = Path(__file__).with_name("materialize_gitleaks_tree.py")
_MATERIALIZER_SPEC = importlib.util.spec_from_file_location(
    "materialize_gitleaks_tree",
    _MATERIALIZER_PATH,
)
if _MATERIALIZER_SPEC is None or _MATERIALIZER_SPEC.loader is None:
    raise RuntimeError("Could not load the Gitleaks tree materializer")
materializer = importlib.util.module_from_spec(_MATERIALIZER_SPEC)
_MATERIALIZER_SPEC.loader.exec_module(materializer)

_MAX_ARCHIVE_DEPTH = 3
_MAX_DECODE_DEPTH = 5
_GITLEAKS_MAX_PROCS = 8
_GITLEAKS_VERSION = "8.28.0"
_DECODE_PROBE_PREFIX = "npa-decode-depth-probe-"
_WRAPPER_SCAN_ROOTS = (
    os.fsdecode(materializer.SCAN_NON_TEST_ROOT) + "/",
    os.fsdecode(materializer.SCAN_TEST_ROOT) + "/",
)
_ENCODING_CANDIDATES = (
    (
        "percent",
        re.compile(rb"%[0-9A-Fa-f]{2}(?:[^\r\n]*%[0-9A-Fa-f]{2})?"),
    ),
    (
        "unicode",
        re.compile(
            rb"(?:(?:U\+[a-fA-F0-9]{4}(?:\s|$))+|(?:\\{1,2}u[a-fA-F0-9]{4})+)",
            re.IGNORECASE,
        ),
    ),
    ("hex", re.compile(rb"[0-9A-Fa-f]{32,}")),
    ("base64", re.compile(rb"[\w/+\-]{16,}={0,2}")),
)
_PROBE_STREAM_MAX_BYTES = 96 * 1024
_PROBE_DECODE_CHUNK_BYTES = 64 * 1024
_NON_PRINTABLE_ASCII = re.compile(rb"[^\x09-\x7e]")
_BASE64_HINT = re.compile(rb"[0-9+/_-]")
_INVALID_STANDARD_BASE64 = re.compile(rb"[^A-Za-z0-9+/=]")
_INVALID_URLSAFE_BASE64 = re.compile(rb"[^A-Za-z0-9_-]")
_HEX_DIGIT = re.compile(rb"[0-9]")
_ARCHIVE_SUFFIXES = {
    ".bz2",
    ".gz",
    ".tar",
    ".xz",
    ".zip",
}
_REJECTED_ARCHIVE_SUFFIXES = {
    ".7z",
    ".apk",
    ".br",
    ".docx",
    ".epub",
    ".jar",
    ".lzip",
    ".lz",
    ".lz4",
    ".mz",
    ".odp",
    ".ods",
    ".odt",
    ".pptx",
    ".rar",
    ".s2",
    ".snappy",
    ".sz",
    ".tbz",
    ".tbz2",
    ".tgz",
    ".tlz4",
    ".txz",
    ".tzst",
    ".war",
    ".whl",
    ".xlsx",
    ".zst",
    ".zstd",
    ".zz",
}
_TAR_SUFFIXES = (".tar", ".tar.bz2", ".tar.gz", ".tar.xz")
_ARCHIVE_READ_BYTES = 64 * 1024
_ARCHIVE_PREFIX_BYTES = 512
_ENCODED_ARCHIVE_MAX_CANDIDATES = 4096
_ENCODED_ARCHIVE_PREFIX_BYTES = 8 * 1024
_INERT_ENCODED_ARCHIVE_SHA256 = {
    # Public empty-metadata gzip fixture in test_curobo_image_verifier.py.
    "4f4fb700ef54461cfa02571ae0db9a0dc1e0cdb5577484a6d75e68dc38e8acc1",
}
_DECODE_PROBE_CONFIG = rb"""title = "NPA decode-depth boundary probe"

[[rules]]
id = "npa-decode-depth-probe-percent"
description = "NPA private percent-decode boundary probe"
regex = '''%[0-9A-Fa-f]{2}(?:.*%[0-9A-Fa-f]{2})?'''

[[rules]]
id = "npa-decode-depth-probe-unicode"
description = "NPA private Unicode-decode boundary probe"
regex = '''(?:(?:U\+[a-fA-F0-9]{4}(?:\s|$))+|(?i:(?:\\{1,2}u[a-fA-F0-9]{4})+))'''

[[rules]]
id = "npa-decode-depth-probe-hex"
description = "NPA private hexadecimal-decode boundary probe"
regex = '''[0-9A-Fa-f]{32,}'''

[[rules]]
id = "npa-decode-depth-probe-base64"
description = "NPA private Base64-decode boundary probe"
regex = '''[\w\/+-]{16,}={0,2}'''
"""


class _ScanError(RuntimeError):
    """Report a tree scan that cannot prove complete fail-closed coverage."""


class _ProbeSourceCounts:
    """Accumulate compact decode-probe source counts."""

    def __init__(self) -> None:
        self.files = 0
        self.wrapper_files = 0
        self.archive_bytes = 0
        self.archive_files = 0
        self.archive_streams = 0

    def render(self, candidates: _CandidateProbeWriter) -> dict[str, int]:
        """Combine source and encoded-candidate counts."""
        return {
            "archive_bytes": self.archive_bytes,
            "archive_files": self.archive_files,
            "archive_streams": self.archive_streams,
            "candidate_bytes": candidates.candidate_bytes,
            "candidate_duplicates": candidates.duplicate_candidates,
            "candidate_matches": candidates.candidates,
            "candidate_streams": candidates.streams,
            "files": self.files,
            "max_candidate_bytes": candidates.max_candidate_bytes,
            "wrapper_files": self.wrapper_files,
        }


class _ScanPlan:
    """Bind exact commit identities and policy bytes for one scan."""

    def __init__(
        self,
        candidate: dict[str, object],
        policy: dict[str, object],
        commits: list[str],
        historical_revisions: list[str],
        commit_range: str,
        history_base: str,
        config_content: bytes,
    ) -> None:
        self.candidate = candidate
        self.policy = policy
        self.commits = commits
        self.historical_revisions = historical_revisions
        self.commit_range = commit_range
        self.history_base = history_base
        self.config_content = config_content


class _MaterializedSources:
    """Collect exact candidate, history, and scanner workspace paths."""

    def __init__(
        self,
        snapshot: Path,
        candidate_receipt: dict[str, object],
        historical_view: Path | None,
        historical_receipt: dict[str, object],
        history_source: Path,
        scanner_work: Path,
    ) -> None:
        self.snapshot = snapshot
        self.candidate_receipt = candidate_receipt
        self.historical_view = historical_view
        self.historical_receipt = historical_receipt
        self.history_source = history_source
        self.scanner_work = scanner_work


class _ProbeSources:
    """Collect compact decode-probe views and their receipts."""

    def __init__(
        self,
        candidate_view: Path,
        candidate_receipt: dict[str, int],
        historical_view: Path | None,
        historical_receipt: dict[str, int],
    ) -> None:
        self.candidate_view = candidate_view
        self.candidate_receipt = candidate_receipt
        self.historical_view = historical_view
        self.historical_receipt = historical_receipt


class _FindingSets:
    """Collect findings from each independently scanned source."""

    def __init__(
        self,
        history: list[dict[str, object]],
        historical_trees: list[dict[str, object]],
        tree: list[dict[str, object]],
    ) -> None:
        self.history = history
        self.historical_trees = historical_trees
        self.tree = tree


def _config_blob(repo: Path, tree: str, relative_path: str) -> bytes:
    """Read one regular committed config blob without checkout transformations."""
    path = PurePosixPath(relative_path)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise _ScanError("Gitleaks config path must stay inside the committed tree")
    return materializer.read_verified_blob(repo, tree, os.fsencode(path.as_posix()))


def _optional_config_blob(
    repo: Path,
    tree: str,
    relative_path: str,
) -> bytes | None:
    """Read one optional regular committed policy blob."""
    path = PurePosixPath(relative_path)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise _ScanError("Gitleaks policy path must stay inside the committed tree")
    return materializer.read_optional_verified_blob(
        repo,
        tree,
        os.fsencode(path.as_posix()),
    )


def _write_private(path: Path, content: bytes) -> None:
    """Write one new private regular file."""
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(content)


class _CandidateProbeWriter:
    """Keep every encoded match inside one complete Gitleaks file fragment."""

    def __init__(self, destination: Path) -> None:
        self.destination = destination
        self.stream: BinaryIO | None = None
        self.stream_bytes = 0
        self.streams = 0
        self.candidates = 0
        self.candidate_bytes = 0
        self.duplicate_candidates = 0
        self.max_candidate_bytes = 0
        self.seen: set[tuple[int, bytes]] = set()

    def _open_stream(self) -> None:
        if self.stream is not None:
            self.stream.close()
        target = self.destination / f"decode-input-candidates-{self.streams}.txt"
        descriptor = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        self.stream = os.fdopen(descriptor, "wb")
        self.stream.write(materializer.SCAN_PREFIX)
        self.stream_bytes = len(materializer.SCAN_PREFIX)
        self.streams += 1

    def _append_candidate(self, candidate: bytes) -> None:
        """Deduplicate and write one bounded codec-valid candidate."""
        identity = (len(candidate), hashlib.sha256(candidate).digest())
        if identity in self.seen:
            self.duplicate_candidates += 1
            return
        self.seen.add(identity)
        payload_bytes = len(candidate) + 2
        if (
            self.stream is None
            or self.stream_bytes + payload_bytes > _PROBE_STREAM_MAX_BYTES
        ):
            self._open_stream()
        if self.stream is None:  # pragma: no cover - invariant guard
            raise _ScanError("Could not open the encoded-candidate stream")
        self.stream.write(candidate)
        self.stream.write(b"\n\n")
        self.stream_bytes += payload_bytes
        self.candidates += 1
        self.candidate_bytes += payload_bytes
        self.max_candidate_bytes = max(self.max_candidate_bytes, len(candidate))

    def append(self, content: bytes | mmap.mmap) -> None:
        """Append exact scanner-supported matches without fragment splitting."""
        for kind, pattern in _ENCODING_CANDIDATES:
            for match in pattern.finditer(content):  # type: ignore[arg-type]
                start, end = match.span()
                candidate_size = end - start
                payload_bytes = candidate_size + 2
                archive_probe_end = min(
                    end,
                    start + _ENCODED_ARCHIVE_PREFIX_BYTES,
                )
                _reject_encoded_archives(bytes(content[start:archive_probe_end]))
                if not _probe_span_decodes(kind, content, start, end):
                    continue
                if payload_bytes + len(materializer.SCAN_PREFIX) > (
                    _PROBE_STREAM_MAX_BYTES
                ):
                    raise _ScanError(
                        "Encoded segment exceeds one complete Gitleaks scan fragment"
                    )
                self._append_candidate(content[start:end])

    def close(self) -> None:
        """Close the current stream when present."""
        if self.stream is not None:
            self.stream.close()
            self.stream = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        _exception_type: object,
        _exception: object,
        _traceback: object,
    ) -> None:
        self.close()


def _archive_probe_names(path: Path) -> set[str]:
    """Select canonical archive names supported by the validated scan boundary."""
    lowered = path.name.lower()
    if any(lowered.endswith(suffix) for suffix in _ARCHIVE_SUFFIXES):
        return {path.name}
    return set()


def _archive_kind(prefix: bytes) -> str | None:
    """Identify archive formats with stable public magic bytes."""
    if prefix.startswith((b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")):
        return "zip"
    if (
        len(prefix) >= 10
        and prefix.startswith(b"\x1f\x8b\x08")
        and not prefix[3] & 0xE0
    ):
        return "gzip"
    if prefix.startswith(b"BZh") and prefix[3:4] in b"123456789":
        return "bzip2"
    if prefix.startswith(b"\xfd7zXZ\x00"):
        return "xz"
    if prefix.startswith(b"7z\xbc\xaf'\x1c"):
        return "7z"
    if prefix.startswith((b"Rar!\x1a\x07\x00", b"Rar!\x1a\x07\x01\x00")):
        return "rar"
    if prefix.startswith(b"\x28\xb5\x2f\xfd"):
        return "zstd"
    if prefix.startswith((b"\x04\x22\x4d\x18", b"\x02\x21\x4c\x18")):
        return "lz4"
    if len(prefix) >= 262 and prefix[257:262] == b"ustar":
        return "tar"
    return None


def _has_archive_suffix(name: str) -> bool:
    """Return whether a path advertises any supported or rejected archive type."""
    lowered = name.lower()
    return any(
        lowered.endswith(suffix)
        for suffix in _ARCHIVE_SUFFIXES | _REJECTED_ARCHIVE_SUFFIXES
    )


def _inspect_archive_stream(stream: BinaryIO) -> tuple[bytes, bool]:
    """Verify a stream and detect encoded or self-extracting nested ZIPs."""
    prefix = bytearray()
    overlap = b""
    is_zip = False
    zip_signatures = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")
    while chunk := stream.read(_ARCHIVE_READ_BYTES):
        if len(prefix) < _ARCHIVE_PREFIX_BYTES:
            missing = _ARCHIVE_PREFIX_BYTES - len(prefix)
            prefix.extend(chunk[:missing])
        window = overlap + chunk
        _reject_encoded_archives(window)
        is_zip = is_zip or any(signature in window for signature in zip_signatures)
        overlap = window[-_ENCODED_ARCHIVE_PREFIX_BYTES:]
    return bytes(prefix), is_zip


def _reject_nested_archive(
    member: str,
    prefix: bytes,
    *,
    is_zip: bool,
) -> None:
    """Reject nested containers whose complete policy traversal is ambiguous."""
    if _has_archive_suffix(member) or _archive_kind(prefix) is not None or is_zip:
        raise _ScanError(
            "Nested archive member cannot prove complete Gitleaks policy coverage"
        )


def _decode_candidate(kind: str, candidate: bytes) -> list[bytes]:
    """Decode one bounded scanner candidate without requiring printable output."""
    decoded: list[bytes] = []
    try:
        if kind == "base64":
            standard_size = len(candidate) - (len(candidate) % 4)
            variants = (
                (None, candidate[:standard_size]),
                (b"-_", candidate + b"=" * (-len(candidate) % 4)),
            )
            for altchars, value in variants:
                if not value:
                    continue
                result = base64.b64decode(
                    value,
                    altchars=altchars,
                    validate=True,
                )
                if result not in decoded:
                    decoded.append(result)
        elif kind == "hex" and len(candidate) % 2 == 0:
            decoded.append(binascii.unhexlify(candidate))
        elif kind == "percent":
            decoded.append(urllib.parse.unquote_to_bytes(candidate))
    except (binascii.Error, ValueError):
        return []
    return decoded


def _reject_encoded_archives(candidate: bytes) -> None:
    """Reject archive bytes hidden behind one or more supported encodings."""
    queue: list[tuple[bytes, int]] = [(candidate, 0)]
    seen: set[tuple[int, bytes]] = set()
    examined = 0
    while queue:
        content, depth = queue.pop()
        identity = (depth, hashlib.sha256(content).digest())
        if identity in seen:
            continue
        seen.add(identity)
        if depth >= _MAX_DECODE_DEPTH:
            continue
        for kind, pattern in _ENCODING_CANDIDATES:
            if kind == "unicode":
                continue
            for match in pattern.finditer(content):
                for decoded in _decode_candidate(kind, match.group()):
                    examined += 1
                    if examined > _ENCODED_ARCHIVE_MAX_CANDIDATES:
                        raise _ScanError(
                            "Encoded archive analysis exceeded its bounded "
                            "candidate count"
                        )
                    archive_kind = _archive_kind(decoded[:_ARCHIVE_PREFIX_BYTES])
                    if archive_kind is not None and (
                        hashlib.sha256(decoded).hexdigest()
                        not in _INERT_ENCODED_ARCHIVE_SHA256
                    ):
                        raise _ScanError(
                            "Encoded archive bytes cannot prove complete "
                            "Gitleaks policy coverage"
                        )
                    if archive_kind is not None:
                        continue
                    queue.append((decoded, depth + 1))


def _validate_zip_archive(path: Path) -> None:
    """Verify one canonical ZIP and reject encrypted or nested containers."""
    try:
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                if member.flag_bits & 0x1:
                    raise _ScanError("Encrypted archive member cannot be scanned")
                with archive.open(member) as stream:
                    prefix, is_zip = _inspect_archive_stream(stream)
                _reject_nested_archive(
                    member.filename,
                    prefix,
                    is_zip=is_zip,
                )
    except _ScanError:
        raise
    except (NotImplementedError, RuntimeError, zipfile.BadZipFile) as exc:
        raise _ScanError("ZIP archive could not be verified") from exc


def _validate_tar_archive(path: Path) -> None:
    """Verify one canonical TAR stream and reject special or nested members."""
    try:
        with tarfile.open(path, mode="r:*") as archive:
            for member in archive:
                if member.isdir():
                    continue
                if not member.isfile():
                    raise _ScanError("Special archive member cannot be scanned")
                stream = archive.extractfile(member)
                if stream is None:
                    raise _ScanError("Archive member could not be read")
                with stream:
                    prefix, is_zip = _inspect_archive_stream(stream)
                _reject_nested_archive(
                    member.name,
                    prefix,
                    is_zip=is_zip,
                )
    except _ScanError:
        raise
    except (OSError, tarfile.TarError) as exc:
        raise _ScanError("TAR archive could not be verified") from exc


def _validate_compressed_stream(path: Path, kind: str) -> None:
    """Verify one non-TAR standard-library stream and reject nested archives."""
    openers = {
        "bzip2": bz2.open,
        "gzip": gzip.open,
        "xz": lzma.open,
    }
    try:
        with openers[kind](path, "rb") as stream:
            prefix, is_zip = _inspect_archive_stream(stream)
    except (EOFError, OSError, lzma.LZMAError) as exc:
        raise _ScanError(f"{kind} archive could not be verified") from exc
    _reject_nested_archive(
        "<compressed-stream>",
        prefix,
        is_zip=is_zip,
    )


def _validate_archive_file(path: Path) -> None:
    """Fail closed on archive names or bytes Gitleaks cannot traverse completely."""
    lowered = path.name.lower()
    if any(lowered.endswith(suffix) for suffix in _REJECTED_ARCHIVE_SUFFIXES):
        raise _ScanError("Archive uses a noncanonical or unsupported extension")
    with path.open("rb") as stream:
        prefix = stream.read(_ARCHIVE_PREFIX_BYTES)
    kind = _archive_kind(prefix)
    is_zip = zipfile.is_zipfile(path)
    if is_zip:
        if not lowered.endswith(".zip"):
            raise _ScanError("ZIP archive bytes require a canonical .zip extension")
        _validate_zip_archive(path)
        return
    if lowered.endswith(".zip"):
        raise _ScanError("ZIP archive is malformed")
    if lowered.endswith(_TAR_SUFFIXES):
        _validate_tar_archive(path)
        return
    compressed_kind = next(
        (
            candidate
            for candidate, suffix in (
                ("bzip2", ".bz2"),
                ("gzip", ".gz"),
                ("xz", ".xz"),
            )
            if lowered.endswith(suffix)
        ),
        None,
    )
    if compressed_kind is not None:
        if kind != compressed_kind:
            raise _ScanError("Archive does not match its canonical extension")
        _validate_compressed_stream(path, compressed_kind)
        return
    if kind is not None:
        raise _ScanError(f"Disguised {kind} archive bytes cannot be scanned")


def _append_probe_file(
    path: Path,
    relative: str,
    is_wrapper: bool,
    candidates: _CandidateProbeWriter,
) -> None:
    """Collect codec-valid spans from one source file and its path."""
    if not is_wrapper:
        candidates.append(os.fsencode(relative))
    if not path.stat().st_size:
        return
    with (
        path.open("rb") as stream,
        mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as content,
    ):
        candidates.append(content)


def _link_probe_archives(
    path: Path,
    relative: str,
    destination: Path,
    counts: _ProbeSourceCounts,
) -> None:
    """Link one raw file under every pinned archive-compatible name."""
    _validate_archive_file(path)
    archive_names = _archive_probe_names(path)
    if not archive_names:
        return
    identity = hashlib.sha256(os.fsencode(relative)).hexdigest()
    archive_root = destination / "decode-input-archives" / identity
    archive_root.mkdir(mode=0o700, parents=True)
    for name in sorted(archive_names):
        os.link(path, archive_root / name)
        counts.archive_streams += 1
    counts.archive_bytes += path.stat().st_size
    counts.archive_files += 1


def _prepare_decode_probe_source(
    source: Path,
    destination: Path,
) -> dict[str, int]:
    """Build a compact view of encoded spans, paths, Unicode, and archives."""
    destination.mkdir(mode=0o700)
    counts = _ProbeSourceCounts()
    with _CandidateProbeWriter(destination) as candidates:
        for path in sorted(source.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(source).as_posix()
            is_wrapper = relative.startswith(_WRAPPER_SCAN_ROOTS)
            if is_wrapper:
                counts.wrapper_files += 1
            else:
                counts.files += 1
            _append_probe_file(path, relative, is_wrapper, candidates)
            if is_wrapper:
                continue
            _link_probe_archives(path, relative, destination, counts)
    return counts.render(candidates)


def _isolated_history_source(repo: Path, destination: Path) -> Path:
    """Expose only Git metadata so checkout ignore files cannot affect history."""
    destination.mkdir(mode=0o700)
    git_directory = materializer.git_directory(repo)
    encoded = os.fsencode(git_directory)
    if b"\n" in encoded or b"\r" in encoded:
        raise _ScanError("Git metadata path cannot be represented safely")
    _write_private(destination / ".git", b"gitdir: " + encoded + b"\n")
    if materializer.git_directory(destination) != git_directory:
        raise _ScanError("Isolated history source resolved different Git metadata")
    return destination


def _verified_executable(executable: str) -> str:
    """Require one explicit regular executable instead of trusting PATH."""
    path = Path(executable)
    if not path.is_absolute():
        raise _ScanError("Gitleaks executable must be an absolute path")
    metadata = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise _ScanError("Gitleaks executable must be a regular file")
    if not os.access(path, os.X_OK):
        raise _ScanError("Gitleaks executable must be executable")
    return os.fsdecode(path)


def _metadata_identity(
    metadata: os.stat_result,
) -> tuple[int, int, int, int, int, int]:
    """Select stat fields that expose replacement or in-place mutation."""
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _executable_metadata(path: Path) -> tuple[int, int, int, int, int, int]:
    """Return identity fields that expose replacement or in-place mutation."""
    metadata = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise _ScanError("Gitleaks executable must remain a regular file")
    return _metadata_identity(metadata)


def _executable_sha256(path: Path) -> str:
    """Hash one scanner executable without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _stage_gitleaks_executable(executable: str, root: Path) -> tuple[str, str]:
    """Copy scanner bytes through one no-follow descriptor into private storage."""
    source_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    source_descriptor = os.open(executable, source_flags)
    destination = root / "gitleaks"
    destination_descriptor = -1
    digest = hashlib.sha256()
    try:
        before = os.fstat(source_descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise _ScanError("Gitleaks executable must remain a regular file")
        destination_descriptor = os.open(
            destination,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o500,
        )
        while content := os.read(source_descriptor, 1024 * 1024):
            digest.update(content)
            view = memoryview(content)
            while view:
                written = os.write(destination_descriptor, view)
                view = view[written:]
        os.fsync(destination_descriptor)
        after = os.fstat(source_descriptor)
        if _metadata_identity(before) != _metadata_identity(after):
            raise _ScanError("Gitleaks executable changed while it was staged")
    finally:
        os.close(source_descriptor)
        if destination_descriptor >= 0:
            os.close(destination_descriptor)
    staged = os.fsdecode(destination)
    staged_digest = _verify_gitleaks_identity(staged)
    if staged_digest != digest.hexdigest():
        raise _ScanError("Staged Gitleaks bytes do not match their copied digest")
    return staged, staged_digest


def _verify_gitleaks_identity(executable: str) -> str:
    """Bind each scan to unchanged bytes of the pinned Gitleaks release."""
    path = Path(executable)
    before = _executable_metadata(path)
    digest = _executable_sha256(path)
    result = subprocess.run(
        [executable, "version"],
        check=False,
        capture_output=True,
        env=_scan_environment(),
    )
    after = _executable_metadata(path)
    if before != after:
        raise _ScanError("Gitleaks executable changed during identity verification")
    if (
        result.returncode != 0
        or result.stdout.strip() != _GITLEAKS_VERSION.encode("ascii")
        or result.stderr.strip()
    ):
        raise _ScanError(
            f"Gitleaks executable must report exact version {_GITLEAKS_VERSION}"
        )
    return digest


def _scan_environment() -> dict[str, str]:
    """Remove caller-controlled Git and Gitleaks process indirection."""
    return {
        "GIT_CONFIG_COUNT": "3",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_KEY_0": "core.commitGraph",
        "GIT_CONFIG_KEY_1": "core.attributesFile",
        "GIT_CONFIG_KEY_2": "diff.external",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_VALUE_0": "false",
        "GIT_CONFIG_VALUE_1": os.devnull,
        "GIT_CONFIG_VALUE_2": "",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GOMAXPROCS": str(_GITLEAKS_MAX_PROCS),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/bin:/bin",
    }


def _gitleaks_base_arguments(
    executable: str,
    source_argument: str,
    config: Path,
    ignore: Path,
    max_decode_depth: int,
) -> list[str]:
    """Build invariant arguments for one pinned Gitleaks invocation."""
    return [
        executable,
        "detect",
        "--source",
        source_argument,
        "--config",
        str(config),
        "--gitleaks-ignore-path",
        str(ignore),
        "--no-banner",
        "--no-color",
        "--ignore-gitleaks-allow",
        "--max-archive-depth",
        str(_MAX_ARCHIVE_DEPTH),
        "--max-decode-depth",
        str(max_decode_depth),
    ]


def _gitleaks_arguments(
    executable: str,
    source_argument: str,
    config: Path,
    ignore: Path,
    report: Path,
    *,
    history_range: str | None = None,
    max_decode_depth: int = _MAX_DECODE_DEPTH,
    redact: bool = True,
) -> list[str]:
    """Build one pinned Gitleaks invocation."""
    arguments = _gitleaks_base_arguments(
        executable,
        source_argument,
        config,
        ignore,
        max_decode_depth,
    )
    if redact:
        arguments.append("--redact")
    if history_range is None:
        arguments.append("--no-git")
    else:
        arguments.extend(
            (
                "--log-opts",
                f"--no-textconv --no-ext-diff {history_range}",
            )
        )
    arguments.extend(
        (
            "--report-format",
            "json",
            "--report-path",
            str(report),
        )
    )
    return arguments


def _load_findings(report: Path) -> list[dict[str, object]]:
    """Validate and load one private Gitleaks JSON report."""
    if report.is_symlink() or not report.is_file():
        raise _ScanError("Gitleaks did not produce a regular JSON report")
    loaded = json.loads(report.read_text(encoding="utf-8"))
    if not isinstance(loaded, list) or not all(
        isinstance(item, dict) for item in loaded
    ):
        raise _ScanError("Gitleaks returned an invalid JSON report")
    findings: list[dict[str, object]] = loaded
    return findings


def _reject_archive_skip(result: subprocess.CompletedProcess[bytes]) -> None:
    """Reject scanner output that reports an archive depth omission."""
    if b"skipping archive: exceeds max archive depth" in result.stderr.lower():
        raise _ScanError(
            "Gitleaks skipped an archive that exceeds the configured depth"
        )


def _run_gitleaks(
    executable: str,
    source: Path,
    config: Path,
    ignore: Path,
    report: Path,
    *,
    history_range: str | None = None,
    working_directory: Path | None = None,
    max_decode_depth: int = _MAX_DECODE_DEPTH,
    redact: bool = True,
) -> tuple[subprocess.CompletedProcess[bytes], list[dict[str, object]]]:
    """Run one scan and parse its private JSON report."""
    process_directory = source if working_directory is None else working_directory
    source_argument = (
        "." if working_directory is None else str(source.resolve(strict=True))
    )
    arguments = _gitleaks_arguments(
        executable,
        source_argument,
        config,
        ignore,
        report,
        history_range=history_range,
        max_decode_depth=max_decode_depth,
        redact=redact,
    )
    result = subprocess.run(
        arguments,
        cwd=process_directory,
        check=False,
        capture_output=True,
        env=_scan_environment(),
    )
    findings = _load_findings(report)
    _reject_archive_skip(result)
    if not redact:
        report.unlink()
    return result, findings


def _encode_layers(payload: bytes, layers: int) -> bytes:
    """Wrap one private control in exactly ``layers`` Base64 encodings."""
    for _ in range(layers):
        payload = base64.b64encode(payload)
    return payload


def _finding_tags(item: dict[str, object]) -> list[str]:
    """Return one finding's string tags without trusting report value types."""
    tags = item.get("Tags")
    if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
        return []
    return tags


def _is_printable_ascii(content: bytes) -> bool:
    """Match Gitleaks' codec definition of printable decoded bytes."""
    return _NON_PRINTABLE_ASCII.search(content) is None


def _base64_variant_decodes(
    content: bytes | mmap.mmap,
    start: int,
    end: int,
    *,
    urlsafe: bool,
) -> bool:
    """Decode one Base64 variant in bounded chunks."""
    size = end - start
    invalid = _INVALID_URLSAFE_BASE64 if urlsafe else _INVALID_STANDARD_BASE64
    if invalid.search(content, start, end):
        return False
    if (urlsafe and size % 4 == 1) or (not urlsafe and size % 4):
        return False
    position = start
    while position < end:
        chunk_end = min(position + _PROBE_DECODE_CHUNK_BYTES, end)
        encoded = content[position:chunk_end]
        if urlsafe and chunk_end == end:
            encoded += b"=" * (-len(encoded) % 4)
        try:
            decoded = base64.b64decode(
                encoded,
                altchars=b"-_" if urlsafe else None,
                validate=True,
            )
        except (binascii.Error, ValueError):
            return False
        if not _is_printable_ascii(decoded):
            return False
        position = chunk_end
    return True


def _base64_span_decodes(
    content: bytes | mmap.mmap,
    start: int,
    end: int,
) -> bool:
    """Match the pinned scanner's standard and raw URL-safe Base64 codecs."""
    if not _BASE64_HINT.search(content, start, end):
        return False
    return _base64_variant_decodes(
        content,
        start,
        end,
        urlsafe=False,
    ) or _base64_variant_decodes(content, start, end, urlsafe=True)


def _hex_span_decodes(
    content: bytes | mmap.mmap,
    start: int,
    end: int,
) -> bool:
    """Validate printable hexadecimal data without allocating the full value."""
    if (end - start) % 2 or not _HEX_DIGIT.search(content, start, end):
        return False
    position = start
    while position < end:
        chunk_end = min(position + _PROBE_DECODE_CHUNK_BYTES, end)
        try:
            decoded = binascii.unhexlify(content[position:chunk_end])
        except (binascii.Error, ValueError):
            return False
        if not _is_printable_ascii(decoded):
            return False
        position = chunk_end
    return True


def _hex_nibble(value: int) -> int:
    """Decode one ASCII hexadecimal nibble or return minus one."""
    if 48 <= value <= 57:
        return value - 48
    if 65 <= value <= 70:
        return value - 55
    if 97 <= value <= 102:
        return value - 87
    return -1


def _percent_span_decodes(
    content: bytes | mmap.mmap,
    start: int,
    end: int,
) -> bool:
    """Validate every percent-decoded byte with the pinned scanner's rules."""
    position = content.find(b"%", start, end)
    while position >= 0:
        if position + 2 < end:
            high = _hex_nibble(content[position + 1])
            low = _hex_nibble(content[position + 2])
            if high >= 0 and low >= 0:
                if not 8 < ((high << 4) | low) < 127:
                    return False
                position = content.find(b"%", position + 3, end)
                continue
        position = content.find(b"%", position + 1, end)
    return True


def _probe_span_decodes(
    kind: str,
    content: bytes | mmap.mmap,
    start: int,
    end: int,
) -> bool:
    """Validate one regex span with pinned Gitleaks codec semantics."""
    if kind == "unicode":
        return True
    if kind == "base64":
        return _base64_span_decodes(content, start, end)
    if kind == "hex":
        return _hex_span_decodes(content, start, end)
    if kind == "percent":
        return _percent_span_decodes(content, start, end)
    return False


def _probe_match_decodes(rule_id: str, value: str) -> bool:
    """Validate one reported encoding-rule match without unbounded decoding."""
    kind = rule_id.removeprefix(_DECODE_PROBE_PREFIX)
    content = value.encode("utf-8")
    return _probe_span_decodes(kind, content, 0, len(content))


def _decode_exhaustions(findings: list[dict[str, object]]) -> list[dict[str, object]]:
    """Select proven decodable matches at the governed recursion ceiling."""
    depth_tag = f"decode-depth:{_MAX_DECODE_DEPTH}"
    exhausted = []
    for item in findings:
        rule_id = item.get("RuleID")
        match = item.get("Match")
        if (
            isinstance(rule_id, str)
            and rule_id.startswith(_DECODE_PROBE_PREFIX)
            and isinstance(match, str)
            and depth_tag in _finding_tags(item)
            and _probe_match_decodes(rule_id, match)
        ):
            exhausted.append(item)
    return exhausted


def _policy_commit_allowlists(content: bytes) -> set[str]:
    """Collect every commit allowlist that a no-Git tree scan cannot represent."""
    document = tomllib.loads(content.decode("utf-8"))
    commits: set[str] = set()

    def visit(value: object) -> None:
        if isinstance(value, dict):
            allowlisted = value.get("commits")
            if allowlisted is not None:
                if not isinstance(allowlisted, list) or not all(
                    isinstance(item, str) for item in allowlisted
                ):
                    raise _ScanError("Gitleaks policy has an invalid commit allowlist")
                commits.update(item.strip().lower() for item in allowlisted)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(document)
    return commits


def _verify_decode_depth(executable: str, root: Path) -> tuple[Path, Path]:
    """Prove the static probe can observe the first unsupported decode layer."""
    control = root / "decode-depth-control"
    control.mkdir(mode=0o700)
    payload = b"safe boundary."
    _write_private(
        control / "at-limit.txt",
        _encode_layers(payload, _MAX_DECODE_DEPTH),
    )
    _write_private(
        control / "past-limit.txt",
        _encode_layers(payload, _MAX_DECODE_DEPTH + 1),
    )
    config = root / "decode-depth-probe.toml"
    _write_private(config, _DECODE_PROBE_CONFIG)
    ignore = root / "decode-depth-probe.ignore"
    _write_private(ignore, b"")
    report = root / "decode-depth-control.json"
    result, findings = _run_gitleaks(
        executable,
        control,
        config,
        ignore,
        report,
        max_decode_depth=_MAX_DECODE_DEPTH,
        redact=False,
    )
    exhaustion_files = {item.get("File") for item in _decode_exhaustions(findings)}
    if (
        result.returncode != 1
        or "past-limit.txt" not in exhaustion_files
        or "at-limit.txt" in exhaustion_files
    ):
        raise _ScanError(
            "Gitleaks could not observe the private decode-depth boundary "
            f"(status {result.returncode})"
        )
    return config, ignore


def _require_no_decode_exhaustion(
    label: str,
    executable: str,
    source: Path,
    probe_config: Path,
    ignore: Path,
    report: Path,
    *,
    history_range: str | None = None,
    working_directory: Path | None = None,
) -> None:
    """Fail if source bytes require recursion beyond the governed scan depth."""
    result, findings = _run_gitleaks(
        executable,
        source,
        probe_config,
        ignore,
        report,
        history_range=history_range,
        working_directory=working_directory,
        max_decode_depth=_MAX_DECODE_DEPTH,
        redact=False,
    )
    if result.returncode not in {0, 1}:
        raise _ScanError(
            f"Gitleaks {label} decode-boundary probe failed with "
            f"status {result.returncode}"
        )
    exhausted = _decode_exhaustions(findings)
    if exhausted:
        raise _ScanError(
            f"Gitleaks {label} still contains {len(exhausted)} decodable "
            f"segment(s) at decode-depth:{_MAX_DECODE_DEPTH}"
        )


def _write_binary_controls(control: Path) -> int:
    """Write private binary, archive, and encoded secret controls."""
    control.mkdir(mode=0o700)
    control_prefix = b"".join((b"Qw7eR9tY", b"2uI4oP6a", b"S8dF0gH1", b"jK3lZ5xC"))
    tokens = [control_prefix + secrets.token_hex(12).encode("ascii") for _ in range(3)]
    payload = (
        materializer.SCAN_PREFIX
        + (b"\0" * 4096)
        + b'\napi_key = "'
        + tokens[0]
        + b'"\n'
    )
    _write_private(
        control / "binary-wrapper-control.txt",
        payload,
    )
    with zipfile.ZipFile(
        control / "archive-control.zip",
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        archive.writestr("payload.txt", b'api_key = "' + tokens[1] + b'"\n')
    _write_private(
        control / "decode-control.txt",
        base64.b64encode(b'api_key = "' + tokens[2] + b'"\n'),
    )
    return len(tokens)


def _verify_binary_control(
    executable: str,
    config: Path,
    ignore: Path,
    root: Path,
) -> None:
    """Prove the selected scan view rejects binary/archive/encoded secrets."""
    control = root / "binary-control"
    control_count = _write_binary_controls(control)
    report = root / "binary-control.json"
    result, findings = _run_gitleaks(
        executable,
        control,
        config,
        ignore,
        report,
    )
    generic = [
        item
        for item in findings
        if item.get("RuleID") == "generic-api-key" and item.get("Secret") == "REDACTED"
    ]
    if result.returncode != 1 or len(generic) < control_count:
        raise _ScanError(
            "Gitleaks did not reject the private binary/archive/decode controls "
            f"(status {result.returncode})"
        )


def _require_clean_scan(
    label: str,
    result: subprocess.CompletedProcess[bytes],
    findings: list[dict[str, object]],
) -> None:
    """Require one scanner process and its JSON report to agree on no findings."""
    if result.stdout:
        sys.stdout.buffer.write(result.stdout)
    if result.stderr:
        sys.stderr.buffer.write(result.stderr)
    if result.returncode or findings:
        summary = [
            {
                "file": item.get("File"),
                "rule": item.get("RuleID"),
                "line": item.get("StartLine"),
            }
            for item in findings
        ]
        print(
            f"Gitleaks {label} findings: " + json.dumps(summary, sort_keys=True),
            file=sys.stderr,
        )
        raise _ScanError(
            f"Gitleaks {label} scan failed with status {result.returncode} "
            f"and {len(findings)} finding(s)"
        )


def _exact_commit(repo: Path, revision: str, label: str) -> dict[str, object]:
    """Verify that a supplied revision is the canonical full commit ID."""
    identity = materializer.resolve_commit(repo, revision)
    if identity["commit"] != revision:
        raise _ScanError(f"{label} must be one canonical full commit ID")
    return identity


def _reject_historical_commit_allowlists(
    config_content: bytes,
    historical_revisions: list[str],
) -> None:
    """Reject commit exceptions that tree-mode scans cannot preserve."""
    allowlisted = _policy_commit_allowlists(config_content)
    if allowlisted.intersection(commit.lower() for commit in historical_revisions):
        raise _ScanError(
            "Policy commit allowlists intersect historical tree materialization; "
            "split the range or replace them with content-scoped policy"
        )


def _plan_scan(
    repo: Path,
    revision: str,
    history_base: str,
    policy_revision: str,
    config_path: str,
) -> _ScanPlan:
    """Validate exact identities and build one immutable scan plan."""
    candidate = _exact_commit(repo, revision, "Candidate revision")
    policy = _exact_commit(repo, policy_revision, "Policy revision")
    if policy["commit"] != candidate["commit"]:
        materializer.history_commits(
            repo,
            str(policy["commit"]),
            str(candidate["commit"]),
        )
    if not history_base:
        raise _ScanError("History base must be one explicit canonical commit ID")
    base = _exact_commit(repo, history_base, "History base")
    commits = materializer.history_commits(
        repo,
        str(base["commit"]),
        str(candidate["commit"]),
    )
    if not commits:
        raise _ScanError("History base must precede the candidate commit")
    commit_range = f"{base['commit']}..{candidate['commit']}"
    history_base_identity = str(base["commit"])
    config_content = _config_blob(repo, str(policy["tree"]), config_path)
    historical_revisions = commits[:-1]
    _reject_historical_commit_allowlists(config_content, historical_revisions)
    return _ScanPlan(
        candidate,
        policy,
        commits,
        historical_revisions,
        commit_range,
        history_base_identity,
        config_content,
    )


def _verify_scan_plan_unchanged(
    repo: Path,
    revision: str,
    history_base: str,
    policy_revision: str,
    config_path: str,
    expected: _ScanPlan,
) -> None:
    """Revalidate mutable repository metadata after all long-running scans."""
    observed = _plan_scan(
        repo,
        revision,
        history_base,
        policy_revision,
        config_path,
    )
    expected_identity = (
        expected.candidate,
        expected.policy,
        expected.commits,
        expected.historical_revisions,
        expected.commit_range,
        expected.history_base,
        expected.config_content,
    )
    observed_identity = (
        observed.candidate,
        observed.policy,
        observed.commits,
        observed.historical_revisions,
        observed.commit_range,
        observed.history_base,
        observed.config_content,
    )
    if observed_identity != expected_identity:
        raise _ScanError("Repository scan plan changed while Gitleaks was running")


def _write_scan_policy(repo: Path, root: Path, plan: _ScanPlan) -> tuple[Path, Path]:
    """Write exact committed policy bytes to private files."""
    config = root / "policy.toml"
    _write_private(config, plan.config_content)
    ignore = root / "policy.ignore"
    ignore_content = _optional_config_blob(
        repo,
        str(plan.policy["tree"]),
        ".gitleaksignore",
    )
    _write_private(ignore, ignore_content or b"")
    return config, ignore


def _empty_historical_receipt(candidate: dict[str, object]) -> dict[str, object]:
    """Return a zero-work historical materialization receipt."""
    return {
        "commits": [],
        "trees": [],
        "object_format": candidate["object_format"],
        "source_entries": 0,
        "unique_blob_contexts": 0,
        "bytes": 0,
        "scan_view": True,
        "scan_prefix_bytes": len(materializer.SCAN_PREFIX),
        "scan_files": 0,
        "scan_bytes": 0,
    }


def _materialize_sources(
    repo: Path,
    root: Path,
    plan: _ScanPlan,
) -> _MaterializedSources:
    """Materialize exact candidate and intermediate tree sources."""
    snapshot = root / "tree"
    snapshot.mkdir(mode=0o700)
    candidate_receipt = materializer.materialize(
        repo,
        snapshot,
        str(plan.candidate["commit"]),
        scan_view=True,
    )
    historical_view: Path | None = None
    historical_receipt = _empty_historical_receipt(plan.candidate)
    if plan.historical_revisions:
        historical_view = root / "historical-trees"
        historical_view.mkdir(mode=0o700)
        historical_receipt = materializer.materialize_history_scan(
            repo,
            historical_view,
            plan.historical_revisions,
        )
    history_source = _isolated_history_source(repo, root / "history-repository")
    scanner_work = root / "scanner-work"
    scanner_work.mkdir(mode=0o700)
    return _MaterializedSources(
        snapshot,
        candidate_receipt,
        historical_view,
        historical_receipt,
        history_source,
        scanner_work,
    )


def _empty_probe_receipt() -> dict[str, int]:
    """Return a zero-work compact decode-probe receipt."""
    return {
        "archive_bytes": 0,
        "archive_files": 0,
        "archive_streams": 0,
        "candidate_bytes": 0,
        "candidate_duplicates": 0,
        "candidate_matches": 0,
        "candidate_streams": 0,
        "files": 0,
        "max_candidate_bytes": 0,
        "wrapper_files": 0,
    }


def _prepare_probe_sources(
    root: Path,
    sources: _MaterializedSources,
) -> _ProbeSources:
    """Build compact candidate and historical decode-probe views."""
    candidate_view = root / "candidate-decode-probe"
    candidate_receipt = _prepare_decode_probe_source(
        sources.snapshot,
        candidate_view,
    )
    historical_view: Path | None = None
    historical_receipt = _empty_probe_receipt()
    if sources.historical_view is not None:
        historical_view = root / "historical-decode-probe"
        historical_receipt = _prepare_decode_probe_source(
            sources.historical_view,
            historical_view,
        )
    return _ProbeSources(
        candidate_view,
        candidate_receipt,
        historical_view,
        historical_receipt,
    )


def _verify_probe_sources(
    executable: str,
    root: Path,
    plan: _ScanPlan,
    sources: _MaterializedSources,
    probes: _ProbeSources,
    probe_config: Path,
    probe_ignore: Path,
) -> None:
    """Require every scan source to stay below the decode recursion ceiling."""
    _require_no_decode_exhaustion(
        "committed-history",
        executable,
        sources.history_source,
        probe_config,
        probe_ignore,
        root / "candidate-history-decode-boundary.json",
        history_range=plan.commit_range,
        working_directory=sources.scanner_work,
    )
    if probes.historical_view is not None:
        _require_no_decode_exhaustion(
            "historical-trees",
            executable,
            probes.historical_view,
            probe_config,
            probe_ignore,
            root / "historical-trees-decode-boundary.json",
        )
    _require_no_decode_exhaustion(
        "committed-tree",
        executable,
        probes.candidate_view,
        probe_config,
        probe_ignore,
        root / "candidate-tree-decode-boundary.json",
    )


def _scan_historical_trees(
    executable: str,
    root: Path,
    sources: _MaterializedSources,
    config: Path,
    ignore: Path,
) -> tuple[subprocess.CompletedProcess[bytes] | None, list[dict[str, object]]]:
    """Scan optional intermediate tree views."""
    if sources.historical_view is None:
        return None, []
    return _run_gitleaks(
        executable,
        sources.historical_view,
        config,
        ignore,
        root / "historical-trees.json",
    )


def _require_clean_content_scans(
    history_scan: tuple[subprocess.CompletedProcess[bytes], list[dict[str, object]]],
    historical_scan: tuple[
        subprocess.CompletedProcess[bytes] | None,
        list[dict[str, object]],
    ],
    tree_scan: tuple[subprocess.CompletedProcess[bytes], list[dict[str, object]]],
) -> None:
    """Require all completed content scans to be clean in source order."""
    _require_clean_scan("committed-history", *history_scan)
    historical_result, historical_findings = historical_scan
    if historical_result is not None:
        _require_clean_scan(
            "historical-trees",
            historical_result,
            historical_findings,
        )
    _require_clean_scan("committed-tree", *tree_scan)


def _run_content_scans(
    executable: str,
    root: Path,
    plan: _ScanPlan,
    sources: _MaterializedSources,
    config: Path,
    ignore: Path,
) -> _FindingSets:
    """Run and validate history, intermediate-tree, and candidate-tree scans."""
    history_scan = _run_gitleaks(
        executable,
        sources.history_source,
        config,
        ignore,
        root / "candidate-history.json",
        history_range=plan.commit_range,
        working_directory=sources.scanner_work,
    )
    historical_scan = _scan_historical_trees(
        executable,
        root,
        sources,
        config,
        ignore,
    )
    tree_scan = _run_gitleaks(
        executable,
        sources.snapshot,
        config,
        ignore,
        root / "candidate-tree.json",
    )
    _require_clean_content_scans(history_scan, historical_scan, tree_scan)
    _, history_findings = history_scan
    _, historical_findings = historical_scan
    _, tree_findings = tree_scan
    return _FindingSets(history_findings, historical_findings, tree_findings)


def _render_scan_receipt(
    plan: _ScanPlan,
    sources: _MaterializedSources,
    probes: _ProbeSources,
    findings: _FindingSets,
    config: Path,
    ignore: Path,
) -> dict[str, object]:
    """Render one non-sensitive complete-scan receipt."""
    finding_count = (
        len(findings.history) + len(findings.historical_trees) + len(findings.tree)
    )
    return {
        **sources.candidate_receipt,
        "binary_control": "passed",
        "decode_depth_control": "passed",
        "history_base": plan.history_base,
        "history_range": plan.commit_range,
        "history_commits": plan.commits,
        "history_findings": len(findings.history),
        "gitleaks_max_procs": _GITLEAKS_MAX_PROCS,
        "candidate_decode_probe": probes.candidate_receipt,
        "historical_decode_probe": probes.historical_receipt,
        "historical_tree_findings": len(findings.historical_trees),
        "historical_tree_scan": sources.historical_receipt,
        "tree_findings": len(findings.tree),
        "gitleaks_findings": finding_count,
        "policy_revision": plan.policy["commit"],
        "policy_tree": plan.policy["tree"],
        "gitleaks_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "gitleaks_ignore_sha256": hashlib.sha256(ignore.read_bytes()).hexdigest(),
    }


def _scan_workspace(
    repo: Path,
    root: Path,
    plan: _ScanPlan,
    executable: str,
) -> dict[str, object]:
    """Execute all controls and scans inside one private workspace."""
    root.chmod(0o700)
    config, ignore = _write_scan_policy(repo, root, plan)
    probe_config, probe_ignore = _verify_decode_depth(executable, root)
    _verify_binary_control(executable, config, ignore, root)
    sources = _materialize_sources(repo, root, plan)
    probes = _prepare_probe_sources(root, sources)
    _verify_probe_sources(
        executable,
        root,
        plan,
        sources,
        probes,
        probe_config,
        probe_ignore,
    )
    findings = _run_content_scans(
        executable,
        root,
        plan,
        sources,
        config,
        ignore,
    )
    return _render_scan_receipt(plan, sources, probes, findings, config, ignore)


def scan(
    repo: Path,
    revision: str,
    history_base: str,
    policy_revision: str,
    executable: str,
    config_path: str,
) -> dict[str, object]:
    """Scan one exact committed range and candidate tree.

    Args:
        repo: Repository containing the exact Git objects.
        revision: Exact candidate commit ID.
        history_base: Exact excluded base commit ID.
        policy_revision: Exact trusted policy commit ID.
        executable: Absolute path to the pinned Gitleaks binary.
        config_path: Policy path inside the trusted commit.
    Returns:
        A non-sensitive complete-scan receipt.
    Raises:
        _ScanError: If complete clean coverage cannot be proven.
    """
    executable = _verified_executable(executable)
    plan = _plan_scan(
        repo,
        revision,
        history_base,
        policy_revision,
        config_path,
    )
    with tempfile.TemporaryDirectory(prefix="npa-gitleaks-tree-") as directory:
        root = Path(directory)
        root.chmod(0o700)
        staged_executable, executable_sha256 = _stage_gitleaks_executable(
            executable,
            root,
        )
        receipt = _scan_workspace(repo, root, plan, staged_executable)
        _verify_scan_plan_unchanged(
            repo,
            revision,
            history_base,
            policy_revision,
            config_path,
            plan,
        )
        if _verify_gitleaks_identity(staged_executable) != executable_sha256:
            raise _ScanError("Staged Gitleaks bytes changed while scanning")
    receipt["gitleaks_binary_sha256"] = executable_sha256
    receipt["gitleaks_version"] = _GITLEAKS_VERSION
    return receipt


def _parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--history-base",
        required=True,
        help="exact excluded base commit; empty or implicit history is rejected",
    )
    parser.add_argument(
        "--policy-revision",
        required=True,
        help="exact trusted commit from which scanner policy must be loaded",
    )
    parser.add_argument("--gitleaks-bin", required=True)
    parser.add_argument("--config-path", default=".gitleaks.toml")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one complete committed-tree scan through a private scan view."""
    args = _parser().parse_args(argv)
    try:
        repo = args.repo_root.resolve(strict=True)
        if not repo.is_dir():
            raise _ScanError("Repository root must be an existing directory")
        receipt = scan(
            repo,
            args.revision,
            args.history_base,
            args.policy_revision,
            args.gitleaks_bin,
            args.config_path,
        )
    except (
        json.JSONDecodeError,
        OSError,
        ValueError,
        subprocess.CalledProcessError,
        materializer.SnapshotError,
        _ScanError,
    ) as exc:
        print(f"Gitleaks committed-tree scan failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
