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
import resource
import secrets
import stat
import struct
import subprocess
import sys
import tarfile
import tempfile
import urllib.parse
import warnings
import zipfile
import zlib
from collections.abc import Callable, Iterator
from functools import partial
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
_MAX_ARCHIVE_MEMBERS = 10_000
_MAX_ARCHIVE_MEMBER_BYTES = 256 * 1024 * 1024
_MAX_ARCHIVE_TOTAL_BYTES = 512 * 1024 * 1024
_MAX_ARCHIVE_MEMBER_NAME_BYTES = 64 * 1024
_MAX_ZIP_DIRECTORY_BYTES = 16 * 1024 * 1024
_GITLEAKS_MAX_PROCS = 8
_GITLEAKS_VERSION = "8.28.0"
_TRUSTED_GITLEAKS_SHA256 = {
    # Official v8.28.0 linux_x64 executable inside the published GitHub
    # release tarball.
    (
        "linux",
        "x86_64",
    ): "5fd1b3b0073269484d40078662e921d07427340ab9e6ed526ccd215a565b3298",
}
_GITLEAKS_FRAGMENT_BYTES = 100_000
_GITLEAKS_FRAGMENT_PEEK_BYTES = 25_000
_GITLEAKS_READER_BUFFER_BYTES = 4096
_FRAGMENT_WINDOW_CONTEXT_BYTES = 45_000
_MAX_GITLEAKS_REPORT_BYTES = 256 * 1024 * 1024
_MAX_GITLEAKS_REPORT_TOTAL_BYTES = 512 * 1024 * 1024
_MAX_GITLEAKS_DIAGNOSTIC_BYTES = 64 * 1024 * 1024
_MAX_POLICY_BLOB_BYTES = 16 * 1024 * 1024
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
    ".lzma",
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
_ENCODED_ARCHIVE_MAX_BYTES = 16 * 1024 * 1024
_ENCODED_ARCHIVE_PREFIX_BYTES = 8 * 1024
_EMBEDDED_ARCHIVE_PROBE_CHUNK_BYTES = 16
_IMMEDIATE_EMBEDDED_ARCHIVE_SIGNATURES = (
    b"PK\x03\x04",
    b"PK\x05\x06",
    b"PK\x07\x08",
    b"\xfd7zXZ\x00",
    b"7z\xbc\xaf'\x1c",
    b"Rar!\x1a\x07\x00",
    b"Rar!\x1a\x07\x01\x00",
    b"\x28\xb5\x2f\xfd",
    b"\x04\x22\x4d\x18",
    b"\x02\x21\x4c\x18",
)
_MAX_ENCODED_CANDIDATE_MATCHES = 1_000_000
_MAX_ENCODED_CANDIDATE_FILES = 100_000
_MAX_ENCODED_CANDIDATE_BYTES = 512 * 1024 * 1024
_MAX_EXACT_PATH_SHARDS = 1024
_INERT_ENCODED_ARCHIVE_SHA256 = {
    # Public empty-metadata gzip fixture in test_curobo_image_verifier.py.
    "4f4fb700ef54461cfa02571ae0db9a0dc1e0cdb5577484a6d75e68dc38e8acc1",
}
_BINARY_CONTROL_FILES = (
    "binary-wrapper-control.txt",
    "archive-control.zip!payload.txt",
    "decode-control.txt",
)
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


class _ReportBudget:
    """Bound aggregate Gitleaks report bytes across one complete scan."""

    def __init__(self) -> None:
        self.bytes = 0

    def remaining_bytes(self) -> int:
        """Return the remaining child-write allowance or fail before launch."""
        remaining = _MAX_GITLEAKS_REPORT_TOTAL_BYTES - self.bytes
        if remaining <= 0:
            raise _ScanError("Gitleaks reports exceed the aggregate byte limit")
        return min(remaining, _MAX_GITLEAKS_REPORT_BYTES)

    def add(self, size: int) -> None:
        """Account one completed report before parsing its contents."""
        self.bytes += size
        if self.bytes > _MAX_GITLEAKS_REPORT_TOTAL_BYTES:
            raise _ScanError("Gitleaks reports exceed the aggregate byte limit")


class _ArchiveBudget:
    """Bound archive members, parser-visible structure, and decompressed bytes."""

    def __init__(self) -> None:
        self.members = 0
        self.total_bytes = 0
        self.structural_bytes = 0

    def start_member(self, declared_size: int | None = None) -> None:
        """Reserve one member and reject an already oversized declaration."""
        self.members += 1
        if self.members > _MAX_ARCHIVE_MEMBERS:
            raise _ScanError("Archive exceeds the bounded member count")
        if declared_size is not None and (
            declared_size < 0 or declared_size > _MAX_ARCHIVE_MEMBER_BYTES
        ):
            raise _ScanError("Archive member exceeds the bounded byte limit")

    def add_bytes(self, member_bytes: int, chunk_bytes: int) -> int:
        """Account one decompressed chunk and return the member total."""
        member_bytes += chunk_bytes
        self.total_bytes += chunk_bytes
        if member_bytes > _MAX_ARCHIVE_MEMBER_BYTES:
            raise _ScanError("Archive member exceeds the bounded byte limit")
        if self.total_bytes > _MAX_ARCHIVE_TOTAL_BYTES:
            raise _ScanError("Archive exceeds the aggregate decompressed byte limit")
        return member_bytes

    def add_structural_bytes(self, chunk_bytes: int) -> None:
        """Bound decompressed container metadata before a parser can allocate it."""
        self.structural_bytes += chunk_bytes
        if self.structural_bytes > _MAX_ARCHIVE_TOTAL_BYTES:
            raise _ScanError("Archive exceeds the aggregate structural byte limit")


class _BoundedArchiveStream:
    """Limit bytes exposed to a streaming archive parser."""

    def __init__(
        self,
        stream: BinaryIO,
        budget: _ArchiveBudget,
        writer: _ArchiveMemberProbeWriter | None = None,
    ) -> None:
        self._stream = stream
        self._budget = budget
        self._writer = writer
        self._archive_overlap = b""

    def read(self, size: int = -1) -> bytes:
        """Read at most one byte beyond the remaining budget, then reject it."""
        remaining = _MAX_ARCHIVE_TOTAL_BYTES - self._budget.structural_bytes
        bounded_size = remaining + 1 if size < 0 else min(size, remaining + 1)
        content = self._stream.read(bounded_size)
        self._budget.add_structural_bytes(len(content))
        window = self._archive_overlap + content
        if _contains_non_tar_archive_signature(window):
            raise _ScanError("Nested archive bytes cannot be scanned")
        self._archive_overlap = window[-_ENCODED_ARCHIVE_PREFIX_BYTES:]
        if self._writer is not None:
            self._writer.append(content)
        return content


class _ProbeSourceCounts:
    """Accumulate compact decode-probe source counts."""

    def __init__(self) -> None:
        self.files = 0
        self.wrapper_files = 0
        self.archive_bytes = 0
        self.archive_fragment_window_bytes = 0
        self.archive_fragment_window_files = 0
        self.archive_files = 0
        self.archive_exact_path_files = 0
        self.archive_exact_path_shards = 0
        self.archive_root_ignore_exact_path_exclusions = 0
        self.archive_member_probe_bytes = 0
        self.archive_member_probe_files = 0
        self.archive_member_transcoded_bytes = 0
        self.archive_member_transcoded_files = 0
        self.archive_streams = 0
        self.fragment_window_bytes = 0
        self.fragment_window_files = 0
        self.fragment_exact_path_bytes = 0
        self.fragment_exact_path_files = 0
        self.fragment_exact_path_shards = 0

    def render(
        self,
        candidates: _CandidateProbeWriter,
        archive_budget: _ArchiveBudget,
    ) -> dict[str, int]:
        """Combine source and encoded-candidate counts."""
        return {
            "archive_bytes": self.archive_bytes,
            "archive_decompressed_bytes": archive_budget.total_bytes,
            "archive_fragment_window_bytes": self.archive_fragment_window_bytes,
            "archive_fragment_window_files": self.archive_fragment_window_files,
            "archive_files": self.archive_files,
            "archive_exact_path_files": self.archive_exact_path_files,
            "archive_exact_path_shards": self.archive_exact_path_shards,
            "archive_root_ignore_exact_path_exclusions": (
                self.archive_root_ignore_exact_path_exclusions
            ),
            "archive_member_probe_bytes": self.archive_member_probe_bytes,
            "archive_member_probe_files": self.archive_member_probe_files,
            "archive_member_transcoded_bytes": self.archive_member_transcoded_bytes,
            "archive_member_transcoded_files": self.archive_member_transcoded_files,
            "archive_members": archive_budget.members,
            "archive_streams": self.archive_streams,
            "archive_structural_bytes": archive_budget.structural_bytes,
            "candidate_bytes": candidates.candidate_bytes,
            "candidate_duplicates": candidates.duplicate_candidates,
            "candidate_exact_path_files": candidates.exact_paths.files,
            "candidate_exact_path_shards": candidates.exact_paths.shards,
            "candidate_examined_matches": candidates.examined_matches,
            "candidate_matches": candidates.candidates,
            "candidate_streams": candidates.streams,
            "files": self.files,
            "fragment_window_bytes": self.fragment_window_bytes,
            "fragment_window_files": self.fragment_window_files,
            "fragment_exact_path_bytes": self.fragment_exact_path_bytes,
            "fragment_exact_path_files": self.fragment_exact_path_files,
            "fragment_exact_path_shards": self.fragment_exact_path_shards,
            "max_candidate_bytes": candidates.max_candidate_bytes,
            "wrapper_files": self.wrapper_files,
        }


def _canonical_archive_member_parts(member: str) -> tuple[str, ...]:
    """Return a safe logical member path or reject ambiguous archive syntax."""
    if (
        not member
        or member.startswith("/")
        or "\\" in member
        or any(ord(value) < 32 for value in member)
    ):
        raise _ScanError("Archive member path is not canonical")
    normalized = member.removesuffix("/")
    parts = tuple(normalized.split("/"))
    if not normalized or any(part in {"", ".", ".."} for part in parts):
        raise _ScanError("Archive member path is not canonical")
    return parts


class _ExactPathAllocator:
    """Pack logical archive-member paths into collision-free scan roots."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.full_paths: list[set[tuple[str, ...]]] = []
        self.strict_prefixes: list[set[tuple[str, ...]]] = []
        self.files = 0

    def allocate(self, member: str) -> Path:
        """Return one target whose source-relative path is exactly ``member``."""
        parts = _canonical_archive_member_parts(member)
        return self.allocate_parts(parts)

    def allocate_parts(self, parts: tuple[str, ...]) -> Path:
        """Return one exact-path target for already validated source parts."""
        if not parts or any(part in {"", ".", ".."} for part in parts):
            raise _ScanError("Exact scanner path is not canonical")
        shard = 0
        while shard < len(self.full_paths):
            full = self.full_paths[shard]
            prefixes = self.strict_prefixes[shard]
            ancestors = {parts[:index] for index in range(1, len(parts))}
            if parts not in full and parts not in prefixes and not ancestors & full:
                break
            shard += 1
        if shard >= _MAX_EXACT_PATH_SHARDS:
            raise _ScanError("Archive exact-path views exceed the shard limit")
        if shard == len(self.full_paths):
            self.full_paths.append(set())
            self.strict_prefixes.append(set())
            (self.root / str(shard)).mkdir(mode=0o700, parents=True)
        self.full_paths[shard].add(parts)
        self.strict_prefixes[shard].update(
            parts[:index] for index in range(1, len(parts))
        )
        self.files += 1
        return self.root.joinpath(str(shard), *parts)

    @property
    def shards(self) -> int:
        """Return the number of scanner roots allocated."""
        return len(self.full_paths)


class _ArchiveMemberProbeWriter:
    """Materialize one bounded archive member as an isolated scanner wrapper."""

    def __init__(
        self,
        destination: Path,
        identity: str,
        member: str,
        test_context: bool,
        counts: _ProbeSourceCounts,
        candidates: _CandidateProbeWriter,
        exact_path_target: Path | None = None,
        fragment_exact_paths: _ExactPathAllocator | None = None,
        logical_scan_path: str | None = None,
    ) -> None:
        member_bytes = member.encode("utf-8", errors="surrogatepass")
        if len(member_bytes) > _MAX_ARCHIVE_MEMBER_NAME_BYTES:
            raise _ScanError("Archive member path exceeds the bounded byte limit")
        context_root = (
            destination / "npa/tests/npa-gitleaks-archive-members"
            if test_context
            else destination / "npa-gitleaks-archive-members"
        )
        work_root = destination / ".npa-gitleaks-archive-member-work"
        context_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        work_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.destination = destination
        self.counts = counts
        self.candidates = candidates
        self.test_context = test_context
        logical_parts = _canonical_archive_member_parts(
            member if logical_scan_path is None else logical_scan_path
        )
        logical_path = Path(*logical_parts)
        if member.endswith("/"):
            logical_path /= ".npa-directory-metadata.txt"
            logical_parts += (".npa-directory-metadata.txt",)
        self.wrapper_path = context_root / f"{identity}.txt"
        self.context_path = context_root / "path-context" / identity / logical_path
        self.exact_path_target = exact_path_target
        self.fragment_exact_paths = (
            fragment_exact_paths if exact_path_target is not None else None
        )
        self.logical_parts = logical_parts
        self.context_path.parent.mkdir(mode=0o700, parents=True)
        self.raw_path = work_root / f"{identity}.raw"
        flags = (
            os.O_CREAT
            | os.O_EXCL
            | os.O_WRONLY
            | getattr(
                os,
                "O_NOFOLLOW",
                0,
            )
        )
        self.raw_stream = os.fdopen(os.open(self.raw_path, flags, 0o600), "wb")
        try:
            self.wrapper_stream = os.fdopen(
                os.open(self.wrapper_path, flags, 0o600),
                "wb",
            )
            self.wrapper_stream.write(materializer.SCAN_PREFIX)
            self.wrapper_stream.write(b"\n" + member_bytes + b"\n")
            self.wrapper_stream.write(materializer._SCAN_BOUNDARY)
        except Exception:
            self.raw_stream.close()
            self.raw_path.unlink(missing_ok=True)
            wrapper_stream = getattr(self, "wrapper_stream", None)
            if wrapper_stream is not None:
                wrapper_stream.close()
            self.wrapper_path.unlink(missing_ok=True)
            raise
        self.finished = False

    def append(self, chunk: bytes) -> None:
        """Copy one exact decompressed chunk into the member scan views."""
        self.raw_stream.write(chunk)
        self.wrapper_stream.write(chunk)

    def finish(self) -> None:
        """Finish raw, Unicode, and overlap views for this member."""
        try:
            self.raw_stream.close()
            before_files = self.counts.fragment_window_files
            before_bytes = self.counts.fragment_window_bytes
            _append_fragment_windows(
                self.raw_path,
                self.raw_path.relative_to(self.destination).as_posix(),
                self.destination,
                self.counts,
                test_context=self.test_context,
            )
            self.counts.archive_fragment_window_files += (
                self.counts.fragment_window_files - before_files
            )
            self.counts.archive_fragment_window_bytes += (
                self.counts.fragment_window_bytes - before_bytes
            )
            if self.raw_path.stat().st_size:
                with (
                    self.raw_path.open("rb") as stream,
                    mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as content,
                ):
                    self.candidates.append(content, self.logical_parts)
            self.wrapper_stream.write(materializer._SCAN_BOUNDARY)
            transcoded = materializer._append_unicode(
                self.raw_path,
                self.wrapper_stream,
            )
            self.wrapper_stream.close()
            os.link(self.wrapper_path, self.context_path)
            if self.exact_path_target is not None:
                self.exact_path_target.parent.mkdir(
                    mode=0o700,
                    parents=True,
                    exist_ok=True,
                )
                os.link(self.wrapper_path, self.exact_path_target)
            self.raw_path.unlink()
            before_files = self.counts.fragment_window_files
            before_bytes = self.counts.fragment_window_bytes
            relative = self.wrapper_path.relative_to(self.destination).as_posix()
            _append_fragment_windows(
                self.wrapper_path,
                relative,
                self.destination,
                self.counts,
                test_context=self.test_context,
                exact_paths=self.fragment_exact_paths,
                exact_path_parts=(
                    self.logical_parts
                    if self.fragment_exact_paths is not None
                    else None
                ),
            )
            context_relative = self.context_path.relative_to(
                self.destination
            ).as_posix()
            _append_fragment_windows(
                self.context_path,
                context_relative,
                self.destination,
                self.counts,
                test_context=self.test_context,
            )
            self.counts.archive_fragment_window_files += (
                self.counts.fragment_window_files - before_files
            )
            self.counts.archive_fragment_window_bytes += (
                self.counts.fragment_window_bytes - before_bytes
            )
            wrapper_bytes = self.wrapper_path.stat().st_size
            wrapper_copies = 2 + int(self.exact_path_target is not None)
            self.counts.archive_member_probe_files += wrapper_copies
            if self.exact_path_target is not None:
                self.counts.archive_exact_path_files += 1
            self.counts.archive_member_probe_bytes += wrapper_copies * wrapper_bytes
            if transcoded:
                self.counts.archive_member_transcoded_files += 1
                self.counts.archive_member_transcoded_bytes += transcoded
            self.finished = True
        except Exception:
            self.abort()
            raise

    def abort(self) -> None:
        """Close and remove partial member views after a validation failure."""
        if self.finished:
            return
        for stream in (self.raw_stream, self.wrapper_stream):
            if not stream.closed:
                stream.close()
        self.raw_path.unlink(missing_ok=True)
        self.wrapper_path.unlink(missing_ok=True)
        self.context_path.unlink(missing_ok=True)
        if self.exact_path_target is not None:
            self.exact_path_target.unlink(missing_ok=True)


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
        candidate_exact_path_view: Path,
        historical_view: Path | None,
        historical_receipt: dict[str, object],
        historical_exact_path_views: list[Path],
        commit_metadata_view: Path,
        commit_metadata_receipt: dict[str, object],
        history_source: Path,
        scanner_work: Path,
    ) -> None:
        self.snapshot = snapshot
        self.candidate_receipt = candidate_receipt
        self.candidate_exact_path_view = candidate_exact_path_view
        self.historical_view = historical_view
        self.historical_receipt = historical_receipt
        self.historical_exact_path_views = historical_exact_path_views
        self.commit_metadata_view = commit_metadata_view
        self.commit_metadata_receipt = commit_metadata_receipt
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
        commit_metadata_view: Path,
        commit_metadata_receipt: dict[str, int],
        archive_exact_path_views: list[Path],
    ) -> None:
        self.candidate_view = candidate_view
        self.candidate_receipt = candidate_receipt
        self.historical_view = historical_view
        self.historical_receipt = historical_receipt
        self.commit_metadata_view = commit_metadata_view
        self.commit_metadata_receipt = commit_metadata_receipt
        self.archive_exact_path_views = archive_exact_path_views


class _FindingSets:
    """Collect findings from each independently scanned source."""

    def __init__(
        self,
        history: list[dict[str, object]],
        historical_trees: list[dict[str, object]],
        tree: list[dict[str, object]],
        commit_metadata: list[dict[str, object]],
        candidate_exact_paths: list[dict[str, object]],
        historical_exact_paths: list[dict[str, object]],
        archive_exact_paths: list[dict[str, object]],
        candidate_probe: list[dict[str, object]],
        historical_probe: list[dict[str, object]],
        commit_metadata_probe: list[dict[str, object]],
    ) -> None:
        self.history = history
        self.historical_trees = historical_trees
        self.tree = tree
        self.commit_metadata = commit_metadata
        self.candidate_exact_paths = candidate_exact_paths
        self.historical_exact_paths = historical_exact_paths
        self.archive_exact_paths = archive_exact_paths
        self.candidate_probe = candidate_probe
        self.historical_probe = historical_probe
        self.commit_metadata_probe = commit_metadata_probe


def _config_blob(repo: Path, tree: str, relative_path: str) -> bytes:
    """Read one regular committed config blob without checkout transformations."""
    path = PurePosixPath(relative_path)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise _ScanError("Gitleaks config path must stay inside the committed tree")
    return materializer.read_verified_blob(
        repo,
        tree,
        os.fsencode(path.as_posix()),
        max_bytes=_MAX_POLICY_BLOB_BYTES,
        bounded_error="Gitleaks policy blob exceeds the bounded byte limit",
    )


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
        max_bytes=_MAX_POLICY_BLOB_BYTES,
        bounded_error="Gitleaks policy blob exceeds the bounded byte limit",
    )


def _write_private(path: Path, content: bytes) -> None:
    """Write one new private regular file."""
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(content)


class _CandidateProbeWriter:
    """Isolate every encoded match and retain required logical-path associations."""

    def __init__(
        self,
        destination: Path,
        path_patterns: tuple[re.Pattern[str], ...] = (),
    ) -> None:
        self.destination = destination
        self.path_patterns = path_patterns
        self.exact_paths = _ExactPathAllocator(
            destination / "npa-gitleaks-candidate-exact-paths"
        )
        self.streams = 0
        self.candidates = 0
        self.candidate_bytes = 0
        self.duplicate_candidates = 0
        self.examined_matches = 0
        self.max_candidate_bytes = 0
        self.seen: dict[tuple[int, bytes], Path] = {}
        self.seen_associations: set[tuple[tuple[int, bytes], tuple[str, ...]]] = set()
        self.seen_match_identities: set[tuple[str, int, bytes]] = set()
        self.source_matches: dict[tuple[int, int], tuple[bytes, ...]] = {}

    def _reserve_output(self, payload_bytes: int) -> None:
        """Bound isolated scanner files and their aggregate scanned bytes."""
        if self.streams + self.exact_paths.files >= _MAX_ENCODED_CANDIDATE_FILES:
            raise _ScanError("Encoded candidate files exceed the bounded file count")
        if self.candidate_bytes + payload_bytes > _MAX_ENCODED_CANDIDATE_BYTES:
            raise _ScanError(
                "Encoded candidates exceed the aggregate bounded byte limit"
            )
        self.candidate_bytes += payload_bytes

    def _append_candidate(
        self,
        candidate: bytes,
        logical_parts: tuple[str, ...] | None,
    ) -> None:
        """Write one candidate per file and preserve each relevant exact path."""
        identity = (len(candidate), hashlib.sha256(candidate).digest())
        target = self.seen.get(identity)
        payload = materializer.SCAN_PREFIX + candidate + b"\n\n"
        if target is not None:
            self.duplicate_candidates += 1
        else:
            self._reserve_output(len(payload))
            target = self.destination / f"decode-input-candidates-{self.streams}.txt"
            _write_private(target, payload)
            self.seen[identity] = target
            self.streams += 1
            self.candidates += 1
            self.max_candidate_bytes = max(self.max_candidate_bytes, len(candidate))
        if logical_parts is None:
            return
        logical_path = "/".join(logical_parts)
        if not any(pattern.search(logical_path) for pattern in self.path_patterns):
            return
        association = (identity, logical_parts)
        if association in self.seen_associations:
            return
        self._reserve_output(len(payload))
        exact_target = self.exact_paths.allocate_parts(logical_parts)
        exact_target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.link(target, exact_target)
        self.seen_associations.add(association)

    def _reserve_match(
        self,
        kind: str,
        content: bytes | mmap.mmap,
        start: int,
        end: int,
    ) -> None:
        """Bound distinct codec work while tolerating repeated scan-view bytes."""
        identity = (
            kind,
            end - start,
            hashlib.sha256(content[start:end]).digest(),
        )
        if identity in self.seen_match_identities:
            return
        self.seen_match_identities.add(identity)
        self.examined_matches += 1
        if self.examined_matches > _MAX_ENCODED_CANDIDATE_MATCHES:
            raise _ScanError(
                "Encoded candidate analysis exceeds the bounded match count"
            )

    def append(
        self,
        content: bytes | mmap.mmap,
        logical_parts: tuple[str, ...] | None = None,
        *,
        source_key: tuple[int, int] | None = None,
    ) -> None:
        """Append exact scanner-supported matches without fragment splitting."""
        if source_key is not None and source_key in self.source_matches:
            for candidate in self.source_matches[source_key]:
                self._append_candidate(candidate, logical_parts)
            return
        accepted: list[bytes] = []
        for kind, pattern in _ENCODING_CANDIDATES:
            for match in pattern.finditer(content):  # type: ignore[arg-type]
                start, end = match.span()
                candidate_size = end - start
                payload_bytes = candidate_size + 2
                if candidate_size > _ENCODED_ARCHIVE_MAX_BYTES:
                    raise _ScanError(
                        "Encoded segment exceeds the bounded archive-analysis "
                        "byte limit"
                    )
                self._reserve_match(kind, content, start, end)
                _reject_encoded_archives(bytes(content[start:end]))
                complete_candidate_fits = (
                    payload_bytes + len(materializer.SCAN_PREFIX)
                    <= _PROBE_STREAM_MAX_BYTES
                )
                if not _probe_span_decodes(kind, content, start, end):
                    continue
                if not complete_candidate_fits:
                    raise _ScanError(
                        "Encoded segment exceeds one complete Gitleaks scan fragment"
                    )
                candidate = bytes(content[start:end])
                accepted.append(candidate)
                self._append_candidate(candidate, logical_parts)
        if source_key is not None:
            self.source_matches[source_key] = tuple(accepted)

    def close(self) -> None:
        """Retain a context-manager boundary for caller symmetry."""

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


def _contains_archive_signature(content: bytes) -> bool:
    """Detect a supported or rejected archive hidden after decoded prefix bytes."""
    return (
        _archive_kind(content[:_ARCHIVE_PREFIX_BYTES]) is not None
        or _contains_non_tar_archive_signature(content)
        or b"ustar" in content
    )


def _contains_non_tar_archive_signature(content: bytes) -> bool:
    """Detect nested container signatures that are not TAR's own header marker."""
    return (
        any(
            signature in content for signature in _IMMEDIATE_EMBEDDED_ARCHIVE_SIGNATURES
        )
        or b"\x1f\x8b\x08" in content
        or re.search(rb"BZh[1-9]", content) is not None
    )


def _contains_tar_header(content: bytes) -> bool:
    """Recognize a checksum-valid TAR header at any offset in one window."""
    position = content.find(b"ustar")
    while position >= 0:
        start = position - 257
        end = start + 512
        if start >= 0 and end <= len(content):
            header = content[start:end]
            raw_checksum = header[148:156].strip(b"\0 ")
            try:
                expected = int(raw_checksum or b"0", 8)
            except ValueError:
                expected = -1
            observed = sum(header[:148]) + (8 * ord(" ")) + sum(header[156:])
            if expected == observed:
                return True
        position = content.find(b"ustar", position + 1)
    return False


def _embedded_compressed_streams(
    content: mmap.mmap,
) -> Iterator[tuple[str, int]]:
    """Yield plausible gzip and bzip2 stream offsets."""
    for kind, signature in (("gzip", b"\x1f\x8b\x08"), ("bzip2", b"BZh")):
        position = content.find(signature)
        while position >= 0:
            if kind != "bzip2" or content[position + 3 : position + 4] in b"123456789":
                yield kind, position
            position = content.find(signature, position + 1)


def _embedded_stream_requires_rejection(path: Path, kind: str, offset: int) -> bool:
    """Reject output, complete streams, and bounded-but-ambiguous candidates."""
    decoder = (
        zlib.decompressobj(16 + zlib.MAX_WBITS)
        if kind == "gzip"
        else bz2.BZ2Decompressor()
    )
    with path.open("rb") as stream:
        stream.seek(offset)
        content = stream.read(_PROBE_DECODE_CHUNK_BYTES)
    for position in range(0, len(content), _EMBEDDED_ARCHIVE_PROBE_CHUNK_BYTES):
        chunk = content[position : position + _EMBEDDED_ARCHIVE_PROBE_CHUNK_BYTES]
        try:
            decoded = decoder.decompress(chunk, max_length=1)
        except (EOFError, OSError, zlib.error):
            return False
        if decoded or decoder.eof:
            return True
    return len(content) == _PROBE_DECODE_CHUNK_BYTES


def _contains_embedded_archive(path: Path) -> bool:
    """Detect a real or unbounded container hidden in a nonarchive source."""
    if not path.stat().st_size:
        return False
    with (
        path.open("rb") as stream,
        mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as content,
    ):
        if any(
            content.find(signature) >= 0
            for signature in _IMMEDIATE_EMBEDDED_ARCHIVE_SIGNATURES
        ) or _contains_tar_header(content):
            return True
        for candidate, (kind, offset) in enumerate(
            _embedded_compressed_streams(content),
            start=1,
        ):
            if candidate > _ENCODED_ARCHIVE_MAX_CANDIDATES:
                return True
            if _embedded_stream_requires_rejection(path, kind, offset):
                return True
    return False


def _has_archive_suffix(name: str) -> bool:
    """Return whether a path advertises any supported or rejected archive type."""
    lowered = name.lower()
    return any(
        lowered.endswith(suffix)
        for suffix in _ARCHIVE_SUFFIXES | _REJECTED_ARCHIVE_SUFFIXES
    )


def _inspect_archive_stream(
    stream: BinaryIO,
    budget: _ArchiveBudget,
    member_writer: _ArchiveMemberProbeWriter | None = None,
) -> tuple[bytes, bool]:
    """Verify a bounded stream and detect encoded or nested containers."""
    prefix = bytearray()
    overlap = b""
    contains_archive = False
    member_bytes = 0
    try:
        while chunk := stream.read(_ARCHIVE_READ_BYTES):
            member_bytes = budget.add_bytes(member_bytes, len(chunk))
            if member_writer is not None:
                member_writer.append(chunk)
            if len(prefix) < _ARCHIVE_PREFIX_BYTES:
                missing = _ARCHIVE_PREFIX_BYTES - len(prefix)
                prefix.extend(chunk[:missing])
            window = overlap + chunk
            _reject_encoded_archives(window)
            contains_archive = contains_archive or _contains_archive_signature(window)
            overlap = window[-_ENCODED_ARCHIVE_PREFIX_BYTES:]
        if member_writer is not None:
            member_writer.finish()
    except Exception:
        if member_writer is not None:
            member_writer.abort()
        raise
    return bytes(prefix), contains_archive


def _reject_nested_archive(
    member: str,
    prefix: bytes,
    *,
    contains_archive: bool,
) -> None:
    """Reject nested containers whose complete policy traversal is ambiguous."""
    if (
        _has_archive_suffix(member)
        or _archive_kind(prefix) is not None
        or contains_archive
    ):
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
        elif kind == "unicode":
            if candidate.startswith(b"U+"):
                codepoints = [
                    int.from_bytes(binascii.unhexlify(value[2:]), "big")
                    for value in candidate.split()
                    if value.startswith(b"U+") and len(value) == 6
                ]
            else:
                codepoints = [
                    int(value, 16)
                    for value in re.findall(
                        rb"\\{1,2}u([0-9A-Fa-f]{4})",
                        candidate,
                    )
                ]
            if codepoints:
                decoded.append("".join(chr(value) for value in codepoints).encode())
    except (binascii.Error, UnicodeEncodeError, ValueError):
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
            for match in pattern.finditer(content):
                for decoded in _decode_candidate(kind, match.group()):
                    examined += 1
                    if examined > _ENCODED_ARCHIVE_MAX_CANDIDATES:
                        raise _ScanError(
                            "Encoded archive analysis exceeded its bounded "
                            "candidate count"
                        )
                    contains_archive = _contains_archive_signature(decoded)
                    if contains_archive and (
                        hashlib.sha256(decoded).hexdigest()
                        not in _INERT_ENCODED_ARCHIVE_SHA256
                    ):
                        raise _ScanError(
                            "Encoded archive bytes cannot prove complete "
                            "Gitleaks policy coverage"
                        )
                    if contains_archive:
                        continue
                    queue.append((decoded, depth + 1))


def _preflight_zip_directory(
    path: Path,
    budget: _ArchiveBudget,
) -> tuple[int, int]:
    """Bound the ZIP central directory before ``ZipFile`` allocates its index."""
    size = path.stat().st_size
    tail_size = min(size, 22 + 65_535)
    with path.open("rb") as stream:
        stream.seek(size - tail_size)
        tail = stream.read(tail_size)
    position = tail.rfind(b"PK\x05\x06")
    if position < 0 or position + 22 > len(tail):
        raise _ScanError("ZIP archive is malformed")
    record = tail[position : position + 22]
    disk = int.from_bytes(record[4:6], "little")
    central_disk = int.from_bytes(record[6:8], "little")
    disk_entries = int.from_bytes(record[8:10], "little")
    total_entries = int.from_bytes(record[10:12], "little")
    central_bytes = int.from_bytes(record[12:16], "little")
    central_offset = int.from_bytes(record[16:20], "little")
    comment_bytes = int.from_bytes(record[20:22], "little")
    if position + 22 + comment_bytes != len(tail):
        raise _ScanError("ZIP archive has malformed end metadata")
    if disk or central_disk or disk_entries != total_entries:
        raise _ScanError("Multi-disk ZIP archives cannot be scanned")
    if (
        total_entries == 0xFFFF
        or central_bytes == 0xFFFFFFFF
        or central_offset == 0xFFFFFFFF
    ):
        raise _ScanError("ZIP64 archive metadata cannot be bounded")
    if total_entries > _MAX_ARCHIVE_MEMBERS:
        raise _ScanError("Archive exceeds the bounded member count")
    if central_bytes > _MAX_ZIP_DIRECTORY_BYTES:
        raise _ScanError("ZIP central directory exceeds the bounded byte limit")
    end_offset = size - tail_size + position
    if central_offset + central_bytes != end_offset:
        raise _ScanError("ZIP central directory metadata is inconsistent")
    budget.add_structural_bytes(central_bytes)
    return central_offset, total_entries


def _reject_zip64_extra(extra: bytes) -> None:
    """Reject malformed or ZIP64 extra records in local and central headers."""
    position = 0
    while position < len(extra):
        if len(extra) - position < 4:
            raise _ScanError("ZIP archive has malformed extra metadata")
        field_id, field_size = struct.unpack_from("<HH", extra, position)
        position += 4
        if field_size > len(extra) - position:
            raise _ScanError("ZIP archive has malformed extra metadata")
        if field_id == 0x0001:
            raise _ScanError("ZIP64 archive metadata cannot be bounded")
        position += field_size


def _validate_zip_local_layout(
    path: Path,
    members: list[zipfile.ZipInfo],
    central_offset: int,
) -> None:
    """Require local records to cover every byte before the central directory."""
    expected_offset = 0
    with path.open("rb") as stream:
        for member in sorted(members, key=lambda item: item.header_offset):
            if member.header_offset != expected_offset:
                raise _ScanError("ZIP archive contains unreferenced local bytes")
            stream.seek(member.header_offset)
            header = stream.read(30)
            if len(header) != 30 or header[:4] != b"PK\x03\x04":
                raise _ScanError("ZIP archive has malformed local metadata")
            (
                _signature,
                _extract_version,
                flag_bits,
                compression,
                _modified_time,
                _modified_date,
                crc,
                compressed_size,
                uncompressed_size,
                name_size,
                extra_size,
            ) = struct.unpack("<4s5H3L2H", header)
            if flag_bits != member.flag_bits or compression != member.compress_type:
                raise _ScanError("ZIP local and central metadata disagree")
            raw_name = stream.read(name_size)
            local_extra = stream.read(extra_size)
            if len(raw_name) != name_size or len(local_extra) != extra_size:
                raise _ScanError("ZIP archive has truncated local metadata")
            encoding = "utf-8" if flag_bits & 0x800 else "cp437"
            try:
                local_name = raw_name.decode(encoding)
            except UnicodeDecodeError as exc:
                raise _ScanError("ZIP member name cannot be decoded") from exc
            if local_name != member.filename:
                raise _ScanError("ZIP local and central member names disagree")
            _reject_archive_metadata(raw_name)
            _reject_archive_metadata(local_extra)
            _reject_zip64_extra(local_extra)
            data_end = member.header_offset + 30 + name_size + extra_size
            data_end += member.compress_size
            if data_end > central_offset:
                raise _ScanError("ZIP member exceeds the local-record boundary")
            stream.seek(data_end)
            if flag_bits & 0x08:
                descriptor = stream.read(16)
                expected = struct.pack(
                    "<LLL",
                    member.CRC,
                    member.compress_size,
                    member.file_size,
                )
                if (
                    descriptor.startswith(b"PK\x07\x08")
                    and descriptor[4:16] == expected
                ):
                    data_end += 16
                elif descriptor[:12] == expected:
                    data_end += 12
                else:
                    raise _ScanError("ZIP data descriptor is malformed")
            elif (
                crc != member.CRC
                or compressed_size != member.compress_size
                or uncompressed_size != member.file_size
            ):
                raise _ScanError("ZIP local and central sizes disagree")
            expected_offset = data_end
    if expected_offset != central_offset:
        raise _ScanError("ZIP archive contains unreferenced local bytes")


def _reject_archive_metadata(content: bytes) -> None:
    """Reject encoded or nested containers hidden in archive metadata fields."""
    if not content:
        return
    _reject_encoded_archives(content)
    if _contains_archive_signature(content):
        raise _ScanError("Nested archive metadata cannot be scanned")


def _validate_zip_archive(
    path: Path,
    budget: _ArchiveBudget,
    member_factory: Callable[[str], _ArchiveMemberProbeWriter] | None,
) -> None:
    """Verify one canonical ZIP and reject encrypted or nested containers."""
    central_offset, declared_entries = _preflight_zip_directory(path, budget)
    try:
        with zipfile.ZipFile(path) as archive:
            _reject_archive_metadata(archive.comment)
            members = archive.infolist()
            if len(members) != declared_entries:
                raise _ScanError("ZIP entry count disagrees with central metadata")
            _validate_zip_local_layout(path, members, central_offset)
            names: set[str] = set()
            for member in members:
                if member.filename in names:
                    raise _ScanError("Archive contains a duplicate member path")
                names.add(member.filename)
                budget.start_member(member.file_size)
                if member.flag_bits & 0x1:
                    raise _ScanError("Encrypted archive member cannot be scanned")
                _reject_archive_metadata(member.comment)
                _reject_archive_metadata(member.extra)
                _reject_zip64_extra(member.extra)
                unix_mode = (
                    member.external_attr >> 16 if member.create_system == 3 else 0
                )
                file_type = stat.S_IFMT(unix_mode)
                if file_type not in {0, stat.S_IFREG, stat.S_IFDIR} or (
                    file_type != 0 and member.is_dir() != (file_type == stat.S_IFDIR)
                ):
                    raise _ScanError("Special archive member cannot be scanned")
                member_writer = (
                    None if member_factory is None else member_factory(member.filename)
                )
                if member.is_dir():
                    if member.file_size:
                        if member_writer is not None:
                            member_writer.abort()
                        raise _ScanError(
                            "Non-empty archive directory member cannot be scanned"
                        )
                    if member_writer is not None:
                        member_writer.finish()
                    continue
                try:
                    with archive.open(member) as stream:
                        prefix, contains_archive = _inspect_archive_stream(
                            stream,
                            budget,
                            member_writer,
                        )
                except Exception:
                    if member_writer is not None:
                        member_writer.abort()
                    raise
                _reject_nested_archive(
                    member.filename,
                    prefix,
                    contains_archive=contains_archive,
                )
    except (materializer.SnapshotError, _ScanError):
        raise
    except (
        EOFError,
        NotImplementedError,
        OSError,
        RuntimeError,
        zipfile.BadZipFile,
        zipfile.LargeZipFile,
    ) as exc:
        raise _ScanError("ZIP archive could not be verified") from exc


def _validate_tar_archive(
    path: Path,
    budget: _ArchiveBudget,
    member_factory: Callable[[str], _ArchiveMemberProbeWriter] | None,
    structure_factory: Callable[[str], _ArchiveMemberProbeWriter] | None,
) -> None:
    """Verify one canonical TAR stream and reject special or nested members."""
    lowered = path.name.lower()
    opener = (
        bz2.open
        if lowered.endswith(".tar.bz2")
        else gzip.open
        if lowered.endswith(".tar.gz")
        else lzma.open
        if lowered.endswith(".tar.xz")
        else Path.open
    )
    structure_writer = (
        None if structure_factory is None else structure_factory("<tar-structure>")
    )
    try:
        with opener(path, "rb") as raw_stream:
            bounded_stream = _BoundedArchiveStream(
                raw_stream,
                budget,
                structure_writer,
            )
            with tarfile.open(fileobj=bounded_stream, mode="r|") as archive:
                names: set[str] = set()
                for member in archive:
                    if member.name in names:
                        raise _ScanError("Archive contains a duplicate member path")
                    names.add(member.name)
                    budget.start_member(member.size)
                    member_writer = (
                        None if member_factory is None else member_factory(member.name)
                    )
                    if member.isdir():
                        if member.size:
                            if member_writer is not None:
                                member_writer.abort()
                            raise _ScanError(
                                "Non-empty archive directory member cannot be scanned"
                            )
                        if member_writer is not None:
                            member_writer.finish()
                        continue
                    if not member.isfile():
                        if member_writer is not None:
                            member_writer.abort()
                        raise _ScanError("Special archive member cannot be scanned")
                    try:
                        stream = archive.extractfile(member)
                    except Exception:
                        if member_writer is not None:
                            member_writer.abort()
                        raise
                    if stream is None:
                        if member_writer is not None:
                            member_writer.abort()
                        raise _ScanError("Archive member could not be read")
                    with stream:
                        prefix, contains_archive = _inspect_archive_stream(
                            stream,
                            budget,
                            member_writer,
                        )
                    _reject_nested_archive(
                        member.name,
                        prefix,
                        contains_archive=contains_archive,
                    )
            while bounded_stream.read(_ARCHIVE_READ_BYTES):
                pass
        if structure_writer is not None:
            structure_writer.finish()
    except _ScanError:
        if structure_writer is not None:
            structure_writer.abort()
        raise
    except (EOFError, OSError, lzma.LZMAError, tarfile.TarError) as exc:
        if structure_writer is not None:
            structure_writer.abort()
        raise _ScanError("TAR archive could not be verified") from exc


def _validate_compressed_stream(
    path: Path,
    kind: str,
    budget: _ArchiveBudget,
    member_factory: Callable[[str], _ArchiveMemberProbeWriter] | None,
) -> None:
    """Verify one non-TAR standard-library stream and reject nested archives."""
    openers = {
        "bzip2": bz2.open,
        "gzip": gzip.open,
        "xz": lzma.open,
    }
    budget.start_member()
    member_writer = (
        None if member_factory is None else member_factory("<compressed-stream>")
    )
    try:
        with openers[kind](path, "rb") as stream:
            prefix, contains_archive = _inspect_archive_stream(
                stream,
                budget,
                member_writer,
            )
    except (EOFError, OSError, lzma.LZMAError) as exc:
        if member_writer is not None:
            member_writer.abort()
        raise _ScanError(f"{kind} archive could not be verified") from exc
    _reject_nested_archive(
        "<compressed-stream>",
        prefix,
        contains_archive=contains_archive,
    )


def _validate_archive_file(
    path: Path,
    budget: _ArchiveBudget | None = None,
    member_factory: Callable[[str], _ArchiveMemberProbeWriter] | None = None,
    structure_factory: Callable[[str], _ArchiveMemberProbeWriter] | None = None,
) -> None:
    """Fail closed on archive names or bytes Gitleaks cannot traverse completely."""
    archive_budget = _ArchiveBudget() if budget is None else budget
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
        _validate_zip_archive(path, archive_budget, member_factory)
        return
    if lowered.endswith(".zip"):
        raise _ScanError("ZIP archive is malformed")
    if lowered.endswith(_TAR_SUFFIXES):
        _validate_tar_archive(
            path,
            archive_budget,
            member_factory,
            structure_factory,
        )
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
        _validate_compressed_stream(
            path,
            compressed_kind,
            archive_budget,
            member_factory,
        )
        return
    if kind is not None:
        raise _ScanError(f"Disguised {kind} archive bytes cannot be scanned")
    if _contains_embedded_archive(path):
        raise _ScanError("Embedded archive bytes cannot prove complete coverage")


def _append_probe_file(
    path: Path,
    relative: str,
    is_wrapper: bool,
    candidates: _CandidateProbeWriter,
    logical_parts: tuple[str, ...] | None,
) -> None:
    """Collect codec-valid spans from one source file and its path."""
    if not is_wrapper:
        candidates.append(os.fsencode(relative))
    metadata = path.stat()
    if not metadata.st_size:
        return
    with (
        path.open("rb") as stream,
        mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as content,
    ):
        candidates.append(
            content,
            logical_parts,
            source_key=(metadata.st_dev, metadata.st_ino),
        )


def _wrapper_exact_path_parts(relative: str) -> tuple[str, ...] | None:
    """Recover the exact logical path carried by one context wrapper."""
    for root in _WRAPPER_SCAN_ROOTS:
        prefix = root + "path-context/"
        if not relative.startswith(prefix):
            continue
        _identity, separator, logical_path = relative.removeprefix(prefix).partition(
            "/"
        )
        if not separator or not logical_path or logical_path == ".gitleaksignore":
            return None
        parts = tuple(logical_path.split("/"))
        if any(part in {"", ".", ".."} for part in parts):
            raise _ScanError("Logical wrapper path is not canonical")
        return parts
    return None


def _historical_raw_logical_paths(source: Path) -> dict[str, tuple[str, ...]]:
    """Recover original paths for collision-free historical raw blob copies."""
    recovered: dict[str, tuple[str, ...]] = {}
    for path in source.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(source).as_posix()
        logical_parts = _wrapper_exact_path_parts(relative)
        if logical_parts is None:
            continue
        _prefix, separator, context = relative.partition("/path-context/")
        if not separator:
            continue
        object_id, separator, _logical_path = context.partition("/")
        if not separator:
            continue
        logical_bytes = os.fsencode("/".join(logical_parts))
        path_digest = hashlib.sha256(logical_bytes).hexdigest()
        history_root = (
            "npa/tests/.npa-gitleaks-history"
            if logical_parts[:2] == ("npa", "tests")
            else ".npa-gitleaks-history"
        )
        historical_relative = (
            f"{history_root}/{object_id}-{path_digest}/{logical_parts[-1]}"
        )
        existing = recovered.setdefault(historical_relative, logical_parts)
        if existing != logical_parts:
            raise _ScanError("Historical raw path recovery is ambiguous")
    return recovered


def _gitleaks_fragment_boundaries(path: Path) -> list[int]:
    """Reproduce pinned Gitleaks' 100-KB read plus 25-KB safe-boundary peek."""
    size = path.stat().st_size
    boundaries: list[int] = []
    position = 0
    whitespace = b" \t\n\r"
    with path.open("rb") as stream:
        buffered = b""
        buffered_position = 0

        def read_chunk() -> bytes:
            nonlocal buffered, buffered_position
            if buffered_position < len(buffered):
                content = buffered[buffered_position:]
                buffered = b""
                buffered_position = 0
                return content
            return stream.read(_GITLEAKS_FRAGMENT_BYTES)

        def read_byte() -> int | None:
            nonlocal buffered, buffered_position
            if buffered_position >= len(buffered):
                buffered = stream.read(_GITLEAKS_READER_BUFFER_BYTES)
                buffered_position = 0
                if not buffered:
                    return None
            value = buffered[buffered_position]
            buffered_position += 1
            return value

        while position < size:
            content = read_chunk()
            if not content:
                raise _ScanError("Could not read a complete scanner fragment")
            boundary = position + len(content)
            if boundary >= size:
                break
            newline_count = 0
            if content[-1] in whitespace:
                for value in reversed(content):
                    if value == ord("\n"):
                        newline_count += 1
                        if newline_count >= 2:
                            break
                    elif value not in whitespace:
                        break
            if newline_count < 2:
                newline_count = 0
                peeked = 0
                value = content[-1]
                while True:
                    if value == ord("\n"):
                        newline_count += 1
                        if newline_count >= 2:
                            break
                    elif value not in whitespace:
                        newline_count = 0
                    if peeked >= _GITLEAKS_FRAGMENT_PEEK_BYTES:
                        break
                    next_value = read_byte()
                    if next_value is None:
                        break
                    value = next_value
                    boundary += 1
                    peeked += 1
            if boundary >= size:
                break
            boundaries.append(boundary)
            position = boundary
    return boundaries


def _append_fragment_windows(
    path: Path,
    relative: str,
    destination: Path,
    counts: _ProbeSourceCounts,
    *,
    test_context: bool | None = None,
    exact_paths: _ExactPathAllocator | None = None,
    exact_path_parts: tuple[str, ...] | None = None,
) -> None:
    """Expose scanner-safe overlap around every pinned scanner boundary."""
    if (exact_paths is None) != (exact_path_parts is None):
        raise _ScanError("Fragment exact-path allocation is incomplete")
    size = path.stat().st_size
    if size <= _GITLEAKS_FRAGMENT_BYTES:
        return
    identity = hashlib.sha256(os.fsencode(relative)).hexdigest()
    if test_context is None:
        test_context = relative.startswith("npa/tests/")
    root = (
        destination / "npa/tests/npa-gitleaks-fragment-windows"
        if test_context
        else destination / "npa-gitleaks-fragment-windows"
    )
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    source_name = path.name
    target_root = root / identity
    target_root.mkdir(mode=0o700)
    context_root = target_root / "path-context"
    context_root.mkdir(mode=0o700)
    with path.open("rb") as stream:
        for boundary in _gitleaks_fragment_boundaries(path):
            start = max(0, boundary - _FRAGMENT_WINDOW_CONTEXT_BYTES)
            end = min(size, boundary + _FRAGMENT_WINDOW_CONTEXT_BYTES)
            stream.seek(start)
            content = stream.read(end - start)
            if len(content) != end - start:
                raise _ScanError("Could not read a complete fragment-boundary window")
            content = materializer.SCAN_PREFIX + content
            if len(content) >= _GITLEAKS_FRAGMENT_BYTES:
                raise _ScanError(
                    "Fragment-boundary window exceeds one scanner fragment"
                )
            neutral_target = target_root / f"{boundary}.txt"
            context_target = context_root / f"{boundary}-{source_name}"
            _write_private(neutral_target, content)
            os.link(neutral_target, context_target)
            if exact_paths is not None and exact_path_parts is not None:
                exact_target = exact_paths.allocate_parts(exact_path_parts)
                exact_target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                os.link(neutral_target, exact_target)
                counts.fragment_exact_path_bytes += len(content)
                counts.fragment_exact_path_files += 1
            counts.fragment_window_files += 2
            counts.fragment_window_bytes += 2 * len(content)


def _link_probe_archives(
    path: Path,
    relative: str,
    logical_relative: str,
    destination: Path,
    counts: _ProbeSourceCounts,
    archive_budget: _ArchiveBudget,
    candidates: _CandidateProbeWriter,
    exact_paths: _ExactPathAllocator,
    fragment_exact_paths: _ExactPathAllocator,
) -> None:
    """Link one raw file under every pinned archive-compatible name."""
    source_identity = os.fsencode(relative)

    def member_factory(
        member: str,
        logical_scan_path_override: str | None = None,
    ) -> _ArchiveMemberProbeWriter:
        member_identity = member.encode("utf-8", errors="surrogatepass")
        identity = hashlib.sha256(source_identity + b"\0" + member_identity).hexdigest()
        if logical_scan_path_override is not None:
            logical_scan_path = logical_scan_path_override
        elif member == "<compressed-stream>":
            logical_scan_path = logical_relative
        elif member.startswith("<"):
            logical_scan_path = None
        else:
            logical_scan_path = f"{logical_relative}!{member}"
        exact_path_target = (
            None
            if logical_scan_path is None or member.endswith("/")
            else exact_paths.allocate(logical_scan_path)
        )
        return _ArchiveMemberProbeWriter(
            destination,
            identity,
            member,
            relative.startswith("npa/tests/"),
            counts,
            candidates,
            exact_path_target,
            fragment_exact_paths,
            logical_scan_path,
        )

    _validate_archive_file(
        path,
        archive_budget,
        member_factory=member_factory,
        structure_factory=lambda member: member_factory(
            f"<container>{member}",
            logical_relative,
        ),
    )
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
    path_patterns: tuple[re.Pattern[str], ...] = (),
) -> dict[str, int]:
    """Build a compact view of encoded spans, paths, Unicode, and archives."""
    destination.mkdir(mode=0o700)
    counts = _ProbeSourceCounts()
    archive_budget = _ArchiveBudget()
    exact_paths = _ExactPathAllocator(destination / "npa-gitleaks-archive-exact-paths")
    fragment_exact_paths = _ExactPathAllocator(
        destination / "npa-gitleaks-fragment-exact-paths"
    )
    historical_raw_paths = _historical_raw_logical_paths(source)
    with _CandidateProbeWriter(destination, path_patterns) as candidates:
        for path in sorted(source.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(source).as_posix()
            is_wrapper = relative.startswith(_WRAPPER_SCAN_ROOTS)
            if is_wrapper:
                counts.wrapper_files += 1
            else:
                counts.files += 1
            exact_path_parts = _wrapper_exact_path_parts(relative)
            if (
                exact_path_parts is None
                and not is_wrapper
                and relative != ".gitleaksignore"
            ):
                exact_path_parts = historical_raw_paths.get(
                    relative,
                    _canonical_archive_member_parts(relative),
                )
            _append_probe_file(
                path,
                relative,
                is_wrapper,
                candidates,
                exact_path_parts,
            )
            _append_fragment_windows(
                path,
                relative,
                destination,
                counts,
                exact_paths=(
                    fragment_exact_paths if exact_path_parts is not None else None
                ),
                exact_path_parts=exact_path_parts,
            )
            if is_wrapper:
                continue
            _link_probe_archives(
                path,
                relative,
                "/".join(exact_path_parts or _canonical_archive_member_parts(relative)),
                destination,
                counts,
                archive_budget,
                candidates,
                exact_paths,
                fragment_exact_paths,
            )
    counts.archive_exact_path_shards = exact_paths.shards
    counts.fragment_exact_path_shards = fragment_exact_paths.shards
    return counts.render(candidates, archive_budget)


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
    platform_identity = (sys.platform, os.uname().machine.lower())
    trusted_digest = _TRUSTED_GITLEAKS_SHA256.get(platform_identity)
    if trusted_digest is None:
        raise _ScanError(
            "No trusted Gitleaks executable digest is pinned for "
            f"{platform_identity[0]}/{platform_identity[1]}"
        )
    if not secrets.compare_digest(digest, trusted_digest):
        raise _ScanError(
            "Gitleaks executable does not match the trusted release digest"
        )
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
    max_archive_depth: int,
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
        str(max_archive_depth),
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
    max_archive_depth: int = _MAX_ARCHIVE_DEPTH,
    redact: bool = True,
) -> list[str]:
    """Build one pinned Gitleaks invocation."""
    arguments = _gitleaks_base_arguments(
        executable,
        source_argument,
        config,
        ignore,
        max_decode_depth,
        max_archive_depth,
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


def _load_findings(
    report: Path,
    report_budget: _ReportBudget | None = None,
) -> list[dict[str, object]]:
    """Validate and load one private Gitleaks JSON report."""
    if report.is_symlink() or not report.is_file():
        raise _ScanError(
            f"Gitleaks did not produce a regular JSON report ({report.name})"
        )
    size = report.stat().st_size
    if size > _MAX_GITLEAKS_REPORT_BYTES:
        raise _ScanError("Gitleaks JSON report exceeds the bounded byte limit")
    if report_budget is not None:
        report_budget.add(size)
    loaded = json.loads(report.read_text(encoding="utf-8"))
    if not isinstance(loaded, list) or not all(
        isinstance(item, dict) for item in loaded
    ):
        raise _ScanError("Gitleaks returned an invalid JSON report")
    findings: list[dict[str, object]] = loaded
    return findings


def _limit_scanner_output_files(max_bytes: int) -> None:
    """Bound each regular file the scanner child can write."""
    _soft, hard = resource.getrlimit(resource.RLIMIT_FSIZE)
    bounded = max_bytes
    if hard != resource.RLIM_INFINITY:
        bounded = min(bounded, hard)
    resource.setrlimit(resource.RLIMIT_FSIZE, (bounded, hard))


def _read_scanner_diagnostic(stream: BinaryIO, label: str) -> bytes:
    """Load one bounded scanner diagnostic sink after the child exits."""
    size = os.fstat(stream.fileno()).st_size
    if size > _MAX_GITLEAKS_DIAGNOSTIC_BYTES:
        raise _ScanError(f"Gitleaks {label} exceeds the bounded byte limit")
    stream.seek(0)
    return stream.read()


def _reject_scanner_diagnostics(
    result: subprocess.CompletedProcess[bytes],
    findings: list[dict[str, object]],
) -> None:
    """Fail closed on warnings, errors, or archive-omission diagnostics."""
    archive_failure_terms = (
        b"could not",
        b"error",
        b"exceed",
        b"fail",
        b"invalid",
        b"malform",
        b"password",
        b"skip",
        b"truncat",
        b"unable",
        b"unsupported",
    )
    for line in (result.stdout + b"\n" + result.stderr).splitlines():
        lowered = line.lower()
        fields = lowered.split()
        if b"archive" in lowered and any(
            term in lowered for term in archive_failure_terms
        ):
            raise _ScanError("Gitleaks emitted an archive coverage diagnostic")
        if b"wrn" in fields:
            summary = re.search(rb"\bwrn\s+leaks found:\s*(\d+)\s*$", lowered)
            expected_finding_summary = (
                result.returncode == 1
                and bool(findings)
                and summary is not None
                and int(summary.group(1)) == len(findings)
            )
            if not expected_finding_summary:
                raise _ScanError("Gitleaks emitted a warning diagnostic")
        if b"err" in fields:
            raise _ScanError("Gitleaks emitted an error diagnostic")


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
    max_archive_depth: int = _MAX_ARCHIVE_DEPTH,
    redact: bool = True,
    report_budget: _ReportBudget | None = None,
) -> tuple[subprocess.CompletedProcess[bytes], list[dict[str, object]]]:
    """Run one scan and parse its private JSON report."""
    budget = _ReportBudget() if report_budget is None else report_budget
    report_byte_limit = budget.remaining_bytes()
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
        max_archive_depth=max_archive_depth,
        redact=redact,
    )
    with (
        tempfile.TemporaryFile() as stdout_sink,
        tempfile.TemporaryFile() as stderr_sink,
    ):
        completed = subprocess.run(
            arguments,
            cwd=process_directory,
            check=False,
            stdout=stdout_sink,
            stderr=stderr_sink,
            env=_scan_environment(),
            preexec_fn=partial(_limit_scanner_output_files, report_byte_limit),
        )
        result = subprocess.CompletedProcess(
            completed.args,
            completed.returncode,
            _read_scanner_diagnostic(stdout_sink, "stdout"),
            _read_scanner_diagnostic(stderr_sink, "stderr"),
        )
    try:
        findings = _load_findings(report, budget)
    except _ScanError as error:
        raise _ScanError(f"{error}; scanner status {result.returncode}") from error
    try:
        _reject_scanner_diagnostics(result, findings)
    except _ScanError as error:
        raise _ScanError(f"{source.name}: {error}") from error
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


def _custom_rule_path_patterns(
    content: bytes,
    *,
    content_rules_only: bool = False,
) -> tuple[re.Pattern[str], ...]:
    """Compile supported positive custom-rule paths from trusted policy bytes."""
    document = tomllib.loads(content.decode("utf-8"))
    rules = document.get("rules", [])
    if not isinstance(rules, list):
        raise _ScanError("Gitleaks policy has an invalid custom rule list")
    patterns = []
    for rule in rules:
        if not isinstance(rule, dict):
            raise _ScanError("Gitleaks policy has an invalid custom rule")
        pattern = rule.get("path")
        if pattern is None:
            continue
        if not isinstance(pattern, str):
            raise _ScanError("Gitleaks policy has an invalid custom rule path")
        if content_rules_only and not isinstance(rule.get("regex"), str):
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", FutureWarning)
                patterns.append(re.compile(pattern))
        except (re.error, FutureWarning) as exc:
            raise _ScanError(
                "Cannot prove exact path coverage for an unsupported custom rule path"
            ) from exc
    return tuple(patterns)


def _policy_path_matches_root_ignore(content: bytes) -> bool:
    """Report whether a committed custom rule selects root ``.gitleaksignore``."""
    return any(
        pattern.search(".gitleaksignore")
        for pattern in _custom_rule_path_patterns(content)
    )


def _verify_decode_depth(
    executable: str,
    root: Path,
    report_budget: _ReportBudget,
) -> tuple[Path, Path]:
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
        report_budget=report_budget,
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
    report_budget: _ReportBudget,
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
        max_archive_depth=0,
        redact=False,
        report_budget=report_budget,
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


def _private_control_tokens(count: int) -> list[bytes]:
    """Generate diverse controls that cannot randomly spell scanner stopwords."""
    prefix = b"".join((b"Qw7eR9tY", b"2uI4oP6a", b"S8dF0gH1", b"jK3lZ5xC"))
    randomizer = secrets.SystemRandom()
    tokens: set[bytes] = set()
    while len(tokens) < count:
        letters = randomizer.sample("ABCDEFGHIJKLMNOPQRSTUVWXYZ", 12)
        digits = [randomizer.choice("0123456789") for _ in letters]
        suffix = "".join(
            character
            for pair in zip(letters, digits, strict=True)
            for character in pair
        ).encode("ascii")
        tokens.add(prefix + suffix)
    return sorted(tokens)


def _write_binary_controls(control: Path) -> tuple[str, ...]:
    """Write private binary, archive, and encoded secret controls."""
    control.mkdir(mode=0o700)
    tokens = _private_control_tokens(3)
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
    return _BINARY_CONTROL_FILES


def _verify_binary_control(
    executable: str,
    config: Path,
    ignore: Path,
    root: Path,
    report_budget: _ReportBudget,
) -> None:
    """Prove the selected scan view rejects binary/archive/encoded secrets."""
    control = root / "binary-control"
    expected_files = set(_write_binary_controls(control))
    report = root / "binary-control.json"
    result, findings = _run_gitleaks(
        executable,
        control,
        config,
        ignore,
        report,
        report_budget=report_budget,
    )
    generic = [
        item
        for item in findings
        if item.get("RuleID") == "generic-api-key" and item.get("Secret") == "REDACTED"
    ]
    observed_files = {
        str(item.get("File")) for item in generic if isinstance(item.get("File"), str)
    }
    missing_files = sorted(expected_files - observed_files)
    if result.returncode != 1 or missing_files:
        rendered_files = sorted(observed_files)
        raise _ScanError(
            "Gitleaks did not reject each distinct private "
            "binary/archive/decode control "
            f"(status {result.returncode}; missing {missing_files!r}; "
            f"files {rendered_files!r})"
        )


def _verify_fragment_boundary_control(
    executable: str,
    config: Path,
    ignore: Path,
    root: Path,
    report_budget: _ReportBudget,
) -> None:
    """Prove raw and archive overlap independently catch one safe-boundary split."""
    tokens = _private_control_tokens(2)
    assignment = b"\napi_key =\n\n"
    filler_bytes = _GITLEAKS_FRAGMENT_BYTES - len(assignment)
    for label, token in zip(("raw", "archive"), tokens, strict=True):
        source = root / f"fragment-boundary-{label}-source"
        source.mkdir(mode=0o700)
        payload = (
            b"\0" + (b"!" * (filler_bytes - 1)) + assignment + b'"' + token + b'"\n'
        )
        if label == "raw":
            _write_private(source / "carrier.bin", payload)
        else:
            with zipfile.ZipFile(
                source / "archive-control.zip",
                "w",
                compression=zipfile.ZIP_DEFLATED,
            ) as archive:
                archive.writestr("carrier.bin", payload)
        probe = root / f"fragment-boundary-{label}-probe"
        receipt = _prepare_decode_probe_source(source, probe)
        expected_count = (
            receipt["archive_fragment_window_files"]
            if label == "archive"
            else receipt["fragment_window_files"]
        )
        if expected_count < 1:
            raise _ScanError(
                f"Fragment-boundary {label} control produced no overlap window"
            )
        result, findings = _run_gitleaks(
            executable,
            probe,
            config,
            ignore,
            root / f"fragment-boundary-{label}.json",
            report_budget=report_budget,
        )
        generic = [
            item
            for item in findings
            if item.get("RuleID") == "generic-api-key"
            and item.get("Secret") == "REDACTED"
        ]
        if result.returncode != 1 or not generic:
            raise _ScanError(
                "Gitleaks did not reject the private "
                f"{label} fragment-boundary control "
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
                "file_sha256": hashlib.sha256(
                    str(item.get("File", "")).encode(
                        "utf-8",
                        errors="surrogatepass",
                    )
                ).hexdigest(),
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
    history_budget = materializer._HistoryBudget()
    candidate = _exact_commit(repo, revision, "Candidate revision")
    if not history_base:
        raise _ScanError("History base must be one explicit canonical commit ID")
    base = _exact_commit(repo, history_base, "History base")
    commits = materializer.history_commits(
        repo,
        str(base["commit"]),
        str(candidate["commit"]),
        history_budget=history_budget,
    )
    if not commits:
        raise _ScanError("History base must precede the candidate commit")
    commit_range = f"{base['commit']}..{candidate['commit']}"
    history_base_identity = str(base["commit"])
    policy = _exact_commit(repo, policy_revision, "Policy revision")
    if policy["commit"] != base["commit"]:
        try:
            materializer.history_commits(
                repo,
                str(policy["commit"]),
                str(base["commit"]),
                history_budget=history_budget,
            )
        except materializer.SnapshotError as error:
            raise _ScanError(
                "Policy revision must equal or precede the excluded history base"
            ) from error
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
        "tree_metadata_bytes": 0,
        "tree_paths": 0,
        "tree_path_bytes": 0,
        "exact_path_shards": 0,
        "root_ignore_exact_path_exclusions": 0,
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
    candidate_exact_path_view = root / "tree-exact-paths"
    candidate_receipt = materializer.materialize(
        repo,
        snapshot,
        str(plan.candidate["commit"]),
        scan_view=True,
        exact_path_view=candidate_exact_path_view,
    )
    historical_view: Path | None = None
    historical_exact_path_views: list[Path] = []
    historical_receipt = _empty_historical_receipt(plan.candidate)
    if plan.historical_revisions:
        historical_view = root / "historical-trees"
        historical_view.mkdir(mode=0o700)
        historical_exact_path_root = root / "historical-exact-paths"
        historical_receipt = materializer.materialize_history_scan(
            repo,
            historical_view,
            plan.historical_revisions,
            exact_path_root=historical_exact_path_root,
        )
        historical_exact_path_views = sorted(
            path for path in historical_exact_path_root.iterdir() if path.is_dir()
        )
    commit_metadata_view = root / "commit-metadata"
    commit_metadata_view.mkdir(mode=0o700)
    commit_metadata_receipt = materializer.materialize_commit_metadata(
        repo,
        commit_metadata_view,
        plan.commits,
    )
    history_source = _isolated_history_source(repo, root / "history-repository")
    scanner_work = root / "scanner-work"
    scanner_work.mkdir(mode=0o700)
    return _MaterializedSources(
        snapshot,
        candidate_receipt,
        candidate_exact_path_view,
        historical_view,
        historical_receipt,
        historical_exact_path_views,
        commit_metadata_view,
        commit_metadata_receipt,
        history_source,
        scanner_work,
    )


def _empty_probe_receipt() -> dict[str, int]:
    """Return a zero-work compact decode-probe receipt."""
    return {
        "archive_bytes": 0,
        "archive_decompressed_bytes": 0,
        "archive_fragment_window_bytes": 0,
        "archive_fragment_window_files": 0,
        "archive_files": 0,
        "archive_exact_path_files": 0,
        "archive_exact_path_shards": 0,
        "archive_root_ignore_exact_path_exclusions": 0,
        "archive_member_probe_bytes": 0,
        "archive_member_probe_files": 0,
        "archive_member_transcoded_bytes": 0,
        "archive_member_transcoded_files": 0,
        "archive_members": 0,
        "archive_streams": 0,
        "archive_structural_bytes": 0,
        "candidate_bytes": 0,
        "candidate_duplicates": 0,
        "candidate_exact_path_files": 0,
        "candidate_exact_path_shards": 0,
        "candidate_examined_matches": 0,
        "candidate_matches": 0,
        "candidate_streams": 0,
        "files": 0,
        "fragment_window_bytes": 0,
        "fragment_window_files": 0,
        "fragment_exact_path_bytes": 0,
        "fragment_exact_path_files": 0,
        "fragment_exact_path_shards": 0,
        "max_candidate_bytes": 0,
        "wrapper_files": 0,
    }


def _prepare_probe_sources(
    root: Path,
    plan: _ScanPlan,
    sources: _MaterializedSources,
) -> _ProbeSources:
    """Build compact candidate and historical decode-probe views."""
    path_patterns = _custom_rule_path_patterns(
        plan.config_content,
        content_rules_only=True,
    )
    candidate_view = root / "candidate-decode-probe"
    candidate_receipt = _prepare_decode_probe_source(
        sources.snapshot,
        candidate_view,
        path_patterns,
    )
    historical_view: Path | None = None
    historical_receipt = _empty_probe_receipt()
    if sources.historical_view is not None:
        historical_view = root / "historical-decode-probe"
        historical_receipt = _prepare_decode_probe_source(
            sources.historical_view,
            historical_view,
            path_patterns,
        )
    commit_metadata_view = root / "commit-metadata-decode-probe"
    commit_metadata_receipt = _prepare_decode_probe_source(
        sources.commit_metadata_view,
        commit_metadata_view,
        path_patterns,
    )
    archive_exact_path_views = []
    for probe_view in (
        candidate_view,
        historical_view,
        commit_metadata_view,
    ):
        if probe_view is None:
            continue
        for exact_root_name in (
            "npa-gitleaks-archive-exact-paths",
            "npa-gitleaks-candidate-exact-paths",
            "npa-gitleaks-fragment-exact-paths",
        ):
            exact_root = probe_view / exact_root_name
            if exact_root.is_dir():
                detached_root = root / f"{probe_view.name}-{exact_root_name}"
                exact_root.rename(detached_root)
                archive_exact_path_views.extend(
                    sorted(path for path in detached_root.iterdir() if path.is_dir())
                )
    return _ProbeSources(
        candidate_view,
        candidate_receipt,
        historical_view,
        historical_receipt,
        commit_metadata_view,
        commit_metadata_receipt,
        archive_exact_path_views,
    )


def _verify_root_ignore_exact_path_coverage(
    plan: _ScanPlan,
    sources: _MaterializedSources,
    probes: _ProbeSources,
) -> None:
    """Fail closed when transformed bytes cannot retain required exact paths."""
    exclusions = (
        int(sources.candidate_receipt["root_ignore_exact_path_exclusions"])
        + int(sources.historical_receipt["root_ignore_exact_path_exclusions"])
        + probes.candidate_receipt["archive_root_ignore_exact_path_exclusions"]
        + probes.historical_receipt["archive_root_ignore_exact_path_exclusions"]
        + probes.commit_metadata_receipt["archive_root_ignore_exact_path_exclusions"]
    )
    if exclusions and _policy_path_matches_root_ignore(plan.config_content):
        raise _ScanError(
            "Cannot prove root .gitleaksignore exact path coverage without "
            "letting candidate bytes control the scanner"
        )


def _decode_boundary_view(
    root: Path,
    label: str,
    source: Path,
    expected_streams: int,
) -> Path:
    """Hardlink each isolated encoded stream into a minimal recursion view."""
    destination = root / f"{label}-decode-candidates"
    destination.mkdir(mode=0o700)
    candidates = sorted(source.glob("decode-input-candidates-*.txt"))
    if len(candidates) != expected_streams:
        raise _ScanError("Decode-boundary candidate inventory is incomplete")
    for candidate in candidates:
        metadata = candidate.lstat()
        if candidate.is_symlink() or not stat.S_ISREG(metadata.st_mode):
            raise _ScanError("Decode-boundary candidate must be a regular file")
        os.link(candidate, destination / candidate.name)
    return destination


def _verify_probe_sources(
    executable: str,
    root: Path,
    plan: _ScanPlan,
    sources: _MaterializedSources,
    probes: _ProbeSources,
    probe_config: Path,
    probe_ignore: Path,
    report_budget: _ReportBudget,
) -> None:
    """Require every scan source to stay below the decode recursion ceiling."""
    candidate_boundary = _decode_boundary_view(
        root,
        "candidate",
        probes.candidate_view,
        probes.candidate_receipt["candidate_streams"],
    )
    _require_no_decode_exhaustion(
        "committed-history",
        executable,
        sources.history_source,
        probe_config,
        probe_ignore,
        root / "candidate-history-decode-boundary.json",
        history_range=plan.commit_range,
        working_directory=sources.scanner_work,
        report_budget=report_budget,
    )
    if probes.historical_view is not None:
        historical_boundary = _decode_boundary_view(
            root,
            "historical",
            probes.historical_view,
            probes.historical_receipt["candidate_streams"],
        )
        _require_no_decode_exhaustion(
            "historical-trees",
            executable,
            historical_boundary,
            probe_config,
            probe_ignore,
            root / "historical-trees-decode-boundary.json",
            report_budget=report_budget,
        )
    _require_no_decode_exhaustion(
        "committed-tree",
        executable,
        candidate_boundary,
        probe_config,
        probe_ignore,
        root / "candidate-tree-decode-boundary.json",
        report_budget=report_budget,
    )
    metadata_boundary = _decode_boundary_view(
        root,
        "commit-metadata",
        probes.commit_metadata_view,
        probes.commit_metadata_receipt["candidate_streams"],
    )
    _require_no_decode_exhaustion(
        "commit-metadata",
        executable,
        metadata_boundary,
        probe_config,
        probe_ignore,
        root / "commit-metadata-decode-boundary.json",
        report_budget=report_budget,
    )


def _scan_historical_trees(
    executable: str,
    root: Path,
    sources: _MaterializedSources,
    config: Path,
    ignore: Path,
    report_budget: _ReportBudget,
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
        max_archive_depth=0,
        report_budget=report_budget,
    )


def _prepare_forced_text_repository(
    view: Path,
    config_content: bytes,
) -> str:
    """Commit one private view with candidate attributes and filters disabled."""
    if (view / ".git").exists():
        raise _ScanError("Exact-path scan view already contains Git metadata")
    environment = materializer._git_environment()
    environment.update(
        {
            "GIT_AUTHOR_DATE": "2000-01-01T00:00:00Z",
            "GIT_AUTHOR_EMAIL": "npa-scan@example.invalid",
            "GIT_AUTHOR_NAME": "NPA exact-path scanner",
            "GIT_COMMITTER_DATE": "2000-01-01T00:00:00Z",
            "GIT_COMMITTER_EMAIL": "npa-scan@example.invalid",
            "GIT_COMMITTER_NAME": "NPA exact-path scanner",
        }
    )

    def run(*arguments: str) -> bytes:
        return subprocess.run(
            materializer._git_command(view, *arguments),
            check=True,
            capture_output=True,
            env=environment,
        ).stdout

    run("init", "--quiet", "--template=", "--initial-branch=npa-scan")
    attributes = view / ".git" / "info" / "attributes"
    attributes.parent.mkdir(mode=0o700)
    _write_private(
        attributes,
        b"* -filter -text -working-tree-encoding diff\n",
    )
    run("add", "--force", "--all", "--", ".")
    run(
        "-c",
        "core.hooksPath=/dev/null",
        "commit",
        "--quiet",
        "--allow-empty",
        "-m",
        "NPA exact-path scanner view",
    )
    commit = run("rev-parse", "HEAD").decode("ascii").strip().lower()
    if commit in _policy_commit_allowlists(config_content):
        raise _ScanError(
            "Policy commit allowlist suppresses a generated exact-path scan view"
        )
    return commit


def _run_gitleaks_exact_view(
    executable: str,
    view: Path,
    config: Path,
    ignore: Path,
    report: Path,
    config_content: bytes,
    report_budget: _ReportBudget,
) -> tuple[subprocess.CompletedProcess[bytes], list[dict[str, object]]]:
    """Scan transformed bytes at exact paths without filename archive dispatch."""
    commit = _prepare_forced_text_repository(view, config_content)
    root_ignore = view / ".gitleaksignore"
    if root_ignore.exists() or root_ignore.is_symlink():
        metadata = root_ignore.lstat()
        if root_ignore.is_symlink() or not stat.S_ISREG(metadata.st_mode):
            raise _ScanError("Exact-path root ignore must be a regular file")
        root_ignore.unlink()
    return _run_gitleaks(
        executable,
        view,
        config,
        ignore,
        report,
        history_range=commit,
        report_budget=report_budget,
    )


def _scan_exact_path_views(
    executable: str,
    root: Path,
    views: list[Path],
    label: str,
    config: Path,
    ignore: Path,
    config_content: bytes,
    report_budget: _ReportBudget,
) -> list[tuple[subprocess.CompletedProcess[bytes], list[dict[str, object]]]]:
    """Scan exact paths in filesystem and forced-text Git modes."""
    results = []
    for index, view in enumerate(views):
        root_ignore = view / ".gitleaksignore"
        detached_ignore = root / f".{label}-{index}-root-ignore"
        has_root_ignore = root_ignore.exists() or root_ignore.is_symlink()
        if has_root_ignore:
            metadata = root_ignore.lstat()
            if root_ignore.is_symlink() or not stat.S_ISREG(metadata.st_mode):
                raise _ScanError("Exact-path root ignore must be a regular file")
            root_ignore.rename(detached_ignore)
        try:
            results.append(
                _run_gitleaks(
                    executable,
                    view,
                    config,
                    ignore,
                    root / f"{label}-{index}-filesystem.json",
                    max_archive_depth=0,
                    report_budget=report_budget,
                )
            )
        finally:
            if has_root_ignore:
                detached_ignore.rename(root_ignore)
        results.append(
            _run_gitleaks_exact_view(
                executable,
                view,
                config,
                ignore,
                root / f"{label}-{index}-git.json",
                config_content,
                report_budget=report_budget,
            )
        )
    return results


def _require_clean_content_scans(
    history_scan: tuple[subprocess.CompletedProcess[bytes], list[dict[str, object]]],
    historical_scan: tuple[
        subprocess.CompletedProcess[bytes] | None,
        list[dict[str, object]],
    ],
    tree_scan: tuple[subprocess.CompletedProcess[bytes], list[dict[str, object]]],
    commit_metadata_scan: tuple[
        subprocess.CompletedProcess[bytes],
        list[dict[str, object]],
    ],
    candidate_exact_path_scans: list[
        tuple[subprocess.CompletedProcess[bytes], list[dict[str, object]]]
    ],
    historical_exact_path_scans: list[
        tuple[subprocess.CompletedProcess[bytes], list[dict[str, object]]]
    ],
    archive_exact_path_scans: list[
        tuple[subprocess.CompletedProcess[bytes], list[dict[str, object]]]
    ],
    candidate_probe_scan: tuple[
        subprocess.CompletedProcess[bytes],
        list[dict[str, object]],
    ],
    historical_probe_scan: tuple[
        subprocess.CompletedProcess[bytes] | None,
        list[dict[str, object]],
    ],
    commit_metadata_probe_scan: tuple[
        subprocess.CompletedProcess[bytes],
        list[dict[str, object]],
    ],
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
    _require_clean_scan("commit-metadata", *commit_metadata_scan)
    for scan in candidate_exact_path_scans:
        _require_clean_scan("candidate-exact-path", *scan)
    for scan in historical_exact_path_scans:
        _require_clean_scan("historical-exact-path", *scan)
    for scan in archive_exact_path_scans:
        _require_clean_scan("archive-exact-path", *scan)
    _require_clean_scan("candidate-probe", *candidate_probe_scan)
    historical_probe_result, historical_probe_findings = historical_probe_scan
    if historical_probe_result is not None:
        _require_clean_scan(
            "historical-probe",
            historical_probe_result,
            historical_probe_findings,
        )
    _require_clean_scan("commit-metadata-probe", *commit_metadata_probe_scan)


def _run_content_scans(
    executable: str,
    root: Path,
    plan: _ScanPlan,
    sources: _MaterializedSources,
    probes: _ProbeSources,
    config: Path,
    ignore: Path,
    report_budget: _ReportBudget,
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
        max_archive_depth=0,
        report_budget=report_budget,
    )
    historical_scan = _scan_historical_trees(
        executable,
        root,
        sources,
        config,
        ignore,
        report_budget,
    )
    tree_scan = _run_gitleaks(
        executable,
        sources.snapshot,
        config,
        ignore,
        root / "candidate-tree.json",
        max_archive_depth=0,
        report_budget=report_budget,
    )
    commit_metadata_scan = _run_gitleaks_exact_view(
        executable,
        sources.commit_metadata_view,
        config,
        ignore,
        root / "commit-metadata.json",
        plan.config_content,
        report_budget=report_budget,
    )
    candidate_exact_path_scans = _scan_exact_path_views(
        executable,
        root,
        [sources.candidate_exact_path_view],
        "candidate-exact-path",
        config,
        ignore,
        plan.config_content,
        report_budget,
    )
    historical_exact_path_scans = _scan_exact_path_views(
        executable,
        root,
        sources.historical_exact_path_views,
        "historical-exact-path",
        config,
        ignore,
        plan.config_content,
        report_budget,
    )
    archive_exact_path_scans = _scan_exact_path_views(
        executable,
        root,
        probes.archive_exact_path_views,
        "archive-exact-path",
        config,
        ignore,
        plan.config_content,
        report_budget,
    )
    candidate_probe_scan = _run_gitleaks(
        executable,
        probes.candidate_view,
        config,
        ignore,
        root / "candidate-probe.json",
        max_archive_depth=0,
        report_budget=report_budget,
    )
    historical_probe_scan: tuple[
        subprocess.CompletedProcess[bytes] | None,
        list[dict[str, object]],
    ] = (None, [])
    if probes.historical_view is not None:
        historical_result, historical_findings = _run_gitleaks(
            executable,
            probes.historical_view,
            config,
            ignore,
            root / "historical-probe.json",
            max_archive_depth=0,
            report_budget=report_budget,
        )
        historical_probe_scan = (historical_result, historical_findings)
    commit_metadata_probe_scan = _run_gitleaks(
        executable,
        probes.commit_metadata_view,
        config,
        ignore,
        root / "commit-metadata-probe.json",
        max_archive_depth=0,
        report_budget=report_budget,
    )
    _require_clean_content_scans(
        history_scan,
        historical_scan,
        tree_scan,
        commit_metadata_scan,
        candidate_exact_path_scans,
        historical_exact_path_scans,
        archive_exact_path_scans,
        candidate_probe_scan,
        historical_probe_scan,
        commit_metadata_probe_scan,
    )
    _, history_findings = history_scan
    _, historical_findings = historical_scan
    _, tree_findings = tree_scan
    _, commit_metadata_findings = commit_metadata_scan
    candidate_exact_path_findings = [
        finding
        for _result, findings in candidate_exact_path_scans
        for finding in findings
    ]
    historical_exact_path_findings = [
        finding
        for _result, findings in historical_exact_path_scans
        for finding in findings
    ]
    archive_exact_path_findings = [
        finding
        for _result, findings in archive_exact_path_scans
        for finding in findings
    ]
    _, candidate_probe_findings = candidate_probe_scan
    _, historical_probe_findings = historical_probe_scan
    _, commit_metadata_probe_findings = commit_metadata_probe_scan
    return _FindingSets(
        history_findings,
        historical_findings,
        tree_findings,
        commit_metadata_findings,
        candidate_exact_path_findings,
        historical_exact_path_findings,
        archive_exact_path_findings,
        candidate_probe_findings,
        historical_probe_findings,
        commit_metadata_probe_findings,
    )


def _render_scan_receipt(
    plan: _ScanPlan,
    sources: _MaterializedSources,
    probes: _ProbeSources,
    findings: _FindingSets,
    config: Path,
    ignore: Path,
    report_budget: _ReportBudget,
) -> dict[str, object]:
    """Render one non-sensitive complete-scan receipt."""
    finding_count = (
        len(findings.history)
        + len(findings.historical_trees)
        + len(findings.tree)
        + len(findings.commit_metadata)
        + len(findings.candidate_exact_paths)
        + len(findings.historical_exact_paths)
        + len(findings.archive_exact_paths)
        + len(findings.candidate_probe)
        + len(findings.historical_probe)
        + len(findings.commit_metadata_probe)
    )
    return {
        **sources.candidate_receipt,
        "binary_control": "passed",
        "decode_depth_control": "passed",
        "fragment_boundary_control": "passed",
        "history_base": plan.history_base,
        "history_range": plan.commit_range,
        "history_commits": plan.commits,
        "history_findings": len(findings.history),
        "gitleaks_max_procs": _GITLEAKS_MAX_PROCS,
        "gitleaks_fragment_bytes": _GITLEAKS_FRAGMENT_BYTES,
        "gitleaks_report_bytes": report_budget.bytes,
        "gitleaks_report_byte_limit": _MAX_GITLEAKS_REPORT_TOTAL_BYTES,
        "candidate_decode_probe": probes.candidate_receipt,
        "historical_decode_probe": probes.historical_receipt,
        "commit_metadata_decode_probe": probes.commit_metadata_receipt,
        "historical_tree_findings": len(findings.historical_trees),
        "historical_tree_scan": sources.historical_receipt,
        "tree_findings": len(findings.tree),
        "commit_metadata_findings": len(findings.commit_metadata),
        "candidate_exact_path_findings": len(findings.candidate_exact_paths),
        "historical_exact_path_findings": len(findings.historical_exact_paths),
        "archive_exact_path_findings": len(findings.archive_exact_paths),
        "commit_metadata_probe_findings": len(findings.commit_metadata_probe),
        "commit_metadata_scan": sources.commit_metadata_receipt,
        "candidate_probe_findings": len(findings.candidate_probe),
        "historical_probe_findings": len(findings.historical_probe),
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
    report_budget = _ReportBudget()
    config, ignore = _write_scan_policy(repo, root, plan)
    probe_config, probe_ignore = _verify_decode_depth(
        executable,
        root,
        report_budget,
    )
    _verify_binary_control(executable, config, ignore, root, report_budget)
    _verify_fragment_boundary_control(
        executable,
        config,
        ignore,
        root,
        report_budget,
    )
    sources = _materialize_sources(repo, root, plan)
    probes = _prepare_probe_sources(root, plan, sources)
    _verify_root_ignore_exact_path_coverage(plan, sources, probes)
    _verify_probe_sources(
        executable,
        root,
        plan,
        sources,
        probes,
        probe_config,
        probe_ignore,
        report_budget,
    )
    findings = _run_content_scans(
        executable,
        root,
        plan,
        sources,
        probes,
        config,
        ignore,
        report_budget,
    )
    return _render_scan_receipt(
        plan,
        sources,
        probes,
        findings,
        config,
        ignore,
        report_budget,
    )


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
