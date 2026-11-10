"""Run Gitleaks over exact committed history and every candidate-tree blob."""

from __future__ import annotations

import argparse
import base64
import binascii
import bz2
import gzip
import hashlib
import importlib.util
import io
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
import zipfile
import zlib
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import partial
from pathlib import Path, PurePosixPath
from typing import BinaryIO, NamedTuple, Self

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
_MAX_NATIVE_HISTORY_INPUT_BYTES = 2 * 1024 * 1024 * 1024
_MAX_NATIVE_HISTORY_DIFF_METADATA_BYTES = 512 * 1024 * 1024
_MAX_NATIVE_HISTORY_DIFF_COMMIT_BYTES = 64 * 1024 * 1024
_NATIVE_HISTORY_SCAN_INVOCATIONS = 2
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
_MAX_XZ_DECOMPRESS_MEMORY_BYTES = 256 * 1024 * 1024
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
_ARCHIVE_SIGNATURE_OVERLAP_BYTES = max(
    _ENCODED_ARCHIVE_PREFIX_BYTES,
    _PROBE_DECODE_CHUNK_BYTES,
)
_EMBEDDED_ARCHIVE_PROBE_CHUNK_BYTES = 16
_MAX_EMBEDDED_TAR_CHECKSUM_CANDIDATES = 100_000
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
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PNG_ADAM7_PASSES = (
    (0, 0, 8, 8),
    (4, 0, 8, 8),
    (0, 4, 4, 8),
    (2, 0, 4, 4),
    (0, 2, 2, 4),
    (1, 0, 2, 2),
    (0, 1, 1, 2),
)
_MAX_PNG_CHUNKS = 100_000
# tarfile.nti() decodes through the first NUL and delegates text syntax to
# int(..., 8), whose accepted ASCII forms include str.strip() whitespace,
# signs, 0o prefixes, and underscores. Keep discovery a parser-complete
# superset; _tar_header_layout_plausible() performs the exact parse.
_TAR_TEXT_CHECKSUM_FIELD = re.compile(
    rb"(?=([+\-0-7_oO\x09-\x0d\x1c-\x20]{8}))"
    rb"(?=[+\-0-7_oO\x09-\x0d\x1c-\x20]{0,7}[0-7])"
)
_TAR_NUL_TERMINATED_CHECKSUM_FIELD = re.compile(
    rb"(?=([+\-0-7_oO\x09-\x0d\x1c-\x20]{1,7}\x00))"
    rb"(?=[+\-0-7_oO\x09-\x0d\x1c-\x20]{0,6}[0-7])"
)
_TAR_BASE256_CHECKSUM_MARKER = re.compile(rb"[\x80\xff]")
_ZLIB_COMPRESSION_METHOD_MARKER = re.compile(rb"[\x08\x18\x28\x38\x48\x58\x68\x78]")
_TAR_STRUCTURAL_TYPES = frozenset(tarfile.SUPPORTED_TYPES) | {
    tarfile.SOLARIS_XHDTYPE,
    tarfile.XGLTYPE,
    tarfile.XHDTYPE,
}
_MAX_ENCODED_CANDIDATE_MATCHES = 1_000_000
_MAX_ENCODED_CANDIDATE_FILES = 100_000
_MAX_ENCODED_CANDIDATE_BYTES = 512 * 1024 * 1024
_MAX_ENCODED_DERIVED_OUTPUT_BYTES = (
    _PROBE_STREAM_MAX_BYTES - len(materializer.SCAN_PREFIX) - 2
)
_MAX_ENCODED_DERIVED_ANALYSIS_BYTES = _MAX_ENCODED_CANDIDATE_BYTES
_MAX_DIRECT_COMPRESSED_CANDIDATES = _MAX_ENCODED_CANDIDATE_MATCHES
_MAX_EXACT_PATH_HISTORY_BYTES = _MAX_NATIVE_HISTORY_INPUT_BYTES
# Each candidate, historical, or commit-metadata probe has four independently
# bounded exact-path populations. Keep deterministic batching from restoring
# one scanner process per view.
_MAX_EXACT_PATH_HISTORY_BATCHES = 12
# Exact-path variants are compacted into independent roots in bounded
# forced-text histories, so their count is bounded by generated files.
_MAX_EXACT_PATH_SHARDS = _MAX_ENCODED_CANDIDATE_FILES
_INERT_ENCODED_ARCHIVE_SHA256 = {
    # Public empty-metadata gzip fixture in test_curobo_image_verifier.py.
    "4f4fb700ef54461cfa02571ae0db9a0dc1e0cdb5577484a6d75e68dc38e8acc1",
    # Public one-pixel PNG fixtures used by smoke, browser, workflow, and USD tests.
    "4b5c5c92cec3b23e6a294fc0eea43234ef5126c5a64f4c6c531ac8430ab0b844",
    "6ae7d93ee3b1d0f0f5a583464e14faf9ed653501de0edcdda1a6bc80145e18d7",
    "431ced6916a2a21a156e38701afe55bbd7f88969fbbfc56d7fe099d47f265460",
    "6b1048f8a6d40bac0b2954c18fefa40c4ea7a96120fc2e54b7317c0e43c2bbec",
    # Public PNG frame payloads embedded in sim2real_sample.mcap.
    "1dd175255efb8aea73319d143f249bf0affb74e9552a00e323990a29b2fb77a4",
    "96aed7926711b877936a49cc737402eded5f59cfc02cc90b5833a824525498a8",
    # Public split IDAT/IEND suffix used by the Sim2Real review controls.
    "480850d1783069d8dab7d0d0b426ebf3c46bc85d852c774278de81547690e677",
}
_INERT_STRICT_PNG_MEDIA_SHA256 = {
    # Public Workbench architecture PNG. Its authenticated ancillary caBX
    # payload has one incidental RFC 1950 candidate that emits a byte.
    "bae993cd4604b6efa7ebf28365fab41d16e9369e22a3e8f7065a4c4d5a597451",
}
_INERT_EMBEDDED_MEDIA_SHA256 = _INERT_STRICT_PNG_MEDIA_SHA256 | {
    # Public LeRobot labeled MP4 and its reviewable Rerun recording. Both contain
    # thousands of incidental RFC 1950 headers, including malformed streams that
    # emit bytes, but their complete immutable public payloads are known inert.
    "803cc4e2bbf2260838354f7475ced9ab9417ab54a99ff62d3e4532e51b894a2b",
    "c6ea539976f254eec55da12d7ac2674272c74b06fa676ffd1fdcac8c392e3218",
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


class _NativeHistoryInput(NamedTuple):
    """Describe one native history scan's bounded Git inputs."""

    patch_bytes: int
    changed_blob_bytes_upper_bound: int
    diff_metadata_bytes: int

    @property
    def bytes_per_scan(self) -> int:
        """Return the patch and possible archive-blob bytes per scan."""
        return self.patch_bytes + self.changed_blob_bytes_upper_bound


class _DirectCompressedAnalysis(NamedTuple):
    """Account one ordinary carrier's bounded embedded-stream analysis."""

    outputs: tuple[bytes, ...]
    compressed_candidates: int
    rfc1950_candidates: int
    complete_empty_streams: int
    invalid_streams: int
    output_occurrences: int
    emitted_bytes: int
    fdict_candidates: int
    fdict_complete_streams: int
    fdict_malformed_streams: int
    fdict_unresolved_streams: int
    nested_compressed_candidates: int
    nested_rfc1950_candidates: int
    nested_complete_empty_streams: int
    nested_invalid_streams: int
    nested_output_occurrences: int
    nested_emitted_bytes: int
    nested_max_depth: int
    nested_fdict_candidates: int
    nested_fdict_complete_streams: int
    nested_fdict_malformed_streams: int
    nested_fdict_unresolved_streams: int


class _RFC1950DecodeResult(NamedTuple):
    """Classify bounded output and FDICT coverage from one RFC 1950 start."""

    output: bytes | None
    strict_rejection: bool
    fdict_status: str | None


class _RecursiveOutputNode(NamedTuple):
    """Retain one output and the immutable identities traversed to reach it."""

    content: bytes
    depth: int
    ancestors: frozenset[tuple[int, bytes]]
    canonical_source: bool


class _CanonicalOutput(NamedTuple):
    """Retain output bytes and whether every source boundary was canonical."""

    content: bytes
    canonical_source: bool


class _RecursiveCompressedAccumulator:
    """Charge nested decompression before deduplicating retained outputs."""

    def __init__(self, max_candidates: int, max_emitted_bytes: int) -> None:
        self.max_candidates = max_candidates
        self.max_emitted_bytes = max_emitted_bytes
        self.compressed_candidates = 0
        self.rfc1950_candidates = 0
        self.complete_empty_streams = 0
        self.invalid_streams = 0
        self.output_occurrences = 0
        self.emitted_bytes = 0
        self.fdict_candidates = 0
        self.fdict_complete_streams = 0
        self.fdict_malformed_streams = 0
        self.fdict_unresolved_streams = 0
        self.max_depth = 0
        self.pending: list[_RecursiveOutputNode] = []
        self.outputs: list[bytes] = []
        self.seen_outputs: set[tuple[int, bytes]] = set()

    def add_root(
        self,
        output: bytes,
        source_identity: tuple[int, bytes],
        *,
        canonical_source: bool = True,
    ) -> None:
        """Queue one first-level output without counting it as nested work."""
        identity = _derived_output_identity(output)
        if identity == source_identity:
            raise _ScanError("Recursive RFC 1950 output cycle cannot be proven safe")
        self.pending.append(
            _RecursiveOutputNode(
                output,
                0,
                frozenset((source_identity, identity)),
                canonical_source,
            )
        )

    def charge_candidate(self, kind: str, depth: int, *, fdict: bool) -> bool:
        """Charge one nested compressed start and identify RFC 1950 work."""
        self.compressed_candidates += 1
        if self.compressed_candidates > self.max_candidates:
            raise _ScanError(
                "Direct compressed-stream analysis exceeded the bounded candidate count"
            )
        if kind != "zlib":
            return False
        self.rfc1950_candidates += 1
        if fdict:
            self.fdict_candidates += 1
        if depth >= _MAX_DECODE_DEPTH:
            raise _ScanError("Recursive RFC 1950 output exceeds the decode-depth bound")
        return True

    def charge_output(
        self,
        result: _RFC1950DecodeResult,
        parent: _RecursiveOutputNode,
        *,
        canonical: bool,
    ) -> None:
        """Account for one nested decoder result and queue every occurrence."""
        if result.fdict_status is not None:
            if result.fdict_status == "complete":
                self.fdict_complete_streams += 1
            elif result.fdict_status == "malformed":
                self.fdict_malformed_streams += 1
            else:
                self.fdict_unresolved_streams += 1
                if canonical:
                    raise _ScanError(
                        "Canonical recursive RFC 1950 output requires "
                        "an external dictionary"
                    )
        output = result.output
        if output is None:
            self.invalid_streams += 1
            return
        if not output:
            self.complete_empty_streams += 1
            return
        depth = parent.depth + 1
        identity = _derived_output_identity(output)
        self.output_occurrences += 1
        self.emitted_bytes += len(output)
        if self.emitted_bytes > self.max_emitted_bytes:
            raise _ScanError(
                "Recursive RFC 1950 derived analysis exceeds the aggregate "
                "bounded byte limit"
            )
        if identity in parent.ancestors:
            raise _ScanError("Recursive RFC 1950 output cycle cannot be proven safe")
        self.max_depth = max(self.max_depth, depth)
        if identity not in self.seen_outputs:
            self.outputs.append(output)
            self.seen_outputs.add(identity)
        self.pending.append(
            _RecursiveOutputNode(
                output,
                depth,
                parent.ancestors | {identity},
                canonical,
            )
        )


class _EncodedCandidateAnalysis(NamedTuple):
    """Return materialized encoded outputs with their recursive accounting."""

    outputs: tuple[bytes, ...]
    emitted_bytes: int
    recursive: _RecursiveCompressedAccumulator
    fdict_candidates: int
    fdict_complete_streams: int
    fdict_malformed_streams: int
    fdict_unresolved_streams: int


class _DirectCompressedAccumulator:
    """Collect first-level RFC 1950 work before recursive closure."""

    def __init__(self, max_candidates: int, max_emitted_bytes: int) -> None:
        self.max_candidates = max_candidates
        self.max_emitted_bytes = max_emitted_bytes
        self.compressed_candidates = 0
        self.rfc1950_candidates = 0
        self.complete_empty_streams = 0
        self.invalid_streams = 0
        self.output_occurrences = 0
        self.emitted_bytes = 0
        self.fdict_candidates = 0
        self.fdict_complete_streams = 0
        self.fdict_malformed_streams = 0
        self.fdict_unresolved_streams = 0
        self.outputs: list[bytes] = []
        self.output_work: list[bytes] = []
        self.output_work_canonical_sources: list[bool] = []
        self.seen_outputs: set[tuple[int, bytes]] = set()

    def inspect(
        self,
        content: bytes | mmap.mmap,
        kind: str,
        absolute_offset: int,
    ) -> None:
        """Classify one direct compressed-stream candidate."""
        self.compressed_candidates += 1
        if self.compressed_candidates > self.max_candidates:
            raise _ScanError(
                "Direct compressed-stream analysis exceeded the bounded candidate count"
            )
        if kind != "zlib":
            if _embedded_stream_requires_rejection(content, kind, absolute_offset):
                raise _ScanError(
                    "Embedded archive bytes cannot prove complete coverage"
                )
            return
        self.rfc1950_candidates += 1
        if _zlib_header_uses_dictionary(content, absolute_offset):
            self.fdict_candidates += 1
        result = _bounded_rfc1950_result(content, absolute_offset)
        if result.fdict_status == "complete":
            self.fdict_complete_streams += 1
        elif result.fdict_status == "malformed":
            self.fdict_malformed_streams += 1
        elif result.fdict_status == "unresolved":
            self.fdict_unresolved_streams += 1
            if absolute_offset == 0:
                raise _ScanError(
                    "Canonical RFC 1950 stream requires an external dictionary"
                )
        self._retain(
            result.output,
            canonical_source=absolute_offset == 0,
        )

    def _retain(
        self,
        output: bytes | None,
        *,
        canonical_source: bool,
    ) -> None:
        """Charge one first-level decoder result before output deduplication."""
        if output is None:
            self.invalid_streams += 1
            return
        if not output:
            self.complete_empty_streams += 1
            return
        self.output_occurrences += 1
        self.emitted_bytes += len(output)
        if self.emitted_bytes > self.max_emitted_bytes:
            raise _ScanError(
                "Direct RFC 1950 derived analysis exceeds the aggregate "
                "bounded byte limit"
            )
        self.output_work.append(output)
        self.output_work_canonical_sources.append(canonical_source)
        identity = _derived_output_identity(output)
        if identity not in self.seen_outputs:
            self.outputs.append(output)
            self.seen_outputs.add(identity)

    def result(self, content: bytes | mmap.mmap) -> _DirectCompressedAnalysis:
        """Close recursive work and return first-level and nested accounting."""
        recursive = _RecursiveCompressedAccumulator(
            self.max_candidates - self.compressed_candidates,
            self.max_emitted_bytes - self.emitted_bytes,
        )
        if self.output_work:
            recursive = _recursive_direct_outputs(
                tuple(self.output_work),
                _derived_output_identity(content),
                self.max_candidates - self.compressed_candidates,
                self.max_emitted_bytes - self.emitted_bytes,
                canonical_sources=tuple(self.output_work_canonical_sources),
            )
        for output in recursive.outputs:
            identity = _derived_output_identity(output)
            if identity not in self.seen_outputs:
                self.outputs.append(output)
                self.seen_outputs.add(identity)
        return _DirectCompressedAnalysis(
            tuple(self.outputs),
            self.compressed_candidates,
            self.rfc1950_candidates,
            self.complete_empty_streams,
            self.invalid_streams,
            self.output_occurrences,
            self.emitted_bytes,
            self.fdict_candidates,
            self.fdict_complete_streams,
            self.fdict_malformed_streams,
            self.fdict_unresolved_streams,
            recursive.compressed_candidates,
            recursive.rfc1950_candidates,
            recursive.complete_empty_streams,
            recursive.invalid_streams,
            recursive.output_occurrences,
            recursive.emitted_bytes,
            recursive.max_depth,
            recursive.fdict_candidates,
            recursive.fdict_complete_streams,
            recursive.fdict_malformed_streams,
            recursive.fdict_unresolved_streams,
        )


def _empty_direct_compressed_analysis() -> _DirectCompressedAnalysis:
    """Return the zero-work receipt for an empty ordinary source file."""
    return _DirectCompressedAnalysis(
        outputs=(),
        compressed_candidates=0,
        rfc1950_candidates=0,
        complete_empty_streams=0,
        invalid_streams=0,
        output_occurrences=0,
        emitted_bytes=0,
        fdict_candidates=0,
        fdict_complete_streams=0,
        fdict_malformed_streams=0,
        fdict_unresolved_streams=0,
        nested_compressed_candidates=0,
        nested_rfc1950_candidates=0,
        nested_complete_empty_streams=0,
        nested_invalid_streams=0,
        nested_output_occurrences=0,
        nested_emitted_bytes=0,
        nested_max_depth=0,
        nested_fdict_candidates=0,
        nested_fdict_complete_streams=0,
        nested_fdict_malformed_streams=0,
        nested_fdict_unresolved_streams=0,
    )


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
    """Bound logical members, parser-visible structure, and decompressed bytes."""

    def __init__(self) -> None:
        self.members = 0
        self.total_bytes = 0
        self.structural_bytes = 0
        self.tar_header_candidates = 0

    def start_member(self, declared_size: int | None = None) -> None:
        """Reserve one logical member and reject an oversized declaration."""
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

    def examine_tar_header_candidate(self) -> bool:
        """Return whether checksum-candidate work exceeded its aggregate limit."""
        self.tar_header_candidates += 1
        return self.tar_header_candidates > _MAX_EMBEDDED_TAR_CHECKSUM_CANDIDATES


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
        self._archive_overlap = window[-_ARCHIVE_SIGNATURE_OVERLAP_BYTES:]
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
            "derived_bytes": candidates.derived_bytes,
            "derived_analysis_bytes": candidates.derived_analysis_bytes,
            "derived_duplicates": candidates.duplicate_derived_outputs,
            "derived_exact_path_bytes": candidates.derived_exact_path_bytes,
            "derived_exact_path_files": candidates.derived_exact_paths.files,
            "derived_exact_path_identity_reuses": (
                candidates.derived_exact_path_identity_reuses
            ),
            "derived_exact_path_shards": candidates.derived_exact_paths.shards,
            "derived_outputs": candidates.derived_outputs,
            "derived_streams": candidates.derived_streams,
            "derived_transcoded_bytes": candidates.derived_transcoded_bytes,
            "derived_transcoded_files": candidates.derived_transcoded_files,
            "encoded_fdict_candidates": candidates.encoded_fdict_candidates,
            "encoded_fdict_complete_streams": (
                candidates.encoded_fdict_complete_streams
            ),
            "encoded_fdict_malformed_streams": (
                candidates.encoded_fdict_malformed_streams
            ),
            "encoded_fdict_unresolved_streams": (
                candidates.encoded_fdict_unresolved_streams
            ),
            "direct_compressed_candidates": candidates.direct_compressed_candidates,
            "direct_complete_empty_streams": (candidates.direct_complete_empty_streams),
            "direct_duplicate_carriers": candidates.direct_duplicate_carriers,
            "direct_emitted_bytes": candidates.direct_emitted_bytes,
            "direct_fdict_candidates": candidates.direct_fdict_candidates,
            "direct_fdict_complete_streams": (candidates.direct_fdict_complete_streams),
            "direct_fdict_malformed_streams": (
                candidates.direct_fdict_malformed_streams
            ),
            "direct_fdict_unresolved_streams": (
                candidates.direct_fdict_unresolved_streams
            ),
            "direct_invalid_streams": candidates.direct_invalid_streams,
            "direct_output_occurrences": candidates.direct_output_occurrences,
            "direct_rfc1950_candidates": candidates.direct_rfc1950_candidates,
            "direct_source_carriers": candidates.direct_source_carriers,
            "files": self.files,
            "fragment_window_bytes": self.fragment_window_bytes,
            "fragment_window_files": self.fragment_window_files,
            "fragment_exact_path_bytes": self.fragment_exact_path_bytes,
            "fragment_exact_path_files": self.fragment_exact_path_files,
            "fragment_exact_path_shards": self.fragment_exact_path_shards,
            "max_candidate_bytes": candidates.max_candidate_bytes,
            "max_derived_bytes": candidates.max_derived_bytes,
            "nested_complete_empty_streams": (candidates.nested_complete_empty_streams),
            "nested_compressed_candidates": candidates.nested_compressed_candidates,
            "nested_emitted_bytes": candidates.nested_emitted_bytes,
            "nested_fdict_candidates": candidates.nested_fdict_candidates,
            "nested_fdict_complete_streams": (candidates.nested_fdict_complete_streams),
            "nested_fdict_malformed_streams": (
                candidates.nested_fdict_malformed_streams
            ),
            "nested_fdict_unresolved_streams": (
                candidates.nested_fdict_unresolved_streams
            ),
            "nested_invalid_streams": candidates.nested_invalid_streams,
            "nested_max_depth": candidates.nested_max_depth,
            "nested_output_occurrences": candidates.nested_output_occurrences,
            "nested_rfc1950_candidates": candidates.nested_rfc1950_candidates,
            "wrapper_files": self.wrapper_files,
        }


def _canonical_archive_member_parts(member: str) -> tuple[str, ...]:
    """Return a safe logical member path or reject ambiguous archive syntax."""
    if (
        not member
        or member.startswith("/")
        or member.endswith("/")
        or "\\" in member
        or any(ord(value) < 32 for value in member)
    ):
        raise _ScanError("Archive member path is not canonical")
    normalized = member
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
        self.next_shards: dict[tuple[str, ...], int] = {}
        self.files = 0

    def allocate(self, member: str) -> Path:
        """Return one target whose source-relative path is exactly ``member``."""
        parts = _canonical_archive_member_parts(member)
        return self.allocate_parts(parts)

    def allocate_parts(self, parts: tuple[str, ...]) -> Path:
        """Return one exact-path target for already validated source parts."""
        if not parts or any(part in {"", ".", ".."} for part in parts):
            raise _ScanError("Exact scanner path is not canonical")
        shard = self.next_shards.get(parts, 0)
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
        self.next_shards[parts] = shard + 1
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
        raw_exact_path_target: Path | None = None,
        transformed_exact_path_target: Path | None = None,
        fragment_exact_paths: _ExactPathAllocator | None = None,
        logical_scan_path: str | None = None,
        *,
        retain_direct_rfc1950_outputs: bool = False,
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
        self.retain_direct_rfc1950_outputs = retain_direct_rfc1950_outputs
        if (raw_exact_path_target is None) != (transformed_exact_path_target is None):
            raise _ScanError("Archive exact-path representations are incomplete")
        logical_parts = _canonical_archive_member_parts(
            member if logical_scan_path is None else logical_scan_path
        )
        logical_path = Path(*logical_parts)
        self.wrapper_path = context_root / f"{identity}.txt"
        self.context_path = context_root / "path-context" / identity / logical_path
        self.raw_exact_path_target = raw_exact_path_target
        self.transformed_exact_path_target = transformed_exact_path_target
        self.fragment_exact_paths = (
            fragment_exact_paths if raw_exact_path_target is not None else None
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
                if self.retain_direct_rfc1950_outputs:
                    self.candidates.append_direct_rfc1950_outputs(
                        self.raw_path,
                        self.logical_parts,
                    )
            self.wrapper_stream.write(materializer._SCAN_BOUNDARY)
            transcoded = materializer._append_unicode(
                self.raw_path,
                self.wrapper_stream,
            )
            self.wrapper_stream.close()
            os.link(self.wrapper_path, self.context_path)
            if self.raw_exact_path_target is not None:
                self.raw_exact_path_target.parent.mkdir(
                    mode=0o700,
                    parents=True,
                    exist_ok=True,
                )
                os.link(self.raw_path, self.raw_exact_path_target)
            if self.transformed_exact_path_target is not None:
                transformed_exact_bytes, _transformed_materialized = (
                    materializer.materialize_transformed_exact_path(
                        self.raw_path,
                        self.transformed_exact_path_target,
                    )
                )
            else:
                transformed_exact_bytes = 0
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
            wrapper_copies = 2
            exact_path_copies = int(self.raw_exact_path_target is not None)
            exact_path_copies += int(self.transformed_exact_path_target is not None)
            self.counts.archive_member_probe_files += wrapper_copies + exact_path_copies
            self.counts.archive_exact_path_files += exact_path_copies
            raw_bytes = (
                self.raw_exact_path_target.stat().st_size
                if self.raw_exact_path_target is not None
                else 0
            )
            transformed_bytes = (
                transformed_exact_bytes
                if self.transformed_exact_path_target is not None
                else 0
            )
            self.counts.archive_member_probe_bytes += (
                (wrapper_copies * wrapper_bytes) + raw_bytes + transformed_bytes
            )
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
        if self.raw_exact_path_target is not None:
            self.raw_exact_path_target.unlink(missing_ok=True)
        if self.transformed_exact_path_target is not None:
            self.transformed_exact_path_target.unlink(missing_ok=True)


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
        candidate_exact_path_views: list[Path],
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
        self.candidate_exact_path_views = candidate_exact_path_views
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
        transformed_exact_path_views: list[Path],
    ) -> None:
        self.candidate_view = candidate_view
        self.candidate_receipt = candidate_receipt
        self.historical_view = historical_view
        self.historical_receipt = historical_receipt
        self.commit_metadata_view = commit_metadata_view
        self.commit_metadata_receipt = commit_metadata_receipt
        self.transformed_exact_path_views = transformed_exact_path_views


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
        transformed_exact_paths: list[dict[str, object]],
        candidate_probe: list[dict[str, object]],
        historical_probe: list[dict[str, object]],
        commit_metadata_probe: list[dict[str, object]],
        candidate_exact_path_scan_invocations: int,
        historical_exact_path_scan_invocations: int,
        transformed_exact_path_scan_invocations: int,
    ) -> None:
        self.history = history
        self.historical_trees = historical_trees
        self.tree = tree
        self.commit_metadata = commit_metadata
        self.candidate_exact_paths = candidate_exact_paths
        self.historical_exact_paths = historical_exact_paths
        self.transformed_exact_paths = transformed_exact_paths
        self.candidate_probe = candidate_probe
        self.historical_probe = historical_probe
        self.commit_metadata_probe = commit_metadata_probe
        self.candidate_exact_path_scan_invocations = (
            candidate_exact_path_scan_invocations
        )
        self.historical_exact_path_scan_invocations = (
            historical_exact_path_scan_invocations
        )
        self.transformed_exact_path_scan_invocations = (
            transformed_exact_path_scan_invocations
        )


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


def _replacement_unicode_projection(
    path: Path,
    content: bytes,
    error: Exception,
) -> bytes:
    """Expose valid text around malformed Unicode while a raw view keeps every byte."""
    if not isinstance(error.__cause__, (UnicodeDecodeError, UnicodeEncodeError)):
        raise error
    encoding = materializer._whole_unicode_encoding(path)
    if encoding is None:
        raise error
    return content.decode(encoding, errors="replace").encode("utf-8")


class _CandidateProbeWriter:
    """Isolate every encoded match and retain required logical-path associations."""

    def __init__(
        self,
        destination: Path,
        preserve_exact_paths: bool = False,
    ) -> None:
        self.destination = destination
        self.preserve_exact_paths = preserve_exact_paths
        self.exact_paths = _ExactPathAllocator(
            destination / "npa-gitleaks-candidate-exact-paths"
        )
        self.derived_exact_paths = _ExactPathAllocator(
            destination / "npa-gitleaks-derived-exact-paths"
        )
        self.streams = 0
        self.candidates = 0
        self.candidate_bytes = 0
        self.duplicate_candidates = 0
        self.derived_streams = 0
        self.derived_outputs = 0
        self.derived_bytes = 0
        self.derived_analysis_bytes = 0
        self.duplicate_derived_outputs = 0
        self.derived_exact_path_bytes = 0
        self.derived_exact_path_identity_reuses = 0
        self.derived_transcoded_bytes = 0
        self.derived_transcoded_files = 0
        self.encoded_fdict_candidates = 0
        self.encoded_fdict_complete_streams = 0
        self.encoded_fdict_malformed_streams = 0
        self.encoded_fdict_unresolved_streams = 0
        self.direct_source_carriers = 0
        self.direct_duplicate_carriers = 0
        self.direct_compressed_candidates = 0
        self.direct_rfc1950_candidates = 0
        self.direct_complete_empty_streams = 0
        self.direct_invalid_streams = 0
        self.direct_output_occurrences = 0
        self.direct_emitted_bytes = 0
        self.direct_fdict_candidates = 0
        self.direct_fdict_complete_streams = 0
        self.direct_fdict_malformed_streams = 0
        self.direct_fdict_unresolved_streams = 0
        self.nested_compressed_candidates = 0
        self.nested_rfc1950_candidates = 0
        self.nested_complete_empty_streams = 0
        self.nested_invalid_streams = 0
        self.nested_output_occurrences = 0
        self.nested_emitted_bytes = 0
        self.nested_max_depth = 0
        self.nested_fdict_candidates = 0
        self.nested_fdict_complete_streams = 0
        self.nested_fdict_malformed_streams = 0
        self.nested_fdict_unresolved_streams = 0
        self.examined_matches = 0
        self.max_candidate_bytes = 0
        self.max_derived_bytes = 0
        self.seen: dict[tuple[int, bytes], Path] = {}
        self.seen_associations: set[tuple[tuple[int, bytes], tuple[str, ...]]] = set()
        self.seen_derived: dict[tuple[int, bytes], Path] = {}
        self.seen_derived_associations: set[
            tuple[tuple[int, bytes], tuple[str, ...]]
        ] = set()
        self.analyzed_matches: dict[
            tuple[str, int, bytes, bool],
            tuple[bytes | None, tuple[bytes, ...]],
        ] = {}
        self.interned_derived_outputs: dict[tuple[int, bytes], bytes] = {}
        self.source_matches: dict[
            tuple[int, int],
            tuple[tuple[bytes | None, tuple[bytes, ...]], ...],
        ] = {}
        self.direct_source_analyses: dict[
            tuple[int, int],
            _DirectCompressedAnalysis,
        ] = {}

    def _reserve_output(self, payload_bytes: int) -> None:
        """Bound isolated scanner files and their aggregate scanned bytes."""
        generated_files = (
            self.streams
            + self.exact_paths.files
            + self.derived_streams
            + self.derived_exact_paths.files
        )
        if generated_files >= _MAX_ENCODED_CANDIDATE_FILES:
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
        if not self.preserve_exact_paths:
            return
        association = (identity, logical_parts)
        if association in self.seen_associations:
            return
        self._reserve_output(len(payload))
        exact_target = self.exact_paths.allocate_parts(logical_parts)
        exact_target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.link(target, exact_target)
        self.seen_associations.add(association)

    def _append_derived_output(
        self,
        output: bytes,
        logical_parts: tuple[str, ...] | None,
    ) -> None:
        """Write fully accounted decompressor output into generic and exact views."""
        if not output:
            return
        identity = (len(output), hashlib.sha256(output).digest())
        target = self.seen_derived.get(identity)
        association = (
            None
            if logical_parts is None or not self.preserve_exact_paths
            else (identity, logical_parts)
        )
        needs_association = (
            association is not None
            and association not in self.seen_derived_associations
        )
        payload: bytes | None = None
        transformed: bytes | None = None
        transcoded = False
        if target is None or needs_association:
            payload, transformed, transcoded = self._render_derived_views(
                output,
                identity,
            )
        if target is not None:
            self.duplicate_derived_outputs += 1
        else:
            assert payload is not None
            self._reserve_output(len(payload))
            target = (
                self.destination / f"decode-derived-output-{self.derived_streams}.txt"
            )
            _write_private(target, payload)
            self.seen_derived[identity] = target
            self.derived_streams += 1
            self.derived_outputs += 1
            self.derived_bytes += len(output)
            self.max_derived_bytes = max(self.max_derived_bytes, len(output))
            if transcoded:
                self.derived_transcoded_files += 1
                self.derived_transcoded_bytes += len(payload) - (
                    len(materializer.SCAN_PREFIX)
                    + (2 * len(materializer._SCAN_BOUNDARY))
                    + len(output)
                )
        if not needs_association:
            return
        assert association is not None and logical_parts is not None
        assert transformed is not None
        self._reserve_output(len(output))
        raw_target = self.derived_exact_paths.allocate_parts(logical_parts)
        raw_target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        _write_private(raw_target, output)
        self.derived_exact_path_bytes += len(output)
        if transformed == output:
            self.derived_exact_path_identity_reuses += 1
        else:
            self._reserve_output(len(transformed))
            transformed_target = self.derived_exact_paths.allocate_parts(logical_parts)
            transformed_target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            _write_private(transformed_target, transformed)
            self.derived_exact_path_bytes += len(transformed)
        self.seen_derived_associations.add(association)

    def _render_derived_views(
        self,
        output: bytes,
        identity: tuple[int, bytes],
    ) -> tuple[bytes, bytes, bool]:
        """Render bounded generic and source-start Unicode projections."""
        work = self.destination / ".npa-gitleaks-derived-work"
        work.mkdir(mode=0o700, exist_ok=True)
        suffix = identity[1].hex()
        raw_path = work / f"{suffix}.raw"
        transformed_path = work / f"{suffix}.transformed"
        try:
            _write_private(raw_path, output)
            generic = io.BytesIO()
            generic.write(materializer.SCAN_PREFIX)
            generic.write(materializer._SCAN_BOUNDARY)
            generic.write(output)
            generic.write(materializer._SCAN_BOUNDARY)
            replacement_projection: bytes | None = None
            try:
                transcoded_bytes = materializer._append_unicode(raw_path, generic)
            except materializer.SnapshotError as error:
                replacement_projection = _replacement_unicode_projection(
                    raw_path,
                    output,
                    error,
                )
                generic.write(replacement_projection)
                transcoded_bytes = len(replacement_projection)
            payload = generic.getvalue()
            if len(payload) > _PROBE_STREAM_MAX_BYTES:
                raise _ScanError(
                    "Encoded RFC 1950 derived scan view exceeds one complete "
                    "Gitleaks scan fragment"
                )
            if replacement_projection is None:
                _transformed_bytes, transformed_materialized = (
                    materializer.materialize_transformed_exact_path(
                        raw_path,
                        transformed_path,
                    )
                )
                transformed = transformed_path.read_bytes()
            else:
                transformed = replacement_projection
                transformed_materialized = True
            projections: dict[tuple[int, bytes], bytes] = {}
            if transcoded_bytes:
                appended = payload[-transcoded_bytes:]
                projections[(len(appended), hashlib.sha256(appended).digest())] = (
                    appended
                )
            if transformed_materialized:
                projections[
                    (len(transformed), hashlib.sha256(transformed).digest())
                ] = transformed
            for projection in projections.values():
                _reject_derived_projection_archives(projection)
            return (
                payload,
                transformed,
                bool(transcoded_bytes or transformed_materialized),
            )
        finally:
            raw_path.unlink(missing_ok=True)
            transformed_path.unlink(missing_ok=True)
            work.rmdir()

    def _intern_derived_outputs(
        self,
        outputs: tuple[bytes, ...],
    ) -> tuple[bytes, ...]:
        """Share exact output objects across duplicate encoded carriers."""
        interned: list[bytes] = []
        for output in outputs:
            identity = (len(output), hashlib.sha256(output).digest())
            canonical = self.interned_derived_outputs.setdefault(identity, output)
            interned.append(canonical)
        return tuple(interned)

    def _analyze_match(
        self,
        kind: str,
        encoded_candidate: bytes,
        *,
        canonical_source: bool,
    ) -> tuple[
        tuple[str, int, bytes, bool],
        tuple[bytes | None, tuple[bytes, ...]],
    ]:
        """Analyze each distinct codec match once and cache its bounded result."""
        identity = (
            kind,
            len(encoded_candidate),
            hashlib.sha256(encoded_candidate).digest(),
            canonical_source,
        )
        cached = self.analyzed_matches.get(identity)
        if cached is not None:
            return identity, cached
        self.examined_matches += 1
        if self.examined_matches > _MAX_ENCODED_CANDIDATE_MATCHES:
            raise _ScanError(
                "Encoded candidate analysis exceeds the bounded match count"
            )
        remaining_analysis_bytes = (
            _MAX_ENCODED_DERIVED_ANALYSIS_BYTES - self.derived_analysis_bytes
        )
        encoded_analysis = _encoded_candidate_analysis(
            encoded_candidate,
            max_emitted_bytes=remaining_analysis_bytes,
            canonical_source=canonical_source,
        )
        discovered_outputs = encoded_analysis.outputs
        emitted_bytes = encoded_analysis.emitted_bytes
        self.derived_analysis_bytes += emitted_bytes
        self.encoded_fdict_candidates += encoded_analysis.fdict_candidates
        self.encoded_fdict_complete_streams += encoded_analysis.fdict_complete_streams
        self.encoded_fdict_malformed_streams += encoded_analysis.fdict_malformed_streams
        self.encoded_fdict_unresolved_streams += (
            encoded_analysis.fdict_unresolved_streams
        )
        recursive = encoded_analysis.recursive
        self.nested_compressed_candidates += recursive.compressed_candidates
        self.nested_rfc1950_candidates += recursive.rfc1950_candidates
        self.nested_complete_empty_streams += recursive.complete_empty_streams
        self.nested_invalid_streams += recursive.invalid_streams
        self.nested_output_occurrences += recursive.output_occurrences
        self.nested_emitted_bytes += recursive.emitted_bytes
        self.nested_max_depth = max(self.nested_max_depth, recursive.max_depth)
        self.nested_fdict_candidates += recursive.fdict_candidates
        self.nested_fdict_complete_streams += recursive.fdict_complete_streams
        self.nested_fdict_malformed_streams += recursive.fdict_malformed_streams
        self.nested_fdict_unresolved_streams += recursive.fdict_unresolved_streams
        derived_outputs = self._intern_derived_outputs(discovered_outputs)
        complete_candidate_fits = (
            len(encoded_candidate) + 2 + len(materializer.SCAN_PREFIX)
            <= _PROBE_STREAM_MAX_BYTES
        )
        scan_candidate: bytes | None = None
        if _probe_span_decodes(
            kind,
            encoded_candidate,
            0,
            len(encoded_candidate),
        ):
            if not complete_candidate_fits:
                raise _ScanError(
                    "Encoded segment exceeds one complete Gitleaks scan fragment"
                )
            scan_candidate = encoded_candidate
        result = (scan_candidate, derived_outputs)
        self.analyzed_matches[identity] = result
        return identity, result

    def append(
        self,
        content: bytes | mmap.mmap,
        logical_parts: tuple[str, ...] | None = None,
        *,
        source_key: tuple[int, int] | None = None,
    ) -> None:
        """Append exact scanner-supported matches without fragment splitting."""
        if source_key is not None and source_key in self.source_matches:
            for candidate, derived_outputs in self.source_matches[source_key]:
                if candidate is not None:
                    self._append_candidate(candidate, logical_parts)
                for output in derived_outputs:
                    self._append_derived_output(output, logical_parts)
            return
        accepted: list[tuple[bytes | None, tuple[bytes, ...]]] = []
        source_identities: set[tuple[str, int, bytes, bool]] = set()
        for kind, pattern in _ENCODING_CANDIDATES:
            for match in pattern.finditer(content):  # type: ignore[arg-type]
                start, end = match.span()
                candidate_size = end - start
                if candidate_size > _ENCODED_ARCHIVE_MAX_BYTES:
                    raise _ScanError(
                        "Encoded segment exceeds the bounded archive-analysis "
                        "byte limit"
                    )
                encoded_candidate = bytes(content[start:end])
                identity, analyzed = self._analyze_match(
                    kind,
                    encoded_candidate,
                    canonical_source=start == 0,
                )
                if identity in source_identities:
                    continue
                source_identities.add(identity)
                scan_candidate, derived_outputs = analyzed
                for output in derived_outputs:
                    self._append_derived_output(output, logical_parts)
                accepted.append(analyzed)
                if scan_candidate is not None:
                    self._append_candidate(scan_candidate, logical_parts)
        if source_key is not None:
            self.source_matches[source_key] = tuple(accepted)

    def _analyze_direct_path(
        self,
        path: Path,
        size: int,
    ) -> _DirectCompressedAnalysis:
        """Analyze one new inode under the remaining shared work budget."""
        if not size:
            return _empty_direct_compressed_analysis()
        remaining_candidates = (
            _MAX_DIRECT_COMPRESSED_CANDIDATES
            - self.direct_compressed_candidates
            - self.nested_compressed_candidates
        )
        remaining_bytes = (
            _MAX_ENCODED_DERIVED_ANALYSIS_BYTES - self.derived_analysis_bytes
        )
        with (
            path.open("rb") as stream,
            mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as content,
        ):
            analysis = _direct_compressed_outputs(
                content,
                max_candidates=remaining_candidates,
                max_emitted_bytes=remaining_bytes,
            )
        return analysis._replace(outputs=self._intern_derived_outputs(analysis.outputs))

    def _record_direct_analysis(self, analysis: _DirectCompressedAnalysis) -> None:
        """Add one uncached inode analysis to the aggregate receipt."""
        self.direct_source_carriers += 1
        self.direct_compressed_candidates += analysis.compressed_candidates
        self.direct_rfc1950_candidates += analysis.rfc1950_candidates
        self.direct_complete_empty_streams += analysis.complete_empty_streams
        self.direct_invalid_streams += analysis.invalid_streams
        self.direct_output_occurrences += analysis.output_occurrences
        self.direct_emitted_bytes += analysis.emitted_bytes
        self.direct_fdict_candidates += analysis.fdict_candidates
        self.direct_fdict_complete_streams += analysis.fdict_complete_streams
        self.direct_fdict_malformed_streams += analysis.fdict_malformed_streams
        self.direct_fdict_unresolved_streams += analysis.fdict_unresolved_streams
        self.nested_compressed_candidates += analysis.nested_compressed_candidates
        self.nested_rfc1950_candidates += analysis.nested_rfc1950_candidates
        self.nested_complete_empty_streams += analysis.nested_complete_empty_streams
        self.nested_invalid_streams += analysis.nested_invalid_streams
        self.nested_output_occurrences += analysis.nested_output_occurrences
        self.nested_emitted_bytes += analysis.nested_emitted_bytes
        self.nested_max_depth = max(self.nested_max_depth, analysis.nested_max_depth)
        self.nested_fdict_candidates += analysis.nested_fdict_candidates
        self.nested_fdict_complete_streams += analysis.nested_fdict_complete_streams
        self.nested_fdict_malformed_streams += analysis.nested_fdict_malformed_streams
        self.nested_fdict_unresolved_streams += analysis.nested_fdict_unresolved_streams
        self.derived_analysis_bytes += (
            analysis.emitted_bytes + analysis.nested_emitted_bytes
        )

    def append_direct_rfc1950_outputs(
        self,
        path: Path,
        logical_parts: tuple[str, ...] | None,
    ) -> None:
        """Retain bounded RFC 1950 output from one ordinary source carrier."""
        metadata = path.stat()
        source_key = (metadata.st_dev, metadata.st_ino)
        analysis = self.direct_source_analyses.get(source_key)
        if analysis is not None:
            self.direct_duplicate_carriers += 1
        else:
            analysis = self._analyze_direct_path(path, metadata.st_size)
            self.direct_source_analyses[source_key] = analysis
            self._record_direct_analysis(analysis)
        for output in analysis.outputs:
            self._append_derived_output(output, logical_parts)

    def append_direct_rfc1950_content(
        self,
        content: bytes,
        logical_parts: tuple[str, ...] | None,
        *,
        maximum_start: int | None = None,
    ) -> None:
        """Retain bounded RFC 1950 output from one in-memory source region."""
        if not content:
            analysis = _empty_direct_compressed_analysis()
        else:
            analysis = _direct_compressed_outputs(
                content,
                max_candidates=(
                    _MAX_DIRECT_COMPRESSED_CANDIDATES
                    - self.direct_compressed_candidates
                    - self.nested_compressed_candidates
                ),
                max_emitted_bytes=(
                    _MAX_ENCODED_DERIVED_ANALYSIS_BYTES - self.derived_analysis_bytes
                ),
                maximum_start=maximum_start,
            )
            analysis = analysis._replace(
                outputs=self._intern_derived_outputs(analysis.outputs)
            )
        self._record_direct_analysis(analysis)
        for output in analysis.outputs:
            self._append_derived_output(output, logical_parts)

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


def _tar_header_checksum_valid(content: bytes) -> bool:
    """Recognize one complete checksum-valid POSIX, GNU, or legacy TAR header."""
    if len(content) < _ARCHIVE_PREFIX_BYTES:
        return False
    header = content[:_ARCHIVE_PREFIX_BYTES]
    try:
        expected = tarfile.nti(header[148:156])
        unsigned, signed = tarfile.calc_chksums(header)
    except (struct.error, tarfile.InvalidHeaderError):
        return False
    if expected not in {unsigned, signed}:
        return False
    try:
        tarfile.TarInfo.frombuf(header, "utf-8", "surrogateescape")
    except (tarfile.TarError, ValueError):
        return False
    return True


def _tar_header_layout_plausible(
    content: bytes | mmap.mmap,
    header_start: int,
) -> bool:
    """Reject random checksum-like fields before charging candidate work."""
    header = content[header_start : header_start + _ARCHIVE_PREFIX_BYTES]
    if (
        len(header) < _ARCHIVE_PREFIX_BYTES
        or bytes(header[156:157]) not in _TAR_STRUCTURAL_TYPES
    ):
        return False
    try:
        for begin, end in (
            (100, 108),
            (108, 116),
            (116, 124),
            (124, 136),
            (136, 148),
            (148, 156),
        ):
            tarfile.nti(header[begin:end])
    except (struct.error, tarfile.InvalidHeaderError):
        return False
    return True


def _archive_kind(prefix: bytes) -> str | None:
    """Identify archive formats with stable magic bytes or a verified header."""
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
    if _zlib_candidate_requires_rejection(prefix, 0, canonical=True):
        return "zlib"
    if _tar_header_checksum_valid(prefix):
        return "tar"
    return None


def _zlib_header_plausible(content: bytes | mmap.mmap, offset: int) -> bool:
    """Recognize every RFC 1950 CM=8 header with a bounded window."""
    if offset < 0 or offset + 2 > len(content):
        return False
    cmf = content[offset]
    flg = content[offset + 1]
    return cmf & 0x0F == 8 and cmf >> 4 <= 7 and ((cmf << 8) | flg) % 31 == 0


def _zlib_header_uses_dictionary(
    content: bytes | mmap.mmap,
    offset: int,
) -> bool:
    """Return the RFC 1950 FDICT bit after validating the complete header."""
    return _zlib_header_plausible(content, offset) and bool(content[offset + 1] & 0x20)


def _zlib_candidate_requires_rejection(
    content: bytes | mmap.mmap,
    offset: int,
    *,
    canonical: bool = False,
    reject_partial_fdict_output: bool = False,
) -> bool:
    """Reject a decodable or bounded-but-ambiguous RFC 1950 stream candidate."""
    if not _zlib_header_plausible(content, offset):
        return False
    try:
        result = _bounded_rfc1950_result(content, offset)
    except _ScanError:
        return True
    if result.fdict_status is not None:
        return (
            result.fdict_status == "complete"
            or canonical
            or (reject_partial_fdict_output and bool(result.output))
        )
    return result.strict_rejection


def _contains_archive_signature(
    content: bytes,
    budget: _ArchiveBudget | None = None,
    *,
    retain_rfc1950_outputs: bool = False,
) -> bool:
    """Detect a supported or rejected archive hidden after decoded prefix bytes."""
    kind = _archive_kind(content[:_ARCHIVE_PREFIX_BYTES])
    return (
        (kind is not None and (kind != "zlib" or not retain_rfc1950_outputs))
        or _contains_non_tar_archive_signature(
            content,
            retain_rfc1950_outputs=retain_rfc1950_outputs,
        )
        or _contains_tar_header(content, budget)
    )


def _contains_non_tar_archive_signature(
    content: bytes,
    *,
    retain_rfc1950_outputs: bool = False,
) -> bool:
    """Detect nested container signatures that are not TAR's own header marker."""
    contains_archive = (
        any(
            signature in content for signature in _IMMEDIATE_EMBEDDED_ARCHIVE_SIGNATURES
        )
        or b"\x1f\x8b\x08" in content
        or re.search(rb"BZh[1-9]", content) is not None
    )
    if retain_rfc1950_outputs:
        return contains_archive
    candidates = 0
    for match in _ZLIB_COMPRESSION_METHOD_MARKER.finditer(content):
        if not _zlib_header_plausible(content, match.start()):
            continue
        candidates += 1
        if candidates > _ENCODED_ARCHIVE_MAX_CANDIDATES:
            return True
        candidate_rejects = _zlib_candidate_requires_rejection(
            content,
            match.start(),
            reject_partial_fdict_output=True,
        )
        contains_archive = contains_archive or candidate_rejects
    return contains_archive


class _TarHeaderSearch:
    """Validate bounded TAR-header candidates in one byte window."""

    def __init__(
        self,
        content: bytes | mmap.mmap,
        budget: _ArchiveBudget | None,
        maximum_header_start: int | None,
    ) -> None:
        self.content = content
        self.budget = budget
        self.maximum_header_start = maximum_header_start
        self.seen: set[int] = set()
        self.examined = 0

    def check(self, checksum_start: int) -> bool:
        """Validate the header associated with one checksum-field match."""
        header_start = checksum_start - 148
        if (
            header_start < 0
            or header_start in self.seen
            or header_start + _ARCHIVE_PREFIX_BYTES > len(self.content)
            or (
                self.maximum_header_start is not None
                and header_start >= self.maximum_header_start
            )
        ):
            return False
        self.seen.add(header_start)
        if not _tar_header_layout_plausible(self.content, header_start):
            return False
        if self._work_exhausted():
            return True
        return _tar_header_checksum_valid(
            self.content[header_start : header_start + _ARCHIVE_PREFIX_BYTES]
        )

    def _work_exhausted(self) -> bool:
        """Charge one plausible header against the shared or local budget."""
        if self.budget is not None:
            return self.budget.examine_tar_header_candidate()
        self.examined += 1
        return self.examined > _MAX_EMBEDDED_TAR_CHECKSUM_CANDIDATES


def _contains_tar_header(
    content: bytes | mmap.mmap,
    budget: _ArchiveBudget | None = None,
    *,
    maximum_header_start: int | None = None,
) -> bool:
    """Recognize any checksum-valid TAR header at any offset in one window."""
    search = _TarHeaderSearch(content, budget, maximum_header_start)
    for pattern in (
        _TAR_TEXT_CHECKSUM_FIELD,
        _TAR_NUL_TERMINATED_CHECKSUM_FIELD,
    ):
        for match in pattern.finditer(content):
            if search.check(match.start(1)):
                return True
    for match in _TAR_BASE256_CHECKSUM_MARKER.finditer(content):
        if search.check(match.start()):
            return True
    return False


def _png_pass_layout(
    width: int,
    height: int,
    bits_per_pixel: int,
    interlace: int,
) -> tuple[int, tuple[tuple[int, int], ...]] | None:
    """Calculate bounded row sizes for sequential or Adam7 image data."""
    passes = ((0, 0, 1, 1),) if interlace == 0 else _PNG_ADAM7_PASSES
    expected = 0
    layout: list[tuple[int, int]] = []
    for x_start, y_start, x_step, y_step in passes:
        if width <= x_start or height <= y_start:
            continue
        pass_width = (width - x_start + x_step - 1) // x_step
        pass_height = (height - y_start + y_step - 1) // y_step
        row_bytes = (pass_width * bits_per_pixel + 7) // 8
        expected += pass_height * (row_bytes + 1)
        if expected > _MAX_ARCHIVE_MEMBER_BYTES:
            return None
        layout.append((row_bytes, pass_height))
    return expected, tuple(layout)


def _png_image_layout(
    ihdr: bytes,
) -> tuple[int, tuple[tuple[int, int], ...]] | None:
    """Return exact row-byte and row-count constraints for one PNG IHDR."""
    if len(ihdr) != 13:
        return None
    width, height, bit_depth, color_type, compression, filtering, interlace = (
        struct.unpack(">IIBBBBB", ihdr)
    )
    channel_counts = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
    allowed_depths = {
        0: {1, 2, 4, 8, 16},
        2: {8, 16},
        3: {1, 2, 4, 8},
        4: {8, 16},
        6: {8, 16},
    }
    if (
        not width
        or not height
        or color_type not in channel_counts
        or bit_depth not in allowed_depths[color_type]
        or compression != 0
        or filtering != 0
        or interlace not in {0, 1}
    ):
        return None
    return _png_pass_layout(
        width,
        height,
        channel_counts[color_type] * bit_depth,
        interlace,
    )


def _consume_png_filtered_bytes(
    content: bytes,
    layout: tuple[tuple[int, int], ...],
    state: list[int],
) -> bool:
    """Consume filtered rows and reject impossible PNG filter markers."""
    pass_index, rows_remaining, row_bytes_remaining = state
    position = 0
    while position < len(content):
        while rows_remaining == 0 and row_bytes_remaining == 0:
            pass_index += 1
            if pass_index >= len(layout):
                return False
            _row_bytes, rows_remaining = layout[pass_index]
        row_bytes = layout[pass_index][0]
        if row_bytes_remaining == 0:
            if content[position] > 4:
                return False
            position += 1
            rows_remaining -= 1
            row_bytes_remaining = row_bytes
            if position >= len(content):
                break
        consumed = min(row_bytes_remaining, len(content) - position)
        position += consumed
        row_bytes_remaining -= consumed
    state[:] = [pass_index, rows_remaining, row_bytes_remaining]
    return True


class _PngChunk(NamedTuple):
    """Describe one checksum-valid PNG chunk."""

    kind: bytes
    data_start: int
    data_end: int
    end: int


class _PngStructure(NamedTuple):
    """Retain the bounded image layout and exact IDAT byte ranges."""

    expected_image_bytes: int
    image_layout: tuple[tuple[int, int], ...]
    idat_ranges: tuple[tuple[int, int], ...]


class _PngParser:
    """Parse the structural subset needed to isolate strict PNG image data."""

    def __init__(self, content: bytes | mmap.mmap) -> None:
        self.content = content
        self.position = len(_PNG_SIGNATURE)
        self.chunk_index = 0
        self.bit_depth: int | None = None
        self.color_type: int | None = None
        self.expected_image_bytes: int | None = None
        self.image_layout: tuple[tuple[int, int], ...] | None = None
        self.idat_ranges: list[tuple[int, int]] = []
        self.seen_idat = False
        self.idat_closed = False
        self.seen_plte = False

    def parse(self) -> _PngStructure | None:
        """Return bounded structural metadata or reject the PNG boundary."""
        while self.position + 12 <= len(self.content):
            if self.chunk_index >= _MAX_PNG_CHUNKS:
                return None
            chunk = self._read_chunk()
            if chunk is None or not self._accept_chunk(chunk):
                return None
            self.position = chunk.end
            self.chunk_index += 1
            if chunk.kind == b"IEND":
                return self._finish(chunk)
        return None

    def _read_chunk(self) -> _PngChunk | None:
        """Read and authenticate one chunk without copying its payload."""
        length = struct.unpack(">I", self.content[self.position : self.position + 4])[0]
        kind = bytes(self.content[self.position + 4 : self.position + 8])
        data_start = self.position + 8
        data_end = data_start + length
        chunk_end = data_end + 4
        if (
            chunk_end > len(self.content)
            or not all(65 <= value <= 90 or 97 <= value <= 122 for value in kind)
            or kind[2] & 0x20
        ):
            return None
        expected_crc = struct.unpack(">I", self.content[data_end:chunk_end])[0]
        actual_crc = zlib.crc32(kind)
        payload = memoryview(self.content)[data_start:data_end]
        actual_crc = zlib.crc32(payload, actual_crc) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            return None
        return _PngChunk(kind, data_start, data_end, chunk_end)

    def _accept_chunk(self, chunk: _PngChunk) -> bool:
        """Apply critical-chunk ordering and palette constraints."""
        if self.chunk_index == 0:
            if not self._accept_header(chunk):
                return False
        elif chunk.kind == b"IHDR":
            return False
        known_critical = {b"IHDR", b"PLTE", b"IDAT", b"IEND"}
        if not chunk.kind[0] & 0x20 and chunk.kind not in known_critical:
            return False
        if chunk.kind == b"PLTE":
            return self._accept_palette(chunk)
        if chunk.kind == b"IDAT":
            if self.idat_closed:
                return False
            self.seen_idat = True
            if chunk.data_start < chunk.data_end:
                self.idat_ranges.append((chunk.data_start, chunk.data_end))
        elif self.seen_idat:
            self.idat_closed = True
        return True

    def _accept_header(self, chunk: _PngChunk) -> bool:
        """Record a valid first IHDR and its bounded image layout."""
        if chunk.kind != b"IHDR":
            return False
        ihdr = bytes(self.content[chunk.data_start : chunk.data_end])
        parsed_layout = _png_image_layout(ihdr)
        if parsed_layout is None:
            return False
        self.expected_image_bytes, self.image_layout = parsed_layout
        self.bit_depth = ihdr[8]
        self.color_type = ihdr[9]
        return True

    def _accept_palette(self, chunk: _PngChunk) -> bool:
        """Validate the one optional or required palette before image data."""
        length = chunk.data_end - chunk.data_start
        if (
            self.seen_plte
            or self.seen_idat
            or not length
            or length % 3
            or length > 768
            or (
                self.color_type == 3
                and self.bit_depth is not None
                and length > 3 * (1 << self.bit_depth)
            )
        ):
            return False
        self.seen_plte = True
        return True

    def _finish(self, chunk: _PngChunk) -> _PngStructure | None:
        """Require an exact IEND and return complete parser state."""
        if (
            chunk.data_start != chunk.data_end
            or not self.seen_idat
            or self.position != len(self.content)
            or self.color_type is None
            or self.expected_image_bytes is None
            or self.image_layout is None
            or (self.color_type == 3 and not self.seen_plte)
            or (self.color_type in {0, 4} and self.seen_plte)
        ):
            return None
        return _PngStructure(
            self.expected_image_bytes,
            self.image_layout,
            tuple(self.idat_ranges),
        )


class _PngImageDecoder:
    """Validate bounded PNG image bytes without retaining decompressed pixels."""

    def __init__(self, structure: _PngStructure) -> None:
        self.structure = structure
        self.decoder = zlib.decompressobj()
        self.decoded_bytes = 0
        self.row_state = [0, structure.image_layout[0][1], 0]

    def validate(self, content: bytes | mmap.mmap) -> bool:
        """Consume every IDAT range and require one exact zlib stream."""
        try:
            for start, end in self.structure.idat_ranges:
                for position in range(start, end, _ARCHIVE_READ_BYTES):
                    chunk_end = min(position + _ARCHIVE_READ_BYTES, end)
                    if not self._consume_compressed(bytes(content[position:chunk_end])):
                        return False
            flushed = self.decoder.flush()
        except zlib.error:
            return False
        return self._complete(flushed)

    def _consume_compressed(self, pending: bytes) -> bool:
        """Consume one bounded input chunk and validate decoded filter rows."""
        while pending:
            remaining = self.structure.expected_image_bytes - self.decoded_bytes + 1
            decoded = self.decoder.decompress(
                pending,
                max_length=min(_ARCHIVE_READ_BYTES, max(1, remaining)),
            )
            self.decoded_bytes += len(decoded)
            if (
                self.decoded_bytes > self.structure.expected_image_bytes
                or self.decoder.unused_data
                or not _consume_png_filtered_bytes(
                    decoded,
                    self.structure.image_layout,
                    self.row_state,
                )
            ):
                return False
            if not self.decoder.unconsumed_tail:
                return True
            if len(self.decoder.unconsumed_tail) >= len(pending) and not decoded:
                return False
            pending = self.decoder.unconsumed_tail
        return True

    def _complete(self, flushed: bytes) -> bool:
        """Require exact output size, valid rows, and no trailing stream."""
        self.decoded_bytes += len(flushed)
        if (
            not self.decoder.eof
            or self.decoder.unconsumed_tail
            or self.decoder.unused_data
            or self.decoded_bytes != self.structure.expected_image_bytes
            or not _consume_png_filtered_bytes(
                flushed,
                self.structure.image_layout,
                self.row_state,
            )
        ):
            return False
        return self.row_state == [len(self.structure.image_layout) - 1, 0, 0]


def _validated_png_idat_ranges(
    content: bytes | mmap.mmap,
) -> tuple[tuple[int, int], ...] | None:
    """Return exact image-data ranges only for one strict, decodable PNG."""
    if (
        len(content) < len(_PNG_SIGNATURE)
        or bytes(content[: len(_PNG_SIGNATURE)]) != _PNG_SIGNATURE
    ):
        return None
    structure = _PngParser(content).parse()
    if structure is None or not _PngImageDecoder(structure).validate(content):
        return None
    return structure.idat_ranges


_EmbeddedScanRegion = tuple[bytes | mmap.mmap, int, int]


def _embedded_scan_regions(
    content: mmap.mmap,
) -> Iterator[_EmbeddedScanRegion]:
    """Yield outside bytes, bounded lookahead, source offset, and start limit."""
    idat_ranges = _validated_png_idat_ranges(content)
    if idat_ranges is None:
        yield content, 0, len(content)
        return
    position = 0
    for start, end in idat_ranges:
        if position < start:
            lookahead_end = min(len(content), start + _ARCHIVE_PREFIX_BYTES - 1)
            yield bytes(content[position:lookahead_end]), position, start - position
        position = end
    if position < len(content):
        yield bytes(content[position:]), position, len(content) - position


def _content_sha256(content: bytes | mmap.mmap) -> str:
    """Hash the same mapped carrier bytes used for embedded discovery."""
    digest = hashlib.sha256()
    for position in range(0, len(content), 1024 * 1024):
        end = min(position + 1024 * 1024, len(content))
        digest.update(memoryview(content)[position:end])
    return digest.hexdigest()


def _inert_embedded_media_approved(content: bytes | mmap.mmap) -> bool:
    """Authenticate exact mapped bytes and any format-specific prerequisite."""
    identity = _content_sha256(content)
    if identity not in _INERT_EMBEDDED_MEDIA_SHA256:
        return False
    return (
        identity not in _INERT_STRICT_PNG_MEDIA_SHA256
        or _validated_png_idat_ranges(content) is not None
    )


def _embedded_compressed_streams(
    content: bytes | mmap.mmap,
) -> Iterator[tuple[str, int]]:
    """Yield plausible gzip, bzip2, and RFC 1950 zlib stream offsets."""
    for kind, signature in (("gzip", b"\x1f\x8b\x08"), ("bzip2", b"BZh")):
        position = content.find(signature)
        while position >= 0:
            if kind != "bzip2" or content[position + 3 : position + 4] in b"123456789":
                yield kind, position
            position = content.find(signature, position + 1)
    for match in _ZLIB_COMPRESSION_METHOD_MARKER.finditer(content):
        if _zlib_header_plausible(content, match.start()):
            yield "zlib", match.start()


def _embedded_stream_requires_rejection(
    content: bytes | mmap.mmap,
    kind: str,
    offset: int,
) -> bool:
    """Reject output, complete streams, and bounded-but-ambiguous candidates."""
    candidate = bytes(content[offset : offset + _PROBE_DECODE_CHUNK_BYTES])
    if kind == "zlib":
        return _zlib_candidate_requires_rejection(
            content,
            offset,
            canonical=offset == 0,
            reject_partial_fdict_output=True,
        )
    if kind == "gzip":
        decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
    elif kind == "bzip2":
        decoder = bz2.BZ2Decompressor()
    else:  # pragma: no cover - yielded kinds are closed above.
        raise _ScanError("Unknown embedded compression kind")
    for position in range(0, len(candidate), _EMBEDDED_ARCHIVE_PROBE_CHUNK_BYTES):
        chunk = candidate[position : position + _EMBEDDED_ARCHIVE_PROBE_CHUNK_BYTES]
        try:
            decoded = decoder.decompress(chunk, max_length=1)
        except (EOFError, OSError, zlib.error):
            return False
        if decoded or decoder.eof:
            return True
    return len(candidate) == _PROBE_DECODE_CHUNK_BYTES


def _regions_contain_immediate_archive(
    regions: tuple[_EmbeddedScanRegion, ...],
) -> bool:
    """Detect fixed signatures and TAR headers from every allowed start."""
    tar_budget = _ArchiveBudget()
    for region, _source_offset, start_limit in regions:
        has_signature = any(
            0 <= region.find(signature) < start_limit
            for signature in _IMMEDIATE_EMBEDDED_ARCHIVE_SIGNATURES
        )
        if has_signature or _contains_tar_header(
            region,
            tar_budget,
            maximum_header_start=start_limit,
        ):
            return True
    return False


def _regions_contain_compressed_archive(
    content: mmap.mmap,
    regions: tuple[_EmbeddedScanRegion, ...],
) -> bool:
    """Probe compressed-stream starts with one aggregate candidate budget."""
    candidates = 0
    inert_media: bool | None = None

    def _inert_media_approved() -> bool:
        """Authenticate the mapped bytes before applying the narrow exception."""
        nonlocal inert_media
        if inert_media is None:
            inert_media = _inert_embedded_media_approved(content)
        return inert_media

    for region, source_offset, start_limit in regions:
        for kind, offset in _embedded_compressed_streams(region):
            if offset >= start_limit:
                continue
            candidates += 1
            if (
                candidates > _ENCODED_ARCHIVE_MAX_CANDIDATES
                and not _inert_media_approved()
            ):
                return True
            if (
                _embedded_stream_requires_rejection(
                    content,
                    kind,
                    source_offset + offset,
                )
                and not _inert_media_approved()
            ):
                return True
    return False


def _contains_embedded_archive(path: Path) -> bool:
    """Detect a real or unbounded container hidden in a nonarchive source."""
    if not path.stat().st_size:
        return False
    with (
        path.open("rb") as stream,
        mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as content,
    ):
        regions = tuple(_embedded_scan_regions(content))
        return _regions_contain_immediate_archive(
            regions
        ) or _regions_contain_compressed_archive(content, regions)


def _derived_output_identity(content: bytes | mmap.mmap) -> tuple[int, bytes]:
    """Return the immutable identity used for cycles and output deduplication."""
    return (len(content), hashlib.sha256(content).digest())


def _reject_recursive_output_boundaries(content: bytes) -> None:
    """Keep non-RFC and encoded containers fail closed in every derived node."""
    if _contains_non_zlib_archive(content):
        raise _ScanError(
            "Nested archive output cannot prove complete Gitleaks policy coverage"
        )
    encoded_outputs, _emitted_bytes = _encoded_candidate_outputs(content)
    if encoded_outputs:
        raise _ScanError(
            "Nested encoded archive output cannot prove complete "
            "Gitleaks policy coverage"
        )


def _inspect_recursive_output(
    node: _RecursiveOutputNode,
    accumulator: _RecursiveCompressedAccumulator,
) -> None:
    """Classify and retain every RFC 1950 child from one derived node."""
    _reject_recursive_output_boundaries(node.content)
    for kind, offset in _embedded_compressed_streams(node.content):
        is_rfc1950 = accumulator.charge_candidate(
            kind,
            node.depth,
            fdict=(
                kind == "zlib" and _zlib_header_uses_dictionary(node.content, offset)
            ),
        )
        if not is_rfc1950:
            if _embedded_stream_requires_rejection(node.content, kind, offset):
                raise _ScanError(
                    "Nested archive output cannot prove complete "
                    "Gitleaks policy coverage"
                )
            continue
        result = _bounded_rfc1950_result(node.content, offset)
        accumulator.charge_output(
            result,
            node,
            canonical=node.canonical_source and offset == 0,
        )


def _recursive_direct_outputs(
    initial_outputs: tuple[bytes, ...],
    source_identity: tuple[int, bytes],
    max_candidates: int,
    max_emitted_bytes: int,
    *,
    canonical_sources: tuple[bool, ...] | None = None,
) -> _RecursiveCompressedAccumulator:
    """Traverse bounded nested RFC 1950 output while retaining every child."""
    if canonical_sources is None:
        canonical_sources = (True,) * len(initial_outputs)
    if len(canonical_sources) != len(initial_outputs):
        raise _ScanError("Recursive RFC 1950 source canonicality is incomplete")
    accumulator = _RecursiveCompressedAccumulator(
        max_candidates,
        max_emitted_bytes,
    )
    for output, canonical_source in zip(
        initial_outputs,
        canonical_sources,
        strict=True,
    ):
        accumulator.add_root(
            output,
            source_identity,
            canonical_source=canonical_source,
        )
    cursor = 0
    while cursor < len(accumulator.pending):
        node = accumulator.pending[cursor]
        cursor += 1
        _inspect_recursive_output(node, accumulator)
    return accumulator


def _direct_compressed_outputs(
    content: bytes | mmap.mmap,
    *,
    max_candidates: int,
    max_emitted_bytes: int,
    maximum_start: int | None = None,
) -> _DirectCompressedAnalysis:
    """Retain direct and recursive RFC 1950 output under shared bounds."""
    if maximum_start is not None and not 0 <= maximum_start <= len(content):
        raise _ScanError("Direct RFC 1950 start limit is outside the source")
    regions = tuple(_embedded_scan_regions(content))  # type: ignore[arg-type]
    if _regions_contain_immediate_archive(regions):
        raise _ScanError("Embedded archive bytes cannot prove complete coverage")
    accumulator = _DirectCompressedAccumulator(max_candidates, max_emitted_bytes)
    for region, source_offset, start_limit in regions:
        for kind, offset in _embedded_compressed_streams(region):
            absolute_offset = source_offset + offset
            if offset >= start_limit or (
                maximum_start is not None and absolute_offset >= maximum_start
            ):
                continue
            accumulator.inspect(content, kind, absolute_offset)
    return accumulator.result(content)


def _has_archive_suffix(name: str) -> bool:
    """Return whether a path advertises any supported or rejected archive type."""
    lowered = name.lower()
    return any(
        lowered.endswith(suffix)
        for suffix in _ARCHIVE_SUFFIXES | _REJECTED_ARCHIVE_SUFFIXES
    )


def _has_nonterminal_archive_token(name: str) -> bool:
    """Reject aliases that the pinned archive stack may redispatch by substring."""
    lowered = name.lower()
    suffixes = _ARCHIVE_SUFFIXES | _REJECTED_ARCHIVE_SUFFIXES
    for suffix in suffixes:
        offset = lowered.find(suffix)
        while offset >= 0:
            end = offset + len(suffix)
            if end != len(lowered):
                compound_tar_suffix = suffix == ".tar" and any(
                    offset == len(lowered) - len(compound)
                    for compound in _TAR_SUFFIXES
                    if lowered.endswith(compound)
                )
                if not compound_tar_suffix:
                    return True
            offset = lowered.find(suffix, offset + 1)
    return False


def _inspect_archive_stream(
    stream: BinaryIO,
    budget: _ArchiveBudget,
    member_writer: _ArchiveMemberProbeWriter | None = None,
    *,
    retain_rfc1950_outputs: bool = False,
) -> tuple[bytes, bool, str]:
    """Verify a bounded stream and detect encoded or nested containers."""
    if retain_rfc1950_outputs and (
        not isinstance(member_writer, _ArchiveMemberProbeWriter)
        or not member_writer.retain_direct_rfc1950_outputs
    ):
        raise _ScanError("Archive RFC 1950 output retention requires a scan sink")
    prefix = bytearray()
    digest = hashlib.sha256()
    overlap = b""
    contains_archive = False
    member_bytes = 0
    try:
        while chunk := stream.read(_ARCHIVE_READ_BYTES):
            member_bytes = budget.add_bytes(member_bytes, len(chunk))
            digest.update(chunk)
            if member_writer is not None:
                member_writer.append(chunk)
            if len(prefix) < _ARCHIVE_PREFIX_BYTES:
                missing = _ARCHIVE_PREFIX_BYTES - len(prefix)
                prefix.extend(chunk[:missing])
            window = overlap + chunk
            _reject_encoded_archives(
                window,
                retain_rfc1950_outputs=retain_rfc1950_outputs,
            )
            window_contains_archive = _contains_archive_signature(
                window,
                budget,
                retain_rfc1950_outputs=retain_rfc1950_outputs,
            )
            contains_archive = contains_archive or window_contains_archive
            overlap = window[-_ARCHIVE_SIGNATURE_OVERLAP_BYTES:]
        if member_writer is not None:
            member_writer.finish()
    except Exception:
        if member_writer is not None:
            member_writer.abort()
        raise
    return bytes(prefix), contains_archive, digest.hexdigest()


def _reject_nested_archive(
    member: str,
    prefix: bytes,
    *,
    contains_archive: bool,
    retain_rfc1950_outputs: bool = False,
) -> None:
    """Reject nested containers whose complete policy traversal is ambiguous."""
    kind = _archive_kind(prefix)
    if (
        _has_archive_suffix(member)
        or _has_nonterminal_archive_token(member)
        or (kind is not None and (kind != "zlib" or not retain_rfc1950_outputs))
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


def _contains_non_zlib_archive(content: bytes) -> bool:
    """Recognize archive bytes that cannot be replaced by a derived zlib view."""
    return (
        any(
            signature in content for signature in _IMMEDIATE_EMBEDDED_ARCHIVE_SIGNATURES
        )
        or b"\x1f\x8b\x08" in content
        or re.search(rb"BZh[1-9]", content) is not None
        or _contains_tar_header(content)
    )


def _bounded_rfc1950_result(
    content: bytes | mmap.mmap,
    offset: int,
) -> _RFC1950DecodeResult:
    """Retain deterministic bytes and classify one bounded RFC 1950 start."""
    if not _zlib_header_plausible(content, offset):
        return _RFC1950DecodeResult(None, False, None)
    fdict = _zlib_header_uses_dictionary(content, offset)
    compressed_start = offset + (6 if fdict else 2)
    if compressed_start > len(content):
        return _RFC1950DecodeResult(
            None,
            fdict,
            "malformed" if fdict else None,
        )
    compressed_end = min(
        len(content),
        compressed_start + _PROBE_DECODE_CHUNK_BYTES,
    )
    window_bits = 8 + (content[offset] >> 4)
    decoder = zlib.decompressobj(-window_bits)
    output = bytearray()

    def retain(decoded: bytes) -> None:
        """Append one decoder result without crossing the scan-fragment bound."""
        output.extend(decoded)
        if len(output) > _MAX_ENCODED_DERIVED_OUTPUT_BYTES:
            raise _ScanError(
                "Encoded RFC 1950 derived output exceeds one complete "
                "Gitleaks scan fragment"
            )

    def failure(error: zlib.error) -> _RFC1950DecodeResult:
        """Preserve prior output and distinguish unavailable dictionary history."""
        rendered = str(error).lower()
        status = (
            "unresolved"
            if fdict and "invalid distance too far back" in rendered
            else "malformed"
        )
        retained = bytes(output) if output else None
        return _RFC1950DecodeResult(
            retained,
            bool(retained) or status == "unresolved",
            status if fdict else None,
        )

    def complete(trailer_start: int) -> _RFC1950DecodeResult:
        """Authenticate dictionary-independent FDICT output when possible."""
        retained = bytes(output)
        if not fdict:
            return _RFC1950DecodeResult(retained, True, None)
        trailer = bytes(content[trailer_start : trailer_start + 4])
        status = (
            "complete"
            if len(trailer) == 4
            and int.from_bytes(trailer, "big") == zlib.adler32(retained)
            else "malformed"
        )
        return _RFC1950DecodeResult(retained, True, status)

    for position in range(compressed_start, compressed_end):
        pending = content[position : position + 1]
        while pending:
            try:
                decoded = decoder.decompress(pending, max_length=1)
            except zlib.error as error:
                return failure(error)
            retain(decoded)
            if decoder.eof:
                return complete(position + 1)
            remaining = decoder.unconsumed_tail
            if remaining and len(remaining) >= len(pending) and not decoded:
                raise _ScanError("Encoded RFC 1950 output accounting made no progress")
            pending = remaining
        while True:
            try:
                decoded = decoder.decompress(b"", max_length=1)
            except zlib.error as error:
                return failure(error)
            if not decoded:
                break
            retain(decoded)
            if decoder.eof:
                return complete(position + 1)
    if compressed_end < len(content):
        raise _ScanError("Encoded RFC 1950 input exceeds the bounded probe limit")
    retained = bytes(output) if output else None
    return _RFC1950DecodeResult(
        retained,
        bool(retained),
        "malformed" if fdict else None,
    )


def _bounded_rfc1950_output(
    content: bytes | mmap.mmap,
    offset: int,
) -> bytes | None:
    """Return deterministic bounded output for compatibility with focused callers."""
    return _bounded_rfc1950_result(content, offset).output


def _decoded_candidate_outputs(
    decoded: bytes,
    compressed_candidates: int,
    emitted_bytes: int,
    max_emitted_bytes: int,
    fdict_counts: dict[str, int],
    *,
    canonical_source: bool,
) -> tuple[tuple[_CanonicalOutput, ...], int, int]:
    """Return bounded RFC 1950 outputs or reject any other decoded container."""
    if hashlib.sha256(decoded).hexdigest() in _INERT_ENCODED_ARCHIVE_SHA256:
        return (), compressed_candidates, emitted_bytes
    if _contains_non_zlib_archive(decoded):
        raise _ScanError(
            "Encoded archive bytes cannot prove complete Gitleaks policy coverage"
        )
    outputs: list[_CanonicalOutput] = []
    for match in _ZLIB_COMPRESSION_METHOD_MARKER.finditer(decoded):
        offset = match.start()
        if not _zlib_header_plausible(decoded, offset):
            continue
        compressed_candidates += 1
        if compressed_candidates > _ENCODED_ARCHIVE_MAX_CANDIDATES:
            raise _ScanError(
                "Encoded archive analysis exceeded its bounded candidate count"
            )
        result = _bounded_rfc1950_result(decoded, offset)
        if result.fdict_status is not None:
            fdict_counts["candidates"] += 1
            fdict_counts[result.fdict_status] += 1
        if result.fdict_status == "unresolved" and canonical_source and offset == 0:
            raise _ScanError(
                "Canonical encoded RFC 1950 output requires an external dictionary"
            )
        output = result.output
        if output is None or not output:
            continue
        emitted_bytes += len(output)
        if emitted_bytes > max_emitted_bytes:
            raise _ScanError(
                "Encoded RFC 1950 derived analysis exceeds the aggregate "
                "bounded byte limit"
            )
        outputs.append(
            _CanonicalOutput(
                output,
                canonical_source and offset == 0,
            )
        )
    return tuple(outputs), compressed_candidates, emitted_bytes


def _encoded_candidate_analysis(
    candidate: bytes,
    *,
    max_emitted_bytes: int = _MAX_ENCODED_DERIVED_ANALYSIS_BYTES,
    canonical_source: bool = True,
) -> _EncodedCandidateAnalysis:
    """Recover every bounded zlib output hidden behind supported text encodings."""
    queue: list[tuple[bytes, int, bool]] = [(candidate, 0, canonical_source)]
    seen: set[tuple[int, bytes, bool]] = set()
    outputs: list[bytes] = []
    output_work: list[bytes] = []
    output_work_canonical_sources: list[bool] = []
    seen_outputs: set[tuple[int, bytes]] = set()
    decoded_candidates = 0
    compressed_candidates = 0
    emitted_bytes = 0
    fdict_counts = {
        "candidates": 0,
        "complete": 0,
        "malformed": 0,
        "unresolved": 0,
    }
    while queue:
        content, depth, content_is_canonical = queue.pop()
        identity = (depth, hashlib.sha256(content).digest(), content_is_canonical)
        if identity in seen or depth >= _MAX_DECODE_DEPTH:
            continue
        seen.add(identity)
        for kind, pattern in _ENCODING_CANDIDATES:
            for match in pattern.finditer(content):
                for decoded in _decode_candidate(kind, match.group()):
                    decoded_is_canonical = content_is_canonical and match.start() == 0
                    decoded_candidates += 1
                    if decoded_candidates > _ENCODED_ARCHIVE_MAX_CANDIDATES:
                        raise _ScanError(
                            "Encoded archive analysis exceeded its bounded "
                            "candidate count"
                        )
                    (
                        derived,
                        compressed_candidates,
                        emitted_bytes,
                    ) = _decoded_candidate_outputs(
                        decoded,
                        compressed_candidates,
                        emitted_bytes,
                        max_emitted_bytes,
                        fdict_counts,
                        canonical_source=decoded_is_canonical,
                    )
                    queue.append((decoded, depth + 1, decoded_is_canonical))
                    for discovered in derived:
                        output = discovered.content
                        output_work.append(output)
                        output_work_canonical_sources.append(
                            discovered.canonical_source
                        )
                        output_identity = (
                            len(output),
                            hashlib.sha256(output).digest(),
                        )
                        if output_identity not in seen_outputs:
                            outputs.append(output)
                            seen_outputs.add(output_identity)
    recursive = _RecursiveCompressedAccumulator(
        _ENCODED_ARCHIVE_MAX_CANDIDATES - compressed_candidates,
        max_emitted_bytes - emitted_bytes,
    )
    if output_work:
        recursive = _recursive_direct_outputs(
            tuple(output_work),
            _derived_output_identity(candidate),
            _ENCODED_ARCHIVE_MAX_CANDIDATES - compressed_candidates,
            max_emitted_bytes - emitted_bytes,
            canonical_sources=tuple(output_work_canonical_sources),
        )
    emitted_bytes += recursive.emitted_bytes
    for output in recursive.outputs:
        identity = _derived_output_identity(output)
        if identity not in seen_outputs:
            outputs.append(output)
            seen_outputs.add(identity)
    return _EncodedCandidateAnalysis(
        tuple(outputs),
        emitted_bytes,
        recursive,
        fdict_counts["candidates"],
        fdict_counts["complete"],
        fdict_counts["malformed"],
        fdict_counts["unresolved"],
    )


def _encoded_candidate_outputs(
    candidate: bytes,
    *,
    max_emitted_bytes: int = _MAX_ENCODED_DERIVED_ANALYSIS_BYTES,
) -> tuple[tuple[bytes, ...], int]:
    """Return the compatibility output view for strict boundary callers."""
    analysis = _encoded_candidate_analysis(
        candidate,
        max_emitted_bytes=max_emitted_bytes,
    )
    return analysis.outputs, analysis.emitted_bytes


def _reject_derived_projection_archives(projection: bytes) -> None:
    """Reject containers exposed only after a derived Unicode projection."""
    encoded_outputs, _emitted_bytes = _encoded_candidate_outputs(projection)
    if _contains_archive_signature(projection) or encoded_outputs:
        raise _ScanError("Derived Unicode projection contains nested archive bytes")


def _reject_encoded_archives(
    candidate: bytes,
    *,
    retain_rfc1950_outputs: bool = False,
) -> None:
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
                    contains_archive = _contains_archive_signature(
                        decoded,
                        retain_rfc1950_outputs=retain_rfc1950_outputs,
                    )
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
                if member.is_dir():
                    if member.file_size:
                        raise _ScanError(
                            "Non-empty archive directory member cannot be scanned"
                        )
                    raise _ScanError(
                        "Archive directory member cannot preserve exact path coverage"
                    )
                _canonical_archive_member_parts(member.filename)
                member_writer = (
                    None if member_factory is None else member_factory(member.filename)
                )
                try:
                    with archive.open(member) as stream:
                        prefix, contains_archive, _content_sha256 = (
                            _inspect_archive_stream(
                                stream,
                                budget,
                                member_writer,
                                retain_rfc1950_outputs=(
                                    member_writer is not None
                                    and member_writer.retain_direct_rfc1950_outputs
                                ),
                            )
                        )
                except Exception:
                    if member_writer is not None:
                        member_writer.abort()
                    raise
                _reject_nested_archive(
                    member.filename,
                    prefix,
                    contains_archive=contains_archive,
                    retain_rfc1950_outputs=(
                        member_writer is not None
                        and member_writer.retain_direct_rfc1950_outputs
                    ),
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


class _MemoryBoundedXzReader:
    """Stream concatenated XZ members with an explicit decoder-memory ceiling."""

    def __init__(self, path: Path) -> None:
        self._stream = path.open("rb")
        self._decompressor = self._new_decompressor()
        self._pending = b""
        self._output = bytearray()
        self._finished = False

    @staticmethod
    def _new_decompressor() -> lzma.LZMADecompressor:
        return lzma.LZMADecompressor(
            format=lzma.FORMAT_XZ,
            memlimit=_MAX_XZ_DECOMPRESS_MEMORY_BYTES,
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *unused: object) -> None:
        self.close()

    def close(self) -> None:
        self._stream.close()

    def _pump(self, maximum: int) -> None:
        starting_output_size = len(self._output)
        while len(self._output) == starting_output_size and not self._finished:
            if self._decompressor.eof:
                compressed = self._decompressor.unused_data
                if not compressed:
                    compressed = self._stream.read(_ARCHIVE_READ_BYTES)
                if not compressed:
                    self._finished = True
                    return
                self._decompressor = self._new_decompressor()
                self._pending = compressed
            if self._decompressor.needs_input:
                compressed = self._pending
                self._pending = b""
                if not compressed:
                    compressed = self._stream.read(_ARCHIVE_READ_BYTES)
                if not compressed:
                    raise lzma.LZMAError("truncated XZ stream")
            else:
                compressed = b""
            self._output.extend(
                self._decompressor.decompress(compressed, max_length=maximum)
            )

    def read(self, size: int = -1) -> bytes:
        if size == 0:
            return b""
        if size < 0:
            chunks = [bytes(self._output)]
            self._output.clear()
            while not self._finished:
                self._pump(_ARCHIVE_READ_BYTES)
                if self._output:
                    chunks.append(bytes(self._output))
                    self._output.clear()
            return b"".join(chunks)
        while len(self._output) < size and not self._finished:
            self._pump(max(size - len(self._output), 1))
        result = bytes(self._output[:size])
        del self._output[:size]
        return result


def _open_memory_bounded_xz(path: Path, mode: str) -> _MemoryBoundedXzReader:
    if mode != "rb":
        raise ValueError("bounded XZ reader only supports binary reads")
    return _MemoryBoundedXzReader(path)


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
        else _open_memory_bounded_xz
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
                archive.ignore_zeros = True
                names: set[str] = set()
                for member in archive:
                    if member.name in names:
                        raise _ScanError("Archive contains a duplicate member path")
                    names.add(member.name)
                    budget.start_member(member.size)
                    if member.isdir():
                        if member.size:
                            raise _ScanError(
                                "Non-empty archive directory member cannot be scanned"
                            )
                        raise _ScanError(
                            "Archive directory member cannot preserve exact path coverage"
                        )
                    _canonical_archive_member_parts(member.name)
                    member_writer = (
                        None if member_factory is None else member_factory(member.name)
                    )
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
                        prefix, contains_archive, _content_sha256 = (
                            _inspect_archive_stream(
                                stream,
                                budget,
                                member_writer,
                                retain_rfc1950_outputs=(
                                    member_writer is not None
                                    and member_writer.retain_direct_rfc1950_outputs
                                ),
                            )
                        )
                    _reject_nested_archive(
                        member.name,
                        prefix,
                        contains_archive=contains_archive,
                        retain_rfc1950_outputs=(
                            member_writer is not None
                            and member_writer.retain_direct_rfc1950_outputs
                        ),
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
        "xz": _open_memory_bounded_xz,
    }
    budget.start_member()
    member_writer = (
        None if member_factory is None else member_factory("<compressed-stream>")
    )
    try:
        with openers[kind](path, "rb") as stream:
            prefix, contains_archive, _content_sha256 = _inspect_archive_stream(
                stream,
                budget,
                member_writer,
                retain_rfc1950_outputs=(
                    member_writer is not None
                    and member_writer.retain_direct_rfc1950_outputs
                ),
            )
    except (EOFError, OSError, lzma.LZMAError) as exc:
        if member_writer is not None:
            member_writer.abort()
        raise _ScanError(f"{kind} archive could not be verified") from exc
    _reject_nested_archive(
        "<compressed-stream>",
        prefix,
        contains_archive=contains_archive,
        retain_rfc1950_outputs=(
            member_writer is not None and member_writer.retain_direct_rfc1950_outputs
        ),
    )


def _validate_archive_file(
    path: Path,
    budget: _ArchiveBudget | None = None,
    member_factory: Callable[[str], _ArchiveMemberProbeWriter] | None = None,
    structure_factory: Callable[[str], _ArchiveMemberProbeWriter] | None = None,
    compressed_stream_factory: (
        Callable[[str], _ArchiveMemberProbeWriter] | None
    ) = None,
    embedded_output_writer: Callable[[Path], None] | None = None,
) -> None:
    """Fail closed on archive names or bytes Gitleaks cannot traverse completely."""
    archive_budget = _ArchiveBudget() if budget is None else budget
    lowered = path.name.lower()
    if _has_nonterminal_archive_token(path.name):
        raise _ScanError("Archive name contains a noncanonical extension alias")
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
            (
                member_factory
                if compressed_stream_factory is None
                else compressed_stream_factory
            ),
        )
        return
    if kind == "zlib" and embedded_output_writer is not None:
        embedded_output_writer(path)
        return
    if kind is not None:
        raise _ScanError(f"Disguised {kind} archive bytes cannot be scanned")
    if embedded_output_writer is not None:
        embedded_output_writer(path)
    elif _contains_embedded_archive(path):
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


def _process_commit_metadata_compressed_boundaries(
    path: Path,
    candidates: _CandidateProbeWriter,
    logical_parts: tuple[str, ...],
) -> None:
    """Scan header output and keep the attacker-controlled message strict."""
    content = path.read_bytes()
    separator = content.find(b"\n\n")
    if separator < 0:
        raise _ScanError("Commit metadata does not contain a message boundary")
    message = content[separator + 2 :]
    if _contains_archive_signature(message):
        raise _ScanError("Embedded archive bytes cannot prove complete coverage")
    candidates.append_direct_rfc1950_content(
        content,
        logical_parts,
        maximum_start=separator,
    )


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
    retain_direct_rfc1950_outputs: bool,
) -> None:
    """Link one raw file under every pinned archive-compatible name."""
    source_identity = os.fsencode(relative)

    def probe_writer(
        member: str,
        logical_scan_path: str,
        identity_domain: bytes,
        *,
        retain_direct_rfc1950_outputs: bool,
    ) -> _ArchiveMemberProbeWriter:
        member_identity = member.encode("utf-8", errors="surrogatepass")
        identity = hashlib.sha256(
            source_identity + b"\0" + identity_domain + b"\0" + member_identity
        ).hexdigest()
        raw_exact_path_target = exact_paths.allocate(logical_scan_path)
        transformed_exact_path_target = exact_paths.allocate(logical_scan_path)
        return _ArchiveMemberProbeWriter(
            destination,
            identity,
            member,
            relative.startswith("npa/tests/"),
            counts,
            candidates,
            raw_exact_path_target,
            transformed_exact_path_target,
            fragment_exact_paths,
            logical_scan_path,
            retain_direct_rfc1950_outputs=retain_direct_rfc1950_outputs,
        )

    def member_factory(member: str) -> _ArchiveMemberProbeWriter:
        return probe_writer(
            member,
            f"{logical_relative}!{member}",
            b"member",
            retain_direct_rfc1950_outputs=True,
        )

    if not retain_direct_rfc1950_outputs:
        _process_commit_metadata_compressed_boundaries(
            path,
            candidates,
            _canonical_archive_member_parts(logical_relative),
        )
        return
    _validate_archive_file(
        path,
        archive_budget,
        member_factory=member_factory,
        structure_factory=lambda member: probe_writer(
            f"<container>{member}",
            logical_relative,
            b"structure",
            retain_direct_rfc1950_outputs=False,
        ),
        compressed_stream_factory=lambda member: probe_writer(
            member,
            logical_relative,
            b"compressed-stream",
            retain_direct_rfc1950_outputs=True,
        ),
        embedded_output_writer=(
            lambda source: candidates.append_direct_rfc1950_outputs(
                source,
                _canonical_archive_member_parts(logical_relative),
            )
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
    preserve_exact_paths: bool = False,
    *,
    retain_direct_rfc1950_outputs: bool = True,
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
    with _CandidateProbeWriter(destination, preserve_exact_paths) as candidates:
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
                retain_direct_rfc1950_outputs,
            )
    counts.archive_exact_path_shards = exact_paths.shards
    counts.fragment_exact_path_shards = fragment_exact_paths.shards
    return counts.render(candidates, archive_budget)


def _git_object_directory(repo: Path) -> Path:
    """Resolve the source object directory as one alternates-file-safe path."""
    raw_objects = materializer._git(repo, "rev-parse", "--git-path", "objects").rstrip(
        b"\n"
    )
    if not raw_objects or b"\n" in raw_objects or b"\r" in raw_objects:
        raise _ScanError("Git object path cannot be represented safely")
    object_directory = Path(os.fsdecode(raw_objects))
    if not object_directory.is_absolute():
        object_directory = repo / object_directory
    object_directory = object_directory.resolve(strict=True)
    if not object_directory.is_dir():
        raise _ScanError("Git object path must resolve to a directory")
    return object_directory


def _initialize_isolated_git_directory(
    destination: Path,
    object_format: bytes,
    object_directory: Path,
) -> Path:
    """Create private Git metadata backed only by the source object database."""
    encoded_objects = os.fsencode(object_directory)
    if b"\n" in encoded_objects or b"\r" in encoded_objects:
        raise _ScanError("Git object path cannot be represented safely")
    git_directory = destination / ".git"
    git_directory.mkdir(mode=0o700)
    (git_directory / "objects").mkdir(mode=0o700)
    (git_directory / "objects" / "info").mkdir(mode=0o700)
    (git_directory / "refs").mkdir(mode=0o700)
    (git_directory / "refs" / "heads").mkdir(mode=0o700)
    extension = (
        b"" if object_format == b"sha1" else b"[extensions]\n\tobjectFormat = sha256\n"
    )
    repository_format = b"0" if object_format == b"sha1" else b"1"
    _write_private(
        git_directory / "config",
        (
            b"[core]\n\trepositoryformatversion = "
            + repository_format
            + b"\n\tbare = false\n\tfilemode = false\n"
            + extension
        ),
    )
    _write_private(git_directory / "HEAD", b"ref: refs/heads/npa-isolated\n")
    _write_private(
        git_directory / "objects" / "info" / "alternates",
        encoded_objects + b"\n",
    )
    return git_directory


def _isolated_history_source(repo: Path, destination: Path) -> Path:
    """Expose objects through new metadata with no caller-controlled local config."""
    destination.mkdir(mode=0o700)
    object_format = materializer._object_format(repo)
    object_directory = _git_object_directory(repo)
    git_directory = _initialize_isolated_git_directory(
        destination,
        object_format,
        object_directory,
    )
    if materializer.git_directory(destination) != git_directory.resolve(strict=True):
        raise _ScanError("Isolated history source resolved different Git metadata")
    if materializer._object_format(destination) != object_format:
        raise _ScanError("Isolated history source changed Git object format")
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


def _read_native_diff_metadata(
    source: Path,
    commit: str,
    remaining_bytes: int,
) -> bytes:
    """Read one bounded raw changed-path inventory."""
    if remaining_bytes <= 0:
        raise _ScanError("Native Git diff metadata exceeds its bounded byte limit")
    limit = min(remaining_bytes, _MAX_NATIVE_HISTORY_DIFF_COMMIT_BYTES)
    arguments = materializer._git_command(
        source,
        "diff-tree",
        "--root",
        "-m",
        "-r",
        "--no-renames",
        "--raw",
        "-z",
        "--no-commit-id",
        commit,
    )
    with (
        tempfile.TemporaryFile() as stdout_sink,
        tempfile.TemporaryFile() as stderr_sink,
    ):
        result = subprocess.run(
            arguments,
            check=False,
            stdout=stdout_sink,
            stderr=stderr_sink,
            env=_scan_environment(),
            preexec_fn=partial(_limit_scanner_output_files, limit),
        )
        size = os.fstat(stdout_sink.fileno()).st_size
        diagnostic_bytes = os.fstat(stderr_sink.fileno()).st_size
        if result.returncode != 0 or diagnostic_bytes or size > limit:
            raise _ScanError(
                "Native Git diff metadata exceeds its bounded byte limit "
                "or could not be verified"
            )
        stdout_sink.seek(0)
        return stdout_sink.read()


def _changed_blob_ids(metadata: bytes, object_format: bytes) -> list[bytes]:
    """Parse every nondeleted blob mode that native Gitleaks can reopen."""
    fields = metadata.split(b"\0")
    if not fields or fields[-1] != b"" or len(fields) % 2 != 1:
        raise _ScanError("Native Git returned invalid raw diff metadata")
    object_ids: list[bytes] = []
    for index in range(0, len(fields) - 1, 2):
        header, path = fields[index : index + 2]
        parts = header.split()
        if len(parts) != 5 or not parts[0].startswith(b":") or not path:
            raise _ScanError("Native Git returned invalid raw diff metadata")
        _old_mode, new_mode, _old_id, new_id, status = parts
        if status not in {b"A", b"D", b"M", b"T", b"U"}:
            raise _ScanError("Native Git returned unsupported raw diff status")
        if status != b"D" and new_mode in materializer._SUPPORTED_MODES:
            materializer._validate_object_id(new_id, object_format)
            object_ids.append(new_id)
    return object_ids


def _request_checked_blob_size(
    process: subprocess.Popen[bytes],
    object_id: bytes,
) -> int:
    """Request and validate one blob size from a batch-check process."""
    if process.stdin is None or process.stdout is None:
        raise _ScanError("Native Git blob-size stream is unavailable")
    process.stdin.write(object_id + b"\n")
    process.stdin.flush()
    fields = process.stdout.readline(1024).removesuffix(b"\n").split()
    if len(fields) != 3 or fields[:2] != [object_id, b"blob"]:
        raise _ScanError("Native Git returned invalid blob-size metadata")
    try:
        size = int(fields[2])
    except ValueError as exc:
        raise _ScanError("Native Git returned invalid blob-size metadata") from exc
    if size < 0:
        raise _ScanError("Native Git returned invalid blob-size metadata")
    return size


def _sum_changed_blob_sizes(
    source: Path,
    object_ids: list[bytes],
    byte_limit: int,
) -> int:
    """Upper-bound archive replay with every changed scannable blob."""
    process = subprocess.Popen(
        materializer._git_command(
            source,
            "cat-file",
            "--batch-check=%(objectname) %(objecttype) %(objectsize)",
        ),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_scan_environment(),
    )
    total = 0
    try:
        for object_id in object_ids:
            total += _request_checked_blob_size(process, object_id)
            if total > byte_limit:
                raise _ScanError(
                    "Native Git history input exceeds its bounded byte limit"
                )
        materializer._finish_blob_process(process)
    finally:
        active_error = sys.exc_info()[1]
        cleanup_errors = materializer._cleanup_blob_process(process, ())
        if cleanup_errors and active_error is None:
            raise _ScanError("Could not clean up native Git blob-size stream") from (
                cleanup_errors[0]
            )
    return total


def _changed_blob_replay_upper_bound(
    source: Path,
    commits: list[str],
    byte_limit: int,
) -> tuple[int, int]:
    """Bound all blobs that native Gitleaks could reopen for archives."""
    object_format = materializer._object_format(source)
    total = 0
    metadata_bytes = 0
    for commit in commits:
        metadata = _read_native_diff_metadata(
            source,
            commit,
            _MAX_NATIVE_HISTORY_DIFF_METADATA_BYTES - metadata_bytes,
        )
        metadata_bytes += len(metadata)
        object_ids = _changed_blob_ids(metadata, object_format)
        total += _sum_changed_blob_sizes(source, object_ids, byte_limit - total)
    return total, metadata_bytes


def _measure_native_history_input(
    source: Path,
    history_range: str,
    commits: list[str],
) -> _NativeHistoryInput:
    """Bound patch and possible archive-blob replay for native scans."""
    per_scan_limit = _MAX_NATIVE_HISTORY_INPUT_BYTES // _NATIVE_HISTORY_SCAN_INVOCATIONS
    arguments = materializer._git_command(
        source,
        "log",
        "-p",
        "-U0",
        "--no-textconv",
        "--no-ext-diff",
        history_range,
    )
    with (
        tempfile.TemporaryFile() as stdout_sink,
        tempfile.TemporaryFile() as stderr_sink,
    ):
        result = subprocess.run(
            arguments,
            check=False,
            stdout=stdout_sink,
            stderr=stderr_sink,
            env=_scan_environment(),
            preexec_fn=partial(_limit_scanner_output_files, per_scan_limit),
        )
        output_bytes = os.fstat(stdout_sink.fileno()).st_size
        diagnostic_bytes = os.fstat(stderr_sink.fileno()).st_size
        if result.returncode != 0:
            raise _ScanError(
                "Native Git history input exceeds its bounded byte limit "
                "or could not be verified"
            )
        if diagnostic_bytes:
            if diagnostic_bytes > _MAX_GITLEAKS_DIAGNOSTIC_BYTES:
                raise _ScanError(
                    "Native Git history diagnostic exceeds the bounded byte limit"
                )
            raise _ScanError("Native Git history preflight emitted a diagnostic")
    blob_bytes, metadata_bytes = _changed_blob_replay_upper_bound(
        source,
        commits,
        per_scan_limit - output_bytes,
    )
    measured = _NativeHistoryInput(output_bytes, blob_bytes, metadata_bytes)
    if measured.bytes_per_scan > per_scan_limit:
        raise _ScanError("Native Git history input exceeds its bounded byte limit")
    return measured


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


def _policy_has_custom_rule_paths(
    content: bytes,
    *,
    content_rules_only: bool = False,
) -> bool:
    """Report whether trusted custom rules require exact-path associations."""
    document = tomllib.loads(content.decode("utf-8"))
    rules = document.get("rules", [])
    if not isinstance(rules, list):
        raise _ScanError("Gitleaks policy has an invalid custom rule list")
    has_path = False
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
        has_path = True
    return has_path


def _policy_path_matches_root_ignore(content: bytes) -> bool:
    """Conservatively report whether a custom rule could select the root ignore."""
    return _policy_has_custom_rule_paths(content)


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
    """Derive diverse controls without random compressed-stream collisions."""
    prefix = b"".join((b"Qw7eR9tY", b"2uI4oP6a", b"S8dF0gH1", b"jK3lZ5xC"))
    letters = "ABCDEFGHIJKLMNPQRSTUVWXYZ"
    digits = "012345679"
    tokens: list[bytes] = []
    for index in range(count):
        seed = hashlib.sha256(
            b"npa-gitleaks-private-control-v1:" + index.to_bytes(8, "big")
        ).digest()
        ranked_letters = sorted(
            letters,
            key=lambda letter: hashlib.sha256(seed + letter.encode()).digest(),
        )
        selected_digits = hashlib.shake_256(seed + b":digits").digest(12)
        suffix = "".join(
            character
            for pair in zip(
                ranked_letters[:12],
                (digits[value % len(digits)] for value in selected_digits),
                strict=True,
            )
            for character in pair
        ).encode("ascii")
        tokens.append(prefix + suffix)
    if len(set(tokens)) != count:
        raise _ScanError("Private scanner controls are not unique")
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
    candidate_exact_path_root = root / "tree-exact-paths"
    candidate_receipt = materializer.materialize(
        repo,
        snapshot,
        str(plan.candidate["commit"]),
        scan_view=True,
        exact_path_view=candidate_exact_path_root,
    )
    candidate_exact_path_views = sorted(
        path for path in candidate_exact_path_root.iterdir() if path.is_dir()
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
        candidate_exact_path_views,
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
        "derived_analysis_bytes": 0,
        "derived_bytes": 0,
        "derived_duplicates": 0,
        "derived_exact_path_bytes": 0,
        "derived_exact_path_files": 0,
        "derived_exact_path_identity_reuses": 0,
        "derived_exact_path_shards": 0,
        "derived_outputs": 0,
        "derived_streams": 0,
        "derived_transcoded_bytes": 0,
        "derived_transcoded_files": 0,
        "encoded_fdict_candidates": 0,
        "encoded_fdict_complete_streams": 0,
        "encoded_fdict_malformed_streams": 0,
        "encoded_fdict_unresolved_streams": 0,
        "direct_complete_empty_streams": 0,
        "direct_compressed_candidates": 0,
        "direct_duplicate_carriers": 0,
        "direct_emitted_bytes": 0,
        "direct_fdict_candidates": 0,
        "direct_fdict_complete_streams": 0,
        "direct_fdict_malformed_streams": 0,
        "direct_fdict_unresolved_streams": 0,
        "direct_invalid_streams": 0,
        "direct_output_occurrences": 0,
        "direct_rfc1950_candidates": 0,
        "direct_source_carriers": 0,
        "files": 0,
        "fragment_window_bytes": 0,
        "fragment_window_files": 0,
        "fragment_exact_path_bytes": 0,
        "fragment_exact_path_files": 0,
        "fragment_exact_path_shards": 0,
        "max_candidate_bytes": 0,
        "max_derived_bytes": 0,
        "nested_complete_empty_streams": 0,
        "nested_compressed_candidates": 0,
        "nested_emitted_bytes": 0,
        "nested_fdict_candidates": 0,
        "nested_fdict_complete_streams": 0,
        "nested_fdict_malformed_streams": 0,
        "nested_fdict_unresolved_streams": 0,
        "nested_invalid_streams": 0,
        "nested_max_depth": 0,
        "nested_output_occurrences": 0,
        "nested_rfc1950_candidates": 0,
        "wrapper_files": 0,
    }


def _prepare_probe_sources(
    root: Path,
    plan: _ScanPlan,
    sources: _MaterializedSources,
) -> _ProbeSources:
    """Build compact candidate and historical decode-probe views."""
    # Generic probe files add scanner framing before each output. Always retain
    # bounded source-start views so pathless absolute anchors see byte zero too.
    preserve_exact_paths = True
    candidate_view = root / "candidate-decode-probe"
    candidate_receipt = _prepare_decode_probe_source(
        sources.snapshot,
        candidate_view,
        preserve_exact_paths,
    )
    historical_view: Path | None = None
    historical_receipt = _empty_probe_receipt()
    if sources.historical_view is not None:
        historical_view = root / "historical-decode-probe"
        historical_receipt = _prepare_decode_probe_source(
            sources.historical_view,
            historical_view,
            preserve_exact_paths,
        )
    commit_metadata_view = root / "commit-metadata-decode-probe"
    commit_metadata_receipt = _prepare_decode_probe_source(
        sources.commit_metadata_view,
        commit_metadata_view,
        preserve_exact_paths,
        retain_direct_rfc1950_outputs=False,
    )
    transformed_exact_path_views = []
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
            "npa-gitleaks-derived-exact-paths",
            "npa-gitleaks-fragment-exact-paths",
        ):
            exact_root = probe_view / exact_root_name
            if exact_root.is_dir():
                detached_root = root / f"{probe_view.name}-{exact_root_name}"
                exact_root.rename(detached_root)
                transformed_exact_path_views.extend(
                    sorted(path for path in detached_root.iterdir() if path.is_dir())
                )
    return _ProbeSources(
        candidate_view,
        candidate_receipt,
        historical_view,
        historical_receipt,
        commit_metadata_view,
        commit_metadata_receipt,
        transformed_exact_path_views,
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
    """Hardlink every isolated encoded or derived stream into a recursion view."""
    destination = root / f"{label}-decode-candidates"
    destination.mkdir(mode=0o700)
    streams = sorted(source.glob("decode-input-candidates-*.txt")) + sorted(
        source.glob("decode-derived-output-*.txt")
    )
    if len(streams) != expected_streams:
        raise _ScanError("Decode-boundary candidate inventory is incomplete")
    for stream in streams:
        metadata = stream.lstat()
        if stream.is_symlink() or not stat.S_ISREG(metadata.st_mode):
            raise _ScanError("Decode-boundary stream must be a regular file")
        os.link(stream, destination / stream.name)
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
        probes.candidate_receipt["candidate_streams"]
        + probes.candidate_receipt["derived_streams"],
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
            probes.historical_receipt["candidate_streams"]
            + probes.historical_receipt["derived_streams"],
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
        probes.commit_metadata_receipt["candidate_streams"]
        + probes.commit_metadata_receipt["derived_streams"],
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
        b"* -filter -ident -text -working-tree-encoding diff\n",
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


def _quote_fast_import_path(path: bytes) -> bytes:
    """Return one byte-exact C-quoted path for Git fast-import."""
    if not path or b"\0" in path or path.startswith(b"/"):
        raise _ScanError("Exact-path history contains a noncanonical path")
    return b'"' + b"".join(f"\\{value:03o}".encode() for value in path) + b'"'


class _ExactPathFile(NamedTuple):
    """Bind one regular exact-path file to its measured content."""

    relative: str
    bytes: int
    sha256: str


class _ExactPathView(NamedTuple):
    """Bind one complete exact-path root to an immutable file manifest."""

    root: Path
    files: tuple[_ExactPathFile, ...]
    bytes: int


class _ExactPathBatch(NamedTuple):
    """Describe one complete-view forced-text history batch."""

    views: tuple[_ExactPathView, ...]
    files: int
    bytes: int


class _PreparedExactPathScan(NamedTuple):
    """Bind one validated generated history to its unique report."""

    repository: Path
    report: Path


class _PreparedExactPathPopulations(NamedTuple):
    """Collect every exact-path population frozen before Gitleaks runs."""

    candidate: tuple[_PreparedExactPathScan, ...]
    historical: tuple[_PreparedExactPathScan, ...]
    transformed: tuple[_PreparedExactPathScan, ...]


class _PreparedScanWorkspace(NamedTuple):
    """Collect all inputs frozen before the first Gitleaks process."""

    report_budget: _ReportBudget
    config: Path
    ignore: Path
    sources: _MaterializedSources
    probes: _ProbeSources
    native_history_input: _NativeHistoryInput
    exact_paths: _PreparedExactPathPopulations


def _stable_stat(metadata: os.stat_result) -> tuple[int, ...]:
    """Return the fields that must not change while a file is read."""
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _enumerate_exact_path_files(
    view: Path,
) -> list[tuple[Path, str, os.stat_result]]:
    """Return one deterministic regular-file inventory without following links."""
    root_metadata = view.lstat()
    if view.is_symlink() or not stat.S_ISDIR(root_metadata.st_mode):
        raise _ScanError("Exact-path scan view must be a regular directory")
    git_metadata = view / ".git"
    if git_metadata.exists() or git_metadata.is_symlink():
        raise _ScanError("Exact-path scan view already contains Git metadata")
    entries: list[tuple[Path, str, os.stat_result]] = []
    paths = sorted(
        view.rglob("*"),
        key=lambda path: os.fsencode(path.relative_to(view).as_posix()),
    )
    for path in paths:
        metadata = path.lstat()
        if path.is_symlink():
            raise _ScanError("Exact-path history contains a symbolic link")
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise _ScanError("Exact-path history contains a non-regular file")
        entries.append((path, path.relative_to(view).as_posix(), metadata))
    return entries


def _hash_exact_path_file(path: Path, metadata: os.stat_result) -> tuple[int, str]:
    """Hash one regular file while proving its opened identity stayed stable."""
    digest = hashlib.sha256()
    total = 0
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as stream:
        opened = os.fstat(stream.fileno())
        if not stat.S_ISREG(opened.st_mode) or (
            opened.st_dev,
            opened.st_ino,
        ) != (metadata.st_dev, metadata.st_ino):
            raise _ScanError("Exact-path file changed while it was measured")
        while chunk := stream.read(_PROBE_DECODE_CHUNK_BYTES):
            digest.update(chunk)
            total += len(chunk)
        closed = os.fstat(stream.fileno())
    if _stable_stat(opened) != _stable_stat(closed) or total != metadata.st_size:
        raise _ScanError("Exact-path file changed while it was measured")
    return total, digest.hexdigest()


def _measure_exact_path_view(view: Path) -> _ExactPathView:
    """Validate and bind one private exact-path view to its content."""
    files: list[_ExactPathFile] = []
    total_bytes = 0
    for path, relative, metadata in _enumerate_exact_path_files(view):
        file_bytes, digest = _hash_exact_path_file(path, metadata)
        files.append(_ExactPathFile(relative, file_bytes, digest))
        total_bytes += file_bytes
        if len(files) > _MAX_ENCODED_CANDIDATE_FILES:
            raise _ScanError("Exact-path view exceeds the bounded file count")
        if total_bytes > _MAX_EXACT_PATH_HISTORY_BYTES:
            raise _ScanError("Exact-path view exceeds the bounded byte count")
    if not files:
        raise _ScanError("Exact-path scan view is empty")
    return _ExactPathView(view, tuple(files), total_bytes)


def _partition_exact_path_views(views: list[Path]) -> list[_ExactPathBatch]:
    """Partition complete views under unchanged per-history bounds."""
    if not views:
        return []
    if len(views) > _MAX_EXACT_PATH_SHARDS:
        raise _ScanError("Archive exact-path views exceed the shard limit")
    batches: list[_ExactPathBatch] = []
    current: list[_ExactPathView] = []
    current_files = 0
    current_bytes = 0
    for view in views:
        measured = _measure_exact_path_view(view)
        view_files = len(measured.files)
        view_bytes = measured.bytes
        exceeds = (
            current_files + view_files > _MAX_ENCODED_CANDIDATE_FILES
            or current_bytes + view_bytes > _MAX_EXACT_PATH_HISTORY_BYTES
        )
        if current and exceeds:
            batches.append(
                _ExactPathBatch(tuple(current), current_files, current_bytes)
            )
            if len(batches) >= _MAX_EXACT_PATH_HISTORY_BATCHES:
                raise _ScanError("Exact-path histories exceed the bounded batch count")
            current = []
            current_files = 0
            current_bytes = 0
        current.append(measured)
        current_files += view_files
        current_bytes += view_bytes
    batches.append(_ExactPathBatch(tuple(current), current_files, current_bytes))
    if len(batches) > _MAX_EXACT_PATH_HISTORY_BATCHES:
        raise _ScanError("Exact-path histories exceed the bounded batch count")
    return batches


def _require_exact_path_inventory(view: _ExactPathView) -> None:
    """Require the current regular-file inventory to match its bound manifest."""
    current = tuple(
        (relative, metadata.st_size)
        for _path, relative, metadata in _enumerate_exact_path_files(view.root)
    )
    expected = tuple((entry.relative, entry.bytes) for entry in view.files)
    if current != expected:
        raise _ScanError("Exact-path view changed after it was measured")


@contextmanager
def _spool_exact_path_file(
    view: _ExactPathView,
    planned: _ExactPathFile,
) -> Iterator[BinaryIO]:
    """Copy one still-bound file into private validated temporary storage."""
    path = view.root.joinpath(*PurePosixPath(planned.relative).parts)
    metadata = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise _ScanError("Exact-path file type changed after it was measured")
    if metadata.st_size != planned.bytes:
        raise _ScanError("Exact-path file size changed after it was measured")
    with tempfile.TemporaryFile() as spool:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        digest = hashlib.sha256()
        total = 0
        with os.fdopen(descriptor, "rb") as source:
            opened = os.fstat(source.fileno())
            if not stat.S_ISREG(opened.st_mode) or (
                opened.st_dev,
                opened.st_ino,
            ) != (metadata.st_dev, metadata.st_ino):
                raise _ScanError("Exact-path file changed after it was measured")
            while chunk := source.read(_PROBE_DECODE_CHUNK_BYTES):
                spool.write(chunk)
                digest.update(chunk)
                total += len(chunk)
            closed = os.fstat(source.fileno())
        if (
            _stable_stat(opened) != _stable_stat(closed)
            or total != planned.bytes
            or digest.hexdigest() != planned.sha256
        ):
            raise _ScanError("Exact-path file changed after it was measured")
        spool.seek(0)
        yield spool


def _stream_exact_path_file(
    stream: BinaryIO,
    view: _ExactPathView,
    planned: _ExactPathFile,
) -> None:
    """Write one validated manifest entry to Git fast-import."""
    with _spool_exact_path_file(view, planned) as spool:
        relative = os.fsencode(planned.relative)
        stream.write(b"M 100644 inline " + _quote_fast_import_path(relative) + b"\n")
        stream.write(f"data {planned.bytes}\n".encode())
        while chunk := spool.read(_PROBE_DECODE_CHUNK_BYTES):
            stream.write(chunk)
        stream.write(b"\n")


def _prepare_forced_text_history(
    repository: Path,
    batch: _ExactPathBatch,
    config_content: bytes,
) -> tuple[str, ...]:
    """Commit every bound exact-path variant as an independent forced-text root."""
    if not batch.views:
        raise _ScanError("Exact-path forced-text history is empty")
    if len(batch.views) > _MAX_EXACT_PATH_SHARDS:
        raise _ScanError("Archive exact-path views exceed the shard limit")
    if sum(len(view.files) for view in batch.views) != batch.files:
        raise _ScanError("Exact-path history file manifest is inconsistent")
    if sum(view.bytes for view in batch.views) != batch.bytes:
        raise _ScanError("Exact-path history byte manifest is inconsistent")
    repository.mkdir(mode=0o700)
    environment = materializer._git_environment()

    def run(*arguments: str) -> bytes:
        return subprocess.run(
            materializer._git_command(repository, *arguments),
            check=True,
            capture_output=True,
            env=environment,
        ).stdout

    run("init", "--quiet", "--template=", "--initial-branch=npa-scan")
    attributes = repository / ".git" / "info" / "attributes"
    attributes.parent.mkdir(mode=0o700)
    _write_private(
        attributes,
        b"* -filter -ident -text -working-tree-encoding diff\n",
    )
    identity = b"NPA exact-path scanner <npa-scan@example.invalid> 946684800 +0000"
    message = b"NPA exact-path scanner view"
    total_files = 0
    total_bytes = 0
    with (
        tempfile.TemporaryFile() as stdout_sink,
        tempfile.TemporaryFile() as stderr_sink,
    ):
        process = subprocess.Popen(
            materializer._git_command(
                repository,
                "fast-import",
                "--quiet",
                "--done",
            ),
            stdin=subprocess.PIPE,
            stdout=stdout_sink,
            stderr=stderr_sink,
            env=environment,
        )
        assert process.stdin is not None
        try:
            for index, view in enumerate(batch.views):
                _require_exact_path_inventory(view)
                process.stdin.write(f"commit refs/npa-exact-path/{index}\n".encode())
                process.stdin.write(b"author " + identity + b"\n")
                process.stdin.write(b"committer " + identity + b"\n")
                process.stdin.write(f"data {len(message)}\n".encode())
                process.stdin.write(message + b"\n")
                process.stdin.write(b"deleteall\n")
                for planned in view.files:
                    total_files += 1
                    total_bytes += planned.bytes
                    if total_files > _MAX_ENCODED_CANDIDATE_FILES:
                        raise _ScanError(
                            "Exact-path history exceeds the bounded file count"
                        )
                    if total_bytes > _MAX_EXACT_PATH_HISTORY_BYTES:
                        raise _ScanError(
                            "Exact-path history exceeds the bounded byte count"
                        )
                    _stream_exact_path_file(process.stdin, view, planned)
                _require_exact_path_inventory(view)
            if total_files != batch.files or total_bytes != batch.bytes:
                raise _ScanError("Exact-path history differs from its manifest totals")
            process.stdin.write(b"done\n")
            process.stdin.close()
            returncode = process.wait()
        except BrokenPipeError as error:
            try:
                process.stdin.close()
            except BrokenPipeError:
                pass
            process.wait()
            diagnostic = _read_scanner_diagnostic(stderr_sink, "Git fast-import")
            raise _ScanError(
                "Could not build exact-path forced-text history: "
                + diagnostic.decode("utf-8", errors="replace")
            ) from error
        except BaseException:
            try:
                process.stdin.close()
            except BrokenPipeError:
                pass
            if process.poll() is None:
                process.terminate()
            process.wait()
            raise
        if returncode:
            diagnostic = _read_scanner_diagnostic(stderr_sink, "Git fast-import")
            raise _ScanError(
                "Could not build exact-path forced-text history: "
                + diagnostic.decode("utf-8", errors="replace")
            )
    commits = tuple(
        line.strip().lower()
        for line in run(
            "for-each-ref",
            "--format=%(objectname)",
            "refs/npa-exact-path/",
        )
        .decode("ascii")
        .splitlines()
        if line.strip()
    )
    if len(commits) != len(batch.views):
        raise _ScanError("Exact-path forced-text history omitted a root")
    if _policy_commit_allowlists(config_content).intersection(commits):
        raise _ScanError(
            "Policy commit allowlist suppresses a generated exact-path scan view"
        )
    return commits


def _run_gitleaks_exact_views(
    executable: str,
    repository: Path,
    views: list[Path],
    config: Path,
    ignore: Path,
    report: Path,
    config_content: bytes,
    report_budget: _ReportBudget,
) -> tuple[subprocess.CompletedProcess[bytes], list[dict[str, object]]]:
    """Scan all exact-path roots once without candidate-controlled Git behavior."""
    batches = _partition_exact_path_views(views)
    if len(batches) != 1:
        raise _ScanError("Exact-path views require multiple bounded histories")
    _prepare_forced_text_history(repository, batches[0], config_content)
    return _run_gitleaks(
        executable,
        repository,
        config,
        ignore,
        report,
        history_range="--all",
        report_budget=report_budget,
    )


def _prepare_exact_path_scans(
    root: Path,
    views: list[Path],
    label: str,
    config_content: bytes,
) -> tuple[_PreparedExactPathScan, ...]:
    """Freeze one population into validated, uniquely named Git histories."""
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    prepared = []
    for index, batch in enumerate(_partition_exact_path_views(views)):
        repository = root / f"{label}-history-{index:02d}"
        report = root / f"{label}-git-{index:02d}.json"
        _prepare_forced_text_history(repository, batch, config_content)
        prepared.append(_PreparedExactPathScan(repository, report))
    return tuple(prepared)


def _scan_prepared_exact_path_scans(
    executable: str,
    prepared: tuple[_PreparedExactPathScan, ...],
    config: Path,
    ignore: Path,
    report_budget: _ReportBudget,
) -> list[tuple[subprocess.CompletedProcess[bytes], list[dict[str, object]]]]:
    """Scan every already validated exact-path history."""
    return [
        _run_gitleaks(
            executable,
            scan.repository,
            config,
            ignore,
            scan.report,
            history_range="--all",
            report_budget=report_budget,
        )
        for scan in prepared
    ]


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
    """Scan complete exact-path views in deterministic bounded histories."""
    prepared = _prepare_exact_path_scans(root, views, label, config_content)
    return _scan_prepared_exact_path_scans(
        executable,
        prepared,
        config,
        ignore,
        report_budget,
    )


def _prepare_exact_path_populations(
    root: Path,
    sources: _MaterializedSources,
    probes: _ProbeSources,
    config_content: bytes,
) -> _PreparedExactPathPopulations:
    """Freeze every exact-path population before any Gitleaks invocation."""
    candidate = _prepare_exact_path_scans(
        root,
        sources.candidate_exact_path_views,
        "candidate-exact-path",
        config_content,
    )
    historical = _prepare_exact_path_scans(
        root,
        sources.historical_exact_path_views,
        "historical-exact-path",
        config_content,
    )
    transformed = _prepare_exact_path_scans(
        root,
        probes.transformed_exact_path_views,
        "transformed-exact-path",
        config_content,
    )
    return _PreparedExactPathPopulations(candidate, historical, transformed)


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
    transformed_exact_path_scans: list[
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
    for scan in transformed_exact_path_scans:
        _require_clean_scan("transformed-exact-path", *scan)
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
    exact_paths: _PreparedExactPathPopulations,
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
    candidate_exact_path_scans = _scan_prepared_exact_path_scans(
        executable,
        exact_paths.candidate,
        config,
        ignore,
        report_budget,
    )
    historical_exact_path_scans = _scan_prepared_exact_path_scans(
        executable,
        exact_paths.historical,
        config,
        ignore,
        report_budget,
    )
    transformed_exact_path_scans = _scan_prepared_exact_path_scans(
        executable,
        exact_paths.transformed,
        config,
        ignore,
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
        transformed_exact_path_scans,
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
    transformed_exact_path_findings = [
        finding
        for _result, findings in transformed_exact_path_scans
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
        transformed_exact_path_findings,
        candidate_probe_findings,
        historical_probe_findings,
        commit_metadata_probe_findings,
        len(candidate_exact_path_scans),
        len(historical_exact_path_scans),
        len(transformed_exact_path_scans),
    )


def _render_scan_receipt(
    plan: _ScanPlan,
    sources: _MaterializedSources,
    probes: _ProbeSources,
    findings: _FindingSets,
    native_history_input: _NativeHistoryInput,
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
        + len(findings.transformed_exact_paths)
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
        "native_history_patch_bytes_per_scan": native_history_input.patch_bytes,
        "native_history_blob_bytes_upper_bound_per_scan": (
            native_history_input.changed_blob_bytes_upper_bound
        ),
        "native_history_diff_metadata_bytes": native_history_input.diff_metadata_bytes,
        "native_history_input_bytes_per_scan": native_history_input.bytes_per_scan,
        "native_history_input_bytes": (
            native_history_input.bytes_per_scan * _NATIVE_HISTORY_SCAN_INVOCATIONS
        ),
        "native_history_input_byte_limit": _MAX_NATIVE_HISTORY_INPUT_BYTES,
        "native_history_scan_invocations": _NATIVE_HISTORY_SCAN_INVOCATIONS,
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
        "transformed_exact_path_findings": len(findings.transformed_exact_paths),
        "candidate_exact_path_scan_invocations": (
            findings.candidate_exact_path_scan_invocations
        ),
        "historical_exact_path_scan_invocations": (
            findings.historical_exact_path_scan_invocations
        ),
        "transformed_exact_path_scan_invocations": (
            findings.transformed_exact_path_scan_invocations
        ),
        "exact_path_history_file_limit": _MAX_ENCODED_CANDIDATE_FILES,
        "exact_path_history_byte_limit": _MAX_EXACT_PATH_HISTORY_BYTES,
        "exact_path_history_batch_limit": _MAX_EXACT_PATH_HISTORY_BATCHES,
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


def _prepare_scan_workspace(
    repo: Path,
    root: Path,
    plan: _ScanPlan,
) -> _PreparedScanWorkspace:
    """Freeze every target-scan input before executing Gitleaks."""
    root.chmod(0o700)
    report_budget = _ReportBudget()
    config, ignore = _write_scan_policy(repo, root, plan)
    sources = _materialize_sources(repo, root, plan)
    probes = _prepare_probe_sources(root, plan, sources)
    _verify_root_ignore_exact_path_coverage(plan, sources, probes)
    native_history_input = _measure_native_history_input(
        sources.history_source,
        plan.commit_range,
        plan.commits,
    )
    exact_paths = _prepare_exact_path_populations(
        root,
        sources,
        probes,
        plan.config_content,
    )
    return _PreparedScanWorkspace(
        report_budget,
        config,
        ignore,
        sources,
        probes,
        native_history_input,
        exact_paths,
    )


def _scan_workspace(
    repo: Path,
    root: Path,
    plan: _ScanPlan,
    executable: str,
    prepared: _PreparedScanWorkspace | None = None,
) -> dict[str, object]:
    """Execute all controls and scans inside one prepared private workspace."""
    if prepared is None:
        prepared = _prepare_scan_workspace(repo, root, plan)
    report_budget = prepared.report_budget
    config = prepared.config
    ignore = prepared.ignore
    sources = prepared.sources
    probes = prepared.probes
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
        prepared.exact_paths,
    )
    return _render_scan_receipt(
        plan,
        sources,
        probes,
        findings,
        prepared.native_history_input,
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
        prepared = _prepare_scan_workspace(repo, root, plan)
        staged_executable, executable_sha256 = _stage_gitleaks_executable(
            executable,
            root,
        )
        receipt = _scan_workspace(
            repo,
            root,
            plan,
            staged_executable,
            prepared,
        )
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
