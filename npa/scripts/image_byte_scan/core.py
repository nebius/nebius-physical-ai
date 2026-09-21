"""Complete image byte accounting with explicit roots and immutable inputs."""

from __future__ import annotations

import argparse
import base64
import collections
import csv
import fcntl
import hashlib
import importlib.machinery
import importlib.util
import io
import json
import multiprocessing
import os
import re
import resource
import signal
import select
import stat
import struct
import subprocess
import sys
import tarfile
import types
import zipfile
import zlib
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict
from pathlib import Path, PurePosixPath

from . import confidentiality as C

_ROOTS = ContextVar("image_byte_scan_authorized_roots", default=None)
CHUNK = 1024 * 1024
MAX_DETECTION_WORKERS = 64


def schedulable_cpus():
    """Report how many CPUs this process may be scheduled on.

    Scans run on Linux, where the helper is a Go child whose default worker count
    follows the same affinity mask. Other platforms only import this module to run
    the hermetic tests and have no affinity call, so they fall back to the
    machine's CPU count.

    Args:
        None.

    Returns:
        The affinity mask size capped at the scanner's worker limit, or the
        equivalently capped machine CPU count where no mask is exposed, and
        never less than one.

    Raises:
        None.
    """
    if hasattr(os, "sched_getaffinity"):
        return max(1, min(len(os.sched_getaffinity(0)), MAX_DETECTION_WORKERS))
    return max(1, min(os.cpu_count() or 1, MAX_DETECTION_WORKERS))


# Records submitted to the helper before their results are collected. The helper
# runs one detection worker per schedulable CPU, so it can only use more than one
# core when more than one record is outstanding. Record sizes inside an image span
# several orders of magnitude, so a depth of one record per worker leaves workers
# idle whenever the outstanding set happens to be small records; several per worker
# keeps them fed. Ledger output stays in stream order for every depth, so this
# bounds resident memory and nothing else.
PIPELINE_RECORDS = 4 * schedulable_cpus()
# Payload bytes held for outstanding records, checked before the next record is
# read. Confidentiality composition needs each complete record, so a run of large
# records reduces the effective depth instead of growing resident memory. This
# bounds the records this process holds for composition; the record being read,
# the helper's own admitted bytes, and the detector's copies of them are separate
# and are bounded on the helper side.
PIPELINE_BYTES = 256 * 1024 * 1024
# PAX/GNU extension bodies are metadata, not file payloads. One MiB is well
# above practical path/xattr limits while bounding attacker-directed allocation.
TAR_EXTENSION_LIMIT = 1024 * 1024
TAR_EXTENSION_CHAIN_LIMIT = 4 * 1024 * 1024
TAR_PAX_KEY_LIMIT = 4096
TAR_PATH_LIMIT = 4096
ZERO_RECORD_LIMIT = 1024 * 1024
GZIP_HEADER_LIMIT = 1024 * 1024
# Public GPU images legitimately contain large native libraries and many files.
# These ceilings leave operational headroom while making every attacker-driven
# decoded population, path set, and complete-record allocation finite.
TAR_ENTRY_LIMIT = 250_000
# Current native-helper measurements peak at 9.85x payload bytes for one record.
# Round that observation up, retain explicit fixed/process headroom, and derive
# the admitted record size from the address-space ceiling. The multiplier is not
# treated as a promise: drift still terminates the worker fail closed at RLIMIT_AS.
DETECTOR_MEMORY_LIMIT = 12 * 1024 * 1024 * 1024
DETECTOR_MEMORY_HEADROOM = 4 * 1024 * 1024 * 1024
DETECTOR_PAYLOAD_EXPANSION = 10
DETECTOR_RECORD_LIMIT = (
    DETECTOR_MEMORY_LIMIT - DETECTOR_MEMORY_HEADROOM
) // DETECTOR_PAYLOAD_EXPANSION
TAR_REGULAR_FILE_LIMIT = DETECTOR_RECORD_LIMIT
DOCKER_SAVE_DECODED_LAYER_LIMIT = 64 * 1024 * 1024 * 1024
CONFIDENTIALITY_RECORD_LIMIT = DETECTOR_RECORD_LIMIT
CONFIDENTIALITY_MEMORY_LIMIT = DETECTOR_MEMORY_LIMIT
HELPER_RESPONSE_LIMIT = 8 * 1024 * 1024
RECORD_FINDING_LIMIT = 4096
SCAN_FINDING_LIMIT = 100_000
LITERAL_MATCH_CHUNK = 64 * 1024
LITERAL_PATTERN_LIMIT = 4096
LITERAL_VALUE_BYTES_LIMIT = 64 * 1024
LITERAL_TOTAL_BYTES_LIMIT = 8 * 1024 * 1024
LITERAL_INVENTORY_JSON_LIMIT = 16 * 1024 * 1024
LEDGER_LINE_LIMIT = 16 * 1024 * 1024
# Docker-save's outer archive contains only short, regular metadata/blob paths.
# Bound the population and every JSON document before handing the archive to
# tarfile, whose PAX/GNU handlers otherwise consume extension bodies eagerly.
DOCKER_SAVE_OUTER_ENTRY_LIMIT = 100_000
DOCKER_SAVE_METADATA_LIMIT = 16 * 1024 * 1024
DOCKER_SAVE_LAYER_LIMIT = 1024
DOCKER_SAVE_LAYER_MEMBER_REPEAT_LIMIT = 8
_CANCEL_REQUESTED = False
_SPAWNING = False
POLICY = "exact-or-short-ascii-token-v1"
REMOVED_PATH_RULES = [
    "freemius-secret-key",
    "hashicorp-tf-password",
    "kubernetes-secret-yaml",
    "nuget-config-password",
]
PKCS12 = re.compile(r"(?i)(?:^|/)[^/]+\.p(?:12|fx)$")
DIGEST = re.compile(r"sha256:[0-9a-f]{64}$")
SHA = re.compile(r"[0-9a-f]{64}$")
PAX_TEXT_KEYS = frozenset(
    {
        "path",
        "linkpath",
        "uid",
        "gid",
        "uname",
        "gname",
        "mtime",
        "atime",
        "ctime",
    }
)
PAX_BINARY_PREFIXES = ("SCHILY.xattr.", "LIBARCHIVE.xattr.")


class ScanError(ValueError):
    pass


class LiteralFindingLimit(ScanError):
    pass


INPUT_ERRORS = (
    OSError,
    ValueError,
    TypeError,
    KeyError,
    EOFError,
    tarfile.TarError,
    zlib.error,
    MemoryError,
    KeyboardInterrupt,
    RuntimeError,
    IndexError,
    AttributeError,
    OverflowError,
    ImportError,
    SyntaxError,
    SystemError,
    zipfile.BadZipFile,
)


def require(condition, code):
    if not condition:
        raise ScanError(code)


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


@contextmanager
def authorized_roots(analysis_root, trusted_root):
    """Caller authorizes private artifacts separately from checked-out source."""
    roots = []
    for value, secret in ((analysis_root, True), (trusted_root, False)):
        path = Path(value).absolute()
        require(".." not in path.parts, "root_parent_component")
        cursor = Path(path.anchor)
        for part in path.parts[1:]:
            cursor /= part
            require(not cursor.is_symlink(), "root_symlink")
        info = path.stat()
        require(
            stat.S_ISDIR(info.st_mode) and info.st_uid == os.geteuid(),
            "root_owner_or_type",
        )
        require(not info.st_mode & (0o077 if secret else 0o022), "root_permissions")
        roots.append(path)
    require(not roots[0].is_relative_to(roots[1]), "analysis_root_inside_build_context")
    token = _ROOTS.set(tuple(roots))
    identities = [(p.stat().st_dev, p.stat().st_ino) for p in roots]
    try:
        yield tuple(roots)
        for path, identity in zip(roots, identities, strict=True):
            current = path.lstat()
            require(
                stat.S_ISDIR(current.st_mode)
                and (current.st_dev, current.st_ino) == identity,
                "authorized_root_changed",
            )
    finally:
        _ROOTS.reset(token)


def source_bindings():
    """Exact executable/configuration closure, excluding generated caches/binaries."""
    roots = _ROOTS.get()
    require(roots is not None, "explicit_roots_required")
    checkout = roots[1]
    require(
        Path(__file__).absolute() == checkout / "npa/scripts/image_byte_scan/core.py",
        "trusted_source_root_mismatch",
    )
    folder = checkout / "npa/scripts/image_byte_scan"
    sources = [
        checkout / ".gitleaks.toml",
        checkout / "npa/scripts/scan_image_bytes.py",
        checkout / "npa/tests/docker/test_image_byte_go_build.py",
    ]
    sources.extend(
        path
        for path in folder.rglob("*")
        if path.is_file()
        and (
            path.suffix in {".py", ".go", ".mod", ".sum", ".json", ".md"}
            or path.name.startswith("LICENSE")
        )
    )
    result = {}
    for path in sorted(sources):
        with open_source_fd(path) as (fd, _info):
            result[str(path.relative_to(checkout))] = {
                "path": str(path),
                "sha256": descriptor_digest(fd),
            }
    return result


@contextmanager
def open_source_fd(path):
    _path, fd, info = open_private_fd(path, secret=False)
    try:
        yield fd, info
        require(
            stat_fingerprint(os.fstat(fd)) == stat_fingerprint(info),
            "source_changed_during_read",
        )
    finally:
        os.close(fd)


def private_path(value, *, secret=True):
    path = Path(value).absolute()
    cursor = Path(path.anchor)
    for component in path.parts[1:]:
        cursor = cursor.parent if component == ".." else cursor / component
        require(not cursor.is_symlink(), "input_symlink")
    path = path.resolve(strict=True)
    roots = _ROOTS.get()
    require(roots is not None, "explicit_roots_required")
    require(
        (path.is_relative_to(roots[0]) and not path.is_relative_to(roots[1]))
        or (not secret and path.is_relative_to(roots[1])),
        "input_outside_authorized_roots",
    )
    info = path.stat()
    require(
        stat.S_ISREG(info.st_mode) and info.st_uid == os.geteuid(),
        "input_ownership_or_type",
    )
    require(not info.st_mode & (0o077 if secret else 0o022), "input_permissions")
    return path


def open_private_fd(value, *, secret=True):
    """Walk each component by nofollow directory descriptor; never block on FIFO."""
    path = private_path(value, secret=secret)
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
    directory_fd = os.open(path.anchor, flags | os.O_DIRECTORY)
    try:
        for component in path.parts[1:-1]:
            child_fd = os.open(component, flags | os.O_DIRECTORY, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = child_fd
        fd = os.open(path.name, flags, dir_fd=directory_fd)
    finally:
        os.close(directory_fd)
    try:
        info = os.fstat(fd)
        require(
            stat.S_ISREG(info.st_mode) and info.st_uid == os.geteuid(),
            "input_ownership_or_type",
        )
        require(not info.st_mode & (0o077 if secret else 0o022), "input_permissions")
        return path, fd, info
    except BaseException:
        os.close(fd)
        raise


def descriptor_digest(fd):
    digest, offset = hashlib.sha256(), 0
    while data := os.pread(fd, CHUNK, offset):
        digest.update(data)
        offset += len(data)
    return digest.hexdigest()


def descriptor_bytes(fd, *, byte_limit=None, limit_code="input_byte_limit"):
    if byte_limit is not None:
        require(type(byte_limit) is int and byte_limit >= 0, limit_code)
    parts, offset = [], 0
    while True:
        amount = CHUNK if byte_limit is None else min(CHUNK, byte_limit + 1 - offset)
        require(amount > 0, limit_code)
        data = os.pread(fd, amount, offset)
        if not data:
            break
        parts.append(data)
        offset += len(data)
        if byte_limit is not None:
            require(offset <= byte_limit, limit_code)
    return b"".join(parts)


@contextmanager
def bound_open(spec, *, secret=True):
    require(
        isinstance(spec, dict) and isinstance(spec.get("path"), str),
        "input_binding_schema",
    )
    require(
        isinstance(spec.get("sha256"), str) and SHA.fullmatch(spec["sha256"]),
        "input_binding_digest",
    )
    path, fd, info = open_private_fd(spec["path"], secret=secret)
    try:
        require(descriptor_digest(fd) == spec["sha256"], "input_binding_changed")
        require(
            stat_fingerprint(os.fstat(fd)) == stat_fingerprint(info),
            "input_changed_during_read",
        )
        yield path, fd, info
        require(
            stat_fingerprint(os.fstat(fd)) == stat_fingerprint(info),
            "input_changed_during_read",
        )
    finally:
        os.close(fd)


def bound_file(spec, *, secret=True):
    with bound_open(spec, secret=secret) as (path, _fd, _info):
        return path


@contextmanager
def sealed_execution_input(source_fd, expected_digest, *, executable=False):
    """Execute/read verified immutable bytes while retaining original-file audits."""
    fd = os.memfd_create(
        "verified-scanner-input", os.MFD_ALLOW_SEALING | os.MFD_CLOEXEC
    )
    try:
        os.fchmod(fd, 0o500 if executable else 0o400)
        offset = 0
        while data := os.pread(source_fd, CHUNK, offset):
            offset += len(data)
            remaining = memoryview(data)
            while remaining:
                written = os.write(fd, remaining)
                require(written > 0, "execution_input_short_write")
                remaining = remaining[written:]
        # Linux UAPI constants, also used for the verified native literal matcher.
        fcntl.fcntl(fd, 1033, 15)
        require(fcntl.fcntl(fd, 1034) == 15, "execution_input_not_sealed")
        require(
            descriptor_digest(fd) == expected_digest, "execution_input_digest_changed"
        )
        os.lseek(fd, 0, os.SEEK_SET)
        yield fd
    finally:
        os.close(fd)


def json_object(data):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate_json_key")
            result[key] = value
        return result

    return json.loads(data, object_pairs_hook=pairs)


def bounded_json_line(stream, limit=HELPER_RESPONSE_LIMIT):
    line = stream.readline(limit + 1)
    require(bool(line), "helper_unexpected_eof")
    require(len(line) <= limit and line.endswith(b"\n"), "helper_response_limit")
    return json_object(line)


def bound_json(spec, *, byte_limit=None, limit_code="json_input_limit"):
    with bound_open(spec) as (_path, fd, info):
        if byte_limit is not None:
            require(
                type(byte_limit) is int
                and byte_limit >= 0
                and info.st_size <= byte_limit,
                limit_code,
            )
        data = descriptor_bytes(
            fd,
            byte_limit=byte_limit,
            limit_code=limit_code,
        )
        require(sha(data) == spec["sha256"], "parsed_input_binding_changed")
        return json_object(data)


class Slice:
    """Position-independent bounded reads from the one opened archive inode."""

    def __init__(self, fd, offset, length):
        self.fd, self.offset, self.length, self.position = fd, offset, length, 0

    def tell(self):
        return self.position

    def read(self, amount=-1):
        remaining = self.length - self.position
        amount = remaining if amount < 0 else min(amount, remaining)
        if not amount:
            return b""
        result = os.pread(self.fd, amount, self.offset + self.position)
        require(result, "archive_short_read")
        self.position += len(result)
        return result


class ZeroReader:
    """Generate a logical zero range without materializing the whole range."""

    def __init__(self, length):
        require(type(length) is int and length >= 0, "zero_reader_length")
        self.remaining = length

    def read(self, amount=-1):
        require(type(amount) is int, "zero_reader_amount")
        requested = self.remaining if amount < 0 else amount
        count = min(requested, self.remaining, CHUNK)
        self.remaining -= count
        return bytes(count)


class HashedReader:
    def __init__(self, source):
        self.source, self.digest, self.position = source, hashlib.sha256(), 0

    def tell(self):
        return self.position

    def read(self, amount=-1):
        data = self.source.read(amount)
        self.digest.update(data)
        self.position += len(data)
        return data


class GzipReader:
    """Exactly one gzip member; zlib verifies CRC32 and ISIZE."""

    def __init__(self, source):
        self.source = source
        self.decoder = zlib.decompressobj(31)
        self.pending = b""
        self.finished = False
        self.position = 0

    def tell(self):
        return self.position

    def read(self, amount):
        require(amount >= 0, "unbounded_decoder_read")
        output = bytearray()
        while len(output) < amount and not self.finished:
            if not self.pending:
                self.pending = self.source.read(CHUNK)
                require(self.pending, "truncated_gzip")
            data = self.decoder.decompress(self.pending, amount - len(output))
            self.pending = self.decoder.unconsumed_tail
            output.extend(data)
            if self.decoder.eof:
                require(
                    not self.decoder.unused_data and not self.pending,
                    "gzip_trailing_member_or_bytes",
                )
                require(not self.source.read(1), "gzip_trailing_member_or_bytes")
                self.finished = True
        self.position += len(output)
        return bytes(output)


def read_exact(reader, count, *, eof=False):
    data = bytearray()
    while len(data) < count:
        part = reader.read(count - len(data))
        if not part:
            require(eof and not data, "truncated_tar_range")
            return b""
        data.extend(part)
    return bytes(data)


def preflight_docker_save_outer_tar(fd, length):
    """Bound outer Docker-save parsing before tarfile sees attacker metadata.

    Docker/OCI save paths fit in one ustar header. PAX and GNU name extensions
    are therefore unnecessary in the outer transport and are rejected before
    tarfile can eagerly allocate their bodies. Layer tar streams retain bounded
    extension support in :func:`walk_tar`.
    """
    require(type(length) is int and length >= 1024, "docker_save_outer_size")
    allowed = {
        tarfile.REGTYPE,
        tarfile.AREGTYPE,
        tarfile.DIRTYPE,
        tarfile.SYMTYPE,
        tarfile.LNKTYPE,
    }
    offset = entries = zero_headers = 0
    while offset + 512 <= length:
        raw = os.pread(fd, 512, offset)
        require(len(raw) == 512, "docker_save_outer_truncated")
        offset += 512
        if not any(raw):
            zero_headers += 1
            if zero_headers < 2:
                continue
            while offset < length:
                data = os.pread(fd, min(CHUNK, length - offset), offset)
                require(bool(data), "docker_save_outer_truncated")
                require(not any(data), "docker_save_outer_nonzero_trailer")
                offset += len(data)
            return entries
        require(zero_headers == 0, "docker_save_outer_incomplete_end_markers")
        info = tarfile.TarInfo.frombuf(raw, encoding="utf-8", errors="strict")
        require(
            info.type
            not in {
                tarfile.XHDTYPE,
                tarfile.XGLTYPE,
                tarfile.GNUTYPE_LONGNAME,
                tarfile.GNUTYPE_LONGLINK,
            },
            "docker_save_outer_extension_unsupported",
        )
        require(
            info.type in allowed and info.size >= 0,
            "docker_save_outer_entry_type",
        )
        entries += 1
        require(
            entries <= DOCKER_SAVE_OUTER_ENTRY_LIMIT,
            "docker_save_outer_entry_limit",
        )
        body_end = offset + info.size
        next_header = body_end + (-info.size) % 512
        require(
            body_end <= next_header <= length,
            "docker_save_outer_member_range",
        )
        offset = next_header
    raise ScanError("docker_save_outer_missing_end_markers")


def gzip_header(reader):
    """RFC1952 header bytes, including every optional field; no filename use."""
    header = bytearray(read_exact(reader, 10))
    require(len(header) <= GZIP_HEADER_LIMIT, "gzip_header_limit")

    def append(data):
        require(
            len(header) + len(data) <= GZIP_HEADER_LIMIT,
            "gzip_header_limit",
        )
        header.extend(data)

    require(header[:3] == b"\x1f\x8b\x08", "unsupported_gzip_method_or_magic")
    flags = header[3]
    require(not flags & 0xE0, "reserved_gzip_header_flags")
    if flags & 4:
        length = read_exact(reader, 2)
        append(length)
        extra = read_exact(reader, int.from_bytes(length, "little"))
        append(extra)
        cursor = 0
        while cursor < len(extra):
            require(len(extra) - cursor >= 4, "malformed_gzip_extra_subfield")
            size = int.from_bytes(extra[cursor + 2 : cursor + 4], "little")
            require(cursor + 4 + size <= len(extra), "malformed_gzip_extra_subfield")
            cursor += 4 + size
    for bit in (8, 16):
        if flags & bit:
            while True:
                value = read_exact(reader, 1)
                append(value)
                if value == b"\0":
                    break
    if flags & 2:
        checksum = read_exact(reader, 2)
        require(
            int.from_bytes(checksum, "little") == zlib.crc32(header) & 0xFFFF,
            "gzip_header_crc_mismatch",
        )
        append(checksum)
    return bytes(header)


def validate_literal_inventory(values):
    require(
        isinstance(values, (list, tuple)) and len(values) <= LITERAL_PATTERN_LIMIT,
        "literal_inventory_pattern_limit",
    )
    encoded = []
    total = 0
    for value in values:
        require(type(value) is str and value, "literal_inventory_schema")
        raw = value.encode("utf-8")
        require(
            len(raw) <= LITERAL_VALUE_BYTES_LIMIT,
            "literal_inventory_value_limit",
        )
        total += len(raw)
        require(total <= LITERAL_TOTAL_BYTES_LIMIT, "literal_inventory_total_limit")
        encoded.append(raw)
    return tuple(encoded)


def compile_literals(values, policy):
    encoded = validate_literal_inventory(values)
    patterns = []
    carry = max((len(raw) for raw in encoded), default=0) + 1
    for index, (value, raw) in enumerate(zip(values, encoded, strict=True)):
        pattern = re.escape(raw)
        if policy == POLICY and len(value) < 6:
            pattern = rb"(?<![A-Za-z0-9_])" + pattern + rb"(?![A-Za-z0-9_])"
        patterns.append((index, sha(raw), re.compile(pattern)))
    return carry, tuple(patterns)


class LiteralMatcher:
    """Finite raw literals with exact accepted short-name token boundaries."""

    def __init__(self, values, policy, *, compiled=None):
        self.carry, self.patterns = (
            compile_literals(values, policy) if compiled is None else compiled
        )
        self.buffer, self.base = b"", 0
        self.next_positions = [0] * len(self.patterns)

    def feed(self, data, *, final=False, finding_limit):
        require(
            type(data) is bytes
            and type(final) is bool
            and type(finding_limit) is int
            and finding_limit >= 0,
            "literal_feed_schema",
        )
        self.buffer += data
        boundary = len(self.buffer) if final else max(0, len(self.buffer) - self.carry)
        found = []
        for index, digest, pattern in self.patterns:
            for match in pattern.finditer(
                self.buffer, max(0, self.next_positions[index] - self.base)
            ):
                start = self.base + match.start()
                if start >= self.base + boundary:
                    break
                if len(found) >= finding_limit:
                    raise LiteralFindingLimit("literal_finding_limit")
                found.append(
                    {
                        "rule_id": "private_literal",
                        "literal_index": index,
                        "literal_sha256": digest,
                        "byte_start": start,
                        "byte_end": self.base + match.end(),
                    }
                )
                self.next_positions[index] = self.base + match.end()
            self.next_positions[index] = max(
                self.next_positions[index], self.base + boundary
            )
        trim = max(0, boundary - 1)
        self.buffer = self.buffer[trim:]
        self.base += trim
        return found


AHO_PINS = {
    "source": "1d722e5a68445fcc35ff1a3b6406f476a2b80e2918ce985ccc5ca9e11815955d",
    "wheel": "9ec1d3465f25a5063c7eaa85ecb106cbe256064669c754e0b13b2483cf613a98",
    "extension": "6c44b1b03f94319834b9294d9720053071ce3ecfab584f3c479407d77249680c",
}
AHO_MEMBER = "ahocorasick.cpython-312-x86_64-linux-gnu.so"
AHO_MODULE = "_npa_authorized_literal_matcher"


def bound_bytes(spec, *, secret=False):
    with bound_open(spec, secret=secret) as (_path, fd, _info):
        data = descriptor_bytes(fd)
        require(sha(data) == spec["sha256"], "parsed_input_binding_changed")
        return data


class AuthorizedAho:
    """Execute exact approved source and an immutable, verified native byte copy."""

    def __init__(self, binding):
        self.fd = self.native = self.module = None
        try:
            require(
                isinstance(binding, dict) and set(binding) == {"kind", *AHO_PINS},
                "literal_engine_schema",
            )
            require(binding["kind"] == "aho-corasick-v1", "literal_engine_kind")
            for role, expected in AHO_PINS.items():
                require(
                    binding[role].get("sha256") == expected,
                    "literal_engine_unreviewed_digest",
                )
            require(
                all(
                    name not in sys.modules
                    for name in ("ahocorasick", "aho_matcher", AHO_MODULE)
                ),
                "literal_engine_preloaded_module",
            )
            source = bound_bytes(binding["source"])
            native = bound_bytes(binding["extension"], secret=True)
            wheel = bound_bytes(binding["wheel"], secret=True)
            with zipfile.ZipFile(io.BytesIO(wheel)) as archive:
                require(
                    archive.namelist().count(AHO_MEMBER) == 1,
                    "literal_engine_wheel_member",
                )
                require(
                    sha(archive.read(AHO_MEMBER)) == sha(native),
                    "literal_engine_wheel_extension_binding",
                )
                names = archive.namelist()
                require(len(names) == len(set(names)), "literal_engine_wheel_duplicate")
                record_name = "pyahocorasick-2.3.1.dist-info/RECORD"
                inventory = list(
                    csv.reader(io.StringIO(archive.read(record_name).decode()))
                )
                require(
                    {row[0] for row in inventory}
                    == {name for name in names if not name.endswith("/")},
                    "literal_engine_wheel_record_inventory",
                )
                require(
                    len(inventory) == len({row[0] for row in inventory}),
                    "literal_engine_wheel_record_duplicate",
                )
                for name, checksum, size in inventory:
                    payload = archive.read(name)
                    if name == record_name:
                        require(
                            not checksum and not size,
                            "literal_engine_wheel_record_self",
                        )
                    else:
                        expected = (
                            "sha256="
                            + base64.urlsafe_b64encode(hashlib.sha256(payload).digest())
                            .rstrip(b"=")
                            .decode()
                        )
                        require(
                            checksum == expected and size == str(len(payload)),
                            "literal_engine_wheel_record_hash",
                        )
                license_files = [name for name in names if name.endswith("/LICENSE")]
                require(
                    len(license_files) == 1
                    and sha(archive.read(license_files[0]))
                    == "2a7f4000fcb22199112e682ae44400f8836b9ff2500c46a8bed7d0edc93b2185",
                    "literal_engine_license_binding",
                )
            self.fd = os.memfd_create(
                "verified-literal-extension", os.MFD_ALLOW_SEALING | os.MFD_CLOEXEC
            )
            view = memoryview(native)
            while view:
                written = os.write(self.fd, view)
                require(written > 0, "literal_engine_memfd_short_write")
                view = view[written:]
            # Linux UAPI: F_ADD_SEALS1033 / F_GET_SEALS1034; SEAL/SHRINK/GROW/WRITE bits1/2/4/8.
            # This CPython build omits the named fcntl seal constants.
            fcntl.fcntl(self.fd, 1033, 15)
            require(
                fcntl.fcntl(self.fd, 1034) == 15
                and descriptor_digest(self.fd) == AHO_PINS["extension"],
                "literal_engine_sealed_binding",
            )
            origin = f"/proc/self/fd/{self.fd}"
            loader = importlib.machinery.ExtensionFileLoader("ahocorasick", origin)
            spec = importlib.util.spec_from_file_location(
                "ahocorasick", origin, loader=loader
            )
            self.native = importlib.util.module_from_spec(spec)
            sys.modules["ahocorasick"] = self.native
            loader.exec_module(self.native)
            require(
                self.native.__file__ == origin and self.native.unicode == 1,
                "literal_engine_native_mode",
            )
            self.module = types.ModuleType(AHO_MODULE)
            sys.modules[AHO_MODULE] = self.module
            exec(
                compile(source, "<authorized-literal-matcher>", "exec"),
                self.module.__dict__,
            )  # noqa: S102 - exact reviewed SHA; execute the verified bytes without a path reread.
        except BaseException:
            self.close()
            raise

    def receipt(self):
        require(
            sys.modules.get("ahocorasick") is self.native
            and sys.modules.get(AHO_MODULE) is self.module,
            "literal_engine_loaded_module_changed",
        )
        require(
            fcntl.fcntl(self.fd, 1034) == 15
            and descriptor_digest(self.fd) == AHO_PINS["extension"],
            "literal_engine_sealed_binding",
        )
        return {
            "kind": "aho-corasick-v1",
            "pinned_sha256": AHO_PINS,
            "sealed_native_copy": True,
        }

    def close(self):
        for name, module in (("ahocorasick", self.native), (AHO_MODULE, self.module)):
            if module is not None and sys.modules.get(name) is module:
                del sys.modules[name]
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


class Detector:
    def __init__(self, authorization, stderr_path):
        global _SPAWNING
        self.process = self.stderr = None
        self.joined = False
        self.ordinal = self.bytes = self.findings = 0
        # Submitted records awaiting their result, oldest first. Responses are
        # buffered from the raw descriptor so that writing record bytes and
        # reading results can never block on each other.
        self.outstanding = collections.deque()
        self.responses = bytearray()
        self.stdout_fd = None
        self.stdin_fd = None
        self.stdout_closed = False
        try:
            parent_fd = directory_fd(stderr_path.parent)
            try:
                stderr_fd = os.open(
                    stderr_path.name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                    0o600,
                    dir_fd=parent_fd,
                )
            finally:
                os.close(parent_fd)
            self.stderr = os.fdopen(stderr_fd, "wb")
            with (
                bound_open(authorization["helper"], secret=True) as (
                    _,
                    helper_source_fd,
                    _,
                ),
                bound_open(authorization["config"], secret=False) as (
                    _,
                    config_source_fd,
                    _,
                ),
                sealed_execution_input(
                    helper_source_fd, authorization["helper"]["sha256"], executable=True
                ) as helper_fd,
                sealed_execution_input(
                    config_source_fd, authorization["config"]["sha256"]
                ) as config_fd,
            ):
                _SPAWNING = True
                try:
                    self.process = subprocess.Popen(
                        [f"/proc/self/fd/{helper_fd}", "--config-fd", str(config_fd)],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=self.stderr,
                        env={"PATH": os.defpath},
                        start_new_session=True,
                        pass_fds=(helper_fd, config_fd),
                    )
                finally:
                    _SPAWNING = False
            if _CANCEL_REQUESTED:
                raise ScanError("scan_cancelled")
            self.stdout_fd = self.process.stdout.fileno()
            self.stdin_fd = self.process.stdin.fileno()
            # Records go out through this descriptor rather than the buffered
            # writer, so a full pipe returns to _transfer instead of parking this
            # process inside a write while the helper waits to be read.
            os.set_blocking(self.stdin_fd, False)
            self._validate_ready(authorization)
        except BaseException:
            self.abort()
            raise

    def _validate_ready(self, authorization):
        self.ready = self._response()
        require(
            self.ready.get("type") == "ready"
            and self.ready.get("protocol") == "whole-file-gitleaks.v1",
            "helper_protocol",
        )
        require(
            sha(canonical(self.ready)) == authorization["helper"]["ready_sha256"],
            "helper_ready_policy_changed",
        )
        require(
            self.ready.get("version") == "8.28.0"
            and self.ready.get("config_sha256") == authorization["config"]["sha256"],
            "helper_config_or_version",
        )
        require(
            self.ready.get("max_target_megabytes") == 0
            and self.ready.get("ignore_inline_allow") is True
            and self.ready.get("redact") == 100,
            "helper_coverage_policy",
        )
        require(
            self.ready.get("removed_content_path_rules") == REMOVED_PATH_RULES,
            "helper_content_path_policy",
        )
        path_rules = self.ready.get("path_rules")
        require(
            isinstance(path_rules, list) and len(path_rules) == 5,
            "helper_path_rule_population",
        )
        require(
            {row["rule_id"] for row in path_rules}
            == {*REMOVED_PATH_RULES, "pkcs12-file"},
            "helper_path_rule_inventory",
        )
        key = next(row for row in path_rules if row["rule_id"] == "pkcs12-file")
        require(
            key.get("has_content_regex") is False
            and key.get("selector") == r"(?i)(?:^|\/)[^\/]+\.p(?:12|fx)$",
            "helper_pkcs12_selector",
        )

    def _absorb(self):
        """Read one batch of helper output into the response buffer.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """
        chunk = os.read(self.stdout_fd, CHUNK)
        if not chunk:
            self.stdout_closed = True
            return
        self.responses += chunk
        require(
            len(self.responses) - self.responses.rfind(b"\n") - 1
            <= HELPER_RESPONSE_LIMIT,
            "helper_response_limit",
        )

    def _transfer(self, data):
        """Write one buffer to the helper while continuing to read its output.

        With several records outstanding, both directions are live at once: the
        helper is writing results for earlier records while this process is still
        writing the bytes of a later one. Draining what is already readable and
        then making a blocking write is not enough, because output that appears
        during that write is never collected and both pipes fill: the helper
        blocks writing results, stops reading records, and this process blocks
        writing records. Waiting for either direction and servicing whichever is
        ready means one side always progresses, so the pipes cannot deadlock
        however few workers the helper actually runs.

        Args:
            data: Bytes to hand to the helper.

        Returns:
            None.

        Raises:
            ScanError: If the helper closed its input before the bytes landed.
        """
        view = memoryview(data)
        while view:
            readable, writable, _ = select.select(
                [] if self.stdout_closed else [self.stdout_fd], [self.stdin_fd], ()
            )
            if readable:
                self._absorb()
            if writable:
                try:
                    view = view[os.write(self.stdin_fd, view) :]
                except BlockingIOError:
                    continue
                except BrokenPipeError:
                    require(False, "helper_unexpected_eof")

    def _response(self):
        while True:
            newline = self.responses.find(b"\n")
            if newline >= 0:
                require(newline + 1 <= HELPER_RESPONSE_LIMIT, "helper_response_limit")
                line = bytes(self.responses[: newline + 1])
                del self.responses[: newline + 1]
                result = json_object(line)
                require(isinstance(result, dict), "helper_response_schema")
                return result
            require(not self.stdout_closed, "helper_unexpected_eof")
            chunk = os.read(self.stdout_fd, CHUNK)
            if not chunk:
                self.stdout_closed = True
                require(False, "helper_unexpected_eof")
            self.responses += chunk
            require(
                len(self.responses) - self.responses.rfind(b"\n") - 1
                <= HELPER_RESPONSE_LIMIT,
                "helper_response_limit",
            )

    def begin(self, length):
        require(type(length) is int and 0 <= length < 2**64, "protocol_length")
        self._transfer(struct.pack(">Q", length))

    def write(self, data):
        self._transfer(data)

    def submit(self, length, digest):
        """Hand one completely written record over without awaiting its result.

        Args:
            length: The record's exact byte count, already written.
            digest: The record's SHA-256, recomputed here for the receipt check.

        Returns:
            None.

        Raises:
            None.
        """
        self.outstanding.append((length, digest))

    def outstanding_records(self):
        """Report how many submitted records have not been collected yet.

        Args:
            None.

        Returns:
            The number of outstanding records.

        Raises:
            None.
        """
        return len(self.outstanding)

    def _checked_findings(self, findings):
        """Validate one record's findings against the accepted schema.

        Args:
            findings: The findings list the helper reported for one record.

        Returns:
            The same list, once every entry matches the schema.

        Raises:
            ScanError: If the list or any finding in it is not exactly the
                accepted shape.
        """
        require(isinstance(findings, list), "helper_findings_schema")
        require(len(findings) <= RECORD_FINDING_LIMIT, "record_finding_limit")
        require(
            self.findings + len(findings) <= SCAN_FINDING_LIMIT,
            "scan_finding_limit",
        )
        allowed = {"rule_id", "start_line", "end_line"}
        for finding in findings:
            require(
                isinstance(finding, dict) and set(finding) == allowed,
                "helper_finding_schema",
            )
            require(
                isinstance(finding["rule_id"], str)
                and re.fullmatch(r"[a-z0-9_-]+", finding["rule_id"]),
                "helper_rule_identifier",
            )
            require(
                type(finding["start_line"]) is int and type(finding["end_line"]) is int,
                "helper_line_schema",
            )
        return findings

    def collect(self):
        """Read and validate the oldest outstanding record's result.

        Args:
            None.

        Returns:
            The record's findings as reported by the helper.

        Raises:
            ScanError: If the helper's receipt does not match the submitted
                record's ordinal, byte count, digest, or finding schema.
        """
        require(bool(self.outstanding), "helper_collect_without_record")
        length, digest = self.outstanding.popleft()
        result = self._response()
        self.ordinal += 1
        require(
            result.get("type") == "result"
            and type(result.get("ordinal")) is int
            and result["ordinal"] == self.ordinal,
            "helper_record_order",
        )
        require(
            type(result.get("bytes")) is int
            and result["bytes"] == length
            and result.get("sha256") == digest,
            "helper_record_byte_receipt",
        )
        findings = self._checked_findings(result.get("findings"))
        self.bytes += length
        self.findings += len(findings)
        return findings

    def finish(self):
        require(not self.outstanding, "helper_unfinished_records")
        self.process.stdin.close()
        result = self._response()
        require(
            result
            == {
                "type": "summary",
                "files": self.ordinal,
                "bytes": self.bytes,
                "findings": self.findings,
            },
            "helper_summary_receipt",
        )
        require(not self.responses, "helper_extra_response")
        require(os.read(self.stdout_fd, 1) == b"", "helper_extra_response")
        code = self.process.wait()
        self.joined = True
        self.process.stdout.close()
        self.stderr.close()
        require(code == (1 if self.findings else 0), "helper_exit_status")
        return result

    def abort(self):
        if self.process is not None:
            if self.process.poll() is None:
                try:
                    self.process.terminate()
                except ProcessLookupError:
                    pass
            self.process.wait()
            self.joined = True
            for stream in (self.process.stdin, self.process.stdout):
                if stream is not None and not stream.closed:
                    try:
                        stream.close()
                    except BrokenPipeError:
                        pass  # The owned child is already reaped; discard its pending pipe buffer.
        self.joined = True
        if self.stderr is not None and not self.stderr.closed:
            self.stderr.close()


class PendingRecord:
    """One submitted record whose ledger lines are held until its result lands.

    Ledger bytes must stay identical to scanning one record at a time, so every
    line produced while a record is outstanding is buffered here and written
    only when that record is finalized, in stream order.
    """

    def __init__(self, ordinal, length, kind, context, raw_parts):
        """Open a buffer for one record.

        Args:
            ordinal: The record's one-based position in the scan.
            length: The record's exact byte count.
            kind: The ledger record kind for this content.
            context: Scope fields repeated on every line for this record.
            raw_parts: Accumulated record bytes when confidentiality
                composition needs the complete record, otherwise ``None``.

        Returns:
            None.

        Raises:
            None.
        """
        self.ordinal = ordinal
        self.length = length
        self.kind = kind
        self.context = context
        self.raw_parts = raw_parts
        self.literal_matches = []
        self.digest = None
        # Lines belonging to the record itself, and lines the scan produced
        # after it moved past the record. Keeping them apart is what lets the
        # record's own result line land between them, exactly where scanning one
        # record at a time would have written it.
        self.lines = []
        self.trailing = []
        self.sealed = False
        self.held_bytes = length if raw_parts is not None else 0

    def append(self, serialized):
        """Buffer one ledger line in this record's correct position.

        Args:
            serialized: The line, newline included.

        Returns:
            None.

        Raises:
            None.
        """
        target = self.trailing if self.sealed else self.lines
        target.append(serialized)

    def ordered_lines(self):
        """Return every buffered line in final ledger order.

        Args:
            None.

        Returns:
            The record's own lines followed by the lines that came after it.

        Raises:
            None.
        """
        return self.lines + self.trailing


def _apply_address_space_limit(requested):
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    effective = requested
    for inherited in (soft, hard):
        if inherited != resource.RLIM_INFINITY:
            effective = min(effective, inherited)
    resource.setrlimit(resource.RLIMIT_AS, (effective, effective))
    return effective


def _confidentiality_worker(
    connection,
    policy_config,
    literal_binding,
    memory_limit,
    finding_limit,
):
    try:
        _apply_address_space_limit(memory_limit)
        policy = C.compile_policy(
            policy_config.get("customer_pattern"),
            policy_config.get("infra_pattern"),
            literal_policy=literal_binding,
        )
        connection.send_bytes(
            canonical(
                {
                    "type": "ready",
                    "policy_sha256": policy.policy_sha256,
                }
            )
        )
        records = 0
        while True:
            command = json_object(connection.recv_bytes(HELPER_RESPONSE_LIMIT))
            require(isinstance(command, dict), "confidentiality_worker_protocol")
            if command == {"type": "finish"}:
                connection.send_bytes(
                    canonical({"type": "summary", "records": records})
                )
                return
            require(
                set(command) == {"type", "length"}
                and command["type"] == "record"
                and type(command["length"]) is int
                and 0 <= command["length"] <= CONFIDENTIALITY_RECORD_LIMIT,
                "confidentiality_worker_protocol",
            )
            length = command["length"]
            raw_buffer = bytearray()
            remaining = length
            while remaining:
                data = connection.recv_bytes(min(CHUNK, remaining))
                require(
                    bool(data) and len(data) <= remaining,
                    "confidentiality_worker_protocol",
                )
                raw_buffer.extend(data)
                remaining -= len(data)
            finish = json_object(connection.recv_bytes(HELPER_RESPONSE_LIMIT))
            require(
                isinstance(finish, dict)
                and set(finish) == {"type", "sha256", "literal_matches"}
                and finish["type"] == "record_end"
                and isinstance(finish["sha256"], str)
                and SHA.fullmatch(finish["sha256"]) is not None
                and isinstance(finish["literal_matches"], list)
                and len(finish["literal_matches"]) <= finding_limit,
                "confidentiality_worker_protocol",
            )
            matches = tuple(
                C.LiteralMatch(
                    item["literal_index"],
                    item["byte_start"],
                    item["byte_end"],
                )
                for item in finish["literal_matches"]
                if isinstance(item, dict)
                and set(item) >= {"literal_index", "byte_start", "byte_end"}
            )
            require(
                len(matches) == len(finish["literal_matches"]),
                "confidentiality_worker_protocol",
            )
            raw = bytes(raw_buffer)
            del raw_buffer
            literal_scan = (
                C.LiteralScan(
                    literal_binding,
                    finish["sha256"],
                    length,
                    matches,
                    True,
                )
                if literal_binding is not None
                else None
            )
            receipt = policy.scan_record(
                raw,
                literal_scan=literal_scan,
                finding_limit=finding_limit,
            )
            response = canonical(
                {
                    "type": "result",
                    "policy_sha256": receipt.policy_sha256,
                    "record_sha256": receipt.record_sha256,
                    "byte_count": receipt.byte_count,
                    "line_count": receipt.line_count,
                    "findings": [asdict(item) for item in receipt.findings],
                }
            )
            require(
                len(response) <= HELPER_RESPONSE_LIMIT,
                "confidentiality_response_limit",
            )
            connection.send_bytes(response)
            records += 1
    except BaseException as exc:
        code = (
            str(exc)
            if isinstance(exc, (ScanError, C.ConfidentialityError))
            else "confidentiality_worker_failure"
        )
        try:
            connection.send_bytes(canonical({"type": "error", "error": code}))
        except (BrokenPipeError, EOFError, OSError):
            pass
        return
    finally:
        connection.close()


class ConfidentialityDetector:
    def __init__(
        self,
        policy_config,
        literal_binding,
        policy_sha256,
        *,
        process_context=None,
        memory_limit=CONFIDENTIALITY_MEMORY_LIMIT,
        finding_limit=RECORD_FINDING_LIMIT,
        autostart=True,
    ):
        self.connection = self.process = self.child_connection = None
        self.policy_sha256 = policy_sha256
        self.joined = False
        context = process_context or multiprocessing.get_context("spawn")
        parent, child = context.Pipe(duplex=True)
        self.connection = parent
        self.child_connection = child
        self.process = context.Process(
            target=_confidentiality_worker,
            args=(
                child,
                policy_config,
                literal_binding,
                memory_limit,
                finding_limit,
            ),
            name="image-byte-confidentiality",
        )
        self.records = 0
        self.finding_limit = finding_limit
        if autostart:
            self.start()

    def start(self):
        require(
            self.process is not None and self.process.pid is None,
            "confidentiality_worker_already_started",
        )
        try:
            self.process.start()
            self.child_connection.close()
            self.child_connection = None
            ready = self._response()
            require(
                ready
                == {
                    "type": "ready",
                    "policy_sha256": self.policy_sha256,
                },
                "confidentiality_worker_ready",
            )
        except BaseException:
            if self.child_connection is not None:
                self.child_connection.close()
                self.child_connection = None
            self.abort()
            raise

    def _response(self):
        try:
            raw = self.connection.recv_bytes(HELPER_RESPONSE_LIMIT)
        except (EOFError, OSError) as exc:
            raise ScanError("confidentiality_worker_failed") from exc
        result = json_object(raw)
        require(isinstance(result, dict), "confidentiality_worker_protocol")
        if result.get("type") == "error":
            code = result.get("error")
            require(
                isinstance(code, str) and re.fullmatch(r"[a-z0-9_-]+", code),
                "confidentiality_worker_protocol",
            )
            raise ScanError(code)
        return result

    def _send(self, payload):
        try:
            self.connection.send_bytes(payload)
        except (BrokenPipeError, EOFError, OSError) as exc:
            raise ScanError("confidentiality_worker_failed") from exc

    def begin(self, length):
        self._send(canonical({"type": "record", "length": length}))

    def write(self, data):
        self._send(data)

    def end(self, length, digest, literal_matches):
        self._send(
            canonical(
                {
                    "type": "record_end",
                    "sha256": digest,
                    "literal_matches": literal_matches,
                }
            )
        )
        result = self._response()
        require(
            set(result)
            == {
                "type",
                "policy_sha256",
                "record_sha256",
                "byte_count",
                "line_count",
                "findings",
            }
            and result["type"] == "result"
            and result["record_sha256"] == digest
            and result["byte_count"] == length
            and type(result["line_count"]) is int
            and result["line_count"] >= 0
            and isinstance(result["findings"], list)
            and len(result["findings"]) <= self.finding_limit,
            "confidentiality_worker_result",
        )
        findings = []
        for item in result["findings"]:
            require(
                isinstance(item, dict)
                and set(item)
                == {
                    "rule_id",
                    "start_byte",
                    "end_byte",
                    "start_line",
                    "end_line",
                    "views",
                }
                and isinstance(item["rule_id"], str)
                and type(item["start_byte"]) is int
                and type(item["end_byte"]) is int
                and type(item["start_line"]) is int
                and type(item["end_line"]) is int
                and isinstance(item["views"], list)
                and all(isinstance(view, str) for view in item["views"]),
                "confidentiality_worker_result",
            )
            findings.append(item)
        self.records += 1
        return result, findings

    def finish(self):
        self._send(canonical({"type": "finish"}))
        result = self._response()
        require(
            result == {"type": "summary", "records": self.records},
            "confidentiality_worker_summary",
        )
        self.connection.close()
        self.process.join()
        self.joined = True
        require(self.process.exitcode == 0, "confidentiality_worker_exit")
        if hasattr(self.process, "close"):
            self.process.close()

    def abort(self):
        if self.child_connection is not None:
            self.child_connection.close()
            self.child_connection = None
        if self.connection is not None:
            self.connection.close()
        if self.process is not None and self.process.pid is not None:
            if self.process.is_alive():
                self.process.terminate()
            self.process.join()
            self.joined = True
            if hasattr(self.process, "close"):
                self.process.close()
        else:
            self.joined = True


class Ledger:
    def __init__(
        self,
        directory,
        detector,
        literals,
        policy,
        literal_engine=None,
        *,
        policy_config=None,
        literal_binding=None,
        record_observer=None,
        defer_confidentiality=False,
    ):
        parent_fd = directory_fd(directory)
        try:
            stream_fd = os.open(
                "records.jsonl",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                0o600,
                dir_fd=parent_fd,
            )
        finally:
            os.close(parent_fd)
        self.stream = os.fdopen(stream_fd, "w", encoding="utf-8")
        self.record_observer = record_observer
        self.detector, self.literals, self.policy = detector, literals, policy
        validate_literal_inventory(literals)
        if literal_engine is None:
            self.compiled_literals = compile_literals(literals, policy)
            self.new_matcher = lambda: LiteralMatcher(
                (), self.policy, compiled=self.compiled_literals
            )
            self.literal_limit_error = LiteralFindingLimit
        else:
            self.compiled_literals = literal_engine.module.compile_literals(
                literals, policy
            )
            self.new_matcher = lambda: literal_engine.module.LiteralMatcher(
                self.compiled_literals
            )
            self.literal_limit_error = literal_engine.module.LiteralFindingLimit
        self.literal_policy_receipt = {
            "kind": policy,
            "inventory_sha256": literal_binding["sha256"] if literal_binding else None,
            "pattern_count": len(literals),
            "matcher_sha256": AHO_PINS["source"]
            if literal_engine
            else source_bindings()["npa/scripts/image_byte_scan/core.py"]["sha256"],
        }
        typed_binding = (
            C.LiteralPolicyBinding(
                sha(canonical(self.literal_policy_receipt)),
                self.literal_policy_receipt["matcher_sha256"],
                len(literals),
            )
            if literal_binding is not None
            else None
        )
        self.typed_literal_binding = typed_binding
        self.confidentiality = (
            C.compile_policy(
                policy_config.get("customer_pattern"),
                policy_config.get("infra_pattern"),
                literal_policy=typed_binding,
            )
            if policy_config is not None
            else None
        )
        self.confidentiality_detector = None
        self.confidentiality_configuration = (
            (policy_config, typed_binding, self.confidentiality.policy_sha256)
            if self.confidentiality is not None
            else None
        )
        if self.confidentiality_configuration is not None and not defer_confidentiality:
            self.start_confidentiality()
        self.zero_run = None
        self.records = self.findings = self.scan_bytes = self.zero_bytes = 0
        self.regular_files = self.regular_bytes = 0
        self.pending = collections.deque()
        self.open_record = None
        self.held_bytes = 0

    def start_confidentiality(self):
        if self.confidentiality_configuration is None:
            return
        require(
            self.confidentiality_detector is None,
            "confidentiality_worker_already_started",
        )
        detector = ConfidentialityDetector(
            *self.confidentiality_configuration,
            autostart=False,
        )
        self.confidentiality_detector = detector
        detector.start()

    def write(self, record):
        serialized = json.dumps(record, sort_keys=True) + "\n"
        require(
            len(serialized.encode("utf-8")) <= LEDGER_LINE_LIMIT,
            "ledger_line_limit",
        )
        if self.open_record is not None:
            self.open_record.append(serialized)
            return
        self.emit(serialized)

    def emit(self, serialized):
        """Write one already-serialized ledger line through to the stream.

        Args:
            serialized: The line, newline included, in final ledger order.

        Returns:
            None.

        Raises:
            OSError: If the ledger stream cannot be written or flushed.
        """
        self.stream.write(serialized)
        self.stream.flush()
        if self.record_observer is not None:
            self.record_observer(serialized.encode("utf-8"))

    def issue(self, code, context):
        require(self.findings < SCAN_FINDING_LIMIT, "scan_finding_limit")
        self.findings += 1
        self.write({"type": "finding", "rule_id": code, **context})

    def literal_batch(self, matcher, data, pending, *, final=False):
        record_remaining = RECORD_FINDING_LIMIT - len(pending.literal_matches)
        scan_remaining = SCAN_FINDING_LIMIT - self.findings
        finding_limit = min(record_remaining, scan_remaining)
        try:
            return matcher.feed(
                data,
                final=final,
                finding_limit=finding_limit,
            )
        except self.literal_limit_error as error:
            code = (
                "record_finding_limit"
                if record_remaining <= scan_remaining
                else "scan_finding_limit"
            )
            raise ScanError(code) from error

    def send(self, reader, length, kind, context):
        self.flush_zeros()
        require(
            self.confidentiality is None or length <= CONFIDENTIALITY_RECORD_LIMIT,
            "confidentiality_record_limit",
        )
        self.records += 1
        self.make_room(length)
        pending = self.stream_record(reader, length, kind, context)
        pending.sealed = True
        self.pending.append(pending)
        self.held_bytes += pending.held_bytes
        return pending.digest

    def make_room(self, length):
        """Collect outstanding records until the next one fits within the bounds.

        Draining before the record is read keeps ``PIPELINE_BYTES`` a bound on
        the bytes held at once rather than a target the next whole record
        overshoots. A record larger than the whole bound is still held alone, so
        no record is ever skipped or truncated for its size.

        Args:
            length: Byte count of the record about to be read.

        Returns:
            None.

        Raises:
            ScanError: If a collected record fails its receipt checks.
        """
        held = length if self.confidentiality is not None else 0
        while self.pending and (
            len(self.pending) >= PIPELINE_RECORDS
            or self.held_bytes + held > PIPELINE_BYTES
        ):
            self.finalize_head()

    def stream_record(self, reader, length, kind, context):
        """Read one complete record, hand it to the helper, and hold its lines.

        Every byte still reaches the detector, the record digest, the literal
        matcher and, when configured, the confidentiality buffer. Only the wait
        for the helper's result is deferred.

        Args:
            reader: Source positioned at the record's first byte.
            length: The record's exact byte count.
            kind: The ledger record kind for this content.
            context: Scope fields repeated on every line for this record.

        Returns:
            The :class:`PendingRecord` awaiting its detector result.

        Raises:
            ScanError: If the source ends before the declared byte count.
        """
        digest = hashlib.sha256()
        matcher = self.new_matcher()
        raw_parts = [] if self.confidentiality is not None else None
        pending = PendingRecord(self.records, length, kind, context, raw_parts)
        self.open_record = pending
        self.detector.begin(length)
        remaining = length
        while remaining:
            data = reader.read(min(CHUNK, remaining))
            require(bool(data) and len(data) <= remaining, "truncated_record")
            self.detector.write(data)
            digest.update(data)
            if raw_parts is not None:
                raw_parts.append(data)
            for offset in range(0, len(data), LITERAL_MATCH_CHUNK):
                self.issue_literals(
                    pending,
                    self.literal_batch(
                        matcher,
                        data[offset : offset + LITERAL_MATCH_CHUNK],
                        pending,
                    ),
                )
            remaining -= len(data)
        self.issue_literals(
            pending,
            self.literal_batch(matcher, b"", pending, final=True),
        )
        pending.digest = digest.hexdigest()
        self.detector.submit(length, pending.digest)
        return pending

    def issue_literals(self, pending, findings):
        """Record literal matches found in one chunk of a record.

        Args:
            pending: The record the matches belong to.
            findings: Literal matches produced by the matcher.

        Returns:
            None.

        Raises:
            None.
        """
        for finding in findings:
            require(
                len(pending.literal_matches) < RECORD_FINDING_LIMIT,
                "record_finding_limit",
            )
            require(self.findings < SCAN_FINDING_LIMIT, "scan_finding_limit")
            pending.literal_matches.append(finding)
            self.findings += 1
            self.write(
                {
                    "type": "finding",
                    "record_ordinal": pending.ordinal,
                    **pending.context,
                    **finding,
                }
            )

    def literal_scan(self, pending):
        """Build the typed literal evidence the confidentiality policy verifies.

        Args:
            pending: The finalized record, holding its literal matches.

        Returns:
            The record's :class:`LiteralScan`, or ``None`` when no literal
            inventory is bound.

        Raises:
            None.
        """
        if self.typed_literal_binding is None:
            return None
        return C.LiteralScan(
            self.typed_literal_binding,
            pending.digest,
            pending.length,
            tuple(
                C.LiteralMatch(row["literal_index"], row["byte_start"], row["byte_end"])
                for row in pending.literal_matches
            ),
            True,
        )

    def compose_confidentiality(self, pending, found):
        """Add the confidentiality policy's findings to a record's results.

        Args:
            pending: The finalized record, holding its complete bytes.
            found: Helper findings for the record, extended in place.

        Returns:
            None.

        Raises:
            ScanError: If the composed receipt does not bind the same record.
        """
        self.confidentiality_detector.begin(pending.length)
        for data in pending.raw_parts:
            self.confidentiality_detector.write(data)
        receipt, confidentiality_findings = self.confidentiality_detector.end(
            pending.length,
            pending.digest,
            pending.literal_matches,
        )
        require(
            receipt["record_sha256"] == pending.digest
            and receipt["byte_count"] == pending.length
            and receipt["policy_sha256"] == self.confidentiality.policy_sha256,
            "confidentiality_record_binding",
        )
        # Literal findings already have durable standalone receipts. The typed
        # composition verifies them; append only the additive regex findings.
        for item in confidentiality_findings:
            if "external_literal" not in item["views"]:
                require(
                    len(pending.literal_matches) + len(found) < RECORD_FINDING_LIMIT,
                    "record_finding_limit",
                )
                require(
                    self.findings + len(found) < SCAN_FINDING_LIMIT,
                    "scan_finding_limit",
                )
                found.append(item)
        self.write(
            {
                "type": "confidentiality_record",
                "record_ordinal": pending.ordinal,
                "policy_sha256": receipt["policy_sha256"],
                "sha256": pending.digest,
                "bytes": pending.length,
                "line_count": receipt["line_count"],
                "composed_findings": len(confidentiality_findings),
            }
        )

    def finalize_head(self):
        """Collect the oldest outstanding record and release its ledger lines.

        Args:
            None.

        Returns:
            None.

        Raises:
            ScanError: If the helper's receipt does not match the record.
        """
        pending = self.pending.popleft()
        self.held_bytes -= pending.held_bytes
        previous, self.open_record = self.open_record, pending
        pending.sealed = False
        try:
            found = self.detector.collect()
            require(
                len(pending.literal_matches) + len(found) <= RECORD_FINDING_LIMIT,
                "record_finding_limit",
            )
            require(
                self.findings + len(found) <= SCAN_FINDING_LIMIT,
                "scan_finding_limit",
            )
            if self.confidentiality is not None:
                self.compose_confidentiality(pending, found)
            self.findings += len(found)
            self.scan_bytes += pending.length
            self.write(
                {
                    "type": "record",
                    "record_ordinal": pending.ordinal,
                    "kind": pending.kind,
                    "bytes": pending.length,
                    "sha256": pending.digest,
                    "findings": found,
                    **pending.context,
                }
            )
        finally:
            pending.sealed = True
            self.open_record = None if previous is pending else previous
        for line in pending.ordered_lines():
            self.emit(line)

    def flush_pending(self):
        """Finalize every outstanding record so the ledger is complete.

        Args:
            None.

        Returns:
            None.

        Raises:
            ScanError: If a collected record fails its receipt checks.
        """
        while self.pending:
            self.finalize_head()

    def data(self, data, kind, context):
        return self.send(io.BytesIO(data), len(data), kind, context)

    def finish(self):
        if (
            self.confidentiality_detector is not None
            and not self.confidentiality_detector.joined
        ):
            self.confidentiality_detector.finish()

    def abort(self):
        if (
            self.confidentiality_detector is not None
            and not self.confidentiality_detector.joined
        ):
            self.confidentiality_detector.abort()

    def flush_zeros(self):
        if self.zero_run is not None:
            run = self.zero_run
            require(
                run["bytes"] <= ZERO_RECORD_LIMIT,
                "zero_record_complete_scan_limit",
            )
            self.zero_run = None
            self.send(
                ZeroReader(run["bytes"]),
                run["bytes"],
                "verified_zero_content",
                run["context"],
            )

    def zeros(self, data, context):
        require(not any(data), "nonzero_range_cannot_be_zero_accounted")
        self.zero_bytes += len(data)
        key = (context.get("scope"), context.get("layer_ordinal"))
        offset = context["tar_offset"]
        if self.zero_run is not None and (
            self.zero_run["key"] != key or self.zero_run["next_offset"] != offset
        ):
            self.flush_zeros()
        if self.zero_run is None:
            self.zero_run = {"key": key, "context": context, "bytes": 0}
        self.zero_run["next_offset"] = offset + len(data)
        self.zero_run["bytes"] += len(data)
        require(
            self.zero_run["bytes"] <= ZERO_RECORD_LIMIT,
            "zero_record_complete_scan_limit",
        )
        self.write(
            {
                "type": "verified_zero_range",
                "bytes": len(data),
                "sha256": sha(data),
                **context,
            }
        )


def safe_name(name):
    require(
        isinstance(name, str)
        and "\x00" not in name
        and len(name.encode("utf-8")) <= TAR_PATH_LIMIT,
        "tar_path_encoding",
    )
    path = PurePosixPath(name)
    require(not path.is_absolute() and ".." not in path.parts, "tar_path_escape")
    return str(path)


def parse_pax(data, *, key_limit=TAR_PAX_KEY_LIMIT):
    result, cursor = {}, 0
    while cursor < len(data):
        space = data.find(b" ", cursor)
        require(space > cursor and data[cursor:space].isdigit(), "pax_record_length")
        length = int(data[cursor:space])
        require(
            length > space - cursor + 2 and cursor + length <= len(data),
            "pax_record_boundary",
        )
        record = data[space + 1 : cursor + length]
        require(record.endswith(b"\n") and b"=" in record, "pax_record_format")
        key, value = record[:-1].split(b"=", 1)
        require(
            len(result) < key_limit and len(key) <= TAR_PATH_LIMIT,
            "tar_pax_key_limit",
        )
        key = key.decode("utf-8")
        require(
            key and key not in result and "\x00" not in key,
            "pax_duplicate_or_invalid_key",
        )
        require(
            key in PAX_TEXT_KEYS or key.startswith(PAX_BINARY_PREFIXES),
            "unsupported_pax_semantics",
        )
        if key in PAX_TEXT_KEYS:
            value = value.decode("utf-8")
            require("\x00" not in value, "pax_duplicate_or_invalid_key")
        # Kernel xattrs carry arbitrary bytes; only their already-scanned record
        # and key participate in this parser's path-safety decisions.
        result[key] = value
        cursor += length
    return result


def walk_tar(reader, sink, scope, file_handler):
    """Read every physical tar byte, including extension records and EOF padding."""
    index, zero_headers = 0, 0
    layer_scope = scope.get("scope") in {"layer", "docker_save_layer"}
    seen = set()
    pending, pending_keys = {}, set()
    extension_chain_bytes = 0
    long_name = long_link = None
    while True:
        offset = reader.tell()
        raw = read_exact(reader, 512, eof=True)
        if layer_scope:
            require(
                reader.tell() <= DOCKER_SAVE_DECODED_LAYER_LIMIT,
                "docker_save_decoded_layer_limit",
            )
        if not raw:
            require(zero_headers >= 2, "tar_missing_end_markers")
            require(
                not pending_keys and long_name is None and long_link is None,
                "orphan_tar_extension",
            )
            break
        context = {**scope, "entry_ordinal": index, "tar_offset": offset}
        if not any(raw):
            sink.zeros(raw, context)
            zero_headers += 1
            continue
        if zero_headers:
            sink.issue("nonzero_tar_trailer", context)
            sink.data(raw, "unexplained_tar_trailer", context)
            while data := reader.read(CHUNK):
                if layer_scope:
                    require(
                        reader.tell() <= DOCKER_SAVE_DECODED_LAYER_LIMIT,
                        "docker_save_decoded_layer_limit",
                    )
                context = {
                    **scope,
                    "entry_ordinal": index,
                    "tar_offset": reader.tell() - len(data),
                }
                if any(data):
                    sink.data(data, "unexplained_tar_trailer", context)
                else:
                    sink.zeros(data, context)
            require(zero_headers >= 2, "tar_incomplete_end_markers")
            break
        require(index < TAR_ENTRY_LIMIT, "tar_entry_limit")
        sink.data(raw, "raw_tar_header", context)
        info = tarfile.TarInfo.frombuf(raw, encoding="utf-8", errors="strict")
        require(info.size >= 0, "negative_tar_size")
        padding = (-info.size) % 512
        if layer_scope:
            require(
                reader.tell() + info.size + padding <= DOCKER_SAVE_DECODED_LAYER_LIMIT,
                "docker_save_decoded_layer_limit",
            )
        require(
            info.type
            in {
                tarfile.REGTYPE,
                tarfile.AREGTYPE,
                tarfile.DIRTYPE,
                tarfile.SYMTYPE,
                tarfile.LNKTYPE,
                tarfile.CHRTYPE,
                tarfile.BLKTYPE,
                tarfile.FIFOTYPE,
                tarfile.XHDTYPE,
                tarfile.XGLTYPE,
                tarfile.GNUTYPE_LONGNAME,
                tarfile.GNUTYPE_LONGLINK,
            },
            "unsupported_tar_entry_type",
        )
        if info.type in {
            tarfile.XHDTYPE,
            tarfile.XGLTYPE,
            tarfile.GNUTYPE_LONGNAME,
            tarfile.GNUTYPE_LONGLINK,
        }:
            require(
                info.size <= TAR_EXTENSION_LIMIT,
                "tar_extension_body_too_large",
            )
            require(
                extension_chain_bytes + info.size <= TAR_EXTENSION_CHAIN_LIMIT,
                "tar_extension_chain_too_large",
            )
            extension_chain_bytes += info.size
            extension_context = {**context, "tar_offset": reader.tell()}
            data = read_exact(reader, info.size)
            sink.data(data, "raw_tar_extension", extension_context)
            if info.type in {tarfile.XHDTYPE, tarfile.XGLTYPE}:
                parsed = parse_pax(
                    data,
                    key_limit=(
                        TAR_PAX_KEY_LIMIT
                        if info.type == tarfile.XGLTYPE
                        else TAR_PAX_KEY_LIMIT - len(pending_keys)
                    ),
                )
                if info.type == tarfile.XGLTYPE:
                    require(
                        not pending_keys
                        and long_name is None
                        and long_link is None
                        and not set(parsed) & {"path", "linkpath"},
                        "global_pax_name_override",
                    )
                else:
                    parsed_keys = set(parsed)
                    require(
                        not parsed_keys & pending_keys,
                        "ambiguous_pax_override",
                    )
                    pending_keys.update(parsed_keys)
                    require(
                        len(pending_keys) <= TAR_PAX_KEY_LIMIT,
                        "tar_pax_key_limit",
                    )
                    for key in ("path", "linkpath"):
                        if key in parsed:
                            pending[key] = parsed[key]
            else:
                require(
                    data.endswith(b"\x00") and b"\x00" not in data[:-1],
                    "gnu_long_name_encoding",
                )
                value = data[:-1].decode("utf-8")
                if info.type == tarfile.GNUTYPE_LONGNAME:
                    require(long_name is None, "ambiguous_gnu_name")
                    long_name = value
                else:
                    require(long_link is None, "ambiguous_gnu_link")
                    long_link = value
        else:
            require(
                not (long_name is not None and "path" in pending)
                and not (long_link is not None and "linkpath" in pending),
                "ambiguous_name_extensions",
            )
            name = safe_name(
                pending.get("path", long_name if long_name is not None else info.name)
            )
            link = pending.get(
                "linkpath", long_link if long_link is not None else info.linkname
            )
            require(
                "\x00" not in link and len(link.encode("utf-8")) <= TAR_PATH_LIMIT,
                "tar_link_encoding",
            )
            require(name not in seen, "duplicate_tar_path")
            seen.add(name)
            sink.data(name.encode("utf-8"), "logical_tar_path", context)
            if link:
                sink.data(link.encode("utf-8"), "logical_tar_link", context)
            if PKCS12.search(name):
                sink.issue("pkcs12-file", context)
            pending, pending_keys = {}, set()
            extension_chain_bytes = 0
            long_name = long_link = None
            if info.isreg():
                require(
                    not layer_scope or info.size <= TAR_REGULAR_FILE_LIMIT,
                    "tar_regular_file_limit",
                )
                file_handler(
                    reader, info.size, name, {**context, "tar_offset": reader.tell()}
                )
            else:
                require(info.size == 0, "nonregular_tar_body")
        if padding:
            context = {**context, "tar_offset": reader.tell()}
            data = read_exact(reader, padding)
            if any(data):
                sink.issue("nonzero_tar_padding", context)
                sink.data(data, "nonzero_tar_padding", context)
            else:
                sink.zeros(data, context)
        index += 1
    sink.flush_zeros()
    return {
        "headers": index,
        "decoded_bytes": reader.tell(),
        "zero_end_blocks": zero_headers,
    }


def validate_docker_save_layer_references(names, diff_ids):
    """Bound logical layer reuse before any physical layer bytes are read."""
    require(
        isinstance(names, list)
        and isinstance(diff_ids, list)
        and len(names) == len(diff_ids)
        and 0 < len(names) <= DOCKER_SAVE_LAYER_LIMIT,
        "docker_save_layer_reference_limit",
    )
    counts = {}
    bindings = {}
    for name, diff_id in zip(names, diff_ids, strict=True):
        require(
            isinstance(name, str)
            and isinstance(diff_id, str)
            and DIGEST.fullmatch(diff_id) is not None,
            "docker_save_layer_reference",
        )
        normalized = safe_name(name)
        counts[normalized] = counts.get(normalized, 0) + 1
        require(
            counts[normalized] <= DOCKER_SAVE_LAYER_MEMBER_REPEAT_LIMIT,
            "docker_save_layer_member_repeat_limit",
        )
        previous = bindings.setdefault(normalized, diff_id)
        require(previous == diff_id, "docker_save_layer_member_conflict")


def graph(fd, length, verification, expected_id):
    """Rebind metadata to the accepted exact archive before scanning its layers."""
    if verification.get("schema_version") == "npa.ncore.oci-verification.v1":
        from . import ncore_verification as N

        result = N.inspect(fd, length, expected_id)
        N.bind(result, verification, expected_id)
        return result["layers"]
    preflight_docker_save_outer_tar(fd, length)
    os.lseek(fd, 0, os.SEEK_SET)
    with (
        os.fdopen(os.dup(fd), "rb") as file,
        tarfile.open(fileobj=file, mode="r:") as archive,
    ):
        members = {}
        for item in archive.getmembers():
            name = safe_name(item.name)
            require(name not in members, "duplicate_outer_path")
            members[name] = item
        require(
            len(members) <= DOCKER_SAVE_OUTER_ENTRY_LIMIT,
            "docker_save_outer_entry_limit",
        )

        def payload(name):
            info = members[name]
            require(
                info.isfile() and info.size <= DOCKER_SAVE_METADATA_LIMIT,
                "graph_metadata_not_regular_or_too_large",
            )
            stream = archive.extractfile(info)
            require(stream is not None, "graph_metadata_not_regular_or_too_large")
            data = stream.read(DOCKER_SAVE_METADATA_LIMIT + 1)
            require(
                len(data) == info.size <= DOCKER_SAVE_METADATA_LIMIT,
                "graph_metadata_not_regular_or_too_large",
            )
            return data

        saved = json_object(payload("manifest.json"))
        require(
            isinstance(saved, list) and len(saved) == 1 and isinstance(saved[0], dict),
            "graph_image_population",
        )
        config_path = safe_name(saved[0]["Config"])
        data = payload(config_path)
        config_digest = "sha256:" + sha(data)
        require(
            config_digest == verification["image_config_digest"], "graph_config_binding"
        )
        config = json_object(data)
        diff_ids = config["rootfs"]["diff_ids"]
        require(
            config["rootfs"]["type"] == "layers"
            and diff_ids == verification["verified_layer_diff_ids"],
            "graph_layer_binding",
        )
        names = saved[0]["Layers"]
        validate_docker_save_layer_references(names, diff_ids)
        require(len(names) == verification["layer_count"], "graph_layer_population")
        manifest_digest = verification.get("image_manifest_digest")
        descriptors = None
        if manifest_digest is not None:
            index = json_object(payload("index.json"))
            require(
                index.get("schemaVersion") == 2 and len(index["manifests"]) == 1,
                "graph_oci_population",
            )
            descriptor = index["manifests"][0]
            require(descriptor["digest"] == manifest_digest, "graph_manifest_selection")
            data = payload("blobs/sha256/" + manifest_digest[7:])
            require(
                len(data) == descriptor["size"]
                and "sha256:" + sha(data) == manifest_digest,
                "graph_manifest_digest",
            )
            manifest = json_object(data)
            require(
                manifest.get("schemaVersion") == 2
                and manifest["mediaType"] == descriptor["mediaType"],
                "graph_manifest_schema",
            )
            require(
                manifest["config"]["digest"] == config_digest
                and manifest["config"]["size"] == members[config_path].size,
                "graph_config_descriptor",
            )
            descriptors = manifest["layers"]
            require(len(descriptors) == len(names), "graph_descriptor_population")
        else:
            require(
                "index.json" not in members and "oci-layout" not in members,
                "unbound_oci_metadata",
            )
        require(
            expected_id in {config_digest, manifest_digest}
            and verification["expected_image_id"] == expected_id,
            "graph_expected_identity",
        )
        result = []
        member_descriptors = {}
        for ordinal, name in enumerate(names):
            name = safe_name(name)
            item = members[name]
            require(
                item.isfile() and item.offset_data + item.size <= length,
                "graph_layer_range",
            )
            descriptor = descriptors[ordinal] if descriptors is not None else None
            if descriptor is not None:
                binding = (
                    descriptor["mediaType"],
                    descriptor["digest"],
                    descriptor["size"],
                )
                previous = member_descriptors.setdefault(name, binding)
                require(previous == binding, "docker_save_layer_member_conflict")
                require(
                    name == "blobs/sha256/" + descriptor["digest"][7:]
                    and item.size == descriptor["size"],
                    "graph_layer_descriptor",
                )
                require(
                    descriptor["mediaType"]
                    in {
                        "application/vnd.oci.image.layer.v1.tar",
                        "application/vnd.oci.image.layer.v1.tar+gzip",
                        "application/vnd.docker.image.rootfs.diff.tar",
                        "application/vnd.docker.image.rootfs.diff.tar.gzip",
                    },
                    "unsupported_layer_codec",
                )
            result.append(
                {
                    "ordinal": ordinal,
                    "name": name,
                    "offset": item.offset_data,
                    "size": item.size,
                    "diff_id": diff_ids[ordinal],
                    "descriptor": descriptor,
                }
            )
        return result


def stat_fingerprint(info):
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
        info.st_mode,
        info.st_uid,
        info.st_gid,
    )


def fingerprint(path):
    return stat_fingerprint(path.stat())


GO_SOURCE_NAMES = (
    "main.go",
    "main_test.go",
    "go.mod",
    "go.sum",
    "build.py",
    "LICENSE-GO",
    "LICENSE-GITLEAKS",
    "README.md",
)
GO_TEST_SOURCE = "npa/tests/docker/test_image_byte_go_build.py"


def current_go_sources():
    checkout = _ROOTS.get()[1]
    files = {
        name: checkout / "npa/scripts/image_byte_scan/go_helper" / name
        for name in GO_SOURCE_NAMES
    }
    files[GO_TEST_SOURCE] = checkout / GO_TEST_SOURCE
    result = {}
    for name, path in files.items():
        with open_source_fd(path) as (fd, _info):
            result[name] = descriptor_digest(fd)
    return result


def verified_tools(spec):
    receipt = bound_json(spec)
    require(
        receipt.get("schema_version") == "npa.image-byte-scan-tools.v1",
        "tools_receipt_schema",
    )
    require(
        receipt.get("source") == current_go_sources(), "tools_receipt_source_changed"
    )
    with open_source_fd(_ROOTS.get()[1] / ".gitleaks.toml") as (fd, _info):
        require(
            receipt["config"]["sha256"] == descriptor_digest(fd),
            "tools_receipt_checkout_config_changed",
        )
    for role in ("helper", "config", "ready"):
        bound_file(receipt[role], secret=role != "config")
    ready = bound_json(receipt["ready"])
    helper = {**receipt["helper"], "ready_sha256": sha(canonical(ready))}
    return helper, receipt["config"]


def input_snapshots(authorization):
    items = [
        ("tools_receipt", authorization["tools_receipt"], True),
        ("verification_report", authorization["verification_report"], True),
        ("helper", authorization["helper"], True),
        ("config", authorization["config"], False),
    ]
    tools_receipt = bound_json(authorization["tools_receipt"])
    items.append(("helper_ready", tools_receipt["ready"], True))
    literal = authorization.get("literal_inventory")
    if literal is not None:
        items.append(("literal_inventory", literal, True))

    engine = authorization.get("literal_engine")
    require(
        "literal_engine" not in authorization or isinstance(engine, dict),
        "literal_engine_schema",
    )
    if engine is not None:
        require(
            isinstance(engine, dict) and set(engine) == {"kind", *AHO_PINS},
            "literal_engine_schema",
        )
        items.extend(
            ("literal_engine_" + role, engine[role], role != "source")
            for role in AHO_PINS
        )
    configured_sources = authorization.get("sources")
    require(configured_sources == source_bindings(), "scanner_source_binding_changed")
    items.extend(
        ("source:" + role, spec, False) for role, spec in configured_sources.items()
    )
    if authorization.get("confidentiality") is not None:
        items.append(("confidentiality", authorization["confidentiality"], True))
    result = []
    for role, spec, secret in items:
        with bound_open(spec, secret=secret) as (path, _fd, info):
            result.append((role, spec, secret, path, stat_fingerprint(info)))
    return result


def recheck_snapshots(snapshots):
    for role, spec, secret, path, before in snapshots:
        with bound_open(spec, secret=secret) as (current_path, _fd, info):
            require(
                current_path == path and stat_fingerprint(info) == before,
                "input_changed_during_scan",
            )


def verification_archive_digest(verification):
    """Keep product verifier identities distinct; neither is a scanner bypass."""
    require(verification.get("valid") is True, "verification_did_not_pass")
    schema = verification.get("schema_version")
    require(
        schema
        in (
            "npa.curobo.image-verification.v1",
            "npa.docker-save.image-verification.v1",
            "npa.ncore.oci-verification.v1",
        ),
        "verification_schema",
    )
    return verification[
        "archive_sha256"
        if schema == "npa.ncore.oci-verification.v1"
        else "docker_save_sha256"
    ]


def _scan(authorization, directory, detector_type=Detector, *, record_observer=None):
    require(
        isinstance(authorization, dict)
        and authorization.get("schema_version")
        == "npa.image-byte-scan-authorization.v1",
        "authorization_schema",
    )
    required = {
        "schema_version",
        "accepted_verification",
        "archive",
        "verification_report",
        "expected_image_id",
        "helper",
        "config",
        "sources",
        "tools_receipt",
    }
    require(
        required
        <= set(authorization)
        <= required | {"literal_inventory", "literal_engine", "confidentiality"},
        "authorization_fields",
    )
    require(
        authorization.get("accepted_verification") is True, "verification_not_accepted"
    )
    helper, config = verified_tools(authorization["tools_receipt"])
    require(
        helper == authorization["helper"] and config == authorization["config"],
        "tools_receipt_authorization_changed",
    )
    snapshots = input_snapshots(authorization)
    verification = bound_json(authorization["verification_report"])
    require(
        authorization["archive"]["sha256"] == verification_archive_digest(verification),
        "archive_verification_binding",
    )
    values, policy, literal_binding = (
        [],
        "exact-substring-v1",
        authorization.get("literal_inventory"),
    )
    if literal_binding is not None:
        inventory = bound_json(
            literal_binding,
            byte_limit=LITERAL_INVENTORY_JSON_LIMIT,
            limit_code="literal_inventory_json_limit",
        )
        values = inventory.get("literals")
        require(
            isinstance(values, list)
            and all(isinstance(value, str) and value for value in values),
            "literal_inventory_schema",
        )
        policy = literal_binding["matching_policy"]
        require(policy in {"exact-substring-v1", POLICY}, "literal_matching_policy")

    require(
        literal_binding is not None or authorization.get("confidentiality") is not None,
        "confidentiality_policy_required",
    )
    policy_config = (
        bound_json(authorization["confidentiality"])
        if authorization.get("confidentiality") is not None
        else None
    )
    require(
        policy_config is not None or bool(values),
        "nonempty_confidentiality_policy_required",
    )
    if policy_config is not None:
        require(
            isinstance(policy_config, dict)
            and set(policy_config) <= {"customer_pattern", "infra_pattern"},
            "confidentiality_schema",
        )
        C.compile_policy(
            policy_config.get("customer_pattern"), policy_config.get("infra_pattern")
        )
    archive_path, fd, initial = open_private_fd(authorization["archive"]["path"])
    try:
        require(
            descriptor_digest(fd) == authorization["archive"]["sha256"],
            "input_binding_changed",
        )
        require(
            stat_fingerprint(os.fstat(fd)) == stat_fingerprint(initial),
            "input_changed_during_read",
        )
    except BaseException:
        os.close(fd)
        raise
    detector = sink = literal_engine = None
    report = {
        "schema_version": "npa.image-byte-scan.v1",
        "valid": False,
        "complete": False,
        "authorization_sha256": sha(canonical(authorization)),
        "archive_sha256": authorization["archive"]["sha256"],
        "image_config_digest": verification["image_config_digest"],
        "image_manifest_digest": verification.get("image_manifest_digest"),
        "expected_image_id": authorization["expected_image_id"],
        "private_literals_configured": literal_binding is not None,
        "private_literal_count": len(values),
        "literal_matching_policy": policy,
        "layers": [],
    }
    try:
        if verification["schema_version"] == "npa.ncore.oci-verification.v1":
            from . import ncore_verification as N

            result = N.inspect(fd, initial.st_size, authorization["expected_image_id"])
            N.bind(result, verification, authorization["expected_image_id"])
            layers = result["layers"]
            report["oci_graph"] = result["receipt"]
        else:
            layers = graph(
                fd, initial.st_size, verification, authorization["expected_image_id"]
            )
        if authorization.get("literal_engine") is not None:
            literal_engine = AuthorizedAho(authorization["literal_engine"])
        detector = detector_type(authorization, directory / "helper-stderr.jsonl")
        sink = Ledger(
            directory,
            detector,
            values,
            policy,
            literal_engine,
            policy_config=policy_config,
            literal_binding=literal_binding,
            defer_confidentiality=True,
            **(
                {"record_observer": record_observer}
                if record_observer is not None
                else {}
            ),
        )
        sink.start_confidentiality()
        report["confidentiality_policy"] = (
            sink.confidentiality.receipt()
            if sink.confidentiality is not None
            else {"mode": "exact-literals-v1", "binding": sink.literal_policy_receipt}
        )
        layer_names = {row["name"] for row in layers}
        locations = {}

        def outer_file(reader, size, name, context):
            offset = reader.tell()
            if name in layer_names:
                if "oci_graph" in report:
                    value = sink.send(reader, size, "outer_regular_content", context)
                else:
                    digest = hashlib.sha256()
                    remaining = size
                    while remaining:
                        data = reader.read(min(CHUNK, remaining))
                        require(data, "truncated_outer_blob")
                        digest.update(data)
                        remaining -= len(data)
                    value = digest.hexdigest()
                locations[name] = {"offset": offset, "size": size, "sha256": value}
                sink.write(
                    {
                        "type": "encoded_layer_blob",
                        "bytes": size,
                        "sha256": value,
                        **context,
                    }
                )
            else:
                sink.send(reader, size, "outer_regular_content", context)

        outer = HashedReader(Slice(fd, 0, initial.st_size))
        report["outer"] = walk_tar(outer, sink, {"scope": "outer"}, outer_file)
        require(
            outer.position == initial.st_size
            and outer.digest.hexdigest() == authorization["archive"]["sha256"],
            "outer_complete_byte_accounting",
        )
        for row in layers:
            location = locations[row["name"]]
            require(
                location["offset"] == row["offset"] and location["size"] == row["size"],
                "manual_graph_range_disagreement",
            )
            if row["descriptor"]:
                require(
                    "sha256:" + location["sha256"] == row["descriptor"]["digest"],
                    "layer_compressed_digest",
                )
            elif row["name"].startswith("blobs/"):
                require(
                    row["name"] == "blobs/sha256/" + location["sha256"],
                    "classic_layer_blob_digest",
                )
            raw = Slice(fd, row["offset"], row["size"])
            header = read_exact(raw, min(10, row["size"]))
            gzip = header[:2] == b"\x1f\x8b"
            if row["descriptor"]:
                declared_gzip = row["descriptor"]["mediaType"].endswith(
                    ("+gzip", ".gzip")
                )
                require(gzip == declared_gzip, "declared_layer_codec_mismatch")
            scope = {"scope": "layer", "layer_ordinal": row["ordinal"]}
            if gzip:
                header = gzip_header(Slice(fd, row["offset"], row["size"]))
                sink.data(header, "raw_gzip_header", {**scope, "compressed_offset": 0})
            raw = Slice(fd, row["offset"], row["size"])
            decoded = HashedReader(GzipReader(raw) if gzip else raw)

            def layer_file(reader, size, name, context):
                sink.send(reader, size, "layer_regular_content", context)
                sink.regular_files += 1
                sink.regular_bytes += size

            details = walk_tar(decoded, sink, scope, layer_file)
            require(
                "sha256:" + decoded.digest.hexdigest() == row["diff_id"],
                "layer_uncompressed_diff_id",
            )
            require(raw.position == row["size"], "unaccounted_compressed_bytes")
            report["layers"].append(
                {
                    "ordinal": row["ordinal"],
                    "diff_id": row["diff_id"],
                    "compressed_sha256": location["sha256"],
                    "compressed_bytes": row["size"],
                    "codec": "gzip" if gzip else "raw",
                    **details,
                }
            )
        require(
            sink.regular_files == verification["regular_files_read"]
            and sink.regular_bytes == verification["content_bytes_read"],
            "verifier_regular_population_disagreement",
        )
        sink.flush_zeros()
        sink.flush_pending()
        report["helper_summary"] = detector.finish()
        sink.finish()
        recheck_snapshots(snapshots)
        require(
            authorization["sources"] == source_bindings(),
            "scanner_source_population_changed",
        )
        report["literal_engine"] = (
            literal_engine.receipt()
            if literal_engine is not None
            else {"kind": "regex-reference-v1"}
        )
        report["input_snapshot_receipts"] = [
            {"role": role, "sha256": spec["sha256"], "stat": list(before)}
            for role, spec, secret, path, before in snapshots
        ]
        report["complete"] = True
        report["valid"] = sink.findings == 0
    except INPUT_ERRORS as error:
        report["failure_code"] = (
            str(error)
            if isinstance(error, ScanError)
            else "uninterpretable_input_or_scanner_failure"
        )
    finally:
        if detector is not None and not detector.joined:
            detector.abort()
        report["helper_joined"] = detector is None or detector.joined
        if sink is not None:
            sink.abort()
            report["confidentiality_worker_joined"] = (
                sink.confidentiality_detector is None
                or sink.confidentiality_detector.joined
            )
            if sink.zero_run is not None:
                report.update(
                    valid=False,
                    complete=False,
                    failure_code=report.get("failure_code", "pending_zero_range_scan"),
                )
            report.update(
                records=sink.records,
                scanned_bytes=sink.scan_bytes,
                verified_zero_bytes=sink.zero_bytes,
                regular_files=sink.regular_files,
                regular_bytes=sink.regular_bytes,
                findings=sink.findings,
            )
            sink.stream.close()
        if literal_engine is not None:
            literal_engine.close()
        try:
            current = os.fstat(fd)
            path_current = archive_path.lstat()
            if (
                stat_fingerprint(initial) != stat_fingerprint(current)
                or not stat.S_ISREG(path_current.st_mode)
                or (current.st_dev, current.st_ino)
                != (path_current.st_dev, path_current.st_ino)
            ):
                report.update(
                    valid=False,
                    complete=False,
                    failure_code="archive_changed_during_scan",
                )
        except OSError:
            report.update(
                valid=False, complete=False, failure_code="archive_changed_during_scan"
            )
        finally:
            os.close(fd)
    return report


@contextmanager
def cancellation_scope():
    """Convert catchable CLI termination into receipt-producing stack unwinding."""
    global _CANCEL_REQUESTED
    previous = signal.getsignal(signal.SIGTERM)
    old_cancel, _CANCEL_REQUESTED = _CANCEL_REQUESTED, False

    def cancelled(signum, frame):
        global _CANCEL_REQUESTED
        if _CANCEL_REQUESTED:
            return
        _CANCEL_REQUESTED = True
        if not _SPAWNING:
            raise ScanError("scan_cancelled")

    signal.signal(signal.SIGTERM, cancelled)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)
        _CANCEL_REQUESTED = old_cancel


class SanitizedArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        raise ScanError("invalid_cli_arguments")


def scan(
    authorization, directory, *, analysis_root, trusted_root, detector_type=Detector
):
    with authorized_roots(analysis_root, trusted_root):
        return _scan(authorization, directory, detector_type=detector_type)


def directory_fd(path):
    """Open a private directory beneath analysis through nofollow descriptors."""
    roots = _ROOTS.get()
    require(roots is not None, "explicit_roots_required")
    path = Path(path).absolute()
    require(
        ".." not in path.parts
        and path.is_relative_to(roots[0])
        and not path.is_relative_to(roots[1]),
        "output_directory_scope",
    )
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC | os.O_DIRECTORY
    fd = os.open(path.anchor, flags)
    try:
        for part in path.parts[1:]:
            next_fd = os.open(part, flags, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        info = os.fstat(fd)
        require(
            info.st_uid == os.geteuid() and not info.st_mode & 0o077,
            "output_directory_permissions",
        )
        return fd
    except BaseException:
        os.close(fd)
        raise


def create_output(path):
    path = Path(path).absolute()
    parent = directory_fd(path.parent)
    try:
        os.mkdir(path.name, 0o700, dir_fd=parent)
        fd = os.open(
            path.name,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_DIRECTORY | os.O_CLOEXEC,
            dir_fd=parent,
        )
        return path, fd
    finally:
        os.close(parent)


def output_identity(directory, held_fd):
    """Require the current path to name the directory held by this invocation."""
    current = directory_fd(directory)
    try:
        expected, observed = os.fstat(held_fd), os.fstat(current)
        require(
            (expected.st_dev, expected.st_ino) == (observed.st_dev, observed.st_ino),
            "output_directory_replaced",
        )
    finally:
        os.close(current)


def verify_private_json(directory, held_fd, name, result, identity):
    """Read the complete published payload through the original directory."""
    require(re.fullmatch(r"[a-zA-Z0-9_.-]+", name) is not None, "output_name")
    output_identity(directory, held_fd)
    fd = os.open(
        name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=held_fd
    )
    try:
        before = os.fstat(fd)
        require(
            stat.S_ISREG(before.st_mode)
            and before.st_uid == os.geteuid()
            and stat.S_IMODE(before.st_mode) == 0o600
            and before.st_nlink == 1
            and stat_fingerprint(before) == identity,
            "output_file_changed",
        )
        payload = (json.dumps(result, sort_keys=True, indent=2) + "\n").encode("utf-8")
        actual = descriptor_bytes(fd)
        after = os.fstat(fd)
        named = os.stat(name, dir_fd=held_fd, follow_symlinks=False)
        require(
            actual == payload
            and stat_fingerprint(before)
            == stat_fingerprint(after)
            == stat_fingerprint(named)
            and after.st_nlink == named.st_nlink == 1,
            "output_bytes_changed",
        )
        output_identity(directory, held_fd)
    finally:
        os.close(fd)


def _private_json_payload(result):
    return (json.dumps(result, sort_keys=True, indent=2) + "\n").encode("utf-8")


def stage_private_json(held_fd, name, result):
    """Write and fsync a private pending receipt through a held directory FD."""
    require(re.fullmatch(r"[a-zA-Z0-9_.-]+", name) is not None, "output_name")
    temporary = name + ".pending"
    output = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
        0o600,
        dir_fd=held_fd,
    )
    try:
        payload = _private_json_payload(result)
        remaining = memoryview(payload)
        while remaining:
            written = os.write(output, remaining)
            require(written > 0, "output_short_write")
            remaining = remaining[written:]
        os.fsync(output)
        identity = stat_fingerprint(os.fstat(output))
    except BaseException:
        try:
            os.unlink(temporary, dir_fd=held_fd)
        except FileNotFoundError:
            pass
        raise
    finally:
        os.close(output)
    return identity


def publish_staged_private_json(held_fd, name, result, identity):
    """Link a verified pending receipt into place through the held directory FD."""
    require(re.fullmatch(r"[a-zA-Z0-9_.-]+", name) is not None, "output_name")
    temporary = name + ".pending"
    pending = os.open(
        temporary,
        os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
        dir_fd=held_fd,
    )
    try:
        before = os.fstat(pending)
        require(
            stat.S_ISREG(before.st_mode)
            and before.st_uid == os.geteuid()
            and stat.S_IMODE(before.st_mode) == 0o600
            and before.st_nlink == 1
            and stat_fingerprint(before) == identity
            and descriptor_bytes(pending) == _private_json_payload(result)
            and stat_fingerprint(os.fstat(pending)) == identity,
            "staged_output_changed",
        )
        os.link(
            temporary,
            name,
            src_dir_fd=held_fd,
            dst_dir_fd=held_fd,
            follow_symlinks=False,
        )
        os.unlink(temporary, dir_fd=held_fd)
        os.fsync(held_fd)
        published = os.stat(name, dir_fd=held_fd, follow_symlinks=False)
        final = os.fstat(pending)
        require(
            final.st_nlink == published.st_nlink == 1
            and (final.st_dev, final.st_ino) == (published.st_dev, published.st_ino)
            and stat_fingerprint(final) == stat_fingerprint(published),
            "published_output_changed",
        )
        return stat_fingerprint(final)
    except BaseException:
        try:
            os.unlink(name, dir_fd=held_fd)
        except FileNotFoundError:
            pass
        raise
    finally:
        os.close(pending)


def discard_private_json(held_fd, name):
    """Remove pending or published output from the originally held directory."""
    require(re.fullmatch(r"[a-zA-Z0-9_.-]+", name) is not None, "output_name")
    for candidate in (name + ".pending", name):
        try:
            os.unlink(candidate, dir_fd=held_fd)
        except FileNotFoundError:
            pass
    os.fsync(held_fd)


def write_private_json(directory, name, result, *, held_fd=None):
    """Publish once through a held directory and return the final identity."""
    require(re.fullmatch(r"[a-zA-Z0-9_.-]+", name) is not None, "output_name")
    owns_fd = held_fd is None
    fd = directory_fd(directory) if owns_fd else held_fd
    try:
        if not owns_fd:
            output_identity(directory, fd)
        identity = stage_private_json(fd, name, result)
        return publish_staged_private_json(fd, name, result, identity)
    finally:
        try:
            try:
                os.unlink(name + ".pending", dir_fd=fd)
            except FileNotFoundError:
                pass
        finally:
            if owns_fd:
                os.close(fd)


def main(argv=None):
    global _CANCEL_REQUESTED
    os.umask(0o077)
    previous_handler = signal.getsignal(signal.SIGTERM)
    previous_cancel = _CANCEL_REQUESTED
    _CANCEL_REQUESTED = False
    directory = output_fd = None
    policy_requested = False
    policy_review = None
    outputs_verified = False

    def cancelled(signum, frame):
        global _CANCEL_REQUESTED
        if _CANCEL_REQUESTED:
            return
        _CANCEL_REQUESTED = True
        if not _SPAWNING:
            raise ScanError("scan_cancelled")

    signal.signal(signal.SIGTERM, cancelled)
    try:
        try:
            parser = SanitizedArgumentParser(description=__doc__)
            parser.add_argument("--analysis-root", type=Path, required=True)
            parser.add_argument("--trusted-root", type=Path, required=True)
            parser.add_argument("--authorization", type=Path, required=True)
            parser.add_argument("--output-dir", type=Path, required=True)
            parser.add_argument("--public-native-policy", type=Path)
            parser.add_argument("--public-native-policy-sha256")
            args = parser.parse_args(argv)
            policy_requested = (
                args.public_native_policy is not None
                or args.public_native_policy_sha256 is not None
            )
            require(
                not policy_requested
                or (
                    args.public_native_policy is not None
                    and args.public_native_policy_sha256 is not None
                ),
                "public_policy_arguments",
            )
            with authorized_roots(args.analysis_root, args.trusted_root):
                directory, output_fd = create_output(args.output_dir)
                try:
                    _path, fd, info = open_private_fd(args.authorization)
                    try:
                        raw = descriptor_bytes(fd)
                        require(
                            stat_fingerprint(os.fstat(fd)) == stat_fingerprint(info),
                            "authorization_changed_during_read",
                        )
                    finally:
                        os.close(fd)
                    authorization = json_object(raw)
                    if policy_requested:
                        from .public_native_policy import FreshPolicyReview

                        policy_review = FreshPolicyReview(
                            args.public_native_policy,
                            args.public_native_policy_sha256,
                            authorization,
                            {"path": str(args.authorization), "sha256": sha(raw)},
                            output_fd=output_fd,
                        )
                    result = (
                        _scan(
                            authorization,
                            directory,
                            record_observer=policy_review.observe,
                        )
                        if policy_review
                        else _scan(authorization, directory)
                    )
                    with bound_open(
                        {"path": str(args.authorization), "sha256": sha(raw)}
                    ) as (_path, _fd, after):
                        require(
                            stat_fingerprint(after) == stat_fingerprint(info),
                            "authorization_changed_during_scan",
                        )
                except INPUT_ERRORS as error:
                    result = {
                        "schema_version": "npa.image-byte-scan.v1",
                        "valid": False,
                        "complete": False,
                        "failure_code": str(error)
                        if isinstance(error, ScanError)
                        else "invalid_scan_configuration",
                    }
                output_identity(directory, output_fd)
                report_identity = write_private_json(
                    directory, "report.json", result, held_fd=output_fd
                )
                verify_private_json(
                    directory, output_fd, "report.json", result, report_identity
                )
                if policy_review is not None:
                    policy_review.accept_fresh_scan(result, directory)
                # Receipt publication can race with changes to any earlier output.
                verify_private_json(
                    directory, output_fd, "report.json", result, report_identity
                )
                if policy_review is not None:
                    policy_review.verify_output(directory)
                output_identity(directory, output_fd)
                outputs_verified = True
        except INPUT_ERRORS:
            outputs_verified = False
            result = {"valid": False, "complete": False}
            if policy_review is not None:
                policy_review.accepted = False
            if policy_requested and output_fd is not None:
                # Revoke only the acceptance name in our original directory; the
                # raw report/ledger and any replacement directory remain intact.
                try:
                    os.unlink("public-policy-acceptance.json", dir_fd=output_fd)
                    os.fsync(output_fd)
                except FileNotFoundError:
                    pass
                except OSError:
                    result["failure_code"] = "public_policy_receipt_cleanup_failed"
        passed = outputs_verified and (
            bool(policy_review and policy_review.accepted)
            if policy_requested
            else result["valid"]
        )
        label = (
            "image byte public-policy gate "
            if policy_requested
            else "complete image byte scan "
        )
        print(label + ("passed" if passed else "failed"))
        return 0 if passed else 1
    finally:
        if output_fd is not None:
            os.close(output_fd)
        signal.signal(signal.SIGTERM, previous_handler)
        _CANCEL_REQUESTED = previous_cancel
