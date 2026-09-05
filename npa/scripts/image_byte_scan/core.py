"""Complete image byte accounting with explicit roots and immutable inputs."""
from __future__ import annotations

import argparse
import base64
import csv
import fcntl
import hashlib
import importlib.machinery
import importlib.util
import io
import json
import os
import re
import signal
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
_CANCEL_REQUESTED = False
_SPAWNING = False
POLICY = "exact-or-short-ascii-token-v1"
REMOVED_PATH_RULES = ["freemius-secret-key", "hashicorp-tf-password", "kubernetes-secret-yaml", "nuget-config-password"]
PKCS12 = re.compile(r"(?i)(?:^|/)[^/]+\.p(?:12|fx)$")
DIGEST = re.compile(r"sha256:[0-9a-f]{64}$")
SHA = re.compile(r"[0-9a-f]{64}$")


class ScanError(ValueError):
    pass


INPUT_ERRORS = (OSError, ValueError, TypeError, KeyError, EOFError, tarfile.TarError,
                zlib.error, MemoryError, KeyboardInterrupt, RuntimeError, IndexError,
                AttributeError, OverflowError, ImportError, SyntaxError, SystemError, zipfile.BadZipFile)


def require(condition, code):
    if not condition:
        raise ScanError(code)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


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
        require(stat.S_ISDIR(info.st_mode) and info.st_uid == os.geteuid(), "root_owner_or_type")
        require(not info.st_mode & (0o077 if secret else 0o022), "root_permissions")
        roots.append(path)
    require(not roots[0].is_relative_to(roots[1]), "analysis_root_inside_build_context")
    token = _ROOTS.set(tuple(roots))
    identities = [(p.stat().st_dev, p.stat().st_ino) for p in roots]
    try:
        yield tuple(roots)
        for path, identity in zip(roots, identities, strict=True):
            current = path.lstat()
            require(stat.S_ISDIR(current.st_mode) and (current.st_dev, current.st_ino) == identity, "authorized_root_changed")
    finally:
        _ROOTS.reset(token)


def source_bindings():
    """Exact executable/configuration closure, excluding generated caches/binaries."""
    roots = _ROOTS.get()
    require(roots is not None, "explicit_roots_required")
    checkout = roots[1]
    require(Path(__file__).absolute() == checkout / "npa/scripts/image_byte_scan/core.py", "trusted_source_root_mismatch")
    folder = checkout / "npa/scripts/image_byte_scan"
    sources = [checkout / ".gitleaks.toml", checkout / "npa/scripts/scan_image_bytes.py",
               checkout / "npa/tests/docker/test_image_byte_go_build.py"]
    sources.extend(path for path in folder.rglob("*") if path.is_file() and
                   (path.suffix in {".py", ".go", ".mod", ".sum", ".json", ".md"} or path.name.startswith("LICENSE")))
    result = {}
    for path in sorted(sources):
        with open_source_fd(path) as (fd, _info):
            result[str(path.relative_to(checkout))] = {"path": str(path), "sha256": descriptor_digest(fd)}
    return result


@contextmanager
def open_source_fd(path):
    _path, fd, info = open_private_fd(path, secret=False)
    try:
        yield fd, info
        require(stat_fingerprint(os.fstat(fd)) == stat_fingerprint(info), "source_changed_during_read")
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
    require((path.is_relative_to(roots[0]) and not path.is_relative_to(roots[1]))
            or (not secret and path.is_relative_to(roots[1])), "input_outside_authorized_roots")
    info = path.stat()
    require(stat.S_ISREG(info.st_mode) and info.st_uid == os.geteuid(), "input_ownership_or_type")
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
        require(stat.S_ISREG(info.st_mode) and info.st_uid == os.geteuid(), "input_ownership_or_type")
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


def descriptor_bytes(fd):
    parts, offset = [], 0
    while data := os.pread(fd, CHUNK, offset):
        parts.append(data)
        offset += len(data)
    return b"".join(parts)


@contextmanager
def bound_open(spec, *, secret=True):
    require(isinstance(spec, dict) and isinstance(spec.get("path"), str), "input_binding_schema")
    require(isinstance(spec.get("sha256"), str) and SHA.fullmatch(spec["sha256"]), "input_binding_digest")
    path, fd, info = open_private_fd(spec["path"], secret=secret)
    try:
        require(descriptor_digest(fd) == spec["sha256"], "input_binding_changed")
        require(stat_fingerprint(os.fstat(fd)) == stat_fingerprint(info), "input_changed_during_read")
        yield path, fd, info
        require(stat_fingerprint(os.fstat(fd)) == stat_fingerprint(info), "input_changed_during_read")
    finally:
        os.close(fd)


def bound_file(spec, *, secret=True):
    with bound_open(spec, secret=secret) as (path, _fd, _info):
        return path


@contextmanager
def sealed_execution_input(source_fd, expected_digest, *, executable=False):
    """Execute/read verified immutable bytes while retaining original-file audits."""
    fd = os.memfd_create("verified-scanner-input", os.MFD_ALLOW_SEALING | os.MFD_CLOEXEC)
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
        require(descriptor_digest(fd) == expected_digest, "execution_input_digest_changed")
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


def bound_json(spec):
    with bound_open(spec) as (_path, fd, _info):
        data = descriptor_bytes(fd)
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
                require(not self.decoder.unused_data and not self.pending, "gzip_trailing_member_or_bytes")
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


def gzip_header(reader):
    """RFC1952 header bytes, including every optional field; no filename use."""
    header = bytearray(read_exact(reader, 10))
    require(header[:3] == b"\x1f\x8b\x08", "unsupported_gzip_method_or_magic")
    flags = header[3]
    require(not flags & 0xE0, "reserved_gzip_header_flags")
    if flags & 4:
        length = read_exact(reader, 2)
        header.extend(length)
        extra = read_exact(reader, int.from_bytes(length, "little"))
        header.extend(extra)
        cursor = 0
        while cursor < len(extra):
            require(len(extra) - cursor >= 4, "malformed_gzip_extra_subfield")
            size = int.from_bytes(extra[cursor + 2:cursor + 4], "little")
            require(cursor + 4 + size <= len(extra), "malformed_gzip_extra_subfield")
            cursor += 4 + size
    for bit in (8, 16):
        if flags & bit:
            while True:
                value = read_exact(reader, 1)
                header.extend(value)
                if value == b"\0":
                    break
    if flags & 2:
        checksum = read_exact(reader, 2)
        require(int.from_bytes(checksum, "little") == zlib.crc32(header) & 0xFFFF, "gzip_header_crc_mismatch")
        header.extend(checksum)
    return bytes(header)


def compile_literals(values, policy):
    patterns = []
    carry = max((len(value.encode("utf-8")) for value in values), default=0) + 1
    for index, value in enumerate(values):
        raw = value.encode("utf-8")
        pattern = re.escape(raw)
        if policy == POLICY and len(value) < 6:
            pattern = rb"(?<![A-Za-z0-9_])" + pattern + rb"(?![A-Za-z0-9_])"
        patterns.append((index, sha(raw), re.compile(pattern)))
    return carry, tuple(patterns)


class LiteralMatcher:
    """Finite raw literals with exact accepted short-name token boundaries."""
    def __init__(self, values, policy, *, compiled=None):
        self.carry, self.patterns = compile_literals(values, policy) if compiled is None else compiled
        self.buffer, self.base = b"", 0
        self.next_positions = [0] * len(self.patterns)

    def feed(self, data, *, final=False):
        self.buffer += data
        boundary = len(self.buffer) if final else max(0, len(self.buffer) - self.carry)
        found = []
        for index, digest, pattern in self.patterns:
            for match in pattern.finditer(self.buffer, max(0, self.next_positions[index] - self.base)):
                start = self.base + match.start()
                if start >= self.base + boundary:
                    break
                found.append({"rule_id": "private_literal", "literal_index": index, "literal_sha256": digest,
                              "byte_start": start, "byte_end": self.base + match.end()})
                self.next_positions[index] = self.base + match.end()
            self.next_positions[index] = max(self.next_positions[index], self.base + boundary)
        trim = max(0, boundary - 1)
        self.buffer = self.buffer[trim:]
        self.base += trim
        return found


AHO_PINS = {
    "source": '372a9a5e0a178b49ba5e5eab709606370d3b06fa8a1ece6a4ce120adc5a4a3e0',
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
            require(isinstance(binding, dict) and set(binding) == {"kind", *AHO_PINS}, "literal_engine_schema")
            require(binding["kind"] == "aho-corasick-v1", "literal_engine_kind")
            for role, expected in AHO_PINS.items():
                require(binding[role].get("sha256") == expected, "literal_engine_unreviewed_digest")
            require(all(name not in sys.modules for name in ("ahocorasick", "aho_matcher", AHO_MODULE)), "literal_engine_preloaded_module")
            source = bound_bytes(binding["source"])
            native = bound_bytes(binding["extension"], secret=True)
            wheel = bound_bytes(binding["wheel"], secret=True)
            with zipfile.ZipFile(io.BytesIO(wheel)) as archive:
                require(archive.namelist().count(AHO_MEMBER) == 1, "literal_engine_wheel_member")
                require(sha(archive.read(AHO_MEMBER)) == sha(native), "literal_engine_wheel_extension_binding")
                names = archive.namelist()
                require(len(names) == len(set(names)), "literal_engine_wheel_duplicate")
                record_name = "pyahocorasick-2.3.1.dist-info/RECORD"
                inventory = list(csv.reader(io.StringIO(archive.read(record_name).decode())))
                require({row[0] for row in inventory} == {name for name in names if not name.endswith("/")}, "literal_engine_wheel_record_inventory")
                require(len(inventory) == len({row[0] for row in inventory}), "literal_engine_wheel_record_duplicate")
                for name, checksum, size in inventory:
                    payload = archive.read(name)
                    if name == record_name:
                        require(not checksum and not size, "literal_engine_wheel_record_self")
                    else:
                        expected = "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=").decode()
                        require(checksum == expected and size == str(len(payload)), "literal_engine_wheel_record_hash")
                license_files = [name for name in names if name.endswith("/LICENSE")]
                require(len(license_files) == 1 and sha(archive.read(license_files[0])) == "2a7f4000fcb22199112e682ae44400f8836b9ff2500c46a8bed7d0edc93b2185", "literal_engine_license_binding")
            self.fd = os.memfd_create("verified-literal-extension", os.MFD_ALLOW_SEALING | os.MFD_CLOEXEC)
            view = memoryview(native)
            while view:
                written = os.write(self.fd, view)
                require(written > 0, "literal_engine_memfd_short_write")
                view = view[written:]
            # Linux UAPI: F_ADD_SEALS1033 / F_GET_SEALS1034; SEAL/SHRINK/GROW/WRITE bits1/2/4/8.
            # This CPython build omits the named fcntl seal constants.
            fcntl.fcntl(self.fd, 1033, 15)
            require(fcntl.fcntl(self.fd, 1034) == 15 and descriptor_digest(self.fd) == AHO_PINS["extension"], "literal_engine_sealed_binding")
            origin = f"/proc/self/fd/{self.fd}"
            loader = importlib.machinery.ExtensionFileLoader("ahocorasick", origin)
            spec = importlib.util.spec_from_file_location("ahocorasick", origin, loader=loader)
            self.native = importlib.util.module_from_spec(spec)
            sys.modules["ahocorasick"] = self.native
            loader.exec_module(self.native)
            require(self.native.__file__ == origin and self.native.unicode == 1, "literal_engine_native_mode")
            self.module = types.ModuleType(AHO_MODULE)
            sys.modules[AHO_MODULE] = self.module
            exec(compile(source, "<authorized-literal-matcher>", "exec"), self.module.__dict__)  # noqa: S102 - exact reviewed SHA; execute the verified bytes without a path reread.
        except BaseException:
            self.close()
            raise

    def receipt(self):
        require(sys.modules.get("ahocorasick") is self.native and sys.modules.get(AHO_MODULE) is self.module, "literal_engine_loaded_module_changed")
        require(fcntl.fcntl(self.fd, 1034) == 15 and descriptor_digest(self.fd) == AHO_PINS["extension"], "literal_engine_sealed_binding")
        return {"kind": "aho-corasick-v1", "pinned_sha256": AHO_PINS, "sealed_native_copy": True}

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
        try:
            parent_fd = directory_fd(stderr_path.parent)
            try:
                stderr_fd = os.open(stderr_path.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=parent_fd)
            finally:
                os.close(parent_fd)
            self.stderr = os.fdopen(stderr_fd, "wb")
            with (
                bound_open(authorization["helper"], secret=True) as (_, helper_source_fd, _),
                bound_open(authorization["config"], secret=False) as (_, config_source_fd, _),
                sealed_execution_input(helper_source_fd, authorization["helper"]["sha256"], executable=True) as helper_fd,
                sealed_execution_input(config_source_fd, authorization["config"]["sha256"]) as config_fd,
            ):
                _SPAWNING = True
                try:
                    self.process = subprocess.Popen([f"/proc/self/fd/{helper_fd}", "--config-fd", str(config_fd)],
                                                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr,
                                                    env={"PATH": os.defpath}, start_new_session=True,
                                                    pass_fds=(helper_fd, config_fd))
                finally:
                    _SPAWNING = False
            if _CANCEL_REQUESTED:
                raise ScanError("scan_cancelled")
            self._validate_ready(authorization)
        except BaseException:
            self.abort()
            raise

    def _validate_ready(self, authorization):
        self.ready = self._response()
        require(self.ready.get("type") == "ready" and self.ready.get("protocol") == "whole-file-gitleaks.v1", "helper_protocol")
        require(sha(canonical(self.ready)) == authorization["helper"]["ready_sha256"], "helper_ready_policy_changed")
        require(self.ready.get("version") == "8.28.0" and self.ready.get("config_sha256") == authorization["config"]["sha256"], "helper_config_or_version")
        require(self.ready.get("max_target_megabytes") == 0 and self.ready.get("ignore_inline_allow") is True and self.ready.get("redact") == 100, "helper_coverage_policy")
        require(self.ready.get("removed_content_path_rules") == REMOVED_PATH_RULES, "helper_content_path_policy")
        path_rules = self.ready.get("path_rules")
        require(isinstance(path_rules, list) and len(path_rules) == 5, "helper_path_rule_population")
        require({row["rule_id"] for row in path_rules} == {*REMOVED_PATH_RULES, "pkcs12-file"}, "helper_path_rule_inventory")
        key = next(row for row in path_rules if row["rule_id"] == "pkcs12-file")
        require(key.get("has_content_regex") is False and key.get("selector") == r"(?i)(?:^|\/)[^\/]+\.p(?:12|fx)$", "helper_pkcs12_selector")

    def _response(self):
        line = self.process.stdout.readline()
        require(bool(line), "helper_unexpected_eof")
        result = json_object(line)
        require(isinstance(result, dict), "helper_response_schema")
        return result

    def begin(self, length):
        require(type(length) is int and 0 <= length < 2**64, "protocol_length")
        self.process.stdin.write(struct.pack(">Q", length))

    def write(self, data):
        self.process.stdin.write(data)

    def end(self, length, digest):
        self.process.stdin.flush()
        result = self._response()
        self.ordinal += 1
        require(result.get("type") == "result" and type(result.get("ordinal")) is int and result["ordinal"] == self.ordinal, "helper_record_order")
        require(type(result.get("bytes")) is int and result["bytes"] == length and result.get("sha256") == digest, "helper_record_byte_receipt")
        findings = result.get("findings")
        require(isinstance(findings, list), "helper_findings_schema")
        allowed = {"rule_id", "start_line", "end_line"}
        for finding in findings:
            require(isinstance(finding, dict) and set(finding) == allowed, "helper_finding_schema")
            require(isinstance(finding["rule_id"], str) and re.fullmatch(r"[a-z0-9_-]+", finding["rule_id"]), "helper_rule_identifier")
            require(type(finding["start_line"]) is int and type(finding["end_line"]) is int, "helper_line_schema")
        self.bytes += length
        self.findings += len(findings)
        return findings

    def finish(self):
        self.process.stdin.close()
        result = self._response()
        require(result == {"type": "summary", "files": self.ordinal, "bytes": self.bytes, "findings": self.findings}, "helper_summary_receipt")
        require(self.process.stdout.read(1) == b"", "helper_extra_response")
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


class Ledger:
    def __init__(self, directory, detector, literals, policy, literal_engine=None, *, policy_config=None, literal_binding=None, record_observer=None):
        parent_fd = directory_fd(directory)
        try:
            stream_fd = os.open("records.jsonl", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=parent_fd)
        finally:
            os.close(parent_fd)
        self.stream = os.fdopen(stream_fd, "w", encoding="utf-8")
        self.record_observer = record_observer
        self.detector, self.literals, self.policy = detector, literals, policy
        if literal_engine is None:
            self.compiled_literals = compile_literals(literals, policy)
            self.new_matcher = lambda: LiteralMatcher((), self.policy, compiled=self.compiled_literals)
        else:
            self.compiled_literals = literal_engine.module.compile_literals(literals, policy)
            self.new_matcher = lambda: literal_engine.module.LiteralMatcher(self.compiled_literals)
        self.literal_policy_receipt = {"kind": policy, "inventory_sha256": literal_binding["sha256"] if literal_binding else None,
                                       "pattern_count": len(literals), "matcher_sha256": AHO_PINS["source"] if literal_engine else source_bindings()["npa/scripts/image_byte_scan/core.py"]["sha256"]}
        typed_binding = C.LiteralPolicyBinding(sha(canonical(self.literal_policy_receipt)), self.literal_policy_receipt["matcher_sha256"], len(literals)) if literal_binding is not None else None
        self.typed_literal_binding = typed_binding
        self.confidentiality = C.compile_policy(policy_config.get("customer_pattern"), policy_config.get("infra_pattern"), literal_policy=typed_binding) if policy_config is not None else None
        self.zero_run = None
        self.records = self.findings = self.scan_bytes = self.zero_bytes = 0
        self.regular_files = self.regular_bytes = 0

    def write(self, record):
        serialized = json.dumps(record, sort_keys=True) + "\n"
        self.stream.write(serialized)
        self.stream.flush()
        if self.record_observer is not None:
            self.record_observer(serialized.encode("utf-8"))

    def issue(self, code, context):
        self.findings += 1
        self.write({"type": "finding", "rule_id": code, **context})

    def send(self, reader, length, kind, context):
        self.flush_zeros()
        self.records += 1
        ordinal = self.records
        digest = hashlib.sha256()
        matcher = self.new_matcher()
        raw_parts = [] if self.confidentiality is not None else None
        literal_matches = []
        self.detector.begin(length)
        remaining = length
        while remaining:
            data = reader.read(min(CHUNK, remaining))
            require(bool(data) and len(data) <= remaining, "truncated_record")
            self.detector.write(data)
            digest.update(data)
            if raw_parts is not None:
                raw_parts.append(data)
            for finding in matcher.feed(data):
                literal_matches.append(finding)
                self.findings += 1
                self.write({"type": "finding", "record_ordinal": ordinal, **context, **finding})
            remaining -= len(data)
        for finding in matcher.feed(b"", final=True):
            literal_matches.append(finding)
            self.findings += 1
            self.write({"type": "finding", "record_ordinal": ordinal, **context, **finding})
        value = digest.hexdigest()
        found = self.detector.end(length, value)
        if self.confidentiality is not None:
            raw_record = b"".join(raw_parts)
            literal_scan = C.LiteralScan(self.typed_literal_binding, value, length,
                                        tuple(C.LiteralMatch(row["literal_index"], row["byte_start"], row["byte_end"]) for row in literal_matches), True) if self.typed_literal_binding is not None else None
            receipt = self.confidentiality.scan_record(raw_record, literal_scan=literal_scan)
            require(receipt.record_sha256 == value and receipt.byte_count == length, "confidentiality_record_binding")
            # Literal findings already have durable standalone receipts. The typed
            # composition verifies them; append only the additive regex findings.
            for item in receipt.findings:
                if "external_literal" not in item.views:
                    found.append(asdict(item))
            self.write({"type": "confidentiality_record", "record_ordinal": ordinal,
                        "policy_sha256": receipt.policy_sha256, "sha256": value, "bytes": length,
                        "line_count": receipt.line_count, "composed_findings": len(receipt.findings)})
        self.findings += len(found)
        self.scan_bytes += length
        self.write({"type": "record", "record_ordinal": ordinal, "kind": kind, "bytes": length,
                    "sha256": value, "findings": found, **context})
        return value

    def data(self, data, kind, context):
        return self.send(io.BytesIO(data), len(data), kind, context)

    def flush_zeros(self):
        if self.zero_run is not None:
            run, self.zero_run = self.zero_run, None
            self.send(io.BytesIO(b"\0" * run["bytes"]), run["bytes"], "verified_zero_content", run["context"])

    def zeros(self, data, context):
        require(not any(data), "nonzero_range_cannot_be_zero_accounted")
        self.zero_bytes += len(data)
        key = (context.get("scope"), context.get("layer_ordinal"))
        offset = context["tar_offset"]
        if self.zero_run is not None and (self.zero_run["key"] != key or self.zero_run["next_offset"] != offset):
            self.flush_zeros()
        if self.zero_run is None:
            self.zero_run = {"key": key, "context": context, "bytes": 0}
        self.zero_run["next_offset"] = offset + len(data)
        self.zero_run["bytes"] += len(data)
        self.write({"type": "verified_zero_range", "bytes": len(data), "sha256": sha(data), **context})


def safe_name(name):
    require(isinstance(name, str) and "\x00" not in name, "tar_path_encoding")
    path = PurePosixPath(name)
    require(not path.is_absolute() and ".." not in path.parts, "tar_path_escape")
    return str(path)


def parse_pax(data):
    result, cursor = {}, 0
    while cursor < len(data):
        space = data.find(b" ", cursor)
        require(space > cursor and data[cursor:space].isdigit(), "pax_record_length")
        length = int(data[cursor:space])
        require(length > space - cursor + 2 and cursor + length <= len(data), "pax_record_boundary")
        record = data[space + 1:cursor + length]
        require(record.endswith(b"\n") and b"=" in record, "pax_record_format")
        key, value = record[:-1].split(b"=", 1)
        key, value = key.decode("utf-8"), value.decode("utf-8")
        require(key and key not in result and "\x00" not in value, "pax_duplicate_or_invalid_key")
        require(key in {"path", "linkpath", "uid", "gid", "uname", "gname", "mtime", "atime", "ctime"}
                or key.startswith(("SCHILY.xattr.", "LIBARCHIVE.xattr.")), "unsupported_pax_semantics")
        result[key] = value
        cursor += length
    return result


def walk_tar(reader, sink, scope, file_handler):
    """Read every physical tar byte, including extension records and EOF padding."""
    index, zero_headers = 0, 0
    seen = set()
    pending, global_pax = {}, {}
    long_name = long_link = None
    while True:
        offset = reader.tell()
        raw = read_exact(reader, 512, eof=True)
        if not raw:
            require(zero_headers >= 2, "tar_missing_end_markers")
            require(not pending and long_name is None and long_link is None, "orphan_tar_extension")
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
                context = {**scope, "entry_ordinal": index, "tar_offset": reader.tell() - len(data)}
                if any(data):
                    sink.data(data, "unexplained_tar_trailer", context)
                else:
                    sink.zeros(data, context)
            require(zero_headers >= 2, "tar_incomplete_end_markers")
            break
        sink.data(raw, "raw_tar_header", context)
        info = tarfile.TarInfo.frombuf(raw, encoding="utf-8", errors="strict")
        require(info.size >= 0, "negative_tar_size")
        require(info.type in {tarfile.REGTYPE, tarfile.AREGTYPE, tarfile.DIRTYPE, tarfile.SYMTYPE,
                              tarfile.LNKTYPE, tarfile.CHRTYPE, tarfile.BLKTYPE, tarfile.FIFOTYPE,
                              tarfile.XHDTYPE, tarfile.XGLTYPE, tarfile.GNUTYPE_LONGNAME, tarfile.GNUTYPE_LONGLINK}, "unsupported_tar_entry_type")
        if info.type in {tarfile.XHDTYPE, tarfile.XGLTYPE, tarfile.GNUTYPE_LONGNAME, tarfile.GNUTYPE_LONGLINK}:
            extension_context = {**context, "tar_offset": reader.tell()}
            data = read_exact(reader, info.size)
            sink.data(data, "raw_tar_extension", extension_context)
            if info.type in {tarfile.XHDTYPE, tarfile.XGLTYPE}:
                parsed = parse_pax(data)
                if info.type == tarfile.XGLTYPE:
                    require(not set(parsed) & {"path", "linkpath"}, "global_pax_name_override")
                    global_pax.update(parsed)
                else:
                    require(not set(parsed) & set(pending), "ambiguous_pax_override")
                    pending.update(parsed)
            else:
                require(data.endswith(b"\x00") and b"\x00" not in data[:-1], "gnu_long_name_encoding")
                value = data[:-1].decode("utf-8")
                if info.type == tarfile.GNUTYPE_LONGNAME:
                    require(long_name is None, "ambiguous_gnu_name")
                    long_name = value
                else:
                    require(long_link is None, "ambiguous_gnu_link")
                    long_link = value
        else:
            require(not (long_name is not None and "path" in pending) and not (long_link is not None and "linkpath" in pending), "ambiguous_name_extensions")
            name = safe_name(pending.get("path", long_name if long_name is not None else info.name))
            link = pending.get("linkpath", long_link if long_link is not None else info.linkname)
            require("\x00" not in link, "tar_link_encoding")
            require(name not in seen, "duplicate_tar_path")
            seen.add(name)
            sink.data(name.encode("utf-8"), "logical_tar_path", context)
            if link:
                sink.data(link.encode("utf-8"), "logical_tar_link", context)
            if PKCS12.search(name):
                sink.issue("pkcs12-file", context)
            pending, long_name, long_link = {}, None, None
            if info.isreg():
                file_handler(reader, info.size, name, {**context, "tar_offset": reader.tell()})
            else:
                require(info.size == 0, "nonregular_tar_body")
        padding = (-info.size) % 512
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
    return {"headers": index, "decoded_bytes": reader.tell(), "zero_end_blocks": zero_headers}


def graph(fd, length, verification, expected_id):
    """Rebind metadata to the accepted exact archive before scanning its layers."""
    os.lseek(fd, 0, os.SEEK_SET)
    with os.fdopen(os.dup(fd), "rb") as file, tarfile.open(fileobj=file, mode="r:") as archive:
        members = {}
        for item in archive.getmembers():
            name = safe_name(item.name)
            require(name not in members, "duplicate_outer_path")
            members[name] = item

        def payload(name):
            info = members[name]
            require(info.isfile(), "graph_metadata_not_regular")
            return archive.extractfile(info).read()

        saved = json_object(payload("manifest.json"))
        require(isinstance(saved, list) and len(saved) == 1 and isinstance(saved[0], dict), "graph_image_population")
        config_path = safe_name(saved[0]["Config"])
        data = payload(config_path)
        config_digest = "sha256:" + sha(data)
        require(config_digest == verification["image_config_digest"], "graph_config_binding")
        config = json_object(data)
        diff_ids = config["rootfs"]["diff_ids"]
        require(config["rootfs"]["type"] == "layers" and diff_ids == verification["verified_layer_diff_ids"], "graph_layer_binding")
        names = saved[0]["Layers"]
        require(isinstance(names, list) and len(names) == len(diff_ids) == verification["layer_count"], "graph_layer_population")
        manifest_digest = verification.get("image_manifest_digest")
        descriptors = None
        if manifest_digest is not None:
            index = json_object(payload("index.json"))
            require(index.get("schemaVersion") == 2 and len(index["manifests"]) == 1, "graph_oci_population")
            descriptor = index["manifests"][0]
            require(descriptor["digest"] == manifest_digest, "graph_manifest_selection")
            data = payload("blobs/sha256/" + manifest_digest[7:])
            require(len(data) == descriptor["size"] and "sha256:" + sha(data) == manifest_digest, "graph_manifest_digest")
            manifest = json_object(data)
            require(manifest.get("schemaVersion") == 2 and manifest["mediaType"] == descriptor["mediaType"], "graph_manifest_schema")
            require(manifest["config"]["digest"] == config_digest and manifest["config"]["size"] == members[config_path].size, "graph_config_descriptor")
            descriptors = manifest["layers"]
            require(len(descriptors) == len(names), "graph_descriptor_population")
        else:
            require("index.json" not in members and "oci-layout" not in members, "unbound_oci_metadata")
        require(expected_id in {config_digest, manifest_digest} and verification["expected_image_id"] == expected_id, "graph_expected_identity")
        result = []
        for ordinal, name in enumerate(names):
            name = safe_name(name)
            item = members[name]
            require(item.isfile() and item.offset_data + item.size <= length, "graph_layer_range")
            descriptor = descriptors[ordinal] if descriptors is not None else None
            if descriptor is not None:
                require(name == "blobs/sha256/" + descriptor["digest"][7:] and item.size == descriptor["size"], "graph_layer_descriptor")
                require(descriptor["mediaType"] in {"application/vnd.oci.image.layer.v1.tar", "application/vnd.oci.image.layer.v1.tar+gzip", "application/vnd.docker.image.rootfs.diff.tar", "application/vnd.docker.image.rootfs.diff.tar.gzip"}, "unsupported_layer_codec")
            result.append({"ordinal": ordinal, "name": name, "offset": item.offset_data, "size": item.size,
                           "diff_id": diff_ids[ordinal], "descriptor": descriptor})
        return result


def stat_fingerprint(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns,
            info.st_mode, info.st_uid, info.st_gid)


def fingerprint(path):
    return stat_fingerprint(path.stat())


GO_SOURCE_NAMES = ("main.go", "main_test.go", "go.mod", "go.sum", "build.py", "LICENSE-GO", "LICENSE-GITLEAKS", "README.md")
GO_TEST_SOURCE = "npa/tests/docker/test_image_byte_go_build.py"


def current_go_sources():
    checkout = _ROOTS.get()[1]
    files = {name: checkout / "npa/scripts/image_byte_scan/go_helper" / name for name in GO_SOURCE_NAMES}
    files[GO_TEST_SOURCE] = checkout / GO_TEST_SOURCE
    result = {}
    for name, path in files.items():
        with open_source_fd(path) as (fd, _info):
            result[name] = descriptor_digest(fd)
    return result


def verified_tools(spec):
    receipt = bound_json(spec)
    require(receipt.get("schema_version") == "npa.image-byte-scan-tools.v1", "tools_receipt_schema")
    require(receipt.get("source") == current_go_sources(), "tools_receipt_source_changed")
    with open_source_fd(_ROOTS.get()[1] / ".gitleaks.toml") as (fd, _info):
        require(receipt["config"]["sha256"] == descriptor_digest(fd), "tools_receipt_checkout_config_changed")
    for role in ("helper", "config", "ready"):
        bound_file(receipt[role], secret=role != "config")
    ready = bound_json(receipt["ready"])
    helper = {**receipt["helper"], "ready_sha256": sha(canonical(ready))}
    return helper, receipt["config"]


def input_snapshots(authorization):
    items = [("tools_receipt", authorization["tools_receipt"], True),
             ("verification_report", authorization["verification_report"], True),
             ("helper", authorization["helper"], True), ("config", authorization["config"], False)]
    tools_receipt = bound_json(authorization["tools_receipt"])
    items.append(("helper_ready", tools_receipt["ready"], True))
    literal = authorization.get("literal_inventory")
    if literal is not None:
        items.append(("literal_inventory", literal, True))

    engine = authorization.get("literal_engine")
    require("literal_engine" not in authorization or isinstance(engine, dict), "literal_engine_schema")
    if engine is not None:
        require(isinstance(engine, dict) and set(engine) == {"kind", *AHO_PINS}, "literal_engine_schema")
        items.extend(("literal_engine_" + role, engine[role], role != "source") for role in AHO_PINS)
    configured_sources = authorization.get("sources")
    require(configured_sources == source_bindings(), "scanner_source_binding_changed")
    items.extend(("source:" + role, spec, False) for role, spec in configured_sources.items())
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
            require(current_path == path and stat_fingerprint(info) == before, "input_changed_during_scan")


def _scan(authorization, directory, detector_type=Detector, *, record_observer=None):
    require(isinstance(authorization, dict) and authorization.get("schema_version") == "npa.image-byte-scan-authorization.v1", "authorization_schema")
    required = {"schema_version", "accepted_verification", "archive", "verification_report", "expected_image_id", "helper", "config", "sources", "tools_receipt"}
    require(required <= set(authorization) <= required | {"literal_inventory", "literal_engine", "confidentiality"}, "authorization_fields")
    require(authorization.get("accepted_verification") is True, "verification_not_accepted")
    helper, config = verified_tools(authorization["tools_receipt"])
    require(helper == authorization["helper"] and config == authorization["config"], "tools_receipt_authorization_changed")
    snapshots = input_snapshots(authorization)
    verification = bound_json(authorization["verification_report"])
    require(verification.get("valid") is True, "verification_did_not_pass")
    require(verification.get("schema_version") == "npa.curobo.image-verification.v1", "verification_schema")
    require(authorization["archive"]["sha256"] == verification["docker_save_sha256"], "archive_verification_binding")
    values, policy, literal_binding = [], "exact-substring-v1", authorization.get("literal_inventory")
    if literal_binding is not None:
        inventory = bound_json(literal_binding)
        values = inventory.get("literals")
        require(isinstance(values, list) and all(isinstance(value, str) and value for value in values), "literal_inventory_schema")
        policy = literal_binding["matching_policy"]
        require(policy in {"exact-substring-v1", POLICY}, "literal_matching_policy")

    require(literal_binding is not None or authorization.get("confidentiality") is not None, "confidentiality_policy_required")
    policy_config = bound_json(authorization["confidentiality"]) if authorization.get("confidentiality") is not None else None
    require(policy_config is not None or bool(values), "nonempty_confidentiality_policy_required")
    if policy_config is not None:
        require(isinstance(policy_config, dict) and set(policy_config) <= {"customer_pattern", "infra_pattern"}, "confidentiality_schema")
        C.compile_policy(policy_config.get("customer_pattern"), policy_config.get("infra_pattern"))
    archive_path, fd, initial = open_private_fd(authorization["archive"]["path"])
    try:
        require(descriptor_digest(fd) == authorization["archive"]["sha256"], "input_binding_changed")
        require(stat_fingerprint(os.fstat(fd)) == stat_fingerprint(initial), "input_changed_during_read")
    except BaseException:
        os.close(fd)
        raise
    detector = sink = literal_engine = None
    report = {"schema_version": "npa.image-byte-scan.v1", "valid": False, "complete": False,
              "authorization_sha256": sha(canonical(authorization)), "archive_sha256": authorization["archive"]["sha256"],
              "image_config_digest": verification["image_config_digest"], "image_manifest_digest": verification.get("image_manifest_digest"),
              "expected_image_id": authorization["expected_image_id"], "private_literals_configured": literal_binding is not None,
              "private_literal_count": len(values), "literal_matching_policy": policy, "layers": []}
    try:
        layers = graph(fd, initial.st_size, verification, authorization["expected_image_id"])
        if authorization.get("literal_engine") is not None:
            literal_engine = AuthorizedAho(authorization["literal_engine"])
        detector = detector_type(authorization, directory / "helper-stderr.jsonl")
        sink = Ledger(directory, detector, values, policy, literal_engine, policy_config=policy_config,
                      literal_binding=literal_binding, **({"record_observer": record_observer} if record_observer is not None else {}))
        report["confidentiality_policy"] = sink.confidentiality.receipt() if sink.confidentiality is not None else {"mode": "exact-literals-v1", "binding": sink.literal_policy_receipt}
        layer_names = {row["name"] for row in layers}
        locations = {}

        def outer_file(reader, size, name, context):
            offset = reader.tell()
            if name in layer_names:
                digest = hashlib.sha256()
                remaining = size
                while remaining:
                    data = reader.read(min(CHUNK, remaining))
                    require(data, "truncated_outer_blob")
                    digest.update(data)
                    remaining -= len(data)
                locations[name] = {"offset": offset, "size": size, "sha256": digest.hexdigest()}
                sink.write({"type": "encoded_layer_blob", "bytes": size, "sha256": digest.hexdigest(), **context})
            else:
                sink.send(reader, size, "outer_regular_content", context)

        outer = HashedReader(Slice(fd, 0, initial.st_size))
        report["outer"] = walk_tar(outer, sink, {"scope": "outer"}, outer_file)
        require(outer.position == initial.st_size and outer.digest.hexdigest() == authorization["archive"]["sha256"], "outer_complete_byte_accounting")
        for row in layers:
            location = locations[row["name"]]
            require(location["offset"] == row["offset"] and location["size"] == row["size"], "manual_graph_range_disagreement")
            if row["descriptor"]:
                require("sha256:" + location["sha256"] == row["descriptor"]["digest"], "layer_compressed_digest")
            elif row["name"].startswith("blobs/"):
                require(row["name"] == "blobs/sha256/" + location["sha256"], "classic_layer_blob_digest")
            raw = Slice(fd, row["offset"], row["size"])
            header = read_exact(raw, min(10, row["size"]))
            gzip = header[:2] == b"\x1f\x8b"
            if row["descriptor"]:
                declared_gzip = row["descriptor"]["mediaType"].endswith(("+gzip", ".gzip"))
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
            require("sha256:" + decoded.digest.hexdigest() == row["diff_id"], "layer_uncompressed_diff_id")
            require(raw.position == row["size"], "unaccounted_compressed_bytes")
            report["layers"].append({"ordinal": row["ordinal"], "diff_id": row["diff_id"], "compressed_sha256": location["sha256"],
                                     "compressed_bytes": row["size"], "codec": "gzip" if gzip else "raw", **details})
        require(sink.regular_files == verification["regular_files_read"] and sink.regular_bytes == verification["content_bytes_read"], "verifier_regular_population_disagreement")
        sink.flush_zeros()
        report["helper_summary"] = detector.finish()
        recheck_snapshots(snapshots)
        require(authorization["sources"] == source_bindings(), "scanner_source_population_changed")
        report["literal_engine"] = literal_engine.receipt() if literal_engine is not None else {"kind": "regex-reference-v1"}
        report["input_snapshot_receipts"] = [{"role": role, "sha256": spec["sha256"], "stat": list(before)} for role, spec, secret, path, before in snapshots]
        report["complete"] = True
        report["valid"] = sink.findings == 0
    except INPUT_ERRORS as error:
        report["failure_code"] = str(error) if isinstance(error, ScanError) else "uninterpretable_input_or_scanner_failure"
    finally:
        if detector is not None and not detector.joined:
            detector.abort()
        report["helper_joined"] = detector is None or detector.joined
        if sink is not None:
            if sink.zero_run is not None:
                report.update(valid=False, complete=False, failure_code=report.get("failure_code", "pending_zero_range_scan"))
            report.update(records=sink.records, scanned_bytes=sink.scan_bytes, verified_zero_bytes=sink.zero_bytes,
                          regular_files=sink.regular_files, regular_bytes=sink.regular_bytes, findings=sink.findings)
            sink.stream.close()
        if literal_engine is not None:
            literal_engine.close()
        try:
            current = os.fstat(fd)
            path_current = archive_path.lstat()
            if (stat_fingerprint(initial) != stat_fingerprint(current)
                    or not stat.S_ISREG(path_current.st_mode)
                    or (current.st_dev, current.st_ino) != (path_current.st_dev, path_current.st_ino)):
                report.update(valid=False, complete=False, failure_code="archive_changed_during_scan")
        except OSError:
            report.update(valid=False, complete=False, failure_code="archive_changed_during_scan")
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


def scan(authorization, directory, *, analysis_root, trusted_root, detector_type=Detector):
    with authorized_roots(analysis_root, trusted_root):
        return _scan(authorization, directory, detector_type=detector_type)


def directory_fd(path):
    """Open a private directory beneath analysis through nofollow descriptors."""
    roots = _ROOTS.get()
    require(roots is not None, "explicit_roots_required")
    path = Path(path).absolute()
    require(".." not in path.parts and path.is_relative_to(roots[0]) and not path.is_relative_to(roots[1]), "output_directory_scope")
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC | os.O_DIRECTORY
    fd = os.open(path.anchor, flags)
    try:
        for part in path.parts[1:]:
            next_fd = os.open(part, flags, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        info = os.fstat(fd)
        require(info.st_uid == os.geteuid() and not info.st_mode & 0o077, "output_directory_permissions")
        return fd
    except BaseException:
        os.close(fd)
        raise


def create_output(path):
    path = Path(path).absolute()
    parent = directory_fd(path.parent)
    try:
        os.mkdir(path.name, 0o700, dir_fd=parent)
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_DIRECTORY | os.O_CLOEXEC, dir_fd=parent)
        return path, fd
    finally:
        os.close(parent)


def write_private_json(directory, name, result):
    require(re.fullmatch(r"[a-zA-Z0-9_.-]+", name) is not None, "output_name")
    fd = directory_fd(directory)
    created = False
    temporary = name + ".pending"
    try:
        output = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=fd)
        created = True
        with os.fdopen(output, "w", encoding="utf-8") as stream:
            json.dump(result, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, name, src_dir_fd=fd, dst_dir_fd=fd, follow_symlinks=False)
        os.fsync(fd)
    finally:
        if created:
            os.unlink(temporary, dir_fd=fd)
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
            policy_requested = args.public_native_policy is not None or args.public_native_policy_sha256 is not None
            require(not policy_requested or (args.public_native_policy is not None and args.public_native_policy_sha256 is not None), "public_policy_arguments")
            with authorized_roots(args.analysis_root, args.trusted_root):
                directory, output_fd = create_output(args.output_dir)
                try:
                    _path, fd, info = open_private_fd(args.authorization)
                    try:
                        raw = descriptor_bytes(fd)
                        require(stat_fingerprint(os.fstat(fd)) == stat_fingerprint(info), "authorization_changed_during_read")
                    finally:
                        os.close(fd)
                    authorization = json_object(raw)
                    if policy_requested:
                        from .public_native_policy import FreshPolicyReview
                        policy_review = FreshPolicyReview(args.public_native_policy, args.public_native_policy_sha256, authorization,
                                                          {"path": str(args.authorization), "sha256": sha(raw)})
                    result = (_scan(authorization, directory, record_observer=policy_review.observe) if policy_review
                              else _scan(authorization, directory))
                    with bound_open({"path": str(args.authorization), "sha256": sha(raw)}) as (_path, _fd, after):
                        require(stat_fingerprint(after) == stat_fingerprint(info), "authorization_changed_during_scan")
                except INPUT_ERRORS as error:
                    result = {"schema_version": "npa.image-byte-scan.v1", "valid": False, "complete": False,
                              "failure_code": str(error) if isinstance(error, ScanError) else "invalid_scan_configuration"}
                current = directory_fd(directory)
                try:
                    require((os.fstat(current).st_dev, os.fstat(current).st_ino) ==
                            (os.fstat(output_fd).st_dev, os.fstat(output_fd).st_ino), "output_directory_replaced")
                finally:
                    os.close(current)
                write_private_json(directory, "report.json", result)
                if policy_review is not None:
                    policy_review.accept_fresh_scan(result, directory)
        except INPUT_ERRORS:
            result = {"valid": False, "complete": False}
        passed = bool(policy_review and policy_review.accepted) if policy_requested else result["valid"]
        label = "image byte public-policy gate " if policy_requested else "complete image byte scan "
        print(label + ("passed" if passed else "failed"))
        return 0 if passed else 1
    finally:
        if output_fd is not None:
            os.close(output_fd)
        signal.signal(signal.SIGTERM, previous_handler)
        _CANCEL_REQUESTED = previous_cancel
