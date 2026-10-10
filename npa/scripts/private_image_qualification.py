#!/usr/bin/env python3
"""Qualify one hash-selected private OCI archive without exporting CI policy."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile

SCANNER_REVISION = "e99f5f2de8224b0d7c21eea1fdfb2c369ec04f0b"
EXPORT_ROOT = Path(".local/share/npa/private-image-qualification/exports")
RECEIPT_ROOT = Path(".local/share/npa/private-image-qualification/receipts")
ADJUDICATION_ROOT = Path(".local/share/npa/private-image-adjudication/requests")
MANIFEST_SCHEMA = "npa.private-image-qualification.v1"
REGISTRY_MANIFEST_SCHEMA = "npa.private-registry-manifest-qualification.v1"
CHUNK = 1024 * 1024
MANIFEST_BYTES = 16384
HEX = re.compile(r"[0-9a-f]{64}")
SSH_OPTIONS = (
    "ssh",
    "-F",
    "/dev/null",
    "-T",
    "-o",
    "BatchMode=yes",
    "-o",
    "IdentitiesOnly=yes",
    "-o",
    "StrictHostKeyChecking=yes",
    "-o",
    "LogLevel=ERROR",
    "-o",
    "GlobalKnownHostsFile=/dev/null",
)
RECEIPT_FILES = (
    "manifest.json",
    "capacity.json",
    "summary.json",
    "graph/verification.json",
    "scan/report.json",
    "scan/records.jsonl",
    "integration/native-checks.json",
    "integration/native-checks-failure.json",
    "phases.jsonl",
    "diagnostics/helper-build.log",
    "diagnostics/native-dependencies.log",
    "diagnostics/native-integration.log",
    "diagnostics/graph-verification.log",
    "diagnostics/image-scan.log",
    "retained/retention.json",
)
PHASES = frozenset(
    {
        "manifest-transfer",
        "capacity",
        "scanner-preparation",
        "policy-check",
        "helper-build",
        "native-dependencies",
        "native-integration",
        "capacity-after-preparation",
        "archive-transfer",
        "image-scan",
        "graph-verification",
        "scan-authorization",
        "scan-report",
        "scan-input-retention",
        "review-request",
        "review-evidence",
        "review-binding",
        "adjudication",
        "receipt-retention",
    }
)
_CURRENT_RUN = ContextVar("private_qualification_run", default=None)
ERROR_CODES = frozenset(
    {
        "archive_digest",
        "adjudication_failed",
        "archive_size",
        "directory_not_private",
        "directory_scope",
        "duplicate_json_key",
        "image_identity",
        "input_changed",
        "input_not_private_regular",
        "input_replaced",
        "insufficient_capacity",
        "insufficient_capacity_after_preparation",
        "invalid_arguments",
        "manifest_capacity",
        "manifest_digest",
        "manifest_schema",
        "manifest_selector",
        "manifest_size",
        "oci_verification_failed",
        "output_replaced",
        "parent_replaced",
        "policy_unavailable",
        "qualification_cancelled",
        "progress_phase",
        "policy_receipt_invalid",
        "private_receipt_not_retained",
        "receipt_capacity",
        "receipt_identity",
        "report_counts",
        "report_identity",
        "report_graph_scope",
        "report_missing",
        "run_identity",
        "scan_authorization_failed",
        "scan_input_retention_failed",
        "helper_build_failed",
        "native_dependency_failed",
        "native_integration_failed",
        "scanner_source_dirty",
        "scanner_source_revision",
        "ssh_authentication_failed",
        "ssh_host_verification_failed",
        "ssh_connection_failed",
        "remote_export_missing",
        "remote_export_permissions",
        "remote_interface_failed",
        "ssh_configuration",
        "ssh_host",
        "ssh_port",
        "ssh_transfer_failed",
        "ssh_user",
        "transfer_arguments",
        "transfer_digest",
        "transfer_oversized",
        "transfer_role",
        "transfer_truncated",
    }
)


class _QualificationError(Exception):
    """A fixed failure code, never subprocess output or private input."""

    def __init__(self, code):
        self.code = code if code in ERROR_CODES else "qualification_failed"
        super().__init__(self.code)


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise _QualificationError("invalid_arguments")


def _require(condition, code):
    if not condition:
        raise _QualificationError(code)


def _failure(error, stage):
    name = type(error).__name__
    known_classes = {
        "_QualificationError",
        "OSError",
        "PermissionError",
        "FileNotFoundError",
        "FileExistsError",
        "NotADirectoryError",
        "IsADirectoryError",
        "BrokenPipeError",
        "ValueError",
        "JSONDecodeError",
        "UnicodeDecodeError",
        "KeyError",
        "TypeError",
    }
    return {
        "failure_stage": stage,
        "failure_code": error.code
        if isinstance(error, _QualificationError)
        else "operation_failed",
        "exception_class": name if name in known_classes else "OperationError",
    }


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _fingerprint(info):
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mode,
        info.st_uid,
        info.st_nlink,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _descriptor_digest(descriptor, length):
    value, offset = hashlib.sha256(), 0
    while offset < length:
        data = os.pread(descriptor, min(CHUNK, length - offset), offset)
        _require(bool(data), "input_changed")
        value.update(data)
        offset += len(data)
    _require(not os.pread(descriptor, 1, length), "input_changed")
    return value.hexdigest()


def _check_parent(path, held):
    current = _directory(path)
    try:
        before, after = os.fstat(held), os.fstat(current)
        _require(
            (before.st_dev, before.st_ino) == (after.st_dev, after.st_ino),
            "parent_replaced",
        )
    finally:
        os.close(current)


def _directory(path, *, create=False):
    _require(path.is_absolute() and ".." not in path.parts, "directory_scope")
    current = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:]:
            if create:
                try:
                    os.mkdir(part, 0o700, dir_fd=current)
                except FileExistsError:
                    pass
            child = os.open(
                part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=current
            )
            os.close(current)
            current = child
        info = os.fstat(current)
        _require(
            info.st_uid == os.getuid() and not info.st_mode & 0o077,
            "directory_not_private",
        )
        result, current = current, -1
        return result
    finally:
        if current >= 0:
            os.close(current)


@contextmanager
def _private_input(path, *, frozen_archive=False):
    parent, descriptor = _directory(path.parent), None
    try:
        descriptor = os.open(
            path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent
        )
        before = os.fstat(descriptor)
        link_allowed = before.st_nlink == 1 or (
            frozen_archive and before.st_nlink > 1 and not before.st_mode & 0o222
        )
        _require(
            stat.S_ISREG(before.st_mode)
            and link_allowed
            and before.st_uid == os.getuid()
            and not before.st_mode & 0o077,
            "input_not_private_regular",
        )
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            yield stream, before
        _require(
            _fingerprint(before) == _fingerprint(os.fstat(descriptor)), "input_changed"
        )
        after = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
        _require(_fingerprint(before) == _fingerprint(after), "input_replaced")
        _check_parent(path.parent, parent)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent)


@contextmanager
def _private_output(path):
    parent = _directory(path.parent)
    try:
        descriptor = os.open(
            path.name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent,
        )
        with os.fdopen(descriptor, "wb") as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
            before = os.fstat(stream.fileno())
            after = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
            _require(
                before.st_nlink == 1 and _fingerprint(before) == _fingerprint(after),
                "output_replaced",
            )
            _check_parent(path.parent, parent)
    finally:
        os.close(parent)


def _write(path, data):
    with _private_output(path) as stream:
        stream.write(data)


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate_json_key")
        result[key] = value
    return result


def _manifest(data, selector):
    _require(HEX.fullmatch(selector) is not None, "manifest_selector")
    _require(len(data) <= MANIFEST_BYTES and _sha(data) == selector, "manifest_digest")
    value = json.loads(data, object_pairs_hook=_unique_object)
    _require(
        type(value) is dict
        and set(value)
        == {
            "schema_version",
            "archive_sha256",
            "archive_bytes",
            "expected_image_id",
            "workspace_bytes",
        },
        "manifest_schema",
    )
    _require(
        value["schema_version"] in (MANIFEST_SCHEMA, REGISTRY_MANIFEST_SCHEMA),
        "manifest_schema",
    )
    _require(
        type(value["archive_sha256"]) is str and HEX.fullmatch(value["archive_sha256"]),
        "archive_digest",
    )
    _require(
        type(value["expected_image_id"]) is str
        and re.fullmatch(r"sha256:[0-9a-f]{64}", value["expected_image_id"]),
        "image_identity",
    )
    for key in ("archive_bytes", "workspace_bytes"):
        _require(type(value[key]) is int and value[key] > 0, "manifest_capacity")
    return value


def _copy_exact(source, destination, size, digest):
    observed, value = 0, hashlib.sha256()
    while observed < size:
        _check_cancelled()
        data = source.read(min(CHUNK, size - observed))
        _require(bool(data), "transfer_truncated")
        destination.write(data)
        value.update(data)
        observed += len(data)
    _require(not source.read(1), "transfer_oversized")
    _require(value.hexdigest() == digest, "transfer_digest")


def _remote_manifest(selector):
    _require(HEX.fullmatch(selector) is not None, "manifest_selector")
    directory = Path.home() / EXPORT_ROOT / selector
    with _private_input(directory / "manifest.json") as (stream, info):
        _require(info.st_size <= MANIFEST_BYTES, "manifest_size")
        data = stream.read(MANIFEST_BYTES + 1)
    return directory, data, _manifest(data, selector)


def _remote_fetch(selector, role):
    directory, data, manifest = _remote_manifest(selector)
    if role == "manifest":
        sys.stdout.buffer.write(data)
        return
    _require(role == "archive", "transfer_role")
    with _private_input(
        directory / "image.tar",
        frozen_archive=True,
    ) as (stream, info):
        _require(info.st_size == manifest["archive_bytes"], "archive_size")
        _copy_exact(stream, sys.stdout.buffer, info.st_size, manifest["archive_sha256"])
        _require(
            _descriptor_digest(stream.fileno(), info.st_size)
            == manifest["archive_sha256"],
            "input_changed",
        )


def _adjudication_request(data, selector):
    _require(
        HEX.fullmatch(selector)
        and _sha(data) == selector
        and len(data) <= MANIFEST_BYTES,
        "manifest_digest",
    )
    value = json.loads(data, object_pairs_hook=_unique_object)
    digests = (
        "image_manifest_sha256",
        "scan_receipt_sha256",
        "retention_sha256",
        "review_tar_sha256",
        "disposition_manifest_sha256",
        "independent_review_sha256",
    )
    sizes = ("scan_receipt_bytes", "review_tar_bytes", "workspace_bytes")
    _require(
        type(value) is dict
        and set(value) == {"schema_version", "scan_run_id", *digests, *sizes},
        "manifest_schema",
    )
    _require(
        value["schema_version"] == "npa.private-image-adjudication.v1",
        "manifest_schema",
    )
    _require(
        all(type(value[key]) is str and HEX.fullmatch(value[key]) for key in digests),
        "manifest_digest",
    )
    _require(
        all(type(value[key]) is int and value[key] > 0 for key in sizes),
        "manifest_capacity",
    )
    _require(re.fullmatch(r"[0-9]+-[0-9]+", value["scan_run_id"]), "run_identity")
    return value


def _remote_review(selector, role):
    _require(HEX.fullmatch(selector), "manifest_selector")
    directory = Path.home() / ADJUDICATION_ROOT / selector
    with _private_input(directory / "request.json") as (stream, _info):
        data = stream.read(MANIFEST_BYTES + 1)
    request = _adjudication_request(data, selector)
    if role == "request":
        sys.stdout.buffer.write(data)
        return
    _require(role in ("receipt", "review"), "transfer_role")
    if role == "receipt":
        _remote_manifest(request["image_manifest_sha256"])
        path = (
            Path.home()
            / RECEIPT_ROOT
            / request["image_manifest_sha256"]
            / request["scan_run_id"]
            / "result.tar"
        )
        size, digest = request["scan_receipt_bytes"], request["scan_receipt_sha256"]
    else:
        path = directory / "review.tar"
        size, digest = request["review_tar_bytes"], request["review_tar_sha256"]
    with _private_input(path) as (stream, info):
        _require(info.st_size == size, "receipt_identity")
        _copy_exact(stream, sys.stdout.buffer, size, digest)
        _require(_descriptor_digest(stream.fileno(), size) == digest, "input_changed")


def _remote_store(selector, run, size, digest):
    _remote_manifest(selector)
    _require(re.fullmatch(r"[0-9]+-[0-9]+", run) is not None, "run_identity")
    _require(HEX.fullmatch(digest) and size > 0, "receipt_identity")
    parent = Path.home() / RECEIPT_ROOT / selector
    descriptor = _directory(parent, create=True)
    try:
        os.mkdir(run, 0o700, dir_fd=descriptor)
    finally:
        os.close(descriptor)
    target = parent / run / "result.tar"
    _require(shutil.disk_usage(target.parent).free >= size, "receipt_capacity")
    with _private_output(target) as stream:
        _copy_exact(sys.stdin.buffer, stream, size, digest)
    with _private_input(target) as (stream, info):
        _require(
            _descriptor_digest(stream.fileno(), info.st_size) == digest, "input_changed"
        )
    print(json.dumps({"sha256": digest, "bytes": size}, sort_keys=True))


def _clean_environment(*, policy=False):
    environment = {
        key: os.environ[key]
        for key in ("PATH", "HOME", "LANG", "TMPDIR")
        if key in os.environ
    }
    if policy:
        for key in ("CUSTOMER_DENYLIST", "INFRA_DENYLIST"):
            if key in os.environ:
                environment[key] = os.environ[key]
    return environment


def _ssh_arguments(root):
    host, user = (
        os.environ.get("DEV_VM_SSH_HOST", ""),
        os.environ.get("DEV_VM_SSH_USER", ""),
    )
    port = os.environ.get("DEV_VM_SSH_PORT") or "22"
    _require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]*", host), "ssh_host")
    _require(re.fullmatch(r"[a-z_][a-z0-9_-]*", user), "ssh_user")
    _require(port.isdecimal() and 1 <= int(port) <= 65535, "ssh_port")
    for variable, name in (
        ("DEV_VM_SSH_PRIVATE_KEY", "ssh-key"),
        ("DEV_VM_SSH_KNOWN_HOSTS", "known-hosts"),
    ):
        value = os.environ.get(variable, "")
        _require(bool(value.strip()), "ssh_configuration")
        _write(root / name, (value + "\n").encode())
    return [
        *SSH_OPTIONS,
        "-p",
        port,
        "-l",
        user,
        "-i",
        str(root / "ssh-key"),
        "-o",
        "UserKnownHostsFile=" + str(root / "known-hosts"),
        "--",
        host,
    ]


def _remote_command(ssh, *arguments):
    # Only reviewed local code executes remotely; image bytes are never source.
    source = Path(__file__).read_text()
    return [
        *ssh,
        shlex.join(["/usr/bin/python3", "-I", "-c", source, "remote", *arguments]),
    ]


def _transfer_failure(errors):
    """Classify failed transport without disclosing SSH output or remote paths."""
    errors.seek(0)
    diagnostic = errors.read(MANIFEST_BYTES)
    for marker, code in (
        (b"Permission denied (", "ssh_authentication_failed"),
        (b"Host key verification failed", "ssh_host_verification_failed"),
        (b"REMOTE HOST IDENTIFICATION HAS CHANGED", "ssh_host_verification_failed"),
        (b"Connection refused", "ssh_connection_failed"),
        (b"Connection timed out", "ssh_connection_failed"),
        (b"No route to host", "ssh_connection_failed"),
        (b"Could not resolve hostname", "ssh_connection_failed"),
    ):
        if marker in diagnostic:
            return _QualificationError(code)
    try:
        remote = json.loads(diagnostic)
    except ValueError:
        return _QualificationError("ssh_transfer_failed")
    if isinstance(remote, dict) and remote.get("status") == "failed":
        if remote.get("exception_class") == "FileNotFoundError":
            return _QualificationError("remote_export_missing")
        if remote.get("exception_class") == "PermissionError":
            return _QualificationError("remote_export_permissions")
        return _QualificationError("remote_interface_failed")
    return _QualificationError("ssh_transfer_failed")


def _fetch(ssh, selector, role, destination, *, size=None, digest=None):
    command = _remote_command(ssh, "fetch", selector, role)
    with _private_output(destination) as output, tempfile.TemporaryFile() as errors:
        with _child(
            command, stdout=subprocess.PIPE, stderr=errors, env=_clean_environment()
        ) as process:
            try:
                if role == "manifest":
                    payload = process.stdout.read(MANIFEST_BYTES + 1)
                    _require(len(payload) <= MANIFEST_BYTES, "manifest_size")
                    if process.wait() != 0:
                        raise _transfer_failure(errors)
                    _manifest(payload, selector)
                    output.write(payload)
                else:
                    try:
                        _copy_exact(process.stdout, output, size, digest)
                    except _QualificationError as error:
                        # These failures mean stdout reached EOF. An oversized
                        # sender may still be writing, so never wait for it here.
                        if error.code in ("transfer_truncated", "transfer_digest"):
                            if process.wait() != 0:
                                raise _transfer_failure(errors) from error
                        raise
            except BaseException:
                process.terminate()
                raise
            if process.wait() != 0:
                raise _transfer_failure(errors)


def _source_binding(scanner):
    result = subprocess.run(
        ["git", "-C", str(scanner), "rev-parse", "HEAD"],
        capture_output=True,
        env=_clean_environment(),
        check=False,
    )
    _require(
        result.returncode == 0 and result.stdout.strip().decode() == SCANNER_REVISION,
        "scanner_source_revision",
    )
    clean = subprocess.run(
        ["git", "-C", str(scanner), "status", "--porcelain", "--ignored"],
        capture_output=True,
        env=_clean_environment(),
        check=False,
    )
    _require(clean.returncode == 0 and not clean.stdout, "scanner_source_dirty")


def _capacity(root, manifest):
    required = manifest["archive_bytes"] + manifest["workspace_bytes"]
    value = {
        "archive_bytes": manifest["archive_bytes"],
        "workspace_bytes": manifest["workspace_bytes"],
        "required_free_bytes": required,
        "available_bytes": shutil.disk_usage(root).free,
    }
    _write(root / "capacity.json", _json_bytes(value))
    print(json.dumps(value, sort_keys=True), flush=True)
    _require(value["available_bytes"] >= required, "insufficient_capacity")


def _scanner_command(scanner, script, root, *arguments, **options):
    command = [sys.executable, str(scanner / "npa/scripts" / script), *arguments]
    for name, value in {
        "analysis_root": root,
        "trusted_root": scanner,
        **options,
    }.items():
        command.extend(["--" + name.replace("_", "-"), str(value)])
    return command


class _RunObserver:
    """Keep fixed progress metadata and cancellation separate from private output."""

    def __init__(self, root, journal):
        self.root = root
        self.journal = journal
        self.phase = "manifest-transfer"
        self.child = None
        self.signum = None
        self.force = False
        self.retaining = False

    def event(self, event, **details):
        value = {"phase": self.phase, "event": event, **details}
        record = {"time": datetime.now(timezone.utc).isoformat(), **value}
        self.journal.write(_json_bytes(record))
        self.journal.flush()
        os.fsync(self.journal.fileno())
        # Fixed phases and numeric exit/signal metadata are safe for Actions.
        print(json.dumps(value, sort_keys=True), flush=True)

    def cancel(self, signum, _frame):
        self.force = self.signum is not None
        self.signum = self.signum or signum
        # The trusted child's own cancellation protocol joins its helpers.
        # A repeated cancellation explicitly forces only the owned child.
        # Do not interrupt the final private receipt transfer with another signal.
        if not self.retaining and self.child is not None:
            self.stop_child()

    def stop_child(self):
        if self.force:
            self.child.kill()
        else:
            self.child.terminate()


@contextmanager
def _observe_run(root):
    (root / "diagnostics").mkdir(mode=0o700)
    with _private_output(root / "phases.jsonl") as journal:
        observer = _RunObserver(root, journal)
        token = _CURRENT_RUN.set(observer)
        previous = {
            number: signal.getsignal(number)
            for number in (signal.SIGTERM, signal.SIGINT)
        }
        try:
            for number in previous:
                signal.signal(number, observer.cancel)
            yield observer
        finally:
            for number, handler in previous.items():
                signal.signal(number, handler)
            _CURRENT_RUN.reset(token)


def _check_cancelled():
    observer = _CURRENT_RUN.get()
    if observer is not None and not observer.retaining:
        _require(observer.signum is None, "qualification_cancelled")


def _phase(name):
    _require(name in PHASES, "progress_phase")
    _check_cancelled()
    observer = _CURRENT_RUN.get()
    if observer is not None:
        observer.phase = name
        observer.event("started")


@contextmanager
def _child(command, **options):
    """Own a child handle across the signal-during-spawn window and always join it."""
    _check_cancelled()
    observer, process = _CURRENT_RUN.get(), None
    try:
        process = subprocess.Popen(command, start_new_session=True, **options)
        if observer is not None:
            observer.child = process
            if observer.signum is not None and not observer.retaining:
                observer.stop_child()
        yield process
    finally:
        if process is not None:
            if process.poll() is None:
                process.terminate()
            process.wait()
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    stream.close()
        if observer is not None:
            observer.child = None


def _execute(command, *, policy=False, phase=None):
    observer = _CURRENT_RUN.get()
    if phase is not None:
        _phase(phase)
    # A policy-bearing command may echo a malformed rule. Never retain its output.
    output_context = (
        _private_output(observer.root / "diagnostics" / (phase + ".log"))
        if observer is not None and phase is not None and not policy
        else tempfile.TemporaryFile()
    )
    with output_context as output:
        with _child(
            command,
            stdout=output,
            stderr=subprocess.STDOUT,
            env=_clean_environment(policy=policy),
        ) as process:
            returncode = process.wait()
    if observer is not None:
        observer.event("child-exited", exit_code=returncode)
    _check_cancelled()
    return returncode


def _check_policy(scanner):
    prepare = scanner / "npa/scripts/image_byte_scan/prepare.py"
    _require(
        _execute(
            [sys.executable, str(prepare), "check-policy", "--policy-mode", "ci-regex"],
            policy=True,
            phase="policy-check",
        )
        == 0,
        "policy_unavailable",
    )


def _prepare_scanner(scanner, root):
    _check_policy(scanner)
    commands = (
        _scanner_command(
            scanner,
            "image_byte_scan/go_helper/build.py",
            root,
            output_dir=root / "tools",
        ),
        _scanner_command(
            scanner,
            "image_byte_scan/prepare.py",
            root,
            "dependencies",
            output_dir=root / "native",
        ),
        _scanner_command(
            scanner,
            "image_byte_scan/real_helper_checks.py",
            root,
            tools_receipt=root / "tools/dependency-receipt.json",
            native_receipt=root / "native/dependencies.json",
            output_dir=root / "integration",
        ),
    )
    codes = (
        "helper_build_failed",
        "native_dependency_failed",
        "native_integration_failed",
    )
    phases = ("helper-build", "native-dependencies", "native-integration")
    for command, code, phase in zip(commands, codes, phases, strict=True):
        _require(_execute(command, phase=phase) == 0, code)


def _scan(scanner, root, manifest):
    _require(
        manifest["schema_version"] in (MANIFEST_SCHEMA, REGISTRY_MANIFEST_SCHEMA),
        "manifest_schema",
    )
    verifier = (
        "registry_manifest_verification.py"
        if manifest["schema_version"] == REGISTRY_MANIFEST_SCHEMA
        else "oci_verification.py"
    )
    graph = _scanner_command(
        scanner,
        "image_byte_scan/" + verifier,
        root,
        archive=root / "image.tar",
        expected_image_id=manifest["expected_image_id"],
        output_dir=root / "graph",
    )
    _require(
        _execute(graph, phase="graph-verification") == 0, "oci_verification_failed"
    )
    _authorize_scan(scanner, root, manifest)
    return _execute(
        _scanner_command(
            scanner,
            "scan_image_bytes.py",
            root,
            authorization=root / "authorization/authorization.json",
            output_dir=root / "scan",
        ),
        phase="image-scan",
    )


def _authorize_scan(scanner, root, manifest):
    command = _scanner_command(
        scanner,
        "image_byte_scan/prepare.py",
        root,
        "authorize",
        tools_receipt=root / "tools/dependency-receipt.json",
        native_receipt=root / "native/dependencies.json",
        archive=root / "image.tar",
        verification_report=root / "graph/verification.json",
        expected_image_id=manifest["expected_image_id"],
        output_dir=root / "authorization",
        policy_mode="ci-regex",
    )
    _require(
        _execute(command, policy=True, phase="scan-authorization") == 0,
        "scan_authorization_failed",
    )


def _read_scan_report(root):
    try:
        with _private_input(root / "scan/report.json") as (stream, _info):
            return json.load(stream)
    except FileNotFoundError as error:
        raise _QualificationError("report_missing") from error


def _policy_summary(report):
    policy = report.get("confidentiality_policy")
    _require(type(policy) is dict, "policy_receipt_invalid")
    _require(
        policy.get("schema") == "npa.image-byte-confidentiality-policy.v1",
        "policy_receipt_invalid",
    )
    digest = policy.get("policy_sha256")
    _require(type(digest) is str and HEX.fullmatch(digest), "policy_receipt_invalid")
    _require(policy.get("customer") == "configured", "policy_receipt_invalid")
    _require(
        policy.get("infra") in ("configured", "not_configured"),
        "policy_receipt_invalid",
    )
    return {key: policy[key] for key in ("policy_sha256", "customer", "infra")}


def _graph_scope_summary(report, manifest):
    if manifest["schema_version"] != REGISTRY_MANIFEST_SCHEMA:
        return {}
    graph = report.get("oci_graph", {})
    attestations = {
        "manifests_in_bound_graph": 0,
        "status": "not-present-in-original-manifest-graph",
        "external_referrers": "not-inspected",
    }
    _require(
        graph.get("identity_kind") == "original-registry-manifest"
        and graph.get("image_manifest_digest") == manifest["expected_image_id"]
        and graph.get("attestations") == attestations,
        "report_graph_scope",
    )
    return {"identity_kind": "original-registry-manifest", "attestations": attestations}


def _scan_summary(root, manifest, selector, returncode):
    report = _read_scan_report(root)
    policy = _policy_summary(report)
    _require(
        report.get("archive_sha256") == manifest["archive_sha256"]
        and report.get("expected_image_id") == manifest["expected_image_id"],
        "report_identity",
    )
    counts = {}
    for key in (
        "records",
        "scanned_bytes",
        "regular_files",
        "regular_bytes",
        "findings",
    ):
        value = report.get(key)
        _require(type(value) is int and value >= 0, "report_counts")
        counts[key] = value
    complete = report.get("complete") is True and report.get("helper_joined") is True
    passed = (
        complete
        and report.get("valid") is True
        and returncode == 0
        and not counts["findings"]
    )
    return {
        "status": "passed" if passed else "not-qualified",
        "complete": complete,
        "manifest_sha256": selector,
        "archive_sha256": manifest["archive_sha256"],
        "scanner_revision": SCANNER_REVISION,
        "confidentiality_policy": policy,
        **_graph_scope_summary(report, manifest),
        **counts,
    }


def _bundle(root):
    target = root / "result.tar"
    with _private_output(target) as output:
        with tarfile.open(fileobj=output, mode="w|") as archive:
            for name in (*RECEIPT_FILES, *_retained_input_names(root)):
                path = root / name
                if not path.exists():
                    continue
                with _private_input(path) as (stream, info):
                    member = tarfile.TarInfo(name)
                    member.size, member.mode = info.st_size, 0o600
                    archive.addfile(member, stream)
    with _private_input(target) as (stream, info):
        value = _descriptor_digest(stream.fileno(), info.st_size)
    return target, info.st_size, value


def _retained_input_names(root):
    path = root / "retained/retention.json"
    if not path.exists():
        return []
    with _private_input(path) as (stream, _info):
        receipt = json.load(stream, object_pairs_hook=_unique_object)
    _require(
        receipt["schema_version"] == "npa.image-byte-retained-inputs.v1"
        and receipt["policy_values_retained"] is False,
        "scan_input_retention_failed",
    )
    names = set(receipt["files"].values())
    for row in receipt["inputs"]:
        _require(
            row["role"] != "confidentiality"
            or (row["delivery"] == "secret" and "file" not in row),
            "scan_input_retention_failed",
        )
        if row["delivery"] == "retained":
            names.add(row["file"])
    _require(
        all(type(name) is str and HEX.fullmatch(name) for name in names),
        "scan_input_retention_failed",
    )
    for name in names:
        with _private_input(root / "retained" / name) as (stream, info):
            _require(
                _descriptor_digest(stream.fileno(), info.st_size) == name,
                "scan_input_retention_failed",
            )
    return ["retained/" + name for name in sorted(names)]


def _retain_scan_inputs(scanner, root, run):
    revision = subprocess.check_output(
        ["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"],
        text=True,
        env=_clean_environment(),
    ).strip()
    command = _scanner_command(
        scanner,
        "image_byte_scan/retained_inputs.py",
        root,
        authorization=root / "authorization/authorization.json",
        report=root / "scan/report.json",
        records=root / "scan/records.jsonl",
        output_dir=root / "retained",
        native_checks=root / "integration/native-checks.json",
        run_id=run,
        interface_revision=revision,
    )
    _require(
        _execute(command, phase="scan-input-retention") == 0,
        "scan_input_retention_failed",
    )


def _retain(ssh, root, selector, run):
    path, size, digest = _bundle(root)
    command = _remote_command(ssh, "store", selector, run, str(size), digest)
    with _private_input(path) as (stream, _info), tempfile.TemporaryFile() as errors:
        with _child(
            command,
            stdin=stream,
            stdout=subprocess.PIPE,
            stderr=errors,
            env=_clean_environment(),
        ) as process:
            stdout, _ = process.communicate()
            returncode = process.returncode
    _require(
        returncode == 0 and json.loads(stdout) == {"sha256": digest, "bytes": size},
        "private_receipt_not_retained",
    )
    return {"receipt_sha256": digest, "receipt_bytes": size}


def _qualification_steps(scanner, root, ssh, selector):
    _phase("manifest-transfer")
    _fetch(ssh, selector, "manifest", root / "manifest.json")
    manifest = _manifest((root / "manifest.json").read_bytes(), selector)
    _phase("capacity")
    _capacity(root, manifest)
    _phase("scanner-preparation")
    _prepare_scanner(scanner, root)
    _phase("capacity-after-preparation")
    _capacity_after_preparation(root, manifest)
    _phase("archive-transfer")
    _fetch(
        ssh,
        selector,
        "archive",
        root / "image.tar",
        size=manifest["archive_bytes"],
        digest=manifest["archive_sha256"],
    )
    _phase("image-scan")
    returncode = _scan(scanner, root, manifest)
    _phase("scan-report")
    return _scan_summary(root, manifest, selector, returncode)


def _cancelled_summary(summary, observer):
    if observer.signum is not None:
        summary.update(status="failed", complete=False, signal=observer.signum)
        summary.update(
            _failure(_QualificationError("qualification_cancelled"), observer.phase)
        )
        observer.event("cancelled", signal=observer.signum)


def _retain_summary(summary, ssh, root, selector, run):
    try:
        summary.update(_retain(ssh, root, selector, run))
    except (OSError, ValueError, KeyError, TypeError, _QualificationError) as error:
        failure = _failure(error, "receipt-retention")
        summary["receipt_retention_failure"] = failure
        if summary["status"] == "passed":
            summary.update(failure)
        summary.update(status="failed", complete=False)


def _qualify(scanner, root, ssh, selector, run):
    summary = {
        "status": "failed",
        "manifest_sha256": selector,
        "scanner_revision": SCANNER_REVISION,
    }
    with _observe_run(root) as observer:
        try:
            summary = _qualification_steps(scanner, root, ssh, selector)
            if summary.get("complete") is True:
                _retain_scan_inputs(scanner, root, run)
            _check_cancelled()
        except (OSError, ValueError, KeyError, TypeError, _QualificationError) as error:
            summary.update(
                _failure(error, observer.phase), status="failed", complete=False
            )
        _cancelled_summary(summary, observer)
        cancelled_before_retention = observer.signum
        observer.retaining = True
        _phase("receipt-retention")
        _write(root / "summary.json", _json_bytes(summary))
        _retain_summary(summary, ssh, root, selector, run)
        # A late signal cannot turn an interrupted run into an accepted result,
        # even when the receipt already contains a completed scanner report.
        if cancelled_before_retention is None:
            _cancelled_summary(summary, observer)
        print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if summary["status"] == "passed" else 1


def _capacity_after_preparation(root, manifest):
    _require(
        shutil.disk_usage(root).free
        >= manifest["archive_bytes"] + manifest["workspace_bytes"],
        "insufficient_capacity_after_preparation",
    )


def _run(args):
    _require(HEX.fullmatch(args.manifest_sha256), "manifest_selector")
    _require(re.fullmatch(r"[0-9]+-[0-9]+", args.run_id), "run_identity")
    _source_binding(args.scanner_root)
    with (
        tempfile.TemporaryDirectory(
            prefix="private-image-qualification-", dir=args.work_parent
        ) as directory,
        tempfile.TemporaryDirectory(
            prefix="private-image-credentials-", dir=args.work_parent
        ) as credentials,
    ):
        root = Path(directory)
        return _qualify(
            args.scanner_root,
            root,
            _ssh_arguments(Path(credentials)),
            args.manifest_sha256,
            args.run_id,
        )


def _arguments(argv):
    parser = _Parser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--manifest-sha256", required=True)
    run.add_argument("--run-id", required=True)
    run.add_argument("--scanner-root", type=Path, required=True)
    run.add_argument("--work-parent", type=Path, required=True)
    remote = commands.add_parser("remote")
    remote.add_argument("operation", choices=("fetch", "store", "review"))
    remote.add_argument("selector")
    remote.add_argument("values", nargs="+")
    return parser.parse_args(argv)


def _main(argv=None):
    os.umask(0o077)
    try:
        args = _arguments(argv)
        if args.command == "run":
            return _run(args)
        if args.operation == "review":
            _require(len(args.values) == 1, "transfer_arguments")
            _remote_review(args.selector, args.values[0])
        elif args.operation == "fetch":
            _require(len(args.values) == 1, "transfer_arguments")
            _remote_fetch(args.selector, args.values[0])
        else:
            _require(len(args.values) == 3, "transfer_arguments")
            _remote_store(
                args.selector, args.values[0], int(args.values[1]), args.values[2]
            )
        return 0
    except (OSError, ValueError, KeyError, TypeError, _QualificationError) as error:
        print(
            json.dumps(
                {"status": "failed", **_failure(error, "interface")}, sort_keys=True
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(_main())
