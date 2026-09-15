"""Execute isolated filesystem qualification actions without inspecting existing data."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import re
import stat
import sys
from contextlib import ExitStack, contextmanager
from typing import Any, Iterator

_PROBE_DIRECTORY = ".npa-storage-probes"
_TOKEN_PATTERN = re.compile(r"[0-9a-f]{32}\Z")
_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
_STATX_LINK_COUNT = 0x0004
_AT_SYMLINK_NOFOLLOW = 0x0100
_AT_STATX_FORCE_SYNC = 0x2000


class _StatxResult(ctypes.Structure):
    """Receive the stable Linux statx prefix within its complete 256-byte buffer."""

    _fields_ = [("mask", ctypes.c_uint32), ("block_size", ctypes.c_uint32),
                ("attributes", ctypes.c_uint64), ("link_count", ctypes.c_uint32),
                ("remaining_fields", ctypes.c_ubyte * 236)]


class StorageProbeError(ValueError):
    """Report a fixed failure category without exposing filesystem information.

    Args:
        category: Public-safe failure category.
    """

    def __init__(self, category: str):
        super().__init__(category)
        self.category = category


def _token(value: Any) -> str:
    if not isinstance(value, str) or not _TOKEN_PATTERN.fullmatch(value):
        raise StorageProbeError("invalid_probe_identity")
    return value


def _open_root(path: str, stack: ExitStack) -> int:
    if not isinstance(path, str) or not path.startswith("/"):
        raise StorageProbeError("invalid_probe_root")
    descriptor = os.open("/", _DIRECTORY_FLAGS)
    stack.callback(os.close, descriptor)
    for part in path.split("/"):
        if not part:
            continue
        if part in {".", ".."}:
            raise StorageProbeError("invalid_probe_root")
        descriptor = os.open(part, _DIRECTORY_FLAGS, dir_fd=descriptor)
        stack.callback(os.close, descriptor)
    return descriptor


def _open_child(parent: int, name: str, create: bool, stack: ExitStack) -> int:
    if create:
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent)
        except FileExistsError:
            pass
    descriptor = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent)
    stack.callback(os.close, descriptor)
    return descriptor


def _verify_marker(directory: int, run_id: str, create: bool) -> None:
    marker = ".owner-" + run_id
    flags = os.O_RDONLY | os.O_NOFOLLOW
    if create:
        try:
            descriptor = os.open(marker, flags | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=directory)
        except FileExistsError:
            descriptor = os.open(marker, flags, dir_fd=directory)
    else:
        descriptor = os.open(marker, flags, dir_fd=directory)
    with os.fdopen(descriptor, "rb") as stream:
        details = os.fstat(stream.fileno())
        if not stat.S_ISREG(details.st_mode) or details.st_size != 0:
            raise StorageProbeError("probe_ownership_mismatch")


@contextmanager
def _owned_directory(configuration: dict[str, Any], create: bool) -> Iterator[tuple[int, int, int]]:
    run_id = _token(configuration["run_id"])
    with ExitStack() as stack:
        root = _open_root(configuration["root_path"], stack)
        parent = _open_child(root, _PROBE_DIRECTORY, create, stack)
        directory = _open_child(parent, run_id, create, stack)
        _verify_marker(directory, run_id, create)
        yield root, parent, directory


def _regular_file(directory: int, filename: str, flags: int) -> int:
    descriptor = os.open(filename, flags | os.O_NOFOLLOW, 0o600, dir_fd=directory)
    details = os.fstat(descriptor)
    if not stat.S_ISREG(details.st_mode) or details.st_nlink != 1:
        os.close(descriptor)
        raise StorageProbeError("probe_file_identity_mismatch")
    return descriptor


def _digest_file(directory: int, filename: str) -> str:
    descriptor = _regular_file(directory, filename, os.O_RDONLY)
    with os.fdopen(descriptor, "rb") as stream:
        if os.fstat(stream.fileno()).st_size != 32:
            raise StorageProbeError("probe_payload_mismatch")
        return hashlib.sha256(stream.read(33)).hexdigest()


def _write_file(directory: int, filename: str) -> str:
    descriptor = _regular_file(directory, filename, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    payload = os.urandom(32)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    checksum = hashlib.sha256(payload).hexdigest()
    if _digest_file(directory, filename) != checksum:
        raise StorageProbeError("probe_payload_mismatch")
    return checksum


def _unlink_file(directory: int, filename: str) -> bool:
    if _absent(directory, filename):
        return False
    try:
        details = os.stat(filename, dir_fd=directory, follow_symlinks=False)
    except FileNotFoundError:
        return False
    if not stat.S_ISREG(details.st_mode) or details.st_nlink != 1:
        raise StorageProbeError("probe_file_identity_mismatch")
    os.unlink(filename, dir_fd=directory)
    return True


def _synchronized_link_count(directory: int, filename: str) -> int | None:
    library = ctypes.CDLL(None, use_errno=True)
    query = getattr(library, "statx", None)
    if query is None:
        raise StorageProbeError("synchronized_absence_unsupported")
    query.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_uint,
                      ctypes.POINTER(_StatxResult)]
    query.restype = ctypes.c_int
    details = _StatxResult()
    flags = _AT_STATX_FORCE_SYNC | _AT_SYMLINK_NOFOLLOW
    ctypes.set_errno(0)
    result = query(directory, os.fsencode(filename), flags, _STATX_LINK_COUNT,
                   ctypes.byref(details))
    if result == -1:
        return _absence_query_error(ctypes.get_errno())
    if result != 0 or details.mask & _STATX_LINK_COUNT != _STATX_LINK_COUNT:
        raise StorageProbeError("synchronized_absence_unverified")
    return details.link_count


def _absence_query_error(error_number: int) -> None:
    if error_number == errno.ENOENT:
        return None
    if error_number in {errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP}:
        raise StorageProbeError("synchronized_absence_unsupported")
    if error_number <= 0:
        raise StorageProbeError("synchronized_absence_unverified")
    raise OSError(error_number, os.strerror(error_number))


def _absent(directory: int, filename: str) -> bool:
    # FUSE can retain positive dentries after another worker deletes the owned path.
    # Forced server attributes distinguish an unlinked inode from a persistent remnant.
    link_count = _synchronized_link_count(directory, filename)
    if link_count is None:
        return True
    return link_count == 0


def _decode_mount_field(field: str) -> str:
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match.group(1), 8)), field)


def _mount_record(configuration: dict[str, Any]) -> dict[str, str]:
    records = []
    with open(configuration["mountinfo_path"], encoding="utf-8") as stream:
        for line in stream:
            before, separator, after = line.strip().partition(" - ")
            if not separator:
                continue
            fields, filesystem = before.split(), after.split()
            if len(fields) < 6 or len(filesystem) < 3:
                continue
            if _decode_mount_field(fields[4]) == configuration["mount_path"]:
                records.append({"type": filesystem[0], "source": _decode_mount_field(filesystem[1]),
                                "options": fields[5], "super_options": filesystem[2],
                                "device": fields[2], "root": _decode_mount_field(fields[3])})
    if len(records) != 1:
        raise StorageProbeError("missing_or_ambiguous_mount")
    record = records[0]
    if record["type"] != "virtiofs":
        raise StorageProbeError("wrong_filesystem_type")
    if record["source"] != configuration["mount_tag"]:
        raise StorageProbeError("wrong_mount_source")
    if "rw" not in record["options"].split(",") or "ro" in record["super_options"].split(","):
        raise StorageProbeError("mount_read_only")
    return record


def _verify_persistence(configuration: dict[str, Any]) -> None:
    matches = []
    with open(configuration["fstab_path"], encoding="utf-8") as stream:
        for line in stream:
            fields = line.split("#", 1)[0].split()
            if len(fields) < 4 or _decode_mount_field(fields[1]) != configuration["mount_path"]:
                continue
            matches.append(fields)
    if len(matches) != 1:
        raise StorageProbeError("missing_or_ambiguous_persistence")
    fields = matches[0]
    if _decode_mount_field(fields[0]) != configuration["mount_tag"] or fields[2] != "virtiofs":
        raise StorageProbeError("persistence_identity_mismatch")
    if "nofail" not in fields[3].split(",") or "ro" in fields[3].split(","):
        raise StorageProbeError("unsafe_mount_persistence")


def _verify_mount_device(configuration: dict[str, Any], record: dict[str, str]) -> None:
    with ExitStack() as stack:
        descriptor = _open_root(configuration["root_path"], stack)
        device = os.fstat(descriptor).st_dev
    if record["device"] != f"{os.major(device)}:{os.minor(device)}":
        raise StorageProbeError("stale_mount_binding")


def _capacity(configuration: dict[str, Any]) -> tuple[int, int, int]:
    requested_gibibytes = configuration["requested_gibibytes"]
    if type(requested_gibibytes) is not int or requested_gibibytes <= 0:
        raise StorageProbeError("invalid_requested_capacity")
    with ExitStack() as stack:
        descriptor = _open_root(configuration["root_path"], stack)
        details = os.fstatvfs(descriptor)
    if details.f_frsize <= 0 or details.f_blocks <= 0:
        raise StorageProbeError("unknown_capacity")
    reported_bytes = details.f_frsize * details.f_blocks
    requested_bytes = requested_gibibytes * 1024**3
    # The filesystem can round a final partial block, but cannot substitute GB for GiB.
    if not requested_bytes <= reported_bytes < requested_bytes + details.f_frsize:
        raise StorageProbeError("capacity_mismatch")
    return reported_bytes, requested_bytes, details.f_frsize


def _host_state(configuration: dict[str, Any]) -> dict[str, Any]:
    record = _mount_record(configuration)
    if record["root"] != "/":
        raise StorageProbeError("wrong_host_mount_root")
    _verify_mount_device(configuration, record)
    _verify_persistence(configuration)
    reported_bytes, requested_bytes, fragment_size = _capacity(configuration)
    return {"capacity_bytes": reported_bytes, "requested_bytes": requested_bytes,
            "filesystem_type": "virtiofs", "source_matches": True, "nofail": True,
            "read_write": True, "fragment_size": fragment_size}


def _host(configuration: dict[str, Any]) -> dict[str, Any]:
    state = _host_state(configuration)
    filename = "host-" + _token(configuration["node_token"])
    with _owned_directory(configuration, create=True) as (_, _, directory):
        try:
            checksum = _write_file(directory, filename)
        finally:
            _unlink_file(directory, filename)
        deleted = _absent(directory, filename)
    if not deleted:
        raise StorageProbeError("host_probe_cleanup_failed")
    return {**state, "checksum": checksum, "probe_deleted": deleted}


def _backing_path(configuration: dict[str, Any]) -> str:
    mount = {**configuration, "mount_path": configuration["root_path"],
             "mountinfo_path": configuration.get("self_mountinfo_path", "/proc/self/mountinfo")}
    record = _mount_record(mount)
    _verify_mount_device(mount, record)
    return _validate_backing_path(record["root"].removeprefix("/"))


def _validate_backing_path(path: Any) -> str:
    if not isinstance(path, str):
        raise StorageProbeError("invalid_backing_path")
    parts = path.split("/")
    if len(parts) != 2 or parts[0] != "csi-mounted-fs-path-data":
        raise StorageProbeError("invalid_backing_path")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", parts[1]):
        raise StorageProbeError("invalid_backing_path")
    return path


def _write(configuration: dict[str, Any]) -> dict[str, Any]:
    backing_path = _backing_path(configuration)
    filename = "payload-" + _token(configuration["node_token"])
    with _owned_directory(configuration, create=True) as (_, _, directory):
        checksum = _write_file(directory, filename)
    return {"checksum": checksum, "written": True, "backing_relative_path": backing_path}


def _expected_checksums(configuration: dict[str, Any]) -> dict[str, str]:
    expected = configuration["expected_checksums"]
    if not isinstance(expected, dict) or not expected:
        raise StorageProbeError("missing_cross_node_evidence")
    for node_token, checksum in expected.items():
        _token(node_token)
        if not isinstance(checksum, str) or not _DIGEST_PATTERN.fullmatch(checksum):
            raise StorageProbeError("invalid_expected_checksum")
    node_tokens = configuration.get("node_tokens")
    if node_tokens is not None and set(expected) != {_token(value) for value in node_tokens}:
        raise StorageProbeError("missing_cross_node_evidence")
    return expected


def _read(configuration: dict[str, Any]) -> dict[str, Any]:
    backing_path = _backing_path(configuration)
    expected = _expected_checksums(configuration)
    with _owned_directory(configuration, create=False) as (_, _, directory):
        for node_token, checksum in expected.items():
            if _digest_file(directory, "payload-" + node_token) != checksum:
                raise StorageProbeError("cross_node_checksum_mismatch")
    return {"verified_checksums": expected, "read_count": len(expected),
            "backing_relative_path": backing_path}


def _known_filenames(configuration: dict[str, Any]) -> list[str]:
    node_tokens = configuration.get("node_tokens", list(configuration.get("expected_checksums", {})))
    if configuration.get("node_token"):
        node_tokens = [*node_tokens, configuration["node_token"]]
    if not isinstance(node_tokens, list) or not node_tokens:
        raise StorageProbeError("missing_cleanup_inventory")
    filenames = []
    for node_token in sorted({_token(value) for value in node_tokens}):
        filenames.extend(["host-" + node_token, "payload-" + node_token])
    return filenames


def _remove_parent(root: int) -> bool:
    try:
        os.rmdir(_PROBE_DIRECTORY, dir_fd=root)
    except FileNotFoundError:
        return True
    except OSError as error:
        if error.errno == errno.ENOTEMPTY:
            return False
        raise
    return True


def _remove_run_directory(run_id: str, parent: int, directory: int) -> None:
    _unlink_file(directory, ".owner-" + run_id)
    try:
        os.rmdir(run_id, dir_fd=parent)
    except OSError:
        # Preserve ownership when an unrecognized entry prevents exact cleanup.
        _verify_marker(directory, run_id, create=True)
        raise


def _verify_probe_mount(configuration: dict[str, Any]) -> str | None:
    if configuration["root_path"] == "/data":
        return _backing_path(configuration)
    _host_state(configuration)
    return None


def _cleanup(configuration: dict[str, Any]) -> dict[str, Any]:
    backing_path = _verify_probe_mount(configuration)
    filenames = _known_filenames(configuration)
    removed_files = 0
    try:
        with _owned_directory(configuration, create=False) as (root, parent, directory):
            for filename in filenames:
                removed_files += _unlink_file(directory, filename)
            if not all(_absent(directory, filename) for filename in filenames):
                raise StorageProbeError("probe_cleanup_failed")
            _remove_run_directory(configuration["run_id"], parent, directory)
            _remove_parent(root)
    except FileNotFoundError:
        # An absent parent or owned directory is the expected result of another node's cleanup.
        pass
    result = _audit(configuration)
    if backing_path:
        result["backing_relative_path"] = backing_path
    return {**result, "removed_files": removed_files, "absent_files": len(filenames)}


def _audit(configuration: dict[str, Any]) -> dict[str, Any]:
    _verify_probe_mount(configuration)
    run_id = _token(configuration["run_id"])
    with ExitStack() as stack:
        root = _open_root(configuration["root_path"], stack)
        if _absent(root, _PROBE_DIRECTORY):
            return {"run_directory_absent": True, "base_directory_absent": True}
        parent = _open_child(root, _PROBE_DIRECTORY, False, stack)
        if not _absent(parent, run_id):
            raise StorageProbeError("probe_cleanup_failed")
    return {"run_directory_absent": True, "base_directory_absent": False}


def _audit_backing(configuration: dict[str, Any]) -> dict[str, Any]:
    _host_state(configuration)
    backing_path = _validate_backing_path(configuration["backing_relative_path"])
    parent_name, leaf = backing_path.split("/")
    with ExitStack() as stack:
        root = _open_root(configuration["root_path"], stack)
        if _absent(root, parent_name):
            return {"backing_directory_absent": True}
        parent = _open_child(root, parent_name, create=False, stack=stack)
        if not _absent(parent, leaf):
            raise StorageProbeError("backing_cleanup_failed")
    return {"backing_directory_absent": True}


def execute_probe(configuration: dict[str, Any]) -> dict[str, Any]:
    """Run one owned probe action and return payload-free evidence.

    Args:
        configuration: Action, ownership tokens, filesystem paths, and expectations.

    Returns:
        Structured checksums, capacity, or independently checked cleanup evidence.

    Raises:
        StorageProbeError: Ownership or verification evidence does not match.
        OSError: A filesystem or probe operation fails.
        KeyError: Required configuration is absent.
    """
    actions = {"host": _host, "write": _write, "read": _read, "cleanup": _cleanup,
               "audit": _audit, "audit_backing": _audit_backing}
    action = configuration["action"]
    if action not in actions:
        raise StorageProbeError("invalid_probe_action")
    evidence = actions[action](configuration)
    return {"passed": True, "action": action, **evidence}


def main() -> int:
    """Read a JSON action argument and emit exactly one sanitized JSON result.

    Returns:
        Zero when qualification passes, or one when it fails.
    """
    try:
        result = execute_probe(json.loads(sys.argv[1]))
    except StorageProbeError as error:
        print(json.dumps({"passed": False, "category": error.category}))
        return 1
    except (OSError, ValueError, TypeError, KeyError, IndexError):
        print(json.dumps({"passed": False, "category": "probe_operation_failed"}))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
