"""Verify OptiX weights and stage a missing runtime payload on the container overlay."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import shutil
import stat
from typing import Any
import uuid

from .errors import IsaacArenaError


_WEIGHTS_PATH = Path("/usr/share/nvidia/nvoptix.bin")
_MOUNTINFO = Path("/proc/self/mountinfo")


def _file_identity(item: os.stat_result) -> tuple[int, ...]:
    return (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns)


def _payload_fingerprint(path: Path) -> dict[str, Any] | None:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise IsaacArenaError("OptiX weights must be a readable regular file") from exc
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode) or before.st_size == 0:
        os.close(descriptor)
        raise IsaacArenaError("OptiX weights must be a nonempty regular file")
    with os.fdopen(descriptor, "rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        after = os.fstat(stream.fileno())
    if _file_identity(before) != _file_identity(after) or _file_identity(
        path.stat(follow_symlinks=False)
    ) != _file_identity(after):
        raise IsaacArenaError("OptiX weights changed during verification")
    return {"sha256": digest.hexdigest(), "bytes": before.st_size}


def _payload_evidence(fingerprint: dict, placement: str) -> dict[str, Any]:
    return {
        **fingerprint,
        "required_path": str(_WEIGHTS_PATH),
        "placement": placement,
        "installed_in_container": placement == "container_overlay",
        "installed_on_node": False,
        "baked": False,
        "published": False,
    }


def _native_optix_weights() -> dict[str, Any] | None:
    fingerprint = _payload_fingerprint(_WEIGHTS_PATH)
    if fingerprint is None:
        return None
    return _payload_evidence(fingerprint, "native_runtime")


def _mounts() -> list[tuple[str, Path, str, str]]:
    try:
        records = []
        for line in _MOUNTINFO.read_text(encoding="utf-8").splitlines():
            left, right = line.split(" - ", 1)
            fields = left.split()
            mountpoint = re.sub(
                r"\\(040|011|012|134)",
                lambda match: chr(int(match.group(1), 8)),
                fields[4],
            )
            records.append((fields[0], Path(mountpoint), right.split()[0], fields[3]))
        return records
    except (OSError, ValueError, IndexError) as exc:
        raise IsaacArenaError(
            "OptiX placement requires readable mount ownership"
        ) from exc


def _container_destination() -> Path:
    parent = _WEIGHTS_PATH.parent
    if parent.resolve() != parent or not parent.is_dir():
        raise IsaacArenaError("OptiX placement requires its image-owned directory")
    ownership = parent.stat()
    if ownership.st_uid != os.geteuid() or ownership.st_mode & 0o022:
        raise IsaacArenaError("OptiX placement requires a private writable directory")
    mounts = _mounts()
    roots = [record for record in mounts if record[1] == Path("/")]
    covering = [
        record
        for record in mounts
        if record[1] == _WEIGHTS_PATH or record[1] in _WEIGHTS_PATH.parents
    ]
    if len(roots) != 1 or roots[0][2:] != ("overlay", "/") or not covering:
        raise IsaacArenaError("OptiX placement requires a private container overlay")
    effective = max(covering, key=lambda record: len(record[1].parts))
    if effective != roots[0]:
        raise IsaacArenaError("OptiX placement refuses a mounted destination")
    return parent


def _copy_verified_payload(source: Path, descriptor: int, fingerprint: dict) -> None:
    source_descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(source_descriptor, "rb") as original:
        with os.fdopen(descriptor, "wb", closefd=False) as copy:
            shutil.copyfileobj(original, copy)
            copy.flush()
            os.fsync(descriptor)
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    for block in iter(lambda: os.read(descriptor, 1024 * 1024), b""):
        digest.update(block)
    if (
        os.fstat(descriptor).st_size != fingerprint["bytes"]
        or digest.hexdigest() != fingerprint["sha256"]
    ):
        raise IsaacArenaError("OptiX weights copy failed hash verification")
    os.fchmod(descriptor, 0o444)


def _assert_destination(directory: int, parent: Path) -> None:
    if parent.stat() != os.fstat(directory) or _container_destination() != parent:
        raise IsaacArenaError("OptiX destination changed during placement")


def _link_payload(temporary: str, directory: int, fingerprint: dict) -> None:
    try:
        os.link(
            temporary,
            _WEIGHTS_PATH.name,
            src_dir_fd=directory,
            dst_dir_fd=directory,
            follow_symlinks=False,
        )
    except FileExistsError:
        # Concurrent evaluations can reuse identical complete bytes, but
        # neither may replace a provider-mounted or different payload.
        pass
    if _payload_fingerprint(_WEIGHTS_PATH) != fingerprint:
        raise IsaacArenaError(
            "OptiX destination differs from the matching driver package"
        )


def _publish_payload(source: Path, parent: Path, fingerprint: dict) -> None:
    directory = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    temporary = f".npa-optix-{uuid.uuid4().hex}"
    try:
        _assert_destination(directory, parent)
        descriptor = os.open(
            temporary, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600, dir_fd=directory
        )
        try:
            _copy_verified_payload(source, descriptor, fingerprint)
        finally:
            os.close(descriptor)
        _assert_destination(directory, parent)
        _link_payload(temporary, directory, fingerprint)
    finally:
        try:
            os.unlink(temporary, dir_fd=directory)
        except FileNotFoundError:
            pass
        os.close(directory)


def _prepare_optix_weights(extracted: Path) -> dict[str, Any]:
    source = extracted / "usr/share/nvidia/nvoptix.bin"
    fingerprint = _payload_fingerprint(source)
    if fingerprint is None:
        raise IsaacArenaError("matching NVIDIA package is missing OptiX weights")
    existing = _payload_fingerprint(_WEIGHTS_PATH)
    if existing is not None:
        if existing != fingerprint:
            raise IsaacArenaError(
                "OptiX destination differs from the matching driver package"
            )
        return _payload_evidence(existing, "native_runtime")
    parent = _container_destination()
    try:
        _publish_payload(source, parent, fingerprint)
    except OSError as exc:
        raise IsaacArenaError("OptiX container-only weights placement failed") from exc
    return _payload_evidence(fingerprint, "container_overlay")
