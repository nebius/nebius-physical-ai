"""Serialize local Fleet recovery operations for one exact target."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import stat


class FleetOperationLockError(RuntimeError):
    """A Fleet target cannot be safely locked for a local recovery operation."""


@contextmanager
def target_operation_lock(install_dir: Path) -> Iterator[None]:
    """Hold an owner-private, nonblocking recovery lock for one target."""
    directory = _private_target_directory(install_dir)
    descriptor = _open_lock(directory / ".npa-fleet-recovery.lock")
    try:
        _claim_lock(descriptor)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _private_target_directory(install_dir: Path) -> Path:
    target = install_dir.expanduser()
    if target.is_symlink():
        raise FleetOperationLockError("Fleet target installation cannot be a link")
    directory = target.resolve()
    try:
        metadata = directory.stat()
    except OSError as exc:
        raise FleetOperationLockError(
            "Fleet target installation is unavailable"
        ) from exc
    if (
        not directory.is_dir()
        or metadata.st_uid != os.getuid()
        or metadata.st_mode & 0o077
    ):
        raise FleetOperationLockError("Fleet target installation is not owner-private")
    return directory


def _open_lock(path: Path) -> int:
    flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise FleetOperationLockError("Fleet recovery lock is unavailable") from exc
    metadata = os.fstat(descriptor)
    if _unsafe_lock(metadata):
        os.close(descriptor)
        raise FleetOperationLockError("Fleet recovery lock is not owner-private")
    return descriptor


def _unsafe_lock(metadata: os.stat_result) -> bool:
    return (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or metadata.st_nlink != 1
        or bool(metadata.st_mode & 0o077)
    )


def _claim_lock(descriptor: int) -> None:
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise FleetOperationLockError(
            "another Fleet recovery operation owns this target"
        ) from exc
