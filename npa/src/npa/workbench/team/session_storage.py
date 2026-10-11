"""Read and atomically replace private user-owned connection and retry records."""

from __future__ import annotations

import json
import os
import re
import stat
import uuid
from contextlib import contextmanager
from pathlib import Path

from .errors import TeamError


def _validate_name(name):
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,159}", name):
        raise TeamError("invalid private record name")


@contextmanager
def _directory(root, *, create=False):
    root = Path(root).expanduser()
    if create:
        root.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        root.mkdir(mode=0o700, exist_ok=True)
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        metadata = os.fstat(descriptor)
        if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) != 0o700:
            raise TeamError("session directory must be owned by you and mode 0700")
        yield descriptor
    finally:
        os.close(descriptor)


def _read_record(directory, name):
    descriptor = os.open(
        name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory
    )
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_uid != os.getuid()
            or metadata.st_nlink != 1
        ):
            raise TeamError("session record must be your private mode-0600 file")
        with os.fdopen(descriptor) as stream:
            descriptor = None
            return json.load(stream)
    finally:
        if descriptor is not None:
            os.close(descriptor)


@contextmanager
def _lock_descriptor(directory, name):
    descriptor = os.open(
        name,
        os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK,
        0o600,
        dir_fd=directory,
    )
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_uid != os.getuid()
            or metadata.st_nlink != 1
        ):
            raise TeamError("session lock must be your private mode-0600 file")
        yield descriptor
    finally:
        os.close(descriptor)


@contextmanager
def private_record_lock(root: Path, name: str):
    """Serialize private record mutations across processes and threads.

    Args:
        root, name: Private directory and safe lock basename.
    Returns:
        A context manager holding the named exclusive advisory lock.
    Raises:
        TeamError: The directory, lock, or lock acquisition is unsafe.
    """
    import fcntl

    _validate_name(name)
    try:
        with _directory(root, create=True) as directory:
            with _lock_descriptor(directory, name) as descriptor:
                fcntl.flock(descriptor, fcntl.LOCK_EX)
                yield
    except OSError as exc:
        raise TeamError("private session lock could not be acquired safely") from exc


def read_private_json(root: Path, name: str):
    """Read one private regular JSON record without following its symlink.

    Args:
        root, name: Private directory and safe record basename.
    Returns:
        Decoded JSON document.
    Raises:
        FileNotFoundError: Directory or record does not exist.
        TeamError: Ownership, permissions, JSON, or file access is unsafe.
    """
    _validate_name(name)
    try:
        with _directory(root) as directory:
            return _read_record(directory, name)
    except FileNotFoundError:
        raise
    except (OSError, ValueError) as exc:
        raise TeamError("private session record could not be read safely") from exc


def _replace_record(directory, name, document):
    temporary = "pending-" + uuid.uuid4().hex + ".json"
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
        dir_fd=directory,
    )
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(document, stream)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, name, src_dir_fd=directory, dst_dir_fd=directory)
        os.fsync(directory)
    finally:
        try:
            os.unlink(temporary, dir_fd=directory)
        except FileNotFoundError:
            pass


def write_private_json(root: Path, name: str, document):
    """Atomically replace a JSON record inside a private user-owned directory.

    Args:
        root, name: Private directory and safe record basename.
        document: JSON-compatible content, potentially containing credentials.
    Returns:
        None.
    Raises:
        TeamError: Directory, serialization, or atomic write is unsafe.
    """
    _validate_name(name)
    try:
        with _directory(root, create=True) as directory:
            _replace_record(directory, name, document)
    except (OSError, ValueError, TypeError) as exc:
        raise TeamError("private session record could not be saved safely") from exc


def remove_private_file(root: Path, name: str) -> bool:
    """Remove an exact private record without following its symlink.

    Args:
        root, name: Private directory and safe record basename.
    Returns:
        Whether the named record existed.
    Raises:
        TeamError: Directory access or removal fails.
    """
    _validate_name(name)
    try:
        with _directory(root) as directory:
            os.unlink(name, dir_fd=directory)
            os.fsync(directory)
        return True
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise TeamError("private session record could not be removed safely") from exc
