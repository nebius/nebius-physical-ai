"""Private local NCore staging; only the current user and root are trusted.

Defaults stay unexpanded until execution, so a workflow uses the pod user's
home, not the machine that rendered it. No source data is reused across runs.
"""

from __future__ import annotations

import os
import stat
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

DEFAULT_COLMAP_CACHE_DIR = Path("~/.cache/npa/ncore/colmap/cache")
DEFAULT_COLMAP_SCRATCH_DIR = Path("~/.cache/npa/ncore/colmap/scratch")


class PrivateStagingError(ValueError):
    """A local staging path does not meet the ownership/permissions contract."""


def _validate_ancestor(parent, uid):
    sticky_root = parent.st_uid == 0 and parent.st_mode & stat.S_ISVTX
    if parent.st_uid not in {0, uid} or (parent.st_mode & 0o022 and not sticky_root):
        raise PrivateStagingError("unsafe staging ancestor ownership or permissions")


def _create_private_tree(path, uid):
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    fd = os.open(path.anchor, flags)
    try:
        for part in path.parts[1:]:
            _validate_ancestor(os.fstat(fd), uid)
            try:
                os.mkdir(part, mode=0o700, dir_fd=fd)
            except FileExistsError:
                # Concurrent creators must validate the winner's actual inode.
                pass
            child_fd = os.open(part, flags, dir_fd=fd)
            os.close(fd)
            fd = child_fd
        info = os.fstat(fd)
        if info.st_uid != uid or stat.S_IMODE(info.st_mode) != 0o700:
            raise PrivateStagingError(
                "staging directory must be owned by the current user with mode 0700"
            )
    finally:
        os.close(fd)


def private_directory(root: Path) -> Path:
    """Create mode-0700 parents and validate paths without following symlinks.

    Descriptor traversal closes mkdir/open races. Ancestors must be owned by
    root or the user, with no other writers except in root-owned sticky
    directories. Existing permissions are never changed automatically.

    Args:
        root: Parent to create or validate, expanded for the executing user.
    Returns:
        Absolute path owned by the current user with mode 0700.
    Raises:
        PrivateStagingError: A path, owner, or permission is unsafe.
    """
    path = root.expanduser().absolute()
    if ".." in path.parts:
        raise PrivateStagingError("staging paths must not contain parent traversal")
    try:
        _create_private_tree(path, os.geteuid())
    except OSError:
        # Tracebacks must not expose local path names either.
        raise PrivateStagingError(
            "staging path must contain only accessible real directories"
        ) from None
    return path


@contextmanager
def private_staging_directory(root: Path, *, prefix: str) -> Iterator[Path]:
    """Allocate and clean up only this invocation's fresh private directory.

    Validated ancestors cannot be replaced by another user. Atomic mkdtemp
    allocation allows simultaneous invocations, including a shared cache and
    scratch parent, without stale bytes or cross-invocation cleanup.

    Args:
        root: Private staging parent to create or validate.
        prefix: Prefix for a securely allocated temporary generation.
    Yields:
        Path of the invocation's private generation.
    Raises:
        PrivateStagingError: The parent is unsafe.
        OSError: Temporary allocation or cleanup fails.
    """
    parent = private_directory(root)
    with tempfile.TemporaryDirectory(prefix=prefix, dir=parent) as directory:
        yield Path(directory)
