"""Build agent deployment bundles from an explicit Git source inventory."""

from __future__ import annotations

import os
import stat
import subprocess
import tarfile
import tempfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

from npa.clients.config import ConfigError

_ROOTS = ("npa", "deploy/cluster", "workflows", "docs", "skills")
_REQUIRED_ROOTS = _ROOTS[:3]


@contextmanager
def _open_source(root_fd: int, relative: PurePosixPath):
    """Open an indexed regular file without following any symlink components."""
    parent_fd = os.dup(root_fd)
    try:
        for component in relative.parts[:-1]:
            next_fd = os.open(
                component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd
            )
            os.close(parent_fd)
            parent_fd = next_fd
        fd = os.open(
            relative.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent_fd
        )
        with os.fdopen(fd, "rb") as source:
            if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
                raise ConfigError("Agent source inventory contains a non-regular file")
            yield source
    finally:
        os.close(parent_fd)


def create_agent_source_archive(repo_root: Path) -> str:
    """Package tracked working-tree bytes; never recursively collect local state.

    New source files must be staged before deployment. A checkout with missing
    indexed files or symlinks fails closed, rather than producing a partial or
    unexpectedly expanded deployment bundle.
    """
    try:
        result = subprocess.run(
            ["git", "--literal-pathspecs", "ls-files", "-z", "--cached", "--", *_ROOTS],
            cwd=repo_root,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ConfigError("Agent deployment requires a Git source inventory") from exc

    paths = sorted(set(os.fsdecode(path) for path in result.stdout.split(b"\0") if path))
    for name in paths:
        path = PurePosixPath(name)
        if (
            path.is_absolute()
            or path.as_posix() != name
            or ".." in path.parts
            or not any(name.startswith(root + "/") for root in _ROOTS)
        ):
            raise ConfigError("Invalid path in agent source inventory")
    for root in _REQUIRED_ROOTS:
        if not any(name.startswith(root + "/") for name in paths):
            raise ConfigError(f"Required agent source inventory is missing: {root}")

    destination = tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False)
    try:
        root_fd = os.open(repo_root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            with destination, tarfile.open(fileobj=destination, mode="w:gz") as archive:
                for name in paths:
                    with _open_source(root_fd, PurePosixPath(name)) as source:
                        info = archive.gettarinfo(fileobj=source, arcname=name)
                        info.uid = info.gid = 0
                        info.uname = info.gname = ""
                        info.mode = 0o755 if info.mode & 0o111 else 0o644
                        archive.addfile(info, source)
        finally:
            os.close(root_fd)
    except (OSError, tarfile.TarError, ConfigError) as exc:
        destination.close()
        Path(destination.name).unlink(missing_ok=True)
        raise ConfigError("Cannot safely package the agent source inventory") from exc
    return destination.name
