"""Build agent deployment bundles from an explicit Git source inventory."""

from __future__ import annotations

import hashlib
import io
import json
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
_MANIFEST = ".npa-source-inventory.json"
_SCHEMA = "npa.agent.source-inventory.v1"


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


def _source_inventory(repo_root: Path) -> tuple[list[str], dict | None]:
    # Deployed bundles carry their exact inventory because they intentionally
    # omit Git metadata. Such bundles may be deployed again, but may not silently
    # collect new local files or modified source bytes.
    if not (repo_root / ".git").exists():
        try:
            root_fd = os.open(repo_root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                with _open_source(root_fd, PurePosixPath(_MANIFEST)) as source:
                    manifest = json.load(source)
            finally:
                os.close(root_fd)
            if not isinstance(manifest, dict) or manifest.get("schema") != _SCHEMA:
                raise ValueError("Invalid source inventory schema")
            files = manifest["files"]
            if not isinstance(files, dict) or any(
                not isinstance(name, str)
                or not isinstance(record, dict)
                or set(record) != {"sha256", "bytes"}
                or not isinstance(record["sha256"], str)
                or len(record["sha256"]) != 64
                or any(char not in "0123456789abcdef" for char in record["sha256"])
                or type(record["bytes"]) is not int
                or record["bytes"] < 0
                for name, record in files.items()
            ):
                raise ValueError("Invalid source inventory records")
            return sorted(files), files
        except (OSError, ValueError, KeyError) as exc:
            raise ConfigError("Agent deployment requires a Git source inventory or verified source bundle") from exc
    try:
        result = subprocess.run(
            ["git", "--literal-pathspecs", "ls-files", "-z", "--cached", "--", *_ROOTS],
            cwd=repo_root,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ConfigError("Agent deployment requires a Git source inventory") from exc

    return sorted(set(os.fsdecode(path) for path in result.stdout.split(b"\0") if path)), None


class _HashingReader:
    def __init__(self, source):
        self.source = source
        self.digest = hashlib.sha256()
        self.bytes = 0

    def read(self, size):
        data = self.source.read(size)
        self.digest.update(data)
        self.bytes += len(data)
        return data


def create_agent_source_archive(repo_root: Path) -> str:
    """Package inventoried source bytes; never recursively collect local state.

    Stage new files in Git before deployment. A previously deployed bundle can
    be forwarded only if every inventoried file still matches its archived hash.
    """
    paths, expected = _source_inventory(repo_root)
    for name in paths:
        path = PurePosixPath(name)
        if (
            path.is_absolute()
            or path.as_posix() != name
            or ".." in path.parts
            or "\0" in name
            or "\\" in name
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
            with destination, tarfile.open(fileobj=destination, mode="w:gz", dereference=True) as archive:
                inventory = {}
                for name in paths:
                    with _open_source(root_fd, PurePosixPath(name)) as source:
                        info = archive.gettarinfo(fileobj=source, arcname=name)
                        info.uid = info.gid = 0
                        info.uname = info.gname = ""
                        info.mode = 0o755 if info.mode & 0o111 else 0o644
                        reader = _HashingReader(source)
                        archive.addfile(info, reader)
                        record = {"sha256": reader.digest.hexdigest(), "bytes": reader.bytes}
                        if expected is not None and record != expected[name]:
                            raise ConfigError("Deployed source differs from its inventory")
                        inventory[name] = record
                payload = json.dumps({"schema": _SCHEMA, "files": inventory}, sort_keys=True).encode()
                info = tarfile.TarInfo(_MANIFEST)
                info.size = len(payload)
                info.mode = 0o644
                archive.addfile(info, io.BytesIO(payload))
        finally:
            os.close(root_fd)
    except (OSError, tarfile.TarError, ConfigError) as exc:
        destination.close()
        Path(destination.name).unlink(missing_ok=True)
        raise ConfigError("Cannot safely package the agent source inventory") from exc
    return destination.name
