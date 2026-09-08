"""Deliver exactly two locked Debian libraries before NCore's real APT bootstrap.

The unprivileged phase caches complete public artifacts. The isolated root phase
accepts only those artifacts on stdin and installs four fixed paths; it never
reads a caller's cache, environment configuration, or Python environment.
"""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import pwd
import re
import secrets
import stat
import subprocess
import sys
import tarfile


LOCK_SHA256 = "8d0b0eec211844d945929d11adb5e2127a5075587a8ecda3bbb215bb2b231172"
SCRIPT_DIRECTORY = Path("/opt/ncore/native")
LOCK_PATH = SCRIPT_DIRECTORY / "native-bootstrap-lock.json"
PYTHON = "/usr/local/bin/python3.12"
LIBRARY_DIRECTORY = Path("/usr/lib/x86_64-linux-gnu")
INSTALL_STATE = Path("/run/npa-ncore-native")
ROOT_UID = 0
ROOT_GID = 0
SELECTION = {
    "libgnutls30": ("libgnutls.so.30.34.3", "libgnutls.so.30"),
    "libssh2-1": ("libssh2.so.1.0.1", "libssh2.so.1"),
}
DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
READ_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _check_directory(info, owner, private=False):
    mode = stat.S_IMODE(info.st_mode)
    sticky_root = info.st_uid == 0 and mode == 0o1777 and owner != 0
    if not stat.S_ISDIR(info.st_mode) or info.st_uid not in (0, owner):
        raise ValueError("unsafe native directory ownership")
    if mode & 0o022 and not sticky_root:
        raise ValueError("writable native directory ancestor")
    if private and (info.st_uid != owner or mode != 0o700):
        raise ValueError("native cache/state directory must be private (0700)")
    if owner == 0 and info.st_gid != 0:
        raise ValueError("native root directory group changed")


@contextmanager
def _directory(path, owner, *, private=False, create=False):
    if not path.is_absolute() or ".." in path.parts or path == Path("/"):
        raise ValueError("native directory must be an absolute non-root path")
    descriptor = os.open("/", DIRECTORY_FLAGS)
    try:
        _check_directory(os.fstat(descriptor), owner)
        for component in path.parts[1:]:
            if create:
                try:
                    os.mkdir(component, 0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
            child = os.open(component, DIRECTORY_FLAGS, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
            _check_directory(os.fstat(descriptor), owner)
        _check_directory(os.fstat(descriptor), owner, private)
        yield descriptor
    finally:
        os.close(descriptor)


def _check_file(info, owner, mode):
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != owner
        or (owner == ROOT_UID and info.st_gid != ROOT_GID)
        or info.st_nlink != 1
        or stat.S_IMODE(info.st_mode) != mode
    ):
        raise ValueError("unsafe native file ownership, type, links or mode")


def _identity(info):
    # Reading may update atime; identity and content metadata must stay stable.
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_uid,
        info.st_gid,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _read_file(directory, name, owner, mode, size):
    descriptor = os.open(name, READ_FLAGS, dir_fd=directory)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        _check_file(before, owner, mode)
        raw = stream.read(size + 1)
        after = os.fstat(stream.fileno())
        current = os.stat(name, dir_fd=directory, follow_symlinks=False)
        if _identity(before) != _identity(after) or _identity(after) != _identity(
            current
        ):
            raise ValueError("native file changed during read")
    return raw


def _read_lock():
    # Root checks the actual fixed entrypoint and its parent directories too.
    with _directory(LOCK_PATH.parent, ROOT_UID) as directory:
        descriptor = os.open(LOCK_PATH.name, READ_FLAGS, dir_fd=directory)
        with os.fdopen(descriptor, "rb") as stream:
            _check_file(os.fstat(stream.fileno()), ROOT_UID, 0o644)
            raw = stream.read()
    if _digest(raw) != LOCK_SHA256:
        raise ValueError("native lock SHA256 mismatch")
    lock = json.loads(raw)
    _validate_selection(lock)
    return lock


def _validate_selection(lock):
    packages = lock["debian_binaries"]
    if lock["schema"] != 1 or len(packages) != len(SELECTION):
        raise ValueError("unexpected native lock schema/selection")
    if {p["name"] for p in packages} != set(SELECTION):
        raise ValueError("unexpected native packages")
    for package in packages:
        library, soname = SELECTION[package["name"]]
        expected = {"usr/lib/x86_64-linux-gnu/" + n for n in (library, soname)}
        if package["architecture"] != "amd64" or len(package["files"]) != 2:
            raise ValueError("unexpected native architecture/member count")
        if {item["path"] for item in package["files"]} != expected:
            raise ValueError("native destination is outside the fixed selection")
        for item in package["files"]:
            if item["path"].endswith("/" + soname):
                if item.get("link") != library or item["mode"] != 0o777:
                    raise ValueError("unexpected native SONAME")
            elif item["mode"] != 0o644 or "link" in item:
                raise ValueError("unexpected native library mode/type")


def _ar_members(raw):
    if not raw.startswith(b"!<arch>\n"):
        raise ValueError("invalid Debian ar magic")
    members = {}
    position = 8
    for expected in ("debian-binary", "control.tar.xz", "data.tar.xz"):
        header = raw[position : position + 60]
        if len(header) != 60 or header[58:] != b"`\n":
            raise ValueError("invalid/truncated Debian ar header")
        name = header[:16].decode("ascii").rstrip().removesuffix("/")
        if name != expected or not re.fullmatch(rb"[0-9]+ *", header[48:58]):
            raise ValueError("unexpected Debian ar member/order/size")
        size = int(header[48:58])
        start, end = position + 60, position + 60 + size
        if end > len(raw) or (size % 2 and raw[end : end + 1] != b"\n"):
            raise ValueError("truncated Debian ar member/padding")
        members[name] = raw[start:end]
        position = end + size % 2
    if position != len(raw) or members["debian-binary"] != b"2.0\n":
        raise ValueError("unexpected Debian ar trailer/version")
    return members


def _tar_path(member):
    name = member.name.removeprefix("./").removesuffix("/")
    if name in ("", ".") and member.isdir():
        return "."
    parts = name.split("/")
    if any(part in ("", ".", "..") for part in parts) or "\x00" in name:
        raise ValueError("unsafe Debian tar path")
    return name


def _check_tar_member(member):
    if member.pax_headers or member.sparse is not None:
        raise ValueError("unexpected Debian tar extensions")
    if not (member.isfile() or member.isdir() or member.issym()):
        raise ValueError("unsafe Debian tar member type")
    if member.mode & ~0o777 or member.uid != 0 or member.gid != 0:
        raise ValueError("unsafe Debian tar mode/ownership")
    if member.issym():
        target = PurePosixPath(member.linkname)
        if not target.parts or target.is_absolute() or ".." in target.parts:
            raise ValueError("unsafe Debian tar link")


def _selected_member(archive, member, item):
    if member.mode != item["mode"]:
        raise ValueError("native member mode mismatch")
    if "link" in item:
        if not member.issym() or member.linkname != item["link"] or member.size:
            raise ValueError("native member SONAME mismatch")
        return None
    if not member.isfile() or member.size != item["size"]:
        raise ValueError("native member type/size mismatch")
    with archive.extractfile(member) as stream:
        raw = stream.read(item["size"] + 1)
    if len(raw) != item["size"] or _digest(raw) != item["sha256"]:
        raise ValueError("native member SHA256 mismatch")
    if not raw.startswith(b"\x7fELF"):
        raise ValueError("native library is not ELF")
    return raw


def _select_libraries(package, raw):
    if len(raw) != package["size"] or _digest(raw) != package["sha256"]:
        raise ValueError("native Debian artifact size/SHA256 mismatch")
    members = _ar_members(raw)
    wanted = {item["path"]: item for item in package["files"]}
    selected, seen = {}, set()
    with tarfile.open(
        fileobj=io.BytesIO(members["data.tar.xz"]), mode="r:xz"
    ) as archive:
        for member in archive:
            name = _tar_path(member)
            if name in seen:
                raise ValueError("duplicate Debian tar path")
            seen.add(name)
            _check_tar_member(member)
            if name in wanted:
                selected[name] = _selected_member(archive, member, wanted[name])
    if selected.keys() != wanted.keys():
        raise ValueError("missing native Debian members")
    return selected


@contextmanager
def _mutex(directory, owner):
    flags = os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
    descriptor = os.open("bootstrap.lock", flags, 0o600, dir_fd=directory)
    try:
        _check_file(os.fstat(descriptor), owner, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        current = os.stat("bootstrap.lock", dir_fd=directory, follow_symlinks=False)
        if os.fstat(descriptor) != current:
            raise ValueError("native bootstrap lock changed while waiting")
        yield
    finally:
        os.close(descriptor)


def _atomic_file(directory, name, raw, owner, mode):
    temporary = ".native-" + secrets.token_hex(16)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
    descriptor = os.open(temporary, flags, 0o600, dir_fd=directory)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fchmod(stream.fileno(), mode)
            os.fsync(stream.fileno())
        staged = _read_file(directory, temporary, owner, mode, len(raw))
        if staged != raw:
            raise ValueError("native staged bytes changed before installation")
        os.replace(temporary, name, src_dir_fd=directory, dst_dir_fd=directory)
        os.fsync(directory)
    finally:
        try:
            os.unlink(temporary, dir_fd=directory)
        except FileNotFoundError:
            pass


class _ArtifactBuffer(io.BytesIO):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def write(self, data):
        if self.tell() + len(data) > self.size:
            raise ValueError("native download exceeds locked artifact size")
        return super().write(data)


def _download(package):
    # -I -S omits script/site paths. Only this root-owned adjacent helper is used.
    sys.path.insert(0, str(SCRIPT_DIRECTORY))
    from _public_https import download_public_https

    with _ArtifactBuffer(package["size"]) as output:
        download_public_https(
            package["url"], output, allowed_hosts=frozenset({"snapshot.debian.org"})
        )
        return output.getvalue()


def _cached_artifacts(lock, cache):
    owner = os.geteuid()
    artifacts = []
    with _directory(cache, owner, private=True, create=True) as directory:
        with _mutex(directory, owner):
            for package in lock["debian_binaries"]:
                name = package["sha256"] + ".deb"
                try:
                    raw = _read_file(directory, name, owner, 0o600, package["size"])
                except FileNotFoundError:
                    raw = _download(package)
                    _select_libraries(package, raw)
                    _atomic_file(directory, name, raw, owner, 0o600)
                _select_libraries(package, raw)
                artifacts.append(raw)
    return artifacts


def _read_input(lock, stream):
    selected = {}
    for package in lock["debian_binaries"]:
        raw = stream.read(package["size"])
        selected.update(_select_libraries(package, raw))
    if stream.read(1):
        raise ValueError("unexpected native installation input trailer")
    return selected


def _installed(directory, item):
    name = Path(item["path"]).name
    try:
        info = os.stat(name, dir_fd=directory, follow_symlinks=False)
    except FileNotFoundError:
        return False
    if "link" in item:
        if (
            not stat.S_ISLNK(info.st_mode)
            or info.st_uid != ROOT_UID
            or info.st_gid != ROOT_GID
            or info.st_nlink != 1
            or stat.S_IMODE(info.st_mode) != 0o777
            or os.readlink(name, dir_fd=directory) != item["link"]
        ):
            raise ValueError("installed native SONAME changed")
    else:
        raw = _read_file(directory, name, ROOT_UID, item["mode"], item["size"])
        if len(raw) != item["size"] or _digest(raw) != item["sha256"]:
            raise ValueError("installed native library SHA256 mismatch")
    return True


def _install_selected(lock, selected):
    items = [item for p in lock["debian_binaries"] for item in p["files"]]
    with _directory(INSTALL_STATE, ROOT_UID, private=True, create=True) as state:
        with (
            _mutex(state, ROOT_UID),
            _directory(LIBRARY_DIRECTORY, ROOT_UID) as directory,
        ):
            present = {item["path"]: _installed(directory, item) for item in items}
            # Publish libraries before SONAME links, and check all existing paths
            # before modifying any. Interrupted installs resume only verified bytes.
            for item in sorted(items, key=lambda item: "link" in item):
                if present[item["path"]]:
                    continue
                name = Path(item["path"]).name
                if "link" in item:
                    os.symlink(item["link"], name, dir_fd=directory)
                else:
                    raw = selected[item["path"]]
                    if len(raw) != item["size"] or _digest(raw) != item["sha256"]:
                        raise ValueError("native selected bytes changed before install")
                    _atomic_file(directory, name, raw, ROOT_UID, item["mode"])
            os.fsync(directory)
            if not all(_installed(directory, item) for item in items):
                raise ValueError("native installation incomplete")


def _check_root_entrypoint():
    if os.geteuid() != ROOT_UID:
        raise ValueError("native installation requires root")
    for path, mode in (
        (Path(PYTHON), 0o755),
        (SCRIPT_DIRECTORY / "native_bootstrap.py", 0o644),
    ):
        with _directory(path.parent, ROOT_UID) as directory:
            descriptor = os.open(path.name, READ_FLAGS, dir_fd=directory)
            try:
                _check_file(os.fstat(descriptor), ROOT_UID, mode)
            finally:
                os.close(descriptor)


def _ensure(lock):
    if os.geteuid() == 0:
        raise ValueError("native download must run as the unprivileged container user")
    home = Path(pwd.getpwuid(os.geteuid()).pw_dir)
    base = Path(os.environ.get("XDG_CACHE_HOME", str(home / ".cache")))
    cache = Path(
        os.environ.get("NPA_NCORE_NATIVE_CACHE", str(base / "npa/ncore/native"))
    )
    artifacts = _cached_artifacts(lock, cache)
    command = [
        "/usr/bin/sudo",
        "-n",
        "/usr/bin/env",
        "-i",
        "PATH=/usr/bin:/bin",
        PYTHON,
        "-I",
        "-S",
        "-B",
        str(SCRIPT_DIRECTORY / "native_bootstrap.py"),
        "install",
    ]
    subprocess.run(command, input=b"".join(artifacts), check=True)


def _main():
    if sys.argv[1:] not in (["ensure"], ["install"], ["verify-lock"]):
        raise ValueError("expected exactly ensure, install or verify-lock")
    if sys.argv[1] == "install":
        _check_root_entrypoint()
    lock = _read_lock()
    if sys.argv[1] == "install":
        selected = _read_input(lock, sys.stdin.buffer)
        _install_selected(lock, selected)
    elif sys.argv[1] == "ensure":
        _ensure(lock)


if __name__ == "__main__":
    try:
        _main()
    except (
        OSError,
        ValueError,
        RuntimeError,
        tarfile.TarError,
        subprocess.SubprocessError,
    ):
        # Host paths, server messages and caller environment never enter diagnostics.
        print("NCore native bootstrap failed; refusing startup", file=sys.stderr)
        sys.exit(1)
