#!/usr/bin/env python3
"""Populate an operator-owned Gymnasium-Robotics runtime cache safely.

The public image contains this neutral bootstrap, not Gymnasium-Robotics,
Shadow Hand assets, MuJoCo, Python wheels, or a populated runtime cache.  A
complete repository-pinned manifest is required before this module opens a
network connection.  Acquisition changes delivery only: it grants no use,
derivative-work, output, or hosted-service rights.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterator
import contextlib
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from typing import BinaryIO, NoReturn
import urllib.error
import urllib.parse
import urllib.request
import venv
import zipfile

MANIFEST_SCHEMA = "npa.gymnasium-robotics.runtime-fetch-lock.v2"
RECEIPT_SCHEMA = "npa.gymnasium-robotics.runtime-cache-receipt.v1"
TREE_MANIFEST_SCHEMA = "npa.gymnasium-robotics.runtime-tree-manifest.v1"
EXPECTED_SOURCE_COMMIT = "4d1ebecbc6436806cfbc0e42ebc36f594d05844e"
EXPECTED_MUJOCO_VERSION = "3.12.0"
EXPECTED_DECISION_SHA256 = (
    "758a29a6fae55075dc4ba879907e81f949b7a4e23fa726b790fd4361241697a3"
)
EXPECTED_WHEEL_COUNT = 19
ALLOWED_HOSTS = frozenset(
    {"github.com", "codeload.github.com", "files.pythonhosted.org"}
)
REQUIRED_ROLES = frozenset({"solution-source", "python-wheel"})
SHA256 = re.compile(r"[0-9a-f]{64}")
REQUIREMENT = re.compile(
    r"[A-Za-z0-9_.-]+(?:\[[A-Za-z0-9_,.-]+\])?==[^\s]+"
    r"(?:\s+--hash=sha256:[0-9a-f]{64})+"
)
MAX_ARTIFACTS = 256
MAX_ARCHIVE_MEMBERS = 100_000
MAX_RUNTIME_ENTRIES = 200_000
RIGHTS_BOUNDARY = (
    "Runtime fetch changes delivery only; it does not grant or resolve use, "
    "derivative-work, output, or hosted-service rights."
)


class BootstrapRefusal(RuntimeError):
    """A fail-closed refusal that must occur without publishing partial state."""


@dataclass(frozen=True)
class Artifact:
    name: str
    role: str
    url: str
    final_url: str
    sha256: str
    size_bytes: int
    filename: str
    archive: str
    strip_prefix: str | None
    max_unpacked_bytes: int | None


@dataclass(frozen=True)
class RuntimeLock:
    raw: bytes
    digest: str
    artifacts: tuple[Artifact, ...]
    requirements_sha256: str


def _refuse(message: str) -> NoReturn:
    raise BootstrapRefusal(message)


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _safe_url(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        _refuse(f"{field} must be a string")
    parsed = urllib.parse.urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.hostname.lower() not in ALLOWED_HOSTS
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        _refuse(f"{field} is not an approved credential-free immutable HTTPS URL")
    return value


def _safe_relative(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        _refuse(f"{field} must be a non-empty POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        _refuse(f"{field} is unsafe")
    return str(path)


def _positive_int(value: object, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        _refuse(f"{field} must be a positive integer")
    return value


def _reject_control_proxies(value: object, *, path: str = "runtime lock") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = re.sub(r"[^a-z0-9]", "", str(key).lower())
            if any(
                token in normalized
                for token in (
                    "accepteula",
                    "acceptterms",
                    "consent",
                    "credential",
                    "password",
                    "secrettoken",
                    "authorization",
                )
            ):
                _refuse(f"{path} contains a credential or consent proxy")
            _reject_control_proxies(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_control_proxies(child, path=f"{path}[{index}]")


def _requirements_hashes(text: str) -> set[str]:
    logical: list[str] = []
    pending = ""
    for source_line in text.splitlines():
        line = source_line.strip()
        if not line or line.startswith("#"):
            continue
        pending = f"{pending} {line}".strip()
        if pending.endswith("\\"):
            pending = pending[:-1].rstrip()
            continue
        logical.append(pending)
        pending = ""
    if pending:
        _refuse("runtime requirements lock ends with an incomplete continuation")
    names: set[str] = set()
    hashes: set[str] = set()
    for requirement in logical:
        if REQUIREMENT.fullmatch(requirement) is None:
            _refuse(f"unhashed or malformed runtime requirement: {requirement}")
        raw_name = requirement.split("==", 1)[0].split("[", 1)[0]
        name = re.sub(r"[-_.]+", "-", raw_name).lower()
        selected = re.findall(r"--hash=sha256:([0-9a-f]{64})", requirement)
        if name in names or len(selected) != 1 or selected[0] in hashes:
            _refuse("runtime requirements must bind one unique wheel per distribution")
        names.add(name)
        hashes.add(selected[0])
    if len(names) != EXPECTED_WHEEL_COUNT:
        _refuse("runtime Python wheel closure is incomplete")
    return hashes


def load_lock(manifest: Path, requirements: Path) -> RuntimeLock:
    """Validate every trust field before any caller can fetch an artifact."""

    try:
        raw = manifest.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        _refuse(f"runtime lock is unavailable or malformed: {error}")
    if not isinstance(payload, dict) or payload.get("schema") != MANIFEST_SCHEMA:
        _refuse("runtime lock schema is unsupported")
    _reject_control_proxies(payload)
    if payload.get("status") != "complete":
        _refuse("runtime lock is incomplete; refusing before network access")
    if payload.get("source_commit") != EXPECTED_SOURCE_COMMIT:
        _refuse("runtime lock source commit changed")
    if payload.get("mujoco_version") != EXPECTED_MUJOCO_VERSION:
        _refuse("runtime lock MuJoCo version changed")
    if payload.get("decision_sha256") != EXPECTED_DECISION_SHA256:
        _refuse("runtime lock manager decision binding changed")
    if payload.get("rights_boundary") != RIGHTS_BOUNDARY:
        _refuse("runtime lock rights boundary is absent or changed")
    if (
        payload.get("expected_python_distribution_count") != EXPECTED_WHEEL_COUNT
        or payload.get("resolved_python_artifact_count") != EXPECTED_WHEEL_COUNT
    ):
        _refuse("runtime Python wheel closure is incomplete")
    requirements_sha256 = payload.get("requirements_lock_sha256")
    if not isinstance(requirements_sha256, str) or not SHA256.fullmatch(
        requirements_sha256
    ):
        _refuse("runtime lock has no exact requirements digest")
    try:
        if _sha256(requirements) != requirements_sha256:
            _refuse("runtime requirements lock bytes changed")
        requirements_text = requirements.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        _refuse(f"runtime requirements lock is unavailable: {error}")
    if (
        "# status: complete" not in requirements_text
        or "--hash=sha256:" not in requirements_text
        or "--require-hashes" in requirements_text
    ):
        _refuse("runtime requirements lock is incomplete or malformed")
    requirement_hashes = _requirements_hashes(requirements_text)

    records = payload.get("artifacts")
    if not isinstance(records, list) or not records or len(records) > MAX_ARTIFACTS:
        _refuse("runtime artifact inventory is absent or too large")
    artifacts: list[Artifact] = []
    names: set[str] = set()
    filenames: set[str] = set()
    roles: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            _refuse(f"artifact {index} is not an object")
        name = _safe_relative(record.get("name"), field=f"artifact {index} name")
        filename = _safe_relative(
            record.get("filename"), field=f"artifact {index} filename"
        )
        if "/" in name or "/" in filename or name in names or filename in filenames:
            _refuse(f"artifact {index} name or filename is duplicated or nested")
        role = record.get("role")
        if role not in REQUIRED_ROLES:
            _refuse(f"artifact {index} has an unsupported role")
        digest = record.get("sha256")
        if not isinstance(digest, str) or not SHA256.fullmatch(digest):
            _refuse(f"artifact {index} has no exact SHA-256")
        archive = record.get("archive")
        if archive not in {"tar.gz", "wheel"}:
            _refuse(f"artifact {index} has an unsupported archive format")
        strip_prefix = record.get("strip_prefix")
        max_unpacked = record.get("max_unpacked_bytes")
        if archive == "tar.gz":
            strip_prefix = _safe_relative(
                strip_prefix, field=f"artifact {index} strip_prefix"
            )
            max_unpacked = _positive_int(
                max_unpacked, field=f"artifact {index} max_unpacked_bytes"
            )
        else:
            if strip_prefix is not None:
                _refuse(f"wheel artifact {index} may not declare strip_prefix")
            max_unpacked = _positive_int(
                max_unpacked, field=f"artifact {index} max_unpacked_bytes"
            )
        artifacts.append(
            Artifact(
                name=name,
                role=role,
                url=_safe_url(record.get("url"), field=f"artifact {index} url"),
                final_url=_safe_url(
                    record.get("final_url"), field=f"artifact {index} final_url"
                ),
                sha256=digest,
                size_bytes=_positive_int(
                    record.get("size_bytes"), field=f"artifact {index} size_bytes"
                ),
                filename=filename,
                archive=archive,
                strip_prefix=strip_prefix,
                max_unpacked_bytes=max_unpacked,
            )
        )
        names.add(name)
        filenames.add(filename)
        roles.add(role)
    if not REQUIRED_ROLES.issubset(roles):
        _refuse("runtime artifact inventory lacks source or wheel roles")
    if sum(item.role == "solution-source" for item in artifacts) != 1:
        _refuse("runtime artifact inventory must contain one solution source")
    if sum(item.role == "python-wheel" for item in artifacts) != EXPECTED_WHEEL_COUNT:
        _refuse("runtime Python wheel closure is incomplete")
    wheel_hashes = {item.sha256 for item in artifacts if item.role == "python-wheel"}
    if wheel_hashes != requirement_hashes:
        _refuse("runtime wheel artifacts do not match requirements hashes")
    by_name = {item.name: item for item in artifacts}
    source_artifact = by_name.get("gymnasium-robotics-source")
    mujoco_artifact = by_name.get("mujoco-3.12.0-cp312-linux-x86_64")
    components = payload.get("components")
    if not isinstance(components, dict):
        _refuse("runtime component identity inventory is absent")
    try:
        expected_source_archive = components["farama_gymnasium_robotics"][
            "archive_sha256"
        ]
        expected_mujoco_wheel = components["mujoco"]["wheel_sha256"]
    except (KeyError, TypeError):
        _refuse("runtime component identity inventory is incomplete")
    if (
        source_artifact is None
        or source_artifact.role != "solution-source"
        or source_artifact.sha256 != expected_source_archive
        or mujoco_artifact is None
        or mujoco_artifact.role != "python-wheel"
        or mujoco_artifact.sha256 != expected_mujoco_wheel
    ):
        _refuse("runtime source or MuJoCo artifact identity changed")
    return RuntimeLock(
        raw=raw,
        digest=hashlib.sha256(raw).hexdigest(),
        artifacts=tuple(artifacts),
        requirements_sha256=requirements_sha256,
    )


def _validate_cache_root(cache_root: Path) -> Path:
    cache_root = cache_root.absolute()
    forbidden = (Path("/"), Path("/opt"), Path("/usr"), Path("/var"))
    if any(cache_root == item or item in cache_root.parents for item in forbidden[1:]):
        _refuse("runtime cache must be an operator-owned external path")
    if cache_root == forbidden[0]:
        _refuse("runtime cache may not be the filesystem root")
    try:
        cache_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        metadata = cache_root.lstat()
    except OSError as error:
        _refuse(f"runtime cache cannot be created: {error}")
    if stat.S_ISLNK(metadata.st_mode):
        _refuse("runtime cache root may not be a symlink")
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid():
        _refuse("runtime cache root is not owned by the runtime operator")
    if stat.S_IMODE(metadata.st_mode) & 0o022:
        _refuse("runtime cache root is group/world writable")
    for parent in (cache_root, *cache_root.parents):
        if parent.is_symlink():
            _refuse("runtime cache path may not traverse a symlink")
    return cache_root


class _RestrictedRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: BinaryIO,
        code: int,
        msg: str,
        headers: object,
        newurl: str,
    ) -> urllib.request.Request | None:
        _safe_url(newurl, field="runtime artifact redirect")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _open_url(url: str) -> contextlib.AbstractContextManager[BinaryIO]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "npa-gymnasium-robotics-bootstrap/1"},
        method="GET",
    )
    opener = urllib.request.build_opener(_RestrictedRedirectHandler())
    return opener.open(request, timeout=60)  # noqa: S310


def _download(
    artifact: Artifact,
    destination: Path,
    *,
    opener: Callable[[str], contextlib.AbstractContextManager[BinaryIO]],
) -> None:
    digest = hashlib.sha256()
    observed = 0
    try:
        with opener(artifact.url) as response, destination.open("xb") as output:
            final_url = getattr(response, "geturl", lambda: artifact.url)()
            if final_url != artifact.final_url:
                _refuse(f"redirect target changed for {artifact.name}")
            _safe_url(final_url, field=f"artifact {artifact.name} resolved URL")
            while chunk := response.read(1024 * 1024):
                observed += len(chunk)
                if observed > artifact.size_bytes:
                    _refuse(f"size mismatch for {artifact.name}")
                digest.update(chunk)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
    except BootstrapRefusal:
        raise
    except (OSError, urllib.error.URLError) as error:
        _refuse(f"runtime artifact access failed for {artifact.name}: {error}")
    if observed != artifact.size_bytes:
        _refuse(f"size mismatch for {artifact.name}")
    if digest.hexdigest() != artifact.sha256:
        _refuse(f"SHA-256 mismatch for {artifact.name}")
    destination.chmod(0o400)


def _member_destination(name: str, *, strip_prefix: str) -> PurePosixPath | None:
    if "\\" in name:
        _refuse("archive member contains a non-POSIX separator")
    source = PurePosixPath(name)
    if source.is_absolute() or ".." in source.parts:
        _refuse(f"unsafe archive member: {name}")
    prefix = PurePosixPath(strip_prefix)
    if source == prefix:
        return None
    try:
        relative = source.relative_to(prefix)
    except ValueError:
        _refuse(f"archive member escapes required prefix: {name}")
    if not relative.parts or "." in relative.parts or ".." in relative.parts:
        _refuse(f"unsafe archive member: {name}")
    return relative


def _extract_source(artifact: Artifact, archive_path: Path, destination: Path) -> None:
    assert artifact.strip_prefix is not None
    assert artifact.max_unpacked_bytes is not None
    destination.mkdir(mode=0o700)
    observed: set[str] = set()
    expanded = 0
    try:
        archive = tarfile.open(archive_path, mode="r:gz")
    except (OSError, tarfile.TarError) as error:
        _refuse(f"source archive is malformed: {error}")
    with archive:
        members = archive.getmembers()
        if not members or len(members) > MAX_ARCHIVE_MEMBERS:
            _refuse("source archive member count is invalid")
        for member in members:
            relative = _member_destination(
                member.name, strip_prefix=artifact.strip_prefix
            )
            if relative is None:
                if not member.isdir():
                    _refuse("source archive prefix is not a directory")
                continue
            normalized = str(relative)
            folded = normalized.casefold()
            if folded in observed:
                _refuse(f"source archive has a duplicate path: {normalized}")
            observed.add(folded)
            target = destination.joinpath(*relative.parts)
            if member.isdir():
                target.mkdir(mode=0o700, parents=True, exist_ok=True)
                continue
            if not member.isfile():
                _refuse(f"source archive contains a link or special file: {normalized}")
            expanded += member.size
            if expanded > artifact.max_unpacked_bytes:
                _refuse("source archive exceeds its expansion limit")
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            stream = archive.extractfile(member)
            if stream is None:
                _refuse(f"source archive member is unreadable: {normalized}")
            with target.open("xb") as output:
                shutil.copyfileobj(stream, output, length=1024 * 1024)
            target.chmod(0o600 | (member.mode & 0o100))
    if not observed:
        _refuse("source archive contains no materialized content")


def _validate_wheel(path: Path, *, max_unpacked_bytes: int) -> None:
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            if not infos:
                _refuse(f"wheel is empty: {path.name}")
            if len(infos) > MAX_ARCHIVE_MEMBERS:
                _refuse(f"wheel exceeds its expansion limit: {path.name}")
            folded: set[str] = set()
            expanded = 0
            for info in infos:
                name = _safe_relative(info.filename, field=f"wheel {path.name} member")
                key = name.casefold()
                if key in folded:
                    _refuse(f"wheel contains a duplicate path: {path.name}")
                folded.add(key)
                if info.flag_bits & 1:
                    _refuse(f"wheel contains an encrypted member: {path.name}")
                mode = info.external_attr >> 16
                if stat.S_ISLNK(mode):
                    _refuse(f"wheel contains a symbolic link: {path.name}")
                if info.is_dir():
                    continue
                with archive.open(info) as member:
                    while chunk := member.read(1024 * 1024):
                        expanded += len(chunk)
                        if expanded > max_unpacked_bytes:
                            _refuse(f"wheel exceeds its expansion limit: {path.name}")
    except (OSError, RuntimeError, zipfile.BadZipFile) as error:
        _refuse(f"wheel is malformed: {path.name}: {error}")


def _remove_venv_compatibility_link(runtime: Path) -> None:
    lib64 = runtime / "lib64"
    if not lib64.is_symlink():
        return
    if os.readlink(lib64) != "lib":
        _refuse("virtual environment contains an unexpected compatibility link")
    lib64.unlink()


def _install_runtime(stage: Path, requirements: Path) -> None:
    runtime = stage / "runtime"
    wheelhouse = stage / "wheelhouse"
    source = stage / "source"
    try:
        venv.EnvBuilder(with_pip=True, clear=False, symlinks=False).create(runtime)
        _remove_venv_compatibility_link(runtime)
        python = runtime / "bin/python"
        subprocess.run(
            [
                str(python),
                "-I",
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-index",
                "--no-deps",
                "--require-hashes",
                "--find-links",
                str(wheelhouse),
                "--requirement",
                str(requirements),
            ],
            check=True,
        )
        subprocess.run(
            [
                str(python),
                "-I",
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-index",
                "--no-deps",
                "--no-build-isolation",
                str(source),
            ],
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        _refuse(f"offline runtime installation failed: {error}")


@contextlib.contextmanager
def _exclusive_lock(cache_root: Path) -> Iterator[None]:
    lock_path = cache_root / ".bootstrap.lock"
    try:
        descriptor = os.open(
            lock_path,
            os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
            0o600,
        )
    except OSError as error:
        _refuse(f"runtime cache lock is unsafe or unavailable: {error}")
    with os.fdopen(descriptor, "a+b") as lock:
        os.fchmod(lock.fileno(), 0o600)
        metadata = os.fstat(lock.fileno())
        if metadata.st_uid != os.geteuid() or not stat.S_ISREG(metadata.st_mode):
            _refuse("runtime cache lock is not an operator-owned regular file")
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        yield


def _read_owned_control(path: Path) -> bytes:
    try:
        before = path.lstat()
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as error:
        _refuse(f"runtime cache control file is unsafe or unavailable: {error}")
    with os.fdopen(descriptor, "rb") as stream:
        observed = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(before.st_mode)
            or not stat.S_ISREG(observed.st_mode)
            or before.st_dev != observed.st_dev
            or before.st_ino != observed.st_ino
            or observed.st_uid != os.geteuid()
            or stat.S_IMODE(observed.st_mode) & 0o222
        ):
            _refuse("runtime cache control file is not sealed and operator-owned")
        return stream.read()


def _write_control(path: Path, content: bytes) -> None:
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_TRUNC | os.O_NOFOLLOW)
    except OSError as error:
        _refuse(f"runtime cache control file cannot be created safely: {error}")
    with os.fdopen(descriptor, "wb") as stream:
        metadata = os.fstat(stream.fileno())
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.geteuid():
            _refuse("runtime cache control file is not operator-owned")
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
        os.fchmod(stream.fileno(), 0o400)


def _seal_runtime_tree(root: Path) -> None:
    paths = sorted(root.rglob("*"), key=lambda path: len(path.parts), reverse=True)
    for path in paths:
        metadata = path.lstat()
        if metadata.st_uid != os.geteuid() or metadata.st_gid != os.getegid():
            _refuse("runtime cache contains an unowned entry")
        if stat.S_ISLNK(metadata.st_mode):
            _refuse("runtime cache contains a symbolic link")
        if stat.S_ISREG(metadata.st_mode):
            if path.parent == root and path.name in {
                "receipt.json",
                "tree-manifest.json",
            }:
                path.chmod(0o600)
            else:
                path.chmod(0o500 if metadata.st_mode & 0o111 else 0o400)
        elif stat.S_ISDIR(metadata.st_mode):
            path.chmod(0o500)
        else:
            _refuse("runtime cache contains a special file")
    metadata = root.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_gid != os.getegid()
    ):
        _refuse("runtime cache staging root is unsafe or unowned")
    root.chmod(0o500)


def _discard_stage(stage: Path) -> None:
    if not stage.exists():
        return
    for path in sorted(stage.rglob("*"), key=lambda item: len(item.parts)):
        with contextlib.suppress(OSError):
            if path.is_dir() and not path.is_symlink():
                path.chmod(0o700)
            elif not path.is_symlink():
                path.chmod(0o600)
    with contextlib.suppress(OSError):
        stage.chmod(0o700)
    shutil.rmtree(stage, ignore_errors=True)


def _discard_published(
    target: Path,
    versions: Path,
    *,
    expected_device: int,
    expected_inode: int,
) -> None:
    """Quarantine and remove only the exact version directory just published."""

    try:
        metadata = target.lstat()
    except FileNotFoundError:
        return
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_dev != expected_device
        or metadata.st_ino != expected_inode
    ):
        _refuse("refusing to remove a changed published runtime target")
    quarantine = versions / f".rejected-{os.getpid()}-{expected_inode}"
    if quarantine.exists() or quarantine.is_symlink():
        _refuse("runtime cache quarantine target already exists")
    target.rename(quarantine)
    moved = quarantine.lstat()
    if moved.st_dev != expected_device or moved.st_ino != expected_inode:
        _refuse("published runtime changed while it was quarantined")
    _discard_stage(quarantine)


def _runtime_tree_entries(root: Path) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    pending: list[tuple[Path, str]] = [(root, "")]
    excluded = {"receipt.json", "tree-manifest.json"}
    while pending:
        directory, prefix = pending.pop()
        try:
            children = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as error:
            _refuse(f"runtime cache tree cannot be traversed safely: {error}")
        for child in children:
            relative = f"{prefix}/{child.name}".lstrip("/")
            if not prefix and child.name in excluded:
                continue
            try:
                metadata = child.stat(follow_symlinks=False)
            except OSError as error:
                _refuse(f"runtime cache entry cannot be inspected: {error}")
            if metadata.st_uid != os.geteuid() or metadata.st_gid != os.getegid():
                _refuse(f"runtime cache entry is unowned: {relative}")
            mode = stat.S_IMODE(metadata.st_mode)
            if mode & 0o222:
                _refuse(f"runtime cache entry is writable: {relative}")
            if stat.S_ISDIR(metadata.st_mode):
                entries.append({"path": relative, "kind": "directory", "mode": mode})
                pending.append((Path(child.path), relative))
            elif stat.S_ISREG(metadata.st_mode):
                try:
                    descriptor = os.open(child.path, os.O_RDONLY | os.O_NOFOLLOW)
                except OSError as error:
                    _refuse(f"runtime cache file cannot be opened safely: {error}")
                with os.fdopen(descriptor, "rb") as stream:
                    opened = os.fstat(stream.fileno())
                    if (
                        opened.st_dev != metadata.st_dev
                        or opened.st_ino != metadata.st_ino
                        or not stat.S_ISREG(opened.st_mode)
                    ):
                        _refuse(f"runtime cache file changed during validation: {relative}")
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                entries.append(
                    {
                        "path": relative,
                        "kind": "regular",
                        "mode": mode,
                        "size_bytes": metadata.st_size,
                        "sha256": digest,
                    }
                )
            else:
                _refuse(f"runtime cache entry has an unsupported type: {relative}")
            if len(entries) > MAX_RUNTIME_ENTRIES:
                _refuse("runtime cache entry count exceeds its validation bound")
    return sorted(entries, key=lambda entry: str(entry["path"]))


def _validated_existing(
    target: Path,
    runtime_lock: RuntimeLock,
    *,
    root_metadata: os.stat_result | None = None,
) -> dict[str, object]:
    metadata = root_metadata if root_metadata is not None else target.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_gid != os.getegid()
        or stat.S_IMODE(metadata.st_mode) & 0o222
    ):
        _refuse("existing runtime cache version is not sealed and operator-owned")
    try:
        receipt_raw = _read_owned_control(target / "receipt.json")
        tree_raw = _read_owned_control(target / "tree-manifest.json")
        receipt = json.loads(receipt_raw)
        tree = json.loads(tree_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        _refuse(f"existing runtime cache is incomplete or malformed: {error}")
    expected_receipt_keys = {
        "schema",
        "status",
        "manifest_sha256",
        "requirements_lock_sha256",
        "source_commit",
        "mujoco_version",
        "artifact_sha256",
        "rights_boundary",
        "tree_manifest_sha256",
    }
    expected_artifacts = {
        item.name: item.sha256 for item in runtime_lock.artifacts
    }
    if (
        set(receipt) != expected_receipt_keys
        or receipt.get("schema") != RECEIPT_SCHEMA
        or receipt.get("status") != "ready"
        or receipt.get("manifest_sha256") != runtime_lock.digest
        or receipt.get("requirements_lock_sha256")
        != runtime_lock.requirements_sha256
        or receipt.get("source_commit") != EXPECTED_SOURCE_COMMIT
        or receipt.get("mujoco_version") != EXPECTED_MUJOCO_VERSION
        or receipt.get("artifact_sha256") != expected_artifacts
        or receipt.get("rights_boundary") != RIGHTS_BOUNDARY
        or receipt.get("tree_manifest_sha256")
        != hashlib.sha256(tree_raw).hexdigest()
    ):
        _refuse("existing runtime cache receipt does not match the pinned runtime")
    if (
        not isinstance(tree, dict)
        or set(tree) != {"schema", "entries"}
        or tree.get("schema") != TREE_MANIFEST_SCHEMA
        or tree.get("entries") != _runtime_tree_entries(target)
    ):
        _refuse("existing runtime cache tree differs from its sealed manifest")
    python = target / "runtime/bin/python"
    if not python.is_file() or not os.access(python, os.X_OK):
        _refuse("existing runtime cache has no executable Python runtime")
    return receipt


def _open_validated_runtime(
    target: Path, runtime_lock: RuntimeLock
) -> tuple[dict[str, object], int, int]:
    """Bind validation and execution to one directory and interpreter inode."""

    try:
        before = target.lstat()
        directory_fd = os.open(
            target,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
    except OSError as error:
        _refuse(f"runtime target cannot be opened safely: {error}")
    python_fd: int | None = None
    try:
        opened = os.fstat(directory_fd)
        if (
            before.st_dev != opened.st_dev
            or before.st_ino != opened.st_ino
            or not stat.S_ISDIR(opened.st_mode)
        ):
            _refuse("runtime target changed while its directory was opened")
        bound_root = Path(f"/proc/self/fd/{directory_fd}")
        if not bound_root.exists():
            _refuse("descriptor-bound runtime traversal is unavailable")
        receipt = _validated_existing(
            bound_root,
            runtime_lock,
            root_metadata=opened,
        )
        after = target.lstat()
        if after.st_dev != opened.st_dev or after.st_ino != opened.st_ino:
            _refuse("runtime target changed while it was validated")
        python_fd = os.open(
            "runtime/bin/python",
            os.O_RDONLY | os.O_NOFOLLOW,
            dir_fd=directory_fd,
        )
        python_metadata = os.fstat(python_fd)
        if (
            not stat.S_ISREG(python_metadata.st_mode)
            or python_metadata.st_uid != os.geteuid()
            or python_metadata.st_gid != os.getegid()
            or stat.S_IMODE(python_metadata.st_mode) & 0o222
            or not stat.S_IMODE(python_metadata.st_mode) & 0o111
        ):
            _refuse("runtime Python is not a sealed operator-owned executable")
        try:
            tree = json.loads(
                _read_owned_control(bound_root / "tree-manifest.json")
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            _refuse(f"runtime tree manifest changed before execution: {error}")
        if not isinstance(tree, dict) or not isinstance(tree.get("entries"), list):
            _refuse("runtime tree manifest changed before execution")
        expected_python = [
            entry
            for entry in tree.get("entries", [])
            if isinstance(entry, dict)
            and entry.get("path") == "runtime/bin/python"
        ]
        with os.fdopen(os.dup(python_fd), "rb") as stream:
            python_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        os.lseek(python_fd, 0, os.SEEK_SET)
        if expected_python != [
            {
                "path": "runtime/bin/python",
                "kind": "regular",
                "mode": stat.S_IMODE(python_metadata.st_mode),
                "size_bytes": python_metadata.st_size,
                "sha256": python_sha256,
            }
        ]:
            _refuse("runtime Python differs from the validated tree manifest")
        os.set_inheritable(directory_fd, True)
        os.set_inheritable(python_fd, True)
        return receipt, directory_fd, python_fd
    except BaseException:
        if python_fd is not None:
            os.close(python_fd)
        os.close(directory_fd)
        raise


def prepare(
    manifest: Path,
    requirements: Path,
    cache_root: Path,
    *,
    opener: Callable[[str], contextlib.AbstractContextManager[BinaryIO]] = _open_url,
    installer: Callable[[Path, Path], None] = _install_runtime,
    retain_runtime_handles: bool = False,
) -> dict[str, object]:
    """Fetch, validate, materialize, and atomically select one runtime version."""

    runtime_lock = load_lock(manifest, requirements)
    cache_root = _validate_cache_root(cache_root)
    versions = cache_root / "versions"
    versions.mkdir(mode=0o700, exist_ok=True)
    metadata = versions.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) & 0o022
    ):
        _refuse("runtime cache control directory is unsafe")
    target = versions / runtime_lock.digest
    with _exclusive_lock(cache_root):
        partials = [
            path
            for path in versions.iterdir()
            if path.name.startswith((".staging-", ".rejected-"))
        ]
        if partials:
            _refuse("runtime cache contains an unreviewed partial publication")
        cache_reused = target.exists()
        if cache_reused:
            receipt = _validated_existing(target, runtime_lock)
        else:
            # A sealed directory cannot be renamed across parents on Linux because
            # its ``..`` entry would change.  Stage under the versions directory so
            # the final rename is both same-parent atomic and performed only after
            # the root itself has become non-writable.
            stage = Path(tempfile.mkdtemp(prefix=".staging-runtime-", dir=versions))
            try:
                downloads = stage / "downloads"
                wheelhouse = stage / "wheelhouse"
                downloads.mkdir(mode=0o700)
                wheelhouse.mkdir(mode=0o700)
                for artifact in runtime_lock.artifacts:
                    fetched = downloads / artifact.filename
                    _download(artifact, fetched, opener=opener)
                    if artifact.archive == "tar.gz":
                        _extract_source(artifact, fetched, stage / "source")
                    else:
                        assert artifact.max_unpacked_bytes is not None
                        _validate_wheel(
                            fetched,
                            max_unpacked_bytes=artifact.max_unpacked_bytes,
                        )
                        os.replace(fetched, wheelhouse / artifact.filename)
                locked_requirements = stage / "requirements.lock"
                shutil.copyfile(requirements, locked_requirements)
                locked_requirements.chmod(0o400)
                installer(stage, locked_requirements)
                python = stage / "runtime/bin/python"
                if not python.is_file() or not os.access(python, os.X_OK):
                    _refuse("offline installer produced no executable Python runtime")
                tree_path = stage / "tree-manifest.json"
                receipt_path = stage / "receipt.json"
                tree_path.touch(mode=0o600, exist_ok=False)
                receipt_path.touch(mode=0o600, exist_ok=False)
                _seal_runtime_tree(stage)
                tree = {
                    "schema": TREE_MANIFEST_SCHEMA,
                    "entries": _runtime_tree_entries(stage),
                }
                tree_raw = (
                    json.dumps(tree, separators=(",", ":"), sort_keys=True) + "\n"
                ).encode()
                _write_control(tree_path, tree_raw)
                receipt = {
                    "schema": RECEIPT_SCHEMA,
                    "status": "ready",
                    "manifest_sha256": runtime_lock.digest,
                    "requirements_lock_sha256": runtime_lock.requirements_sha256,
                    "source_commit": EXPECTED_SOURCE_COMMIT,
                    "mujoco_version": EXPECTED_MUJOCO_VERSION,
                    "artifact_sha256": {
                        item.name: item.sha256 for item in runtime_lock.artifacts
                    },
                    "rights_boundary": RIGHTS_BOUNDARY,
                    "tree_manifest_sha256": hashlib.sha256(tree_raw).hexdigest(),
                }
                _write_control(
                    receipt_path,
                    (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode(),
                )
                staged = stage.lstat()
                if stat.S_IMODE(staged.st_mode) & 0o222:
                    _refuse("runtime staging root was not sealed before publication")
                os.replace(stage, target)
                try:
                    receipt = _validated_existing(target, runtime_lock)
                except BaseException:
                    _discard_published(
                        target,
                        versions,
                        expected_device=staged.st_dev,
                        expected_inode=staged.st_ino,
                    )
                    raise
            except BaseException:
                _discard_stage(stage)
                raise
        link = cache_root / "current"
        temporary_link = cache_root / f".current-{os.getpid()}"
        try:
            temporary_link.unlink(missing_ok=True)
            temporary_link.symlink_to(Path("versions") / runtime_lock.digest)
            os.replace(temporary_link, link)
        finally:
            temporary_link.unlink(missing_ok=True)
        handles: tuple[int, int] | None = None
        if retain_runtime_handles:
            receipt, directory_fd, python_fd = _open_validated_runtime(
                target, runtime_lock
            )
            handles = (directory_fd, python_fd)
    result: dict[str, object] = {
        **receipt,
        "runtime_root": str(target),
        "cache_root": str(cache_root),
        "cache_reused": cache_reused,
    }
    if handles is not None:
        result["_runtime_directory_fd"] = handles[0]
        result["_runtime_python_fd"] = handles[1]
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--requirements", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--json", type=Path)
    parser.add_argument("command", choices=("prepare", "exec"))
    parser.add_argument("args", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        receipt = prepare(
            args.manifest,
            args.requirements,
            args.cache_root,
            retain_runtime_handles=args.command == "exec",
        )
        directory_fd = receipt.pop("_runtime_directory_fd", None)
        python_fd = receipt.pop("_runtime_python_fd", None)
        rendered = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
        if args.json:
            args.json.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            args.json.write_text(rendered, encoding="utf-8")
            args.json.chmod(0o600)
        if args.command == "exec":
            command = list(args.args)
            if command[:1] == ["--"]:
                command.pop(0)
            if not command:
                _refuse("exec requires a script or module argument")
            if not isinstance(directory_fd, int) or not isinstance(python_fd, int):
                _refuse("exec did not retain descriptor-bound runtime handles")
            if os.execve not in os.supports_fd:
                _refuse("this platform cannot execute a descriptor-bound runtime")
            os.environ["NPA_GYMNASIUM_RUNTIME_ROOT"] = (
                f"/proc/self/fd/{directory_fd}"
            )
            python_name = f"/proc/self/fd/{python_fd}"
            try:
                os.execve(
                    python_fd,
                    [python_name, "-I", "-B", *command],
                    os.environ,
                )
            except OSError as error:
                os.close(python_fd)
                os.close(directory_fd)
                _refuse(f"descriptor-bound runtime execution failed: {error}")
        print(rendered, end="")
        return 0
    except BootstrapRefusal as error:
        print(f"Gymnasium-Robotics runtime bootstrap refused: {error}", file=sys.stderr)
        return 65


if __name__ == "__main__":
    raise SystemExit(main())
