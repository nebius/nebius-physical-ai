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
EXPECTED_SOURCE_COMMIT = "4d1ebecbc6436806cfbc0e42ebc36f594d05844e"
EXPECTED_MUJOCO_VERSION = "3.12.0"
EXPECTED_DECISION_SHA256 = (
    "758a29a6fae55075dc4ba879907e81f949b7a4e23fa726b790fd4361241697a3"
)
EXPECTED_WHEEL_COUNT = 26
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
            if (
                len(infos) > MAX_ARCHIVE_MEMBERS
                or sum(item.file_size for item in infos) > max_unpacked_bytes
            ):
                _refuse(f"wheel exceeds its expansion limit: {path.name}")
            folded: set[str] = set()
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
            if archive.testzip() is not None:
                _refuse(f"wheel CRC validation failed: {path.name}")
    except (OSError, zipfile.BadZipFile) as error:
        _refuse(f"wheel is malformed: {path.name}: {error}")


def _install_runtime(stage: Path, requirements: Path) -> None:
    runtime = stage / "runtime"
    wheelhouse = stage / "wheelhouse"
    source = stage / "source"
    try:
        venv.EnvBuilder(with_pip=True, clear=False, symlinks=False).create(runtime)
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


def _validated_existing(target: Path, runtime_lock: RuntimeLock) -> dict[str, object]:
    metadata = target.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
    ):
        _refuse("existing runtime cache version is not an operator-owned directory")
    try:
        receipt = json.loads((target / "receipt.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        _refuse(f"existing runtime cache is incomplete or malformed: {error}")
    if (
        receipt.get("schema") != RECEIPT_SCHEMA
        or receipt.get("manifest_sha256") != runtime_lock.digest
        or receipt.get("source_commit") != EXPECTED_SOURCE_COMMIT
        or receipt.get("mujoco_version") != EXPECTED_MUJOCO_VERSION
    ):
        _refuse("existing runtime cache receipt does not match the pinned runtime")
    python = target / "runtime/bin/python"
    if not python.is_file() or not os.access(python, os.X_OK):
        _refuse("existing runtime cache has no executable Python runtime")
    return receipt


def prepare(
    manifest: Path,
    requirements: Path,
    cache_root: Path,
    *,
    opener: Callable[[str], contextlib.AbstractContextManager[BinaryIO]] = _open_url,
    installer: Callable[[Path, Path], None] = _install_runtime,
) -> dict[str, object]:
    """Fetch, validate, materialize, and atomically select one runtime version."""

    runtime_lock = load_lock(manifest, requirements)
    cache_root = _validate_cache_root(cache_root)
    versions = cache_root / "versions"
    staging = cache_root / ".staging"
    versions.mkdir(mode=0o700, exist_ok=True)
    staging.mkdir(mode=0o700, exist_ok=True)
    for directory in (versions, staging):
        metadata = directory.lstat()
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) & 0o022
        ):
            _refuse("runtime cache control directory is unsafe")
    target = versions / runtime_lock.digest
    with _exclusive_lock(cache_root):
        cache_reused = target.exists()
        if cache_reused:
            receipt = _validated_existing(target, runtime_lock)
        else:
            stage = Path(tempfile.mkdtemp(prefix="runtime-", dir=staging))
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
                }
                receipt_path = stage / "receipt.json"
                receipt_path.write_text(
                    json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                receipt_path.chmod(0o400)
                os.replace(stage, target)
            except BaseException:
                shutil.rmtree(stage, ignore_errors=True)
                raise
        link = cache_root / "current"
        temporary_link = cache_root / f".current-{os.getpid()}"
        try:
            temporary_link.unlink(missing_ok=True)
            temporary_link.symlink_to(Path("versions") / runtime_lock.digest)
            os.replace(temporary_link, link)
        finally:
            temporary_link.unlink(missing_ok=True)
    return {
        **receipt,
        "runtime_root": str(target),
        "cache_root": str(cache_root),
        "cache_reused": cache_reused,
    }


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
        receipt = prepare(args.manifest, args.requirements, args.cache_root)
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
            python = Path(str(receipt["runtime_root"])) / "runtime/bin/python"
            os.environ["NPA_GYMNASIUM_RUNTIME_ROOT"] = str(receipt["runtime_root"])
            os.execv(str(python), [str(python), "-I", "-B", *command])
        print(rendered, end="")
        return 0
    except BootstrapRefusal as error:
        print(f"Gymnasium-Robotics runtime bootstrap refused: {error}", file=sys.stderr)
        return 65


if __name__ == "__main__":
    raise SystemExit(main())
