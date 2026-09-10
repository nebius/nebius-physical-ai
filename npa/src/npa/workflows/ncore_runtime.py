"""Fetch the NCore CPU dependency closure into an operator-owned runtime cache.

Only stdlib imports belong here: the image contains no application wheels.
A cache is local/ephemeral unless the operator mounts durable storage. Cache
contents must never be copied into a published image or shared with other users.
"""

from __future__ import annotations

import base64
from contextlib import contextmanager
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import urllib.parse

from npa._public_https import download_public_https
from npa.workbench.ncore_staging import PrivateStagingError, private_directory


RUNTIME_LOCK = Path("/usr/share/doc/npa-ncore/runtime-lock.json")
RUNTIME_LOCK_SHA256 = "e5385557190c68c7425c96d535d6759b6a66257559d024be67f8c14a0951af7b"
READY_MARKER = ".npa-ncore-ready.json"
SOURCE_ROOTS = (
    "/opt/npa/src",
    "/opt/ncore/src/ncore",
    "/opt/ncore/src/pycolmap",
    "/opt/ncore/src/pycolmap/pycolmap",
)


class NcoreRuntimeError(RuntimeError):
    """A reviewed, complete runtime could not be established."""


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_lock(path: Path) -> dict:
    if sha256(path) != RUNTIME_LOCK_SHA256:
        raise NcoreRuntimeError("NCore runtime lock SHA256 mismatch")
    lock = json.loads(path.read_bytes())
    if lock.get("schema") != 1:
        raise NcoreRuntimeError("unsupported NCore runtime lock")
    names = set()
    for item in lock["artifacts"]:
        url = urllib.parse.urlsplit(item["url"])
        if (
            url.scheme != "https"
            or url.netloc != "files.pythonhosted.org"
            or url.query
            or url.fragment
            or Path(item["filename"]).name != item["filename"]
            or item["filename"] in ("", ".", "..")
            or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
            or not re.fullmatch(r"[a-z0-9-]+", item["name"])
            or not re.fullmatch(r"[a-zA-Z0-9.+]+", item["version"])
            or item["name"] in names
            or item["phase"] not in ("build", "runtime")
        ):
            raise NcoreRuntimeError("invalid NCore runtime artifact lock")
        names.add(item["name"])
    return lock


def source_identity() -> str:
    digest = hashlib.sha256()
    for path in (
        Path("/opt/ncore/src/source-inventory.json"),
        Path("/usr/share/doc/npa-ncore/npa-source-sha"),
        Path(__file__),
    ):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def cache_directory() -> Path:
    if value := os.environ.get("NPA_NCORE_RUNTIME_CACHE"):
        return Path(value).expanduser()
    if value := os.environ.get("NPA_MODEL_CACHE_DIR"):
        return Path(value).expanduser() / "ncore/runtime"
    return (
        Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
        / "npa/ncore/runtime"
    )


def download_artifact(item: dict, destination: Path) -> None:
    try:
        with destination.open("xb") as output:
            download_public_https(
                item["url"], output, allowed_hosts=frozenset({"files.pythonhosted.org"})
            )
        if sha256(destination) != item["sha256"]:
            raise NcoreRuntimeError("NCore runtime artifact SHA256 mismatch")
    except Exception:
        destination.unlink(missing_ok=True)
        raise


def _run(argv: list[str], *, env: dict | None = None) -> None:
    # Package tools can print credentials or ambient index configuration. Keep
    # diagnostics private and emit only the failing phase to callers.
    with tempfile.TemporaryFile() as diagnostics:
        result = subprocess.run(
            argv, env=env, stdout=diagnostics, stderr=diagnostics, check=False
        )
    if result.returncode:
        raise NcoreRuntimeError("NCore runtime dependency command failed")


def install_runtime(destination: Path, lock: dict) -> None:
    downloads = destination / "downloads"
    downloads.mkdir()
    for item in lock["artifacts"]:
        download_artifact(item, downloads / item["filename"])
    # No ambient credentials, index, pip config, Python path or user site reach
    # the installer. All input bytes have already passed their exact hashes.
    env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": str(destination),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PIP_CONFIG_FILE": os.devnull,
        "PIP_NO_CACHE_DIR": "1",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    }
    _run(
        [sys.executable, "-I", "-m", "venv", "--without-pip", str(destination)], env=env
    )
    executable = str(destination / "bin/python")
    # Bootstrap pip from its verified wheel using zipimport, independent of the
    # base image's pip version. -S also prevents loading ambient site hooks.
    pip_wheel = next(
        downloads / a["filename"] for a in lock["artifacts"] if a["name"] == "pip"
    )
    bootstrap = "import runpy,sys; sys.path.insert(0,sys.argv.pop(1)); runpy.run_module('pip',run_name='__main__')"
    for phase in ("build", "runtime"):
        requirements = destination / f"{phase}-requirements.lock"
        requirements.write_text(
            "".join(
                f"{a['name']}=={a['version']} --hash=sha256:{a['sha256']}\n"
                for a in lock["artifacts"]
                if a["phase"] == phase
            )
        )
        _run(
            [
                executable,
                "-I",
                "-S",
                "-c",
                bootstrap,
                str(pip_wheel),
                "--isolated",
                "--python",
                executable,
                "install",
                "--no-index",
                "--find-links",
                str(downloads),
                "--require-hashes",
                "--no-deps",
                "--no-build-isolation",
                "--no-compile",
                "-r",
                str(requirements),
            ],
            env=env,
        )
    _run([executable, "-I", "-m", "pip", "check"], env=env)
    site = destination / "lib/python3.12/site-packages"
    (site / "ncore-source.pth").write_text("\n".join(SOURCE_ROOTS) + "\n")
    # Wheels, sdists, their license records and installer locks stay in this
    # private runtime generation for provenance; none enter image layers.


def verify_runtime(destination: Path) -> None:
    executable = str(destination / "bin/python")
    _run(
        [
            executable,
            "-I",
            "-B",
            "-c",
            (
                "import importlib.util; import numpy as np; import pycolmap; "
                "from ncore.data.v4 import SequenceLoaderV4; "
                "from npa.workbench.nurec.colmap import ColmapConversionRequest, validate_ncore_sequence; "
                "from npa.workbench.nurec.ncore_rig import derive_rig_poses; "
                "from npa.clients.storage import StorageClient; "
                "assert pycolmap.SceneManager.INVALID_POINT3D == np.iinfo(np.uint64).max; "
                "assert all(importlib.util.find_spec(x) is None for x in "
                "('torch','av','rerun','lancedb','pyarrow')); "
                "assert SequenceLoaderV4 and ColmapConversionRequest and StorageClient"
            ),
        ]
    )
    _run(
        [
            executable,
            "-I",
            "-B",
            "-m",
            "tools.data_converter.colmap.converter",
            "--help",
        ]
    )
    with tempfile.TemporaryDirectory(prefix="ncore-schema-") as help_output:
        _run(
            [
                executable,
                "-I",
                "-B",
                "-m",
                "tools.data_converter.colmap.converter",
                "--output-dir",
                help_output,
                "colmap-v4",
                "--help",
            ]
        )


def _files(root: Path) -> dict:
    files = {}
    for path in sorted(root.rglob("*")):
        if path == root / READY_MARKER:
            continue
        name = path.relative_to(root).as_posix()
        if path.is_symlink():
            files[name] = {"link": os.readlink(path)}
        elif path.is_file():
            files[name] = {"sha256": sha256(path)}
    return files


def verify_platform() -> None:
    if (
        sys.implementation.name != "cpython"
        or sys.version_info[:2] != (3, 12)
        or sys.platform != "linux"
        or platform.machine() not in ("x86_64", "amd64")
    ):
        raise NcoreRuntimeError("NCore runtime requires CPython 3.12 on Linux x86_64")


def relocate_scripts(stage: Path, ready: Path) -> None:
    """Relocate generated venv entrypoints and their installed RECORD hashes."""
    changed = {}
    for script in (stage / "bin").iterdir():
        if script.is_symlink() or not script.is_file():
            continue
        raw = script.read_bytes()
        if raw.startswith(b"#!") or script.name in (
            "activate",
            "activate.csh",
            "activate.fish",
            "Activate.ps1",
        ):
            rewritten = raw.replace(str(stage).encode(), str(ready).encode())
            if rewritten != raw:
                script.write_bytes(rewritten)
                changed[script.resolve()] = rewritten
    for record in (stage / "lib/python3.12/site-packages").glob("*.dist-info/RECORD"):
        with record.open(newline="") as stream:
            rows = list(csv.reader(stream))
        updated = False
        for row in rows:
            raw = changed.get((record.parent.parent / row[0]).resolve())
            if raw is not None:
                row[1] = "sha256=" + base64.urlsafe_b64encode(
                    hashlib.sha256(raw).digest()
                ).decode().rstrip("=")
                row[2] = str(len(raw))
                updated = True
        if updated:
            with record.open("w", newline="") as stream:
                csv.writer(stream).writerows(rows)


@contextmanager
def _runtime_lock(path: Path):
    # The cache is private; a separate regular inode coordinates publication
    # across processes without following stale links into other files.
    descriptor = None
    try:
        try:
            descriptor = os.open(
                path,
                os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
                0o600,
            )
            info = os.fstat(descriptor)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.geteuid()
                or stat.S_IMODE(info.st_mode) != 0o600
                or info.st_nlink != 1
            ):
                raise NcoreRuntimeError(
                    "NCore runtime cache lock is not private and regular"
                )
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        except OSError:
            raise NcoreRuntimeError("NCore runtime cache lock is unavailable") from None
        yield
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _runtime_key() -> str:
    identity = {
        "lock_sha256": RUNTIME_LOCK_SHA256,
        "source_sha256": source_identity(),
        "python": sys.version,
        "source_roots": SOURCE_ROOTS,
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def _read_generation_marker(ready: Path) -> dict:
    descriptor = os.open(
        ready / READY_MARKER, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
    )
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.geteuid()
            or info.st_nlink != 1
        ):
            raise ValueError("runtime marker must be an owned regular file")
        return json.load(stream)


def _verify_generation(ready: Path, key: str) -> None:
    try:
        info = ready.lstat()
        if (
            not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) != 0o700
        ):
            raise ValueError("runtime generation must be private")
        receipt = _read_generation_marker(ready)
        if receipt["key"] != key or receipt["files"] != _files(ready):
            raise ValueError("changed runtime generation")
    except (OSError, ValueError, KeyError, TypeError):
        raise NcoreRuntimeError("NCore runtime cache integrity check failed") from None


def _install_generation(cache: Path, ready: Path, key: str, lock: dict) -> None:
    stage = Path(tempfile.mkdtemp(prefix=f"{key}.partial-", dir=cache))
    try:
        install_runtime(stage, lock)
        verify_runtime(stage)
        relocate_scripts(stage, ready)
        (stage / READY_MARKER).write_text(
            json.dumps({"key": key, "files": _files(stage)}, sort_keys=True) + "\n"
        )
        stage.rename(ready)
    except Exception:
        shutil.rmtree(stage)
        raise NcoreRuntimeError("NCore runtime preparation failed") from None


def ensure_runtime(
    *, lock_path: Path = RUNTIME_LOCK, cache_root: Path | None = None
) -> Path:
    """Reuse or atomically install a hash-verified runtime under private ownership.

    Args:
        lock_path: Reviewed artifact lock bound to the embedded SHA256.
        cache_root: Optional private cache parent; defaults to operator cache settings.
    Returns:
        Complete verified generation containing the runtime interpreter.
    Raises:
        NcoreRuntimeError: Platform, path, lock, installation, or integrity checks fail.
    """
    lock = read_lock(lock_path)
    verify_platform()
    key = _runtime_key()
    try:
        cache = private_directory(cache_root or cache_directory())
    except PrivateStagingError:
        raise NcoreRuntimeError(
            "NCore runtime cache directory is not private"
        ) from None
    ready = cache / key
    with _runtime_lock(cache / f"{key}.lock"):
        if ready.exists() or ready.is_symlink():
            _verify_generation(ready, key)
        else:
            _install_generation(cache, ready, key, lock)
    return ready


def exec_runtime(module: str, argv: list[str]) -> None:
    runtime = ensure_runtime()
    executable = str(runtime / "bin/python")
    env = dict(os.environ)
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    os.execve(executable, [executable, "-I", "-B", "-m", module, *argv], env)


def main() -> None:
    try:
        if sys.argv[1:2] == ["converter"]:
            exec_runtime("tools.data_converter.colmap.converter", sys.argv[2:])
        else:
            raise NcoreRuntimeError("expected NCore runtime command: converter")
    except (NcoreRuntimeError, OSError):
        print("NCore runtime preparation or integrity check failed", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
