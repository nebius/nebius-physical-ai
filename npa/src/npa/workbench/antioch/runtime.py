"""Runtime-only installation of the proprietary Antioch CLI.

The public adapter image contains this downloader but no ``antioch-sim`` bytes.
The operator fetches the exact vendor wheel directly into a mounted cache.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import venv
from pathlib import Path

from .vendor_cli import AntiochCli, AntiochCliError


ANTIOCH_CLI_VERSION = "0.3.47"
ANTIOCH_CLI_SHA256 = "9037118b94d1b8241ca7e693b34a7a6ccffa7f47492b929b56c6f3b813032e8c"
ANTIOCH_CLI_URL = (
    "https://files.pythonhosted.org/packages/e5/61/"
    "577030f1ebdbd8591947bfc326a4eb357020e885ffa1011986eececae2c4/"
    "antioch_sim-0.3.47-py3-none-any.whl"
)


class AntiochRuntimeError(RuntimeError):
    """The pinned vendor runtime is unavailable or failed verification."""


def runtime_cache_root() -> Path:
    return Path(
        os.environ.get("NPA_ANTIOCH_RUNTIME_CACHE", "/workspace/.cache/npa/antioch")
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_executable(path: Path, expected_version: str) -> Path:
    if not path.is_file() or not os.access(path, os.X_OK):
        raise AntiochRuntimeError("configured Antioch CLI executable is not usable")
    try:
        version = AntiochCli(path).version()
    except AntiochCliError as exc:
        raise AntiochRuntimeError(str(exc)) from exc
    if version != expected_version:
        raise AntiochRuntimeError(
            f"Antioch CLI version mismatch: expected {expected_version}, received {version or '<unknown>'}"
        )
    return path


def ensure_runtime(*, expected_version: str = ANTIOCH_CLI_VERSION) -> Path:
    """Return an exact Antioch CLI, populating the runtime cache if necessary."""

    explicit = os.environ.get("NPA_ANTIOCH_CLI", "").strip()
    if explicit:
        return _verified_executable(Path(explicit), expected_version)
    if expected_version != ANTIOCH_CLI_VERSION:
        raise AntiochRuntimeError(
            f"unsupported Antioch CLI version {expected_version}; this adapter pins {ANTIOCH_CLI_VERSION}"
        )

    root = runtime_cache_root()
    version_root = root / expected_version
    executable = version_root / "venv" / "bin" / "antioch"
    ready = version_root / ".complete"
    if (
        ready.is_file()
        and ready.read_text(encoding="utf-8").strip() == ANTIOCH_CLI_SHA256
    ):
        return _verified_executable(executable, expected_version)
    if os.environ.get("NPA_ANTIOCH_RUNTIME_OFFLINE", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }:
        raise AntiochRuntimeError(
            "Antioch runtime cache is cold and offline mode forbids download"
        )

    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".install.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if (
            ready.is_file()
            and ready.read_text(encoding="utf-8").strip() == ANTIOCH_CLI_SHA256
        ):
            return _verified_executable(executable, expected_version)
        url = os.environ.get("NPA_ANTIOCH_CLI_URL", ANTIOCH_CLI_URL).strip()
        expected_sha = (
            os.environ.get("NPA_ANTIOCH_CLI_SHA256", ANTIOCH_CLI_SHA256).strip().lower()
        )
        if not url:
            raise AntiochRuntimeError("NPA_ANTIOCH_CLI_URL must not be empty")
        if expected_sha != ANTIOCH_CLI_SHA256:
            raise AntiochRuntimeError(
                "NPA_ANTIOCH_CLI_SHA256 must match the adapter's reviewed 0.3.47 wheel digest"
            )

        with tempfile.TemporaryDirectory(
            prefix=".antioch-install-", dir=root
        ) as temp_name:
            temp = Path(temp_name)
            wheel = temp / "antioch_sim-0.3.47-py3-none-any.whl"
            try:
                with (
                    urllib.request.urlopen(url) as response,
                    wheel.open("wb") as output,
                ):
                    shutil.copyfileobj(response, output)
            except Exception as exc:
                raise AntiochRuntimeError(
                    "could not fetch the pinned Antioch CLI wheel"
                ) from exc
            actual_sha = _sha256(wheel)
            if actual_sha != expected_sha:
                raise AntiochRuntimeError(
                    "downloaded Antioch CLI wheel failed SHA-256 verification"
                )

            install_root = temp / "published"
            environment = install_root / "venv"
            venv.EnvBuilder(with_pip=True, clear=True).create(environment)
            pip = environment / "bin" / "pip"
            result = subprocess.run(
                [
                    str(pip),
                    "install",
                    "--disable-pip-version-check",
                    "--no-cache-dir",
                    str(wheel),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            if result.returncode:
                raise AntiochRuntimeError(
                    "pinned Antioch CLI runtime installation failed"
                )
            (install_root / ".complete").write_text(
                expected_sha + "\n", encoding="utf-8"
            )
            _verified_executable(environment / "bin" / "antioch", expected_version)
            if version_root.exists():
                raise AntiochRuntimeError(
                    "Antioch runtime cache contains an incomplete pinned version"
                )
            os.replace(install_root, version_root)

    return _verified_executable(executable, expected_version)


def runtime_has_proprietary_distribution() -> bool:
    """Detect a mistakenly baked system-level proprietary distribution."""

    script = (
        "import importlib.metadata as m,sys; "
        "sys.exit(0 if any(d.metadata.get('Name','').lower()=='antioch-sim' for d in m.distributions()) else 1)"
    )
    return subprocess.run([sys.executable, "-c", script], check=False).returncode == 0
