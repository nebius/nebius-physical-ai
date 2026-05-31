"""CLI helpers for managing the isolated SkyPilot runtime."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

import typer
from rich.console import Console

from npa.orchestration.skypilot._bin import REQUIRED_SKYPILOT_VERSION

app = typer.Typer(
    name="skypilot",
    help="Manage the isolated SkyPilot runtime used by NPA workflows.",
    no_args_is_help=True,
)

console = Console(stderr=True)

SKYPILOT_VERSION = REQUIRED_SKYPILOT_VERSION
UTC = timezone.utc
SKYPILOT_EXTRAS = ("nebius", "kubernetes")
SKYPILOT_PACKAGE = f"skypilot[{','.join(SKYPILOT_EXTRAS)}]=={SKYPILOT_VERSION}"
DEFAULT_VENV_PATH = Path.home() / ".npa" / "skypilot-venv"
VENV_PATH_ENV = "NPA_SKYPILOT_VENV_PATH"
PYTHON_ENV = "NPA_SKYPILOT_PYTHON"
MARKER_FILE = ".npa-bootstrap-ok"


class SkyPilotBootstrapError(RuntimeError):
    """Raised when the isolated SkyPilot runtime cannot be bootstrapped."""


@dataclass(frozen=True)
class VenvState:
    path: Path
    python_bin: Path
    pip_bin: Path
    sky_bin: Path
    exists: bool
    has_python: bool
    has_pip: bool
    has_sky: bool
    version: str | None
    importable: bool
    marker_path: Path

    @property
    def installed(self) -> bool:
        return self.version == SKYPILOT_VERSION and self.importable and self.has_sky


@dataclass(frozen=True)
class BootstrapResult:
    path: Path
    sky_bin: Path
    installed: bool
    reused: bool
    marker_path: Path


@app.command("bootstrap")
def bootstrap_cmd(
    path: Path | None = typer.Option(
        None,
        "--path",
        help=f"SkyPilot venv path. Defaults to {VENV_PATH_ENV} or ~/.npa/skypilot-venv.",
    ),
    python: str = typer.Option(
        "",
        "--python",
        help=f"Python executable used to create the isolated venv. Defaults to {PYTHON_ENV} or this interpreter.",
    ),
) -> None:
    """Install SkyPilot into an isolated, idempotent virtualenv."""

    try:
        result = bootstrap_skypilot(venv_path=path, python_bin=python or None)
    except SkyPilotBootstrapError as exc:
        _fail(str(exc))
        return

    state = "already installed" if result.reused else "installed"
    typer.echo(f"SkyPilot {SKYPILOT_VERSION} {state} at {result.path}")
    typer.echo(str(result.sky_bin))
    typer.echo(f"export NPA_SKYPILOT_BIN={shlex.quote(str(result.sky_bin))}")
    typer.echo(f"marker: {result.marker_path}")


@app.command("status")
def status_cmd(
    path: Path | None = typer.Option(
        None,
        "--path",
        help=f"SkyPilot venv path. Defaults to {VENV_PATH_ENV} or ~/.npa/skypilot-venv.",
    ),
    bin_path: bool = typer.Option(False, "--bin-path", help="Print only the resolved sky binary path."),
) -> None:
    """Report the isolated SkyPilot runtime status."""

    state = inspect_venv(_resolve_venv_path(path))
    if bin_path:
        if not state.has_sky:
            _fail(f"SkyPilot binary is not installed at {state.sky_bin}. Run `npa skypilot bootstrap`.")
            return
        typer.echo(str(state.sky_bin))
        return

    if not state.installed:
        detail = f"found version {state.version}" if state.version else "sky binary missing or not executable"
        _fail(f"SkyPilot {SKYPILOT_VERSION} is not ready in {state.path}: {detail}. Run `npa skypilot bootstrap`.")
        return

    marker_age = _format_marker_age(state.marker_path)
    typer.echo(f"venv_path: {state.path}")
    typer.echo(f"sky_bin: {state.sky_bin}")
    typer.echo(f"version: {state.version}")
    typer.echo(f"marker: {state.marker_path}")
    typer.echo(f"marker_age: {marker_age}")

    result = _run_no_raise([str(state.sky_bin), "check"])
    summary = _summarize_completed_process(result)
    typer.echo(f"sky_check: {summary}")


@app.command("verify")
def verify_cmd(
    path: Path | None = typer.Option(
        None,
        "--path",
        help=f"SkyPilot venv path. Defaults to {VENV_PATH_ENV} or ~/.npa/skypilot-venv.",
    ),
) -> None:
    """Run `sky check` against the isolated SkyPilot runtime."""

    state = inspect_venv(_resolve_venv_path(path))
    if not state.installed:
        detail = f"found version {state.version}" if state.version else "sky binary missing or not executable"
        _fail(f"SkyPilot {SKYPILOT_VERSION} is not ready in {state.path}: {detail}. Run `npa skypilot bootstrap`.")
        return

    result = _run_no_raise([str(state.sky_bin), "check"])
    if result.stdout:
        typer.echo(result.stdout.rstrip())
    if result.stderr:
        console.print(result.stderr.rstrip())
    raise typer.Exit(result.returncode)


def bootstrap_skypilot(
    *,
    venv_path: Path | str | None = None,
    python_bin: str | os.PathLike[str] | None = None,
    package_spec: str = SKYPILOT_PACKAGE,
    expected_version: str = SKYPILOT_VERSION,
    extras: tuple[str, ...] = SKYPILOT_EXTRAS,
) -> BootstrapResult:
    """Create or reuse an isolated SkyPilot virtualenv."""

    path = _resolve_venv_path(venv_path)
    _reject_npa_environment(path)
    if path.exists() and not path.is_dir():
        raise SkyPilotBootstrapError(
            f"Path collision: {path} exists and is not a directory. "
            "Suggested action: choose a different --path or remove the file."
        )

    state = inspect_venv(path)
    if state.version and state.version != expected_version:
        raise SkyPilotBootstrapError(
            f"Version conflict: {state.sky_bin} reports SkyPilot {state.version}, "
            f"but NPA requires {expected_version}. Suggested action: remove {path} "
            "or choose a new --path."
        )
    if state.installed:
        _write_marker(state, package_spec=package_spec, expected_version=expected_version, extras=extras, reused=True)
        return BootstrapResult(path=state.path, sky_bin=state.sky_bin, installed=True, reused=True, marker_path=state.marker_path)

    if state.exists and not state.has_python:
        raise SkyPilotBootstrapError(
            f"Path collision: {path} exists but is not a Python virtualenv. "
            "Suggested action: choose a different --path or remove the directory."
        )

    if not state.exists:
        _create_venv(path, python_bin)

    state = inspect_venv(path)
    _ensure_pip(state)
    _install_package(state, package_spec)

    state = inspect_venv(path)
    if state.version and state.version != expected_version:
        raise SkyPilotBootstrapError(
            f"Version conflict after install: expected SkyPilot {expected_version}, got {state.version}. "
            "Suggested action: remove the venv and retry."
        )
    if not state.importable:
        raise SkyPilotBootstrapError(
            f"SkyPilot installed in {path}, but `import sky` failed. "
            "Suggested action: inspect the venv Python and pip install logs, then retry bootstrap."
        )
    if not state.has_sky:
        raise SkyPilotBootstrapError(
            f"SkyPilot installed in {path}, but no executable sky binary was found. "
            "Suggested action: inspect pip console-script installation and retry bootstrap."
        )

    _write_marker(state, package_spec=package_spec, expected_version=expected_version, extras=extras, reused=False)
    return BootstrapResult(path=state.path, sky_bin=state.sky_bin, installed=True, reused=False, marker_path=state.marker_path)


def inspect_venv(path: Path | str) -> VenvState:
    resolved = Path(path).expanduser().resolve(strict=False)
    bin_dir = _venv_bin_dir(resolved)
    python_bin = bin_dir / ("python.exe" if os.name == "nt" else "python")
    pip_bin = bin_dir / ("pip.exe" if os.name == "nt" else "pip")
    sky_bin = bin_dir / ("sky.exe" if os.name == "nt" else "sky")
    has_python = _is_executable(python_bin)
    has_pip = _is_executable(pip_bin)
    has_sky = _is_executable(sky_bin)
    version = _sky_version(sky_bin) if has_sky else None
    importable = _sky_importable(python_bin) if has_python else False
    return VenvState(
        path=resolved,
        python_bin=python_bin,
        pip_bin=pip_bin,
        sky_bin=sky_bin,
        exists=resolved.exists(),
        has_python=has_python,
        has_pip=has_pip,
        has_sky=has_sky,
        version=version,
        importable=importable,
        marker_path=resolved / MARKER_FILE,
    )


def _resolve_venv_path(path: Path | str | None) -> Path:
    value = path or os.environ.get(VENV_PATH_ENV) or DEFAULT_VENV_PATH
    return Path(value).expanduser().resolve(strict=False)


def _reject_npa_environment(path: Path) -> None:
    prefixes = [Path(sys.prefix).expanduser().resolve(strict=False)]
    if os.environ.get("VIRTUAL_ENV"):
        prefixes.append(Path(os.environ["VIRTUAL_ENV"]).expanduser().resolve(strict=False))
    for prefix in prefixes:
        if path == prefix or prefix in path.parents:
            raise SkyPilotBootstrapError(
                f"Refusing to install SkyPilot into the NPA Python environment: {path}. "
                "Suggested action: use the default ~/.npa/skypilot-venv path or pass a separate --path."
            )


def _create_venv(path: Path, python_bin: str | os.PathLike[str] | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    executable = os.fspath(python_bin or os.environ.get(PYTHON_ENV) or sys.executable)
    result = _run_no_raise([executable, "-m", "venv", str(path)])
    if result.returncode != 0:
        detail = _combined_output(result) or "no output"
        raise SkyPilotBootstrapError(
            f"Unable to create SkyPilot venv with {executable}: {detail}. "
            "Suggested action: install Python with venv support or pass --python."
        )


def _ensure_pip(state: VenvState) -> None:
    if not state.has_python:
        raise SkyPilotBootstrapError(
            f"Missing Python in SkyPilot venv: {state.python_bin}. "
            "Suggested action: remove the venv and rerun bootstrap."
        )
    result = _run_no_raise([str(state.python_bin), "-m", "pip", "--version"])
    if result.returncode == 0:
        return
    ensurepip = _run_no_raise([str(state.python_bin), "-m", "ensurepip", "--upgrade"])
    if ensurepip.returncode == 0:
        return
    detail = _combined_output(ensurepip) or _combined_output(result) or "no output"
    raise SkyPilotBootstrapError(
        f"Missing pip in SkyPilot venv {state.path}: {detail}. "
        "Suggested action: install Python with ensurepip support or recreate the venv."
    )


def _install_package(state: VenvState, package_spec: str) -> None:
    result = _run_no_raise([str(state.python_bin), "-m", "pip", "install", package_spec])
    if result.returncode == 0:
        return
    detail = _combined_output(result) or "no output"
    if _looks_like_network_failure(detail):
        raise SkyPilotBootstrapError(
            f"Network failure while installing {package_spec}: {detail}. "
            "Suggested action: verify package index connectivity and rerun bootstrap."
        )
    raise SkyPilotBootstrapError(
        f"pip failed while installing {package_spec}: {detail}. "
        "Suggested action: inspect the pip error above, fix the environment, and rerun bootstrap."
    )


def _write_marker(
    state: VenvState,
    *,
    package_spec: str,
    expected_version: str,
    extras: tuple[str, ...],
    reused: bool,
) -> None:
    payload = {
        "version": expected_version,
        "install_timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "extras": list(extras),
        "package": package_spec,
        "sky_bin": str(state.sky_bin),
        "reused_existing_venv": reused,
    }
    state.marker_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _sky_version(sky_bin: Path) -> str | None:
    result = _run_no_raise([str(sky_bin), "--version"])
    output = _combined_output(result)
    match = re.search(r"(\d+\.\d+\.\d+)", output)
    return match.group(1) if match else None


def _sky_importable(python_bin: Path) -> bool:
    result = _run_no_raise([str(python_bin), "-c", "import sky"])
    return result.returncode == 0


def _run_no_raise(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        return subprocess.CompletedProcess(cmd, 127, stdout="", stderr=str(exc))
    except OSError as exc:
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr=str(exc))


def _summarize_completed_process(result: subprocess.CompletedProcess[str]) -> str:
    status = "passed" if result.returncode == 0 else f"failed ({result.returncode})"
    output = _combined_output(result).splitlines()
    first_line = output[0] if output else "no output"
    return f"{status}: {first_line}"


def _combined_output(result: subprocess.CompletedProcess[str]) -> str:
    return "\n".join(part.strip() for part in (result.stdout, result.stderr) if part and part.strip())


def _looks_like_network_failure(detail: str) -> bool:
    lowered = detail.lower()
    needles = (
        "temporary failure",
        "name resolution",
        "connection",
        "network",
        "timed out",
        "timeout",
        "unreachable",
    )
    return any(needle in lowered for needle in needles)


def _format_marker_age(marker_path: Path) -> str:
    if not marker_path.exists():
        return "missing"
    age_seconds = max(0, int(time.time() - marker_path.stat().st_mtime))
    return f"{age_seconds}s"


def _venv_bin_dir(path: Path) -> Path:
    return path / ("Scripts" if os.name == "nt" else "bin")


def _is_executable(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def _fail(message: str, code: int = 1) -> None:
    console.print(f"[red]Error:[/red] {message}")
    raise typer.Exit(code)
