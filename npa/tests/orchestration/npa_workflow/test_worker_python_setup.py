"""Execute worker setup against old, compatible, and unavailable runtimes."""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys

import pytest

from npa.orchestration.npa_workflow.skypilot_render import (
    default_npa_setup,
    render_task_run_script,
)


def _executable(path: Path, body: str) -> Path:
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(0o755)
    return path


@pytest.fixture
def worker(tmp_path: Path) -> dict[str, str]:
    binaries = tmp_path / "bin"
    binaries.mkdir()
    for command in ("mkdir", "mktemp", "chmod", "sed", "ln", "cat"):
        (binaries / command).symlink_to(shutil.which(command))
    _executable(binaries / "python3", "exit 93\n")
    compatible = _executable(
        binaries / "compatible python",
        'if [ "$1" = -m ] && [ "$2" = pip ]; then\n'
        '  printf \'%s\\n\' "$*" >> "$INSTALL_LOG"\n'
        '  if [ "$3" = install ]; then : > "$INSTALL_LOG.done"; fi\n'
        "  exit 0\n"
        "fi\n"
        'if [ "$2" = "import npa.cli.main" ] && [ "$REQUIRE_INSTALL" = 1 ] '
        '&& [ ! -f "$INSTALL_LOG.done" ]; then exit 1; fi\n'
        f'exec {shlex.quote(sys.executable)} "$@"\n',
    )
    _executable(
        binaries / "uv",
        'printf \'%s\\n\' "$*" >> "$UV_LOG"\n'
        'if [ "$UV_EXIT" != 0 ]; then exit "$UV_EXIT"; fi\n'
        'for target in "$@"; do :; done\n'
        'mkdir -p "$target/bin"\n'
        'ln -s "$COMPATIBLE_PYTHON" "$target/bin/python"\n',
    )
    runtime = tmp_path / "runtime"
    source = runtime / "npa-src/src"
    cli = source / "npa/cli"
    cli.mkdir(parents=True)
    for module in (source / "npa/__init__.py", cli / "__init__.py", cli / "main.py"):
        module.write_text("")
    (cli / "entry.py").write_text(
        "import sys\ndef main(): print('prepared-control-cli', *sys.argv[1:])\n"
    )
    return {
        "HOME": str(tmp_path / "home"),
        "PATH": str(binaries),
        "PYTHONPATH": str(source),
        "COMPATIBLE_PYTHON": str(compatible),
        "INSTALL_LOG": str(tmp_path / "install.log"),
        "UV_LOG": str(tmp_path / "uv.log"),
        "UV_EXIT": "0",
        "REQUIRE_INSTALL": "0",
        "NPA_SRC_OVERLAY": "",
        "NPA_SRC_S3_URI": "",
        "HISTFILE": os.devnull,
    }


def _isolated(script: str, tmp_path: Path) -> str:
    worker_paths = set()
    for value in re.findall(r"/[A-Za-z0-9_./-]+", script):
        parts = PurePosixPath(value).parts
        if len(parts) >= 3 and parts[1] == "tmp" and parts[2].startswith("npa-"):
            worker_paths.add(PurePosixPath(*parts[:3]))
    assert worker_paths
    for path in sorted(worker_paths, key=str, reverse=True):
        script = script.replace(str(path), str(tmp_path / "runtime" / path.name))
    return (
        script.replace("/opt/", str(tmp_path / "absent-opt") + "/")
        .replace("/usr/local/bin/npa", str(tmp_path / "npa-link"))
        .replace("/etc/profile.d", str(tmp_path / "absent-profiles"))
    )


def _setup(tmp_path: Path, environment: dict[str, str], *, interactive: bool = False):
    arguments = [shutil.which("bash")]
    script = _isolated(default_npa_setup(), tmp_path)
    if interactive:
        arguments.extend(["--noprofile", "--norc", "-i"])
    else:
        arguments.extend(["-c", script])
    return subprocess.run(
        arguments,
        input=script if interactive else None,
        env=environment,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("stale_launcher", [False, True])
def test_old_worker_prepares_and_records_a_separate_control_runtime(
    tmp_path: Path, worker: dict[str, str], stale_launcher: bool
) -> None:
    worker["REQUIRE_INSTALL"] = "1"
    if stale_launcher:
        _executable(tmp_path / "bin/npa", "exit 94\n")

    result = _setup(tmp_path, worker)

    assert result.returncode == 0, result.stderr
    recorded = (tmp_path / "runtime/npa-python").read_text().strip()
    assert Path(recorded).is_relative_to(tmp_path / "runtime")
    assert recorded.endswith("/bin/python")
    assert Path(worker["UV_LOG"]).read_text().startswith("venv --python 3.12 --seed ")
    assert Path(worker["INSTALL_LOG"]).read_text().splitlines() == [
        "-m pip --version",
        f"-m pip install -q -e {tmp_path / 'runtime/npa-src'}",
    ]
    stage = _isolated(
        render_task_run_script(["/bin/sh", "-c", "python3 --version && npa --version"]),
        tmp_path,
    )
    executed = subprocess.run(
        [shutil.which("bash"), "-c", stage],
        env=worker,
        capture_output=True,
        text=True,
    )
    assert executed.returncode == 0, executed.stderr
    assert (
        f"Python {sys.version_info.major}.{sys.version_info.minor}" in executed.stdout
    )
    assert "prepared-control-cli --version" in executed.stdout


@pytest.mark.parametrize("source", ["baked", "python3.12", "python3.11"])
def test_compatible_worker_does_not_bootstrap_another_runtime(
    tmp_path: Path, worker: dict[str, str], source: str
) -> None:
    if source == "baked":
        worker["NPA_BAKED_PYTHON"] = worker["COMPATIBLE_PYTHON"]
    else:
        (tmp_path / "bin" / source).symlink_to(worker["COMPATIBLE_PYTHON"])

    result = _setup(tmp_path, worker)

    assert result.returncode == 0, result.stderr
    assert not Path(worker["UV_LOG"]).exists()
    recorded = (tmp_path / "runtime/npa-python").read_text().strip()
    assert Path(recorded).resolve() == Path(worker["COMPATIBLE_PYTHON"])


@pytest.mark.parametrize("failure", ["uv-missing", "download-failed"])
def test_unavailable_runtime_fails_before_installing_source(
    tmp_path: Path, worker: dict[str, str], failure: str
) -> None:
    if failure == "uv-missing":
        (tmp_path / "bin/uv").unlink()
    else:
        worker["UV_EXIT"] = "17"

    result = _setup(tmp_path, worker)

    assert result.returncode != 0
    assert not Path(worker["INSTALL_LOG"]).exists()
    assert not (tmp_path / "runtime/npa-python").exists()
    if failure == "uv-missing":
        assert "supply a compatible control interpreter or uv" in result.stderr
    else:
        assert result.returncode == 17


def test_baked_launcher_survives_interactive_history_expansion(
    tmp_path: Path, worker: dict[str, str]
) -> None:
    worker["NPA_BAKED_PYTHON"] = worker["COMPATIBLE_PYTHON"]

    result = _setup(tmp_path, worker, interactive=True)

    assert result.returncode == 0, result.stderr
    assert "event not found" not in result.stderr
    launcher = Path(worker["HOME"]) / ".local/bin/npa"
    assert launcher.read_text().startswith("#!/bin/sh\n")
    executed = subprocess.run(
        [str(launcher), "--version"], env=worker, capture_output=True, text=True
    )
    assert executed.returncode == 0, executed.stderr
    assert "prepared-control-cli --version" in executed.stdout
