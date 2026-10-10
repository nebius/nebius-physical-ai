"""Execute rendered stage shells against competing baked and submitted packages."""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path, PurePosixPath

import pytest

from npa.orchestration.npa_workflow.skypilot_render import render_task_run_script


@pytest.mark.parametrize("record_baked", [False, True])
def test_submitted_source_wins_over_baked_package(
    tmp_path: Path, record_baked: bool
) -> None:
    overlay = tmp_path / "npa-src-overlay"
    baked = tmp_path / "baked"
    for directory, marker in ((overlay / "src", "submitted"), (baked, "baked")):
        package = directory / "npa"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text(f"SOURCE = {marker!r}\n")
    (tmp_path / "npa-src-root").write_text(str(overlay))
    if record_baked:
        (tmp_path / "npa-baked-pythonpath").write_text(str(baked))
    script = render_task_run_script(
        [sys.executable, "-c", "import npa; assert npa.SOURCE == 'submitted'"]
    )
    # Follow the renderer's real control directory without using shared host files.
    shim = re.search(r"^  mkdir -p (.+/npa-shim)$", script, flags=re.MULTILINE)
    assert shim is not None
    control_prefix = str(PurePosixPath(shim[1]).with_name("npa"))
    script = script.replace(control_prefix, str(tmp_path / "npa"))
    script = script.replace("/etc/profile.d", str(tmp_path / "profiles"))
    environment = {**os.environ, "PYTHONPATH": str(baked), "HOME": str(tmp_path)}
    result = subprocess.run(
        ["bash", "-c", script], env=environment, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _write_partial_baked_sdk(project: Path, destination: Path) -> None:
    """Create the deliberately incomplete NPA tree copied by the LeRobot image."""

    source = destination / "src" / "npa"
    source.mkdir(parents=True)
    shutil.copy2(project / "pyproject.toml", destination / "pyproject.toml")
    for relative in (
        Path("__init__.py"),
        Path("clients/__init__.py"),
        Path("server/__init__.py"),
        Path("smoke/__init__.py"),
    ):
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        candidate = project / "src" / "npa" / relative
        if candidate.is_file():
            shutil.copy2(candidate, target)
        else:
            target.write_text("", encoding="utf-8")
    assert not (source / "workflow_build.py").exists()


def _copy_complete_declared_source(project: Path, destination: Path) -> None:
    shutil.copytree(project / "src", destination / "src")
    shutil.copy2(project / "pyproject.toml", destination / "pyproject.toml")
    assert (destination / "src" / "npa" / "workflow_build.py").is_file()


def _write_local_s3_sitecustomize(directory: Path) -> None:
    """Route only this fixture's S3 client to its local complete source tree."""

    directory.mkdir()
    (directory / "sitecustomize.py").write_text(
        textwrap.dedent(
            """\
            from pathlib import Path
            import os
            import shutil
            import boto3

            _real_client = boto3.client

            class _LocalS3:
                def list_objects_v2(self, **kwargs):
                    root = Path(os.environ['NPA_TEST_DECLARED_SOURCE'])
                    prefix = kwargs['Prefix'].rstrip('/')
                    return {'Contents': [
                        {'Key': prefix + '/' + path.relative_to(root).as_posix()}
                        for path in sorted(root.rglob('*')) if path.is_file()
                    ], 'IsTruncated': False}

                def download_file(self, bucket, key, destination):
                    prefix = os.environ['NPA_TEST_SOURCE_PREFIX'].strip('/')
                    relative = key.removeprefix(prefix).lstrip('/')
                    target = Path(destination)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(Path(os.environ['NPA_TEST_DECLARED_SOURCE']) / relative, target)

            def client(name, *args, **kwargs):
                if name == 's3' and os.environ.get('NPA_TEST_LOCAL_S3') == '1':
                    return _LocalS3()
                return _real_client(name, *args, **kwargs)

            boto3.client = client
            """
        ),
        encoding="utf-8",
    )


def _write_pip_intercepting_python(path: Path) -> None:
    """Forward normal Python calls while failing all editable installs locally."""

    path.write_text(
        "#!/bin/sh\n"
        'printf "%s:%s\\n" "$0" "$*" >> "$NPA_TEST_TRACE"\n'
        'if [ "$1" = "-m" ] && [ "$2" = "pip" ]; then\n'
        '  [ "$3" = "--version" ] && exit 0\n'
        '  if [ "$3" = "install" ]; then\n'
        '    npa_target=""; npa_next=0\n'
        '    for npa_arg in "$@"; do\n'
        '      [ "$npa_next" = 1 ] && { npa_target="$npa_arg"; break; }\n'
        '      [ "$npa_arg" = "-e" ] && npa_next=1\n'
        "    done\n"
        '    if [ "$npa_target" = "$NPA_TEST_PARTIAL" ]; then\n'
        "      echo 'OSError: Build script does not exist: src/npa/workflow_build.py' >&2\n"
        "      exit 1\n"
        "    fi\n"
        '    echo "unexpected pip install: $npa_target" >&2\n'
        "    exit 90\n"
        "  fi\n"
        "fi\n"
        f'exec {shlex.quote(sys.executable)} "$@"\n',
        encoding="utf-8",
    )
    path.chmod(0o755)


def _replace_setup_paths(setup: str, paths: dict[str, Path]) -> str:
    for original, replacement in paths.items():
        setup = setup.replace(original, str(replacement))
    return setup


def _run_setup(
    setup: str, environment: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", setup], env=environment, capture_output=True, text=True
    )


@dataclass(frozen=True)
class _OverlaySetupFixture:
    setup: str
    broken_setup: str
    environment: dict[str, str]
    overlay: Path
    source_root: Path
    recorded_python: Path
    system_python: Path
    vendor_python: Path
    trace: Path


def _render_overlay_setup(
    partial: Path,
    overlay: Path,
    source_root: Path,
    recorded_python: Path,
    vendor_python: Path,
    overlay_venv: Path,
) -> tuple[str, str]:
    from npa.orchestration.npa_workflow.skypilot_render import (
        default_npa_setup,
        render_vendor_interpreter_setup,
        tool_vendor_interpreters,
    )

    setup = default_npa_setup() + render_vendor_interpreter_setup(
        tool_vendor_interpreters("workbench.lerobot.policy_train")
    )
    setup = _replace_setup_paths(
        setup,
        {
            "/opt/npa": partial,
            "/opt/lerobot/venv/bin/python": vendor_python,
            str(Path(tempfile.gettempdir()) / "npa-src-overlay"): overlay,
            str(Path(tempfile.gettempdir()) / "npa-src-root"): source_root,
            str(Path(tempfile.gettempdir()) / "npa-python"): recorded_python,
            str(Path(tempfile.gettempdir()) / "npa-overlay-venv"): overlay_venv,
        },
    )
    branch = (
        '  if [ "$NPA_SRC_OVERLAY" = "1" ] && [ -n "$NPA_SRC_S3_URI" ]; then\n'
        "    : # stage the explicitly selected source before any image-local install\n"
    )
    return setup, setup.replace(branch, "", 1).replace(
        f"  elif [ -f {partial}/pyproject.toml ]",
        f"  if [ -f {partial}/pyproject.toml ]",
        1,
    )


def _overlay_environment(
    tmp_path: Path,
    binaries: Path,
    site: Path,
    declared: Path,
    partial: Path,
    trace: Path,
) -> dict[str, str]:
    environment = {
        **os.environ,
        "HOME": str(tmp_path / "home"),
        "PATH": f"{binaries}:/usr/bin:/bin",
        "PYTHONPATH": str(site),
        "PIP_CONFIG_FILE": os.devnull,
        "PIP_NO_INDEX": "1",
        "NPA_SRC_OVERLAY": "1",
        "NPA_SRC_S3_URI": "s3://test-bucket/npa-src",
        "NPA_TEST_DECLARED_SOURCE": str(declared),
        "NPA_TEST_SOURCE_PREFIX": "npa-src",
        "NPA_TEST_LOCAL_S3": "1",
        "NPA_TEST_PARTIAL": str(partial),
        "NPA_TEST_TRACE": str(trace),
    }
    environment.pop("NPA_BAKED_PYTHON", None)
    return environment


def _prepare_overlay_setup_fixture(tmp_path: Path) -> _OverlaySetupFixture:
    project = Path(__file__).resolve().parents[4] / "npa"
    partial, declared = tmp_path / "partial-baked-npa", tmp_path / "declared-source"
    _write_partial_baked_sdk(project, partial)
    _copy_complete_declared_source(project, declared)
    binaries, site = tmp_path / "bin", tmp_path / "sitecustomize"
    binaries.mkdir()
    _write_local_s3_sitecustomize(site)
    system_python, vendor_python = binaries / "python3", binaries / "vendor-python"
    _write_pip_intercepting_python(system_python)
    _write_pip_intercepting_python(vendor_python)
    overlay, source_root = tmp_path / "staged-overlay", tmp_path / "npa-src-root"
    recorded_python, trace = tmp_path / "npa-python", tmp_path / "interpreter-trace"
    setup, broken_setup = _render_overlay_setup(
        partial,
        overlay,
        source_root,
        recorded_python,
        vendor_python,
        tmp_path / "overlay-venv",
    )
    return _OverlaySetupFixture(
        setup,
        broken_setup,
        _overlay_environment(tmp_path, binaries, site, declared, partial, trace),
        overlay,
        source_root,
        recorded_python,
        system_python,
        vendor_python,
        trace,
    )


def test_overlay_setup_stages_declared_source_before_partial_baked_project(
    tmp_path: Path,
) -> None:
    """The guarded overlay must win before a partial image-local project."""

    fixture = _prepare_overlay_setup_fixture(tmp_path)
    failed = _run_setup(fixture.broken_setup, fixture.environment)
    assert failed.returncode != 0
    assert "Build script does not exist: src/npa/workflow_build.py" in failed.stderr
    assert not fixture.overlay.exists()
    fixture.trace.write_text("", encoding="utf-8")
    result = _run_setup(fixture.setup, fixture.environment)
    assert result.returncode == 0, result.stdout + result.stderr
    assert fixture.source_root.read_text(encoding="utf-8") == str(fixture.overlay)
    assert fixture.recorded_python.read_text(encoding="utf-8").strip() == str(
        fixture.vendor_python
    )
    assert (fixture.overlay / "src" / "npa" / "workflow_build.py").is_file()
    calls = fixture.trace.read_text(encoding="utf-8").splitlines()
    assert any(line.startswith(f"{fixture.system_python}:") for line in calls)
    assert any(line.startswith(f"{fixture.vendor_python}:") for line in calls)
    assert not any("-m pip" in line for line in calls), "\n".join(calls)
