"""Keep the Pillow compatibility repair bound to reviewed release bytes."""

import importlib.util
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest


@pytest.fixture
def compat():
    path = (
        Path(__file__).resolve().parents[2]
        / "docker/workbench/common/moviepy_pillow_compat.py"
    )
    spec = importlib.util.spec_from_file_location("moviepy_compat", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_unknown_metadata_is_rejected(compat):
    with pytest.raises(RuntimeError, match="reviewed 2.2.1 bytes"):
        compat.patched_metadata(b"Requires-Dist: pillow<12.0,>=9.2.0\n")


def test_unknown_moviepy_version_is_rejected(compat, monkeypatch):
    monkeypatch.setattr(
        compat.metadata, "distribution", lambda _: SimpleNamespace(version="2.2.2")
    )
    with pytest.raises(RuntimeError, match="requires MoviePy 2.2.1"):
        compat.main()


@pytest.mark.parametrize("pillow_version", ["11.3.0", "12.2.0", "13.0.0"])
def test_unpatched_or_unreviewed_pillow_version_is_rejected(
    compat, monkeypatch, pillow_version
):
    monkeypatch.setattr(
        compat.metadata, "distribution", lambda _: SimpleNamespace(version="2.2.1")
    )
    monkeypatch.setattr(compat.metadata, "version", lambda _: pillow_version)
    with pytest.raises(RuntimeError, match="requires patched Pillow"):
        compat.main()


@pytest.mark.parametrize("closure", ["isaac", "leisaac", "unreviewed-moviepy"])
def test_shared_installer_repairs_only_closures_that_include_moviepy(tmp_path, closure):
    common = Path(__file__).resolve().parents[2] / "docker/workbench/common"
    requirements = (common / "isaac-oss-deps.txt").read_text()
    if closure == "leisaac":
        requirements = "\n".join(
            line
            for line in requirements.splitlines()
            if not line.startswith(("moviepy==", "imageio-ffmpeg=="))
        )
    elif closure == "unreviewed-moviepy":
        requirements = requirements.replace("moviepy==2.2.1", "moviepy==2.2.2")
    selected = tmp_path / "selected-closure.txt"
    selected.write_text(requirements)
    interpreter = tmp_path / "venv/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$@" > "$REPAIR_CALL"\nexit "$REPAIR_EXIT"\n'
    )
    interpreter.chmod(0o700)
    calls = tmp_path / "repair-call.txt"
    installer = (common / "install_isaac_runtime_base.sh").read_text()
    hooks = installer.split(
        '"$ISAAC_VENV/bin/python" -m pip uninstall --yes wheel\n', 1
    )[1]
    hooks = hooks.split('"$ISAAC_VENV/bin/python" -m pip check', 1)[0]
    result = subprocess.run(
        ["bash", "-eu", "-c", hooks],
        env={
            **os.environ,
            "ISAAC_PYTHON_MINOR": "3.11",
            "ISAAC_VENV": str(interpreter.parent.parent),
            "OSS_DEPS_FILE": str(selected),
            "COMMON_DIR": str(common),
            "REPAIR_CALL": str(calls),
            "REPAIR_EXIT": "23" if closure == "unreviewed-moviepy" else "0",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == (23 if closure == "unreviewed-moviepy" else 0)
    if closure == "leisaac":
        assert not calls.exists()
    else:
        assert calls.read_text().strip() == str(common / "moviepy_pillow_compat.py")
