"""Keep the Pillow compatibility repair bound to reviewed release bytes."""

import base64
import csv
import hashlib
import importlib.util
import io
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


def test_published_and_patched_metadata_hashes_are_exact(compat):
    assert (
        compat.ORIGINAL_SHA256
        == "ba08814030a33c196589cb232d87518b2be874b342e8e5f6dfefb464e52e5569"
    )
    assert (
        compat.PATCHED_SHA256
        == "40bc8558f5dba3d3c3d31831b8f32253e1b9ad74527ced0242130c201a2e2595"
    )


def test_unknown_moviepy_version_is_rejected(compat, monkeypatch):
    monkeypatch.setattr(
        compat.metadata, "distribution", lambda _: SimpleNamespace(version="2.2.2")
    )
    with pytest.raises(RuntimeError, match="requires MoviePy 2.2.1"):
        compat.main()


@pytest.mark.parametrize("pillow_version", ["11.3.0", "12.3.0", "13.0.1", "14.0.0"])
def test_unqualified_pillow_version_is_rejected(compat, monkeypatch, pillow_version):
    monkeypatch.setattr(
        compat.metadata, "distribution", lambda _: SimpleNamespace(version="2.2.1")
    )
    monkeypatch.setattr(compat.metadata, "version", lambda _: pillow_version)
    with pytest.raises(RuntimeError, match="requires Pillow 13.0.0"):
        compat.main()


def test_exact_pillow13_rebinds_metadata_record(compat, monkeypatch, tmp_path):
    original = (
        b"Metadata-Version: 2.4\n"
        b"Name: moviepy\n"
        b"Version: 2.2.1\n"
        b"Requires-Dist: pillow<12.0,>=9.2.0\n"
    )
    patched = original.replace(
        b"Requires-Dist: pillow<12.0,>=9.2.0",
        b"Requires-Dist: pillow==13.0.0",
    )
    monkeypatch.setattr(compat, "ORIGINAL_SHA256", hashlib.sha256(original).hexdigest())
    monkeypatch.setattr(compat, "PATCHED_SHA256", hashlib.sha256(patched).hexdigest())
    relative = "moviepy-2.2.1.dist-info/METADATA"
    target = tmp_path / relative
    target.parent.mkdir()
    target.write_bytes(original)
    record = target.parent / "RECORD"
    record.write_text(f"{relative},sha256=old,{len(original)}\n")
    distribution = SimpleNamespace(
        version="2.2.1",
        locate_file=lambda _: tmp_path,
    )
    monkeypatch.setattr(compat.metadata, "distribution", lambda _: distribution)
    monkeypatch.setattr(compat.metadata, "version", lambda _: "13.0.0")

    compat.main()

    assert target.read_bytes() == patched
    rows = list(csv.reader(io.StringIO(record.read_text())))
    expected_digest = (
        base64.urlsafe_b64encode(hashlib.sha256(patched).digest()).rstrip(b"=").decode()
    )
    assert rows == [[relative, f"sha256={expected_digest}", str(len(patched))]]


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
