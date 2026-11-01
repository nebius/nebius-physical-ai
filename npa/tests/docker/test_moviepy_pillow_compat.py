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


def test_patched_metadata_is_idempotent(compat, monkeypatch):
    original = b"Requires-Dist: pillow<12.0,>=9.2.0\n"
    patched = b"Requires-Dist: pillow==13.0.0\n"
    monkeypatch.setattr(compat, "ORIGINAL_SHA256", hashlib.sha256(original).hexdigest())
    monkeypatch.setattr(compat, "PATCHED_SHA256", hashlib.sha256(patched).hexdigest())
    assert compat.patched_metadata(original) == patched
    assert compat.patched_metadata(patched) == patched


@pytest.mark.parametrize(
    "record",
    (
        b"other.py,sha256=other,1\n",
        (
            b"moviepy-2.2.1.dist-info/METADATA,sha256=one,1\n"
            b"moviepy-2.2.1.dist-info/METADATA,sha256=two,2\n"
        ),
    ),
)
def test_missing_or_duplicate_metadata_record_is_rejected(compat, record):
    with pytest.raises(RuntimeError, match="exactly one METADATA entry"):
        compat.patched_record(
            record,
            "moviepy-2.2.1.dist-info/METADATA",
            b"patched metadata",
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
    before = (
        b"moviepy/__init__.py,sha256=unchanged,123\r\n"
        + f"{relative},sha256=old,{len(original)}\r\n".encode()
        + b"moviepy-2.2.1.dist-info/RECORD,,\r\n"
    )
    record.write_bytes(before)
    distribution = SimpleNamespace(
        version="2.2.1",
        locate_file=lambda _: tmp_path,
    )
    monkeypatch.setattr(compat.metadata, "distribution", lambda _: distribution)
    monkeypatch.setattr(compat.metadata, "version", lambda _: "13.0.0")

    compat.main()

    assert target.read_bytes() == patched
    first_record = record.read_bytes()
    rows = list(csv.reader(io.StringIO(first_record.decode())))
    expected_digest = (
        base64.urlsafe_b64encode(hashlib.sha256(patched).digest()).rstrip(b"=").decode()
    )
    assert rows == [
        ["moviepy/__init__.py", "sha256=unchanged", "123"],
        [relative, f"sha256={expected_digest}", str(len(patched))],
        ["moviepy-2.2.1.dist-info/RECORD", "", ""],
    ]
    assert (
        first_record.splitlines(keepends=True)[0] == before.splitlines(keepends=True)[0]
    )
    assert (
        first_record.splitlines(keepends=True)[2] == before.splitlines(keepends=True)[2]
    )
    assert first_record.count(b"\r\n") == 3

    compat.main()

    assert target.read_bytes() == patched
    assert record.read_bytes() == first_record


@pytest.mark.parametrize(
    "closure", ["isaac", "isaac3", "leisaac", "unreviewed-moviepy"]
)
def test_shared_installer_repairs_only_closures_that_include_moviepy(tmp_path, closure):
    common = Path(__file__).resolve().parents[2] / "docker/workbench/common"
    python_minor = "3.12" if closure == "isaac3" else "3.11"
    requirements_file = (
        "isaac3-oss-deps.txt" if closure == "isaac3" else "isaac-oss-deps.txt"
    )
    requirements = (common / requirements_file).read_text()
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
        '#!/bin/sh\nprintf "%s\\n" "$@" >> "$REPAIR_CALL"\nexit "$REPAIR_EXIT"\n'
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
            "ISAAC_PYTHON_MINOR": python_minor,
            "ISAAC_VENV": str(interpreter.parent.parent),
            "OSS_DEPS_FILE": str(selected),
            "COMMON_DIR": str(common),
            "SKIP_TORCH": "1",
            "REPAIR_CALL": str(calls),
            "REPAIR_EXIT": "23" if closure == "unreviewed-moviepy" else "0",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == (23 if closure == "unreviewed-moviepy" else 0)
    moviepy_repair = str(common / "moviepy_pillow_compat.py")
    if closure in {"isaac3", "leisaac"}:
        assert (
            not calls.exists() or moviepy_repair not in calls.read_text().splitlines()
        )
    else:
        assert calls.read_text().splitlines() == [moviepy_repair]


def _assert_groot_keeps_qualified_moviepy_pillow_state(text):
    assert '"pillow>=13.0.0,<14.0"' not in text
    assert text.count('"pillow==13.0.0"') == 2
    overlay = text[text.index("RUN /opt/isaac-lab/venv/bin/python -m pip install") :]
    repair = "/opt/npa/docker/workbench/common/moviepy_pillow_compat.py"
    final_check = "/opt/isaac-lab/venv/bin/python -m pip check"
    assert repair in overlay
    assert final_check in overlay
    assert overlay.index('"pillow==13.0.0"') < overlay.index(repair)
    assert overlay.index(repair) < overlay.index(final_check)


def test_groot_revalidates_moviepy_after_its_pillow_overlay():
    dockerfile = (
        Path(__file__).resolve().parents[2] / "docker/workbench/groot/Dockerfile"
    ).read_text()
    _assert_groot_keeps_qualified_moviepy_pillow_state(dockerfile)


def test_groot_moviepy_post_overlay_contract_rejects_mutations():
    dockerfile = (
        Path(__file__).resolve().parents[2] / "docker/workbench/groot/Dockerfile"
    ).read_text()
    mutations = (
        dockerfile.replace('"pillow==13.0.0"', '"pillow>=13.0.0,<14.0"', 1),
        dockerfile.replace(
            "/opt/npa/docker/workbench/common/moviepy_pillow_compat.py",
            "/opt/npa/docker/workbench/common/omitted_moviepy_repair.py",
            1,
        ),
        dockerfile.replace(
            "/opt/isaac-lab/venv/bin/python -m pip check",
            "/opt/isaac-lab/venv/bin/python -m pip --version",
            1,
        ),
    )
    for mutation in mutations:
        with pytest.raises(AssertionError):
            _assert_groot_keeps_qualified_moviepy_pillow_state(mutation)
