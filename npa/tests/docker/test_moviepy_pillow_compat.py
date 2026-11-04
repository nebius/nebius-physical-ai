"""Keep the Pillow compatibility repair bound to reviewed release bytes."""

import base64
import csv
import hashlib
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import sys
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


@pytest.fixture
def selector():
    path = (
        Path(__file__).resolve().parents[2]
        / "docker/workbench/common/select_moviepy_pillow_closure.py"
    )
    spec = importlib.util.spec_from_file_location("moviepy_selector", path)
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


@pytest.mark.parametrize(
    "pillow_version",
    ["11.3.0", "12.3.0", "13", "13.0", "13.0.0.0", "13.0.1", "14.0.0"],
)
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
            "MOVIEPY_PILLOW_QUALIFIED": ("1" if closure == "isaac" else "0"),
            "ISAAC_PYTHON_MINOR": python_minor,
            "ISAAC_VENV": str(interpreter.parent.parent),
            "OSS_DEPS_FILE": str(selected),
            "COMMON_DIR": str(common),
            "SKIP_TORCH": "1",
            "REPAIR_CALL": str(calls),
            "REPAIR_EXIT": "0",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    moviepy_repair = str(common / "moviepy_pillow_compat.py")
    if closure in {"isaac3", "leisaac", "unreviewed-moviepy"}:
        assert (
            not calls.exists() or moviepy_repair not in calls.read_text().splitlines()
        )
    else:
        assert calls.read_text().splitlines() == [moviepy_repair]


def test_moviepy_pillow_wheels_are_exactly_hash_locked():
    common = Path(__file__).resolve().parents[2] / "docker/workbench/common"
    lock = (common / "moviepy-pillow311-requirements.lock").read_text()
    assert lock == (
        "# Exact PyPI wheels qualified for the Python 3.11 linux/amd64 Isaac closure.\n"
        "moviepy==2.2.1 \\\n"
        "    --hash=sha256:"
        "6b56803fec2ac54b557404126ac1160e65448e03798fa282bd23e8fab3795060\n"
        "pillow==13.0.0 \\\n"
        "    --hash=sha256:"
        "a625df9a2a5597fa5e5266e01aefc82604bb85d291e3547981e863560d75da6c\n"
    )

    installer = (common / "install_isaac_runtime_base.sh").read_text()
    assert "select_moviepy_pillow_closure.py" in installer
    assert "--force-reinstall" in installer
    assert "--only-binary=:all:" in installer
    assert "--require-hashes" in installer
    assert '-r "${COMMON_DIR}/moviepy-pillow311-requirements.lock"' in installer

    scanner_path = (
        Path(__file__).resolve().parents[3] / "scripts/security_dependencies.py"
    )
    spec = importlib.util.spec_from_file_location("security_dependencies", scanner_path)
    assert spec and spec.loader
    scanner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scanner)
    assert scanner._is_python_dependency_manifest(
        common / "moviepy-pillow311-requirements.lock"
    )


def test_selector_uses_canonical_identity_and_emits_reviewed_closure(selector):
    selected, qualified = selector.select_closure(
        "# reviewed\n"
        "ImageIO_FFMPEG==0.6.0\n"
        "MoviePy==2.2.1\n"
        "PILLOW==13.0.0\n"
        "requests==2.34.2\n",
        "3.11",
    )

    assert qualified is True
    assert selected == "# reviewed\nrequests==2.34.2\n"


def test_selector_keeps_pillow_when_moviepy_is_absent(selector):
    selected, qualified = selector.select_closure(
        "imageio-ffmpeg==0.6.0\npillow==13.0.0\nrequests==2.34.2\n",
        "3.11",
    )

    assert qualified is False
    assert selected == "pillow==13.0.0\nrequests==2.34.2\n"


def test_selector_classifies_the_committed_isaac_locks(selector):
    """Exercise the selector against every real closure shape on its build path."""
    common = Path(__file__).resolve().parents[2] / "docker/workbench/common"
    isaac = (common / "isaac-oss-deps.txt").read_text()
    isaac3 = (common / "isaac3-oss-deps.txt").read_text()
    leisaac = "\n".join(
        line
        for line in isaac.splitlines()
        if not line.startswith(("moviepy==", "imageio-ffmpeg=="))
    )
    if isaac.endswith("\n"):
        leisaac += "\n"

    selected_isaac, qualified_isaac = selector.select_closure(isaac, "3.11")
    selected_isaac3, qualified_isaac3 = selector.select_closure(isaac3, "3.12")
    selected_leisaac, qualified_leisaac = selector.select_closure(leisaac, "3.11")

    assert qualified_isaac is True
    assert "moviepy==" not in selected_isaac and "pillow==" not in selected_isaac
    assert qualified_isaac3 is False
    assert "moviepy==1.0.3" in selected_isaac3
    assert qualified_leisaac is False
    assert "moviepy==" not in selected_leisaac
    assert "pillow==13.0.0" in selected_leisaac


@pytest.mark.parametrize(
    ("requirements", "message"),
    (
        (
            "MoviePy==2.2.2\npillow==13.0.0\n",
            "requires exactly moviepy==2.2.1",
        ),
        (
            "moviepy==2.2.1\npillow==12.0.0\n",
            "requires exactly pillow==13.0.0",
        ),
        (
            "moviepy @ https://example.invalid/moviepy.whl\npillow==13.0.0\n",
            "direct references",
        ),
        ("-r nested.txt\n", "includes"),
        ("moviepy===2.2.1\n", "alternate comparison forms"),
        (
            "moviepy==2.2.1\nMoviePy==2.2.1\npillow==13.0.0\n",
            "duplicate canonical requirement",
        ),
        (
            "moviepy==2.2.1; python_version >= '3.11'\npillow==13.0.0\n",
            "markers",
        ),
    ),
)
def test_selector_rejects_ambiguous_or_unqualified_forms(
    selector, requirements, message
):
    with pytest.raises(ValueError, match=message):
        selector.select_closure(requirements, "3.11")


def test_unknown_moviepy_closure_fails_before_install(tmp_path):
    common = Path(__file__).resolve().parents[2] / "docker/workbench/common"
    output = tmp_path / "selected.txt"
    marker = tmp_path / "qualified.txt"
    output.write_text("stale output")
    marker.write_text("stale marker")
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("moviepy==2.2.2\npillow==13.0.0\n")

    result = subprocess.run(
        [
            sys.executable,
            str(common / "select_moviepy_pillow_closure.py"),
            "--requirements",
            str(requirements),
            "--output",
            str(output),
            "--qualified-marker",
            str(marker),
            "--python-minor",
            "3.11",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "requires exactly moviepy==2.2.1" in result.stderr
    assert not output.exists()
    assert not marker.exists()


def _assert_groot_keeps_qualified_moviepy_pillow_state(text):
    assert '"pillow>=13.0.0,<14.0"' not in text
    assert text.count('"pillow==13.0.0"') == 1
    overlay = text[text.index("RUN /opt/isaac-lab/venv/bin/python -m pip install") :]
    repair = "/opt/npa/docker/workbench/common/moviepy_pillow_compat.py"
    assert '"pillow==13.0.0"' not in overlay
    assert repair in overlay
    # This overlay intentionally uses --no-deps against a vendor environment whose
    # full dependency graph is not internally consistent. The bounded MoviePy/Pillow
    # verifier is the applicable postcondition; a venv-wide pip check would fail on
    # unrelated packages such as wheel/packaging and make GR00T unbuildable.
    assert "/opt/isaac-lab/venv/bin/python -m pip check" not in overlay


def test_groot_revalidates_moviepy_without_replacing_locked_pillow():
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
            '      "python-multipart>=0.0.30" \\\n',
            '      "pillow==13.0.0" \\\n      "python-multipart>=0.0.30" \\\n',
            1,
        ),
        dockerfile.replace(
            "/opt/npa/docker/workbench/common/moviepy_pillow_compat.py",
            "/opt/npa/docker/workbench/common/omitted_moviepy_repair.py",
            1,
        ),
    )
    for mutation in mutations:
        with pytest.raises(AssertionError):
            _assert_groot_keeps_qualified_moviepy_pillow_state(mutation)
