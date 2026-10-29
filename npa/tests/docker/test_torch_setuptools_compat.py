"""Keep the Torch/Setuptools compatibility repair exact and RECORD-correct."""

import base64
import csv
import hashlib
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def compat():
    path = (
        Path(__file__).resolve().parents[2]
        / "docker/workbench/common/torch_setuptools_compat.py"
    )
    spec = importlib.util.spec_from_file_location("torch_setuptools_compat", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reviewed_metadata_constants_are_pinned(compat):
    assert compat.ORIGINAL_SHA256 == (
        "de881a744ff66b27fffc338c1624660cf7c87d3019574ad71e3dda886b2a172b"
    )
    assert compat.PATCHED_SHA256 == (
        "63d6d6eaa7465a24d0178ca256c85831a4e9e90708301b889a263e1238b961ee"
    )
    assert compat.ORIGINAL_REQUIREMENT == b"Requires-Dist: setuptools<82"
    assert compat.PATCHED_REQUIREMENT == b"Requires-Dist: setuptools>=77.0.3"


def test_unknown_metadata_is_rejected(compat):
    with pytest.raises(RuntimeError, match="reviewed 2.11.0\\+cu128 bytes"):
        compat.patched_metadata(b"Requires-Dist: setuptools<82\n")


def test_patched_metadata_updates_exact_record_row(compat, monkeypatch, tmp_path):
    original = b"Name: torch\nRequires-Dist: setuptools<82\n"
    patched = original.replace(compat.ORIGINAL_REQUIREMENT, compat.PATCHED_REQUIREMENT)
    monkeypatch.setattr(compat, "ORIGINAL_SHA256", hashlib.sha256(original).hexdigest())
    monkeypatch.setattr(compat, "PATCHED_SHA256", hashlib.sha256(patched).hexdigest())
    relative = "torch-2.11.0+cu128.dist-info/METADATA"
    dist_info = tmp_path / "torch-2.11.0+cu128.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_bytes(original)
    (dist_info / "RECORD").write_text(f"{relative},old,1\n")
    monkeypatch.setattr(
        compat.metadata,
        "distribution",
        lambda _: SimpleNamespace(
            version=compat.TORCH_VERSION, locate_file=lambda _: tmp_path
        ),
    )
    monkeypatch.setattr(compat.metadata, "version", lambda _: compat.SETUPTOOLS_VERSION)

    compat.main()

    assert (dist_info / "METADATA").read_bytes() == patched
    rows = list(csv.reader(io.StringIO((dist_info / "RECORD").read_text())))
    digest = base64.urlsafe_b64encode(hashlib.sha256(patched).digest()).rstrip(b"=")
    assert rows == [[relative, "sha256=" + digest.decode(), str(len(patched))]]


@pytest.mark.parametrize(
    ("torch_version", "setuptools_version", "message"),
    [
        ("2.11.1+cu128", "84.2.1", "requires Torch"),
        ("2.11.0+cu128", "84.1.0", "requires secure Setuptools"),
    ],
)
def test_unreviewed_versions_are_rejected(
    compat, monkeypatch, torch_version, setuptools_version, message
):
    monkeypatch.setattr(
        compat.metadata,
        "distribution",
        lambda _: SimpleNamespace(version=torch_version),
    )
    monkeypatch.setattr(compat.metadata, "version", lambda _: setuptools_version)
    with pytest.raises(RuntimeError, match=message):
        compat.main()


def test_installer_repairs_only_the_exact_isaac3_cu128_closure():
    common = Path(__file__).resolve().parents[2] / "docker/workbench/common"
    installer = (common / "install_isaac_runtime_base.sh").read_text()
    repair = '"$ISAAC_VENV/bin/python" "${COMMON_DIR}/torch_setuptools_compat.py"'
    manifest = (
        '"$ISAAC_VENV/bin/python" "${COMMON_DIR}/backport_setuptools_manifest.py"'
    )
    assert repair in installer
    assert '[ "$TORCH_VERSION" = 2.11.0 ]' in installer
    assert '[[ "$TORCH_INDEX_URL" == */cu128 ]]' in installer
    assert installer.index(manifest) < installer.index(repair)
    assert installer.index(repair) < installer.rindex(
        '"$ISAAC_VENV/bin/python" -m pip check'
    )
