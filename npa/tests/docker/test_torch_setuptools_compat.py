"""Bind the Torch runtime metadata repair to reviewed upstream bytes."""

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


def test_unknown_metadata_is_rejected(compat):
    with pytest.raises(RuntimeError, match=r"reviewed 2.11.0\+cu128 bytes"):
        compat.patched_metadata(compat.ORIGINAL_REQUIREMENT)


def test_exact_metadata_transformation_constants_match_review(compat):
    assert compat.ORIGINAL_SHA256 == (
        "de881a744ff66b27fffc338c1624660cf7c87d3019574ad71e3dda886b2a172b"
    )
    assert compat.PATCHED_SHA256 == (
        "63d6d6eaa7465a24d0178ca256c85831a4e9e90708301b889a263e1238b961ee"
    )
    assert compat.ORIGINAL_REQUIREMENT == b"Requires-Dist: setuptools<82\n"
    assert compat.PATCHED_REQUIREMENT == b"Requires-Dist: setuptools>=77.0.3\n"


def test_reviewed_transformation_replaces_only_runtime_bound(compat, monkeypatch):
    source = b"Name: torch\nRequires-Dist: setuptools<82\nRequires-Dist: filelock\n"
    expected = source.replace(compat.ORIGINAL_REQUIREMENT, compat.PATCHED_REQUIREMENT)
    monkeypatch.setattr(compat, "ORIGINAL_SHA256", hashlib.sha256(source).hexdigest())
    monkeypatch.setattr(compat, "PATCHED_SHA256", hashlib.sha256(expected).hexdigest())
    assert compat.patched_metadata(source) == expected


def test_unknown_torch_version_is_rejected(compat, monkeypatch):
    monkeypatch.setattr(
        compat.metadata, "distribution", lambda _: SimpleNamespace(version="2.12.0")
    )
    with pytest.raises(RuntimeError, match=r"requires Torch 2.11.0\+cu128"):
        compat.main()


def test_main_updates_exact_record_after_reviewed_patch(compat, monkeypatch, tmp_path):
    metadata_dir = tmp_path / "torch-2.11.0+cu128.dist-info"
    metadata_dir.mkdir()
    target = metadata_dir / "METADATA"
    record = metadata_dir / "RECORD"
    target.write_bytes(b"reviewed-original")
    record.write_text(f"{compat.METADATA_RELATIVE},sha256=old,17\n")
    distribution = SimpleNamespace(
        version=compat.TORCH_VERSION,
        locate_file=lambda _: tmp_path,
    )
    monkeypatch.setattr(compat.metadata, "distribution", lambda _: distribution)
    monkeypatch.setattr(compat, "patched_metadata", lambda _: b"reviewed-patched")

    compat.main()

    rows = list(csv.reader(io.StringIO(record.read_text())))
    digest = (
        base64.urlsafe_b64encode(hashlib.sha256(b"reviewed-patched").digest())
        .rstrip(b"=")
        .decode()
    )
    assert target.read_bytes() == b"reviewed-patched"
    assert rows == [[compat.METADATA_RELATIVE, "sha256=" + digest, "16"]]


def test_main_rejects_duplicate_record_without_mutation(compat, monkeypatch, tmp_path):
    metadata_dir = tmp_path / "torch-2.11.0+cu128.dist-info"
    metadata_dir.mkdir()
    target = metadata_dir / "METADATA"
    record = metadata_dir / "RECORD"
    target.write_bytes(b"reviewed-original")
    original_record = f"{compat.METADATA_RELATIVE},,\n" * 2
    record.write_text(original_record)
    distribution = SimpleNamespace(
        version=compat.TORCH_VERSION,
        locate_file=lambda _: tmp_path,
    )
    monkeypatch.setattr(compat.metadata, "distribution", lambda _: distribution)
    monkeypatch.setattr(compat, "patched_metadata", lambda _: b"reviewed-patched")

    with pytest.raises(RuntimeError, match="exactly one METADATA"):
        compat.main()
    assert target.read_bytes() == b"reviewed-original"
    assert record.read_text() == original_record


def test_python312_installer_repairs_then_probes_then_checks():
    common = Path(__file__).resolve().parents[2] / "docker/workbench/common"
    installer = (common / "install_isaac_runtime_base.sh").read_text()
    repair = '"$ISAAC_VENV/bin/python" "${COMMON_DIR}/torch_setuptools_compat.py"'
    manifest = (
        '"$ISAAC_VENV/bin/python" "${COMMON_DIR}/backport_setuptools_manifest.py"'
    )
    probe = "from torch.utils.cpp_extension import BuildExtension"
    final_check = '"$ISAAC_VENV/bin/python" -m pip check'
    assert installer.index(repair) < installer.index(manifest)
    assert installer.index(manifest) < installer.index(probe)
    assert installer.index(probe) < installer.rindex(final_check)
    assert "setuptools==84.2.1" in (common / "isaac3-oss-deps.txt").read_text()
