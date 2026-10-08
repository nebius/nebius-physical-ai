"""Protect exact-source metadata exclusions and the stock Genesis native CPU gate."""

from __future__ import annotations

import base64
import csv
import hashlib
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
RECIPE = ROOT / "npa/docker/workbench/genesis"


@pytest.fixture
def runtime_source():
    spec = importlib.util.spec_from_file_location(
        "genesis_public_runtime", RECIPE / "prepare_public_runtime.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _installed_source(root, module, monkeypatch):
    sources = {}
    for name, source in module.SOURCES.items():
        directory = root / name
        directory.mkdir()
        original = (
            "Metadata-Version: 2.4\nLicense-File: LICENSE\n"
            "Requires-Dist: numpy>=1.24\n"
            + "".join(
                f"Requires-Dist: {dependency}>=0.1\n"
                for dependency in sorted(source["excluded_dependencies"])
            )
        ).encode()
        (directory / "METADATA").write_bytes(original)
        (directory / "LICENSE").write_text("retained license notice")
        (directory / "RECORD").write_text(
            f"{name}/METADATA,sha256=original,{len(original)}\n"
            f"{name}/LICENSE,sha256=untouched,23\n"
        )
        sources[name] = {
            **source,
            "metadata_sha256": hashlib.sha256(original).hexdigest(),
        }
    monkeypatch.setattr(module, "SOURCES", sources)


def test_exclusions_update_record_and_preserve_other_edges_and_notices(
    runtime_source, monkeypatch, tmp_path
):
    _installed_source(tmp_path, runtime_source, monkeypatch)
    receipt = runtime_source._prepare(tmp_path, check_only=False)
    assert len(receipt) == 2
    for name, source in runtime_source.SOURCES.items():
        directory = tmp_path / name
        metadata = (directory / "METADATA").read_bytes()
        assert b"Requires-Dist: numpy>=1.24" in metadata
        assert b"License-File: LICENSE" in metadata
        assert (directory / "LICENSE").read_text() == "retained license notice"
        for dependency in source["excluded_dependencies"]:
            assert f"Requires-Dist: {dependency}".encode() not in metadata
        rows = list(csv.reader((directory / "RECORD").read_text().splitlines()))
        digest = (
            base64.urlsafe_b64encode(hashlib.sha256(metadata).digest())
            .decode()
            .rstrip("=")
        )
        assert rows[0][1:] == ["sha256=" + digest, str(len(metadata))]
        assert rows[1][1:] == ["sha256=untouched", "23"]


def test_unknown_second_source_fails_before_any_metadata_changes(
    runtime_source, monkeypatch, tmp_path
):
    _installed_source(tmp_path, runtime_source, monkeypatch)
    originals = {path: path.read_bytes() for path in tmp_path.glob("*/METADATA")}
    second = list(originals)[1]
    second.write_bytes(originals[second] + b"Requires-Dist: unexpected\n")
    originals[second] = second.read_bytes()
    with pytest.raises(RuntimeError, match="unreviewed dependency metadata"):
        runtime_source._prepare(tmp_path, check_only=False)
    assert all(path.read_bytes() == content for path, content in originals.items())


def test_apply_requires_excluded_distributions_to_be_uninstalled(
    runtime_source, monkeypatch, tmp_path
):
    _installed_source(tmp_path, runtime_source, monkeypatch)
    (tmp_path / "tetgen-0.8.2.dist-info").mkdir()
    original = next(tmp_path.glob("*/METADATA")).read_bytes()
    assert runtime_source._prepare(tmp_path, check_only=True)
    with pytest.raises(RuntimeError, match="excluded distribution is installed"):
        runtime_source._prepare(tmp_path, check_only=False)
    assert next(tmp_path.glob("*/METADATA")).read_bytes() == original


def test_stock_recipe_enforces_reduced_closure_in_the_install_layer():
    text = (RECIPE / "Dockerfile").read_text()
    installation = text.split("RUN python3 -m venv", 1)[1].split("\nENV ", 1)[0]
    assert "NPA_GENESIS_SUPPORTED_STUDENT_POLICIES=act" in text
    assert "--check-only" in installation
    assert "pip uninstall -y tetgen diffusers wandb torchcodec" in installation
    assert "public-runtime-dependencies.json" in installation
    assert "pip check" in installation
    assert "diffusers>=0.38.0" not in text
    assert "RUN python /opt/npa/docker/workbench/genesis/smoke_public_cpu.py" in text
    smoke = (RECIPE / "smoke_public_cpu.py").read_text()
    assert (
        "from lerobot.policies.factory import make_policy, make_pre_post_processors"
        in smoke
    )
    assert "policy = make_policy(configuration)" in smoke
