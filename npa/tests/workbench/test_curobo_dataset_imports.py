"""Real import machinery binds verified benchmark data without ambient paths."""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
import sys

import pytest

from npa.workbench.curobo import runner
from npa.workbench.curobo.artifacts import CuroboError
from npa.workbench.curobo.schemas import DATASET_REVISION, SOURCE_REVISION


@pytest.fixture
def source_trees(tmp_path, monkeypatch):
    source = tmp_path / "source"
    dataset = tmp_path / "dataset"
    package = dataset / "robometrics"
    package.mkdir(parents=True)
    (source / "benchmark").mkdir(parents=True)
    (source / "NPA_SOURCE_REVISION").write_text(SOURCE_REVISION)
    (dataset / "NPA_SOURCE_REVISION").write_text(DATASET_REVISION)
    (package / "__init__.py").write_text("")
    (package / "datasets.py").write_text(
        "from pathlib import Path\n"
        "def motion_benchmaker_raw():\n"
        "    return (Path(__file__).parent / 'content/dataset/test.yaml').read_text()\n"
    )
    (source / "benchmark/motion_plan_benchmark.py").write_text(
        "from robometrics.datasets import motion_benchmaker_raw\n"
        "OBSERVED_DATA = motion_benchmaker_raw()\n"
    )
    (package / "content/dataset").mkdir(parents=True)
    data = b"verified synthetic importer fixture\n"
    (package / "content/dataset/test.yaml").write_bytes(data)
    monkeypatch.setattr(runner, "DATASET_FILES", {"test": ("test.yaml", hashlib.sha256(data).hexdigest())})
    monkeypatch.setattr(runner, "version", lambda _: "0.8.0")
    monkeypatch.setenv("NPA_CUROBO_SOURCE", str(source))
    monkeypatch.setenv("NPA_CUROBO_DATASET_SOURCE", str(dataset))
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.setattr(sys, "path", list(sys.path))
    saved = {name: module for name, module in sys.modules.items() if name == "robometrics" or name.startswith("robometrics.")}
    for name in saved:
        sys.modules.pop(name)
    yield source, dataset
    for name in tuple(sys.modules):
        if name == "robometrics" or name.startswith("robometrics."):
            sys.modules.pop(name)
    sys.modules.update(saved)


def test_scrubbed_pythonpath_loads_only_verified_raw_dataset(source_trees):
    source, dataset = source_trees
    before = {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
    before.update({p: p.read_bytes() for p in dataset.rglob("*") if p.is_file()})
    module = runner._benchmark_module()
    assert module.OBSERVED_DATA == "verified synthetic importer fixture\n"
    assert Path(importlib.import_module("robometrics.datasets").__file__) == dataset / "robometrics/datasets.py"
    assert all(path.read_bytes() == payload for path, payload in before.items())


def test_ambient_import_path_cannot_shadow_verified_dataset(source_trees, tmp_path):
    _, dataset = source_trees
    shadow = tmp_path / "shadow"
    (shadow / "robometrics").mkdir(parents=True)
    (shadow / "robometrics/__init__.py").write_text("raise AssertionError('ambient package executed')\n")
    sys.path.insert(0, str(shadow))
    assert runner._benchmark_module().OBSERVED_DATA == "verified synthetic importer fixture\n"
    assert sys.path[0] == str(dataset)


def test_cached_foreign_package_is_rejected_before_upstream_execution(source_trees, tmp_path):
    _, dataset = source_trees
    shadow = tmp_path / "shadow"
    (shadow / "robometrics").mkdir(parents=True)
    (shadow / "robometrics/__init__.py").write_text("")
    sys.path.insert(0, str(shadow))
    foreign = importlib.import_module("robometrics")
    before_path = list(sys.path)
    with pytest.raises(CuroboError, match="outside the verified dataset tree"):
        runner._benchmark_module()
    assert sys.modules["robometrics"] is foreign
    assert str(dataset) not in sys.path and sys.path == before_path


@pytest.mark.parametrize("mutation", ["revision", "bytes"])
def test_bad_dataset_identity_fails_before_import_or_path_change(source_trees, mutation):
    _, dataset = source_trees
    target = dataset / ("NPA_SOURCE_REVISION" if mutation == "revision" else "robometrics/content/dataset/test.yaml")
    target.write_text("different identity")
    before_path = list(sys.path)
    with pytest.raises(CuroboError, match="(revision|YAML bytes)"):
        runner._benchmark_module()
    assert "robometrics" not in sys.modules
    assert sys.path == before_path
