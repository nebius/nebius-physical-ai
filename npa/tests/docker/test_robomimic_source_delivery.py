"""Prove source delivery rejects missing archives and uncovered package versions."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
IMAGE = ROOT / "npa/docker/workbench/robomimic"
SPEC = importlib.util.spec_from_file_location(
    "robomimic_source_delivery", IMAGE / "source_delivery.py"
)
assert SPEC and SPEC.loader
delivery = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(delivery)


def _material(tmp_path: Path):
    artifact = {"filename": "demo.tar.xz", "sha256": "a" * 64, "size": 5}
    package = {
        "name": "demo-bin",
        "version": "1",
        "architecture": "amd64",
        "source": "demo",
        "source_version": "1",
    }
    lock = {
        "parent_diff_ids": ["sha256:" + "b" * 64],
        "packages": [package],
        "sources": [{"name": "demo", "version": "1", "artifacts": [artifact]}],
        "cpython": artifact,
    }
    path = tmp_path / "lock.json"
    path.write_text(json.dumps(lock))
    files = {
        f"{delivery.SOURCE_ROOT}/{'a' * 64}/demo.tar.xz": {
            "type": "0",
            "sha256": "a" * 64,
            "size": 5,
        },
        f"{delivery.SOURCE_ROOT}/corresponding-source.lock.json": {
            "type": "0",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size": path.stat().st_size,
        },
    }
    inventory = {
        "archive_sha256": "c" * 64,
        "final_files": files,
        "layers": [{"diff_id": "sha256:" + "b" * 64, "debian_packages": [package]}],
    }
    return lock, path, inventory


@pytest.mark.parametrize(
    "mutation", ["archive", "version", "parent", "source", "symlink"]
)
def test_source_delivery_rejects_incomplete_correspondence(
    tmp_path, monkeypatch, mutation
):
    lock, path, original = _material(tmp_path)
    inventory = copy.deepcopy(original)
    if mutation == "archive":
        next(iter(inventory["final_files"].values()))["sha256"] = "d" * 64
    elif mutation == "version":
        inventory["layers"][0]["debian_packages"][0]["version"] = "2"
    elif mutation == "parent":
        inventory["layers"][0]["diff_id"] = "sha256:" + "e" * 64
    elif mutation == "source":
        lock["sources"] = []
    else:
        next(iter(inventory["final_files"].values()))["type"] = "2"
    monkeypatch.setattr(delivery, "_inventory", lambda _archive: inventory)
    with pytest.raises(ValueError):
        delivery._verify(lock, path, tmp_path / "image.tar")


def test_source_delivery_accepts_complete_correspondence(tmp_path, monkeypatch):
    lock, path, inventory = _material(tmp_path)
    monkeypatch.setattr(delivery, "_inventory", lambda _archive: inventory)
    proof = delivery._verify(lock, path, tmp_path / "image.tar")
    assert proof["distributed_package_identities"] == 1
    assert proof["delivered_archives"] == 1


def test_real_source_lock_covers_inherited_and_added_packages():
    lock = json.loads((IMAGE / "corresponding-source.lock.json").read_bytes())
    assert len(lock["parent_diff_ids"]) == 4
    required = {(p["source"], p["source_version"]) for p in lock["packages"]}
    assert required == {(s["name"], s["version"]) for s in lock["sources"]}
    added = json.loads((IMAGE / "debian-packages.lock").read_bytes())["packages"]
    actual = {(p["name"], p["version"]) for p in lock["packages"]}
    assert {(p["name"], p["version"]) for p in added} <= actual
    assert all(delivery._artifacts(lock).values())
