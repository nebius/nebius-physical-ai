"""Bind RoboCasa's CUDA verifier to real archive parsing and its own notice path."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
IMAGE = ROOT / "npa/docker/workbench/robocasa"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VERIFIER = _load("robocasa_image_verifier", IMAGE / "verify_image.py")
ARCHIVES = _load("robocasa_archive_builders", Path(__file__).with_name("test_curobo_image_verifier.py"))


@pytest.fixture
def payload(monkeypatch):
    contract, entries, excluded = ARCHIVES.payload.__wrapped__()
    original_path = contract["nvshmem_notice"]["path"]
    path = VERIFIER._runtime_contract()["nvshmem_notice"]["path"]
    contract["nvshmem_notice"]["path"] = path
    entries = [(path if name == original_path else name, *rest) for name, *rest in entries]
    monkeypatch.setattr(VERIFIER, "_runtime_contract", lambda: copy.deepcopy(contract))
    return contract, entries, excluded


def test_contract_uses_same_exact_pinned_wheels_with_robocasa_notice():
    contract = VERIFIER._runtime_contract()
    shared = json.loads((IMAGE.parent / "curobo/runtime-payload.json").read_text())
    assert contract["nvshmem_notice"]["path"] == "usr/share/doc/npa-robocasa/NVSHMEM-LICENSE.txt"
    contract["nvshmem_notice"]["path"] = shared["nvshmem_notice"]["path"]
    assert contract == shared
    lock = (IMAGE / "requirements.lock").read_text()
    for pin in ("torch==2.13.0+cu130", "nvidia-cudnn-cu13==9.20.0.48", "nvidia-nvshmem-cu13==3.4.5"):
        assert pin in lock


@pytest.mark.parametrize("compressed", [False, True])
def test_complete_real_archive_requires_correct_notice_location(tmp_path, payload, compressed):
    archive, image_id = ARCHIVES.save_image(tmp_path, [payload[1]], compressed=compressed)
    report = VERIFIER.verify_image(archive, expected_image_id=image_id)
    assert report["valid"]
    assert report["tool"] == "robocasa"
    assert report["required_payload_count"] == 10
    assert report["content_bytes_read"] == sum(len(row[1]) for row in payload[1])


@pytest.mark.parametrize("mutation", ["old-notice-path", "changed-notice", "hidden-sdk"])
def test_missing_attribution_or_hidden_sdk_cannot_pass(tmp_path, payload, mutation):
    entries = copy.deepcopy(payload[1])
    layers = []
    if mutation == "old-notice-path":
        entries[-1] = ARCHIVES.entry("usr/share/doc/npa-curobo/NVSHMEM-LICENSE.txt", entries[-1][1])
    elif mutation == "changed-notice":
        entries[-1] = ARCHIVES.entry(entries[-1][0], b"truncated")
    else:
        layers.append([ARCHIVES.entry("opt/renamed-sdk", payload[2][0])])
        entries.insert(0, ARCHIVES.entry("opt/.wh.renamed-sdk"))
    archive, image_id = ARCHIVES.save_image(tmp_path, [*layers, entries])
    report = VERIFIER.verify_image(archive, expected_image_id=image_id)
    assert not report["valid"]
    expected = {"old-notice-path": "required_payload_missing",
                "changed-notice": "retained_payload_hash_mismatch",
                "hidden-sdk": "excluded_cudnn_sdk_bytes"}[mutation]
    assert expected in {row["code"] for row in report["findings"]}
