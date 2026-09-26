"""Tests for the reusable Comet profile qualification core."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from npa.workflows.behavior_challenge import comet_policy

sys.modules["comet_policy"] = comet_policy
comet_server = importlib.import_module("npa.workflows.behavior_challenge.comet_server")
ROOT = Path(__file__).parents[3]
IMPLEMENTATION = ROOT / "workflows/implementations/behavior-policy-qualification"
ADAPTER = ROOT / "npa/src/npa/workflows/behavior_challenge"
PACKAGED_ADAPTER_FILES = (
    "comet_policy.py",
    "comet_server.py",
    "evaluator_versions.py",
    "evaluator_wire.py",
    "comet12-checkpoint.json",
    "comet50-checkpoint.json",
)


def load_script(monkeypatch, name: str):
    monkeypatch.setitem(sys.modules, "comet_policy", comet_policy)
    monkeypatch.setitem(sys.modules, "comet_server", comet_server)
    path = IMPLEMENTATION / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fetch_receipt(profile) -> dict:
    schemas = {
        "comet12": "npa.behavior.private-comet-fetch.v1",
        "comet50": "npa.behavior.private-comet50-fetch.v1",
    }
    inventories = {
        "comet12": "2ec0c083c0cfdacdf7a7ba02fe2185f76e8d91c9aafea88469042806bd131fda",
        "comet50": "3498c1d4f1fc5dbb9bb391554d7d794085c94b0f07e817a5a60700f06c612087",
    }
    return {
        "schema": schemas[profile.kind],
        "status": "bytes_verified_not_gpu_qualified",
        "source_commit": comet_policy.SOURCE_COMMIT,
        "model_repository": comet_policy.MODEL_REPOSITORY,
        "model_revision": profile.revision,
        "inventory_sha256": inventories[profile.kind],
        "files": profile.file_count,
        "checkpoint_bytes": profile.total_bytes,
        "readback_verified": True,
        "archive": {"sha256": "a" * 64, "bytes": 1, "path": "private.zip"},
    }


@pytest.mark.parametrize("kind", ["comet12", "comet50"])
def test_fetch_receipt_binds_profile_schema_and_archive(monkeypatch, tmp_path, kind):
    module = load_script(monkeypatch, "qualify_comet")
    profile = comet_policy.get_profile(kind)
    path = tmp_path / "fetch.json"
    path.write_text(json.dumps(fetch_receipt(profile)))
    assert (
        module._verify_fetch_receipt(path, profile)["model_revision"]
        == profile.revision
    )


@pytest.mark.parametrize("field,value", [("model_revision", "bad"), ("schema", "bad")])
def test_fetch_receipt_rejects_profile_mismatch(monkeypatch, tmp_path, field, value):
    module = load_script(monkeypatch, "qualify_comet")
    receipt = fetch_receipt(comet_policy.COMET50_PROFILE)
    receipt[field] = value
    path = tmp_path / "fetch.json"
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="fetch receipt differs"):
        module._verify_fetch_receipt(path, comet_policy.COMET50_PROFILE)


@pytest.mark.parametrize(
    "archive",
    [
        {"sha256": "x" * 64, "bytes": 1, "path": "x"},
        {"sha256": "a" * 64, "bytes": 0, "path": "x"},
        {"sha256": "a" * 64, "bytes": 1, "path": ""},
    ],
)
def test_fetch_receipt_rejects_malformed_archive(monkeypatch, tmp_path, archive):
    module = load_script(monkeypatch, "qualify_comet")
    receipt = fetch_receipt(comet_policy.COMET12_PROFILE)
    receipt["archive"] = archive
    path = tmp_path / "fetch.json"
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="archive"):
        module._verify_fetch_receipt(path, comet_policy.COMET12_PROFILE)


def action_args(tmp_path, action=None, expected=None):
    output = tmp_path / "output"
    output.mkdir()
    action = np.arange(23, dtype=np.float64) if action is None else action
    np.save(output / "action-1.npy", action, allow_pickle=False)
    np.save(output / "action-2.npy", action, allow_pickle=False)
    raw = hashlib.sha256(action.tobytes(order="C")).hexdigest()
    return SimpleNamespace(output=output, expected_action_sha256=expected), raw


def test_actions_are_finite_exact_bytes_and_match_optional_expected(
    monkeypatch, tmp_path
):
    module = load_script(monkeypatch, "qualify_comet")
    args, raw = action_args(tmp_path)
    args.expected_action_sha256 = raw
    result = module._compare_actions(args, [{"action": {"raw_sha256": raw}}] * 2)
    assert result["equal_across_two_fresh_processes"] is True
    assert result["equal_to_wrapper_expected_action"] is True


def test_action_comparison_rejects_signed_zero_byte_drift(monkeypatch, tmp_path):
    module = load_script(monkeypatch, "qualify_comet")
    action = np.zeros(23, dtype=np.float64)
    args, raw = action_args(tmp_path, action)
    right = action.copy()
    right[0] = -0.0
    np.save(args.output / "action-2.npy", right, allow_pickle=False)
    with pytest.raises(ValueError, match="first-action bytes differ"):
        module._compare_actions(args, [{"action": {"raw_sha256": raw}}] * 2)


@pytest.mark.parametrize(
    "action",
    [np.full(23, np.nan), np.zeros(22), np.arange(23, dtype=np.int32)],
)
def test_action_comparison_rejects_nonfinite_shape_or_dtype(
    monkeypatch, tmp_path, action
):
    module = load_script(monkeypatch, "qualify_comet")
    args, raw = action_args(tmp_path, action)
    with pytest.raises(ValueError, match="action"):
        module._compare_actions(args, [{"action": {"raw_sha256": raw}}] * 2)


def test_action_comparison_rejects_receipt_mismatch(monkeypatch, tmp_path):
    module = load_script(monkeypatch, "qualify_comet")
    args, _ = action_args(tmp_path)
    with pytest.raises(ValueError, match="receipts differ"):
        module._compare_actions(args, [{"action": {"raw_sha256": "0" * 64}}] * 2)


def test_child_action_identity_records_raw_and_npy_bytes(monkeypatch, tmp_path):
    module = load_script(monkeypatch, "process_action")
    action = np.arange(23, dtype=np.float64)
    path = tmp_path / "action.npy"
    np.save(path, action, allow_pickle=False)
    identity = module._action_identity(action, path)
    assert identity["shape"] == [23]
    assert identity["bytes"] == path.stat().st_size
    assert identity["raw_sha256"] != identity["npy_sha256"]


def test_scripts_import_from_checkout_and_packaged_layout(tmp_path):
    subprocess.run(
        [sys.executable, str(IMPLEMENTATION / "qualify_comet.py"), "--help"], check=True
    )
    package = tmp_path / "package"
    package.mkdir()
    for name in ("qualify_comet.py", "process_action.py"):
        shutil.copyfile(IMPLEMENTATION / name, package / name)
    for name in PACKAGED_ADAPTER_FILES:
        shutil.copyfile(ADAPTER / name, package / name)
    for name in ("qualify_comet.py", "process_action.py"):
        subprocess.run([sys.executable, str(package / name), "--help"], check=True)
