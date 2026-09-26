from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from npa.workflows.behavior_challenge.native_training_checkpoint import (
    checkpoint_inventory,
)
from npa.workflows.behavior_challenge.native_training_restore import restore_latest


class Storage:
    def __init__(self, values):
        self.values = values

    def read_bytes_with_etag(self, uri):
        return (self.values[uri], "etag") if uri in self.values else None

    def download_file(self, uri, path):
        Path(path).write_bytes(self.values[uri])


def _array_inventory(leaves: dict) -> dict:
    encoded = json.dumps(leaves, separators=(",", ":"), sort_keys=True).encode()
    return {
        "leaves": leaves,
        "row_count": len(leaves),
        "element_count": sum(row["elements"] for row in leaves.values()),
        "total_bytes": sum(row["bytes"] for row in leaves.values()),
        "content_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _full_state(step: int) -> dict:
    leaf = {
        "x": {
            "shape": [1],
            "dtype": "float32",
            "elements": 1,
            "bytes": 4,
            "sha256": "a" * 64,
        }
    }
    return {
        "schema": "npa.behavior.comet-native-full-train-state.v1",
        "step": step,
        "params": _array_inventory(leaf),
        "optimizer_state": _array_inventory(leaf),
        "adamw_mu": _array_inventory(leaf),
        "adamw_nu": _array_inventory(leaf),
        "optimizer_scalar_progress": {"count": step},
        "ema_params_present": False,
    }


def _identity(value):
    return {
        "bytes": len(value),
        "sha256": hashlib.sha256(value).hexdigest(),
        "provider_readback": True,
    }


def _checkpoint_fixture(step):
    member, optimizer = b"complete parameters", b"complete optimizer"
    rows = [
        {"path": f"{step - 1}/params/value", **_identity(member), "mode": "0o644"},
        {
            "path": f"{step - 1}/train_state/value",
            **_identity(optimizer),
            "mode": "0o644",
        },
    ]
    for row in rows:
        row.pop("provider_readback")
    inventory = {
        "files": rows,
        "file_count": 2,
        "total_bytes": len(member) + len(optimizer),
        "max_member_bytes": max(len(member), len(optimizer)),
        "content_sha256": hashlib.sha256(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    return member, optimizer, inventory


def _record_fixture(step, inventory, admission, static, workflow_inputs):
    cursor = {"epoch": 0, "batch_offset": step, "committed_global_batch": step}
    record = {
        "logical_update_count": step,
        "manager_step": step - 1,
        "cursor": cursor,
        "checkpoint": inventory,
        "admission": admission,
        "static_reconstruction": static,
        "workflow_inputs": workflow_inputs,
        "full_state_resume_ready": True,
        "train_state": _full_state(step),
        "durable_milestones_before": {"0": {"role": "parent"}},
    }
    return cursor, record, (json.dumps(record) + "\n").encode()


def _provider_files(prefix, step, member, optimizer, receipt):
    base = f"{prefix}/milestones/step-{step:05d}/originals/"
    return {
        f"checkpoint/{step - 1}/params/value": {
            **_identity(member),
            "uri": base + f"checkpoint/{step - 1}/params/value",
        },
        f"checkpoint/{step - 1}/train_state/value": {
            **_identity(optimizer),
            "uri": base + f"checkpoint/{step - 1}/train_state/value",
        },
        "milestone-receipt.json": {
            **_identity(receipt),
            "uri": base + "milestone-receipt.json",
        },
    }


def _fixture(step=2):
    admission = {"bytes": 7, "sha256": "a" * 64}
    static, workflow_inputs = (
        {"seed": 42, "optimizer": {"name": "adamw"}},
        {"schema": "inputs", "dataset": "bound"},
    )
    member, optimizer, inventory = _checkpoint_fixture(step)
    cursor, _record, receipt = _record_fixture(
        step, inventory, admission, static, workflow_inputs
    )
    prefix = "s3://example/run"
    files = _provider_files(prefix, step, member, optimizer, receipt)
    manifest = {
        "schema": "npa.behavior.comet-native-training-milestone.v1",
        "status": "complete_full_state_and_serving_params_provider_readback",
        "logical_update_count": step,
        "manager_step": step - 1,
        "full_state_resume_ready": True,
        "checkpoint": inventory,
        "cursor": cursor,
        "admission": admission,
        "static_reconstruction": static,
        "workflow_inputs": workflow_inputs,
        "files": files,
    }
    uri = f"{prefix}/milestones/step-{step:05d}/output-manifest.json"
    values = {
        uri: (json.dumps(manifest) + "\n").encode(),
        files[f"checkpoint/{step - 1}/params/value"]["uri"]: member,
        files[f"checkpoint/{step - 1}/train_state/value"]["uri"]: optimizer,
        files["milestone-receipt.json"]["uri"]: receipt,
    }
    return prefix, Storage(values), admission, static, workflow_inputs


def _restore(tmp_path, storage, prefix, admission, static, workflow_inputs):
    return restore_latest(
        storage,
        prefix=prefix,
        milestones=(0, 2),
        checkpoint_base=tmp_path / "checkpoints",
        selection=tmp_path / "selection.json",
        expected_admission=admission,
        expected_static_reconstruction=static,
        expected_workflow_inputs=workflow_inputs,
    )


def test_restore_reuses_exact_local_state_without_redownload(tmp_path):
    prefix, storage, admission, static, workflow_inputs = _fixture()
    first = _restore(tmp_path, storage, prefix, admission, static, workflow_inputs)
    checkpoint = Path(first["resume_checkpoint"])
    original_download = storage.download_file
    storage.download_file = lambda *_args: (_ for _ in ()).throw(
        AssertionError("redownload")
    )
    (tmp_path / "selection.json").unlink()
    second = _restore(tmp_path, storage, prefix, admission, static, workflow_inputs)
    assert second["resume_checkpoint"] == str(checkpoint)
    assert checkpoint_inventory(checkpoint)["file_count"] == 2
    storage.download_file = original_download


def test_restore_rejects_changed_admission_before_model(tmp_path):
    prefix, storage, admission, static, workflow_inputs = _fixture()
    with pytest.raises(ValueError, match="admission/static/workflow inputs"):
        _restore(
            tmp_path,
            storage,
            prefix,
            {**admission, "sha256": "b" * 64},
            static,
            workflow_inputs,
        )


def test_restore_resumes_safe_partial_transaction(tmp_path):
    prefix, storage, admission, static, workflow_inputs = _fixture()
    partial = tmp_path / "checkpoints/.restore-step-00002.partial/checkpoint/1/params"
    partial.mkdir(parents=True)
    partial.joinpath("value").write_bytes(b"complete parameters")
    result = _restore(tmp_path, storage, prefix, admission, static, workflow_inputs)
    assert Path(result["resume_checkpoint"]).is_dir()


def test_restore_retries_half_written_owned_download(tmp_path):
    prefix, storage, admission, static, workflow_inputs = _fixture()
    original = storage.download_file
    calls = 0

    def interrupted(uri, path):
        nonlocal calls
        calls += 1
        if calls == 1:
            Path(path).write_bytes(b"half")
            raise OSError("interrupted")
        original(uri, path)

    storage.download_file = interrupted
    with pytest.raises(OSError, match="interrupted"):
        _restore(tmp_path, storage, prefix, admission, static, workflow_inputs)
    result = _restore(tmp_path, storage, prefix, admission, static, workflow_inputs)
    assert Path(result["resume_checkpoint"]).is_dir()


def test_restore_finishes_checkpoint_rename_before_receipt_rename(tmp_path):
    prefix, storage, admission, static, workflow_inputs = _fixture()
    result = _restore(tmp_path, storage, prefix, admission, static, workflow_inputs)
    target = Path(result["resume_checkpoint"])
    receipt = Path(result["resume_receipt"])
    transaction = target.parent / ".restore-step-00002.partial"
    transaction.mkdir()
    receipt.replace(transaction / "milestone-receipt.json")
    recovered = _restore(tmp_path, storage, prefix, admission, static, workflow_inputs)
    assert Path(recovered["resume_receipt"]).is_file()


def test_restore_applies_manifest_modes_under_restrictive_umask(tmp_path):
    prefix, storage, admission, static, workflow_inputs = _fixture()
    previous = os.umask(0o077)
    try:
        result = _restore(tmp_path, storage, prefix, admission, static, workflow_inputs)
    finally:
        os.umask(previous)
    checkpoint = Path(result["resume_checkpoint"])
    assert {
        oct(path.stat().st_mode & 0o777)
        for path in checkpoint.rglob("*")
        if path.is_file()
    } == {"0o644"}


def test_restore_rejects_invalid_checkpoint_member_mode(tmp_path):
    prefix, storage, admission, static, workflow_inputs = _fixture()
    uri = f"{prefix}/milestones/step-00002/output-manifest.json"
    manifest = json.loads(storage.values[uri])
    manifest["checkpoint"]["files"][0]["mode"] = "0o4755"
    storage.values[uri] = (json.dumps(manifest) + "\n").encode()
    with pytest.raises(ValueError, match="checkpoint member mode differs"):
        _restore(tmp_path, storage, prefix, admission, static, workflow_inputs)
