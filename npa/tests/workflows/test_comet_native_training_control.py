from __future__ import annotations

import hashlib
import json
import os
import sys

import pytest
from npa.workflows.behavior_challenge.native_training_checkpoint import (
    atomic_json,
    checkpoint_inventory,
    file_identity,
    validate_full_state_contract,
)
from npa.workflows.behavior_challenge.native_training_control import (
    publish_with_control,
)

WORKER = """
import argparse,hashlib,json,os
from pathlib import Path
def identity(path):
    data=path.read_bytes();return {"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest()}
p=argparse.ArgumentParser();p.add_argument("operation");p.add_argument("--request",type=Path);p.add_argument("--output",type=Path);a=p.parse_args()
r=json.loads(a.request.read_text());binding={"request":identity(a.request),"receipt":r["receipt"],"checkpoint":r["checkpoint"],"prefix":r["prefix"],"logical_update_count":r["logical_update_count"]}
if os.environ.get("CORRUPT_BINDING")=="1":binding["logical_update_count"]+=1
manifest={"fixture":True};payload=(json.dumps(manifest,indent=2,sort_keys=True)+"\\n").encode();provider={"provider_readback":True,"uri":r["prefix"]+"/output-manifest.json","bytes":len(payload),"sha256":hashlib.sha256(payload).hexdigest()}
a.output.write_text(json.dumps({"schema":"npa.behavior.comet-native-storage-result.v1","binding":binding,"manifest":manifest,"provider_manifest":provider}))
"""


def fixture(tmp_path, monkeypatch):
    checkpoint = tmp_path / "step-00002"
    (checkpoint / "1/params").mkdir(parents=True)
    (checkpoint / "1/params/x").write_bytes(b"x")
    receipt = tmp_path / "step-00002.json"
    atomic_json(
        receipt,
        {"logical_update_count": 2, "checkpoint": checkpoint_inventory(checkpoint)},
    )
    worker = tmp_path / "worker.py"
    worker.write_text(WORKER)
    monkeypatch.setenv("NPA_STORAGE_CONTROL_PYTHON", sys.executable)
    monkeypatch.setenv("NPA_STORAGE_CONTROL_WORKER", str(worker))
    monkeypatch.setenv("NPA_STORAGE_CONTROL_PYTHONPATH", os.pathsep.join(sys.path))
    return checkpoint, receipt


def test_control_bridge_binds_fresh_request_and_result(tmp_path, monkeypatch):
    checkpoint, receipt = fixture(tmp_path, monkeypatch)
    result = publish_with_control(
        checkpoint, receipt, "s3://bucket/run/milestones/step-00002", tmp_path
    )
    assert result["binding"]["receipt"] == file_identity(receipt)


def test_control_bridge_rejects_stale_result(tmp_path, monkeypatch):
    checkpoint, receipt = fixture(tmp_path, monkeypatch)
    (tmp_path / "publish-step-00002-result.json").write_text("{}")
    with pytest.raises(ValueError, match="binding differs"):
        publish_with_control(
            checkpoint, receipt, "s3://bucket/run/milestones/step-00002", tmp_path
        )


def test_control_bridge_rejects_corrupt_child_binding(tmp_path, monkeypatch):
    checkpoint, receipt = fixture(tmp_path, monkeypatch)
    monkeypatch.setenv("CORRUPT_BINDING", "1")
    with pytest.raises(ValueError, match="binding differs"):
        publish_with_control(
            checkpoint, receipt, "s3://bucket/run/milestones/step-00002", tmp_path
        )


def test_control_bridge_reuses_exact_completed_transaction(tmp_path, monkeypatch):
    checkpoint, receipt = fixture(tmp_path, monkeypatch)
    first = publish_with_control(
        checkpoint, receipt, "s3://bucket/run/milestones/step-00002", tmp_path
    )
    monkeypatch.setenv(
        "NPA_STORAGE_CONTROL_WORKER", str(tmp_path / "missing-worker.py")
    )
    with pytest.raises(ValueError, match="worker differs"):
        publish_with_control(
            checkpoint, receipt, "s3://bucket/run/milestones/step-00002", tmp_path
        )
    monkeypatch.setenv("NPA_STORAGE_CONTROL_WORKER", str(tmp_path / "worker.py"))
    second = publish_with_control(
        checkpoint, receipt, "s3://bucket/run/milestones/step-00002", tmp_path
    )
    assert second == first


def _array_inventory(leaves):
    payload = json.dumps(leaves, separators=(",", ":"), sort_keys=True).encode()
    return {
        "leaves": leaves,
        "row_count": len(leaves),
        "element_count": sum(row["elements"] for row in leaves.values()),
        "total_bytes": sum(row["bytes"] for row in leaves.values()),
        "content_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _state_contract():
    row = {
        "shape": [2],
        "dtype": "float32",
        "elements": 2,
        "bytes": 8,
        "sha256": "a" * 64,
    }
    leaves = {"params/a": row}
    return {
        "schema": "npa.behavior.comet-native-full-train-state.v1",
        "step": 2,
        "params": _array_inventory(leaves),
        "optimizer_state": _array_inventory(leaves),
        "adamw_mu": _array_inventory(leaves),
        "adamw_nu": _array_inventory(leaves),
        "optimizer_scalar_progress": {"count": 2},
        "ema_params_present": False,
    }


@pytest.mark.parametrize("mutation", ["summary", "moment_path", "dtype"])
def test_full_state_validator_rejects_receipt_only_corruption(mutation):
    value = _state_contract()
    if mutation == "summary":
        value["params"]["total_bytes"] += 1
    elif mutation == "moment_path":
        value["adamw_mu"]["leaves"] = {
            "wrong/path": value["adamw_mu"]["leaves"].pop("params/a")
        }
        leaves = value["adamw_mu"]["leaves"]
        value["adamw_mu"] = _array_inventory(leaves)
    else:
        value["params"]["leaves"]["params/a"]["dtype"] = "bfloat16"
        value["params"] = _array_inventory(value["params"]["leaves"])
    with pytest.raises(ValueError):
        validate_full_state_contract(value, 2)
