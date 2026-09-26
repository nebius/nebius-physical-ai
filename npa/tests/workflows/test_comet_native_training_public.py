from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import tarfile
from pathlib import Path, PurePosixPath

import pytest
from npa.workflows.behavior_challenge import comet_native_workflow as workflow
from npa.workflows.behavior_challenge.native_training import NativeTrainingPlan


class Storage:
    def __init__(self, objects: dict[str, bytes] | None = None):
        self.objects = dict(objects or {})

    def read_bytes_with_etag(self, uri: str):
        return (self.objects[uri], "etag") if uri in self.objects else None

    def download_file(self, uri: str, path: str) -> None:
        Path(path).write_bytes(self.objects[uri])

    def put_bytes_conditional(self, payload: bytes, uri: str, **_kwargs) -> None:
        if uri in self.objects:
            raise RuntimeError("overwrite")
        self.objects[uri] = payload


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


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _tar(name: str, payload: bytes, mode: int = 0o755) -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        row = tarfile.TarInfo(name)
        row.mode = mode
        row.size = len(payload)
        archive.addfile(row, io.BytesIO(payload))
    return stream.getvalue()


def _locked_runtime_receipt(python_bytes: bytes, *, base_relative: str) -> bytes:
    value = {
        "schema": "npa.behavior.locked-scientific-runtime.v1",
        "status": "complete",
        "python": {
            "relative": "work/locked-venv/bin/python",
            "link_target": "unused-for-regular-file",
            "base_relative": base_relative,
            "base_sha256": _sha(python_bytes),
            "base_bytes": len(python_bytes),
        },
        "runtime_base": {
            "schema": "npa.behavior.runtime-ready.v2",
            "python_directory": base_relative.rsplit("/bin/", 1)[0],
        },
        "installed_distributions": {
            "count": 217,
            "bytes": 10,
            "sha256": "1" * 64,
            "canonical_sha256": "2" * 64,
        },
    }
    return json.dumps(value, sort_keys=True).encode()


def _runtime_args(tmp_path, manifest_uri, manifest, receipt, python_bytes):
    home = tmp_path / "home"
    home.mkdir()
    return argparse.Namespace(
        runtime_manifest_uri=manifest_uri,
        runtime_manifest_sha256=_sha(manifest),
        home=home,
        runtime_workspace=home / "work",
        scientific_python_relative="work/locked-venv/bin/python",
        scientific_python_link_target="unused-for-regular-file",
        scientific_python_base_relative=".local/python/bin/python3.11",
        scientific_python_base_sha256=_sha(python_bytes),
        scientific_python_base_bytes=len(python_bytes),
        scientific_runtime_receipt_relative="work/locked-venv/runtime-receipt.json",
        scientific_runtime_receipt_sha256=_sha(receipt),
        scientific_runtime_receipt_bytes=len(receipt),
    )


def _runtime_case(tmp_path):
    python_bytes = b"qualified-python"
    archive = _tar(".local/python/bin/python3.11", python_bytes)
    receipt = _locked_runtime_receipt(
        python_bytes, base_relative=".local/python/bin/python3.11"
    )
    runtime_archive = _bundle(
        {
            "work/locked-venv/bin/python": python_bytes,
            "work/locked-venv/runtime-receipt.json": receipt,
        }
    )
    archive_uri, runtime_uri = (
        "s3://example/runtime.tar.gz",
        "s3://example/runtime-work.tar.gz",
    )
    value = {
        "schema": "npa.behavior.private-runtime.v1",
        "allowed_directories": [".local/python", "work"],
        "archives": [
            {"uri": archive_uri, "sha256": _sha(archive), "bytes": len(archive)},
            {
                "uri": runtime_uri,
                "sha256": _sha(runtime_archive),
                "bytes": len(runtime_archive),
            },
        ],
    }
    manifest = (json.dumps(value, sort_keys=True) + "\n").encode()
    manifest_uri = "s3://example/runtime.json"
    storage = Storage(
        {manifest_uri: manifest, archive_uri: archive, runtime_uri: runtime_archive}
    )
    args = _runtime_args(tmp_path, manifest_uri, manifest, receipt, python_bytes)
    return args, storage, python_bytes


def test_runtime_uses_real_ready_v2_python_directory(tmp_path: Path) -> None:
    args, storage, python_bytes = _runtime_case(tmp_path)
    python, receipt = workflow._runtime(args, storage)
    assert receipt["schema"] == "npa.behavior.runtime-ready.v2"
    assert receipt["python_directory"] == ".local/python"
    assert python.read_bytes() == python_bytes
    assert "python_executable" not in receipt


@pytest.mark.parametrize("operation", ["preflight", "train"])
def test_operation_failure_publishes_originals_then_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    storage = Storage()
    workspace = tmp_path / operation
    args = argparse.Namespace(
        attempt_id=f"{operation}-1",
        operation=operation,
        workspace=workspace,
        prefix="s3://example/run",
    )

    def fail(_args):
        workspace.mkdir()
        _args._workspace_owned = True
        (workspace / f"{operation}.log").write_text("complete failure log\n")
        raise ValueError("fixture failure")

    monkeypatch.setattr(workflow, "execute", fail)
    monkeypatch.setattr(workflow.StorageClient, "from_environment", lambda: storage)
    with pytest.raises(ValueError, match="fixture failure"):
        workflow.run(args)
    root = f"s3://example/run/attempts/{operation}-1"
    manifest = json.loads(storage.objects[f"{root}/failure-manifest.json"])
    assert manifest["status"] == "failed_originals_provider_readback"
    assert set(manifest["originals"]) == {f"{operation}.log", "failure.json"}
    for row in manifest["originals"].values():
        assert row["provider_readback"] is True
        assert storage.objects[row["uri"]]


def test_existing_workspace_is_never_modified_or_published(tmp_path, monkeypatch):
    workspace = tmp_path / "owned-by-someone-else"
    workspace.mkdir()
    original = workspace / "failure.json"
    original.write_bytes(b"original evidence")
    storage = Storage()
    monkeypatch.setattr(workflow.StorageClient, "from_environment", lambda: storage)
    args = argparse.Namespace(
        attempt_id="retry-1",
        operation="train",
        workspace=workspace,
        prefix="s3://example/run",
    )
    with pytest.raises(ValueError, match="workspace must be new"):
        workflow.run(args)
    assert original.read_bytes() == b"original evidence"
    assert storage.objects == {}


def test_matching_canonical_receipt_is_reused_without_child(tmp_path):
    contract = {
        "schema": "npa.behavior.comet-native-workflow-inputs.v1",
        "runtime_ready": {"schema": "ready"},
    }
    receipt = {
        "schema": "npa.behavior.comet-native-input-preflight.v1",
        "status": "real_overlay_config_dataset_verified_before_policy_initialization",
        "config_name": "verified-config",
        "dataset_size": 8,
        "source_overlay": "ephemeral_verified_source_overlay",
        "admission": {"sha256": "a" * 64},
        "workflow_inputs": contract,
        "full_policy_initialized": False,
        "optimizer_updates": 0,
        "jax_backend": "cpu",
    }
    uri = "s3://example/run/preflight/receipt.json"
    storage = Storage({uri: (json.dumps(receipt) + "\n").encode()})
    args = argparse.Namespace(
        prefix="s3://example/run",
        operation="preflight",
        config_name="verified-config",
    )
    reused = workflow._reuse_completed_operation(storage, args, contract)
    assert reused["reused_completed_canonical_receipt"] is True


def _train_files(base):
    return {
        "checkpoint/1/params/value": {
            "uri": base + "checkpoint/1/params/value",
            "provider_readback": True,
            "bytes": 4,
            "sha256": "a" * 64,
        },
        "milestone-receipt.json": {
            "uri": base + "milestone-receipt.json",
            "provider_readback": True,
            "bytes": 8,
            "sha256": "c" * 64,
        },
        "checkpoint/1/train_state/value": {
            "uri": base + "checkpoint/1/train_state/value",
            "provider_readback": True,
            "bytes": 4,
            "sha256": "b" * 64,
        },
    }


def _train_manifest(contract):
    checkpoint = {
        "files": [
            {"path": "1/params/value", "bytes": 4, "sha256": "a" * 64},
            {"path": "1/train_state/value", "bytes": 4, "sha256": "b" * 64},
        ]
    }
    base = "s3://example/run/milestones/step-00002/originals/"
    return {
        "schema": "npa.behavior.comet-native-training-milestone.v1",
        "status": "complete_full_state_and_serving_params_provider_readback",
        "logical_update_count": 2,
        "manager_step": 1,
        "checkpoint": checkpoint,
        "files": _train_files(base),
        "full_state_resume_ready": True,
        "outcome": "success",
        "serving_export_qualified": False,
        "scoring_executed": False,
        "admission": {"sha256": "b" * 64},
        "workflow_inputs": contract,
    }


def _train_durable(manifest):
    payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    binding = {
        "request": {"bytes": 1, "sha256": "c" * 64},
        "receipt": {"bytes": 2, "sha256": "d" * 64},
        "checkpoint": manifest["checkpoint"],
        "prefix": "s3://example/run/milestones/step-00002",
        "logical_update_count": 2,
    }
    provider = {
        "bytes": len(payload),
        "sha256": _sha(payload),
        "uri": "s3://example/run/milestones/step-00002/output-manifest.json",
        "provider_readback": True,
    }
    return {
        "schema": "npa.behavior.comet-native-storage-result.v1",
        "binding": binding,
        "manifest": manifest,
        "provider_manifest": provider,
    }


def _metric(step, loss, learning_rate):
    return {
        "step": step,
        "loss": loss,
        "grad_norm": 1.0,
        "param_norm": 1.0,
        "learning_rate": learning_rate,
        "loader_consumer_wait_seconds": 0.1,
        "synchronized_native_update_seconds": 0.2,
    }


def _completed_train(contract: dict) -> dict:
    manifest = _train_manifest(contract)
    return {
        "schema": "npa.behavior.comet-native-full-training.v1",
        "status": "native_updates_and_declared_milestones_durable",
        "start_logical_update": 0,
        "logical_updates": 2,
        "milestones": [0, 2],
        "manager_index_semantics": "manager_step_equals_logical_update_minus_one",
        "checkpoints": {
            "0": {"role": "released_parent_nonresumable"},
            "2": {"durable": _train_durable(manifest)},
        },
        "hot_path_metrics": [_metric(1, 1.0, 0.0), _metric(2, 0.5, 0.1)],
        "runtime": {"admission": manifest["admission"], "workflow_inputs": contract},
        "selection_or_scoring_executed": False,
        "serving_export_qualified": False,
    }


def test_matching_complete_training_receipt_is_reused() -> None:
    contract = {
        "schema": "npa.behavior.comet-native-workflow-inputs.v1",
        "runtime_ready": {"schema": "ready"},
    }
    receipt = _completed_train(contract)
    uri = "s3://example/run/train/receipt.json"
    storage = Storage({uri: (json.dumps(receipt) + "\n").encode()})
    args = argparse.Namespace(
        prefix="s3://example/run", operation="train", final_step=2, milestones="0,2"
    )
    reused = workflow._reuse_completed_operation(storage, args, contract)
    assert reused["reused_completed_canonical_receipt"] is True


@pytest.mark.parametrize(
    "mutation",
    ["missing_metrics", "bogus_metrics", "wrong_final", "missing_final", "bad_file"],
)
def test_incomplete_training_receipt_never_skips_work(mutation: str) -> None:
    contract = {
        "schema": "npa.behavior.comet-native-workflow-inputs.v1",
        "runtime_ready": {"schema": "ready"},
    }
    receipt = _completed_train(contract)
    if mutation == "missing_metrics":
        receipt.pop("hot_path_metrics")
    elif mutation == "bogus_metrics":
        receipt["hot_path_metrics"][0] = {"step": 1, "bogus": 0.0}
    elif mutation == "wrong_final":
        receipt["logical_updates"] = 1
    elif mutation == "missing_final":
        receipt["checkpoints"].pop("2")
    else:
        manifest = receipt["checkpoints"]["2"]["durable"]["manifest"]
        manifest["files"]["milestone-receipt.json"].pop("sha256")
    uri = "s3://example/run/train/receipt.json"
    storage = Storage({uri: (json.dumps(receipt) + "\n").encode()})
    args = argparse.Namespace(
        prefix="s3://example/run", operation="train", final_step=2, milestones="0,2"
    )
    with pytest.raises(
        ValueError, match="canonical completed training|milestone provider"
    ):
        workflow._reuse_completed_operation(storage, args, contract)


def test_plan_rejects_bool_and_noncanonical_milestones() -> None:
    with pytest.raises(ValueError):
        NativeTrainingPlan(True, (0, 1))
    with pytest.raises(ValueError):
        NativeTrainingPlan(2, (0, 1))


def test_materialization_capacity_fails_before_archive_download(tmp_path: Path) -> None:
    admission = json.dumps({"minimum_materialization_free_bytes": 3}).encode()
    storage = Storage({"s3://example/admission": admission})
    args = argparse.Namespace(
        workspace=tmp_path,
        admission_uri="s3://example/admission",
        admission_sha256=_sha(admission),
        admission_bytes=len(admission),
        openpi_bytes=1,
        worker_bytes=1,
        dataset_bytes=1,
        parent_bytes=1,
    )
    with pytest.raises(ValueError, match="capacity contract"):
        workflow._stage_archives(args, storage)
    assert set(storage.objects) == {"s3://example/admission"}


def test_input_contract_binds_every_scientific_archive_and_runtime(
    tmp_path: Path,
) -> None:
    python = tmp_path / "python"
    python.write_bytes(b"python")
    values = {
        "runtime_manifest_uri": "s3://example/runtime",
        "runtime_manifest_sha256": "f" * 64,
        "scientific_python_relative": "work/locked-venv/bin/python",
        "scientific_python_link_target": "unused",
        "scientific_python_base_relative": ".local/python/bin/python3.11",
        "dataset_relative": "view",
        "split_relative": "split.json",
        "split_sha256": "e" * 64,
        "config_name": "verified-config",
        "final_step": 20_000,
        "milestones": "0,5000,10000,15000,20000",
    }
    for index, name in enumerate(
        ("openpi", "worker", "dataset", "parent", "admission"), 1
    ):
        values[f"{name}_uri"] = f"s3://example/{name}"
        values[f"{name}_sha256"] = str(index) * 64
        values[f"{name}_bytes"] = index
    contract = workflow._input_contract(
        argparse.Namespace(**values),
        {"schema": "npa.behavior.runtime-ready.v2"},
        python,
    )
    assert set(contract["archives"]) == {
        "openpi",
        "worker",
        "dataset",
        "parent",
        "admission",
    }
    assert contract["runtime_manifest"]["sha256"] == "f" * 64
    assert contract["scientific_python"]["sha256"] == _sha(b"python")


def _bundle(rows: dict[str, bytes], executable: set[str] | None = None) -> bytes:
    stream = io.BytesIO()
    executable = executable or set()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for name, payload in rows.items():
            row = tarfile.TarInfo(name)
            row.mode = 0o755 if name in executable else 0o644
            row.size = len(payload)
            archive.addfile(row, io.BytesIO(payload))
    return stream.getvalue()


def _portable_archives(python_bytes, runtime_receipt):
    runtime = _bundle(
        {
            ".local/python/bin/python": python_bytes,
            "work/locked-venv/bin/python": python_bytes,
            "work/locked-venv/runtime-receipt.json": runtime_receipt,
        },
        {".local/python/bin/python", "work/locked-venv/bin/python"},
    )
    stub = b"import argparse,json\nfrom pathlib import Path\np=argparse.ArgumentParser();p.add_argument('operation');p.add_argument('--workflow-input-contract',type=Path);p.add_argument('--output',type=Path);a,_=p.parse_known_args();c=json.loads(a.workflow_input_contract.read_text());a.output.write_text(json.dumps({'schema':'npa.behavior.comet-native-input-preflight.v1','status':'real_overlay_config_dataset_verified_before_policy_initialization','workflow_inputs':c})+'\\n')\n"
    archives = {
        "openpi": _bundle(
            {"bundle/packages/openpi-client/src/openpi_client/__init__.py": b""}
        ),
        "worker": _bundle(
            {
                "bundle/workflows/implementations/behavior-comet12/train_comet_native.py": stub
            }
        ),
        "dataset": _bundle(
            {"bundle/view/data.bin": b"data", "bundle/split.json": b"{}"}
        ),
        "parent": _bundle({"bundle/params/value": b"parent"}),
    }
    return runtime, archives


def _portable_storage(runtime, archives, admission):
    runtime_uri = "s3://example/runtime.tar.gz"
    value = {
        "schema": "npa.behavior.private-runtime.v1",
        "allowed_directories": [".local/python", "work"],
        "archives": [
            {"uri": runtime_uri, "sha256": _sha(runtime), "bytes": len(runtime)}
        ],
    }
    manifest = (json.dumps(value, sort_keys=True) + "\n").encode()
    objects = {
        "s3://example/runtime.json": manifest,
        runtime_uri: runtime,
        "s3://example/admission": admission,
    }
    bindings = {}
    for name, payload in archives.items():
        uri = f"s3://example/{name}.tar.gz"
        objects[uri] = payload
        bindings.update(
            {
                f"{name}_uri": uri,
                f"{name}_sha256": _sha(payload),
                f"{name}_bytes": len(payload),
            }
        )
    return Storage(objects), manifest, bindings


def _portable_args(tmp_path, python_bytes, receipt, admission, manifest, bindings):
    home = tmp_path / "home"
    home.mkdir()
    return argparse.Namespace(
        operation="preflight",
        attempt_id="preflight-1",
        prefix="s3://example/run",
        admission_uri="s3://example/admission",
        admission_sha256=_sha(admission),
        admission_bytes=len(admission),
        runtime_manifest_uri="s3://example/runtime.json",
        runtime_manifest_sha256=_sha(manifest),
        scientific_python_relative="work/locked-venv/bin/python",
        scientific_python_link_target="unused-for-regular-file",
        scientific_python_base_relative=".local/python/bin/python",
        scientific_python_base_sha256=_sha(python_bytes),
        scientific_python_base_bytes=len(python_bytes),
        scientific_runtime_receipt_relative="work/locked-venv/runtime-receipt.json",
        scientific_runtime_receipt_sha256=_sha(receipt),
        scientific_runtime_receipt_bytes=len(receipt),
        home=home,
        runtime_workspace=home / "work",
        workspace=tmp_path / "operation",
        checkpoint_root=home / "checkpoints",
        dataset_relative="view",
        split_relative="split.json",
        split_sha256=_sha(b"{}"),
        config_name="fixture",
        final_step=2,
        milestones="0,1,2",
        preflight_uri="s3://example/run/preflight/receipt.json",
        **bindings,
    )


def test_portable_preflight_entrypoint_restores_runtime_and_runs_bound_python(
    tmp_path, monkeypatch
):
    import sys

    python_bytes = Path(sys.executable).read_bytes()
    receipt = _locked_runtime_receipt(
        python_bytes, base_relative=".local/python/bin/python"
    )
    runtime, archives = _portable_archives(python_bytes, receipt)
    admission_value = {
        "schema": "npa.behavior.comet-native-training-admission.v1",
        "status": "qualified_inputs_bound_for_native_training",
        "minimum_materialization_free_bytes": sum(map(len, archives.values())),
        "minimum_checkpoint_free_bytes": 1,
        "static_reconstruction": {"seed": 42},
    }
    admission = (json.dumps(admission_value) + "\n").encode()
    storage, manifest, bindings = _portable_storage(runtime, archives, admission)
    monkeypatch.setattr(workflow.StorageClient, "from_environment", lambda: storage)
    args = _portable_args(
        tmp_path, python_bytes, receipt, admission, manifest, bindings
    )
    result = workflow.run(args)
    published = json.loads(storage.objects["s3://example/run/preflight/receipt.json"])
    assert result["runtime"]["schema"] == "npa.behavior.runtime-ready.v2"
    assert published["workflow_inputs"]["archives"]["worker"]["sha256"] == _sha(
        archives["worker"]
    )
    assert (args.workspace / "preflight.log").is_file()


def test_restart_publishes_complete_saved_milestone_without_retraining(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "checkpoints/step-00002"
    (checkpoint / "1/params").mkdir(parents=True)
    (checkpoint / "1/params/value").write_bytes(b"params")
    (checkpoint / "1/train_state").mkdir()
    (checkpoint / "1/train_state/value").write_bytes(b"optimizer")
    admission = tmp_path / "admission.json"
    admission_value = {"static_reconstruction": {"seed": 42}}
    admission.write_text(json.dumps(admission_value))
    record = {
        "logical_update_count": 2,
        "manager_step": 1,
        "checkpoint": workflow.checkpoint_inventory(checkpoint),
        "admission": workflow.file_identity(admission),
        "static_reconstruction": admission_value["static_reconstruction"],
        "full_state_resume_ready": True,
        "train_state": _full_state(2),
        "workflow_inputs": {"bound": True},
    }
    receipt = tmp_path / "checkpoints/step-00002.json"
    receipt.write_text(json.dumps(record))
    calls = []
    monkeypatch.setattr(
        workflow, "publish_with_control", lambda *args: calls.append(args)
    )
    args = argparse.Namespace(
        milestones="0,2",
        checkpoint_root=tmp_path / "checkpoints",
        prefix="s3://example/run",
    )
    workflow._recover_saved_milestones(
        args, {"admission": admission}, {}, {"bound": True}, Storage()
    )
    assert len(calls) == 1
    assert calls[0][0] == checkpoint and calls[0][1] == receipt


def _retired_local_state(tmp_path):
    root = tmp_path / "checkpoints"
    checkpoint = root / "step-00002"
    for name, payload in (("1/params/x", b"p"), ("1/train_state/x", b"o")):
        path = checkpoint / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    inventory = workflow.checkpoint_inventory(checkpoint)
    admission = tmp_path / "admission.json"
    value = {"static_reconstruction": {"seed": 42}}
    admission.write_text(json.dumps(value))
    return root, checkpoint, inventory, admission, value


def _retired_manifest(inventory, receipt, record):
    base = "s3://example/run/milestones/step-00002/originals/"
    files = {
        f"checkpoint/{row['path']}": {
            **{key: row[key] for key in ("bytes", "sha256")},
            "uri": base + f"checkpoint/{row['path']}",
            "provider_readback": True,
        }
        for row in inventory["files"]
    }
    files["milestone-receipt.json"] = {
        **workflow.file_identity(receipt),
        "uri": base + "milestone-receipt.json",
        "provider_readback": True,
    }
    return {
        "schema": "npa.behavior.comet-native-training-milestone.v1",
        "status": "complete_full_state_and_serving_params_provider_readback",
        "logical_update_count": 2,
        "manager_step": 1,
        "full_state_resume_ready": True,
        "checkpoint": inventory,
        "files": files,
        "admission": record["admission"],
        "static_reconstruction": record["static_reconstruction"],
        "workflow_inputs": record["workflow_inputs"],
    }


def _retired_provider_fixture(tmp_path: Path) -> tuple:
    root, checkpoint, inventory, admission, admission_value = _retired_local_state(
        tmp_path
    )
    contract = {"schema": "bound-inputs"}
    record = {
        "logical_update_count": 2,
        "manager_step": 1,
        "checkpoint": inventory,
        "admission": workflow.file_identity(admission),
        "static_reconstruction": admission_value["static_reconstruction"],
        "workflow_inputs": contract,
        "full_state_resume_ready": True,
        "train_state": _full_state(2),
    }
    receipt = root / "step-00002.json"
    receipt.write_text(json.dumps(record))
    manifest = _retired_manifest(inventory, receipt, record)
    shutil.rmtree(checkpoint)
    return root, receipt, admission, admission_value, contract, manifest


def test_provider_durable_retired_receipt_is_crossbound_before_unlink(tmp_path):
    values = _retired_provider_fixture(tmp_path)
    root, receipt, admission, _admission_value, contract, manifest = values
    uri = "s3://example/run/milestones/step-00002/output-manifest.json"
    storage = Storage({uri: (json.dumps(manifest) + "\n").encode()})
    args = argparse.Namespace(
        milestones="0,2", checkpoint_root=root, prefix="s3://example/run"
    )
    workflow._recover_saved_milestones(
        args, {"admission": admission}, {}, contract, storage
    )
    assert not receipt.exists()


@pytest.mark.parametrize(
    "field", ["admission", "static_reconstruction", "workflow_inputs"]
)
def test_provider_durable_retired_receipt_rejects_unbound_manifest_before_unlink(
    tmp_path, field
):
    values = _retired_provider_fixture(tmp_path)
    root, receipt, admission, _admission_value, contract, manifest = values
    manifest.pop(field)
    uri = "s3://example/run/milestones/step-00002/output-manifest.json"
    storage = Storage({uri: (json.dumps(manifest) + "\n").encode()})
    args = argparse.Namespace(
        milestones="0,2", checkpoint_root=root, prefix="s3://example/run"
    )
    with pytest.raises(ValueError, match="input binding"):
        workflow._recover_saved_milestones(
            args, {"admission": admission}, {}, contract, storage
        )
    assert receipt.exists()


@pytest.mark.parametrize("field", ["python", "runtime_base", "installed_distributions"])
def test_scientific_receipt_rejects_semantic_runtime_drift(tmp_path, field):
    python_bytes = b"qualified-python"
    payload = json.loads(
        _locked_runtime_receipt(
            python_bytes, base_relative=".local/python/bin/python3.11"
        )
    )
    if field == "python":
        payload[field]["relative"] = "work/other/bin/python"
    elif field == "runtime_base":
        payload[field]["python_directory"] = ".local/other"
    else:
        payload[field]["canonical_sha256"] = "invalid"
    raw = json.dumps(payload, sort_keys=True).encode()
    receipt = tmp_path / "work/locked-venv/runtime-receipt.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_bytes(raw)
    args = argparse.Namespace(
        home=tmp_path,
        scientific_runtime_receipt_relative="work/locked-venv/runtime-receipt.json",
        scientific_runtime_receipt_sha256=_sha(raw),
        scientific_runtime_receipt_bytes=len(raw),
        scientific_python_relative="work/locked-venv/bin/python",
        scientific_python_link_target="unused-for-regular-file",
        scientific_python_base_relative=".local/python/bin/python3.11",
        scientific_python_base_sha256=_sha(python_bytes),
        scientific_python_base_bytes=len(python_bytes),
    )
    ready = {
        "schema": "npa.behavior.runtime-ready.v2",
        "python_directory": ".local/python",
    }
    with pytest.raises(ValueError, match="semantic binding"):
        workflow._scientific_receipt(args, (PurePosixPath("work"),), ready)


def _manifest_worker_archive(*, undeclared_pyc: bool) -> tuple[bytes, str]:
    path = "workflows/implementations/behavior-comet12/train_comet_native.py"
    source = b"raise AssertionError('science child must not start')\n"
    manifest = {
        "files": {
            path: {
                "bytes": len(source),
                "sha256": _sha(source),
                "mode": "0o644",
            }
        }
    }
    manifest_bytes = (json.dumps(manifest, sort_keys=True) + "\n").encode()
    rows = {f"bundle/{path}": source, "bundle/MANIFEST.json": manifest_bytes}
    if undeclared_pyc:
        rows["bundle/__pycache__/worker.cpython-311.pyc"] = b"undeclared"
    return _bundle(rows), _sha(manifest_bytes)


@pytest.mark.parametrize(
    ("undeclared_pyc", "package_root", "message"),
    [(True, "bundle", "file sets differ"), (False, ".", "explicit root")],
)
@pytest.mark.parametrize("operation", ["preflight", "train"])
def test_operations_reject_worker_package_before_science_runtime(
    tmp_path, monkeypatch, undeclared_pyc, package_root, message, operation
):
    import sys

    python_bytes = Path(sys.executable).read_bytes()
    receipt, admission, storage, runtime_manifest, bindings = _bad_worker_inputs(
        python_bytes, undeclared_pyc
    )
    args = _portable_args(
        tmp_path, python_bytes, receipt, admission, runtime_manifest, bindings
    )
    args.worker_package_root = package_root
    args.worker_manifest_sha256 = bindings.pop("worker_manifest_sha256")
    args.operation = operation
    reached_runtime = []
    monkeypatch.setattr(workflow.StorageClient, "from_environment", lambda: storage)
    monkeypatch.setattr(workflow, "_runtime", lambda *_: reached_runtime.append(True))
    with pytest.raises(ValueError, match=message):
        workflow.run(args)
    assert reached_runtime == []


def test_worker_package_contract_is_optional_as_a_complete_pair():
    assert workflow._worker_package_contract(argparse.Namespace()) is None
    partial = argparse.Namespace(worker_package_root="bundle")
    with pytest.raises(ValueError, match="contract is incomplete"):
        workflow._worker_package_contract(partial)


def _bad_worker_inputs(python_bytes, undeclared_pyc):
    receipt = _locked_runtime_receipt(
        python_bytes, base_relative=".local/python/bin/python"
    )
    runtime, archives = _portable_archives(python_bytes, receipt)
    archives["worker"], manifest_sha = _manifest_worker_archive(
        undeclared_pyc=undeclared_pyc
    )
    minimum = sum(map(len, archives.values()))
    admission = (
        json.dumps(
            {
                "minimum_materialization_free_bytes": minimum,
                "minimum_checkpoint_free_bytes": 1,
            }
        )
        + "\n"
    ).encode()
    storage, runtime_manifest, bindings = _portable_storage(
        runtime, archives, admission
    )
    bindings["worker_manifest_sha256"] = manifest_sha
    return receipt, admission, storage, runtime_manifest, bindings
