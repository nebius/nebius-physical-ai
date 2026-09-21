"""Exercise RLC transfer boundaries, published weight identity and episode resets."""

import argparse
import asyncio
import builtins
import contextlib
import functools
import hashlib
import inspect
import importlib.util
import json
from pathlib import Path
import sys
import types
from types import SimpleNamespace
from unittest.mock import Mock

import ml_dtypes
import numpy as np
import pytest

from npa.workflows.behavior_challenge import (
    policy,
    rlc_correlation,
    rlc_observations,
    rlc_policy,
    rlc_selected,
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _selected_artifacts(tmp_path: Path) -> SimpleNamespace:
    export = tmp_path / "export-receipt.json"
    export.write_text(
        json.dumps(
            {
                "schema": rlc_selected.EXPORT_SCHEMA,
                "selected_step": 3599,
                "status": "holdout_selected_not_rollout_evaluated",
                "files": {},
            }
        )
        + "\n"
    )
    array = np.zeros(rlc_selected.CORRELATION_SHAPE, dtype=ml_dtypes.bfloat16)
    artifact = tmp_path / "native-correlation.bf16"
    artifact.write_bytes(array.tobytes())
    source_hashes = {"validator": "1" * 64}
    evidence_hashes = {"selected_export_receipt": _digest(export)}
    manifest = {
        "schema": rlc_selected.MANIFEST_SCHEMA,
        "adapter_version": 1,
        "selected_step": 3599,
        "artifact": {
            "path": artifact.name,
            **rlc_selected.array_identity(array),
        },
        "native_correlation": {
            **rlc_selected.array_identity(array),
            "path": "action_correlation_cholesky",
            "variable_type": "Intermediate",
        },
        "direct_correlation_before": {
            **rlc_selected.array_identity(
                np.zeros(rlc_selected.CORRELATION_SHAPE, dtype=np.float32)
            ),
            "path": "action_correlation_cholesky",
            "variable_type": "Intermediate",
        },
        "source_sha256": source_hashes,
        "evidence_sha256": evidence_hashes,
    }
    manifest_path = tmp_path / "correlation-manifest.json"
    manifest_path.write_text(json.dumps(manifest) + "\n")
    adapter_root = Path(rlc_policy.__file__).parent
    validation = {
        "schema": rlc_selected.VALIDATION_SCHEMA,
        "selected_step": 3599,
        "status": "selected_serving_validated",
        "candidate_rollout_eligible": True,
        "adapter": {"manifest_sha256": _digest(manifest_path)},
        "typed_state": {
            "leaves": 2,
            "after_adapter": {
                "equal": True,
                "differing": [],
                "direct_only": [],
                "native_only": [],
            },
        },
        "metrics": {
            "native_vs_aligned_selected_export": {
                "equal": True,
                "parts": [
                    {
                        "equal": True,
                        "dtype_equal": True,
                        "shape_equal": True,
                        "max_abs_delta": 0.0,
                        "mean_abs_delta": 0.0,
                    }
                    for _ in range(4)
                ],
            }
        },
        "actions": {
            "native_vs_aligned_selected_export": {
                "equal": True,
                "parts": [
                    {
                        "equal": True,
                        "dtype_equal": True,
                        "shape_equal": True,
                        "max_abs_delta": 0.0,
                        "mean_abs_delta": 0.0,
                    }
                    for _ in range(2)
                ],
            },
            "native_vs_existing_adapter_policy_jit": {
                "equal": True,
                "parts": [
                    {
                        "equal": True,
                        "dtype_equal": True,
                        "shape_equal": True,
                        "max_abs_delta": 0.0,
                        "mean_abs_delta": 0.0,
                    }
                    for _ in range(2)
                ],
            },
            "existing_adapter_smoke": {
                "finite": True,
                "dtype": "float32",
                "shape": [23],
            },
        },
        "source_sha256": source_hashes,
        "evidence_sha256": evidence_hashes,
        "existing_adapter": {
            "source_sha256": {
                name: _digest(adapter_root / name)
                for name in ("rlc_server.py", "rlc_observations.py")
            }
        },
    }
    validation_path = tmp_path / "validation-receipt.json"
    validation_path.write_text(json.dumps(validation) + "\n")
    return SimpleNamespace(
        export=export,
        manifest=manifest_path,
        validation=validation_path,
        adapter_root=adapter_root,
        array=array,
    )


def _install_fake_jax(monkeypatch) -> None:
    jax = types.ModuleType("jax")
    jax_numpy = types.ModuleType("jax.numpy")
    jax_numpy.asarray = np.asarray
    jax.numpy = jax_numpy
    monkeypatch.setitem(sys.modules, "jax", jax)
    monkeypatch.setitem(sys.modules, "jax.numpy", jax_numpy)


@pytest.fixture
def observation():
    value = {"robot_r1::proprio": np.arange(61, dtype=np.float32)}
    for camera in rlc_observations.CAMERAS:
        size = 720 if camera == "zed_link" else 480
        value[f"robot_r1::robot_r1:{camera}:Camera:0::rgb"] = np.zeros(
            (size, size, 4), dtype=np.uint8
        )
    return value


def test_policy_cannot_receive_task_state_or_instance_metadata(observation):
    observation.update(task_id=[49], instance_id=311, object_pose="forbidden")
    result = rlc_observations.policy_observation(observation)
    assert len(result) == 4
    assert "task_id" not in result
    assert "instance_id" not in result
    assert "object_pose" not in result
    assert result["robot_r1::proprio"] is observation["robot_r1::proprio"]
    for name, value in result.items():
        if name.endswith("::rgb"):
            assert value.shape[-1] == 3
            assert np.shares_memory(value, observation[name])


@pytest.mark.parametrize(
    "state", [np.zeros(256), np.full(61, np.nan), np.zeros((1, 61))]
)
def test_old_or_invalid_proprioception_is_rejected(observation, state):
    observation["robot_r1::proprio"] = state
    with pytest.raises(ValueError, match="61-element"):
        rlc_observations.policy_observation(observation)


@pytest.mark.parametrize(
    "image", [np.zeros((224, 224, 3), dtype=np.uint8), np.zeros((720, 720, 3))]
)
def test_unexpected_camera_contract_is_rejected(observation, image):
    observation["robot_r1::robot_r1:zed_link:Camera:0::rgb"] = image
    with pytest.raises(ValueError, match="camera"):
        rlc_observations.policy_observation(observation)


def test_proprioception_selects_correct_arms_grippers_and_trunk(observation):
    state = observation["robot_r1::proprio"]
    indices = rlc_observations.PROPRIOCEPTION_INDICES["R1Pro"]
    assert state[indices["arm_left_qpos"]].tolist() == list(range(3, 10))
    assert state[indices["arm_right_qpos"]].tolist() == list(range(28, 35))
    assert state[indices["gripper_left_qpos"]].tolist() == [24, 25]
    assert state[indices["gripper_right_qpos"]].tolist() == [49, 50]
    assert state[indices["trunk_qpos"]].tolist() == [53, 54, 55, 56]
    assert state[indices["base_qvel"]].tolist() == [0, 1, 2]


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setitem(sys.modules, "rlc_observations", rlc_observations)
    path = Path(rlc_policy.__file__).with_name("rlc_server.py")
    spec = importlib.util.spec_from_file_location("test_rlc_server", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_adapter_preserves_original_checkout(tmp_path, server, monkeypatch):
    old = "from omnigibson.learning.utils.eval_utils import PROPRIOCEPTION_INDICES"
    relative = (
        "src/b1k/policies/b1k_policy.py",
        "src/b1k/shared/eval_b1k_wrapper.py",
        "openpi/src/openpi/policies/b1k_policy.py",
    )
    for name in relative:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(old + "\n# frozen model code\n")
    monkeypatch.setattr(sys, "path", sys.path.copy())
    server._policy_source(tmp_path, tmp_path / "overlay")
    for name in relative:
        assert (tmp_path / name).read_text().startswith(old)
        module = name.removeprefix("openpi/").removeprefix("src/")
        patched = (tmp_path / "overlay" / module).read_text()
        assert "from rlc_observations import PROPRIOCEPTION_INDICES" in patched
        assert patched.endswith("# frozen model code\n")


def test_stock_server_execution_variant_parser_keeps_native_default(server):
    common = [
        "--source-root",
        "/source",
        "--checkpoint",
        "/checkpoint",
        "--task-id",
        "1",
        "--port",
        "8000",
    ]

    native = server.parser().parse_args(common)
    assert native.execution_variant == "native"
    assert native.correlation_asset is None
    assert native.correlation_sha256 is None
    for variant in ("final-stage-backtrack", "adaptive-short-chunk"):
        parsed = server.parser().parse_args([*common, "--execution-variant", variant])
        assert parsed.execution_variant == variant


def test_stock_server_default_keeps_native_loader(server, monkeypatch):
    native = object()
    monkeypatch.setattr(server, "_load_policy", lambda args: native)

    loaded, receipt = server._load_stock_policy(
        SimpleNamespace(correlation_asset=None, correlation_sha256=None)
    )

    assert loaded is native
    assert receipt is None


def test_adapter_inventory_stages_execution_module_for_both_weight_paths(tmp_path):
    stock = tmp_path / "stock"
    stock.mkdir()
    selected = tmp_path / "selected"
    selected.mkdir()

    stock_files = rlc_policy._adapter_files(stock, selected=False)
    selected_files = rlc_policy._adapter_files(selected, selected=True)

    assert stock_files["rlc_execution.py"] == _digest(stock / "rlc_execution.py")
    assert selected_files["rlc_execution.py"] == _digest(selected / "rlc_execution.py")
    assert stock_files["rlc_correlation.py"] == _digest(stock / "rlc_correlation.py")
    assert selected_files["rlc_correlation.py"] == _digest(
        selected / "rlc_correlation.py"
    )
    assert "rlc_selected_server.py" not in stock_files
    assert "rlc_selected_server.py" in selected_files


def test_fp32_correlation_loader_requires_exact_regular_bytes(tmp_path):
    array = np.zeros(rlc_correlation.CORRELATION_SHAPE, dtype="<f4")
    artifact = tmp_path / "correlation.float32.bin"
    artifact.write_bytes(array.tobytes(order="C"))
    expected = _digest(artifact)

    loaded = rlc_correlation.load_fp32_correlation(artifact, expected)

    assert loaded.shape == rlc_correlation.CORRELATION_SHAPE
    assert loaded.dtype == np.dtype("float32")
    assert rlc_correlation.correlation_identity(loaded)["sha256"] == expected
    with pytest.raises(ValueError, match="SHA-256 differs"):
        rlc_correlation.load_fp32_correlation(artifact, "1" * 64)
    link = tmp_path / "link.bin"
    link.symlink_to(artifact)
    with pytest.raises(ValueError, match="regular file"):
        rlc_correlation.load_fp32_correlation(link, expected)


def test_stock_server_installs_explicit_correlation_before_policy(
    tmp_path, server, monkeypatch
):
    artifact = tmp_path / "correlation.bin"
    artifact.write_bytes(np.zeros(rlc_correlation.CORRELATION_SHAPE, dtype="<f4"))
    expected = _digest(artifact)
    active = {"value": False}

    @contextlib.contextmanager
    def install(policy_type, correlation, digest):
        assert policy_type is FakePiBehavior
        assert correlation.shape == rlc_correlation.CORRELATION_SHAPE
        assert digest == expected
        active["value"] = True
        yield [{"installed": True}]
        active["value"] = False

    class FakePiBehavior:
        pass

    wrapper = SimpleNamespace(policy=SimpleNamespace(_model=object()))
    monkeypatch.setattr(server, "_load_policy", lambda args: wrapper)
    fake = SimpleNamespace(
        load_fp32_correlation=lambda path, digest: np.zeros(
            rlc_correlation.CORRELATION_SHAPE, dtype=np.float32
        ),
        pre_policy_fp32_correlation=install,
        verify_captured_correlation=lambda value, digest: {
            "sha256": digest,
            "loaded_inside_context": active["value"] is False,
        },
    )
    monkeypatch.setitem(sys.modules, "rlc_correlation", fake)
    monkeypatch.setitem(sys.modules, "b1k", types.ModuleType("b1k"))
    models = types.ModuleType("b1k.models")
    pi_behavior = types.ModuleType("b1k.models.pi_behavior")
    pi_behavior.PiBehavior = FakePiBehavior
    monkeypatch.setitem(sys.modules, "b1k.models", models)
    monkeypatch.setitem(sys.modules, "b1k.models.pi_behavior", pi_behavior)
    args = SimpleNamespace(correlation_asset=artifact, correlation_sha256=expected)

    loaded, receipt = server._load_stock_policy(args)

    assert loaded is wrapper
    assert receipt["installation"] == {"installed": True}
    assert receipt["captured"]["sha256"] == expected


def test_connection_resets_memory_without_sending_reset_response(
    server, observation, monkeypatch
):
    codec = SimpleNamespace(
        Packer=lambda: SimpleNamespace(pack=lambda x: x), unpackb=lambda x: x
    )
    monkeypatch.setitem(
        sys.modules, "openpi_client", SimpleNamespace(msgpack_numpy=codec)
    )
    policy = Mock()
    policy.act.return_value.cpu.return_value.numpy.return_value = np.zeros(23)

    class Socket:
        def __init__(self):
            self.sent = []

        async def send(self, value):
            self.sent.append(value)

        async def __aiter__(self):
            for value in ({"reset": True}, observation, {"reset": True}, observation):
                yield value

    socket = Socket()
    asyncio.run(server._connection(socket, policy))
    assert policy.reset.call_count == 4
    assert policy.act.call_count == 2
    assert len(socket.sent) == 3
    assert all("action" in response for response in socket.sent[1:])


@pytest.mark.parametrize(
    "split,tasks",
    [("report", ["radio"]), ("development", "all"), ("development", ["a", "b"])],
)
def test_unvalidated_transfer_cannot_enter_reporting(split, tasks, tmp_path):
    plan = {"recipe": {"split": split, "tasks": tasks}}
    with pytest.raises(ValueError, match="one development task"):
        rlc_policy.prepare_policy(argparse.Namespace(), plan, tmp_path)


def test_published_file_verifier_rejects_substituted_model(tmp_path, monkeypatch):
    weights = tmp_path / "weights"
    weights.write_bytes(b"different weights")
    identity = {"weights": {"size": 17, "sha256": "0" * 64}}
    manifest = {
        "revision": rlc_policy.MODEL_REVISION,
        "checkpoints": {"test": identity},
    }
    monkeypatch.setattr(rlc_policy.json, "loads", lambda _: manifest)
    with pytest.raises(ValueError, match="bytes differ"):
        rlc_policy._verify_published_files(
            tmp_path,
            "test",
            {"weights": hashlib.sha256(weights.read_bytes()).hexdigest()},
        )


def test_task_mapping_rejects_new_tasks_and_reordered_registry(tmp_path):
    old = tmp_path / "BEHAVIOR-1K/docs/challenge/task_data.json"
    new = tmp_path / "docs/challenge/task_data.json"
    for path, count in ((old, 50), (new, 100)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"tasks": [{"id": str(i)} for i in range(count)]}))
    with pytest.raises(ValueError, match="original 50"):
        rlc_policy._task_checkpoint(tmp_path, tmp_path, "51")
    data = json.loads(new.read_text())
    data["tasks"][0], data["tasks"][1] = data["tasks"][1], data["tasks"][0]
    new.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="original 50"):
        rlc_policy._task_checkpoint(tmp_path, tmp_path, "0")


def test_selected_policy_is_limited_to_checkpoint_2_tasks(tmp_path, monkeypatch):
    old = tmp_path / "BEHAVIOR-1K/docs/challenge/task_data.json"
    new = tmp_path / "docs/challenge/task_data.json"
    tasks = [{"id": str(task_id)} for task_id in range(100)]
    old.parent.mkdir(parents=True)
    new.parent.mkdir(parents=True)
    old.write_text(json.dumps({"tasks": tasks[:50]}))
    new.write_text(json.dumps({"tasks": tasks}))
    supported = [0, 1, 7, 8, 9, 12, 16, 17, 18, 20, 21, 22, 26, 30, 43, 45]
    unsupported = [task_id for task_id in range(50) if task_id not in supported]
    (tmp_path / "task_checkpoint_mapping.json").write_text(
        json.dumps(
            {
                "checkpoints": {
                    "checkpoint_2": {"tasks": supported},
                    "other": {"tasks": unsupported},
                }
            }
        )
    )
    monkeypatch.setattr(rlc_policy, "_verify_checkout", lambda *_: None)
    args = SimpleNamespace(
        policy_kind="rlc-selected",
        policy_root=tmp_path,
        upstream_root=tmp_path,
    )
    for task_id in supported:
        plan = {"recipe": {"split": "development", "tasks": [str(task_id)]}}
        assert rlc_policy._verify_task(args, plan) == (task_id, "selected")
    with pytest.raises(ValueError, match="checkpoint_2 tasks only"):
        rlc_policy._verify_task(
            args, {"recipe": {"split": "development", "tasks": ["2"]}}
        )


def test_selected_correlation_is_bound_to_export_and_existing_adapter(tmp_path):
    selected = _selected_artifacts(tmp_path)
    array, manifest = rlc_selected.load_validated_correlation(
        selected.manifest,
        selected.validation,
        selected.export,
        selected.adapter_root,
    )
    assert rlc_selected.array_identity(array) == {
        field: manifest["native_correlation"][field]
        for field in rlc_selected.ARRAY_IDENTITY_FIELDS
    }

    selected.export.write_text(selected.export.read_text() + " ")
    with pytest.raises(ValueError, match="does not bind the selected export"):
        rlc_selected.load_validated_correlation(
            selected.manifest,
            selected.validation,
            selected.export,
            selected.adapter_root,
        )


def test_host_selected_validation_needs_no_policy_numeric_runtime(
    tmp_path, monkeypatch
):
    selected = _selected_artifacts(tmp_path)
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name in {"jax", "jax.numpy", "ml_dtypes"}:
            raise AssertionError(f"host validation imported {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    artifact, manifest = rlc_selected.validate_selected_correlation(
        selected.manifest,
        selected.validation,
        selected.export,
        selected.adapter_root,
    )
    assert artifact.name == manifest["artifact"]["path"]


def test_selected_validation_requires_typed_state_metrics_and_actions(tmp_path):
    selected = _selected_artifacts(tmp_path)
    validation = json.loads(selected.validation.read_text())
    validation["actions"]["native_vs_existing_adapter_policy_jit"]["equal"] = False
    selected.validation.write_text(json.dumps(validation) + "\n")
    with pytest.raises(ValueError, match="action contract"):
        rlc_selected.validate_selected_correlation(
            selected.manifest,
            selected.validation,
            selected.export,
            selected.adapter_root,
        )


def test_legacy_3599_receipts_remain_compatible(tmp_path):
    selected = _selected_artifacts(tmp_path)
    manifest = json.loads(selected.manifest.read_text())
    manifest["schema"] = rlc_selected.LEGACY_MANIFEST_SCHEMA
    selected.manifest.write_text(json.dumps(manifest) + "\n")
    validation = json.loads(selected.validation.read_text())
    validation["schema"] = rlc_selected.LEGACY_VALIDATION_SCHEMA
    validation["status"] = "selected_3599_serving_validated"
    validation["adapter"]["manifest_sha256"] = _digest(selected.manifest)
    selected.validation.write_text(json.dumps(validation) + "\n")

    array, loaded = rlc_selected.load_validated_correlation(
        selected.manifest,
        selected.validation,
        selected.export,
        selected.adapter_root,
    )
    assert loaded["selected_step"] == 3599
    assert np.isfinite(array.astype(np.float32)).all()


def test_generic_selected_receipts_support_other_positive_steps(tmp_path):
    selected = _selected_artifacts(tmp_path)
    export = json.loads(selected.export.read_text())
    export["selected_step"] = 1200
    selected.export.write_text(json.dumps(export) + "\n")
    manifest = json.loads(selected.manifest.read_text())
    manifest["selected_step"] = 1200
    manifest["evidence_sha256"]["selected_export_receipt"] = _digest(selected.export)
    selected.manifest.write_text(json.dumps(manifest) + "\n")
    validation = json.loads(selected.validation.read_text())
    validation["selected_step"] = 1200
    validation["evidence_sha256"] = manifest["evidence_sha256"]
    validation["adapter"]["manifest_sha256"] = _digest(selected.manifest)
    selected.validation.write_text(json.dumps(validation) + "\n")

    _, loaded = rlc_selected.load_validated_correlation(
        selected.manifest,
        selected.validation,
        selected.export,
        selected.adapter_root,
    )
    assert loaded["selected_step"] == 1200


def test_selected_correlation_rejects_nonfinite_asset(tmp_path):
    selected = _selected_artifacts(tmp_path)
    array = selected.array.copy()
    array[0, 0] = np.inf
    artifact = selected.manifest.parent / "native-correlation.bf16"
    artifact.write_bytes(array.tobytes())
    manifest = json.loads(selected.manifest.read_text())
    identity = rlc_selected.array_identity(array)
    manifest["artifact"].update(identity)
    manifest["native_correlation"].update(identity)
    selected.manifest.write_text(json.dumps(manifest) + "\n")
    validation = json.loads(selected.validation.read_text())
    validation["adapter"]["manifest_sha256"] = _digest(selected.manifest)
    selected.validation.write_text(json.dumps(validation) + "\n")

    with pytest.raises(ValueError, match="non-finite"):
        rlc_selected.load_validated_correlation(
            selected.manifest,
            selected.validation,
            selected.export,
            selected.adapter_root,
        )


def test_selected_export_requires_exact_checkpoint_inventory(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    weights = checkpoint / "params.bin"
    weights.write_bytes(b"selected params")
    digest = _digest(weights)
    receipt = tmp_path / "export.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": rlc_selected.EXPORT_SCHEMA,
                "selected_step": 3599,
                "status": "holdout_selected_not_rollout_evaluated",
                "files": {
                    weights.name: {"sha256": digest, "bytes": weights.stat().st_size}
                },
            }
        )
    )
    args = SimpleNamespace(
        policy_selected_export_receipt=receipt,
        policy_checkpoint=checkpoint,
    )
    assert (
        rlc_policy._verify_selected_export(args, {weights.name: digest})[
            "selected_step"
        ]
        == 3599
    )
    with pytest.raises(ValueError, match="contract differs"):
        rlc_policy._verify_selected_export(
            args, {weights.name: digest, "unexpected": "0" * 64}
        )
    invalid = json.loads(receipt.read_text())
    invalid["selected_step"] = True
    receipt.write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match="contract differs"):
        rlc_policy._verify_selected_export(args, {weights.name: digest})


def test_selected_correlation_rejects_boolean_step(tmp_path):
    selected = _selected_artifacts(tmp_path)
    export = json.loads(selected.export.read_text())
    export["selected_step"] = True
    selected.export.write_text(json.dumps(export) + "\n")
    manifest = json.loads(selected.manifest.read_text())
    manifest["selected_step"] = True
    manifest["evidence_sha256"]["selected_export_receipt"] = _digest(selected.export)
    selected.manifest.write_text(json.dumps(manifest) + "\n")
    validation = json.loads(selected.validation.read_text())
    validation["selected_step"] = True
    validation["evidence_sha256"] = manifest["evidence_sha256"]
    validation["adapter"]["manifest_sha256"] = _digest(selected.manifest)
    selected.validation.write_text(json.dumps(validation) + "\n")

    with pytest.raises(ValueError, match="positive integer"):
        rlc_selected.validate_selected_correlation(
            selected.manifest,
            selected.validation,
            selected.export,
            selected.adapter_root,
        )


def test_selected_policy_installs_correlation_before_policy_capture(
    tmp_path, monkeypatch
):
    selected = _selected_artifacts(tmp_path)
    native, manifest = rlc_selected.load_validated_correlation(
        selected.manifest,
        selected.validation,
        selected.export,
        selected.adapter_root,
    )
    _install_fake_jax(monkeypatch)

    class PiBehavior:
        def __init__(self):
            self.action_correlation_cholesky = SimpleNamespace(value=None)

        def load_correlation_matrix(self, norm_stats):
            del norm_stats
            self.action_correlation_cholesky.value = np.zeros(
                rlc_selected.CORRELATION_SHAPE, dtype=np.float32
            )

    pi_module = types.ModuleType("b1k.models.pi_behavior")
    pi_module.PiBehavior = PiBehavior
    monkeypatch.setitem(sys.modules, "b1k", types.ModuleType("b1k"))
    monkeypatch.setitem(sys.modules, "b1k.models", types.ModuleType("b1k.models"))
    monkeypatch.setitem(sys.modules, "b1k.models.pi_behavior", pi_module)
    original = PiBehavior.load_correlation_matrix

    def load_policy(args):
        del args
        model = PiBehavior()
        model.load_correlation_matrix({})
        return SimpleNamespace(policy=SimpleNamespace(_model=model))

    wrapper = rlc_selected.load_selected_policy(
        SimpleNamespace(_load_policy=load_policy), object(), native, manifest
    )
    assert PiBehavior.load_correlation_matrix is original
    assert rlc_selected.array_identity(
        wrapper.policy._model.action_correlation_cholesky.value
    ) == {
        field: manifest["native_correlation"][field]
        for field in rlc_selected.ARRAY_IDENTITY_FIELDS
    }


def test_selected_policy_rejects_initializer_that_skips_correlation(
    tmp_path, monkeypatch
):
    selected = _selected_artifacts(tmp_path)
    native, manifest = rlc_selected.load_validated_correlation(
        selected.manifest,
        selected.validation,
        selected.export,
        selected.adapter_root,
    )
    _install_fake_jax(monkeypatch)

    class PiBehavior:
        def load_correlation_matrix(self, norm_stats):
            del norm_stats

    pi_module = types.ModuleType("b1k.models.pi_behavior")
    pi_module.PiBehavior = PiBehavior
    monkeypatch.setitem(sys.modules, "b1k", types.ModuleType("b1k"))
    monkeypatch.setitem(sys.modules, "b1k.models", types.ModuleType("b1k.models"))
    monkeypatch.setitem(sys.modules, "b1k.models.pi_behavior", pi_module)
    original = PiBehavior.load_correlation_matrix

    with pytest.raises(ValueError, match="exactly one"):
        rlc_selected.load_selected_policy(
            SimpleNamespace(_load_policy=lambda args: object()),
            object(),
            native,
            manifest,
        )
    assert PiBehavior.load_correlation_matrix is original


def test_selected_hook_precedes_real_pinned_nnx_jit_capture(monkeypatch):
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    flax = pytest.importorskip("flax")
    from flax import nnx

    if jax.__version__ != "0.5.3" or flax.__version__ != "0.10.2":
        pytest.skip("requires the selected checkpoint's pinned JAX/Flax runtime")

    class PiBehavior(nnx.Module):
        def __init__(self):
            self.action_correlation_cholesky = nnx.Intermediate(
                jnp.zeros((2, 2), dtype=jnp.float32)
            )

        def load_correlation_matrix(self, norm_stats):
            del norm_stats
            self.action_correlation_cholesky.value = jnp.zeros(
                (2, 2), dtype=jnp.float32
            )

        def sample_actions(self, value):
            return value + jnp.sum(self.action_correlation_cholesky.value)

    def module_jit(method):
        assert inspect.ismethod(method) and isinstance(method.__self__, nnx.Module)
        graph, state = nnx.split(method.__self__)

        def call(frozen, value):
            module = nnx.merge(graph, frozen)
            return method.__func__(module, value)

        compiled = jax.jit(call)

        @functools.wraps(method)
        def captured(value):
            return compiled(state, value)

        return captured

    pi_module = types.ModuleType("b1k.models.pi_behavior")
    pi_module.PiBehavior = PiBehavior
    monkeypatch.setitem(sys.modules, "b1k", types.ModuleType("b1k"))
    monkeypatch.setitem(sys.modules, "b1k.models", types.ModuleType("b1k.models"))
    monkeypatch.setitem(sys.modules, "b1k.models.pi_behavior", pi_module)
    native = jnp.full((2, 2), 2, dtype=jnp.bfloat16)
    manifest = {
        "direct_correlation_before": {
            **rlc_selected.array_identity(np.zeros((2, 2), dtype=np.float32)),
            "path": "action_correlation_cholesky",
            "variable_type": "Intermediate",
        }
    }

    def load_policy(args):
        del args
        model = PiBehavior()
        model.load_correlation_matrix({})
        return SimpleNamespace(
            policy=SimpleNamespace(
                _model=model,
                _sample_actions=module_jit(model.sample_actions),
            )
        )

    original = PiBehavior.load_correlation_matrix
    wrapper = rlc_selected.load_selected_policy(
        SimpleNamespace(_load_policy=load_policy), object(), native, manifest
    )
    assert PiBehavior.load_correlation_matrix is original
    assert wrapper.policy._model.action_correlation_cholesky.value.dtype == jnp.bfloat16
    assert float(wrapper.policy._sample_actions(jnp.array(1.0))) == 9.0
    wrapper.policy._model.action_correlation_cholesky.value = jnp.zeros(
        (2, 2), dtype=jnp.bfloat16
    )
    assert float(wrapper.policy._sample_actions(jnp.array(1.0))) == 9.0

    for calls in (0, 2):
        with pytest.raises(ValueError, match="exactly one"):
            with rlc_selected.pre_policy_correlation(PiBehavior, native):
                model = PiBehavior()
                for _ in range(calls):
                    model.load_correlation_matrix({})
        assert PiBehavior.load_correlation_matrix is original

    def fail_after_install(args):
        del args
        model = PiBehavior()
        model.load_correlation_matrix({})
        raise RuntimeError("constructor failure")

    with pytest.raises(RuntimeError, match="constructor failure"):
        rlc_selected.load_selected_policy(
            SimpleNamespace(_load_policy=fail_after_install),
            object(),
            native,
            manifest,
        )
    assert PiBehavior.load_correlation_matrix is original


def test_selected_server_command_uses_explicit_receipts(tmp_path):
    staged = {
        "selected_export": tmp_path / "export.json",
        "correlation_manifest": tmp_path / "manifest.json",
        "validation_receipt": tmp_path / "validation.json",
    }
    args = SimpleNamespace(
        policy_python=Path("/runtime/python"),
        policy_root=Path("/runtime/source"),
        policy_checkpoint=Path("/runtime/selected-model"),
        policy_execution_variant="transition-refresh",
        port=9000,
    )
    command = rlc_policy._command(args, 22, tmp_path, staged)
    assert command[:6] == [
        "/runtime/python",
        str(tmp_path / "rlc_selected_server.py"),
        "--source-root",
        "/runtime/source",
        "--adapter-root",
        str(tmp_path),
    ]
    assert command[-8:] == [
        "--selected-export-receipt",
        str(staged["selected_export"]),
        "--correlation-manifest",
        str(staged["correlation_manifest"]),
        "--validation-receipt",
        str(staged["validation_receipt"]),
        "--execution-variant",
        "transition-refresh",
    ]
    published = rlc_policy._command(args, 22, tmp_path)
    assert published[1] == str(tmp_path / "rlc_server.py")
    assert "--adapter-root" not in published
    assert published[-2:] == ["--execution-variant", "transition-refresh"]


def test_stock_provenance_records_experimental_execution_scope(tmp_path, monkeypatch):
    monkeypatch.setattr(
        rlc_policy,
        "_adapter_files",
        lambda output, selected: {"selected": str(selected)},
    )
    command = ["server", "--execution-variant", "adaptive-short-chunk"]
    rlc_policy._record(
        tmp_path,
        command,
        {"params": "a" * 64},
        "checkpoint_2",
        {"recipe": {"policy_checkpoint_sha256": "b" * 64}},
    )

    evidence = json.loads((tmp_path / "policy-provenance.json").read_text())
    variant = evidence["execution_variant"]
    assert variant["name"] == "adaptive-short-chunk"
    assert (
        variant["provenance"]["evaluation"]
        == "experimental_no_aggregate_gain_established"
    )


@pytest.mark.parametrize(
    "variant", ["native", "transition-refresh", "final-stage-backtrack"]
)
def test_selected_provenance_records_execution_validation_scope(
    tmp_path, monkeypatch, variant
):
    module = "npa.workflows.behavior_challenge.rlc_transition"
    monkeypatch.delitem(sys.modules, module, raising=False)
    staged = {}
    for name in ("selected_export", "correlation_manifest"):
        path = tmp_path / f"{name}.json"
        path.write_text("{}")
        staged[name] = path
    validation = tmp_path / "validation.json"
    validation.write_text("{}")
    staged["validation_receipt"] = validation
    monkeypatch.setattr(
        rlc_policy,
        "_adapter_files",
        lambda output, selected: {"selected": str(selected)},
    )
    command = ["server", "--execution-variant", variant]
    rlc_policy._record_selected(
        tmp_path,
        command,
        {"params": "a" * 64},
        {"selected_step": 3599},
        {"recipe": {"policy_checkpoint_sha256": "b" * 64}},
        staged,
    )

    evidence = json.loads((tmp_path / "policy-provenance.json").read_text())
    assert evidence["execution_variant"]["name"] == variant
    if variant == "native":
        assert module not in sys.modules
        assert evidence["execution_variant"]["provenance"] is None
        assert evidence["execution_variant"]["transition_refresh_provenance"] is None
        assert evidence["status"] == "serving_validated_not_rollout_evaluated"
    else:
        assert evidence["execution_variant"]["provenance"]
        expected_transition = (
            evidence["execution_variant"]["provenance"]
            if variant == "transition-refresh"
            else None
        )
        assert (
            evidence["execution_variant"]["transition_refresh_provenance"]
            == expected_transition
        )
        assert (
            evidence["status"]
            == "selected_state_validated_execution_variant_unevaluated"
        )


def test_selected_policy_arguments_are_all_or_nothing(tmp_path):
    base = {
        "policy_kind": "rlc-selected",
        "policy_root": tmp_path,
        "policy_python": tmp_path / "python",
        "policy_checkpoint": tmp_path / "checkpoint",
        "policy_archive": tmp_path / "archive",
        "policy_selected_export_receipt": tmp_path / "export.json",
        "policy_correlation_manifest": tmp_path / "manifest.json",
        "policy_validation_receipt": None,
    }
    with pytest.raises(ValueError, match="export, correlation, and validation"):
        with policy.managed_policy(
            argparse.Namespace(**base), {"recipe": {}}, tmp_path / "output"
        ):
            pass

    base["policy_kind"] = "rlc"
    base["policy_validation_receipt"] = tmp_path / "validation.json"
    with pytest.raises(ValueError, match="require --policy-kind"):
        with policy.managed_policy(
            argparse.Namespace(**base), {"recipe": {}}, tmp_path / "output"
        ):
            pass


def test_stock_correlation_arguments_are_opt_in_and_all_or_nothing(tmp_path):
    base = {
        "policy_kind": "rlc",
        "policy_execution_variant": "native",
        "policy_root": None,
        "policy_python": None,
        "policy_checkpoint": None,
        "policy_archive": None,
        "policy_selected_export_receipt": None,
        "policy_correlation_manifest": None,
        "policy_validation_receipt": None,
        "policy_stock_correlation_asset": tmp_path / "correlation.bin",
        "policy_stock_correlation_sha256": None,
    }
    with pytest.raises(ValueError, match="artifact and SHA-256"):
        with policy.managed_policy(
            argparse.Namespace(**base), {"recipe": {}}, tmp_path / "output"
        ):
            pass

    base["policy_stock_correlation_sha256"] = "1" * 64
    base["policy_kind"] = "official"
    with pytest.raises(ValueError, match="requires --policy-kind rlc"):
        with policy.managed_policy(
            argparse.Namespace(**base), {"recipe": {}}, tmp_path / "output"
        ):
            pass


def test_stock_correlation_is_staged_and_added_to_server_command(tmp_path):
    artifact = tmp_path / "canonical.bin"
    artifact.write_bytes(np.zeros(rlc_correlation.CORRELATION_SHAPE, dtype="<f4"))
    expected = _digest(artifact)
    output = tmp_path / "output"
    output.mkdir()
    args = SimpleNamespace(
        policy_stock_correlation_asset=artifact,
        policy_stock_correlation_sha256=expected,
        policy_python=Path("/runtime/python"),
        policy_root=Path("/source"),
        policy_checkpoint=Path("/checkpoint"),
        policy_execution_variant="native",
        port=8000,
    )

    staged = rlc_policy._stage_stock_correlation(args, output)
    command = rlc_policy._command(args, 1, output, stock_correlation=staged)

    assert staged == output / "stock-correlation.float32.bin"
    assert _digest(staged) == expected
    assert command[command.index("--correlation-asset") + 1] == str(staged)
    assert command[command.index("--correlation-sha256") + 1] == expected


@pytest.mark.parametrize(
    ("kind", "variant"),
    [
        ("official", "final-stage-backtrack"),
        ("rlc", "transition-refresh"),
        ("rlc-selected", "adaptive-short-chunk"),
    ],
)
def test_execution_variant_rejects_wrong_policy_family(tmp_path, kind, variant):
    args = argparse.Namespace(
        policy_kind=kind,
        policy_execution_variant=variant,
        policy_root=None,
        policy_python=None,
        policy_checkpoint=None,
        policy_archive=None,
        policy_selected_export_receipt=None,
        policy_correlation_manifest=None,
        policy_validation_receipt=None,
    )

    with pytest.raises(ValueError, match="Unsupported"):
        with policy.managed_policy(args, {"recipe": {}}, tmp_path):
            pass


def test_execution_variant_requires_managed_policy_paths(tmp_path):
    args = argparse.Namespace(
        policy_kind="rlc",
        policy_execution_variant="final-stage-backtrack",
        policy_root=None,
        policy_python=None,
        policy_checkpoint=None,
        policy_archive=None,
        policy_selected_export_receipt=None,
        policy_correlation_manifest=None,
        policy_validation_receipt=None,
    )

    with pytest.raises(ValueError, match="all four managed policy paths"):
        with policy.managed_policy(args, {"recipe": {}}, tmp_path):
            pass
