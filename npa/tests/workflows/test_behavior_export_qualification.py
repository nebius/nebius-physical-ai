from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parents[3]
IMPLEMENTATION = ROOT / "workflows/implementations/behavior-anchored-training"
SPEC = importlib.util.spec_from_file_location(
    "behavior_export_qualification", IMPLEMENTATION / "export_qualification.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_qualifies_exact_actions_and_documented_stage_mask():
    actions = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    stages = np.full((2, 5), -np.inf, dtype=np.float32)
    stages[0, :2] = [0.5, 1.5]
    stages[1, :3] = [0.25, 1.25, 2.25]
    proof = MODULE.qualify_policy_output(
        (actions, stages),
        task_ids=np.asarray([0, 1], dtype=np.int32),
        task_num_stages=np.asarray([2, 3], dtype=np.int32),
        expected_actions=MODULE.array_identity(actions),
        expected_stage_prediction=MODULE.array_identity(stages),
    )
    assert proof["exact"] is True
    assert proof["stage_prediction"]["selected_task_stage_counts"] == [2, 3]
    assert proof["stage_prediction"]["semantically_valid"] is True


@pytest.mark.parametrize(
    ("row", "column", "value"),
    [(0, 0, np.nan), (0, 0, np.inf), (0, 0, -np.inf), (0, 2, 0.0)],
)
def test_rejects_invalid_stage_value_or_mask(row, column, value):
    stages = np.full((1, 4), -np.inf, dtype=np.float32)
    stages[0, :2] = [0.5, 1.5]
    stages[row, column] = value
    with pytest.raises(ValueError):
        MODULE.stage_mask_proof(
            stages,
            task_ids=np.asarray([0], dtype=np.int32),
            task_num_stages=np.asarray([2], dtype=np.int32),
        )


def test_rejects_changed_output_bytes():
    actions = np.zeros((1, 2, 3), dtype=np.float32)
    stages = np.asarray([[0.5, -np.inf]], dtype=np.float32)
    expected = MODULE.array_identity(actions)
    changed = actions.copy()
    changed[0, 0, 0] = 1.0
    with pytest.raises(ValueError, match="action bytes differ"):
        MODULE.qualify_policy_output(
            (changed, stages),
            task_ids=np.asarray([0], dtype=np.int32),
            task_num_stages=np.asarray([1], dtype=np.int32),
            expected_actions=expected,
            expected_stage_prediction=MODULE.array_identity(stages),
        )


def test_export_inventory_rejects_extra_and_symlink(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "params").write_bytes(b"params")
    expected = {"params": MODULE.file_identity(model / "params")}
    assert MODULE.validate_export_files(model, expected) == expected
    (model / "extra").write_bytes(b"unexpected")
    with pytest.raises(ValueError, match="file set differs"):
        MODULE.validate_export_files(model, expected)


def test_export_inventory_rejects_symlink(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    target = tmp_path / "outside"
    target.write_bytes(b"params")
    (model / "params").symlink_to(target)
    expected = {"params": MODULE.file_identity(target)}
    with pytest.raises(ValueError, match="symlink"):
        MODULE.validate_export_files(model, expected)


def test_array_bundle_round_trip_and_archive_tamper(tmp_path):
    path = tmp_path / "observation.npz"
    manifest = tmp_path / "observation.json"
    MODULE.write_array_bundle(
        path,
        manifest,
        schema="test.observation.v1",
        arrays={"image": {"front": np.arange(6, dtype=np.uint8).reshape(2, 3)}},
        metadata={"seed": 4049},
    )
    arrays, receipt = MODULE.load_array_bundle(
        path, manifest, expected_schema="test.observation.v1"
    )
    np.testing.assert_array_equal(
        arrays["image/front"], np.arange(6, dtype=np.uint8).reshape(2, 3)
    )
    assert receipt["metadata"] == {"seed": 4049}
    with path.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="NPZ identity differs"):
        MODULE.load_array_bundle(path, manifest, expected_schema="test.observation.v1")


def test_typed_key_and_bfloat16_bundle_round_trip(tmp_path):
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    key = jax.random.key(4049)
    record = MODULE.serialize_prng_key(jax, key)
    restored = MODULE.restore_prng_key(jax, record)
    np.testing.assert_array_equal(
        np.asarray(jax.random.key_data(restored)), np.asarray(jax.random.key_data(key))
    )
    assert str(jax.random.key_impl(restored)) == record["implementation"]

    path = tmp_path / "bfloat.npz"
    manifest = tmp_path / "bfloat.json"
    MODULE.write_array_bundle(
        path,
        manifest,
        schema="test.bfloat.v1",
        arrays={"stage": jnp.asarray([1.0, -1.0], dtype=jnp.bfloat16)},
        metadata={"key": record},
    )
    arrays, loaded = MODULE.load_array_bundle(
        path, manifest, expected_schema="test.bfloat.v1"
    )
    assert str(arrays["stage"].dtype) == "bfloat16"
    assert loaded["metadata"]["key"] == record


def diagnostic_arrays():
    actions = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    stages = np.full((2, 5), -np.inf, dtype=np.float32)
    stages[0, :2] = [0.5, 1.5]
    stages[1, :3] = [0.25, 1.25, 2.25]
    capture = {
        "actions": actions,
        "same_load_repeat_actions": actions.copy(),
        "stage_prediction": stages,
        "same_load_repeat_stage_prediction": stages.copy(),
    }
    repeat = {
        "frozen_actions": actions.copy(),
        "frozen_same_load_repeat_actions": actions.copy(),
        "frozen_stage_prediction": stages.copy(),
        "frozen_same_load_repeat_stage_prediction": stages.copy(),
        "reconstructed_actions": actions.copy(),
        "reconstructed_same_load_repeat_actions": actions.copy(),
        "reconstructed_stage_prediction": stages.copy(),
        "reconstructed_same_load_repeat_stage_prediction": stages.copy(),
    }
    return capture, repeat


def test_loader_diagnostic_proves_repeatability_without_native_parity_claim():
    capture, repeat = diagnostic_arrays()
    observation = {"state": np.ones((2, 4), dtype=np.float32)}
    result = MODULE.compare_loader_diagnostics(
        captured_observation=observation,
        reconstructed_observation={"state": observation["state"].copy()},
        capture=capture,
        repeat=repeat,
        task_ids=np.asarray([0, 1], dtype=np.int32),
        task_num_stages=np.asarray([2, 3], dtype=np.int32),
    )
    assert result["independent_load_repeat_exact"] is True
    assert result["native_export_parity_resolved"] is False
    assert result["rollout_eligibility_claimed"] is False


def test_loader_diagnostic_rejects_reconstruction_and_output_byte_changes():
    capture, repeat = diagnostic_arrays()
    observation = {"state": np.ones((2, 4), dtype=np.float32)}
    changed_observation = {"state": observation["state"].copy()}
    changed_observation["state"][0, 0] = 2
    with pytest.raises(ValueError, match="reconstructed observation"):
        MODULE.compare_loader_diagnostics(
            captured_observation=observation,
            reconstructed_observation=changed_observation,
            capture=capture,
            repeat=repeat,
            task_ids=np.asarray([0, 1], dtype=np.int32),
            task_num_stages=np.asarray([2, 3], dtype=np.int32),
        )
    repeat["frozen_actions"] = repeat["frozen_actions"].copy()
    repeat["frozen_actions"][0, 0, 0] += 1
    repeat["frozen_same_load_repeat_actions"] = repeat["frozen_actions"].copy()
    with pytest.raises(ValueError, match="independent loader actions"):
        MODULE.compare_loader_diagnostics(
            captured_observation=observation,
            reconstructed_observation=observation,
            capture=capture,
            repeat=repeat,
            task_ids=np.asarray([0, 1], dtype=np.int32),
            task_num_stages=np.asarray([2, 3], dtype=np.int32),
        )
