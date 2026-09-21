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
