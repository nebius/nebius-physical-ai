"""Verify Arena accepts the upstream replay contract without requiring optional motion history."""

from __future__ import annotations

import hashlib
from pathlib import Path

import h5py
import numpy as np
import pytest

from npa.workbench.isaac_arena.errors import IsaacArenaError
from npa.workbench.isaac_arena.replay_input import (
    _prepare_replay_execution_input,
    _replay_input_evidence,
)


def _replay(
    path: Path, actions: np.ndarray, *, states: np.ndarray | None = None
) -> None:
    with h5py.File(path, "w") as dataset:
        episode = dataset.create_group("data/demo_0")
        episode.create_dataset("actions", data=actions)
        episode.create_dataset(
            "initial_state/articulation/robot/joint_position",
            data=np.array([[0.1, 0.2]], dtype=np.float32),
        )
        if states is not None:
            episode.create_dataset(
                "states/articulation/robot/joint_position", data=states
            )
        episode.attrs["success"] = False


def test_minimal_upstream_replay_needs_no_optional_state_history(
    tmp_path: Path,
) -> None:
    path = tmp_path / "minimal.hdf5"
    _replay(path, np.array([[0.1, 0.2], [0.2, 0.3]], dtype=np.float32))
    evidence = _replay_input_evidence(path)
    assert evidence["steps"] == 2
    assert evidence["nonzero_actions"] is True
    assert evidence["recorded_state_available"] is False
    assert evidence["recorded_state_changes"] is False
    assert evidence["state_max_range"] is None
    assert evidence["runtime_outcome_claim"] is False


def test_zero_action_replay_is_a_factual_evaluation_input(tmp_path: Path) -> None:
    path = tmp_path / "zero.hdf5"
    _replay(path, np.zeros((20, 2), dtype=np.float32))
    evidence = _replay_input_evidence(path)
    assert evidence["nonzero_actions"] is False
    assert evidence["action_abs_max"] == 0.0
    assert evidence["action_nonzero_fraction"] == 0.0
    assert evidence["source_recorded_success"] is False
    assert evidence["runtime_outcome_claim"] is False


def test_static_optional_state_history_is_diagnostic_not_a_rejection(
    tmp_path: Path,
) -> None:
    path = tmp_path / "static-state.hdf5"
    _replay(path, np.ones((20, 2)), states=np.ones((20, 2)))
    evidence = _replay_input_evidence(path)
    assert evidence["recorded_state_available"] is True
    assert evidence["recorded_state_changes"] is False
    assert evidence["state_max_range"] == 0


@pytest.mark.parametrize(
    "actions",
    [
        np.array([[0, np.nan], [0, 1]]),
        np.array([[0, np.inf], [0, 1]]),
        np.array([[1, 2]]),
        np.array([1, 2]),
        np.empty((2, 0)),
        np.array([[1j, 2], [3, 4]]),
    ],
)
def test_invalid_action_tensor_still_fails_before_simulator(
    tmp_path: Path, actions: np.ndarray
) -> None:
    path = tmp_path / "invalid-actions.hdf5"
    _replay(path, actions)
    with pytest.raises(IsaacArenaError, match="finite multi-step tensor"):
        _replay_input_evidence(path)


@pytest.mark.parametrize("initial", [None, np.array([np.nan]), np.array([])])
def test_initial_state_remains_required_and_finite(
    tmp_path: Path, initial: np.ndarray | None
) -> None:
    path = tmp_path / "invalid-initial.hdf5"
    _replay(path, np.zeros((2, 2)))
    with h5py.File(path, "a") as dataset:
        del dataset["data/demo_0/initial_state"]
        if initial is not None:
            dataset.create_dataset(
                "data/demo_0/initial_state/articulation/robot/joint_position",
                data=initial,
            )
    with pytest.raises(IsaacArenaError, match="initial_state"):
        _replay_input_evidence(path)


def test_minimal_replay_normalization_preserves_every_action_and_initial_tensor(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.hdf5"
    actions = np.arange(160, dtype=np.float32).reshape(80, 2)
    _replay(source, actions)
    original_bytes = source.read_bytes()
    evidence = {
        "trajectory": _replay_input_evidence(source),
        "sha256": hashlib.sha256(original_bytes).hexdigest(),
    }
    execution, record = _prepare_replay_execution_input(
        source, tmp_path / "private", source_evidence=evidence
    )
    with h5py.File(source, "r") as input_file, h5py.File(execution, "r") as executed:
        np.testing.assert_array_equal(executed["data/demo_0/actions"], actions)
        initial_key = "data/demo_0/initial_state/articulation/robot/joint_position"
        np.testing.assert_array_equal(executed[initial_key], input_file[initial_key])
        assert set(executed["data/demo_0"]) == {"actions", "initial_state"}
    assert source.read_bytes() == original_bytes
    assert execution.stat().st_mode & 0o777 == 0o600
    assert record["prepared_steps"] == record["source_steps"] == 80
    assert record["action_padding_steps"] == 0
    assert record["published"] is False


def test_integer_action_diagnostics_do_not_overflow_or_mutate_replay(
    tmp_path: Path,
) -> None:
    path = tmp_path / "integer.hdf5"
    _replay(path, np.array([[-128, -128], [127, 127]], dtype=np.int8))
    evidence = _replay_input_evidence(path)
    assert evidence["nonzero_actions"] is True
    assert evidence["action_abs_max"] == 128
    assert evidence["action_step_delta_mean"] == 255
    with h5py.File(path, "r") as source:
        assert source["data/demo_0/actions"].dtype == np.dtype("int8")
