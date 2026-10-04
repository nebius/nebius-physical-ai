"""Reject corrupted physical evidence before it can admit a training episode."""

from __future__ import annotations

import numpy as np
import pytest

from npa.workbench.token_factory.robot_sim import physics_checks


def _trace():
    count = 20
    positions = np.tile([0.1, 0.1, 0.425], (count, 1))
    positions[5, 2] += 0.15
    states = np.zeros((count, 9))
    states[:, -2:] = 0.05
    return {
        "state": states.copy(),
        "next_state": states,
        "actions": np.zeros((count, 4)),
        "object_position": np.tile([-0.1, -0.1, 0.425], (count, 1)),
        "next_object_position": positions,
        "goal": np.array([0.1, 0.1, 0.425]),
        "next_gripper_position": np.tile([0.1, 0.1, 0.575], (count, 1)),
        "finger_contacts": np.full(count, 2.0),
        "environment_success": np.ones(count, dtype=bool),
    }


def test_complete_physical_evidence_still_passes():
    result = physics_checks(_trace())
    assert result["accepted"] and all(result["checks"].values())


@pytest.mark.parametrize(
    "key",
    [
        "object_position",
        "next_gripper_position",
        "finger_contacts",
        "environment_success",
    ],
)
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_evidence_never_passes(key, value):
    arrays = _trace()
    arrays[key] = arrays[key].astype(float)
    arrays[key].flat[0] = value
    try:
        result = physics_checks(arrays)
    except ValueError:
        return
    assert result["accepted"] is False


@pytest.mark.parametrize("key", [key for key in _trace() if key != "goal"])
def test_misaligned_trace_lengths_raise_value_error(key):
    arrays = _trace()
    arrays[key] = arrays[key][:-1]
    with pytest.raises(ValueError):
        physics_checks(arrays)


@pytest.mark.parametrize(
    "key", ["state", "next_state", "actions", "next_object_position"]
)
def test_wrong_feature_width_raises_value_error(key):
    arrays = _trace()
    arrays[key] = arrays[key][:, :-1]
    with pytest.raises(ValueError):
        physics_checks(arrays)


def test_one_transition_cannot_establish_settling():
    arrays = {
        key: value if key == "goal" else value[:1] for key, value in _trace().items()
    }
    with pytest.raises(ValueError):
        physics_checks(arrays)


@pytest.mark.parametrize("key", list(_trace()))
def test_required_physics_evidence_cannot_be_omitted(key):
    arrays = _trace()
    del arrays[key]
    with pytest.raises(ValueError):
        physics_checks(arrays)


@pytest.mark.parametrize("key", ["actions", "state", "finger_contacts"])
def test_object_arrays_cannot_hide_nonfinite_physics(key):
    arrays = _trace()
    arrays[key] = arrays[key].astype(object)
    arrays[key].flat[0] = np.nan
    try:
        result = physics_checks(arrays)
    except ValueError:
        return
    assert result["accepted"] is False


@pytest.mark.parametrize("key", ["state", "actions"])
@pytest.mark.parametrize("dtype", [str, complex])
def test_physical_vectors_require_real_numeric_values(key, dtype):
    arrays = _trace()
    arrays[key] = arrays[key].astype(dtype)
    try:
        result = physics_checks(arrays)
    except ValueError:
        return
    assert result["accepted"] is False


@pytest.mark.parametrize("key", ["finger_contacts", "environment_success"])
def test_scalar_measurements_have_one_value_per_timestep(key):
    arrays = _trace()
    arrays[key] = arrays[key][:, None]
    with pytest.raises(ValueError):
        physics_checks(arrays)
