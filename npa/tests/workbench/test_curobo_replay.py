"""Independent cuRobo replay helpers stay separate and fail closed."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from npa.workbench.curobo import replay


def test_quaternion_replay_distance_is_sign_invariant():
    identity = np.asarray([1.0, 0.0, 0.0, 0.0])
    assert replay._quaternion_distance(identity, identity) == 0.0
    assert replay._quaternion_distance(identity, -identity) == 0.0
    assert replay._quaternion_distance(
        identity, np.asarray([0.0, 1.0, 0.0, 0.0])
    ) == pytest.approx(np.pi)


def test_replay_comparison_rejects_shape_and_value_changes():
    expected = np.asarray([[0.0, 0.1], [0.2, 0.3]])
    assert (
        replay._require_close(expected.copy(), expected, name="fixture", atol=1e-9)
        == 0.0
    )
    with pytest.raises(replay.ReplayError, match="shape"):
        replay._require_close(expected[:, :1], expected, name="fixture", atol=1e-9)
    changed = expected.copy()
    changed[1, 1] += 1e-4
    with pytest.raises(replay.ReplayError, match="differs"):
        replay._require_close(changed, expected, name="fixture", atol=1e-9)


def test_replay_does_not_import_producer_or_validator_helpers():
    source = inspect.getsource(replay)
    assert ".runner import" not in source
    assert ".artifacts import" not in source
    assert ".audit import" not in source
