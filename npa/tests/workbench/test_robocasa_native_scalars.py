"""Native outcomes must not acquire success through Python truthiness."""

from types import SimpleNamespace

import numpy as np
import pytest

from npa.workbench.robocasa import capabilities


MALFORMED = [
    "false",
    "true",
    [False],
    None,
    2,
    -1,
    0.5,
    np.nan,
    np.inf,
    np.array([1]),
    np.array(1),
]


@pytest.mark.parametrize("value", MALFORMED)
@pytest.mark.parametrize("source", ["success", "is_success", "goal_reached", "checker"])
def test_native_success_rejects_malformed_scalar(value, source):
    env = SimpleNamespace()
    info = {source: value}
    if source == "checker":
        env._check_success = lambda: value
        info = {}
    with pytest.raises(capabilities.RoboCasaError, match="boolean or exact numeric"):
        capabilities._native_task_success(env, info, 0.25)


@pytest.mark.parametrize(
    "value",
    [
        False,
        True,
        np.bool_(False),
        np.bool_(True),
        0,
        1,
        0.0,
        1.0,
        np.int64(1),
        np.float32(0),
    ],
)
def test_native_success_accepts_only_explicit_boolean_forms(value):
    result, sources = capabilities._native_task_success(
        SimpleNamespace(_check_success=lambda: value), {"success": value}, 0.25
    )
    assert type(result) is bool
    assert result == bool(value)
    assert sources == ["environment._check_success", "info.success"]


@pytest.mark.parametrize("value", MALFORMED)
@pytest.mark.parametrize("field", ["terminated", "truncated"])
def test_episode_rejects_malformed_termination_without_mutating_outcome(value, field):
    outcome = capabilities._empty_episode_outcome()
    arguments = {"reward": 0.0, "terminated": False, "truncated": False, "info": {}}
    arguments[field] = value
    with pytest.raises(capabilities.RoboCasaError, match="boolean or exact numeric"):
        capabilities._update_episode_outcome(outcome, SimpleNamespace(), **arguments)
    assert outcome == capabilities._empty_episode_outcome()


@pytest.mark.parametrize(
    "value",
    [
        "0",
        True,
        np.bool_(False),
        [0],
        np.array(0),
        np.array([0]),
        None,
        np.nan,
        np.inf,
        complex(0, 0),
    ],
)
def test_episode_rejects_non_numeric_or_nonfinite_reward(value):
    outcome = capabilities._empty_episode_outcome()
    with pytest.raises(capabilities.RoboCasaError, match="reward"):
        capabilities._update_episode_outcome(
            outcome, SimpleNamespace(), value, False, False, {}
        )
    assert outcome == capabilities._empty_episode_outcome()


def test_episode_rejects_reward_sum_overflow_before_mutation():
    outcome = capabilities._empty_episode_outcome()
    outcome["reward_sum"] = 1e308
    with pytest.raises(capabilities.RoboCasaError, match="non-finite"):
        capabilities._update_episode_outcome(
            outcome, SimpleNamespace(), 1e308, False, False, {"success": False}
        )
    assert outcome["reward_sum"] == 1e308
    assert outcome["final_reward"] == 0.0
