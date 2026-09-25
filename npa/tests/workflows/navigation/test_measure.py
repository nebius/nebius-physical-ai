"""Check navigation metrics and peer-probe decisions using measurement fixtures."""

from types import SimpleNamespace

import numpy as np
import pytest

from npa.workflows.navigation import measure


def state(recipe, offset=0.0):
    return {
        "position_m": np.array([case.position_m for case in recipe.eval_cases])
        + offset,
        "goal_m": np.array([case.goal_m for case in recipe.eval_cases]),
        "heading_rad": np.zeros(recipe.num_envs),
        "obstacle_contact": np.zeros(recipe.num_envs),
        "peer_contact": np.zeros(recipe.num_envs),
        "active": np.ones(recipe.num_envs, dtype=bool),
    }


def test_metrics_use_collision_free_goal_distance_not_reward(recipe):
    initial, final = state(recipe), state(recipe)
    final["position_m"][:, :2] = final["goal_m"]
    final["obstacle_contact"][1] = 2.0
    rows = measure.episode_rows(
        recipe.eval_cases, [initial, final], recipe.goal_tolerance_m
    )
    assert rows[0]["success"] is True
    assert rows[1]["success"] is False
    assert rows[0]["path_length_m"] == 5.0
    assert rows[1]["collision_steps"] == 1


def test_success_after_collision_is_not_recovered(recipe):
    initial, collision, reset = state(recipe), state(recipe), state(recipe)
    collision["obstacle_contact"][0] = 1
    reset["position_m"][:, :2] = reset["goal_m"]
    reset["active"][0] = False
    rows = measure.episode_rows(recipe.eval_cases, [initial, collision, reset], 0.2)
    assert rows[0]["success"] is False
    assert rows[0]["steps"] == 1


@pytest.mark.parametrize(
    "key", ["position_m", "goal_m", "obstacle_contact", "peer_contact"]
)
def test_missing_measurements_rejected(recipe, key):
    data = state(recipe)
    del data[key]
    with pytest.raises(KeyError):
        measure.snapshot(SimpleNamespace(measure=lambda _: data), None, 2)


def test_negative_contacts_are_not_silently_treated_as_zero(recipe):
    data = state(recipe)
    data["peer_contact"][0] = -1
    with pytest.raises(ValueError, match="negative"):
        measure.snapshot(SimpleNamespace(measure=lambda _: data), None, 2)


@pytest.mark.parametrize(
    "group,key",
    [
        ("state", "position_m"),
        ("state", "peer_contact"),
        ("observations", "critic"),
        ("observations", "policy"),
    ],
)
def test_peer_motion_and_all_observation_streams_checked(recipe, group, key):
    baseline = {
        "state": state(recipe),
        "observations": {"policy": np.ones((2, 3)), "critic": np.ones((2, 4))},
    }
    baseline["state"].pop("active")
    changed = {
        name: {field: a.copy() for field, a in values.items()}
        for name, values in baseline.items()
    }
    changed[group][key][0] += 1
    with pytest.raises(ValueError):
        measure._compare_traces([baseline], [changed], 1e-5)


def test_probe_requires_motion_and_positive_obstacle_contact(recipe, monkeypatch):
    def trace(*args):
        row = {"state": state(recipe), "observations": {"policy": np.ones((2, 3))}}
        row["state"].pop("active")
        row["state"]["position_m"][0] = recipe.probe.free.position_m
        return [row, row]

    monkeypatch.setattr(measure, "_probe_trace", trace)
    with pytest.raises(ValueError, match="did not move"):
        measure.probe_isolation(None, None, None, recipe)
    baseline = trace()
    with pytest.raises(ValueError, match="no physical contact"):
        measure._probe_controls(
            None, None, None, recipe, [recipe.probe.free] * 2, baseline
        )


def test_reset_pose_and_goal_are_independently_checked(recipe):
    data = state(recipe)
    data["goal_m"][0] += 1
    adapter = SimpleNamespace(reset=lambda *args: None, measure=lambda _: data)
    with pytest.raises(ValueError, match="goal_m"):
        measure.verify_reset(
            adapter,
            SimpleNamespace(unwrapped=SimpleNamespace()),
            recipe.eval_cases,
            1e-5,
        )


@pytest.mark.parametrize("obs", [{}, np.array([[np.nan]]), np.array([])])
def test_empty_or_nonfinite_observations_fail(obs):
    with pytest.raises(ValueError):
        measure.observations(obs)
