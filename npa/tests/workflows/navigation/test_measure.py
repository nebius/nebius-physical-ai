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
        "physical_failure": np.zeros(recipe.num_envs),
        "upright_cosine": np.ones(recipe.num_envs),
        "ground_clearance_m": np.full(recipe.num_envs, 0.6),
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


def test_goal_xy_while_fallen_is_never_success(recipe):
    initial, fallen, later = state(recipe), state(recipe), state(recipe)
    fallen["position_m"][:, :2] = fallen["goal_m"]
    fallen["position_m"][:, 2] = -3.0
    fallen["physical_failure"][:] = 1
    later["position_m"][:, :2] = later["goal_m"]
    later["active"][:] = False
    rows = measure.episode_rows(recipe.eval_cases, [initial, fallen, later], 0.2)
    assert all(not row["success"] for row in rows)
    assert all(row["physical_failure_steps"] == 1 for row in rows)
    assert all(row["collision_steps"] == 0 for row in rows)


@pytest.mark.parametrize("value", [-1, 0.5, 2])
def test_physical_failure_must_be_boolean(recipe, value):
    data = state(recipe)
    data["physical_failure"][0] = value
    with pytest.raises(ValueError, match="boolean"):
        measure.snapshot(SimpleNamespace(measure=lambda _: data), None, 2)


@pytest.mark.parametrize(
    "key",
    [
        "position_m",
        "goal_m",
        "obstacle_contact",
        "peer_contact",
        "physical_failure",
        "upright_cosine",
        "ground_clearance_m",
    ],
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


def test_probe_actions_allow_subsequent_native_metric_resets(recipe, monkeypatch):
    torch = pytest.importorskip("torch")
    low_level_policy = torch.nn.Linear(3, 2).eval()
    metrics = {"error_pos": torch.zeros(recipe.num_envs)}

    def reset(*args):
        metrics["error_pos"][torch.arange(recipe.num_envs)] = 0.0

    def step(actions):
        targets = low_level_policy(actions)
        assert not targets.requires_grad
        metrics["error_pos"] = torch.linalg.vector_norm(targets, dim=-1)
        return targets, None, torch.zeros(recipe.num_envs, dtype=torch.bool), {}

    monkeypatch.setattr(measure, "verify_reset", reset)
    adapter = SimpleNamespace(measure=lambda _: state(recipe))
    env = SimpleNamespace(unwrapped=SimpleNamespace(device="cpu"))
    wrapped = SimpleNamespace(
        get_observations=lambda: torch.zeros((recipe.num_envs, 2)), step=step
    )
    with torch.enable_grad():
        for _ in range(2):
            trace = measure._probe_trace(
                adapter, env, wrapped, recipe.eval_cases, [[0.6, 0.0, 0.0]], 1e-3
            )
        reset()
        assert torch.is_grad_enabled()
    assert len(trace) == 2
    assert trace[-1]["observations"]["value"].shape == (recipe.num_envs, 2)


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
