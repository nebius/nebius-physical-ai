"""Protect batched MJLab accounting, finite-value checks and recurrent resets."""

from types import SimpleNamespace

import pytest

from npa.workbench.mjlab.rollout import measure_episodes
from npa.workbench.mjlab.schemas import EvalRequest, MjlabError


class _Environment:
    def __init__(self, torch, lanes):
        self.torch = torch
        self.num_envs = lanes
        self.max_episode_length = 4
        self.unwrapped = self
        self.steps = 0
        self.done_history = []

    def get_observations(self):
        return self.torch.zeros((self.num_envs, 1))

    def step(self, actions):
        self.steps += 1
        self.reset_terminated = self.torch.zeros(self.num_envs, dtype=self.torch.bool)
        self.reset_time_outs = self.torch.ones(self.num_envs, dtype=self.torch.bool)
        dones = self.torch.ones(self.num_envs, dtype=self.torch.bool)
        self.done_history.append(dones)
        return self.get_observations(), self.torch.ones(self.num_envs), dones, {}


def _request(lanes, episodes):
    return EvalRequest(
        checkpoint="s3://fixture/model.pt",
        output_path="s3://fixture/evaluation",
        num_envs=lanes,
        episodes=episodes,
    )


def test_scalar_reads_do_not_grow_with_environment_count(tmp_path):
    """Prevent CUDA scalar synchronization from scaling with vectorized lanes."""
    torch = pytest.importorskip("torch")
    scalar_reads = []
    for lanes in (1, 64):
        env = _Environment(torch, lanes)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as p:
            report = measure_episodes(
                env, lambda obs: torch.zeros_like(obs), _request(lanes, lanes), tmp_path
            )
        scalar_reads.append(
            sum(
                e.count
                for e in p.key_averages()
                if e.key == "aten::_local_scalar_dense"
            )
        )
        assert report["episodes_completed"] == lanes
        assert report["score"] == 1
    assert scalar_reads[1] == scalar_reads[0]


class _UnevenEnvironment(_Environment):
    def step(self, actions):
        self.steps += 1
        torch = self.torch
        self.reset_terminated = torch.tensor(
            [True, self.steps == 2, False], dtype=torch.bool
        )
        self.reset_time_outs = torch.tensor(
            [False, self.steps % 2 == 0, self.steps % 3 == 0]
        )
        dones = torch.tensor([True, self.steps % 2 == 0, self.steps % 3 == 0])
        self.done_history.append(dones)
        rewards = torch.tensor([1.0 if self.steps <= 2 else float("nan"), 2.0, 3.0])
        return self.get_observations(), rewards, dones, {}


def test_uneven_quotas_and_recurrent_resets_preserve_episode_records(tmp_path):
    """Ignore completed lanes while retaining native reset tensors and quotas."""
    torch = pytest.importorskip("torch")
    env = _UnevenEnvironment(torch, 3)
    reset_calls = []

    class Policy:
        def __call__(self, observations):
            return torch.zeros_like(observations)

        def reset(self, dones):
            reset_calls.append(dones)

    report = measure_episodes(env, Policy(), _request(3, 5), tmp_path)
    actual = [
        (r["environment"], r["episode"], r["return"], r["length"], r["survived"])
        for r in report["episodes"]
    ]
    assert actual == [
        (0, 0, 1.0, 1, False),
        (0, 1, 1.0, 1, False),
        (1, 0, 4.0, 2, False),
        (2, 0, 9.0, 3, True),
        (1, 1, 4.0, 2, True),
    ]
    assert len(reset_calls) == env.steps == 4
    assert all(a is b for a, b in zip(reset_calls, env.done_history))
    assert report["score"] == pytest.approx(2 / 5)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_actions_are_rejected_before_simulation(tmp_path, value):
    """Reject invalid policy outputs before they reach the simulator."""
    torch = pytest.importorskip("torch")
    env = _Environment(torch, 2)
    with pytest.raises(MjlabError, match="non-finite actions"):
        measure_episodes(
            env,
            lambda obs: torch.full_like(obs, value),
            _request(2, 2),
            tmp_path,
        )
    assert env.steps == 0


def test_noncontiguous_step_vectors_preserve_reward_precision():
    """Copy noncontiguous vectors without narrowing high precision rewards."""
    torch = pytest.importorskip("torch")
    from npa.workbench.mjlab.rollout import _host_step_values

    rewards = torch.tensor([1.000000000001, 0, -2.000000000001, 0], dtype=torch.float64)
    dones = torch.tensor([False, True, False, True])
    values = _host_step_values(SimpleNamespace(), rewards[::2], dones[::2])
    assert values == ([1.000000000001, -2.000000000001], [False, False], None, None)
