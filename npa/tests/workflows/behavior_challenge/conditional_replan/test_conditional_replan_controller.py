"""Tests for rejection rollback and accepted proposal commit semantics."""

from __future__ import annotations

from collections import deque

import numpy as np
import pytest
from npa.workflows.behavior_challenge.conditional_replan.controller import (
    ConditionalQueueController,
)


class _Policy:
    _is_pytorch_model = False

    def __init__(self) -> None:
        self._rng = 0

    def _input_transform(self, observation):
        return {"state": np.full(32, observation["observation/state"][0], np.float32)}

    def infer(self, batch):
        self._rng += 1
        return {"actions": np.full((32, 23), self._rng, np.float64)}


class _Wrapper:
    control_mode, max_len, fine_grained_level = "receeding_horizon", 32, 0

    def __init__(self) -> None:
        self.policy, self.action_queue, self.last_action = _Policy(), deque(), None
        self.task_prompt = "synthetic task"

    def process_obs(self, observation):
        return {
            "observation": np.zeros((1, 3, 2, 2, 3), np.uint8),
            "proprio": np.asarray([[observation["state"]]], np.float32),
        }

    def reset(self) -> None:
        self.action_queue.clear()

    def act(self, observation):
        if not self.action_queue:
            processed = self.process_obs(observation)
            batch = {"observation/state": processed["proprio"][0]}
            result = self.policy.infer(self.policy._input_transform(batch))
            self.action_queue = deque(result["actions"])
        return self.action_queue.popleft()


def _controller(decide) -> ConditionalQueueController:
    wrapper = _Wrapper()
    return ConditionalQueueController(
        wrapper, lambda value: value.astype(np.float32), decide
    )


def test_rejection_preserves_remaining_actions_and_rng() -> None:
    controller = _controller(lambda _features: False)
    emitted = [controller.act({"state": float(index)}) for index in range(17)]
    assert all(np.array_equal(action, np.ones(23)) for action in emitted)
    assert controller.wrapper.policy._rng == 1
    assert len(controller.wrapper.action_queue) == 15


def test_acceptance_commits_new_queue_and_advanced_rng() -> None:
    controller = _controller(lambda _features: True)
    emitted = [controller.act({"state": float(index)}) for index in range(17)]
    assert np.array_equal(emitted[-1], np.full(23, 2.0))
    assert controller.wrapper.policy._rng == 2
    assert len(controller.wrapper.action_queue) == 31


def test_decision_error_rolls_back_rng() -> None:
    def reject(_features):
        raise RuntimeError("decision failed")

    controller = _controller(reject)
    for index in range(16):
        controller.act({"state": float(index)})
    with pytest.raises(RuntimeError, match="decision failed"):
        controller.act({"state": 16.0})
    assert controller.wrapper.policy._rng == 1
