"""Run the official TRAIN evaluator with lossless experience hooks."""

from __future__ import annotations

import os
import json
from pathlib import Path
import time
from typing import Any

try:
    from train_experience import EvaluatorExperienceRecorder
except ModuleNotFoundError:
    from .train_experience import EvaluatorExperienceRecorder

_ROOT_ENV = "NPA_TRAIN_EXPERIENCE_ROOT"


def install() -> None:
    """Install process-local recording hooks on the official evaluator.

    Args:
        None.
    Returns:
        None.
    Raises:
        KeyError: The run-owned experience root is absent.
        ImportError: The pinned official evaluator is unavailable.
    """
    from omnigibson.eval.evaluator import BatchedEvaluator

    BatchedEvaluator.__init__ = _initialize_hook(BatchedEvaluator.__init__)
    BatchedEvaluator._apply_actions = _apply_hook(BatchedEvaluator._apply_actions)
    BatchedEvaluator.run = _run_hook(BatchedEvaluator.run)


def _initialize_hook(original):
    def initialize(evaluator, config):
        _validate_config(config)
        original(evaluator, config)
        recorder = EvaluatorExperienceRecorder(Path(os.environ[_ROOT_ENV]))
        evaluator.policy = _recording_policy(evaluator.policy, recorder)
        evaluator._npa_train_experience = recorder

    return initialize


def _apply_hook(original):
    def apply(evaluator, actions, active_indices):
        _active_environment(active_indices)
        started = time.monotonic_ns()
        result = original(evaluator, actions, active_indices)
        completed = time.monotonic_ns()
        evaluator._npa_train_experience.record_applied(
            evaluator._batch_obs(), actions[0], started, completed
        )
        return result

    return apply


def _run_hook(original):
    def run(evaluator, instances, **kwargs):
        recorder = evaluator._npa_train_experience
        expected_instance = recorder.config["case"]["instance_id"]
        if list(instances) != [expected_instance] or kwargs.get("rollout_id") != 0:
            raise ValueError("TRAIN experience requires one prescribed rollout")
        result = original(evaluator, instances, **kwargs)
        recorder.close()
        return result

    return run


def _active_environment(indices) -> None:
    if list(indices) != [0]:
        raise ValueError("TRAIN experience requires active environment zero")


def _recording_policy(delegate: Any, recorder: EvaluatorExperienceRecorder) -> Any:
    class RecordingPolicy:
        def reset(self) -> None:
            delegate.reset()
            recorder.reset()

        def forward(self, *, obs):
            started = time.monotonic_ns()
            action = delegate.forward(obs=obs)
            completed = time.monotonic_ns()
            recorder.record_policy(obs, action, started, completed)
            return action

    return RecordingPolicy()


def _get(value: Any, name: str) -> Any:
    return value.get(name) if hasattr(value, "get") else getattr(value, name)


def _validate_config(config: Any) -> None:
    experience = json.loads((Path(os.environ[_ROOT_ENV]) / "config.json").read_text())
    task = _get(config, "task")
    model = _get(config, "model")
    observed = (
        _get(config, "mode"),
        int(_get(config, "num_envs")),
        _get(task, "name"),
        bool(_get(config, "write_video")),
        _get(config, "policy_name"),
        int(_get(model, "action_chunk_size")),
    )
    expected = (
        "train",
        1,
        experience["case"]["task"],
        True,
        "websocket",
        0,
    )
    if observed != expected:
        raise ValueError("TRAIN experience evaluator scope differs")


def main() -> None:
    """Install recording before invoking the unchanged official CLI.

    Args:
        None.
    Returns:
        None.
    Raises:
        ValueError: Official TRAIN scope differs.
    """
    install()
    from omnigibson.eval.eval import main as official_main

    official_main()


if __name__ == "__main__":
    main()
