"""Run the official TRAIN evaluator with lossless experience hooks."""

from __future__ import annotations

import os
import inspect
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

try:
    from train_experience import (
        GOAL_PROGRESS_SOURCE,
        EvaluatorExperienceRecorder,
        file_identity,
    )
except ModuleNotFoundError:
    from .train_experience import (
        GOAL_PROGRESS_SOURCE,
        EvaluatorExperienceRecorder,
        file_identity,
    )

try:
    from train_official_q import bind_official_q_observer
except ModuleNotFoundError:
    from .train_official_q import bind_official_q_observer

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

    source = _official_goal_source()
    BatchedEvaluator.__init__ = _initialize_hook(BatchedEvaluator.__init__, source)
    BatchedEvaluator._apply_actions = _apply_hook(BatchedEvaluator._apply_actions)
    BatchedEvaluator.run = _run_hook(BatchedEvaluator.run)


def _initialize_hook(original, progress_source=None):
    def initialize(evaluator, config):
        _validate_config(config)
        original(evaluator, config)
        official_q = bind_official_q_observer(evaluator)
        recorder = EvaluatorExperienceRecorder(
            Path(os.environ[_ROOT_ENV]),
            progress_source=progress_source,
            official_q_source=official_q.source,
        )
        evaluator.policy = _recording_policy(evaluator.policy, recorder)
        evaluator._npa_train_experience = recorder
        evaluator._npa_train_official_q = official_q

    return initialize


def _apply_hook(original):
    def apply(evaluator, actions, active_indices):
        _active_environment(active_indices)
        started = time.monotonic_ns()
        result = original(evaluator, actions, active_indices)
        completed = time.monotonic_ns()
        recorder = evaluator._npa_train_experience
        step = _official_step(result) if recorder.progress_enabled else None
        official_q = (
            evaluator._npa_train_official_q.observe(recorder.frame)
            if recorder.official_q_enabled
            else None
        )
        recorder.record_applied(
            evaluator._batch_obs(),
            actions[0],
            started,
            completed,
            official_step=step,
            official_q=official_q,
        )
        return result

    return apply


def _official_goal_source() -> dict[str, Any]:
    from bddl.condition_evaluation import evaluate_state
    from omnigibson.envs.env_base import Environment
    from omnigibson.eval.evaluator import BatchedEvaluator
    from omnigibson.tasks.behavior_task import BehaviorTask
    from omnigibson.tasks.task_base import BaseTask
    from omnigibson.termination_conditions.predicate_goal import PredicateGoal

    classes = (
        BatchedEvaluator,
        BehaviorTask,
        BaseTask,
        Environment,
        PredicateGoal,
        evaluate_state,
    )
    suffixes = tuple(GOAL_PROGRESS_SOURCE["files"])
    observed = {}
    for cls, suffix in zip(classes, suffixes, strict=True):
        path = Path(inspect.getsourcefile(cls) or "")
        if not path.as_posix().endswith(suffix):
            raise ValueError("Official goal-status source path differs")
        observed[suffix] = file_identity(path)
    if observed != GOAL_PROGRESS_SOURCE["files"]:
        raise ValueError("Official goal-status source bytes differ")
    return json.loads(json.dumps(GOAL_PROGRESS_SOURCE))


def _official_step(result: Any) -> dict[str, Any]:
    if not isinstance(result, tuple) or len(result) != 3:
        raise ValueError("Official evaluator step result differs")
    terminated, truncated, info = result
    if not isinstance(info, list) or len(info) != 1:
        raise ValueError("Official evaluator step info differs")
    done = info[0].get("done") if isinstance(info[0], dict) else None
    if not isinstance(done, dict):
        raise ValueError("Official evaluator done info differs")
    return {
        "goal_status": done.get("goal_status"),
        "terminated": _single_bool(terminated),
        "truncated": _single_bool(truncated),
    }


def _single_bool(value: Any) -> bool:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.shape != (1,) or array.dtype != np.bool_:
        raise ValueError("Official evaluator termination array differs")
    return bool(array[0])


def _run_hook(original):
    def run(evaluator, instances, **kwargs):
        recorder = evaluator._npa_train_experience
        expected_instance = recorder.config["case"]["instance_id"]
        if list(instances) != [expected_instance] or kwargs.get("rollout_id") != 0:
            raise ValueError("TRAIN experience requires one prescribed rollout")
        result = original(evaluator, instances, **kwargs)
        recorder.close(result)
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
