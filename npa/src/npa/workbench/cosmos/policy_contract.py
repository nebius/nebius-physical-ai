"""Pinned native LIBERO policy contracts and evaluation evidence validation."""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

FRAMEWORK_REVISION = "2a8339d46a6e10e96f26c98509e6080d04ead490"
LIBERO_REVISION = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
DATASET_REVISION = "e5907374380b8f96511957e6ba5582be52a1e179"
MODEL_REVISION = "7a312c868bcce8e40b3eb40861300a9d0ba3fde1"
VAE_REVISION = "921dbaf3f1674a56f47e83fb80a34bac8a8f203e"
EXPERIMENT = "action_policy_libero_nano"
RECIPE = "examples/toml/sft_config/action_policy_libero_10_nano.toml"
STATS = "cosmos_framework/data/generator/action/normalizer_stats/libero_native_frame_wise_relative_rot6d.json"
ACTION_CONTRACT = {
    "task_suite": "libero_10",
    "fps": 20,
    "action_dim": 10,
    "action_space": "frame_wise_relative",
    "rotation_space": "6d",
    "normalization": "quantile_rot",
    "coordinate_frame": "native",
    "camera": "agentview,wrist",
    "image_size": 256,
}


class TrainSettings(BaseModel):
    """Typed recipe changes; unknown keys cannot silently change native behavior."""

    model_config = ConfigDict(extra="forbid", strict=True)
    processes: int = Field(default=8, ge=1)
    iterations: int = Field(default=2000, ge=1)
    save_every: int = Field(default=500, ge=1)
    samples_per_rank: int = Field(default=128, ge=1)
    gradient_accumulation: int = Field(default=2, ge=1)
    seed: int = Field(default=42, ge=0)


class EvalSettings(BaseModel):
    """Native evaluation protocol, preserved alongside every measurement."""

    model_config = ConfigDict(extra="forbid", strict=True)
    trials_per_task: int = Field(default=50, ge=1, le=50)
    seed: int = Field(default=0, ge=0)
    task_ids: list[int] = Field(default_factory=lambda: list(range(10)))

    @field_validator("task_ids")
    @classmethod
    def valid_tasks(cls, value: list[int]) -> list[int]:
        """Require unique native LIBERO-10 task identifiers."""
        if (
            not value
            or len(set(value)) != len(value)
            or not set(value) <= set(range(10))
        ):
            raise ValueError(
                "task_ids must be unique LIBERO-10 IDs between zero and nine"
            )
        return value


def validate_summary(summary: dict[str, Any], settings: EvalSettings) -> None:
    """Reject incomplete or internally inconsistent native evaluation evidence.

    Args:
        summary: Native LIBERO summary.json object.
        settings: Requested task set and trial count.
    Returns:
        None when every requested episode has a consistent result.
    Raises:
        ValueError: Missing tasks, invalid counts, or incompatible action contract.
    """
    for key in ("task_suite", "action_dim", "action_space", "rotation_space"):
        if summary.get(key) != ACTION_CONTRACT[key]:
            raise ValueError(f"evaluation action contract mismatch: {key}")
    expected = sorted(settings.task_ids)
    if (
        not expected
        or len(set(expected)) != len(expected)
        or not set(expected) <= set(range(10))
    ):
        raise ValueError("evaluation requires unique LIBERO-10 task IDs")
    tasks = summary.get("task_results", [])
    if sorted(task["task_id"] for task in tasks) != expected:
        raise ValueError("evaluation did not complete the requested tasks")
    if sorted(summary.get("selected_task_ids", [])) != expected:
        raise ValueError("evaluation selected tasks differ from requested protocol")
    for task in tasks:
        _validate_task(task, settings.trials_per_task)
    episodes = sum(task["episodes"] for task in tasks)
    successes = sum(task["successes"] for task in tasks)
    if (
        summary.get("total_episodes") != episodes
        or summary.get("total_successes") != successes
    ):
        raise ValueError("evaluation totals disagree with per-task evidence")
    if summary.get("num_trials_per_task") != settings.trials_per_task:
        raise ValueError("evaluation trial count differs from requested protocol")
    _validate_rate(summary.get("overall_success_rate"), successes / episodes)


def _validate_rate(actual: Any, expected: float) -> None:
    if isinstance(actual, bool) or not isinstance(actual, (int, float)):
        raise ValueError("success rate must be numeric")
    if not math.isfinite(actual) or not math.isclose(actual, expected, abs_tol=1e-9):
        raise ValueError("success rate disagrees with episode counts")


def _validate_task(task: dict[str, Any], trials: int) -> None:
    results = task.get("episode_results", [])
    if task.get("episodes") != trials or len(results) != trials:
        raise ValueError("evaluation contains missing episodes")
    ids = [result.get("episode") for result in results]
    if sorted(ids) != list(range(trials)):
        raise ValueError("evaluation contains repeated or missing episode IDs")
    if any(type(result.get("success")) is not bool for result in results):
        raise ValueError("episode success must be a boolean")
    if any(result.get("error") is not None for result in results):
        raise ValueError("evaluation contains simulator or policy-server errors")
    if any(
        type(result.get("steps")) is not int or result["steps"] <= 0
        for result in results
    ):
        raise ValueError(
            "evaluation requires completed simulator steps for every episode"
        )
    successes = sum(result["success"] for result in results)
    if task.get("successes") != successes:
        raise ValueError("task success count disagrees with episode evidence")
    if (
        not isinstance(task.get("task_description"), str)
        or not task["task_description"].strip()
    ):
        raise ValueError("task description is required for feedback")
    _validate_rate(task.get("success_rate"), successes / trials)
