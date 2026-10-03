"""Validated camera-capture and PPO runtime parameters for Sim2Real.

These values originate in the canonical compositional Sim2Real workflow and are consumed by
the real Isaac rollout/eval/trainer sibling Jobs.  Keeping parsing here makes
submission, Job materialization, and artifact metadata agree on one contract.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

DEFAULT_CAPTURE_WIDTH = 640
DEFAULT_CAPTURE_HEIGHT = 480
DEFAULT_ROLLOUT_CAPTURE_STRIDE = 1
DEFAULT_HELDOUT_CAPTURE_STRIDE = 20
DEFAULT_PNG_COMPRESS_LEVEL = 3
DEFAULT_CAPTURE_FPS = 10.0
DEFAULT_PPO_NUM_ENVS = 1024
# The canonical loop resumes three inner passes.  A 2,000-update pass gives the
# manipulation curriculum enough runway to reach the grasp/place rewards while
# still publishing a validation-ranked checkpoint after every pass.
DEFAULT_PPO_ITERATIONS = 2_000
DEFAULT_PPO_STEPS_PER_ENV = 24


def _integer(
    values: Mapping[str, Any], name: str, default: int, *, minimum: int, maximum: int
) -> int:
    raw = values.get(name, default)
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}], got {value}")
    return value


def _number(
    values: Mapping[str, Any],
    name: str,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    raw = values.get(name, default)
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}], got {value}")
    return value


def capture_settings(values: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return the validated, artifact-ready Isaac capture contract."""

    source = os.environ if values is None else values
    return {
        "width": _integer(
            source,
            "NPA_SIM2REAL_CAPTURE_WIDTH",
            DEFAULT_CAPTURE_WIDTH,
            minimum=320,
            maximum=4096,
        ),
        "height": _integer(
            source,
            "NPA_SIM2REAL_CAPTURE_HEIGHT",
            DEFAULT_CAPTURE_HEIGHT,
            minimum=240,
            maximum=2160,
        ),
        "rollout_stride": _integer(
            source,
            "NPA_SIM2REAL_ROLLOUT_CAPTURE_STRIDE",
            DEFAULT_ROLLOUT_CAPTURE_STRIDE,
            minimum=1,
            maximum=10_000,
        ),
        "heldout_stride": _integer(
            source,
            "NPA_SIM2REAL_HELDOUT_CAPTURE_STRIDE",
            DEFAULT_HELDOUT_CAPTURE_STRIDE,
            minimum=1,
            maximum=10_000,
        ),
        "png_compress_level": _integer(
            source,
            "NPA_SIM2REAL_PNG_COMPRESS_LEVEL",
            DEFAULT_PNG_COMPRESS_LEVEL,
            minimum=0,
            maximum=9,
        ),
        "fps": _number(
            source,
            "NPA_SIM2REAL_CAPTURE_FPS",
            DEFAULT_CAPTURE_FPS,
            minimum=0.1,
            maximum=240.0,
        ),
    }


def ppo_settings(values: Mapping[str, Any] | None = None) -> dict[str, int]:
    """Return validated RSL-RL PPO workload dimensions."""

    source = os.environ if values is None else values
    num_envs = _integer(
        source,
        "NPA_BYO_ISAAC_NUM_ENVS",
        DEFAULT_PPO_NUM_ENVS,
        minimum=1,
        maximum=65_536,
    )
    iterations = _integer(
        source,
        "NPA_BYO_ISAAC_ITERATIONS",
        DEFAULT_PPO_ITERATIONS,
        minimum=1,
        maximum=1_000_000,
    )
    steps_per_env = _integer(
        source,
        "NPA_BYO_ISAAC_STEPS_PER_ENV",
        DEFAULT_PPO_STEPS_PER_ENV,
        minimum=1,
        maximum=16_384,
    )
    return {
        "num_envs": num_envs,
        "iterations": iterations,
        "steps_per_env": steps_per_env,
        "total_environment_steps": num_envs * iterations * steps_per_env,
    }


def runtime_parameter_metadata(
    values: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Combined parameters persisted into reports and visualization provenance."""

    return {"capture": capture_settings(values), "ppo": ppo_settings(values)}
