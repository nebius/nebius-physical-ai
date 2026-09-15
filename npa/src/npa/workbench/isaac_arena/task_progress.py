"""Register environment-specific progress semantics for visual qualification."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

from .errors import IsaacArenaError

OPEN_MICROWAVE_SUCCESS_THRESHOLD = 0.8
OPEN_MICROWAVE_MINIMUM_DELTA = 0.5

ProgressQualifier = Callable[[dict, Mapping[str, Any], int | None, bool], dict]


@dataclass(frozen=True)
class TaskProgressAdapter:
    """One explicit environment-to-native-progress qualification contract."""

    name: str
    environment: str
    signal_names: tuple[str, ...]
    qualifier: ProgressQualifier
    thresholds: Mapping[str, float]


def _microwave_trace(signals: Mapping[str, Any], expected_steps: int | None) -> Any:
    import numpy as np

    trace = signals.get("revolute_joint_state")
    if trace is None:
        raise IsaacArenaError(
            "microwave simulator ground truth has no revolute-joint trace"
        )
    values = np.asarray(trace, dtype=float)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 1 or values.size < 2:
        raise IsaacArenaError(
            "microwave simulator ground truth has no usable revolute-joint trace"
        )
    if expected_steps is not None and values.size - 1 > expected_steps:
        raise IsaacArenaError(
            "microwave simulator trace exceeds the exact replay action horizon"
        )
    return values


def _microwave_progress_interval(trace: Any, total_steps: int) -> dict[str, int]:
    import numpy as np

    changed = np.flatnonzero(trace > float(trace[0]) + 0.02)
    opened = np.flatnonzero(trace > OPEN_MICROWAVE_SUCCESS_THRESHOLD)
    if not changed.size or not opened.size:
        raise IsaacArenaError(
            "microwave simulator trace has no open-door progress interval"
        )
    return {
        "start_action_step": max(0, int(changed[0]) - 1),
        "end_action_step": int(opened[0]),
        "total_action_steps": total_steps,
    }


def _microwave_statistics(values: Any, require_progress: bool) -> dict[str, float]:
    initial, final = float(values[0]), float(values[-1])
    minimum, maximum = float(values.min()), float(values.max())
    if final <= OPEN_MICROWAVE_SUCCESS_THRESHOLD:
        raise IsaacArenaError(
            "microwave success is not supported by final simulator door openness"
        )
    delta = maximum - initial
    if require_progress and delta < OPEN_MICROWAVE_MINIMUM_DELTA:
        raise IsaacArenaError(
            "microwave simulator door progress is below the acceptance delta"
        )
    return {
        "initial": initial,
        "final": final,
        "minimum": minimum,
        "maximum": maximum,
        "delta": delta,
    }


def _microwave_motion_record(
    record: dict, values: Any, statistics: dict[str, float], total_steps: int
) -> dict:
    qualified = statistics["delta"] >= OPEN_MICROWAVE_MINIMUM_DELTA
    interval = _microwave_progress_interval(values, total_steps) if qualified else None
    return {
        "adapter": "arena.open-door.revolute-joint.v1",
        "environment": "gr1_open_microwave",
        "kind": "revolute_joint_openness",
        "source": "Arena simulator metric recorder",
        "signal": "revolute_joint_state",
        "episode": record["episode"],
        "episode_length": record["episode_length"],
        "initial_openness": statistics["initial"],
        "final_openness": statistics["final"],
        "minimum_openness": statistics["minimum"],
        "maximum_openness": statistics["maximum"],
        "openness_delta": statistics["delta"],
        "success_threshold": OPEN_MICROWAVE_SUCCESS_THRESHOLD,
        "minimum_required_delta": OPEN_MICROWAVE_MINIMUM_DELTA,
        "samples": int(values.size),
        "task_success": True,
        "progress_interval": interval,
        "visual_progress_qualified": qualified,
        "video_capture": record.get("video_capture"),
    }


def _qualify_open_microwave(
    record: dict,
    signals: Mapping[str, Any],
    expected_steps: int | None,
    require_progress: bool,
) -> dict:
    values = _microwave_trace(signals, expected_steps)
    if int(values.size - 1) != record["episode_length"]:
        raise IsaacArenaError(
            "simulator trace length disagrees with upstream episode length"
        )
    statistics = _microwave_statistics(values, require_progress)
    total_steps = expected_steps or int(values.size - 1)
    return _microwave_motion_record(record, values, statistics, total_steps)


_TASK_PROGRESS_ADAPTERS: Mapping[str, TaskProgressAdapter] = MappingProxyType(
    {
        "gr1_open_microwave": TaskProgressAdapter(
            name="arena.open-door.revolute-joint.v1",
            environment="gr1_open_microwave",
            signal_names=("revolute_joint_state",),
            qualifier=_qualify_open_microwave,
            thresholds=MappingProxyType(
                {
                    "final_openness_greater_than": OPEN_MICROWAVE_SUCCESS_THRESHOLD,
                    "minimum_peak_minus_initial_openness": OPEN_MICROWAVE_MINIMUM_DELTA,
                }
            ),
        )
    }
)


def task_progress_adapter(environment: str) -> TaskProgressAdapter | None:
    """Resolve an explicit environment adapter; unregistered tasks return None."""

    return _TASK_PROGRESS_ADAPTERS.get(environment)


def task_progress_capabilities() -> list[dict[str, Any]]:
    """Describe registered adapters without exposing implementation callables."""

    return [
        {
            "name": adapter.name,
            "environment": adapter.environment,
            "signal_names": list(adapter.signal_names),
            "thresholds": dict(adapter.thresholds),
        }
        for adapter in _TASK_PROGRESS_ADAPTERS.values()
    ]


def qualify_task_progress(
    adapter: TaskProgressAdapter,
    episodes: Sequence[tuple[dict, Mapping[str, Any]]],
    *,
    expected_action_steps: int | None,
    require_success: bool,
) -> dict | None:
    """Apply one adapter only to a native successful episode."""

    for record, signals in episodes:
        if record["success"]:
            return adapter.qualifier(
                record, signals, expected_action_steps, require_success
            )
    if require_success:
        raise IsaacArenaError(
            f"{adapter.environment} produced no upstream-defined successful task episode"
        )
    return None
