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
    supported_policy_types: tuple[str, ...]
    maximum_trailing_held_action_fraction: float
    visual_interval_strategy: str
    visual_context_steps: int
    visual_progress_signal: str
    visual_progress_region: Mapping[str, Any]
    visual_association_radius_fraction: float
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


def task_visual_interval(
    adapter: TaskProgressAdapter,
    progress_interval: Mapping[str, Any],
    total_steps: int,
) -> dict[str, int]:
    """Resolve the adapter-declared visual horizon for one scored episode."""

    keys = ("start_action_step", "end_action_step", "total_action_steps")
    if (
        type(total_steps) is not int
        or total_steps <= 0
        or any(type(progress_interval.get(key)) is not int for key in keys)
        or progress_interval["total_action_steps"] != total_steps
        or not 0
        <= progress_interval["start_action_step"]
        < progress_interval["end_action_step"]
        <= total_steps
        or type(adapter.visual_context_steps) is not int
        or adapter.visual_context_steps < 0
    ):
        raise IsaacArenaError("task progress adapter has an invalid visual interval")
    if adapter.visual_interval_strategy == "native_scored_episode":
        return {
            "start_action_step": 0,
            "end_action_step": total_steps,
            "total_action_steps": total_steps,
        }
    if adapter.visual_interval_strategy == "task_progress":
        return dict(progress_interval)
    if adapter.visual_interval_strategy == "task_progress_with_leading_context":
        return {
            "start_action_step": max(
                0,
                progress_interval["start_action_step"] - adapter.visual_context_steps,
            ),
            "end_action_step": progress_interval["end_action_step"],
            "total_action_steps": total_steps,
        }
    raise IsaacArenaError(
        "task progress adapter has an invalid visual interval strategy"
    )


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
    # The native terminal is authoritative for the scored/captured episode.
    # A replay may contain an unused suffix, but that suffix is outside this
    # episode and cannot lengthen its progress or visual evidence interval.
    total_steps = int(values.size - 1)
    return _microwave_motion_record(record, values, statistics, total_steps)


_TASK_PROGRESS_ADAPTERS: Mapping[str, TaskProgressAdapter] = MappingProxyType(
    {
        "gr1_open_microwave": TaskProgressAdapter(
            name="arena.open-door.revolute-joint.v1",
            environment="gr1_open_microwave",
            supported_policy_types=("replay", "rsl_rl"),
            maximum_trailing_held_action_fraction=0.25,
            visual_interval_strategy="task_progress_with_leading_context",
            visual_context_steps=30,
            visual_progress_signal="monotonic_structural_change",
            visual_progress_region=MappingProxyType(
                {
                    "name": "microwave_door_workspace",
                    "left": 0.25,
                    "top": 0.3,
                    "right": 0.7,
                    "bottom": 0.95,
                }
            ),
            visual_association_radius_fraction=0.03,
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
            "supported_policy_types": list(adapter.supported_policy_types),
            "maximum_trailing_held_action_fraction": (
                adapter.maximum_trailing_held_action_fraction
            ),
            "visual_interval_strategy": adapter.visual_interval_strategy,
            "visual_context_steps": adapter.visual_context_steps,
            "visual_progress_signal": adapter.visual_progress_signal,
            "visual_progress_region": dict(adapter.visual_progress_region),
            "visual_association_radius_fraction": (
                adapter.visual_association_radius_fraction
            ),
            "signal_names": list(adapter.signal_names),
            "thresholds": dict(adapter.thresholds),
        }
        for adapter in _TASK_PROGRESS_ADAPTERS.values()
    ]


def qualify_task_progress(
    adapter: TaskProgressAdapter,
    episodes: Sequence[tuple[dict, Mapping[str, Any]]],
    *,
    policy_type: str,
    expected_action_steps: int | None,
    require_success: bool,
) -> dict | None:
    """Apply one adapter only to a native successful episode."""

    if require_success and policy_type not in adapter.supported_policy_types:
        raise IsaacArenaError(
            f"{adapter.environment} does not register {policy_type} for visual qualification"
        )
    for record, signals in episodes:
        if record["success"]:
            motion = adapter.qualifier(
                record, signals, expected_action_steps, require_success
            )
            interval = motion.get("progress_interval")
            if motion.get("visual_progress_qualified"):
                if not isinstance(interval, Mapping):
                    raise IsaacArenaError(
                        "task progress adapter has no qualified progress interval"
                    )
                total_steps = motion.get("episode_length")
                if type(total_steps) is not int or total_steps <= 0:
                    raise IsaacArenaError(
                        "task progress adapter has an invalid scored horizon"
                    )
                motion["visual_interval_strategy"] = adapter.visual_interval_strategy
                motion["visual_context_steps"] = adapter.visual_context_steps
                motion["visual_progress_signal"] = adapter.visual_progress_signal
                motion["visual_progress_region"] = dict(adapter.visual_progress_region)
                motion["visual_association_radius_fraction"] = (
                    adapter.visual_association_radius_fraction
                )
                motion["visual_interval"] = task_visual_interval(
                    adapter, interval, total_steps
                )
            return motion
    if require_success:
        raise IsaacArenaError(
            f"{adapter.environment} produced no upstream-defined successful task episode"
        )
    return None
