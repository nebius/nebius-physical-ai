"""Validate current-run Arena metric traces against scored episode outcomes."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .errors import IsaacArenaError
from .hashing import file_sha256

MICROWAVE_SUCCESS_THRESHOLD = 0.8
MICROWAVE_MINIMUM_OPENNESS_DELTA = 0.5


def _read_signals(episode: Any) -> dict[str, Any]:
    import h5py
    import numpy as np

    arrays: dict[str, Any] = {}

    def inspect(name: str, item: Any) -> None:
        if not isinstance(item, h5py.Dataset):
            return
        values = np.asarray(item)
        numeric = np.issubdtype(values.dtype, np.number)
        boolean = np.issubdtype(values.dtype, np.bool_)
        if not (numeric or boolean) or not np.isfinite(values).all():
            raise IsaacArenaError(f"invalid simulator ground-truth signal: {name}")
        arrays[name] = values

    episode.visititems(inspect)
    return arrays


def _signal_summary(values: Any) -> dict[str, Any]:
    result = {
        "shape": list(values.shape),
        "samples": int(values.shape[0]) if values.ndim else 1,
    }
    if values.size:
        result.update(minimum=float(values.min()), maximum=float(values.max()))
    return result


def _episode_record(path: Path, name: str, episode: Any) -> tuple[dict, Any]:
    arrays = _read_signals(episode)
    success_values = arrays.get("success")
    if success_values is None or success_values.size != 1:
        raise IsaacArenaError("simulator ground truth requires one success flag per episode")
    success_value = success_values.reshape(-1)[0]
    if success_value not in (0, 1):
        raise IsaacArenaError("simulator ground-truth success flag must be boolean or 0/1")
    success = bool(success_value)
    if "success" in episode.attrs and bool(episode.attrs["success"]) != success:
        raise IsaacArenaError("simulator ground-truth success metadata disagrees")
    record = {
        "file": path.name, "episode": name, "success": success,
        "signals": {name: _signal_summary(values) for name, values in arrays.items()},
    }
    record["video_capture"] = _video_capture_steps(arrays)
    return record, arrays.get("revolute_joint_state")


def _video_capture_steps(arrays: dict[str, Any]) -> dict | None:
    import numpy as np

    names = ("initial_action_step", "action_step", "terminal_action_step")
    if not any(f"npa_video/{name}" in arrays for name in names):
        return None
    values = {name: arrays.get(f"npa_video/{name}") for name in names}
    if any(item is None or not np.issubdtype(item.dtype, np.integer) for item in values.values()):
        raise IsaacArenaError("simulator video requires integer initial/action/terminal step signals")
    initial, steps, terminal = (values[name].reshape(-1) for name in names)
    if initial.size != 1 or int(initial[0]) != 0 or terminal.size != 1:
        raise IsaacArenaError("simulator video requires one initial zero step and one terminal step")
    if not np.array_equal(steps, np.arange(1, int(terminal[0]) + 1)):
        raise IsaacArenaError("simulator video action steps are not contiguous")
    return {"initial_action_step": 0, "action_steps": steps.tolist(),
            "terminal_action_step": int(terminal[0])}


def _match_episode_results(path: Path, records: list[dict], traces: list[Any]) -> None:
    rank = re.fullmatch(r"simulator_ground_truth_rank([0-9]+)\.hdf5", path.name)
    if rank is None:
        raise IsaacArenaError("invalid simulator ground-truth rank filename")
    journal = path.with_name(f"episode_results_rank{rank[1]}.jsonl")
    try:
        entries = [json.loads(line) for line in journal.read_text().splitlines() if line.strip()]
    except (OSError, ValueError) as exc:
        raise IsaacArenaError("simulator ground truth has no matching episode JSONL") from exc
    if len(entries) != len(records):
        raise IsaacArenaError("simulator ground-truth episode count disagrees with upstream episode JSONL")
    for entry, record, trace in zip(entries, records, traces, strict=True):
        if not isinstance(entry, dict) or type(entry.get("success")) is not bool or entry["success"] != record["success"]:
            raise IsaacArenaError("simulator ground-truth success flags disagree with upstream episode JSONL")
        length = entry.get("episode_length")
        if type(length) is not int or length <= 0:
            raise IsaacArenaError("upstream episode length must be a positive integer")
        if trace is not None and length != len(trace) - 1:
            raise IsaacArenaError("simulator trace length disagrees with upstream episode length")


def _read_file(path: Path) -> tuple[list[dict], list[Any]]:
    import h5py

    records, traces = [], []
    try:
        with h5py.File(path, "r") as dataset:
            data = dataset.get("data")
            if not isinstance(data, h5py.Group) or len(data) == 0:
                raise IsaacArenaError("simulator ground truth has no completed episodes")
            if any(re.fullmatch(r"demo_[0-9]+", name) is None for name in data):
                raise IsaacArenaError("invalid simulator ground-truth episode name")
            for name in sorted(data, key=lambda value: int(value.removeprefix("demo_"))):
                record, trace = _episode_record(path, name, data[name])
                records.append(record)
                traces.append(trace)
    except OSError as exc:
        raise IsaacArenaError(f"unreadable simulator ground truth: {path.name}") from exc
    _match_episode_results(path, records, traces)
    return records, traces


def _read_evidence(run_dir: Path) -> tuple[list[dict], list[dict], list[Any]]:
    paths = sorted(run_dir.glob("simulator_ground_truth_rank*.hdf5"))
    if not paths:
        raise IsaacArenaError("upstream evaluation retained no simulator ground-truth HDF5")
    files, records, traces = [], [], []
    for path in paths:
        digest = file_sha256(path)
        files.append({"path": path.name, "bytes": path.stat().st_size, "sha256": digest})
        file_records, file_traces = _read_file(path)
        records.extend(file_records)
        traces.extend(file_traces)
    return files, records, traces


def _microwave_trace(trace: Any, *, executed_steps: int | None) -> Any:
    import numpy as np

    if trace is None:
        raise IsaacArenaError("microwave simulator ground truth has no revolute-joint trace")
    values = np.asarray(trace, dtype=float)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 1 or values.size < 2:
        raise IsaacArenaError("microwave simulator ground truth has no usable revolute-joint trace")
    if executed_steps is not None and values.size - 1 > executed_steps:
        raise IsaacArenaError("microwave simulator trace exceeds the exact replay action horizon")
    return values


def _progress_interval(trace: Any, total_steps: int) -> dict[str, int]:
    import numpy as np

    changed = np.flatnonzero(trace > float(trace[0]) + 0.02)
    opened = np.flatnonzero(trace > MICROWAVE_SUCCESS_THRESHOLD)
    if not changed.size or not opened.size:
        raise IsaacArenaError("microwave simulator trace has no open-door progress interval")
    return {
        "start_action_step": max(0, int(changed[0]) - 1),
        "end_action_step": int(opened[0]),
        "total_action_steps": total_steps,
    }


def _door_motion(record: dict, trace: Any, executed_steps: int | None, require_progress: bool) -> dict:
    values = _microwave_trace(trace, executed_steps=executed_steps)
    initial, final, maximum = float(values[0]), float(values[-1]), float(values.max())
    if final <= MICROWAVE_SUCCESS_THRESHOLD:
        raise IsaacArenaError("microwave success is not supported by final simulator door openness")
    delta = maximum - initial
    if require_progress and delta < MICROWAVE_MINIMUM_OPENNESS_DELTA:
        raise IsaacArenaError("microwave simulator door progress is below the acceptance delta")
    return {
        "kind": "revolute_joint_openness", "source": "Arena simulator metric recorder",
        "episode": record["episode"], "initial_openness": initial, "final_openness": final,
        "minimum_openness": float(values.min()), "maximum_openness": maximum,
        "openness_delta": delta, "success_threshold": MICROWAVE_SUCCESS_THRESHOLD,
        "minimum_required_delta": MICROWAVE_MINIMUM_OPENNESS_DELTA,
        "samples": int(values.size), "task_success": True,
        "progress_interval": (_progress_interval(values, executed_steps or int(values.size - 1))
                              if delta >= MICROWAVE_MINIMUM_OPENNESS_DELTA else None),
        "visual_progress_qualified": delta >= MICROWAVE_MINIMUM_OPENNESS_DELTA,
        "video_capture": record.get("video_capture"),
    }


def _task_motion(records: list[dict], traces: list[Any], executed_steps: int | None,
                 require_success: bool) -> dict | None:
    for record, trace in zip(records, traces, strict=True):
        if record["success"]:
            return _door_motion(record, trace, executed_steps, require_success)
    if require_success:
        raise IsaacArenaError("gr1_open_microwave produced no upstream-defined successful task episode")
    return None


def simulator_ground_truth(run_dir: Path, *, environment: str, expected_episodes: int,
                           expected_successes: int, executed_steps: int | None,
                           require_task_success: bool) -> dict[str, Any]:
    """Bind retained simulator metrics to the current run's episode JSONL.

    Args:
        run_dir: Fresh upstream output directory, never the replay input directory.
        environment: Requested registered Arena environment.
        expected_episodes: Number of completed JSONL episodes.
        expected_successes: Number of JSONL successes.
        executed_steps: Exact replay action count, if applicable.
        require_task_success: Require demonstrable microwave opening for visual proof.

    Returns:
        Hashed trace files, measured states, and task progress interval.

    Raises:
        IsaacArenaError: Missing, malformed, inconsistent, or insufficient evidence.
    """
    files, records, traces = _read_evidence(run_dir)
    successes = sum(item["success"] for item in records)
    if len(records) != expected_episodes:
        raise IsaacArenaError("simulator ground-truth episode count disagrees with upstream episode JSONL")
    if successes != expected_successes:
        raise IsaacArenaError("simulator ground-truth success flags disagree with upstream episode JSONL")
    motion = None
    if environment == "gr1_open_microwave":
        motion = _task_motion(records, traces, executed_steps, require_task_success)
    return {
        "source": "current-run Arena metric-recorder HDF5", "files": files,
        "episodes": records, "successes": successes, "task_motion": motion,
    }
