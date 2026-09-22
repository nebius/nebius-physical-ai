"""Validate current-run Arena metric traces against scored episode outcomes."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .errors import IsaacArenaError
from .hashing import file_sha256
from .task_progress import qualify_task_progress, task_progress_adapter


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


def _episode_record(path: Path, name: str, episode: Any) -> tuple[dict, dict[str, Any]]:
    arrays = _read_signals(episode)
    success_values = arrays.get("success")
    if success_values is None or success_values.size != 1:
        raise IsaacArenaError(
            "simulator ground truth requires one success flag per episode"
        )
    success_value = success_values.reshape(-1)[0]
    if success_value not in (0, 1):
        raise IsaacArenaError(
            "simulator ground-truth success flag must be boolean or 0/1"
        )
    success = bool(success_value)
    if "success" in episode.attrs and bool(episode.attrs["success"]) != success:
        raise IsaacArenaError("simulator ground-truth success metadata disagrees")
    record = {
        "file": path.name,
        "episode": name,
        "success": success,
        "signals": {name: _signal_summary(values) for name, values in arrays.items()},
    }
    record["video_capture"] = _video_capture_steps(arrays)
    return record, arrays


def _video_capture_steps(arrays: dict[str, Any]) -> dict | None:
    import numpy as np

    names = ("initial_action_step", "action_step", "terminal_action_step")
    if not any(f"npa_video/{name}" in arrays for name in names):
        return None
    values = {name: arrays.get(f"npa_video/{name}") for name in names}
    if any(
        item is None or not np.issubdtype(item.dtype, np.integer)
        for item in values.values()
    ):
        raise IsaacArenaError(
            "simulator video requires integer initial/action/terminal step signals"
        )
    initial, steps, terminal = (values[name].reshape(-1) for name in names)
    if initial.size != 1 or int(initial[0]) != 0 or terminal.size != 1:
        raise IsaacArenaError(
            "simulator video requires one initial zero step and one terminal step"
        )
    if not np.array_equal(steps, np.arange(1, int(terminal[0]) + 1)):
        raise IsaacArenaError("simulator video action steps are not contiguous")
    return {
        "initial_action_step": 0,
        "action_steps": steps.tolist(),
        "terminal_action_step": int(terminal[0]),
    }


def _match_episode_results(path: Path, records: list[dict]) -> None:
    rank = re.fullmatch(r"simulator_ground_truth_rank([0-9]+)\.hdf5", path.name)
    if rank is None:
        raise IsaacArenaError("invalid simulator ground-truth rank filename")
    journal = path.with_name(f"episode_results_rank{rank[1]}.jsonl")
    try:
        entries = [
            json.loads(line)
            for line in journal.read_text().splitlines()
            if line.strip()
        ]
    except (OSError, ValueError) as exc:
        raise IsaacArenaError(
            "simulator ground truth has no matching episode JSONL"
        ) from exc
    if len(entries) != len(records):
        raise IsaacArenaError(
            "simulator ground-truth episode count disagrees with upstream episode JSONL"
        )
    for entry, record in zip(entries, records, strict=True):
        if (
            not isinstance(entry, dict)
            or type(entry.get("success")) is not bool
            or entry["success"] != record["success"]
        ):
            raise IsaacArenaError(
                "simulator ground-truth success flags disagree with upstream episode JSONL"
            )
        length = entry.get("episode_length")
        if type(length) is not int or length <= 0:
            raise IsaacArenaError("upstream episode length must be a positive integer")
        record["episode_length"] = length


def _read_file(path: Path) -> list[tuple[dict, dict[str, Any]]]:
    import h5py

    episodes = []
    try:
        with h5py.File(path, "r") as dataset:
            data = dataset.get("data")
            if not isinstance(data, h5py.Group) or len(data) == 0:
                raise IsaacArenaError(
                    "simulator ground truth has no completed episodes"
                )
            if any(re.fullmatch(r"demo_[0-9]+", name) is None for name in data):
                raise IsaacArenaError("invalid simulator ground-truth episode name")
            for name in sorted(
                data, key=lambda value: int(value.removeprefix("demo_"))
            ):
                episodes.append(_episode_record(path, name, data[name]))
    except OSError as exc:
        raise IsaacArenaError(
            f"unreadable simulator ground truth: {path.name}"
        ) from exc
    _match_episode_results(path, [record for record, _signals in episodes])
    return episodes


def _read_evidence(
    run_dir: Path,
) -> tuple[list[dict], list[tuple[dict, dict[str, Any]]]]:
    paths = sorted(run_dir.glob("simulator_ground_truth_rank*.hdf5"))
    if not paths:
        raise IsaacArenaError(
            "upstream evaluation retained no simulator ground-truth HDF5"
        )
    files, episodes = [], []
    for path in paths:
        digest = file_sha256(path)
        files.append(
            {"path": path.name, "bytes": path.stat().st_size, "sha256": digest}
        )
        episodes.extend(_read_file(path))
    return files, episodes


def simulator_ground_truth(
    run_dir: Path,
    *,
    environment: str,
    policy_type: str,
    expected_episodes: int,
    expected_successes: int,
    expected_action_steps: int | None,
    require_task_success: bool,
) -> dict[str, Any]:
    """Bind retained simulator metrics to the current run's episode JSONL.

    Args:
        run_dir: Fresh upstream output directory, never the replay input directory.
        environment: Requested registered Arena environment.
        policy_type: Requested upstream policy adapter.
        expected_episodes: Number of completed JSONL episodes.
        expected_successes: Number of JSONL successes.
        expected_action_steps: Prepared replay length used to bound the observed trace.
        require_task_success: Require a registered, successful native progress proof.

    Returns:
        Hashed trace files, measured states, and task progress interval.

    Raises:
        IsaacArenaError: Missing, malformed, inconsistent, or insufficient evidence.
    """
    files, episodes = _read_evidence(run_dir)
    records = [record for record, _signals in episodes]
    successes = sum(item["success"] for item in records)
    if len(records) != expected_episodes:
        raise IsaacArenaError(
            "simulator ground-truth episode count disagrees with upstream episode JSONL"
        )
    if successes != expected_successes:
        raise IsaacArenaError(
            "simulator ground-truth success flags disagree with upstream episode JSONL"
        )
    adapter = task_progress_adapter(environment)
    if require_task_success and adapter is None:
        raise IsaacArenaError(
            f"visual task qualification has no registered progress adapter for {environment}"
        )
    motion = (
        qualify_task_progress(
            adapter,
            episodes,
            policy_type=policy_type,
            expected_action_steps=expected_action_steps,
            require_success=require_task_success,
        )
        if adapter is not None
        else None
    )
    return {
        "source": "current-run Arena metric-recorder HDF5",
        "files": files,
        "episodes": records,
        "successes": successes,
        "task_progress_adapter": adapter.name if adapter else None,
        "task_motion": motion,
    }
