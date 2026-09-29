"""Build task-isolated LeRobot views for Comet training and holdout scoring.

This module owns only the dataset boundary. The pinned Comet source still owns
its B1K state transform, normalization, model, and training loop.
"""

from __future__ import annotations

import bisect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import multiprocessing
from pathlib import Path
from typing import Any

import numpy as np


DATASET_REPOSITORY = "behavior-1k/2026-challenge-demos"
DATASET_REVISION = "4f50b44796641a4d526a19d9aeadc8aa51e2f2c2"
TASK_ID = 1
TASK_NAME = "picking_up_trash"
TASK_NAMES = {
    0: "turning_on_radio",
    TASK_ID: TASK_NAME,
    22: "putting_shoes_on_rack",
}
FPS = 30
ACTION_HORIZON = 32
ACTION_DIMENSION = 23
STATE_DIMENSION = 61
ALIGNMENT_TOLERANCE_SECONDS = 5e-4
CAMERAS = {
    "head": "zed_link",
    "left_wrist": "left_realsense_link",
    "right_wrist": "right_realsense_link",
}
CAMERA_SIZES = {"head": 720, "left_wrist": 480, "right_wrist": 480}
DATA_RECONSTRUCTION_SCHEMA = "npa.behavior.comet-native-data-reconstruction.v1"


class DeliveredSampleDataset:
    """Attach the source index to each transformed Comet sample.

    This class lives in an importable package because Torch ``spawn`` workers
    must reconstruct the dataset without the dynamically loaded adapter module.

    Args:
        dataset: Map-style dataset whose rows can be copied into dictionaries.

    Returns:
        A map-style dataset that includes ``_npa_sample_index`` in every row.

    Raises:
        TypeError: A source row cannot be converted to a dictionary.
    """

    def __init__(self, dataset: Any) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        """Return the source dataset length.

        Args:
            None.
        Returns:
            Number of source samples.
        Raises:
            TypeError: The source dataset has no length.
        """
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Copy one row and attach its delivered source index.

        Args:
            index: Zero-based source dataset index.
        Returns:
            Copied sample with its source index.
        Raises:
            IndexError: The source dataset rejects ``index``.
            TypeError: The source row cannot be converted to a dictionary.
        """
        value = dict(self.dataset[index])
        value["_npa_sample_index"] = index
        return value


def _read_spawn_sample(dataset: Any, connection: Any) -> None:
    try:
        value = dataset[0]
        connection.send({"status": "sample_read", "type": type(value).__name__})
    except Exception as exc:
        connection.send({"status": "failed", "error": type(exc).__name__})
    finally:
        connection.close()


def _close_spawn_process(process: multiprocessing.Process) -> None:
    if process.pid is None:
        return
    if process.is_alive():
        process.terminate()
        process.join()
    process.close()


def verify_spawn_dataset_sample(dataset: Any) -> dict[str, object]:
    """Verify one real dataset sample in a fresh spawn process.

    Args:
        dataset: Nonempty map-style dataset passed to Torch loader workers.
    Returns:
        Spawn method and successful sample-read status.
    Raises:
        ValueError: The dataset is empty, cannot be reconstructed, or cannot
            produce its first sample in a spawned process.
    """
    if len(dataset) < 1:
        raise ValueError("spawn dataset probe requires a nonempty dataset")
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_read_spawn_sample, args=(dataset, sender))
    try:
        try:
            process.start()
        except Exception as exc:
            raise ValueError("spawn dataset worker could not start") from exc
        sender.close()
        process.join()
        try:
            result = receiver.recv() if receiver.poll() else None
        except (EOFError, OSError) as exc:
            raise ValueError("spawn dataset worker ended without a result") from exc
        if process.exitcode != 0 or not isinstance(result, dict):
            raise ValueError("spawn dataset worker failed before returning a sample")
        if result.get("status") != "sample_read":
            raise ValueError(f"spawn dataset sample failed: {result.get('error')}")
        return {"start_method": "spawn", "status": result["status"], "sample_index": 0}
    finally:
        receiver.close()
        sender.close()
        _close_spawn_process(process)


def validate_data_reconstruction(value: object) -> dict[str, object]:
    """Validate one explicit Comet task data contract.

    Args:
        value: Candidate admission data reconstruction.
    Returns:
        The validated contract.
    Raises:
        ValueError: Fixed data or task identity differs.
    """
    fixed = {
        "schema": DATA_RECONSTRUCTION_SCHEMA,
        "dataset_repository": DATASET_REPOSITORY,
        "dataset_revision": DATASET_REVISION,
        "modalities": ["rgb"],
        "tolerance_s": ALIGNMENT_TOLERANCE_SECONDS,
        "prompt_from_task": True,
        "fine_grained_level": 0,
    }
    if not isinstance(value, dict) or any(
        value.get(key) != expected for key, expected in fixed.items()
    ):
        raise ValueError("Comet data reconstruction contract differs")
    task_id = value.get("task_id")
    if (
        type(task_id) is not int
        or task_id not in TASK_NAMES
        or TASK_NAMES[task_id] != value.get("task_name")
    ):
        raise ValueError("Comet task identity differs")
    return value


def load_task1_episodes(
    path: Path, partition: str, *, expected_split_sha256: str
) -> tuple[int, ...]:
    """Load one immutable task-1 partition from a reviewed episode split.

    Args:
        path: Reviewed JSON split containing task 1.
        partition: Either ``training`` or ``holdout``.
        expected_split_sha256: Approved SHA-256 of the raw split bytes.
    Returns:
        Ordered episode identifiers for the requested partition.
    Raises:
        ValueError: The partition or split contract differs.
    """
    return load_task_episodes(
        path, partition, task_id=TASK_ID, expected_split_sha256=expected_split_sha256
    )


def load_task_episodes(
    path: Path, partition: str, *, task_id: int, expected_split_sha256: str
) -> tuple[int, ...]:
    """Load one immutable task partition from a reviewed split.

    Args:
        path: Reviewed JSON split containing the requested task.
        partition: Either ``training`` or ``holdout``.
        task_id: Supported task 0, 1, or 22.
        expected_split_sha256: Approved SHA-256 of the raw split bytes.
    Returns:
        Ordered episode identifiers for the requested task and partition.
    Raises:
        ValueError: Task, partition, split identity, or membership differs.
    """
    _task_name(task_id)
    split = _load_bound_split(path, expected_split_sha256)
    return _partition_episodes(split, partition, task_id=task_id)


def _task_name(task_id: int) -> str:
    if type(task_id) is not int or task_id not in TASK_NAMES:
        raise ValueError("Comet dataset supports only task IDs 0, 1, and 22")
    return TASK_NAMES[task_id]


def _partition_episodes(
    split: Mapping[str, Any], partition: str, *, task_id: int = TASK_ID
) -> tuple[int, ...]:
    if partition not in {"training", "holdout"}:
        raise ValueError("partition must be training or holdout")
    if (
        split.get("schema") != "npa.behavior.panel-episode-split.v1"
        or split.get("revision") != DATASET_REVISION
    ):
        raise ValueError("Episode split identity differs")
    row = split.get("tasks", {}).get(str(task_id), {})
    training = row.get("training")
    holdout = row.get("holdout")
    if (
        row.get("name") != _task_name(task_id)
        or not isinstance(training, list)
        or not isinstance(holdout, list)
        or len(training) != 180
        or len(holdout) != 20
    ):
        raise ValueError(f"Task-{task_id} episode split differs")
    combined = training + holdout
    if (
        any(type(value) is not int or value < 0 for value in combined)
        or len(set(combined)) != 200
        or set(training) & set(holdout)
    ):
        raise ValueError(
            f"Task-{task_id} episode split overlaps or contains invalid IDs"
        )
    return tuple(training if partition == "training" else holdout)


def _load_bound_split(path: Path, expected_sha256: str) -> dict[str, Any]:
    if len(expected_sha256) != 64 or any(
        c not in "0123456789abcdef" for c in expected_sha256
    ):
        raise ValueError("Expected split SHA-256 is malformed")
    if path.is_symlink() or not path.is_file():
        raise ValueError("Episode split must be a regular file")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Episode split bytes differ")
    split = json.loads(raw)
    if not isinstance(split, dict):
        raise ValueError("Episode split must be a JSON object")
    return split


def action_valid_mask(
    frame_index: int, episode_length: int, horizon: int = ACTION_HORIZON
) -> np.ndarray:
    """Return true for action rows that precede the episode boundary.

    Args:
        frame_index: Zero-based frame within one episode.
        episode_length: Number of frames in that episode.
        horizon: Required Comet action horizon.
    Returns:
        Boolean mask with one entry per action row.
    Raises:
        TypeError: Frame metadata is not integer typed.
        ValueError: Metadata is out of range or the horizon differs.
    """
    if type(frame_index) is not int or type(episode_length) is not int:
        raise TypeError("Frame index and episode length must be integers")
    if horizon != ACTION_HORIZON:
        raise ValueError("Comet requires a 32-step action horizon")
    if episode_length <= 0 or not 0 <= frame_index < episode_length:
        raise ValueError("Frame index lies outside its episode")
    valid = min(horizon, episode_length - frame_index)
    return np.arange(horizon) < valid


def _episode_lengths(metadata: Any, episodes: Sequence[int]) -> dict[int, int]:
    rows = getattr(metadata, "episodes", None)
    if not isinstance(rows, Mapping):
        raise ValueError("LeRobot metadata does not expose episode rows")
    lengths = {}
    for episode in episodes:
        row = rows.get(episode, rows.get(str(episode)))
        if not isinstance(row, Mapping) or type(row.get("length")) is not int:
            raise ValueError("LeRobot episode length metadata differs")
        length = row["length"]
        if length <= 0:
            raise ValueError("LeRobot episode length must be positive")
        lengths[episode] = length
    return lengths


def _default_dataset_factory(*args, **kwargs):
    if kwargs.pop("return_uint8", None) is not True:
        raise ValueError("Comet dataset decoding must request uint8 RGB")
    return _Uint8LeRobotDataset(_PackedV3Dataset(*args, **kwargs))


@dataclass(frozen=True)
class _EpisodeLocation:
    episode: int
    length: int
    dataset_from: int
    data_path: Path
    video_paths: dict[str, Path]
    video_offsets: dict[str, float]


def _regular_json(path: Path) -> Any:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"LeRobot v3 metadata file is absent: {path.name}")
    return json.loads(path.read_text())


def _validate_v3_info(root: Path) -> dict[str, Any]:
    info = _regular_json(root / "meta/info.json")
    expected = {
        "codebase_version": "v3.0",
        "fps": FPS,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": (
            "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
        ),
    }
    if not isinstance(info, dict) or any(
        info.get(key) != value for key, value in expected.items()
    ):
        raise ValueError("LeRobot v3 info contract differs")
    features = info.get("features", {})
    required = {
        "observation.state": ("float32", [STATE_DIMENSION]),
        "action": ("float32", [ACTION_DIMENSION]),
    }
    if any(
        (features.get(key) or {}).get("dtype") != dtype
        or (features.get(key) or {}).get("shape") != shape
        for key, (dtype, shape) in required.items()
    ):
        raise ValueError("LeRobot v3 state/action feature contract differs")
    return info


def _validate_task_metadata(root: Path, task_id: int = TASK_ID) -> None:
    tasks_path = root / "meta/tasks.jsonl"
    if tasks_path.is_symlink() or not tasks_path.is_file():
        raise ValueError("LeRobot v3 task metadata is absent")
    rows = [json.loads(line) for line in tasks_path.read_text().splitlines()]
    matches = [row for row in rows if row.get("task_index") == task_id]
    if (
        len(matches) != 1
        or type(matches[0].get("task_index")) is not int
        or matches[0].get("task_name") != _task_name(task_id)
    ):
        raise ValueError(f"LeRobot v3 task-{task_id} identity differs")


def _episode_metadata_rows(root: Path, task_id: int = TASK_ID) -> list[dict[str, Any]]:
    import pyarrow.parquet as parquet

    path = root / f"meta/episodes/chunk-{task_id:03d}/file-000.parquet"
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"LeRobot v3 task-{task_id} episode metadata is absent")
    rows = parquet.read_table(path).to_pylist()
    if len(rows) != 200 or len({row.get("episode_index") for row in rows}) != 200:
        raise ValueError(f"LeRobot v3 task-{task_id} episode inventory differs")
    return rows


def _packed_path(
    root: Path, template: str, row: Mapping[str, Any], prefix: str
) -> Path:
    chunk = row.get(f"{prefix}/chunk_index")
    file = row.get(f"{prefix}/file_index")
    if type(chunk) is not int or chunk < 0 or type(file) is not int or file < 0:
        raise ValueError("LeRobot v3 packed file index differs")
    relative = template.format(
        video_key=prefix.removeprefix("videos/"),
        chunk_index=chunk,
        file_index=file,
    )
    path = root / relative
    if not path.is_file():
        raise ValueError(f"LeRobot v3 packed file is absent: {relative}")
    return path


def _episode_location(
    root: Path, info: Mapping[str, Any], row: Mapping[str, Any]
) -> _EpisodeLocation:
    length = row.get("length")
    start, stop = row.get("dataset_from_index"), row.get("dataset_to_index")
    episode = row.get("episode_index")
    if (
        type(episode) is not int
        or type(length) is not int
        or length <= 0
        or type(start) is not int
        or start < 0
        or type(stop) is not int
        or stop - start != length
    ):
        raise ValueError("LeRobot v3 episode frame interval differs")
    data = _packed_path(root, info["data_path"], row, "data")
    videos, offsets = {}, {}
    for camera in CAMERAS.values():
        key = f"observation.rgb.{camera}_camera_0"
        prefix = f"videos/{key}"
        videos[key] = _packed_path(root, info["video_path"], row, prefix)
        offsets[key] = _video_interval_start(row, prefix, length)
    return _EpisodeLocation(episode, length, start, data, videos, offsets)


def _video_interval_start(row: Mapping[str, Any], prefix: str, length: int) -> float:
    begin = row.get(f"{prefix}/from_timestamp")
    end = row.get(f"{prefix}/to_timestamp")
    if (
        not _is_numeric_scalar(begin)
        or not _is_numeric_scalar(end)
        or not np.isfinite([begin, end]).all()
        or begin < 0
        or not np.isclose(
            end - begin,
            length / FPS,
            atol=ALIGNMENT_TOLERANCE_SECONDS,
            rtol=0.0,
        )
    ):
        raise ValueError("LeRobot v3 video interval differs from episode length")
    return float(begin)


def _is_numeric_scalar(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _load_locations(
    root: Path, episodes: Sequence[int], *, task_id: int = TASK_ID
) -> dict[int, _EpisodeLocation]:
    task_name = _task_name(task_id)
    info = _validate_v3_info(root)
    _validate_task_metadata(root, task_id)
    _regular_json(root / "meta/stats.json")
    requested = set(episodes)
    rows = [
        row
        for row in _episode_metadata_rows(root, task_id)
        if row.get("episode_index") in requested
    ]
    if {row.get("episode_index") for row in rows} != requested:
        raise ValueError(
            f"Requested task-{task_id} episodes are absent from LeRobot v3 metadata"
        )
    if any(
        type(row.get("task_index")) is not int
        or row.get("task_index") != task_id
        or row.get("tasks") != [task_name]
        for row in rows
    ):
        raise ValueError("LeRobot v3 episode task identity differs")
    return {row["episode_index"]: _episode_location(root, info, row) for row in rows}


class _PackedV3Metadata:
    def __init__(self, locations: Mapping[int, _EpisodeLocation]) -> None:
        self.episodes = {
            episode: {"length": row.length} for episode, row in locations.items()
        }


class _PackedV3Dataset:
    """Read immutable packed LeRobot v3 bytes through the pinned runtime APIs."""

    def __init__(self, repo_id: str, **kwargs: Any) -> None:
        if repo_id != DATASET_REPOSITORY:
            raise ValueError("LeRobot repository identity differs")
        root, episodes = Path(kwargs.pop("root")), tuple(kwargs.pop("episodes"))
        task_id = kwargs.pop("task_id", TASK_ID)
        self._validate_options(kwargs)
        self.locations = _load_locations(root, episodes, task_id=task_id)
        self.episodes = episodes
        self.meta = _PackedV3Metadata(self.locations)
        self._starts = self._episode_starts()
        self._tables = self._load_tables()

    @staticmethod
    def _validate_options(options: Mapping[str, Any]) -> None:
        expected = {
            "revision": DATASET_REVISION,
            "video_backend": "pyav",
            "tolerance_s": ALIGNMENT_TOLERANCE_SECONDS,
            "delta_timestamps": {
                "action": [step / FPS for step in range(ACTION_HORIZON)]
            },
        }
        if dict(options) != expected:
            raise ValueError("LeRobot v3 compatibility options differ")

    def _episode_starts(self) -> list[int]:
        starts, total = [], 0
        for episode in self.episodes:
            starts.append(total)
            total += self.locations[episode].length
        starts.append(total)
        return starts

    def _load_tables(self) -> dict[Path, Any]:
        from datasets import Dataset

        result = {}
        for path in sorted({row.data_path for row in self.locations.values()}):
            table = Dataset.from_parquet(str(path))
            _validate_packed_features(table.features)
            table.set_format("numpy")
            result[path] = table
        return result

    def __len__(self) -> int:
        return self._starts[-1]

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode_position = bisect.bisect_right(self._starts, index) - 1
        if episode_position < 0 or episode_position >= len(self.episodes):
            raise IndexError(index)
        episode = self.episodes[episode_position]
        frame = index - self._starts[episode_position]
        location = self.locations[episode]
        sample = self._low_dimensional_sample(location, frame)
        sample.update(self._rgb_sample(location, frame, float(sample["timestamp"])))
        return sample

    def _low_dimensional_sample(
        self, location: _EpisodeLocation, frame: int
    ) -> dict[str, Any]:
        table = self._tables[location.data_path]
        first_index = int(table[0]["index"])
        local = location.dataset_from - first_index + frame
        current = dict(table[local])
        indices = [
            local + min(frame + step, location.length - 1) - frame
            for step in range(ACTION_HORIZON)
        ]
        action_rows = table[indices]
        actions = np.asarray(action_rows["action"], dtype=np.float32)
        if (
            int(current["episode_index"]) != location.episode
            or int(current["frame_index"]) != frame
            or int(current["index"]) != location.dataset_from + frame
        ):
            raise ValueError("LeRobot v3 packed data interval differs")
        expected_frames = [
            min(frame + step, location.length - 1) for step in range(ACTION_HORIZON)
        ]
        if not np.array_equal(
            action_rows["episode_index"],
            np.full(ACTION_HORIZON, location.episode),
        ) or not np.array_equal(action_rows["frame_index"], expected_frames):
            raise ValueError("LeRobot v3 action horizon crosses an episode boundary")
        if not np.isclose(
            float(current["timestamp"]),
            frame / FPS,
            atol=ALIGNMENT_TOLERANCE_SECONDS,
            rtol=0.0,
        ):
            raise ValueError("LeRobot v3 frame timestamp differs")
        current["action"] = actions
        return current

    def _rgb_sample(
        self, location: _EpisodeLocation, frame: int, timestamp: float
    ) -> dict[str, Any]:
        result = {}
        for key, path in location.video_paths.items():
            query = [location.video_offsets[key] + timestamp]
            result[key] = _decode_rgb(path, query)
        return result


def _validate_packed_features(features: Mapping[str, Any]) -> None:
    expected = {
        "index": ("int64", None),
        "episode_index": ("int64", None),
        "frame_index": ("int64", None),
        "timestamp": ("float32", None),
        "observation.state": ("float32", STATE_DIMENSION),
        "action": ("float32", ACTION_DIMENSION),
    }
    for name, (dtype, length) in expected.items():
        feature = features.get(name)
        leaf = getattr(feature, "feature", feature)
        if getattr(leaf, "dtype", None) != dtype:
            raise ValueError(f"LeRobot v3 packed feature differs: {name}")
        declared_length = getattr(feature, "length", None)
        if length is not None and declared_length not in {None, -1, length}:
            raise ValueError(f"LeRobot v3 packed feature length differs: {name}")


def _decode_rgb(path: Path, query: list[float]) -> Any:
    from lerobot.datasets.video_utils import decode_video_frames

    frames = decode_video_frames(path, query, ALIGNMENT_TOLERANCE_SECONDS, "pyav")
    return frames.squeeze(0)


class _Uint8LeRobotDataset:
    """Apply the pinned Comet float-to-uint8 rule to LeRobot RGB."""

    def __init__(self, dataset: Any) -> None:
        self.dataset = dataset
        self.meta = dataset.meta

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = dict(self.dataset[index])
        for camera in CAMERAS.values():
            key = f"observation.rgb.{camera}_camera_0"
            pixels = np.asarray(sample[key])
            if pixels.dtype == np.uint8:
                continue
            if not np.issubdtype(pixels.dtype, np.floating):
                raise ValueError("Released Comet RGB dtype differs")
            if not np.isfinite(pixels).all() or pixels.min() < 0 or pixels.max() > 1:
                raise ValueError("Released Comet RGB floating decode differs")
            sample[key] = (255 * pixels).astype(np.uint8)
        return sample


class CometTaskDataset:
    """Expose one isolated task partition with explicit terminal actions.

    Args:
        root: Extracted LeRobot dataset root.
        split_path: Reviewed 180/20 episode split.
        task_id: Supported task 0, 1, or 22.
        expected_split_sha256: Approved SHA-256 of the raw split bytes.
        partition: Isolated partition to expose.
        dataset_factory: Optional LeRobot-compatible factory for testing or the
            pinned runtime, including a task_id keyword for anchor tasks.
    Returns:
        None.
    Raises:
        ValueError: Split or episode metadata violates the contract.
    """

    def __init__(
        self,
        root: str | Path,
        split_path: str | Path,
        *,
        task_id: int,
        expected_split_sha256: str,
        partition: str = "training",
        dataset_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.task_name = _task_name(task_id)
        self.episodes = load_task_episodes(
            Path(split_path),
            partition,
            task_id=task_id,
            expected_split_sha256=expected_split_sha256,
        )
        factory = dataset_factory or _default_dataset_factory
        task_options = {} if task_id == TASK_ID else {"task_id": task_id}
        self.dataset = factory(
            DATASET_REPOSITORY,
            root=Path(root),
            episodes=list(self.episodes),
            revision=DATASET_REVISION,
            delta_timestamps={"action": [step / FPS for step in range(ACTION_HORIZON)]},
            video_backend="pyav",
            return_uint8=True,
            tolerance_s=ALIGNMENT_TOLERANCE_SECONDS,
            **task_options,
        )
        self.episode_lengths = _episode_lengths(self.dataset.meta, self.episodes)

    def __len__(self) -> int:
        """Return the number of frames in this partition."""
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one mapped Comet sample and its diagnostic terminal mask.

        Args:
            index: Dataset frame index.
        Returns:
            State, action chunk, prompt, RGB, identity, and padding arrays.
        Raises:
            ValueError: The sample escapes the partition or differs in shape,
                type, finiteness, or camera layout.
        """
        sample = self.dataset[index]
        episode = _scalar_int(sample, "episode_index")
        frame = _scalar_int(sample, "frame_index")
        if episode not in self.episode_lengths:
            raise ValueError("LeRobot sample escaped the requested episode partition")

        state, actions = _validated_state_actions(sample)
        valid = action_valid_mask(frame, self.episode_lengths[episode])
        actions = _repeat_terminal_action(actions, valid)

        result: dict[str, Any] = {
            "observation.state": state,
            "action": actions,
            "action_valid_mask": valid,
            "action_is_pad": ~valid,
            "prompt": self.task_name,
            "episode_index": np.asarray(episode),
            "frame_index": np.asarray(frame),
        }
        for role, camera in CAMERAS.items():
            result[f"observation.images.rgb.{role}"] = _validated_rgb(
                sample, role, camera
            )
        return result


class CometTask1Dataset(CometTaskDataset):
    """Retain the task-1 reader interface for existing training workflows.

    Args:
        root: Extracted LeRobot dataset root.
        split_path: Reviewed 180/20 episode split.
        expected_split_sha256: Approved SHA-256 of the raw split bytes.
        partition: Either ``training`` or ``holdout``.
        dataset_factory: Optional LeRobot-compatible dataset factory.
    Returns:
        None.
    Raises:
        ValueError: Split or episode metadata violates the task-1 contract.
    """

    def __init__(
        self,
        root: str | Path,
        split_path: str | Path,
        *,
        expected_split_sha256: str,
        partition: str = "training",
        dataset_factory: Callable[..., Any] | None = None,
    ) -> None:
        super().__init__(
            root,
            split_path,
            task_id=TASK_ID,
            expected_split_sha256=expected_split_sha256,
            partition=partition,
            dataset_factory=dataset_factory,
        )


def _validated_state_actions(
    sample: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    state = np.asarray(sample["observation.state"])
    actions = np.asarray(sample["action"])
    if state.shape != (STATE_DIMENSION,) or actions.shape != (
        ACTION_HORIZON,
        ACTION_DIMENSION,
    ):
        raise ValueError("Released Comet state/action shape differs")
    if state.dtype != np.float32 or actions.dtype != np.float32:
        raise ValueError("Released Comet state/action dtype differs")
    if not np.isfinite(state).all() or not np.isfinite(actions).all():
        raise ValueError("Released Comet state/action contains non-finite values")
    return state, actions


def _validated_rgb(sample: Mapping[str, Any], role: str, camera: str) -> np.ndarray:
    pixels = np.asarray(sample[f"observation.rgb.{camera}_camera_0"])
    size = CAMERA_SIZES[role]
    if pixels.dtype != np.uint8:
        raise ValueError("Released Comet RGB dtype differs")
    if pixels.shape not in {(size, size, 3), (3, size, size)}:
        raise ValueError("Released Comet RGB shape differs")
    return pixels


def _repeat_terminal_action(actions: np.ndarray, valid: np.ndarray) -> np.ndarray:
    result = actions.copy()
    # The pinned Behavior loader clamps delta queries at the episode boundary.
    # Apply that rule explicitly so another backend cannot cross an episode.
    if not valid.all():
        result[~valid] = result[np.flatnonzero(valid)[-1]]
    return result


def _scalar_int(sample: Mapping[str, Any], name: str) -> int:
    value = np.asarray(sample[name])
    if value.shape != () or not np.issubdtype(value.dtype, np.integer):
        raise ValueError(f"LeRobot {name} must be an integer scalar")
    return int(value)
