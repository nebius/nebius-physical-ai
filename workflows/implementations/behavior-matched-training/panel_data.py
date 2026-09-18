"""LeRobot adapter for matched teacher-bin and parent-replay training arms."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
from stage_conditioning import ApplyStageCondition, FrameKey, ReplayRow

DATA_REVISION = "4f50b44796641a4d526a19d9aeadc8aa51e2f2c2"
TASKS = {
    0: "turning_on_radio",
    1: "picking_up_trash",
    22: "putting_shoes_on_rack",
}
CAMERAS = {
    "head": "zed_link",
    "left_wrist": "left_realsense_link",
    "right_wrist": "right_realsense_link",
}
_STAGE_REPLAY_TRACE: dict[FrameKey, ReplayRow] | None = None
_STAGE_REPLAY_SPLIT = "training"


def digest(path: Path) -> str:
    """Return the SHA-256 digest of a file.

    Args:
        path: File to hash.

    Returns:
        Lowercase hexadecimal SHA-256 digest.

    Raises:
        OSError: The file cannot be opened or read.
    """

    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_split(path: Path) -> dict:
    """Load and validate the frozen 180/20 episode split.

    Args:
        path: JSON split manifest.

    Returns:
        Validated split payload.

    Raises:
        OSError: The manifest cannot be read.
        ValueError: Its schema, task membership, or counts differ.
    """

    split = json.loads(path.read_text())
    if (
        split.get("schema") != "npa.behavior.panel-episode-split.v1"
        or split.get("revision") != DATA_REVISION
    ):
        raise ValueError("Panel split identity differs")
    if set(split.get("tasks", {})) != {str(task_id) for task_id in TASKS}:
        raise ValueError("Panel split task set differs")
    seen = set()
    for task_id, name in TASKS.items():
        row = split["tasks"][str(task_id)]
        training, holdout = row.get("training", []), row.get("holdout", [])
        if row.get("name") != name or len(training) != 180 or len(holdout) != 20:
            raise ValueError(f"Task {task_id} split must remain 180/20")
        if set(training) & set(holdout) or len(set(training + holdout)) != 200:
            raise ValueError(f"Task {task_id} split overlaps or contains duplicates")
        if seen & set(training + holdout):
            raise ValueError("Episode IDs overlap across tasks")
        seen.update(training + holdout)
    if split.get("training_episodes") != 540 or split.get("holdout_episodes") != 60:
        raise ValueError("Panel split totals differ")
    return split


class PanelDataset:
    """Expose three released task datasets through one stable global index."""

    def __init__(
        self,
        root: str | Path,
        split_path: str | Path,
        action_horizon: int = 30,
        split_name: str = "training",
    ):
        if action_horizon != 30:
            raise ValueError("Published checkpoint requires a 30-step action horizon")
        if split_name not in {"training", "holdout"}:
            raise ValueError("dataset split must be training or holdout")
        self.root = Path(root)
        self.split = load_split(Path(split_path))
        self.datasets = []
        self.task_ids = tuple(TASKS)
        metadata = {}
        self.offsets = []
        total = 0
        for task_id in self.task_ids:
            episodes = self.split["tasks"][str(task_id)][split_name]
            dataset = _task_dataset(self.root, episodes, action_horizon)
            self.datasets.append(dataset)
            self.offsets.append(total)
            total += len(dataset)
            rows = pq.read_table(
                self.root / f"meta/episodes/chunk-{task_id:03d}/file-000.parquet"
            ).to_pylist()
            metadata.update({row["episode_index"]: row for row in rows})
        self.task_lengths = tuple(len(dataset) for dataset in self.datasets)
        if not all(self.task_lengths):
            raise ValueError("Every panel task must contain training frames")
        self.meta = SimpleNamespace(episodes=metadata)
        self._length = total

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> dict:
        index = int(index)
        if not 0 <= index < self._length:
            raise IndexError(index)
        slot = max(
            position for position, offset in enumerate(self.offsets) if offset <= index
        )
        sample = self.datasets[slot][index - self.offsets[slot]]
        task_id = self.task_ids[slot]
        result = {
            key: np.asarray(sample[key])
            for key in (
                "observation.state",
                "action",
                "task_index",
                "timestamp",
                "episode_index",
                "index",
            )
        }
        if result["observation.state"].shape != (61,) or result["action"].shape != (
            30,
            23,
        ):
            raise ValueError("Released panel state/action shape differs")
        if (
            not np.isfinite(result["observation.state"]).all()
            or not np.isfinite(result["action"]).all()
        ):
            raise ValueError("Released panel state/action contains non-finite values")
        if int(result["task_index"]) != task_id:
            raise ValueError("Released panel sample task identity differs")
        for role, camera in CAMERAS.items():
            pixels = np.asarray(sample[f"observation.rgb.{camera}_camera_0"])
            if pixels.ndim != 3 or 3 not in (pixels.shape[0], pixels.shape[-1]):
                raise ValueError("Released panel RGB shape differs")
            result[f"observation.images.rgb.{role}"] = pixels
        return result

    def key_for_index(self, index: int, *, sample: dict | None = None) -> FrameKey:
        """Map one global panel index to its episode-relative frame key."""
        sample = self[index] if sample is None else sample
        task_id = int(sample["task_index"])
        episode = int(sample["episode_index"])
        slot = self.task_ids.index(task_id)
        dataset = self.datasets[slot]
        episode_slot = list(dataset.episodes).index(episode)
        local_index = int(index) - self.offsets[slot]
        start = int(dataset.episode_data_index["from"][episode_slot])
        end = int(dataset.episode_data_index["to"][episode_slot])
        if not start <= local_index < end:
            raise IndexError(index)
        return FrameKey(task_id, episode, local_index - start)


def _task_dataset(root: Path, episodes: list[int], action_horizon: int):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset(
        "behavior-1k/2026-challenge-demos",
        root=root,
        episodes=episodes,
        revision=DATA_REVISION,
        delta_timestamps={"action": [step / 30.0 for step in range(action_horizon)]},
        video_backend="pyav",
        return_uint8=True,
        tolerance_s=5e-4,
    )


class ReplanDataset:
    """Restrict a panel split to traced replan keys and attach stage conditions."""

    def __init__(self, panel: PanelDataset, trace: dict[FrameKey, ReplayRow]):
        self.panel = panel
        self.trace = trace
        self.offsets = panel.offsets
        self.task_ids = panel.task_ids
        self.episode_slots = [
            {int(episode): slot for slot, episode in enumerate(dataset.episodes)}
            for dataset in panel.datasets
        ]
        self.indices_by_task: list[list[int]] = []
        used: set[FrameKey] = set()
        for slot, task_id in enumerate(panel.task_ids):
            dataset = panel.datasets[slot]
            indices = []
            starts = np.asarray(dataset.episode_data_index["from"], dtype=np.int64)
            ends = np.asarray(dataset.episode_data_index["to"], dtype=np.int64)
            if len(starts) != len(dataset.episodes) or len(ends) != len(
                dataset.episodes
            ):
                raise ValueError(
                    "episode index boundaries differ from selected episodes"
                )
            for episode, start, end in zip(dataset.episodes, starts, ends, strict=True):
                for frame in range(0, int(end - start), 20):
                    key = FrameKey(task_id, int(episode), frame)
                    if key not in trace:
                        raise ValueError(f"trace lacks released replan key: {key}")
                    used.add(key)
                    indices.append(panel.offsets[slot] + int(start) + frame)
            self.indices_by_task.append(indices)
        if used != set(trace):
            raise ValueError(
                "trace contains held-out, duplicate-split, or unmapped keys"
            )
        self.flat_indices = tuple(
            index for task in self.indices_by_task for index in task
        )

    def __len__(self) -> int:
        return len(self.panel)

    def __getitem__(self, index: int) -> dict:
        sample = self.panel[index]
        key = self.key_for_index(index, sample=sample)
        row = self.trace.get(key)
        if row is None:
            raise IndexError(f"sampler selected an untraced frame: {key}")
        return {
            **sample,
            "teacher_stage": np.asarray(row.teacher_stage, dtype=np.int32),
            "replay_stage": np.asarray(row.replay_stage, dtype=np.int32),
        }

    def key_for_index(self, index: int, *, sample: dict | None = None) -> FrameKey:
        """Return the immutable frame key for one underlying panel index."""
        return self.panel.key_for_index(index, sample=sample)


class TaskBalancedSampler:
    """Yield one seeded sample per task in every three-sample block."""

    def __init__(self, dataset, seed: int):
        while hasattr(dataset, "_dataset"):
            dataset = dataset._dataset
        if not isinstance(dataset, ReplanDataset):
            raise TypeError("TaskBalancedSampler requires ReplanDataset")
        self.indices_by_task = dataset.indices_by_task
        self.lengths = tuple(len(indices) for indices in self.indices_by_task)
        self.seed = seed
        self.epoch = 0
        self.samples_per_task = max(self.lengths)

    def __len__(self) -> int:
        return len(self.lengths) * self.samples_per_task

    def __iter__(self):
        generator = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        permutations = [
            generator.permutation(length).tolist() for length in self.lengths
        ]
        cursors = [0] * len(self.lengths)
        for _ in range(self.samples_per_task):
            task_order = generator.permutation(len(self.lengths)).tolist()
            for task_slot in task_order:
                if cursors[task_slot] == len(permutations[task_slot]):
                    permutations[task_slot] = generator.permutation(
                        self.lengths[task_slot]
                    ).tolist()
                    cursors[task_slot] = 0
                local = permutations[task_slot][cursors[task_slot]]
                cursors[task_slot] += 1
                yield self.indices_by_task[task_slot][local]

    def prefix_counts(self, samples: int) -> dict[int, int]:
        return balanced_prefix_counts(samples)


def balanced_prefix_counts(samples: int) -> dict[int, int]:
    """Count each task's samples in a balanced prefix.

    Args:
        samples: Total number of samples in the prefix.

    Returns:
        Sample count keyed by task ID.
    """

    complete, remainder = divmod(samples, len(TASKS))
    return {
        task_id: complete + (1 if slot < remainder else 0)
        for slot, task_id in enumerate(TASKS)
    }


def create_dataset(data_config, action_horizon: int, seed: int | None = None):
    """Create the panel dataset expected by the native loader.

    Args:
        data_config: Native data configuration carrying the dataset root.
        action_horizon: Number of target actions per starting frame.
        seed: Native seam parameter; sampling owns the seed separately.

    Returns:
        Training or holdout dataset restricted to replay-trace keys.

    Raises:
        ValueError: The horizon or replay trace contract differs.
    """

    del seed
    split_path = Path(data_config.behavior_dataset_root).parent / "episode-split.json"
    panel = PanelDataset(
        data_config.behavior_dataset_root,
        split_path,
        action_horizon,
        split_name=_STAGE_REPLAY_SPLIT,
    )
    if _STAGE_REPLAY_TRACE is None:
        raise ValueError("data config lacks a validated stage replay trace")
    return ReplanDataset(panel, _STAGE_REPLAY_TRACE)


def install_balanced_loader(
    module,
    *,
    arm: str,
    trace: dict[FrameKey, ReplayRow],
    split: str = "training",
) -> None:
    """Patch the pinned native module at its documented dataset/loader seams."""
    _install_trace(trace, split)
    native_loader = module.TorchDataLoader
    if getattr(native_loader, "_panel_balanced", False):
        raise RuntimeError("refusing to install over an existing balanced loader")
    balanced_loader = _balanced_loader(native_loader)
    balanced_loader._panel_balanced = True
    module.create_behavior_dataset = create_dataset
    module.TorchDataLoader = balanced_loader

    original_transform = module.transform_dataset

    def transform_dataset(dataset, data_config, **kwargs):
        return transform_conditioned_dataset(
            module,
            dataset,
            data_config,
            arm=arm,
            transform=original_transform,
            **kwargs,
        )

    module.transform_dataset = transform_dataset


def _balanced_loader(native_loader):
    def balanced_loader(dataset, local_batch_size, **kwargs):
        seed = int(kwargs.get("seed", 0))
        kwargs["shuffle"] = False
        kwargs["sampler"] = TaskBalancedSampler(dataset, seed)
        return native_loader(dataset, local_batch_size, **kwargs)

    return balanced_loader


def _install_trace(trace: dict[FrameKey, ReplayRow], split: str) -> None:
    global _STAGE_REPLAY_SPLIT, _STAGE_REPLAY_TRACE
    if _STAGE_REPLAY_TRACE is not None:
        raise RuntimeError("stage replay loader is already installed")
    _STAGE_REPLAY_TRACE = trace
    _STAGE_REPLAY_SPLIT = split


def transform_conditioned_dataset(
    module, dataset, data_config, *, arm: str, transform=None, **kwargs
):
    """Apply native transforms and inject the selected stage condition.

    Args:
        module: Pinned native data-loader module.
        dataset: Dataset with teacher and replay stage fields.
        data_config: Native transform configuration.
        arm: ``teacher`` or ``replay``.
        transform: Optional native transform function.
        **kwargs: Arguments forwarded to the native transform.

    Returns:
        Native transformed dataset with the chosen stage token.
    """

    import dataclasses

    from openpi import transforms as openpi_transforms

    transform = transform or module.transform_dataset
    repack = data_config.repack_transforms.inputs[0]
    structure = {
        **repack.structure,
        "teacher_stage": "teacher_stage",
        "replay_stage": "replay_stage",
    }
    repack_group = openpi_transforms.Group(
        inputs=(openpi_transforms.RepackTransform(structure),),
        outputs=data_config.repack_transforms.outputs,
    )
    model_group = data_config.model_transforms.push(inputs=(ApplyStageCondition(arm),))
    configured = dataclasses.replace(
        data_config,
        repack_transforms=repack_group,
        model_transforms=model_group,
    )
    return transform(dataset, configured, **kwargs)
