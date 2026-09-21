"""Build the released task-1 LeRobot view used by Comet training.

This module owns only the dataset boundary. The pinned Comet source still owns
its B1K state transform, normalization, model, and training loop.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


DATASET_REPOSITORY = "behavior-1k/2026-challenge-demos"
DATASET_REVISION = "4f50b44796641a4d526a19d9aeadc8aa51e2f2c2"
TASK_ID = 1
TASK_NAME = "picking_up_trash"
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
    split = _load_bound_split(path, expected_split_sha256)
    return _partition_episodes(split, partition)


def _partition_episodes(split: Mapping[str, Any], partition: str) -> tuple[int, ...]:
    if partition not in {"training", "holdout"}:
        raise ValueError("partition must be training or holdout")
    if (
        split.get("schema") != "npa.behavior.panel-episode-split.v1"
        or split.get("revision") != DATASET_REVISION
    ):
        raise ValueError("Episode split identity differs")
    row = split.get("tasks", {}).get(str(TASK_ID), {})
    training = row.get("training")
    holdout = row.get("holdout")
    if (
        row.get("name") != TASK_NAME
        or not isinstance(training, list)
        or not isinstance(holdout, list)
        or len(training) != 180
        or len(holdout) != 20
    ):
        raise ValueError("Task-1 episode split differs")
    combined = training + holdout
    if (
        any(type(value) is not int or value < 0 for value in combined)
        or len(set(combined)) != 200
        or set(training) & set(holdout)
    ):
        raise ValueError("Task-1 episode split overlaps or contains invalid IDs")
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
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if kwargs.pop("return_uint8", None) is not True:
        raise ValueError("Comet dataset decoding must request uint8 RGB")
    return _Uint8LeRobotDataset(LeRobotDataset(*args, **kwargs))


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


class CometTask1Dataset:
    """Expose one isolated task-1 partition with explicit terminal actions.

    Args:
        root: Extracted LeRobot dataset root.
        split_path: Reviewed 180/20 episode split.
        expected_split_sha256: Approved SHA-256 of the raw split bytes.
        partition: Isolated partition to expose.
        dataset_factory: Optional LeRobot-compatible factory for testing or the
            pinned runtime.
    Raises:
        ValueError: Split or episode metadata violates the contract.
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
        self.episodes = load_task1_episodes(
            Path(split_path),
            partition,
            expected_split_sha256=expected_split_sha256,
        )
        factory = dataset_factory or _default_dataset_factory
        self.dataset = factory(
            DATASET_REPOSITORY,
            root=Path(root),
            episodes=list(self.episodes),
            revision=DATASET_REVISION,
            delta_timestamps={"action": [step / FPS for step in range(ACTION_HORIZON)]},
            video_backend="pyav",
            return_uint8=True,
            tolerance_s=ALIGNMENT_TOLERANCE_SECONDS,
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
            "prompt": TASK_NAME,
            "episode_index": np.asarray(episode),
            "frame_index": np.asarray(frame),
        }
        for role, camera in CAMERAS.items():
            result[f"observation.images.rgb.{role}"] = _validated_rgb(
                sample, role, camera
            )
        return result


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
