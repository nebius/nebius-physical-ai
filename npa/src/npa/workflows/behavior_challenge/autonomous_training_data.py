"""Project finalized TRAIN experience into the existing Comet sample contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .comet_training_data import (
    ACTION_DIMENSION,
    ACTION_HORIZON,
    CAMERAS,
    STATE_DIMENSION,
    action_valid_mask,
)
from .train_experience import file_identity, validate_finalized_experience
from .train_prompt import (
    legacy_prompt_derivation_sha256,
    validate_legacy_prompt_derivation,
    validate_prompt_binding,
)


def _dataset_prompt_binding(
    config: dict,
    legacy: dict | None,
    expected_legacy_sha256: str | None,
    config_path: Path,
) -> dict:
    if config["schema"] == "npa.behavior.train-experience-config.v2":
        if legacy is not None or expected_legacy_sha256 is not None:
            raise ValueError("Legacy prompt binding supplied for config v2")
        return validate_prompt_binding(config["prompt_binding"])
    if legacy is None or expected_legacy_sha256 is None:
        raise ValueError("Legacy TRAIN prompt requires pinned derivation")
    if legacy_prompt_derivation_sha256(legacy) != expected_legacy_sha256:
        raise ValueError("Legacy TRAIN prompt derivation identity differs")
    derivation = validate_legacy_prompt_derivation(legacy, config_path)
    return derivation["prompt_binding"]


class AutonomousCometDataset:
    """Read model-decision observations with official applied-action targets.

    Args:
        root: Finalized TRAIN experience root.
        legacy_prompt_derivation: Explicit receipt joining a v1 recording to
            its original qualification, source, and effective prompt.
        legacy_prompt_derivation_sha256: Independently admitted canonical
            identity of that receipt.
    Returns:
        None.
    Raises:
        ValueError: Manifest, prompt lineage, cadence, shards, or arrays differ.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        legacy_prompt_derivation: dict | None = None,
        legacy_prompt_derivation_sha256: str | None = None,
    ) -> None:
        self.root = Path(root)
        self.manifest = _manifest(self.root)
        self.config = self.manifest["config"]
        self.prompt_binding = _dataset_prompt_binding(
            self.config,
            legacy_prompt_derivation,
            legacy_prompt_derivation_sha256,
            self.root / "config.json",
        )
        self.decisions = _decision_locations(self.root)
        self.actions = _applied_actions(self.root)
        self.frame_count = self.manifest["frame_count"]
        _validate_decision_frames(self.decisions, self.frame_count)

    def __len__(self) -> int:
        """Return the number of recorded model decisions.

        Args:
            None.
        Returns:
            Number of decision-cadence samples.
        Raises:
            None.
        """
        return len(self.decisions)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one exact Comet sample at model-decision cadence.

        Args:
            index: Zero-based decision sample index.
        Returns:
            Existing Comet RGB/state/action/mask sample mapping.
        Raises:
            IndexError: The sample index is outside the experience.
            ValueError: Stored arrays differ from the Comet contract.
        """
        path, offset, frame = self.decisions[index]
        observation = _decision_arrays(path, offset)
        state = np.array(observation["robot_r1::proprio"], copy=True)
        if state.shape != (STATE_DIMENSION,) or state.dtype != np.float32:
            raise ValueError("Autonomous Comet state contract differs")
        valid = action_valid_mask(frame, self.frame_count)
        actions = _action_target(self.actions, frame, valid)
        result = {
            "observation.state": state,
            "action": actions,
            "action_valid_mask": valid,
            "action_is_pad": ~valid,
            "prompt": self.prompt_binding["effective_prompt"],
            "episode_index": np.asarray(0),
            "frame_index": np.asarray(frame),
        }
        result.update(_rgb_arrays(observation))
        return result

    def terminal_observation(self) -> dict[str, np.ndarray]:
        """Return the lossless final post-apply allowed observation.

        Args:
            None.
        Returns:
            Final frame index, RGB, proprioception, and enabled depth arrays.
        Raises:
            ValueError: The final observation shard differs.
        """
        directory = self.root / "evaluator/final-observation"
        rows = _index(directory)
        if len(rows) != 1 or rows[0].get("count") != 1:
            raise ValueError("Autonomous Comet final observation differs")
        path = _shard_path(directory, rows[0])
        _verify_shard(path, rows[0])
        with np.load(path, allow_pickle=False) as shard:
            result = {name: np.array(shard[name][0], copy=True) for name in shard.files}
        if int(result["frame_index"]) != self.frame_count:
            raise ValueError("Autonomous Comet final observation index differs")
        return result


class OfficialGoalProgressDataset:
    """Read post-action official TRAIN progress without changing policy inputs.

    Args:
        root: Finalized TRAIN experience root with v3.9.3 progress records.
    Returns:
        None.
    Raises:
        ValueError: Progress evidence is absent, invalid, or misaligned.
    """

    def __init__(self, root: str | Path) -> None:
        """Validate and open one official TRAIN goal-progress recording.

        Args:
            root: Finalized TRAIN experience root with v3.9.3 progress records.
        Returns:
            None.
        Raises:
            ValueError: Progress evidence is absent, invalid, or misaligned.
        """
        self.root = Path(root)
        self.manifest = validate_finalized_experience(self.root)
        annotations = self.manifest.get("training_annotations", {})
        if "official_goal_status" not in annotations:
            raise ValueError("TRAIN experience has no official goal progress")
        self.rows = _progress_rows(self.root)
        if len(self.rows) != self.manifest["frame_count"]:
            raise ValueError("TRAIN goal progress action alignment differs")

    def __len__(self) -> int:
        """Return the number of officially applied actions.

        Args:
            None.
        Returns:
            Number of action-aligned post-step labels.
        Raises:
            None.
        """
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one exact post-action label and derived potential target.

        Args:
            index: Zero-based officially applied action index.
        Returns:
            Exact official status plus its satisfied-goal fraction.
        Raises:
            IndexError: The index is outside the recorded rollout.
        """
        row = self.rows[index]
        result = json.loads(json.dumps(row))
        result["potential"] = np.float32(row["satisfied_count"] / row["goal_count"])
        return result


def _manifest(root: Path) -> dict[str, Any]:
    return validate_finalized_experience(root)


def _progress_rows(root: Path) -> list[dict[str, Any]]:
    path = root / "evaluator/goal-progress.jsonl"
    if path.is_symlink() or not path.is_file():
        raise ValueError("TRAIN goal progress records differ")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError("TRAIN goal progress records differ")
    return rows


def _index(root: Path) -> list[dict[str, Any]]:
    value = json.loads((root / "index.json").read_text())
    rows = value.get("shards") if isinstance(value, dict) else None
    if not isinstance(rows, list):
        raise ValueError("Autonomous Comet shard index differs")
    return rows


def _decision_locations(root: Path) -> list[tuple[Path, int, int]]:
    directory = root / "evaluator/decisions"
    result = []
    for row in _index(directory):
        path = _shard_path(directory, row)
        _verify_shard(path, row)
        with np.load(path, allow_pickle=False) as shard:
            frames = np.asarray(shard["frame_index"])
        result.extend((path, offset, int(frame)) for offset, frame in enumerate(frames))
    return result


def _applied_actions(root: Path) -> np.ndarray:
    directory = root / "evaluator/transitions"
    parts = []
    for row in _index(directory):
        path = _shard_path(directory, row)
        _verify_shard(path, row)
        with np.load(path, allow_pickle=False) as shard:
            parts.append(np.array(shard["applied_action"], copy=True))
    actions = np.concatenate(parts)
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIMENSION:
        raise ValueError("Autonomous Comet applied actions differ")
    if not np.issubdtype(actions.dtype, np.floating) or not np.isfinite(actions).all():
        raise ValueError("Autonomous Comet applied-action dtype differs")
    return actions


def _shard_path(directory: Path, row: object) -> Path:
    if not isinstance(row, dict) or not isinstance(row.get("path"), str):
        raise ValueError("Autonomous Comet shard row differs")
    relative = Path(row["path"])
    path = directory / relative
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or path.is_symlink()
        or not path.is_file()
        or not path.resolve().is_relative_to(directory.resolve())
    ):
        raise ValueError("Autonomous Comet shard path differs")
    return path


def _verify_shard(path: Path, row: dict[str, Any]) -> None:
    expected = {name: row[name] for name in ("bytes", "sha256")}
    if file_identity(path) != expected:
        raise ValueError("Autonomous Comet shard bytes differ")


def _validate_decision_frames(rows, frame_count: int) -> None:
    frames = [frame for _, _, frame in rows]
    expected = list(range(0, frame_count, ACTION_HORIZON))
    if frames != expected:
        raise ValueError("Autonomous Comet decision cadence differs")


def _decision_arrays(path: Path, offset: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as shard:
        return {
            name: np.array(shard[name][offset], copy=True)
            for name in shard.files
            if name != "frame_index"
        }


def _action_target(actions: np.ndarray, frame: int, valid: np.ndarray) -> np.ndarray:
    indices = np.minimum(np.arange(frame, frame + ACTION_HORIZON), len(actions) - 1)
    result = np.asarray(actions[indices], dtype=np.float32)
    if result.shape != (ACTION_HORIZON, ACTION_DIMENSION) or not valid.any():
        raise ValueError("Autonomous Comet action horizon differs")
    result[~valid] = result[np.flatnonzero(valid)[-1]]
    return result


def _rgb_arrays(observation: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    result = {}
    for role, camera in CAMERAS.items():
        source = f"robot_r1::robot_r1:{camera}:Camera:0::rgb"
        pixels = observation[source]
        if (
            pixels.dtype != np.uint8
            or pixels.ndim != 3
            or pixels.shape[-1] not in {3, 4}
        ):
            raise ValueError("Autonomous Comet RGB contract differs")
        result[f"observation.images.rgb.{role}"] = np.array(pixels[..., :3], copy=True)
    return result


__all__ = ["AutonomousCometDataset", "OfficialGoalProgressDataset"]
