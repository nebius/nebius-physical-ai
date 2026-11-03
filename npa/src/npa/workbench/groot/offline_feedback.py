"""Validate real-robot feedback and prepare immutable N1.5 training weights."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator


class EpisodeFeedback(BaseModel):
    """An explicitly split episode with observed outcome and optional progress.

    Args:
        episode_index: LeRobot episode identifier.
        split: Training or held-out evaluation membership.
        success: Observed episode outcome.
        group: Collection group kept entirely within one split.
        progress: Optional calibrated per-frame task progress in [0, 1].
        progress_source: Whether progress was measured or predicted by ARM.
        reward_model_sha256: Required identity for ARM-produced progress.
    Returns:
        None.
    Raises:
        ValueError: Feedback has invalid values or missing provenance.
    """

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    episode_index: int = Field(ge=0)
    split: Literal["train", "eval"]
    success: bool
    group: str = Field(min_length=1)
    progress: list[float] | None = None
    progress_source: Literal["measured", "arm"] | None = None
    reward_model_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _check_progress(self):
        if self.progress is None:
            if self.progress_source or self.reward_model_sha256:
                raise ValueError("progress provenance requires per-frame progress")
            return self
        if not self.progress or any(not 0 <= value <= 1 for value in self.progress):
            raise ValueError("progress must contain finite values in [0, 1]")
        if self.progress_source is None:
            raise ValueError("progress_source is required with progress")
        if self.progress_source == "arm" and not self.reward_model_sha256:
            raise ValueError("ARM progress requires reward_model_sha256")
        return self


class PreparedEpisode(BaseModel):
    """An episode's split, length, and per-action-chunk training weights.

    Args:
        episode_index: LeRobot episode identifier.
        split: Training or evaluation membership.
        length: Number of recorded frames.
        weights: One nonnegative feedback weight per frame.
    Returns:
        None.
    Raises:
        ValueError: Weights do not match the episode length.
    """

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    episode_index: int = Field(ge=0)
    split: Literal["train", "eval"]
    length: int = Field(gt=0)
    weights: list[float]

    @model_validator(mode="after")
    def _check_weights(self):
        if len(self.weights) != self.length or any(value < 0 for value in self.weights):
            raise ValueError("weights must be nonnegative and match episode length")
        if self.split == "eval" and any(value != 1 for value in self.weights):
            raise ValueError("held-out weights must remain one")
        return self


class OfflineManifest(BaseModel):
    """A content-bound offline training input, separate from policy evaluation.

    Args:
        schema_version: Manifest format identity.
        dataset_sha256: Digest of all meta, data, and video file bytes.
        feedback_sha256: Digest of the source feedback file.
        mode: Outcome-return or progress-gain weighting.
        action_horizon: Contiguous action chunk length.
        temperature: Positive exponential-weighting temperature.
        episodes: Explicit split membership and frame weights.
    Returns:
        None.
    Raises:
        ValueError: Split membership or training weights are invalid.
    """

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    schema_version: Literal["npa.groot-offline/v1"] = "npa.groot-offline/v1"
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    feedback_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    mode: Literal["outcome", "progress"]
    action_horizon: int = Field(gt=0)
    temperature: float = Field(gt=0)
    episodes: list[PreparedEpisode]

    @model_validator(mode="after")
    def _check_splits(self):
        identifiers = [episode.episode_index for episode in self.episodes]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("duplicate episode_index")
        if {episode.split for episode in self.episodes} != {"train", "eval"}:
            raise ValueError("both train and eval episodes are required")
        weights = [w for e in self.episodes if e.split == "train" for w in e.weights]
        if not weights or sum(weights) <= 0:
            raise ValueError("training weights must have positive total mass")
        return self


def dataset_digest(dataset_root: Path) -> str:
    """Hash dataset paths and bytes without storing observations in the manifest.

    Args:
        dataset_root: Local, staged LeRobot dataset directory.
    Returns:
        SHA-256 covering meta, data, and videos, including their relative paths.
    Raises:
        ValueError: A dataset path is a symlink or no files exist.
        OSError: Dataset bytes cannot be read.
    """
    digest = hashlib.sha256()
    files = []
    for name in ("meta", "data", "videos"):
        directory = dataset_root / name
        for path in [directory, *sorted(directory.rglob("*"))]:
            if path.is_symlink():
                raise ValueError("dataset symlinks are unsupported")
            if path.is_file():
                files.append(path)
    if not files:
        raise ValueError("dataset has no meta, data, or video files")
    for path in files:
        relative = path.relative_to(dataset_root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(file_digest(path)))
    return digest.hexdigest()


def file_digest(path: Path) -> str:
    """Hash a file using bounded memory.

    Args:
        path: File to hash.
    Returns:
        Its SHA-256 hex digest.
    Raises:
        OSError: File cannot be read.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_feedback(
    dataset_root: Path,
    feedback_path: Path,
    output_path: Path,
    *,
    action_horizon: int,
    mode: Literal["outcome", "progress"],
    temperature: float = 1.0,
) -> OfflineManifest:
    """Prepare feedback weights without modifying the source LeRobot dataset.

    Args:
        dataset_root: Staged native N1.5-compatible LeRobot v2 dataset.
        feedback_path: JSON array containing every episode's feedback.
        output_path: New manifest file outside the dataset directory.
        action_horizon: Number of actions predicted per contiguous chunk.
        mode: Outcome-return or progress-gain weighting.
        temperature: Positive temperature; lower values favor higher returns.
    Returns:
        Validated, content-bound training manifest written to output_path.
    Raises:
        ValueError: Data, feedback, or weighting parameters are invalid.
        OSError: Inputs cannot be read or output already exists.
    """
    _check_preparation_paths(dataset_root, output_path)
    if type(action_horizon) is not int or action_horizon <= 0:
        raise ValueError("action_horizon must be a positive integer")
    if (
        isinstance(temperature, bool)
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("temperature must be positive and finite")
    if mode not in {"outcome", "progress"}:
        raise ValueError("mode must be outcome or progress")
    feedback = TypeAdapter(list[EpisodeFeedback]).validate_json(
        feedback_path.read_text()
    )
    digest = dataset_digest(dataset_root)
    lengths = _read_episode_lengths(dataset_root)
    _validate_feedback(feedback, lengths, mode)
    episodes = _prepare_episodes(feedback, lengths, action_horizon, mode, temperature)
    manifest = OfflineManifest(
        dataset_sha256=digest,
        feedback_sha256=file_digest(feedback_path),
        mode=mode,
        action_horizon=action_horizon,
        temperature=temperature,
        episodes=episodes,
    )
    with output_path.open("x") as stream:
        stream.write(manifest.model_dump_json(indent=2) + "\n")
    return manifest


def _check_preparation_paths(dataset_root, output_path):
    if output_path.resolve().is_relative_to(dataset_root.resolve()):
        raise ValueError("output must be outside the immutable dataset")
    if output_path.exists():
        raise FileExistsError(output_path)


def _read_episode_lengths(dataset_root):
    import pyarrow.parquet as parquet

    info = json.loads((dataset_root / "meta/info.json").read_text())
    if info.get("codebase_version") not in {"v2.0", "v2.1"}:
        raise ValueError("native N1.5 preparation requires LeRobot v2 data")
    chunk_size = info.get("chunks_size")
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("chunks_size must be a positive integer")
    lengths = {}
    for line in (dataset_root / "meta/episodes.jsonl").read_text().splitlines():
        episode = json.loads(line)
        identifier, length = episode["episode_index"], episode["length"]
        if type(identifier) is not int or identifier < 0 or identifier in lengths:
            raise ValueError("invalid or duplicate episode_index in metadata")
        if type(length) is not int or length <= 0:
            raise ValueError("episode length must be a positive integer")
        bindings = {
            "episode_chunk": identifier // chunk_size,
            "episode_index": identifier,
        }
        path = _bound_dataset_path(dataset_root, "data", info["data_path"], bindings)
        _validate_video_paths(dataset_root, info, bindings)
        table = parquet.read_table(path, columns=["episode_index", "frame_index"])
        import pyarrow

        if not all(pyarrow.types.is_integer(field.type) for field in table.schema):
            raise ValueError(
                "episode_index and frame_index must have integer parquet types"
            )
        if table["episode_index"].to_pylist() != [identifier] * length:
            raise ValueError("parquet episode_index or length differs from metadata")
        if table["frame_index"].to_pylist() != list(range(length)):
            raise ValueError("frame_index must be contiguous and start at zero")
        lengths[identifier] = length
    return lengths


def _bound_dataset_path(root, directory, pattern, bindings):
    relative = Path(pattern.format(**bindings))
    path = root / relative
    if relative.is_absolute() or not path.resolve().is_relative_to(
        (root / directory).resolve()
    ):
        raise ValueError(
            f"{directory} path must stay inside the hashed {directory} directory"
        )
    return path


def _validate_video_paths(root, info, bindings):
    if "video_path" not in info:
        return
    keys = [
        key
        for key, value in info.get("features", {}).items()
        if value.get("dtype") == "video"
    ]
    for key in keys or ["observation.images.example"]:
        path = _bound_dataset_path(
            root, "videos", info["video_path"], dict(bindings, video_key=key)
        )
        if keys and not path.is_file():
            raise ValueError("declared observation video is missing")


def _validate_feedback(feedback, lengths, mode):
    identifiers = [episode.episode_index for episode in feedback]
    if len(identifiers) != len(set(identifiers)) or set(identifiers) != set(lengths):
        raise ValueError("feedback must cover every dataset episode exactly once")
    groups = {}
    for episode in feedback:
        previous = groups.setdefault(episode.group, episode.split)
        if previous != episode.split:
            raise ValueError("a collection group cannot cross train/eval splits")
        if (
            episode.progress is not None
            and len(episode.progress) != lengths[episode.episode_index]
        ):
            raise ValueError("progress must match recorded episode length")
        if mode == "progress" and episode.progress is None:
            raise ValueError(
                "progress mode requires explicit progress for every episode"
            )
    training = [episode for episode in feedback if episode.split == "train"]
    if not training or not any(episode.split == "eval" for episode in feedback):
        raise ValueError("both train and eval episodes are required")
    if mode == "outcome" and not any(episode.success for episode in training):
        raise ValueError("outcome mode needs at least one successful training episode")


def _prepare_episodes(feedback, lengths, horizon, mode, temperature):
    gains = {}
    for episode in feedback:
        length = lengths[episode.episode_index]
        if episode.split == "eval":
            continue
        if mode == "outcome":
            gains[episode.episode_index] = [float(episode.success)] * length
        else:
            progress = episode.progress
            gains[episode.episode_index] = [
                progress[min(frame + horizon, length - 1)] - progress[frame]
                for frame in range(length)
            ]
    maximum = max(value for values in gains.values() for value in values)
    weights = {
        identifier: [math.exp((v - maximum) / temperature) for v in values]
        for identifier, values in gains.items()
    }
    mean = sum(sum(values) for values in weights.values()) / sum(
        map(len, weights.values())
    )
    return [
        PreparedEpisode(
            episode_index=e.episode_index,
            split=e.split,
            length=lengths[e.episode_index],
            weights=[w / mean for w in weights[e.episode_index]]
            if e.split == "train"
            else [1.0] * lengths[e.episode_index],
        )
        for e in feedback
    ]
