"""Fail-closed Antioch trajectory validation and LeRobot conversion seam."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from npa.adapter.sim_to_lerobot import AdapterError, convert

from .schemas import DATASET_PROVENANCE_SCHEMA, EpisodeProvenance


class AntiochDatasetError(RuntimeError):
    pass


REQUIRED = {
    "observation_state",
    "observation_image_workspace",
    "observation_image_wrist",
    "action",
    "reward",
    "terminated",
    "truncated",
    "timestamp",
    "provenance",
}


def validate_episode(path: Path) -> tuple[dict[str, np.ndarray], EpisodeProvenance]:
    """Validate the explicit offline-trajectory contract; arbitrary telemetry is rejected."""

    try:
        bundle = np.load(path, allow_pickle=False)
        missing = REQUIRED.difference(bundle.files)
        if missing:
            raise AntiochDatasetError(
                f"trajectory is missing required arrays: {', '.join(sorted(missing))}"
            )
        arrays = {name: bundle[name] for name in REQUIRED - {"provenance"}}
        provenance_raw = bundle["provenance"].item()
        provenance = EpisodeProvenance.model_validate_json(str(provenance_raw))
    except AntiochDatasetError:
        raise
    except Exception as exc:
        raise AntiochDatasetError(
            "trajectory bundle is malformed or has invalid provenance"
        ) from exc
    length = arrays["observation_state"].shape[0]
    if length < 2 or any(value.shape[0] != length for value in arrays.values()):
        raise AntiochDatasetError(
            "trajectory arrays must have the same non-trivial episode length"
        )
    if arrays["observation_state"].ndim != 2 or arrays["action"].ndim != 2:
        raise AntiochDatasetError("observation_state and action must be rank-2 arrays")
    if arrays["action"].shape[1] != len(provenance.action_schema):
        raise AntiochDatasetError(
            "action width does not match provenance action_schema"
        )
    if arrays["action"].shape[1] < 2:
        raise AntiochDatasetError(
            "the pinned LeRobot ACT trainer requires at least two action channels; "
            "refusing to pad or duplicate a one-channel control"
        )
    for camera in ("observation_image_workspace", "observation_image_wrist"):
        value = arrays[camera]
        if value.ndim != 4 or value.shape[-1] != 3 or value.dtype != np.uint8:
            raise AntiochDatasetError(f"{camera} must be uint8 [T,H,W,3]")
    for name in ("reward", "terminated", "truncated", "timestamp"):
        if arrays[name].ndim != 1:
            raise AntiochDatasetError(f"{name} must be rank 1")
    if not np.all(np.isfinite(arrays["observation_state"])) or not np.all(
        np.isfinite(arrays["action"])
    ):
        raise AntiochDatasetError(
            "observations and actions must contain only finite values"
        )
    timestamps = arrays["timestamp"].astype(np.float64)
    if not np.all(np.isfinite(timestamps)) or not np.all(np.diff(timestamps) > 0):
        raise AntiochDatasetError("timestamps must be finite and strictly increasing")
    terminated = arrays["terminated"].astype(bool)
    truncated = arrays["truncated"].astype(bool)
    if np.any(terminated[:-1] | truncated[:-1]) or bool(terminated[-1]) == bool(
        truncated[-1]
    ):
        raise AntiochDatasetError(
            "exactly one terminal flag must be set, on the final frame only"
        )
    return arrays, provenance


def convert_episodes(
    trajectories: list[Path],
    output: Path,
    *,
    robot_type: str,
    task: str,
    source_sha256: str,
    asset_hashes: dict[str, str],
) -> dict[str, object]:
    """Convert validated offline episodes to the real LeRobotDataset v3 contract."""

    if not trajectories:
        raise AntiochDatasetError("no trajectory .npz artifacts were collected")
    source = output.parent / "validated-episodes"
    source.mkdir()
    provenance_records: list[dict[str, object]] = []
    fps: int | None = None
    for index, trajectory in enumerate(sorted(trajectories)):
        arrays, provenance = validate_episode(trajectory)
        if provenance.source_sha256 != source_sha256:
            raise AntiochDatasetError(
                "episode provenance does not match the immutable project source"
            )
        if provenance.assets_sha256 != asset_hashes:
            raise AntiochDatasetError(
                "episode provenance does not match the immutable project assets"
            )
        if fps is None:
            fps = provenance.fps
        elif fps != provenance.fps:
            raise AntiochDatasetError(
                "all episodes in one dataset must use the same fps"
            )
        episode = source / f"episode_{index:04d}"
        episode.mkdir()
        np.save(episode / "state.npy", arrays["observation_state"])
        np.save(episode / "actions.npy", arrays["action"])
        np.save(episode / "obs_workspace.npy", arrays["observation_image_workspace"])
        np.save(episode / "obs_wrist.npy", arrays["observation_image_wrist"])
        provenance_records.append(provenance.model_dump(mode="json"))
    try:
        convert(source, output, fps=int(fps or 1), robot_type=robot_type, task=task)
    except AdapterError as exc:
        raise AntiochDatasetError(f"LeRobot conversion failed: {exc}") from exc
    record = {
        "schema_name": DATASET_PROVENANCE_SCHEMA,
        "episodes": provenance_records,
        "conversion": "npa.adapter.sim_to_lerobot",
        "training_semantics": "offline_trajectory_imitation",
    }
    (output / "meta" / "antioch-provenance.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record
