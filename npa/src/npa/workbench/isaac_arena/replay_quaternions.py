"""Translate declared legacy GR1 Pink poses into Isaac Lab 3's XYZW format."""

from __future__ import annotations

from numbers import Integral
from typing import Any

from .errors import IsaacArenaError


def dataset_format_version(dataset: Any) -> int:
    """Match the pinned Lab loader's missing-version convention, without guessing."""
    version = dataset.attrs.get("format_version", 0)
    if (
        isinstance(version, bool)
        or not isinstance(version, Integral)
        or version not in (0, 1)
    ):
        raise IsaacArenaError("replay format_version must be integer 0 or 1")
    return int(version)


def _validate_quaternions(values: Any, np: Any) -> None:
    norms = np.linalg.norm(values.astype(np.float64), axis=-1)
    if not np.isfinite(norms).all() or not (norms > 0).all():
        raise IsaacArenaError(
            "GR1 Pink replay poses require finite nonzero quaternions"
        )


def _pink_actions(episode: Any, legacy: bool, np: Any) -> list[list[int]]:
    actions = np.asarray(episode["actions"])
    if actions.ndim != 2 or actions.shape[1] != 36:
        raise IsaacArenaError("gr1_pink replay requires exactly 36 action columns")
    slices = [(3, 7), (10, 14)]
    for start, stop in slices:
        _validate_quaternions(actions[:, start:stop], np)
        if legacy:
            actions[:, start:stop] = np.roll(actions[:, start:stop], -1, axis=-1)
    if legacy:
        episode["actions"][...] = actions
    return [list(bounds) for bounds in slices] if legacy else []


def _initial_root_poses(episode: Any, legacy: bool, h5py: Any, np: Any) -> int:
    if not isinstance(
        episode.get("initial_state/articulation/robot/root_pose"), h5py.Dataset
    ):
        raise IsaacArenaError("GR1 Pink replay requires its initial robot root_pose")
    converted = 0

    def inspect(name: str, item: Any) -> None:
        nonlocal converted
        if not isinstance(item, h5py.Dataset) or name.rsplit("/", 1)[-1] != "root_pose":
            return
        values = np.asarray(item)
        if values.shape != (1, 7):
            raise IsaacArenaError(
                "GR1 Pink replay root_pose must have single-environment shape (1, 7)"
            )
        _validate_quaternions(values[..., 3:7], np)
        if legacy:
            values[..., 3:7] = np.roll(values[..., 3:7], -1, axis=-1)
            item[...] = values
            converted += 1

    episode["initial_state"].visititems(inspect)
    return converted


def normalize_pose_representation(
    source: Any, execution: Any, episode_name: str, embodiment: str
) -> dict[str, Any]:
    """Convert only the resolved Pink contract; joint actions keep native handling."""
    import h5py
    import numpy as np

    version = dataset_format_version(source)
    record = {
        "source_format_version": version,
        "source_version_defaulted": "format_version" not in source.attrs,
        "execution_format_version": version,
        "embodiment": embodiment,
        "action_quaternion_slices": [],
        "initial_root_poses_converted": 0,
        "conversion": "upstream_root_pose_only" if version == 0 else "none",
    }
    if embodiment != "gr1_pink":
        return record
    episode = execution["data"][episode_name]
    record["action_quaternion_slices"] = _pink_actions(episode, version == 0, np)
    record["initial_root_poses_converted"] = _initial_root_poses(
        episode, version == 0, h5py, np
    )
    execution.attrs["format_version"] = 1
    record["execution_format_version"] = 1
    record["conversion"] = "wxyz_to_xyzw_gr1_pink" if version == 0 else "none"
    return record
