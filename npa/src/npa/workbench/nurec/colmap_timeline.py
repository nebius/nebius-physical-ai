"""Represent independent still cameras on a nonsynchronized virtual rig timeline."""

from __future__ import annotations

from typing import Any

import numpy as np

FRAME_POSE_MODE = "independent-camera-virtual-time-v2"


def camera_timestamps(cameras: dict[str, Any]) -> dict[str, np.ndarray]:
    """Assign disjoint photographic indices, sharing indices with downsample aliases.

    Args:
        cameras: Audited source camera records with targets and ordered frames.
    Returns:
        Per-camera virtual microsecond arrays; these are not capture times.
    Raises:
        ValueError: A camera alias has no original or differs in frame count.
    """
    result = {}
    offset = 0
    for name in sorted(cameras):
        camera = cameras[name]
        if camera["target"] != "world":
            continue
        count = len(camera["frames"])
        result[name] = np.arange(offset, offset + count, dtype=np.uint64) * 1_000_000
        offset += count
    for name, camera in cameras.items():
        if name in result:
            continue
        target = camera["target"]
        if target not in result or len(camera["frames"]) != len(result[target]):
            raise ValueError("virtual camera alias differs from its source")
        result[name] = result[target].copy()
    return result


def merge_world_trajectories(trajectories: dict[str, tuple[Any, Any]]) -> tuple:
    """Merge independently timed world trajectories into one exact-knot virtual rig.

    Args:
        trajectories: Original camera world poses and virtual timestamps.
    Returns:
        Ordered poses and timestamps for the virtual rig.
    Raises:
        ValueError: Camera times overlap or do not form the declared virtual grid.
    """
    if not trajectories or any(
        np.shape(poses) != (len(times), 4, 4) for poses, times in trajectories.values()
    ):
        raise ValueError("virtual rig trajectory dimensions differ")
    poses = np.concatenate([trajectories[name][0] for name in sorted(trajectories)])
    times = np.concatenate([trajectories[name][1] for name in sorted(trajectories)])
    expected = np.arange(len(times), dtype=np.uint64) * 1_000_000
    if not np.array_equal(times, expected) or poses.shape != (len(times), 4, 4):
        raise ValueError(
            "independent camera timelines overlap or differ from virtual grid"
        )
    if not np.isfinite(poses).all():
        raise ValueError("virtual rig poses must be finite")
    return poses, times


def source_time_mapping(cameras: dict[str, Any]) -> list[dict[str, Any]]:
    """Bind each source photograph and original order to its new virtual timestamp.

    Args:
        cameras: Audited source camera records, including encoded image hashes.
    Returns:
        Explicit per-frame lineage, without a physical capture-time claim.
    Raises:
        ValueError: Camera alias metadata is inconsistent.
    """
    timestamps = camera_timestamps(cameras)
    return [
        {
            "camera": camera,
            "source_image": frame["name"],
            "source_image_sha256": frame["encoded_sha256"],
            "source_frame_index": index,
            "original_virtual_timestamp_us": index * 1_000_000,
            "virtual_timestamp_us": int(timestamps[camera][index]),
        }
        for camera in sorted(cameras)
        for index, frame in enumerate(cameras[camera]["frames"])
    ]
