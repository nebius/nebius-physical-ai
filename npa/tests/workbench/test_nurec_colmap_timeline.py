"""Prove independent photographic timelines cannot alias native rig pose knots."""

import numpy as np
import pytest

from npa.workbench.nurec.colmap_timeline import (
    camera_timestamps,
    merge_world_trajectories,
    source_time_mapping,
)


def _cameras():
    def camera(count, target="world"):
        return {
            "target": target,
            "frames": [
                {"name": f"image-{index}.png", "encoded_sha256": "a" * 64}
                for index in range(count)
            ],
        }

    return {
        "camera2": camera(2),
        "camera1": camera(3),
        "camera1_2": camera(3, "camera1"),
    }


def test_virtual_times_preserve_order_and_downsample_aliases():
    actual = camera_timestamps(_cameras())
    np.testing.assert_array_equal(actual["camera1"], [0, 1_000_000, 2_000_000])
    np.testing.assert_array_equal(actual["camera2"], [3_000_000, 4_000_000])
    np.testing.assert_array_equal(actual["camera1_2"], actual["camera1"])
    assert all(value.dtype == np.uint64 for value in actual.values())


@pytest.mark.parametrize("corruption", ["missing-parent", "count"])
def test_virtual_times_reject_invalid_alias(corruption):
    cameras = _cameras()
    if corruption == "missing-parent":
        cameras["camera1_2"]["target"] = "unknown"
    else:
        cameras["camera1_2"]["frames"].pop()
    with pytest.raises(ValueError, match="alias differs"):
        camera_timestamps(cameras)


def test_source_mapping_binds_original_and_virtual_indices_without_capture_claim():
    cameras = _cameras()
    rows = source_time_mapping(cameras)
    assert len(rows) == 8
    second = [row for row in rows if row["camera"] == "camera2"]
    assert second[0] == {
        "camera": "camera2",
        "source_image": "image-0.png",
        "source_image_sha256": "a" * 64,
        "source_frame_index": 0,
        "original_virtual_timestamp_us": 0,
        "virtual_timestamp_us": 3_000_000,
    }
    assert all("capture" not in key for row in rows for key in row)


def _trajectories():
    poses = np.repeat(np.eye(4)[None], 5, axis=0)
    poses[:, 0, 3] = np.arange(5) * 3
    timestamps = np.arange(5, dtype=np.uint64) * 1_000_000
    return {
        "camera2": (poses[3:], timestamps[3:]),
        "camera1": (poses[:3], timestamps[:3]),
    }


def test_virtual_rig_matches_every_source_camera_at_its_exact_knots():
    source = _trajectories()
    poses, times = merge_world_trajectories(source)
    for expected, timestamps in source.values():
        np.testing.assert_array_equal(
            poses[np.searchsorted(times, timestamps)], expected
        )
    assert len(poses) == 5


@pytest.mark.parametrize("corruption", ["old-overlap", "gap", "nan", "shape"])
def test_virtual_rig_rejects_ambiguous_or_invalid_source(corruption):
    source = _trajectories()
    poses, times = source["camera2"]
    if corruption == "old-overlap":
        times = np.array([0, 1_000_000], dtype=np.uint64)
    elif corruption == "gap":
        times = times + 1_000_000
    elif corruption == "nan":
        poses[0, 0, 0] = np.nan
    else:
        poses = poses[:, :3]
    source["camera2"] = poses, times
    with pytest.raises(ValueError, match="virtual|overlap"):
        merge_world_trajectories(source)
