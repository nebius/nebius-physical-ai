"""Reject legacy SfM data that cannot retain every world-frame colored point."""

from types import SimpleNamespace

import numpy as np
import pytest

from npa.workbench.nurec import ncore_legacy_sfm as legacy
from npa.workbench.nurec.nurec import NurecError


class RayReader:
    frames_count = 1

    def __init__(self):
        self.frames_timestamps_us = np.array([[0, 3]], dtype=np.uint64)
        self.direction = np.array([[1, 0, 0], [0, 0.6, 0.8]], dtype=np.float32)
        self.distance = np.array([[2, 3]], dtype=np.float32)
        self.timestamps = np.array([0, 3], dtype=np.uint64)
        self.rgb = np.array([[1, 2, 3], [254, 255, 0]], dtype=np.uint8)
        self.valid = np.ones((1, 2), dtype=bool)
        self.ray_count = 2
        self.return_count = 1
        self.names = ["rgb"]

    def get_frame_ray_bundle_data(self, timestamp, name):
        assert timestamp == 3
        return {"direction": self.direction, "timestamp_us": self.timestamps}[name]

    def get_frame_ray_bundle_return_data(self, timestamp, name, return_index):
        assert name == "distance_m" and return_index is None
        return self.distance

    def get_frame_ray_bundle_return_valid_mask(self, timestamp):
        return self.valid

    def get_frame_ray_bundle_count(self, timestamp):
        return self.ray_count

    def get_frame_ray_bundle_return_count(self, timestamp):
        return self.return_count

    def get_frame_generic_data_names(self, timestamp):
        return self.names

    def get_frame_generic_data(self, timestamp, name):
        assert name == "rgb"
        return self.rgb


def test_legacy_decode_retains_float32_source_geometry_and_colors():
    source = RayReader()
    points = legacy._decode_sensor(source)
    assert points.pcs_count == 1
    assert points.get_pc_reference_frame_id(0) == "world"
    assert points.get_pc_xyz(0).dtype == np.float32
    np.testing.assert_array_equal(
        points.get_pc_xyz(0), source.direction * source.distance[0, :, None]
    )
    np.testing.assert_array_equal(points.get_pc_attribute(0, "rgb"), source.rgb)
    assert len(points.source_sha256) == 64
    source.rgb[0, 0] = 99
    assert points.source_sha256 != legacy._decode_sensor(source).source_sha256
    assert points.get_pc_attribute(0, "rgb")[0, 0] == 1


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("return_count", 2, "one valid return"),
        ("ray_count", 0, "one valid return"),
        ("valid", np.array([[True, False]]), "one valid return"),
        ("valid", np.ones((2, 2), dtype=bool), "one valid return"),
        ("valid", np.ones((1, 2), dtype=np.uint8), "one valid return"),
        ("direction", np.ones((2, 3), dtype=np.float32), "finite unit rays"),
        ("direction", np.eye(2, 3, dtype=np.float64), "finite unit rays"),
        ("direction", np.array([[np.nan, 0, 1]] * 2, np.float32), "finite unit rays"),
        ("direction", np.empty((0, 3), np.float32), "finite unit rays"),
        ("distance", np.array([[2, -3]], np.float32), "finite unit rays"),
        ("distance", np.array([[2, np.inf]], np.float32), "finite unit rays"),
        ("distance", np.array([2, 3], np.float32), "finite unit rays"),
        ("distance", np.array([[2, 3]], np.float64), "finite unit rays"),
        ("timestamps", np.array([0, 4], np.uint64), "outside their frame"),
        ("timestamps", np.array([0, 3], np.int64), "outside their frame"),
        ("timestamps", np.array([[0, 3]], np.uint64), "outside their frame"),
        ("rgb", np.zeros((2, 3), np.float32), "one uint8 RGB"),
        ("rgb", np.zeros((1, 3), np.uint8), "one uint8 RGB"),
        ("names", [], "missing RGB"),
        ("frames_timestamps_us", np.array([[4, 3]], np.uint64), "frame timestamps"),
        ("frames_timestamps_us", np.array([[0, 3]], np.int64), "frame timestamps"),
        ("frames_timestamps_us", np.empty((0, 2), np.uint64), "frame timestamps"),
        ("frames_count", 2, "frame timestamps"),
    ],
)
def test_legacy_decode_rejects_malformed_data_without_filtering(field, value, message):
    source = RayReader()
    setattr(source, field, value)
    with pytest.raises(NurecError, match=message):
        legacy._decode_sensor(source)


def pose_group(static=(), dynamic=()):
    return SimpleNamespace(
        get_static_poses=lambda: iter(static), get_dynamic_poses=lambda: iter(dynamic)
    )


def test_legacy_requires_world_identity_in_all_pose_groups():
    identity = (("virtual_lidar", "world"), np.eye(4, dtype=np.float32))
    legacy._require_world_identity(
        {"default": pose_group([identity]), "npa_rig": pose_group([identity])}
    )
    translated = np.eye(4)
    translated[0, 3] = 5
    with pytest.raises(NurecError, match="identity"):
        legacy._require_world_identity(
            {
                "default": pose_group([identity]),
                "npa_rig": pose_group([(("virtual_lidar", "world"), translated)]),
            }
        )


@pytest.mark.parametrize("mode", ["missing", "inverse", "dynamic", "nonfinite"])
def test_legacy_rejects_unsupported_pose_semantics(mode):
    static, dynamic = [], []
    if mode == "inverse":
        static = [(("world", "virtual_lidar"), np.eye(4))]
    if mode == "dynamic":
        dynamic = [(("virtual_lidar", "world"), (np.eye(4)[None], [0]))]
    if mode == "nonfinite":
        static = [(("virtual_lidar", "world"), np.full((4, 4), np.nan))]
    with pytest.raises(NurecError, match="world|identity"):
        legacy._require_world_identity({"default": pose_group(static, dynamic)})
