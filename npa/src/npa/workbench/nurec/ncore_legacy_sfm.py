"""Decode unchanged legacy NCore virtual-LiDAR SfM points in the world frame.

Uses NVIDIA's public LidarSensorComponent API and its direction/distance
contract: https://github.com/NVIDIA/ncore/blob/59c698d206da92b406a4f72619fce3b3a2c64bfd/ncore/impl/data/v4/components.py
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from npa.workbench.nurec.nurec import NurecError

_SENSOR = "virtual_lidar"
_SOURCE = "NCore LidarSensorComponent virtual_lidar SfM direction/distance and rgb"


@dataclass
class _LegacySfmPoints:
    clouds: list[tuple[np.ndarray, np.ndarray]]
    source_sha256: str
    source_description = _SOURCE
    attribute_names = ("rgb",)

    @property
    def pcs_count(self) -> int:
        return len(self.clouds)

    def get_pc_xyz(self, index: int) -> np.ndarray:
        return self.clouds[index][0]

    def get_pc_attribute(self, index: int, name: str) -> np.ndarray:
        if name != "rgb":
            raise NurecError("legacy SfM points contain only the RGB attribute")
        return self.clouds[index][1]

    def get_pc_reference_frame_id(self, index: int) -> str:
        return "world"


def legacy_sfm_readers(reader: Any) -> dict[str, Any]:
    """Adapt legacy colored SfM rays when no modern point component exists.

    Args:
        reader: Public NCore V4 component-groups reader.
    Returns:
        A point-reader adapter for every frame of the virtual SfM sensor.
    Raises:
        NurecError: Frames cannot preserve every colored source point in world.
    """
    from ncore.data.v4 import LidarSensorComponent, PosesComponent

    lidars = reader.open_component_readers(LidarSensorComponent.Reader)
    if not lidars:
        return {}
    if set(lidars) != {_SENSOR}:
        raise NurecError("legacy SfM initialization requires only virtual_lidar")
    _require_world_identity(reader.open_component_readers(PosesComponent.Reader))
    return {_SENSOR: _decode_sensor(lidars[_SENSOR])}


def _require_world_identity(poses: dict[str, Any]) -> None:
    found = False
    for group in poses.values():
        for (source, target), _ in group.get_dynamic_poses():
            if _SENSOR in (source, target):
                raise NurecError(
                    "legacy SfM initialization requires a static world pose"
                )
        for (source, target), matrix in group.get_static_poses():
            if _SENSOR not in (source, target):
                continue
            if (source, target) != (_SENSOR, "world") or not np.array_equal(
                np.asarray(matrix), np.eye(4)
            ):
                raise NurecError(
                    "legacy SfM initialization requires virtual_lidar -> world identity"
                )
            found = True
    if not found:
        raise NurecError("legacy SfM initialization is missing its world identity pose")


def _decode_sensor(sensor: Any) -> _LegacySfmPoints:
    frames = np.asarray(sensor.frames_timestamps_us)
    if (
        frames.dtype != np.uint64
        or frames.ndim != 2
        or frames.shape[1] != 2
        or not len(frames)
        or len(frames) != sensor.frames_count
        or np.any(frames[:, 0] > frames[:, 1])
        or len(np.unique(frames[:, 1])) != len(frames)
    ):
        raise NurecError("legacy SfM initialization has invalid frame timestamps")
    digest = hashlib.sha256(frames.astype("<u8").tobytes())
    clouds = [_decode_frame(sensor, frame, digest) for frame in frames]
    return _LegacySfmPoints(clouds, digest.hexdigest())


def _decode_frame(
    sensor: Any, frame: np.ndarray, digest: Any
) -> tuple[np.ndarray, np.ndarray]:
    timestamp = int(frame[1])
    directions = np.asarray(sensor.get_frame_ray_bundle_data(timestamp, "direction"))
    distances = np.asarray(
        sensor.get_frame_ray_bundle_return_data(timestamp, "distance_m", None)
    )
    valid = np.asarray(sensor.get_frame_ray_bundle_return_valid_mask(timestamp))
    count = sensor.get_frame_ray_bundle_count(timestamp)
    returns = sensor.get_frame_ray_bundle_return_count(timestamp)
    _validate_rays(directions, distances, valid, count, returns)
    timestamps = np.asarray(sensor.get_frame_ray_bundle_data(timestamp, "timestamp_us"))
    _validate_timestamps(timestamps, frame, count)
    if "rgb" not in sensor.get_frame_generic_data_names(timestamp):
        raise NurecError("legacy SfM initialization point cloud is missing RGB colors")
    rgb = np.asarray(sensor.get_frame_generic_data(timestamp, "rgb"))
    if rgb.dtype != np.uint8 or rgb.shape != (count, 3):
        raise NurecError(
            "legacy SfM initialization requires one uint8 RGB color per point"
        )
    for array in (directions, distances, valid, timestamps, rgb):
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.astype(array.dtype.newbyteorder("<")).tobytes())
    # NCore computes XYZ in the original float32 representation before export.
    xyz = directions * distances[0, :, None]
    if not np.isfinite(xyz).all():
        raise NurecError("legacy SfM initialization produced nonfinite XYZ points")
    return xyz, rgb.copy()


def _validate_rays(
    directions: np.ndarray,
    distances: np.ndarray,
    valid: np.ndarray,
    count: int,
    returns: int,
) -> None:
    if (
        count <= 0
        or returns != 1
        or valid.dtype != np.bool_
        or valid.shape != (1, count)
        or not valid.all()
    ):
        raise NurecError(
            "legacy SfM initialization requires one valid return for every ray"
        )
    if (
        directions.dtype != np.float32
        or directions.shape != (count, 3)
        or not np.isfinite(directions).all()
        or not np.all(np.abs(np.sum(directions**2, axis=1) - 1) < 1e-4)
        or distances.dtype != np.float32
        or distances.shape != (1, count)
        or not np.isfinite(distances).all()
        or np.any(distances < 0)
    ):
        raise NurecError(
            "legacy SfM initialization requires finite unit rays and distances"
        )


def _validate_timestamps(timestamps: np.ndarray, frame: np.ndarray, count: int) -> None:
    if (
        timestamps.dtype != np.uint64
        or timestamps.shape != (count,)
        or np.any(timestamps < frame[0])
        or np.any(timestamps > frame[1])
    ):
        raise NurecError(
            "legacy SfM initialization point timestamps are outside their frame"
        )
