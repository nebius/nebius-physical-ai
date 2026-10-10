"""Bind independent photographs to exact knots consumed by NRE's virtual rig."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from npa.workbench.nurec.colmap_timeline import FRAME_POSE_MODE
from npa.workbench.nurec.nurec import NurecConfig, NurecError, read_rig_sidecar

_OVERRIDE = "dataset.frame_generic_data_pose_overwrite"


def _camera_frame_poses(reader: Any) -> set[str]:
    from ncore.data.v4 import CameraSensorComponent, SequenceLoaderV4

    loader = SequenceLoaderV4(
        reader,
        poses_component_group_name="default",
        masks_component_group_name=None,
        cuboids_component_group_name=None,
    )
    cameras = reader.open_component_readers(CameraSensorComponent.Reader)
    for name, camera in cameras.items():
        sensor = loader.get_camera_sensor(name)
        for index, timestamp in enumerate(camera.frames_timestamps_us[:, 1]):
            try:
                actual = np.asarray(
                    camera.get_frame_generic_data(int(timestamp), "T_sensor_worlds")
                )
            except KeyError as exc:
                raise NurecError("required frame world pose data is missing") from exc
            expected = sensor.get_frames_T_sensor_target(
                "world", index, frame_timepoint=None
            )
            _require_frame_poses(actual, expected)
    if not cameras:
        raise NurecError("frame pose input has no cameras")
    return set(cameras)


def _require_frame_poses(actual: np.ndarray, expected: np.ndarray) -> None:
    if (
        actual.shape != (2, 4, 4)
        or not np.issubdtype(actual.dtype, np.floating)
        or not np.isfinite(actual).all()
        or not np.allclose(actual, expected, rtol=1e-5, atol=1e-5)
    ):
        raise NurecError("frame world poses differ from the original camera trajectory")


def _write_static_calibrations(writer: Any, cameras: set[str]) -> None:
    # These are virtual calibrations, not a physical multi-camera rig. Each
    # photograph has its own exact world-pose knot on the shared virtual timeline.
    for camera in sorted(cameras):
        writer.store_static_pose(camera, "rig", np.eye(4, dtype=np.float32))


def _validate_static_calibrations(reader: Any, cameras: set[str]) -> None:
    from ncore.data.v4 import PosesComponent

    poses = reader.open_component_readers(PosesComponent.Reader)["npa_rig"]
    static = dict(poses.get_static_poses())
    if set(static) != {(camera, "rig") for camera in cameras} or any(
        not np.array_equal(matrix, np.eye(4)) for matrix in static.values()
    ):
        raise NurecError(
            "virtual camera calibrations differ from the frame pose contract"
        )
    dynamic = dict(poses.get_dynamic_poses())
    if set(dynamic) != {("rig", "world")}:
        raise NurecError("virtual rig group contains an unexpected dynamic camera edge")
    _require_reference_coverage(reader, dynamic[("rig", "world")][1])
    _require_rig_frame_knots(reader, *dynamic[("rig", "world")])


def _require_rig_frame_knots(reader: Any, poses: np.ndarray, times: np.ndarray) -> None:
    from ncore.data.v4 import CameraSensorComponent

    for camera in reader.open_component_readers(CameraSensorComponent.Reader).values():
        for timestamp in camera.frames_timestamps_us:
            if timestamp[0] != timestamp[1]:
                raise NurecError("virtual photographic frames require global shutter")
            indices = np.flatnonzero(times == timestamp[1])
            if len(indices) != 1:
                raise NurecError(
                    "virtual rig must contain each frame's exact time knot"
                )
            actual = camera.get_frame_generic_data(int(timestamp[1]), "T_sensor_worlds")
            _require_frame_poses(
                np.asarray(actual), np.repeat(poses[indices[0]][None], 2, axis=0)
            )


def _require_reference_coverage(reader: Any, timestamps: np.ndarray) -> None:
    from ncore.data.v4 import CameraSensorComponent

    for camera in reader.open_component_readers(CameraSensorComponent.Reader).values():
        frame_times = camera.frames_timestamps_us
        if (
            not frame_times.size
            or frame_times.min() < timestamps.min()
            or frame_times.max() > timestamps.max()
        ):
            raise NurecError("reference trajectory does not cover every camera frame")


def _require_native_overrides(config: NurecConfig) -> None:
    for override in config.extra_overrides:
        key, separator, value = override.partition("=")
        key = key.strip()
        if key.lstrip("+~") == "dataset":
            raise NurecError("native frame pose dataset cannot be replaced")
        if key.lstrip("+~") == _OVERRIDE and (
            key.startswith("~") or not separator or value != "true"
        ):
            raise NurecError("native frame pose override cannot be disabled")
        if key.lstrip("+~") == "dataset.poses_component_group" and (
            key.startswith("~") or not separator or value != "npa_rig"
        ):
            raise NurecError("native frame pose group cannot be replaced")


def plan_frame_poses(config: NurecConfig, ncore_json: str) -> NurecConfig:
    """Validate the shared rig and per-frame paths against the same exact poses.

    Args:
        config: Resolved reconstruction configuration.
        ncore_json: Local sequence metadata with its conversion sidecar.
    Returns:
        Configuration selecting the validated virtual rig and frame world poses.
    Raises:
        NurecError: The declared pose contract or an explicit override conflicts.
    """
    sidecar = read_rig_sidecar(ncore_json)
    mode = sidecar.get("frame_pose_mode")
    if mode is None:
        return config
    if mode != FRAME_POSE_MODE or config.poses_component_group not in ("", "npa_rig"):
        raise NurecError("unsupported or conflicting NCore frame pose contract")
    _require_native_overrides(config)
    from ncore.data.v4 import SequenceComponentGroupsReader
    from upath import UPath

    reader = SequenceComponentGroupsReader([UPath(Path(ncore_json))])
    _validate_static_calibrations(reader, _camera_frame_poses(reader))
    return replace(
        config,
        poses_component_group="npa_rig",
        extra_overrides=(f"{_OVERRIDE}=true", *config.extra_overrides),
    )
