"""Fuse synchronized world clouds with exact source camera and pixel provenance."""

from __future__ import annotations

import numpy as np
from PIL import Image

from .contract import _contained, _keys
from .geometry import backproject

_METHOD = "concatenate_world_backprojections"
_ARRAYS = {"xyz_world_m", "rgb", "camera_index", "pixel_index"}


def _view_points(root, camera, view):
    artifacts = view["artifacts"]
    depth = np.load(_contained(root, artifacts["depth"]), allow_pickle=False)
    with Image.open(_contained(root, artifacts["rgb"])) as image:
        rgb = np.asarray(image).copy()
    points, colors = backproject(
        depth, rgb, camera["intrinsics"], view["T_world_camera"]
    )
    pixels = np.flatnonzero(depth.reshape(-1) > 0).astype(np.uint64)
    return points, colors, pixels


def _write_fused_cloud(root, index, cameras, views):
    arrays = {name: [] for name in _ARRAYS}
    for camera_index, (camera, view) in enumerate(zip(cameras, views, strict=True)):
        points, colors, pixels = _view_points(root, camera, view)
        arrays["xyz_world_m"].append(points)
        arrays["rgb"].append(colors)
        arrays["pixel_index"].append(pixels)
        arrays["camera_index"].append(
            np.full(len(points), camera_index, dtype=np.uint32)
        )
    path = root / "frames" / f"{index:06d}" / "fused.npz"
    np.savez(path, **{name: np.concatenate(parts) for name, parts in arrays.items()})
    return {
        "path": path.relative_to(root).as_posix(),
        "method": _METHOD,
        "camera_ids": [camera["id"] for camera in cameras],
    }


def _validate_segment(cloud, offset, camera_index, expected):
    points, colors, pixels = expected
    segment = slice(offset, offset + len(points))
    actual = cloud["xyz_world_m"][segment]
    if actual.shape != points.shape or not np.allclose(
        actual, points, atol=1e-7, rtol=1e-7
    ):
        raise ValueError("fused cloud is not the source RGB-D world backprojection")
    if not np.array_equal(cloud["rgb"][segment], colors):
        raise ValueError("fused cloud colors differ from source RGB")
    if not np.array_equal(cloud["pixel_index"][segment], pixels) or not np.all(
        cloud["camera_index"][segment] == camera_index
    ):
        raise ValueError("fused cloud source camera/pixel provenance differs")
    return offset + len(points)


def _cloud_arrays(archive):
    if set(archive.files) != _ARRAYS:
        raise ValueError("fused cloud requires points, colors and source camera/pixel")
    cloud = {name: archive[name] for name in _ARRAYS}
    count = len(cloud["xyz_world_m"])
    expected = {
        "xyz_world_m": ((count, 3), np.float64),
        "rgb": ((count, 3), np.uint8),
        "camera_index": ((count,), np.uint32),
        "pixel_index": ((count,), np.uint64),
    }
    for name, (shape, dtype) in expected.items():
        if cloud[name].shape != shape or cloud[name].dtype != dtype:
            raise ValueError("fused cloud array shape or dtype differs from contract")
    return cloud


def _validate_fused_cloud(root, frame, cameras, manifest):
    fused = frame["fused_cloud"]
    if not manifest["request"]["pointcloud"]:
        if fused is not None:
            raise ValueError("fused cloud must be null when pointcloud is false")
        return 0
    _keys(fused, "path method camera_ids", "fused cloud")
    if fused["method"] != _METHOD or fused["camera_ids"] != [
        camera["id"] for camera in cameras
    ]:
        raise ValueError("fused cloud method or source camera order differs")
    if fused["path"] not in manifest["files"]:
        raise ValueError("fused cloud references an unhashed dataset file")
    with np.load(_contained(root, fused["path"]), allow_pickle=False) as archive:
        cloud = _cloud_arrays(archive)
    offset = 0
    for index, (camera, view) in enumerate(zip(cameras, frame["views"], strict=True)):
        expected = _view_points(root, camera, view)
        offset = _validate_segment(cloud, offset, index, expected)
    if offset != len(cloud["xyz_world_m"]):
        raise ValueError("fused cloud contains missing or additional source points")
    return offset
