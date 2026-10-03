"""Validate rigid transforms and backproject axial metric depth into world space."""

from __future__ import annotations

import numpy as np


def _array(value, shape, name):
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a numeric array of shape {shape}")
    if any(
        isinstance(item, (bool, np.bool_))
        for item in np.asarray(value, dtype=object).flat
    ):
        raise ValueError(f"{name} must not contain booleans")
    array = array.astype(np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite numbers")
    return array


def _transform(value, name="transform"):
    matrix = _array(value, (4, 4), name)
    rotation = matrix[:3, :3]
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-7, rtol=0):
        raise ValueError(f"{name} must have homogeneous last row [0,0,0,1]")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0):
        raise ValueError(f"{name} rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1, atol=1e-6, rtol=0):
        raise ValueError(f"{name} rotation must be right handed")
    return matrix


def _intrinsics(value, width, height):
    matrix = _array(value, (3, 3), "intrinsics")
    if not np.array_equal(matrix[2], [0, 0, 1]):
        raise ValueError("intrinsics must have last row [0,0,1]")
    if matrix[0, 1] != 0 or matrix[1, 0] != 0:
        raise ValueError("only zero-skew pinhole intrinsics are supported")
    if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        raise ValueError("intrinsics focal lengths must be positive")
    if not (0 <= matrix[0, 2] < width and 0 <= matrix[1, 2] < height):
        raise ValueError("intrinsics principal point must lie inside the image")
    return matrix


def _masked_depth(raw, camera):
    depth = np.asarray(raw)
    if depth.shape != (camera["height"], camera["width"]) or depth.dtype.kind != "f":
        raise ValueError("depth must be a floating HxW distance_to_image_plane image")
    near, far = camera["depth_range_m"]
    valid = np.isfinite(depth) & (depth >= near) & (depth <= far)
    if not valid.any():
        raise ValueError(f"camera {camera['id']} has no valid metric depth pixels")
    return np.where(valid, depth, 0).astype(np.float32), valid


def backproject(depth, rgb, intrinsics, world_from_camera):
    """Backproject valid optical-Z pixels to colored world-frame points in meters.

    Args:
        depth: HxW metric axial depth; zero denotes invalid data.
        rgb: Matching HxWx3 uint8 colors.
        intrinsics: Zero-skew 3x3 calibration; pixel centers have integer u,v.
        world_from_camera: Rigid 4x4 column-vector optical-to-world transform.

    Returns:
        Tuple of Nx3 float64 world points and Nx3 uint8 colors, in row-major order.

    Raises:
        ValueError: Calibration, pose, depth, or color data violate the contract.
    """
    depth, rgb = np.asarray(depth), np.asarray(rgb)
    if depth.ndim != 2 or depth.dtype.kind != "f" or not np.isfinite(depth).all():
        raise ValueError("depth must be finite floating HxW with zero invalid pixels")
    if (depth < 0).any() or rgb.shape != (*depth.shape, 3) or rgb.dtype != np.uint8:
        raise ValueError("invalid depth or aligned uint8 RGB data")
    height, width = depth.shape
    calibration = _intrinsics(intrinsics, width, height)
    transform = _transform(world_from_camera)
    rows, columns = np.nonzero(depth > 0)
    if not len(rows):
        raise ValueError("cannot backproject an empty depth image")
    rays = np.stack([columns, rows, np.ones_like(rows)], axis=1)
    camera_points = (rays @ np.linalg.inv(calibration).T) * depth[rows, columns, None]
    points = camera_points @ transform[:3, :3].T + transform[:3, 3]
    if not np.isfinite(points).all():
        raise ValueError("backprojection produced non-finite points")
    return points, rgb[rows, columns].copy()


def _usd_camera_parameters(camera):
    width, height = camera["width"], camera["height"]
    matrix = _intrinsics(camera["intrinsics"], width, height)
    focal = 24.0
    return {
        "focal": focal,
        "aperture": [focal * width / matrix[0, 0], focal * height / matrix[1, 1]],
        # USD film coordinates use image edges; our pixel centers start at (0,0).
        "offset": [
            (width / 2 - matrix[0, 2] - 0.5) * focal / matrix[0, 0],
            (matrix[1, 2] + 0.5 - height / 2) * focal / matrix[1, 1],
        ],
    }


def _optical_to_usd(transform):
    return _transform(transform) @ np.diag([1.0, -1.0, -1.0, 1.0])
