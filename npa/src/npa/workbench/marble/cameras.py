"""Produce explicit OpenCV camera poses for a short exploratory world sweep."""

import numpy as np


def camera_sweep(count, width, height):
    """Build a rotating camera path and calibrated pinhole intrinsics.

    Args: Frame count and output dimensions in pixels.
    Returns: Camera-to-world matrices and one intrinsic matrix.
    Raises: ValueError for nonpositive dimensions.
    """
    if min(count, width, height) <= 0:
        raise ValueError("Camera dimensions and count must be positive")
    poses = []
    for angle in np.linspace(0, 2 * np.pi, count, endpoint=False):
        forward = np.array([np.sin(angle), 0.0, -np.cos(angle)])
        right = np.cross(forward, [0, 1, 0])
        down = np.cross(forward, right)
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = np.stack([right, down, forward], axis=1)
        pose[:3, 3] = [0.25 * np.sin(angle), 0.15, 0.25 * np.cos(angle)]
        poses.append(pose)
    focal = width / (2 * np.tan(np.deg2rad(75) / 2))
    intrinsic = np.array(
        [[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], dtype=np.float32
    )
    return np.stack(poses), intrinsic


def transform_splats(cloud, transform):
    """Apply uniform scale, axis reflection, and translation to decoded splats.

    Args: Niantic GaussianCloud and explicit diagonal world transform.
    Returns: Means, log scales, XYZW rotations, opacity logits, and SH DC colors.
    Raises: ValueError when scaling is not uniform in magnitude.
    """
    scale = np.asarray(transform["scale"], dtype=np.float32)
    if not np.allclose(np.abs(scale), abs(scale[0])) or scale[0] == 0:
        raise ValueError("Splat transform requires nonzero uniform scale magnitude")
    means = cloud.positions.reshape(-1, 3) * scale + transform["translation"]
    logs = cloud.scales.reshape(-1, 3) + np.log(abs(scale[0]))
    rotations = cloud.rotations.reshape(-1, 4).copy()
    signs = np.sign(scale)
    rotations[:, :3] *= np.prod(signs) * signs
    return means, logs, rotations, cloud.alphas, cloud.colors.reshape(-1, 3)
