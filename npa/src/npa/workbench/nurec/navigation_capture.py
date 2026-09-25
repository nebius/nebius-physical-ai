"""Validate calibrated metric RGB-D captures before native surface reconstruction."""

from __future__ import annotations

import json
import math
from pathlib import Path
import re

import numpy as np

from npa.workbench.nurec.navigation_assets import contained_file, sha256
from npa.workbench.nurec.navigation_geometry import rigid_transform


def _positive(value, name: str) -> float:
    if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _calibration(capture: dict) -> None:
    if capture.get("world") != {"meters_per_unit": 1, "up_axis": "Z"}:
        raise ValueError("capture requires explicit metric Z-up world coordinates")
    if capture.get("camera_convention") != "optical_x_right_y_down_z_forward":
        raise ValueError("capture requires optical camera coordinates")
    intrinsic = capture["intrinsics"]
    for key in ("width", "height"):
        if type(intrinsic.get(key)) is not int or intrinsic[key] <= 0:
            raise ValueError("camera dimensions must be positive integers")
    for key in ("fx", "fy"):
        _positive(intrinsic.get(key), key)
    for key, bound in (("cx", "width"), ("cy", "height")):
        value = intrinsic.get(key)
        if type(value) not in (int, float) or not 0 <= value < intrinsic[bound]:
            raise ValueError("principal point must lie within the image")
    if intrinsic.get("distortion") != [0, 0, 0, 0, 0]:
        raise ValueError("RGB and registered depth must already be undistorted")


def _parameters(capture: dict) -> None:
    for key in ("depth_units_per_meter", "depth_max_m", "voxel_size_m", "sdf_trunc_m"):
        _positive(capture.get(key), key)
    if capture["sdf_trunc_m"] <= capture["voxel_size_m"]:
        raise ValueError("TSDF truncation must exceed voxel size")
    quality = capture["validation"]
    stride = quality.get("pixel_stride")
    if type(stride) is not int or not 1 <= stride < min(
        capture["intrinsics"]["width"], capture["intrinsics"]["height"]
    ):
        raise ValueError("validation pixel_stride must sample the image")
    for key in ("min_coverage", "min_inlier_fraction"):
        if not 0 < _positive(quality.get(key), key) <= 1:
            raise ValueError("validation fractions must lie in (0, 1]")
    for key in ("max_mean_error_m", "distance_tolerance_m"):
        _positive(quality.get(key), key)


def _frame(root: Path, frame: dict) -> None:
    if not isinstance(frame, dict) or not re.fullmatch(
        r"[A-Za-z0-9_-]+", str(frame.get("id", ""))
    ):
        raise ValueError("each frame requires a plain unique id")
    if frame.get("split") not in {"integration", "validation"}:
        raise ValueError("each frame requires an integration or validation split")
    timestamp = frame.get("timestamp_s")
    if type(timestamp) not in (float, int) or not math.isfinite(timestamp):
        raise ValueError("frame timestamp must be finite")
    rigid_transform(np.asarray(frame.get("camera_to_world"), dtype=float).T)
    for kind in ("rgb", "depth"):
        path = contained_file(root, frame.get(kind, ""))
        digest = frame.get(f"{kind}_sha256", "")
        if not re.fullmatch("[0-9a-f]{64}", str(digest)) or sha256(path) != digest:
            raise ValueError(f"{kind} input hash differs from capture manifest")


def read_capture(root: Path) -> dict:
    """Verify calibration, disjoint split membership, poses, and every input hash.

    Args:
        root: Contained input directory with capture.json and image pairs.
    Returns:
        Validated capture contract using column-vector optical camera poses.
    Raises:
        ValueError: Inputs, calibration, scale, or split membership are invalid.
        OSError: A manifest or image cannot be read.
        KeyError: A required calibration or validation field is absent.
    """
    capture = json.loads(contained_file(root, "capture.json").read_text())
    if (
        not isinstance(capture, dict)
        or capture.get("schema") != "npa.navigation.rgbd_capture.v1"
    ):
        raise ValueError("capture.json requires npa.navigation.rgbd_capture.v1")
    _calibration(capture)
    _parameters(capture)
    frames = capture.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("capture requires image frames")
    for frame in frames:
        _frame(root, frame)
    _splits(frames)
    return capture


def _splits(frames: list[dict]) -> None:
    if len({frame["id"] for frame in frames}) != len(frames):
        raise ValueError("frame ids must be unique")
    groups = {}
    for split in ("integration", "validation"):
        selected = [frame for frame in frames if frame["split"] == split]
        if not selected:
            raise ValueError("integration and held-out validation frames are required")
        groups[split] = {
            (frame["rgb_sha256"], frame["depth_sha256"]) for frame in selected
        }
    if groups["integration"] & groups["validation"]:
        raise ValueError("identical image pairs cannot cross the held-out split")


def read_images(root: Path, capture: dict, frame: dict):
    """Decode a calibrated RGB and axial-depth pair without implicit conversion.

    Args:
        root: Contained capture directory.
        capture: Validated capture calibration.
        frame: One frame record with verified input hashes.
    Returns:
        RGB uint8 and registered depth uint16 NumPy arrays.
    Raises:
        ValueError: Images have incompatible dimensions, channels, or data types.
        OSError: Images cannot be decoded.
        ImportError: Pillow is unavailable.
    """
    from PIL import Image

    with Image.open(contained_file(root, frame["rgb"])) as image:
        rgb = np.asarray(image).copy()
    with Image.open(contained_file(root, frame["depth"])) as image:
        depth = np.asarray(image).copy()
    intrinsic = capture["intrinsics"]
    shape = (intrinsic["height"], intrinsic["width"])
    if rgb.dtype != np.uint8 or rgb.shape != (*shape, 3):
        raise ValueError("RGB must be calibrated uint8 three-channel imagery")
    if depth.dtype != np.uint16 or depth.shape != shape:
        raise ValueError(
            "depth must be calibrated registered uint16 axial-depth imagery"
        )
    if not np.any(
        (depth > 0)
        & (depth / capture["depth_units_per_meter"] < capture["depth_max_m"])
    ):
        raise ValueError("frame has no valid metric depth observations")
    return rgb, depth
