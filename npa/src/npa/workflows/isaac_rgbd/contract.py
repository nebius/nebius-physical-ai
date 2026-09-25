"""Define strict calibrated-rig requests and contained, hash-verified file contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re

from .geometry import _array, _intrinsics, _transform

REQUEST_SCHEMA = "npa.isaac.rgbd.request.v1"
DATASET_SCHEMA = "npa.isaac.rgbd.dataset.v1"
DEPTH_CONVENTION = "optical_z_m_zero_invalid"
CAMERA_FRAME = "optical_x_right_y_down_z_forward"
ISAAC_SIM_VERSION = "6.0.1.0"


def _keys(value, fields, name):
    if not isinstance(value, dict) or set(value) != set(fields.split()):
        raise ValueError(f"{name} requires exactly these fields: {fields}")


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _relative(name):
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_./-]+", name):
        raise ValueError("asset path must use plain relative POSIX components")
    path = PurePosixPath(name)
    if path.is_absolute() or path.as_posix() != name or ".." in path.parts:
        raise ValueError("asset path must be normalized and contained")
    if not path.parts or any(part.startswith(".") for part in path.parts):
        raise ValueError("asset path cannot contain hidden or dot components")
    return path


def _contained(root, name):
    base = Path(root).resolve()
    path = base / _relative(name)
    if not path.resolve().is_relative_to(base):
        raise ValueError("asset path escapes its bundle")
    if any(
        parent.is_symlink()
        for parent in [path, *path.parents]
        if parent != base and parent.is_relative_to(base)
    ):
        raise ValueError("asset path cannot traverse symlinks")
    return path


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value):
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()


def _read_json(path):
    def reject_constant(_value):
        raise ValueError("JSON contains NaN or Infinity")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("JSON contains duplicate keys")
            result[key] = value
        return result

    return json.loads(
        Path(path).read_text(),
        parse_constant=reject_constant,
        object_pairs_hook=unique_object,
    )


def _file_hashes(files):
    if not isinstance(files, dict) or not files:
        raise ValueError("files must declare a nonempty asset-to-SHA256 mapping")
    for name, digest in files.items():
        _relative(name)
        if not isinstance(digest, str) or not re.fullmatch("[a-f0-9]{64}", digest):
            raise ValueError("every file requires a lowercase SHA256")
    folded = [name.casefold() for name in files]
    if len(set(folded)) != len(folded):
        raise ValueError("asset paths must not collide on case-insensitive filesystems")
    for name in files:
        if any(str(parent) in files for parent in PurePosixPath(name).parents):
            raise ValueError("asset paths contain a file/directory collision")


def _camera(camera):
    _keys(camera, "id width height intrinsics T_rig_camera depth_range_m", "camera")
    if not isinstance(camera["id"], str) or not re.fullmatch(
        "[a-z][a-z0-9_]*", camera["id"]
    ):
        raise ValueError("camera id must be a lowercase identifier")
    width = _integer(camera["width"], "width", 1)
    height = _integer(camera["height"], "height", 1)
    _intrinsics(camera["intrinsics"], width, height)
    _transform(camera["T_rig_camera"], "T_rig_camera")
    near, far = _array(camera["depth_range_m"], (2,), "depth_range_m")
    if not 0 < near < far:
        raise ValueError("depth_range_m requires 0 < near < far")


def _trajectory(samples):
    if not isinstance(samples, list) or not samples:
        raise ValueError("trajectory must contain at least one sampled pose")
    previous = -1
    for sample in samples:
        _keys(sample, "timestamp_ns T_world_rig", "trajectory sample")
        timestamp = _integer(sample["timestamp_ns"], "timestamp_ns")
        if timestamp <= previous or timestamp > 2**53:
            raise ValueError(
                "timestamps must strictly increase within exact float64 integer range"
            )
        previous = timestamp
        _transform(sample["T_world_rig"], "T_world_rig")


def validate_request(request):
    """Reject unsupported scenes, malformed calibration, and invalid sampled trajectories.

    Args:
        request: Decoded npa.isaac.rgbd.request.v1 object.

    Returns:
        The validated request, without normalization or reordering.

    Raises:
        ValueError: A field is missing, unsupported, non-finite, or inconsistent.
    """
    _keys(request, "schema scene files cameras trajectory pointcloud", "request")
    if request["schema"] != REQUEST_SCHEMA or type(request["pointcloud"]) is not bool:
        raise ValueError("unsupported request schema or non-boolean pointcloud")
    _file_hashes(request["files"])
    scene = str(_relative(request["scene"]))
    if scene not in request["files"] or Path(scene).suffix not in {
        ".usd",
        ".usda",
        ".usdc",
        ".usdz",
    }:
        raise ValueError("scene must name a declared USD layer or USDZ package")
    if "request.json" in request["files"]:
        raise ValueError("request.json is reserved for the request manifest")
    cameras = request["cameras"]
    if not isinstance(cameras, list) or not cameras:
        raise ValueError("cameras must contain at least one calibrated sensor")
    for camera in cameras:
        _camera(camera)
    if len({camera["id"] for camera in cameras}) != len(cameras):
        raise ValueError("camera ids must be unique")
    _trajectory(request["trajectory"])
    return request
