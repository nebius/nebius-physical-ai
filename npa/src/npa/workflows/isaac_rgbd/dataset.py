"""Write and independently decode synchronized RGB-D datasets and colored points."""

from __future__ import annotations

from fractions import Fraction
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from .contract import (
    CAMERA_FRAME,
    DATASET_SCHEMA,
    DEPTH_CONVENTION,
    ISAAC_SIM_VERSION,
    _contained,
    _file_hashes,
    _integer,
    _json_bytes,
    _keys,
    _read_json,
    _sha256,
    validate_request,
)
from .geometry import (
    _array,
    _masked_depth,
    _optical_to_usd,
    _usd_camera_parameters,
    backproject,
)


def _reference_time(value):
    _keys(value, "numerator denominator", "render_reference_time")
    numerator = _integer(value["numerator"], "reference time numerator")
    denominator = _integer(value["denominator"], "reference time denominator", 1)
    return Fraction(numerator, denominator)


def _check_render_metadata(metadata, camera, world_from_camera):
    _keys(
        metadata,
        "aperture offset focal view_transform resolution",
        "render calibration",
    )
    expected = _usd_camera_parameters(camera)
    for name in ("aperture", "offset", "focal"):
        actual = _array(metadata[name], np.shape(expected[name]), f"render {name}")
        if not np.allclose(actual, expected[name], atol=1e-5, rtol=1e-5):
            raise ValueError(f"rendered {name} differs from requested calibration")
    if metadata["resolution"] != [camera["width"], camera["height"]]:
        raise ValueError("rendered resolution differs from calibration")
    view = _array(metadata["view_transform"], (4, 4), "render view transform")
    # CameraParams uses Gf row-vector world-to-view; USD cameras look down -Z.
    expected_view = np.linalg.inv(_optical_to_usd(world_from_camera)).T
    if not np.allclose(view, expected_view, atol=1e-5, rtol=1e-5):
        raise ValueError("rendered camera pose differs from the sampled rig pose")


def _save_array(path, value):
    with path.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)


def _write_view(root, camera, sample, frame_index, snapshot, pointcloud):
    rgb = np.asarray(snapshot["rgb"])
    shape = (camera["height"], camera["width"])
    if rgb.dtype != np.uint8 or rgb.shape not in {(*shape, 3), (*shape, 4)}:
        raise ValueError("renderer must return a decoded HxWx3/4 uint8 RGB image")
    rgb = rgb[:, :, :3].copy()
    depth, mask = _masked_depth(snapshot["depth"], camera)
    pose = np.asarray(sample["T_world_rig"]) @ np.asarray(camera["T_rig_camera"])
    _check_render_metadata(snapshot["render_calibration"], camera, pose)
    directory = root / "frames" / f"{frame_index:06d}" / camera["id"]
    directory.mkdir(parents=True)
    Image.fromarray(rgb).save(directory / "rgb.png")
    _save_array(directory / "depth.npy", depth)
    _save_array(directory / "mask.npy", mask)
    names = ["rgb.png", "depth.npy", "mask.npy"]
    if pointcloud:
        points, colors = backproject(depth, rgb, camera["intrinsics"], pose)
        np.savez(directory / "points.npz", xyz_world_m=points, rgb=colors)
        names.append("points.npz")
    return {
        "camera_id": camera["id"],
        "T_world_camera": pose.tolist(),
        "render_reference_time": snapshot["render_reference_time"],
        "render_calibration": snapshot["render_calibration"],
        "artifacts": {
            Path(name).stem: (directory / name).relative_to(root).as_posix()
            for name in names
        },
    }


def _write_frame(root, request, index, snapshots, simulation_time_s):
    cameras = sorted(request["cameras"], key=lambda camera: camera["id"])
    if set(snapshots) != {camera["id"] for camera in cameras}:
        raise ValueError("renderer did not return every camera exactly once")
    sample = request["trajectory"][index]
    return {
        "index": index,
        "timestamp_ns": sample["timestamp_ns"],
        "simulation_time_s": simulation_time_s,
        "T_world_rig": sample["T_world_rig"],
        "views": [
            _write_view(
                root,
                camera,
                sample,
                index,
                snapshots[camera["id"]],
                request["pointcloud"],
            )
            for camera in cameras
        ],
    }


def _manifest(root, request, frames, provenance):
    files = {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    return {
        "schema": DATASET_SCHEMA,
        "engine": "isaac-sim-replicator",
        "isaac_sim_version": ISAAC_SIM_VERSION,
        "depth_convention": DEPTH_CONVENTION,
        "camera_frame": CAMERA_FRAME,
        "world_frame": "usd_z_up_meters",
        "request": request,
        "provenance": provenance,
        "frames": frames,
        "files": files,
    }


def _verify_files(root, files):
    _file_hashes(files)
    for name, digest in files.items():
        path = _contained(root, name)
        if not path.is_file() or _sha256(path) != digest:
            raise ValueError(f"missing or SHA256-mismatched dataset file: {name}")


def _decode_view(root, camera, artifacts, files, pointcloud):
    expected = "rgb depth mask points" if pointcloud else "rgb depth mask"
    _keys(artifacts, expected, "view artifacts")
    if any(name not in files for name in artifacts.values()):
        raise ValueError("view references an unhashed dataset file")
    with Image.open(_contained(root, artifacts["rgb"])) as image:
        if image.format != "PNG" or image.mode != "RGB":
            raise ValueError("RGB must decode as an RGB PNG")
        rgb = np.asarray(image).copy()
    depth = np.load(_contained(root, artifacts["depth"]), allow_pickle=False)
    mask = np.load(_contained(root, artifacts["mask"]), allow_pickle=False)
    shape = (camera["height"], camera["width"])
    if rgb.shape != (*shape, 3) or depth.shape != shape or mask.shape != shape:
        raise ValueError("RGB/depth/mask resolution mismatch")
    if depth.dtype != np.float32 or mask.dtype != np.bool_:
        raise ValueError("depth must be float32 and mask must be bool")
    clean, valid = _masked_depth(depth, camera)
    if not np.array_equal(clean, depth) or not np.array_equal(valid, mask):
        raise ValueError("invalid-depth pixels must be zero and match the mask")
    return rgb, depth, mask


def _validate_points(root, name, depth, rgb, camera, pose):
    expected, colors = backproject(depth, rgb, camera["intrinsics"], pose)
    with np.load(_contained(root, name), allow_pickle=False) as cloud:
        if set(cloud.files) != {"xyz_world_m", "rgb"}:
            raise ValueError("point cloud requires xyz_world_m and rgb arrays")
        points, actual_colors = cloud["xyz_world_m"], cloud["rgb"]
        if points.dtype.kind != "f" or points.shape != expected.shape:
            raise ValueError("point count does not match valid depth pixels")
        if not np.isfinite(points).all() or not np.allclose(
            points, expected, atol=1e-7, rtol=1e-7
        ):
            raise ValueError("point cloud is not the world-frame backprojection")
        if actual_colors.dtype != np.uint8 or not np.array_equal(actual_colors, colors):
            raise ValueError("point cloud colors do not match RGB")


def _validate_view(root, view, camera, sample, manifest):
    _keys(
        view,
        "camera_id T_world_camera render_reference_time render_calibration artifacts",
        "view",
    )
    pose = np.asarray(sample["T_world_rig"]) @ np.asarray(camera["T_rig_camera"])
    actual_pose = _array(view["T_world_camera"], (4, 4), "T_world_camera")
    if not np.allclose(actual_pose, pose, atol=1e-7, rtol=0):
        raise ValueError("camera extrinsics do not compose with the rig trajectory")
    _check_render_metadata(view["render_calibration"], camera, pose)
    pointcloud = manifest["request"]["pointcloud"]
    rgb, depth, mask = _decode_view(
        root, camera, view["artifacts"], manifest["files"], pointcloud
    )
    if pointcloud:
        _validate_points(root, view["artifacts"]["points"], depth, rgb, camera, pose)
    return int(mask.sum())


def _validate_frame(root, frame, index, manifest):
    _keys(frame, "index timestamp_ns simulation_time_s T_world_rig views", "frame")
    request = manifest["request"]
    sample = request["trajectory"][index]
    if (
        _integer(frame["index"], "frame index") != index
        or frame["timestamp_ns"] != sample["timestamp_ns"]
    ):
        raise ValueError("frame index/timestamp is not aligned with trajectory")
    time = _array(frame["simulation_time_s"], (), "simulation_time_s")
    if abs(float(time) - sample["timestamp_ns"] / 1e9) > 1e-9:
        raise ValueError("rendered timeline does not match trajectory time")
    if not np.array_equal(frame["T_world_rig"], sample["T_world_rig"]):
        raise ValueError("frame rig pose differs from trajectory")
    cameras = sorted(request["cameras"], key=lambda camera: camera["id"])
    if [view["camera_id"] for view in frame["views"]] != [
        camera["id"] for camera in cameras
    ]:
        raise ValueError("cross-camera coverage/order mismatch")
    times = [_reference_time(view["render_reference_time"]) for view in frame["views"]]
    if len(set(times)) != 1:
        raise ValueError("cross-camera render reference times are not synchronized")
    pixels = sum(
        _validate_view(root, view, camera, sample, manifest)
        for camera, view in zip(cameras, frame["views"], strict=True)
    )
    return pixels, times[0]


def _validate_header(manifest):
    _keys(
        manifest,
        "schema engine isaac_sim_version depth_convention camera_frame world_frame request provenance frames files",
        "dataset",
    )
    expected = {
        "schema": DATASET_SCHEMA,
        "engine": "isaac-sim-replicator",
        "isaac_sim_version": ISAAC_SIM_VERSION,
        "depth_convention": DEPTH_CONVENTION,
        "camera_frame": CAMERA_FRAME,
        "world_frame": "usd_z_up_meters",
    }
    if any(manifest[name] != value for name, value in expected.items()):
        raise ValueError(
            "unsupported dataset engine, version, or coordinate convention"
        )
    validate_request(manifest["request"])
    _validate_provenance(manifest)
    if len(manifest["frames"]) != len(manifest["request"]["trajectory"]):
        raise ValueError("dataset must contain every trajectory sample")


def _validate_provenance(manifest):
    provenance = manifest["provenance"]
    _keys(
        provenance,
        "request_sha256 input_manifest_sha256 scope replicator_version",
        "provenance",
    )
    if (
        provenance["scope"] not in {"procedural-room", "supplied-usd"}
        or not provenance["replicator_version"]
    ):
        raise ValueError("missing renderer provenance")
    _file_hashes({"request.json": provenance["request_sha256"]})
    _file_hashes({"input.json": provenance["input_manifest_sha256"]})
    if (
        provenance["request_sha256"]
        != hashlib.sha256(_json_bytes(manifest["request"])).hexdigest()
    ):
        raise ValueError(
            "request provenance hash does not match calibration/trajectory"
        )


def validate_dataset(root, manifest=None):
    """Decode all RGB-D bytes and verify hashes, geometry, synchronization, and coverage.

    Args:
        root: Directory holding manifest.json and all declared relative artifacts.
        manifest: Optional decoded manifest for validation before publication.

    Returns:
        A data-validation report with measured frame, view, and valid-pixel counts.

    Raises:
        ValueError: Data are missing, corrupt, misaligned, or geometrically inconsistent.
        OSError: An artifact cannot be read or decoded.
    """
    root = Path(root)
    manifest = _read_json(root / "manifest.json") if manifest is None else manifest
    _validate_header(manifest)
    _verify_files(root, manifest["files"])
    pixels, previous, used = 0, None, []
    for index, frame in enumerate(manifest["frames"]):
        count, reference = _validate_frame(root, frame, index, manifest)
        if previous is not None and reference <= previous:
            raise ValueError("render reference time did not advance; stale capture")
        pixels, previous = pixels + count, reference
        used.extend(
            name for view in frame["views"] for name in view["artifacts"].values()
        )
    if len(set(used)) != len(used) or set(used) != set(manifest["files"]):
        raise ValueError(
            "dataset artifacts must have unique, exhaustive frame ownership"
        )
    return {
        "schema": "npa.isaac.rgbd.validation.v1",
        "validated": True,
        "frames": len(manifest["frames"]),
        "cameras": len(manifest["request"]["cameras"]),
        "views": sum(len(frame["views"]) for frame in manifest["frames"]),
        "valid_depth_pixels": pixels,
    }


def _finalize(root, request, frames, provenance):
    manifest = _manifest(root, request, frames, provenance)
    validate_dataset(root, manifest)
    (root / "manifest.json").write_bytes(_json_bytes(manifest))
    return manifest
