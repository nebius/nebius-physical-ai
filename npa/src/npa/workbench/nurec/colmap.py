"""Delegate COLMAP ingestion to NVIDIA NCore and verify a portable V4 sequence.

The executable is the native Bazel target //tools/data_converter/colmap:convert
from NVIDIA/ncore. This Apache-2.0 ingestion step is separate from proprietary
NRE reconstruction. Reader calls follow that pinned upstream source, including
trueprice's SceneManager (not the unrelated PyPI COLMAP bindings).
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import shutil
import stat
import struct
import subprocess
import tempfile
import zipfile
from dataclasses import fields
from pathlib import Path, PurePosixPath
from typing import Any, Literal
from urllib.parse import urlparse

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator

from npa.cli.path_contract import validate_read_path, validate_write_path
from npa.errors import NpaError
from npa.workbench.ncore_staging import (
    DEFAULT_COLMAP_CACHE_DIR,
    DEFAULT_COLMAP_SCRATCH_DIR,
    private_staging_directory,
)
from npa.workbench.nurec.nurec import NurecError

NCORE_REVISION = "59c698d206da92b406a4f72619fce3b3a2c64bfd"
DEFAULT_CONVERTER = "/opt/ncore/bin/colmap-convert"
CONVERSION_REPORT = "conversion.json"
PUBLICATION_CLAIM = ".npa-colmap-claim.json"


class NcoreConversionError(NurecError, NpaError):
    """The source, runtime, output verification, or publication failed."""


def _relative(value: str, *, allow_dot: bool = False) -> Path:
    # UPath also interprets protocols: reject those before constructing any path.
    parts = value.split("/")
    if value == "." and allow_dot:
        return Path(".")
    if (
        not value
        or "\\" in value
        or ":" in value
        or "\x00" in value
        or any(part in {"", ".", ".."} for part in parts)
        or PurePosixPath(value).is_absolute()
    ):
        raise ValueError("expected a safe relative path")
    return Path(*parts)


class ColmapConversionRequest(BaseModel):
    """S3 handoffs with local directories used only as disposable staging space."""

    model_config = ConfigDict(extra="forbid")
    input_path: str
    output_path: str
    cache_dir: Path = DEFAULT_COLMAP_CACHE_DIR
    scratch_dir: Path = DEFAULT_COLMAP_SCRATCH_DIR
    dataset_root: str = "."
    colmap_dir: str = "sparse/0"
    images_dir: str = "images"
    masks_dir: str = ""
    rig_mode: Literal["derive", "preserve"] = "derive"
    reference_camera: str = ""
    include_downsampled_images: bool = True

    @field_validator("input_path", "output_path")
    @classmethod
    def handoff(cls, value: str, info: Any) -> str:
        if info.field_name == "input_path":
            value = validate_read_path(
                value, tool="nurec convert-colmap", allow_hf=False
            )
        else:
            value = validate_write_path(
                value, tool="nurec convert-colmap", required=True
            )
        parsed = urlparse(value)
        if parsed.query or parsed.fragment or parsed.username or parsed.password:
            raise ValueError(
                "S3 handoffs must not contain authentication, query, or fragment"
            )
        _relative(parsed.path.lstrip("/").rstrip("/"))
        return value

    @field_validator("dataset_root", "colmap_dir", "images_dir", "masks_dir")
    @classmethod
    def layout(cls, value: str, info: Any) -> str:
        if info.field_name == "masks_dir" and not value:
            return value
        _relative(value, allow_dot=info.field_name in {"dataset_root", "colmap_dir"})
        return value


def _regular_files(root: Path) -> list[Path]:
    result = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise NcoreConversionError("dataset contains a link or non-regular entry")
        if path.is_file():
            result.append(path)
    return result


def extract_colmap_zip(archive: Path, destination: Path) -> None:
    """Preflight every member before extraction; never follow links or overwrite."""
    try:
        with zipfile.ZipFile(archive) as bundle:
            seen: set[Path] = set()
            entries = []
            for member in bundle.infolist():
                relative = _relative(member.filename.rstrip("/"))
                kind = stat.S_IFMT(member.external_attr >> 16)
                if kind not in {0, stat.S_IFREG, stat.S_IFDIR} or relative in seen:
                    raise ValueError("link, special, or duplicate archive entry")
                seen.add(relative)
                target = destination / relative
                if (
                    not target.resolve().is_relative_to(destination.resolve())
                    or target.exists()
                ):
                    raise ValueError("archive entry escapes or overwrites destination")
                entries.append((member, target))
            # Catch file/directory collisions independently of member order.
            file_paths = {path for member, path in entries if not member.is_dir()}
            if any(
                parent in file_paths for _, path in entries for parent in path.parents
            ):
                raise ValueError("archive file/directory collision")
            for member, target in entries:
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with bundle.open(member) as source, target.open("xb") as output:
                        shutil.copyfileobj(source, output)
    except (ValueError, OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise NcoreConversionError("unsafe or unreadable COLMAP archive") from exc


def find_colmap_root(
    root: Path, dataset_root: str, colmap_dir: str, images_dir: str
) -> Path:
    """Select one complete reconstruction, respecting an explicit relative layout."""

    def valid(path: Path) -> bool:
        sparse = path / colmap_dir
        return (path / images_dir).is_dir() and any(
            all(
                (sparse / f"{name}.{ext}").is_file()
                for name in ("cameras", "images", "points3D")
            )
            for ext in ("bin", "txt")
        )

    _regular_files(root)
    selected = root / _relative(dataset_root, allow_dot=True)
    if valid(selected):
        return selected
    if dataset_root != ".":
        raise NcoreConversionError(
            "selected dataset root lacks COLMAP cameras/images/points3D and images"
        )
    candidates = [p for p in sorted(root.rglob("*")) if p.is_dir() and valid(p)]
    if len(candidates) != 1:
        reason = "multiple" if candidates else "no"
        raise NcoreConversionError(
            f"found {reason} COLMAP datasets; set --dataset-root and layout options"
        )
    return candidates[0]


def _finite(value: Any, label: str) -> np.ndarray:
    array = np.asarray(value)
    if not np.isfinite(array).all():
        raise NcoreConversionError(f"non-finite {label}")
    return array


def _transforms(value: Any) -> np.ndarray:
    array = _finite(value, "transforms")
    if array.shape[-2:] != (4, 4) or not np.allclose(
        array[..., 3, :], [0, 0, 0, 1], atol=1e-5
    ):
        raise NcoreConversionError("invalid homogeneous transforms")
    rotations = array[..., :3, :3]
    if not np.allclose(
        rotations.swapaxes(-1, -2) @ rotations, np.eye(3), atol=1e-4
    ) or not np.allclose(np.linalg.det(rotations), 1, atol=1e-4):
        raise NcoreConversionError("invalid rotation transforms")
    return array


def validate_binary_images(path: Path) -> None:
    """Validate binary image IDs, paths and byte boundaries before vendor loading.

    trueprice's reader loops until a NUL byte, without checking for EOF. A
    truncated images.bin would otherwise hang rather than produce a domain error.
    Counts are checked against remaining file bytes, never an arbitrary cap.

    Args:
        path: COLMAP images.bin file.
    Returns:
        None.
    Raises:
        NcoreConversionError: Records are malformed, unsafe or duplicated.
    """
    _raw_binary_model_ids(path, "images")


def _binary_fields(stream: Any, layout: str) -> tuple[Any, ...]:
    size = struct.calcsize(layout)
    data = stream.read(size)
    if len(data) != size:
        raise ValueError("truncated record")
    return struct.unpack(layout, data)


def _skip_binary_values(stream: Any, size: int, count: int, stride: int) -> None:
    if count > (size - stream.tell()) // stride:
        raise ValueError("truncated variable-length record")
    stream.seek(count * stride, 1)


def _binary_camera_id(stream: Any, size: int) -> int:
    record_id, model, _, _ = _binary_fields(stream, "<IiQQ")
    # The pinned reader and converter support precisely these six COLMAP models.
    parameter_counts = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8}
    if model not in parameter_counts:
        raise ValueError("unsupported camera model")
    _skip_binary_values(stream, size, parameter_counts[model], 8)
    return record_id


def _binary_image_id(stream: Any, size: int) -> int:
    record_id = _binary_fields(stream, "<I4d3dI")[0]
    name = bytearray()
    while (character := _binary_fields(stream, "<c")[0]) != b"\0":
        name.extend(character)
    _relative(name.decode("utf-8"))
    observations = _binary_fields(stream, "<Q")[0]
    _skip_binary_values(stream, size, observations, 24)
    return record_id


def _binary_point_id(stream: Any, size: int) -> int:
    record = _binary_fields(stream, "<Q3d3BdQ")
    _skip_binary_values(stream, size, record[-1], 8)
    return record[0]


def _raw_binary_model_ids(path: Path, component: str) -> set[int]:
    """Parse IDs independently, bounding every declared count by actual bytes."""
    parsers = {
        "cameras": (_binary_camera_id, 48),
        "images": (_binary_image_id, 73),
        "points3D": (_binary_point_id, 51),
    }
    parse_record, minimum_size = parsers[component]
    ids: set[int] = set()
    try:
        size = path.stat().st_size
        with path.open("rb") as stream:
            count = _binary_fields(stream, "<Q")[0]
            if count > (size - stream.tell()) // minimum_size:
                raise ValueError("record count exceeds available bytes")
            for _ in range(count):
                record_id = parse_record(stream, size)
                if record_id in ids:
                    raise ValueError("duplicate record ID")
                ids.add(record_id)
            if stream.tell() != size:
                raise ValueError("unexpected trailing data")
    except (ValueError, OSError) as exc:
        raise NcoreConversionError(
            f"invalid COLMAP raw binary {component} records: {exc}"
        ) from exc
    return ids


def _raw_text_model_ids(path: Path, component: str) -> set[int]:
    """Check text record boundaries independently of the shared vendor reader.

    A blank images.txt observation line is a complete empty array; EOF is not.
    Camera and point files have one record per nonblank, noncomment line.
    """
    ids: set[int] = set()
    awaiting_observations = False
    try:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                line = line.strip()
                if line.startswith("#"):
                    continue
                if awaiting_observations:
                    observations = line.split()
                    if len(observations) % 3:
                        raise ValueError("incomplete observation triple")
                    for offset in range(0, len(observations), 3):
                        float(observations[offset])
                        float(observations[offset + 1])
                        int(observations[offset + 2])
                    awaiting_observations = False
                    continue
                if not line:
                    continue
                fields = (
                    line.split(maxsplit=9) if component == "images" else line.split()
                )
                if component == "images":
                    if len(fields) != 10:
                        raise ValueError("incomplete image header")
                    for value in fields[1:8]:
                        float(value)
                    int(fields[8])
                    awaiting_observations = True
                elif component == "cameras":
                    if len(fields) < 5:
                        raise ValueError("incomplete camera record")
                elif len(fields) < 8 or (len(fields) - 8) % 2:
                    raise ValueError("incomplete point record")
                record_id = int(fields[0])
                if record_id in ids:
                    raise ValueError("duplicate record ID")
                ids.add(record_id)
    except (ValueError, UnicodeError) as exc:
        raise NcoreConversionError(
            f"invalid COLMAP raw text {component} records"
        ) from exc
    if awaiting_observations:
        raise NcoreConversionError("incomplete COLMAP text image record")
    return ids


def _load_colmap_scene(root: Path, request: ColmapConversionRequest) -> Any:
    # Match each of SceneManager's binary-before-text selections independently.
    model_dir = root / request.colmap_dir
    raw_ids = {}
    for component in ("cameras", "images", "points3D"):
        binary = model_dir / f"{component}.bin"
        raw_ids[component] = (
            _raw_binary_model_ids(binary, component)
            if binary.is_file()
            else _raw_text_model_ids(model_dir / f"{component}.txt", component)
        )
    try:
        import pycolmap
    except ImportError as exc:
        raise NcoreConversionError(
            "runtime requires NCore's trueprice pycolmap reader"
        ) from exc
    if not hasattr(pycolmap, "SceneManager"):
        raise NcoreConversionError(
            "runtime requires trueprice pycolmap.SceneManager, not pip pycolmap bindings"
        )
    scene = pycolmap.SceneManager(
        str(root / request.colmap_dir), image_path=str(root / request.images_dir)
    )
    scene.load()
    _validate_parsed_model_ids(scene, raw_ids)
    return scene


def _validate_parsed_model_ids(scene: Any, raw_ids: dict[str, set[int]]) -> None:
    parsed_ids = {
        "cameras": list(scene.cameras),
        "images": list(scene.images),
        "points3D": list(scene.point3D_ids),
    }
    for component, expected_ids in raw_ids.items():
        actual_ids = parsed_ids[component]
        if len(actual_ids) != len(expected_ids) or set(actual_ids) != expected_ids:
            raise NcoreConversionError(
                f"COLMAP raw {component} IDs/counts differ from parsed model"
            )
    if len(scene.points3D) != len(raw_ids["points3D"]):
        raise NcoreConversionError(
            "COLMAP raw points3D count differs from parsed model"
        )


def inspect_colmap_source(
    root: Path, request: ColmapConversionRequest
) -> dict[str, Any]:
    """Read trueprice's actual COLMAP model and validate all file references."""
    scene = _load_colmap_scene(root, request)
    cameras: dict[str, dict[str, Any]] = {}
    for image in scene.images.values():
        relative = _relative(image.name)
        if not (root / request.images_dir / relative).is_file():
            raise NcoreConversionError("COLMAP references a missing image")
        camera = scene.cameras[image.camera_id]
        if camera.camera_type not in {0, 1, 2, 3, 4, 5}:
            raise NcoreConversionError(
                "COLMAP camera model is unsupported by the official converter"
            )
        intrinsic = _finite(
            [camera.width, camera.height, camera.fx, camera.fy, camera.cx, camera.cy],
            "source calibration",
        )
        if np.any(intrinsic[:4] <= 0):
            raise NcoreConversionError(
                "invalid source calibration dimensions or focal length"
            )
        rotation = _finite(image.R(), "source rotation")
        translation = _finite(image.tvec, "source translation").reshape(3)
        transform = np.eye(4)
        transform[:3, :3] = rotation.T
        transform[:3, 3] = -rotation.T @ translation
        _transforms(transform)
        camera_id = "camera" + str(image.camera_id)
        entry = cameras.setdefault(
            camera_id,
            {
                "frames": [],
                "resolution": [int(camera.width), int(camera.height)],
                "focal_length": [float(camera.fx), float(camera.fy)],
                "principal_point": [float(camera.cx), float(camera.cy)],
                **_source_distortion(camera),
                "target": "world",
            },
        )
        entry["frames"].append(
            {
                "name": image.name,
                "pose": transform.tolist(),
                "encoded_sha256": _hash_file(root / request.images_dir / relative),
            }
        )
    if not cameras:
        raise NcoreConversionError("COLMAP source contains no registered camera images")
    if request.include_downsampled_images:
        for factor in (2, 4, 8):
            image_dir = root / f"images_{factor}"
            if not image_dir.is_dir():
                continue
            for camera_id, original in list(cameras.items()):
                if original["target"] != "world":
                    continue
                for frame in original["frames"]:
                    if not (image_dir / _relative(frame["name"])).is_file():
                        raise NcoreConversionError(
                            "COLMAP references a missing downsampled image"
                        )
                # Upstream adjusts downsampled resolution to the actual first image.
                from PIL import Image

                with Image.open(image_dir / original["frames"][0]["name"]) as image:
                    resolution = list(image.size)
                cameras[f"{camera_id}_{factor}"] = {
                    **original,
                    "frames": [
                        {
                            "name": frame["name"],
                            "pose": np.eye(4).tolist(),
                            "encoded_sha256": _hash_file(image_dir / frame["name"]),
                        }
                        for frame in original["frames"]
                    ],
                    "resolution": resolution,
                    "focal_length": [v / factor for v in original["focal_length"]],
                    "principal_point": [
                        v / factor for v in original["principal_point"]
                    ],
                    "target": camera_id,
                }
    xyz = _finite(scene.points3D, "source points").astype(np.float32).reshape(-1, 3)
    _finite(xyz, "source points after float32 conversion")
    retained_mask = np.linalg.norm(xyz, axis=1) > 1e-6
    retained = xyz[retained_mask]
    rgb = _point_rgb(scene.point3D_colors, len(xyz))[retained_mask]
    images = sum(len(camera["frames"]) for camera in cameras.values())
    return {
        "cameras": cameras,
        "counts": {
            "images": images,
            "cameras": len(cameras),
            "poses": images,
            "points": len(retained),
        },
        "points_sha256": hashlib.sha256(retained.astype("<f4").tobytes()).hexdigest(),
        "points_rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
        "source_points": len(xyz),
        "origin_points_filtered": len(xyz) - len(retained),
    }


def _source_distortion(camera: Any) -> dict[str, Any]:
    """Map all six supported COLMAP models to the official V4 intrinsic schema."""
    camera_type = int(camera.camera_type)
    if camera_type == 5:
        coefficients = {
            "radial_coeffs": [camera.k1, camera.k2, camera.k3, camera.k4],
        }
    else:
        coefficients = {
            "radial_coeffs": [
                camera.k1 if camera_type > 1 else 0,
                camera.k2 if camera_type > 2 else 0,
                0,
                0,
                0,
                0,
            ],
            "tangential_coeffs": [camera.p1, camera.p2] if camera_type == 4 else [0, 0],
            "thin_prism_coeffs": [0, 0, 0, 0],
        }
    return {
        "colmap_camera_type": camera_type,
        "model_type": "opencv-fisheye" if camera_type == 5 else "opencv-pinhole",
        **{
            name: _finite(value, "source distortion").astype(np.float32).tolist()
            for name, value in coefficients.items()
        },
    }


def sequence_members(meta_path: Path) -> list[Path]:
    """Resolve only local regular itar stores named by a V4 sequence meta-file."""
    try:
        meta = json.loads(meta_path.read_text())
        if meta.get("version") != "v4" or not meta.get("component_stores"):
            raise ValueError("missing V4 component stores")
        members = []
        for store in meta["component_stores"]:
            relative = _relative(store["path"])
            path = meta_path.parent / relative
            if (
                len(relative.parts) != 1
                or not path.name.endswith(".zarr.itar")
                or path.is_symlink()
                or not path.is_file()
            ):
                raise ValueError("unsupported or missing component store")
            if path in members:
                raise ValueError("duplicate component store")
            members.append(path)
        return members
    except (ValueError, KeyError, TypeError, OSError) as exc:
        raise NcoreConversionError(
            "invalid or unsafe NCore component references"
        ) from exc


def _rig_reference(source: dict[str, Any], preferred: str) -> str:
    candidates = {
        name: camera
        for name, camera in source["cameras"].items()
        if camera["target"] == "world" and camera["frames"]
    }
    if not candidates or (preferred and preferred not in candidates):
        raise NcoreConversionError("derived rig reference has no source world poses")
    if preferred:
        return preferred
    return min(candidates, key=lambda name: (-len(candidates[name]["frames"]), name))


def _validate_rig_sidecar(
    meta_path: Path, source: dict[str, Any], sequence_id: str, reference: str
) -> None:
    expected = {
        "derived_by": "npa.workbench.nurec.ncore_rig",
        "sequence_id": sequence_id,
        "reference_camera": reference,
        "poses_component_group": "npa_rig",
        "pose_count": len(source["cameras"][reference]["frames"]),
        "cameras": sorted(
            name
            for name, camera in source["cameras"].items()
            if camera["target"] == "world"
        ),
    }
    try:
        sidecar = json.loads((meta_path.parent / "npa-rig.json").read_text())
    except (OSError, ValueError) as exc:
        raise NcoreConversionError("invalid or missing derived rig sidecar") from exc
    if not isinstance(sidecar, dict) or any(
        type(sidecar.get(key)) is not type(value) or sidecar[key] != value
        for key, value in expected.items()
    ):
        raise NcoreConversionError("derived rig sidecar differs from source selection")


def _validate_derived_rig(
    meta_path: Path,
    source: dict[str, Any],
    reader: Any,
    trajectories: dict,
    reference_camera: str,
) -> None:
    reference = _rig_reference(source, reference_camera)
    _validate_rig_sidecar(meta_path, source, str(reader.sequence_id), reference)
    frames = source["cameras"][reference]["frames"]
    expected_ts = np.arange(len(frames), dtype=np.uint64) * 1_000_000
    edge = ("rig", "world")
    if edge not in trajectories["npa_rig"]:
        raise NcoreConversionError("derived rig edge is absent")
    poses, timestamps = trajectories["npa_rig"][edge]
    if (
        len(frames) < 2
        or poses.shape != (len(frames), 4, 4)
        or not np.array_equal(timestamps, expected_ts)
        or not np.allclose(
            poses, [frame["pose"] for frame in frames], rtol=1e-4, atol=1e-4
        )
    ):
        raise NcoreConversionError(
            "derived rig trajectory differs from source reference"
        )


def _point_rgb(value: Any, point_count: int) -> np.ndarray:
    rgb = _finite(value, "point RGB")
    if rgb.shape != (point_count, 3) or rgb.dtype != np.dtype("uint8"):
        raise NcoreConversionError("point RGB must have shape (N, 3) and uint8 values")
    return rgb


def _validate_point_clouds(reader: Any, source: dict[str, Any]) -> int:
    from ncore.data.v4 import PointCloudsComponent

    point_digest, rgb_digest = hashlib.sha256(), hashlib.sha256()
    count = 0
    for points in reader.open_component_readers(PointCloudsComponent.Reader).values():
        _finite(points.pc_timestamps_us, "point timestamps")
        for index in range(points.pcs_count):
            xyz = _finite(points.get_pc_xyz(index), "points")
            if (
                xyz.ndim != 2
                or xyz.shape[1] != 3
                or points.get_pc_reference_frame_id(index) != "world"
            ):
                raise NcoreConversionError("invalid point geometry or reference frame")
            if "rgb" not in points.attribute_names:
                raise NcoreConversionError("NCore sparse point RGB is absent")
            rgb = _point_rgb(points.get_pc_attribute(index, "rgb"), len(xyz))
            for attribute in points.attribute_names:
                values = _finite(
                    points.get_pc_attribute(index, attribute), "point attributes"
                )
                if len(values) != len(xyz):
                    raise NcoreConversionError("point attribute count differs")
            point_digest.update(xyz.astype("<f4").tobytes())
            rgb_digest.update(rgb.tobytes())
            count += len(xyz)
    if (
        source.get("points_sha256")
        and point_digest.hexdigest() != source["points_sha256"]
    ):
        raise NcoreConversionError("source and NCore point coordinates differ")
    if rgb_digest.hexdigest() != source.get("points_rgb_sha256"):
        raise NcoreConversionError("source and NCore point RGB differ")
    return count


def validate_ncore_sequence(
    meta_path: Path, source: dict[str, Any], *, rig_mode: str = "preserve",
    reference_camera: str = "",
) -> dict[str, int]:
    """Independently reopen every V4 frame, calibration, pose and point array.

    Args:
        meta_path: Portable V4 sequence metadata.
        source: Source inventory returned by inspect_colmap_source.
        rig_mode: Whether to require the derived rig trajectory and sidecar.
        reference_camera: Explicit rig reference, or longest source world trajectory
            with camera ID breaking ties.
    Returns:
        Verified camera, image, pose and retained sparse point counts.
    Raises:
        NcoreConversionError: Output data or rig provenance differs from source.
    """
    sequence_members(meta_path)
    try:
        from ncore.data.v4 import (
            CameraSensorComponent,
            IntrinsicsComponent,
            PosesComponent,
            SequenceComponentGroupsReader,
        )
        from upath import UPath
    except ImportError as exc:
        raise NcoreConversionError(
            "runtime requires the pinned NVIDIA ncore V4 reader"
        ) from exc
    reader = SequenceComponentGroupsReader([UPath(meta_path)])
    cameras = reader.open_component_readers(CameraSensorComponent.Reader)
    intrinsics = reader.open_component_readers(IntrinsicsComponent.Reader)
    pose_readers = reader.open_component_readers(PosesComponent.Reader)
    if set(cameras) != set(source["cameras"]):
        raise NcoreConversionError("source and NCore camera counts or IDs differ")
    counts = {"cameras": len(cameras), "images": 0, "poses": 0, "points": 0}
    trajectories: dict[str, dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]] = {}
    for group, poses_reader in pose_readers.items():
        trajectories[group] = {}
        for edge, (poses, timestamps) in poses_reader.get_dynamic_poses():
            poses = _transforms(poses)
            timestamps = _finite(timestamps, "pose timestamps")
            if (
                poses.shape != (len(timestamps), 4, 4)
                or len(timestamps) == 0
                or np.any(timestamps[1:] <= timestamps[:-1])
            ):
                raise NcoreConversionError("invalid pose timeline")
            trajectories[group][edge] = (poses, timestamps)
        for _, transform in poses_reader.get_static_poses():
            _transforms(transform)
    group = "npa_rig" if rig_mode == "derive" else "default"
    if group not in trajectories:
        raise NcoreConversionError("required NCore poses component group is absent")
    if rig_mode == "derive":
        _validate_derived_rig(meta_path, source, reader, trajectories, reference_camera)
    for camera_id, camera in cameras.items():
        expected = source["cameras"][camera_id]
        calibrations = []
        for calibration in intrinsics.values():
            try:
                calibrations.append(calibration.get_camera_model_parameters(camera_id))
            except KeyError:
                continue
        if len(calibrations) != 1:
            raise NcoreConversionError(
                "camera must have exactly one intrinsic calibration"
            )
        model = calibrations[0]
        for item in fields(model):
            value = getattr(model, item.name)
            if isinstance(value, (np.ndarray, float, int)):
                _finite(value, "calibration")
        if (
            model.type() != expected["model_type"]
            or model.shutter_type.name != "GLOBAL"
            or model.external_distortion_parameters is not None
        ):
            raise NcoreConversionError("source and NCore calibration model differ")
        calibration_fields = [
            "resolution",
            "focal_length",
            "principal_point",
            "radial_coeffs",
        ]
        if expected["model_type"] == "opencv-pinhole":
            calibration_fields.extend(["tangential_coeffs", "thin_prism_coeffs"])
        else:
            from ncore.impl.data.types import OpenCVFisheyeCameraModelParameters

            # This bound depends on source resolution, K and all four source
            # coefficients. Compute it from the source, never the output model.
            expected = {
                **expected,
                "max_angle": OpenCVFisheyeCameraModelParameters.compute_max_angle(
                    np.asarray(expected["resolution"], dtype=np.uint64),
                    np.asarray(expected["focal_length"], dtype=np.float32),
                    np.asarray(expected["principal_point"], dtype=np.float32),
                    np.asarray(expected["radial_coeffs"], dtype=np.float32),
                ),
            }
            calibration_fields.append("max_angle")
        for name in calibration_fields:
            actual = np.asarray(getattr(model, name))
            if actual.shape != np.shape(expected[name]) or not np.allclose(
                actual, expected[name], rtol=1e-5, atol=1e-7
            ):
                raise NcoreConversionError("source and NCore calibration differ")
        timestamps = _finite(camera.frames_timestamps_us, "image timestamps")
        n_frames = len(expected["frames"])
        if timestamps.shape != (n_frames, 2) or camera.frames_count != n_frames:
            raise NcoreConversionError("source and NCore image counts differ")
        expected_ts = np.arange(n_frames, dtype=np.uint64) * 1_000_000
        if not np.array_equal(timestamps[:, 0], expected_ts) or not np.array_equal(
            timestamps[:, 1], expected_ts
        ):
            raise NcoreConversionError(
                "NCore image timeline differs from upstream's 1 FPS mapping"
            )
        edge = (camera_id, expected["target"])
        # Check original AND copied derived trajectories; neither may hide bad geometry.
        for poses_group in trajectories.values():
            if edge not in poses_group:
                raise NcoreConversionError("camera pose edge is absent")
            poses, pose_ts = poses_group[edge]
            if not np.array_equal(pose_ts, expected_ts) or not np.allclose(
                poses,
                [frame["pose"] for frame in expected["frames"]],
                rtol=1e-4,
                atol=1e-4,
            ):
                raise NcoreConversionError("source and NCore camera transforms differ")
        for index, timestamp in enumerate(timestamps[:, 1]):
            expected_hash = expected["frames"][index].get("encoded_sha256")
            if (
                expected_hash
                and hashlib.sha256(
                    camera.get_frame_data(int(timestamp)).get_encoded_image_data()
                ).hexdigest()
                != expected_hash
            ):
                raise NcoreConversionError("source and NCore image bytes differ")
            with camera.get_frame_image(int(timestamp)) as decoded:
                decoded.load()
                if tuple(decoded.size) != tuple(expected["resolution"]):
                    raise NcoreConversionError(
                        "decoded image dimensions differ from calibration"
                    )
            for name in camera.get_frame_generic_data_names(int(timestamp)):
                value = _finite(
                    camera.get_frame_generic_data(int(timestamp), name),
                    "camera frame data",
                )
                if name == "mask" and value.shape != tuple(
                    reversed(expected["resolution"])
                ):
                    raise NcoreConversionError(
                        "frame mask dimensions differ from image"
                    )
            counts["images"] += 1
        counts["poses"] += n_frames
    counts["points"] = _validate_point_clouds(reader, source)
    if counts != source["counts"]:
        raise NcoreConversionError("source and NCore counts differ")
    return counts


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inventory(root: Path, members: list[Path]) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _hash_file(path),
        }
        for path in sorted(members)
    ]


def _claim_destination(client: Any, base: str, report_sha256: str) -> None:
    """Reserve an empty immutable prefix; claims are never removed or reused.

    A failed/ambiguous publication requires a fresh output prefix. In particular,
    do not expire claims: a paused original writer could resume after expiry.
    """
    from npa.clients.storage import StoragePreconditionFailed

    parsed = urlparse(base)
    prefix = parsed.path.lstrip("/") + "/"
    claim_key = prefix + PUBLICATION_CLAIM

    def occupied(*, claimed: bool = False) -> bool:
        paginator = client.s3.get_paginator("list_objects_v2")
        return any(
            not claimed or item["Key"] != claim_key
            for page in paginator.paginate(Bucket=parsed.netloc, Prefix=prefix)
            for item in page.get("Contents", [])
        )

    unavailable = "COLMAP destination is occupied or claimed; use a fresh output prefix"
    # Avoid even adding a claim to an existing legacy generation. This check is
    # advisory: only the provider's conditional PutObject grants ownership.
    if occupied():
        raise NcoreConversionError(unavailable)
    try:
        client.put_bytes_conditional(
            json.dumps({"schema_version": 1, "report_sha256": report_sha256}).encode(),
            f"{base}/{PUBLICATION_CLAIM}",
            if_none_match=True,
            content_type="application/json",
        )
    except StoragePreconditionFailed as exc:
        raise NcoreConversionError(unavailable) from exc
    # Detect legacy/nonparticipating writes between the advisory check and claim.
    # Retain the claim on any failure; no converter-owned member has been written.
    if occupied(claimed=True):
        raise NcoreConversionError(unavailable)


def verify_conversion_inventory(root: Path) -> None:
    """Verify a COLMAP handoff without importing NCore or invoking its reader.

    Older, otherwise complete conversion reports remain verifiable without a
    claim. Ordinary NuRec sequences without either sidecar remain supported.
    """
    report_path = root / CONVERSION_REPORT
    claim_path = root / PUBLICATION_CLAIM
    if not any(
        path.exists() or path.is_symlink() for path in (report_path, claim_path)
    ):
        return
    try:
        files = set(_regular_files(root))
        report = json.loads(report_path.read_text())
        if (
            not isinstance(report, dict)
            or report["schema_version"] != 1
            or report["status"] != "ok"
            or report["engine"] != "nvidia-ncore-colmap"
            or not isinstance(report["ncore_meta"], str)
        ):
            raise ValueError("invalid conversion report")
        meta_name = _relative(report["ncore_meta"])
        if len(meta_name.parts) != 1 or meta_name.suffix != ".json":
            raise ValueError("invalid metadata name")
        meta = root / meta_name
        expected = {meta, *sequence_members(meta)}
        if report["poses_component_group"] == "npa_rig":
            expected.add(root / "npa-rig.json")
        elif report["poses_component_group"] != "default":
            raise ValueError("invalid poses group")
        members = report["members"]
        if not isinstance(members, list) or not members:
            raise ValueError("missing inventory")
        inventoried = set()
        for item in members:
            if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                raise ValueError("invalid inventory entry")
            relative = _relative(item["path"])
            path = root / relative
            if len(relative.parts) != 1 or path in inventoried or path not in files:
                raise ValueError("invalid inventory member")
            if (
                type(item["bytes"]) is not int
                or path.stat().st_size != item["bytes"]
                or _hash_file(path) != item["sha256"]
            ):
                raise ValueError("inventory bytes differ")
            inventoried.add(path)
        if inventoried != expected:
            raise ValueError("incomplete inventory")
        expected.add(report_path)
        publication = report.get("publication")
        if publication is not None and publication != {
            "mode": "immutable-prefix-v1",
            "claim": PUBLICATION_CLAIM,
        }:
            raise ValueError("unsupported publication")
        if publication is not None or claim_path.exists():
            claim = json.loads(claim_path.read_text())
            if (
                not isinstance(claim, dict)
                or claim["schema_version"] != 1
                or claim["report_sha256"] != _hash_file(report_path)
            ):
                raise ValueError("publication claim differs")
            expected.add(claim_path)
        if files != expected:
            raise ValueError("mixed conversion generation")
    except (ValueError, TypeError, KeyError, OSError, NcoreConversionError) as exc:
        raise NcoreConversionError(
            "COLMAP conversion inventory is incomplete or inconsistent"
        ) from exc


def _runtime_fingerprints() -> dict[str, str]:
    import inspect
    import pycolmap
    from ncore.data.v4 import SequenceComponentGroupsReader

    return {
        "executable_sha256": _hash_file(Path(DEFAULT_CONVERTER)),
        "v4_reader_source_sha256": _hash_file(
            Path(inspect.getfile(SequenceComponentGroupsReader))
        ),
        "colmap_reader_source_sha256": _hash_file(
            Path(inspect.getfile(pycolmap.SceneManager))
        ),
    }


def _derive_in_place(meta: Path, reference_camera: str) -> None:
    from npa.workbench.nurec.ncore_rig import derive_rig_poses

    # Same directory means original regular shards remain in place: no symlinks,
    # duplicate sequence metas, or cross-pod absolute paths are introduced.
    result = derive_rig_poses(
        meta,
        output_dir=meta.parent,
        reference_camera=reference_camera,
        sequence_meta_name=meta.name,
    )
    if not result.ok:
        raise NcoreConversionError("NCore rig derivation failed")


def convert_colmap(
    request: ColmapConversionRequest, *, storage_client: Any = None
) -> dict[str, Any]:
    """Run the real converter, validate every frame, then publish at the exact URI."""
    from npa.clients.storage import StorageClient

    phase = "staging"
    try:
        client = storage_client or StorageClient.from_environment()
        # Fresh directories avoid stale source bytes and racing invocations. The
        # caller chooses scratch/cache placement; no input files are deleted.
        with (
            private_staging_directory(
                request.cache_dir, prefix="colmap-"
            ) as cache,
            private_staging_directory(
                request.scratch_dir, prefix="ncore-"
            ) as scratch,
        ):
            staged = Path(cache) / "dataset"
            staged.mkdir()
            if urlparse(request.input_path).path.lower().endswith(".zip"):
                archive = Path(cache) / "input.zip"
                client.download_file(request.input_path, str(archive))
                extract_colmap_zip(archive, staged)
                archive_hash = _hash_file(archive)
            else:
                client.download_directory(
                    request.input_path.rstrip("/") + "/", str(staged)
                )
                archive_hash = None
            root = find_colmap_root(
                staged, request.dataset_root, request.colmap_dir, request.images_dir
            )
            source_members = _inventory(root, _regular_files(root))
            phase = "source validation"
            # Some vendor readers print source names; never leak them into the
            # CLI's stdout contract or publish private raw runtime diagnostics.
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                source = inspect_colmap_source(root, request)
            output = Path(scratch) / "converted"
            argv = [
                DEFAULT_CONVERTER,
                "--root-dir",
                str(root),
                "--output-dir",
                str(output),
                "colmap-v4",
                "--colmap-dir",
                request.colmap_dir,
                "--images-dir",
                request.images_dir,
                "--store-type",
                "itar",
                "--sequence-meta",
                "--include-3d-points",
                "--include-downsampled-images"
                if request.include_downsampled_images
                else "--no-include-downsampled-images",
            ]
            if request.masks_dir:
                argv.extend(["--masks-dir", request.masks_dir])
            phase = "official converter"
            with tempfile.TemporaryFile() as diagnostics:
                completed = subprocess.run(
                    argv, stdout=diagnostics, stderr=diagnostics, check=False
                )
            if completed.returncode != 0:
                raise NcoreConversionError(
                    f"official NCore converter failed with exit code {completed.returncode}"
                )
            upstream_meta = output / root.name / f"{root.name}.json"
            sequence_members(upstream_meta)
            # A fixed discovery name cannot collide with our provenance/rig
            # sidecars when the source directory happens to share their names.
            meta = upstream_meta.with_name("sequence.json")
            if upstream_meta != meta:
                upstream_meta.rename(meta)
            phase = "rig derivation"
            if request.rig_mode == "derive":
                _derive_in_place(meta, request.reference_camera)
            phase = "NCore validation"
            counts = validate_ncore_sequence(
                meta, source, rig_mode=request.rig_mode,
                reference_camera=request.reference_camera,
            )
            members = [meta, *sequence_members(meta)]
            sidecar = meta.parent / "npa-rig.json"
            if request.rig_mode == "derive":
                if not sidecar.is_file() or sidecar.is_symlink():
                    raise NcoreConversionError("derived rig sidecar is missing")
                members.append(sidecar)
            inventory = _inventory(meta.parent, members)
            report = {
                "schema_version": 1,
                "status": "ok",
                "engine": "nvidia-ncore-colmap",
                "converter": {
                    "repository": "https://github.com/NVIDIA/ncore",
                    "revision": NCORE_REVISION,
                    "target": "//tools/data_converter/colmap:convert",
                    "runtime_sha256": _runtime_fingerprints(),
                    "license": "Apache-2.0",
                },
                "reader": "ncore.data.v4.SequenceComponentGroupsReader",
                "source": {
                    "input_uri_sha256": hashlib.sha256(
                        request.input_path.encode()
                    ).hexdigest(),
                    "archive_sha256": archive_hash,
                    "members": source_members,
                    "counts": source["counts"],
                    "origin_points_filtered": source.get("origin_points_filtered", 0),
                },
                "options": request.model_dump(
                    mode="json",
                    exclude={"input_path", "output_path", "cache_dir", "scratch_dir"},
                ),
                "counts": counts,
                "members": inventory,
                "publication": {
                    "mode": "immutable-prefix-v1",
                    "claim": PUBLICATION_CLAIM,
                },
                "ncore_meta": meta.name,
                "poses_component_group": "npa_rig"
                if request.rig_mode == "derive"
                else "default",
                "time_mapping": "upstream assigns per-camera image order timestamps at 1 FPS",
                "point_filter": "upstream excludes float32 SfM points whose norm is <= 1e-6",
            }
            report_path = meta.parent / CONVERSION_REPORT
            report_path.write_text(
                json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
            )
            phase = "publication"
            base = request.output_path.rstrip("/")
            _claim_destination(client, base, _hash_file(report_path))
            # Publish the discovery meta last, after every referenced byte and
            # provenance document is durable in our exclusively owned prefix.
            # The permanent claim prevents replacement, even after partial failure.
            for member in [path for path in members if path != meta] + [
                report_path,
                meta,
            ]:
                client.upload_file(str(member), f"{base}/{member.name}")
            return {
                "status": "ok",
                "engine": report["engine"],
                "output_path": base + "/",
                "ncore_meta_uri": f"{base}/{meta.name}",
                "conversion_uri": f"{base}/{CONVERSION_REPORT}",
                "conversion_sha256": _hash_file(report_path),
                "counts": counts,
                "poses_component_group": report["poses_component_group"],
                "objects": len(members) + 2,
            }
    except NcoreConversionError:
        raise
    except Exception as exc:
        # Library/S3/subprocess exceptions may include credentials, endpoints or
        # source data. The public domain error carries only the failing phase.
        raise NcoreConversionError(
            f"COLMAP conversion failed during {phase} ({type(exc).__name__})"
        ) from exc
