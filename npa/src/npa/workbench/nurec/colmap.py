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
from npa.workbench.nurec.nurec import NurecError

NCORE_REVISION = "59c698d206da92b406a4f72619fce3b3a2c64bfd"
DEFAULT_CONVERTER = "/opt/ncore/bin/colmap-convert"
CONVERSION_REPORT = "conversion.json"


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
    cache_dir: Path = Path("/tmp/npa-ncore-cache")
    scratch_dir: Path = Path("/tmp/npa-ncore-scratch")
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
    """Check the upstream reader's binary layout before its NUL-name loop.

    trueprice's reader loops until a NUL byte, without checking for EOF. A
    truncated images.bin would otherwise hang rather than produce a domain error.
    Counts are checked against remaining file bytes, never an arbitrary cap.
    """
    size = path.stat().st_size
    with path.open("rb") as stream:

        def take(count: int) -> bytes:
            value = stream.read(count)
            if len(value) != count:
                raise NcoreConversionError("truncated COLMAP binary images model")
            return value

        count = struct.unpack("<Q", take(8))[0]
        for _ in range(count):
            take(struct.calcsize("<I4d3dI"))
            name = bytearray()
            while (character := take(1)) != b"\0":
                name.extend(character)
            try:
                _relative(name.decode("utf-8"))
            except (ValueError, UnicodeError) as exc:
                raise NcoreConversionError(
                    "unsafe COLMAP binary image reference"
                ) from exc
            points = struct.unpack("<Q", take(8))[0]
            remaining = size - stream.tell()
            if points > remaining // 24:
                raise NcoreConversionError("truncated COLMAP binary image observations")
            stream.seek(points * 24, 1)
        if stream.tell() != size:
            raise NcoreConversionError("unexpected trailing COLMAP binary image data")


def inspect_colmap_source(
    root: Path, request: ColmapConversionRequest
) -> dict[str, Any]:
    """Read trueprice's actual COLMAP model and validate all file references."""
    binary_images = root / request.colmap_dir / "images.bin"
    if binary_images.is_file():
        validate_binary_images(binary_images)
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
    retained = xyz[np.linalg.norm(xyz, axis=1) > 1e-6]
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
        "source_points": len(xyz),
        "origin_points_filtered": len(xyz) - len(retained),
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


def validate_ncore_sequence(
    meta_path: Path, source: dict[str, Any], *, rig_mode: str = "preserve"
) -> dict[str, int]:
    """Independently reopen every V4 frame, calibration, pose and point array."""
    sequence_members(meta_path)
    try:
        from ncore.data.v4 import (
            CameraSensorComponent,
            IntrinsicsComponent,
            PointCloudsComponent,
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
    if rig_mode == "derive" and ("rig", "world") not in trajectories[group]:
        raise NcoreConversionError("derived rig edge is absent")
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
        for name in ("resolution", "focal_length", "principal_point"):
            if not np.allclose(
                getattr(model, name), expected[name], rtol=1e-5, atol=1e-5
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
    point_readers = reader.open_component_readers(PointCloudsComponent.Reader)
    point_digest = hashlib.sha256()
    for points in point_readers.values():
        _finite(points.pc_timestamps_us, "point timestamps")
        for index in range(points.pcs_count):
            xyz = _finite(points.get_pc_xyz(index), "points")
            if (
                xyz.ndim != 2
                or xyz.shape[1] != 3
                or points.get_pc_reference_frame_id(index) != "world"
            ):
                raise NcoreConversionError("invalid point geometry or reference frame")
            for attribute in points.attribute_names:
                values = _finite(
                    points.get_pc_attribute(index, attribute), "point attributes"
                )
                if len(values) != len(xyz):
                    raise NcoreConversionError("point attribute count differs")
            point_digest.update(xyz.astype("<f4").tobytes())
            counts["points"] += len(xyz)
    if (
        source.get("points_sha256")
        and point_digest.hexdigest() != source["points_sha256"]
    ):
        raise NcoreConversionError("source and NCore point coordinates differ")
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
        request.cache_dir.mkdir(parents=True, exist_ok=True)
        request.scratch_dir.mkdir(parents=True, exist_ok=True)
        # Fresh directories avoid stale source bytes and racing invocations. The
        # caller chooses scratch/cache placement; no input files are deleted.
        with (
            tempfile.TemporaryDirectory(
                prefix="colmap-", dir=request.cache_dir
            ) as cache,
            tempfile.TemporaryDirectory(
                prefix="ncore-", dir=request.scratch_dir
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
            counts = validate_ncore_sequence(meta, source, rig_mode=request.rig_mode)
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
            # Publish the discovery meta last, after every referenced byte and
            # provenance document is durable. No hidden sequence/ subdirectory.
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
                "objects": len(members) + 1,
            }
    except NcoreConversionError:
        raise
    except Exception as exc:
        # Library/S3/subprocess exceptions may include credentials, endpoints or
        # source data. The public domain error carries only the failing phase.
        raise NcoreConversionError(
            f"COLMAP conversion failed during {phase} ({type(exc).__name__})"
        ) from exc
