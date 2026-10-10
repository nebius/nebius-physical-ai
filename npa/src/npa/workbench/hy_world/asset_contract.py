"""Fail-closed evidence for HY-World's released image-to-world pipeline.

This module deliberately verifies outputs that only the full released pipeline
can produce: a generated panorama, WorldStereo expansion video(s), camera
metadata, a trained Gaussian-splat asset, and a newly rendered camera video.
It does not turn a WorldMirror-only reconstruction or a plausible placeholder
mesh into a world-generation claim.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

SOURCE_REPOSITORY = "https://github.com/Tencent-Hunyuan/HY-World-2.0"
SOURCE_REF = "df9988efb87bfc0f4947eb3889411cf957478b06"
HY_WORLD_MODEL_REPOSITORY = "tencent/HY-World-2.0"
HY_WORLD_MODEL_REF = "d78a16c91c7a56488894a1c8de4f5c7cc28aa8b0"
WORLD_STEREO_REPOSITORY = "hanshanxue/WorldStereo"
WORLD_STEREO_REF = "ac2ad97ecb043fe80c2f19cd1898006becb9d66e"
QWEN_IMAGE_REPOSITORY = "Qwen/Qwen-Image-Edit-2509"
QWEN_IMAGE_REF = "d3968ef930e841f4c73640fb8afa3b306a78167e"
QWEN_VLM_REPOSITORY = "Qwen/Qwen3-VL-8B-Instruct"
QWEN_VLM_REF = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
ZIM_REPOSITORY = "naver-iv/zim-anything-vitl"
ZIM_REF = "667e2d7c233f6f1cacd12ccc64bdf6cc7b5aa16d"
GROUNDING_DINO_REPOSITORY = "IDEA-Research/grounding-dino-tiny"
GROUNDING_DINO_REF = "a2bb814dd30d776dcf7e30523b00659f4f141c71"
SAM3_REPOSITORY = "facebook/sam3"
SAM3_REF = "3c879f39826c281e95690f02c7821c4de09afae7"
MOGE_REPOSITORY = "Ruicheng/moge-2-vitl-normal"
MOGE_REF = "cb0e8bbd6b1e243589717c78e750b1ba4c093acf"
UNI3C_REPOSITORY = "ewrfcas/Uni3C"
UNI3C_REF = "fca895f7fb454f4dfed43956cab09180d0946318"
PYTORCH3D_REPOSITORY = "https://github.com/facebookresearch/pytorch3d.git"
PYTORCH3D_REF = "88e182f989c80836f4bd744e0d9cb1852762ce01"
FLASH_ATTN_VERSION = "2.8.3"

EVIDENCE_SCHEMA = "npa.workbench.byof.hy_world_image_to_world.v1"
PRIMARY_CAPABILITY = "hy_world_2_image_conditioned_world_generation"
RENDER_CAPABILITY = "hy_world_2_generated_scene_camera_render"
REPORT_CAPABILITY = "hy_world_2_factual_scene_report"
_MIN_PLY_BYTES = 256


class HyWorldEvidenceError(RuntimeError):
    """Raised when an alleged HY-World result lacks real upstream evidence."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HyWorldEvidenceError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record(path: Path, *, root: Path) -> dict[str, Any]:
    _require(path.is_file(), f"missing required file: {path}")
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _load_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HyWorldEvidenceError(f"invalid JSON evidence {path}: {exc}") from exc
    _require(isinstance(payload, dict), f"camera evidence must be an object: {path}")
    return payload


def _matrix_count(value: Any) -> int:
    if not isinstance(value, list) or len(value) not in {3, 4}:
        return 0
    if not all(isinstance(row, list) and len(row) == 4 for row in value):
        return 0
    values = [entry for row in value for entry in row]
    return int(
        all(
            isinstance(entry, (int, float)) and math.isfinite(entry) for entry in values
        )
    )


def _camera_matrix_count(payload: Any) -> int:
    if _matrix_count(payload):
        return 1
    if isinstance(payload, Mapping):
        return sum(_camera_matrix_count(value) for value in payload.values())
    if isinstance(payload, list):
        return sum(_camera_matrix_count(value) for value in payload)
    return 0


def _validate_ply(path: Path) -> None:
    _require(path.stat().st_size >= _MIN_PLY_BYTES, f"PLY is too small: {path}")
    with path.open("rb") as stream:
        header = stream.read(16 * 1024)
    _require(header.startswith(b"ply\n"), f"PLY header missing: {path}")
    _require(b"end_header\n" in header, f"PLY header incomplete: {path}")
    match = re.search(rb"^element vertex ([0-9]+)$", header, re.MULTILINE)
    _require(match is not None, f"PLY has no vertex declaration: {path}")
    _require(int(match.group(1)) > 0, f"PLY has no vertices: {path}")


def _validate_spz(path: Path) -> None:
    """Reject a missing, empty, or obviously placeholder compressed splat."""

    _require(path.stat().st_size >= 16, f"SPZ is too small: {path}")
    with path.open("rb") as stream:
        header = stream.read(4)
    _require(header not in {b"", b"\x00\x00\x00\x00"}, f"SPZ header is empty: {path}")


def probe_image(path: Path) -> dict[str, Any]:
    """Decode an image and return factual dimensions for evidence."""

    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "stream=codec_type,width,height",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, check=False, text=True)
    _require(
        result.returncode == 0,
        f"ffprobe failed for {path.name}: {result.stderr.strip()}",
    )
    try:
        streams = json.loads(result.stdout).get("streams", [])
    except json.JSONDecodeError as exc:
        raise HyWorldEvidenceError(
            f"ffprobe emitted invalid JSON for {path.name}"
        ) from exc
    stream = next((item for item in streams if item.get("codec_type") == "video"), None)
    _require(isinstance(stream, dict), f"no decodable image stream in {path.name}")
    width, height = int(stream.get("width") or 0), int(stream.get("height") or 0)
    _require(width >= 512 and height >= 256, f"generated panorama is too small: {path}")
    _require(
        1.5 <= width / height <= 2.5,
        f"generated panorama is not equirectangular: {path}",
    )
    decode = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-frames:v", "1", "-f", "null", "-"],
        capture_output=True,
        check=False,
        text=True,
    )
    _require(decode.returncode == 0, f"ffmpeg decode failed for {path.name}")
    return {"width": width, "height": height}


def probe_video(path: Path) -> dict[str, Any]:
    """Decode a video and return only factual stream fields used in evidence."""

    command = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-show_entries",
        "stream=codec_type,width,height,nb_read_frames,duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, check=False, text=True)
    _require(
        result.returncode == 0,
        f"ffprobe failed for {path.name}: {result.stderr.strip()}",
    )
    try:
        streams = json.loads(result.stdout).get("streams", [])
    except json.JSONDecodeError as exc:
        raise HyWorldEvidenceError(
            f"ffprobe emitted invalid JSON for {path.name}"
        ) from exc
    stream = next((item for item in streams if item.get("codec_type") == "video"), None)
    _require(isinstance(stream, dict), f"no video stream in {path.name}")
    frames = int(stream.get("nb_read_frames") or 0)
    width, height = int(stream.get("width") or 0), int(stream.get("height") or 0)
    _require(
        width >= 64 and height >= 64 and frames >= 3,
        f"invalid render stream: {path.name}",
    )
    decode = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-frames:v", "3", "-f", "null", "-"],
        capture_output=True,
        check=False,
        text=True,
    )
    _require(decode.returncode == 0, f"ffmpeg decode failed for {path.name}")
    return {"width": width, "height": height, "decoded_frames": frames}


def _find_files(root: Path, pattern: str) -> list[Path]:
    return sorted(path for path in root.glob(pattern) if path.is_file())


def _required_one(paths: Iterable[Path], label: str) -> Path:
    result = list(paths)
    _require(bool(result), f"missing {label}")
    return result[0]


def _validate_camera_files(
    paths: Iterable[Path], *, root: Path
) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        matrices = _camera_matrix_count(_load_object(path))
        _require(matrices > 0, f"camera file has no finite 3x4/4x4 matrix: {path}")
        record = _record(path, root=root)
        record["finite_matrix_count"] = matrices
        records.append(record)
    _require(bool(records), "no upstream trajectory camera.json files were produced")
    return records


def _model_provenance(metadata: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "source_repository": SOURCE_REPOSITORY,
        "source_ref": SOURCE_REF,
        "hy_world_model": {
            "repository": HY_WORLD_MODEL_REPOSITORY,
            "ref": HY_WORLD_MODEL_REF,
        },
        "worldstereo_model": {
            "repository": WORLD_STEREO_REPOSITORY,
            "ref": WORLD_STEREO_REF,
        },
        "qwen_image_model": {
            "repository": QWEN_IMAGE_REPOSITORY,
            "ref": QWEN_IMAGE_REF,
        },
        "qwen_vlm": {"repository": QWEN_VLM_REPOSITORY, "ref": QWEN_VLM_REF},
        "zim_model": {"repository": ZIM_REPOSITORY, "ref": ZIM_REF},
        "grounding_dino_model": {
            "repository": GROUNDING_DINO_REPOSITORY,
            "ref": GROUNDING_DINO_REF,
        },
        "sam3_model": {"repository": SAM3_REPOSITORY, "ref": SAM3_REF},
        "moge_model": {"repository": MOGE_REPOSITORY, "ref": MOGE_REF},
        "uni3c_model": {"repository": UNI3C_REPOSITORY, "ref": UNI3C_REF},
        "runtime_build_dependencies": {
            "pytorch3d": {
                "repository": PYTORCH3D_REPOSITORY,
                "ref": PYTORCH3D_REF,
            },
            "flash_attn": {"package": "flash-attn", "version": FLASH_ATTN_VERSION},
            "gsplat_maskgaussian": {"source_ref": SOURCE_REF},
            "navmesh": {"source_ref": SOURCE_REF},
        },
    }
    for name, value in expected.items():
        _require(metadata.get(name) == value, f"runtime provenance drifted at {name}")
    image = str(metadata.get("container_image") or "")
    _require(
        "@sha256:" in image or image.startswith("sha256:"),
        "container image is not immutable",
    )
    inventories: dict[str, dict[str, str]] = {}
    for name, filename in (
        ("resolved_python_packages", "npa_resolved_inventory.txt"),
        ("submodules", "npa_submodules.txt"),
        ("model_snapshot_revisions", "npa_model_snapshot_revisions.json"),
    ):
        record = metadata.get(name)
        _require(isinstance(record, Mapping), f"runtime metadata lacks {name}")
        sha256 = str(record.get("sha256") or "")
        _require(
            record.get("path") == filename
            and re.fullmatch(r"[0-9a-f]{64}", sha256) is not None,
            f"runtime metadata has invalid {name}",
        )
        inventories[name] = {"path": filename, "sha256": sha256}
    return {**expected, "container_image": image, "runtime_inventories": inventories}


def build_evidence(
    *,
    scene_dir: Path,
    result_dir: Path,
    input_image: Path,
    runtime_metadata: Mapping[str, Any],
    image_checker: Callable[[Path], dict[str, Any]] = probe_image,
    video_checker: Callable[[Path], dict[str, Any]] = probe_video,
) -> dict[str, Any]:
    """Validate an actual image-to-world run and make its portable JSON record."""

    _require(scene_dir.is_dir(), f"scene directory is missing: {scene_dir}")
    _require(result_dir.is_dir(), f"result directory is missing: {result_dir}")
    _require(input_image.is_file(), f"input image is missing: {input_image}")
    _require(
        runtime_metadata.get("pipeline_mode") == "image_to_world", "not image-to-world"
    )
    _require(
        runtime_metadata.get("stages")
        == ["pano", "traj", "render", "expand", "gs_data", "gs_train"],
        "incomplete upstream pipeline",
    )
    provenance = _model_provenance(runtime_metadata)
    panorama = scene_dir / "panorama.png"
    _require(
        panorama.is_file() and panorama.stat().st_size > 256,
        "generated panorama missing",
    )
    worldstereo = _find_files(
        scene_dir, "render_results/**/worldstereo-memory-dmd_result.mp4"
    )
    _required_one(worldstereo, "WorldStereo expansion video")
    cameras = _find_files(scene_dir, "render_results/**/camera.json")
    splat = _required_one(_find_files(result_dir, "**/*.ply"), "trained 3DGS PLY")
    compressed = _required_one(
        _find_files(result_dir, "**/*.spz"), "compressed 3DGS SPZ"
    )
    rendered = _required_one(
        _find_files(result_dir, "videos/*.mp4"), "3DGS camera render"
    )
    _validate_ply(splat)
    _validate_spz(compressed)
    generated_videos = [
        {**_record(path, root=scene_dir), "decode": video_checker(path)}
        for path in worldstereo
    ]
    render_record = {
        **_record(rendered, root=result_dir),
        "decode": video_checker(rendered),
    }
    return {
        "schema": EVIDENCE_SCHEMA,
        "solution": "hy-world-2.0",
        "capability": PRIMARY_CAPABILITY,
        "capabilities_exercised": [
            PRIMARY_CAPABILITY,
            RENDER_CAPABILITY,
            REPORT_CAPABILITY,
        ],
        "pipeline_mode": "image_to_world",
        "source_baked": False,
        "weights_baked": False,
        "provenance": provenance,
        "input": _record(input_image, root=input_image.parent),
        "generated_panorama": {
            **_record(panorama, root=scene_dir),
            "decode": image_checker(panorama),
        },
        "worldstereo_expansion_videos": generated_videos,
        "trajectory_cameras": _validate_camera_files(cameras, root=scene_dir),
        "scene_assets": {
            "ply": _record(splat, root=result_dir),
            "spz": _record(compressed, root=result_dir),
        },
        "rendered_camera_dataset": render_record,
        "coordinates": {
            "camera_matrices": "upstream camera.json finite 3x4/4x4 matrices; c2w uses the upstream OpenCV convention when emitted as c2w",
            "scale": "upstream world-space normalization is uncalibrated; no metric scale claim",
        },
        "not_claimed": [
            "text_to_panorama_or_text_to_world",
            "worldmirror_only_multiview_reconstruction_as_world_generation",
            "mesh_collision_validity",
            "robot_policy_simulation",
            "metric_or_calibrated_scene_scale",
        ],
    }


def write_evidence(path: Path, **kwargs: Any) -> dict[str, Any]:
    """Build and atomically save generated-scene evidence after all checks pass."""

    evidence = build_evidence(**kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
    return evidence
