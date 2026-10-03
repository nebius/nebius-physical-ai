"""Build a compact offline RGB, depth and point preview from validated capture bytes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from npa.workflows.preview_html import image_preview, write_preview

from .contract import _contained


def _depth_preview(path: Path, far: float) -> str:
    depth = np.load(path, allow_pickle=False)
    intensity = (np.clip(depth / far, 0, 1) * 255).astype(np.uint8)
    colors = np.stack([intensity, 255 - intensity, np.full_like(intensity, 180)], -1)
    colors[depth <= 0] = 0
    return image_preview(Image.fromarray(colors), width=320)


def _points(root: Path, frame: dict) -> list:
    if not frame["fused_cloud"]:
        return []
    path = _contained(root, frame["fused_cloud"]["path"])
    with np.load(path, allow_pickle=False) as cloud:
        points, colors = cloud["xyz_world_m"], cloud["rgb"]
        indices = np.linspace(0, len(points) - 1, min(len(points), 2048), dtype=int)
        return [
            [*point.round(5).tolist(), *color.tolist()]
            for point, color in zip(points[indices], colors[indices], strict=True)
        ]


def _frame(root: Path, frame: dict, cameras: dict) -> dict:
    images = []
    for view in frame["views"]:
        camera = cameras[view["camera_id"]]
        far = camera["depth_range_m"][1]
        with Image.open(_contained(root, view["artifacts"]["rgb"])) as image:
            images.append(
                {
                    "label": f"{camera['id']} · RGB",
                    "data": image_preview(image, width=320),
                }
            )
        images.append(
            {
                "label": f"{camera['id']} · axial depth, 0–{far:g} m (black = invalid)",
                "data": _depth_preview(
                    _contained(root, view["artifacts"]["depth"]), far
                ),
            }
        )
    return {
        "label": f"Pose {frame['index']} · {frame['timestamp_ns'] / 1e9:g} s",
        "images": images,
        "points": _points(root, frame),
    }


def _timeline(root, manifest):
    frames = manifest["frames"]
    indices = np.linspace(0, len(frames) - 1, min(len(frames), 32), dtype=int)
    cameras = {camera["id"]: camera for camera in manifest["request"]["cameras"]}
    return {
        "title": "Synchronized camera rig",
        "note": "Evenly spaced preview poses from the full capture. The timeline uses recorded simulation timestamps; playback is a slideshow. Depth colors use each camera's fixed metric range.",
        "frames": [_frame(root, frames[index], cameras) for index in indices],
    }


def write_capture_preview(
    root: Path, manifest: dict, validation: dict, output: Path
) -> dict:
    """Create a small visual companion after the full dataset passes validation.

    Args:
        root: Downloaded, hash-verified capture directory.
        manifest: Validated capture manifest.
        validation: Successful full decoded validation result.
        output: Local HTML destination.

    Returns:
        Display-sampling counts, separate from the full dataset measurements.

    Raises:
        ValueError: The supplied validation report did not pass.
        OSError: A source media artifact cannot be read.
    """
    if validation.get("validated") is not True:
        raise ValueError("capture preview requires successful full validation")
    timeline = _timeline(root, manifest)
    metrics = {
        name.replace("_", " "): validation[name]
        for name in ("frames", "cameras", "views", "valid_depth_pixels", "fused_points")
    }
    metrics["preview poses"] = len(timeline["frames"])
    write_preview(
        output,
        title="Synthetic RGB, depth and colored points",
        metrics=metrics,
        summary="All capture artifacts passed decoded geometry, calibration, synchronization and hash checks. This is sensor rendering along a supplied route; it does not establish robot navigation or collision-free travel.",
        groups=[timeline],
    )
    return {"poses": len(timeline["frames"]), "maximum_points_per_pose": 2048}
