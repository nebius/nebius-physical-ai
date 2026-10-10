"""Collect verified policy-driven Go1 observations using the existing Marble world."""

import json
from pathlib import Path
import tempfile
import time

import numpy as np

from .api import MarbleError
from .gpu import _capture, _cuda, _result_metadata
from .quadruped_assets import acquire_robot
from .quadruped_physics import simulate_quadruped
from .quadruped_render import render_observer
from .rover import _depth, _save_result
from .runtime import _materialize, _paths, _publish


def _sensors(root, world, request, records, torch):
    for name in ("frames", "depth-frames", "observer"):
        (root / name).mkdir(exist_ok=True)
    poses = np.asarray([r["camera_to_world"] for r in records], dtype=np.float32)
    observer = np.asarray(
        [r["observer_camera_to_world"] for r in records], dtype=np.float32
    )
    focal = request.width / (2 * np.tan(np.deg2rad(60) / 2))
    intrinsic = np.array(
        [[focal, 0, request.width / 2], [0, focal, request.height / 2], [0, 0, 1]],
        dtype=np.float32,
    )
    hero = render_observer(root, world, request, observer, intrinsic, torch)
    rgb = _capture(root, world, request, poses, intrinsic, torch)
    depth = _depth(root, world, request, poses, intrinsic)
    events = (
        np.array(rgb["cuda_frame_ms"])
        + depth["cuda_frame_ms"]
        + hero["render"]["cuda_frame_ms"]
    )
    return (
        poses,
        intrinsic,
        {
            "rgb": rgb,
            "depth": depth,
            "observer": hero,
            "cuda_frame_ms": events.tolist(),
        },
    )


def process_quadruped(root, request):
    """Run actual articulated physics, the upstream policy, and GPU sensor/render work.

    Args: Materialized World API bundle directory and QuadrupedRequest.
    Returns: Complete measured result with artifact hashes.
    Raises: MarbleError on unsupported world, motion, CUDA, or renderer evidence.
    """
    world = json.loads((root / "world.json").read_text())
    if (
        world["source_kind"] != "world-api"
        or world["mesh_transform"] != world["splat_transform"]
    ):
        raise MarbleError("Quadruped collection requires an aligned World API collider")
    torch, started = _cuda(), time.perf_counter()
    _, assets = acquire_robot(root)
    records, physics = simulate_quadruped(root, world, request)
    poses, intrinsic, metrics = _sensors(root, world, request, records, torch)
    metrics["physics"] = physics
    result = _result_metadata(
        request, "quadruped", world, torch, metrics, poses, intrinsic, started
    )
    result["robot_assets"] = assets
    result["limitations"] = (
        "Synthetic Marble world; approximate geometry. CPU rigid-body dynamics and pretrained policy inference; CUDA sensors and CUDA robot rendering. CPU denoising. Authored lighting. No physical robot or new policy training."
    )
    return _save_result(root, result)


def quadruped_collect(request):
    """Execute headless quadruped capture through the shared S3 artifact contract.

    Args: QuadrupedRequest with input/output paths and capture parameters.
    Returns: Structured measured GPU, physics, and artifact summary.
    Raises: MarbleError or runtime error when a real component fails.
    """
    _paths(request)
    with tempfile.TemporaryDirectory(prefix="npa-marble-go1-") as directory:
        root = Path(directory) / "collection"
        root.mkdir()
        _materialize(request.input_path, "world.json", root)
        result = process_quadruped(root, request)
        _publish(root, request.output_path)
    return {
        "run_id": request.run_id,
        "frames": result["frames"],
        "gpu": result["gpu"],
        "physics": result["metrics"]["physics"],
    }
