"""Collect aligned GPU observations along a physically simulated rover trajectory."""

import hashlib
import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .api import MarbleError
from .gpu import _capture, _cuda, _point_cloud, _result_metadata, _save_frame
from .rover_physics import simulate_rover
from .runtime import _materialize, _paths, _publish


def _depth(root, world, request, poses, intrinsic):
    from .raycast import scan_mesh

    depths, timing, triangles = scan_mesh(
        root / "collider.glb",
        world["mesh_transform"],
        poses,
        intrinsic,
        request.width,
        request.height,
    )
    hits = np.isfinite(depths) & (depths > 0)
    if hits.mean() < 0.5:
        raise MarbleError("Sensor depth coverage is below 50%; inspect mesh alignment")
    np.savez_compressed(
        root / "depth.npz", depth=depths, camera_to_world=poses, intrinsics=intrinsic
    )
    ceiling = float(np.percentile(depths[hits], 95))
    for index, frame in enumerate(depths):
        normalized = np.nan_to_num(np.clip(frame / ceiling, 0, 1), nan=0)
        rgb = np.stack(
            [normalized**0.5, np.sin(normalized * np.pi), 1 - normalized], axis=-1
        )
        rgb[~hits[index]] = 0
        _save_frame(rgb, root / "depth-frames" / f"{index:04d}.jpg")
    _point_cloud(root, depths, poses, intrinsic)
    return {
        "engine": "nvidia-warp",
        "cuda_frame_ms": timing,
        "triangles": triangles,
        "rays": int(depths.size),
        "hits": int(hits.sum()),
        "coverage": hits.mean(axis=(1, 2)).tolist(),
        "range_definition": "Euclidean ray distance in provider-estimated meters; no return is NaN",
    }


def _observer(root, world, request, poses, intrinsic, torch):
    from .raycast import scan_mesh

    render = _capture(root, world, request, poses, intrinsic, torch, "observer")
    depths, timing, _ = scan_mesh(
        root / "collider.glb",
        world["mesh_transform"],
        poses,
        intrinsic,
        request.width,
        request.height,
    )
    yy, xx = np.mgrid[: request.height, : request.width]
    ray_scale = np.sqrt(
        1
        + ((xx + 0.5 - intrinsic[0, 2]) / intrinsic[0, 0]) ** 2
        + ((yy + 0.5 - intrinsic[1, 2]) / intrinsic[1, 1]) ** 2
    )
    for index, scene_depth in enumerate(depths):
        actor = np.load(root / "actors" / f"{index:04d}.npz")
        visible = actor["mask"] & (
            ~np.isfinite(scene_depth) | (actor["z"] * ray_scale < scene_depth + 0.03)
        )
        path = root / "observer" / f"{index:04d}.jpg"
        background = np.asarray(Image.open(path)).copy()
        background[visible] = actor["rgb"][visible]
        Image.fromarray(background).save(path, quality=92)
    shutil.rmtree(root / "actors")
    return {
        "render": render,
        "occlusion_cuda_frame_ms": timing,
        "composition": "gsplat CUDA background, TinyRenderer CPU rover at recorded rigid-body pose, Warp CUDA mesh depth occlusion",
    }


def _metrics(capture, depth, observer, physics):
    measured = np.array(capture["cuda_frame_ms"]) + depth["cuda_frame_ms"]
    measured += observer["render"]["cuda_frame_ms"]
    measured += observer["occlusion_cuda_frame_ms"]
    return {
        "cuda_frame_ms": measured.tolist(),
        "rgb": capture,
        "depth": depth,
        "observer": observer,
        "physics": physics,
    }


def _save_result(root, result):
    if result["kind"] == "rover":
        result["limitations"] = (
            "Synthetic warehouse and approximate metric geometry. CPU rigid-body simulation; GPU RGB and depth. No physical robot, trained policy, or measured sim-to-real success."
        )
    result["files"] = {
        str(path.relative_to(root)): {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "result.json"
    }
    (root / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    return result


def process_rover(root, request):
    """Execute physics and CUDA sensor kernels and persist auditable data.

    Args: Verified local bundle and RoverRequest.
    Returns: A result manifest with hashes, poses, contacts, and CUDA timings.
    Raises: MarbleError for unaligned geometry, absent CUDA, or invalid motion.
    """
    world = json.loads((root / "world.json").read_text())
    if (
        world["source_kind"] != "world-api"
        or world["mesh_transform"] != world["splat_transform"]
    ):
        raise MarbleError(
            "Rover requires a generated world with aligned metric mesh and splats"
        )
    torch = _cuda()
    started = time.perf_counter()
    records, physics = simulate_rover(root, world, request)
    for name in ["frames", "depth-frames", "observer"]:
        (root / name).mkdir(exist_ok=True)
    poses = np.asarray([row["camera_to_world"] for row in records], dtype=np.float32)
    observer_poses = np.asarray(
        [row["observer_camera_to_world"] for row in records], dtype=np.float32
    )
    focal = request.width / (2 * np.tan(np.deg2rad(75) / 2))
    intrinsic = np.array(
        [[focal, 0, request.width / 2], [0, focal, request.height / 2], [0, 0, 1]],
        dtype=np.float32,
    )
    capture = _capture(root, world, request, poses, intrinsic, torch)
    depth = _depth(root, world, request, poses, intrinsic)
    observer = _observer(root, world, request, observer_poses, intrinsic, torch)
    metrics = _metrics(capture, depth, observer, physics)
    result = _result_metadata(
        request, "rover", world, torch, metrics, poses, intrinsic, started
    )
    return _save_result(root, result)


def rover_collect(request):
    """Run headless embodied collection through the common Workbench S3 contract.

    Args: RoverRequest with world and output S3 prefixes.
    Returns: Actual device, frame count, and traveled distance.
    Raises: MarbleError for invalid inputs, physics, CUDA, or storage evidence.
    """
    _paths(request)
    with tempfile.TemporaryDirectory(prefix="npa-marble-rover-") as directory:
        root = Path(directory)
        _materialize(request.input_path, "world.json", root)
        result = process_rover(root, request)
        _publish(root, request.output_path)
    return {
        "run_id": request.run_id,
        "frames": result["frames"],
        "gpu": result["gpu"],
        "physics": result["metrics"]["physics"],
    }
