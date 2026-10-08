"""Render real Gaussian splats or raycast collision geometry on a required CUDA device."""

import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .api import MarbleError
from .cameras import camera_sweep, transform_splats


def _cuda():
    import torch

    if not torch.cuda.is_available():
        raise MarbleError(
            "This stage requires a real CUDA GPU; there is no CPU fallback"
        )
    torch.cuda.init()
    major, minor = torch.cuda.get_device_capability()
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    return torch


def _save_frame(array, path):
    pixels = np.clip(array * 255, 0, 255).astype(np.uint8)
    Image.fromarray(pixels).save(path, quality=92)


def _load_gaussians(root, world, torch):
    import spz

    cloud = spz.load_spz(str(root / "world.spz"))
    if cloud.num_points <= 0:
        raise MarbleError("SPZ decoder returned an empty world")
    values = transform_splats(cloud, world["splat_transform"])
    tensors = [
        torch.as_tensor(np.ascontiguousarray(v), dtype=torch.float32, device="cuda")
        for v in values
    ]
    means, logs, rotations, logits, colors = tensors
    return dict(
        means=means,
        scales=logs.exp(),
        quats=rotations[:, [3, 0, 1, 2]],
        opacities=logits.sigmoid(),
        colors=(colors * 0.28209479177387814 + 0.5).clamp(0, 1),
    )


def _capture(root, world, request, poses, intrinsic, torch):
    from gsplat import rasterization

    gaussians = _load_gaussians(root, world, torch)
    viewmats = torch.linalg.inv(torch.as_tensor(poses, device="cuda"))
    intrinsics = torch.as_tensor(intrinsic[None], device="cuda")
    milliseconds = []
    coverage = []
    for index, view in enumerate(viewmats):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        with torch.no_grad():
            rendered, alpha, _ = rasterization(
                **gaussians,
                viewmats=view[None],
                Ks=intrinsics,
                width=request.width,
                height=request.height,
                packed=True,
                render_mode="RGB",
                near_plane=0.05,
                far_plane=1000,
            )
        end.record()
        torch.cuda.synchronize()
        milliseconds.append(start.elapsed_time(end))
        coverage.append(float((alpha > 0.5).float().mean().item()))
        _save_frame(rendered[0].cpu().numpy(), root / f"frames/{index:04d}.jpg")
    return {
        "engine": "gsplat",
        "engine_version": importlib.metadata.version("gsplat"),
        "gaussians": len(gaussians["means"]),
        "cuda_frame_ms": milliseconds,
        "coverage": coverage,
        "color_mode": "SH degree 0; view-dependent harmonics omitted",
    }


def _scan(root, world, request, poses, intrinsic, torch):
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
    np.savez_compressed(
        root / "depth.npz", depth=depths, camera_to_world=poses, intrinsics=intrinsic
    )
    valid = depths[hits]
    if valid.size == 0:
        raise MarbleError("GPU raycast produced no mesh intersections")
    ceiling = float(np.percentile(valid, 95))
    for index, depth in enumerate(depths):
        normalized = np.where(hits[index], np.clip(depth / ceiling, 0, 1), 0)
        color = np.stack(
            [normalized**0.5, np.sin(normalized * np.pi), 1 - normalized], axis=-1
        )
        color[~hits[index]] = 0
        _save_frame(color, root / f"frames/{index:04d}.jpg")
    _point_cloud(root, depths, poses, intrinsic)
    return {
        "engine": "nvidia-warp",
        "engine_version": importlib.metadata.version("warp-lang"),
        "triangles": triangles,
        "rays": int(depths.size),
        "hits": int(hits.sum()),
        "cuda_frame_ms": timing,
        "coverage": hits.mean(axis=(1, 2)).tolist(),
        "range_percentiles": np.percentile(valid, [5, 25, 50, 75, 95]).tolist(),
        "range_units": world["units"],
        "range_definition": "Euclidean ray distance; no-return pixels are NaN",
    }


def _point_cloud(root, depths, poses, intrinsic):
    height, width = depths.shape[1:]
    yy, xx = np.mgrid[0:height:8, 0:width:8]
    rays = np.stack(
        [
            (xx + 0.5 - intrinsic[0, 2]) / intrinsic[0, 0],
            (yy + 0.5 - intrinsic[1, 2]) / intrinsic[1, 1],
            np.ones_like(xx),
        ],
        axis=-1,
    )
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    points = []
    for index in range(0, len(poses), max(1, len(poses) // 16)):
        distance = depths[index, ::8, ::8]
        local = rays * distance[..., None]
        hit = np.isfinite(distance) & (distance > 0)
        points.extend(
            (local[hit] @ poses[index, :3, :3].T + poses[index, :3, 3]).tolist()
        )
    (root / "scan-points.json").write_text(json.dumps(points, separators=(",", ":")))


def _result_metadata(request, kind, world, torch, metrics, poses, intrinsic, started):
    return {
        "schema_version": "npa.marble.result.v1",
        "kind": kind,
        "run_id": request.run_id,
        "world": world,
        "frames": request.frames,
        "width": request.width,
        "height": request.height,
        "gpu": {
            "name": torch.cuda.get_device_name(),
            "cuda": torch.version.cuda,
            "torch": torch.__version__,
            "compute_capability": list(torch.cuda.get_device_capability()),
            "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        },
        "wall_seconds": time.perf_counter() - started,
        "metrics": metrics,
        "camera_to_world": poses.tolist(),
        "intrinsics": intrinsic.tolist(),
        "camera_convention": "OpenCV camera axes; explicit camera-to-world matrices",
        "limitations": "Exploratory camera sweep, not a validated robot path or calibrated simulator.",
    }


def process_world(root: Path, request, kind):
    """Run CUDA work, persist camera outputs, and record actual device and timings.

    Args: A local verified bundle, RunRequest, and capture or scan kind.
    Returns: A report containing measured GPU evidence and output hashes.
    Raises: MarbleError if CUDA or valid world geometry is unavailable.
    """
    torch = _cuda()
    world = json.loads((root / "world.json").read_text())
    (root / "frames").mkdir(exist_ok=True)
    poses, intrinsic = camera_sweep(request.frames, request.width, request.height)
    started = time.perf_counter()
    operation = _capture if kind == "capture" else _scan
    metrics = operation(root, world, request, poses, intrinsic, torch)
    report = _result_metadata(
        request, kind, world, torch, metrics, poses, intrinsic, started
    )
    report["files"] = {
        str(p.relative_to(root)): {
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "bytes": p.stat().st_size,
        }
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name != "result.json"
    }
    (root / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return report
