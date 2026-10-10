"""Raycast actual collision triangles with NVIDIA Warp CUDA kernels."""

import numpy as np
import warp as wp


@wp.kernel
def _rays(
    mesh: wp.uint64,
    camera: wp.array2d(dtype=float),
    focal_x: float,
    focal_y: float,
    center_x: float,
    center_y: float,
    width: int,
    depth: wp.array(dtype=float),
):
    index = wp.tid()
    x = float(index % width) + 0.5
    y = float(index // width) + 0.5
    local = wp.normalize(
        wp.vec3((x - center_x) / focal_x, (y - center_y) / focal_y, 1.0)
    )
    direction = wp.vec3(
        camera[0, 0] * local[0] + camera[0, 1] * local[1] + camera[0, 2] * local[2],
        camera[1, 0] * local[0] + camera[1, 1] * local[1] + camera[1, 2] * local[2],
        camera[2, 0] * local[0] + camera[2, 1] * local[1] + camera[2, 2] * local[2],
    )
    origin = wp.vec3(camera[0, 3], camera[1, 3], camera[2, 3])
    result = wp.mesh_query_ray(mesh, origin, direction, 1000.0)
    depth[index] = -1.0
    if result.result:
        depth[index] = result.t


def _load_collider(path, transform, device):
    import trimesh

    scene = trimesh.load(str(path), force="scene")
    mesh = scene.to_geometry()
    vertices = (
        np.asarray(mesh.vertices, dtype=np.float32) * transform["scale"]
        + transform["translation"]
    )
    faces = np.asarray(mesh.faces, dtype=np.int32)
    if len(faces) == 0 or not np.isfinite(vertices).all():
        raise ValueError("Collider must contain finite triangle geometry")
    collider = wp.Mesh(
        points=wp.array(vertices, dtype=wp.vec3, device=device),
        indices=wp.array(faces.flatten(), dtype=wp.int32, device=device),
    )
    return collider, len(faces)


def _cast_frame(collider, pose, intrinsic, width, output, device, torch):
    camera = wp.array(pose, dtype=float, device=device)
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    wp.synchronize_device(device)
    start.record()
    wp.launch(
        _rays,
        dim=output.size,
        inputs=[
            collider.id,
            camera,
            float(intrinsic[0, 0]),
            float(intrinsic[1, 1]),
            float(intrinsic[0, 2]),
            float(intrinsic[1, 2]),
            width,
            output,
        ],
        device=device,
        stream=wp.stream_from_torch(torch.cuda.current_stream()),
    )
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def scan_mesh(path, transform, poses, intrinsic, width, height):
    """Measure ray distance for every pixel of every camera on CUDA.

    Args: GLB path, world transform, camera poses/intrinsics, and dimensions.
    Returns: Depth arrays, CUDA event durations, and triangle count.
    Raises: ValueError on invalid geometry; RuntimeError on absent CUDA.
    """
    import torch

    device = wp.get_device("cuda:0")
    collider, triangle_count = _load_collider(path, transform, device)
    output = wp.empty(width * height, dtype=float, device=device)
    depths, timing = [], []
    for pose in poses:
        timing.append(
            _cast_frame(collider, pose, intrinsic, width, output, device, torch)
        )
        result = output.numpy().reshape(height, width).copy()
        result[result < 0] = np.nan
        depths.append(result)
    return np.stack(depths), timing, triangle_count
