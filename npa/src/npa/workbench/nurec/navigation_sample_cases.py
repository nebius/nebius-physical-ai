"""Derive supported navigation resets from the measured public office surface."""

from __future__ import annotations

import math
from pathlib import Path
import random

import numpy as np

from npa.workbench.nurec.navigation_assets import sha256
from npa.workbench.nurec.navigation_sample_parking import select_parked_pose

_FOOTPRINT = np.array(
    [[0, 0], [-0.35, -0.23], [-0.35, 0.23], [0.35, -0.23], [0.35, 0.23]]
)


def _scene(surface_path):
    import open3d as o3d

    with np.load(surface_path, allow_pickle=False) as data:
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(data["points"]),
            o3d.utility.Vector3iVector(data["triangles"]),
        )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene


def _rays(scene, origins, directions):
    import open3d as o3d

    rays = np.column_stack((origins, directions)).astype(np.float32)
    return scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()


def _support(scene, points):
    heights = []
    for offset in _FOOTPRINT:
        origins = np.column_stack((points + offset, np.full(len(points), 0.3)))
        distances = _rays(scene, origins, np.tile([0, 0, -1], (len(points), 1)))
        heights.append(0.3 - distances)
    floor = np.array(heights)
    valid = np.all(np.isfinite(floor) & (abs(floor) < 0.1), axis=0)
    valid &= np.ptp(np.where(np.isfinite(floor), floor, 0), axis=0) < 0.04
    return valid, floor[0]


def _clearance(scene, points):
    angles = np.arange(0, 2 * np.pi, np.pi / 32)
    directions = np.column_stack((np.cos(angles), np.sin(angles), np.zeros(64)))
    clearance = np.full(len(points), np.inf)
    for height in (0.15, 0.4, 0.7):
        origins = np.repeat(
            np.column_stack((points, np.full(len(points), height))), 64, axis=0
        )
        distances = _rays(scene, origins, np.tile(directions, (len(points), 1)))
        clearance = np.minimum(clearance, distances.reshape(-1, 64).min(axis=1))
    return clearance


def _free_grid(scene):
    from scipy.ndimage import label

    xx, yy = np.meshgrid(np.arange(-3.5, 2.6, 0.15), np.arange(-2.5, 3.1, 0.15))
    points = np.column_stack((xx.ravel(), yy.ravel()))
    supported, floor = _support(scene, points)
    clearance = _clearance(scene, points)
    labels, count = label((supported & (clearance > 0.5)).reshape(xx.shape))
    if count == 0:
        raise ValueError("public scan has no measured connected support region")
    largest = max(
        range(1, count + 1), key=lambda value: np.count_nonzero(labels == value)
    )
    indices = np.flatnonzero(labels.ravel() == largest)
    return [
        {
            "xy_m": points[index].tolist(),
            "floor_z_m": float(floor[index]),
            "obstacle_clearance_m": float(clearance[index]),
        }
        for index in indices
    ]


def _supported_pairs(scene, pairs):
    origins = []
    for start, goal in pairs:
        yaw = math.atan2(
            goal["xy_m"][1] - start["xy_m"][1], goal["xy_m"][0] - start["xy_m"][0]
        )
        rotation = np.array(
            [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
        )
        xy = _FOOTPRINT @ rotation.T + start["xy_m"]
        origins.extend(np.column_stack((xy, np.full(5, start["floor_z_m"] + 0.6))))
    distance = _rays(scene, origins, np.tile([0, 0, -1], (len(origins), 1))).reshape(
        -1, 5
    )
    accepted = np.all(
        np.isfinite(distance) & (distance > 0.45) & (distance < 0.75), axis=1
    )
    accepted &= np.ptp(np.where(np.isfinite(distance), distance, 0), axis=1) < 0.04
    return [pair for pair, keep in zip(pairs, accepted, strict=True) if keep]


def _case(name, seed, position, goal, yaw=None):
    heading = math.atan2(goal[1] - position[1], goal[0] - position[0])
    return {
        "id": name,
        "seed": seed,
        "position_m": position,
        "goal_m": goal,
        "heading_rad": heading if yaw is None else yaw,
    }


def _probes(scene, surface_path, grid):
    probes = {
        "free": _case("free", 42, [-3.05, -2.2, 0.56624765], [-3.05, 1.4], math.pi / 2),
        "obstacle": _case(
            "obstacle", 43, [-3.05, -0.85, 0.59664259], [-0.2, 1.5], 0.687223393
        ),
        "actions": [[0.6, 0.0, 0.0]] * 30,
        "tolerance": 0.001,
    }
    position, parking = select_parked_pose(surface_path, scene, grid, probes["free"])
    probes["parked"] = _case("parked", 44, position, [position[0], 1.4], math.pi / 2)
    for name in ("free", "obstacle", "parked"):
        _probe_support(scene, probes[name])
    obstacle = probes["obstacle"]
    origin = obstacle["position_m"]
    yaw = obstacle["heading_rad"]
    distance = _rays(scene, [origin], [[math.cos(yaw), math.sin(yaw), 0]])[0]
    if not np.isfinite(distance) or not 0.7 < distance < 2.5:
        raise ValueError("public scan obstacle control lost its measured surface")
    return probes, parking


def _probe_support(scene, case):
    position = case["position_m"]
    yaw = case["heading_rad"]
    rotation = np.array(
        [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
    )
    origins = np.column_stack(
        (_FOOTPRINT @ rotation.T + position[:2], np.full(5, position[2]))
    )
    distance = _rays(scene, origins, np.tile([0, 0, -1], (5, 1)))
    if not np.all(np.isfinite(distance) & (distance > 0.45) & (distance < 0.75)):
        raise ValueError("public scan control footprint lacks measured support")


def build_cases(surface_path: Path, count: int = 4000) -> tuple[dict, dict]:
    """Generate disjoint train/evaluation cases on the actual reconstructed mesh.

    Args:
        surface_path: Measured public TUM office surface.npz from full TSDF fusion.
        count: Training case count and equally sized held-out case count.
    Returns:
        Typed recipe fields and a report of measured support and split provenance.
    Raises:
        ValueError: The scan cannot support enough distinct cases or native controls.
        ImportError: Open3D or SciPy is unavailable.
        OSError: The measured surface cannot be read.
    """
    if type(count) is not int or count < 2:
        raise ValueError("navigation sample requires at least two cases per split")
    scene = _scene(Path(surface_path))
    grid = _free_grid(scene)
    pairs = [
        (start, goal)
        for start in grid
        for goal in grid
        if math.dist(start["xy_m"], goal["xy_m"]) >= 1.0
    ]
    supported = _supported_pairs(scene, pairs)
    random.Random(260925).shuffle(supported)
    if len(supported) < 2 * count:
        raise ValueError("public scan has insufficient disjoint supported reset pairs")
    probes, parking = _probes(scene, surface_path, grid)
    cases = _split_cases(probes, supported, count)
    report = _support_report(surface_path, pairs, supported, grid, count)
    report["parked_support"] = parking
    return cases, report


def _split_cases(probes, supported, count):
    rows = [
        _case(
            f"office-{index}",
            3000000 + index,
            [*start["xy_m"], start["floor_z_m"] + 0.6],
            goal["xy_m"],
        )
        for index, (start, goal) in enumerate(supported[: 2 * count])
    ]
    return {
        "train_cases": rows[:count],
        "eval_cases": rows[count:],
        "probe": probes,
    }


def _support_report(surface_path, pairs, supported, grid, count):
    return {
        "schema": "npa.navigation.sample_cases.v1",
        "surface_sha256": sha256(Path(surface_path)),
        "candidate_pairs": len(pairs),
        "supported_pairs": len(supported),
        "train_cases": count,
        "eval_cases": count,
        "free_grid_points": grid,
        "method": "5 downward rays under rotated 0.70 x 0.46 m footprint; support 0.45–0.75 m; variation <0.04 m; 64 clearance rays at 0.15/0.4/0.7 m",
        "scope": "initial support only; native controls and rollouts must establish physical validity",
    }
