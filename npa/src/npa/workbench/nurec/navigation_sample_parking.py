"""Choose a stationary control pose from measured stance support and source topology."""

from __future__ import annotations

import math

import numpy as np

from npa.workflows.navigation.contact_surface_geometry import _topology

# A stationary native foot can reach beyond the five-ray reset footprint.
_HALF_EXTENTS = (0.65, 0.35)
_SPACING = 0.05
# Include source faces within more than the native 1.2 mm foot contact offset.
_MARGIN = 0.002
_HEIGHT_BAND = 0.042
_YAW = math.pi / 2


def _rotation(yaw):
    return np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])


def _stance_offsets():
    axes = [
        np.linspace(-half, half, round(2 * half / _SPACING) + 1)
        for half in _HALF_EXTENTS
    ]
    xx, yy = np.meshgrid(*axes)
    return np.column_stack((xx.ravel(), yy.ravel())) @ _rotation(_YAW).T


def _floor_evidence(distances, normals):
    floor = 0.3 - distances
    if not np.all(np.isfinite(floor) & (abs(floor) < 0.1)):
        return None
    if not np.all(np.isfinite(normals)) or np.min(normals[:, 2]) < 0.7:
        return None
    height_range = float(np.ptp(floor))
    if height_range >= 0.04:
        return None
    return {
        "supported_rays": len(floor),
        "height_range_m": height_range,
        "minimum_normal_z": float(np.min(normals[:, 2])),
        "support_rays": {"floor_z_m": floor.tolist(), "normals": normals.tolist()},
    }


def _support(scene, xy):
    import open3d as o3d

    footprint = _stance_offsets() + xy
    rays = np.column_stack(
        (
            footprint,
            np.full(len(footprint), 0.3),
            np.tile([0, 0, -1], (len(footprint), 1)),
        )
    )
    result = scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32)))
    evidence = _floor_evidence(
        result["t_hit"].numpy(), result["primitive_normals"].numpy()
    )
    if evidence is not None:
        evidence["support_rays"]["source_face_indices"] = (
            result["primitive_ids"].numpy().tolist()
        )
    return evidence


def _surface_index(surface_path):
    with np.load(surface_path, allow_pickle=False) as surface:
        points, triangles = surface["points"], surface["triangles"]
    neighbors, normals = _topology(points, triangles)
    faces = points[triangles]
    return (
        faces.min(axis=1),
        faces.max(axis=1),
        np.any(neighbors < 0, axis=1) | (normals[:, 2] < 0.7),
    )


def _topology_evidence(index, point):
    minimum, maximum, unsafe = index
    footprint = _stance_offsets() + point["xy_m"]
    lower = np.r_[footprint.min(axis=0) - _MARGIN, point["floor_z_m"] - _HEIGHT_BAND]
    upper = np.r_[footprint.max(axis=0) + _MARGIN, point["floor_z_m"] + _HEIGHT_BAND]
    nearby = np.all(maximum >= lower, axis=1) & np.all(minimum <= upper, axis=1)
    if not nearby.any() or np.any(unsafe & nearby):
        return None
    return {
        "nearby_source_faces": int(nearby.sum()),
        "unsafe_source_faces": 0,
        "source_query_min_m": lower.tolist(),
        "source_query_max_m": upper.tolist(),
    }


def _footprints_separated(xy, focal):
    corners = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]]) * _HALF_EXTENTS
    parked_rotation = _rotation(_YAW)
    focal_rotation = _rotation(focal["heading_rad"])
    parked = corners @ parked_rotation.T + xy
    free = corners @ focal_rotation.T + focal["position_m"][:2]
    for axis in np.column_stack((parked_rotation, focal_rotation)).T:
        parked_interval, free_interval = parked @ axis, free @ axis
        if (
            parked_interval.max() + _MARGIN < free_interval.min()
            or free_interval.max() + _MARGIN < parked_interval.min()
        ):
            return True
    return False


def _candidates(scene, grid, index, focal):
    accepted = []
    counts = {"grid": len(grid), "dense_support": 0, "topology": 0, "separated": 0}
    for point in grid:
        support = _support(scene, point["xy_m"])
        if support is None:
            continue
        counts["dense_support"] += 1
        topology = _topology_evidence(index, point)
        if topology is None:
            continue
        counts["topology"] += 1
        if not _footprints_separated(point["xy_m"], focal):
            continue
        counts["separated"] += 1
        distance = math.dist(point["xy_m"], focal["position_m"][:2])
        accepted.append(
            {**point, **support, **topology, "focal_origin_distance_m": distance}
        )
    return accepted, counts


def _report(selected, counts, position):
    return {
        "schema": "npa.navigation.parked_support.v1",
        "selection_rule": "farthest supported separated center from the free-control initial origin; ties use ascending x, y, floor z",
        "candidate_counts": counts,
        "position_m": position,
        "heading_rad": _YAW,
        "local_half_extents_m": list(_HALF_EXTENTS),
        "ray_spacing_m": _SPACING,
        "support_bounds": {
            "absolute_floor_z_below_m": 0.1,
            "height_range_below_m": 0.04,
            "minimum_normal_z": 0.7,
        },
        "topology_bounds": {
            "xy_margin_m": _MARGIN,
            "center_floor_half_height_m": _HEIGHT_BAND,
        },
        "measured": selected,
        "initial_footprints_separated": True,
        "scope": "measured initial parking support only; unchanged native controls must still pass before learning",
    }


def select_parked_pose(surface_path, scene, grid, focal):
    """Select a deterministic supported parking pose without altering the surface.

    Args:
        surface_path: Exact measured surface.npz used by the ray scene.
        scene: Open3D ray scene containing that same surface.
        grid: Measured points in the existing largest connected support region.
        focal: Unchanged free-control case, used for initial stance separation.
    Returns:
        Parked position and source-support evidence for publication before learning.
    Raises:
        ValueError: No candidate has dense, upward, boundary-free stance support.
        ImportError: Open3D is unavailable.
        OSError: The measured surface cannot be read.
    """
    index = _surface_index(surface_path)
    candidates, counts = _candidates(scene, grid, index, focal)
    if not candidates:
        raise ValueError(
            "public scan has no separated parked stance with measured topology support"
        )
    selected = min(
        candidates,
        key=lambda point: (
            -point["focal_origin_distance_m"],
            *point["xy_m"],
            point["floor_z_m"],
        ),
    )
    position = [*selected["xy_m"], selected["floor_z_m"] + 0.6]
    return position, _report(selected, counts, position)
