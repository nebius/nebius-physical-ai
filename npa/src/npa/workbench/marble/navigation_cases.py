"""Derive supported navigation resets and physical controls from a Marble mesh."""

from collections import deque
import math

import numpy as np

from .api import MarbleError

_FOOTPRINT = np.array([[0, 0], [-0.4, -0.3], [-0.4, 0.3], [0.4, -0.3], [0.4, 0.3]])


def _hits(client, origins, ends):
    result = []
    for offset in range(0, len(origins), 4096):
        result.extend(
            client.rayTestBatch(
                origins[offset : offset + 4096], ends[offset : offset + 4096]
            )
        )
    return result


def _supported(client, points):
    footprint = (points[:, None, :] + _FOOTPRINT).reshape(-1, 2)
    starts = np.column_stack((footprint, np.full(len(footprint), 0.8)))
    ends = np.column_stack((footprint, np.full(len(footprint), -0.5)))
    hits = _hits(client, starts.tolist(), ends.tolist())
    floor = np.array([hit[3][2] if hit[0] >= 0 else np.nan for hit in hits]).reshape(
        -1, 5
    )
    normals = np.array([hit[4][2] for hit in hits]).reshape(-1, 5)
    valid = np.all(np.isfinite(floor) & (abs(floor) < 0.35) & (normals > 0.85), axis=1)
    valid &= np.ptp(np.nan_to_num(floor), axis=1) < 0.04
    return valid, floor[:, 0]


def _clearance(client, points, floor):
    angles = np.arange(32) * math.tau / 32
    directions = np.column_stack((np.cos(angles), np.sin(angles), np.zeros(32)))
    clear = np.ones(len(points), dtype=bool)
    for height in (0.2, 0.6):
        origins = np.repeat(np.column_stack((points, floor + height)), 32, axis=0)
        ends = origins + np.tile(directions, (len(points), 1)) * 0.7
        hits = _hits(client, origins.tolist(), ends.tolist())
        clear &= np.array([hit[0] < 0 for hit in hits]).reshape(-1, 32).all(axis=1)
    return clear


def _connected(mask):
    unseen = set(map(tuple, np.argwhere(mask)))
    components = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        queue, component = deque([start]), [start]
        while queue:
            x, y = queue.popleft()
            for neighbor in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    queue.append(neighbor)
                    component.append(neighbor)
        components.append(component)
    if not components:
        raise MarbleError(
            "Marble collider has no supported connected navigation region"
        )
    return sorted(max(components, key=len))


def _grid(client, radius):
    axis = np.arange(-radius, radius + 0.2, 0.4)
    xx, yy = np.meshgrid(axis, axis)
    points = np.column_stack((xx.ravel(), yy.ravel()))
    valid, floor = _supported(client, points)
    valid &= np.linalg.norm(points, axis=1) <= radius
    indices = np.flatnonzero(valid)
    valid[indices] &= _clearance(client, points[indices], floor[indices])
    connected = _connected(valid.reshape(xx.shape))
    indices = np.array([row * len(axis) + column for row, column in connected])
    return np.column_stack((points[indices], floor[indices]))


def _case(name, seed, start, goal, heading=None):
    yaw = math.atan2(goal[1] - start[1], goal[0] - start[0])
    return {
        "id": name,
        "seed": int(seed),
        "position_m": [*map(float, start[:2]), float(start[2] + 0.6)],
        "goal_m": list(map(float, goal[:2])),
        "heading_rad": float(yaw if heading is None else heading),
    }


def _split(grid, count, seed):
    generator = np.random.default_rng(seed)
    order = generator.permutation(len(grid))
    cut = len(order) * 4 // 5
    if cut < 2 or len(order) - cut < 2:
        raise MarbleError(
            "Navigation requires enough separate train and evaluation goal locations"
        )
    splits = []
    for label, goals in (("train", order[:cut]), ("eval", order[cut:])):
        pairs = [
            (i, int(j))
            for i in range(len(grid))
            for j in goals
            if np.linalg.norm(grid[i, :2] - grid[j, :2]) >= 2.0
        ]
        if len(pairs) < count:
            raise MarbleError(
                "Marble collider has insufficient distinct supported navigation cases"
            )
        selected = generator.choice(len(pairs), count, replace=False)
        base = seed + (0 if label == "train" else count)
        splits.append(
            [
                _case(
                    f"{label}-{k}",
                    base + k,
                    grid[pairs[index][0]],
                    grid[pairs[index][1]],
                )
                for k, index in enumerate(selected)
            ]
        )
    return splits


def _free_probe(client, grid):
    for point in grid:
        for yaw in np.arange(8) * math.tau / 8:
            direction = np.array([math.cos(yaw), math.sin(yaw)])
            path = point[:2] + np.arange(0, 5.25, 0.25)[:, None] * direction
            valid, floor = _supported(client, path)
            if valid.all() and _clearance(client, path, floor).all():
                return _case("free", 42, point, point[:2] + 5 * direction), path
    raise MarbleError("Marble collider lacks a supported five-meter free-space control")


def _obstacle_probe(client, grid):
    for point in grid:
        for yaw in np.arange(32) * math.tau / 32:
            direction = np.array([math.cos(yaw), math.sin(yaw), 0])
            starts = [point + [0, 0, height] for height in (0.3, 0.6)]
            hits = _hits(
                client,
                np.asarray(starts).tolist(),
                (np.asarray(starts) + direction * 2).tolist(),
            )
            distances = [hit[2] * 2 for hit in hits]
            if all(hit[0] >= 0 and abs(hit[4][2]) < 0.4 for hit in hits):
                if (
                    all(0.9 < distance < 1.8 for distance in distances)
                    and abs(distances[0] - distances[1]) < 0.2
                ):
                    return _case(
                        "obstacle", 43, point, point[:2] + direction[:2] * 4, yaw
                    )
    raise MarbleError(
        "Marble collider lacks a supported obstacle-contact positive control"
    )


def _probes(client, grid):
    free, path = _free_probe(client, grid)
    distances = np.linalg.norm(grid[:, None, :2] - path[None, :, :], axis=2).min(axis=1)
    point = grid[int(distances.argmax())]
    if distances.max() < 3:
        raise MarbleError(
            "Marble collider lacks a separated supported peer parking location"
        )
    return {
        "free": free,
        "obstacle": _obstacle_probe(client, grid),
        "parked": _case("parked", 44, point, point[:2] + [1, 0]),
        "actions": [[0.6, 0.0, 0.0]] * 30,
        "tolerance": 0.001,
    }


def build_cases(vertices, faces, *, count, seed, radius):
    """Measure supported resets with disjoint goal locations and native control cases.

    Args: Exact Z-up mesh vertices/faces, cases per split, random seed, and sampling radius.
    Returns: Native reset fields and measured preparation evidence.
    Raises: MarbleError when mesh support, clearance, splits, or control requirements fail.
    """
    import pybullet
    from pybullet_utils.bullet_client import BulletClient

    client = BulletClient(connection_mode=pybullet.DIRECT)
    try:
        shape = client.createCollisionShape(
            pybullet.GEOM_MESH,
            vertices=vertices.tolist(),
            indices=faces.reshape(-1).tolist(),
            flags=pybullet.GEOM_FORCE_CONCAVE_TRIMESH,
        )
        client.createMultiBody(baseMass=0, baseCollisionShapeIndex=shape)
        grid = _grid(client, radius)
        train, evaluation = _split(grid, count, seed)
        probes = _probes(client, grid)
    finally:
        client.disconnect()
    evidence = {
        "supported_positions": grid.tolist(),
        "train_cases": count,
        "eval_cases": count,
        "goal_locations_disjoint": True,
        "grid_spacing_m": 0.4,
        "sampling_radius_m": radius,
        "scope": "CPU mesh support checks; native physics, controls, and learning remain unverified",
    }
    return {"train_cases": train, "eval_cases": evaluation, "probe": probes}, evidence
