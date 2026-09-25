"""Build a public cluttered warehouse with real static triangle-mesh collision."""

import random


def warehouse(path):
    """Create one metric Z-up scene with floor, boundary walls and central racks.

    Args:
        path: Fresh local USDZ output path.
    Returns:
        None.
    Raises:
        RuntimeError: USD packaging fails.
    """
    from pxr import Sdf, Usd, UsdGeom, UsdUtils

    layer = path.with_suffix(".usda")
    stage = Usd.Stage.CreateNew(str(layer))
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/World").GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    points, indices = _warehouse_triangles()
    _collision_mesh(stage, points, indices)
    stage.GetRootLayer().Save()
    if not UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(str(layer)), str(path)):
        raise RuntimeError("could not package the public collision scene")
    layer.unlink()


def _warehouse_triangles():
    points, indices = [], []
    boxes = [
        ((0, 0, -0.1), (24, 16, 0.2)),
        ((0, -8, 1), (24, 0.2, 2)),
        ((0, 8, 1), (24, 0.2, 2)),
        ((-12, 0, 1), (0.2, 16, 2)),
        ((12, 0, 1), (0.2, 16, 2)),
        ((0, -2.75, 1), (2, 3, 2)),
        ((0, 2.75, 1), (2, 3, 2)),
    ]
    for center, size in boxes:
        _box(center, size, points, indices)
    return points, indices


def _collision_mesh(stage, points, indices):
    from pxr import UsdGeom, UsdPhysics

    mesh = UsdGeom.Mesh.Define(stage, "/World/Collision/Warehouse")
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([3] * (len(indices) // 3))
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDisplayColorAttr([(0.38, 0.48, 0.58)])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("none")


def _box(center, size, points, indices):
    offset = len(points)
    corners = [
        (-1, -1, -1),
        (1, -1, -1),
        (1, 1, -1),
        (-1, 1, -1),
        (-1, -1, 1),
        (1, -1, 1),
        (1, 1, 1),
        (-1, 1, 1),
    ]
    points.extend(
        tuple(center[d] + corner[d] * size[d] / 2 for d in range(3))
        for corner in corners
    )
    triangles = [
        (0, 2, 1),
        (0, 3, 2),
        (4, 5, 6),
        (4, 6, 7),
        (0, 1, 5),
        (0, 5, 4),
        (1, 2, 6),
        (1, 6, 5),
        (2, 3, 7),
        (2, 7, 6),
        (3, 0, 4),
        (3, 4, 7),
    ]
    indices.extend(offset + index for triangle in triangles for index in triangle)


def cases(count):
    """Define disjoint seeded training and held-out routes through real clutter.

    Args:
        count: Number of concurrent robots and held-out cases.
    Returns:
        Training cases, held-out cases and physical isolation controls.
    Raises:
        None.
    """
    return {
        "train_cases": [_case(index, 1000) for index in range(count)],
        "eval_cases": [_case(index, 1000000) for index in range(count)],
        "probe": {
            "free": _probe("free", [-8.0, -6.0, 0.6], [-4.0, -6.0]),
            "obstacle": _probe("obstacle", [-2.0, -3.0, 0.6], [3.0, -3.0]),
            "parked": _probe("parked", [-8.0, 6.0, 0.6], [-7.0, 6.0]),
            "actions": [[0.6, 0.0, 0.0]] * 30,
            "tolerance": 0.001,
        },
    }


def _case(index, offset):
    import math

    rng = random.Random(offset + index)
    direction = 1.0 if index % 2 == 0 else -1.0
    start = [-direction * rng.uniform(4.0, 6.0), rng.uniform(-5.0, 5.0), 0.6]
    goal = [direction * rng.uniform(4.0, 6.0), rng.uniform(-5.0, 5.0)]
    return {
        "id": f"route-{offset + index}",
        "seed": offset + index,
        "position_m": start,
        "goal_m": goal,
        "heading_rad": math.atan2(goal[1] - start[1], goal[0] - start[0]),
    }


def _probe(name, position, goal):
    return {
        "id": name,
        "seed": 42,
        "position_m": position,
        "heading_rad": 0.0,
        "goal_m": goal,
    }
