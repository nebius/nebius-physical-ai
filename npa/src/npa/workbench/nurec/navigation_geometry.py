"""Validate registered static geometry and author exact USD triangle colliders."""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np


def rigid_transform(value: Any) -> np.ndarray:
    """Validate a row-vector source-meters to Z-up world-meters transform.

    Args:
        value: Four-by-four numeric rigid transformation matrix.
    Returns:
        Finite proper rigid transform.
    Raises:
        ValueError: Matrix is missing, singular, scaled, reflected, or nonfinite.
    """
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("registration requires a finite 4x4 rigid matrix")
    if not np.allclose(matrix[:, 3], [0, 0, 0, 1], atol=1e-8):
        raise ValueError("registration must use USD row-vector affine convention")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-8):
        raise ValueError("registration must be rigid; author scale with metersPerUnit")
    if not math.isclose(np.linalg.det(rotation), 1, abs_tol=1e-8):
        raise ValueError("registration must preserve orientation")
    return matrix


def validate_probes(value: Any) -> list[dict]:
    """Validate explicit expected physics ray intersections in world meters.

    Args:
        value: Nonempty list of origin/direction/distance interval records.
    Returns:
        Validated probe records.
    Raises:
        ValueError: Missing, malformed, nonfinite, or unbounded expectation.
    """
    if not isinstance(value, list) or not value:
        raise ValueError("ray_probes requires at least one expected mesh intersection")
    for probe in value:
        if not isinstance(probe, dict):
            raise ValueError("ray probe must be an object")
        vectors = np.asarray([probe.get("origin"), probe.get("direction")], dtype=float)
        if vectors.shape != (2, 3) or not np.isfinite(vectors).all():
            raise ValueError("ray probe vectors must be finite triples")
        if not math.isclose(np.linalg.norm(vectors[1]), 1, abs_tol=1e-6):
            raise ValueError("ray probe direction must be a unit vector")
        bounds = [probe.get("min_distance"), probe.get("max_distance")]
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in bounds):
            raise ValueError("ray probe distances must be finite numbers")
        if not 0 <= bounds[0] < bounds[1]:
            raise ValueError("ray probe distances require 0 <= min < max")
    return value


def open_static_stage(path):
    """Open a pre-audited static USD with explicit scale and coordinate metadata.

    Args:
        path: Dependency-audited USD or USDZ path.
    Returns:
        Stage with a transformable default prim and explicit metric scale.
    Raises:
        ValueError: Stage, metadata, transforms, or physics are unsupported.
    """
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(path))
    if not stage or stage.GetCompositionErrors():
        raise ValueError("USD composition failed")
    root = stage.GetDefaultPrim()
    if not root or not root.IsA(UsdGeom.Xformable):
        raise ValueError("USD requires an Xformable default prim")
    _check_stage_metadata(stage)
    if list(stage.GetPseudoRoot().GetChildren()) != [root]:
        raise ValueError("all scene content must be beneath the default prim")
    for prim in stage.Traverse():
        _check_source_prim(prim)
    return stage


def _check_stage_metadata(stage) -> None:
    from pxr import UsdGeom

    scale = UsdGeom.GetStageMetersPerUnit(stage)
    if (
        not stage.HasAuthoredMetadata("metersPerUnit")
        or not math.isfinite(scale)
        or scale <= 0
    ):
        raise ValueError("USD requires explicit positive finite metersPerUnit")
    if not stage.HasAuthoredMetadata("upAxis") or UsdGeom.GetStageUpAxis(stage) not in (
        "Y",
        "Z",
    ):
        raise ValueError("USD requires explicit Y or Z upAxis")


def _check_source_prim(prim) -> None:
    from pxr import UsdGeom, UsdPhysics

    if prim.IsInstance() or prim.IsA(UsdGeom.PointInstancer):
        raise ValueError("instanced geometry is unsupported; expand instances first")
    authored = prim.GetMetadata("apiSchemas")
    schemas = set(prim.GetAppliedSchemas())
    if authored:
        schemas.update(authored.GetAppliedItems())
    if any(
        "physics" in schema.lower() or "physx" in schema.lower() for schema in schemas
    ):
        raise ValueError(
            "source USD must not contain preexisting physics; supply geometry only"
        )
    if prim.IsA(UsdPhysics.Scene) or prim.IsA(UsdPhysics.Joint):
        raise ValueError("source USD must not contain a physics scene or joint")
    _check_transform(prim)


def _check_transform(prim) -> None:
    from pxr import UsdGeom

    xform = UsdGeom.Xformable(prim)
    if not xform:
        return
    if xform.GetResetXformStack():
        raise ValueError("resetXformStack would break registration")
    matrix = np.asarray(xform.GetLocalTransformation())
    if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
        raise ValueError("source transform is nonfinite or singular")
    if not np.allclose(matrix[:, 3], [0, 0, 0, 1], atol=1e-12):
        raise ValueError("source transform must be affine")


def _triangles(mesh) -> tuple[np.ndarray, np.ndarray]:
    if mesh.GetSubdivisionSchemeAttr().Get() != "none":
        raise ValueError("collision mesh subdivisionScheme must be none")
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("collision mesh requires finite 3D points")
    if counts.ndim != 1 or not counts.size or not np.all(counts == 3):
        raise ValueError("collision mesh must contain explicitly triangulated faces")
    if (
        indices.size != counts.size * 3
        or indices.min() < 0
        or indices.max() >= len(points)
    ):
        raise ValueError("collision mesh face indices are invalid")
    if mesh.GetHoleIndicesAttr().Get():
        raise ValueError("collision mesh holes must be triangulated explicitly")
    triangles = indices.reshape(-1, 3)
    vertices = points[triangles]
    area = np.linalg.norm(
        np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]),
        axis=1,
    )
    if not np.isfinite(area).all() or np.any(area <= 0):
        raise ValueError("collision mesh contains degenerate triangles")
    return points, triangles


def author_colliders(source, target, registration: np.ndarray) -> list[dict]:
    """Bake registered collision meshes into static metric triangle colliders.

    Args:
        source: Validated source stage with triangulated USD meshes.
        target: Output stage in meters with Z up.
        registration: Source-meters to world-meters row-vector transform.
    Returns:
        Collider paths, measured bounds, and triangle counts.
    Raises:
        ValueError: Geometry is absent, nontriangulated, or invalid.
    """
    from pxr import Usd, UsdGeom

    meshes = [
        UsdGeom.Mesh(prim)
        for prim in Usd.PrimRange(source.GetDefaultPrim())
        if prim.IsA(UsdGeom.Mesh)
    ]
    if not meshes:
        raise ValueError(
            "explicit collision mesh required; Gaussian splats are not collision geometry"
        )
    records = []
    for index, mesh in enumerate(meshes):
        points, triangles = _registered_triangles(source, mesh, registration)
        records.append(_author_mesh(target, index, points, triangles))
    return records


def _registered_triangles(source, mesh, registration):
    from pxr import UsdGeom

    points, triangles = _triangles(mesh)
    local = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(mesh.GetPrim()))
    vertices = np.column_stack([points, np.ones(len(points))]) @ local
    vertices[:, :3] *= UsdGeom.GetStageMetersPerUnit(source)
    world = (vertices @ registration)[:, :3]
    if not np.isfinite(world).all() or np.max(np.abs(world)) > np.finfo(np.float32).max:
        raise ValueError(
            "registered collision vertices exceed finite USD point precision"
        )
    if mesh.GetOrientationAttr().Get() == "leftHanded":
        triangles = triangles[:, ::-1]
    if np.linalg.det(local[:3, :3]) < 0:
        triangles = triangles[:, ::-1]
    return world, triangles


def _geometry_sha256(points, triangles) -> str:
    digest = hashlib.sha256(np.asarray(points, dtype="<f4").tobytes())
    digest.update(np.asarray(triangles, dtype="<i4").tobytes())
    return digest.hexdigest()


def _author_mesh(stage, index, points, triangles) -> dict:
    from pxr import UsdGeom, UsdPhysics

    path = f"/World/Collision/Mesh_{index}"
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(points.tolist())
    mesh.CreateFaceVertexCountsAttr([3] * len(triangles))
    mesh.CreateFaceVertexIndicesAttr(triangles.flatten().tolist())
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateVisibilityAttr("invisible")
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("none")
    return {
        "path": path,
        "triangles": len(triangles),
        "geometry_sha256": _geometry_sha256(points, triangles),
        "bounds_m": [points.min(axis=0).tolist(), points.max(axis=0).tolist()],
    }
