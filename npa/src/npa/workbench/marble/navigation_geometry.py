"""Convert a verified Marble collider to the native navigation scene frame."""

import json
from pathlib import Path

import numpy as np

from .api import MarbleError


def collision_geometry(root: Path, world: dict):
    """Read the actual generated collision mesh in provider-scaled Z-up coordinates.

    Args: Materialized world directory and verified world manifest.
    Returns: Finite vertices and triangle indices in the native navigation frame.
    Raises: MarbleError for unsupported provenance, transforms, or geometry.
    """
    import trimesh

    if world.get("source_kind") != "world-api":
        raise MarbleError("Navigation requires a World Labs API world")
    transform = world.get("mesh_transform", {})
    if transform != world.get("splat_transform"):
        raise MarbleError("Navigation requires aligned splat and collider transforms")
    scale = np.asarray(transform.get("scale"), dtype=float)
    translation = np.asarray(transform.get("translation"), dtype=float)
    if scale.shape != (3,) or translation.shape != (3,):
        raise MarbleError(
            "World transforms must contain three scale and translation values"
        )
    if (
        not np.isfinite(scale).all()
        or not np.isfinite(translation).all()
        or not np.all(scale)
    ):
        raise MarbleError("World transforms must be finite and invertible")
    mesh = trimesh.load(root / "collider.glb", force="scene").to_geometry()
    vertices = np.asarray(mesh.vertices) * scale + translation
    vertices = vertices[:, [0, 2, 1]] * [1, -1, 1]
    faces = np.asarray(mesh.faces, dtype=np.int32)
    if not np.isfinite(vertices).all() or not len(faces):
        raise MarbleError("Navigation requires finite collision triangles")
    return vertices, faces


def write_scene(path: Path, vertices, faces, provenance: dict):
    """Package the original triangles as a static native USDZ collision scene.

    Args: Destination USDZ, transformed vertices, triangle indices, and source hashes.
    Returns: None.
    Raises: RuntimeError on failed USD packaging; ImportError without OpenUSD.
    """
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils

    layer = path.with_suffix(".usda")
    stage = Usd.Stage.CreateNew(str(layer))
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/World").GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, "Z")
    stage.GetRootLayer().customLayerData = {
        "npa_marble_source": json.dumps(provenance, sort_keys=True)
    }
    mesh = UsdGeom.Mesh.Define(stage, "/World/Collision/Warehouse")
    mesh.CreatePointsAttr(vertices.tolist())
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr(faces.reshape(-1).tolist())
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDisplayColorAttr([(0.38, 0.48, 0.58)])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("none")
    stage.GetRootLayer().Save()
    if not UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(str(layer)), str(path)):
        raise RuntimeError("Could not package the Marble navigation scene")
    layer.unlink()
