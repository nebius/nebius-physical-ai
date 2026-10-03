"""Identify the actual static collision surface independently of USD package metadata."""

import hashlib


def collision_identity(scene):
    """Hash exact world-space triangles, independent of mesh ordering and ZIP headers.

    Args:
        scene: Local metric, Z-up static collision USDZ.
    Returns:
        Canonical triangle SHA-256 and measured triangle count.
    Raises:
        ValueError: Geometry is absent, nonfinite, dynamic or not triangulated.
        OSError: The sealed scene cannot be opened.
    """
    import numpy as np
    from pxr import Usd, UsdGeom, UsdPhysics
    from npa.workflows.navigation.reference_geometry import validate_scene_frame

    validate_scene_frame(str(scene))
    stage = Usd.Stage.Open(str(scene))
    triangles = []
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError("frozen replay geometry must be static")
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
            continue
        if not prim.IsA(UsdGeom.Mesh):
            raise ValueError("frozen replay colliders must be triangle meshes")
        triangles.append(_triangles(prim))
    if not triangles:
        raise ValueError("frozen replay scene has no collision geometry")
    rows = np.concatenate(triangles).reshape(-1, 9)
    rows = np.ascontiguousarray(rows, dtype="<f8")
    rows[rows == 0] = 0  # Canonicalize signed zero without rounding geometry.
    records = rows.view(np.dtype((np.void, 72))).reshape(-1)
    digest = hashlib.sha256(np.sort(records).tobytes()).hexdigest()
    return {"triangles_sha256": digest, "triangle_count": len(records)}


def _triangles(prim):
    import numpy as np
    from pxr import UsdGeom

    mesh = UsdGeom.Mesh(prim)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
    if not len(counts) or not np.all(counts == 3):
        raise ValueError("frozen replay collision mesh must be triangulated")
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    matrix = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim))
    points = points @ matrix[:3, :3] + matrix[3, :3]
    if not np.isfinite(points).all():
        raise ValueError("frozen replay geometry must be finite")
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
    triangles = points[indices]
    order = np.lexsort((triangles[:, :, 2], triangles[:, :, 1], triangles[:, :, 0]))
    cyclic = (order[:, :1] + np.arange(3)) % 3
    return np.take_along_axis(triangles, cyclic[:, :, None], axis=1)
