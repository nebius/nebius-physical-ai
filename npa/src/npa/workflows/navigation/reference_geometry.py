"""Derive one static ray mesh from the exact imported scene's collision geometry."""


def spawn_scene(prim_path, cfg, translation=None, orientation=None):
    """Spawn one USDZ and derive a noncolliding sensor mesh from its real colliders.

    Args:
        prim_path: Global warehouse root.
        cfg: Native UsdFileCfg with the sealed asset path.
        translation: Optional root translation.
        orientation: Optional root quaternion.
    Returns:
        Spawned USD scene prim.
    Raises:
        ValueError: Collision geometry is missing or is not static triangle meshes.
    """
    from isaaclab.sim import spawn_from_usd
    from pxr import Usd, UsdGeom, UsdPhysics

    root = spawn_from_usd(prim_path, cfg, translation, orientation)
    points, faces = [], []
    stage = root.GetStage()
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        if not prim.GetPath().HasPrefix(root.GetPath()):
            continue
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError(
                "reference scene must contain only static collision geometry"
            )
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            if not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
                continue
            if not prim.IsA(UsdGeom.Mesh):
                raise ValueError("reference scene colliders must be triangle meshes")
            _append_mesh(prim, points, faces)
    if not faces:
        raise ValueError("reference scene has no collision mesh triangles")
    _sensor_mesh(stage, prim_path, points, faces)
    return root


def validate_scene_frame(scene_file):
    """Reject source geometry whose coordinate frame differs from metric Z-up cases.

    Args:
        scene_file: Local self-contained USDZ.
    Returns:
        None.
    Raises:
        ValueError: The stage is missing, has no default root or uses another frame.
    """
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(scene_file)
    if not stage or not stage.GetDefaultPrim():
        raise ValueError("reference scene requires a valid USD default prim")
    if (
        UsdGeom.GetStageMetersPerUnit(stage) != 1.0
        or UsdGeom.GetStageUpAxis(stage) != "Z"
    ):
        raise ValueError("reference scene must use meters and Z-up coordinates")


def _append_mesh(prim, points, faces):
    from pxr import Gf, UsdGeom

    mesh = UsdGeom.Mesh(prim)
    counts = mesh.GetFaceVertexCountsAttr().Get()
    indices = mesh.GetFaceVertexIndicesAttr().Get()
    if any(count != 3 for count in counts):
        raise ValueError("reference collision mesh must be triangulated")
    transform = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    offset = len(points)
    points.extend(
        transform.Transform(Gf.Vec3d(point)) for point in mesh.GetPointsAttr().Get()
    )
    faces.extend(int(index) + offset for index in indices)


def _sensor_mesh(stage, root, points, faces):
    from pxr import Gf, UsdGeom

    if stage.GetPrimAtPath(root + "/NpaRaycastMesh"):
        raise ValueError("source scene already occupies the reserved ray-mesh path")
    parent_inverse = (
        UsdGeom.XformCache()
        .GetLocalToWorldTransform(stage.GetPrimAtPath(root))
        .GetInverse()
    )
    mesh = UsdGeom.Mesh.Define(stage, root + "/NpaRaycastMesh")
    mesh.CreatePointsAttr(
        [Gf.Vec3f(parent_inverse.Transform(point)) for point in points]
    )
    mesh.CreateFaceVertexCountsAttr([3] * (len(faces) // 3))
    mesh.CreateFaceVertexIndicesAttr(faces)
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateVisibilityAttr("invisible")
