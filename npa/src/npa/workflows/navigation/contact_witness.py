"""Verify live sphere colliders and derive their terrain-side native contact witness."""

import math


def _sphere_record(stage, body_path, native_max_shapes):
    from pxr import Usd, UsdPhysics

    record = {
        "native_path": body_path,
        "collider_path": None,
        "geometry_type": None,
        "radius_m": 0.0,
        "scale": [0.0, 0.0, 0.0],
        "body_local_center_m": [0.0, 0.0, 0.0],
        "status": "unsupported_native_shape_count",
    }
    if native_max_shapes != 1:
        return record
    body = stage.GetPrimAtPath(body_path)
    if not body.IsValid() or not body.HasAPI(UsdPhysics.RigidBodyAPI):
        return {**record, "status": "missing_native_rigid_body"}
    prims = list(
        Usd.PrimRange(body, Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate))
    )
    if any(not prim.IsActive() or not prim.IsLoaded() for prim in prims):
        return {**record, "status": "inactive_or_unloaded_descendant"}
    if any(prim != body and prim.HasAPI(UsdPhysics.RigidBodyAPI) for prim in prims):
        return {**record, "status": "nested_rigid_body"}
    colliders = [prim for prim in prims if prim.HasAPI(UsdPhysics.CollisionAPI)]
    if len(colliders) != 1:
        return {**record, "status": "ambiguous_collider_count"}
    return _collider_record(record, body, colliders[0])


def _collider_record(record, body, collider):
    from pxr import UsdGeom, UsdPhysics

    record = {
        **record,
        "collider_path": str(collider.GetPath()),
        "geometry_type": collider.GetTypeName(),
    }
    if not UsdPhysics.CollisionAPI(collider).GetCollisionEnabledAttr().Get():
        return {**record, "status": "disabled_collision"}
    if not collider.IsA(UsdGeom.Sphere):
        return {**record, "status": "unsupported_geometry"}
    radius = UsdGeom.Sphere(collider).GetRadiusAttr().Get()
    if radius is None or not math.isfinite(radius) or radius <= 0:
        return {**record, "status": "invalid_sphere_radius"}
    transform = _sphere_transform(body, collider)
    if transform is None:
        return {**record, "status": "unsupported_sphere_transform"}
    scale, center = transform
    return {
        **record,
        "radius_m": float(radius),
        "scale": scale,
        "body_local_center_m": center,
        "status": "verified_live_sphere",
    }


def _sphere_transform(body, collider):
    import numpy as np
    from pxr import UsdGeom

    cache = UsdGeom.XformCache()
    world = cache.GetLocalToWorldTransform(collider)
    basis = np.asarray(world, dtype=float)[:3, :3]
    scale = np.linalg.norm(basis, axis=1)
    if not np.isfinite(basis).all() or not (scale > 0).all():
        return None
    # Only roundoff from native float transform authoring is tolerated, not shear.
    normalized = basis / scale[:, None]
    if np.linalg.det(normalized) <= 0 or not np.allclose(
        normalized @ normalized.T, np.eye(3), rtol=0, atol=8 * np.finfo(np.float32).eps
    ):
        return None
    if not np.allclose(scale, scale[0], rtol=8 * np.finfo(np.float32).eps, atol=0):
        return None
    relative = world * cache.GetLocalToWorldTransform(body).GetInverse()
    center = np.asarray(relative.ExtractTranslation(), dtype=float)
    if not np.isfinite(center).all():
        return None
    return scale.tolist(), center.tolist()


def _terrain_witness(points, normals, separation, signed_force, verified_spheres):
    import torch

    normal_squared = (normals * normals).sum(dim=1)
    valid = (
        torch.isfinite(points).all(dim=1)
        & torch.isfinite(normals).all(dim=1)
        & torch.isfinite(separation)
        & torch.isfinite(signed_force)
        & (signed_force != 0)
        & ((normal_squared - 1).abs() <= 8 * torch.finfo(normals.dtype).eps)
    )
    # PhysX keeps pair-oriented normals and changes only force sign for actor 1.
    witness = points - (separation * signed_force.sign())[:, None] * normals
    valid &= torch.isfinite(witness).all(dim=1)
    reason = torch.zeros_like(separation, dtype=torch.int32)
    reason[~valid] = 13
    reason[~verified_spheres] = 12
    valid &= verified_spheres
    return torch.where(valid[:, None], witness, 0.0).contiguous(), valid, reason
