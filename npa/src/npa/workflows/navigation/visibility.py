"""Hide all robot Gprims while preserving native physics and camera ancestors."""

from __future__ import annotations

import re


def _beneath(path, root):
    return path == root or path.startswith(root + "/")


def _paths(stage, adapter, env, recipe):
    from pxr import UsdGeom

    inventory = adapter.visibility_paths(env)
    roots = inventory["robot_roots"] + inventory["attachment_roots"]
    cameras = inventory["camera_prims"]
    if (
        len(inventory["robot_roots"]) != recipe.num_envs
        or len(cameras) != recipe.num_envs
    ):
        raise ValueError(
            "visibility paths need one robot root and observing camera per robot"
        )
    if not roots or len(set(roots + cameras)) != len(roots + cameras):
        raise ValueError("visibility paths must be unique")
    for path in roots + cameras:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid() or not prim.IsActive():
            raise ValueError("visibility path is missing or inactive")
    for root in roots:
        if _beneath(recipe.scene_prim, root) or _beneath(root, recipe.scene_prim):
            raise ValueError("robot roots and static warehouse must be disjoint")
        if any(root != other and _beneath(root, other) for other in roots):
            raise ValueError("robot and attachment roots must not overlap")
    if any(not stage.GetPrimAtPath(path).IsA(UsdGeom.Camera) for path in cameras):
        raise ValueError("camera paths must identify real USD cameras")
    _verify_camera_bindings(env, cameras)
    return inventory, roots, cameras


def _verify_camera_bindings(env, cameras):
    native = env.unwrapped
    patterns = [
        sensor.cfg.prim_path
        for sensor in native.scene.sensors.values()
        if "camera" in type(sensor).__name__.lower()
    ]
    namespace = getattr(native.scene, "env_regex_ns", "")
    patterns = [pattern.replace("{ENV_REGEX_NS}", namespace) for pattern in patterns]
    if any(
        not any(re.fullmatch(pattern, camera) for pattern in patterns)
        for camera in cameras
    ):
        raise ValueError("declared camera prim is not bound to a task camera sensor")


def _robot_geometry(stage, roots, scene_prim):
    from pxr import Usd, UsdGeom, UsdLux, UsdPhysics

    geometry, articulations = [], []
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        path = str(prim.GetPath())
        robot = any(_beneath(path, root) for root in roots)
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            if not robot:
                raise ValueError("articulation is outside declared robot roots")
            articulations.append(path)
        if robot:
            _reject_unsupported(prim, UsdGeom, UsdLux)
        if prim.IsA(UsdGeom.Gprim):
            if robot:
                geometry.append(prim)
            elif not _beneath(path, scene_prim):
                raise ValueError(
                    "render geometry is outside warehouse and declared robot roots"
                )
    if not geometry:
        raise ValueError("declared robot roots contain no render geometry")
    return geometry, articulations


def _reject_unsupported(prim, geom, lux):
    if prim.IsInstance() or prim.IsInstanceProxy():
        raise ValueError(
            "robot instance proxies require an explicit pre-cloning adapter"
        )
    kind = prim.GetTypeName().lower()
    if prim.HasAPI(lux.LightAPI) or any(
        token in kind for token in ("light", "particle", "volume")
    ):
        raise ValueError("robot lights/effects are unsupported")
    if prim.IsA(geom.PointInstancer):
        raise ValueError("robot point instancers are unsupported")


def _hide_geometry(geometry, cameras, author):
    from pxr import UsdGeom, UsdPhysics

    for prim in geometry:
        path = str(prim.GetPath())
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI) or prim.HasAPI(
            UsdPhysics.RigidBodyAPI
        ):
            # Gprim link roots would also hide arbitrary future sensor children.
            raise ValueError(
                "robot Gprim is a physics link ancestor; use separate visual children"
            )
        if any(_beneath(camera, path) for camera in cameras):
            raise ValueError("camera cannot be beneath a hidden robot Gprim")
        collision = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr()
        before = collision.Get() if collision else None
        imageable = UsdGeom.Imageable(prim)
        if author:
            imageable.MakeInvisible()
        if imageable.ComputeVisibility() != UsdGeom.Tokens.invisible:
            raise ValueError("robot geometry visibility could not be authored")
        if collision and collision.Get() != before:
            raise ValueError("visibility authoring changed collision state")
    for camera in cameras:
        if (
            UsdGeom.Imageable(
                geometry[0].GetStage().GetPrimAtPath(camera)
            ).ComputeVisibility()
            == UsdGeom.Tokens.invisible
        ):
            raise ValueError("observing camera inherits hidden visibility")


def hide_robot_geometry(adapter, env, recipe, *, author: bool = True) -> dict:
    """Check full stage coverage and author only robot geometry visibility.

    Args:
        adapter: Trusted module supplying robot, attachment and camera USD paths.
        env: Native Isaac environment with a live USD stage.
        recipe: Validated RGB-D recipe and shared warehouse path.
        author: Author visibility initially; False checks reset/step preservation.
    Returns:
        Geometry and camera inventory checked on the current stage.
    Raises:
        ValueError: Coverage, editability, instancing or sensor ancestry is unsafe.
    """
    stage = env.unwrapped.sim.stage
    inventory, roots, cameras = _paths(stage, adapter, env, recipe)
    geometry, articulations = _robot_geometry(stage, roots, recipe.scene_prim)
    if len(articulations) != recipe.num_envs:
        raise ValueError("live articulation count differs from declared robot count")
    _hide_geometry(geometry, cameras, author)
    return {
        **inventory,
        "hidden_gprims": [str(p.GetPath()) for p in geometry],
        "camera_visibility": "all_robot_geometry_hidden",
    }
