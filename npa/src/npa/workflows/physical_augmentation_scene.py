"""Build a camera-ready physical studio and verify its native solver configuration."""

from __future__ import annotations

import numpy as np


def configure_scene(config, recipe: dict) -> None:
    """Configure real collision geometry, studio lighting and an HD RTX camera.

    Args:
        config: Native Isaac task configuration before scene construction.
        recipe: Sealed presentation settings.
    Returns:
        None.
    Raises:
        ImportError: The native Isaac runtime is unavailable.
    """
    import isaaclab.sim as sim
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.sensors import TiledCameraCfg

    config.scene.table = _surface(
        "Table", (1.25, 0.92, 0.1), (0.4, 0, -0.05), (0.09, 0.15, 0.18)
    )
    config.scene.plane = _surface(
        "StudioFloor", (200, 200, 0.1), (0, 0, -1.05), (0.045, 0.06, 0.075)
    )
    config.scene.light.spawn = sim.DomeLightCfg(color=(0.82, 0.9, 1), intensity=1200)
    config.scene.npa_key_light = AssetBaseCfg(
        prim_path="/World/StudioKey",
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.2, -0.4, 0, 0.894427191)),
        spawn=sim.DistantLightCfg(color=(1, 0.94, 0.84), intensity=2500),
    )
    config.scene.npa_rollout_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/NpaRolloutCamera",
        data_types=["rgb"],
        width=recipe["presentation"]["width"],
        height=recipe["presentation"]["height"],
        update_period=0.0,
        spawn=sim.PinholeCameraCfg(
            focal_length=24, horizontal_aperture=24, clipping_range=(0.05, 200)
        ),
    )


def _surface(name, size, position, color):
    import isaaclab.sim as sim
    from isaaclab.assets import AssetBaseCfg

    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/" + name,
        init_state=AssetBaseCfg.InitialStateCfg(pos=position),
        spawn=sim.CuboidCfg(
            size=size,
            collision_props=sim.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim.RigidBodyMaterialCfg(
                static_friction=1, dynamic_friction=1
            ),
            visual_material=sim.PreviewSurfaceCfg(diffuse_color=color, roughness=0.7),
        ),
    )


def orient_camera(env) -> None:
    """Frame the actual robot and manipulation workspace after each reset.

    Args:
        env: Initialized native environment.
    Returns:
        None.
    Raises:
        RuntimeError: Native rendering fails.
    """
    import torch

    origin = env.scene.env_origins[:1]
    eye = torch.tensor([[1.25, -1.2, 0.95]], device=origin.device) + origin
    target = torch.tensor([[0.32, 0, 0.22]], device=origin.device) + origin
    env.scene["npa_rollout_camera"].set_world_poses_from_view(eyes=eye, targets=target)
    env.sim.render()


def camera_frame(env) -> np.ndarray:
    """Copy one actual RTX frame with the configured image dimensions.

    Args:
        env: Initialized native environment with the studio camera.
    Returns:
        RGB uint8 pixels before the next applied action.
    Raises:
        ValueError: The renderer returned the wrong shape or dtype.
    """
    camera = env.scene["npa_rollout_camera"]
    frame = camera.data.output["rgb"].torch[0, ..., :3].detach().cpu().numpy()
    if (
        frame.shape != (camera.cfg.height, camera.cfg.width, 3)
        or frame.dtype != np.uint8
    ):
        raise ValueError("RTX camera output differs from the configured presentation")
    return frame.copy()


def solver_evidence(env, recipe: dict) -> dict:
    """Verify the velocity solve on the composed robot and manipuland.

    Args:
        env: Initialized native environment.
        recipe: Sealed solver settings.
    Returns:
        Solver profile verified against native configuration and composed USD.
    Raises:
        ValueError: Native or composed solver settings differ from the recipe.
        ImportError: Native USD APIs are unavailable.
    """
    from pxr import UsdPhysics

    expected = recipe["physics_solver"]
    if (
        env.cfg.sim.physics.solver_type != 1
        or env.cfg.sim.physics.enable_external_forces_every_iteration is not True
    ):
        raise ValueError("Native solver differs from sealed TGS configuration")
    for name, schema, prefix in (
        ("robot", UsdPhysics.ArticulationRootAPI, "physxArticulation:"),
        ("object", UsdPhysics.RigidBodyAPI, "physxRigidBody:"),
    ):
        _check_composed_solver(env.scene[name].cfg.prim_path, schema, prefix, expected)
    return dict(expected)


def _check_composed_solver(path, schema, prefix, expected):
    from isaaclab.sim.utils.queries import find_first_matching_prim
    from pxr import Usd

    root = find_first_matching_prim(path)
    bodies = (
        [prim for prim in Usd.PrimRange(root) if prim.HasAPI(schema)] if root else []
    )
    if len(bodies) != 1:
        raise ValueError(f"Expected one composed solver body for {path}")
    for field, attribute in (
        ("position_iterations", "solverPositionIterationCount"),
        ("velocity_iterations", "solverVelocityIterationCount"),
    ):
        if bodies[0].GetAttribute(prefix + attribute).Get() != expected[field]:
            raise ValueError(f"Composed {path} solver differs from sealed {field}")
