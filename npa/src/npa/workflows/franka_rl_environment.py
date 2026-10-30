"""Configure the real Isaac Franka embodiment, physics variation, and RTX camera."""

from __future__ import annotations

import math
from pathlib import Path


POLICY_INTERCHANGE_SCAFFOLD_TASK = "Isaac-Lift-Cube-Franka-IK-Rel-v0"
POLICY_INTERCHANGE_SIMULATION_HZ = 60.0


def _policy_interchange_contact_prim_paths(
    robot_profile, rigid_body_prim_paths: dict[str, str]
) -> tuple[str, str]:
    """Resolve live contact-sensor paths from the converted USD hierarchy."""

    resolved: list[str] = []
    for link in robot_profile.gripper.finger_links:
        relative = rigid_body_prim_paths.get(link)
        if not isinstance(relative, str) or not relative:
            raise ValueError(f"resolved RobotSpec USD lacks finger rigid body {link!r}")
        resolved.append(f"{{ENV_REGEX_NS}}/Robot/{relative}")
    if len(resolved) != 2:
        raise ValueError("surface placement requires exactly two finger contact bodies")
    return resolved[0], resolved[1]


def policy_interchange_timing(*, control_hz: float) -> dict[str, float | int]:
    """Resolve an exact controller cadence against the owned physics rate."""

    frequency = float(control_hz)
    ratio = POLICY_INTERCHANGE_SIMULATION_HZ / frequency
    decimation = round(ratio)
    if (
        not math.isfinite(frequency)
        or frequency <= 0.0
        or decimation <= 0
        or abs(ratio - decimation) > 1.0e-9
    ):
        raise ValueError(
            "policy control_hz must divide the 60 Hz interchange physics rate exactly"
        )
    return {
        "simulation_hz": POLICY_INTERCHANGE_SIMULATION_HZ,
        "simulation_dt_s": 1.0 / POLICY_INTERCHANGE_SIMULATION_HZ,
        "control_hz": frequency,
        "decimation": decimation,
        "episode_length_s": 240.0,
    }


def policy_interchange_environment_config(*, seed: int, control_hz: float):
    """Load only the native scene/IK scaffold used by typed interchange.

    The caller must replace the scaffold robot, home state, actuators, action
    mapping, support surface, rigid object, goal, reset, and cameras before the
    environment is created. This avoids importing Franka reach, controller, or
    task settings through an unrelated training recipe.
    """

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import load_cfg_from_registry
    from npa.workflows.sim2real.isaac_assets_compat import remap_moved_franka_usd

    config = load_cfg_from_registry(
        POLICY_INTERCHANGE_SCAFFOLD_TASK, "env_cfg_entry_point"
    )
    remap_moved_franka_usd(config)
    timing = policy_interchange_timing(control_hz=control_hz)
    config.scene.num_envs = 1
    config.seed = seed
    config.sim.device = "cuda:0"
    config.sim.dt = timing["simulation_dt_s"]
    config.decimation = timing["decimation"]
    config.episode_length_s = timing["episode_length_s"]
    config.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)
    config.observations.policy.enable_corruption = False
    return config


def configure_policy_interchange_robot(
    config,
    robot_profile,
    robot_spec: dict,
    robot_usd_path: Path,
    *,
    controller_representation: str = "absolute_joint_position",
) -> dict:
    """Replace the template articulation with the exact resolved RobotSpec asset.

    The Lift environment is only a scene/controller scaffold here.  Morphology,
    home state, actuator gains, action order, gripper semantics, and link names
    all come from the immutable RobotSpec used to produce ``robot_usd_path``.
    """

    from copy import deepcopy
    import re

    import isaaclab.sim as sim
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.envs import mdp
    from npa.workflows.sim2real.isaac_robot_asset import (
        resolved_rigid_body_prim_paths,
    )

    names = list(robot_spec.get("joint_names") or ())
    expected = [*robot_profile.arm_joint_names, *robot_profile.gripper.joint_names]
    arrays = {
        "home_qpos": robot_spec.get("home_qpos"),
        "kp": robot_spec.get("kp"),
        "kv": robot_spec.get("kv"),
        "force_lower": robot_spec.get("force_lower"),
        "force_upper": robot_spec.get("force_upper"),
    }
    if names != expected or any(
        not isinstance(values, list) or len(values) != len(names)
        for values in arrays.values()
    ):
        raise ValueError(
            "resolved RobotSpec joint order or actuator arrays differ from the policy profile"
        )
    if (
        robot_spec.get("base_link") != robot_profile.base_link
        or robot_spec.get("ee_link") != robot_profile.ee_link
        or list(robot_spec.get("finger_links") or ())
        != list(robot_profile.gripper.finger_links)
    ):
        raise ValueError("resolved RobotSpec links differ from the policy profile")
    if not robot_usd_path.is_file():
        raise ValueError("resolved RobotSpec USD is missing")
    rigid_body_paths = resolved_rigid_body_prim_paths(
        robot_usd_path,
        (
            robot_profile.base_link,
            robot_profile.ee_link,
            *robot_profile.gripper.finger_links,
        ),
    )

    def spawned_prim_path(relative: str) -> str:
        root = "{ENV_REGEX_NS}/Robot"
        return f"{root}/{relative}" if relative else root

    old_spawn = config.scene.robot.spawn
    spawn = sim.UsdFileCfg(usd_path=str(robot_usd_path))
    for attribute in ("articulation_props", "rigid_props", "activate_contact_sensors"):
        if hasattr(old_spawn, attribute) and hasattr(spawn, attribute):
            setattr(spawn, attribute, deepcopy(getattr(old_spawn, attribute)))
    config.scene.robot.spawn = spawn
    config.scene.robot.init_state.pos = (0.0, 0.0, 0.0)
    config.scene.robot.init_state.rot = (0.0, 0.0, 0.0, 1.0)
    config.scene.robot.init_state.joint_pos = dict(
        zip(names, arrays["home_qpos"], strict=True)
    )
    config.scene.robot.actuators = {}
    for index, name in enumerate(names):
        effort = max(
            abs(float(arrays["force_lower"][index])),
            abs(float(arrays["force_upper"][index])),
        )
        config.scene.robot.actuators[f"robot_spec_joint_{index:02d}"] = (
            ImplicitActuatorCfg(
                joint_names_expr=["^" + re.escape(name) + "$"],
                stiffness=float(arrays["kp"][index]),
                damping=float(arrays["kv"][index]),
                effort_limit=effort,
                effort_limit_sim=effort,
            )
        )
    if controller_representation == "absolute_joint_position":
        config.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=list(robot_profile.arm_joint_names),
            preserve_order=True,
            scale=1.0,
            use_default_offset=False,
        )
    elif controller_representation == "relative_ee_pose":
        action = config.actions.arm_action
        if (
            not hasattr(action, "body_name")
            or not hasattr(action, "body_offset")
            or not hasattr(action, "joint_names")
        ):
            raise ValueError(
                "geometry-aware teacher requires the native relative differential-IK "
                "action scaffold"
            )
        action.joint_names = list(robot_profile.arm_joint_names)
        action.body_name = robot_profile.ee_link
        action.body_offset.pos = (0.0, 0.0, 0.0)
        action.body_offset.rot = (0.0, 0.0, 0.0, 1.0)
    else:
        raise ValueError(
            f"unsupported policy-interchange controller {controller_representation!r}"
        )
    config.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=list(robot_profile.gripper.joint_names),
        open_command_expr=dict(
            zip(
                robot_profile.gripper.joint_names,
                robot_profile.gripper.open_joint_positions,
                strict=True,
            )
        ),
        close_command_expr=dict(
            zip(
                robot_profile.gripper.joint_names,
                robot_profile.gripper.closed_joint_positions,
                strict=True,
            )
        ),
    )
    config.commands.object_pose.body_name = robot_profile.ee_link
    config.scene.ee_frame.prim_path = spawned_prim_path(
        rigid_body_paths[robot_profile.base_link]
    )
    target = config.scene.ee_frame.target_frames[0]
    target.prim_path = spawned_prim_path(rigid_body_paths[robot_profile.ee_link])
    target.offset.pos = (0.0, 0.0, 0.0)
    target.offset.rot = (0.0, 0.0, 0.0, 1.0)
    return {
        "joint_names": names,
        "home_qpos": list(arrays["home_qpos"]),
        "kp": list(arrays["kp"]),
        "kv": list(arrays["kv"]),
        "force_lower": list(arrays["force_lower"]),
        "force_upper": list(arrays["force_upper"]),
        "usd_path": str(robot_usd_path),
        "rigid_body_prim_paths": rigid_body_paths,
        "controller_representation": controller_representation,
    }


def environment_config(
    recipe: dict,
    *,
    training: bool,
    condition: str = "nominal",
    capture: bool = False,
    asset_root: Path | None = None,
):
    """Build the pinned Franka lift task with explicit physical perturbations.

    Args:
        recipe: Sealed experiment settings.
        training: Apply the training distribution and vector environment count.
        condition: Held-out physics condition when evaluating.
        capture: Add a genuine RTX camera to a single environment.
        asset_root: Materialized directory containing the sealed USD asset bundle.
    Returns:
        Isaac Lab environment configuration.
    Raises:
        ImportError: Isaac runtime is unavailable.
        ValueError: An unknown condition is requested.
    """
    config = _base_task_config(recipe)
    config.scene.num_envs = recipe["num_envs"] if training else recipe["eval_episodes"]
    config.seed = recipe["seed"] if training else recipe["validation_seed"]
    config.sim.device = "cuda:0"
    if "physics_capacity" in recipe:
        config.sim.physics.gpu_total_aggregate_pairs_capacity = recipe[
            "physics_capacity"
        ]["gpu_total_aggregate_pairs_capacity"]
    config.commands.object_pose.resampling_time_range = (5.0, 5.0)
    _assets(config, recipe, asset_root)
    _physics_events(config, recipe, training, condition)
    from npa.workflows.franka_rl_learning import configure_learning

    configure_learning(config, recipe, training=training)
    from npa.workflows.franka_rl_validity import configure_validity

    configure_validity(config, recipe)
    if not training:
        config.observations.policy.enable_corruption = False
    if capture:
        config.scene.num_envs = 1
        _camera_config(config)
    return config


def _base_task_config(recipe):
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import load_cfg_from_registry
    from npa.workflows.franka_rl_embodiments import configure_embodiment
    from npa.workflows.franka_rl_physics import configure_stability
    from npa.workflows.sim2real.isaac_assets_compat import remap_moved_franka_usd

    config = load_cfg_from_registry(recipe["task"], "env_cfg_entry_point")
    configure_embodiment(config, recipe)
    configure_stability(config, recipe)
    remap_moved_franka_usd(config)
    return config


def _assets(config, recipe: dict, asset_root: Path | None) -> None:
    if "assets" not in recipe:
        return
    from npa.workflows.franka_rl_assets import configure_assets

    if asset_root is None:
        raise ValueError("Franka asset bundle must be materialized before simulation")
    configure_assets(config, recipe["assets"], asset_root)


def _physics_events(config, recipe: dict, training: bool, condition: str) -> None:
    from isaaclab.envs import mdp
    from isaaclab.managers import EventTermCfg, SceneEntityCfg

    shift = recipe["conditions"][condition]
    mass = tuple(recipe["train_mass_range"]) if training else (shift["mass_scale"],) * 2
    friction = (
        tuple(recipe["train_friction_range"]) if training else (shift["friction"],) * 2
    )
    config.events.npa_object_mass = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": mass,
            "operation": "scale",
        },
    )
    config.events.npa_object_material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": friction,
            "dynamic_friction_range": friction,
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64 if training else 1,
        },
    )


def _camera_config(config) -> None:
    import isaaclab.sim as sim
    from isaaclab.sensors import TiledCameraCfg

    config.scene.npa_rollout_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/NpaRolloutCamera",
        offset=TiledCameraCfg.OffsetCfg(pos=(1.5, -1.5, 1.3), convention="world"),
        data_types=["rgb"],
        width=640,
        height=480,
        update_period=0.0,
        spawn=sim.PinholeCameraCfg(
            focal_length=24.0, horizontal_aperture=20.955, clipping_range=(0.05, 10.0)
        ),
    )


def configure_policy_interchange_cameras(config, robot_profile) -> None:
    """Install calibrated world-pose cameras from a sealed robot profile.

    Both sensors use world-rooted camera prims. The native policy execution
    stage republishes the wrist sensor pose from its declared live mount link
    before every capture, avoiding a stale USD child transform on moving
    articulations.
    """

    import re

    import isaaclab.sim as sim
    from isaaclab.sensors import TiledCameraCfg

    config.scene.num_envs = 1
    for camera in robot_profile.cameras:
        if re.fullmatch(r"[a-z][a-z0-9_]*", camera.key) is None:
            raise ValueError(
                "policy camera keys must be lowercase identifiers for Isaac routing"
            )
        width, height = camera.image_size
        fx, fy, cx, cy = camera.intrinsics
        if (
            abs(fx - fy) > 1.0e-6
            or abs(cx - (width - 1) / 2.0) > 1.0
            or abs(cy - (height - 1) / 2.0) > 1.0
        ):
            raise ValueError(
                "Isaac interchange currently requires square pixels and centered intrinsics"
            )
        aperture = 20.955
        focal_length = fx * aperture / width
        sensor = TiledCameraCfg(
            prim_path=f"{{ENV_REGEX_NS}}/NpaPolicyCamera_{camera.key}",
            offset=TiledCameraCfg.OffsetCfg(
                pos=tuple(camera.mount_from_camera.translation_m),
                rot=tuple(camera.mount_from_camera.quaternion_xyzw),
                convention="ros",
            ),
            data_types=["rgb"],
            width=width,
            height=height,
            update_period=0.0,
            update_latest_camera_pose=True,
            spawn=sim.PinholeCameraCfg(
                focal_length=focal_length,
                horizontal_aperture=aperture,
                clipping_range=(0.03, 10.0),
            ),
        )
        setattr(config.scene, f"npa_policy_camera_{camera.key}", sensor)


def configure_policy_interchange_surface_task(
    config,
    robot_profile,
    object_profile,
    task,
    *,
    object_usd_path: Path,
    robot_rigid_body_prim_paths: dict[str, str],
) -> None:
    """Bind a non-Franka Isaac lift template to the released-surface protocol."""

    from copy import deepcopy

    import isaaclab.sim as sim
    from isaaclab.sensors import ContactSensorCfg
    from npa.workflows.sim2real.pi05_isaac import (
        configure_destination_marker,
        configure_support_surface,
        configure_surface_goal,
    )

    configure_support_surface(config)
    configure_surface_goal(config, task.target_world_xyz_m)
    configure_destination_marker(config, task.target_world_xyz_m)
    if not object_usd_path.is_file():
        raise ValueError("sealed rigid-object USD is missing")
    old_spawn = config.scene.object.spawn
    config.scene.object.spawn = sim.UsdFileCfg(
        usd_path=str(object_usd_path),
        scale=tuple(object_profile.scale_xyz),
        rigid_props=deepcopy(old_spawn.rigid_props),
        mass_props=sim.MassPropertiesCfg(mass=object_profile.mass_kg),
        physics_material=sim.RigidBodyMaterialCfg(
            static_friction=object_profile.static_friction,
            dynamic_friction=object_profile.dynamic_friction,
            restitution=object_profile.restitution,
        ),
    )
    config.scene.object.init_state.pos = tuple(
        object_profile.world_from_root.translation_m
    )
    config.scene.object.init_state.rot = tuple(
        object_profile.world_from_root.quaternion_xyzw
    )
    reset = getattr(config.events, "reset_object_position", None)
    params = getattr(reset, "params", None)
    if not isinstance(params, dict) or "pose_range" not in params:
        raise ValueError("Isaac task lacks an explicit object reset pose range")
    params["pose_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    config.scene.robot.spawn.activate_contact_sensors = True
    config.scene.object.spawn.activate_contact_sensors = True
    config.scene.table.spawn.activate_contact_sensors = True
    left_path, right_path = _policy_interchange_contact_prim_paths(
        robot_profile, robot_rigid_body_prim_paths
    )
    config.scene.npa_left_object_contact = ContactSensorCfg(
        prim_path=left_path,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        history_length=3,
    )
    config.scene.npa_right_object_contact = ContactSensorCfg(
        prim_path=right_path,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        history_length=3,
    )
    config.scene.npa_object_support_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Table"],
        history_length=3,
    )


def build_runner(env, recipe: dict, output=None):
    """Construct native RSL-RL PPO with the task's version-migrated configuration.

    Args:
        env: Instantiated Isaac environment.
        recipe: Sealed PPO iteration and checkpoint settings.
        output: Training directory, or None during evaluation.
    Returns:
        Wrapped vector environment, real PPO runner, and resolved agent settings.
    Raises:
        ImportError: Required native training libraries are unavailable.
        RuntimeError: Native runner creation fails.
    """
    from rsl_rl.runners import OnPolicyRunner
    from npa.workflows.franka_rl_embodiments import embodiment_evidence

    env.unwrapped.npa_embodiment_evidence = embodiment_evidence(env, recipe)
    from npa.workflows.franka_rl_physics import stability_evidence

    env.unwrapped.npa_stability_evidence = stability_evidence(env.unwrapped, recipe)
    profile = env.unwrapped.npa_embodiment_evidence["profile"]
    if "sensor_source_body" in profile or "learning" in recipe:
        env.unwrapped.npa_tool_frame_check = _tool_frame_check(env.unwrapped, profile)
    config = _learner_config(recipe)
    settings = config.to_dict()
    from npa.workflows.franka_rl_validity import build_validated_wrapper

    wrapped = build_validated_wrapper(env, clip_actions=config.clip_actions)
    runner = OnPolicyRunner(
        wrapped, settings, log_dir=str(output) if output else None, device="cuda:0"
    )
    return wrapped, runner, settings


def _learner_config(recipe):
    from importlib.metadata import version
    from isaaclab_tasks.utils import load_cfg_from_registry
    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
    from npa.workflows.franka_rl_learning import configure_learner

    config = load_cfg_from_registry(recipe["task"], "rsl_rl_cfg_entry_point")
    config = handle_deprecated_rsl_rl_cfg(config, version("rsl-rl-lib"))
    configure_learner(config, recipe)
    config.seed = recipe["seed"]
    config.num_steps_per_env = recipe["steps_per_env"]
    config.save_interval = recipe["checkpoint_interval"]
    config.max_iterations = recipe["iterations"]
    return config


def _tool_frame_check(native, profile: dict) -> dict:
    """Check world TCP against the articulation's independent XYZW link pose."""
    import numpy as np

    robot = native.scene["robot"]
    index = robot.body_names.index(profile["tool_body"])
    positions = robot.data.body_link_pos_w.torch[:, index].detach().cpu().numpy()
    quaternions = robot.data.body_link_quat_w.torch[:, index].detach().cpu().numpy()
    measured = (
        native.scene["ee_frame"].data.target_pos_w.torch[:, 0].detach().cpu().numpy()
    )
    offset = np.broadcast_to(
        np.asarray(profile["tool_offset_m"], dtype=positions.dtype), positions.shape
    )
    cross = 2 * np.cross(quaternions[:, :3], offset)
    expected = (
        positions
        + offset
        + quaternions[:, 3:] * cross
        + np.cross(quaternions[:, :3], cross)
    )
    errors = np.linalg.norm(measured - expected, axis=1)
    if (
        not np.isfinite(errors).all()
        or not np.allclose(np.linalg.norm(quaternions, axis=1), 1, atol=1e-4)
        or float(errors.max()) > 1e-4
    ):
        raise ValueError(
            "Frame sensor world TCP differs from the articulation link pose and grasp offset"
        )
    return {
        "source_body": profile.get("sensor_source_body", profile["base_body"]),
        "tool_body": profile["tool_body"],
        "environments_checked": len(errors),
        "maximum_error_m": float(errors.max()),
        "tolerance_m": 1e-4,
        "world_tcp_verified": True,
    }


def physics_evidence(env) -> dict:
    """Read applied mass and friction directly from the simulator's rigid-body view.

    Args:
        env: Live Isaac environment.
    Returns:
        Physical mass and material ranges, including applied event names.
    Raises:
        RuntimeError: Physics events were not activated or tensors are nonfinite.
    """
    import torch
    import warp as wp

    unwrapped = env.unwrapped
    terms = unwrapped.event_manager.active_terms["startup"]
    if not {"npa_object_mass", "npa_object_material"}.issubset(terms):
        raise RuntimeError("Franka physics randomization events are missing")
    view = unwrapped.scene["object"].root_view
    material = wp.to_torch(view.get_material_properties())
    values = {
        "mass_kg": wp.to_torch(view.get_masses()),
        "static_friction": material[..., 0],
        "dynamic_friction": material[..., 1],
        "restitution": material[..., 2],
    }
    evidence = {
        "startup_terms": list(terms),
        "embodiment": unwrapped.npa_embodiment_evidence,
        "physics_capacity": {
            "gpu_total_aggregate_pairs_capacity": unwrapped.cfg.sim.physics.gpu_total_aggregate_pairs_capacity
        },
    }
    evidence.update(_integrity_evidence(unwrapped))
    for name, value in values.items():
        if not torch.isfinite(value).all():
            raise RuntimeError("Applied Franka physics is nonfinite")
        evidence[name] = {"min": float(value.min()), "max": float(value.max())}
    return evidence


def _integrity_evidence(native):
    from npa.workflows.franka_rl_validity import validity_evidence

    evidence = {"simulation_validity": validity_evidence(native)}
    for field, attribute in {
        "stability": "npa_stability_evidence",
        "tool_frame_check": "npa_tool_frame_check",
        "frozen_observation_normalization": "npa_normalization_evidence",
        "frozen_action_distribution": "npa_distribution_evidence",
    }.items():
        if getattr(native, attribute, None):
            evidence[field] = getattr(native, attribute)
    if hasattr(native.cfg, "npa_simulation_validity"):
        from npa.workflows.franka_rl_assets import asset_physics_evidence

        evidence["object_solver_properties"] = asset_physics_evidence(native)
    return evidence


def load_checkpoint(runner, path) -> None:
    """Load tensor-only native PPO weights with strict architecture matching.

    Args:
        runner: Native RSL-RL runner.
        path: Checksum-verified native checkpoint.
    Returns:
        None.
    Raises:
        ValueError: The checkpoint does not contain native PPO state.
        RuntimeError: Tensor decoding or strict state loading fails.
    """
    import torch

    payload = torch.load(path, map_location="cuda:0", weights_only=True)
    if not isinstance(payload, dict) or type(payload.get("iter")) is not int:
        raise ValueError("Franka checkpoint lacks native PPO iteration state")
    if runner.alg.load(payload, load_cfg=None, strict=True):
        runner.current_learning_iteration = payload["iter"]
