"""Configure the real Isaac Franka embodiment, physics variation, and RTX camera."""

from __future__ import annotations

from pathlib import Path


def environment_config(recipe: dict, *, training: bool, condition: str = "nominal",
                       capture: bool = False, asset_root: Path | None = None):
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
        config.sim.physics.gpu_total_aggregate_pairs_capacity = recipe["physics_capacity"]["gpu_total_aggregate_pairs_capacity"]
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
    friction = tuple(recipe["train_friction_range"]) if training else (shift["friction"],) * 2
    config.events.npa_object_mass = EventTermCfg(
        func=mdp.randomize_rigid_body_mass, mode="startup",
        params={"asset_cfg": SceneEntityCfg("object"), "mass_distribution_params": mass,
                "operation": "scale"},
    )
    config.events.npa_object_material = EventTermCfg(
        func=mdp.randomize_rigid_body_material, mode="startup",
        params={"asset_cfg": SceneEntityCfg("object"), "static_friction_range": friction,
                "dynamic_friction_range": friction, "restitution_range": (0.0, 0.0),
                "num_buckets": 64 if training else 1},
    )


def _camera_config(config) -> None:
    import isaaclab.sim as sim
    from isaaclab.sensors import TiledCameraCfg

    config.scene.npa_rollout_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/NpaRolloutCamera",
        offset=TiledCameraCfg.OffsetCfg(pos=(1.5, -1.5, 1.3), convention="world"),
        data_types=["rgb"], width=640, height=480, update_period=0.0,
        spawn=sim.PinholeCameraCfg(focal_length=24.0, horizontal_aperture=20.955,
                                  clipping_range=(0.05, 10.0)),
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
    runner = OnPolicyRunner(wrapped, settings, log_dir=str(output) if output else None, device="cuda:0")
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
    measured = native.scene["ee_frame"].data.target_pos_w.torch[:, 0].detach().cpu().numpy()
    offset = np.broadcast_to(np.asarray(profile["tool_offset_m"], dtype=positions.dtype), positions.shape)
    cross = 2 * np.cross(quaternions[:, :3], offset)
    expected = positions + offset + quaternions[:, 3:] * cross + np.cross(quaternions[:, :3], cross)
    errors = np.linalg.norm(measured - expected, axis=1)
    if (not np.isfinite(errors).all() or not np.allclose(np.linalg.norm(quaternions, axis=1), 1, atol=1e-4)
            or float(errors.max()) > 1e-4):
        raise ValueError("Frame sensor world TCP differs from the articulation link pose and grasp offset")
    return {"source_body": profile.get("sensor_source_body", profile["base_body"]), "tool_body": profile["tool_body"],
            "environments_checked": len(errors), "maximum_error_m": float(errors.max()),
            "tolerance_m": 1e-4, "world_tcp_verified": True}


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
    values = {"mass_kg": wp.to_torch(view.get_masses()), "static_friction": material[..., 0],
              "dynamic_friction": material[..., 1], "restitution": material[..., 2]}
    evidence = {"startup_terms": list(terms), "embodiment": unwrapped.npa_embodiment_evidence, "physics_capacity": {
        "gpu_total_aggregate_pairs_capacity": unwrapped.cfg.sim.physics.gpu_total_aggregate_pairs_capacity}}
    evidence.update(_integrity_evidence(unwrapped))
    for name, value in values.items():
        if not torch.isfinite(value).all():
            raise RuntimeError("Applied Franka physics is nonfinite")
        evidence[name] = {"min": float(value.min()), "max": float(value.max())}
    return evidence


def _integrity_evidence(native):
    from npa.workflows.franka_rl_validity import validity_evidence

    evidence = {"simulation_validity": validity_evidence(native)}
    for field, attribute in {"stability": "npa_stability_evidence", "tool_frame_check": "npa_tool_frame_check",
            "frozen_observation_normalization": "npa_normalization_evidence",
            "frozen_action_distribution": "npa_distribution_evidence"}.items():
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
