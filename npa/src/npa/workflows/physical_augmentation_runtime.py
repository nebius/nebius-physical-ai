"""Execute public Franka physics variants and record genuine RTX transitions."""

from __future__ import annotations

import argparse
from importlib.metadata import version
import os
from pathlib import Path

import numpy as np

from npa.workflows.franka_rl_validity import (
    SimulationValidityError,
    configure_validity,
    validate_simulation_state,
)
from npa.workflows.lerobot_transfer_data import write_json
from npa.workflows.physical_augmentation_contract import (
    read_recipe,
    LiftController,
    accepted_steps,
    longest_hold,
)


def _configuration(recipe: dict, condition: str):
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import load_cfg_from_registry
    from npa.workflows.physical_augmentation_scene import configure_scene
    from npa.workflows.sim2real.isaac_assets_compat import remap_moved_franka_usd

    config = load_cfg_from_registry(recipe["task"], "env_cfg_entry_point")
    remap_moved_franka_usd(config)
    config.scene.num_envs = 1
    config.seed = recipe["seed"]
    config.sim.device = "cuda:0"
    _configure_clock(config, recipe)
    config.episode_length_s = (
        (recipe["episode_steps"] + 100) * config.sim.dt * config.decimation
    )
    config.commands.object_pose.debug_vis = False
    config.observations.policy.enable_corruption = False
    _align_tool_frame(config, recipe)
    _configure_gripper(config, recipe)
    _configure_object(config, recipe["conditions"][condition])
    _configure_solver(config, recipe)
    configure_scene(config, recipe)
    configure_validity(config, recipe)
    return config


def _configure_clock(config, recipe):
    clock = recipe["physics_solver"]
    config.sim.dt = clock["physics_dt_s"]
    config.decimation = round(clock["control_dt_s"] / config.sim.dt)
    config.sim.render_interval = config.decimation


def _configure_object(config, case):
    import isaaclab.sim as sim

    config.scene.object.spawn = sim.CuboidCfg(
        size=(0.05, 0.05, 0.05),
        rigid_props=config.scene.object.spawn.rigid_props,
        collision_props=sim.CollisionPropertiesCfg(collision_enabled=True),
        mass_props=sim.MassPropertiesCfg(mass=case["mass_kg"]),
        physics_material=sim.RigidBodyMaterialCfg(
            static_friction=case["friction"], dynamic_friction=case["friction"]
        ),
        visual_material=sim.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.1)),
    )
    x, y = case["offset_xy_m"]
    config.scene.object.init_state.pos = (0.5 + x, y, 0.035)
    config.events.reset_object_position.params["pose_range"] = {
        "x": (-0.02, 0.02),
        "y": (-0.02, 0.02),
        "z": (0.0, 0.0),
    }


def _configure_gripper(config, recipe: dict) -> None:
    from isaaclab.utils.string import ResolvableString

    config.actions.gripper_action.class_type = ResolvableString(
        "npa.workflows.physical_augmentation_servo:RampedGripperAction"
    )
    config.npa_gripper_servo = recipe["gripper_servo"]
    config.scene.robot.actuators["panda_hand"].effort_limit_sim = recipe[
        "gripper_servo"
    ]["effort_limit_n"]


def _configure_solver(config, recipe: dict) -> None:
    if config.sim.physics.solver_type != 1:
        raise ValueError("Physical augmentation requires the native TGS solver")
    # Native Isaac warns that skipping these forces produces noisy velocities.
    # Improve the solver's velocity updates rather than loosening hold criteria.
    config.sim.physics.enable_external_forces_every_iteration = recipe[
        "physics_solver"
    ]["external_forces_every_iteration"]
    # The upstream Franka requests zero velocity iterations. Contacts can then
    # retain biased velocities even when the measured object pose is stationary.
    for props in (
        config.scene.robot.spawn.articulation_props,
        config.scene.object.spawn.rigid_props,
    ):
        props.solver_position_iteration_count = recipe["physics_solver"][
            "position_iterations"
        ]
        props.solver_velocity_iteration_count = recipe["physics_solver"][
            "velocity_iterations"
        ]


def _align_tool_frame(config, recipe: dict) -> None:
    arm = config.actions.arm_action
    expected = recipe["tcp_contract"]
    if (
        arm.body_name != expected["body"]
        or list(arm.body_offset.pos) != expected["offset_m"]
    ):
        raise ValueError("Native IK tool frame differs from the sealed action contract")
    # The stock frame sensor uses 0.1034m while IK controls 0.107m. That bias is
    # larger than one speed-bounded action, so feedback from it can reverse motion.
    frame = config.scene.ee_frame.target_frames[0]
    frame.offset.pos = tuple(arm.body_offset.pos)
    frame.offset.rot = tuple(arm.body_offset.rot)


def _physics(env, requested: dict) -> dict:
    import warp as wp

    view = env.scene["object"].root_view
    mass = wp.to_torch(view.get_masses()).cpu().numpy()
    material = wp.to_torch(view.get_material_properties()).cpu().numpy()
    if not np.allclose(mass, requested["mass_kg"], atol=1e-5):
        raise ValueError("Simulator object mass differs from the requested condition")
    if not np.allclose(material[..., :2], requested["friction"], atol=1e-5):
        raise ValueError("Simulator friction differs from the requested condition")
    return {
        "mass_kg": mass.tolist(),
        "material_static_dynamic_restitution": material.tolist(),
        "object_geometry": "procedural 5cm collision-bearing cube",
        "composed_physics": _composed_physics(env),
    }


def _composed_physics(env) -> dict:
    from isaaclab.sim.utils.queries import find_first_matching_prim
    from pxr import Usd, UsdPhysics, PhysxSchema

    root = find_first_matching_prim(env.scene["object"].cfg.prim_path)
    if not root:
        raise ValueError("Missing composed object prim")
    bodies = [
        prim for prim in Usd.PrimRange(root) if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    collisions = [
        prim for prim in Usd.PrimRange(root) if prim.HasAPI(UsdPhysics.CollisionAPI)
    ]
    if len(bodies) != 1 or not collisions:
        raise ValueError("Object requires one rigid body and collision geometry")
    body = UsdPhysics.RigidBodyAPI(bodies[0])
    no_gravity = PhysxSchema.PhysxRigidBodyAPI(bodies[0]).GetDisableGravityAttr().Get()
    enabled = all(
        UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
        for prim in collisions
    )
    if (
        no_gravity
        or body.GetKinematicEnabledAttr().Get()
        or not body.GetRigidBodyEnabledAttr().Get()
        or not enabled
    ):
        raise ValueError(
            "Manipulation object must be dynamic with gravity and collisions"
        )
    return {
        "source": "composed USD schema",
        "dynamic": True,
        "gravity_enabled": True,
        "collision_prims": len(collisions),
        "collision_enabled": bool(enabled),
    }


def _snapshot(env) -> dict:
    from isaaclab.utils.math import subtract_frame_transforms

    robot, obj = env.scene["robot"].data, env.scene["object"].data
    frame = env.scene["ee_frame"].data
    tcp, quat = subtract_frame_transforms(
        robot.root_pos_w.torch,
        robot.root_quat_w.torch,
        frame.target_pos_w.torch[:, 0],
        frame.target_quat_w.torch[:, 0],
    )
    cube, _ = subtract_frame_transforms(
        robot.root_pos_w.torch, robot.root_quat_w.torch, obj.root_pos_w.torch
    )
    _verify_tool_frame(env, tcp, quat)
    return {
        name: value[0].detach().cpu().numpy().copy()
        for name, value in {
            "state": robot.joint_pos.torch,
            "object": cube,
            "tcp": tcp,
            "quaternion": quat,
            "velocity": obj.root_lin_vel_w.torch,
        }.items()
    }


def _verify_tool_frame(env, tcp, quaternion) -> None:
    import torch

    position, rotation = env.action_manager.get_term("arm_action")._compute_frame_pose()
    if not torch.allclose(position, tcp, atol=1e-4, rtol=0) or not torch.allclose(
        (rotation * quaternion).sum(dim=-1).abs(),
        torch.ones_like(rotation[:, 0]),
        atol=1e-4,
        rtol=0,
    ):
        raise ValueError("Measured TCP differs from the native IK control frame")


def _step(env, action):
    import torch

    _, _, terminated, truncated, _ = env.step(
        torch.as_tensor(action, device=env.device).unsqueeze(0)
    )
    # A reset observation must never be recorded as the preceding action's outcome.
    done = bool(terminated[0] or truncated[0])
    if not done and not np.array_equal(
        env.action_manager.action[0].cpu().numpy(), action
    ):
        raise ValueError("Native action manager differs from the recorded command")
    return done


def _settle(env, seed: int) -> dict:
    from npa.workflows.physical_augmentation_scene import orient_camera

    env.reset(seed=seed)
    orient_camera(env)
    observed = _snapshot(env)
    action = np.asarray(
        [*observed["tcp"], *observed["quaternion"], 1.0], dtype=np.float32
    )
    for _ in range(25):
        if _step(env, action):
            raise RuntimeError("Environment terminated during initial settling")
    validate_simulation_state(env, phase="settled_initial_state")
    return _snapshot(env)


def _transition(
    env, controller, observed: dict, history: dict, step: int
) -> tuple[dict, bool]:
    from npa.workflows.physical_augmentation_scene import camera_frame

    action = controller.action(
        observed["tcp"], observed["object"], observed["quaternion"]
    )
    frame = camera_frame(env)
    try:
        if _step(env, action):
            return observed, True
    except SimulationValidityError as error:
        error.evidence["input_action"] = action.tolist()
        error.evidence["controller_phase"] = controller.phase
        error.evidence["last_valid_observation"] = {
            key: value.tolist() for key, value in observed.items()
        }
        raise
    after = _snapshot(env)
    row = {
        "state": observed["state"],
        "object": observed["object"],
        "tcp": observed["tcp"],
        "rgb": frame,
        "actions": action,
        "next_state": after["state"],
        "next_object": after["object"],
        "next_tcp": after["tcp"],
        "next_velocity": after["velocity"],
        "timestamp": step * float(env.step_dt),
    }
    for key, value in row.items():
        history.setdefault(key, []).append(value)
    return after, False


def _episode(env, recipe: dict, condition: str, index: int, output: Path) -> dict:
    initial = _settle(env, recipe["seed"] + index)
    controller = LiftController(float(env.step_dt))
    observed, history, terminated = initial, {}, False
    for step in range(recipe["episode_steps"]):
        observed, terminated = _transition(env, controller, observed, history, step)
        if terminated:
            break
        tail = {
            key: np.asarray(value[-recipe["success"]["hold_steps"] :])
            for key, value in history.items()
            if key != "rgb"
        }
        if (
            longest_hold(accepted_steps(tail, initial["object"], recipe["success"]))
            >= recipe["success"]["hold_steps"]
        ):
            break
    if not history:
        raise RuntimeError("Attempt produced no non-reset transitions")
    arrays = {key: np.asarray(value) for key, value in history.items()}
    hold = longest_hold(accepted_steps(arrays, initial["object"], recipe["success"]))
    output.mkdir(parents=True)
    for key, value in arrays.items():
        np.save(output / f"{key}.npy", value, allow_pickle=False)
    result = {
        "condition": condition,
        "attempt": index,
        "seed": recipe["seed"] + index,
        "length": len(arrays["actions"]),
        "terminated": terminated,
        "controller_phase": controller.phase,
        "initial_object_m": initial["object"].tolist(),
        "longest_hold_steps": hold,
        "success": not terminated and hold >= recipe["success"]["hold_steps"],
    }
    write_json(output / "result.json", result)
    return result


def _collect(config, recipe: dict, condition: str, output: Path) -> None:
    import gymnasium as gym
    import torch
    from npa.workflows.physical_augmentation_scene import solver_evidence

    env = gym.make(recipe["task"], cfg=config).unwrapped
    solver = solver_evidence(env, recipe)
    with torch.inference_mode():
        results = [
            _episode(env, recipe, condition, index, output / f"episode_{index:06d}")
            for index in range(recipe["episodes_per_condition"])
        ]
    metadata = {
        "schema": "npa.physical-augmentation.capture.v1",
        "condition": condition,
        "episodes": results,
        "state_names": list(env.scene["robot"].joint_names),
        "control_dt": float(env.step_dt),
        "runtime_version": version("isaaclab"),
        "physics": _physics(env, recipe["conditions"][condition]),
        "simulation_validity_checks": env.npa_validity_checks,
        "tcp_contract": recipe["tcp_contract"],
        "tool_frame_checked": True,
        "gripper_servo": recipe["gripper_servo"],
        "physics_solver": solver,
        "presentation": recipe["presentation"],
    }
    write_json(output / "capture.json", metadata)
    env.close()


def _collect_with_fault_record(config, recipe, condition, output: Path) -> None:
    # Isaac's launcher may consume exceptions on context exit. Retain the
    # measured fault before it crosses that boundary.
    try:
        _collect(config, recipe, condition, output)
    except SimulationValidityError as error:
        write_json(output / "simulation-validity-failure.json", error.evidence)
        raise


def main(argv: list[str] | None = None) -> int:
    """Launch the pinned native Isaac runtime for one physical condition.

    Args:
        argv: Local prepared recipe, condition, output, and launcher arguments.
    Returns:
        Zero after real simulation and complete artifact capture.
    Raises:
        RuntimeError: Simulation or camera capture fails.
        ValueError: Physics, state validity, or requested settings are invalid.
    """
    from isaaclab_tasks.utils import add_launcher_args, launch_simulation
    from isaaclab.utils.seed import configure_seed

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    add_launcher_args(parser)
    args = parser.parse_args(argv)
    recipe = read_recipe(args.input_path / "recipe.json")
    configure_seed(recipe["seed"])
    os.environ["OMNI_TELEMETRY_DISABLE_ANONYMOUS_DATA"] = "1"
    config = _configuration(recipe, args.condition)
    args.enable_cameras = True
    with launch_simulation(config, args):
        _collect_with_fault_record(config, recipe, args.condition, args.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
