"""Record real MuJoCo Fetch pick-and-place trajectories and physics acceptance checks."""

from __future__ import annotations

from contextlib import contextmanager
from importlib import metadata
from pathlib import Path

import numpy as np

from npa.adapter.sim_to_lerobot import encode_video
from npa.workbench.token_factory import TokenFactoryToolError
from .robot_scene import COLORS

ENVIRONMENT = "FetchPickAndPlace-v4"
FPS = 25
JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "upperarm_roll_joint",
    "elbow_flex_joint",
    "forearm_roll_joint",
    "wrist_flex_joint",
    "wrist_roll_joint",
    "r_gripper_finger_joint",
    "l_gripper_finger_joint",
]
ACTION_NAMES = ["cartesian_dx", "cartesian_dy", "cartesian_dz", "gripper"]


def runtime_versions():
    """Check the optional simulation runtime before spending hosted inference calls.

    Args:
        None.
    Returns:
        Installed simulator and environment package versions.
    Raises:
        TokenFactoryToolError: A simulation dependency is missing.
    """
    try:
        return {
            name: metadata.version(name) for name in ("mujoco", "gymnasium-robotics")
        }
    except metadata.PackageNotFoundError:
        raise TokenFactoryToolError(
            "Install the npa[robot-sdg] extra for robot simulation"
        ) from None


def _configure_scene(env, scene):
    import mujoco

    base = env.initial_gripper_xpos[:2]
    pose = env.data.joint("object0:joint").qpos.copy()
    pose[:2] = base + [scene["object_x"], scene["object_y"]]
    env.data.joint("object0:joint").qpos[:] = pose
    env.goal = np.r_[base + [scene["goal_x"], scene["goal_y"]], env.height_offset]
    cube = env.model.geom("object0")
    cube.rgba[:] = COLORS[scene["object_color"]]
    cube.matid[:] = -1
    target = env.model.site("target0")
    target.rgba[:] = COLORS[scene["target_color"]]
    target.size[:] = [0.035, 0.002, 0.0]
    target.type[:] = int(mujoco.mjtGeom.mjGEOM_CYLINDER)
    env.model.light_diffuse[:] *= scene["lighting"]
    env.model.light_ambient[:] *= scene["lighting"]
    mujoco.mj_forward(env.model, env.data)
    _mount_wrist_camera(env)
    env._render_callback()


def _mount_wrist_camera(env):
    import mujoco

    camera = env.model.camera("gripper_camera_rgb")
    body = env.data.body(int(camera.bodyid[0]))
    gripper = env._get_obs()["observation"][:3]
    position = gripper + [0, -0.16, 0.035]
    target = gripper + [0, 0, -0.035]
    backward = position - target
    backward /= np.linalg.norm(backward)
    right = np.cross([0, 0, 1], backward)
    right /= np.linalg.norm(right)
    up = np.cross(backward, right)
    body_rotation = body.xmat.reshape(3, 3)
    camera.pos[:] = body_rotation.T @ (position - body.xpos)
    rotation = body_rotation.T @ np.column_stack((right, up, backward))
    mujoco.mju_mat2Quat(camera.quat, rotation.ravel())
    camera.fovy[:] = 65


@contextmanager
def _world(scene, seed, width, height):
    import gymnasium as gym
    import gymnasium_robotics
    import mujoco

    gym.register_envs(gymnasium_robotics)
    env = gym.make(ENVIRONMENT, width=width, height=height).unwrapped
    renderer = None
    try:
        env.reset(seed=seed)
        _configure_scene(env, scene)
        env.model.vis.global_.offwidth = width
        env.model.vis.global_.offheight = height
        renderer = mujoco.Renderer(env.model, height=height, width=width)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [1.30, 0.75, 0.48]
        camera.distance, camera.azimuth, camera.elevation = 1.65, 132, -27
        yield env, renderer, camera
    finally:
        if renderer is not None:
            renderer.close()
        env.close()


def _phases(object_position, goal):
    return [
        ("approach", object_position + [0, 0, 0.12], 1.0, 25),
        ("descend", object_position, 1.0, 25),
        ("grasp", object_position, -1.0, 15),
        ("lift", object_position + [0, 0, 0.15], -1.0, 25),
        ("transfer", goal + [0, 0, 0.15], -1.0, 30),
        ("lower", goal, -1.0, 25),
        ("release", goal, 1.0, 15),
        ("retreat", goal + [0, 0, 0.15], 1.0, 25),
    ]


def _robot_state(env):
    return np.array(
        [env.data.joint("robot0:" + name).qpos[0] for name in JOINT_NAMES],
        dtype=np.float32,
    )


def _finger_contacts(env):
    object_id = env.model.geom("object0").id
    fingers = set()
    for contact in env.data.contact:
        if object_id not in (contact.geom1, contact.geom2):
            continue
        other = contact.geom2 if contact.geom1 == object_id else contact.geom1
        body_name = env.model.body(int(env.model.geom_bodyid[other])).name
        if "gripper_finger" in body_name:
            fingers.add(body_name)
    return len(fingers)


def _frames(env, renderer, camera):
    renderer.update_scene(env.data, camera=camera)
    workspace = renderer.render().copy()
    renderer.update_scene(env.data, camera="gripper_camera_rgb")
    wrist = renderer.render().copy()
    return workspace, wrist


def _sample_step(world, observation, target, gripper):
    env, renderer, camera = world
    action = np.r_[
        np.clip((target - observation["observation"][:3]) * 8, -1, 1), gripper
    ].astype(np.float32)
    workspace, wrist = _frames(env, renderer, camera)
    state = _robot_state(env)
    next_observation, reward, _, _, info = env.step(action)
    sample = {
        "obs_workspace": workspace,
        "obs_wrist": wrist,
        "state": state,
        "actions": action,
        "next_state": _robot_state(env),
        "object_position": observation["achieved_goal"],
        "next_object_position": next_observation["achieved_goal"],
        "gripper_position": observation["observation"][:3],
        "next_gripper_position": next_observation["observation"][:3],
        "finger_contacts": _finger_contacts(env),
        "reward": float(reward),
        "environment_success": bool(info["is_success"]),
    }
    return next_observation, sample


def _rollout(world):
    env = world[0]
    observation = env._get_obs()
    samples, phases = [], []
    for name, target, gripper, steps in _phases(observation["achieved_goal"], env.goal):
        for _ in range(steps):
            observation, sample = _sample_step(world, observation, target, gripper)
            samples.append(sample)
            phases.append(name)
    arrays = {key: np.stack([sample[key] for sample in samples]) for key in samples[0]}
    arrays["phase"] = np.array(phases)
    arrays["goal"] = env.goal.copy()
    return arrays


def physics_checks(arrays):
    """Judge grasp, lift, placement, release, and settling from recorded simulator state.

    Args:
        arrays: Synchronized observations, actions, contacts, phases, and next states.
    Returns:
        Numeric measurements and independent boolean checks; all must pass.
    Raises:
        ValueError: Required trace arrays have invalid shapes.
    """
    positions = arrays["next_object_position"]
    final_distance = float(np.linalg.norm(positions[-1] - arrays["goal"]))
    lift_height = float(positions[:, 2].max() - arrays["object_position"][0, 2])
    settled_speed = float(
        np.linalg.norm(np.diff(positions[-11:], axis=0), axis=1).max() * FPS
    )
    checks = {
        "finite": all(
            np.isfinite(arrays[key]).all()
            for key in ("state", "actions", "next_state", "next_object_position")
        ),
        "bilateral_grasp_contact": bool(np.any(arrays["finger_contacts"] >= 2)),
        "lifted": lift_height >= 0.08,
        "placed": final_distance <= 0.025,
        "released": bool(np.all(arrays["next_state"][-1, -2:] >= 0.04)),
        "retreated": bool(
            arrays["next_gripper_position"][-1, 2] - positions[-1, 2] >= 0.10
        ),
        "settled": settled_speed <= 0.02,
        "environment_success": bool(arrays["environment_success"][-1]),
    }
    return {
        "accepted": all(checks.values()),
        "checks": checks,
        "final_distance_m": final_distance,
        "lift_height_m": lift_height,
        "settled_max_speed_m_s": settled_speed,
        "bilateral_contact_frames": int(
            np.count_nonzero(arrays["finger_contacts"] >= 2)
        ),
        "judge": "mujoco_state_and_contacts",
    }


def _write_recording(arrays, output):
    output.mkdir(parents=True, exist_ok=False)
    for name in ("obs_workspace", "obs_wrist", "state", "actions"):
        np.save(output / f"{name}.npy", arrays[name], allow_pickle=False)
    trace = {key: value for key, value in arrays.items() if not key.startswith("obs_")}
    np.savez_compressed(output / "physics.npz", **trace)
    preview = np.concatenate((arrays["obs_workspace"], arrays["obs_wrist"]), axis=2)
    encode_video(preview, output / "preview.mp4", FPS)


def simulate_robot_episode(scene, *, seed, output, width=480, height=360):
    """Run one real Fetch demonstration and record aligned RGB, state, and actions.

    Args:
        scene: Validated RobotScene mapping.
        seed: Simulator reset seed.
        output: New episode directory for raw arrays, physics trace, and preview MP4.
        width, height: Resolution of each camera stream, in pixels.
    Returns:
        Physics acceptance, frame count, task, action and observation contracts.
    Raises:
        RuntimeError: MuJoCo simulation or rendering fails.
        OSError: Recording or video publication fails.
    """
    with _world(scene, seed, width, height) as world:
        arrays = _rollout(world)
    verdict = physics_checks(arrays)
    _write_recording(arrays, Path(output))
    return {
        **verdict,
        "frames": len(arrays["actions"]),
        "fps": FPS,
        "environment": ENVIRONMENT,
        "robot_type": "fetch",
        "task": f"Pick up the {scene['object_color']} cube and place it on the {scene['target_color']} target, then release it.",
        "state_names": JOINT_NAMES,
        "action_names": ACTION_NAMES,
        "action_contract": "normalized Cartesian delta xyz (scale 0.05 m) and gripper command in [-1, 1]",
        "alignment": "observation_t and state_t precede action_t; physics.npz also records state_t+1",
        "controller": "scripted_privileged_state_pick_place_v1",
    }
