"""Independently replay Fetch recordings through upstream MuJoCo without candidate helpers."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import numpy as np

_JOINTS = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "upperarm_roll_joint",
    "elbow_flex_joint",
    "forearm_roll_joint",
    "wrist_flex_joint",
    "wrist_roll_joint",
    "r_gripper_finger_joint",
    "l_gripper_finger_joint",
)
_COLORS = {
    "red": [0.85, 0.12, 0.10, 1],
    "green": [0.12, 0.70, 0.22, 1],
    "blue": [0.10, 0.30, 0.85, 1],
    "orange": [0.95, 0.45, 0.08, 1],
}
_TRACE_SHAPES = {
    "state": (185, 9),
    "actions": (185, 4),
    "next_state": (185, 9),
    "object_position": (185, 3),
    "next_object_position": (185, 3),
    "gripper_position": (185, 3),
    "next_gripper_position": (185, 3),
    "finger_contacts": (185,),
    "reward": (185,),
    "environment_success": (185,),
    "phase": (185,),
    "goal": (3,),
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _configure(env, scene):
    import mujoco

    base = env.initial_gripper_xpos[:2]
    env.data.joint("object0:joint").qpos[:2] = base + [
        scene["object_x"],
        scene["object_y"],
    ]
    env.goal = np.r_[base + [scene["goal_x"], scene["goal_y"]], env.height_offset]
    env.model.geom("object0").rgba[:] = _COLORS[scene["object_color"]]
    env.model.geom("object0").matid[:] = -1
    marker = env.model.site("target0")
    marker.rgba[:], marker.size[:] = _COLORS[scene["target_color"]], [0.035, 0.002, 0]
    marker.type[:] = int(mujoco.mjtGeom.mjGEOM_CYLINDER)
    env.model.light_diffuse[:] *= scene["lighting"]
    env.model.light_ambient[:] *= scene["lighting"]
    mujoco.mj_forward(env.model, env.data)
    _wrist_camera(env)
    env._render_callback()


def _wrist_camera(env):
    import mujoco

    camera = env.model.camera("gripper_camera_rgb")
    body = env.data.body(int(camera.bodyid[0]))
    gripper = env._get_obs()["observation"][:3]
    position, target = gripper + [0, -0.16, 0.035], gripper + [0, 0, -0.035]
    backward = position - target
    backward /= np.linalg.norm(backward)
    right = np.cross([0, 0, 1], backward)
    right /= np.linalg.norm(right)
    up = np.cross(backward, right)
    rotation = body.xmat.reshape(3, 3)
    camera.pos[:] = rotation.T @ (position - body.xpos)
    mujoco.mju_mat2Quat(
        camera.quat, (rotation.T @ np.column_stack((right, up, backward))).ravel()
    )
    camera.fovy[:] = 65


@contextmanager
def _world(case):
    import gymnasium as gym
    import gymnasium_robotics
    import mujoco

    gym.register_envs(gymnasium_robotics)
    env = gym.make("FetchPickAndPlace-v4", width=480, height=360).unwrapped
    renderer = None
    try:
        env.reset(seed=case["seed"])
        _configure(env, case["scene"])
        env.model.vis.global_.offwidth, env.model.vis.global_.offheight = 480, 360
        renderer = mujoco.Renderer(env.model, height=360, width=480)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [1.30, 0.75, 0.48]
        camera.distance, camera.azimuth, camera.elevation = 1.65, 132, -27
        yield env, renderer, camera
    finally:
        if renderer is not None:
            renderer.close()
        env.close()


def _state(env):
    return np.array(
        [env.data.joint("robot0:" + name).qpos[0] for name in _JOINTS], dtype=np.float32
    )


def _contact_count(env):
    cube = env.model.geom("object0").id
    fingers = set()
    for contact in env.data.contact:
        if cube in (contact.geom1, contact.geom2):
            other = contact.geom2 if contact.geom1 == cube else contact.geom1
            name = env.model.body(int(env.model.geom_bodyid[other])).name
            if "gripper_finger" in name:
                fingers.add(name)
    return len(fingers)


def _schedule(position, goal, controller):
    rows = [
        ("approach", position + [0, 0, 0.12], 1, 25),
        ("descend", position, 1, 25),
        ("grasp", position, -1, 15),
        ("lift", position + [0, 0, 0.15], -1, 25),
        ("transfer", goal + [0, 0, 0.15], -1, 30),
        ("lower", goal, -1, 25),
        ("release", goal, 1, 15),
        ("retreat", goal + [0, 0, 0.15], 1, 25),
    ]
    return [
        (name, target, 1 if controller == "open_gripper" else gripper)
        for name, target, gripper, count in rows
        for _ in range(count)
    ]


def _arrays(episode):
    result = {
        name: np.load(episode / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        for name in ("state", "actions", "obs_workspace", "obs_wrist")
    }
    with np.load(episode / "physics.npz", allow_pickle=False) as trace:
        _require(set(trace.files) == set(_TRACE_SHAPES), "physics trace fields differ")
        for name in trace.files:
            if name in result:
                np.testing.assert_array_equal(
                    result[name], trace[name], err_msg="trace/raw disagreement"
                )
            result[name] = trace[name]
    _trace_shapes(result)
    _require(
        np.isfinite(result["actions"]).all() and (abs(result["actions"]) <= 1).all(),
        "invalid actions",
    )
    for name in ("obs_workspace", "obs_wrist"):
        _require(
            result[name].shape == (185, 360, 480, 3) and result[name].dtype == np.uint8,
            "camera array contract",
        )
    return result


def _trace_shapes(arrays):
    for name, shape in _TRACE_SHAPES.items():
        value = arrays[name]
        _require(value.shape == shape, f"physics trace shape differs: {name}")
        if name == "phase":
            _require(value.dtype.kind == "U", "phase trace must contain text")
        elif name == "environment_success":
            _require(value.dtype == np.bool_, "native success trace must be boolean")
        else:
            _require(
                value.dtype.kind in "iuf" and np.isfinite(value).all(),
                f"nonfinite or nonnumeric trace: {name}",
            )


def _before_step(world, arrays, index, target, gripper):
    env, renderer, camera = world
    obs = env._get_obs()
    expected = np.r_[
        np.clip((target - obs["observation"][:3]) * 8, -1, 1), gripper
    ].astype(np.float32)
    np.testing.assert_allclose(
        arrays["actions"][index],
        expected,
        rtol=0,
        atol=1e-6,
        err_msg="controller action",
    )
    np.testing.assert_allclose(
        arrays["state"][index],
        _state(env),
        rtol=0,
        atol=1e-6,
        err_msg="pre-action joint state",
    )
    np.testing.assert_allclose(
        arrays["object_position"][index], obs["achieved_goal"], rtol=0, atol=1e-6
    )
    np.testing.assert_allclose(
        arrays["gripper_position"][index], obs["observation"][:3], rtol=0, atol=1e-6
    )
    for key, view in (("obs_workspace", camera), ("obs_wrist", "gripper_camera_rgb")):
        renderer.update_scene(env.data, camera=view)
        np.testing.assert_array_equal(
            arrays[key][index], renderer.render(), err_msg="pre-action camera frame"
        )


def _after_step(env, arrays, index):
    observation, reward, _, _, info = env.step(arrays["actions"][index])
    state = _state(env)
    np.testing.assert_allclose(
        arrays["next_state"][index],
        state,
        rtol=0,
        atol=1e-6,
        err_msg="post-action joint state",
    )
    np.testing.assert_allclose(
        arrays["next_object_position"][index],
        observation["achieved_goal"],
        rtol=0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        arrays["next_gripper_position"][index],
        observation["observation"][:3],
        rtol=0,
        atol=1e-6,
    )
    _require(
        arrays["finger_contacts"][index] == _contact_count(env), "contact trace differs"
    )
    _require(arrays["reward"][index] == reward, "reward trace differs")
    _require(
        bool(arrays["environment_success"][index]) == bool(info["is_success"]),
        "native success trace differs",
    )


def _physics(arrays):
    positions = arrays["next_object_position"]
    distance = float(np.linalg.norm(positions[-1] - arrays["goal"]))
    lift = float(positions[:, 2].max() - arrays["object_position"][0, 2])
    speed = float(np.linalg.norm(np.diff(positions[-11:], axis=0), axis=1).max() * 25)
    checks = {
        "finite": all(
            np.isfinite(arrays[name]).all()
            for name in ("state", "actions", "next_state", "next_object_position")
        ),
        "bilateral_grasp_contact": bool(np.any(arrays["finger_contacts"] >= 2)),
        "lifted": lift >= 0.08,
        "placed": distance <= 0.025,
        "released": bool(np.all(arrays["next_state"][-1, -2:] >= 0.04)),
        "retreated": bool(
            arrays["next_gripper_position"][-1, 2] - positions[-1, 2] >= 0.10
        ),
        "settled": speed <= 0.02,
        "environment_success": bool(arrays["environment_success"][-1]),
    }
    return {
        "accepted": all(checks.values()),
        "checks": checks,
        "final_distance_m": distance,
        "lift_height_m": lift,
        "settled_max_speed_m_s": speed,
        "bilateral_contact_frames": int(
            np.count_nonzero(arrays["finger_contacts"] >= 2)
        ),
        "judge": "mujoco_state_and_contacts",
    }


def replay_episode(episode: Path, case: dict) -> dict:
    """Replay every native transition and both pre-action camera streams independently.

    Args: episode: Recorded local arrays. case: Frozen scene/reset/controller contract.
    Returns: Independently measured physics checks and verified frame counts.
    Raises: ValueError, AssertionError, RuntimeError: Artifact or native replay differs.
    """
    arrays = _arrays(episode)
    with _world(case) as world:
        env = world[0]
        np.testing.assert_allclose(arrays["goal"], env.goal, rtol=0, atol=1e-6)
        schedule = _schedule(
            env._get_obs()["achieved_goal"].copy(), env.goal.copy(), case["controller"]
        )
        for index, (phase, target, gripper) in enumerate(schedule):
            _require(arrays["phase"][index] == phase, "controller phase differs")
            _before_step(world, arrays, index, target, gripper)
            _after_step(env, arrays, index)
    result = _physics(arrays)
    _require(
        result["accepted"] == case["expected_accepted"],
        "physical outcome differs from prepared matrix",
    )
    return {**result, "replayed_steps": 185, "replayed_camera_frames": 370}
