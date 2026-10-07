"""Real Isaac collector for the pi0.5 released-surface pick-and-place task.

The expert commands the Franka end effector through Isaac's differential-IK
action term.  It never writes object pose or velocity.  Every admitted training
row contains synchronous exterior/wrist RTX pixels and the arm controller's
actual absolute joint target, rather than the expert's Cartesian command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

from npa.workflows.sim2real.pi05_contract import (
    CONTROL_HZ,
    EPISODE_SCHEMA,
    JOINT_NAMES,
    MAX_GRIPPER_WIDTH_M,
    build_contract,
)


EXPERT_BASE_TASK_ID = "Isaac-Lift-Cube-Franka-IK-Rel-v0"
EXPERT_SURFACE_TASK_ID = "NPA-Pi05-SurfacePlace-Franka-IK-Rel-v0"
POLICY_BASE_TASK_ID = "Isaac-Lift-Cube-Franka-v0"
POLICY_SURFACE_TASK_ID = "NPA-Pi05-SurfacePlace-Franka-v0"


def register_surface_task(gym: Any, *, base_id: str, task_id: str) -> None:
    """Register an owned surface-task ID over an explicitly configured base."""

    try:
        gym.spec(task_id)
        return
    except gym.error.Error:
        base = gym.spec(base_id)
    gym.register(id=task_id, entry_point=base.entry_point, disable_env_checker=True)


def configure_surface_goal(cfg: Any, target: tuple[float, float, float]) -> None:
    """Replace the stock airborne goal with one object center on the support."""

    command = cfg.commands.object_pose
    command.resampling_time_range = (1.0e9, 1.0e9)
    command.ranges.pos_x = (target[0], target[0])
    command.ranges.pos_y = (target[1], target[1])
    command.ranges.pos_z = (target[2], target[2])


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _camera_config(env_cfg: Any) -> None:
    """Install a fixed exterior camera and a real panda-hand child camera."""

    import isaaclab.sim as sim_utils
    from isaaclab.sensors import TiledCameraCfg

    common = {
        "data_types": ["rgb", "semantic_segmentation"],
        "width": 224,
        "height": 224,
        "update_period": 0.0,
        "spawn": sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            horizontal_aperture=20.955,
            clipping_range=(0.03, 10.0),
        ),
    }
    env_cfg.scene.pi05_exterior = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Pi05ExteriorCamera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(1.25, -1.25, 1.05),
            rot=(0.683013, 0.183013, 0.183013, 0.683013),
            convention="world",
        ),
        **common,
    )
    # The prim is below panda_hand, so its world extrinsics advance with every
    # articulation step.  A world/static camera with a wrist-like label cannot
    # satisfy this contract.
    env_cfg.scene.pi05_wrist = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/Pi05WristCamera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.045, 0.0, 0.035),
            rot=(0.0, 0.0, 1.0, 0.0),
            convention="world",
        ),
        **common,
    )


def _contact_config(env_cfg: Any) -> None:
    """Enable filtered bilateral finger/object and object/table reporting."""

    from isaaclab.sensors import ContactSensorCfg

    env_cfg.scene.robot.spawn.activate_contact_sensors = True
    env_cfg.scene.object.spawn.activate_contact_sensors = True
    env_cfg.scene.left_object_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        history_length=3,
    )
    env_cfg.scene.right_object_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        history_length=3,
    )
    env_cfg.scene.object_support_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Table/.*"],
        history_length=3,
    )


def _physics_randomization(env_cfg: Any) -> None:
    from isaaclab.envs import mdp
    from isaaclab.managers import EventTermCfg, SceneEntityCfg

    env_cfg.events.pi05_object_mass = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.85, 1.15),
            "operation": "scale",
        },
    )
    env_cfg.events.pi05_object_material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (0.45, 1.25),
            "dynamic_friction_range": (0.45, 1.25),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 32,
        },
    )


def _rgb(native: Any, sensor_name: str) -> Any:
    import numpy as np

    value = native.scene[sensor_name].data.output["rgb"]
    if hasattr(value, "torch"):
        value = value.torch
    frame = value[0, ..., :3].detach().cpu().numpy()
    if frame.shape != (224, 224, 3) or frame.dtype != np.uint8:
        raise RuntimeError(
            f"{sensor_name} did not produce uint8[224,224,3]: {frame.shape} {frame.dtype}"
        )
    if int(frame.max()) - int(frame.min()) < 8:
        raise RuntimeError(f"{sensor_name} produced a blank RTX observation")
    return frame.copy()


def _target_pixel_count(native: Any, sensor_name: str) -> int:
    import numpy as np

    sensor = native.scene[sensor_name]
    labels = sensor.data.info.get("semantic_segmentation", {}).get("idToLabels", {})
    target_ids = [
        int(label_id)
        for label_id, value in labels.items()
        if "pi05_target_object" in str(value)
    ]
    if not target_ids:
        raise RuntimeError(f"{sensor_name} semantic output has no target-object label")
    segmentation = sensor.data.output["semantic_segmentation"][0]
    if hasattr(segmentation, "detach"):
        segmentation = segmentation.detach().cpu().numpy()
    return int(np.isin(np.asarray(segmentation), target_ids).sum())


def _camera_pose_sample(native: Any) -> dict[str, list[float]]:
    robot = native.scene["robot"]
    hand_index = robot.body_names.index("panda_hand")

    def values(tensor: Any) -> list[float]:
        return tensor.detach().cpu().numpy().astype(float).tolist()

    return {
        "hand_position_m": values(robot.data.body_pos_w[0, hand_index]),
        "wrist_camera_position_m": values(native.scene["pi05_wrist"].data.pos_w[0]),
        "exterior_camera_position_m": values(
            native.scene["pi05_exterior"].data.pos_w[0]
        ),
    }


def _verify_camera_motion(traces: list[dict[str, Any]]) -> dict[str, float]:
    import numpy as np

    hand = np.asarray([row["camera_pose"]["hand_position_m"] for row in traces])
    wrist = np.asarray(
        [row["camera_pose"]["wrist_camera_position_m"] for row in traces]
    )
    exterior = np.asarray(
        [row["camera_pose"]["exterior_camera_position_m"] for row in traces]
    )
    wrist_world_motion = float(np.linalg.norm(wrist - wrist[0], axis=1).max())
    hand_world_motion = float(np.linalg.norm(hand - hand[0], axis=1).max())
    relative = np.linalg.norm(wrist - hand, axis=1)
    relative_drift = float(np.ptp(relative))
    exterior_drift = float(np.linalg.norm(exterior - exterior[0], axis=1).max())
    if wrist_world_motion < 0.05 or hand_world_motion < 0.05:
        raise RuntimeError("wrist camera/hand did not move through the manipulation")
    if relative_drift > 0.002 or exterior_drift > 1.0e-4:
        raise RuntimeError(
            "camera extrinsics violate moving-wrist/static-exterior contract"
        )
    if min(int(row["wrist_target_pixels"]) for row in traces) < 4:
        raise RuntimeError("wrist camera did not keep the target object in view")
    if min(int(row["exterior_target_pixels"]) for row in traces) < 4:
        raise RuntimeError("exterior camera did not keep the target object in view")
    return {
        "wrist_world_motion_m": wrist_world_motion,
        "hand_world_motion_m": hand_world_motion,
        "wrist_mount_distance_drift_m": relative_drift,
        "exterior_world_drift_m": exterior_drift,
    }


def _action_layout(native: Any) -> tuple[int, int, int]:
    names = list(native.action_manager.active_terms)
    dimensions = [int(v) for v in native.action_manager.action_term_dim]
    if "arm_action" not in names or "gripper_action" not in names:
        raise RuntimeError("Isaac task lacks named arm_action/gripper_action terms")
    arm_index, grip_index = names.index("arm_action"), names.index("gripper_action")
    arm_start = sum(dimensions[:arm_index])
    grip_start = sum(dimensions[:grip_index])
    if dimensions[arm_index] != 6 or dimensions[grip_index] != 1:
        raise RuntimeError("expert requires 6D differential IK plus binary gripper")
    return sum(dimensions), arm_start, grip_start


def _controller_target(native: Any) -> Any:
    import numpy as np

    robot = native.scene["robot"]
    names = list(robot.joint_names)
    indices = [names.index(name) for name in JOINT_NAMES]
    value = (
        robot.data.joint_pos_target[0, indices]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=True)
    )
    if value.shape != (7,) or not np.isfinite(value).all():
        raise RuntimeError(f"invalid absolute controller target: {value.shape}")
    return value


def _gripper_width(native: Any) -> float:
    import numpy as np

    robot = native.scene["robot"]
    names = list(robot.joint_names)
    indices = [
        names.index(name) for name in ("panda_finger_joint1", "panda_finger_joint2")
    ]
    q = robot.data.joint_pos[0, indices].detach().cpu().numpy()
    width = float(np.sum(q))
    if not 0.0 <= width <= MAX_GRIPPER_WIDTH_M + 0.005:
        raise RuntimeError(f"measured physical finger width is invalid: {width}")
    return width


def _finger_controller_target(native: Any) -> Any:
    import numpy as np

    robot = native.scene["robot"]
    names = list(robot.joint_names)
    indices = [
        names.index(name) for name in ("panda_finger_joint1", "panda_finger_joint2")
    ]
    target = robot.data.joint_pos_target[0, indices].detach().cpu().numpy()
    if target.shape != (2,) or not np.isfinite(target).all():
        raise RuntimeError("Isaac did not expose finite two-finger controller targets")
    return target.astype(np.float32, copy=True)


def _wrist_mount_evidence(native: Any) -> dict[str, Any]:
    import numpy as np

    wrist = native.scene["pi05_wrist"].data
    hand_index = native.scene["robot"].body_names.index("panda_hand")
    hand = native.scene["robot"].data.body_pos_w[0, hand_index].detach().cpu().numpy()
    camera = wrist.pos_w[0].detach().cpu().numpy()
    distance = float(np.linalg.norm(camera - hand))
    if not 0.02 <= distance <= 0.12:
        raise RuntimeError("wrist camera world pose is not attached near panda_hand")
    return {
        "parent_prim": "{ENV_REGEX_NS}/Robot/panda_hand",
        "mount_translation_m": [0.045, 0.0, 0.035],
        "mount_quaternion_wxyz": [0.0, 0.0, 1.0, 0.0],
        "measured_camera_to_hand_distance_m": distance,
        "moving_with_end_effector": True,
    }


def _contact_forces(native: Any) -> tuple[float, float, float]:
    import torch

    values = []
    for name in (
        "left_object_contact",
        "right_object_contact",
        "object_support_contact",
    ):
        sensor = native.scene[name]
        force = getattr(sensor.data, "force_matrix_w", None)
        if force is None:
            raise RuntimeError(f"{name} did not publish filtered contact forces")
        values.append(float(torch.linalg.norm(force).item()))
    return values[0], values[1], values[2]


def _sim_time(native: Any) -> float:
    return float(native.common_step_counter) * float(native.step_dt)


def _camera_time(native: Any, name: str) -> float:
    sensor = native.scene[name]
    value = getattr(sensor, "timestamp", None)
    if value is None:
        value = getattr(sensor, "_timestamp", None)
    if value is None:
        raise RuntimeError(f"{name} exposes no simulator capture timestamp")
    if hasattr(value, "detach"):
        value = value[0].detach().cpu().item()
    return float(value)


def _expert_waypoints(
    object_position: Any, target: Any
) -> list[tuple[str, Any, float]]:
    return [
        ("approach", object_position + [0.0, 0.0, 0.14], 0.0),
        ("contact", object_position + [0.0, 0.0, 0.025], 0.0),
        ("grasp", object_position + [0.0, 0.0, 0.025], 1.0),
        ("lift", object_position + [0.0, 0.0, 0.16], 1.0),
        ("transport", target + [0.0, 0.0, 0.16], 1.0),
        ("lower", target + [0.0, 0.0, 0.025], 1.0),
        ("release", target + [0.0, 0.0, 0.025], 0.0),
        ("retreat", target + [0.0, 0.0, 0.16], 0.0),
    ]


def _causal_observation(native: Any, joint_indices: list[int]) -> dict[str, Any]:
    import numpy as np

    now = _sim_time(native)
    exterior_time = _camera_time(native, "pi05_exterior")
    wrist_time = _camera_time(native, "pi05_wrist")
    if max(abs(exterior_time - now), abs(wrist_time - now)) > native.step_dt:
        raise RuntimeError(
            "camera captures are stale relative to the causal observation"
        )
    q = native.scene["robot"].data.joint_pos[0, joint_indices]
    return {
        "exterior": _rgb(native, "pi05_exterior"),
        "wrist": _rgb(native, "pi05_wrist"),
        "exterior_pixels": _target_pixel_count(native, "pi05_exterior"),
        "wrist_pixels": _target_pixel_count(native, "pi05_wrist"),
        "camera_pose": _camera_pose_sample(native),
        "time": now,
        "exterior_time": exterior_time,
        "wrist_time": wrist_time,
        "q": q.detach().cpu().numpy().astype(np.float32, copy=True),
        "width": _gripper_width(native),
    }


def _expert_command(
    context: dict[str, Any], goal: Any, gripper: float
) -> tuple[Any, float]:
    import torch

    native = context["native"]
    ee = native.scene["ee_frame"].data.target_pos_w[0, 0, :3]
    error = torch.as_tensor(goal, device=ee.device) - ee
    command = torch.zeros((1, context["total_dim"]), device=ee.device)
    start = context["arm_start"]
    command[0, start : start + 3] = torch.clamp(error * 5.0, -0.35, 0.35)
    command[0, context["grip_start"]] = -1.0 if gripper > 0.5 else 1.0
    return command, float(torch.linalg.norm(error).item())


def _step_command(
    context: dict[str, Any], command: Any, phase: str
) -> tuple[float, float]:
    native, env = context["native"], context["env"]
    command_time = _sim_time(native)
    _, _, terminated, truncated, _ = env.step(command)
    resulting_time = _sim_time(native)
    measured_hold = resulting_time - command_time
    if abs(measured_hold - 1.0 / CONTROL_HZ) > 1.0e-6:
        raise RuntimeError(f"Isaac control cadence mismatch: {measured_hold} seconds")
    if bool(terminated[0]) or bool(truncated[0]):
        raise RuntimeError(f"Isaac reset during expert phase {phase}")
    return command_time, resulting_time


def _hand_object_distance(native: Any, object_position: Any) -> float:
    import numpy as np

    robot = native.scene["robot"]
    hand_index = list(robot.body_names).index("panda_hand")
    hand = robot.data.body_pos_w[0, hand_index].detach().cpu().numpy()
    return float(np.linalg.norm(hand - object_position))


def _physics_sample(
    context: dict[str, Any], phase: str, intent: float
) -> dict[str, Any]:
    import numpy as np

    native = context["native"]
    obj = native.scene["object"].data.root_pos_w[0, :3]
    obj_np = obj.detach().cpu().numpy()
    velocity = native.scene["object"].data.root_lin_vel_w[0].detach().cpu().numpy()
    speed = float(np.linalg.norm(velocity))
    left, right, support = _contact_forces(native)
    width = _gripper_width(native)
    bilateral = left > 1.0e-3 and right > 1.0e-3
    lifted = float(obj_np[2] - context["initial_z"]) >= 0.05
    xy_error = float(np.linalg.norm(obj_np[:2] - context["target"][:2]))
    released = width >= 0.06 and phase in {"release", "retreat"}
    retreat_distance = _hand_object_distance(native, obj_np)
    retreated = phase == "retreat" and retreat_distance >= 0.10
    supported = support > 1.0e-3 and not bilateral and released and retreated
    return {
        "object_position_m": obj_np.tolist(),
        "object_speed_m_s": speed,
        "left_finger_object_force_n": left,
        "right_finger_object_force_n": right,
        "object_support_force_n": support,
        "bilateral_finger_contact": bilateral,
        "support_contact": support > 1.0e-3,
        "gripper_width_m": width,
        "hand_object_distance_m": retreat_distance,
        "finger_controller_target_m": _finger_controller_target(native).tolist(),
        "target_xy_distance_m": xy_error,
        "lifted": lifted,
        "transported": xy_error < 0.08 and lifted,
        "released": released,
        "grasped": bilateral
        and intent > 0.5
        and abs(width - context["object_width_m"]) <= 0.015
        and (lifted or speed > 0.005),
        "supported": supported,
    }


def _update_events(context: dict[str, Any], sample: dict[str, Any]) -> None:
    stable = sample["supported"] and sample["target_xy_distance_m"] < 0.05
    stable = stable and sample["object_speed_m_s"] < 0.03
    context["stable_steps"] = context["stable_steps"] + 1 if stable else 0
    conditions = (
        ("contact", sample["bilateral_finger_contact"]),
        ("grasp", sample["grasped"]),
        ("lift", sample["lifted"]),
        ("transport", sample["transported"]),
        ("release", sample["released"]),
        ("support_contact", sample["supported"]),
        ("stable", context["stable_steps"] >= 3),
    )
    for name, condition in conditions:
        if condition and name not in context["buffers"]["events"]:
            context["buffers"]["events"].append(name)


def _record_step(
    context: dict[str, Any],
    observation: dict[str, Any],
    intent: float,
    phase: str,
    command_time: float,
    resulting_time: float,
) -> None:
    import numpy as np

    buffers, native = context["buffers"], context["native"]
    target_q = _controller_target(native)
    actual_q = native.scene["robot"].data.joint_pos[0, context["joint_indices"]]
    actual_q = actual_q.detach().cpu().numpy()
    buffers["controller_errors"].append(float(np.max(np.abs(target_q - actual_q))))
    buffers["frames_exterior"].append(observation["exterior"])
    buffers["frames_wrist"].append(observation["wrist"])
    state = np.r_[observation["q"], 1.0 - observation["width"] / MAX_GRIPPER_WIDTH_M]
    buffers["states"].append(state.astype(np.float32))
    buffers["actions"].append(np.r_[target_q, intent].astype(np.float32))
    buffers["timestamps"].append(observation["time"])
    sample = _physics_sample(context, phase, intent)
    _update_events(context, sample)
    sample.update(
        {
            "step": len(buffers["actions"]) - 1,
            "phase": phase,
            "observation_time_s": observation["time"],
            "exterior_capture_time_s": observation["exterior_time"],
            "wrist_capture_time_s": observation["wrist_time"],
            "command_application_time_s": command_time,
            "resulting_state_time_s": resulting_time,
            "camera_pose": observation["camera_pose"],
            "exterior_target_pixels": observation["exterior_pixels"],
            "wrist_target_pixels": observation["wrist_pixels"],
            "stable_steps": context["stable_steps"],
        }
    )
    buffers["traces"].append(sample)


def _run_phase(context: dict[str, Any], phase: str, goal: Any, intent: float) -> None:
    reached = 0
    for _ in range(90):
        command, distance = _expert_command(context, goal, intent)
        observation = _causal_observation(context["native"], context["joint_indices"])
        command_time, resulting_time = _step_command(context, command, phase)
        _record_step(context, observation, intent, phase, command_time, resulting_time)
        reached = reached + 1 if distance < 0.012 else 0
        hold = 12 if phase in {"grasp", "release"} else 3
        if reached >= hold or (phase == "retreat" and context["stable_steps"] >= 3):
            return
    raise RuntimeError(f"expert failed to reach {phase} waypoint")


def _write_episode_arrays(episode: Path, buffers: dict[str, list[Any]]) -> None:
    import numpy as np

    np.save(episode / "obs_workspace.npy", np.stack(buffers["frames_exterior"]))
    np.save(episode / "obs_wrist.npy", np.stack(buffers["frames_wrist"]))
    np.save(episode / "state.npy", np.stack(buffers["states"]).astype(np.float32))
    np.save(episode / "actions.npy", np.stack(buffers["actions"]).astype(np.float32))
    np.save(
        episode / "timestamps.npy", np.asarray(buffers["timestamps"], dtype=np.float64)
    )
    trace = buffers["traces"]
    keys = (
        "object_position_m",
        "object_speed_m_s",
        "left_finger_object_force_n",
        "right_finger_object_force_n",
        "object_support_force_n",
        "gripper_width_m",
        "hand_object_distance_m",
        "finger_controller_target_m",
    )
    np.savez(
        episode / "physics_trace.npz",
        **{
            key: np.asarray([row[key] for row in trace], dtype=np.float32)
            for key in keys
        },
    )


def _step_records(buffers: dict[str, list[Any]]) -> list[dict[str, Any]]:
    records = []
    for index, timestamp in enumerate(buffers["timestamps"]):
        trace = buffers["traces"][index]
        records.append(
            {
                "step": index,
                "timestamp_s": timestamp,
                "exterior_timestamp_s": trace["exterior_capture_time_s"],
                "wrist_timestamp_s": trace["wrist_capture_time_s"],
                "command_application_timestamp_s": trace["command_application_time_s"],
                "resulting_state_timestamp_s": trace["resulting_state_time_s"],
                "joint_position": buffers["states"][index][:7].tolist(),
                "gripper_position": [float(buffers["states"][index][7])],
                "action": buffers["actions"][index].tolist(),
                "exterior_image_1_left": {
                    "path": f"obs_workspace.npy#{index}",
                    "shape": [224, 224, 3],
                    "dtype": "uint8",
                },
                "wrist_image_left": {
                    "path": f"obs_wrist.npy#{index}",
                    "shape": [224, 224, 3],
                    "dtype": "uint8",
                },
            }
        )
    return records


def _controller_success_evidence(context: dict[str, Any]) -> dict[str, Any]:
    buffers = context["buffers"]
    events = buffers["events"]
    expected = [
        "contact",
        "grasp",
        "lift",
        "transport",
        "release",
        "support_contact",
        "stable",
    ]
    return {
        "controller": {
            "labels_are_actual_absolute_controller_targets": True,
            "maximum_tracking_error_rad": max(buffers["controller_errors"]),
            "hold_seconds": 1.0 / CONTROL_HZ,
            "gripper_convention": "DROID 0=open, 1=closed; predictions >0.5 close",
            "two_finger_targets_recorded_in_physics_trace": True,
        },
        "success_evidence": {
            "ordered_events": events,
            "released_supported_surface_placement": events == expected
            and context["stable_steps"] >= 3,
            "stable_steps": context["stable_steps"],
            "target_xy_distance_m_lt": 0.05,
            "object_speed_m_s_lt": 0.03,
            "physics_trace": "physics_trace.npz",
        },
    }


def _episode_evidence(
    context: dict[str, Any], mounted: dict[str, Any]
) -> dict[str, Any]:
    buffers = context["buffers"]
    count = len(buffers["actions"])
    return {
        "camera": {
            "exterior": {"mount": "world"},
            "wrist": mounted,
            "motion_and_visibility": _verify_camera_motion(buffers["traces"]),
            "semantic_target_visible_every_step": True,
        },
        "dense_alignment": {
            name: count
            for name in (
                "exterior_frames",
                "wrist_frames",
                "states",
                "actions",
                "timestamps",
            )
        },
        **_controller_success_evidence(context),
    }


def _episode_metadata(
    context: dict[str, Any],
    identity: str,
    scene_id: str,
    attempt: int,
    mounted: dict[str, Any],
) -> dict[str, Any]:
    buffers = context["buffers"]
    scene_digest = _sha(
        {
            "scene_id": scene_id,
            "target": context["target"].tolist(),
            "attempt": attempt,
            "initial_z": context["initial_z"],
            "scene_seed": context["scene_seed"],
        }
    )
    return {
        "schema": EPISODE_SCHEMA,
        "episode_id": f"episode-{attempt:06d}",
        "source_backend": "isaac",
        "object_motion": "physics_only",
        "direct_object_pose_writes": 0,
        "action_semantics": "absolute_joint_position_plus_normalized_gripper",
        "joint_order": list(JOINT_NAMES),
        "control_hz": CONTROL_HZ,
        "step_count": len(buffers["actions"]),
        "object_identity": identity,
        "object_width_m": context["object_width_m"],
        "initial_object_position_m": context["initial_object_position"].tolist(),
        "target_position_m": context["target"].tolist(),
        "scene_configuration_digest": scene_digest,
        "scene_seed": context["scene_seed"],
        "instruction": build_contract()["task"]["instruction"],
        **_episode_evidence(context, mounted),
        "steps": _step_records(buffers),
    }


def _attempt_context(
    native: Any,
    env: Any,
    buffers: dict[str, list[Any]],
    object_position: Any,
    target: Any,
    object_width_m: float,
) -> dict[str, Any]:
    total_dim, arm_start, grip_start = _action_layout(native)
    robot = native.scene["robot"]
    return {
        "native": native,
        "env": env,
        "buffers": buffers,
        "target": target,
        "initial_object_position": object_position.copy(),
        "initial_z": float(object_position[2]),
        "object_width_m": object_width_m,
        "total_dim": total_dim,
        "arm_start": arm_start,
        "grip_start": grip_start,
        "joint_indices": [list(robot.joint_names).index(name) for name in JOINT_NAMES],
        "stable_steps": 0,
    }


def _capture_attempt(
    native: Any,
    env: Any,
    output: Path,
    attempt: int,
    *,
    object_identity: str,
    scene_id: str,
    buffers: dict[str, list[Any]],
    object_width_m: float,
    scene_seed: int,
) -> dict[str, Any]:
    import numpy as np

    object_position = (
        native.scene["object"].data.root_pos_w[0, :3].detach().cpu().numpy()
    )
    target = np.asarray([0.50, 0.18, float(object_position[2])], dtype=np.float32)
    context = _attempt_context(
        native, env, buffers, object_position, target, object_width_m
    )
    context["scene_seed"] = scene_seed
    mounted = _wrist_mount_evidence(native)
    for phase, goal, intent in _expert_waypoints(object_position, target):
        _run_phase(context, phase, goal, intent)
    episode = output / f"episode-{attempt:06d}"
    episode.mkdir(parents=True)
    _write_episode_arrays(episode, buffers)
    metadata = _episode_metadata(context, object_identity, scene_id, attempt, mounted)
    (episode / "episode.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    return metadata


def _new_attempt_buffers() -> dict[str, list[Any]]:
    return {
        name: []
        for name in (
            "frames_exterior",
            "frames_wrist",
            "states",
            "actions",
            "timestamps",
            "controller_errors",
            "events",
            "traces",
        )
    }


def _save_partial_attempt(
    root: Path, attempt: int, buffers: dict[str, list[Any]], error: Exception
) -> dict[str, Any]:
    import numpy as np

    episode = root / f"episode-{attempt:06d}-failed"
    episode.mkdir(parents=True, exist_ok=True)
    array_names = {
        "frames_exterior": "obs_workspace.npy",
        "frames_wrist": "obs_wrist.npy",
        "states": "state.npy",
        "actions": "actions.npy",
        "timestamps": "timestamps.npy",
    }
    for source, filename in array_names.items():
        if buffers[source]:
            np.save(episode / filename, np.asarray(buffers[source]))
    failure = {
        "schema": "npa.sim2real.pi05.failed_attempt.v1",
        "episode_id": f"episode-{attempt:06d}",
        "failure": f"{type(error).__name__}: {error}",
        "captured_steps": len(buffers["actions"]),
        "partial_frames_preserved": bool(buffers["frames_exterior"]),
        "training_admitted": False,
        "trace": buffers["traces"],
    }
    (episode / "failure.json").write_text(
        json.dumps(failure, indent=2, sort_keys=True) + "\n"
    )
    return failure


def _make_environment(
    seed: int, object_usd: str, object_scale: tuple[float, float, float]
) -> tuple[Any, Any]:
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, enable_cameras=True).app
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    from npa.workflows.sim2real.isaac_assets_compat import remap_moved_franka_usd

    cfg = parse_env_cfg(EXPERT_BASE_TASK_ID, device="cuda:0", num_envs=1)
    cfg.seed, cfg.sim.dt, cfg.decimation = seed, 1.0 / 60.0, 4
    cfg.episode_length_s = 240.0
    remap_moved_franka_usd(cfg)
    if object_usd:
        cfg.scene.object.spawn.usd_path = object_usd
    cfg.scene.object.spawn.scale = object_scale
    cfg.scene.object.spawn.semantic_tags = [("class", "pi05_target_object")]
    configure_surface_goal(cfg, (0.50, 0.18, 0.025 * object_scale[2]))
    _camera_config(cfg)
    _contact_config(cfg)
    _physics_randomization(cfg)
    register_surface_task(
        gym, base_id=EXPERT_BASE_TASK_ID, task_id=EXPERT_SURFACE_TASK_ID
    )
    env = gym.make(EXPERT_SURFACE_TASK_ID, cfg=cfg)
    env.reset(seed=seed)
    return app, env


def _collect_attempts(
    env: Any,
    work: Path,
    episodes: int,
    seed: int,
    identity: str,
    scene_id: str,
    scale: tuple[float, float, float],
) -> list[dict[str, Any]]:
    attempts = []
    for attempt in range(episodes):
        if attempt:
            env.reset(seed=seed + attempt)
        buffers = _new_attempt_buffers()
        try:
            result = _capture_attempt(
                env.unwrapped,
                env,
                work,
                attempt,
                object_identity=identity,
                scene_id=scene_id,
                buffers=buffers,
                object_width_m=0.05 * scale[1],
                scene_seed=seed + attempt,
            )
        except Exception as exc:
            failure = _save_partial_attempt(work, attempt, buffers, exc)
            result = {
                **failure,
                "success_evidence": {"released_supported_surface_placement": False},
            }
        attempts.append(result)
    return attempts


def _collection_report(
    attempts: list[dict[str, Any]],
    identity: str,
    object_usd: str,
    scale: tuple[float, float, float],
    scene_id: str,
) -> dict[str, Any]:
    successes = sum(
        bool(
            row.get("success_evidence", {}).get("released_supported_surface_placement")
        )
        for row in attempts
    )
    scenarios = [
        {
            "scene_seed": row["scene_seed"],
            "scene_configuration_digest": row["scene_configuration_digest"],
            "initial_object_position_m": row["initial_object_position_m"],
            "target_position_m": row["target_position_m"],
        }
        for row in attempts
        if row.get("success_evidence", {}).get("released_supported_surface_placement")
    ]
    return {
        "schema": "npa.sim2real.pi05.expert_collection.v1",
        "task_contract": build_contract(),
        "attempt_count": len(attempts),
        "successful_episode_count": successes,
        "attempts": attempts,
        "training_admission": "successful episodes only; failures retained",
        "object_asset": {
            "identity": identity,
            "usd": object_usd or "task_default",
            "scale": list(scale),
        },
        "scene_id": scene_id,
        "successful_scenarios": scenarios,
    }


def collect(
    *,
    output_uri: str,
    episodes: int,
    seed: int,
    object_identity: str,
    object_usd: str,
    object_scale: tuple[float, float, float],
    scene_id: str,
) -> dict[str, Any]:
    """Collect physics attempts. Args: collection settings. Returns: Report. Raises: RuntimeError."""

    from npa.clients.storage import StorageClient

    work = Path(tempfile.mkdtemp(prefix="npa-pi05-isaac-"))
    app, env = _make_environment(seed, object_usd, object_scale)
    try:
        attempts = _collect_attempts(
            env, work, episodes, seed, object_identity, scene_id, object_scale
        )
        report = _collection_report(
            attempts, object_identity, object_usd, object_scale, scene_id
        )
        path = work / "collection.json"
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        StorageClient.from_environment().upload_directory(str(work), output_uri)
        if report["successful_episode_count"] < 1:
            raise RuntimeError(
                "expert produced no physics-verified released placements"
            )
        return report
    finally:
        env.close()
        app.close()


def main(argv: list[str] | None = None) -> int:
    """Run the collector CLI. Args: argv. Returns: Exit status. Raises: RuntimeError."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-uri", required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--object-identity", required=True)
    parser.add_argument("--object-usd", default="")
    parser.add_argument("--object-scale", default="1,1,1")
    parser.add_argument("--scene-id", required=True)
    args = parser.parse_args(argv)
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    try:
        scale = tuple(float(value) for value in args.object_scale.split(","))
    except ValueError as exc:
        parser.error(f"--object-scale must be three comma-separated numbers: {exc}")
    if len(scale) != 3 or any(value <= 0 for value in scale):
        parser.error("--object-scale must contain three positive values")
    report = collect(
        output_uri=args.output_uri,
        episodes=args.episodes,
        seed=args.seed,
        object_identity=args.object_identity,
        object_usd=args.object_usd,
        object_scale=scale,
        scene_id=args.scene_id,
    )
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
