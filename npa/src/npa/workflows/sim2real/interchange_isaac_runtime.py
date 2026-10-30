"""Run a sealed typed SmolVLA policy on a non-Franka Isaac surface task."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import time
import traceback
from typing import Any
from urllib.parse import urlparse
import zlib

from npa.workflows.lerobot_transfer_data import write_json
from npa.workflows.sim2real.interchange_contract import (
    policy_profile_from_dict,
    rigid_object_profile_from_dict,
    robot_sensor_profile_from_dict,
    surface_placement_task_from_dict,
    validate_observation,
)
from npa.workflows.sim2real.interchange_isaac import (
    action_layout_from_live_isaac,
    apply_absolute_joint_target,
    assert_live_robot_matches_profile,
    live_camera_world_pose,
    measured_policy_state,
    surface_teacher_waypoints,
)
from npa.workflows.sim2real.isaac_kit_state import validated_kit_args
from npa.workflows.sim2real.smolvla_policy_client import SmolVLAPolicyClient


PROMPT = "Pick up the object and place it inside the cyan circular target on the table."
CONTACT_FORCE_N = 1.0e-3
TARGET_DISTANCE_M = 0.05
TRANSPORT_DISTANCE_M = 0.08
STABLE_SPEED_M_S = 0.03
STABLE_STEPS = 3


@dataclass
class TeacherTrace:
    """Mutable evidence streams for one native teacher rollout."""

    controls: list[dict[str, Any]] = field(default_factory=list)
    physics: list[dict[str, Any]] = field(default_factory=list)
    camera_hashes: list[dict[str, str]] = field(default_factory=list)
    camera_clocks: list[dict[str, Any]] = field(default_factory=list)
    states: list[list[float]] = field(default_factory=list)
    actions: list[list[float]] = field(default_factory=list)
    frames: list[dict[str, tuple[bytes, int, int, str]]] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)
    visuals: list[dict[str, Any]] = field(default_factory=list)
    step: int = 0


@dataclass(frozen=True)
class RuntimeInputs:
    """Identity-bound inputs consumed by one native policy rollout."""

    receipt: dict[str, Any]
    robot_manifest: dict[str, Any]
    robot_spec: dict[str, Any]
    robot: Any
    prop: Any
    task: Any
    policy: Any
    loaded: dict[str, Any]
    object_usd_path: Path


@dataclass
class PolicyTrace:
    """Mutable evidence streams for one native policy rollout."""

    controls: list[dict[str, Any]] = field(default_factory=list)
    physics: list[dict[str, Any]] = field(default_factory=list)
    inference: list[dict[str, Any]] = field(default_factory=list)
    camera_hashes: list[dict[str, str]] = field(default_factory=list)
    observation_clocks: list[dict[str, Any]] = field(default_factory=list)
    action_chunks: list[dict[str, Any]] = field(default_factory=list)
    visuals: list[dict[str, Any]] = field(default_factory=list)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain one JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _local_object_asset(root: Path, prop: Any) -> Path:
    """Resolve one exact sealed object byte stream without trusting its filename."""

    name = Path(urlparse(prop.source_uri).path).name
    candidates = [path for path in root.rglob(name) if path.is_file()]
    matches = [path for path in candidates if _sha256(path) == prop.source_sha256]
    if len(matches) != 1:
        raise RuntimeError(
            "prepared simulation must contain exactly one object asset matching "
            "the sealed source URI and SHA-256"
        )
    return matches[0]


def _camera_frame(native: Any, key: str, profile: Any) -> tuple[Any, float, dict]:
    import numpy as np
    import torch

    sensor = native.scene[f"npa_policy_camera_{key}"]
    pose = live_camera_world_pose(native, profile, key)
    sensor.set_world_poses(
        positions=torch.tensor(
            [pose.translation_m], dtype=torch.float32, device=native.device
        ),
        orientations=torch.tensor(
            [pose.quaternion_xyzw], dtype=torch.float32, device=native.device
        ),
        convention="ros",
    )
    native.sim.render()
    pixels = sensor.data.output["rgb"]
    if hasattr(pixels, "torch"):
        pixels = pixels.torch
    frame = np.ascontiguousarray(pixels[0, ..., :3].detach().cpu().numpy())
    camera = next(item for item in profile.cameras if item.key == key)
    width, height = camera.image_size
    if frame.shape != (height, width, 3) or frame.dtype != np.uint8:
        raise RuntimeError(f"camera {key!r} returned invalid RGB bytes")
    if int(np.ptp(frame.astype(np.int16))) < 8:
        raise RuntimeError(f"camera {key!r} returned a blank frame")
    return frame, _sim_time(native), pose.to_dict()


def _camera_payloads(
    native: Any, robot: Any, state_time: float
) -> tuple[
    dict[str, Any],
    dict[str, str],
    dict[str, tuple[bytes, int, int, str]],
    dict[str, float],
    dict[str, dict],
]:
    """Capture all declared views with hashes, roles, poses, and clocks."""

    cameras: dict[str, Any] = {}
    hashes: dict[str, str] = {}
    frames: dict[str, tuple[bytes, int, int, str]] = {}
    capture_timestamps: dict[str, float] = {}
    camera_world_poses: dict[str, dict] = {}
    for camera in robot.cameras:
        frame, capture_time, world_pose = _camera_frame(native, camera.key, robot)
        payload = frame.tobytes()
        digest = hashlib.sha256(payload).hexdigest()
        hashes[camera.key] = digest
        capture_timestamps[camera.key] = capture_time
        camera_world_poses[camera.key] = world_pose
        frames[camera.key] = (
            payload,
            camera.image_size[0],
            camera.image_size[1],
            camera.role,
        )
        cameras[camera.key] = {
            "layout": camera.layout,
            "image_size_wh": list(camera.image_size),
            "mask": True,
            "capture_timestamp_s": capture_time,
            "state_timestamp_s": state_time,
            "rgb8": payload,
            "image_sha256": digest,
        }
    return cameras, hashes, frames, capture_timestamps, camera_world_poses


def _observation(
    native: Any, robot: Any, policy: Any, decision: int
) -> tuple[Any, dict, dict[str, tuple[bytes, int, int, str]], dict[str, Any]]:
    """Build one causal typed observation and its native camera evidence."""

    state = measured_policy_state(native, robot)
    state_time = _sim_time(native)
    cameras, hashes, frames, capture_timestamps, camera_world_poses = _camera_payloads(
        native, robot, state_time
    )
    observation = validate_observation(
        robot=robot,
        policy=policy,
        camera_payloads=cameras,
        measured_state=state,
        prompt=PROMPT,
        input_identity=f"non-franka-surface-decision-{decision}",
        task_identity=policy.task_identity,
    )
    return (
        observation,
        hashes,
        frames,
        {
            "decision": decision,
            "state_timestamp_s": state_time,
            "capture_timestamps_s": capture_timestamps,
            "camera_world_poses": camera_world_poses,
        },
    )


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def _write_png_rgb8(path: Path, payload: bytes, width: int, height: int) -> dict:
    """Write exact HWC RGB8 bytes as PNG without a runtime imaging dependency."""

    if width < 1 or height < 1 or len(payload) != width * height * 3:
        raise RuntimeError("native visual evidence has invalid RGB dimensions")
    stride = width * 3
    rows = b"".join(
        b"\x00" + payload[offset : offset + stride]
        for offset in range(0, len(payload), stride)
    )
    encoded = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(
            "IHDR".encode(), struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
        )
        + _png_chunk("IDAT".encode(), zlib.compress(rows, level=6))
        + _png_chunk("IEND".encode(), b"")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return {
        "path": path.as_posix(),
        "bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "raw_rgb_sha256": hashlib.sha256(payload).hexdigest(),
        "width": width,
        "height": height,
    }


def _visual_evidence_decisions(decisions: int) -> frozenset[int]:
    if decisions < 1:
        raise ValueError("decisions must be positive")
    return frozenset((0, decisions // 2, decisions - 1))


def _force(native: Any, name: str) -> float:
    import torch

    matrix = native.scene[name].data.force_matrix_w
    if hasattr(matrix, "torch"):
        matrix = matrix.torch
    return float(torch.linalg.vector_norm(matrix[0], dim=-1).max().detach().cpu())


def _reference_position(native: Any, prop: Any) -> tuple[float, float, float]:
    obj = native.scene["object"].data
    position = obj.root_pos_w
    quaternion = obj.root_quat_w
    if hasattr(position, "torch"):
        position = position.torch
    if hasattr(quaternion, "torch"):
        quaternion = quaternion.torch
    from npa.workflows.sim2real.interchange_isaac import camera_world_pose

    pose = camera_world_pose(
        link_position_m=position[0, :3].detach().cpu().tolist(),
        link_quaternion_xyzw=quaternion[0].detach().cpu().tolist(),
        mount_from_camera=prop.root_from_reference,
    )
    return pose.translation_m


def _sample(native: Any, robot: Any, prop: Any, task: Any, initial_z: float) -> dict:
    import torch

    reference = _reference_position(native, prop)
    velocity = native.scene["object"].data.root_lin_vel_w
    if hasattr(velocity, "torch"):
        velocity = velocity.torch
    speed = float(torch.linalg.vector_norm(velocity[0]).detach().cpu())
    state = measured_policy_state(native, robot)
    closed = robot.gripper.model_value_is_closed(state[robot.measured_state_names[-1]])
    left = _force(native, "npa_left_object_contact")
    right = _force(native, "npa_right_object_contact")
    support = _force(native, "npa_object_support_contact")
    bilateral = left > CONTACT_FORCE_N and right > CONTACT_FORCE_N
    target_xy = task.target_world_xyz_m[:2]
    distance = math.dist(reference[:2], target_xy)
    ee_index = native.scene["robot"].body_names.index(robot.ee_link)
    ee = native.scene["robot"].data.body_link_pos_w
    if hasattr(ee, "torch"):
        ee = ee.torch
    retreat = math.dist(ee[0, ee_index, :3].detach().cpu().tolist(), reference)
    return {
        "simulator_time_s": _sim_time(native),
        "object_reference_world_m": list(reference),
        "object_speed_m_s": speed,
        "left_finger_object_force_n": left,
        "right_finger_object_force_n": right,
        "object_support_force_n": support,
        "bilateral_finger_contact": bilateral,
        "gripper_closed": closed,
        "lifted": reference[2] - initial_z >= task.lift_delta_m,
        "target_xy_distance_m": distance,
        "hand_object_distance_m": retreat,
        "support_contact": support > CONTACT_FORCE_N,
    }


def _update_events(context: dict[str, Any], sample: dict[str, Any], step: int) -> None:
    """Advance ordered physical success events and stability state."""

    events = context["ordered_event_steps"]
    for name, condition in _event_conditions(context, sample):
        predecessors = list(
            context["required_events"][: context["required_events"].index(name)]
        )
        if (
            condition
            and name not in events
            and all(item in events for item in predecessors)
        ):
            events[name] = step
    stable = (
        "support_contact" in events
        and sample["target_xy_distance_m"] < TARGET_DISTANCE_M
        and sample["object_speed_m_s"] < STABLE_SPEED_M_S
    )
    context["stable_steps"] = context["stable_steps"] + 1 if stable else 0
    if context["stable_steps"] >= STABLE_STEPS and "stable" not in events:
        events["stable"] = step


def _event_conditions(
    context: dict[str, Any], sample: dict[str, Any]
) -> tuple[tuple[str, bool], ...]:
    """Evaluate physical event predicates without mutating chronology."""

    events = context["ordered_event_steps"]
    return (
        ("contact", sample["bilateral_finger_contact"]),
        (
            "grasp",
            sample["bilateral_finger_contact"]
            and sample["gripper_closed"]
            and (sample["lifted"] or sample["object_speed_m_s"] > 0.005),
        ),
        ("lift", sample["lifted"]),
        (
            "transport",
            sample["lifted"] and sample["target_xy_distance_m"] < TRANSPORT_DISTANCE_M,
        ),
        (
            "release",
            "transport" in events
            and not sample["gripper_closed"]
            and sample["hand_object_distance_m"] >= context["retreat_distance_m"],
        ),
        (
            "support_contact",
            "release" in events and sample["object_support_force_n"] > CONTACT_FORCE_N,
        ),
    )


def _native_tensor(value: Any) -> Any:
    if hasattr(value, "torch"):
        return value.torch
    return value


def _teacher_layout(native: Any, robot: Any) -> dict[str, Any]:
    """Bind the native differential-IK action to the sealed morphology."""

    names, dimensions, arm, gripper, arm_index, gripper_index = _teacher_terms(native)
    arm_names = tuple(
        getattr(getattr(arm, "IO_descriptor", None), "joint_names", ())
        or getattr(arm, "_joint_names", ())
    )
    gripper_names = tuple(
        getattr(getattr(gripper, "IO_descriptor", None), "joint_names", ())
        or getattr(gripper, "_joint_names", ())
    )
    _require_teacher_action_order(arm_names, gripper_names, robot)
    if str(arm.cfg.body_name) != robot.ee_link:
        raise RuntimeError("teacher IK end-effector differs from the sealed robot")
    values = _teacher_scale(arm)
    return {
        "total_dimension": sum(dimensions),
        "arm_start": sum(dimensions[:arm_index]),
        "gripper_start": sum(dimensions[:gripper_index]),
        "scale": values,
        "arm_joint_names": list(robot.arm_joint_names),
        "gripper_joint_names": list(robot.gripper.joint_names),
        "ee_link": robot.ee_link,
    }


def _require_teacher_action_order(
    arm_names: tuple[str, ...], gripper_names: tuple[str, ...], robot: Any
) -> None:
    """Reject either native controller order mismatch explicitly."""

    if arm_names != robot.arm_joint_names:
        raise RuntimeError("teacher IK joint order differs from the sealed robot")
    if gripper_names != robot.gripper.joint_names:
        raise RuntimeError("teacher gripper order differs from the sealed robot")


def _teacher_terms(native: Any) -> tuple[list[str], list[int], Any, Any, int, int]:
    """Resolve the named six-axis IK and binary gripper terms."""

    names = list(native.action_manager.active_terms)
    dimensions = [int(value) for value in native.action_manager.action_term_dim]
    if names.count("arm_action") != 1 or names.count("gripper_action") != 1:
        raise RuntimeError("teacher requires one named arm and gripper action")
    arm_index = names.index("arm_action")
    gripper_index = names.index("gripper_action")
    if dimensions[arm_index] != 6 or dimensions[gripper_index] != 1:
        raise RuntimeError("teacher requires six-axis relative IK plus gripper")
    return (
        names,
        dimensions,
        native.action_manager.get_term("arm_action"),
        native.action_manager.get_term("gripper_action"),
        arm_index,
        gripper_index,
    )


def _teacher_scale(arm: Any) -> Any:
    """Validate and return the native six-axis IK scale."""

    scale = _native_tensor(getattr(arm, "_scale", None))
    if scale is None or not hasattr(scale, "detach"):
        raise RuntimeError("teacher IK action does not expose its native scale")
    values = scale[0].detach().cpu().numpy().astype(float)
    if values.shape != (6,) or not all(
        math.isfinite(item) and item > 0 for item in values
    ):
        raise RuntimeError("teacher IK native scale is invalid")
    return values


def _axis_angle_error(current: Any, target: tuple[float, ...]) -> Any:
    import numpy as np

    current = np.asarray(current, dtype=float)
    target_array = np.asarray(target, dtype=float)
    inverse = np.asarray([-current[0], -current[1], -current[2], current[3]])
    x1, y1, z1, w1 = target_array
    x2, y2, z2, w2 = inverse
    relative = np.asarray(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ]
    )
    if relative[3] < 0.0:
        relative *= -1.0
    vector_norm = float(np.linalg.norm(relative[:3]))
    if vector_norm < 1.0e-12:
        return np.zeros(3, dtype=float)
    angle = 2.0 * math.atan2(vector_norm, max(0.0, float(relative[3])))
    return relative[:3] * (angle / vector_norm)


def _teacher_command(
    native: Any, layout: dict[str, Any], waypoint: Any
) -> tuple[Any, float, float]:
    import numpy as np
    import torch

    frame = native.scene["ee_frame"].data
    position = _native_tensor(frame.target_pos_w)[0, 0, :3].detach().cpu().numpy()
    quaternion = _native_tensor(frame.target_quat_w)[0, 0, :4].detach().cpu().numpy()
    target = waypoint.world_from_ee
    translation = np.asarray(target.translation_m, dtype=float) - position
    rotation = _axis_angle_error(quaternion, target.quaternion_xyzw)
    processed = np.r_[
        np.clip(translation, -0.07, 0.07),
        np.clip(rotation, -0.25, 0.25),
    ]
    raw = np.clip(processed / layout["scale"], -1.0, 1.0)
    command = torch.zeros(
        (1, layout["total_dimension"]), dtype=torch.float32, device=native.device
    )
    start = layout["arm_start"]
    command[0, start : start + 6] = torch.as_tensor(raw, device=native.device)
    command[0, layout["gripper_start"]] = -1.0 if waypoint.gripper_closed else 1.0
    return command, float(np.linalg.norm(translation)), float(np.linalg.norm(rotation))


def _teacher_camera_capture(
    native: Any, robot: Any
) -> tuple[
    dict[str, str],
    dict[str, tuple[bytes, int, int, str]],
    dict[str, float],
    dict[str, dict],
]:
    hashes: dict[str, str] = {}
    frames: dict[str, tuple[bytes, int, int, str]] = {}
    timestamps: dict[str, float] = {}
    poses: dict[str, dict] = {}
    for camera in robot.cameras:
        frame, capture_time, world_pose = _camera_frame(native, camera.key, robot)
        payload = frame.tobytes()
        hashes[camera.key] = hashlib.sha256(payload).hexdigest()
        timestamps[camera.key] = capture_time
        poses[camera.key] = world_pose
        frames[camera.key] = (
            payload,
            camera.image_size[0],
            camera.image_size[1],
            camera.role,
        )
    return hashes, frames, timestamps, poses


def _teacher_views(
    camera_frames: list[dict[str, tuple[bytes, int, int, str]]],
) -> dict[str, list[Any]]:
    """Decode exactly one exterior and one wrist view per teacher step."""

    import numpy as np

    by_role: dict[str, list[Any]] = {"exterior": [], "wrist": []}
    for frame_set in camera_frames:
        roles: dict[str, Any] = {}
        for payload, width, height, role in frame_set.values():
            if role in roles:
                raise RuntimeError(f"teacher episode has multiple {role!r} cameras")
            roles[role] = (
                np.frombuffer(payload, dtype=np.uint8).reshape(height, width, 3).copy()
            )
        if set(roles) != set(by_role):
            raise RuntimeError(
                "teacher episode requires one exterior and one wrist view"
            )
        for role, pixels in roles.items():
            by_role[role].append(pixels)
    return by_role


def _write_teacher_arrays(
    directory: Path,
    views: dict[str, list[Any]],
    states: list[list[float]],
    actions: list[list[float]],
    timestamps: list[float],
) -> None:
    """Write native teacher arrays without replacing existing evidence."""

    import numpy as np

    names = (
        "obs_workspace.npy",
        "obs_wrist.npy",
        "state.npy",
        "actions.npy",
        "timestamps.npy",
    )
    if any((directory / name).exists() for name in names):
        raise RuntimeError("teacher episode output already exists")
    np.save(directory / "obs_workspace.npy", np.stack(views["exterior"]))
    np.save(directory / "obs_wrist.npy", np.stack(views["wrist"]))
    np.save(directory / "state.npy", np.asarray(states, dtype=np.float32))
    np.save(directory / "actions.npy", np.asarray(actions, dtype=np.float32))
    np.save(directory / "timestamps.npy", np.asarray(timestamps, dtype=np.float64))


def _teacher_metadata(
    robot: Any, prop: Any, task: Any, count: int, teacher: dict[str, Any]
) -> dict[str, Any]:
    """Build model-neutral teacher episode metadata."""

    return {
        "schema": "npa.sim2real.policy_interchange.teacher_episode.v1",
        "robot_identity": robot.identity,
        "object_identity": prop.identity,
        "task_identity": task.identity,
        "task": PROMPT,
        "frames": count,
        "observation_state_names": list(robot.measured_state_names),
        "observation_state_units": list(robot.measured_state_units),
        "action_representation": "absolute_joint_position",
        "action_names": list(robot.measured_state_names),
        "action_units": list(robot.measured_state_units),
        "gripper_semantics": robot.gripper.model_semantics,
        "privileged_object_state_policy_input": False,
        "ordered_event_steps": teacher["ordered_event_steps"],
    }


def _write_teacher_episode(
    *,
    directory: Path,
    robot: Any,
    task: Any,
    prop: Any,
    states: list[list[float]],
    actions: list[list[float]],
    camera_frames: list[dict[str, tuple[bytes, int, int, str]]],
    timestamps: list[float],
    teacher: dict[str, Any],
) -> dict[str, Any]:
    """Materialize one native teacher rollout in the generic adapter layout."""
    count = _validate_teacher_episode(
        robot, teacher, states, actions, camera_frames, timestamps
    )
    directory.mkdir(parents=True, exist_ok=True)
    outputs = (
        "obs_workspace.npy",
        "obs_wrist.npy",
        "state.npy",
        "actions.npy",
        "timestamps.npy",
        "metadata.json",
        "native_teacher.json",
    )
    if any((directory / name).exists() for name in outputs):
        raise RuntimeError("teacher episode output already exists")
    _write_teacher_arrays(
        directory, _teacher_views(camera_frames), states, actions, timestamps
    )
    metadata = _teacher_metadata(robot, prop, task, count, teacher)
    write_json(directory / "metadata.json", metadata)
    write_json(directory / "native_teacher.json", teacher)
    return metadata


def _validate_teacher_episode(
    robot: Any,
    teacher: dict[str, Any],
    states: list[list[float]],
    actions: list[list[float]],
    camera_frames: list[dict[str, Any]],
    timestamps: list[float],
) -> int:
    """Validate synchronized successful teacher streams before materialization."""

    if not teacher.get("task_success"):
        raise RuntimeError("only successful native teacher episodes may enter training")
    count = len(states)
    if not count or not (
        len(actions) == len(camera_frames) == len(timestamps) == count
    ):
        raise RuntimeError("teacher dataset streams have inconsistent lengths")
    expected_dimension = len(robot.measured_state_names)
    if any(len(row) != expected_dimension for row in [*states, *actions]):
        raise RuntimeError("teacher state/action width differs from the sealed robot")
    return count


def _teacher_environment(
    gym: Any,
    args: argparse.Namespace,
    robot: Any,
    robot_spec: dict[str, Any],
    prop: Any,
    task: Any,
    object_usd_path: Path,
    control_hz: float,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Configure the embodiment-aware relative-IK teacher environment."""

    from npa.workflows.franka_rl_environment import (
        POLICY_INTERCHANGE_SCAFFOLD_TASK,
        configure_policy_interchange_cameras,
        configure_policy_interchange_robot,
        configure_policy_interchange_surface_task,
        policy_interchange_environment_config,
        policy_interchange_timing,
    )

    timing = policy_interchange_timing(control_hz=control_hz)
    config = policy_interchange_environment_config(
        seed=args.seed, control_hz=control_hz
    )
    configuration = configure_policy_interchange_robot(
        config,
        robot,
        robot_spec,
        args.robot_usd_path,
        controller_representation="relative_ee_pose",
    )
    configure_policy_interchange_surface_task(
        config,
        robot,
        prop,
        task,
        object_usd_path=object_usd_path,
        robot_rigid_body_prim_paths=configuration["rigid_body_prim_paths"],
    )
    configure_policy_interchange_cameras(config, robot)
    return gym.make(POLICY_INTERCHANGE_SCAFFOLD_TASK, cfg=config), timing, configuration


def _teacher_visuals(
    args: argparse.Namespace,
    trace: TeacherTrace,
    frames: dict[str, tuple[bytes, int, int, str]],
    phase: str,
) -> None:
    """Write one factual camera pair at the start of a teacher phase."""

    for key, (payload, width, height, role) in frames.items():
        relative = Path("visuals") / f"teacher-{phase}-{role}.png"
        trace.visuals.append(
            {
                "step": trace.step,
                "phase": phase,
                "camera_key": key,
                "camera_role": role,
                **_write_png_rgb8(args.output_path / relative, payload, width, height),
                "path": relative.as_posix(),
            }
        )


def _teacher_step(
    *,
    env: Any,
    native: Any,
    args: argparse.Namespace,
    robot: Any,
    prop: Any,
    task: Any,
    layout: dict[str, Any],
    waypoint: Any,
    initial_z: float,
    context: dict[str, Any],
    trace: TeacherTrace,
    first: bool,
) -> tuple[float, float]:
    """Capture, apply, measure, and record one causal native teacher control."""

    hashes, frames, captures, poses, measured_before = _teacher_step_inputs(
        native, robot, args, trace, waypoint, first
    )
    command, errors, times, sample, measured_after = _apply_teacher_step(
        env, native, robot, prop, task, layout, waypoint, initial_z, captures
    )
    _update_events(context, sample, trace.step)
    _record_teacher_transition(
        trace,
        waypoint,
        command,
        times,
        errors,
        sample,
        hashes,
        captures,
        poses,
        measured_before,
        measured_after,
        robot,
        frames,
    )
    return errors


def _record_teacher_transition(
    trace: TeacherTrace,
    waypoint: Any,
    command: Any,
    times: tuple[float, float],
    errors: tuple[float, float],
    sample: dict[str, Any],
    hashes: dict[str, str],
    captures: dict[str, float],
    poses: dict[str, dict],
    measured_before: dict[str, float],
    measured_after: dict[str, float],
    robot: Any,
    frames: dict[str, Any],
) -> None:
    _record_teacher_step(
        trace=trace,
        waypoint=waypoint,
        command=command,
        before=times[0],
        after=times[1],
        position_error=errors[0],
        orientation_error=errors[1],
        sample=sample,
        hashes=hashes,
        captures=captures,
        poses=poses,
        measured_before=measured_before,
        measured_after=measured_after,
        robot=robot,
        frames=frames,
    )


def _teacher_step_inputs(
    native: Any,
    robot: Any,
    args: argparse.Namespace,
    trace: TeacherTrace,
    waypoint: Any,
    first: bool,
) -> tuple[dict, dict, dict, dict, dict]:
    """Capture causal cameras and the pre-control measured robot state."""

    hashes, frames, captures, poses = _teacher_camera_capture(native, robot)
    measured_before = measured_policy_state(native, robot)
    if first:
        _teacher_visuals(args, trace, frames, waypoint.phase)
    return hashes, frames, captures, poses, measured_before


def _apply_teacher_step(
    env: Any,
    native: Any,
    robot: Any,
    prop: Any,
    task: Any,
    layout: dict[str, Any],
    waypoint: Any,
    initial_z: float,
    captures: dict[str, float],
) -> tuple[
    Any, tuple[float, float], tuple[float, float], dict[str, Any], dict[str, float]
]:
    """Apply one teacher command after proving its cameras are causal."""

    command, position_error, orientation_error = _teacher_command(
        native, layout, waypoint
    )
    before = _sim_time(native)
    if any(value > before + 1.0e-9 for value in captures.values()):
        raise RuntimeError("teacher camera capture followed its control")
    _, _, terminated, truncated, _ = env.step(command)
    after = _sim_time(native)
    if bool(terminated[0]) or bool(truncated[0]):
        raise RuntimeError(f"native teacher reset during {waypoint.phase}")
    sample = _sample(native, robot, prop, task, initial_z)
    measured_after = measured_policy_state(native, robot)
    return (
        command,
        (position_error, orientation_error),
        (before, after),
        sample,
        measured_after,
    )


def _record_teacher_step(
    *,
    trace: TeacherTrace,
    waypoint: Any,
    command: Any,
    before: float,
    after: float,
    position_error: float,
    orientation_error: float,
    sample: dict[str, Any],
    hashes: dict[str, str],
    captures: dict[str, float],
    poses: dict[str, dict],
    measured_before: dict[str, float],
    measured_after: dict[str, float],
    robot: Any,
    frames: dict[str, tuple[bytes, int, int, str]],
) -> None:
    """Append one native teacher transition to every evidence stream."""

    trace.controls.append(
        _teacher_control_receipt(
            trace.step,
            waypoint,
            command,
            before,
            after,
            position_error,
            orientation_error,
        )
    )
    trace.physics.append(sample)
    trace.camera_hashes.append(hashes)
    trace.camera_clocks.append(
        _teacher_clock(trace.step, captures, poses, before, after)
    )
    _record_teacher_arrays(
        trace, robot, measured_before, measured_after, frames, before
    )
    trace.step += 1


def _record_teacher_arrays(
    trace: TeacherTrace,
    robot: Any,
    measured_before: dict[str, float],
    measured_after: dict[str, float],
    frames: dict[str, Any],
    before: float,
) -> None:
    trace.states.append([measured_before[name] for name in robot.measured_state_names])
    trace.actions.append([measured_after[name] for name in robot.measured_state_names])
    trace.frames.append(frames)
    trace.timestamps.append(before)


def _teacher_control_receipt(
    step: int,
    waypoint: Any,
    command: Any,
    before: float,
    after: float,
    position_error: float,
    orientation_error: float,
) -> dict[str, Any]:
    return {
        "step": step,
        "phase": waypoint.phase,
        "command_application_time_s": before,
        "resulting_state_time_s": after,
        "native_action": command[0].detach().cpu().tolist(),
        "position_error_m": position_error,
        "orientation_error_rad": orientation_error,
        "gripper_closed": waypoint.gripper_closed,
    }


def _teacher_clock(
    step: int,
    captures: dict[str, float],
    poses: dict[str, dict],
    before: float,
    after: float,
) -> dict[str, Any]:
    return {
        "step": step,
        "capture_timestamps_s": captures,
        "camera_world_poses": poses,
        "command_application_time_s": before,
        "resulting_state_time_s": after,
    }


def _teacher_waypoint(
    *,
    env: Any,
    native: Any,
    args: argparse.Namespace,
    robot: Any,
    prop: Any,
    task: Any,
    layout: dict[str, Any],
    waypoint: Any,
    initial_z: float,
    context: dict[str, Any],
    trace: TeacherTrace,
) -> None:
    """Drive one geometry-derived waypoint to convergence."""

    consecutive = 0
    for phase_step in range(90):
        position, orientation = _teacher_step(
            env=env,
            native=native,
            args=args,
            robot=robot,
            prop=prop,
            task=task,
            layout=layout,
            waypoint=waypoint,
            initial_z=initial_z,
            context=context,
            trace=trace,
            first=phase_step == 0,
        )
        consecutive = consecutive + 1 if position < 0.012 and orientation < 0.08 else 0
        hold = 12 if waypoint.phase in {"grasp", "release"} else 3
        if waypoint.phase == "retreat" and context["stable_steps"] >= 3:
            return
        if consecutive >= hold:
            return
    raise RuntimeError(f"geometry-aware teacher failed to reach {waypoint.phase}")


def _teacher_result(
    *,
    task: Any,
    prop: Any,
    timing: dict[str, Any],
    configuration: dict[str, Any],
    layout: dict[str, Any],
    context: dict[str, Any],
    trace: TeacherTrace,
) -> dict[str, Any]:
    """Build the native teacher execution and camera provenance report."""

    from npa.workflows.franka_rl_environment import POLICY_INTERCHANGE_SCAFFOLD_TASK

    return {
        "schema": "npa.sim2real.policy_interchange.native_teacher.v1",
        "execution_valid": bool(trace.controls),
        "task_success": tuple(context["ordered_event_steps"]) == task.success_events,
        "ordered_event_steps": context["ordered_event_steps"],
        "native_scaffold_task": POLICY_INTERCHANGE_SCAFFOLD_TASK,
        "timing": timing,
        "controller": configuration,
        "layout": _serializable_layout(layout),
        "waypoints": _teacher_waypoint_receipts(prop, task),
        "controls": trace.controls,
        "physics": trace.physics,
        "camera_hashes": trace.camera_hashes,
        "camera_clocks": trace.camera_clocks,
        "visual_evidence": trace.visuals,
        "privileged_object_state_policy_input": False,
    }


def _serializable_layout(layout: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.tolist() if hasattr(value, "tolist") else value
        for key, value in layout.items()
    }


def _teacher_waypoint_receipts(prop: Any, task: Any) -> list[dict[str, Any]]:
    return [
        {
            "phase": item.phase,
            "world_from_ee": item.world_from_ee.to_dict(),
            "gripper_closed": item.gripper_closed,
        }
        for item in surface_teacher_waypoints(prop, task)
    ]


def _run_teacher(
    *,
    gym: Any,
    args: argparse.Namespace,
    robot: Any,
    robot_spec: dict[str, Any],
    prop: Any,
    task: Any,
    object_usd_path: Path,
    control_hz: float,
    episode_directory: Path | None = None,
) -> dict[str, Any]:
    """Execute one geometry-aware native teacher before policy evaluation."""

    env, timing, configuration = _teacher_environment(
        gym, args, robot, robot_spec, prop, task, object_usd_path, control_hz
    )
    try:
        env.reset(seed=args.seed)
        native = env.unwrapped
        layout = _teacher_layout(native, robot)
        initial_z = _reference_position(native, prop)[2]
        context = _event_context(task)
        trace = _execute_teacher_waypoints(
            env, native, args, robot, prop, task, layout, initial_z, context
        )
        return _finish_teacher(
            robot=robot,
            prop=prop,
            task=task,
            timing=timing,
            configuration=configuration,
            layout=layout,
            context=context,
            trace=trace,
            episode_directory=episode_directory,
        )
    finally:
        env.close()


def _event_context(task: Any) -> dict[str, Any]:
    return {
        "ordered_event_steps": {},
        "required_events": task.success_events,
        "retreat_distance_m": task.retreat_distance_m,
        "stable_steps": 0,
    }


def _execute_teacher_waypoints(
    env: Any,
    native: Any,
    args: argparse.Namespace,
    robot: Any,
    prop: Any,
    task: Any,
    layout: dict[str, Any],
    initial_z: float,
    context: dict[str, Any],
) -> TeacherTrace:
    """Execute every geometry-derived phase into one evidence trace."""

    trace = TeacherTrace()
    for waypoint in surface_teacher_waypoints(prop, task):
        _teacher_waypoint(
            env=env,
            native=native,
            args=args,
            robot=robot,
            prop=prop,
            task=task,
            layout=layout,
            waypoint=waypoint,
            initial_z=initial_z,
            context=context,
            trace=trace,
        )
    return trace


def _finish_teacher(
    *,
    robot: Any,
    prop: Any,
    task: Any,
    timing: dict[str, Any],
    configuration: dict[str, Any],
    layout: dict[str, Any],
    context: dict[str, Any],
    trace: TeacherTrace,
    episode_directory: Path | None,
) -> dict[str, Any]:
    """Finalize teacher evidence and optionally seal its dataset episode."""

    result = _teacher_result(
        task=task,
        prop=prop,
        timing=timing,
        configuration=configuration,
        layout=layout,
        context=context,
        trace=trace,
    )
    if episode_directory is not None:
        result["dataset_episode"] = _write_teacher_episode(
            directory=episode_directory,
            robot=robot,
            task=task,
            prop=prop,
            states=trace.states,
            actions=trace.actions,
            camera_frames=trace.frames,
            timestamps=trace.timestamps,
            teacher=result,
        )
    return result


def _runtime_inputs(args: argparse.Namespace) -> RuntimeInputs:
    """Load and bind all immutable rollout inputs before Isaac starts."""

    from npa.workflows.sim2real.interchange_contract import assert_policy_compatible

    preparation = _load(args.preparation_path)
    receipt = _load(args.service_receipt_path)
    robot_manifest = _load(args.robot_asset_manifest_path)
    robot_asset = preparation.get("robot_asset") or {}
    robot_spec = robot_asset.get("robot_spec")
    if not isinstance(robot_spec, dict):
        raise RuntimeError("preparation lacks normalized RobotSpec bytes")
    robot = robot_sensor_profile_from_dict(preparation.get("robot") or {})
    _assert_robot_asset_binding(args, robot_manifest, robot)
    prop = rigid_object_profile_from_dict(preparation.get("object") or {})
    task = surface_placement_task_from_dict(preparation.get("task") or {})
    policy = policy_profile_from_dict(preparation.get("policy") or {})
    loaded = receipt.get("interchange") or {}
    policy = _bound_policy(receipt, loaded, robot, task, policy)
    assert_policy_compatible(robot, task, policy)
    return RuntimeInputs(
        receipt=receipt,
        robot_manifest=robot_manifest,
        robot_spec=robot_spec,
        robot=robot,
        prop=prop,
        task=task,
        policy=policy,
        loaded=loaded,
        object_usd_path=_local_object_asset(args.input_path, prop),
    )


def _assert_robot_asset_binding(
    args: argparse.Namespace, manifest: dict[str, Any], robot: Any
) -> None:
    """Reject robot bytes that do not match the resolved profile."""

    if (
        manifest.get("embodiment_digest") != robot.identity
        or manifest.get("source_tree_sha256") != robot.bundle_sha256
        or manifest.get("usd_sha256") != _sha256(args.robot_usd_path)
    ):
        raise RuntimeError("fetched robot asset manifest does not bind this profile")


def _bound_policy(
    receipt: dict[str, Any], loaded: dict[str, Any], robot: Any, task: Any, policy: Any
) -> Any:
    """Bind the declared policy to the exact ready service checkpoint."""

    if (
        receipt.get("status") != "ready"
        or policy.backend != "lerobot_smolvla"
        or loaded.get("policy_identity") != policy.identity
        or loaded.get("robot_identity") != robot.identity
        or loaded.get("task_identity") != task.identity
    ):
        raise RuntimeError("service, robot, task and policy identities are not bound")
    return replace(
        policy,
        checkpoint_sha256=str(loaded.get("checkpoint_sha256") or ""),
        normalization_identity=str(loaded.get("normalization_identity") or ""),
    )


def _policy_environment(
    gym: Any, args: argparse.Namespace, inputs: RuntimeInputs
) -> tuple[Any, dict]:
    """Configure the sealed embodiment for absolute-joint policy control."""

    from npa.workflows.franka_rl_environment import (
        POLICY_INTERCHANGE_SCAFFOLD_TASK,
        configure_policy_interchange_cameras,
        configure_policy_interchange_robot,
        configure_policy_interchange_surface_task,
        policy_interchange_environment_config,
    )

    config = policy_interchange_environment_config(
        seed=args.seed, control_hz=inputs.policy.action.control_hz
    )
    robot_configuration = configure_policy_interchange_robot(
        config, inputs.robot, inputs.robot_spec, args.robot_usd_path
    )
    configure_policy_interchange_surface_task(
        config,
        inputs.robot,
        inputs.prop,
        inputs.task,
        object_usd_path=inputs.object_usd_path,
        robot_rigid_body_prim_paths=robot_configuration["rigid_body_prim_paths"],
    )
    configure_policy_interchange_cameras(config, inputs.robot)
    return gym.make(POLICY_INTERCHANGE_SCAFFOLD_TASK, cfg=config), robot_configuration


def _live_binding(
    native: Any, inputs: RuntimeInputs, configuration: dict
) -> tuple[dict, dict]:
    """Bind the live Isaac articulation to the sealed robot asset."""

    layout = action_layout_from_live_isaac(native, inputs.robot)
    binding = assert_live_robot_matches_profile(
        robot_joint_names=native.scene["robot"].joint_names,
        robot_body_names=native.scene["robot"].body_names,
        layout=layout,
        profile=inputs.robot,
    )
    manifest = inputs.robot_manifest
    embodiment = {
        "schema": "npa.sim2real.exact_robot_asset_binding.v1",
        "robot_identity": inputs.robot.identity,
        "source_sha256": inputs.robot.source_sha256,
        "source_tree_sha256": inputs.robot.bundle_sha256,
        "usd_sha256": manifest["usd_sha256"],
        "usd_size_bytes": manifest["usd_size_bytes"],
        "asset_tree_sha256": manifest["asset_tree_sha256"],
        "converter": manifest["converter"],
        "configuration": configuration,
        "live_joint_names": list(native.scene["robot"].joint_names),
        "live_body_names": list(native.scene["robot"].body_names),
        "exact_resolved_usd_consumed": True,
    }
    return binding, embodiment


def _policy_visuals(
    args: argparse.Namespace,
    trace: PolicyTrace,
    decision: int,
    frames: dict[str, tuple[bytes, int, int, str]],
) -> None:
    """Persist one selected factual policy camera pair."""

    for key, (payload, width, height, role) in frames.items():
        relative = Path("visuals") / f"decision-{decision:06d}-{role}.png"
        trace.visuals.append(
            {
                "decision": decision,
                "camera_key": key,
                "camera_role": role,
                **_write_png_rgb8(args.output_path / relative, payload, width, height),
                "path": relative.as_posix(),
            }
        )


def _policy_step(
    *,
    env: Any,
    native: Any,
    args: argparse.Namespace,
    inputs: RuntimeInputs,
    client: SmolVLAPolicyClient,
    decision: int,
    initial_z: float,
    context: dict[str, Any],
    trace: PolicyTrace,
    evidence_decisions: set[int],
) -> None:
    """Capture, infer, apply, and record one causal policy decision."""

    observation, hashes, frames, clock = _observation(
        native, inputs.robot, inputs.policy, decision
    )
    if decision in evidence_decisions:
        _policy_visuals(args, trace, decision, frames)
    output, inference = client.infer_typed(
        observation=observation, policy=inputs.policy
    )
    action, action_receipt = _select_policy_action(
        output,
        decision=decision,
        strategy=args.action_selection,
        history=context.setdefault("action_chunk_history", []),
    )
    control = apply_absolute_joint_target(env, action, profile=inputs.robot)
    if control["terminated"] or control["truncated"]:
        raise RuntimeError("native policy environment reset during evaluation")
    sample = _sample(native, inputs.robot, inputs.prop, inputs.task, initial_z)
    _update_events(context, sample, decision)
    trace.controls.append(control)
    trace.physics.append(sample)
    trace.inference.append(inference)
    trace.camera_hashes.append(hashes)
    trace.observation_clocks.append(clock)
    trace.action_chunks.append(action_receipt)


def _select_policy_action(
    output: Any,
    *,
    decision: int,
    strategy: str,
    history: list[dict[str, Any]],
) -> tuple[tuple[float, ...], dict[str, Any]]:
    """Select one causal action while retaining the complete native chunk."""

    import numpy as np

    values = np.ascontiguousarray(output.action_chunk, dtype=np.float64)
    if (
        output.action.execute_prefix != 1
        or values.ndim != 2
        or values.shape[0] < 1
        or values.shape[0] > output.action.horizon
        or values.shape[1] != len(output.action.names)
        or not np.isfinite(values).all()
    ):
        raise RuntimeError("native service returned an invalid current action chunk")
    if decision == 0:
        history.clear()
    elif not history or int(history[-1]["origin_decision"]) != decision - 1:
        raise RuntimeError("native action-chunk history is not episode-contiguous")
    chunk_sha256 = hashlib.sha256(values.tobytes()).hexdigest()
    history.append(
        {
            "origin_decision": decision,
            "actions": values.copy(),
            "chunk_sha256": chunk_sha256,
        }
    )
    if strategy == "first_action":
        selected = values[0].copy()
        contributors = [
            {
                "origin_decision": decision,
                "action_index": 0,
                "chunk_sha256": chunk_sha256,
            }
        ]
        history[:] = history[-1:]
        method = "native_fresh_chunk_first_action"
    elif strategy == "uniform_temporal_ensemble":
        retained, candidates, contributors = [], [], []
        for row in history:
            offset = decision - int(row["origin_decision"])
            if 0 <= offset < row["actions"].shape[0]:
                retained.append(row)
                candidates.append(row["actions"][offset])
                contributors.append(
                    {
                        "origin_decision": int(row["origin_decision"]),
                        "action_index": offset,
                        "chunk_sha256": row["chunk_sha256"],
                    }
                )
        history[:] = retained
        if not candidates or contributors[-1]["action_index"] != 0:
            raise RuntimeError("native temporal ensemble lacks the current prediction")
        selected = np.ascontiguousarray(np.mean(np.stack(candidates), axis=0))
        method = "native_fresh_chunk_uniform_temporal_mean"
    else:
        raise RuntimeError(f"unsupported native action-selection strategy {strategy!r}")
    return tuple(float(value) for value in selected), {
        "schema": "npa.sim2real.native_action_chunk.v1",
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "sha256": chunk_sha256,
        "actions": values.tolist(),
        "selection_strategy": strategy,
        "selection_method": method,
        "selected_action_index": 0 if strategy == "first_action" else None,
        "contributors": contributors,
        "contributor_count": len(contributors),
        "weight_per_contributor": 1.0 / len(contributors),
        "selected_action_sha256": hashlib.sha256(selected.tobytes()).hexdigest(),
        "selected_action": selected.tolist(),
    }


def _current_policy_action(output: Any) -> tuple[tuple[float, ...], dict[str, Any]]:
    """Compatibility helper for callers explicitly selecting chunk row zero."""

    return _select_policy_action(
        output, decision=0, strategy="first_action", history=[]
    )


def _policy_rollout(
    env: Any,
    native: Any,
    args: argparse.Namespace,
    inputs: RuntimeInputs,
    client: SmolVLAPolicyClient,
) -> tuple[dict[str, Any], PolicyTrace, set[int]]:
    """Run the fixed decision cohort and retain every causal receipt."""

    context = {
        "ordered_event_steps": {},
        "required_events": inputs.task.success_events,
        "retreat_distance_m": inputs.task.retreat_distance_m,
        "stable_steps": 0,
    }
    trace = PolicyTrace()
    evidence_decisions = _visual_evidence_decisions(args.decisions)
    initial_z = _reference_position(native, inputs.prop)[2]
    for decision in range(args.decisions):
        _policy_step(
            env=env,
            native=native,
            args=args,
            inputs=inputs,
            client=client,
            decision=decision,
            initial_z=initial_z,
            context=context,
            trace=trace,
            evidence_decisions=evidence_decisions,
        )
    return context, trace, evidence_decisions


def _success_contract(task: Any) -> dict[str, Any]:
    return {
        "required_event_order": list(task.success_events),
        "target_xy_distance_m_lt": TARGET_DISTANCE_M,
        "object_speed_m_s_lt": STABLE_SPEED_M_S,
        "stable_steps_gte": STABLE_STEPS,
        "contact_force_n_gt": CONTACT_FORCE_N,
        "lift_delta_m_gte": task.lift_delta_m,
        "retreat_distance_m_gte": task.retreat_distance_m,
    }


def _object_asset(prop: Any) -> dict[str, Any]:
    return {
        "identity": prop.identity,
        "source_sha256": prop.source_sha256,
        "bundle_sha256": prop.bundle_sha256,
        "mass_kg": prop.mass_kg,
        "static_friction": prop.static_friction,
        "dynamic_friction": prop.dynamic_friction,
        "restitution": prop.restitution,
        "exact_local_bytes_consumed": True,
    }


def _service_provenance(inputs: RuntimeInputs) -> dict[str, Any]:
    image = str(inputs.receipt.get("runtime_image") or "")
    return {
        "source_commit": inputs.receipt.get("source_commit"),
        "runtime_image": image,
        "image_digest": image.rsplit("@sha256:", 1)[-1],
        "checkpoint_sha256": inputs.loaded.get("checkpoint_sha256"),
        "checkpoint_manifest_sha256": inputs.loaded.get("manifest_sha256"),
    }


def _execution_report(
    *,
    args: argparse.Namespace,
    inputs: RuntimeInputs,
    teacher: dict[str, Any],
    binding: dict,
    embodiment: dict,
    reset: dict,
    context: dict[str, Any],
    trace: PolicyTrace,
    evidence_decisions: set[int],
) -> dict[str, Any]:
    """Assemble the immutable execution, provenance, and camera report."""

    from npa.workflows.franka_rl_environment import POLICY_INTERCHANGE_SCAFFOLD_TASK

    report = {
        "schema": "npa.sim2real.policy_interchange.non_franka_execution.v1",
        "execution_valid": len(trace.controls) == args.decisions,
        "task_success": tuple(context["ordered_event_steps"])
        == inputs.task.success_events,
        "ordered_event_steps": context["ordered_event_steps"],
        "success_contract": _success_contract(inputs.task),
        "robot_binding": binding,
        "embodiment": embodiment,
        "robot_asset": inputs.robot_manifest,
        "object_asset": _object_asset(inputs.prop),
        **_report_identity_block(inputs, teacher),
        "native_scaffold_task": POLICY_INTERCHANGE_SCAFFOLD_TASK,
        "service_provenance": _service_provenance(inputs),
        "service_reset": reset,
        "teacher": teacher,
        "seed": args.seed,
        "decisions": args.decisions,
        "action_selection_strategy": args.action_selection,
    }
    report.update(_trace_report(args, trace, evidence_decisions))
    return report


def _report_identity_block(
    inputs: RuntimeInputs, teacher: dict[str, Any]
) -> dict[str, Any]:
    return {
        "robot_identity": inputs.robot.identity,
        "object_identity": inputs.prop.identity,
        "task_identity": inputs.task.identity,
        "timing": teacher["timing"],
        "policy_identity": inputs.policy.identity,
        "checkpoint_sha256": inputs.policy.checkpoint_sha256,
        "normalization_identity": inputs.policy.normalization_identity,
    }


def _trace_report(
    args: argparse.Namespace, trace: PolicyTrace, evidence_decisions: set[int]
) -> dict[str, Any]:
    return {
        "controls": trace.controls,
        "physics": trace.physics,
        "inference": trace.inference,
        "action_chunks": trace.action_chunks,
        "camera_hashes": trace.camera_hashes,
        "observation_clocks": trace.observation_clocks,
        "visual_evidence": trace.visuals,
        "visual_evidence_decisions": sorted(evidence_decisions),
        "all_camera_decisions_hashed": len(trace.camera_hashes) == args.decisions,
        "privileged_object_state_policy_input": False,
        "physical_robot_tested": False,
    }


def _runtime_boundary_smoke(
    gym: Any, args: argparse.Namespace, inputs: RuntimeInputs
) -> dict[str, Any]:
    """Exercise one real camera-to-model-to-controller boundary before trials."""

    env, configuration = _policy_environment(gym, args, inputs)
    try:
        env.reset(seed=args.seed)
        native = env.unwrapped
        binding, embodiment = _live_binding(native, inputs, configuration)
        client = _policy_client(inputs)
        reset = client.reset(policy=inputs.policy)
        context = {
            "ordered_event_steps": {},
            "required_events": inputs.task.success_events,
            "retreat_distance_m": inputs.task.retreat_distance_m,
            "stable_steps": 0,
        }
        trace = PolicyTrace()
        _policy_step(
            env=env,
            native=native,
            args=args,
            inputs=inputs,
            client=client,
            decision=0,
            initial_z=_reference_position(native, inputs.prop)[2],
            context=context,
            trace=trace,
            evidence_decisions=set(),
        )
        if not all(
            len(rows) == 1
            for rows in (
                trace.controls,
                trace.physics,
                trace.inference,
                trace.action_chunks,
                trace.camera_hashes,
                trace.observation_clocks,
            )
        ):
            raise RuntimeError(
                "native runtime boundary smoke emitted incomplete evidence"
            )
        receipt = {
            "schema": "npa.sim2real.policy_interchange.runtime_boundary_smoke.v1",
            "status": "passed",
            "training_or_evaluation_row": False,
            "excluded_from_task_metrics": True,
            "action_selection_strategy": args.action_selection,
            "robot_binding": binding,
            "embodiment": embodiment,
            "service_reset": reset,
            "camera_hashes": trace.camera_hashes[0],
            "observation_clock": trace.observation_clocks[0],
            "inference": trace.inference[0],
            "action_chunk": trace.action_chunks[0],
            "controller_proof": trace.controls[0],
            "physics_observation": trace.physics[0],
        }
        write_json(args.output_path / "runtime-boundary-smoke.json", receipt)
        return receipt
    finally:
        env.close()


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the sealed teacher and SmolVLA policy in native Isaac."""

    import gymnasium as gym

    inputs = _runtime_inputs(args)
    args.output_path.mkdir(parents=True, exist_ok=False)
    boundary_smoke = _runtime_boundary_smoke(gym, args, inputs)
    teacher = _policy_teacher(gym, args, inputs)
    env, configuration = _policy_environment(gym, args, inputs)
    try:
        env.reset(seed=args.seed)
        native = env.unwrapped
        binding, embodiment = _live_binding(native, inputs, configuration)
        client = _policy_client(inputs)
        reset = client.reset(policy=inputs.policy)
        context, trace, evidence = _policy_rollout(env, native, args, inputs, client)
        report = _execution_report(
            args=args,
            inputs=inputs,
            teacher=teacher,
            binding=binding,
            embodiment=embodiment,
            reset=reset,
            context=context,
            trace=trace,
            evidence_decisions=evidence,
        )
        report["runtime_boundary_smoke"] = boundary_smoke
        write_json(args.output_path / "report.json", report)
        return report
    finally:
        env.close()


def _policy_teacher(
    gym: Any, args: argparse.Namespace, inputs: RuntimeInputs
) -> dict[str, Any]:
    return _run_teacher(
        gym=gym,
        args=args,
        robot=inputs.robot,
        robot_spec=inputs.robot_spec,
        prop=inputs.prop,
        task=inputs.task,
        object_usd_path=inputs.object_usd_path,
        control_hz=inputs.policy.action.control_hz,
    )


def _policy_client(inputs: RuntimeInputs) -> SmolVLAPolicyClient:
    return SmolVLAPolicyClient(
        str(inputs.receipt["service_host"]), int(inputs.receipt.get("port", 8080))
    )


def _sim_time(native: Any) -> float:
    return float(native.common_step_counter) * float(native.step_dt)


def build_parser() -> argparse.ArgumentParser:
    """Build the native Isaac policy-rollout argument parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--preparation-path", type=Path, required=True)
    parser.add_argument("--service-receipt-path", type=Path, required=True)
    parser.add_argument("--robot-usd-path", type=Path, required=True)
    parser.add_argument("--robot-asset-manifest-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--decisions", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--action-selection",
        choices=("first_action", "uniform_temporal_ensemble"),
        default="first_action",
    )
    return parser


def _failure_receipt(args: argparse.Namespace, error: BaseException) -> dict[str, Any]:
    """Persist the native failure before Isaac cleanup can terminate."""

    receipt = {
        "schema": "npa.sim2real.policy_interchange.isaac_runtime_failure.v1",
        "status": "failed",
        "error_type": type(error).__name__,
        "error": str(error),
        "system_exit_code": error.code if isinstance(error, SystemExit) else None,
        "report_written": (args.output_path / "report.json").is_file(),
    }
    args.output_path.mkdir(parents=True, exist_ok=True)
    write_json(args.output_path / "runtime_failure.json", receipt)
    print(
        "NPA_INTERCHANGE_ISAAC_RUNTIME_FAILURE " + json.dumps(receipt, sort_keys=True),
        file=sys.stderr,
        flush=True,
    )
    traceback.print_exception(error, file=sys.stderr)
    return receipt


def _run_with_app(args: argparse.Namespace, app: Any) -> dict[str, Any]:
    """Run policy evaluation without allowing Isaac cleanup to erase failure."""

    result: dict[str, Any] | None = None
    failure: BaseException | None = None
    started = time.monotonic()
    try:
        result = run(args)
        result["duration_seconds"] = time.monotonic() - started
        write_json(args.output_path / "report.json", result)
    except BaseException as error:
        failure = error
        _failure_receipt(args, error)
    try:
        app.close()
    except BaseException as close_error:
        if failure is None:
            raise
        print(
            "NPA_INTERCHANGE_ISAAC_RUNTIME_CLOSE_FAILURE "
            + json.dumps(
                {"error_type": type(close_error).__name__, "error": str(close_error)},
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
    if failure is not None:
        if isinstance(failure, SystemExit):
            raise RuntimeError(
                "native policy runtime exited before completing its report "
                f"(SystemExit code={failure.code!r})"
            ) from failure
        raise failure.with_traceback(failure.__traceback__)
    if result is None:  # pragma: no cover - defensive invariant
        raise RuntimeError("native policy runtime returned no result")
    return result


def main(argv: list[str] | None = None) -> int:
    """Run one native Isaac rollout and emit its factual report."""

    args = build_parser().parse_args(argv)
    if args.decisions < 1 or args.seed < 0:
        raise ValueError("decisions must be positive and seed nonnegative")
    os.environ["OMNI_TELEMETRY_DISABLE_ANONYMOUS_DATA"] = "1"
    kit_args = validated_kit_args()
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, enable_cameras=True, kit_args=kit_args).app
    result = _run_with_app(args, app)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
