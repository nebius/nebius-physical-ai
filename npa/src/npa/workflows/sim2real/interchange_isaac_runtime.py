"""Run a sealed typed SmolVLA policy on a non-Franka Isaac surface task."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any

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
)
from npa.workflows.sim2real.smolvla_policy_client import SmolVLAPolicyClient


PROMPT = "Pick up the object and place it inside the cyan circular target on the table."
CONTACT_FORCE_N = 1.0e-3
TARGET_DISTANCE_M = 0.05
TRANSPORT_DISTANCE_M = 0.08
STABLE_SPEED_M_S = 0.03
STABLE_STEPS = 3


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain one JSON object")
    return value


def _camera_frame(native: Any, key: str, profile: Any) -> tuple[Any, float]:
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
    return frame, _sim_time(native)


def _observation(
    native: Any, robot: Any, policy: Any, decision: int
) -> tuple[Any, dict]:
    state = measured_policy_state(native, robot)
    state_time = _sim_time(native)
    cameras: dict[str, Any] = {}
    hashes: dict[str, str] = {}
    for camera in robot.cameras:
        frame, capture_time = _camera_frame(native, camera.key, robot)
        payload = frame.tobytes()
        digest = hashlib.sha256(payload).hexdigest()
        hashes[camera.key] = digest
        cameras[camera.key] = {
            "layout": camera.layout,
            "image_size_wh": list(camera.image_size),
            "mask": True,
            "capture_timestamp_s": capture_time,
            "state_timestamp_s": max(state_time, capture_time),
            "rgb8": payload,
            "image_sha256": digest,
        }
    observation = validate_observation(
        robot=robot,
        policy=policy,
        camera_payloads=cameras,
        measured_state=state,
        prompt=PROMPT,
        input_identity=f"non-franka-surface-decision-{decision}",
        task_identity=policy.task_identity,
    )
    return observation, hashes


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
    closed = (
        state[robot.measured_state_names[-1]]
        > (robot.gripper.model_open + robot.gripper.model_closed) / 2.0
    )
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
    events = context["ordered_event_steps"]
    ordered = (
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
    for name, condition in ordered:
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


def run(args: argparse.Namespace) -> dict[str, Any]:
    import gymnasium as gym

    from npa.workflows.franka_rl_embodiments import embodiment_evidence
    from npa.workflows.franka_rl_environment import (
        configure_policy_interchange_cameras,
        configure_policy_interchange_surface_task,
        environment_config,
    )
    from npa.workflows.sim2real.interchange_contract import assert_policy_compatible

    preparation = _load(args.preparation_path)
    receipt = _load(args.service_receipt_path)
    recipe = _load(args.input_path / "recipe.json")
    robot = robot_sensor_profile_from_dict(preparation.get("robot") or {})
    prop = rigid_object_profile_from_dict(preparation.get("object") or {})
    task = surface_placement_task_from_dict(preparation.get("task") or {})
    policy = policy_profile_from_dict(preparation.get("policy") or {})
    loaded = receipt.get("interchange") or {}
    if (
        receipt.get("status") != "ready"
        or policy.backend != "lerobot_smolvla"
        or loaded.get("policy_identity") != policy.identity
        or loaded.get("robot_identity") != robot.identity
        or loaded.get("task_identity") != task.identity
    ):
        raise RuntimeError("service, robot, task and policy identities are not bound")
    policy = replace(
        policy,
        checkpoint_sha256=str(loaded.get("checkpoint_sha256") or ""),
        normalization_identity=str(loaded.get("normalization_identity") or ""),
    )
    assert_policy_compatible(robot, task, policy)
    config = environment_config(
        recipe,
        training=False,
        capture=False,
        asset_root=args.input_path,
    )
    config.scene.num_envs = 1
    config.seed = args.seed
    configure_policy_interchange_surface_task(config, robot, prop, task)
    configure_policy_interchange_cameras(config, robot)
    env = gym.make(recipe["task"], cfg=config)
    try:
        env.reset(seed=args.seed)
        native = env.unwrapped
        layout = action_layout_from_live_isaac(native, robot)
        binding = assert_live_robot_matches_profile(
            robot_joint_names=native.scene["robot"].joint_names,
            robot_body_names=native.scene["robot"].body_names,
            layout=layout,
            profile=robot,
        )
        embodiment = embodiment_evidence(env, recipe)
        client = SmolVLAPolicyClient(
            str(receipt["service_host"]), int(receipt.get("port", 8080))
        )
        reset = client.reset(policy=policy)
        initial_z = _reference_position(native, prop)[2]
        context = {
            "ordered_event_steps": {},
            "required_events": task.success_events,
            "retreat_distance_m": task.retreat_distance_m,
            "stable_steps": 0,
        }
        controls, physics, inference, camera_hashes = [], [], [], []
        for decision in range(args.decisions):
            observation, hashes = _observation(native, robot, policy, decision)
            output, inference_receipt = client.infer_typed(
                observation=observation, policy=policy
            )
            if len(output.action_chunk) != 1:
                raise RuntimeError(
                    "native service must return one current queued action"
                )
            control = apply_absolute_joint_target(
                env, output.action_chunk[0], profile=robot
            )
            sample = _sample(native, robot, prop, task, initial_z)
            _update_events(context, sample, decision)
            controls.append(control)
            physics.append(sample)
            inference.append(inference_receipt)
            camera_hashes.append(hashes)
        success = tuple(context["ordered_event_steps"]) == task.success_events
        report = {
            "schema": "npa.sim2real.policy_interchange.non_franka_execution.v1",
            "execution_valid": len(controls) == args.decisions,
            "task_success": success,
            "ordered_event_steps": context["ordered_event_steps"],
            "success_contract": {
                "required_event_order": list(task.success_events),
                "target_xy_distance_m_lt": TARGET_DISTANCE_M,
                "object_speed_m_s_lt": STABLE_SPEED_M_S,
                "stable_steps_gte": STABLE_STEPS,
                "contact_force_n_gt": CONTACT_FORCE_N,
                "lift_delta_m_gte": task.lift_delta_m,
                "retreat_distance_m_gte": task.retreat_distance_m,
            },
            "robot_binding": binding,
            "embodiment": embodiment,
            "robot_identity": robot.identity,
            "object_identity": prop.identity,
            "task_identity": task.identity,
            "policy_identity": policy.identity,
            "checkpoint_sha256": policy.checkpoint_sha256,
            "normalization_identity": policy.normalization_identity,
            "service_reset": reset,
            "seed": args.seed,
            "decisions": args.decisions,
            "controls": controls,
            "physics": physics,
            "inference": inference,
            "camera_hashes": camera_hashes,
            "privileged_object_state_policy_input": False,
            "physical_robot_tested": False,
        }
        args.output_path.mkdir(parents=True, exist_ok=False)
        write_json(args.output_path / "report.json", report)
        return report
    finally:
        env.close()


def _sim_time(native: Any) -> float:
    return float(native.common_step_counter) * float(native.step_dt)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--preparation-path", type=Path, required=True)
    parser.add_argument("--service-receipt-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--decisions", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.decisions < 1 or args.seed < 0:
        raise ValueError("decisions must be positive and seed nonnegative")
    os.environ["OMNI_TELEMETRY_DISABLE_ANONYMOUS_DATA"] = "1"
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, enable_cameras=True).app
    try:
        started = time.monotonic()
        result = run(args)
        result["duration_seconds"] = time.monotonic() - started
        write_json(args.output_path / "report.json", result)
    finally:
        app.close()
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
