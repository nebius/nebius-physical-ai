"""Closed-loop Isaac evaluation of a private pinned OpenPI pi0.5 service."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from npa.workflows.sim2real.pi05_contract import (
    ACTION_HORIZON,
    CONTROL_HZ,
    EXECUTION_PREFIX,
    JOINT_NAMES,
    SCHEMA as PI05_TASK_SCHEMA,
    TASK_ID as PI05_TASK_ID,
    absolute_to_isaac_action,
    assert_contract,
    build_contract,
    release_width_threshold,
    verify_controller_roundtrip,
    width_to_droid_gripper,
)
from npa.workflows.sim2real.pi05_isaac import (
    POLICY_BASE_TASK_ID,
    POLICY_SURFACE_TASK_ID,
    RESET_SETTLE_CONTROL_STEPS,
    _camera_config,
    _bind_rgb_clock,
    _camera_pose_sample,
    _camera_time,
    _contact_config,
    _destination_pixel_count,
    _contact_forces,
    _finger_in_frustum,
    _finger_pixel_counts,
    _finger_controller_target,
    _gripper_width,
    _graphics_preflight,
    _hand_origin_control_receipt,
    _launch_camera_app,
    _verify_scene_semantics,
    _physics_randomization,
    _object_asset_geometry,
    _robot_plant_receipt,
    _proxy_torch,
    _rgb,
    _rgb_receipt,
    _reset_settle_sample,
    _require_expert_phase_visibility,
    _semantic_encoding_receipt,
    _sim_time,
    _sensor_clock_reference,
    _synchronize_wrist_before_capture,
    _target_pixel_count,
    _validate_capture_ages,
    _validate_camera_motion_metrics,
    _validate_reset_settle,
    _verify_camera_motion,
    _wrist_mount_receipt,
    configure_surface_goal,
    configure_destination_marker,
    configure_finger_semantics,
    configure_hand_origin_frames,
    configure_object_reset,
    configure_robot_plant,
    configure_support_surface,
    register_surface_task,
    support_surface_contract,
)
from npa.workflows.sim2real.pi05_policy_client import Pi05PolicyClient
from npa.workflows.sim2real.interchange_contract import (
    policy_profile_from_dict,
    robot_sensor_profile_from_dict,
    validate_observation,
)
from npa.workflows.sim2real.interchange_object_asset import (
    assert_collection_object_binding,
    materialize_prepared_object,
)
from npa.workflows.sim2real.smolvla_policy_client import SmolVLAPolicyClient


PROMPT = (
    "Pick up the object and release it inside the cyan circular target on the table."
)
WRIST_ABLATION_MODES = ("fresh", "frozen", "pixel_channel_shuffle")
LEGACY_DEFAULT_OBJECT_ASSET = (
    "Isaac/Props/Blocks/MultiColorCube/multi_color_cube_instanceable.usd"
)
OBJECT_SOURCE_FIELDS = (
    "profile_identity",
    "source_uri",
    "source_sha256",
    "bundle_sha256",
)
PI05_SAMPLER_NOISE_SHAPE = [15, 32]
CAMERA_MOTION_METRICS = (
    "wrist_world_motion_m",
    "hand_world_motion_m",
    "wrist_renderer_world_motion_m",
    "wrist_mount_distance_drift_m",
    "wrist_mount_angular_drift_rad",
    "exterior_world_drift_m",
    "wrist_renderer_position_error_m",
    "wrist_renderer_orientation_error_rad",
)
EFFECTIVENESS_PROTOCOL_FIELDS = {
    "task",
    "split",
    "seed",
    "episodes",
    "decisions",
    "object_identity",
    "object_usd",
    "object_scale",
    "object_width_m",
    "object_height_m",
    "object_geometry",
    "scene_id",
    "destination_marker_contract",
    "support_surface_contract",
    "robot_plant",
    "control_hz",
    "execution_prefix",
    "action_selection",
    "scenarios",
}
MANIPULATION_EVENT_ORDER = (
    "contact",
    "grasp",
    "lift",
    "transport",
    "release",
    "support_contact",
    "stable",
)


def _action_selection_protocol() -> dict[str, Any]:
    return {
        "schema": "npa.sim2real.pi05.temporal_action_ensemble.v1",
        "method": "uniform_mean",
        "model_horizon": ACTION_HORIZON,
        "fresh_observation_per_control": True,
        "causality": "current_and_past_predictions_only",
    }


def _vector(value: Any, size: int, label: str) -> list[float]:
    import numpy as np

    value = _proxy_torch(value)
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise RuntimeError(f"{label} must contain {size} finite values")
    return [float(item) for item in array]


def _action_parameter_row(value: Any, size: int, label: str) -> list[float]:
    value = _proxy_torch(value)
    shape = tuple(int(dimension) for dimension in getattr(value, "shape", ()))
    if shape != (1, size):
        raise RuntimeError(
            f"{label} must have single-environment shape (1, {size}), got {shape}"
        )
    return _vector(value[0], size, label)


def _action_scale(value: Any, size: int) -> list[float]:
    import math
    import numpy as np

    value = _proxy_torch(value)
    shape = tuple(int(dimension) for dimension in getattr(value, "shape", ()))
    if shape == (1, size):
        scales = _vector(value[0], size, "arm action scales")
    elif shape == ():
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        scalar = float(np.asarray(value).item())
        scales = [scalar] * size
    else:
        raise RuntimeError(
            f"arm action scales must be a scalar or shape (1, {size}), got {shape}"
        )
    if any(not math.isfinite(scale) or scale == 0.0 for scale in scales):
        raise RuntimeError("arm action scales must be finite and nonzero")
    return scales


def _reject_configured_action_clip(arm: Any) -> None:
    cfg_clip = getattr(getattr(arm, "cfg", None), "clip", None)
    runtime_clip = getattr(arm, "_clip", None)
    if cfg_clip is not None or runtime_clip is not None:
        raise RuntimeError(
            "pi0.5 absolute targets require an unclipped Isaac affine action term"
        )


def _joint_bridge(native: Any) -> tuple[list[float], list[float], int, int, int]:
    names = list(native.action_manager.active_terms)
    dimensions = [int(value) for value in native.action_manager.action_term_dim]
    if "arm_action" not in names or "gripper_action" not in names:
        raise RuntimeError("policy task lacks named arm and gripper actions")
    arm_index, grip_index = names.index("arm_action"), names.index("gripper_action")
    if dimensions[arm_index] != 7 or dimensions[grip_index] != 1:
        raise RuntimeError(
            "policy task must expose seven joint actions and one gripper"
        )
    arm = native.action_manager.get_term("arm_action")
    _reject_configured_action_clip(arm)
    offset, scale = getattr(arm, "_offset", None), getattr(arm, "_scale", None)
    if offset is None or scale is None:
        raise RuntimeError("arm action term does not expose configured offsets/scales")
    defaults = _action_parameter_row(offset, 7, "arm action offsets")
    scales = _action_scale(scale, 7)
    return (
        defaults,
        scales,
        sum(dimensions),
        sum(dimensions[:arm_index]),
        sum(dimensions[:grip_index]),
    )


def _policy_observation(native: Any) -> tuple[Any, Any, Any, Any]:
    import numpy as np

    _synchronize_wrist_before_capture(native)
    robot = native.scene["robot"]
    indices = [list(robot.joint_names).index(name) for name in JOINT_NAMES]
    joint = (
        _proxy_torch(robot.data.joint_pos)[0, indices]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    gripper = np.asarray(
        [width_to_droid_gripper(_gripper_width(native))], dtype=np.float32
    )
    return _rgb(native, "pi05_exterior"), _rgb(native, "pi05_wrist"), joint, gripper


def _apply_target(
    env: Any,
    target: Any,
    *,
    defaults: list[float],
    scales: list[float],
    total_dim: int,
    arm_start: int,
    grip_start: int,
) -> dict[str, Any]:
    import torch

    native = env.unwrapped
    rendered = absolute_to_isaac_action(target, defaults=defaults, scales=scales)
    action = torch.zeros((1, total_dim), device=native.device)
    action[0, arm_start : arm_start + 7] = torch.as_tensor(
        rendered[:7], device=native.device
    )
    action[0, grip_start] = rendered[7]
    before = _sim_time(native)
    _, _, terminated, truncated, _ = env.step(action)
    after = _sim_time(native)
    if abs((after - before) - 1.0 / CONTROL_HZ) > 1.0e-6:
        raise RuntimeError("policy action hold does not match 15 Hz controller cadence")
    proof = _target_proof(native, target, defaults, scales)
    proof.update(
        {
            "command_application_time_s": before,
            "resulting_state_time_s": after,
            "terminated": bool(terminated[0]),
            "truncated": bool(truncated[0]),
        }
    )
    return proof


def _target_proof(
    native: Any, target: Any, defaults: list[float], scales: list[float]
) -> dict[str, Any]:
    robot = native.scene["robot"]
    indices = [list(robot.joint_names).index(name) for name in JOINT_NAMES]
    controller = (
        _proxy_torch(robot.data.joint_pos_target)[0, indices].detach().cpu().numpy()
    )
    return verify_controller_roundtrip(
        target,
        defaults=defaults,
        scales=scales,
        controller_target=controller,
        finger_controller_target_m=_finger_controller_target(native),
    )


def _physics_sample(
    native: Any, initial_z: float, target_xy: Any, object_width_m: float, closing: bool
) -> dict[str, Any]:
    import numpy as np
    import torch

    obj = _proxy_torch(native.scene["object"].data.root_pos_w)[0, :3]
    speed = float(
        torch.linalg.norm(_proxy_torch(native.scene["object"].data.root_lin_vel_w)[0])
    )
    left, right, support = _contact_forces(native)
    width = _gripper_width(native)
    position = obj.detach().cpu().numpy()
    robot = native.scene["robot"]
    hand_index = list(robot.body_names).index("panda_hand")
    hand = _proxy_torch(robot.data.body_pos_w)[0, hand_index].detach().cpu().numpy()
    return {
        "object_position_m": position.tolist(),
        "object_speed_m_s": speed,
        "object_lift_m": float(position[2] - initial_z),
        "target_xy_distance_m": float(np.linalg.norm(position[:2] - target_xy)),
        "left_finger_object_force_n": left,
        "right_finger_object_force_n": right,
        "object_support_force_n": support,
        "gripper_width_m": width,
        "hand_object_distance_m": float(np.linalg.norm(hand - position)),
        "bilateral_finger_contact": left > 1.0e-3 and right > 1.0e-3,
        "gripper_closing_intent": closing,
        "object_width_m": object_width_m,
    }


def _qualified_grasp(row: dict[str, Any]) -> bool:
    return (
        row["bilateral_finger_contact"]
        and row["gripper_closing_intent"]
        and abs(row["gripper_width_m"] - row["object_width_m"]) <= 0.015
        and (row["object_speed_m_s"] > 0.005 or row["object_lift_m"] > 0.002)
    )


def _supported_release(row: dict[str, Any]) -> tuple[bool, bool]:
    released = row["gripper_width_m"] >= release_width_threshold(row["object_width_m"])
    released = released and not row["bilateral_finger_contact"]
    supported = released and row["object_support_force_n"] > 1.0e-3
    supported = supported and row["target_xy_distance_m"] < 0.05
    supported = supported and row["object_speed_m_s"] < 0.03
    supported = supported and row["hand_object_distance_m"] >= 0.10
    return released, supported


def _success_result(
    trace: list[dict[str, Any]], events: dict[str, int], stable_steps: int
) -> dict[str, Any]:
    order = MANIPULATION_EVENT_ORDER
    return {
        **{name: name in events for name in order[:4]},
        "released": "release" in events,
        "support_contact": "support_contact" in events,
        "stable_steps": stable_steps,
        "ordered_event_steps": events,
        "required_event_order": list(order),
        "released_supported_surface_placement": "stable" in events,
        "old_strict_5cm_diagnostic_only": any(
            row["target_xy_distance_m"] < 0.05 for row in trace
        ),
    }


def _success(trace: list[dict[str, Any]]) -> dict[str, Any]:
    events, _stage, maximum_stable = _manipulation_progress(trace)
    return _success_result(trace, events, maximum_stable)


def _manipulation_progress(
    trace: list[dict[str, Any]],
) -> tuple[dict[str, int], str, int]:
    events: dict[str, int] = {}
    grasp_run = stable = maximum_stable = 0
    stage = "contact"
    for index, row in enumerate(trace):
        if stage == "contact" and row["bilateral_finger_contact"]:
            events[stage], stage = index, "grasp"
            continue
        grasp_now = _qualified_grasp(row)
        grasp_run = grasp_run + 1 if stage == "grasp" and grasp_now else 0
        if stage == "grasp" and grasp_run >= 2:
            events[stage], stage = index, "lift"
            continue
        elif stage == "lift" and row["object_lift_m"] >= 0.05:
            events[stage], stage = index, "transport"
            continue
        elif (
            stage == "transport"
            and row["target_xy_distance_m"] < 0.08
            and row["object_lift_m"] >= 0.05
        ):
            events[stage], stage = index, "release"
            continue
        released_now, supported = _supported_release(row)
        if stage == "release" and released_now:
            events[stage], stage, stable = index, "support_contact", 0
            continue
        if stage == "support_contact" and supported:
            events[stage], stage, stable = index, "stable", 1
        elif stage == "stable":
            stable = stable + 1 if supported else 0
        maximum_stable = max(maximum_stable, stable)
        if stage == "stable" and stable >= 3:
            events[stage], stage = index, "complete"
    return events, stage, maximum_stable


def _new_episode_record() -> dict[str, Any]:
    return {
        "queries": [],
        "physics_trace": [],
        "camera_trace": [],
        "wrist_mount_receipts": [],
        "_exterior_frames": [],
        "_wrist_frames": [],
    }


def _camera_query_receipts(native: Any, exterior: Any, wrist: Any) -> tuple[dict, dict]:
    observation_time = _sim_time(native)
    exterior_time = _camera_time(native, "pi05_exterior")
    wrist_time = _camera_time(native, "pi05_wrist")
    clock = _sensor_clock_reference(native)
    _validate_capture_ages((exterior_time, wrist_time), clock, native.step_dt)
    timing = {
        "observation_time_s": observation_time,
        "exterior_capture_time_s": exterior_time,
        "wrist_capture_time_s": wrist_time,
    }
    receipts = {
        "exterior": _rgb_receipt(exterior, exterior_time, observation_time),
        "wrist": _rgb_receipt(wrist, wrist_time, observation_time),
    }
    for receipt in receipts.values():
        _bind_rgb_clock(receipt, clock)
    return timing, receipts


def _policy_phase(record: dict[str, Any]) -> str:
    _events, stage, _stable = _manipulation_progress(record["physics_trace"])
    return {
        "contact": "approach",
        "grasp": "grasp",
        "lift": "lift",
        "transport": "transport",
        "release": "lower",
        "support_contact": "release",
        "stable": "retreat",
        "complete": "retreat",
    }[stage]


def _record_query_camera(native: Any, record: dict, receipts: dict, wire: dict) -> None:
    camera = _camera_pose_sample(native)
    destination_pixels = _destination_pixel_count(native, "pi05_exterior")
    phase = _policy_phase(record)
    if destination_pixels < 16 and phase != "retreat":
        raise RuntimeError("cyan destination marker is not observable in exterior RGB")
    camera.update(
        {
            "wrist_target_pixels": _target_pixel_count(native, "pi05_wrist"),
            "exterior_target_pixels": _target_pixel_count(native, "pi05_exterior"),
            "exterior_destination_pixels": destination_pixels,
            "wrist_finger_in_frustum": _finger_in_frustum(native),
            "wrist_finger_pixels": _finger_pixel_counts(native, "pi05_wrist"),
            "semantic_encoding": {
                name: _semantic_encoding_receipt(native, name)
                for name in ("pi05_exterior", "pi05_wrist")
            },
            "phase": phase,
        }
    )
    record["camera_trace"].append(
        {
            "camera_pose": camera,
            "exterior_rgb": receipts["exterior"],
            "wrist_rgb": receipts["wrist"],
            "wire_input": wire,
            **camera,
        }
    )


def _wrist_variant(
    context: dict[str, Any], wrist: Any, decision: int
) -> tuple[Any, dict]:
    import numpy as np

    mode = context["wrist_mode"]
    if context.get("frozen_wrist") is None:
        context["frozen_wrist"] = np.asarray(wrist).copy()
    if mode == "fresh":
        value = wrist
    elif mode == "frozen":
        value = context["frozen_wrist"]
    elif mode == "pixel_channel_shuffle":
        value = np.asarray(wrist).copy().reshape(-1)
        rng = np.random.default_rng(context["ablation_seed"] + decision)
        value = value[rng.permutation(value.size)].reshape(np.asarray(wrist).shape)
    else:
        raise RuntimeError(f"unsupported wrist ablation mode: {mode}")
    return value, {
        "mode": mode,
        "decision": decision,
        "source_sha256": _rgb_receipt(wrist, 0.0, 0.0)["sha256"],
        "policy_sha256": _rgb_receipt(value, 0.0, 0.0)["sha256"],
    }


def _decision_noise(context: dict[str, Any], decision: int) -> tuple[Any, str]:
    import numpy as np

    protocol = context["client"].metadata.get("npa_explicit_noise_protocol") or {}
    shape = tuple(protocol.get("shape") or ())
    if protocol.get("schema") != "npa.openpi.pi05-explicit-noise.v1" or len(shape) != 2:
        raise RuntimeError(
            "pi0.5 service lacks validated explicit model-noise metadata"
        )
    seed = np.random.SeedSequence([context["ablation_seed"], decision])
    noise = np.random.default_rng(seed).standard_normal(shape, dtype=np.float32)
    sequence = f"scene-{context['scenario_seed']}-decision-{decision}"
    return noise, sequence


def _query_policy(
    context: dict[str, Any], decision: int
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    native, record = context["native"], context["record"]
    exterior, wrist, joint, gripper = _policy_observation(native)
    timing, receipts = _camera_query_receipts(native, exterior, wrist)
    timing["exterior_rgb_sha256"] = receipts["exterior"]["sha256"]
    timing["wrist_rgb_sha256"] = receipts["wrist"]["sha256"]
    try:
        actions, inference = _infer_query(
            context, decision, (exterior, wrist, joint, gripper), receipts
        )
        _record_query_camera(native, record, receipts, inference["wire_input"])
    except Exception as error:
        _retain_failed_camera(record, exterior, wrist, timing, error)
        raise
    record["_exterior_frames"].append(exterior)
    record["_wrist_frames"].append(wrist)
    return actions, inference, timing


def _infer_query(context: dict, decision: int, observation: tuple, receipts: dict):
    exterior, wrist, joint, gripper = observation
    policy_wrist, intervention = _wrist_variant(context, wrist, decision)
    if context.get("policy_backend") == "lerobot_smolvla":
        return _infer_smolvla_query(
            context,
            decision,
            exterior,
            policy_wrist,
            receipts,
            intervention,
        )
    sampling_noise, sampling_sequence = _decision_noise(context, decision)
    actions, inference = context["client"].infer(
        exterior=exterior,
        wrist=policy_wrist,
        joint_position=joint,
        gripper_position=gripper,
        prompt=PROMPT,
        capture_age_s=receipts["wrist"]["age_s"],
        source_wrist_sha256=intervention["policy_sha256"],
        sampling_noise=sampling_noise,
        sampling_sequence=sampling_sequence,
    )
    inference["wrist_intervention"] = intervention
    return actions, inference


def _infer_smolvla_query(
    context: dict[str, Any],
    decision: int,
    exterior: Any,
    wrist: Any,
    receipts: dict[str, Any],
    intervention: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Run the typed SmolVLA seam without inventing OpenPI sampler noise."""

    import numpy as np

    native = context["native"]
    robot_profile = context["robot_profile"]
    policy_profile = context["policy_profile"]
    robot = native.scene["robot"]
    names = list(robot.joint_names)
    arm_state = (
        _proxy_torch(robot.data.joint_pos)[
            0, [names.index(name) for name in robot_profile.arm_joint_names]
        ]
        .detach()
        .cpu()
        .numpy()
    )
    gripper_positions = (
        _proxy_torch(robot.data.joint_pos)[
            0, [names.index(name) for name in robot_profile.gripper.joint_names]
        ]
        .detach()
        .cpu()
        .numpy()
    )
    width = robot_profile.gripper.joint_positions_to_width(gripper_positions)
    state = np.r_[arm_state, robot_profile.gripper.width_to_model(width)]
    state_map = {
        name: float(value)
        for name, value in zip(robot_profile.measured_state_names, state, strict=True)
    }
    frames_by_role = {"exterior": exterior, "wrist": wrist}
    camera_payloads = {}
    for camera in robot_profile.cameras:
        frame = np.ascontiguousarray(frames_by_role[camera.role], dtype=np.uint8)
        receipt = receipts[camera.role]
        camera_payloads[camera.key] = {
            "layout": camera.layout,
            "image_size_wh": list(camera.image_size),
            "mask": True,
            "capture_timestamp_s": receipt["capture_time_s"],
            "state_timestamp_s": receipt["observation_time_s"],
            "rgb8": frame.tobytes(),
            "image_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
        }
    observation = validate_observation(
        robot=robot_profile,
        policy=policy_profile,
        camera_payloads=camera_payloads,
        measured_state=state_map,
        prompt=PROMPT,
        input_identity=(
            f"scene-{context['scenario_seed']}-decision-{decision}-"
            f"{context['wrist_mode']}"
        ),
        task_identity=policy_profile.task_identity,
    )
    output, inference = context["client"].infer_typed(
        observation=observation, policy=policy_profile
    )
    actions = np.asarray(output.action_chunk, dtype=np.float64)
    inference["wire_input"] = {
        "schema": "npa.sim2real.policy_observation_receipt.v1",
        "input_identity": observation.input_identity,
        "exterior_rgb_sha256": next(
            view.rgb_sha256 for view in observation.views if view.role == "exterior"
        ),
        "wrist_rgb_sha256": next(
            view.rgb_sha256 for view in observation.views if view.role == "wrist"
        ),
        "capture_age_s": next(
            view.state_timestamp_s - view.capture_timestamp_s
            for view in observation.views
            if view.role == "wrist"
        ),
        "measured_state_names": list(observation.measured_state_names),
    }
    inference["wrist_intervention"] = intervention
    return actions, inference


def _retain_failed_camera(
    record: dict, exterior: Any, wrist: Any, timing: dict, error: Exception
) -> None:
    record.setdefault("failed_camera_observations", []).append(
        {**timing, "failure": _execution_error(error), "policy_query_recorded": False}
    )
    record.setdefault("_failed_exterior_frames", []).append(exterior)
    record.setdefault("_failed_wrist_frames", []).append(wrist)


def _write_failed_camera_evidence(work: Path, episodes: list[dict]) -> list[dict]:
    import numpy as np

    artifacts = []
    for index, episode in enumerate(episodes):
        for camera in ("exterior", "wrist"):
            frames = episode.pop(f"_failed_{camera}_frames", [])
            if not frames:
                continue
            path = work / f"episode-{index:06d}-failed-{camera}.npy"
            np.save(path, np.asarray(frames, dtype=np.uint8), allow_pickle=False)
            artifacts.append(
                {
                    "path": path.name,
                    "episode": index,
                    "camera": camera,
                    "frames": len(frames),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    return artifacts


def _execute_prefix(context: dict[str, Any], actions: Any, query: dict) -> bool:
    bridge = context["bridge"]
    if context.get("policy_backend") == "lerobot_smolvla":
        target, selection = _current_action_target(actions, query)
    else:
        target, selection = _temporally_ensembled_target(context, actions, query)
    query["action_selection"] = selection
    for index, target in enumerate([target]):
        query["attempted_prefix"] = index + 1
        proof = _apply_target(
            context["env"],
            target,
            defaults=bridge[0],
            scales=bridge[1],
            total_dim=bridge[2],
            arm_start=bridge[3],
            grip_start=bridge[4],
        )
        query["controller_proofs"].append(proof)
        query["executed_prefix"] = len(query["controller_proofs"])
        try:
            sample = _physics_sample(
                context["native"],
                context["initial_z"],
                context["target_xy"],
                context["object_width_m"],
                bool(float(target[7]) > 0.5),
            )
        except Exception as error:
            proof["physics_observation_status"] = "failed"
            proof["physics_observation_error"] = _execution_error(error)
            raise
        proof["physics_observation_status"] = "recorded"
        proof["physics_trace_index"] = len(context["record"]["physics_trace"])
        sample["simulator_time_s"] = proof["resulting_state_time_s"]
        context["record"]["physics_trace"].append(sample)
        if proof["terminated"] or proof["truncated"]:
            return True
    return False


def _current_action_target(
    actions: Any, query: dict[str, Any]
) -> tuple[Any, dict[str, Any]]:
    """Apply the current native SmolVLA action; its queue lives in the runtime."""

    import numpy as np

    value = np.ascontiguousarray(actions, dtype=np.float64)
    if value.shape != (1, 8) or not np.isfinite(value).all():
        raise RuntimeError("SmolVLA must return one finite current 8D action")
    target = value[0].copy()
    return target, {
        "schema": "npa.sim2real.current_native_action.v1",
        "method": "native_policy_queue_current_action",
        "fresh_observation_per_control": True,
        "causality": "current_observation_and_native_policy_history",
        "decision": int(query["decision"]),
        "contributor_count": 1,
        "contributors": [
            {
                "origin_decision": int(query["decision"]),
                "action_index": 0,
                "chunk_sha256": query["action_receipt"]["sha256"],
            }
        ],
        "weight_per_contributor": 1.0,
        "applied_action_sha256": hashlib.sha256(target.tobytes()).hexdigest(),
        "applied_action_dtype": str(target.dtype),
        "applied_action_shape": list(target.shape),
        "applied_action": target.tolist(),
        "applied_gripper_score": float(target[7]),
        "applied_gripper_command": int(float(target[7]) > 0.5),
    }


def _temporally_ensembled_target(
    context: dict[str, Any], actions: Any, query: dict[str, Any]
) -> tuple[Any, dict[str, Any]]:
    """Select one causal target from overlapping fresh-observation chunks."""

    import numpy as np

    decision = int(query["decision"])
    value = np.ascontiguousarray(actions)
    if value.shape != (ACTION_HORIZON, 8) or not np.isfinite(value).all():
        raise RuntimeError("policy action chunk is not finite 15x8 data")
    history = context.setdefault("action_chunks", [])
    if decision == 0:
        history.clear()
    elif not history or int(history[-1]["origin_decision"]) != decision - 1:
        raise RuntimeError("temporal action ensemble history is not episode-contiguous")
    history.append(
        {
            "origin_decision": decision,
            "actions": value.copy(),
            "chunk_sha256": query["action_receipt"]["sha256"],
        }
    )
    history[:] = [
        row
        for row in history
        if 0 <= decision - int(row["origin_decision"]) < ACTION_HORIZON
    ]
    contributors, candidates = [], []
    for row in history:
        offset = decision - int(row["origin_decision"])
        candidate = row["actions"][offset]
        candidates.append(candidate)
        contributors.append(
            {
                "origin_decision": int(row["origin_decision"]),
                "action_index": offset,
                "chunk_sha256": row["chunk_sha256"],
            }
        )
    if not candidates or contributors[-1]["action_index"] != 0:
        raise RuntimeError("temporal action ensemble lacks the current prediction")
    target = np.ascontiguousarray(np.mean(np.stack(candidates), axis=0))
    if target.shape != (8,) or not np.isfinite(target).all():
        raise RuntimeError("temporal action ensemble produced an invalid target")
    receipt = {
        **_action_selection_protocol(),
        "decision": decision,
        "contributor_count": len(contributors),
        "contributors": contributors,
        "weight_per_contributor": 1.0 / len(contributors),
        "applied_action_sha256": hashlib.sha256(target.tobytes()).hexdigest(),
        "applied_action_dtype": str(target.dtype),
        "applied_action_shape": list(target.shape),
        "applied_action": target.tolist(),
        "applied_gripper_score": float(target[7]),
        "applied_gripper_command": int(float(target[7]) > 0.5),
    }
    return target, receipt


def _action_receipt(actions: Any) -> dict[str, Any]:
    import numpy as np

    value = np.ascontiguousarray(actions)
    scores = value[:, 7].tolist()
    return {
        "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "actions": value.tolist(),
        "gripper_scores": scores,
        "thresholded_gripper_commands": [int(score > 0.5) for score in scores],
    }


def _execution_error(error: Exception) -> str:
    import re

    message = f"{type(error).__name__}: {error}"
    return re.sub(r"\b(?:s3|gs|https?)://[^\s'\"\\]+", "<redacted-uri>", message)


def _decision_step(context: dict[str, Any], decision: int) -> bool:
    actions, inference, timing = _query_policy(context, decision)
    query = {
        "decision": decision,
        **timing,
        "inference": inference,
        "action_receipt": _action_receipt(actions),
        "execution_status": "pending",
        "attempted_prefix": 0,
        "executed_prefix": 0,
        "controller_proofs": [],
    }
    context["record"]["queries"].append(query)
    try:
        terminal = _execute_prefix(context, actions, query)
    except Exception as error:
        query["execution_status"] = "failed"
        query["execution_error"] = _execution_error(error)
        raise
    query["execution_status"] = "completed"
    return terminal


def _finish_episode(record: dict[str, Any]) -> dict[str, Any]:
    record["success"] = _success(record["physics_trace"])
    camera_trace = record["camera_trace"]
    record["camera_evidence"] = _verify_camera_motion(
        camera_trace, require_expert_phases=False
    )
    record.pop("camera_trace")
    record["measured_inference_round_trip_ms"] = [
        row["inference"]["round_trip_ms"] for row in record["queries"]
    ]
    times = [row["observation_time_s"] for row in record["queries"]]
    record["measured_requery_sim_seconds"] = [b - a for a, b in zip(times, times[1:])]
    record["receding_horizon"] = {
        "model_horizon": record.get("model_horizon", ACTION_HORIZON),
        "configured_execute_prefix": EXECUTION_PREFIX,
        "action_selection": record.get(
            "action_selection_protocol", _action_selection_protocol()
        ),
    }
    return record


def _episode(
    env: Any,
    client: Pi05PolicyClient,
    decisions: int,
    record: dict[str, Any],
    scenario: dict[str, Any],
    wrist_mode: str,
    ablation_seed: int,
    object_width_m: float,
    *,
    robot_profile: Any = None,
    policy_profile: Any = None,
) -> dict[str, Any]:
    native = env.unwrapped
    if getattr(client, "metadata", {}).get("backend") == "lerobot_smolvla":
        record["policy_reset"] = client.reset(policy=policy_profile)
    bridge = _joint_bridge(native)
    settle = _settle_policy_reset(env, bridge)
    record["hand_origin_control"] = _hand_origin_control_receipt(
        native, require_ik=False
    )
    record["robot_plant"] = _robot_plant_receipt(native)
    initial = _verified_initial_object_position(native, scenario)
    _wrist_mount_receipt(native, record)
    context = _episode_context(
        env,
        native,
        client,
        record,
        bridge,
        initial,
        scenario,
        object_width_m,
        wrist_mode,
        ablation_seed,
        robot_profile=robot_profile,
        policy_profile=policy_profile,
    )
    record["wrist_ablation_mode"] = wrist_mode
    record["scenario_seed"] = int(scenario["scene_seed"])
    record["reset_settle"] = settle
    for decision in range(decisions):
        if _decision_step(context, decision):
            break
    return _finish_episode(record)


def _episode_context(
    env: Any,
    native: Any,
    client: Pi05PolicyClient,
    record: dict[str, Any],
    bridge: Any,
    initial: Any,
    scenario: dict[str, Any],
    object_width_m: float,
    wrist_mode: str,
    ablation_seed: int,
    *,
    robot_profile: Any = None,
    policy_profile: Any = None,
) -> dict[str, Any]:
    import numpy as np

    backend = getattr(client, "metadata", {}).get("backend", "openpi_pi05")
    if backend == "lerobot_smolvla":
        record["model_horizon"] = policy_profile.action.horizon
        record["action_selection_protocol"] = {
            "schema": "npa.sim2real.current_native_action.v1",
            "method": "native_policy_queue_current_action",
            "fresh_observation_per_control": True,
            "causality": "current_observation_and_native_policy_history",
        }
    return {
        "env": env,
        "native": native,
        "client": client,
        "record": record,
        "bridge": bridge,
        "initial_z": float(initial[2]),
        "target_xy": np.asarray(scenario["target_position_m"][:2]),
        "object_width_m": object_width_m,
        "wrist_mode": wrist_mode,
        "ablation_seed": ablation_seed,
        "scenario_seed": int(scenario["scene_seed"]),
        "frozen_wrist": None,
        "policy_backend": backend,
        "robot_profile": robot_profile,
        "policy_profile": policy_profile,
    }


def _open_joint_target(native: Any) -> Any:
    import numpy as np

    robot = native.scene["robot"]
    indices = [list(robot.joint_names).index(name) for name in JOINT_NAMES]
    joints = _proxy_torch(robot.data.joint_pos)[0, indices].detach().cpu().numpy()
    return np.r_[joints, 0.0].astype(np.float32)


def _settle_policy_reset(env: Any, bridge: tuple) -> dict[str, Any]:
    """Hold measured joints open through the declared reset-settle protocol."""

    native = env.unwrapped
    target = _open_joint_target(native)
    before = int(_proxy_torch(native.episode_length_buf)[0].item())
    proofs, samples = [], []
    for _ in range(RESET_SETTLE_CONTROL_STEPS):
        command_time = _sim_time(native)
        proof = _apply_target(
            env,
            target,
            defaults=bridge[0],
            scales=bridge[1],
            total_dim=bridge[2],
            arm_start=bridge[3],
            grip_start=bridge[4],
        )
        proofs.append(proof)
        samples.append(_reset_settle_sample(native, command_time))
    after = int(_proxy_torch(native.episode_length_buf)[0].item())
    if any(proof["terminated"] or proof["truncated"] for proof in proofs):
        raise RuntimeError("policy environment failed its declared reset settle")
    _validate_reset_settle(samples, after - before)
    return {
        "control_steps": RESET_SETTLE_CONTROL_STEPS,
        "physics_steps": RESET_SETTLE_CONTROL_STEPS * int(native.cfg.decimation),
        "start_control_step": before,
        "result_control_step": after,
        "object_speed_m_s": samples[-1]["object_speed_m_s"],
        "object_support_force_n": samples[-1]["object_support_force_n"],
        "samples": samples,
        "training_or_evaluation_row": False,
        "controller_proofs": proofs,
    }


def _verified_initial_object_position(native: Any, scenario: dict[str, Any]) -> Any:
    import numpy as np

    initial = (
        _proxy_torch(native.scene["object"].data.root_pos_w)[0, :3]
        .detach()
        .cpu()
        .numpy()
    )
    expected = np.asarray(scenario["initial_object_position_m"])
    if not np.allclose(initial, expected, atol=1.0e-4, rtol=0.0):
        raise RuntimeError("evaluation reset does not reproduce the gold scenario")
    return initial


def _write_video(work: Path, episodes: list[dict[str, Any]]) -> tuple[Path, int]:
    import numpy as np
    from PIL import Image, ImageDraw

    from npa.adapter.sim_to_lerobot import encode_video

    rows = [
        row
        for row in episodes
        if row.get("_exterior_frames")
        and row.get("_wrist_frames")
        and row.get("queries")
    ]
    if not rows:
        raise RuntimeError(
            "closed-loop evaluation produced no reviewable camera frames"
        )
    panels = []
    for episode_index, row in enumerate(rows):
        for query_index, (exterior, wrist, query) in enumerate(
            zip(row["_exterior_frames"], row["_wrist_frames"], row["queries"])
        ):
            canvas = Image.new("RGB", (448, 256), "black")
            canvas.paste(Image.fromarray(np.asarray(exterior)), (0, 32))
            canvas.paste(Image.fromarray(np.asarray(wrist)), (224, 32))
            text = (
                f"episode={episode_index} query={query_index} "
                f"t={query['observation_time_s']:.3f}s ext|wrist"
            )
            ImageDraw.Draw(canvas).text((4, 8), text, fill="white")
            panels.append(np.asarray(canvas))
    preview = np.stack(panels)
    mp4 = work / "rollouts.mp4"
    encode_video(preview, mp4, max(1, round(CONTROL_HZ / EXECUTION_PREFIX)))
    if not mp4.is_file() or mp4.stat().st_size == 0:
        raise RuntimeError("closed-loop MP4 readback is empty")
    return mp4, len(preview)


def _write_rrd(
    work: Path, episodes: list[dict[str, Any]], role: str
) -> tuple[Path, int]:
    import rerun as rr

    rrd = work / "rollouts.rrd"
    step = 0
    with rr.RecordingStream(f"pi05-{role}") as recording:
        recording.save(rrd)
        for episode_index, episode in enumerate(episodes):
            step += _log_rrd_physics(recording, rr, episode, episode_index)
            _log_rrd_queries(recording, rr, episode, episode_index)
    if not rrd.is_file() or rrd.stat().st_size == 0 or step < 1:
        raise RuntimeError("closed-loop Rerun readback is empty")
    return rrd, step


def _rrd_time(
    recording: Any, episode_index: int, index_name: str, index: int, time_s: float
) -> None:
    import math

    if not math.isfinite(time_s) or time_s < 0.0:
        raise RuntimeError(f"Rerun {index_name} simulator time is invalid")
    recording.reset_time()
    recording.set_time("episode", sequence=episode_index)
    recording.set_time(index_name, sequence=index)
    recording.set_time("sim_time", duration=time_s)


def _physics_rows(episode: dict) -> list[tuple[dict, float]]:
    trace = list(episode.get("physics_trace", []))
    proofs = [
        proof
        for query in episode.get("queries", [])
        for proof in query.get("controller_proofs", [])
    ]
    recorded = [
        proof
        for proof in proofs
        if proof.get("physics_observation_status") == "recorded"
    ]
    if any(
        proof.get("physics_observation_status") not in {"recorded", "failed"}
        for proof in proofs
    ):
        raise RuntimeError("Rerun controller proof lacks physics observation status")
    if len(trace) != len(recorded):
        raise RuntimeError("Rerun physics rows do not match recorded controller proofs")
    rows = []
    previous = -1.0
    for index, (sample, proof) in enumerate(zip(trace, recorded)):
        command = proof.get("command_application_time_s")
        result = proof.get("resulting_state_time_s")
        if not all(isinstance(value, (int, float)) for value in (command, result)):
            raise RuntimeError("Rerun controller proof lacks simulator times")
        if not previous <= float(command) < float(result):
            raise RuntimeError("Rerun controller proof times are not ordered")
        if proof.get("physics_trace_index") != index:
            raise RuntimeError("Rerun physics trace index binding is invalid")
        if sample.get("simulator_time_s") != result:
            raise RuntimeError(
                "Rerun physics sample time differs from controller result"
            )
        previous = float(result)
        rows.append((sample, previous))
    return rows


def _log_rrd_physics(recording: Any, rr: Any, episode: dict, episode_index: int) -> int:
    rows = _physics_rows(episode)
    for step, (sample, result_time) in enumerate(rows):
        _rrd_time(recording, episode_index, "physics_step", step, result_time)
        root = f"episodes/{episode_index}/physics/{step}"
        recording.log(
            root + "/object_position", rr.Points3D([sample["object_position_m"]])
        )
        recording.log(
            root + "/object_speed_m_s", rr.Scalars([sample["object_speed_m_s"]])
        )
        recording.log(
            root + "/target_xy_distance_m", rr.Scalars([sample["target_xy_distance_m"]])
        )
    return len(rows)


def _log_rrd_queries(
    recording: Any, rr: Any, episode: dict, episode_index: int
) -> None:
    queries = list(episode.get("queries", []))
    exterior = episode.get("_exterior_frames", [])
    wrist = episode.get("_wrist_frames", [])
    if len(exterior) != len(queries) or len(wrist) != len(queries):
        raise RuntimeError("Rerun camera frames do not match policy queries")
    for query_index, query in enumerate(queries):
        root = f"episodes/{episode_index}/queries/{query_index}"
        _rrd_time(
            recording,
            episode_index,
            "query",
            query_index,
            float(query["exterior_capture_time_s"]),
        )
        recording.log(root + "/exterior", rr.Image(exterior[query_index]))
        _rrd_time(
            recording,
            episode_index,
            "query",
            query_index,
            float(query["wrist_capture_time_s"]),
        )
        recording.log(root + "/wrist", rr.Image(wrist[query_index]))
        _rrd_time(
            recording,
            episode_index,
            "query",
            query_index,
            float(query["observation_time_s"]),
        )
        recording.log(
            root + "/capture_age_s",
            rr.Scalars([query["inference"]["wire_input"]["capture_age_s"]]),
        )


def _decode_artifacts(mp4: Path, rrd: Path) -> tuple[dict[str, Any], str]:
    import subprocess
    import sys

    rerun = [sys.executable, "-m", "rerun"]
    subprocess.run([*rerun, "rrd", "verify", str(rrd)], check=True)
    decoded = subprocess.run(
        [*rerun, "rrd", "print", "-v", str(rrd)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "stream=codec_name,nb_frames,r_frame_rate",
        "-of",
        "json",
        str(mp4),
    ]
    probe = json.loads(
        subprocess.run(command, check=True, capture_output=True, text=True).stdout
    )
    if not probe.get("streams") or "object_position" not in decoded:
        raise RuntimeError(
            "review artifact independent decode lacks expected structure"
        )
    return probe, decoded


def _write_review_artifacts(
    work: Path, episodes: list[dict[str, Any]], checkpoint_role: str
) -> dict[str, Any]:
    failed_cameras = _write_failed_camera_evidence(work, episodes)
    mp4, frames = _write_video(work, episodes)
    rrd, steps = _write_rrd(work, episodes, checkpoint_role)
    probe, _ = _decode_artifacts(mp4, rrd)
    for episode in episodes:
        episode.pop("_exterior_frames", None)
        episode.pop("_wrist_frames", None)
    return {
        "failed_camera_observations": failed_cameras,
        "mp4": {
            "path": mp4.name,
            "bytes": mp4.stat().st_size,
            "frames": frames,
            "ffprobe": probe["streams"],
        },
        "rrd": {
            "path": rrd.name,
            "bytes": rrd.stat().st_size,
            "timeline_rows": steps,
            "rerun_verify": True,
            "decoded_expected_entity": True,
        },
    }


def _persist_partial_episodes(work: Path, episodes: list[dict[str, Any]]) -> None:
    _write_failed_camera_evidence(work, episodes)
    receipts = [
        {
            key: value
            for key, value in episode.items()
            if key not in {"_exterior_frames", "_wrist_frames"}
        }
        for episode in episodes
    ]
    (work / "partial-episodes.json").write_text(
        json.dumps(receipts, indent=2, sort_keys=True, default=_json_value) + "\n"
    )


def _json_value(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"unsupported partial-evidence value: {type(value).__name__}")


def _sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _load_gold_protocol(
    args: argparse.Namespace,
    work: Path,
    preparation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from npa.workflows.sim2real.workflow_io import read_json

    collection = read_json(args.gold_collection_uri, directory=work)
    if collection.get("schema") != "npa.sim2real.pi05.expert_collection.v1":
        raise RuntimeError("gold collection has the wrong task schema")
    asset = collection.get("object_asset") or {}
    geometry = asset.get("geometry") or {}
    support = collection.get("support_surface_contract") or {}
    if support != support_surface_contract():
        raise RuntimeError("gold collection support geometry differs from evaluator")
    scale = _vector(asset.get("scale"), 3, "gold object scale")
    scenarios = collection.get("successful_scenarios") or []
    if len(scenarios) < args.episodes:
        raise RuntimeError("gold collection lacks enough physics-qualified scenarios")
    scenarios = scenarios[: args.episodes]
    _validate_gold_scenarios(scenarios, args.seed)
    robot_plant = _validated_robot_plant(collection.get("robot_plant"))
    protocol = {
        "task": "released_supported_surface_placement_v1",
        "split": args.split,
        "seed": args.seed,
        "episodes": args.episodes,
        "decisions": args.decisions,
        **_protocol_object_fields(asset, geometry, scale),
        "scene_id": collection.get("scene_id"),
        "destination_marker_contract": collection.get("destination_marker_contract"),
        "support_surface_contract": support,
        "robot_plant": robot_plant,
        "control_hz": CONTROL_HZ,
        "execution_prefix": EXECUTION_PREFIX,
        "action_selection": _action_selection_protocol(),
        "scenarios": scenarios,
    }
    if not protocol["object_identity"] or not protocol["scene_id"]:
        raise RuntimeError("gold collection lacks object or scene identity")
    _validate_object_source(protocol["object_usd"])
    runtime_object = None
    if preparation is not None:
        binding, runtime_object = _resolve_interchange_object(
            collection, asset, protocol, preparation, work
        )
        protocol["object_binding"] = binding
    _bind_protocol_hashes(protocol)
    if runtime_object is not None:
        protocol["_runtime_object_usd"] = str(runtime_object)
    return protocol


def _resolve_interchange_object(
    collection: dict[str, Any],
    asset: dict[str, Any],
    protocol: dict[str, Any],
    preparation: dict[str, Any],
    work: Path,
) -> tuple[dict[str, Any], Path | None]:
    values = [asset.get(field) for field in OBJECT_SOURCE_FIELDS]
    if any(values) and not all(values):
        raise RuntimeError("gold collection has partial object source provenance")
    if all(values):
        prepared = materialize_prepared_object(preparation, work / "runtime-object")
        assert_collection_object_binding(asset, prepared.profile)
        _validate_prepared_object_geometry(protocol, prepared.profile)
        return {
            "authority": "prepared_profile",
            "profile_identity": prepared.profile.identity,
        }, prepared.local_path
    return _legacy_collection_object_binding(collection, asset), None


def _legacy_collection_object_binding(
    collection: dict[str, Any], asset: dict[str, Any]
) -> dict[str, Any]:
    expected = {
        "family": "isaac-multicolor-cube",
        "identity": "isaac-multicolor-cube-072mm",
        "usd": "task_default",
    }
    if any(asset.get(key) != value for key, value in expected.items()):
        raise RuntimeError("legacy gold collection object is not the canonical cube")
    contract = collection.get("task_contract") or {}
    assert_contract(contract)
    current = build_contract()
    if (
        contract.get("schema") != PI05_TASK_SCHEMA
        or contract.get("task_id") != PI05_TASK_ID
        or contract.get("success") != current.get("success")
    ):
        raise RuntimeError(
            "legacy gold task or success contract differs from evaluator"
        )
    return {
        "authority": "sealed_legacy_collection",
        "runtime_asset": LEGACY_DEFAULT_OBJECT_ASSET,
        "collection_contract_sha256": contract["contract_sha256"],
        "source_hashes_available": False,
    }


def _protocol_object_fields(
    asset: dict[str, Any], geometry: dict[str, Any], scale: list[float]
) -> dict[str, Any]:
    result = {
        "object_identity": asset.get("identity"),
        "object_usd": asset.get("usd"),
        "object_scale": scale,
        "object_width_m": _measured_dimension(geometry, "grasp_width_m", 0.08),
        "object_height_m": _measured_dimension(geometry, "support_height_m", 0.25),
        "object_geometry": _validated_geometry_receipt(geometry, scale),
    }
    if asset.get("profile_identity"):
        result.update(
            {
                "object_profile_identity": asset.get("profile_identity"),
                "object_source_sha256": asset.get("source_sha256"),
                "object_bundle_sha256": asset.get("bundle_sha256"),
            }
        )
    return result


def _validate_prepared_object_geometry(protocol: dict[str, Any], profile: Any) -> None:
    """Require prepared gold-scale geometry to match the composed gold asset."""

    expected_scale = [float(value) for value in profile.scale_xyz]
    if expected_scale != protocol["object_scale"]:
        raise RuntimeError("prepared object scale differs from the gold protocol")
    axis = ("x", "y", "z").index(profile.grasp_axis)
    if abs(profile.grasp_extents_m[axis] - protocol["object_width_m"]) > 1.0e-6:
        raise RuntimeError("prepared object grasp extent differs from gold geometry")
    if abs(profile.grasp_extents_m[2] - protocol["object_height_m"]) > 1.0e-6:
        raise RuntimeError("prepared object height differs from gold geometry")


def _validate_object_source(value: Any) -> None:
    if not isinstance(value, str) or not value:
        raise RuntimeError("gold collection lacks the exact object USD source")


def _validated_robot_plant(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError("gold collection lacks the robot plant receipt")
    if value.get("schema") != "npa.sim2real.pi05.robot_plant.v1":
        raise RuntimeError("gold collection robot plant schema is invalid")
    if value.get("source") != "isaaclab_assets.FRANKA_PANDA_HIGH_PD_CFG":
        raise RuntimeError("gold collection uses an unsupported Franka plant")
    gravity = value.get("observed_disable_gravity") or {}
    if (
        value.get("configured_disable_robot_gravity") is not True
        or gravity.get("all_bodies_disabled") is not True
        or not value.get("actuators")
    ):
        raise RuntimeError("gold collection robot plant is incomplete")
    return value


def _validate_gold_scenarios(scenarios: list[dict[str, Any]], seed: int) -> None:
    for index, scenario in enumerate(scenarios):
        if int(scenario.get("scene_seed", -1)) != seed + index:
            raise RuntimeError("gold scenario seed does not match evaluation protocol")
        _vector(scenario.get("initial_object_position_m"), 3, "initial object")
        _vector(scenario.get("target_position_m"), 3, "surface target")
        marker = scenario.get("destination_marker") or {}
        if (
            marker.get("semantic_label") != "pi05_destination_marker"
            or marker.get("physics_collision") is not False
        ):
            raise RuntimeError("gold scenario lacks the rendered destination contract")
        if scenario.get("support_surface") != support_surface_contract():
            raise RuntimeError(
                "gold scenario lacks the exact one-body support contract"
            )


def _measured_dimension(geometry: dict[str, Any], name: str, maximum: float) -> float:
    value = geometry.get(name)
    if not isinstance(value, (int, float)) or not 0.01 <= float(value) <= maximum:
        raise RuntimeError(f"gold collection lacks valid measured {name}")
    return float(value)


def _validated_geometry_receipt(
    geometry: dict[str, Any], scale: list[float]
) -> dict[str, Any]:
    vectors = {
        name: _vector(geometry.get(name), 3, f"gold {name}")
        for name in (
            "source_untransformed_extent_m",
            "effective_local_extent_m",
            "source_untransformed_center_m",
            "declared_spawn_scale",
        )
    }
    if vectors["declared_spawn_scale"] != scale:
        raise RuntimeError("gold geometry scale differs from object asset scale")
    strings = {
        "grasp_width_axis": "object_local_y",
        "measurement": "UsdGeom.BBoxCache composed runtime prim",
    }
    if any(geometry.get(key) != value for key, value in strings.items()):
        raise RuntimeError("gold geometry measurement provenance is invalid")
    return {
        **geometry,
        **vectors,
        "grasp_width_m": _measured_dimension(geometry, "grasp_width_m", 0.08),
        "support_height_m": _measured_dimension(geometry, "support_height_m", 0.25),
    }


def _bind_protocol_hashes(protocol: dict[str, Any]) -> None:
    protocol["sha256"] = _sha(protocol)
    protocol["scenario_sha256"] = [
        _sha({"protocol": protocol["sha256"], "seed": protocol["seed"] + index})
        for index in range(protocol["episodes"])
    ]


def _configure_eval(
    cfg: Any, args: argparse.Namespace, protocol: dict[str, Any]
) -> None:
    from npa.workflows.sim2real.isaac_assets_compat import remap_moved_franka_usd

    cfg.seed, cfg.sim.dt, cfg.decimation = args.seed, 1.0 / 60.0, 4
    cfg.episode_length_s = 240.0
    configure_robot_plant(cfg)
    remap_moved_franka_usd(cfg)
    _configure_eval_object(cfg, protocol)
    configure_object_reset(cfg)
    configure_finger_semantics(cfg)
    configure_hand_origin_frames(cfg, require_ik=False)
    target = tuple(protocol["scenarios"][0]["target_position_m"])
    if any(
        scenario["target_position_m"] != list(target)
        for scenario in protocol["scenarios"]
    ):
        raise RuntimeError("gold scenarios do not share one declared surface target")
    configure_support_surface(cfg)
    configure_surface_goal(cfg, target)
    configure_destination_marker(cfg, target)
    _camera_config(cfg)
    _contact_config(cfg)
    _physics_randomization(cfg)


def _configure_eval_object(cfg: Any, protocol: dict[str, Any]) -> None:
    runtime_source = protocol.get("_runtime_object_usd") or protocol["object_usd"]
    if runtime_source != "task_default":
        cfg.scene.object.spawn.usd_path = runtime_source
    cfg.scene.object.spawn.scale = tuple(protocol["object_scale"])
    cfg.scene.object.spawn.semantic_tags = [("class", "pi05_target_object")]


def _validate_eval_geometry(native: Any, protocol: dict[str, Any]) -> dict[str, Any]:
    """Remeasure the composed evaluator asset against collection provenance."""

    import numpy as np

    actual = _object_asset_geometry(native, tuple(protocol["object_scale"]))
    expected = protocol["object_geometry"]
    vector_keys = (
        "source_untransformed_extent_m",
        "effective_local_extent_m",
        "source_untransformed_center_m",
        "declared_spawn_scale",
    )
    if any(
        not np.allclose(actual.get(key), expected.get(key), atol=1.0e-6, rtol=0.0)
        for key in vector_keys
    ):
        raise RuntimeError("evaluator object geometry differs from gold collection")
    scalar_keys = ("grasp_width_m", "support_height_m")
    if any(
        abs(float(actual[key]) - float(expected[key])) > 1.0e-6 for key in scalar_keys
    ):
        raise RuntimeError("evaluator object dimensions differ from gold collection")
    string_keys = ("grasp_width_axis", "measurement")
    if any(actual.get(key) != expected.get(key) for key in string_keys):
        raise RuntimeError("evaluator object provenance differs from gold collection")
    return actual


def _validate_eval_object_source(native: Any, protocol: dict[str, Any]) -> str:
    spawn = native.scene["object"].cfg.spawn
    actual = str(getattr(spawn, "usd_path", "") or "")
    binding = protocol.get("object_binding") or {}
    if binding.get("authority") == "sealed_legacy_collection":
        expected = str(binding.get("runtime_asset") or "")
        if not expected or not actual.endswith(expected):
            raise RuntimeError("evaluator legacy object asset differs from collection")
    elif binding.get("authority") == "prepared_profile":
        expected = str(protocol.get("_runtime_object_usd") or "")
        if not expected or actual != expected:
            raise RuntimeError("evaluator prepared object source differs from profile")
    return actual


def _validate_eval_robot_plant(native: Any, protocol: dict[str, Any]) -> dict[str, Any]:
    """Require evaluator runtime dynamics to match the expert collection."""

    actual = _robot_plant_receipt(native)
    if actual != protocol["robot_plant"]:
        raise RuntimeError("evaluator robot plant differs from gold collection")
    return actual


def _evaluate(
    env: Any,
    client: Pi05PolicyClient,
    args: argparse.Namespace,
    protocol: dict[str, Any],
    *,
    robot_profile: Any = None,
    policy_profile: Any = None,
) -> list[dict[str, Any]]:
    episodes = []
    for mode in args.wrist_ablation_modes:
        for index in range(args.episodes):
            env.reset(seed=args.seed + index)
            record = _new_episode_record()
            try:
                result = _episode(
                    env,
                    client,
                    args.decisions,
                    record,
                    protocol["scenarios"][index],
                    mode,
                    args.ablation_seed + index,
                    float(protocol["object_width_m"]),
                    **(
                        {
                            "robot_profile": robot_profile,
                            "policy_profile": policy_profile,
                        }
                        if robot_profile is not None or policy_profile is not None
                        else {}
                    ),
                )
            except Exception as exc:
                record.update(
                    {
                        "wrist_ablation_mode": mode,
                        "success": {"released_supported_surface_placement": False},
                        "failure": _execution_error(exc),
                        "partial_rollout_preserved": bool(
                            record.get("_exterior_frames")
                            or record.get("_wrist_frames")
                            or record.get("physics_trace")
                        ),
                    }
                )
                result = record
            episodes.append(result)
    return episodes


def _checkpoint_provenance(receipt: dict[str, Any]) -> dict[str, Any]:
    if receipt.get("schema") == "npa.sim2real.smolvla_service_start.v1":
        interchange = receipt.get("interchange") or {}
        result = {
            "source_commit": receipt.get("source_commit"),
            "config_name": interchange.get("policy_identity"),
            "checkpoint_role": receipt.get("checkpoint_role"),
            "checkpoint_sha256": interchange.get("checkpoint_sha256"),
            "checkpoint_manifest_sha256": interchange.get("manifest_sha256"),
            "normalization_identity": interchange.get("normalization_identity"),
            "image_digest": receipt.get("runtime_image"),
        }
        if any(not result.get(key) for key in result):
            raise RuntimeError("SmolVLA service receipt lacks immutable provenance")
        return result
    keys = (
        "source_commit",
        "config_name",
        "base_config_name",
        "selected_policy",
        "checkpoint_role",
        "checkpoint_sha256",
        "checkpoint_manifest_sha256",
        "image_digest",
    )
    result = {key: receipt.get(key) for key in keys if receipt.get(key)}
    if not result.get("source_commit") or not result.get("image_digest"):
        raise RuntimeError("service receipt lacks immutable source/image provenance")
    return result


def _result(
    args: argparse.Namespace,
    episodes: list[dict[str, Any]],
    artifacts: dict[str, Any],
    protocol: dict[str, Any],
    receipt: dict[str, Any],
) -> dict[str, Any]:
    passes = sum(_episode_passed(row) for row in episodes)
    grouped = _ablation_groups(episodes, args.wrist_ablation_modes)
    policy_backend = getattr(args, "policy_backend", "openpi_pi05")
    if policy_backend == "lerobot_smolvla":
        noise_pairing = {
            "sampler_noise_identity_verified": False,
            "sampler_noise_protocol": "unavailable_for_native_smolvla",
            "paired_trial_protocol": "independently_seeded_repeated_matched_scenarios",
            "paired_trial_seeds": sorted(
                {int(row["scenario_seed"]) for row in episodes}
            ),
        }
        qualification = "matched_repeated_trials_without_noise_coupling"
        schema = "npa.sim2real.policy_interchange.closed_loop_eval.v1"
        comparison = "same surface task and physical controller semantics only"
    else:
        noise_pairing = _sampler_noise_pairing(episodes, args.wrist_ablation_modes)
        qualification = (
            "matched_effectiveness_eligible"
            if noise_pairing["sampler_noise_identity_verified"]
            else "effectiveness_claim_rejected_without_sampler_noise_identity"
        )
        schema = "npa.sim2real.pi05.closed_loop_eval.v1"
        comparison = "same pi0.5 surface task only"
    public_protocol = {
        key: value for key, value in protocol.items() if not key.startswith("_")
    }
    return {
        "schema": schema,
        "policy_backend": policy_backend,
        "checkpoint_role": args.checkpoint_role,
        "split": args.split,
        "checkpoint_provenance": _checkpoint_provenance(receipt),
        "protocol": public_protocol,
        "episode_count": len(episodes),
        "released_surface_placements": passes,
        "success_rate": passes / len(episodes),
        "wrist_ablation": {
            "modes": args.wrist_ablation_modes,
            "same_checkpoint_protocol_scenarios_and_reset_seeds": True,
            "same_initial_object_poses": True,
            **noise_pairing,
            "per_mode": grouped,
            "qualification": qualification,
        },
        "task_comparison": comparison,
        "physical_robot_deployment": False,
        "episodes": episodes,
        "review_artifacts": artifacts,
    }


def _ablation_groups(
    episodes: list[dict[str, Any]], modes: list[str]
) -> dict[str, dict[str, Any]]:
    result = {}
    for mode in modes:
        rows = [row for row in episodes if row["wrist_ablation_mode"] == mode]
        if not rows:
            continue
        passes = sum(_episode_passed(row) for row in rows)
        result[mode] = {
            "episodes": len(rows),
            "released_surface_placements": passes,
            "success_rate": passes / len(rows),
        }
    return result


def _episode_passed(row: dict[str, Any]) -> bool:
    return bool(row["success"].get("released_supported_surface_placement"))


def _noise_identities(
    episodes: list[dict],
) -> tuple[dict[tuple[int, int], dict[str, str | None]], bool]:
    identities: dict[tuple[int, int], dict[str, str | None]] = {}
    duplicate = False
    for episode in episodes:
        for query in episode.get("queries", []):
            key = (int(episode["scenario_seed"]), int(query["decision"]))
            by_mode = identities.setdefault(key, {})
            mode = str(episode["wrist_ablation_mode"])
            duplicate = duplicate or mode in by_mode
            by_mode[mode] = query["inference"].get("sampler_noise_sha256")
    return identities, duplicate


def _sampler_noise_pairing(episodes: list[dict], modes: list[str]) -> dict[str, Any]:
    identities, duplicate = _noise_identities(episodes)
    expected = set(modes)
    noise_complete = bool(identities) and not duplicate
    noise_complete = noise_complete and all(
        set(by_mode) == expected
        and all(_valid_receipt_hash(value) for value in by_mode.values())
        and len(set(by_mode.values())) == 1
        for by_mode in identities.values()
    )
    execution_complete = bool(episodes) and all(
        episode.get("queries")
        and not episode.get("failure")
        and all(_query_execution_complete(query) for query in episode["queries"])
        for episode in episodes
    )
    complete = noise_complete and execution_complete
    return {
        "sampling_noise_controlled": complete,
        "sampler_noise_identity_verified": complete,
        "controller_execution_complete": execution_complete,
        "sampler_noise_identity_receipt_count": sum(
            value is not None
            for by_mode in identities.values()
            for value in by_mode.values()
        ),
        "sampling_noise_limitation": (
            None
            if complete
            else "paired noise or complete controller execution evidence is absent"
        ),
        "same_state_action_sensitivity_only": True,
        "closed_loop_states_and_exterior_diverge_after_different_actions": True,
    }


def _query_execution_complete(query: dict[str, Any]) -> bool:
    proofs = query.get("controller_proofs") or []
    return bool(
        query.get("execution_status") == "completed"
        and query.get("attempted_prefix") == EXECUTION_PREFIX
        and query.get("executed_prefix") == EXECUTION_PREFIX
        and len(proofs) == EXECUTION_PREFIX
        and all(
            proof.get("physics_observation_status") == "recorded" for proof in proofs
        )
        and not any(
            proof.get("terminated") or proof.get("truncated") for proof in proofs
        )
    )


def _valid_receipt_hash(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_wrist_interventions(episodes: list[dict[str, Any]]) -> None:
    for episode in episodes:
        mode = episode.get("wrist_ablation_mode")
        frozen_hash = None
        for query in episode.get("queries") or []:
            intervention = (query.get("inference") or {}).get(
                "wrist_intervention"
            ) or {}
            source = intervention.get("source_sha256")
            policy = intervention.get("policy_sha256")
            if (
                intervention.get("mode") != mode
                or intervention.get("decision") != query.get("decision")
                or not _valid_receipt_hash(source)
                or not _valid_receipt_hash(policy)
                or (mode == "fresh" and source != policy)
                or (mode == "pixel_channel_shuffle" and source == policy)
            ):
                raise RuntimeError("wrist intervention receipt is invalid")
            if mode == "frozen":
                frozen_hash = policy if frozen_hash is None else frozen_hash
                if policy != frozen_hash:
                    raise RuntimeError(
                        "frozen wrist intervention changed within episode"
                    )


def _validate_query_lineage(episode: dict[str, Any], query: dict[str, Any]) -> None:
    inference = query.get("inference") or {}
    intervention = inference.get("wrist_intervention") or {}
    wire = inference.get("wire_input") or {}
    transform = inference.get("model_wrist_transform") or {}
    seed, decision = int(episode["scenario_seed"]), int(query["decision"])
    expected_sequence = f"scene-{seed}-decision-{decision}"
    hashes = (
        wire.get("payload_sha256"),
        wire.get("exterior_rgb_sha256"),
        wire.get("wrist_rgb_sha256"),
        transform.get("left_wrist_0_rgb_sha256"),
        transform.get("right_wrist_0_rgb_sha256"),
    )
    if (
        not all(_valid_receipt_hash(value) for value in hashes)
        or query.get("wrist_rgb_sha256") != intervention.get("source_sha256")
        or query.get("exterior_rgb_sha256") != wire.get("exterior_rgb_sha256")
        or wire.get("wrist_rgb_sha256") != intervention.get("policy_sha256")
        or wire.get("wrist_source_hash_matched") is not True
        or transform.get("schema") != "npa.openpi.pi05-live-wrist-transform.v1"
        or transform.get("source_wrist_sha256") != intervention.get("policy_sha256")
        or transform.get("left_wrist_mask") is not True
        or transform.get("right_wrist_mask") is not False
        or inference.get("sampling_sequence") != expected_sequence
    ):
        raise RuntimeError("policy query wrist lineage is invalid")
    _validate_query_numeric_receipts(query, inference, wire)


def _validate_query_numeric_receipts(
    query: dict[str, Any], inference: dict[str, Any], wire: dict[str, Any]
) -> None:
    import math

    shape = inference.get("sampler_noise_shape")
    age = wire.get("capture_age_s")
    action = query.get("action_receipt") or {}
    selection = query.get("action_selection") or {}
    horizon = inference.get("shape")
    if (
        shape != PI05_SAMPLER_NOISE_SHAPE
        or not isinstance(age, (int, float))
        or not math.isfinite(float(age))
        or not 0.0 <= float(age) <= 1.0 / CONTROL_HZ
        or inference.get("dtype") != "float64"
        or inference.get("finite") is not True
        or not isinstance(horizon, list)
        or len(horizon) != 2
        or horizon[0] < ACTION_HORIZON
        or horizon[1] != 8
        or action.get("shape") != horizon
        or not _valid_action_chunk_receipt(action)
        or not _valid_action_selection_receipt(selection, query.get("decision"))
    ):
        raise RuntimeError("policy query numeric receipt is invalid")


def _valid_action_chunk_receipt(receipt: dict[str, Any]) -> bool:
    import numpy as np

    try:
        actions = np.asarray(receipt.get("actions"), dtype=np.float64)
    except (TypeError, ValueError):
        return False
    value = np.ascontiguousarray(actions)
    return bool(
        receipt.get("shape") == [ACTION_HORIZON, 8]
        and receipt.get("dtype") == "float64"
        and value.shape == (ACTION_HORIZON, 8)
        and np.isfinite(value).all()
        and receipt.get("sha256") == hashlib.sha256(value.tobytes()).hexdigest()
        and receipt.get("gripper_scores") == value[:, 7].tolist()
        and receipt.get("thresholded_gripper_commands")
        == [int(score > 0.5) for score in value[:, 7]]
    )


def _valid_action_selection_receipt(selection: dict, decision: Any) -> bool:
    import numpy as np

    try:
        applied = np.asarray(selection.get("applied_action"), dtype=np.float64)
    except (TypeError, ValueError):
        return False
    digest = hashlib.sha256(np.ascontiguousarray(applied).tobytes()).hexdigest()
    score = selection.get("applied_gripper_score")
    return bool(
        selection.get("schema") == _action_selection_protocol()["schema"]
        and selection.get("method") == "uniform_mean"
        and selection.get("model_horizon") == ACTION_HORIZON
        and selection.get("fresh_observation_per_control") is True
        and selection.get("causality") == "current_and_past_predictions_only"
        and selection.get("decision") == decision
        and selection.get("applied_action_shape") == [8]
        and selection.get("applied_action_dtype") == "float64"
        and applied.shape == (8,)
        and np.isfinite(applied).all()
        and selection.get("applied_action_sha256") == digest
        and isinstance(score, (int, float))
        and float(score) == float(applied[7])
        and selection.get("applied_gripper_command") == int(float(score) > 0.5)
    )


def _validate_action_selection_history(episode: dict[str, Any]) -> None:
    import numpy as np

    queries = episode.get("queries") or []
    receipts = {
        int(query["decision"]): query.get("action_receipt") or {} for query in queries
    }
    for query in queries:
        decision = int(query["decision"])
        selection = query.get("action_selection") or {}
        contributors = selection.get("contributors") or []
        origins = list(range(max(0, decision - ACTION_HORIZON + 1), decision + 1))
        if (
            selection.get("contributor_count") != len(origins)
            or len(contributors) != len(origins)
            or selection.get("weight_per_contributor") != 1.0 / len(origins)
        ):
            raise RuntimeError("temporal action ensemble count is invalid")
        for contributor, origin in zip(contributors, origins):
            if (
                contributor.get("origin_decision") != origin
                or contributor.get("action_index") != decision - origin
                or contributor.get("chunk_sha256")
                != receipts.get(origin, {}).get("sha256")
                or not _valid_receipt_hash(contributor.get("chunk_sha256"))
            ):
                raise RuntimeError("temporal action ensemble lineage is invalid")
        candidates = [
            np.asarray(receipts[origin]["actions"], dtype=np.float64)[decision - origin]
            for origin in origins
        ]
        expected = np.ascontiguousarray(np.mean(np.stack(candidates), axis=0))
        applied = np.asarray(selection["applied_action"], dtype=np.float64)
        if not np.array_equal(applied, expected):
            raise RuntimeError("temporal action ensemble value is invalid")


def _validate_camera_evidence(episode: dict[str, Any]) -> None:
    import math

    evidence = episode.get("camera_evidence") or {}
    visibility = evidence.get("phase_visibility") or {}
    if any(
        not isinstance(evidence.get(name), (int, float))
        or not math.isfinite(float(evidence[name]))
        for name in CAMERA_MOTION_METRICS
    ):
        raise RuntimeError("camera motion metric is nonfinite or absent")
    _validate_camera_motion_metrics(evidence)
    samples = 0
    for phase in visibility.values():
        count = phase.get("samples")
        counters = (
            phase.get("wrist_visible_samples"),
            phase.get("exterior_visible_samples"),
            phase.get("destination_exterior_visible_samples"),
            phase.get("both_fingers_rendered_samples"),
        )
        if not isinstance(count, int) or count < 1:
            raise RuntimeError("camera phase evidence is invalid")
        if any(
            not isinstance(value, int) or not 0 <= value <= count for value in counters
        ):
            raise RuntimeError("camera phase visibility count is invalid")
        if phase.get("destination_exterior_visible_samples") != count and not phase.get(
            "occlusion_permitted", False
        ):
            raise RuntimeError("destination marker was absent at a policy query")
        samples += count
    if not visibility or samples != len(episode.get("queries") or []):
        raise RuntimeError("camera evidence does not bind every policy query")
    success = _success(episode.get("physics_trace") or [])
    if (
        episode.get("wrist_ablation_mode") == "fresh"
        and success["released_supported_surface_placement"]
    ):
        _require_expert_phase_visibility(visibility)


def _validate_effectiveness_cohort(episodes: list[dict[str, Any]]) -> None:
    keys = [
        (episode.get("scenario_seed"), episode.get("wrist_ablation_mode"))
        for episode in episodes
    ]
    if len(set(keys)) != len(keys):
        raise RuntimeError("wrist evaluation contains duplicate scenario modes")
    seed_sets = [
        {
            episode.get("scenario_seed")
            for episode in episodes
            if episode.get("wrist_ablation_mode") == mode
        }
        for mode in WRIST_ABLATION_MODES
    ]
    if not seed_sets[0] or any(seeds != seed_sets[0] for seeds in seed_sets[1:]):
        raise RuntimeError("wrist evaluation scenario cohorts are not paired")
    for episode in episodes:
        _validate_camera_evidence(episode)
        for query in episode.get("queries") or []:
            _validate_query_lineage(episode, query)
        _validate_action_selection_history(episode)
        _physics_rows(episode)


def _validate_effectiveness_artifacts(
    artifacts: dict[str, Any], episodes: list[dict[str, Any]]
) -> None:
    mp4, rrd = artifacts.get("mp4") or {}, artifacts.get("rrd") or {}
    query_count = sum(len(episode.get("queries") or []) for episode in episodes)
    physics_count = sum(len(episode.get("physics_trace") or []) for episode in episodes)
    if (
        int(mp4.get("bytes", 0)) < 1
        or int(mp4.get("frames", 0)) != query_count
        or not mp4.get("ffprobe")
        or int(rrd.get("bytes", 0)) < 1
        or int(rrd.get("timeline_rows", 0)) != physics_count
        or rrd.get("rerun_verify") is not True
        or rrd.get("decoded_expected_entity") is not True
    ):
        raise RuntimeError("wrist evaluation review artifacts are incomplete")


def _validate_effectiveness_summaries(
    report: dict[str, Any], episodes: list[dict[str, Any]], ablation: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    for episode in episodes:
        if episode.get("success") != _success(episode.get("physics_trace") or []):
            raise RuntimeError("wrist evaluation success differs from physics trace")
    grouped = _ablation_groups(episodes, list(WRIST_ABLATION_MODES))
    passes = sum(_episode_passed(episode) for episode in episodes)
    if (
        ablation.get("per_mode") != grouped
        or report.get("released_surface_placements") != passes
        or report.get("success_rate") != passes / len(episodes)
    ):
        raise RuntimeError("wrist evaluation summaries differ from episodes")
    pairing = _sampler_noise_pairing(episodes, list(WRIST_ABLATION_MODES))
    for key, value in pairing.items():
        if ablation.get(key) != value:
            raise RuntimeError("wrist evaluation pairing summary is invalid")
    if not pairing["sampler_noise_identity_verified"]:
        raise RuntimeError("wrist evaluation lacks paired execution evidence")
    return grouped


def _validate_checkpoint_provenance(report: dict[str, Any], role: str) -> None:
    provenance = report.get("checkpoint_provenance") or {}
    image = str(provenance.get("image_digest", "")).removeprefix("sha256:")
    source = str(provenance.get("source_commit", ""))
    if (
        provenance.get("checkpoint_role") != role
        or not _valid_receipt_hash(provenance.get("checkpoint_sha256"))
        or not _valid_receipt_hash(provenance.get("checkpoint_manifest_sha256"))
        or not _valid_receipt_hash(image)
        or len(source) != 40
        or any(character not in "0123456789abcdef" for character in source)
        or not provenance.get("config_name")
    ):
        raise RuntimeError("closed-loop checkpoint provenance is invalid")


def _expected_scenario_hashes(protocol: dict[str, Any]) -> list[str]:
    seed, count = protocol.get("seed"), protocol.get("episodes")
    digest = protocol.get("sha256")
    if not isinstance(seed, int) or not isinstance(count, int) or count < 1:
        return []
    return [_sha({"protocol": digest, "seed": seed + index}) for index in range(count)]


def _validate_effectiveness_protocol(report: dict[str, Any], role: str) -> None:
    protocol = report.get("protocol") or {}
    scenarios = protocol.get("scenario_sha256") or []
    payload = {
        key: value
        for key, value in protocol.items()
        if key not in {"sha256", "scenario_sha256"}
    }
    expected_scenarios = _expected_scenario_hashes(protocol)
    if (
        not EFFECTIVENESS_PROTOCOL_FIELDS.issubset(payload)
        or protocol.get("task") != "released_supported_surface_placement_v1"
        or protocol.get("split") != "gold"
        or protocol.get("control_hz") != CONTROL_HZ
        or protocol.get("execution_prefix") != EXECUTION_PREFIX
        or protocol.get("episodes") != len(protocol.get("scenarios") or [])
        or protocol.get("sha256") != _sha(payload)
        or scenarios != expected_scenarios
    ):
        raise RuntimeError("closed-loop protocol provenance is invalid")
    _validate_protocol_contents(protocol)
    _validate_checkpoint_provenance(report, role)


def _validate_protocol_contents(protocol: dict[str, Any]) -> None:
    scale = _vector(protocol.get("object_scale"), 3, "comparison object scale")
    geometry = _validated_geometry_receipt(protocol.get("object_geometry") or {}, scale)
    if (
        protocol.get("object_width_m") != geometry["grasp_width_m"]
        or protocol.get("object_height_m") != geometry["support_height_m"]
        or protocol.get("support_surface_contract") != support_surface_contract()
        or not protocol.get("destination_marker_contract")
        or not protocol.get("object_identity")
        or not protocol.get("scene_id")
        or not isinstance(protocol.get("decisions"), int)
        or protocol["decisions"] < 1
        or protocol.get("action_selection") != _action_selection_protocol()
    ):
        raise RuntimeError("closed-loop task protocol is incomplete")
    _validate_object_source(protocol.get("object_usd"))
    _validated_robot_plant(protocol.get("robot_plant"))
    _validate_gold_scenarios(protocol["scenarios"], int(protocol["seed"]))


def _validate_effectiveness_header(
    report: dict[str, Any], expected_role: str, episodes: list[dict[str, Any]]
) -> None:
    ablation = report.get("wrist_ablation") or {}
    if (
        report.get("schema") != "npa.sim2real.pi05.closed_loop_eval.v1"
        or report.get("checkpoint_role") != expected_role
        or report.get("split") != "gold"
        or report.get("episode_count") != len(episodes)
        or not episodes
        or ablation.get("modes") != list(WRIST_ABLATION_MODES)
        or ablation.get("same_checkpoint_protocol_scenarios_and_reset_seeds")
        is not True
        or ablation.get("same_initial_object_poses") is not True
        or ablation.get("qualification") != "matched_effectiveness_eligible"
    ):
        raise RuntimeError("closed-loop effectiveness report header is invalid")


def validate_effectiveness_report(
    report: dict[str, Any], expected_role: str
) -> dict[str, Any]:
    """Validate a closed-loop report before making an efficacy comparison.

    Args:
        report: Persisted closed-loop evaluation report.
        expected_role: Checkpoint role required for this comparison arm.

    Returns:
        Recomputed fresh-wrist cohort metrics.

    Raises:
        RuntimeError: The report lacks matched execution or review evidence.
    """

    episodes = report.get("episodes") or []
    ablation = report.get("wrist_ablation") or {}
    _validate_effectiveness_header(report, expected_role, episodes)
    _validate_effectiveness_cohort(episodes)
    _validate_effectiveness_protocol(report, expected_role)
    scenario_seeds = {episode["scenario_seed"] for episode in episodes}
    protocol_seeds = {row["scene_seed"] for row in report["protocol"]["scenarios"]}
    if scenario_seeds != protocol_seeds:
        raise RuntimeError("evaluation episodes differ from protocol scenarios")
    _validate_wrist_interventions(episodes)
    grouped = _validate_effectiveness_summaries(report, episodes, ablation)
    _validate_effectiveness_artifacts(report.get("review_artifacts") or {}, episodes)
    return grouped["fresh"]


def _persist_evaluation(
    args: argparse.Namespace,
    work: Path,
    result: dict[str, Any],
    protocol: dict[str, Any],
) -> None:
    (work / "report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    from npa.clients.storage import StorageClient
    from npa.workflows.sim2real.workflow_io import (
        PI05_COMPONENT_SCHEMA,
        POLICY_INTERCHANGE_COMPONENT_SCHEMA,
        publish_component_record,
    )

    StorageClient.from_environment().upload_directory(
        str(work), args.artifact_root_uri, require_empty=True
    )
    stage = 6 if args.checkpoint_role == "base" else 11
    interchange = getattr(args, "policy_backend", "openpi_pi05") == "lerobot_smolvla"
    publish_component_record(
        component_schema=(
            POLICY_INTERCHANGE_COMPONENT_SCHEMA
            if interchange
            else PI05_COMPONENT_SCHEMA
        ),
        root_uri=args.component_root_uri,
        stage=stage,
        name=(
            f"smolvla_{args.checkpoint_role}_closed_loop_gold"
            if interchange
            else f"pi05_{args.checkpoint_role}_closed_loop_gold"
        ),
        tier="WORKS",
        require_gpu=True,
        evidence=(
            "Ran native SmolVLA current-action control and measured released-support physics."
            if interchange
            else "Ran receding-horizon pi0.5 control and measured released-support physics."
        ),
        artifacts={
            "report": args.output_uri,
            "artifacts": args.artifact_root_uri,
            "protocol_sha256": protocol["sha256"],
        },
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run closed-loop evaluation. Args: args. Returns: Report. Raises: RuntimeError."""
    app = None
    with tempfile.TemporaryDirectory(prefix="npa-pi05-closed-loop-") as directory:
        work = Path(directory)
        try:
            graphics = _graphics_preflight()
            (work / "graphics-preflight.json").write_text(
                json.dumps(graphics, indent=2, sort_keys=True) + "\n"
            )
            app = _launch_camera_app()
            import gymnasium as gym
            import isaaclab_tasks  # noqa: F401
            from isaaclab_tasks.utils import parse_env_cfg

            return _run_from_receipt(args, Path(directory), gym, parse_env_cfg)
        except Exception as error:
            _persist_evaluation_failure(args, work, error)
            raise
        finally:
            if app is not None:
                app.close()


def _persist_evaluation_failure(
    args: argparse.Namespace, work: Path, error: Exception
) -> None:
    from npa.clients.storage import StorageClient
    from npa.workflows.sim2real.pi05_isaac import _failure_report

    phase = (
        "graphics_preflight"
        if not (work / "graphics-preflight.json").exists()
        else "evaluation_runtime"
    )
    report = _failure_report(error, "closed_loop_gold", phase)
    report["schema"] = "npa.sim2real.pi05.closed_loop_failure.v1"
    (work / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    StorageClient.from_environment().upload_directory(
        str(work), args.artifact_root_uri, require_empty=True
    )


def _run_from_receipt(
    args: argparse.Namespace, work: Path, gym: Any, parse_env_cfg: Any
) -> dict[str, Any]:
    from npa.workflows.sim2real.workflow_io import read_json

    receipt = read_json(args.service_receipt_uri, directory=work)
    if (
        receipt.get("status") != "ready"
        or receipt.get("public_ingress") is not False
        or receipt.get("application_authentication") != "kubernetes_secret_bearer"
    ):
        raise RuntimeError("policy service receipt is not a ready private service")
    robot_profile = policy_profile = preparation = None
    if args.policy_backend == "lerobot_smolvla":
        if not args.preparation_uri:
            raise RuntimeError("SmolVLA evaluation requires --preparation-uri")
        preparation = read_json(args.preparation_uri, directory=work / "preparation")
        robot_profile = robot_sensor_profile_from_dict(preparation.get("robot") or {})
        policy_profile = policy_profile_from_dict(preparation.get("policy") or {})
        loaded = receipt.get("interchange") or {}
        if (
            policy_profile.backend != "lerobot_smolvla"
            or loaded.get("policy_identity") != policy_profile.identity
            or loaded.get("robot_identity") != robot_profile.identity
            or loaded.get("task_identity") != policy_profile.task_identity
            or loaded.get("action") != policy_profile.action.to_dict()
            or loaded.get("measured_state_names")
            != list(policy_profile.measured_state_names)
            or [row.get("key") for row in loaded.get("cameras") or []]
            != list(policy_profile.camera_keys)
        ):
            raise RuntimeError("SmolVLA service differs from prepared policy")
        policy_profile = replace(
            policy_profile,
            checkpoint_sha256=str(loaded.get("checkpoint_sha256") or ""),
            normalization_identity=str(loaded.get("normalization_identity") or ""),
        )
        client = SmolVLAPolicyClient(str(receipt["service_host"]), int(receipt["port"]))
    else:
        client = Pi05PolicyClient(
            str(receipt["service_host"]), int(receipt["service_port"])
        )
    protocol = _load_gold_protocol(args, work, preparation)
    return _run_evaluation(
        args,
        work,
        receipt,
        client,
        protocol,
        gym,
        parse_env_cfg,
        robot_profile=robot_profile,
        policy_profile=policy_profile,
    )


def _run_evaluation(
    args: argparse.Namespace,
    work: Path,
    receipt: dict[str, Any],
    client: Pi05PolicyClient,
    protocol: dict[str, Any],
    gym: Any,
    parse_env_cfg: Any,
    *,
    robot_profile: Any = None,
    policy_profile: Any = None,
) -> dict[str, Any]:
    cfg = parse_env_cfg(POLICY_BASE_TASK_ID, device="cuda:0", num_envs=1)
    _configure_eval(cfg, args, protocol)
    register_surface_task(
        gym, base_id=POLICY_BASE_TASK_ID, task_id=POLICY_SURFACE_TASK_ID
    )
    env = gym.make(POLICY_SURFACE_TASK_ID, cfg=cfg)
    try:
        _verify_scene_semantics(env.unwrapped)
        _validate_eval_object_source(env.unwrapped, protocol)
        _validate_eval_geometry(env.unwrapped, protocol)
        _validate_eval_robot_plant(env.unwrapped, protocol)
        episodes = _evaluate(
            env,
            client,
            args,
            protocol,
            robot_profile=robot_profile,
            policy_profile=policy_profile,
        )
        try:
            artifacts = _write_review_artifacts(work, episodes, args.checkpoint_role)
            result = _result(args, episodes, artifacts, protocol, receipt)
        except Exception:
            _persist_partial_episodes(work, episodes)
            raise
        _persist_evaluation(args, work, result, protocol)
        return result
    finally:
        env.close()


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI. Args: None. Returns: Parser. Raises: None."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--service-receipt-uri", required=True)
    parser.add_argument(
        "--policy-backend",
        choices=("openpi_pi05", "lerobot_smolvla"),
        default="openpi_pi05",
    )
    parser.add_argument("--preparation-uri", default="")
    parser.add_argument("--gold-collection-uri", required=True)
    parser.add_argument("--output-uri", required=True)
    parser.add_argument("--artifact-root-uri", required=True)
    parser.add_argument("--component-root-uri", required=True)
    parser.add_argument("--checkpoint-role", choices=("base", "adapted"), required=True)
    parser.add_argument("--split", choices=("validation", "gold"), required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--decisions", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--wrist-ablation-modes", default="fresh,frozen,pixel_channel_shuffle"
    )
    parser.add_argument("--ablation-seed", type=int, required=True)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse and validate the closed-loop CLI contract."""
    parser = build_parser()
    args = parser.parse_args(argv)
    args.wrist_ablation_modes = args.wrist_ablation_modes.split(",")
    expected_modes = {"fresh", "frozen", "pixel_channel_shuffle"}
    if set(args.wrist_ablation_modes) != expected_modes:
        parser.error(
            "--wrist-ablation-modes must be fresh,frozen,pixel_channel_shuffle"
        )
    if min(args.episodes, args.decisions) < 1:
        parser.error("episodes and decisions must be positive")
    expected_output = f"{args.artifact_root_uri.rstrip('/')}/report.json"
    if args.output_uri != expected_output:
        parser.error("--output-uri must be <artifact-root-uri>/report.json")
    return args


def validate_success_report(report: dict[str, Any], args: argparse.Namespace) -> None:
    """Reject absent, failed, or incomplete post-Kit output."""
    expected = args.episodes * len(args.wrist_ablation_modes)
    artifacts = report.get("review_artifacts") or {}
    expected_schema = (
        "npa.sim2real.policy_interchange.closed_loop_eval.v1"
        if getattr(args, "policy_backend", "openpi_pi05") == "lerobot_smolvla"
        else "npa.sim2real.pi05.closed_loop_eval.v1"
    )
    if (
        report.get("schema") != expected_schema
        or report.get("checkpoint_role") != args.checkpoint_role
        or report.get("split") != args.split
        or report.get("episode_count") != expected
        or len(report.get("episodes") or []) != expected
    ):
        raise RuntimeError("closed-loop report is absent, failed, or incomplete")
    for name in ("mp4", "rrd"):
        if int((artifacts.get(name) or {}).get("bytes", 0)) < 1:
            raise RuntimeError(f"closed-loop report lacks decoded {name} evidence")


def main(argv: list[str] | None = None) -> int:
    """Run inside Kit. The outer entrypoint validates remote report bytes."""
    args = parse_args(argv)
    try:
        result = run(args)
        validate_success_report(result, args)
    except Exception:
        if os.environ.get("NPA_PI05_CLOSED_LOOP_CHILD") == "1":
            import traceback

            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)
        raise
    print(json.dumps(result, sort_keys=True), flush=True)
    if os.environ.get("NPA_PI05_CLOSED_LOOP_CHILD") == "1":
        os._exit(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
