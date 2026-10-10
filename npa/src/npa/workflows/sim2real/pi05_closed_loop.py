"""Closed-loop Isaac evaluation of a private pinned OpenPI pi0.5 service."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from npa.workflows.sim2real.pi05_contract import (
    CONTROL_HZ,
    EXECUTION_PREFIX,
    JOINT_NAMES,
    absolute_to_isaac_action,
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
    _semantic_encoding_receipt,
    _sim_time,
    _sensor_clock_reference,
    _synchronize_wrist_before_capture,
    _target_pixel_count,
    _validate_capture_ages,
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


PROMPT = (
    "Pick up the object and release it inside the cyan circular target on the table."
)
MANIPULATION_EVENT_ORDER = (
    "contact",
    "grasp",
    "lift",
    "transport",
    "release",
    "support_contact",
    "stable",
)


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
    if destination_pixels < 16:
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
            "phase": _policy_phase(record),
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
    native, client, record = context["native"], context["client"], context["record"]
    exterior, wrist, joint, gripper = _policy_observation(native)
    record["_exterior_frames"].append(exterior)
    record["_wrist_frames"].append(wrist)
    timing, receipts = _camera_query_receipts(native, exterior, wrist)
    policy_wrist, intervention = _wrist_variant(context, wrist, decision)
    sampling_noise, sampling_sequence = _decision_noise(context, decision)
    actions, inference = client.infer(
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
    _record_query_camera(native, record, receipts, inference["wire_input"])
    timing["exterior_rgb_sha256"] = receipts["exterior"]["sha256"]
    timing["wrist_rgb_sha256"] = receipts["wrist"]["sha256"]
    return actions, inference, timing


def _execute_prefix(context: dict[str, Any], actions: Any) -> list[dict[str, Any]]:
    applied, bridge = [], context["bridge"]
    for target in actions[:EXECUTION_PREFIX]:
        proof = _apply_target(
            context["env"],
            target,
            defaults=bridge[0],
            scales=bridge[1],
            total_dim=bridge[2],
            arm_start=bridge[3],
            grip_start=bridge[4],
        )
        applied.append(proof)
        context["record"]["physics_trace"].append(
            _physics_sample(
                context["native"],
                context["initial_z"],
                context["target_xy"],
                context["object_width_m"],
                bool(float(target[7]) > 0.5),
            )
        )
        if proof["terminated"] or proof["truncated"]:
            break
    return applied


def _decision_step(context: dict[str, Any], decision: int) -> bool:
    actions, inference, timing = _query_policy(context, decision)
    applied = _execute_prefix(context, actions)
    context["record"]["queries"].append(
        {
            "decision": decision,
            **timing,
            "inference": inference,
            "executed_prefix": len(applied),
            "controller_proofs": applied,
        }
    )
    return bool(applied and (applied[-1]["terminated"] or applied[-1]["truncated"]))


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
        "model_horizon": 15,
        "executed_prefix": EXECUTION_PREFIX,
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
) -> dict[str, Any]:
    native = env.unwrapped
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
) -> dict[str, Any]:
    import numpy as np

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

    recording = rr.RecordingStream(f"pi05-{role}")
    rrd = work / "rollouts.rrd"
    recording.save(rrd)
    step = 0
    for episode_index, episode in enumerate(episodes):
        for sample in episode.get("physics_trace", []):
            recording.set_time("step", sequence=step)
            recording.log(
                f"episodes/{episode_index}/object_position",
                rr.Points3D([sample["object_position_m"]]),
            )
            recording.log(
                f"episodes/{episode_index}/object_speed_m_s",
                rr.Scalars([sample["object_speed_m_s"]]),
            )
            recording.log(
                f"episodes/{episode_index}/target_xy_distance_m",
                rr.Scalars([sample["target_xy_distance_m"]]),
            )
            step += 1
        _log_rrd_queries(recording, rr, episode, episode_index)
    recording.flush()
    if not rrd.is_file() or rrd.stat().st_size == 0 or step < 1:
        raise RuntimeError("closed-loop Rerun readback is empty")
    return rrd, step


def _log_rrd_queries(
    recording: Any, rr: Any, episode: dict, episode_index: int
) -> None:
    for query_index, query in enumerate(episode.get("queries", [])):
        recording.set_time("query", sequence=query_index)
        root = f"episodes/{episode_index}/queries/{query_index}"
        recording.log(
            root + "/exterior", rr.Image(episode["_exterior_frames"][query_index])
        )
        recording.log(root + "/wrist", rr.Image(episode["_wrist_frames"][query_index]))
        recording.log(
            root + "/capture_age_s",
            rr.Scalars([query["inference"]["wire_input"]["capture_age_s"]]),
        )


def _decode_artifacts(mp4: Path, rrd: Path) -> tuple[dict[str, Any], str]:
    import subprocess
    import sys

    rerun = str(Path(sys.executable).with_name("rerun"))
    subprocess.run([rerun, "rrd", "verify", str(rrd)], check=True)
    decoded = subprocess.run(
        [rerun, "rrd", "print", "-v", str(rrd)],
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
    mp4, frames = _write_video(work, episodes)
    rrd, steps = _write_rrd(work, episodes, checkpoint_role)
    probe, _ = _decode_artifacts(mp4, rrd)
    for episode in episodes:
        episode.pop("_exterior_frames", None)
        episode.pop("_wrist_frames", None)
    return {
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


def _sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _load_gold_protocol(args: argparse.Namespace, work: Path) -> dict[str, Any]:
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
        "scenarios": scenarios,
    }
    if not protocol["object_identity"] or not protocol["scene_id"]:
        raise RuntimeError("gold collection lacks object or scene identity")
    _validate_object_source(protocol["object_usd"])
    _bind_protocol_hashes(protocol)
    return protocol


def _protocol_object_fields(
    asset: dict[str, Any], geometry: dict[str, Any], scale: list[float]
) -> dict[str, Any]:
    return {
        "object_identity": asset.get("identity"),
        "object_usd": asset.get("usd"),
        "object_scale": scale,
        "object_width_m": _measured_dimension(geometry, "grasp_width_m", 0.08),
        "object_height_m": _measured_dimension(geometry, "support_height_m", 0.25),
        "object_geometry": _validated_geometry_receipt(geometry, scale),
    }


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
    if protocol["object_usd"] != "task_default":
        cfg.scene.object.spawn.usd_path = protocol["object_usd"]
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
                )
            except Exception as exc:
                record.update(
                    {
                        "wrist_ablation_mode": mode,
                        "success": {"released_supported_surface_placement": False},
                        "failure": f"{type(exc).__name__}: {exc}",
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
    keys = (
        "source_commit",
        "config_name",
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
    noise_pairing = _sampler_noise_pairing(episodes, args.wrist_ablation_modes)
    return {
        "schema": "npa.sim2real.pi05.closed_loop_eval.v1",
        "checkpoint_role": args.checkpoint_role,
        "split": args.split,
        "checkpoint_provenance": _checkpoint_provenance(receipt),
        "protocol": protocol,
        "episode_count": len(episodes),
        "released_surface_placements": passes,
        "success_rate": passes / len(episodes),
        "wrist_ablation": {
            "modes": args.wrist_ablation_modes,
            "same_checkpoint_protocol_scenarios_and_reset_seeds": True,
            "same_initial_object_poses": True,
            **noise_pairing,
            "per_mode": grouped,
            "qualification": (
                "matched_effectiveness_eligible"
                if noise_pairing["sampler_noise_identity_verified"]
                else "effectiveness_claim_rejected_without_sampler_noise_identity"
            ),
        },
        "task_comparison": "same pi0.5 surface task only",
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


def _sampler_noise_pairing(episodes: list[dict], modes: list[str]) -> dict[str, Any]:
    identities: dict[tuple[int, int], list[str | None]] = {}
    for episode in episodes:
        for query in episode.get("queries", []):
            key = (int(episode["scenario_seed"]), int(query["decision"]))
            identities.setdefault(key, []).append(
                query["inference"].get("sampler_noise_sha256")
            )
    complete = bool(identities) and all(
        len(values) == len(modes) and None not in values and len(set(values)) == 1
        for values in identities.values()
    )
    return {
        "sampling_noise_controlled": complete,
        "sampler_noise_identity_verified": complete,
        "sampler_noise_identity_receipt_count": sum(
            value is not None for values in identities.values() for value in values
        ),
        "sampling_noise_limitation": (
            None
            if complete
            else "scoped service did not return a complete paired explicit-noise identity"
        ),
        "same_state_action_sensitivity_only": True,
        "closed_loop_states_and_exterior_diverge_after_different_actions": True,
    }


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
    from npa.workflows.sim2real.workflow_io import publish_component_record

    StorageClient.from_environment().upload_directory(
        str(work), args.artifact_root_uri, require_empty=True
    )
    stage = 6 if args.checkpoint_role == "base" else 11
    publish_component_record(
        root_uri=args.component_root_uri,
        stage=stage,
        name=f"pi05_{args.checkpoint_role}_closed_loop_gold",
        tier="WORKS",
        require_gpu=True,
        evidence="Ran receding-horizon pi0.5 control and measured released-support physics.",
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
        raise RuntimeError("OpenPI service receipt is not a ready private service")
    client = Pi05PolicyClient(
        str(receipt["service_host"]), int(receipt["service_port"])
    )
    protocol = _load_gold_protocol(args, work)
    return _run_evaluation(args, work, receipt, client, protocol, gym, parse_env_cfg)


def _run_evaluation(
    args: argparse.Namespace,
    work: Path,
    receipt: dict[str, Any],
    client: Pi05PolicyClient,
    protocol: dict[str, Any],
    gym: Any,
    parse_env_cfg: Any,
) -> dict[str, Any]:
    cfg = parse_env_cfg(POLICY_BASE_TASK_ID, device="cuda:0", num_envs=1)
    _configure_eval(cfg, args, protocol)
    register_surface_task(
        gym, base_id=POLICY_BASE_TASK_ID, task_id=POLICY_SURFACE_TASK_ID
    )
    env = gym.make(POLICY_SURFACE_TASK_ID, cfg=cfg)
    try:
        _verify_scene_semantics(env.unwrapped)
        _validate_eval_geometry(env.unwrapped, protocol)
        _validate_eval_robot_plant(env.unwrapped, protocol)
        episodes = _evaluate(env, client, args, protocol)
        artifacts = _write_review_artifacts(work, episodes, args.checkpoint_role)
        result = _result(args, episodes, artifacts, protocol, receipt)
        _persist_evaluation(args, work, result, protocol)
        return result
    finally:
        env.close()


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI. Args: None. Returns: Parser. Raises: None."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--service-receipt-uri", required=True)
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
    if (
        report.get("schema") != "npa.sim2real.pi05.closed_loop_eval.v1"
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
