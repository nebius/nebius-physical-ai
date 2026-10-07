"""Closed-loop Isaac evaluation of a private pinned OpenPI pi0.5 service."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

from npa.workflows.sim2real.pi05_contract import (
    CONTROL_HZ,
    EXECUTION_PREFIX,
    JOINT_NAMES,
    absolute_to_isaac_action,
    verify_controller_roundtrip,
    width_to_droid_gripper,
)
from npa.workflows.sim2real.pi05_isaac import (
    POLICY_BASE_TASK_ID,
    POLICY_SURFACE_TASK_ID,
    _camera_config,
    _camera_pose_sample,
    _camera_time,
    _contact_config,
    _contact_forces,
    _finger_controller_target,
    _gripper_width,
    _physics_randomization,
    _rgb,
    _sim_time,
    _target_pixel_count,
    _verify_camera_motion,
    configure_surface_goal,
    register_surface_task,
)
from npa.workflows.sim2real.pi05_policy_client import Pi05PolicyClient


PROMPT = "Pick up the object and release it inside the target area on the table."


def _vector(value: Any, size: int, label: str) -> list[float]:
    import numpy as np

    array = np.asarray(value).reshape(-1)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise RuntimeError(f"{label} must contain {size} finite values")
    return [float(item) for item in array]


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
    defaults = _vector(offset[0], 7, "arm action offsets")
    scales = _vector(scale[0], 7, "arm action scales")
    return (
        defaults,
        scales,
        sum(dimensions),
        sum(dimensions[:arm_index]),
        sum(dimensions[:grip_index]),
    )


def _policy_observation(native: Any) -> tuple[Any, Any, Any, Any]:
    import numpy as np

    robot = native.scene["robot"]
    indices = [list(robot.joint_names).index(name) for name in JOINT_NAMES]
    joint = robot.data.joint_pos[0, indices].detach().cpu().numpy().astype(np.float32)
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
    controller = robot.data.joint_pos_target[0, indices].detach().cpu().numpy()
    return verify_controller_roundtrip(
        target,
        defaults=defaults,
        scales=scales,
        controller_target=controller,
        finger_controller_target_m=_finger_controller_target(native),
    )


def _physics_sample(native: Any, initial_z: float, target_xy: Any) -> dict[str, Any]:
    import numpy as np
    import torch

    obj = native.scene["object"].data.root_pos_w[0, :3]
    speed = float(torch.linalg.norm(native.scene["object"].data.root_lin_vel_w[0]))
    left, right, support = _contact_forces(native)
    width = _gripper_width(native)
    position = obj.detach().cpu().numpy()
    robot = native.scene["robot"]
    hand_index = list(robot.body_names).index("panda_hand")
    hand = robot.data.body_pos_w[0, hand_index].detach().cpu().numpy()
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
    }


def _success(trace: list[dict[str, Any]]) -> dict[str, Any]:
    contact = any(row["bilateral_finger_contact"] for row in trace)
    lifted = any(row["object_lift_m"] >= 0.05 for row in trace)
    transported = any(
        row["target_xy_distance_m"] < 0.08 and row["object_lift_m"] >= 0.05
        for row in trace
    )
    released = False
    stable = maximum_stable = 0
    for row in trace:
        released_now = row["gripper_width_m"] >= 0.06
        released_now = released_now and not row["bilateral_finger_contact"]
        released |= released_now
        supported = released_now and row["object_support_force_n"] > 1.0e-3
        supported = supported and row["target_xy_distance_m"] < 0.05
        supported = supported and row["object_speed_m_s"] < 0.03
        supported = supported and row["hand_object_distance_m"] >= 0.10
        stable = stable + 1 if supported else 0
        maximum_stable = max(maximum_stable, stable)
    passed = contact and lifted and transported and maximum_stable >= 3
    return {
        "contact": contact,
        "lift": lifted,
        "transport": transported,
        "released": released,
        "stable_steps": maximum_stable,
        "released_supported_surface_placement": passed,
        "old_strict_5cm_diagnostic_only": any(
            row["target_xy_distance_m"] < 0.05 for row in trace
        ),
    }


def _new_episode_record() -> dict[str, Any]:
    return {
        "queries": [],
        "physics_trace": [],
        "camera_trace": [],
        "_exterior_frames": [],
        "_wrist_frames": [],
    }


def _query_policy(
    context: dict[str, Any],
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    native, client, record = context["native"], context["client"], context["record"]
    exterior, wrist, joint, gripper = _policy_observation(native)
    record["_exterior_frames"].append(exterior)
    record["_wrist_frames"].append(wrist)
    timing = {
        "observation_time_s": _sim_time(native),
        "exterior_capture_time_s": _camera_time(native, "pi05_exterior"),
        "wrist_capture_time_s": _camera_time(native, "pi05_wrist"),
    }
    if (
        max(
            abs(timing["exterior_capture_time_s"] - timing["observation_time_s"]),
            abs(timing["wrist_capture_time_s"] - timing["observation_time_s"]),
        )
        > native.step_dt
    ):
        raise RuntimeError("policy cameras are stale at the causal observation")
    actions, inference = client.infer(
        exterior=exterior,
        wrist=wrist,
        joint_position=joint,
        gripper_position=gripper,
        prompt=PROMPT,
    )
    camera = _camera_pose_sample(native)
    camera.update(
        {
            "wrist_target_pixels": _target_pixel_count(native, "pi05_wrist"),
            "exterior_target_pixels": _target_pixel_count(native, "pi05_exterior"),
        }
    )
    record["camera_trace"].append({"camera_pose": camera, **camera})
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
                context["native"], context["initial_z"], context["target_xy"]
            )
        )
        if proof["terminated"] or proof["truncated"]:
            break
    return applied


def _decision_step(context: dict[str, Any], decision: int) -> bool:
    actions, inference, timing = _query_policy(context)
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
    record["camera_evidence"] = _verify_camera_motion(record.pop("camera_trace"))
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
) -> dict[str, Any]:
    import numpy as np

    native = env.unwrapped
    initial = native.scene["object"].data.root_pos_w[0, :3].detach().cpu().numpy()
    expected_initial = np.asarray(scenario["initial_object_position_m"])
    if not np.allclose(initial, expected_initial, atol=1.0e-4, rtol=0.0):
        raise RuntimeError("evaluation reset does not reproduce the gold scenario")
    context = {
        "env": env,
        "native": native,
        "client": client,
        "record": record,
        "bridge": _joint_bridge(native),
        "initial_z": float(initial[2]),
        "target_xy": np.asarray(scenario["target_position_m"][:2]),
    }
    for decision in range(decisions):
        if _decision_step(context, decision):
            break
    return _finish_episode(record)


def _write_video(work: Path, episodes: list[dict[str, Any]]) -> tuple[Path, int]:
    import numpy as np

    from npa.adapter.sim_to_lerobot import encode_video

    rows = [
        row
        for row in episodes
        if row.get("_exterior_frames") and row.get("_wrist_frames")
    ]
    if not rows:
        raise RuntimeError(
            "closed-loop evaluation produced no reviewable camera frames"
        )
    exterior = np.concatenate([np.stack(row.pop("_exterior_frames")) for row in rows])
    wrist = np.concatenate([np.stack(row.pop("_wrist_frames")) for row in rows])
    preview = np.concatenate([exterior, wrist], axis=2)
    mp4 = work / "rollouts.mp4"
    encode_video(preview, mp4, int(CONTROL_HZ))
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
    recording.flush()
    if not rrd.is_file() or rrd.stat().st_size == 0 or step < 1:
        raise RuntimeError("closed-loop Rerun readback is empty")
    return rrd, step


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
    scale = _vector(asset.get("scale"), 3, "gold object scale")
    scenarios = collection.get("successful_scenarios") or []
    if len(scenarios) < args.episodes:
        raise RuntimeError("gold collection lacks enough physics-qualified scenarios")
    scenarios = scenarios[: args.episodes]
    for index, scenario in enumerate(scenarios):
        if int(scenario.get("scene_seed", -1)) != args.seed + index:
            raise RuntimeError("gold scenario seed does not match evaluation protocol")
        _vector(scenario.get("initial_object_position_m"), 3, "initial object")
        _vector(scenario.get("target_position_m"), 3, "surface target")
    protocol = {
        "task": "released_supported_surface_placement_v1",
        "split": args.split,
        "seed": args.seed,
        "episodes": args.episodes,
        "decisions": args.decisions,
        "object_identity": asset.get("identity"),
        "object_scale": scale,
        "scene_id": collection.get("scene_id"),
        "control_hz": CONTROL_HZ,
        "execution_prefix": EXECUTION_PREFIX,
        "scenarios": scenarios,
    }
    if not protocol["object_identity"] or not protocol["scene_id"]:
        raise RuntimeError("gold collection lacks object or scene identity")
    protocol["sha256"] = _sha(protocol)
    protocol["scenario_sha256"] = [
        _sha({"protocol": protocol["sha256"], "seed": args.seed + index})
        for index in range(args.episodes)
    ]
    return protocol


def _configure_eval(
    cfg: Any, args: argparse.Namespace, protocol: dict[str, Any]
) -> None:
    from npa.workflows.sim2real.isaac_assets_compat import remap_moved_franka_usd

    cfg.seed, cfg.sim.dt, cfg.decimation = args.seed, 1.0 / 60.0, 4
    cfg.episode_length_s = 240.0
    remap_moved_franka_usd(cfg)
    cfg.scene.object.spawn.scale = tuple(protocol["object_scale"])
    cfg.scene.object.spawn.semantic_tags = [("class", "pi05_target_object")]
    target = tuple(protocol["scenarios"][0]["target_position_m"])
    if any(
        scenario["target_position_m"] != list(target)
        for scenario in protocol["scenarios"]
    ):
        raise RuntimeError("gold scenarios do not share one declared surface target")
    configure_surface_goal(cfg, target)
    _camera_config(cfg)
    _contact_config(cfg)
    _physics_randomization(cfg)


def _evaluate(
    env: Any,
    client: Pi05PolicyClient,
    args: argparse.Namespace,
    protocol: dict[str, Any],
) -> list[dict[str, Any]]:
    episodes = []
    for index in range(args.episodes):
        env.reset(seed=args.seed + index)
        record = _new_episode_record()
        try:
            result = _episode(
                env, client, args.decisions, record, protocol["scenarios"][index]
            )
        except Exception as exc:
            record.pop("camera_trace", None)
            record.update(
                {
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
    passes = sum(
        bool(row["success"].get("released_supported_surface_placement"))
        for row in episodes
    )
    return {
        "schema": "npa.sim2real.pi05.closed_loop_eval.v1",
        "checkpoint_role": args.checkpoint_role,
        "split": args.split,
        "checkpoint_provenance": _checkpoint_provenance(receipt),
        "protocol": protocol,
        "episode_count": len(episodes),
        "released_surface_placements": passes,
        "success_rate": passes / len(episodes),
        "task_comparison": "same pi0.5 surface task only",
        "physical_robot_deployment": False,
        "episodes": episodes,
        "review_artifacts": artifacts,
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
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, enable_cameras=True).app
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    from npa.workflows.sim2real.workflow_io import read_json

    work = Path(tempfile.mkdtemp(prefix="npa-pi05-closed-loop-"))
    receipt = read_json(args.service_receipt_uri, directory=work)
    if receipt.get("status") != "ready" or receipt.get("public_ingress") is not False:
        raise RuntimeError("OpenPI service receipt is not a ready private service")
    client = Pi05PolicyClient(
        str(receipt["service_host"]), int(receipt["service_port"])
    )
    protocol = _load_gold_protocol(args, work)
    env = None
    try:
        cfg = parse_env_cfg(POLICY_BASE_TASK_ID, device="cuda:0", num_envs=1)
        _configure_eval(cfg, args, protocol)
        register_surface_task(
            gym, base_id=POLICY_BASE_TASK_ID, task_id=POLICY_SURFACE_TASK_ID
        )
        env = gym.make(POLICY_SURFACE_TASK_ID, cfg=cfg)
        episodes = _evaluate(env, client, args, protocol)
        artifacts = _write_review_artifacts(work, episodes, args.checkpoint_role)
        result = _result(args, episodes, artifacts, protocol, receipt)
        _persist_evaluation(args, work, result, protocol)
        return result
    finally:
        if env is not None:
            env.close()
        app.close()


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
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI. Args: argv. Returns: Exit status. Raises: RuntimeError."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if min(args.episodes, args.decisions) < 1:
        parser.error("episodes and decisions must be positive")
    expected_output = f"{args.artifact_root_uri.rstrip('/')}/report.json"
    if args.output_uri != expected_output:
        parser.error("--output-uri must be <artifact-root-uri>/report.json")
    result = run(args)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
