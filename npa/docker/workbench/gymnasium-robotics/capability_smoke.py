"""Exact Gymnasium-Robotics Shadow Hand MuJoCo/EGL capability gate.

This script is packaged for a future authorized image transaction. Phase A
does not execute it and its presence is not capability evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any

import gymnasium as gym
import gymnasium_robotics
import mujoco
import numpy as np

ENV_ID = "HandManipulateBlockRotateXYZ_ContinuousTouchSensors-v1"
RESET_SEED = 20260910
ACTION_SEED = 11092026
ROLLOUT_STEPS = 120
MIN_TRANSITION_DELTA = 1e-6
EXPECTED_SOURCE = "4d1ebecbc6436806cfbc0e42ebc36f594d05844e"
EXPECTED_MUJOCO_COMMIT = "13827e9ee56f097f57acf69ae52b078f9839682d"
EXPECTED_MUJOCO_WHEEL = (
    "7ec16ce408871a0a9157cc556958ab66cd34db9fc1dccd3ef07717170163a4e0"
)
EXPECTED_ASSET_LOCK = "e22eb62fc690a5e1d1ea931bab950392ca480caf3d51c7f16fd8cb4133d65568"


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _exact_files(root: Path, expected: dict[str, str]) -> dict[str, str]:
    observed = {name: _sha256(root / name) for name in sorted(expected)}
    if observed != dict(sorted(expected.items())):
        raise RuntimeError("directly loaded upstream asset closure changed")
    return observed


def _digest_from_reference(value: str, field: str) -> str:
    matches = re.findall(r"sha256:[0-9a-f]{64}", value.lower())
    if not matches:
        raise RuntimeError(f"{field} is not digest-pinned")
    return matches[-1]


def _gpu() -> dict[str, Any]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,compute_cap,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [row for row in completed.stdout.splitlines() if row.strip()]
    if len(rows) != 1:
        raise RuntimeError(f"expected one GPU, observed {len(rows)}")
    name, capability, driver = (part.strip() for part in rows[0].split(","))
    if "RTX PRO 6000" not in name or "Blackwell" not in name or capability != "12.0":
        raise RuntimeError(f"wrong GPU for EGL gate: {name}, compute {capability}")
    return {
        "name": name,
        "architecture": "Blackwell",
        "compute_capability": capability,
        "driver_version": driver,
        "count": 1,
    }


def _quaternion_angle(left: np.ndarray, right: np.ndarray) -> float:
    left = left / np.linalg.norm(left)
    right = right / np.linalg.norm(right)
    dot = float(np.clip(abs(np.dot(left, right)), 0.0, 1.0))
    return float(2.0 * math.acos(dot))


def _graphics_libraries() -> list[dict[str, str]]:
    paths: set[Path] = set()
    for line in Path("/proc/self/maps").read_text(encoding="utf-8").splitlines():
        path = Path(line.rsplit(" ", 1)[-1])
        if any(marker in path.name for marker in ("libEGL", "libGL", "libOpenGL")):
            paths.add(path)
    observed = [{"path": str(path), "sha256": _sha256(path)} for path in sorted(paths)]
    if not any(
        "libegl" in item["path"].lower() and "nvidia" in item["path"].lower()
        for item in observed
    ):
        raise RuntimeError("render did not load a host NVIDIA EGL library")
    return observed


def _canonical_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _source_evidence() -> tuple[Path, dict[str, Any], dict[str, Any]]:
    package_root = Path(gymnasium_robotics.__file__).resolve().parent
    source_root = package_root.parent
    locks = Path("/opt/npa/gymnasium-robotics")
    source_lock = json.loads((locks / "source-lock.json").read_text(encoding="utf-8"))
    asset_lock_path = locks / "asset-lock.json"
    if _sha256(asset_lock_path) != EXPECTED_ASSET_LOCK:
        raise RuntimeError("approved asset lock bytes changed")
    asset_lock = json.loads(asset_lock_path.read_text(encoding="utf-8"))
    component = source_lock["components"]["farama_gymnasium_robotics"]
    if (
        component["commit"] != EXPECTED_SOURCE
        or gymnasium_robotics.__version__ != "1.4.2"
    ):
        raise RuntimeError(
            "installed Gymnasium-Robotics source is not the reviewed pin"
        )
    if source_lock["status"] != "complete":
        raise RuntimeError("image source lock was not completed")
    return source_root, source_lock, asset_lock


def _render(env: gym.Env, hashes: list[str], shapes: list[list[int]]) -> float:
    started = time.perf_counter()
    frame = np.asarray(env.render())
    elapsed = time.perf_counter() - started
    if frame.dtype != np.uint8 or frame.shape != (240, 320, 3):
        raise RuntimeError(f"invalid real RGB frame: {frame.shape}/{frame.dtype}")
    hashes.append(hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest())
    shapes.append(list(frame.shape))
    return elapsed


def _require_state_transition(
    position_changes: list[float],
    orientation_changes: list[float],
    state_changes: list[float],
) -> None:
    if (
        not position_changes
        or max(position_changes) <= MIN_TRANSITION_DELTA
        or not orientation_changes
        or max(orientation_changes) <= MIN_TRANSITION_DELTA
        or not state_changes
        or max(state_changes) <= MIN_TRANSITION_DELTA
    ):
        raise RuntimeError(
            "rollout did not evolve object position, orientation, and simulator state"
        )


def _rollout(env: gym.Env, observation: dict[str, np.ndarray]) -> dict[str, Any]:
    raw = env.unwrapped
    initial_qpos = np.asarray(raw.data.qpos, dtype=np.float64).copy()
    initial_goal = np.asarray(observation["achieved_goal"], dtype=np.float64).copy()
    touch_ids = np.asarray(raw._touch_sensor_id, dtype=np.int64)
    if touch_ids.shape != (92,):
        raise RuntimeError(f"expected 92 touch sensors, got {touch_ids.shape}")
    frames: list[str] = []
    shapes: list[list[int]] = []
    render_seconds = _render(env, frames, shapes)
    rng = np.random.default_rng(ACTION_SEED)
    action = np.zeros(env.action_space.shape, dtype=np.float64)
    rewards: list[float] = []
    contacts: list[int] = []
    touches: list[int] = []
    touch_maxima: list[float] = []
    positions: list[float] = []
    orientations: list[float] = []
    state_changes: list[float] = []
    step_seconds = 0.0
    for index in range(ROLLOUT_STEPS):
        action = np.clip(
            0.65 * action + 0.35 * rng.uniform(-0.9, 0.9, action.shape), -1, 1
        )
        started = time.perf_counter()
        observation, reward, terminated, truncated, _ = env.step(action)
        step_seconds += time.perf_counter() - started
        reward = float(reward)
        if not math.isfinite(reward) or terminated or truncated:
            raise RuntimeError(f"invalid rollout transition at step {index + 1}")
        rewards.append(reward)
        contacts.append(int(raw.data.ncon))
        sensor = np.asarray(raw.data.sensordata[touch_ids], dtype=np.float64)
        if sensor.shape != (92,) or not np.isfinite(sensor).all():
            raise RuntimeError("invalid live touch sensor data")
        touches.append(int(np.count_nonzero(sensor > 0)))
        touch_maxima.append(float(sensor.max()))
        achieved = np.asarray(observation["achieved_goal"], dtype=np.float64)
        positions.append(float(np.linalg.norm(initial_goal[:3] - achieved[:3])))
        orientations.append(_quaternion_angle(initial_goal[3:], achieved[3:]))
        state_changes.append(
            float(np.linalg.norm(np.asarray(raw.data.qpos) - initial_qpos))
        )
        if index % 4 == 3:
            render_seconds += _render(env, frames, shapes)
    if not any(contacts) or not any(touches) or max(touch_maxima) <= 0:
        raise RuntimeError("rollout did not produce contact and touch response")
    _require_state_transition(positions, orientations, state_changes)
    if len(set(frames)) < 2:
        raise RuntimeError("EGL output did not contain two distinct RGB frames")
    return {
        "contacts": contacts,
        "frame_hashes": frames,
        "frame_shapes": shapes,
        "orientation_changes": orientations,
        "position_changes": positions,
        "render_seconds": render_seconds,
        "rewards": rewards,
        "state_changes": state_changes,
        "initial_qpos": initial_qpos,
        "initial_goal": initial_goal,
        "final_qpos": np.asarray(raw.data.qpos, dtype=np.float64).copy(),
        "final_goal": np.asarray(observation["achieved_goal"], dtype=np.float64).copy(),
        "step_seconds": step_seconds,
        "touch_maxima": touch_maxima,
        "touch_nonzero": touches,
    }


def main() -> None:
    if (
        os.environ.get("MUJOCO_GL") != "egl"
        or os.environ.get("PYOPENGL_PLATFORM") != "egl"
    ):
        raise RuntimeError("MuJoCo and PyOpenGL must use EGL")
    source_root, source_lock, asset_lock = _source_evidence()
    expected = _digest_from_reference(os.environ.get("BYOF_IMAGE", ""), "BYOF_IMAGE")
    observed = _digest_from_reference(
        os.environ.get("NPA_BYOF_POD_IMAGE_ID", ""), "pod imageID"
    )
    if expected != observed:
        raise RuntimeError("pod-observed image digest differs from requested digest")
    gpu = _gpu()
    assets_root = source_root / "gymnasium_robotics" / "envs" / "assets"
    xml = _exact_files(assets_root, asset_lock["directly_loaded_xml"])
    material = _exact_files(assets_root, asset_lock["directly_loaded_mesh_texture"])
    if _sha256(assets_root / "LICENSE.md") != asset_lock["asset_notice_sha256"]:
        raise RuntimeError("Shadow Hand asset notice changed")
    if mujoco.__version__ != "3.12.0" or mujoco.mj_versionString() != "3.12.0":
        raise RuntimeError("MuJoCo runtime version changed")
    gym.register_envs(gymnasium_robotics)
    spec = gym.spec(ENV_ID)
    if "MujocoHandBlockTouchSensorsEnv" not in str(spec.entry_point):
        raise RuntimeError("registered hard-gate entry point changed")
    env = gym.make(
        ENV_ID, render_mode="rgb_array", width=320, height=240, max_episode_steps=121
    )
    observation, reset_info = env.reset(seed=RESET_SEED)
    shapes = {
        "action": list(env.action_space.shape),
        "observation": list(observation["observation"].shape),
        "achieved_goal": list(observation["achieved_goal"].shape),
        "desired_goal": list(observation["desired_goal"].shape),
    }
    if shapes != {
        "action": [20],
        "observation": [153],
        "achieved_goal": [7],
        "desired_goal": [7],
    }:
        raise RuntimeError(f"registered environment shapes changed: {shapes}")
    rollout = _rollout(env, observation)
    raw = env.unwrapped
    substeps = ROLLOUT_STEPS * int(raw.n_substeps)
    if substeps != 2400:
        raise RuntimeError(f"expected 2,400 MuJoCo substeps, observed {substeps}")
    libraries = _graphics_libraries()
    mujoco_libraries = sorted(
        Path(mujoco.__file__).resolve().parent.rglob("libmujoco.so*")
    )
    if len(mujoco_libraries) != 1:
        raise RuntimeError("expected exactly one pinned MuJoCo shared library")
    result: dict[str, Any] = {
        "solution": "gymnasium-robotics",
        "capability": ENV_ID,
        "capabilities_exercised": [
            "registered_shadow_hand_environment",
            "mujoco_physics_steps",
            "continuous_touch_sensor_response",
            "mujoco_contacts",
            "egl_rgb_rendering",
            "rtx_pro_6000_blackwell_execution",
        ],
        "source": {
            "repository": source_lock["components"]["farama_gymnasium_robotics"][
                "repository"
            ],
            "commit": EXPECTED_SOURCE,
            "package_version": gymnasium_robotics.__version__,
            "license_sha256": {
                "LICENSE": source_lock["components"]["farama_gymnasium_robotics"][
                    "license_sha256"
                ],
                "gymnasium_robotics/envs/assets/LICENSE.md": asset_lock[
                    "asset_notice_sha256"
                ],
            },
            "directly_loaded_xml_sha256": xml,
            "directly_loaded_mesh_texture_sha256": material,
            "asset_notice_sha256": asset_lock["asset_notice_sha256"],
        },
        "mujoco": {
            "python_version": mujoco.__version__,
            "native_version": mujoco.mj_versionString(),
            "release_commit": EXPECTED_MUJOCO_COMMIT,
            "wheel_sha256": EXPECTED_MUJOCO_WHEEL,
            "license_sha256": source_lock["components"]["mujoco"]["license_sha256"],
            "third_party_notices_sha256": source_lock["components"]["mujoco"][
                "third_party_notices_sha256"
            ],
            "shared_library": str(mujoco_libraries[0]),
            "shared_library_sha256": _sha256(mujoco_libraries[0]),
            "license": "Apache-2.0",
        },
        "environment": {
            "id": ENV_ID,
            "entry_point": str(spec.entry_point),
            "reset_seed": RESET_SEED,
            "action_seed": ACTION_SEED,
            "action_shape": shapes["action"],
            "observation_shape": shapes["observation"],
            "achieved_goal_shape": shapes["achieved_goal"],
            "desired_goal_shape": shapes["desired_goal"],
            "reset_info_keys": sorted(map(str, reset_info)),
            "synthetic_only_fixture": False,
        },
        "physics": {
            "environment_steps": ROLLOUT_STEPS,
            "substeps_per_environment_step": int(raw.n_substeps),
            "physics_substeps": substeps,
            "simulated_seconds": float(raw.data.time),
            "contact_count_sum": sum(rollout["contacts"]),
            "contact_count_max": max(rollout["contacts"]),
            "steps_with_contacts": sum(value > 0 for value in rollout["contacts"]),
            "finite_reward_count": len(rollout["rewards"]),
            "reward_min": min(rollout["rewards"]),
            "reward_max": max(rollout["rewards"]),
            "step_calls_per_second": ROLLOUT_STEPS / rollout["step_seconds"],
            "physics_substeps_per_second": substeps / rollout["step_seconds"],
            "terminated": False,
            "truncated": False,
        },
        "touch_sensors": {
            "shape": [92],
            "nonzero_reading_count": sum(rollout["touch_nonzero"]),
            "steps_with_nonzero_readings": sum(
                value > 0 for value in rollout["touch_nonzero"]
            ),
            "max_reading": max(rollout["touch_maxima"]),
        },
        "state_transition": {
            "initial_object_pose": rollout["initial_goal"].tolist(),
            "final_object_pose": rollout["final_goal"].tolist(),
            "object_position_delta_m": float(
                np.linalg.norm(rollout["final_goal"][:3] - rollout["initial_goal"][:3])
            ),
            "object_orientation_delta_rad": _quaternion_angle(
                rollout["initial_goal"][3:], rollout["final_goal"][3:]
            ),
            "full_qpos_delta_l2": float(
                np.linalg.norm(rollout["final_qpos"] - rollout["initial_qpos"])
            ),
            "max_object_orientation_delta_rad": max(rollout["orientation_changes"]),
            "max_object_position_delta_m": max(rollout["position_changes"]),
            "max_full_qpos_delta_l2": max(rollout["state_changes"]),
        },
        "rendering": {
            "renderer": f"{type(raw.mujoco_renderer).__module__}.{type(raw.mujoco_renderer).__name__}",
            "backend": "egl",
            "rgb_frame_count": len(rollout["frame_hashes"]),
            "rgb_frame_shapes": sorted(
                {tuple(shape) for shape in rollout["frame_shapes"]}
            ),
            "rgb_frame_sha256": rollout["frame_hashes"],
            "distinct_rgb_frame_sha256": sorted(set(rollout["frame_hashes"])),
            "render_calls_per_second": len(rollout["frame_hashes"])
            / rollout["render_seconds"],
            "render_seconds": rollout["render_seconds"],
            "loaded_gl_egl_libraries": libraries,
        },
        "runtime": {
            "expected_image_digest": expected,
            "pod_observed_image_digest": observed,
            "gpu_count": 1,
            "gpus": [gpu],
            "exit_status": 0,
        },
    }
    env.close()
    result["artifact"] = {
        "filename": "gymnasium-robotics-smoke.json",
        "media_type": "application/json",
        "sha256": "0" * 64,
        "sha256_scope": "canonical JSON with artifact.sha256 replaced by 64 zeroes",
        "size_bytes": 0,
    }
    output_dir = Path(os.environ["NPA_SMOKE_OUTPUT_DIR"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "gymnasium-robotics-smoke.json"
    for _ in range(8):
        encoded = _canonical_bytes(result)
        if result["artifact"]["size_bytes"] == len(encoded):
            break
        result["artifact"]["size_bytes"] = len(encoded)
    else:
        raise RuntimeError("artifact byte size did not stabilize")
    normalized = _canonical_bytes(result)
    result["artifact"]["sha256"] = hashlib.sha256(normalized).hexdigest()
    final = _canonical_bytes(result)
    if len(final) != result["artifact"]["size_bytes"]:
        raise RuntimeError("artifact size changed after hash insertion")
    output.write_bytes(final)
    print(final.decode(), end="", flush=True)


if __name__ == "__main__":
    main()
