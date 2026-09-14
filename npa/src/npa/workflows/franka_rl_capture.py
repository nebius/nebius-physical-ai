"""Record synchronized Franka state, applied policy actions, and genuine RTX pixels."""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path

import numpy as np

from npa.workflows.franka_rl_environment import build_runner, environment_config, load_checkpoint, physics_evidence
from npa.workflows.franka_rl_metrics import PlacementMetrics
from npa.workflows.lerobot_transfer_data import file_sha256, write_json


def _orient_camera(env) -> None:
    import torch

    scene = env.unwrapped.scene
    origin = scene.env_origins[:1]
    eye = torch.tensor([[1.5, -1.5, 1.3]], device=origin.device) + origin
    target = torch.tensor([[0.45, 0.0, 0.25]], device=origin.device) + origin
    scene["npa_rollout_camera"].set_world_poses_from_view(eyes=eye, targets=target)
    env.unwrapped.sim.render()


def _frame(env) -> np.ndarray:
    pixels = env.unwrapped.scene["npa_rollout_camera"].data.output["rgb"].torch[0, ..., :3]
    frame = pixels.detach().cpu().numpy()
    if frame.shape != (480, 640, 3) or frame.dtype != np.uint8:
        raise RuntimeError(f"Invalid Franka RTX frame: {frame.shape}, {frame.dtype}")
    return frame.copy()


def _capture_episode(wrapped, policy, output: Path, recipe: dict, *, condition: str = "nominal",
                     episode_index: int = 0) -> dict:
    import torch

    from npa.workflows.franka_rl_eval import _initial_state_hashes, _observe

    states, actions, frames, geometry = [], [], [], []
    with torch.inference_mode():
        # Isaac can retain tensors created by the preceding inference rollout.
        wrapped.unwrapped.seed(recipe["capture_seed"] + episode_index)
        obs, _ = wrapped.reset()
        initial_hash = _initial_state_hashes(wrapped)[0]
        _orient_camera(wrapped)
        metrics = PlacementMetrics(1, recipe)
        metrics.closest = _observe(wrapped)[0]
        previous = torch.zeros((1, wrapped.num_actions), device=wrapped.device)
        for _ in range(recipe["episode_steps"]):
            action = policy(obs)
            if wrapped.clip_actions is not None:
                action = action.clamp(-wrapped.clip_actions, wrapped.clip_actions)
            applied = previous if recipe["conditions"][condition]["action_delay"] else action
            states.append(wrapped.unwrapped.scene["robot"].data.joint_pos.torch[0].cpu().numpy().copy())
            actions.append(applied[0].cpu().numpy().copy())
            frames.append(_frame(wrapped))
            geometry.append([value[0] for value in _observe(wrapped)])
            obs, _, done, _ = wrapped.step(applied)
            metrics.update(*_observe(wrapped), done.cpu().numpy())
            previous = action.clone()
            if bool(done[0]):
                break
    return _save_episode(output, states, actions, frames, geometry, metrics, initial_hash)


def _save_episode(output, states, actions, frames, geometry, metrics, initial_hash) -> dict:
    output.mkdir(parents=True)
    for name, values in (("state", states), ("actions", actions), ("rgb", frames), ("object_metrics", geometry)):
        np.save(output / f"{name}.npy", np.stack(values))
    rgb = np.stack(frames)
    if np.mean(np.ptp(rgb.astype(np.int16), axis=(1, 2, 3)) >= 8) < 0.9:
        raise RuntimeError("Franka RTX camera produced blank frames")
    if len(frames) > 1 and not np.any(np.diff(rgb.astype(np.int16), axis=0)):
        raise RuntimeError("Franka RTX recording contains no temporal change")
    return {"length": len(states), "success": bool(metrics.success[0]),
            "lifted": bool(metrics.lifted[0]), "longest_stable_steps": int(metrics.longest[0]),
            "closest_distance_m": float(metrics.closest[0]), "rgb_frame_count": len(frames),
            "initial_state_sha256": initial_hash}


def capture_policy(checkpoint: Path, output: Path, recipe: dict, *, arm: str = "trained",
                   condition: str = "nominal") -> None:
    """Export a trained policy on separate capture resets with measured success labels.

    Args:
        checkpoint: Exact validation-selected PPO weights.
        output: New numpy trajectory directory.
        recipe: Sealed experiment, camera, and capture settings.
        arm: Initial or validation-selected policy identity, withheld from the VLM.
        condition: Actual mass, friction, and actuation condition to capture.
    Returns:
        None.
    Raises:
        RuntimeError: Policy loading or genuine RGB capture fails.
        OSError: Trajectory arrays cannot be written.
    """
    import gymnasium as gym
    from isaaclab.utils.seed import configure_seed

    configure_seed(recipe["capture_seed"])
    config = environment_config(recipe, training=False, capture=True, condition=condition, asset_root=checkpoint.parent)
    config.seed = recipe["capture_seed"]
    env = gym.make(recipe["task"], cfg=config)
    try:
        wrapped, runner, _ = build_runner(env, recipe)
        load_checkpoint(runner, checkpoint)
        policy = runner.get_inference_policy(device="cuda:0")
        episodes = [_capture_episode(wrapped, policy, output / f"episode_{index:06d}", recipe,
                                     condition=condition, episode_index=index)
                    for index in range(recipe["capture_episodes"])]
        for index, row in enumerate(episodes):
            row.update(arm=arm, condition=condition, reset_seed=recipe["capture_seed"] + index,
                       capture_index=index, checkpoint_sha256=file_sha256(checkpoint))
        _write_metadata(env, checkpoint, output, recipe, episodes)
    finally:
        env.close()


def _write_metadata(env, checkpoint: Path, output: Path, recipe: dict, episodes: list[dict]) -> None:
    joint_names = list(env.unwrapped.scene["robot"].joint_names)
    step_dt = float(env.unwrapped.step_dt)
    write_json(output / "meta.json", {
        "format": "npa_isaac_lab_rollout_v2", "task": recipe["task"], "robot_type": "franka", "run_id": recipe["run_id"],
        "policy_loaded": True, "runtime_version": version("isaaclab"),
        "checkpoint_sha256": file_sha256(checkpoint), "fps": 1.0 / step_dt,
        "assets": recipe.get("assets"),
        "physics": physics_evidence(env),
        "task_description": "Lift the " + recipe.get("assets", {}).get("description", "cube") + " and hold it steady",
        "control_dt": step_dt, "source_joint_names": joint_names, "state_names": joint_names,
        "action_names": [f"panda_joint{i}_normalized_target" for i in range(1, 8)] + ["gripper_open_close"],
        "action_semantics": "Isaac joint position: default offset + 0.5 * arm action; binary gripper",
        "num_episodes": len(episodes), "episode_lengths": [row["length"] for row in episodes],
        "total_frames": sum(row["length"] for row in episodes), "episode_results": episodes,
        "capture_seed": recipe["capture_seed"], "source_split": "capture",
        "rgb_enabled": True, "rgb_frame_count": sum(row["length"] for row in episodes),
        "rgb_dimensions": [480, 640, 3], "genuine_simulator_pixels": True,
        "renderer": "isaac_sim_tiled_camera_rtx", "timeline": "episode_index/frame_index/timestamp",
        "physical_robot_tested": False, "expert_success_assumed": False,
    })
