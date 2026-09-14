"""Select Franka PPO weights on validation and evaluate untouched paired physics shifts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from npa.workflows.franka_rl_environment import build_runner, environment_config, load_checkpoint, physics_evidence
from npa.workflows.franka_rl_metrics import PlacementMetrics, rank_checkpoint
from npa.workflows.lerobot_transfer_data import file_sha256, write_json


def _observe(env) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch
    from isaaclab.utils.math import combine_frame_transforms

    scene = env.unwrapped.scene
    robot = scene["robot"].data
    obj = scene["object"].data
    command = env.unwrapped.command_manager.get_command("object_pose")
    goal, _ = combine_frame_transforms(robot.root_pos_w, robot.root_quat_w, command[:, :3])
    values = (torch.linalg.vector_norm(obj.root_pos_w - goal, dim=1),
              torch.linalg.vector_norm(obj.root_lin_vel_w, dim=1),
              obj.root_pos_w[:, 2] - scene.env_origins[:, 2])
    return tuple(value.detach().cpu().numpy() for value in values)


def _rows(metrics: PlacementMetrics, *, split: str, condition: str, checkpoint: Path,
          seed: int, iteration: int, initial_hashes: list[str]) -> list[dict]:
    digest = file_sha256(checkpoint)
    return [{"split": split, "condition": condition, "checkpoint_sha256": digest,
             "iteration": iteration, "reset_seed": seed, "env_index": index,
             "initial_state_sha256": initial_hashes[index],
             "success": bool(metrics.success[index]), "lifted": bool(metrics.lifted[index]),
             "closest_distance_m": float(metrics.closest[index]),
             "longest_stable_steps": int(metrics.longest[index]), "steps": int(metrics.steps[index])}
            for index in range(len(metrics.success))]


def _initial_state_hashes(env) -> list[str]:
    import torch

    scene = env.unwrapped.scene
    state = torch.cat([scene["robot"].data.joint_pos, scene["object"].data.root_pos_w,
                       scene["object"].data.root_quat_w,
                       env.unwrapped.command_manager.get_command("object_pose")], dim=1)
    return [hashlib.sha256(row.tobytes()).hexdigest() for row in state.detach().cpu().numpy()]


def _rollout(checkpoint: Path, recipe: dict, *, split: str, condition: str,
             seed: int, iteration: int) -> tuple[list[dict], dict]:
    import gymnasium as gym
    import torch
    from isaaclab.utils.seed import configure_seed

    configure_seed(seed)
    config = environment_config(recipe, training=False, condition=condition)
    config.seed = seed
    env = gym.make(recipe["task"], cfg=config)
    try:
        wrapped, runner, _ = build_runner(env, recipe)
        load_checkpoint(runner, checkpoint)
        policy = runner.get_inference_policy(device="cuda:0")
        obs, _ = wrapped.reset()
        initial_hashes = _initial_state_hashes(env)
        metrics = PlacementMetrics(recipe["eval_episodes"], recipe)
        metrics.closest = _observe(env)[0]
        previous = torch.zeros((wrapped.num_envs, wrapped.num_actions), device="cuda:0")
        with torch.inference_mode():
            for _ in range(recipe["episode_steps"]):
                action = policy(obs)
                applied = previous if recipe["conditions"][condition]["action_delay"] else action
                obs, _, done, _ = wrapped.step(applied)
                metrics.update(*_observe(env), done.cpu().numpy())
                previous = action.clone()
        rows = _rows(metrics, split=split, condition=condition, checkpoint=checkpoint,
                     seed=seed, iteration=iteration, initial_hashes=initial_hashes)
        return rows, physics_evidence(env)
    finally:
        env.close()


def _validation(training: Path, output: Path, recipe: dict) -> tuple[Path, list[dict]]:
    candidates = sorted((training / "checkpoints").glob("model_*.pt"),
                        key=lambda path: int(path.stem.split("_")[-1]))
    if not candidates:
        raise ValueError("Franka training contains no PPO checkpoints")
    all_rows, ranked = [], []
    for checkpoint in candidates:
        iteration = int(checkpoint.stem.split("_")[-1])
        rows, physics = _rollout(checkpoint, recipe, split="validation", condition="nominal",
                                 seed=recipe["validation_seed"], iteration=iteration)
        ranked.append((rank_checkpoint(rows), checkpoint))
        all_rows.extend(rows)
        write_json(output / f"validation/iteration-{iteration}.json", {"trials": rows, "physics": physics})
        print(f"Validated PPO iteration {iteration}: {sum(row['success'] for row in rows)}/{len(rows)}", flush=True)
    return max(ranked, key=lambda item: item[0])[1], all_rows


def _test(training: Path, selected: Path, output: Path, recipe: dict) -> list[dict]:
    all_rows = []
    for arm, checkpoint in (("initial", training / "initial.pt"), ("trained", selected)):
        iteration = -1 if arm == "initial" else int(checkpoint.stem.split("_")[-1])
        for condition in recipe["conditions"]:
            rows, physics = _rollout(checkpoint, recipe, split="test", condition=condition,
                                     seed=recipe["test_seed"], iteration=iteration)
            for row in rows:
                row["arm"] = arm
            all_rows.extend(rows)
            write_json(output / f"test/{arm}-{condition}.json", {"trials": rows, "physics": physics})
            print(f"Test {arm}/{condition}: {sum(row['success'] for row in rows)}/{len(rows)}", flush=True)
    return all_rows


def evaluate_checkpoints(training: Path, output: Path, recipe: dict, runtime: dict) -> None:
    """Select on validation before opening gold resets, then export separate capture episodes.

    Args:
        training: Verified stage containing native PPO checkpoints.
        output: New evaluation artifact directory.
        recipe: Sealed experiment protocol.
        runtime: Measured runtime and accelerator versions.
    Returns:
        None.
    Raises:
        ValueError: Training evidence or checkpoint selection is invalid.
        RuntimeError: A real policy, simulator, or camera fails.
        OSError: Artifacts cannot be written.
    """
    from npa.workflows.franka_rl_capture import capture_policy

    manifest = json.loads((training / "training.json").read_text())
    if manifest["recipe"] != recipe:
        raise ValueError("Training and evaluation recipes differ")
    for relative, digest in manifest["checkpoints"].items():
        if file_sha256(training / relative) != digest:
            raise ValueError("Franka checkpoint bytes changed before evaluation")
    selected, validation = _validation(training, output, recipe)
    write_json(output / "selection.json", {
        "selected_checkpoint_sha256": file_sha256(selected), "iteration": int(selected.stem.split("_")[-1]),
        "selection_rule": "validation success, then closest distance, then earlier iteration",
        "test_data_used": False,
    })
    trials = _test(training, selected, output, recipe)
    shutil.copy2(selected, output / "selected.pt")
    capture_policy(selected, output / "trajectories", recipe)
    write_json(output / "evaluation.json", {
        "schema": "npa.franka-rl.evaluation.v1", "recipe": recipe, "runtime": runtime,
        "validation": validation, "trials": trials, "training": manifest,
        "selection": json.loads((output / "selection.json").read_text()),
        "physical_robot_tested": False,
    })
