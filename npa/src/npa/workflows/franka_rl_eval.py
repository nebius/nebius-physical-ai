"""Select Franka PPO weights on validation and evaluate untouched paired physics shifts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from npa.workflows.franka_rl_environment import (
    build_runner,
    environment_config,
    load_checkpoint,
    physics_evidence,
)
from npa.workflows.franka_rl_metrics import PlacementMetrics, rank_checkpoint
from npa.workflows.franka_rl_learning import (
    distribution_evidence,
    normalization_evidence,
)
from npa.workflows.franka_rl_validity import task_domain_exits
from npa.workflows.lerobot_transfer_data import file_sha256, write_json


def _observe(env) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch
    from isaaclab.utils.math import combine_frame_transforms

    scene = env.unwrapped.scene
    robot = scene["robot"].data
    obj = scene["object"].data
    command = env.unwrapped.command_manager.get_command("object_pose")
    goal, _ = combine_frame_transforms(
        robot.root_pos_w.torch, robot.root_quat_w.torch, command[:, :3]
    )
    values = (
        torch.linalg.vector_norm(obj.root_pos_w.torch - goal, dim=1),
        torch.linalg.vector_norm(obj.root_lin_vel_w.torch, dim=1),
        obj.root_pos_w.torch[:, 2] - scene.env_origins[:, 2],
    )
    return tuple(value.detach().cpu().numpy() for value in values)


def _rows(
    metrics: PlacementMetrics,
    *,
    split: str,
    condition: str,
    checkpoint: Path,
    seed: int,
    iteration: int,
    initial_hashes: list[str],
) -> list[dict]:
    digest = file_sha256(checkpoint)
    return [
        {
            "split": split,
            "condition": condition,
            "checkpoint_sha256": digest,
            "iteration": iteration,
            "reset_seed": seed,
            "env_index": index,
            "initial_state_sha256": initial_hashes[index],
            "success": bool(metrics.success[index]),
            "lifted": bool(metrics.lifted[index]),
            "task_domain_exit": bool(metrics.domain_exit[index]),
            "closest_distance_m": float(metrics.closest[index]),
            "longest_stable_steps": int(metrics.longest[index]),
            "steps": int(metrics.steps[index]),
        }
        for index in range(len(metrics.success))
    ]


def _initial_state_hashes(env) -> list[str]:
    import torch

    scene = env.unwrapped.scene
    state = torch.cat(
        [
            scene["robot"].data.joint_pos.torch,
            scene["object"].data.root_pos_w.torch,
            scene["object"].data.root_quat_w.torch,
            env.unwrapped.command_manager.get_command("object_pose"),
        ],
        dim=1,
    )
    return [
        hashlib.sha256(row.tobytes()).hexdigest()
        for row in state.detach().cpu().numpy()
    ]


def _verify_frozen_inference(env, runner, recipe, normalization, distribution) -> None:
    if normalization != normalization_evidence(runner, recipe):
        raise RuntimeError("Evaluation changed the checkpoint observation normalizer")
    if distribution != distribution_evidence(runner, recipe):
        raise RuntimeError("Evaluation changed the checkpoint action distribution")
    env.unwrapped.npa_normalization_evidence = normalization
    env.unwrapped.npa_distribution_evidence = distribution


def _rollout(
    checkpoint: Path,
    recipe: dict,
    *,
    split: str,
    condition: str,
    seed: int,
    iteration: int,
) -> tuple[list[dict], dict]:
    import gymnasium as gym
    import torch
    from isaaclab.utils.seed import configure_seed

    configure_seed(seed)
    root = (
        checkpoint.parent.parent
        if checkpoint.parent.name == "checkpoints"
        else checkpoint.parent
    )
    config = environment_config(
        recipe, training=False, condition=condition, asset_root=root
    )
    config.seed = seed
    env = gym.make(recipe["task"], cfg=config)
    try:
        wrapped, runner, _ = build_runner(env, recipe)
        load_checkpoint(runner, checkpoint)
        policy = runner.get_inference_policy(device="cuda:0")
        normalization = normalization_evidence(runner, recipe)
        distribution = distribution_evidence(runner, recipe)
        obs, _ = wrapped.reset()
        initial_hashes = _initial_state_hashes(env)
        metrics = PlacementMetrics(recipe["eval_episodes"], recipe)
        metrics.closest = _observe(env)[0]
        previous = torch.zeros((wrapped.num_envs, wrapped.num_actions), device="cuda:0")
        with torch.inference_mode():
            for _ in range(recipe["episode_steps"]):
                action = policy(obs)
                applied = (
                    previous
                    if recipe["conditions"][condition]["action_delay"]
                    else action
                )
                obs, _, done, _ = wrapped.step(applied)
                metrics.update(
                    *_observe(env),
                    done.cpu().numpy(),
                    task_domain_exits(env.unwrapped).cpu().numpy(),
                )
                previous = action.clone()
        _verify_frozen_inference(env, runner, recipe, normalization, distribution)
        rows = _rows(
            metrics,
            split=split,
            condition=condition,
            checkpoint=checkpoint,
            seed=seed,
            iteration=iteration,
            initial_hashes=initial_hashes,
        )
        physics = physics_evidence(env)
        for row in rows:
            row["simulation_validity"] = physics["simulation_validity"]
        return rows, physics
    finally:
        env.close()


def _validation(training: Path, output: Path, recipe: dict) -> tuple[Path, list[dict]]:
    candidates = sorted(
        (training / "checkpoints").glob("model_*.pt"),
        key=lambda path: int(path.stem.split("_")[-1]),
    )
    if not candidates:
        raise ValueError("Franka training contains no PPO checkpoints")
    all_rows, ranked = [], []
    for checkpoint in candidates:
        iteration = int(checkpoint.stem.split("_")[-1])
        rows, physics = _rollout(
            checkpoint,
            recipe,
            split="validation",
            condition="nominal",
            seed=recipe["validation_seed"],
            iteration=iteration,
        )
        ranked.append((rank_checkpoint(rows), checkpoint))
        all_rows.extend(rows)
        write_json(
            output / f"validation/iteration-{iteration}.json",
            {"trials": rows, "physics": physics},
        )
        print(
            f"Validated PPO iteration {iteration}: {sum(row['success'] for row in rows)}/{len(rows)}",
            flush=True,
        )
    return max(ranked, key=lambda item: item[0])[1], all_rows


def _test(training: Path, selected: Path, output: Path, recipe: dict) -> list[dict]:
    all_rows = []
    for arm, checkpoint in (
        ("initial", training / "initial.pt"),
        ("trained", selected),
    ):
        iteration = -1 if arm == "initial" else int(checkpoint.stem.split("_")[-1])
        for condition in recipe["conditions"]:
            rows, physics = _rollout(
                checkpoint,
                recipe,
                split="test",
                condition=condition,
                seed=recipe["test_seed"],
                iteration=iteration,
            )
            for row in rows:
                row["arm"] = arm
            all_rows.extend(rows)
            write_json(
                output / f"test/{arm}-{condition}.json",
                {"trials": rows, "physics": physics},
            )
            print(
                f"Test {arm}/{condition}: {sum(row['success'] for row in rows)}/{len(rows)}",
                flush=True,
            )
    return all_rows


def validate_checkpoints(
    training: Path, output: Path, recipe: dict, runtime: dict
) -> None:
    """Freeze a checkpoint selection using only the validation reset stream.

    Args:
        training: Verified stage containing native PPO checkpoints.
        output: New evaluation artifact directory.
        recipe: Sealed experiment protocol.
        runtime: Measured runtime and accelerator versions.
    Returns:
        None.
    Raises:
        ValueError: Training evidence or checkpoint selection is invalid.
        RuntimeError: A real policy or simulator fails.
        OSError: Artifacts cannot be written.
    """
    manifest = json.loads((training / "training.json").read_text())
    if manifest["recipe"] != recipe:
        raise ValueError("Training and evaluation recipes differ")
    for relative, digest in manifest["checkpoints"].items():
        if file_sha256(training / relative) != digest:
            raise ValueError("Franka checkpoint bytes changed before evaluation")
    selected, validation = _validation(training, output, recipe)
    write_json(
        output / "selection.json",
        {
            "selected_checkpoint_sha256": file_sha256(selected),
            "iteration": int(selected.stem.split("_")[-1]),
            "selection_rule": "validation success, then closest distance, then earlier iteration",
            "test_data_used": False,
        },
    )
    shutil.copy2(selected, output / "selected.pt")
    write_json(
        output / "validation.json",
        {
            "schema": "npa.franka-rl.validation.v1",
            "recipe": recipe,
            "runtime": runtime,
            "validation": validation,
            "training": manifest,
            "selection": json.loads((output / "selection.json").read_text()),
        },
    )


def evaluate_checkpoints(
    training: Path, output: Path, recipe: dict, runtime: dict
) -> None:
    """Evaluate a frozen validation selection on untouched paired physics shifts.

    Args:
        training: Verified stage containing native PPO checkpoints.
        output: Directory containing the completed validation and selection.
        recipe: Sealed experiment protocol.
        runtime: Measured runtime and accelerator versions.
    Returns:
        None.
    Raises:
        ValueError: Selection evidence or checkpoint bytes changed.
        RuntimeError: A real policy or simulator fails.
        OSError: Artifacts cannot be read or written.
    """
    validated = json.loads((output / "validation.json").read_text())
    selection = json.loads((output / "selection.json").read_text())
    selected = training / "checkpoints" / f"model_{selection['iteration']}.pt"
    if (
        validated["recipe"] != recipe
        or validated["selection"] != selection
        or selection["test_data_used"]
    ):
        raise ValueError("Franka validation selection changed before test")
    expected = selection["selected_checkpoint_sha256"]
    if (
        file_sha256(selected) != expected
        or file_sha256(output / "selected.pt") != expected
    ):
        raise ValueError("Selected Franka checkpoint bytes changed before test")
    trials = _test(training, selected, output, recipe)
    write_json(
        output / "evaluation.json",
        {
            "schema": "npa.franka-rl.evaluation.v1",
            "recipe": recipe,
            "runtime": runtime,
            "validation": validated["validation"],
            "trials": trials,
            "training": validated["training"],
            "selection": selection,
            "physical_robot_tested": False,
        },
    )
