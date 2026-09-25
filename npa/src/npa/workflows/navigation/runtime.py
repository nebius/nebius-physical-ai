"""Train and evaluate an operator-owned navigation task inside native Isaac Lab."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from npa.workflows.navigation.artifacts import file_sha256, write_json
from npa.workflows.navigation.contract import read_recipe
from npa.workflows.navigation.measure import (
    episode_rows,
    probe_isolation,
    snapshot,
    verify_reset,
)
from npa.workflows.navigation.native import (
    build_runner,
    inspect_scene,
    parameters,
    policy_state_digest,
    task_adapter,
    validate_scene_package,
)


def _train(env, wrapped, runner, adapter, recipe, output, source=None):
    import torch
    from npa.workflows.navigation.initialization import initialize_runner
    from npa.workflows.navigation.learning_evidence import learn

    initialization = initialize_runner(runner, recipe, source)
    before = parameters(runner)
    probe = _probes(adapter, env, wrapped, recipe, output)
    env.reset(seed=recipe.train_cases[0].seed)
    performance = learn(runner, env, recipe, output)
    delta = float(torch.linalg.vector_norm(parameters(runner) - before))
    if not 0 < delta < float("inf"):
        raise RuntimeError("native learner produced no finite policy parameter update")
    checkpoint = output / "policy.pt"
    runner.save(str(checkpoint))
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise RuntimeError("native learner did not write a checkpoint")
    return {
        "schema": "npa.navigation.training.v1",
        "iterations": recipe.iterations,
        "policy_parameter_delta_l2": delta,
        "checkpoint_sha256": file_sha256(checkpoint),
        "isolation": probe,
        "heldout_used_for_training": False,
        "initialization": initialization,
        "performance": performance,
    }


def _evaluate(env, wrapped, runner, adapter, recipe, source, output):
    from npa.workflows.navigation.initialization import load_native_checkpoint

    training = json.loads((source / "training.json").read_text())
    checkpoint = source / "policy.pt"
    if file_sha256(checkpoint) != training["checkpoint_sha256"]:
        raise ValueError("evaluation checkpoint SHA-256 differs from training evidence")
    _verify_training_binding(training, recipe, source)
    load_native_checkpoint(runner, checkpoint)
    return _score_checkpoint(env, wrapped, runner, adapter, recipe, checkpoint, output)


def _score_checkpoint(env, wrapped, runner, adapter, recipe, checkpoint, output):
    from npa.workflows.navigation.rollout_video import recording

    before = policy_state_digest(runner)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    probe = _probes(adapter, env, wrapped, recipe, output)
    with recording(env, recipe, output) as frame:
        trajectory = _rollout(env, wrapped, policy, adapter, recipe, frame)
    if before != policy_state_digest(runner):
        raise RuntimeError(
            "evaluation changed policy parameters or normalization buffers"
        )
    rows = episode_rows(recipe.eval_cases, trajectory, recipe.goal_tolerance_m)
    rate = sum(row["success"] for row in rows) / len(rows)
    write_json(
        output / "trajectory.json",
        [{k: v.tolist() for k, v in s.items()} for s in trajectory],
    )
    return {
        "schema": "npa.navigation.evaluation.v1",
        "policy_loaded": True,
        "checkpoint_sha256": file_sha256(checkpoint),
        "isolation": probe,
        "episodes": rows,
        "success_rate": rate,
        "passed": rate >= recipe.minimum_success_rate,
        "evaluation_inputs_sha256": _case_digest(recipe),
    }


def _verify_training_binding(training, recipe, source):
    expected = {
        "schema": "npa.navigation.training.v1",
        "task": recipe.task,
        "image": recipe.image,
        "adapter_sha256": recipe.adapter_sha256,
        "recipe_sha256": file_sha256(source / "recipe.json"),
    }
    if any(training.get(key) != value for key, value in expected.items()):
        raise ValueError(
            "training evidence does not bind this task/image/adapter/recipe"
        )
    if training.get("runtime", {}).get("scene_sha256") != recipe.scene_sha256:
        raise ValueError("training evidence does not bind the evaluated scene")


def _case_digest(recipe):
    import hashlib

    data = json.dumps([c.model_dump() for c in recipe.eval_cases], sort_keys=True)
    return hashlib.sha256(data.encode()).hexdigest()


def _rollout(env, wrapped, policy, adapter, recipe, frame=None):
    import torch

    initial = verify_reset(adapter, env, recipe.eval_cases, recipe.probe.tolerance)
    active = np.ones(recipe.num_envs, dtype=bool)
    trajectory = [{**initial, "active": active.copy()}]
    if frame:
        frame(initial, 0)
    obs = wrapped.get_observations()
    for _ in range(recipe.episode_steps):
        with torch.inference_mode():
            actions = policy(obs)
            if not torch.isfinite(actions).all():
                raise ValueError("checkpoint policy produced nonfinite actions")
            obs, _, done, _ = wrapped.step(actions)
        if bool(done.any()):
            raise ValueError(
                "evaluation task auto-reset; adapter must disable evaluation terminations"
            )
        state = snapshot(adapter, env, recipe.num_envs)
        if frame:
            frame(state, len(trajectory))
        if not np.allclose(
            state["goal_m"], initial["goal_m"], rtol=0, atol=recipe.probe.tolerance
        ):
            raise ValueError("held-out navigation goal changed during evaluation")
        trajectory.append({**state, "active": active.copy()})
        reached = np.linalg.norm(state["position_m"][:, :2] - state["goal_m"], axis=1)
        active &= (reached > recipe.goal_tolerance_m) & (state["obstacle_contact"] == 0)
        active &= state["peer_contact"] == 0
        active &= state["physical_failure"] == 0
        if not active.any():
            break
    return trajectory


def _install_visibility_guard(adapter, env, recipe):
    if recipe.sensor_mode != "rgbd":
        return
    from npa.workflows.navigation.visibility import hide_robot_geometry

    hide_robot_geometry(adapter, env, recipe)
    native = env.unwrapped
    original = native.step

    def checked_step(actions):
        hide_robot_geometry(adapter, env, recipe, author=False)
        result = original(actions)
        hide_robot_geometry(adapter, env, recipe, author=False)
        return result

    native.step = checked_step
    native.npa_navigation_visibility_check = lambda: hide_robot_geometry(
        adapter, env, recipe, author=False
    )


def _probes(adapter, env, wrapped, recipe, output):
    with adapter.probe_mode(env):
        result = probe_isolation(adapter, env, wrapped, recipe)
        if recipe.sensor_mode == "rgbd":
            from npa.workflows.navigation.cameras import probe_cameras

            result["camera"] = probe_cameras(adapter, env, recipe, output)
            result["camera_isolation_verified"] = True
    return result


def _execute(args, recipe, adapter, config):
    import gymnasium as gym

    env = gym.make(recipe.task, cfg=config)
    try:
        runtime = inspect_scene(env, recipe, args.input_path / recipe.scene_file)
        _install_visibility_guard(adapter, env, recipe)
        wrapped, runner, settings = build_runner(
            env,
            recipe,
            args.output_path / "checkpoints" if args.stage == "train" else None,
        )
        evidence = _run_environment_stage(
            args, recipe, env, wrapped, runner, adapter, settings
        )
        _bind_evidence(evidence, runtime, recipe, args.input_path)
        if args.stage == "train":
            write_json(args.output_path / "agent.json", settings)
        name = "training.json" if args.stage == "train" else "evaluation.json"
        write_json(args.output_path / name, evidence)
    finally:
        env.close()


def _run_environment_stage(args, recipe, env, wrapped, runner, adapter, settings):
    if args.stage == "train":
        return _train(
            env, wrapped, runner, adapter, recipe, args.output_path, args.input_path
        )
    if args.stage == "evaluate-checkpoint":
        return _evaluate_initial(
            env, wrapped, runner, adapter, recipe, args.input_path, args.output_path
        )
    if json.loads((args.input_path / "agent.json").read_text()) != settings:
        raise ValueError("evaluation learner config differs from trained policy")
    return _evaluate(
        env, wrapped, runner, adapter, recipe, args.input_path, args.output_path
    )


def _bind_evidence(evidence, runtime, recipe, source):
    evidence.update(
        runtime=runtime,
        task=recipe.task,
        image=recipe.image,
        adapter_sha256=recipe.adapter_sha256,
        source_bundle_sha256=recipe.source_bundle_sha256,
        runtime_source_sha256=file_sha256(Path(__file__)),
        workbench_sources={
            p.name: file_sha256(p) for p in Path(__file__).parent.glob("*.py")
        },
        recipe_sha256=file_sha256(source / "recipe.json"),
    )


def _evaluate_initial(env, wrapped, runner, adapter, recipe, source, output):
    from npa.workflows.navigation.initialization import initialize_runner

    if recipe.initial_checkpoint is None:
        raise ValueError("evaluate-checkpoint requires a sealed initial_checkpoint")
    initialization = initialize_runner(runner, recipe, source)
    checkpoint = source / recipe.initial_checkpoint.file
    evidence = _score_checkpoint(
        env, wrapped, runner, adapter, recipe, checkpoint, output
    )
    evidence["initialization"] = initialization
    return evidence


def main(argv: list[str] | None = None) -> int:
    """Launch the installed Isaac runtime and execute one stateless GPU stage.

    Args:
        argv: Stage, materialized bundle and output arguments.
    Returns:
        Zero after native completion; qualification is recorded separately.
    Raises:
        ValueError: Input, shared scene, isolation or checkpoint checks fail.
        RuntimeError: Isaac or RSL-RL execution fails.
        ImportError: Required native dependencies are missing.
    """
    from isaaclab_tasks.utils import launch_simulation
    from isaaclab.utils.seed import configure_seed

    args = _arguments(argv)
    recipe = read_recipe(args.input_path)
    validate_scene_package(args.input_path / recipe.scene_file)
    adapter = task_adapter(recipe)
    training = args.stage == "train"
    cases = recipe.train_cases if training else recipe.eval_cases
    configure_seed(cases[0].seed)
    config = adapter.configure(
        task=recipe.task,
        scene_file=str(args.input_path / recipe.scene_file),
        scene_prim=recipe.scene_prim,
        num_envs=recipe.num_envs,
        cases=[case.model_dump() for case in cases],
        training=training,
    )
    args.enable_cameras = recipe.sensor_mode == "rgbd" or (
        not training and recipe.adapter_module == "npa.workflows.navigation.reference"
    )
    args.output_path.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.input_path / "recipe.json", args.output_path / "recipe.json")
    with launch_simulation(config, args):
        _execute(args, recipe, adapter, config)
    return 0


def _arguments(argv):
    from isaaclab_tasks.utils import add_launcher_args

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "evaluate", "evaluate-checkpoint"))
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    add_launcher_args(parser)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
