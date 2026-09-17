"""Execute native Isaac Lab PPO or evaluation within one workflow-owned GPU task."""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import time

from npa.workflows.lerobot_transfer_data import file_sha256, write_json


def _runtime_versions() -> dict:
    import torch

    return {"isaaclab": version("isaaclab"), "rsl_rl": version("rsl-rl-lib"),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability())}


def _policy_parameters(runner):
    import torch

    return torch.cat([parameter.detach().flatten().cpu() for parameter in runner.alg.get_policy().parameters()])


def _save_initial(runner, path: Path) -> None:
    """Save native PPO state before its training logger has been initialized."""
    import torch

    payload = runner.alg.save()
    payload.update(iter=runner.current_learning_iteration, infos=None)
    torch.save(payload, path)


def _train(args, recipe: dict, config) -> None:
    import gymnasium as gym
    import torch

    from npa.workflows.franka_rl_environment import build_runner, physics_evidence
    from npa.workflows.franka_rl_learning import distribution_evidence
    from npa.workflows.franka_rl_validity import validity_evidence

    output = Path(args.output_path)
    env = gym.make(recipe["task"], cfg=config)
    try:
        wrapped, runner, agent = build_runner(env, recipe, output / "checkpoints")
        initial = output / "initial.pt"
        _save_initial(runner, initial)
        initial_distribution = distribution_evidence(runner, recipe)
        parameters_before = _policy_parameters(runner)
        applied = physics_evidence(env)
        started = time.monotonic()
        runner.learn(num_learning_iterations=recipe["iterations"], init_at_random_ep_len="learning" not in recipe)
        final = output / "checkpoints" / f"model_{recipe['iterations']}.pt"
        runner.save(str(final))
        torch.cuda.synchronize()
        checkpoints, parameter_delta = _verify_training_checkpoints(runner, parameters_before, initial, final)
        applied["simulation_validity"] = validity_evidence(env.unwrapped)
        write_json(output / "training.json", {
            "schema": "npa.franka-rl.training.v1", "recipe": recipe,
            "agent": agent, "runtime": _runtime_versions(), "physics": applied,
            "duration_seconds": time.monotonic() - started,
            "policy_parameter_delta_l2": parameter_delta,
            "transitions": recipe["iterations"] * recipe["num_envs"] * recipe["steps_per_env"],
            "checkpoints": {p.relative_to(output).as_posix(): file_sha256(p) for p in [initial, *checkpoints]},
            "policy_loaded": True, "physical_robot_tested": False,
            **({"exploration": {"initial": initial_distribution,
                                "final": distribution_evidence(runner, recipe)}} if initial_distribution else {}),
            **_learning_evidence(env),
        })
    finally:
        env.close()


def _verify_training_checkpoints(runner, before, initial, final):
    import torch

    delta = torch.linalg.vector_norm(_policy_parameters(runner) - before).item()
    if not 0 < delta < float("inf"):
        raise RuntimeError("Franka PPO policy parameters did not change finitely")
    checkpoints = sorted(final.parent.glob("model_*.pt"))
    if not checkpoints or file_sha256(initial) == file_sha256(final):
        raise RuntimeError("PPO did not produce a changed checkpoint")
    return checkpoints, delta


def _learning_evidence(env) -> dict:
    reward = getattr(env.unwrapped, "npa_learning_reward", None)
    if reward is None:
        return {}
    curriculum = reward.curriculum
    return {"learning": {"settings": reward.settings, "completed_episodes": reward.completed_episodes,
                         "strict_successes": reward.completed_successes,
                         "difficulty": curriculum.fraction, "history": curriculum.history,
                         "pending_window_episodes": curriculum.episodes,
                         "pending_window_successes": curriculum.successes,
                         "evaluation_used": False}}


def _execute_stage(args, recipe: dict, config) -> None:
    if args.stage == "train":
        _train(args, recipe, config)
    elif args.stage == "capture":
        from npa.workflows.franka_rl_capture import capture_policy

        filename = "initial.pt" if args.capture_arm == "initial" else "selected.pt"
        capture_policy(args.input_path / filename, args.output_path, recipe,
                       arm=args.capture_arm, condition=args.condition)
    else:
        from npa.workflows.franka_rl_eval import evaluate_checkpoints, validate_checkpoints

        operation = validate_checkpoints if args.stage == "validate" else evaluate_checkpoints
        operation(args.input_path, args.output_path, recipe, _runtime_versions())


def main(argv: list[str] | None = None) -> int:
    """Launch the pinned Isaac runtime and perform real simulation work.

    Args:
        argv: Native stage and Isaac launcher arguments.
    Returns:
        Zero after a complete native stage.
    Raises:
        RuntimeError: Runtime, training, checkpoint loading, or rendering fails.
        ValueError: The experiment configuration is invalid.
    """
    from isaaclab_tasks.utils import launch_simulation
    from isaaclab.utils.seed import configure_seed

    from npa.workflows.franka_rl_environment import environment_config
    from npa.workflows.franka_rl_validity import SimulationValidityError

    args = _arguments(argv)
    recipe = json.loads((args.input_path / "recipe.json").read_text())
    seed_key = {"train": "seed", "validate": "validation_seed", "test": "test_seed", "capture": "capture_seed"}
    configure_seed(recipe[seed_key[args.stage]])
    os.environ["OMNI_TELEMETRY_DISABLE_ANONYMOUS_DATA"] = "1"
    config = environment_config(recipe, training=args.stage == "train", capture=args.stage == "capture",
                                condition=args.condition, asset_root=args.input_path)
    config.seed = recipe[seed_key[args.stage]]
    args.enable_cameras = args.stage == "capture"
    args.output_path.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.input_path / "recipe.json", args.output_path / "recipe.json")
    if "assets" in recipe:
        shutil.copytree(args.input_path / "assets", args.output_path / "assets", dirs_exist_ok=True)
    try:
        with launch_simulation(config, args):
            _execute_stage(args, recipe, config)
    except SimulationValidityError as error:
        write_json(args.output_path / "simulation-validity-failure.json", error.evidence)
        raise
    return 0


def _arguments(argv):
    from isaaclab_tasks.utils import add_launcher_args

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "validate", "test", "capture"))
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--capture-arm", choices=("initial", "trained"), default="trained")
    parser.add_argument("--condition", default="nominal")
    add_launcher_args(parser)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
