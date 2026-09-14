"""Run stateless Franka PPO stages through the standard workflow runtime."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

from npa.workflows.lerobot_transfer_data import materialize, publish, write_json


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--run-id", required=True)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--iterations", type=int, default=1500)
    prepare.add_argument("--num-envs", type=int, default=4096)
    prepare.add_argument("--eval-episodes", type=int, default=128)
    prepare.add_argument("--minimum-success", type=float, default=0.7)
    for stage in ("train", "evaluate", "report"):
        command = commands.add_parser(stage)
        command.add_argument("--input-path", required=True)
        command.add_argument("--output-path", required=True)
    prepare.add_argument("--output-path", required=True)
    return parser


def _recipe(args: argparse.Namespace) -> dict:
    if min(args.iterations, args.num_envs, args.eval_episodes) <= 0 or args.seed < 0:
        raise ValueError("Iterations and environment counts must be positive; seed nonnegative")
    if not 0 < args.minimum_success <= 1:
        raise ValueError("Minimum success must be in (0, 1]")
    if args.seed in {100_000, 200_000, 300_000}:
        raise ValueError("Training seed overlaps a reserved evaluation or capture stream")
    return {
        "schema": "npa.franka-rl.recipe.v1", "task": "Isaac-Lift-Cube-Franka-v0", "run_id": args.run_id,
        "seed": args.seed, "iterations": args.iterations, "num_envs": args.num_envs,
        "eval_episodes": args.eval_episodes, "minimum_success": args.minimum_success,
        "steps_per_env": 24, "checkpoint_interval": max(1, args.iterations // 3),
        "validation_seed": 100_000, "test_seed": 200_000, "capture_seed": 300_000,
        "episode_steps": 250, "stable_steps": 20, "success_distance_m": 0.05,
        "maximum_object_speed_m_s": 0.03, "minimum_object_height_m": 0.1,
        "train_mass_range": [0.7, 1.3], "train_friction_range": [0.5, 1.5],
        "conditions": {
            "nominal": {"mass_scale": 1.0, "friction": 1.0, "action_delay": 0},
            "heavy": {"mass_scale": 2.0, "friction": 1.0, "action_delay": 0},
            "slippery": {"mass_scale": 1.0, "friction": 0.25, "action_delay": 0},
            "delay": {"mass_scale": 1.0, "friction": 1.0, "action_delay": 1},
        },
        "capture_episodes": 4,
        "reference_workflow": "workflows/main/sim2real.yaml",
        "physical_robot_tested": False,
    }


def _run_native(stage: str, prepared: Path, output: Path) -> None:
    interpreter = os.environ.get("ISAAC_LAB_PYTHON", "/isaac-sim/python.sh")
    argv = [interpreter, "-m", "npa.workflows.franka_rl_runtime", stage,
            "--input-path", str(prepared), "--output-path", str(output),
            "--visualizer", "none"]
    output.mkdir(parents=True)
    with (output / "runtime.log").open("w") as log:
        result = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=False)
    receipt = output / ("training.json" if stage == "train" else "evaluation.json")
    if result.returncode or not receipt.is_file():
        print((output / "runtime.log").read_text()[-12_000:], flush=True)
        raise RuntimeError(f"Franka {stage} failed with exit {result.returncode}; required {receipt.name}")
    metadata = json.loads(receipt.read_text())
    expected = "training" if stage == "train" else "evaluation"
    if metadata.get("schema") != f"npa.franka-rl.{expected}.v1":
        raise ValueError(f"Franka {stage} produced an invalid completion record")


def main(argv: list[str] | None = None) -> int:
    """Execute and publish one Franka workflow stage.

    Args:
        argv: Stage arguments; process arguments when omitted.
    Returns:
        Zero after verified artifact publication.
    Raises:
        ValueError: Recipe or input hashes are invalid.
        RuntimeError: Native simulation or training fails.
        OSError: Artifact exchange fails.
    """
    args = _parser().parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="franka-rl-") as temporary:
        workspace = Path(temporary)
        output = workspace / "output"
        if args.stage == "prepare":
            write_json(output / "recipe.json", _recipe(args))
        else:
            prepared = materialize(args.input_path, workspace / "input")
            if args.stage == "report":
                from npa.workflows.franka_rl_report import report_results

                report_results(prepared, output)
            else:
                _run_native(args.stage, prepared, output)
        publish(output, args.output_path)
        print(json.dumps({"stage": args.stage, "status": "complete"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
