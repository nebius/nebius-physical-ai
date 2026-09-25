"""Run stateless Franka PPO stages through the standard workflow runtime."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import uuid

from npa.workflows.lerobot_transfer_data import materialize, publish, write_json
from npa.workflows.franka_rl_embodiments import EMBODIMENTS, embodiment_profile
from npa.workflows.franka_rl_learning import LEARNING_RECIPES, learning_profile
from npa.workflows.franka_rl_validity import validity_contract
from npa.workflows.franka_rl_physics import stability_profile


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
    prepare.add_argument(
        "--asset", choices=("spool", "hex_nut", "bottle"), default="spool"
    )
    prepare.add_argument("--embodiment", choices=EMBODIMENTS, default="franka")
    prepare.add_argument(
        "--learning-recipe", choices=LEARNING_RECIPES, default="joint-baseline"
    )
    prepare.add_argument("--vlm-model", default="MiniMaxAI/MiniMax-M3")
    for stage in ("train", "evaluate", "visual-evaluate", "report"):
        command = commands.add_parser(stage)
        command.add_argument("--input-path", required=True)
        command.add_argument("--output-path", required=True)
        if stage == "report":
            command.add_argument("--vlm-path", required=True)
        if stage == "visual-evaluate":
            command.add_argument("--prior-judgments-path")
    prepare.add_argument("--output-path", required=True)
    return parser


def _recipe(args: argparse.Namespace) -> dict:
    if min(args.iterations, args.num_envs, args.eval_episodes) <= 0 or args.seed < 0:
        raise ValueError(
            "Iterations and environment counts must be positive; seed nonnegative"
        )
    if not 0 < args.minimum_success <= 1:
        raise ValueError("Minimum success must be in (0, 1]")
    if args.seed in {100_000, 200_000, 300_000}:
        raise ValueError(
            "Training seed overlaps a reserved evaluation or capture stream"
        )
    return {
        "schema": "npa.franka-rl.recipe.v1",
        "task": "Isaac-Lift-Cube-Franka-v0",
        "run_id": args.run_id,
        "embodiment": embodiment_profile(getattr(args, "embodiment", "franka")),
        "simulation_validity": validity_contract(),
        **(
            {"stability": stability_profile("ur10e-mimic-asset-v1")}
            if getattr(args, "embodiment", "franka") == "ur10e_robotiq85"
            else {}
        ),
        **_learning_settings(args),
        "seed": args.seed,
        "iterations": args.iterations,
        "num_envs": args.num_envs,
        "eval_episodes": args.eval_episodes,
        "minimum_success": args.minimum_success,
        "steps_per_env": 24,
        "checkpoint_interval": max(1, args.iterations // 3),
        "validation_seed": 100_000,
        "test_seed": 200_000,
        "capture_seed": 300_000,
        "episode_steps": 250,
        "stable_steps": 20,
        "success_distance_m": 0.05,
        "maximum_object_speed_m_s": 0.03,
        "minimum_object_height_m": 0.1,
        "train_mass_range": [0.7, 1.3],
        "train_friction_range": [0.5, 1.5],
        "physics_capacity": {"gpu_total_aggregate_pairs_capacity": 2**21},
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


def _learning_settings(args: argparse.Namespace) -> dict:
    settings = learning_profile(getattr(args, "learning_recipe", "joint-baseline"))
    return {"learning": settings} if settings else {}


def _prepare(args, output: Path) -> None:
    from npa.workbench.vlm_eval.temporal import RUBRIC, RUBRIC_VERSION
    from npa.workflows.franka_rl_assets import write_assets
    import hashlib

    recipe = _recipe(args)
    recipe["assets"] = write_assets(output, args.asset)
    recipe["visual_eval"] = {
        "model": args.vlm_model,
        "frame_count": 16,
        "rubric_version": RUBRIC_VERSION,
        "rubric_sha256": hashlib.sha256(RUBRIC.encode()).hexdigest(),
        "minimum_lift_agreement": 0.8,
        "role": "independent_capture_audit",
        "arms": ["initial", "trained"],
        "conditions": list(recipe["conditions"]),
    }
    write_json(output / "recipe.json", recipe)


def _native_step(stage: str, prepared: Path, output: Path, *extra: str) -> None:
    interpreter = os.environ.get("ISAAC_LAB_PYTHON", "/isaac-sim/python.sh")
    argv = [
        interpreter,
        "-m",
        "npa.workflows.franka_rl_runtime",
        stage,
        "--input-path",
        str(prepared),
        "--output-path",
        str(output),
        "--visualizer",
        "none",
        *extra,
    ]
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / ("runtime.log" if stage == "train" else f"{stage}.log")
    receipts = {
        "train": ("training.json", "schema", "npa.franka-rl.training.v1"),
        "validate": ("validation.json", "schema", "npa.franka-rl.validation.v1"),
        "test": ("evaluation.json", "schema", "npa.franka-rl.evaluation.v1"),
        "capture": ("meta.json", "format", "npa_isaac_lab_rollout_v2"),
    }
    filename, key, expected = receipts[stage]
    with log_path.open("w") as log:
        result = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=False)
    if any(
        "PhysX error:" in line or "simulation will miss interactions" in line
        for line in log_path.read_text().splitlines()
    ):
        raise RuntimeError(
            f"Franka {stage} reported invalid physics; inspect {log_path.name}"
        )
    receipt = output / filename
    if result.returncode or not receipt.is_file():
        print(log_path.read_text()[-12_000:], flush=True)
        raise RuntimeError(
            f"Franka {stage} failed with exit {result.returncode}; required {receipt.name}"
        )
    metadata = json.loads(receipt.read_text())
    if metadata.get(key) != expected:
        raise ValueError(f"Franka {stage} produced an invalid completion record")
    print(f"Native Franka {stage} completed", flush=True)


def _run_native(stage: str, prepared: Path, output: Path) -> None:
    if stage == "train":
        _native_step("train", prepared, output)
        return
    # Each reset stream owns a fresh simulator and Replicator graph.
    _native_step("validate", prepared, output)
    _native_step("test", prepared, output)
    recipe = json.loads((prepared / "recipe.json").read_text())
    if "visual_eval" not in recipe:
        _native_step("capture", output, output / "trajectories")
        return
    from npa.workflows.franka_rl_capture_merge import merge_captures

    shutil.copy2(prepared / "initial.pt", output / "initial.pt")
    for arm in recipe["visual_eval"]["arms"]:
        for condition in recipe["visual_eval"]["conditions"]:
            _native_step(
                "capture",
                output,
                output / "captures" / f"{arm}-{condition}",
                "--capture-arm",
                arm,
                "--condition",
                condition,
            )
    merge_captures(output, recipe)


def _run_stage(args, prepared: Path, output: Path, workspace: Path) -> None:
    try:
        _run_stage_payload(args, prepared, output, workspace)
    except Exception as error:
        _publish_stage_failure(output, args.output_path, error, stage=args.stage)
        raise


def _run_stage_payload(args, prepared: Path, output: Path, workspace: Path) -> None:
    if args.stage == "report":
        from npa.workflows.franka_rl_report import report_results

        visual = materialize(args.vlm_path, workspace / "visual")
        report_results(prepared, output, visual=visual)
    elif args.stage == "visual-evaluate":
        from npa.workflows.franka_rl_vlm import evaluate_captures

        previous = (
            materialize(args.prior_judgments_path, workspace / "previous")
            if args.prior_judgments_path
            else None
        )
        evaluate_captures(prepared, output, previous=previous)
    else:
        _run_native(args.stage, prepared, output)


def _publish_stage_failure(
    output: Path, destination: str, error: Exception, *, stage: str = "visual-evaluate"
) -> None:
    """Retain diagnostic bytes separately without claiming stage completion."""
    try:
        output.mkdir(parents=True, exist_ok=True)
        for name in (
            "training.json",
            "evaluation.json",
            "visual-evaluation.json",
            "report.json",
        ):
            (output / name).unlink(missing_ok=True)
        failure_destination = (
            destination.rstrip("/") + "-failures/" + uuid.uuid4().hex + "/"
        )
        write_json(
            output / "failure.json",
            {
                "schema": "npa.franka-rl.stage-failure.v1",
                "status": "failed",
                "stage": stage,
                "error_type": type(error).__name__,
                "completed_judgments": len(list(output.glob("episode-*/verdict.json"))),
                "complete_stage": False,
            },
        )
        publish(output, failure_destination)
        print(
            json.dumps({"status": "failed", "evidence_uri": failure_destination}),
            flush=True,
        )
    except Exception as publication_error:
        # Publication failure must never replace the stage's original traceback.
        print(
            json.dumps(
                {
                    "status": "failed",
                    "failure_evidence_published": False,
                    "publication_error_type": type(publication_error).__name__,
                }
            ),
            flush=True,
        )


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
            _prepare(args, output)
        else:
            prepared = materialize(args.input_path, workspace / "input")
            _run_stage(args, prepared, output, workspace)
        publish(output, args.output_path)
        print(json.dumps({"stage": args.stage, "status": "complete"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
