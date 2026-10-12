"""Run sealed physical augmentation stages through the standard Workbench runtime."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from npa.workflows.lerobot_transfer_data import materialize, publish, write_json
from npa.workflows.physical_augmentation_contract import make_recipe, read_recipe


def build_parser() -> argparse.ArgumentParser:
    """Describe stateless prepare, collect, and report stage arguments.

    Args:
        None.
    Returns:
        Parser shared by execution and workflow argv validation.
    Raises:
        None.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--run-id", required=True)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--episodes-per-condition", type=int, default=3)
    prepare.add_argument("--episode-steps", type=int, default=600)
    prepare.add_argument("--output-path", required=True)
    for name in ("collect", "report"):
        stage = commands.add_parser(name)
        stage.add_argument("--input-path", required=True)
        stage.add_argument("--output-path", required=True)
    return parser


def _collect(source: Path, output: Path) -> None:
    recipe = read_recipe(source / "recipe.json")
    output.mkdir(parents=True)
    shutil.copy2(source / "recipe.json", output / "recipe.json")
    if "scene_binding" in recipe:
        shutil.copy2(source / "scene.usdc", output / "scene.usdc")
    for condition in recipe["conditions"]:
        _collect_condition(source, output, condition)
    write_json(
        output / "collection.json",
        {
            "schema": "npa.physical-augmentation.collection.v1",
            "conditions": list(recipe["conditions"]),
            "run_id": recipe["run_id"],
        },
    )


def _collect_condition(source: Path, output: Path, condition: str) -> None:
    interpreter = os.environ.get("ISAAC_LAB_PYTHON", "/isaac-sim/python.sh")
    argv = [
        interpreter,
        "-m",
        "npa.workflows.physical_augmentation_runtime",
        "--input-path",
        str(source),
        "--output-path",
        str(output / condition),
        "--condition",
        condition,
        "--visualizer",
        "none",
    ]
    log_path = output / f"{condition}.log"
    with log_path.open("w") as log:
        result = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=False)
    log_text = log_path.read_text()
    if result.returncode or not (output / condition / "capture.json").is_file():
        print(log_text[-12000:], flush=True)
        raise RuntimeError(
            f"Native {condition} capture failed or lacks its completion record"
        )
    if "PhysX error:" in log_text or "simulation will miss interactions" in log_text:
        raise RuntimeError(f"Invalid native physics in {condition}")
    capture = json.loads((output / condition / "capture.json").read_text())
    if (capture.get("schema"), capture.get("condition")) != (
        "npa.physical-augmentation.capture.v1",
        condition,
    ):
        raise ValueError("Native capture completion record has the wrong identity")
    print(f"Captured physical condition: {condition}", flush=True)


def _execute(args, workspace: Path, output: Path) -> None:
    if args.stage == "prepare":
        recipe = make_recipe(
            args.run_id, args.seed, args.episodes_per_condition, args.episode_steps
        )
        write_json(output / "recipe.json", recipe)
        return
    source = materialize(args.input_path, workspace / "input")
    if args.stage == "collect":
        _collect(source, output)
    else:
        from npa.workflows.physical_augmentation_report import report_results

        report_results(source, output)


def main(argv: list[str] | None = None) -> int:
    """Execute one workflow stage and publish content-verified artifacts.

    Args:
        argv: Stage arguments, or process arguments when omitted.
    Returns:
        Zero after successful publication.
    Raises:
        ValueError: Recipe, artifact, or acceptance validation fails.
        RuntimeError: Native simulation fails.
        OSError: Artifact exchange fails.
    """
    from npa.workflows.franka_rl import _publish_stage_failure

    args = build_parser().parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="physical-augmentation-") as temporary:
        workspace = Path(temporary)
        output = workspace / "output"
        try:
            _execute(args, workspace, output)
        except Exception as error:
            _publish_stage_failure(
                output,
                args.output_path,
                error,
                stage=args.stage,
                schema="npa.physical-augmentation.stage-failure.v1",
            )
            raise
        publish(output, args.output_path)
        print(json.dumps({"stage": args.stage, "status": "complete"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
