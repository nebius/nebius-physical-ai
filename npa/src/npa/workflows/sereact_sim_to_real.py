"""Controller-loop helpers for the Sereact sim-to-real SkyPilot workflow."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Callable


@dataclass(frozen=True)
class ControllerSettings:
    run_id: str
    input_path: str
    output_path: str
    max_iterations: int = 3
    success_threshold: float = 0.8
    dry_run: bool = True
    cosmos_prompt: str = "Generate sim augmentation candidates for Sereact grasp policy gaps."


@dataclass(frozen=True)
class ControllerStep:
    iteration: int
    stage: str
    command: list[str]
    output_path: str


@dataclass(frozen=True)
class ControllerResult:
    status: str
    run_id: str
    dry_run: bool
    iterations_planned: int
    steps: list[dict[str, Any]]
    generated_at: str


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


def build_iteration_steps(settings: ControllerSettings, iteration: int) -> list[ControllerStep]:
    """Build the command plan for one controller-loop iteration."""
    if iteration < 1:
        raise ValueError("iteration must be positive")
    root = settings.output_path.rstrip("/")
    imported = f"{root}/iter-{iteration:02d}/imported-data/"
    cosmos_out = f"{root}/iter-{iteration:02d}/cosmos-candidates/"
    eval_out = f"{root}/iter-{iteration:02d}/vlm-eval/"
    return [
        ControllerStep(
            iteration=iteration,
            stage="data_import",
            command=[
                "npa",
                "workbench",
                "data",
                "sync",
                "--input-path",
                settings.input_path,
                "--output-path",
                imported,
                "--output",
                "json",
            ],
            output_path=imported,
        ),
        ControllerStep(
            iteration=iteration,
            stage="cosmos_generate",
            command=[
                "npa",
                "workbench",
                "cosmos",
                "infer",
                "--prompt",
                settings.cosmos_prompt,
                "--input-path",
                imported,
                "--output-path",
                cosmos_out,
                "--output",
                "json",
            ],
            output_path=cosmos_out,
        ),
        ControllerStep(
            iteration=iteration,
            stage="vlm_eval",
            command=[
                "npa",
                "workbench",
                "vlm-eval",
                "run",
                "--input-path",
                cosmos_out,
                "--output-path",
                eval_out,
                "--success-threshold",
                str(settings.success_threshold),
                "--output",
                "json",
            ],
            output_path=eval_out,
        ),
    ]


def build_controller_plan(settings: ControllerSettings) -> ControllerResult:
    """Return the full controller plan without executing commands."""
    _validate_settings(settings)
    steps = [
        asdict(step)
        for iteration in range(1, settings.max_iterations + 1)
        for step in build_iteration_steps(settings, iteration)
    ]
    return ControllerResult(
        status="planned",
        run_id=settings.run_id,
        dry_run=True,
        iterations_planned=settings.max_iterations,
        steps=steps,
        generated_at=_now_iso(),
    )


def run_controller(
    settings: ControllerSettings,
    *,
    runner: CommandRunner = subprocess.run,
) -> ControllerResult:
    """Run the controller loop or return the dry-run plan."""
    _validate_settings(settings)
    if settings.dry_run:
        return build_controller_plan(settings)

    executed: list[dict[str, Any]] = []
    for iteration in range(1, settings.max_iterations + 1):
        for step in build_iteration_steps(settings, iteration):
            result = runner(
                step.command,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            record = asdict(step)
            record.update(
                {
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
            )
            executed.append(record)
            if result.returncode != 0:
                return ControllerResult(
                    status="failed",
                    run_id=settings.run_id,
                    dry_run=False,
                    iterations_planned=iteration,
                    steps=executed,
                    generated_at=_now_iso(),
                )
        if _last_vlm_step_passed(executed):
            return ControllerResult(
                status="succeeded",
                run_id=settings.run_id,
                dry_run=False,
                iterations_planned=iteration,
                steps=executed,
                generated_at=_now_iso(),
            )
    return ControllerResult(
        status="needs_iteration",
        run_id=settings.run_id,
        dry_run=False,
        iterations_planned=settings.max_iterations,
        steps=executed,
        generated_at=_now_iso(),
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    settings = ControllerSettings(
        run_id=args.run_id,
        input_path=args.input_path,
        output_path=args.output_path,
        max_iterations=args.max_iterations,
        success_threshold=args.success_threshold,
        dry_run=args.dry_run or _env_dry_run(),
        cosmos_prompt=args.cosmos_prompt,
    )
    try:
        result = run_controller(settings)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    if result.status in {"failed", "needs_iteration"}:
        return 1
    return 0


def _validate_settings(settings: ControllerSettings) -> None:
    if not settings.run_id:
        raise ValueError("run_id is required")
    if not settings.input_path.startswith("s3://"):
        raise ValueError("input_path must be an s3:// URI")
    if not settings.output_path.startswith("s3://"):
        raise ValueError("output_path must be an s3:// URI")
    if settings.max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    if not 0.0 <= settings.success_threshold <= 1.0:
        raise ValueError("success_threshold must be between 0 and 1")


def _last_vlm_step_passed(executed: list[dict[str, Any]]) -> bool:
    for step in reversed(executed):
        if step.get("stage") != "vlm_eval":
            continue
        try:
            payload = json.loads(str(step.get("stdout") or "{}"))
        except json.JSONDecodeError:
            return False
        return bool(payload.get("passed"))
    return False


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--success-threshold", type=float, default=0.8)
    parser.add_argument("--cosmos-prompt", default=ControllerSettings.cosmos_prompt)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _env_dry_run() -> bool:
    return os.environ.get("NPA_DRY_RUN", "").lower() in {"1", "true", "yes"} or os.environ.get(
        "DRY_RUN", ""
    ).lower() in {"1", "true", "yes"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    sys.exit(main())
