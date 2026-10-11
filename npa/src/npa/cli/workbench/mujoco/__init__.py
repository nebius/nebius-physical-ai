"""npa mujoco — MuJoCo contact-rich manipulation rollouts.

Runs scripted policies on programmatic MuJoCo scenes (peg insertion, screw
driving, reach) with real contact solving, headless by default.  Runs locally;
the ``mujoco`` package is required at run time but the CLI imports without it.
"""

from __future__ import annotations

import typer
from rich.console import Console

from npa.cli.path_contract import PathContractError, validate_write_path

app = typer.Typer(
    name="mujoco",
    help="MuJoCo contact-rich manipulation: scripted-policy rollouts with real contact solving.",
    no_args_is_help=True,
)

console = Console(stderr=True)


@app.command("run")
def run_cmd(
    task: str = typer.Option(
        ...,
        "--task",
        help="Manipulation task: 'peg_insertion', 'screw_driving', or 'reach'.",
    ),
    policy: str = typer.Option(
        ...,
        "--policy",
        help="Scripted policy: 'scripted:<expert|noisy|random>'.",
    ),
    episodes: int = typer.Option(10, "--episodes", help="Number of rollout episodes."),
    seed: int = typer.Option(
        0, "--seed", help="Base random seed (episode i uses seed+i)."
    ),
    output_path: str = typer.Option(
        ...,
        "--output-path",
        "--output-uri",
        help="S3 output path for the trajectories + metrics JSON report (--output-uri is a compatibility alias).",
    ),
    max_steps: int = typer.Option(
        500, "--max-steps", help="Maximum control steps per episode."
    ),
) -> None:
    """Run N episodes of a scripted policy; write trajectories + metrics."""
    from npa.workflows.byof.mujoco_pipeline import run as stage_run

    try:
        output_path = validate_write_path(
            output_path,
            tool="mujoco run",
            option="--output-path",
            required=True,
        )
    except PathContractError as exc:
        raise typer.BadParameter(str(exc), param_hint="--output-path") from exc
    result = stage_run(
        task=task,
        policy=policy,
        output_path=output_path,
        episodes=episodes,
        seed=seed,
        max_steps=max_steps,
    )
    summary = result["summary"]
    console.print(
        f"[green]done:[/green] {task} x {policy}: "
        f"success_rate={summary['success_rate']:.3f} "
        f"({summary['successes']}/{summary['episodes']}) -> {output_path}"
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
