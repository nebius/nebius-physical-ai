"""npa newton — Newton physics engine workbench commands.

Runs locally by default.  The three core stages (``train-teacher``,
``generate-demos``, ``eval``) are currently stubs: they expose the
intended argument signatures but exit with a clear "not yet implemented"
message until the Newton simulation pipeline lands (tracking issue
nebius/nebius-physical-ai#499).
"""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console

app = typer.Typer(
    name="newton",
    help="Newton physics engine: teacher training, demo generation, evaluation.",
    no_args_is_help=True,
)

console = Console(stderr=True)

_ISSUE_REF = "nebius/nebius-physical-ai#499"


def _not_implemented(stage: str) -> None:
    """Print the stub notice and exit non-zero."""
    console.print(
        f"[yellow]not yet implemented:[/yellow] Newton workbench stage "
        f"'{stage}' is a stub in this release. Tracking issue: {_ISSUE_REF}."
    )
    raise typer.Exit(code=2)


@app.command("train-teacher")
def train_teacher_cmd(
    dataset_uri: str = typer.Option(
        ..., "--dataset-uri", help="URI of the training dataset (s3:// or file://)."
    ),
    output_uri: str = typer.Option(
        ..., "--output-uri", help="URI where the trained teacher artifact is written."
    ),
    config_name: str = typer.Option(
        "newton_double_pendulum",
        "--config-name",
        help="Newton scenario / model config name.",
    ),
    train_steps: int = typer.Option(
        100, "--train-steps", help="Number of teacher training steps."
    ),
    seed: Optional[int] = typer.Option(
        None, "--seed", help="Random seed for reproducibility."
    ),
) -> None:
    """Train a teacher policy in Newton simulation."""
    _not_implemented("train-teacher")


@app.command("generate-demos")
def generate_demos_cmd(
    checkpoint_uri: str = typer.Option(
        ...,
        "--checkpoint-uri",
        help="URI of the trained teacher checkpoint to roll out.",
    ),
    output_uri: str = typer.Option(
        ..., "--output-uri", help="URI where generated demonstrations are written."
    ),
    num_demos: int = typer.Option(
        10, "--num-demos", help="Number of demonstrations to generate."
    ),
    seed: Optional[int] = typer.Option(
        None, "--seed", help="Random seed for reproducibility."
    ),
) -> None:
    """Generate demonstration rollouts from a Newton teacher policy."""
    _not_implemented("generate-demos")


@app.command("eval")
def eval_cmd(
    checkpoint_uri: str = typer.Option(
        ..., "--checkpoint-uri", help="URI of the policy checkpoint to evaluate."
    ),
    dataset_uri: str = typer.Option(
        ..., "--dataset-uri", help="URI of the evaluation dataset (s3:// or file://)."
    ),
    output_uri: str = typer.Option(
        ..., "--output-uri", help="URI where evaluation results are written."
    ),
    num_episodes: int = typer.Option(
        5, "--num-episodes", help="Number of evaluation episodes."
    ),
    seed: Optional[int] = typer.Option(
        None, "--seed", help="Random seed for reproducibility."
    ),
) -> None:
    """Evaluate a policy in Newton simulation."""
    _not_implemented("eval")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
