"""Thin CLI bindings for native Cosmos policy training, evaluation, and feedback."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import typer

from npa.cli.path_contract import validate_read_path, validate_write_path
from npa.lifecycle_intent import json_stdout_contract
from npa.workbench.cosmos.policy_eval import evaluate_policy
from npa.workbench.cosmos.policy_feedback import (
    generate_failure_candidates,
    policy_feedback,
)
from npa.workbench.cosmos.policy_train import train_policy


def _invoke(
    function: Callable[..., dict[str, Any]],
    input_path: str,
    output_path: str,
    **kwargs: Any,
) -> None:
    try:
        if input_path:
            validate_read_path(input_path, tool="cosmos3 policy", allow_hf=False)
        validate_write_path(output_path, tool="cosmos3 policy", required=True)
        result = function(input_path=input_path, output_path=output_path, **kwargs)
    except Exception as exc:
        typer.echo(json.dumps({"status": "failed", "error_type": type(exc).__name__}))
        raise typer.Exit(1) from exc
    typer.echo(json.dumps(result))


@json_stdout_contract
def policy_train_cmd(
    output_path: str = typer.Option(..., "--output-path"),
    input_path: str = typer.Option(
        "", "--input-path", help="Optional typed training settings JSON."
    ),
) -> None:
    """Run native LIBERO-10 action-policy SFT and publish its complete DCP."""
    _invoke(train_policy, input_path, output_path)


@json_stdout_contract
def policy_eval_cmd(
    input_path: str = typer.Option(
        ..., "--input-path", help="Completed training.json URI."
    ),
    output_path: str = typer.Option(..., "--output-path"),
    trials_per_task: int = typer.Option(50, "--trials-per-task", min=1, max=50),
    task_ids: str = typer.Option("0,1,2,3,4,5,6,7,8,9", "--task-ids"),
    seed: int = typer.Option(0, "--seed", min=0),
) -> None:
    """Measure the trained policy in native LIBERO closed-loop simulation."""
    _invoke(
        evaluate_policy,
        input_path,
        output_path,
        trials_per_task=trials_per_task,
        task_ids=task_ids,
        seed=seed,
    )


@json_stdout_contract
def policy_feedback_cmd(
    input_path: str = typer.Option(..., "--input-path"),
    output_path: str = typer.Option(..., "--output-path"),
    minimum_success_rate: float = typer.Option(
        0.9, "--minimum-success-rate", min=0, max=1
    ),
) -> None:
    """Qualify complete evaluation evidence and derive failed-task video targets."""
    _invoke(
        policy_feedback,
        input_path,
        output_path,
        minimum_success_rate=minimum_success_rate,
    )


@json_stdout_contract
def failure_candidates_cmd(
    input_path: str = typer.Option(..., "--input-path"),
    output_path: str = typer.Option(..., "--output-path"),
    seed: int = typer.Option(0, "--seed", min=0),
) -> None:
    """Generate guarded video candidates from measured failed-task feedback."""
    _invoke(generate_failure_candidates, input_path, output_path, seed=seed)
