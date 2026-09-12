"""CLI for real Isaac Lab-Arena policy evaluation."""

from __future__ import annotations

import json
from enum import Enum

import typer

from npa.cli._typer_defaults import resolve_typer_defaults
from npa.lifecycle_intent import OperationIntent, intent_boundary, json_stdout_contract
from npa.workbench.isaac_arena import (
    IsaacArenaError,
    IsaacArenaRequest,
    capabilities,
    evaluate,
)

app = typer.Typer(
    name="isaac-arena",
    help="Isaac Lab-Arena policy evaluation with durable results.",
    no_args_is_help=True,
)


class OutputFormat(str, Enum):
    json = "json"
    text = "text"


@app.command("capabilities")
@resolve_typer_defaults
@intent_boundary(OperationIntent.OBSERVE)
def capabilities_cmd() -> None:
    """Print the pinned upstream surface and NPA support status as JSON."""

    typer.echo(json.dumps(capabilities(), indent=2, sort_keys=True))


@app.command("evaluate")
@resolve_typer_defaults
@json_stdout_contract
@intent_boundary(OperationIntent.MUTATE)
def evaluate_cmd(
    output_path: str = typer.Option(
        ..., "--output-path", help="Local directory or s3:// prefix."
    ),
    environment: str = typer.Option("cube_goal_pose", "--environment"),
    policy_type: str = typer.Option("zero_action", "--policy-type"),
    input_path: str = typer.Option(
        "", "--input-path", help="Replay file or RSL-RL checkpoint tree."
    ),
    num_episodes: int = typer.Option(1, "--num-episodes", min=1),
    num_envs: int = typer.Option(1, "--num-envs", min=1),
    seed: int = typer.Option(42, "--seed"),
    embodiment: str = typer.Option("", "--embodiment"),
    object_name: str = typer.Option("", "--object"),
    record_video: bool = typer.Option(False, "--record-video/--no-record-video"),
    run_id: str = typer.Option("", "--run-id"),
    runtime_image: str = typer.Option("", "--runtime-image"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    output_format: OutputFormat = typer.Option(OutputFormat.json, "--output-format"),
) -> None:
    """Evaluate a zero, replay, or RSL-RL policy with upstream's runner."""

    try:
        result = evaluate(
            IsaacArenaRequest(
                output_path=output_path,
                environment=environment,
                policy_type=policy_type,
                input_path=input_path,
                num_episodes=num_episodes,
                num_envs=num_envs,
                seed=seed,
                embodiment=embodiment,
                object_name=object_name,
                record_video=record_video,
                run_id=run_id,
                runtime_image=runtime_image,
                dry_run=dry_run,
            )
        )
    except IsaacArenaError as exc:
        typer.echo(f"Isaac Arena evaluation failed: {exc}", err=True)
        raise typer.Exit(1) from exc
    if output_format == OutputFormat.json:
        typer.echo(json.dumps(result, indent=2, sort_keys=True))
    else:
        summary = result.get("summary") or {}
        typer.echo(
            f"status={result['status']} episodes={summary.get('episodes', 0)} "
            f"success_rate={summary.get('success_rate', 'n/a')}"
        )


@app.command("terms")
@resolve_typer_defaults
def terms_cmd() -> None:
    """Describe source and runtime redistribution boundaries."""

    typer.echo(
        json.dumps(
            {
                "arena_source": {
                    "license": "Apache-2.0",
                    "baked": True,
                    "version": "0.3.0",
                    "release_channel": "alpha",
                },
                "isaac_sim_and_lab": {
                    "license": "NVIDIA proprietary software terms",
                    "baked": False,
                    "runtime_fetch": True,
                    "acceptance_environment": "ACCEPT_EULA",
                },
                "operator_inputs": {"baked": False, "redistribution": False},
            },
            indent=2,
            sort_keys=True,
        )
    )
