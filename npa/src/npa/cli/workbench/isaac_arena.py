"""CLI for real Isaac Lab-Arena policy evaluation."""

from __future__ import annotations

import json
from enum import Enum
from typing import Any

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


_TERMS = {
    "arena_source": {
        "license": "Apache-2.0",
        "baked": True,
        "version": "0.3.0",
        "release_channel": "alpha",
    },
    "isaac_sim_and_lab": {
        "license": "Isaac Lab wheel: BSD-3-Clause; Isaac Sim and proprietary runtime: NVIDIA terms",
        "baked": False,
        "runtime_fetch": True,
        "acceptance_environment": "ACCEPT_EULA",
    },
    "lightwheel_sdk": {
        "license": "Apache-2.0",
        "version": "1.0.3",
        "baked": True,
    },
    "lightwheel_registry_assets": {
        "license": "upstream-provider-controlled",
        "baked": False,
        "runtime_fetch": True,
        "redistribution": False,
        "authorization": (
            "operator responsibility; NPA supplies no credential or license grant"
        ),
    },
    "nvidia_viewport_graphics_userspace": {
        "license": "NVIDIA driver package terms",
        "baked": False,
        "runtime_fetch": True,
        "redistribution": False,
        "scope": "viewport evaluation only",
        "source": "exact-driver-matched Ubuntu signed package",
        "installed_on_node": False,
        "container_weights_placement": (
            "Missing nvoptix.bin is copied only into the verified private root overlay; "
            "the image contains an empty /usr/share/nvidia directory."
        ),
        "retention": "Copied weights remain only for the worker container lifetime.",
        "readiness": "Native EGL/Vulkan, libnvoptix.so.1 and readable regular nonempty weights; no denoising-success claim.",
    },
    "operator_inputs": {"baked": False, "redistribution": False},
}


class OutputFormat(str, Enum):
    """Select the representation printed by the evaluation command.

    Args:
        value: ``json`` for the result document or ``text`` for its summary.

    Returns:
        The matching output-format member.

    Raises:
        ValueError: The value is not a supported output format.
    """

    json = "json"
    text = "text"


@app.command("capabilities")
@resolve_typer_defaults
@intent_boundary(OperationIntent.OBSERVE)
def capabilities_cmd() -> None:
    """Print the pinned upstream surface and NPA support status as JSON.

    Args:
        None.

    Returns:
        None. Writes the capability document to standard output.

    Raises:
        None.
    """

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
    execution_device: str = typer.Option(
        "cuda:0",
        "--execution-device",
        help=(
            "Arena physics/policy device. Use cuda:0 for CUDA state evaluation; "
            "the GR1 replay workflow uses cpu following upstream's tutorial. "
            "Viewport recording still requires an RTX GPU."
        ),
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
    """Evaluate a zero, replay, or RSL-RL policy with upstream's runner.

    Args:
        output_path: Local result directory or S3 output prefix.
        environment: Registered Arena environment name.
        policy_type: ``zero_action``, ``replay``, or ``rsl_rl`` adapter.
        input_path: Local or S3 replay file/checkpoint input, when required.
        execution_device: ``cuda:0`` or ``cpu`` physics/policy device, separate from viewport rendering.
        num_episodes: Number of scored episodes; replay requires one.
        num_envs: Number of simulation environments; replay requires one.
        seed: Upstream random seed.
        embodiment: Optional registered robot override.
        object_name: Optional task-compatible object override.
        record_video: Require viewport video and its visual qualification.
        run_id: Optional identifier binding the resulting evidence.
        runtime_image: Optional exact image identity recorded in the manifest.
        dry_run: Build the request without executing the simulator.
        output_format: JSON document or concise text summary.

    Returns:
        None. Writes the evaluation result to standard output.

    Raises:
        typer.Exit: Evaluation fails; exits with status 1.
    """
    request = IsaacArenaRequest(
        output_path=output_path,
        environment=environment,
        policy_type=policy_type,
        input_path=input_path,
        execution_device=execution_device,
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
    _evaluate_and_emit(request, output_format)


def _evaluate_and_emit(request: IsaacArenaRequest, output_format: OutputFormat) -> None:
    try:
        result = evaluate(request)
    except IsaacArenaError as exc:
        typer.echo(f"Isaac Arena evaluation failed: {exc}", err=True)
        raise typer.Exit(1) from exc
    _emit_evaluation(result, output_format)


def _emit_evaluation(result: dict[str, Any], output_format: OutputFormat) -> None:
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
    """Describe source and runtime redistribution boundaries.

    Args:
        None.

    Returns:
        None. Writes the terms document to standard output.

    Raises:
        None.
    """

    typer.echo(json.dumps(_TERMS, indent=2, sort_keys=True))
