"""Thin CLI for MJLab training, measured evaluation and policy export."""

from __future__ import annotations

import json
from enum import Enum
import os

import typer

from npa.lifecycle_intent import json_stdout_contract
from npa.workbench.mjlab.client import invoke
from npa.workbench.mjlab.schemas import (
    DEFAULT_TASK,
    EvalRequest,
    ExportRequest,
    MjlabError,
    TrainRequest,
)

app = typer.Typer(
    name="mjlab",
    help="MJLab GPU robot learning, evaluation and ONNX export.",
    no_args_is_help=True,
)


class OutputFormat(str, Enum):
    """Supported machine-readable CLI format."""

    json = "json"


def _call(operation, model, values, endpoint, token_env, dry_run=False):
    try:
        request = model(**values) if model else None
        planned = dry_run or os.environ.get("NPA_DRY_RUN", "").lower() in {
            "1",
            "true",
            "yes",
        }
        payload = invoke(
            operation, request, endpoint=endpoint, token_env=token_env, dry_run=planned
        )
    except (ValueError, MjlabError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    typer.echo(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


@app.command("train")
@json_stdout_contract
def train_cmd(
    output_path: str = typer.Option(..., "--output-path"),
    task: str = typer.Option(DEFAULT_TASK, "--task"),
    input_path: str | None = typer.Option(None, "--input-path"),
    checkpoint: str | None = typer.Option(None, "--checkpoint"),
    iterations: int | None = typer.Option(None, "--iterations"),
    num_envs: int | None = typer.Option(None, "--num-envs"),
    gpu_count: int = typer.Option(1, "--gpu-count"),
    learning_rate: float | None = typer.Option(None, "--learning-rate"),
    seed: int = typer.Option(42, "--seed"),
    endpoint: str = typer.Option("", "--endpoint"),
    token_env: str = typer.Option("MJLAB_TOKEN", "--token-env"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    output_format: OutputFormat = typer.Option(
        OutputFormat.json, "--output-format", "--output"
    ),
) -> None:
    """Train or resume a native MJLab policy on the current GPU node.

    Args:
        CLI options: Task, storage, training and service settings shown in help.
    Returns:
        None; emits one JSON manifest.
    Raises:
        typer.Exit: Invalid request or failed operation.
    """
    values = dict(
        task=task,
        input_path=input_path,
        output_path=output_path,
        checkpoint=checkpoint,
        iterations=iterations,
        num_envs=num_envs,
        gpu_count=gpu_count,
        learning_rate=learning_rate,
        seed=seed,
    )
    _call("train", TrainRequest, values, endpoint, token_env, dry_run)


@app.command("eval")
@json_stdout_contract
def eval_cmd(
    checkpoint: str = typer.Option(..., "--checkpoint", "--checkpoint-path"),
    output_path: str = typer.Option(..., "--output-path", "-o"),
    task: str = typer.Option(DEFAULT_TASK, "--task"),
    input_path: str | None = typer.Option(None, "--input-path"),
    episodes: int = typer.Option(8, "--episodes"),
    num_envs: int = typer.Option(1, "--num-envs"),
    success_threshold: float = typer.Option(0.75, "--success-threshold"),
    seed: int = typer.Option(42, "--seed"),
    device: str = typer.Option("cuda:0", "--device"),
    video: bool = typer.Option(False, "--video"),
    endpoint: str = typer.Option("", "--endpoint"),
    token_env: str = typer.Option("MJLAB_TOKEN", "--token-env"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    output_format: OutputFormat = typer.Option(
        OutputFormat.json, "--output-format", "--output"
    ),
) -> None:
    """Measure complete episodes from a native MJLab/RSL-RL checkpoint.

    Args: Checkpoint, task, evaluation, storage and service CLI settings.
    Returns: None; emits one measured JSON report or dry-run plan.
    Raises: typer.Exit for an invalid request or failed operation.
    """
    values = dict(
        task=task,
        input_path=input_path,
        output_path=output_path,
        checkpoint=checkpoint,
        episodes=episodes,
        num_envs=num_envs,
        success_threshold=success_threshold,
        seed=seed,
        device=device,
        video=video,
    )
    _call("eval", EvalRequest, values, endpoint, token_env, dry_run)


@app.command("export")
@json_stdout_contract
def export_cmd(
    checkpoint: str = typer.Option(..., "--checkpoint"),
    output_path: str = typer.Option(..., "--output-path"),
    task: str = typer.Option(DEFAULT_TASK, "--task"),
    input_path: str | None = typer.Option(None, "--input-path"),
    seed: int = typer.Option(42, "--seed"),
    device: str = typer.Option("cuda:0", "--device"),
    endpoint: str = typer.Option("", "--endpoint"),
    token_env: str = typer.Option("MJLAB_TOKEN", "--token-env"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    output_format: OutputFormat = typer.Option(
        OutputFormat.json, "--output-format", "--output"
    ),
) -> None:
    """Export a policy to checked ONNX with robot metadata.

    Args:
        CLI options: Native checkpoint, task, storage and service settings.
    Returns:
        None; emits one JSON manifest.
    Raises:
        typer.Exit: Invalid request or failed operation.
    """
    values = dict(
        task=task,
        input_path=input_path,
        checkpoint=checkpoint,
        output_path=output_path,
        seed=seed,
        device=device,
    )
    _call("export", ExportRequest, values, endpoint, token_env, dry_run)


@app.command("status")
@json_stdout_contract
def status_cmd(
    endpoint: str = typer.Option("", "--endpoint"),
    token_env: str = typer.Option("MJLAB_TOKEN", "--token-env"),
    output_format: OutputFormat = typer.Option(
        OutputFormat.json, "--output-format", "--output"
    ),
) -> None:
    """Report installation or remote status.

    Args:
        endpoint: Optional service URL.
        token_env: Bearer-token variable name.
        output_format: JSON output.
    Returns:
        None; writes status.
    Raises:
        typer.Exit: Service failure.
    """
    _call("status", None, {}, endpoint, token_env)


@app.command("system-info")
@json_stdout_contract
def system_info_cmd(
    endpoint: str = typer.Option("", "--endpoint"),
    token_env: str = typer.Option("MJLAB_TOKEN", "--token-env"),
    output_format: OutputFormat = typer.Option(
        OutputFormat.json, "--output-format", "--output"
    ),
) -> None:
    """Inspect dependency versions.

    Args:
        endpoint: Optional service URL.
        token_env: Bearer-token variable name.
        output_format: JSON output.
    Returns:
        None; writes inventory.
    Raises:
        typer.Exit: Service failure.
    """
    _call("system-info", None, {}, endpoint, token_env)


@app.command("list")
@json_stdout_contract
def list_cmd(
    endpoint: str = typer.Option("", "--endpoint"),
    token_env: str = typer.Option("MJLAB_TOKEN", "--token-env"),
    output_format: OutputFormat = typer.Option(
        OutputFormat.json, "--output-format", "--output"
    ),
) -> None:
    """List tasks from the installed MJLab registry.

    Args:
        endpoint: Optional service URL.
        token_env: Bearer-token variable name.
        output_format: JSON output.
    Returns:
        None; writes task IDs.
    Raises:
        typer.Exit: Missing simulator or service failure.
    """
    _call("list", None, {}, endpoint, token_env)


@app.command("workflow")
@json_stdout_contract
def workflow_cmd(
    output_format: OutputFormat = typer.Option(
        OutputFormat.json, "--output-format", "--output"
    ),
) -> None:
    """Locate the MJLab workflow templates.

    Args:
        output_format: JSON output.
    Returns:
        None; writes paths.
    Raises:
        None.
    """
    typer.echo(
        json.dumps(
            {
                "workflow": "workflows/testing/mjlab-eval.yaml",
                "training_workflow": "workflows/testing/mjlab-train-eval.yaml",
                "render_workflow": "workflows/testing/mjlab-render.yaml",
            }
        )
    )


@app.command("deploy")
@json_stdout_contract
def deploy_cmd(
    image: str = typer.Option(..., "--image"),
    token_secret: str = typer.Option(..., "--token-secret"),
    storage_secret: str = typer.Option(..., "--storage-secret"),
    allowed_s3_roots: str = typer.Option(..., "--allowed-s3-roots"),
    accelerator: str = typer.Option(..., "--accelerator"),
    name: str = typer.Option("mjlab", "--name"),
    namespace: str = typer.Option("workbench", "--namespace"),
    gpu_count: int = typer.Option(1, "--gpu-count"),
    kubeconfig: str = typer.Option("", "--kubeconfig"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    output_format: OutputFormat = typer.Option(
        OutputFormat.json, "--output-format", "--output"
    ),
) -> None:
    """Deploy a private service on an existing GPU cluster using existing Secrets.

    Args:
        CLI options: Image, GPU selector, Secret names, S3 scope and cluster settings.
    Returns:
        None; emits the concrete deployment manifest.
    Raises:
        typer.Exit: Invalid configuration or failed kubectl apply.
    """
    _deploy(
        dict(
            image=image,
            token_secret=token_secret,
            storage_secret=storage_secret,
            allowed_s3_roots=allowed_s3_roots,
            accelerator=accelerator,
            name=name,
            namespace=namespace,
            gpu_count=gpu_count,
        ),
        kubeconfig,
        dry_run,
    )


def _deploy(values, kubeconfig, dry_run):
    from npa.workbench.mjlab.deployment import DeployRequest, deploy

    try:
        dry_run = dry_run or os.environ.get("NPA_DRY_RUN", "").lower() in {
            "1",
            "true",
            "yes",
        }
        request = DeployRequest(**values)
        typer.echo(
            json.dumps(
                deploy(request, kubeconfig=kubeconfig, dry_run=dry_run), indent=2
            )
        )
    except (ValueError, MjlabError, OSError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
