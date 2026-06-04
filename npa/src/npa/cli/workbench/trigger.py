"""npa workbench trigger - S3 retriggers for Workbench pipelines."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from npa.workflows.sim_to_real import DEFAULT_GPU_FAILOVER, DEFAULT_GPU_TYPE
from npa.workflows.sim_to_real_trigger import (
    DEFAULT_TRIGGER_POLL_INTERVAL,
    DEFAULT_TRIGGER_SUBMIT_TIMEOUT,
    SimToRealTriggerError,
    TriggerConfig,
    run_once as run_trigger_once,
    watch as watch_trigger,
)

app = typer.Typer(
    name="trigger",
    help="Watch S3-compatible data prefixes and retrigger Workbench workflows.",
    no_args_is_help=True,
)
console = Console(stderr=True)


class OutputFormat(str, Enum):
    text = "text"
    json = "json"


class TaskCloudOption(str, Enum):
    kubernetes = "kubernetes"
    nebius = "nebius"


@app.command("run")
def run_cmd(
    s3_endpoint: str = typer.Option(..., "--s3-endpoint", help="S3-compatible endpoint URL."),
    s3_bucket: str = typer.Option(..., "--s3-bucket", help="Bucket containing the LeRobot dataset prefix."),
    s3_prefix: str = typer.Option(..., "--s3-prefix", help="Prefix to poll for LeRobot-format objects."),
    watermark_uri: str = typer.Option("", "--watermark-uri", help="S3 URI or local path for the trigger cursor."),
    pipeline_yaml: Path | None = typer.Option(
        None,
        "--pipeline-yaml",
        help="SkyPilot YAML for the pipeline entrypoint. Defaults to the bundled sim-to-real YAML.",
    ),
    pipeline_bucket: str = typer.Option(
        "",
        "--pipeline-bucket",
        help="Bucket for pipeline outputs. Defaults to --s3-bucket.",
    ),
    pipeline_s3_prefix: str = typer.Option(
        "",
        "--pipeline-s3-prefix",
        help="Output prefix template for pipeline runs; supports {run_id}.",
    ),
    pipeline_input_data_uri: str = typer.Option(
        "",
        "--pipeline-input-data-uri",
        help="Dataset URI passed to the pipeline. Defaults to s3://bucket/prefix/.",
    ),
    pipeline_render_only: bool = typer.Option(
        False,
        "--pipeline-render-only",
        help="Render the pipeline YAML instead of submitting it.",
    ),
    task_cloud: TaskCloudOption = typer.Option(
        TaskCloudOption.kubernetes,
        "--task-cloud",
        help="Cloud backend rendered into the pipeline task resources.",
    ),
    controller_backend: TaskCloudOption = typer.Option(
        TaskCloudOption.kubernetes,
        "--controller-backend",
        help="SkyPilot managed-jobs controller backend.",
    ),
    sky_bin: str = typer.Option(
        "",
        "--sky-bin",
        help="SkyPilot executable path. Defaults to NPA_SKYPILOT_BIN when set.",
    ),
    gpu: str = typer.Option(DEFAULT_GPU_TYPE, "--gpu", help="Primary SkyPilot accelerator."),
    gpu_failover: str = typer.Option(
        DEFAULT_GPU_FAILOVER,
        "--gpu-failover",
        help="Comma-separated fallback SkyPilot accelerators.",
    ),
    submit_timeout: int = typer.Option(
        DEFAULT_TRIGGER_SUBMIT_TIMEOUT,
        "--submit-timeout",
        help="Pipeline submission timeout in seconds.",
    ),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Poll once and launch one pipeline run if new LeRobot data is present."""

    try:
        result = run_trigger_once(
            _config(
                s3_endpoint=s3_endpoint,
                s3_bucket=s3_bucket,
                s3_prefix=s3_prefix,
                watermark_uri=watermark_uri,
                pipeline_yaml=pipeline_yaml,
                pipeline_bucket=pipeline_bucket,
                pipeline_s3_prefix=pipeline_s3_prefix,
                pipeline_input_data_uri=pipeline_input_data_uri,
                pipeline_render_only=pipeline_render_only,
                task_cloud=task_cloud.value,
                controller_backend=controller_backend.value,
                sky_bin=sky_bin,
                gpu=gpu,
                gpu_failover=gpu_failover,
                submit_timeout=submit_timeout,
            )
        )
    except SimToRealTriggerError as exc:
        _fail(str(exc))
        return
    _emit(result.to_payload(), output)


@app.command("watch")
def watch_cmd(
    s3_endpoint: str = typer.Option(..., "--s3-endpoint", help="S3-compatible endpoint URL."),
    s3_bucket: str = typer.Option(..., "--s3-bucket", help="Bucket containing the LeRobot dataset prefix."),
    s3_prefix: str = typer.Option(..., "--s3-prefix", help="Prefix to poll for LeRobot-format objects."),
    watermark_uri: str = typer.Option("", "--watermark-uri", help="S3 URI or local path for the trigger cursor."),
    pipeline_yaml: Path | None = typer.Option(
        None,
        "--pipeline-yaml",
        help="SkyPilot YAML for the pipeline entrypoint. Defaults to the bundled sim-to-real YAML.",
    ),
    pipeline_bucket: str = typer.Option(
        "",
        "--pipeline-bucket",
        help="Bucket for pipeline outputs. Defaults to --s3-bucket.",
    ),
    pipeline_s3_prefix: str = typer.Option(
        "",
        "--pipeline-s3-prefix",
        help="Output prefix template for pipeline runs; supports {run_id}.",
    ),
    pipeline_input_data_uri: str = typer.Option(
        "",
        "--pipeline-input-data-uri",
        help="Dataset URI passed to the pipeline. Defaults to s3://bucket/prefix/.",
    ),
    pipeline_render_only: bool = typer.Option(
        False,
        "--pipeline-render-only",
        help="Render the pipeline YAML instead of submitting it.",
    ),
    task_cloud: TaskCloudOption = typer.Option(
        TaskCloudOption.kubernetes,
        "--task-cloud",
        help="Cloud backend rendered into the pipeline task resources.",
    ),
    controller_backend: TaskCloudOption = typer.Option(
        TaskCloudOption.kubernetes,
        "--controller-backend",
        help="SkyPilot managed-jobs controller backend.",
    ),
    sky_bin: str = typer.Option(
        "",
        "--sky-bin",
        help="SkyPilot executable path. Defaults to NPA_SKYPILOT_BIN when set.",
    ),
    gpu: str = typer.Option(DEFAULT_GPU_TYPE, "--gpu", help="Primary SkyPilot accelerator."),
    gpu_failover: str = typer.Option(
        DEFAULT_GPU_FAILOVER,
        "--gpu-failover",
        help="Comma-separated fallback SkyPilot accelerators.",
    ),
    submit_timeout: int = typer.Option(
        DEFAULT_TRIGGER_SUBMIT_TIMEOUT,
        "--submit-timeout",
        help="Pipeline submission timeout in seconds.",
    ),
    poll_interval: int = typer.Option(
        DEFAULT_TRIGGER_POLL_INTERVAL,
        "--poll-interval",
        help="Seconds between polls.",
    ),
    max_polls: int = typer.Option(0, "--max-polls", help="Maximum polls before exiting; 0 means forever."),
    max_launches: int = typer.Option(
        0,
        "--max-launches",
        help="Maximum launched pipeline runs before exiting; 0 means forever.",
    ),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Poll continuously and launch one pipeline run per new LeRobot data batch."""

    try:
        results = watch_trigger(
            _config(
                s3_endpoint=s3_endpoint,
                s3_bucket=s3_bucket,
                s3_prefix=s3_prefix,
                watermark_uri=watermark_uri,
                pipeline_yaml=pipeline_yaml,
                pipeline_bucket=pipeline_bucket,
                pipeline_s3_prefix=pipeline_s3_prefix,
                pipeline_input_data_uri=pipeline_input_data_uri,
                pipeline_render_only=pipeline_render_only,
                task_cloud=task_cloud.value,
                controller_backend=controller_backend.value,
                sky_bin=sky_bin,
                gpu=gpu,
                gpu_failover=gpu_failover,
                submit_timeout=submit_timeout,
            ),
            poll_interval=poll_interval,
            max_polls=max_polls,
            max_launches=max_launches,
        )
    except SimToRealTriggerError as exc:
        _fail(str(exc))
        return
    _emit({"polls": [result.to_payload() for result in results]}, output)


def _config(
    *,
    s3_endpoint: str,
    s3_bucket: str,
    s3_prefix: str,
    watermark_uri: str,
    pipeline_yaml: Path | None,
    pipeline_bucket: str,
    pipeline_s3_prefix: str,
    pipeline_input_data_uri: str,
    pipeline_render_only: bool,
    task_cloud: str,
    controller_backend: str,
    sky_bin: str,
    gpu: str,
    gpu_failover: str,
    submit_timeout: int,
) -> TriggerConfig:
    return TriggerConfig(
        s3_endpoint=s3_endpoint,
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix,
        watermark_uri=watermark_uri,
        pipeline_yaml=str(pipeline_yaml) if pipeline_yaml else "",
        pipeline_bucket=pipeline_bucket,
        pipeline_s3_prefix=pipeline_s3_prefix,
        pipeline_input_data_uri=pipeline_input_data_uri,
        pipeline_render_only=pipeline_render_only,
        task_cloud=task_cloud,
        controller_backend=controller_backend,
        sky_bin=sky_bin,
        gpu=gpu,
        gpu_failover=gpu_failover,
        submit_timeout=submit_timeout,
    )


def _emit(payload: dict[str, Any], output: OutputFormat) -> None:
    if output == OutputFormat.json:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    for key, value in payload.items():
        if key == "new_objects":
            typer.echo(f"  new_objects: {len(value)}")
        elif key == "polls":
            typer.echo(f"  polls: {len(value)}")
        else:
            typer.echo(f"  {key}: {value}")


def _fail(message: str) -> None:
    console.print(f"[red]Error:[/red] {message}")
    raise typer.Exit(1)
