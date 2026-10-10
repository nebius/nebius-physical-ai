"""Typer adapter for the shared Gemini Robotics hosted-client and workflow stages."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from npa.cli.path_contract import validate_read_path, validate_write_path
from npa.clients.gemini_robotics import (
    BASE_URL_ENV,
    DEFAULT_MODEL_ID,
    GeminiRoboticsClient,
    GeminiRoboticsError,
    resolve_config,
)
from npa.clients.storage import StorageClient, StorageError
from npa.workflows.byof.gemini_robotics_pipeline import (
    GeminiRoboticsPipelineConfig,
    GeminiRoboticsPipelineError,
    read_eval_input,
    run_er_planning_stage,
    run_eval_stage,
)

app = typer.Typer(
    name="gemini-robotics",
    help=(
        "Gemini Robotics hosted planning and evaluation (plan, eval). "
        "Bring your own GOOGLE_API_KEY; model and base URL default to the "
        "live-validated endpoint."
    ),
    no_args_is_help=True,
)
console = Console(stderr=True)


def _fail(message: str) -> None:
    console.print(f"[red]error:[/red] {message}")
    raise typer.Exit(1)


@app.command("plan")
def plan_cmd(
    task: str = typer.Argument(..., help="Natural-language task for the ER planner."),
    image: list[str] = typer.Option(
        [],
        "--image",
        help="Exact s3:// scene-observation image URI (repeatable).",
    ),
    model: str = typer.Option(
        DEFAULT_MODEL_ID,
        "--model",
        help=f"Gemini API model id (default: {DEFAULT_MODEL_ID}).",
    ),
    api_base_url: str = typer.Option(
        "",
        "--api-base-url",
        help=f"Gemini API base URL (or set {BASE_URL_ENV}).",
    ),
    output_path: str = typer.Option(
        ..., "--output-path", help="Exact s3:// URI for the plan receipt."
    ),
) -> dict[str, Any]:
    """Run advisory ER planning and publish one immutable S3 receipt."""

    try:
        validate_write_path(output_path, tool="gemini-robotics plan", required=True)
        for image_path in image:
            validate_read_path(
                image_path,
                tool="gemini-robotics plan",
                option="--image",
                allow_hf=False,
            )
        stage_config = GeminiRoboticsPipelineConfig(
            task=task,
            output_path=output_path,
            model=model,
        )
        stage_config.require_model()
        config = resolve_config(base_url=api_base_url or None)
        client = GeminiRoboticsClient(config)
        storage = StorageClient.from_environment()
        with tempfile.TemporaryDirectory(prefix="npa-gemini-robotics-") as temp_dir:
            local_images: list[str] = []
            for index, image_path in enumerate(image):
                local_path = Path(temp_dir) / f"{index}-{Path(image_path).name}"
                storage.download_file(image_path, str(local_path))
                local_images.append(str(local_path))
            stage_config.images = local_images
            receipt = run_er_planning_stage(
                stage_config,
                client,
                storage,
            )
    except (
        GeminiRoboticsError,
        GeminiRoboticsPipelineError,
        OSError,
        StorageError,
        ValueError,
    ) as exc:
        _fail(str(exc))
        raise  # pragma: no cover - _fail raises
    console.print(json.dumps(receipt, indent=2))
    return receipt


@app.command("eval")
def eval_cmd(
    input_path: str = typer.Option(
        ...,
        "--input-path",
        help="Exact s3:// JSON input with non-empty plan_text and rubric strings.",
    ),
    model: str = typer.Option(
        DEFAULT_MODEL_ID,
        "--model",
        help=f"Gemini API model id (default: {DEFAULT_MODEL_ID}).",
    ),
    api_base_url: str = typer.Option(
        "",
        "--api-base-url",
        help=f"Gemini API base URL (or set {BASE_URL_ENV}).",
    ),
    output_path: str = typer.Option(
        ..., "--output-path", help="Exact s3:// URI for the eval receipt."
    ),
) -> dict[str, Any]:
    """Evaluate one durable plan/rubric input and publish an S3 receipt."""

    try:
        validate_write_path(output_path, tool="gemini-robotics eval", required=True)
        stage_config = GeminiRoboticsPipelineConfig(
            task=input_path,
            output_path=output_path,
            model=model,
        )
        stage_config.require_model()
        config = resolve_config(base_url=api_base_url or None)
        client = GeminiRoboticsClient(config)
        storage = StorageClient.from_environment()
        plan_receipt, source_uri, source_etag, source_sha256 = read_eval_input(
            input_path, storage
        )
        receipt = run_eval_stage(
            stage_config,
            plan_receipt,
            str(plan_receipt["rubric"]),
            client,
            storage,
            input_path=source_uri,
            input_etag=source_etag,
            input_sha256=source_sha256,
        )
    except (
        GeminiRoboticsError,
        GeminiRoboticsPipelineError,
        OSError,
        StorageError,
        ValueError,
    ) as exc:
        _fail(str(exc))
        raise  # pragma: no cover - _fail raises
    console.print(json.dumps(receipt, indent=2))
    return receipt
