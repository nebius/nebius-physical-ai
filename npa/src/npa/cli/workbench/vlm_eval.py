"""npa workbench vlm-eval - VLM rollout evaluation commands."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from npa.workbench.vlm_eval import (
    DEFAULT_API_KEY_ENV,
    DEFAULT_BACKEND,
    DEFAULT_FRAME_SELECTION,
    DEFAULT_MAX_FRAMES,
    DEFAULT_MODEL,
    DEFAULT_RUBRIC,
    DEFAULT_TIMEOUT_S,
    SUPPORTED_BACKENDS,
    SUPPORTED_FRAME_SELECTIONS,
    VlmEvalError,
    evaluate_vlm,
    write_result,
)

app = typer.Typer(
    name="vlm-eval",
    help="VLM evaluation for sim-to-real pipeline gating.",
    no_args_is_help=True,
)
console = Console(stderr=True)
WORKFLOW_PATH = Path("npa/workflows/workbench/skypilot/vlm-eval.yaml")


class OutputFormat(str, Enum):
    text = "text"
    json = "json"


class BackendName(str, Enum):
    self_hosted = "self-hosted"
    api = "api"
    stub = "stub"


class FrameSelection(str, Enum):
    final = "final"
    keyframes = "keyframes"
    sequence = "sequence"


@app.command("run")
def run_cmd(
    input_path: str = typer.Option(..., "--input-path", help="S3 or local artifact path to score."),
    output_path: str = typer.Option(..., "--output-path", help="S3 or local path for eval JSON."),
    task: str = typer.Option("sim-to-real", "--task", help="Evaluation task label."),
    backend: BackendName = typer.Option(
        BackendName.self_hosted,
        "--backend",
        help="VLM backend: self-hosted, api, or stub.",
    ),
    model: str = typer.Option(DEFAULT_MODEL, "--model", help="VLM model name."),
    endpoint_url: str = typer.Option(
        "",
        "--endpoint-url",
        help="OpenAI-compatible base URL or /chat/completions URL.",
    ),
    api_key_env: str = typer.Option(
        DEFAULT_API_KEY_ENV,
        "--api-key-env",
        help="Environment variable containing the API key for --backend api.",
    ),
    frame_selection: FrameSelection = typer.Option(
        FrameSelection.keyframes,
        "--frame-selection",
        help="Rollout frame selection: final, keyframes, or sequence.",
    ),
    max_frames: int = typer.Option(
        DEFAULT_MAX_FRAMES,
        "--max-frames",
        help="Maximum frames sent to the VLM.",
    ),
    rubric: str = typer.Option(
        DEFAULT_RUBRIC,
        "--rubric",
        help="Scoring rubric text.",
    ),
    rubric_path: str = typer.Option(
        "",
        "--rubric-path",
        help="Path to a scoring rubric text file.",
    ),
    success_threshold: float = typer.Option(
        0.8,
        "--success-threshold",
        help="Score threshold required to pass.",
    ),
    score: float = typer.Option(
        -1.0,
        "--score",
        help="Override score for tests and dry validation; skips the VLM call.",
    ),
    timeout_s: float = typer.Option(
        DEFAULT_TIMEOUT_S,
        "--timeout-s",
        help="VLM request timeout in seconds.",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Do not write the result artifact."),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Score a rollout artifact with a VLM backend."""

    try:
        result = evaluate_vlm(
            input_path=input_path,
            output_path=output_path,
            task=task,
            backend=_enum_value(backend),
            model=model,
            endpoint_url=endpoint_url,
            api_key_env=api_key_env,
            frame_selection=_enum_value(frame_selection),
            max_frames=max_frames,
            rubric=rubric,
            rubric_path=rubric_path,
            success_threshold=success_threshold,
            timeout_s=timeout_s,
            score=None if score < 0 else score,
        )
        payload = asdict(result)
        effective_dry_run = dry_run or _env_dry_run()
        payload["dry_run"] = effective_dry_run
        if not effective_dry_run:
            payload["written_uri"] = write_result(payload, result_uri=result.result_uri)
    except VlmEvalError as exc:
        _fail(str(exc))
        return
    _emit(payload, output)


@app.command("workflow")
def workflow_cmd(
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Show the SkyPilot YAML template for VLM evaluation."""

    _emit({"workflow": str(WORKFLOW_PATH)}, output)


@app.command("status")
def status_cmd(
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Show VLM eval backend status."""

    _emit(
        {
            "backend": DEFAULT_BACKEND,
            "status": "configured",
            "real_vlm_backend": True,
            "default_model": DEFAULT_MODEL,
            "default_frame_selection": DEFAULT_FRAME_SELECTION,
            "workflow": str(WORKFLOW_PATH),
        },
        output,
    )


@app.command("list")
def list_cmd(
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """List available VLM eval backends."""

    _emit(
        {
            "backends": [
                {"name": "self-hosted", "real_backend": True, "default": True},
                {"name": "api", "real_backend": True, "default": False},
                {"name": "stub", "real_backend": False, "default": False},
            ],
            "frame_selections": list(SUPPORTED_FRAME_SELECTIONS),
            "supported_backends": list(SUPPORTED_BACKENDS),
        },
        output,
    )


def _emit(payload: dict[str, Any], output: OutputFormat) -> None:
    if output == OutputFormat.json:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    for key, value in payload.items():
        typer.echo(f"  {key}: {value}")


def _env_dry_run() -> bool:
    return os.environ.get("NPA_DRY_RUN", "").lower() in {"1", "true", "yes"} or os.environ.get(
        "DRY_RUN", ""
    ).lower() in {"1", "true", "yes"}


def _enum_value(value: Any) -> str:
    return value.value if hasattr(value, "value") else str(value)


def _fail(message: str) -> None:
    console.print(f"[red]Error:[/red] {message}")
    raise typer.Exit(1)
