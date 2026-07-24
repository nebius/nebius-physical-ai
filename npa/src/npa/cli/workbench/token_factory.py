"""npa workbench token-factory - Nebius Token Factory hosted-inference commands."""

from __future__ import annotations

import json
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from npa.clients.token_factory import (
    DEFAULT_BASE_URL,
    DEFAULT_REASONER_MODEL,
    DEFAULT_TEXT_MODEL,
    DEFAULT_VISION_MODEL,
    TokenFactoryError,
    resolve_config,
)
from npa.workbench.token_factory import (
    DEFAULT_CAPTION_INSTRUCTION,
    DEFAULT_GENERATE_SYSTEM_PROMPT,
    DEFAULT_MAX_IMAGES,
    DEFAULT_MAX_TOKENS,
    DEFAULT_REASON_MAX_IMAGES,
    DEFAULT_REASON_MAX_TOKENS,
    DEFAULT_REASON_SYSTEM_PROMPT,
    DEFAULT_REASON_TASK,
    TokenFactoryToolError,
    caption_images,
    generate_text,
    list_models,
    reason_scene,
    write_captions,
    write_generations,
    write_reason,
)

app = typer.Typer(
    name="token-factory",
    help="Nebius Token Factory hosted inference (zero-GPU, OpenAI-compatible).",
    no_args_is_help=True,
)
console = Console(stderr=True)

CAPTION_WORKFLOW_PATH = Path("npa/src/npa/workflows/skypilot/token-factory-caption.yaml")
GENERATE_WORKFLOW_PATH = Path("npa/src/npa/workflows/skypilot/token-factory-generate.yaml")
REASON_WORKFLOW_PATH = Path("npa/src/npa/workflows/skypilot/token-factory-cosmos-reason.yaml")
VLM_EVAL_WORKFLOW_PATH = Path("npa/src/npa/workflows/skypilot/vlm-eval-token-factory.yaml")


class OutputFormat(str, Enum):
    text = "text"
    json = "json"


@app.command("caption")
def caption_cmd(
    input_path: str = typer.Option(..., "--input-path", help="S3 or local path to images."),
    output_path: str = typer.Option(..., "--output-path", help="S3 or local path for captions JSON."),
    model: str = typer.Option(DEFAULT_VISION_MODEL, "--model", help="Token Factory vision model."),
    instruction: str = typer.Option(
        DEFAULT_CAPTION_INSTRUCTION, "--instruction", help="Captioning instruction prompt."
    ),
    max_images: int = typer.Option(DEFAULT_MAX_IMAGES, "--max-images", help="Maximum images to caption."),
    max_tokens: int = typer.Option(DEFAULT_MAX_TOKENS, "--max-tokens", help="Max tokens per caption."),
    temperature: float = typer.Option(0.2, "--temperature", help="Sampling temperature."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Do not write the result artifact."),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Caption a folder of images with a hosted Token Factory vision model."""

    try:
        result = caption_images(
            input_path=input_path,
            output_path=output_path,
            model=model,
            instruction=instruction,
            max_images=max_images,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        payload = asdict(result)
        payload["dry_run"] = dry_run
        if not dry_run:
            payload["written_uri"] = write_captions(payload, result_uri=result.result_uri)
    except (TokenFactoryToolError, TokenFactoryError) as exc:
        _fail(str(exc))
        return
    _emit(payload, output)


@app.command("generate")
def generate_cmd(
    input_path: str = typer.Option(..., "--input-path", help="S3 or local JSONL/text prompt file."),
    output_path: str = typer.Option(..., "--output-path", help="S3 or local path for generations JSONL."),
    model: str = typer.Option(DEFAULT_TEXT_MODEL, "--model", help="Token Factory text model."),
    system_prompt: str = typer.Option(
        DEFAULT_GENERATE_SYSTEM_PROMPT, "--system-prompt", help="System prompt applied to every request."
    ),
    max_prompts: int = typer.Option(0, "--max-prompts", help="Limit prompts processed (0 = all)."),
    max_tokens: int = typer.Option(DEFAULT_MAX_TOKENS, "--max-tokens", help="Max tokens per completion."),
    temperature: float = typer.Option(0.7, "--temperature", help="Sampling temperature."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Do not write the result artifact."),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Generate completions for each prompt in a JSONL/text file."""

    try:
        result = generate_text(
            input_path=input_path,
            output_path=output_path,
            model=model,
            system_prompt=system_prompt,
            max_prompts=max_prompts,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        payload = asdict(result)
        payload["dry_run"] = dry_run
        if not dry_run:
            rows = [asdict(item) for item in result.generations]
            payload["written_uri"] = write_generations(rows, result_uri=result.result_uri)
    except (TokenFactoryToolError, TokenFactoryError) as exc:
        _fail(str(exc))
        return
    _emit(payload, output)


@app.command("reason")
def reason_cmd(
    input_path: str = typer.Option(..., "--input-path", help="S3 or local path to scene images."),
    output_path: str = typer.Option(..., "--output-path", help="S3 or local path for the reasoning JSON."),
    task: str = typer.Option(DEFAULT_REASON_TASK, "--task", help="Task / question for the reasoner."),
    model: str = typer.Option(DEFAULT_REASONER_MODEL, "--model", help="Token Factory reasoning model."),
    system_prompt: str = typer.Option(
        DEFAULT_REASON_SYSTEM_PROMPT, "--system-prompt", help="System prompt for the reasoner."
    ),
    max_images: int = typer.Option(
        DEFAULT_REASON_MAX_IMAGES, "--max-images", help="Max scene images sent in one request."
    ),
    max_tokens: int = typer.Option(DEFAULT_REASON_MAX_TOKENS, "--max-tokens", help="Max tokens in the answer."),
    temperature: float = typer.Option(0.2, "--temperature", help="Sampling temperature."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Do not write the result artifact."),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Reason over scene images for physical understanding and a plan of action.

    Defaults to nvidia/Cosmos3-Super-Reasoner: point it at images of a scene and
    ask what a robot should do there.
    """

    try:
        result = reason_scene(
            input_path=input_path,
            output_path=output_path,
            task=task,
            model=model,
            system_prompt=system_prompt,
            max_images=max_images,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        payload = asdict(result)
        payload["dry_run"] = dry_run
        if not dry_run:
            payload["written_uri"] = write_reason(payload, result_uri=result.result_uri)
    except (TokenFactoryToolError, TokenFactoryError) as exc:
        _fail(str(exc))
        return
    _emit(payload, output)


@app.command("models")
def models_cmd(
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """List models available to the configured Token Factory API key."""

    try:
        models = list_models()
    except (TokenFactoryToolError, TokenFactoryError) as exc:
        _fail(str(exc))
        return
    _emit({"count": len(models), "models": models}, output)


@app.command("verify")
def verify_cmd(
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Verify Token Factory authentication with a live models call.

    Confirms NEBIUS_TOKEN_FACTORY_KEY is resolvable and the key authenticates against the
    configured base URL. Exits non-zero on any auth or connectivity failure.
    """

    config = resolve_config(require_api_key=False)
    if not config.api_key:
        _fail(
            "NEBIUS_TOKEN_FACTORY_KEY is not set. Add it under tokens: in "
            "~/.npa/credentials.yaml (run `npa configure`) or export "
            "NEBIUS_TOKEN_FACTORY_KEY."
        )
        return
    try:
        models = list_models()
    except (TokenFactoryToolError, TokenFactoryError) as exc:
        _fail(str(exc))
        return
    _emit(
        {
            "authenticated": True,
            "base_url": config.base_url,
            "model_count": len(models),
            "sample_models": models[:5],
        },
        output,
    )


@app.command("status")
def status_cmd(
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Show Token Factory connection status (no network call)."""

    config = resolve_config(require_api_key=False)
    _emit(
        {
            "provider": "nebius-token-factory",
            "base_url": config.base_url,
            "api_key_configured": bool(config.api_key),
            "default_text_model": DEFAULT_TEXT_MODEL,
            "default_vision_model": DEFAULT_VISION_MODEL,
            "default_reasoner_model": DEFAULT_REASONER_MODEL,
            "caption_workflow": str(CAPTION_WORKFLOW_PATH),
            "generate_workflow": str(GENERATE_WORKFLOW_PATH),
            "reason_workflow": str(REASON_WORKFLOW_PATH),
            "vlm_eval_workflow": str(VLM_EVAL_WORKFLOW_PATH),
        },
        output,
    )


@app.command("list")
def list_cmd(
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """List Token Factory tool capabilities."""

    _emit(
        {
            "default_base_url": DEFAULT_BASE_URL,
            "capabilities": [
                {"name": "caption", "kind": "vision", "default_model": DEFAULT_VISION_MODEL},
                {"name": "generate", "kind": "text", "default_model": DEFAULT_TEXT_MODEL},
                {"name": "reason", "kind": "physical-reasoning", "default_model": DEFAULT_REASONER_MODEL},
                {"name": "models", "kind": "discovery"},
            ],
        },
        output,
    )


@app.command("workflow")
def workflow_cmd(
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Show the checked-in Token Factory SkyPilot workflow templates."""

    _emit(
        {
            "caption_workflow": str(CAPTION_WORKFLOW_PATH),
            "generate_workflow": str(GENERATE_WORKFLOW_PATH),
            "reason_workflow": str(REASON_WORKFLOW_PATH),
            "vlm_eval_workflow": str(VLM_EVAL_WORKFLOW_PATH),
        },
        output,
    )


def _emit(payload: dict[str, Any], output: OutputFormat) -> None:
    if output == OutputFormat.json:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    for key, value in payload.items():
        typer.echo(f"  {key}: {value}")


def _fail(message: str) -> None:
    console.print(f"[red]Error:[/red] {message}")
    raise typer.Exit(1)
