"""CLI for flex-pi world-action policy inference."""

from __future__ import annotations

import json

import typer

from npa.cli.path_contract import validate_read_path, validate_write_path
from npa.workbench.flex_pi.runtime import (
    DEFAULT_CHECKPOINT_ID,
    DEFAULT_CHECKPOINT_REVISION,
    DEFAULT_INPUT_MANIFEST,
    FlexPiError,
    FlexPiRequest,
    run_inference,
)

app = typer.Typer(
    name="flex-pi", help="Flex-pi multi-stream world-action policy inference.",
    no_args_is_help=True,
)


@app.command("infer")
def infer_cmd(
    input_path: str = typer.Option(DEFAULT_INPUT_MANIFEST, "--input-path"),
    output_path: str = typer.Option(..., "--output-path"),
    checkpoint_id: str = typer.Option(DEFAULT_CHECKPOINT_ID, "--checkpoint-id"),
    checkpoint_revision: str = typer.Option(DEFAULT_CHECKPOINT_REVISION, "--checkpoint-revision"),
    num_inference_steps: int = typer.Option(4, "--num-inference-steps", min=1),
    seed: int = typer.Option(42, "--seed"),
    torch_compile: bool = typer.Option(False, "--torch-compile/--no-torch-compile"),
    expected_gpu: str = typer.Option("", "--expected-gpu"),
    run_id: str = typer.Option("", "--run-id"),
    runtime_image: str = typer.Option("", "--runtime-image"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Run genuine action-only inference and publish verified artifacts.

    Args:
        input_path: Local or S3 public-observation manifest.
        output_path: Local directory or S3 artifact prefix.
        checkpoint_id: Hugging Face checkpoint repository.
        checkpoint_revision: Immutable checkpoint commit.
        num_inference_steps: Flow-matching denoising steps.
        seed: Action-noise seed.
        torch_compile: Compile the denoising step after model load.
        expected_gpu: Optional normalized GPU-name substring.
        run_id: Workflow provenance identifier.
        runtime_image: Runtime image provenance override.
        dry_run: Resolve the plan without model execution.
    Returns:
        None; prints one JSON result.
    Raises:
        typer.Exit: Validation or inference fails.
    """
    try:
        if input_path != DEFAULT_INPUT_MANIFEST:
            validate_read_path(input_path, tool="flex-pi")
        validate_write_path(output_path, tool="flex-pi")
        result = run_inference(FlexPiRequest(**locals()))
    except (FlexPiError, ValueError) as exc:
        typer.echo(f"flex-pi inference failed: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


@app.command("terms")
def terms_cmd() -> None:
    """Print separately applicable source, model, and public-input terms."""
    typer.echo(json.dumps({
        "source": {"license": "MIT", "baked": True},
        "checkpoint": {"license": "MIT", "runtime_fetch": True},
        "wan_runtime_assets": {"license": "Apache-2.0", "runtime_fetch": True},
        "dinov3_runtime_assets": {"license": "upstream-specific", "runtime_fetch": True},
        "public_input": {"dataset": "flex-pi/robotwin_3d", "license": "not-declared", "runtime_fetch": True, "redistribution": False},
    }, indent=2, sort_keys=True))
