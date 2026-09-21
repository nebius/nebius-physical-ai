"""CLI for Flex-Pi policy inference and pinned public training."""

from __future__ import annotations

import json

import typer

from npa.cli.path_contract import validate_read_path, validate_write_path
from npa.lifecycle_intent import json_stdout_contract
from npa.workbench.flex_pi.runtime import (
    DEFAULT_CHECKPOINT_ID,
    DEFAULT_CHECKPOINT_REVISION,
    DEFAULT_INPUT_MANIFEST,
    FlexPiError,
    FlexPiRequest,
    run_inference,
)

app = typer.Typer(
    name="flex-pi",
    help="Flex-Pi multi-stream policy inference and public training.",
    no_args_is_help=True,
)


@app.command("train")
@json_stdout_contract
def train_cmd(
    output_path: str = typer.Option(..., "--output-path"),
    mode: str = typer.Option("train", "--mode", help="train, profile, profile-resume"),
    microbatch_per_rank: int = typer.Option(
        1, "--microbatch-per-rank", help="1, or 3 after fixed-batch parity checks."
    ),
    compile_mode: str = typer.Option("off", "--compile-mode"),
    normalization_path: str = typer.Option("", "--normalization-path"),
    normalization_sha256: str = typer.Option("", "--normalization-sha256"),
    num_workers: int = typer.Option(4, "--num-workers", min=0),
    prefetch_factor: int = typer.Option(4, "--prefetch-factor", min=1),
    optimizer: str = typer.Option("default", "--optimizer"),
    run_id: str = typer.Option("", "--run-id"),
    runtime_image: str = typer.Option("", "--runtime-image"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    output_format: str = typer.Option("json", "--output-format"),
) -> None:
    """Train the pinned real public YAM workload on exactly four GPUs.

    Args:
        output_path: Authorized run-scoped S3 artifacts and checkpoint prefix.
        mode: Full train/validation/resume or the fixed profiling protocol.
        microbatch_per_rank: One anchor, or parity-qualified groups of three.
        compile_mode: Off, or parity-qualified RMSNorm compilation.
        normalization_path: Original run's statistics as an exact S3 object.
        normalization_sha256: Required content hash when reusing statistics.
        num_workers: Loader workers per participating GPU.
        prefetch_factor: Batches prefetched per loader worker.
        optimizer: Default, foreach, or fused AdamW execution.
        run_id: Workflow provenance identifier.
        runtime_image: Exact runtime image provenance.
        dry_run: Resolve the immutable contract without downloads or execution.
        output_format: JSON output contract.
    Returns:
        None; prints one JSON result.
    Raises:
        typer.Exit: Validation, training or any acceptance gate fails.
    """
    _execute_training_options(locals())


def _execute_training_options(options):
    from npa.workbench.flex_pi.training import TrainingRequest

    output_format = options.pop("output_format")
    _execute_training_request(output_format, TrainingRequest(**options))


def _execute_training_request(output_format, request):
    from npa.workbench.flex_pi.training import run_training

    try:
        if output_format != "json":
            raise ValueError("output-format must be json")
        validate_write_path(request.output_path, tool="flex-pi")
        result = run_training(request)
    except (FlexPiError, ValueError) as exc:
        typer.echo(f"flex-pi training failed: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


@app.command("infer")
@json_stdout_contract
def infer_cmd(
    input_path: str = typer.Option(DEFAULT_INPUT_MANIFEST, "--input-path"),
    output_path: str = typer.Option(..., "--output-path"),
    checkpoint_id: str = typer.Option(DEFAULT_CHECKPOINT_ID, "--checkpoint-id"),
    checkpoint_revision: str = typer.Option(
        DEFAULT_CHECKPOINT_REVISION, "--checkpoint-revision"
    ),
    num_inference_steps: int = typer.Option(4, "--num-inference-steps", min=1),
    seed: int = typer.Option(42, "--seed"),
    torch_compile: bool = typer.Option(False, "--torch-compile/--no-torch-compile"),
    expected_gpu: str = typer.Option("", "--expected-gpu"),
    run_id: str = typer.Option("", "--run-id"),
    runtime_image: str = typer.Option("", "--runtime-image"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    output_format: str = typer.Option(
        "json", "--output-format", help="Result format; must be json."
    ),
) -> None:
    """Run genuine action-only inference and publish verified artifacts.

    Args:
        input_path: Local or S3 public-observation manifest.
        output_path: Authorized S3 artifact prefix.
        checkpoint_id: Hugging Face checkpoint repository.
        checkpoint_revision: Immutable checkpoint commit.
        num_inference_steps: Flow-matching denoising steps.
        seed: Action-noise seed.
        torch_compile: Compile the denoising step after model load.
        expected_gpu: Optional normalized GPU-name substring.
        run_id: Workflow provenance identifier.
        runtime_image: Runtime image provenance override.
        dry_run: Resolve the plan without model execution.
        output_format: Machine-readable JSON result format.
    Returns:
        None; prints one JSON result.
    Raises:
        typer.Exit: Validation or inference fails.
    """
    try:
        if output_format != "json":
            raise ValueError("output-format must be json")
        if input_path != DEFAULT_INPUT_MANIFEST:
            validate_read_path(input_path, tool="flex-pi")
        validate_write_path(output_path, tool="flex-pi")
        request = FlexPiRequest(
            input_path=input_path,
            output_path=output_path,
            checkpoint_id=checkpoint_id,
            checkpoint_revision=checkpoint_revision,
            num_inference_steps=num_inference_steps,
            seed=seed,
            torch_compile=torch_compile,
            expected_gpu=expected_gpu,
            run_id=run_id,
            runtime_image=runtime_image,
            dry_run=dry_run,
        )
        result = run_inference(request)
    except (FlexPiError, ValueError) as exc:
        typer.echo(f"flex-pi inference failed: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


@app.command("terms")
def terms_cmd() -> None:
    """Print separately applicable source, model, and public-input terms."""
    typer.echo(
        json.dumps(
            {
                "source": {"license": "MIT", "baked": True},
                "checkpoint": {"license": "MIT", "runtime_fetch": True},
                "wan_runtime_assets": {"license": "Apache-2.0", "runtime_fetch": True},
                "dinov3_runtime_assets": {
                    "license": "upstream-specific",
                    "runtime_fetch": True,
                },
                "public_input": {
                    "dataset": "flex-pi/robotwin_3d",
                    "license": "not-declared",
                    "runtime_fetch": True,
                    "redistribution": False,
                },
                "public_training_input": {
                    "dataset": "flex-pi/sort_utensils",
                    "revision": "0780dd0a0b281df91abcef9434c4b3ac2757448c",
                    "license": "CC-BY-4.0",
                    "runtime_fetch": True,
                    "redistribution": False,
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
