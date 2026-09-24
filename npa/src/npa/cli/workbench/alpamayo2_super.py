"""CLI for NVIDIA Alpamayo 2 Super."""

from __future__ import annotations

import json
from contextlib import redirect_stdout
import sys

import typer
from npa.cli.path_contract import validate_read_path, validate_write_path

from npa.workbench.alpamayo2_super.runtime import (
    DEFAULT_DATASET_REVISION,
    DEFAULT_MANIFEST,
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    Alpamayo2SuperError,
    Alpamayo2SuperRequest,
    run_inference,
)

app = typer.Typer(
    name="alpamayo2-super",
    help="NVIDIA Alpamayo 2 Super trajectory-inference workbench.",
    no_args_is_help=True,
)


@app.command("sweep")
def sweep_cmd(
    output_path: str = typer.Option(..., "--output-path"),
    run_id: str = typer.Option(..., "--run-id"),
    sample_indices: str = typer.Option("0,1", "--sample-indices"),
    seeds: str = typer.Option("42,43", "--seeds"),
    diffusion_steps: str = typer.Option("10,20", "--diffusion-steps"),
    workers: int = typer.Option(1, "--workers", min=1),
    input_path: str = typer.Option("", "--input-path"),
    minimum_ade: float = typer.Option(2.0, "--minimum-ade", min=0),
) -> None:
    """Sweep scenarios with Ray; optionally refine a baseline report's hard cases.

    Args:
        output_path: S3 report destination.
        run_id: Experiment identity.
        sample_indices: Comma-separated manifest indices for a baseline.
        seeds: Comma-separated seeds; refinement inherits baseline seeds.
        diffusion_steps: Comma-separated integration-step counts.
        workers: GPU actor count on this allocated node.
        input_path: Optional S3 baseline report.json.
        minimum_ade: Refine baseline scenarios above this mean ADE in meters.
    Returns:
        None; writes a JSON report to stdout.
    Raises:
        typer.Exit: Input validation or inference fails.
    """
    _sweep_command(locals())


def _sweep_command(values: dict) -> None:
    from npa.workbench.alpamayo2_super.ray_sweep import AlpamayoSweepRequest, run_sweep

    try:
        validate_write_path(values["output_path"], tool="alpamayo2-super")
        if values["input_path"]:
            validate_read_path(values["input_path"], tool="alpamayo2-super")
        for name in ("sample_indices", "seeds", "diffusion_steps"):
            values[name] = [int(value.strip()) for value in values[name].split(",")]
        with redirect_stdout(sys.stderr):
            result = run_sweep(AlpamayoSweepRequest(**values))
    except (ValueError, Alpamayo2SuperError) as exc:
        typer.echo(f"Alpamayo sweep failed: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


@app.command("infer")
def infer_cmd(
    output_path: str = typer.Option(
        ..., "--output-path", help="Local directory or s3:// prefix."
    ),
    model_id: str = typer.Option(DEFAULT_MODEL_ID, "--model-id"),
    model_revision: str = typer.Option(DEFAULT_MODEL_REVISION, "--model-revision"),
    dataset_revision: str = typer.Option(
        DEFAULT_DATASET_REVISION, "--dataset-revision"
    ),
    manifest: str = typer.Option(DEFAULT_MANIFEST, "--manifest"),
    sample_index: int = typer.Option(0, "--sample-index"),
    diffusion_steps: int = typer.Option(10, "--diffusion-steps"),
    seed: int = typer.Option(42, "--seed"),
    figure_style: str = typer.Option("blog", "--figure-style"),
    require_camera_projection: bool = typer.Option(
        True, "--require-camera-projection/--allow-missing-camera-projection"
    ),
    run_id: str = typer.Option("", "--run-id"),
    runtime_image: str = typer.Option("", "--runtime-image"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Run the real upstream expert trajectory inference and publish artifacts."""

    try:
        payload = run_inference(
            Alpamayo2SuperRequest(
                output_path=output_path,
                model_id=model_id,
                model_revision=model_revision,
                dataset_revision=dataset_revision,
                manifest=manifest,
                sample_index=sample_index,
                diffusion_steps=diffusion_steps,
                seed=seed,
                figure_style=figure_style,
                require_camera_projection=require_camera_projection,
                run_id=run_id,
                runtime_image=runtime_image,
                dry_run=dry_run,
            )
        )
    except Alpamayo2SuperError as exc:
        typer.echo(f"Alpamayo 2 Super inference failed: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@app.command("terms")
def terms_cmd() -> None:
    """Print separately applicable source, model, and dataset terms."""

    typer.echo(
        json.dumps(
            {
                "source": {"license": "Apache-2.0", "baked": True},
                "model": {
                    "id": DEFAULT_MODEL_ID,
                    "revision": DEFAULT_MODEL_REVISION,
                    "license": "OpenMDW-1.1",
                    "acceptance": "by exercising rights under the agreement",
                    "runtime_fetch": True,
                },
                "dataset": {
                    "id": "nvidia/PhysicalAI-Autonomous-Vehicles",
                    "revision": DEFAULT_DATASET_REVISION,
                    "license": "NVIDIA Autonomous Vehicle Dataset License Agreement",
                    "acceptance": "interactive on the Hugging Face dataset page",
                    "runtime_fetch": True,
                    "redistribution": False,
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
