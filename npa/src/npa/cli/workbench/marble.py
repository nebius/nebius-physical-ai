"""Expose hosted Marble acquisition and GPU consumers through thin CLI commands."""

import json

import typer

from npa.lifecycle_intent import json_stdout_contract
from npa.workbench.marble import runtime
from npa.workbench.marble.api import MarbleError
from npa.workbench.marble.schemas import (
    AcquireRequest,
    PalletBenchmarkRequest,
    RunRequest,
    RoverRequest,
)

app = typer.Typer(
    help="World Labs Marble worlds, CUDA camera datasets, and spatial scans.",
    no_args_is_help=True,
)


@app.command("rover-collect")
@json_stdout_contract
def rover_collect_cmd(
    input_path: str = typer.Option(..., "--input-path"),
    output_path: str = typer.Option(..., "--output-path"),
    run_id: str = typer.Option(..., "--run-id"),
    frames: int = typer.Option(240, "--frames"),
    width: int = typer.Option(960, "--width"),
    height: int = typer.Option(540, "--height"),
    sensor_hz: int = typer.Option(12, "--sensor-hz"),
    output_format: str = typer.Option("json", "--output-format"),
):
    """Collect RGB, depth, actions, and contact states from a wheel-driven rover.

    Args: World/result prefixes, identity, dimensions, sensor rate, and format.
    Returns: None; writes measured execution JSON.
    Raises: typer.Exit on invalid geometry, physics, CUDA, or storage evidence.
    """
    from npa.workbench.marble.rover import rover_collect

    _call(
        rover_collect,
        RoverRequest(
            input_path=input_path,
            output_path=output_path,
            run_id=run_id,
            frames=frames,
            width=width,
            height=height,
            sensor_hz=sensor_hz,
        ),
    )


def _call(operation, request):
    try:
        result = operation(request)
    except (MarbleError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    typer.echo(json.dumps(result, sort_keys=True))


@app.command("acquire")
@json_stdout_contract
def acquire_cmd(
    output_path: str = typer.Option(..., "--output-path"),
    run_id: str = typer.Option(..., "--run-id"),
    prompt: str = typer.Option(..., "--prompt"),
    source: str = typer.Option("generate", "--source"),
    model: str = typer.Option("marble-1.1", "--model"),
    output_format: str = typer.Option("json", "--output-format"),
):
    """Acquire a hosted world or the explicitly attributed upstream example.

    Args: Source, prompt, model, S3 destination, run identity, and JSON format.
    Returns: None; writes one JSON result.
    Raises: typer.Exit if acquisition fails.
    """
    _call(
        runtime.acquire,
        AcquireRequest(
            output_path=output_path,
            run_id=run_id,
            prompt=prompt,
            source=source,
            model=model,
        ),
    )


@app.command("capture")
@json_stdout_contract
def capture_cmd(
    input_path: str = typer.Option(..., "--input-path"),
    output_path: str = typer.Option(..., "--output-path"),
    run_id: str = typer.Option(..., "--run-id"),
    frames: int = typer.Option(120, "--frames"),
    width: int = typer.Option(960, "--width"),
    height: int = typer.Option(540, "--height"),
    output_format: str = typer.Option("json", "--output-format"),
):
    """Render real Gaussian splats with gsplat on a required CUDA GPU.

    Args: World/output S3 prefixes, run identity, image dimensions, and format.
    Returns: None; writes measured execution JSON.
    Raises: typer.Exit on invalid inputs or runtime failure.
    """
    _call(
        runtime.capture,
        RunRequest(
            input_path=input_path,
            output_path=output_path,
            run_id=run_id,
            frames=frames,
            width=width,
            height=height,
        ),
    )


@app.command("scan")
@json_stdout_contract
def scan_cmd(
    input_path: str = typer.Option(..., "--input-path"),
    output_path: str = typer.Option(..., "--output-path"),
    run_id: str = typer.Option(..., "--run-id"),
    frames: int = typer.Option(120, "--frames"),
    width: int = typer.Option(480, "--width"),
    height: int = typer.Option(270, "--height"),
    output_format: str = typer.Option("json", "--output-format"),
):
    """Measure depth and clearance against real mesh triangles on CUDA.

    Args: World/output S3 prefixes, run identity, scan dimensions, and format.
    Returns: None; writes measured execution JSON.
    Raises: typer.Exit on invalid inputs or runtime failure.
    """
    _call(
        runtime.scan,
        RunRequest(
            input_path=input_path,
            output_path=output_path,
            run_id=run_id,
            frames=frames,
            width=width,
            height=height,
        ),
    )


@app.command("report")
@json_stdout_contract
def report_cmd(
    input_path: str = typer.Option(..., "--input-path"),
    output_path: str = typer.Option(..., "--output-path"),
    run_id: str = typer.Option(..., "--run-id"),
    output_format: str = typer.Option("json", "--output-format"),
):
    """Build the interactive HTML report from verified GPU results.

    Args: GPU result and report prefixes, run identity, and JSON format.
    Returns: None; writes publication JSON.
    Raises: typer.Exit on incomplete or invalid evidence.
    """
    _call(
        runtime.report,
        RunRequest(input_path=input_path, output_path=output_path, run_id=run_id),
    )


@app.command("pallet-preflight")
@json_stdout_contract
def pallet_preflight_cmd(
    input_path: str = typer.Option(..., "--input-path"),
    output_path: str = typer.Option(..., "--output-path"),
    run_id: str = typer.Option(..., "--run-id"),
    output_format: str = typer.Option("json", "--output-format"),
):
    """Require the World API key and snapshot disjoint labeled pallet data.

    Args: Dataset manifest URI, snapshot prefix, run identity, and format.
    Returns: None; writes validated dataset counts as JSON.
    Raises: typer.Exit for missing auth, invalid inputs, or data leakage.
    """
    from npa.workbench.marble.pallet_data import pallet_preflight

    _call(
        pallet_preflight,
        RunRequest(input_path=input_path, output_path=output_path, run_id=run_id),
    )


@app.command("pallet-benchmark")
@json_stdout_contract
def pallet_benchmark_cmd(
    input_path: str = typer.Option(..., "--input-path"),
    dataset_path: str = typer.Option(..., "--dataset-path"),
    output_path: str = typer.Option(..., "--output-path"),
    run_id: str = typer.Option(..., "--run-id"),
    epochs: int = typer.Option(10, "--epochs"),
    batch_size: int = typer.Option(2, "--batch-size"),
    learning_rate: float = typer.Option(0.005, "--learning-rate"),
    seed: int = typer.Option(42, "--seed"),
    output_format: str = typer.Option("json", "--output-format"),
):
    """Train matched-budget pallet detectors and compare held-out real-image AP.

    Args: Capture/snapshot/output prefixes, run identity, and training settings.
    Returns: None; writes the measured signed AP changes as JSON.
    Raises: typer.Exit on invalid inputs or runtime failure.
    """
    from npa.workbench.marble.pallet_benchmark import pallet_benchmark

    _call(
        pallet_benchmark,
        PalletBenchmarkRequest(
            input_path=input_path,
            dataset_path=dataset_path,
            output_path=output_path,
            run_id=run_id,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
        ),
    )


@app.command("pallet-report")
@json_stdout_contract
def pallet_report_cmd(
    input_path: str = typer.Option(..., "--input-path"),
    output_path: str = typer.Option(..., "--output-path"),
    run_id: str = typer.Option(..., "--run-id"),
    output_format: str = typer.Option("json", "--output-format"),
):
    """Publish the measured pallet comparison as static JSON and HTML.

    Args: Benchmark/report prefixes, run identity, and format.
    Returns: None; writes the publication outcome as JSON.
    Raises: typer.Exit for invalid evidence or failed publication.
    """
    from npa.workbench.marble.pallet_report import pallet_report

    _call(
        pallet_report,
        RunRequest(input_path=input_path, output_path=output_path, run_id=run_id),
    )
