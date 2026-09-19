"""Thin Open3D CLI; all capability behavior lives in the workbench module."""

from __future__ import annotations

import json
from enum import Enum
from typing import Any, Callable

import typer

from npa.lifecycle_intent import json_stdout_contract
from npa.workbench.open3d import runtime
from npa.workbench.open3d.artifacts import Open3dError
from npa.workbench.open3d.schemas import (
    PrepareRequest,
    ReconstructRequest,
    RunRequest,
    StageDemoRequest,
)

app = typer.Typer(
    name="open3d",
    help="Open3D point-cloud registration and surface reconstruction.",
    no_args_is_help=True,
)


class OutputFormat(str, Enum):
    json = "json"
    text = "text"


def _call(operation: Callable[[Any], dict[str, Any]], request: Any) -> None:
    try:
        result = operation(request)
    except (Open3dError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    typer.echo(json.dumps(result, sort_keys=True, indent=2))


@app.command("prepare")
@json_stdout_contract
def prepare_cmd(
    input_path: str = typer.Option(
        ...,
        "--input-path",
        help="Prefix holding at least two .pcd/.ply scans to register.",
    ),
    output_path: str = typer.Option(
        ..., "--output-path", help="Prefix to write manifest.json under."
    ),
    voxel_size: float = typer.Option(
        0.05,
        "--voxel-size",
        help="voxel_down_sample edge length, in the scans' own units.",
    ),
    icp_estimation: str = typer.Option(
        "point_to_plane",
        "--icp-estimation",
        help="ICP estimation method: point_to_plane or point_to_point.",
    ),
    output_format: OutputFormat = typer.Option(
        OutputFormat.json, "--output-format", help="Output format."
    ),
) -> None:
    """Index the scans under a prefix into a registration manifest."""
    _call(
        runtime.prepare,
        PrepareRequest(
            input_path=input_path,
            output_path=output_path,
            voxel_size=voxel_size,
            icp_estimation=icp_estimation,
        ),
    )


@app.command("stage-demo")
@json_stdout_contract
def stage_demo_cmd(
    output_path: str = typer.Option(
        ...,
        "--output-path",
        help="Prefix to publish the staged scans and manifest.json under.",
    ),
    voxel_size: float = typer.Option(
        0.05,
        "--voxel-size",
        help="voxel_down_sample edge length, in the scans' own units.",
    ),
    icp_estimation: str = typer.Option(
        "point_to_plane",
        "--icp-estimation",
        help="ICP estimation method: point_to_plane or point_to_point.",
    ),
    output_format: OutputFormat = typer.Option(
        OutputFormat.json, "--output-format", help="Output format."
    ),
) -> None:
    """Publish the upstream open3d.data DemoICPPointClouds scans as input."""
    _call(
        runtime.stage_demo,
        StageDemoRequest(
            output_path=output_path,
            voxel_size=voxel_size,
            icp_estimation=icp_estimation,
        ),
    )


@app.command("register")
@json_stdout_contract
def register_cmd(
    input_path: str = typer.Option(
        ..., "--input-path", help="Prefix holding manifest.json from prepare."
    ),
    output_path: str = typer.Option(..., "--output-path"),
    run_id: str = typer.Option(..., "--run-id"),
    output_format: OutputFormat = typer.Option(
        OutputFormat.json, "--output-format", help="Output format."
    ),
) -> None:
    """Register consecutive pairs with RANSAC/FPFH, then refine with ICP."""
    _call(
        runtime.register,
        RunRequest(input_path=input_path, output_path=output_path, run_id=run_id),
    )


@app.command("multiway")
@json_stdout_contract
def multiway_cmd(
    input_path: str = typer.Option(
        ..., "--input-path", help="Prefix holding manifest.json from prepare."
    ),
    output_path: str = typer.Option(..., "--output-path"),
    run_id: str = typer.Option(..., "--run-id"),
    output_format: OutputFormat = typer.Option(
        OutputFormat.json, "--output-format", help="Output format."
    ),
) -> None:
    """Build the full pairwise pose graph, optimize it, and fuse the scans."""
    _call(
        runtime.multiway,
        RunRequest(input_path=input_path, output_path=output_path, run_id=run_id),
    )


@app.command("validate")
@json_stdout_contract
def validate_cmd(
    input_path: str = typer.Option(
        ..., "--input-path", help="Prefix holding a register or multiway result."
    ),
    output_path: str = typer.Option(..., "--output-path"),
    run_id: str = typer.Option(..., "--run-id"),
    output_format: OutputFormat = typer.Option(
        OutputFormat.json, "--output-format", help="Output format."
    ),
) -> None:
    """Re-verify a published registration from its artifacts alone."""
    _call(
        runtime.validate,
        RunRequest(input_path=input_path, output_path=output_path, run_id=run_id),
    )


@app.command("reconstruct")
@json_stdout_contract
def reconstruct_cmd(
    input_path: str = typer.Option(
        ..., "--input-path", help="Prefix holding a multiway result and fused.ply."
    ),
    output_path: str = typer.Option(..., "--output-path"),
    run_id: str = typer.Option(..., "--run-id"),
    poisson_depth: int = typer.Option(
        9, "--poisson-depth", help="create_from_point_cloud_poisson octree depth."
    ),
    density_quantile: float = typer.Option(
        0.02,
        "--density-quantile",
        help="Drop this lowest-density fraction of Poisson vertices.",
    ),
    output_format: OutputFormat = typer.Option(
        OutputFormat.json, "--output-format", help="Output format."
    ),
) -> None:
    """Reconstruct a Poisson surface from the fused multiway cloud."""
    _call(
        runtime.reconstruct,
        ReconstructRequest(
            input_path=input_path,
            output_path=output_path,
            run_id=run_id,
            poisson_depth=poisson_depth,
            density_quantile=density_quantile,
        ),
    )


@app.command("visualize")
@json_stdout_contract
def visualize_cmd(
    input_path: str = typer.Option(
        ..., "--input-path", help="Prefix holding a reconstruct result and mesh.ply."
    ),
    output_path: str = typer.Option(..., "--output-path"),
    run_id: str = typer.Option(..., "--run-id"),
    output_format: OutputFormat = typer.Option(
        OutputFormat.json, "--output-format", help="Output format."
    ),
) -> None:
    """Build and verify an RRD of the optimized scans, fusion and surface."""
    _call(
        runtime.visualize,
        RunRequest(input_path=input_path, output_path=output_path, run_id=run_id),
    )
