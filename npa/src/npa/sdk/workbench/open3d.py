"""Open3D SDK, using the same request models and operations as CLI/workflow."""

from __future__ import annotations

from typing import Any

from npa.workbench.open3d import runtime
from npa.workbench.open3d.schemas import (
    PrepareRequest,
    ReconstructRequest,
    RunRequest,
    StageDemoRequest,
)


def prepare(
    *,
    input_path: str,
    output_path: str,
    voxel_size: float = 0.05,
    icp_estimation: str = "point_to_plane",
) -> dict[str, Any]:
    return runtime.prepare(
        PrepareRequest(
            input_path=input_path,
            output_path=output_path,
            voxel_size=voxel_size,
            icp_estimation=icp_estimation,
        )
    )


def stage_demo(
    *,
    output_path: str,
    voxel_size: float = 0.05,
    icp_estimation: str = "point_to_plane",
) -> dict[str, Any]:
    return runtime.stage_demo(
        StageDemoRequest(
            output_path=output_path,
            voxel_size=voxel_size,
            icp_estimation=icp_estimation,
        )
    )


def register(*, input_path: str, output_path: str, run_id: str) -> dict[str, Any]:
    return runtime.register(
        RunRequest(input_path=input_path, output_path=output_path, run_id=run_id)
    )


def multiway(*, input_path: str, output_path: str, run_id: str) -> dict[str, Any]:
    return runtime.multiway(
        RunRequest(input_path=input_path, output_path=output_path, run_id=run_id)
    )


def validate(*, input_path: str, output_path: str, run_id: str) -> dict[str, Any]:
    return runtime.validate(
        RunRequest(input_path=input_path, output_path=output_path, run_id=run_id)
    )


def reconstruct(
    *,
    input_path: str,
    output_path: str,
    run_id: str,
    poisson_depth: int = 9,
    density_quantile: float = 0.02,
) -> dict[str, Any]:
    return runtime.reconstruct(
        ReconstructRequest(
            input_path=input_path,
            output_path=output_path,
            run_id=run_id,
            poisson_depth=poisson_depth,
            density_quantile=density_quantile,
        )
    )


def visualize(*, input_path: str, output_path: str, run_id: str) -> dict[str, Any]:
    return runtime.visualize(
        RunRequest(input_path=input_path, output_path=output_path, run_id=run_id)
    )
