"""SDK clients for NVIDIA NCore ingestion and NuRec / NRE reconstruction.

Neural reconstruction: a real sensor capture (NCore V4) becomes a 3D Gaussian
reconstruction, a renderable USDZ, and novel-view renders, driven by NVIDIA's
proprietary NRE container on an RT-core GPU. Apache-2.0 NCore ingestion
is a separate CPU capability. See
``skills/workflows/neural-reconstruction/SKILL.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from npa._sdk import make_cli_wrapper


def convert_colmap(
    input_path: str,
    output_path: str,
    *,
    cache_dir: Path | str = "/tmp/npa-ncore-cache",
    scratch_dir: Path | str = "/tmp/npa-ncore-scratch",
    dataset_root: str = ".",
    colmap_dir: str = "sparse/0",
    images_dir: str = "images",
    masks_dir: str = "",
    rig_mode: Literal["derive", "preserve"] = "derive",
    reference_camera: str = "",
    include_downsampled_images: bool = True,
) -> dict[str, Any]:
    """Convert an S3 COLMAP dataset with NVIDIA NCore and publish a verified V4 sequence."""
    from npa.workbench.nurec.colmap import (
        ColmapConversionRequest,
        convert_colmap as run,
    )

    return run(
        ColmapConversionRequest(
            input_path=input_path,
            output_path=output_path,
            cache_dir=Path(cache_dir),
            scratch_dir=Path(scratch_dir),
            dataset_root=dataset_root,
            colmap_dir=colmap_dir,
            images_dir=images_dir,
            masks_dir=masks_dir,
            rig_mode=rig_mode,
            reference_camera=reference_camera,
            include_downsampled_images=include_downsampled_images,
        )
    )


check = make_cli_wrapper(
    "npa.cli.nurec",
    "check_cmd",
    "Check NRE container access, dataset download rights, and GPU suitability.",
)
fetch = make_cli_wrapper(
    "npa.cli.nurec",
    "fetch_cmd",
    "Download real NCore V4 shards and derive the rig->world pose edge NRE needs.",
)
reconstruct = make_cli_wrapper(
    "npa.cli.nurec",
    "reconstruct_cmd",
    "Train a 3DGUT Gaussian reconstruction into a renderable USDZ.",
)
render = make_cli_wrapper(
    "npa.cli.nurec",
    "render_cmd",
    "Render novel views from a trained reconstruction.",
)
visualize = make_cli_wrapper(
    "npa.cli.nurec",
    "visualize_cmd",
    "Build the run's Rerun recording for the NPA agent viewer.",
)
finalize = make_cli_wrapper(
    "npa.cli.nurec", "finalize_cmd", "Aggregate a NuRec run tree into a final report."
)
status = make_cli_wrapper(
    "npa.cli.nurec", "status_cmd", "Summarize a NuRec run prefix, stage by stage."
)

__all__ = [
    "check",
    "convert_colmap",
    "fetch",
    "finalize",
    "reconstruct",
    "render",
    "status",
    "visualize",
]
