"""npa.workbench.nurec - NVIDIA Omniverse NuRec / Neural Reconstruction Engine.

Turns a real sensor recording (NCore V4) into a 3D Gaussian reconstruction, a
renderable USDZ, and novel-view renders by driving NVIDIA's public NRE container
on a Nebius RT-core GPU.
"""

from __future__ import annotations

from npa._sdk import make_cli_wrapper
from npa.workbench.nurec.nurec import (
    DEFAULT_CONFIG_NAME as DEFAULT_CONFIG_NAME,
    DEFAULT_DATASET_ID as DEFAULT_DATASET_ID,
    DEFAULT_NRE_ENTRYPOINT as DEFAULT_NRE_ENTRYPOINT,
    DEFAULT_NRE_IMAGE as DEFAULT_NRE_IMAGE,
    DEFAULT_NRE_TOOLS_IMAGE as DEFAULT_NRE_TOOLS_IMAGE,
    DEFAULT_SCENE as DEFAULT_SCENE,
    DEFAULT_VARIANT as DEFAULT_VARIANT,
    NO_LIDAR_SENTINEL as NO_LIDAR_SENTINEL,
    NurecCheckResult as NurecCheckResult,
    NurecConfig as NurecConfig,
    NurecError as NurecError,
    NurecFetchResult as NurecFetchResult,
    NurecReconstructResult as NurecReconstructResult,
    NurecRenderResult as NurecRenderResult,
    NurecStatusResult as NurecStatusResult,
    build_docker_wrapper,
    build_nre_export_gt_args,
    build_nre_render_args,
    build_nre_train_args,
    check_nurec_access,
    count_render_frames,
    extract_archive,
    fetch_nurec_dataset,
    find_ncore_json,
    find_scene_dir,
    has_rt_cores,
    ncore_sensor_ids,
    nre_command,
    nurec_run_status,
    parse_metrics_yaml,
    parse_offset,
    reconstruct_scene,
    render_novel_views,
    resolve_nre_run_dir,
)

check = make_cli_wrapper(
    "npa.cli.nurec", "check_cmd", "Check NRE container, dataset, and GPU access."
)
fetch = make_cli_wrapper(
    "npa.cli.nurec", "fetch_cmd", "Fetch real NCore V4 shards for a scene."
)
reconstruct = make_cli_wrapper(
    "npa.cli.nurec", "reconstruct_cmd", "Train a 3DGUT reconstruction into a USDZ."
)
render = make_cli_wrapper(
    "npa.cli.nurec", "render_cmd", "Render novel views from a trained USDZ."
)
visualize = make_cli_wrapper(
    "npa.cli.nurec", "visualize_cmd", "Build the run's Rerun recording."
)
finalize = make_cli_wrapper(
    "npa.cli.nurec", "finalize_cmd", "Write the run's aggregate report."
)
status = make_cli_wrapper("npa.cli.nurec", "status_cmd", "Summarize a NuRec run prefix.")

__all__ = [
    "DEFAULT_CONFIG_NAME",
    "DEFAULT_DATASET_ID",
    "DEFAULT_NRE_ENTRYPOINT",
    "DEFAULT_NRE_IMAGE",
    "DEFAULT_NRE_TOOLS_IMAGE",
    "DEFAULT_SCENE",
    "DEFAULT_VARIANT",
    "NO_LIDAR_SENTINEL",
    "NurecCheckResult",
    "NurecConfig",
    "NurecError",
    "NurecFetchResult",
    "NurecReconstructResult",
    "NurecRenderResult",
    "NurecStatusResult",
    "build_docker_wrapper",
    "build_nre_export_gt_args",
    "build_nre_render_args",
    "build_nre_train_args",
    "check",
    "check_nurec_access",
    "count_render_frames",
    "extract_archive",
    "fetch",
    "fetch_nurec_dataset",
    "finalize",
    "find_ncore_json",
    "find_scene_dir",
    "has_rt_cores",
    "ncore_sensor_ids",
    "nre_command",
    "nurec_run_status",
    "parse_metrics_yaml",
    "parse_offset",
    "reconstruct",
    "reconstruct_scene",
    "render",
    "render_novel_views",
    "resolve_nre_run_dir",
    "status",
    "visualize",
]
