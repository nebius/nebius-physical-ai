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
from npa.workbench.ncore_staging import (
    DEFAULT_COLMAP_CACHE_DIR,
    DEFAULT_COLMAP_SCRATCH_DIR,
)


def convert_colmap(
    input_path: str,
    output_path: str,
    *,
    cache_dir: Path | str = DEFAULT_COLMAP_CACHE_DIR,
    scratch_dir: Path | str = DEFAULT_COLMAP_SCRATCH_DIR,
    dataset_root: str = ".",
    colmap_dir: str = "sparse/0",
    images_dir: str = "images",
    masks_dir: str = "",
    expected_archive_sha256: str = "",
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
            expected_archive_sha256=expected_archive_sha256,
            rig_mode=rig_mode,
            reference_camera=reference_camera,
            include_downsampled_images=include_downsampled_images,
        )
    )


def audit_colmap(
    input_path: str,
    conversion_path: str,
    output_path: str,
    *,
    expected_archive_sha256: str,
    cache_dir: Path | str = DEFAULT_COLMAP_CACHE_DIR,
    scratch_dir: Path | str = DEFAULT_COLMAP_SCRATCH_DIR,
    dataset_root: str = ".",
    colmap_dir: str = "sparse/0",
    images_dir: str = "images",
    masks_dir: str = "",
    rig_mode: Literal["derive", "preserve"] = "derive",
    reference_camera: str = "",
    include_downsampled_images: bool = True,
) -> dict[str, Any]:
    """Independently re-download and decode a published NCore V4 generation."""
    from npa.workbench.nurec.ncore_audit import (
        ColmapAuditRequest,
        audit_colmap_conversion,
    )

    return audit_colmap_conversion(
        ColmapAuditRequest(
            input_path=input_path,
            conversion_path=conversion_path,
            output_path=output_path,
            expected_archive_sha256=expected_archive_sha256,
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


def probe_storage(prefix: str, receipt_path: Path | str) -> dict[str, Any]:
    """Prove conditional create/read/list/delete on one fresh S3 run prefix."""
    from npa.workbench.nurec.s3_probe import probe_s3_handoff

    return probe_s3_handoff(prefix, Path(receipt_path))


def control_source(
    input_path: str,
    output_path: str,
    *,
    expected_archive_sha256: str,
    receipt_path: Path | str,
    cache_dir: Path | str = DEFAULT_COLMAP_CACHE_DIR,
    scratch_dir: Path | str = DEFAULT_COLMAP_SCRATCH_DIR,
) -> dict[str, Any]:
    """Prove a wrong source hash is rejected before extraction and publication."""
    from npa.workbench.nurec.colmap import ColmapConversionRequest
    from npa.workbench.nurec.source_control import run_wrong_source_control

    request = ColmapConversionRequest(
        input_path=input_path,
        output_path=output_path,
        expected_archive_sha256=expected_archive_sha256,
        cache_dir=Path(cache_dir),
        scratch_dir=Path(scratch_dir),
    )
    return run_wrong_source_control(
        request,
        expected_archive_sha256=expected_archive_sha256,
        receipt_path=Path(receipt_path),
    )


def stage_source(
    source_path: Path | str,
    output_path: str,
    *,
    expected_archive_sha256: str,
    receipt_path: Path | str,
    scratch_dir: Path | str = DEFAULT_COLMAP_SCRATCH_DIR,
) -> dict[str, Any]:
    """Conditionally stage and independently read back a pinned source ZIP."""
    from npa.workbench.nurec.source_staging import stage_source_archive

    return stage_source_archive(
        Path(source_path),
        output_path,
        expected_sha256=expected_archive_sha256,
        receipt_path=Path(receipt_path),
        scratch_dir=Path(scratch_dir),
    )


def observe_runtime(
    *,
    stage: str,
    namespace: str,
    expected_image: str,
    managed_job_name: str,
    managed_job_id: str,
    context: str,
    receipt_path: Path | str,
    pod_name: str = "",
    container_name: str = "ray-node",
    max_wait_seconds: float = 0,
    poll_seconds: float = 5,
    kubectl_bin: str = "kubectl",
) -> dict[str, Any]:
    """Observe one NRE stage image/GPU identity through Kubernetes."""
    from npa.workbench.nurec.runtime_attestation import observe_kubernetes_stage

    return observe_kubernetes_stage(
        stage=stage,
        pod_name=pod_name,
        namespace=namespace,
        container_name=container_name,
        expected_image=expected_image,
        managed_job_name=managed_job_name,
        managed_job_id=managed_job_id,
        output_path=Path(receipt_path),
        context=context,
        max_wait_seconds=max_wait_seconds,
        poll_seconds=poll_seconds,
        kubectl_bin=kubectl_bin,
    )


def bundle_runtime(
    reconstruct_receipt: Path | str,
    render_receipt: Path | str,
    receipt_path: Path | str,
) -> dict[str, Any]:
    """Bind reconstruct and render control-plane observations."""
    from npa.workbench.nurec.runtime_attestation import bundle_runtime_attestations

    return bundle_runtime_attestations(
        reconstruct_path=Path(reconstruct_receipt),
        render_path=Path(render_receipt),
        output_path=Path(receipt_path),
    )


def cleanup_qualification(
    *,
    run_id: str,
    job_id: str,
    context: str,
    namespace: str,
    storage_prefix: str,
    local_image: str,
    builder: str,
    build_receipt: Path | str,
    source_sha: str,
    receipt_path: Path | str,
    isolated_config_dir: Path | str | None = None,
    config_path: Path | str | None = None,
    sky_bin: str | None = None,
    docker_bin: str = "docker",
    kubectl_bin: str = "kubectl",
) -> dict[str, Any]:
    """Cancel/down the exact run and prove owned local runtime absence."""
    from npa.workbench.nurec.qualification_cleanup import (
        cleanup_qualification as run,
    )

    return run(
        run_id=run_id,
        job_id=job_id,
        context=context,
        namespace=namespace,
        storage_prefix=storage_prefix,
        local_image=local_image,
        builder=builder,
        build_receipt_path=Path(build_receipt),
        source_sha=source_sha,
        output_path=Path(receipt_path),
        isolated_config_dir=(
            Path(isolated_config_dir) if isolated_config_dir is not None else None
        ),
        config_path=Path(config_path) if config_path is not None else None,
        sky_bin=sky_bin,
        docker_bin=docker_bin,
        kubectl_bin=kubectl_bin,
    )


def audit_qualification(
    root: Path | str,
    *,
    recording_id: str,
    expected_image: str,
    expected_source_sha256: str,
    receipt_path: Path | str,
) -> dict[str, Any]:
    """Reopen and byte-bind the complete downloaded NCore/NRE qualification."""
    from npa.workbench.nurec.qualification_audit import audit_qualification as run

    return run(
        Path(root),
        recording_id=recording_id,
        expected_image=expected_image,
        expected_source_sha256=expected_source_sha256,
        output_path=Path(receipt_path),
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
    "audit_colmap",
    "audit_qualification",
    "bundle_runtime",
    "check",
    "cleanup_qualification",
    "control_source",
    "convert_colmap",
    "fetch",
    "finalize",
    "observe_runtime",
    "probe_storage",
    "reconstruct",
    "render",
    "stage_source",
    "status",
    "visualize",
]
