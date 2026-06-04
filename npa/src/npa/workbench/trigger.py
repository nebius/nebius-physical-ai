"""S3-compatible retrigger helpers for Workbench sim-to-real loops."""

from __future__ import annotations

from typing import Any

from npa.workflows.sim_to_real_trigger import (
    PipelineLauncher,
    TriggerResult,
    WatermarkStore,
    build_config_from_env,
    run_once as _run_once,
    watch as _watch,
)


def run_once(
    *,
    s3_endpoint: str = "",
    s3_bucket: str = "",
    s3_prefix: str = "",
    watermark_uri: str = "",
    pipeline_yaml: str = "",
    pipeline_bucket: str = "",
    pipeline_s3_prefix: str = "",
    pipeline_input_data_uri: str = "",
    pipeline_render_only: bool = False,
    task_cloud: str = "kubernetes",
    controller_backend: str = "kubernetes",
    sky_bin: str = "",
    gpu: str = "",
    gpu_failover: str = "",
    submit_timeout: int | None = None,
    s3_client: Any | None = None,
    watermark_store: WatermarkStore | None = None,
    launcher: PipelineLauncher | None = None,
) -> TriggerResult:
    """Poll once and launch one sim-to-real pipeline run when new LeRobot data lands."""

    overrides: dict[str, Any] = {
        "s3_endpoint": s3_endpoint,
        "s3_bucket": s3_bucket,
        "s3_prefix": s3_prefix,
        "watermark_uri": watermark_uri,
        "pipeline_yaml": pipeline_yaml,
        "pipeline_bucket": pipeline_bucket,
        "pipeline_s3_prefix": pipeline_s3_prefix,
        "pipeline_input_data_uri": pipeline_input_data_uri,
        "pipeline_render_only": pipeline_render_only,
        "task_cloud": task_cloud,
        "controller_backend": controller_backend,
        "sky_bin": sky_bin,
        "gpu": gpu,
        "gpu_failover": gpu_failover,
    }
    if submit_timeout is not None:
        overrides["submit_timeout"] = submit_timeout
    config = build_config_from_env(**overrides)
    return _run_once(config, s3_client=s3_client, watermark_store=watermark_store, launcher=launcher)


def watch(
    *,
    poll_interval: int = 60,
    max_polls: int = 0,
    max_launches: int = 0,
    s3_endpoint: str = "",
    s3_bucket: str = "",
    s3_prefix: str = "",
    watermark_uri: str = "",
    pipeline_yaml: str = "",
    pipeline_bucket: str = "",
    pipeline_s3_prefix: str = "",
    pipeline_input_data_uri: str = "",
    pipeline_render_only: bool = False,
    task_cloud: str = "kubernetes",
    controller_backend: str = "kubernetes",
    sky_bin: str = "",
    gpu: str = "",
    gpu_failover: str = "",
    submit_timeout: int | None = None,
    s3_client: Any | None = None,
    watermark_store: WatermarkStore | None = None,
    launcher: PipelineLauncher | None = None,
) -> list[TriggerResult]:
    """Watch for LeRobot data and retrigger the sim-to-real pipeline."""

    overrides: dict[str, Any] = {
        "s3_endpoint": s3_endpoint,
        "s3_bucket": s3_bucket,
        "s3_prefix": s3_prefix,
        "watermark_uri": watermark_uri,
        "pipeline_yaml": pipeline_yaml,
        "pipeline_bucket": pipeline_bucket,
        "pipeline_s3_prefix": pipeline_s3_prefix,
        "pipeline_input_data_uri": pipeline_input_data_uri,
        "pipeline_render_only": pipeline_render_only,
        "task_cloud": task_cloud,
        "controller_backend": controller_backend,
        "sky_bin": sky_bin,
        "gpu": gpu,
        "gpu_failover": gpu_failover,
    }
    if submit_timeout is not None:
        overrides["submit_timeout"] = submit_timeout
    config = build_config_from_env(**overrides)
    return _watch(
        config,
        poll_interval=poll_interval,
        max_polls=max_polls,
        max_launches=max_launches,
        s3_client=s3_client,
        watermark_store=watermark_store,
        launcher=launcher,
    )


__all__ = ["run_once", "watch"]
