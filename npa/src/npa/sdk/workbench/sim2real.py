"""SDK helpers for the Sim2Real VLM-to-RL workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from npa.workflows.sim2real import (
    Sim2RealLoopConfig,
    artifact_uris,
    build_config_from_env,
    byo_seams,
    convert_vlm_eval_to_rl_signal,
    run_finalize,
    run_full_loop,
    run_inner_loop,
    run_preamble,
    run_single_outer_iteration,
    signal_mapping_rules,
)


def run(
    *,
    run_id: str = "sim2real-sdk",
    output_dir: str | Path | None = None,
    upload_artifacts: bool = False,
    **overrides: Any,
) -> dict[str, Any]:
    """Run the full Sim2Real Stage 1-13 workflow."""

    config = build_config_from_env(
        run_id=run_id,
        output_dir=output_dir,
        upload_artifacts=upload_artifacts,
        **overrides,
    )
    return run_full_loop(config)


def inner_loop(
    *,
    run_id: str = "sim2real-inner-sdk",
    output_dir: str | Path,
    initial_quality: float = 0.38,
    **overrides: Any,
) -> dict[str, Any]:
    """Run only Stage 7-9 VLM eval, signal conversion, and policy update."""

    config = build_config_from_env(run_id=run_id, output_dir=output_dir, **overrides)
    return run_inner_loop(config, local_dir=Path(output_dir), initial_quality=initial_quality)


def preamble(
    *,
    run_id: str = "sim2real-staged-sdk",
    output_dir: str | Path | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Run Stage 1-6 and persist workflow state."""

    config = build_config_from_env(run_id=run_id, output_dir=output_dir, **overrides)
    return run_preamble(config)


def outer_iteration(
    *,
    run_id: str = "sim2real-staged-sdk",
    output_dir: str | Path,
    outer_iteration: int,
    initial_quality: float,
    **overrides: Any,
) -> dict[str, Any]:
    """Run one Stage 7-11 iteration for staged execution."""

    local_dir = Path(output_dir)
    config = build_config_from_env(run_id=run_id, output_dir=local_dir, **overrides)
    return run_single_outer_iteration(
        config,
        local_dir=local_dir,
        outer_iteration=outer_iteration,
        initial_quality=initial_quality,
    )


def finalize(
    *,
    run_id: str = "sim2real-staged-sdk",
    output_dir: str | Path,
    stage_records: list[dict[str, Any]],
    components: list[dict[str, Any]],
    outer_history: list[dict[str, Any]],
    final_inner: dict[str, Any],
    final_eval: dict[str, Any],
    final_decision: dict[str, Any],
    upload_artifacts: bool = False,
    **overrides: Any,
) -> dict[str, Any]:
    """Run Stage 12-13/report/upload for staged execution."""

    local_dir = Path(output_dir)
    config = build_config_from_env(
        run_id=run_id,
        output_dir=local_dir,
        upload_artifacts=upload_artifacts,
        **overrides,
    )
    return run_finalize(
        config,
        local_dir=local_dir,
        stage_records=stage_records,
        components=components,
        outer_history=outer_history,
        final_inner=final_inner,
        final_eval=final_eval,
        final_decision=final_decision,
    )


def output_paths(**overrides: Any) -> dict[str, str]:
    """Return run-scoped S3 artifact URIs."""

    return artifact_uris(build_config_from_env(**overrides))


def status(
    *,
    run_id: str,
    watch: bool = False,
    interval: float = 10.0,
    **overrides: Any,
) -> dict[str, Any]:
    """Live stage progress for a staged cluster run."""

    from npa.workflows.sim2real.monitor import watch_sim2real_status

    return watch_sim2real_status(
        run_id,
        watch=watch,
        interval=interval,
        s3_bucket=str(overrides.get("s3_bucket") or ""),
        s3_prefix=str(overrides.get("s3_prefix") or "sim2real-b"),
        s3_endpoint=str(overrides.get("s3_endpoint") or ""),
        k8s_context=str(overrides.get("k8s_context") or ""),
        k8s_namespace=str(overrides.get("k8s_namespace") or "default"),
    )


def seams(**overrides: Any) -> dict[str, Any]:
    """Return all BYO plug points for the workflow."""

    return byo_seams(build_config_from_env(**overrides))


__all__ = [
    "Sim2RealLoopConfig",
    "artifact_uris",
    "build_config_from_env",
    "byo_seams",
    "convert_vlm_eval_to_rl_signal",
    "inner_loop",
    "output_paths",
    "outer_iteration",
    "preamble",
    "finalize",
    "run",
    "signal_mapping_rules",
    "seams",
    "status",
]
