"""SDK helpers for the Sim2Real VLM-to-RL workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from npa.workflows.sim2real_loop import (
    Sim2RealLoopConfig,
    artifact_uris,
    build_config_from_env,
    byo_seams,
    convert_vlm_eval_to_rl_signal,
    run_full_loop,
    run_inner_loop,
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


def output_paths(**overrides: Any) -> dict[str, str]:
    """Return run-scoped S3 artifact URIs."""

    return artifact_uris(build_config_from_env(**overrides))


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
    "run",
    "signal_mapping_rules",
    "seams",
]
