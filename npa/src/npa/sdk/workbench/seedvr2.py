"""Python SDK for the shared SeedVR2 workbench implementation."""

from __future__ import annotations

from typing import Any

from npa.workbench.seedvr2 import artifacts, runtime
from npa.workbench.seedvr2.schemas import (
    ConditioningMode,
    ExpectedGPU,
    RestoreRequest,
    VideoArtifactRequest,
)


def probe(*, input_path: str, output_path: str, run_id: str) -> dict[str, Any]:
    """Decode an input and publish its media manifest."""

    return artifacts.probe(
        VideoArtifactRequest(
            input_path=input_path, output_path=output_path, run_id=run_id
        )
    )


def restore(
    *,
    input_path: str,
    output_path: str,
    run_id: str,
    probe_path: str = "",
    output_height: int = 480,
    output_width: int = 640,
    seed: int = 666,
    conditioning_mode: ConditioningMode = ConditioningMode.sample,
    expected_gpu: ExpectedGPU = ExpectedGPU.h100,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run official SeedVR2-3B through the shared implementation."""

    return runtime.restore(
        RestoreRequest(
            input_path=input_path,
            output_path=output_path,
            run_id=run_id,
            probe_path=probe_path,
            output_height=output_height,
            output_width=output_width,
            seed=seed,
            conditioning_mode=conditioning_mode,
            expected_gpu=expected_gpu,
            dry_run=dry_run,
        )
    )


def verify(*, input_path: str, output_path: str, run_id: str) -> dict[str, Any]:
    """Recompute the delivered video identity and decode contract."""

    return artifacts.verify(
        VideoArtifactRequest(
            input_path=input_path, output_path=output_path, run_id=run_id
        )
    )


def review(*, input_path: str, output_path: str, run_id: str) -> dict[str, Any]:
    """Build the non-blended matched review package."""

    return artifacts.review(
        VideoArtifactRequest(
            input_path=input_path, output_path=output_path, run_id=run_id
        )
    )


__all__ = [
    "RestoreRequest",
    "VideoArtifactRequest",
    "probe",
    "restore",
    "review",
    "verify",
]
