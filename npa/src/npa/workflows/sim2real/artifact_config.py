"""Resolve existing Sim2Real artifact coordinates without execution image defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from npa.workflows.sim2real.constants import (
    DEFAULT_K8S_GPU_PRODUCT,
    DEFAULT_OUTER_ITERATIONS,
    DEFAULT_PREFIX,
    DEFAULT_S3_ENDPOINT,
)
from npa.workflows.sim2real.models import new_run_id
from npa.workflows.sim2real.utils import _split_csv


@dataclass(frozen=True)
class Sim2RealArtifactConfig:
    """Storage coordinates and visualization metadata for an existing run.

    Args:
        run_id: Existing run identifier.
        s3_bucket: Bucket holding the run's artifacts.
        s3_prefix: Parent prefix containing the run.
        s3_endpoint: Storage endpoint used for reads and optional publication.
        outer_iterations: Final outer iteration to discover.
        k8s_gpu_product: Configured GPU product retained in viewer metadata.
        k8s_gpu_candidates: Configured placement alternatives retained in reports.

    Returns:
        None.
    Raises:
        None.
    """

    run_id: str
    s3_bucket: str
    s3_prefix: str = DEFAULT_PREFIX
    s3_endpoint: str = DEFAULT_S3_ENDPOINT
    outer_iterations: int = DEFAULT_OUTER_ITERATIONS
    k8s_gpu_product: str = DEFAULT_K8S_GPU_PRODUCT
    k8s_gpu_candidates: tuple[str, ...] = ()


def build_artifact_config_from_env(
    *,
    run_id: str,
    s3_bucket: str = "",
    s3_prefix: str | None = None,
    s3_endpoint: str = "",
    outer_iterations: int | None = None,
    k8s_gpu_product: str = "",
    k8s_gpu_candidates: Any = "",
) -> Sim2RealArtifactConfig:
    """Resolve artifact settings while leaving execution image policy untouched.

    Args:
        run_id: Existing run identifier.
        s3_bucket: Explicit bucket, or the existing environment fallback.
        s3_prefix: Explicit parent prefix, otherwise the existing environment fallback.
        s3_endpoint: Explicit endpoint, or the existing environment fallback.
        outer_iterations: Explicit final outer iteration, or the environment fallback.
        k8s_gpu_product: Explicit GPU product, or the environment fallback.
        k8s_gpu_candidates: Explicit GPU alternatives, or the environment fallback.
    Returns:
        Image-free settings for artifact download or visualization regeneration.
    Raises:
        ValueError: The configured outer iteration count is not an integer.
    """

    return Sim2RealArtifactConfig(
        run_id=run_id or os.environ.get("NPA_SIM2REAL_RUN_ID") or new_run_id(),
        s3_bucket=_artifact_bucket(s3_bucket),
        s3_prefix=_artifact_prefix(s3_prefix),
        s3_endpoint=_artifact_endpoint(s3_endpoint),
        outer_iterations=int(
            outer_iterations
            if outer_iterations is not None
            else os.environ.get("OUTER_ITERATIONS", DEFAULT_OUTER_ITERATIONS)
        ),
        k8s_gpu_product=k8s_gpu_product
        or os.environ.get("NPA_SIM2REAL_K8S_GPU_PRODUCT")
        or DEFAULT_K8S_GPU_PRODUCT,
        k8s_gpu_candidates=tuple(
            _split_csv(
                k8s_gpu_candidates
                or os.environ.get("NPA_SIM2REAL_K8S_GPU_CANDIDATES", "")
            )
        ),
    )


def _artifact_bucket(override: str) -> str:
    return (
        override
        or os.environ.get("NPA_SIM2REAL_BUCKET")
        or os.environ.get("NPA_S3_BUCKET")
        or os.environ.get("S3_BUCKET", "")
    )


def _artifact_prefix(override: str | None) -> str:
    if override is not None:
        return override
    return os.environ.get("NPA_SIM2REAL_PREFIX", DEFAULT_PREFIX)


def _artifact_endpoint(override: str) -> str:
    return (
        override
        or os.environ.get("AWS_ENDPOINT_URL")
        or os.environ.get("S3_ENDPOINT_URL")
        or DEFAULT_S3_ENDPOINT
    )
