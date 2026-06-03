"""SDK helpers for the tiered sim-to-real workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from npa.workflows.sim_to_real import (
    SimToRealConfig,
    SimToRealReport,
    artifact_uris,
    build_config_from_env,
    build_policy_container_contract,
    run_structural_spine,
)


def local_smoke(
    *,
    run_id: str = "sim-to-real-sdk",
    output_dir: str | Path | None = None,
    attempt_s3_roundtrip: bool = False,
    **overrides: Any,
) -> SimToRealReport:
    """Run the local structural sim-to-real spine and return a tiered report."""

    config = build_config_from_env(run_id=run_id, output_dir=output_dir, **overrides)
    return run_structural_spine(config, attempt_s3_roundtrip=attempt_s3_roundtrip)


def output_paths(**overrides: Any) -> dict[str, str]:
    """Return the run-scoped S3 artifact layout for a sim-to-real config."""

    return artifact_uris(build_config_from_env(**overrides))


def policy_container_contract(**overrides: Any) -> dict[str, Any]:
    """Return the BYO LeRobot policy image I/O contract."""

    return build_policy_container_contract(build_config_from_env(**overrides))


__all__ = [
    "SimToRealConfig",
    "SimToRealReport",
    "local_smoke",
    "output_paths",
    "policy_container_contract",
]
