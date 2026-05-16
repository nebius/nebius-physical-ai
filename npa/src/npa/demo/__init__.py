"""npa.demo - demo artifact staging and verification."""

from __future__ import annotations

from pathlib import Path

from npa.cli.demo import DEFAULT_MANIFEST, stage_artifacts, verify_artifacts


def stage(
    *,
    target_bucket: str,
    manifest: str | Path = DEFAULT_MANIFEST,
    source_project: str | None = None,
    target_project: str | None = None,
    allow_host_creds: bool = False,
    s3_client=None,
    host_s3_client=None,
) -> list[dict[str, str]]:
    """Stage demo artifacts into an operator-owned bucket."""
    return stage_artifacts(
        target_bucket=target_bucket,
        manifest_path=Path(manifest),
        source_project=source_project,
        target_project=target_project,
        allow_host_creds=allow_host_creds,
        s3_client=s3_client,
        host_s3_client=host_s3_client,
    )


def verify(
    *,
    target_bucket: str,
    manifest: str | Path = DEFAULT_MANIFEST,
    target_project: str | None = None,
    allow_host_creds: bool = False,
    s3_client=None,
) -> list[str]:
    """Verify staged demo artifacts without downloading object contents."""
    return verify_artifacts(
        target_bucket=target_bucket,
        manifest_path=Path(manifest),
        target_project=target_project,
        allow_host_creds=allow_host_creds,
        s3_client=s3_client,
    )


__all__ = ["stage", "verify"]
