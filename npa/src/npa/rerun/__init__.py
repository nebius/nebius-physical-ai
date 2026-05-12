"""npa.rerun - hosted Rerun recording sharing."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from npa.cli.rerun import (
    MAX_TTL_HOURS,
    RerunHostResult,
    RerunShareListItem,
    host_recording,
    list_share_items,
    revoke_share,
    share_recording,
)


def host(
    rrd_path: str | Path,
    *,
    target_bucket: str = "",
    ttl_hours: int = 1,
    allow_host_creds: bool = False,
    source_project: str | None = None,
    target_project: str | None = None,
    s3_client=None,
    host_s3_client=None,
    now: datetime | None = None,
) -> RerunHostResult:
    """Upload or reference a Rerun recording and return a hosted viewer URL."""
    return host_recording(
        str(rrd_path),
        target_bucket=target_bucket,
        ttl_hours=ttl_hours,
        allow_host_creds=allow_host_creds,
        source_project=source_project,
        target_project=target_project,
        s3_client=s3_client,
        host_s3_client=host_s3_client,
        now=now,
    )


def share(
    rrd_path: str | Path,
    *,
    target_bucket: str = "",
    ttl_hours: int = MAX_TTL_HOURS,
    label: str = "",
    workspace: str = "default",
    allow_host_creds: bool = False,
    source_project: str | None = None,
    target_project: str | None = None,
    s3_client=None,
    host_s3_client=None,
    now: datetime | None = None,
) -> RerunHostResult:
    """Create a durable S3-backed Rerun share URL."""
    return share_recording(
        str(rrd_path),
        target_bucket=target_bucket,
        ttl_hours=ttl_hours,
        label=label,
        workspace=workspace,
        allow_host_creds=allow_host_creds,
        source_project=source_project,
        target_project=target_project,
        s3_client=s3_client,
        host_s3_client=host_s3_client,
        now=now,
    )


def list_shares(
    *,
    target_bucket: str = "",
    s3_client=None,
    host_s3_client=None,
    allow_host_creds: bool = False,
    target_project: str | None = None,
    now: datetime | None = None,
) -> list[RerunShareListItem]:
    """List shared Rerun recordings stored in the operator bucket."""
    return list_share_items(
        target_bucket=target_bucket,
        s3_client=s3_client,
        host_s3_client=host_s3_client,
        allow_host_creds=allow_host_creds,
        target_project=target_project,
        now=now,
    )


def revoke(
    identifier: str,
    *,
    target_bucket: str = "",
    s3_client=None,
    host_s3_client=None,
    allow_host_creds: bool = False,
    target_project: str | None = None,
) -> int:
    """Delete matching shared Rerun recordings from S3."""
    return revoke_share(
        identifier,
        target_bucket=target_bucket,
        s3_client=s3_client,
        host_s3_client=host_s3_client,
        allow_host_creds=allow_host_creds,
        target_project=target_project,
    )


__all__ = ["host", "share", "list_shares", "revoke"]
