"""npa.rerun - hosted Rerun recording sharing."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from npa.clients.config import resolve_environment, resolve_project_storage
from npa.clients.nebius import (
    BucketCorsPlan,
    apply_bucket_rerun_cors,
    plan_bucket_rerun_cors,
)
from npa.clients.scoped_credentials import bucket_from_s3_uri
from npa.cli.rerun import (
    MAX_TTL_HOURS,
    RerunHostResult,
    RerunShareListItem,
    host_recording,
    list_share_items,
    revoke_share,
    share_recording,
)


def configure_browser_cors(
    *,
    target_bucket: str = "",
    target_project: str | None = None,
    target_project_id: str = "",
    apply: bool = False,
) -> BucketCorsPlan:
    """Plan or apply the bucket-admin CORS rule required by app.rerun.io.

    This uses the Nebius control-plane identity selected by the active CLI
    profile. It deliberately never uses the scoped S3 object credentials that
    :func:`host` and :func:`share` use for recording data.
    """

    project_id = target_project_id.strip()
    if project_id and not target_bucket:
        raise ValueError("target_bucket is required with target_project_id")
    if not project_id:
        environment = resolve_environment(target_project)
        project_id = str(getattr(environment, "project_id", "") or "").strip()
    if not project_id:
        alias = f" for target_project {target_project!r}" if target_project else ""
        raise ValueError(
            f"Target project is not configured{alias}. Pass target_project_id, "
            "or run `npa configure` to configure the project."
        )
    configured = target_bucket
    if not configured:
        storage = resolve_project_storage(target_project)
        configured = storage.checkpoint_bucket
    bucket = (
        bucket_from_s3_uri(configured)
        if configured.startswith("s3://")
        else configured.split("/", 1)[0]
    )
    if not bucket:
        raise ValueError(
            "Target bucket is not configured. Pass target_bucket or configure project storage."
        )
    operation = apply_bucket_rerun_cors if apply else plan_bucket_rerun_cors
    return operation(project_id, bucket)


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


__all__ = ["configure_browser_cors", "host", "share", "list_shares", "revoke"]
