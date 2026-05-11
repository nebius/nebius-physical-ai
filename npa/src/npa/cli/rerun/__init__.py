"""Hosted Rerun recording sharing commands."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
import hashlib
import json
import logging
from pathlib import Path
from urllib.parse import quote, urlparse

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError, NoCredentialsError
import typer

from npa.clients.config import StorageConfig, resolve_project_storage
from npa.clients.scoped_credentials import (
    bucket_from_s3_uri,
    run_with_host_credential_fallback,
)

app = typer.Typer(
    name="rerun",
    help="Host and share Rerun .rrd recordings through app.rerun.io.",
    no_args_is_help=True,
)

logger = logging.getLogger(__name__)

# Keep this pinned to the rerun-sdk version used to produce current recordings.
RERUN_VERSION = "0.31.4"
MAX_TTL_HOURS = 168


class RerunHostError(ValueError):
    pass


class OutputFormat(str, Enum):
    text = "text"
    json = "json"


@dataclass(frozen=True)
class RerunHostResult:
    share_url: str
    rrd_s3_uri: str
    presigned_url: str
    ttl_expires_at: str
    sha256: str
    rerun_version: str


@dataclass(frozen=True)
class RerunShareListItem:
    label: str
    workspace: str
    age: str
    age_seconds: int
    sha256: str
    s3_uri: str


def host_recording(
    recording_path: str,
    *,
    target_bucket: str = "",
    ttl_hours: int = 1,
    allow_host_creds: bool = False,
    s3_client=None,
    host_s3_client=None,
    now: datetime | None = None,
) -> RerunHostResult:
    if ttl_hours <= 0:
        raise RerunHostError("--ttl-hours must be positive")
    if ttl_hours > MAX_TTL_HOURS:
        raise RerunHostError("--ttl-hours cannot exceed 168 hours (7 days)")

    storage = resolve_project_storage()
    endpoint = storage.endpoint_url
    target_bucket = target_bucket or _default_target_bucket(storage)
    scoped_s3 = s3_client or _s3_client(storage)
    fallback_s3 = host_s3_client or _host_s3_client(endpoint)

    if _is_s3_uri(recording_path):
        bucket, key, data, sha = _prepare_s3_recording(
            scoped_s3,
            fallback_s3,
            recording_path,
            allow_host_creds=allow_host_creds,
        )
    else:
        bucket, key, data, sha = _prepare_local_recording(recording_path, target_bucket)

    uri = f"s3://{bucket}/{key}"
    _ensure_uploaded(
        scoped_s3,
        fallback_s3,
        bucket,
        key,
        data,
        sha,
        allow_host_creds=allow_host_creds,
    )
    presigned = _presign(scoped_s3, bucket, key, ttl_hours)
    expires_at = ((now or datetime.now(UTC)) + timedelta(hours=ttl_hours)).replace(
        microsecond=0
    )
    share_url = _share_url(presigned)
    return RerunHostResult(
        share_url=share_url,
        rrd_s3_uri=uri,
        presigned_url=presigned,
        ttl_expires_at=expires_at.isoformat().replace("+00:00", "Z"),
        sha256=sha,
        rerun_version=RERUN_VERSION,
    )


def share_recording(
    recording_path: str,
    *,
    target_bucket: str = "",
    ttl_hours: int = MAX_TTL_HOURS,
    label: str = "",
    workspace: str = "default",
    allow_host_creds: bool = False,
    s3_client=None,
    host_s3_client=None,
    now: datetime | None = None,
) -> RerunHostResult:
    if ttl_hours <= 0:
        raise RerunHostError("--ttl-hours must be positive")
    if ttl_hours > MAX_TTL_HOURS:
        raise RerunHostError("--ttl-hours cannot exceed 168 hours (7 days)")

    storage = resolve_project_storage()
    endpoint = storage.endpoint_url
    target_bucket = target_bucket or _default_target_bucket(storage)
    scoped_s3 = s3_client or _s3_client(storage)
    fallback_s3 = host_s3_client or _host_s3_client(endpoint)

    if _is_s3_uri(recording_path):
        _source_bucket, _source_key, data, sha = _prepare_s3_recording(
            scoped_s3,
            fallback_s3,
            recording_path,
            allow_host_creds=allow_host_creds,
        )
    else:
        data, sha = _read_local_recording(recording_path)

    bucket, prefix = _target_bucket_prefix(target_bucket)
    workspace_name = _normalize_workspace(workspace)
    key = "/".join(
        part for part in (prefix, "rerun-shares", workspace_name, f"{sha}.rrd") if part
    )
    metadata = {"sha256": sha, "rerun-workspace": workspace_name}
    if label:
        metadata["rerun-label"] = label

    _ensure_uploaded(
        scoped_s3,
        fallback_s3,
        bucket,
        key,
        data,
        sha,
        allow_host_creds=allow_host_creds,
        metadata=metadata,
    )
    presigned = _presign(scoped_s3, bucket, key, ttl_hours)
    expires_at = ((now or datetime.now(UTC)) + timedelta(hours=ttl_hours)).replace(
        microsecond=0
    )
    return RerunHostResult(
        share_url=_share_url(presigned),
        rrd_s3_uri=f"s3://{bucket}/{key}",
        presigned_url=presigned,
        ttl_expires_at=expires_at.isoformat().replace("+00:00", "Z"),
        sha256=sha,
        rerun_version=RERUN_VERSION,
    )


def list_share_items(
    *,
    target_bucket: str = "",
    s3_client=None,
    host_s3_client=None,
    allow_host_creds: bool = False,
    now: datetime | None = None,
) -> list[RerunShareListItem]:
    storage = resolve_project_storage()
    endpoint = storage.endpoint_url
    target_bucket = target_bucket or _default_target_bucket(storage)
    scoped_s3 = s3_client or _s3_client(storage)
    fallback_s3 = host_s3_client or _host_s3_client(endpoint)
    bucket, configured_prefix = _target_bucket_prefix(target_bucket)
    list_prefix = "/".join(part for part in (configured_prefix, "rerun-shares") if part)
    if list_prefix:
        list_prefix = f"{list_prefix}/"
    current_time = now or datetime.now(UTC)
    items: list[RerunShareListItem] = []

    for obj in _list_objects(
        scoped_s3,
        fallback_s3,
        bucket,
        list_prefix,
        allow_host_creds=allow_host_creds,
    ):
        key = obj.get("Key", "")
        if not key.endswith(".rrd"):
            continue
        head = _head_object(
            scoped_s3,
            fallback_s3,
            bucket,
            key,
            allow_host_creds=allow_host_creds,
        )
        if head is None:
            continue
        metadata = head.get("Metadata", {}) or {}
        workspace = metadata.get(
            "rerun-workspace", _workspace_from_key(key, list_prefix)
        )
        sha = metadata.get("sha256") or Path(key).stem
        age_seconds = _age_seconds(current_time, obj.get("LastModified"))
        items.append(
            RerunShareListItem(
                label=metadata.get("rerun-label", ""),
                workspace=workspace,
                age=_format_age(age_seconds),
                age_seconds=age_seconds,
                sha256=sha,
                s3_uri=f"s3://{bucket}/{key}",
            )
        )

    return sorted(items, key=lambda item: (item.workspace, item.label, item.sha256))


def revoke_share(
    identifier: str,
    *,
    target_bucket: str = "",
    s3_client=None,
    host_s3_client=None,
    allow_host_creds: bool = False,
) -> int:
    storage = resolve_project_storage()
    endpoint = storage.endpoint_url
    target_bucket = target_bucket or _default_target_bucket(storage)
    scoped_s3 = s3_client or _s3_client(storage)
    fallback_s3 = host_s3_client or _host_s3_client(endpoint)
    bucket, _prefix = _target_bucket_prefix(target_bucket)
    matches = [
        item
        for item in list_share_items(
            target_bucket=target_bucket,
            s3_client=scoped_s3,
            host_s3_client=fallback_s3,
            allow_host_creds=allow_host_creds,
        )
        if item.sha256 == identifier or item.label == identifier
    ]
    for item in matches:
        _delete_object(
            scoped_s3,
            fallback_s3,
            bucket,
            _key_from_s3_uri(item.s3_uri),
            allow_host_creds=allow_host_creds,
        )
    return len(matches)


@app.command("host")
def host_cmd(
    path: str = typer.Argument(..., help="Local or s3:// path to a .rrd recording."),
    target_bucket: str = typer.Option(
        "",
        "--target-bucket",
        help="Target bucket or s3://bucket/prefix for local uploads. Defaults to configured project storage.",
    ),
    ttl_hours: int = typer.Option(
        1, "--ttl-hours", help="Presigned URL lifetime in hours, max 168."
    ),
    output: OutputFormat = typer.Option(
        OutputFormat.text, "--output", help="Output format."
    ),
    allow_host_creds: bool = typer.Option(
        False,
        "--allow-host-creds",
        help="Use --allow-host-creds to fall back to host credentials for S3 upload.",
    ),
) -> None:
    """Upload or reference a Rerun .rrd and print an app.rerun.io URL."""
    try:
        result = host_recording(
            path,
            target_bucket=target_bucket,
            ttl_hours=ttl_hours,
            allow_host_creds=allow_host_creds,
        )
    except Exception as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc

    if output == OutputFormat.json:
        typer.echo(json.dumps(asdict(result), indent=2))
        return
    typer.echo(result.share_url)


@app.command("share")
def share_cmd(
    path: str = typer.Argument(..., help="Local or s3:// path to a .rrd recording."),
    target_bucket: str = typer.Option(
        "",
        "--target-bucket",
        help="Target bucket or s3://bucket/prefix for shared recordings. Defaults to configured project storage.",
    ),
    ttl_hours: int = typer.Option(
        MAX_TTL_HOURS,
        "--ttl-hours",
        help="Presigned URL lifetime in hours, max 168.",
    ),
    label: str = typer.Option("", "--label", help="Human-readable share label."),
    workspace: str = typer.Option(
        "default", "--workspace", help="Workspace name under rerun-shares/."
    ),
    output: OutputFormat = typer.Option(
        OutputFormat.text, "--output", help="Output format."
    ),
    allow_host_creds: bool = typer.Option(
        False,
        "--allow-host-creds",
        help="Use --allow-host-creds to fall back to host credentials for S3 upload.",
    ),
) -> None:
    """Create a durable S3-backed Rerun share URL, capped at 7 days."""
    try:
        result = share_recording(
            path,
            target_bucket=target_bucket,
            ttl_hours=ttl_hours,
            label=label,
            workspace=workspace,
            allow_host_creds=allow_host_creds,
        )
    except Exception as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc

    if output == OutputFormat.json:
        typer.echo(json.dumps(asdict(result), indent=2))
        return
    typer.echo(result.share_url)


@app.command("list-shares")
def list_shares_cmd(
    target_bucket: str = typer.Option(
        "",
        "--target-bucket",
        help="Target bucket or s3://bucket/prefix to list. Defaults to configured project storage.",
    ),
    output: OutputFormat = typer.Option(
        OutputFormat.text, "--output", help="Output format."
    ),
    allow_host_creds: bool = typer.Option(
        False,
        "--allow-host-creds",
        help="Use --allow-host-creds to fall back to host credentials for S3 listing.",
    ),
) -> None:
    """List shared Rerun recordings stored in the operator bucket."""
    try:
        items = list_share_items(
            target_bucket=target_bucket,
            allow_host_creds=allow_host_creds,
        )
    except Exception as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc

    if output == OutputFormat.json:
        typer.echo(json.dumps([asdict(item) for item in items], indent=2))
        return
    if not items:
        typer.echo("No Rerun shares found.")
        return
    typer.echo(f"{'label':24} {'workspace':20} {'age':>8} sha256")
    for item in items:
        label = item.label or "-"
        typer.echo(
            f"{label[:24]:24} {item.workspace[:20]:20} {item.age:>8} {item.sha256}"
        )


@app.command("revoke")
def revoke_cmd(
    identifier: str = typer.Argument(..., help="Share sha256 or label to revoke."),
    target_bucket: str = typer.Option(
        "",
        "--target-bucket",
        help="Target bucket or s3://bucket/prefix to search. Defaults to configured project storage.",
    ),
    allow_host_creds: bool = typer.Option(
        False,
        "--allow-host-creds",
        help="Use --allow-host-creds to fall back to host credentials for S3 deletion.",
    ),
) -> None:
    """Delete matching shared Rerun recordings from S3."""
    try:
        deleted = revoke_share(
            identifier,
            target_bucket=target_bucket,
            allow_host_creds=allow_host_creds,
        )
    except Exception as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Deleted {deleted} matching Rerun share(s).")


def _read_local_recording(recording_path: str) -> tuple[bytes, str]:
    path = Path(recording_path)
    if not path.exists():
        raise RerunHostError(f"Rerun recording does not exist: {recording_path}")
    if not path.is_file():
        raise RerunHostError(f"Rerun recording must be a file: {recording_path}")
    if path.suffix.lower() != ".rrd":
        raise RerunHostError(f"Rerun recording must end in .rrd: {recording_path}")
    data = path.read_bytes()
    return data, hashlib.sha256(data).hexdigest()


def _prepare_local_recording(
    recording_path: str, target_bucket: str
) -> tuple[str, str, bytes, str]:
    data, sha = _read_local_recording(recording_path)
    bucket, prefix = _target_bucket_prefix(target_bucket)
    key = "/".join(part for part in (prefix, "rerun-shared", f"{sha}.rrd") if part)
    return bucket, key, data, sha


def _prepare_s3_recording(
    scoped_s3,
    host_s3,
    recording_path: str,
    *,
    allow_host_creds: bool,
) -> tuple[str, str, bytes, str]:
    bucket, key = _parse_s3_uri(recording_path)
    if not key.lower().endswith(".rrd"):
        raise RerunHostError(f"Rerun recording must end in .rrd: {recording_path}")
    data = _get_object(
        scoped_s3,
        host_s3,
        bucket,
        key,
        allow_host_creds=allow_host_creds,
    )
    sha = hashlib.sha256(data).hexdigest()
    return bucket, key, data, sha


def _ensure_uploaded(
    scoped_s3,
    host_s3,
    bucket: str,
    key: str,
    data: bytes,
    sha: str,
    *,
    allow_host_creds: bool,
    metadata: dict[str, str] | None = None,
) -> None:
    upload_metadata = metadata or {"sha256": sha}
    head = _head_object(
        scoped_s3,
        host_s3,
        bucket,
        key,
        allow_host_creds=allow_host_creds,
    )
    existing_metadata = head.get("Metadata", {}) if head else {}
    metadata_matches = all(
        existing_metadata.get(key) == value for key, value in upload_metadata.items()
    )
    if metadata_matches and int(head.get("ContentLength", -1)) == len(data):
        return

    def scoped_put() -> None:
        scoped_s3.put_object(
            Bucket=bucket, Key=key, Body=data, Metadata=upload_metadata
        )

    def host_put() -> None:
        host_s3.put_object(Bucket=bucket, Key=key, Body=data, Metadata=upload_metadata)

    run_with_host_credential_fallback(
        scoped_put,
        host_put,
        bucket=bucket,
        operation=f"Rerun recording upload to s3://{bucket}/{key}",
        allow_host_creds=allow_host_creds,
        logger=logger,
    )


def _head_object(scoped_s3, host_s3, bucket: str, key: str, *, allow_host_creds: bool):
    def scoped_head():
        try:
            return scoped_s3.head_object(Bucket=bucket, Key=key)
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            if code in {"404", "NoSuchKey", "NotFound"}:
                return None
            raise

    def host_head():
        try:
            return host_s3.head_object(Bucket=bucket, Key=key)
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            if code in {"404", "NoSuchKey", "NotFound"}:
                return None
            raise

    return run_with_host_credential_fallback(
        scoped_head,
        host_head,
        bucket=bucket,
        operation=f"Rerun recording head s3://{bucket}/{key}",
        allow_host_creds=allow_host_creds,
        logger=logger,
    )


def _get_object(
    scoped_s3, host_s3, bucket: str, key: str, *, allow_host_creds: bool
) -> bytes:
    def scoped_get() -> bytes:
        return scoped_s3.get_object(Bucket=bucket, Key=key)["Body"].read()

    def host_get() -> bytes:
        return host_s3.get_object(Bucket=bucket, Key=key)["Body"].read()

    return run_with_host_credential_fallback(
        scoped_get,
        host_get,
        bucket=bucket,
        operation=f"Rerun recording read s3://{bucket}/{key}",
        allow_host_creds=allow_host_creds,
        logger=logger,
    )


def _list_objects(
    scoped_s3,
    host_s3,
    bucket: str,
    prefix: str,
    *,
    allow_host_creds: bool,
) -> list[dict]:
    def scoped_list() -> list[dict]:
        return _list_objects_with_client(scoped_s3, bucket, prefix)

    def host_list() -> list[dict]:
        return _list_objects_with_client(host_s3, bucket, prefix)

    return run_with_host_credential_fallback(
        scoped_list,
        host_list,
        bucket=bucket,
        operation=f"Rerun share list s3://{bucket}/{prefix}",
        allow_host_creds=allow_host_creds,
        logger=logger,
    )


def _list_objects_with_client(s3, bucket: str, prefix: str) -> list[dict]:
    objects: list[dict] = []
    token: str | None = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        response = s3.list_objects_v2(**kwargs)
        objects.extend(response.get("Contents", []))
        if not response.get("IsTruncated"):
            return objects
        token = response.get("NextContinuationToken")
        if not token:
            return objects


def _delete_object(
    scoped_s3,
    host_s3,
    bucket: str,
    key: str,
    *,
    allow_host_creds: bool,
) -> None:
    def scoped_delete() -> None:
        _delete_with_client(scoped_s3, bucket, key)

    def host_delete() -> None:
        _delete_with_client(host_s3, bucket, key)

    run_with_host_credential_fallback(
        scoped_delete,
        host_delete,
        bucket=bucket,
        operation=f"Rerun share delete s3://{bucket}/{key}",
        allow_host_creds=allow_host_creds,
        logger=logger,
    )


def _delete_with_client(s3, bucket: str, key: str) -> None:
    try:
        s3.delete_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code", ""))
        if code in {"404", "NoSuchKey", "NotFound"}:
            return
        raise


def _presign(s3, bucket: str, key: str, ttl_hours: int) -> str:
    try:
        return s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=ttl_hours * 3600,
        )
    except (ClientError, NoCredentialsError) as exc:
        raise RerunHostError(f"Failed to presign s3://{bucket}/{key}: {exc}") from exc


def _share_url(presigned_url: str) -> str:
    return f"https://app.rerun.io/version/{RERUN_VERSION}/?url={quote(presigned_url, safe='')}"


def _s3_client(storage: StorageConfig):
    return boto3.client(
        "s3",
        endpoint_url=storage.endpoint_url,
        aws_access_key_id=storage.aws_access_key_id or None,
        aws_secret_access_key=storage.aws_secret_access_key or None,
        config=BotoConfig(signature_version="s3v4"),
    )


def _host_s3_client(endpoint_url: str):
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url or None,
        config=BotoConfig(signature_version="s3v4"),
    )


def _is_s3_uri(path: str) -> bool:
    return path.startswith("s3://")


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.lstrip("/"):
        raise RerunHostError(f"Expected s3://bucket/key URI, got: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def _key_from_s3_uri(uri: str) -> str:
    return _parse_s3_uri(uri)[1]


def _target_bucket_prefix(target_bucket: str) -> tuple[str, str]:
    if target_bucket.startswith("s3://"):
        parsed = urlparse(target_bucket)
        if not parsed.netloc:
            raise RerunHostError(f"Invalid --target-bucket: {target_bucket}")
        return parsed.netloc, parsed.path.lstrip("/").rstrip("/")
    if not target_bucket:
        raise RerunHostError("Target bucket is not configured. Pass --target-bucket.")
    return target_bucket, ""


def _default_target_bucket(storage: StorageConfig) -> str:
    configured = storage.checkpoint_bucket
    if configured:
        return (
            bucket_from_s3_uri(configured)
            if configured.startswith("s3://")
            else configured
        )
    raise RerunHostError("Target bucket is not configured. Pass --target-bucket.")


def _normalize_workspace(workspace: str) -> str:
    name = workspace.strip().strip("/")
    if not name or ".." in name:
        raise RerunHostError("--workspace must be a non-empty name without '..'")
    return name


def _workspace_from_key(key: str, list_prefix: str) -> str:
    suffix = key[len(list_prefix) :] if key.startswith(list_prefix) else key
    parts = suffix.split("/")
    return parts[0] if len(parts) > 1 and parts[0] else "default"


def _age_seconds(now: datetime, last_modified) -> int:
    if not isinstance(last_modified, datetime):
        return 0
    if last_modified.tzinfo is None:
        last_modified = last_modified.replace(tzinfo=UTC)
    return max(0, int((now - last_modified.astimezone(UTC)).total_seconds()))


def _format_age(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    if hours < 48:
        return f"{hours}h"
    return f"{hours // 24}d"
