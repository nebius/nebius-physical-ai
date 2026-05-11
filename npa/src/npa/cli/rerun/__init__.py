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


def _prepare_local_recording(
    recording_path: str, target_bucket: str
) -> tuple[str, str, bytes, str]:
    path = Path(recording_path)
    if not path.exists():
        raise RerunHostError(f"Rerun recording does not exist: {recording_path}")
    if not path.is_file():
        raise RerunHostError(f"Rerun recording must be a file: {recording_path}")
    if path.suffix.lower() != ".rrd":
        raise RerunHostError(f"Rerun recording must end in .rrd: {recording_path}")
    data = path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
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
) -> None:
    head = _head_object(
        scoped_s3,
        host_s3,
        bucket,
        key,
        allow_host_creds=allow_host_creds,
    )
    metadata = head.get("Metadata", {}) if head else {}
    if metadata.get("sha256") == sha and int(head.get("ContentLength", -1)) == len(
        data
    ):
        return

    def scoped_put() -> None:
        scoped_s3.put_object(
            Bucket=bucket, Key=key, Body=data, Metadata={"sha256": sha}
        )

    def host_put() -> None:
        host_s3.put_object(Bucket=bucket, Key=key, Body=data, Metadata={"sha256": sha})

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
