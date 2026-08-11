"""Artifact-storage credential selection for NPA agent lifecycle commands."""

from __future__ import annotations

import secrets
from typing import NoReturn

import typer

from npa.clients.config import (
    ConfigError,
    resolve_project_storage,
    resolve_terraform_state,
)
from npa.clients.credentials import load_credentials


def _fail(message: str) -> NoReturn:
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(code=1)


def _storage_credentials_allow_writes(
    *,
    bucket: str,
    endpoint: str,
    access_key: str,
    secret_key: str,
    region: str,
    prefix: str = "",
) -> bool:
    """Return True when credentials can list, write, and delete in the bucket."""
    bucket_name = str(bucket or "").strip()
    if not bucket_name:
        return False
    endpoint_url = str(endpoint or "").strip()
    if not endpoint_url:
        endpoint_url = (
            f"https://storage.{str(region or '').strip() or 'eu-north1'}.nebius.cloud"
        )
    try:
        import boto3
    except Exception:
        return False
    client_options = {
        "endpoint_url": endpoint_url,
        "aws_access_key_id": str(access_key or "").strip(),
        "aws_" + "secret_access_key": str(secret_key or "").strip(),
        "region_name": str(region or "").strip() or None,
    }
    client = boto3.client("s3", **client_options)
    normalized_prefix = str(prefix or "").strip().strip("/")
    probe_base = "/".join(
        part for part in (normalized_prefix, "npa-agent/probe") if part
    )
    probe_key = f"{probe_base}/{secrets.token_hex(8)}.txt"
    try:
        client.list_objects_v2(
            Bucket=bucket_name,
            Prefix=(probe_base + "/") if probe_base else "",
            MaxKeys=1,
        )
        client.put_object(Bucket=bucket_name, Key=probe_key, Body=b"ok")
        client.delete_object(Bucket=bucket_name, Key=probe_key)
        return True
    except Exception:
        return False


def _split_bucket_uri(raw_bucket: str) -> tuple[str, str]:
    value = str(raw_bucket or "").strip()
    if not value.startswith("s3://"):
        return value, ""
    bucket, _sep, prefix = value[len("s3://") :].partition("/")
    return bucket, prefix.strip("/")


def _resolve_deploy_storage_credentials(
    *,
    region: str,
    bootstrap_creds: dict[str, str],
    project_alias: str = "",
) -> dict[str, str]:
    """Prefer project artifact storage, then shared or bootstrap credentials."""
    candidate = dict(bootstrap_creds)
    project_name = str(project_alias or "").strip()
    if project_name:
        try:
            project_storage = resolve_project_storage(
                project_name, include_shared_credentials=False
            )
        except ConfigError:
            project_storage = None
        if project_storage is not None:
            project_bucket, project_prefix = _split_bucket_uri(
                project_storage.checkpoint_bucket
            )
            project_endpoint = str(
                project_storage.endpoint_url or f"https://storage.{region}.nebius.cloud"
            ).strip()
            project_access_key = str(project_storage.aws_access_key_id or "").strip()
            project_secret_key = str(
                project_storage.aws_secret_access_key or ""
            ).strip()
            if project_bucket and _storage_credentials_allow_writes(
                bucket=project_bucket,
                endpoint=project_endpoint,
                access_key=project_access_key,
                secret_key=project_secret_key,
                region=region,
                prefix=project_prefix,
            ):
                typer.echo(
                    "  Using project-configured artifact storage credentials for the agent."
                )
                candidate.update(
                    s3_bucket=project_bucket,
                    s3_prefix=project_prefix,
                    s3_endpoint=project_endpoint,
                    nebius_api_key=project_access_key,
                    nebius_secret_key=project_secret_key,
                )
                return candidate

    shared = load_credentials(environ={})
    shared_bucket, shared_prefix = _split_bucket_uri(shared.s3_bucket)
    shared_endpoint = str(
        shared.s3_endpoint or f"https://storage.{region}.nebius.cloud"
    ).strip()
    shared_access_key = str(shared.s3_access_key_id or "").strip()
    shared_secret_key = str(shared.s3_secret_access_key or "").strip()
    if shared_bucket and _storage_credentials_allow_writes(
        bucket=shared_bucket,
        endpoint=shared_endpoint,
        access_key=shared_access_key,
        secret_key=shared_secret_key,
        region=region,
        prefix=shared_prefix,
    ):
        typer.echo(
            "  Using shared configured artifact storage credentials for the agent."
        )
        candidate.update(
            s3_bucket=shared_bucket,
            s3_prefix=shared_prefix,
            s3_endpoint=shared_endpoint,
            nebius_api_key=shared_access_key,
            nebius_secret_key=shared_secret_key,
        )
        return candidate

    if _storage_credentials_allow_writes(
        bucket=str(candidate.get("s3_bucket", "")).strip(),
        endpoint=str(candidate.get("s3_endpoint", "")).strip(),
        access_key=str(candidate.get("nebius_api_key", "")).strip(),
        secret_key=str(candidate.get("nebius_secret_key", "")).strip(),
        region=region,
        prefix=str(candidate.get("s3_prefix", "")),
    ):
        return candidate
    if project_name:
        try:
            saved_state = resolve_terraform_state(project_name)
        except ConfigError:
            saved_state = None
        if saved_state is not None:
            saved_bucket = str(getattr(saved_state, "bucket", "") or "").strip()
            saved_endpoint = str(getattr(saved_state, "endpoint", "") or "").strip()
            saved_access_key = str(getattr(saved_state, "access_key", "") or "").strip()
            saved_secret_key = str(getattr(saved_state, "secret_key", "") or "").strip()
            if _storage_credentials_allow_writes(
                bucket=saved_bucket,
                endpoint=saved_endpoint,
                access_key=saved_access_key,
                secret_key=saved_secret_key,
                region=region,
            ):
                typer.echo(
                    "  Bootstrap S3 key has no data-plane access; "
                    "falling back to saved project terraform_state credentials."
                )
                candidate.update(
                    s3_bucket=saved_bucket,
                    s3_endpoint=saved_endpoint,
                    nebius_api_key=saved_access_key,
                    nebius_secret_key=saved_secret_key,
                )
                return candidate
    _fail(
        "unable to verify writable S3 credentials for deploy; "
        "configure object-storage credentials with data-plane access before deploying the agent"
    )


def _resolve_artifact_project_storage(
    project_alias: str, *, region: str
) -> tuple[str, str, str, str, str]:
    """Resolve one explicit project's writable artifact namespace."""
    alias = str(project_alias or "").strip()
    if not alias:
        raise ConfigError("artifact project alias is required")
    storage = resolve_project_storage(alias, include_shared_credentials=False)
    bucket, prefix = _split_bucket_uri(storage.checkpoint_bucket)
    endpoint = str(
        storage.endpoint_url
        or f"https://storage.{str(region or '').strip() or 'eu-north1'}.nebius.cloud"
    ).strip()
    access_key = str(storage.aws_access_key_id or "").strip()
    secret_key = str(storage.aws_secret_access_key or "").strip()
    if not (bucket and access_key and secret_key):
        raise ConfigError(f"artifact storage is incomplete for project {alias!r}")
    if not _storage_credentials_allow_writes(
        bucket=bucket,
        endpoint=endpoint,
        access_key=access_key,
        secret_key=secret_key,
        region=region,
        prefix=prefix,
    ):
        raise ConfigError(f"artifact storage is not writable for project {alias!r}")
    return bucket, prefix, endpoint, access_key, secret_key


def _validate_artifact_prefix(prefix: str) -> str:
    """Normalize a configured namespace root without accepting traversal."""
    value = str(prefix or "").strip().strip("/")
    if not value:
        return ""
    if "\\" in value or any(ord(ch) < 32 for ch in value):
        raise ValueError("artifact prefix contains unsupported characters")
    if any(segment in {"", ".", ".."} for segment in value.split("/")):
        raise ValueError("artifact prefix contains traversal segments")
    return value
