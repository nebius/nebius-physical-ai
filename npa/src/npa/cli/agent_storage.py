"""Storage credential resolution used by the NPA Agent lifecycle commands."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def storage_credentials_allow_writes(
    *,
    bucket: str,
    endpoint: str,
    access_key: str,
    secret_key: str,
    region: str,
    prefix: str = "",
) -> bool:
    """Return whether a credential can list, write, and delete under a prefix.

    Args:
        bucket: Object-storage bucket name.
        endpoint: S3-compatible endpoint, if explicitly configured.
        access_key: S3 access key ID.
        secret_key: S3 secret access key.
        region: Nebius storage region.
        prefix: Optional bounded object prefix for the probe.

    Returns:
        Whether the data-plane probe succeeded.
    """

    bucket_name = str(bucket or "").strip()
    if not bucket_name:
        return False
    endpoint_url = str(endpoint or "").strip()
    if not endpoint_url:
        endpoint_url = (
            f"https://storage.{str(region or '').strip() or 'eu-north1'}.nebius.cloud"
        )
    from npa.clients.storage_validation import probe_storage_write

    normalized_prefix = str(prefix or "").strip().strip("/")
    probe_prefix = "/".join(
        part for part in (normalized_prefix, "npa-agent/preflight") if part
    )
    probe = probe_storage_write(
        bucket=bucket_name,
        endpoint_url=endpoint_url,
        access_key_id=str(access_key or "").strip(),
        secret_access_key=str(secret_key or "").strip(),
        region=str(region or "").strip(),
        prefix=probe_prefix,
    )
    return bool(probe.ok)


def resolve_deploy_storage_credentials(
    *,
    region: str,
    bootstrap_creds: dict[str, str] | None = None,
    project_alias: str = "",
    emit_status: bool = True,
    resolve_project_storage: Callable[..., Any],
    resolve_terraform_state: Callable[[str], Any],
    config_error: type[Exception],
    can_write: Callable[..., bool],
    emit: Callable[[str], None],
    credential_error: type[Exception],
) -> dict[str, str]:
    """Resolve the exact writable storage credentials Agent deploy will use.

    Project-scoped credentials take precedence over shared or bootstrap
    credentials. Runtime dependencies are supplied by ``agent.py`` so the CLI's
    compatibility seams and focused tests retain their existing patch points.

    Args:
        region: Nebius region for default endpoint derivation.
        bootstrap_creds: Fresh credentials from bootstrap, if available.
        project_alias: Explicit project alias, if selected.
        emit_status: Whether to emit the selected credential source.
        resolve_project_storage: Project-storage resolver callback.
        resolve_terraform_state: Durable project-state resolver callback.
        config_error: Configuration exception type raised by the resolvers.
        can_write: Data-plane credential probe callback.
        emit: Status-output callback.
        credential_error: Error type raised if no writable credentials exist.

    Returns:
        The deploy credential mapping with a verified writable storage target.

    Raises:
        credential_error: If no candidate has verified data-plane access.
    """

    candidate = dict(bootstrap_creds or {})
    from npa.clients.credentials import load_credentials

    project_name = str(project_alias or "").strip()
    if project_name:
        try:
            project_storage = resolve_project_storage(
                project_name,
                include_shared_credentials=False,
            )
        except config_error:
            project_storage = None
        if project_storage is not None:
            project_bucket = str(project_storage.checkpoint_bucket or "").strip()
            project_prefix = ""
            if project_bucket.startswith("s3://"):
                rest = project_bucket[len("s3://") :]
                project_bucket, _separator, project_prefix = rest.partition("/")
                project_prefix = project_prefix.strip("/")
            project_endpoint = str(
                project_storage.endpoint_url or f"https://storage.{region}.nebius.cloud"
            ).strip()
            project_access_key = str(project_storage.aws_access_key_id or "").strip()
            project_secret_key = str(
                project_storage.aws_secret_access_key or ""
            ).strip()
            if project_bucket and can_write(
                bucket=project_bucket,
                endpoint=project_endpoint,
                access_key=project_access_key,
                secret_key=project_secret_key,
                region=region,
                prefix=project_prefix,
            ):
                if emit_status:
                    emit(
                        "  Using health-verified project artifact storage credentials."
                    )
                candidate.update(
                    {
                        "s3_bucket": project_bucket,
                        "s3_prefix": project_prefix,
                        "s3_endpoint": project_endpoint,
                        "nebius_api_key": project_access_key,
                        "nebius_secret_key": project_secret_key,
                    }
                )
                return candidate
    if not project_name:
        try:
            configured = resolve_project_storage(None)
        except config_error:
            configured = None
        configured_bucket = str(
            getattr(configured, "checkpoint_bucket", "") or ""
        ).strip()
        configured_prefix = ""
        if configured_bucket.startswith("s3://"):
            rest = configured_bucket[len("s3://") :]
            configured_bucket, _separator, configured_prefix = rest.partition("/")
            configured_prefix = configured_prefix.strip("/")
        configured_endpoint = str(
            getattr(configured, "endpoint_url", "")
            or f"https://storage.{region}.nebius.cloud"
        ).strip()
        configured_access_key = str(
            getattr(configured, "aws_access_key_id", "") or ""
        ).strip()
        configured_secret_key = str(
            getattr(configured, "aws_secret_access_key", "") or ""
        ).strip()
        if configured_bucket and can_write(
            bucket=configured_bucket,
            endpoint=configured_endpoint,
            access_key=configured_access_key,
            secret_key=configured_secret_key,
            region=region,
            prefix=configured_prefix,
        ):
            if emit_status:
                emit("  Using health-verified configured artifact storage credentials.")
            candidate.update(
                {
                    "s3_bucket": configured_bucket,
                    "s3_prefix": configured_prefix,
                    "s3_endpoint": configured_endpoint,
                    "nebius_api_key": configured_access_key,
                    "nebius_secret_key": configured_secret_key,
                }
            )
            return candidate
    if not project_name:
        shared = load_credentials(environ={})
        shared_bucket = str(shared.s3_bucket or "").strip()
        shared_prefix = ""
        if shared_bucket.startswith("s3://"):
            rest = shared_bucket[len("s3://") :]
            shared_bucket, _separator, shared_prefix = rest.partition("/")
            shared_prefix = shared_prefix.strip("/")
        shared_endpoint = str(
            shared.s3_endpoint or f"https://storage.{region}.nebius.cloud"
        ).strip()
        shared_access_key = str(shared.s3_access_key_id or "").strip()
        shared_secret_key = str(shared.s3_secret_access_key or "").strip()
        if shared_bucket and can_write(
            bucket=shared_bucket,
            endpoint=shared_endpoint,
            access_key=shared_access_key,
            secret_key=shared_secret_key,
            region=region,
            prefix=shared_prefix,
        ):
            if emit_status:
                emit("  Using health-verified shared artifact storage credentials.")
            candidate.update(
                {
                    "s3_bucket": shared_bucket,
                    "s3_prefix": shared_prefix,
                    "s3_endpoint": shared_endpoint,
                    "nebius_api_key": shared_access_key,
                    "nebius_secret_key": shared_secret_key,
                }
            )
            return candidate

    bucket = str(candidate.get("s3_bucket", "")).strip()
    endpoint = str(candidate.get("s3_endpoint", "")).strip()
    access_key = str(candidate.get("nebius_api_key", "")).strip()
    secret_key = str(candidate.get("nebius_secret_key", "")).strip()
    if can_write(
        bucket=bucket,
        endpoint=endpoint,
        access_key=access_key,
        secret_key=secret_key,
        region=region,
        prefix=str(candidate.get("s3_prefix", "")),
    ):
        return candidate
    if project_name:
        try:
            saved_state = resolve_terraform_state(project_name)
        except config_error:
            saved_state = None
        if saved_state is not None:
            saved_bucket = str(getattr(saved_state, "bucket", "") or "").strip()
            saved_endpoint = str(getattr(saved_state, "endpoint", "") or "").strip()
            saved_access_key = str(getattr(saved_state, "access_key", "") or "").strip()
            saved_secret_key = str(getattr(saved_state, "secret_key", "") or "").strip()
            if can_write(
                bucket=saved_bucket,
                endpoint=saved_endpoint,
                access_key=saved_access_key,
                secret_key=saved_secret_key,
                region=region,
            ):
                if emit_status:
                    emit(
                        "  Bootstrap S3 key has no data-plane access; falling back "
                        "to saved project terraform_state credentials."
                    )
                candidate.update(
                    {
                        "s3_bucket": saved_bucket,
                        "s3_endpoint": saved_endpoint,
                        "nebius_api_key": saved_access_key,
                        "nebius_secret_key": saved_secret_key,
                    }
                )
                return candidate
    raise credential_error(
        "unable to verify writable S3 credentials for deploy; "
        "configure object-storage credentials with data-plane access before deploying the agent"
    )


__all__ = ["resolve_deploy_storage_credentials", "storage_credentials_allow_writes"]
