"""Durable exact artifact-source configuration for agent bootstrap."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import typer

from npa.clients.project_credential_store import project_credential_record
from npa.cli.agent_access import normalize_configured_artifact_sources
from npa.cli.agent_env_files import _load_agent_artifact_sources_file


class AgentStorageCredentialError(RuntimeError):
    """Configured/bootstrap storage cannot satisfy the data-plane contract."""


@dataclass(frozen=True)
class ArtifactStorageCredentialResolution:
    """One artifact-read identity and its observable compatibility mode."""

    credentials: tuple[str, str, str, str, str, str]
    mode: str

    @property
    def migration_warning(self) -> str:
        if self.mode != "deployment-write-migration":
            return ""
        return (
            "Warning: artifact discovery is using an explicit deployment-write "
            "credential migration scoped to the configured source."
        )

    def bootstrap_kwargs(self, region: str) -> dict[str, str]:
        """Render only the isolated artifact channel's remote bootstrap fields."""

        bucket, _prefix, endpoint, access_key, secret_key, _service_account = (
            self.credentials
        )
        return {
            "artifact_s3_bucket": bucket,
            "artifact_s3_endpoint": endpoint,
            "artifact_s3_access_key": access_key,
            "artifact_s3_secret_key": secret_key,
            "artifact_s3_region": region,
            "artifact_credential_mode": self.mode,
        }


def artifact_credential_summary(
    mode: str, *, sources_configured: bool
) -> dict[str, str]:
    """Return the non-secret status view shared by agent operator surfaces."""

    normalized = str(mode or "unconfigured").strip() or "unconfigured"
    if normalized == "isolated-read" and sources_configured:
        return {"mode": normalized, "status": "ready"}
    if normalized == "deployment-write-migration" and sources_configured:
        return {"mode": normalized, "status": "warning"}
    return {"mode": normalized, "status": "blocked"}


def artifact_credential_record_summary(record: dict[str, Any]) -> dict[str, str]:
    """Summarize persisted mode together with its exact-source registration."""

    return artifact_credential_summary(
        str(record.get("artifact_credential_mode") or "unconfigured"),
        sources_configured=bool(record.get("artifact_sources")),
    )


def emit_artifact_migration_warning(
    resolution: ArtifactStorageCredentialResolution,
) -> None:
    """Make an explicitly retained writer migration visible to the operator."""

    if resolution.migration_warning:
        typer.echo(resolution.migration_warning, err=True)


def artifact_source_file_option():
    """Reusable owner-only exact-source file option for Agent bootstrap."""

    return typer.Option(
        "",
        "--artifact-source-file",
        help=(
            "Owner-only JSON file containing exact read-only artifact source "
            "project/bucket/prefix tuples; persisted for future bootstraps."
        ),
    )


def artifact_write_identity_migration_option():
    """Reusable explicit same-project writer-migration option."""

    return typer.Option(
        False,
        "--allow-artifact-write-identity-migration",
        help=(
            "Explicitly retain the deployment-write S3 identity for one exact "
            "same-project artifact source during migration; status remains warned."
        ),
    )


def resolve_agent_service_account_id(project_alias: str, record: dict[str, Any]) -> str:
    """Resolve the attached service-account id from exact owner state."""
    stored = str(record.get("service_account_id", "")).strip()
    if stored:
        return stored
    creds = record.get("credentials", {})
    if isinstance(creds, dict):
        from_record = str(creds.get("service_account_id", "")).strip()
        if from_record:
            return from_record
    from npa.clients.nebius import resolve_service_account_id

    project_id = str(record.get("project_id", "")).strip()
    return str(resolve_service_account_id(project_id) or "") if project_id else ""


def resolve_agent_storage_credentials(
    project_alias: str,
    record: dict[str, Any],
    *,
    resolve_terraform_state: Callable[[str], Any],
    resolve_service_account_id: Callable[[str, dict[str, Any]], str],
    config_error: type[Exception],
) -> tuple[str, str, str, str, str, str]:
    """Resolve the deployment project's legacy or Terraform S3 credentials."""
    creds = record.get("credentials", {})
    if isinstance(creds, dict):
        access_key = str(creds.get("access_key", "")).strip()
        secret_key = str(creds.get("secret_key", "")).strip()
        bucket = str(creds.get("s3_bucket", "")).strip()
        prefix = str(creds.get("s3_prefix", "")).strip().strip("/")
        endpoint = str(creds.get("s3_endpoint", "")).strip()
        service_account_id = str(
            creds.get("service_account_id", record.get("service_account_id", ""))
        ).strip()
        if bucket and access_key and secret_key:
            if not service_account_id:
                service_account_id = resolve_service_account_id(project_alias, record)
            return (
                bucket,
                prefix,
                endpoint,
                access_key,
                secret_key,
                service_account_id,
            )
    try:
        tf_state = resolve_terraform_state(project_alias)
    except config_error:
        return ("", "", "", "", "", resolve_service_account_id(project_alias, record))
    return (
        str(getattr(tf_state, "bucket", "") or ""),
        "",
        str(getattr(tf_state, "endpoint", "") or ""),
        str(getattr(tf_state, "access_key", "") or ""),
        str(getattr(tf_state, "secret_key", "") or ""),
        resolve_service_account_id(project_alias, record),
    )


def resolve_configured_artifact_storage_identity(
    artifact_sources: tuple[dict[str, str], ...] | list[dict[str, str]],
    *,
    deployment_project_id: str,
    current: tuple[str, str, str, str, str, str],
    allow_deployment_write_migration: bool = False,
) -> ArtifactStorageCredentialResolution:
    """Select an exact source project's owner-stored artifact-read identity.

    A source tuple is not a grant. The normal path therefore requires a
    separately recorded ``storage.viewer`` identity for the exact project,
    bucket, and source prefixes. Reuse of the deployment writer exists only as
    an explicit same-project migration mode and is never an implicit fallback.
    """
    sources = normalize_configured_artifact_sources(artifact_sources)
    if not sources:
        if allow_deployment_write_migration:
            raise AgentStorageCredentialError(
                "artifact credential migration requires an exact configured source"
            )
        return ArtifactStorageCredentialResolution(
            credentials=("", "", "", "", "", ""), mode="unconfigured"
        )
    source_projects = {item["project_id"] for item in sources}
    if len(source_projects) != 1:
        raise AgentStorageCredentialError(
            "configured artifact sources must use one exact credential project"
        )
    source_project = next(iter(source_projects))
    source_buckets = {item["bucket"] for item in sources}
    if len(source_buckets) != 1:
        raise AgentStorageCredentialError(
            "configured artifact sources must use one exact credential bucket"
        )
    source_bucket = next(iter(source_buckets))
    source_prefixes = tuple(sorted({item["resolved_prefix"] for item in sources}))
    source_prefix = source_prefixes[0] if len(source_prefixes) == 1 else ""
    current_bucket, _prefix, endpoint, access_key, secret_key, service_account_id = (
        current
    )
    record = project_credential_record(source_project, migrate_legacy=False)
    raw_storage = (
        record.get("artifact_read_storage") if isinstance(record, dict) else None
    )
    if raw_storage is None:
        if source_project != str(deployment_project_id or "").strip():
            raise AgentStorageCredentialError(
                "owner credential store has no exact matching artifact read credentials"
            )
        if not allow_deployment_write_migration:
            raise AgentStorageCredentialError(
                "configured artifact sources require an independent read-only "
                "artifact identity; deployment-write fallback is disabled"
            )
        if current_bucket != source_bucket:
            raise AgentStorageCredentialError(
                "deployment-write migration bucket does not match the exact artifact source"
            )
        if not (endpoint and access_key and secret_key):
            raise AgentStorageCredentialError(
                "deployment project has no owner-stored S3 credentials for the "
                "configured artifact source"
            )
        return ArtifactStorageCredentialResolution(
            credentials=(
                source_bucket,
                source_prefix,
                endpoint,
                access_key,
                secret_key,
                service_account_id,
            ),
            mode="deployment-write-migration",
        )
    if not isinstance(raw_storage, dict):
        raise AgentStorageCredentialError(
            "owner credential store has malformed artifact read credentials"
        )
    storage = raw_storage
    storage = storage if isinstance(storage, dict) else {}
    saved_bucket = (
        str(storage.get("bucket") or storage.get("s3_bucket") or "")
        .removeprefix("s3://")
        .strip("/")
    )
    saved_endpoint = str(
        storage.get("endpoint_url") or storage.get("endpoint") or ""
    ).strip()
    saved_access_key = str(
        storage.get("aws_access_key_id") or storage.get("access_key_id") or ""
    ).strip()
    saved_secret_key = str(
        storage.get("aws_secret_access_key") or storage.get("secret_access_key") or ""
    ).strip()
    saved_project = str(storage.get("project_id") or source_project).strip()
    saved_role = str(storage.get("iam_role") or "").strip()
    saved_prefixes = tuple(
        sorted(
            {
                str(item or "").strip().strip("/")
                for item in (storage.get("resolved_prefixes") or ())
                if str(item or "").strip().strip("/")
            }
        )
    )
    if saved_bucket != source_bucket or not (
        saved_endpoint and saved_access_key and saved_secret_key
    ):
        raise AgentStorageCredentialError(
            "owner credential store has no exact matching artifact read credentials"
        )
    if (
        saved_project != source_project
        or saved_role != "storage.viewer"
        or not set(source_prefixes).issubset(saved_prefixes)
    ):
        raise AgentStorageCredentialError(
            "artifact read credential scope does not match the exact configured source"
        )
    return ArtifactStorageCredentialResolution(
        credentials=(
            source_bucket,
            source_prefix,
            saved_endpoint,
            saved_access_key,
            saved_secret_key,
            str(storage.get("service_account_id") or "").strip(),
        ),
        mode="isolated-read",
    )


def resolve_configured_artifact_storage_credentials(
    artifact_sources: tuple[dict[str, str], ...] | list[dict[str, str]],
    *,
    deployment_project_id: str,
    current: tuple[str, str, str, str, str, str],
    allow_deployment_write_migration: bool = False,
) -> tuple[str, str, str, str, str, str]:
    """Compatibility wrapper returning only the resolved credential tuple."""

    return resolve_configured_artifact_storage_identity(
        artifact_sources,
        deployment_project_id=deployment_project_id,
        current=current,
        allow_deployment_write_migration=allow_deployment_write_migration,
    ).credentials


def resolve_bootstrap_artifact_storage_identity(
    artifact_sources: tuple[dict[str, str], ...] | list[dict[str, str]],
    *,
    deployment_project_id: str,
    current: tuple[str, str, str, str, str, str],
    persisted_mode: str,
    migration_requested: bool,
) -> ArtifactStorageCredentialResolution:
    """Resolve bootstrap identity while making persisted migration explicit."""

    return resolve_configured_artifact_storage_identity(
        artifact_sources,
        deployment_project_id=deployment_project_id,
        current=current,
        allow_deployment_write_migration=(
            migration_requested
            or str(persisted_mode or "").strip() == "deployment-write-migration"
        ),
    )


def resolve_agent_artifact_sources(
    record: dict[str, Any], *, artifact_source_file: str = ""
) -> tuple[dict[str, str], ...]:
    """Prefer an explicit owner file, otherwise reuse the durable agent record."""
    if str(artifact_source_file or "").strip():
        return _load_agent_artifact_sources_file(artifact_source_file)
    return normalize_configured_artifact_sources(record.get("artifact_sources") or ())


__all__ = [
    "AgentStorageCredentialError",
    "ArtifactStorageCredentialResolution",
    "artifact_credential_record_summary",
    "artifact_credential_summary",
    "artifact_source_file_option",
    "artifact_write_identity_migration_option",
    "emit_artifact_migration_warning",
    "resolve_agent_artifact_sources",
    "resolve_agent_service_account_id",
    "resolve_agent_storage_credentials",
    "resolve_bootstrap_artifact_storage_identity",
    "resolve_configured_artifact_storage_credentials",
    "resolve_configured_artifact_storage_identity",
]
