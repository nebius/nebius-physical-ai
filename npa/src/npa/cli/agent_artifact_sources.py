"""Durable exact artifact-source configuration for agent bootstrap."""

from __future__ import annotations

from typing import Any, Callable

from npa.clients.project_credential_store import project_credential_record
from npa.cli.agent_access import normalize_configured_artifact_sources
from npa.cli.agent_env_files import _load_agent_artifact_sources_file


class AgentStorageCredentialError(RuntimeError):
    """Configured/bootstrap storage cannot satisfy the data-plane contract."""


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


def resolve_configured_artifact_storage_credentials(
    artifact_sources: tuple[dict[str, str], ...] | list[dict[str, str]],
    *,
    deployment_project_id: str,
    current: tuple[str, str, str, str, str, str],
) -> tuple[str, str, str, str, str, str]:
    """Select an exact source project's owner-stored S3 identity.

    A source tuple is not a grant. Cross-project defaults therefore resolve
    credentials only from that exact project's private credential record.
    """
    sources = normalize_configured_artifact_sources(artifact_sources)
    if not sources:
        return current
    source_projects = {item["project_id"] for item in sources}
    if len(source_projects) != 1:
        raise AgentStorageCredentialError(
            "configured artifact sources must use one exact credential project"
        )
    source_project = next(iter(source_projects))
    source_bucket = sources[0]["bucket"]
    source_prefix = sources[0]["resolved_prefix"]
    _current_bucket, _prefix, endpoint, access_key, secret_key, service_account_id = (
        current
    )
    if source_project == str(deployment_project_id or "").strip():
        if not (endpoint and access_key and secret_key):
            raise AgentStorageCredentialError(
                "deployment project has no owner-stored S3 credentials for the "
                "configured artifact source"
            )
        return (
            source_bucket,
            source_prefix,
            endpoint,
            access_key,
            secret_key,
            service_account_id,
        )

    record = project_credential_record(source_project, migrate_legacy=False)
    storage = record.get("storage") if isinstance(record, dict) else None
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
    if saved_bucket != source_bucket or not (
        saved_endpoint and saved_access_key and saved_secret_key
    ):
        raise AgentStorageCredentialError(
            "owner credential store has no exact matching artifact source credentials"
        )
    return (
        source_bucket,
        source_prefix,
        saved_endpoint,
        saved_access_key,
        saved_secret_key,
        service_account_id,
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
    "resolve_agent_artifact_sources",
    "resolve_agent_service_account_id",
    "resolve_agent_storage_credentials",
    "resolve_configured_artifact_storage_credentials",
]
