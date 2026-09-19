"""Durable exact artifact-source configuration for agent bootstrap."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import typer

from npa.clients.project_credential_store import project_credential_record
from npa.cli.agent_access import (
    ACCESS_SCHEMA,
    ACCESS_STATES,
    normalize_configured_artifact_sources,
)
from npa.cli.agent_env_files import (
    _load_agent_artifact_sources_file,
    _write_agent_artifact_sources_env,
)


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


def write_artifact_sources_env(
    ssh: Any,
    artifact_sources: tuple[dict[str, str], ...] | list[dict[str, str]],
    resolution: ArtifactStorageCredentialResolution | None,
    region: str,
) -> None:
    """Stage the resolved read identity without expanding it in the CLI module."""

    credentials = resolution.credentials if resolution else ("", "", "", "", "", "")
    bucket, _prefix, endpoint, access_key, secret_key, _service_account = credentials
    _write_agent_artifact_sources_env(
        ssh,
        artifact_sources=artifact_sources,
        bucket=bucket,
        endpoint=endpoint,
        access_key=access_key,
        secret_key=secret_key,
        region=region,
        credential_mode=resolution.mode if resolution else "unconfigured",
    )


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


def validate_live_artifact_credentials(
    record: dict[str, Any], payload: dict[str, Any]
) -> None:
    """Require live artifact-read status to match the persisted bootstrap mode."""

    credentials = payload.get("artifact_credentials")
    if not isinstance(credentials, dict):
        raise AgentStorageCredentialError(
            "agent access endpoint did not report artifact credential separation"
        )
    expected_mode = str(record.get("artifact_credential_mode") or "unconfigured")
    if credentials.get("mode") != expected_mode:
        raise AgentStorageCredentialError(
            "agent artifact credential mode differs from bootstrap state"
        )
    if record.get("artifact_sources") and credentials.get("status") not in {
        "ready",
        "warning",
    }:
        raise AgentStorageCredentialError(
            "configured artifact source has no usable artifact read identity"
        )
    if expected_mode == "isolated-read" and credentials.get("status") != "ready":
        raise AgentStorageCredentialError(
            "isolated artifact read identity is not ready"
        )


def validate_live_access_payload(
    record: dict[str, Any], payload: Any
) -> dict[str, Any]:
    """Validate the access endpoint schema and its artifact-read identity."""

    if not isinstance(payload, dict) or payload.get("apiVersion") != ACCESS_SCHEMA:
        raise AgentStorageCredentialError(
            "agent access endpoint returned an invalid schema"
        )
    if payload.get("status") not in ACCESS_STATES:
        raise AgentStorageCredentialError(
            "agent access endpoint returned an invalid status"
        )
    if not isinstance(payload.get("projects"), list):
        raise AgentStorageCredentialError(
            "agent access endpoint did not return a projects list"
        )
    validate_live_artifact_credentials(record, payload)
    return payload


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


def _provider_verified_artifact_read_identity(
    *,
    storage: dict[str, Any],
    source_project_id: str,
    project_sources: list[dict[str, str]],
    access_key: str,
    service_account_id: str,
) -> bool:
    """Verify key ownership and exact read-only bucket policy from Nebius state."""

    group_id = str(storage.get("iam_group_id") or "").strip()
    credential_project = str(
        storage.get("credential_project_id") or source_project_id
    ).strip()
    if not all((group_id, credential_project, access_key, service_account_id)):
        return False
    try:
        from npa.clients import nebius

        if not nebius._group_has_member(group_id, service_account_id):
            return False
        key_matches = False
        for key in nebius.list_access_keys_for_service_account(
            credential_project, service_account_id, strict=True
        ):
            resource_id = str(key.get("id") or "").strip()
            if not resource_id:
                continue
            live = nebius._run_json(
                ["iam", "v2", "access-key", "get", "--id", resource_id]
            )
            if (
                str((live.get("status") or {}).get("aws_access_key_id") or "")
                == access_key
            ):
                key_matches = True
                break
        if not key_matches:
            return False
        for source in project_sources:
            bucket = str(source.get("bucket") or "").strip()
            prefix = str(source.get("resolved_prefix") or "").strip().strip("/")
            item = nebius.get_bucket_by_name(source_project_id, bucket)
            spec = (item or {}).get("spec") if isinstance(item, dict) else None
            policy = spec.get("bucket_policy") if isinstance(spec, dict) else None
            rules = policy.get("rules") if isinstance(policy, dict) else None
            expected_path = f"{prefix}/*"
            if not isinstance(rules, list) or not any(
                isinstance(rule, dict)
                and str(rule.get("group_id") or "") == group_id
                and sorted(str(value) for value in rule.get("paths", []))
                == [expected_path]
                and sorted(str(value) for value in rule.get("roles", []))
                == ["storage.viewer"]
                and not rule.get("anonymous")
                for rule in rules
            ):
                return False
    except Exception:  # noqa: BLE001 - fail closed without provider diagnostics
        return False
    return True


def resolve_configured_artifact_storage_identity(
    artifact_sources: tuple[dict[str, str], ...] | list[dict[str, str]],
    *,
    deployment_project_id: str,
    current: tuple[str, str, str, str, str, str],
    allow_deployment_write_migration: bool = False,
) -> ArtifactStorageCredentialResolution:
    """Select one shared owner-stored identity for exact artifact sources.

    A source tuple is not a grant. The normal path therefore requires a
    separately recorded ``storage.viewer`` scope for every exact project,
    bucket, and source prefix. Multiple source records may reference the same
    dedicated key, but mixed identities fail closed because the shipped backend
    has one artifact-read channel. Reuse of the deployment writer exists only
    as an explicit single-source same-project migration mode and is never an
    implicit fallback.
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
    primary_source = sources[0]
    source_bucket = primary_source["bucket"]
    source_prefix = primary_source["resolved_prefix"] if len(sources) == 1 else ""
    current_bucket, _prefix, endpoint, access_key, secret_key, service_account_id = (
        current
    )
    shared_identity: tuple[str, str, str, str] | None = None
    for source_project in sorted(source_projects):
        project_sources = [
            item for item in sources if item["project_id"] == source_project
        ]
        record = project_credential_record(source_project, migrate_legacy=False)
        raw_storage = (
            record.get("artifact_read_storage") if isinstance(record, dict) else None
        )
        if raw_storage is None:
            single_source_migration = (
                len(sources) == 1
                and source_project == str(deployment_project_id or "").strip()
            )
            if not single_source_migration:
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
        saved_endpoint = str(
            storage.get("endpoint_url") or storage.get("endpoint") or ""
        ).strip()
        saved_access_key = str(
            storage.get("aws_access_key_id") or storage.get("access_key_id") or ""
        ).strip()
        saved_secret_key = str(
            storage.get("aws_secret_access_key")
            or storage.get("secret_access_key")
            or ""
        ).strip()
        saved_service_account = str(storage.get("service_account_id") or "").strip()
        saved_project = str(
            storage.get("source_project_id")
            or storage.get("project_id")
            or source_project
        ).strip()
        saved_role = str(storage.get("iam_role") or "").strip()
        raw_scopes = storage.get("source_scopes")
        if raw_scopes is None:
            saved_bucket = (
                str(storage.get("bucket") or storage.get("s3_bucket") or "")
                .removeprefix("s3://")
                .strip("/")
            )
            raw_scopes = [
                {
                    "bucket": saved_bucket,
                    "resolved_prefixes": storage.get("resolved_prefixes") or (),
                }
            ]
        if not isinstance(raw_scopes, (list, tuple)):
            raise AgentStorageCredentialError(
                "owner credential store has malformed artifact read credentials"
            )
        scopes: dict[str, set[str]] = {}
        for scope in raw_scopes:
            if not isinstance(scope, dict):
                raise AgentStorageCredentialError(
                    "owner credential store has malformed artifact read credentials"
                )
            bucket = (
                str(scope.get("bucket") or scope.get("s3_bucket") or "")
                .removeprefix("s3://")
                .strip("/")
            )
            prefixes = scope.get("resolved_prefixes")
            if not bucket or not isinstance(prefixes, (list, tuple)):
                raise AgentStorageCredentialError(
                    "owner credential store has malformed artifact read credentials"
                )
            scopes.setdefault(bucket, set()).update(
                str(item or "").strip().strip("/") for item in prefixes
            )
        if not (saved_endpoint and saved_access_key and saved_secret_key):
            raise AgentStorageCredentialError(
                "owner credential store has no exact matching artifact read credentials"
            )
        if (
            saved_project != source_project
            or saved_role != "storage.viewer"
            or not saved_service_account
            or any(
                item["resolved_prefix"] not in scopes.get(item["bucket"], set())
                for item in project_sources
            )
        ):
            raise AgentStorageCredentialError(
                "artifact read credential scope does not match the exact configured source"
            )
        if not _provider_verified_artifact_read_identity(
            storage=storage,
            source_project_id=source_project,
            project_sources=project_sources,
            access_key=saved_access_key,
            service_account_id=saved_service_account,
        ):
            raise AgentStorageCredentialError(
                "artifact read credential identity or storage.viewer binding could not be verified"
            )
        identity = (
            saved_endpoint,
            saved_access_key,
            saved_secret_key,
            saved_service_account,
        )
        if shared_identity is not None and identity != shared_identity:
            raise AgentStorageCredentialError(
                "configured artifact sources do not share one read-only identity"
            )
        shared_identity = identity
    if shared_identity is None:
        raise AgentStorageCredentialError(
            "owner credential store has no exact matching artifact read credentials"
        )
    saved_endpoint, saved_access_key, saved_secret_key, saved_service_account = (
        shared_identity
    )
    return ArtifactStorageCredentialResolution(
        credentials=(
            source_bucket,
            source_prefix,
            saved_endpoint,
            saved_access_key,
            saved_secret_key,
            saved_service_account,
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
