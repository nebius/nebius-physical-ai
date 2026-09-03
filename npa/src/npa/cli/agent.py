"""Top-level CLI for deploying and operating the NPA agent VM."""

from __future__ import annotations

import base64
import functools
import hashlib
import inspect
import json
import os
import secrets
import shlex
import shutil
import subprocess
import ipaddress
import tarfile
import tempfile
from pathlib import Path
from typing import Any, NoReturn

import httpx
import typer

from npa.cli._typer_defaults import resolve_typer_defaults
from npa.clients.project_credential_store import (
    persist_agent_service_account_id as _persist_project_agent_service_account_id,
    persist_agent_terraform_credentials,
)
from npa.lifecycle_intent import OperationIntent, intent_boundary
from npa.cli.agent_errors import _agent_deploy_failure_hint
from npa.cli.agent_quota import (
    _agent_check_compute_instance_quota,  # noqa: F401 - compatibility re-export
    _agent_check_public_ip_quota,  # noqa: F401 - compatibility re-export
    _agent_check_whole_path_capacity,
    _agent_compute_instance_quota_result,  # noqa: F401 - compatibility re-export
    _agent_public_ip_quota_result,  # noqa: F401 - compatibility re-export
    _agent_whole_path_capacity_result,
)
from npa.cli.agent_assets import (  # noqa: F401 - re-exported for tests/callers
    _agent_public_login_form_html,
    _lichtblick_default_layout_json,
)
from npa.cli.agent_env_files import (  # noqa: F401 - re-exported for tests/callers
    _stage_private_text,
    _write_agent_nebius_env,
    _write_agent_operator_profile,
    _write_agent_s3_env,
)
from npa.cli.agent_destroy import destroy_cmd as _destroy_cmd_impl
from npa.cli.agent_auth import auth_profile_cmd
from npa.cli.agent_inventory import agent_list_cmd
from npa.cli.agent_preflight import (
    _agent_hard_prereq_results,
    _agent_nebius_auth_result,
    _agent_storage_result,
    _agent_token_factory_result,
    _render_agent_cloud_init,  # noqa: F401 - compatibility re-export for tests
    _render_agent_checks,
)
from npa.cli.agent_records import (  # noqa: F401 - compatibility re-exports
    agent_record as _agent_record,
    remove_agent_record as _remove_agent_record,
    resolve_project_agents,
    store_agent_record as _store_agent_record,
)
from npa.cli.agent_network import (
    _agent_ssh_egress_result,
)
from npa.cli.agent_payloads import (
    agent_credentials_payload as _agent_credentials_payload,
    coerce_cli_list as _coerce_cli_list,
    tool_catalog_payload as _tool_catalog_payload,
)
from npa.cli.agent_setup_convergence import (
    converge_remote_agent_setup,
    reconcile_agent_setup as _reconcile_agent_setup,
)
from npa.cli.agent_terraform import (
    _agent_terraform_state_exists,  # noqa: F401 - recovery hook re-export
    _apply_agent_terraform,
    _cleanup_orphan_agent_instances,  # noqa: F401 - recovery hook re-export
    _destroy_agent_terraform,
    _ensure_terraform_state_bucket,
    _persist_agent_project_config,
    _record_agent_destroy_event,  # noqa: F401 - recovery hook re-export
    _resolve_destroy_tf_vars,  # noqa: F401 - recovery hook re-export
)
from npa.clients.config import (
    ConfigError,
    resolve_environment,
    resolve_project_storage,
    resolve_ssh_config,
    resolve_terraform_state,
    write_config,
)
from npa.clients.env import redact_value
from npa.clients.network import (
    NetworkIngressError,
    ensure_ingress,
    remove_ingress_for_instance,
    remove_npa_ingress_for_instance_ports,
    resolve_instance_network_context,
)
from npa.clients.ssh import SSHClient, SSHError
from npa.cli.agent_public import (
    AgentConfig,
    build_agent_urls,
    record_public_https as _record_public_https,
    record_tls_verify as _record_tls_verify,
    record_customer_url as _record_customer_url,
    resolve_record_public_ip,
)
from npa.agent_backend.shipping import render_shipped_backend_install
from npa.cli import agent_resources
from npa.cli.agent_access import (
    ACCESS_SCHEMA,
    ACCESS_STATES,
    consistent_agent_service_account_id,
)
from npa.cli.agent_contracts import (  # noqa: F401 - public compatibility exports
    AGENT_CHAT_QUEUE_CONTRACT,
    AGENT_FOXGLOVE_CONTRACT,
    AGENT_LEISAAC_CONTRACT,
    AGENT_MEDIA_PREVIEW_CONTRACT,
    AGENT_READABLE_COLOR_CONTRACT,
    AGENT_RERUN_NO_BUNDLE_SPLASH_CONTRACT,
    AGENT_STAGES_RUN_PICKER_CONTRACT,
    AGENT_VIEWER_CHAT_DRAWER_CONTRACT,
    AGENT_VISUAL_FEEDBACK_CONTRACT,
    LEISAAC_CONTROL_READINESS_CONTRACT,
    _embedded_agent_artifact_content_source,
    _embedded_agent_artifacts_source,
    _embedded_agent_chat_source,
    _embedded_agent_provenance_source,
    _embedded_agent_recordings_source,
    _embedded_agent_routing_source,
    _embedded_agent_rrd_proxy_source,
    _embedded_agent_s3_guard_source,
    _embedded_agent_stages_source,
    _embedded_agent_state_source,
    _embedded_agent_visual_feedback_source,
    _embedded_agent_workflow_source,
    rendered_agent_ui_html,
)
from npa.cli.agent_embed import embedded_python_source
from npa.cli.agent_site import (
    DEFAULT_LICHTBLICK_PORT,
    nginx_agent_site_body as _nginx_agent_site_body,
)
from npa.cli.agent_deployment import (
    DeploymentIdentityError,
    assert_remote_owner_if_present,
    build_deployment_manifest,
    verify_remote_deployment,
)
from npa.deploy import provisioner
from npa.deploy.images import container_image_candidates
from npa.deploy.provisioner import ProvisionerError
from npa.provisioning_journal import (
    ProvisioningOperation,
    current_operation,
    emit_recovery_summary,
    operation_context,
    operation_heartbeats,
)
from npa.cli import agent_foxglove_config

app = typer.Typer(
    name="agent",
    help="Deploy and operate a public NPA chat agent VM.",
    no_args_is_help=True,
)
app.command("auth-profile")(auth_profile_cmd)
app.command("list")(agent_list_cmd)

DEFAULT_AGENT_PORT = 8088
DEFAULT_BACKEND_PORT = 8787
DEFAULT_RERUN_PORT = 9090
DEFAULT_PROJECT_ALIAS = "us-central1"
DEFAULT_AGENT_NAME = "agent"
# The public agent VM is CPU-only (gpu_platform="cpu-d3" below). Use a
# driverless (non-CUDA) Ubuntu image: it is the correct base for a CPU host,
# boots faster (no GPU driver install), and — unlike the CUDA image families —
# is published across Nebius regions. The Terraform default is a CUDA image
# (variables.tf: ubuntu24.04-cuda12), which is absent outside a few regions and
# makes the boot-disk create fail with "Image family ... not found" (e.g. in
# uk-south1). Pinning a driverless family here keeps agent deploy region-portable.
DEFAULT_AGENT_IMAGE_FAMILY = "ubuntu24.04-driverless"
DEFAULT_AGENT_USER = "npa"
DEFAULT_LLM_PROVIDER = "token_factory"
DEFAULT_LLM_MODEL = "nvidia/Cosmos3-Super-Reasoner"
# Cost-ordered ladder; per-turn routing reorders it, while no-routing paths and
# the model picker retain the cheap workhorse as their default.
DEFAULT_LLM_MODELS = (
    "Qwen/Qwen3-32B",
    "meta-llama/Llama-3.3-70B-Instruct",
    DEFAULT_LLM_MODEL,
    "Qwen/Qwen2.5-VL-72B-Instruct",
)
AGENT_UI_VERSION = "2026082901"
ARTIFACT_DISCOVERY_CONTRACT = "s3-source-qualified-v1"
DEFAULT_HTTPS_PORT = 443
AGENT_SOURCE_ROOT = "/opt/npa-agent/npa-src"
_AGENT_TERRAFORM_RUNTIME_ONLY_VARS = frozenset(
    {
        "s3_prefix",
        "s3_session_token",
        # Backend HMAC credentials are supplied to Terraform only through the
        # scrubbed subprocess environment. They must never become input vars,
        # plan/state values, or compute-instance user-data.
        "nebius_api_key",
        "nebius_secret_key",
    }
)


class AgentStorageCredentialError(RuntimeError):
    """Configured/bootstrap storage cannot satisfy the deploy data-plane contract."""


# Contract markers that must stay in the embedded agent UI/backend. verify-live,
# smoke, and unit tests share this list so media-preview regressions cannot
# silently disappear after a bootstrap drift or template edit.
_AGENT_CHAT_EMBED = "__NPA_AGENT_CHAT_EMBED__"
_AGENT_RECORDINGS_EMBED = "__NPA_AGENT_RECORDINGS_EMBED__"
_AGENT_BACKEND_SHIP = "__NPA_AGENT_BACKEND_SHIP__"
_AGENT_WORKFLOW_EMBED = "__NPA_AGENT_WORKFLOW_EMBED__"
_AGENT_ARTIFACTS_EMBED = "__NPA_AGENT_ARTIFACTS_EMBED__"
_AGENT_ACCESS_EMBED = "__NPA_AGENT_ACCESS_EMBED__"
_AGENT_ACCESS_RUNTIME_EMBED = "__NPA_AGENT_ACCESS_RUNTIME_EMBED__"
_AGENT_ARTIFACT_CONTENT_EMBED = "__NPA_AGENT_ARTIFACT_CONTENT_EMBED__"
_AGENT_ROUTING_EMBED = "__NPA_AGENT_ROUTING_EMBED__"
_AGENT_VISUAL_FEEDBACK_EMBED = "__NPA_AGENT_VISUAL_FEEDBACK_EMBED__"
_AGENT_RRD_PROXY_EMBED = "__NPA_AGENT_RRD_PROXY_EMBED__"
_AGENT_STATE_EMBED = "__NPA_AGENT_STATE_EMBED__"
_AGENT_S3_GUARD_EMBED = "__NPA_AGENT_S3_GUARD_EMBED__"
_AGENT_STAGES_EMBED = "__NPA_AGENT_STAGES_EMBED__"
_AGENT_STAGE_RUNTIME_EMBED = "__NPA_AGENT_STAGE_RUNTIME_EMBED__"
_AGENT_VIEWER_RUNTIME_EMBED = "__NPA_AGENT_VIEWER_RUNTIME_EMBED__"
_AGENT_PROVENANCE_EMBED = "__NPA_AGENT_PROVENANCE_EMBED__"
_AGENT_UI_HTML_EMBED = "__NPA_AGENT_UI_HTML__"


def _resolve_record_public_ip(record: dict[str, Any]) -> str:
    return resolve_record_public_ip(record, resolver=resolve_instance_network_context)


def _embedded_agent_module_source(filename: str) -> str:
    """Return one standalone agent module embedded into the backend."""
    return embedded_python_source(
        Path(__file__).with_name(filename), strip_standalone=True
    )


_embedded_agent_stage_runtime_source = functools.partial(
    _embedded_agent_module_source, "agent_stage_runtime.py"
)
_embedded_agent_viewer_runtime_source = functools.partial(
    _embedded_agent_module_source, "agent_viewer_runtime.py"
)
_embedded_agent_access_source = functools.partial(
    _embedded_agent_module_source, "agent_access.py"
)
_embedded_agent_access_runtime_source = functools.partial(
    _embedded_agent_module_source, "agent_access_runtime.py"
)


def _fail(message: str) -> NoReturn:
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(code=1)


def _resolve_foxglove_settings_or_fail(**settings: Any) -> dict[str, str]:
    """Translate expected Foxglove setting errors at the CLI boundary."""
    try:
        return agent_foxglove_config.resolve_settings(**settings)
    except agent_foxglove_config.FoxgloveSettingsError as exc:
        _fail(str(exc))


def _looks_like_compute_permission_denied(message: str) -> bool:
    lowered = str(message or "").lower()
    return "permissiondenied" in lowered and "service compute" in lowered


def _resolve_project_alias(project: str) -> str:
    """Resolve an agent ``--project``: explicit value, else the configured
    ``default_project``, else the only configured project, else the static
    ``DEFAULT_PROJECT_ALIAS`` fallback.

    Commands default ``--project`` to "" so an omitted flag targets the operator's
    configured default project rather than a hard-coded ``us-central1`` alias
    (which risked tearing down / inspecting the wrong thing).
    """
    alias = str(project or "").strip()
    if alias:
        return alias
    try:
        from npa.clients.config import default_project_name, list_projects

        configured = str(default_project_name() or "").strip()
        projects = list_projects()
    except Exception:  # noqa: BLE001 - best-effort; fall back to the static default
        return DEFAULT_PROJECT_ALIAS
    if configured and configured in projects:
        return configured
    # `default_project_name()` returns the literal "default" when the config has no
    # default_project, which names no real stanza. A single configured project is
    # the unambiguous target (matching `npa agent setup`).
    if len(projects) == 1:
        return next(iter(projects))
    return configured or DEFAULT_PROJECT_ALIAS


def _agent_extra_ingress_ports(
    *,
    agent_port: int,
    rerun_port: int,
    public_https: bool,
) -> list[int]:
    extra = [rerun_port]
    if public_https:
        extra.append(DEFAULT_HTTPS_PORT)
    return sorted({port for port in extra if port != agent_port})


def _cleanup_agent_ingress(instance_id: str) -> None:
    if not str(instance_id or "").strip():
        return
    try:
        remove_ingress_for_instance(
            str(instance_id).strip(),
            on_status=lambda msg: typer.echo(f"  {msg}"),
        )
    except NetworkIngressError as exc:
        typer.echo(f"  Warning: could not remove npa ingress rules: {exc}", err=True)


def _auth_secret_path(project_alias: str, name: str) -> Path:
    return Path.home() / ".npa" / "agents" / project_alias / name / "auth.env"


def _cleanup_agent_local_files(project_alias: str, name: str) -> None:
    """Remove the local agent state + Terraform workdir after a destroy.

    Two trees live under ``~/.npa`` for an agent: ``agents/<alias>/<name>/``
    (auth.env + secrets — live basic-auth credentials, a stale-credential leak
    if left) and ``workbenches/<alias>/<name>/`` (the Terraform workdir with the
    provider cache and, in a local backend, ``terraform.tfstate``). Terraform has
    already destroyed the VM by the time this runs, so both are safe to remove;
    leaving the workdir behind was the teardown-report leftover.
    """
    agent_dir = Path.home() / ".npa" / "agents" / project_alias / name
    shutil.rmtree(agent_dir, ignore_errors=True)

    from npa.deploy import provisioner

    tf_dir = provisioner.working_dir_path(project_alias, name)
    shutil.rmtree(tf_dir, ignore_errors=True)

    # Drop the now-empty <alias> parents so tearing down the last agent leaves no
    # empty ~/.npa/{agents,workbenches}/<alias>/ tree behind (a sibling agent
    # under the same alias keeps its parent non-empty, so it is preserved).
    for parent in (agent_dir.parent, tf_dir.parent):
        try:
            if parent.is_dir() and not any(parent.iterdir()):
                parent.rmdir()
        except OSError:
            pass


def _write_auth_secret(
    *, project_alias: str, name: str, user: str, password: str
) -> Path:
    path = _auth_secret_path(project_alias, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"AGENT_USER={user}\nAGENT_PASSWORD={password}\n", encoding="utf-8")
    path.chmod(0o600)
    return path


def _load_auth_secret(path: str) -> tuple[str, str]:
    secret_path = Path(path).expanduser()
    if not secret_path.exists():
        raise ValueError(f"auth secret not found: {secret_path}")
    values: dict[str, str] = {}
    for raw in secret_path.read_text(encoding="utf-8").splitlines():
        if "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        values[key.strip()] = value.strip()
    user = values.get("AGENT_USER", "")
    password = values.get("AGENT_PASSWORD", "")
    if not user or not password:
        raise ValueError(
            f"auth secret missing AGENT_USER/AGENT_PASSWORD: {secret_path}"
        )
    return user, password


def _resolve_deploy_llm_credentials() -> tuple[str, str]:
    """Return Token Factory API key and default model for agent VM bootstrap."""
    from npa.clients.credentials import load_credentials

    creds = load_credentials()
    return creds.token_factory_api_key, DEFAULT_LLM_MODEL


def _normalize_llm_models(models: list[str] | tuple[str, ...] | str) -> list[str]:
    """Return an ordered, unique model list from repeated or comma-separated values."""
    if isinstance(models, str):
        raw_items = [models]
    else:
        raw_items = list(models)
    normalized: list[str] = []
    for raw in raw_items:
        for chunk in str(raw).replace("\n", ",").split(","):
            value = chunk.strip()
            if value and value not in normalized:
                normalized.append(value)
    if not normalized:
        normalized = list(DEFAULT_LLM_MODELS)
    if DEFAULT_LLM_MODEL not in normalized:
        normalized.insert(0, DEFAULT_LLM_MODEL)
    return normalized


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


def _resolve_deploy_storage_credentials(
    *,
    region: str,
    bootstrap_creds: dict[str, str] | None = None,
    project_alias: str = "",
    emit_status: bool = True,
) -> dict[str, str]:
    """Resolve the exact writable-storage credentials agent deploy will use.

    Project-scoped credentials are evaluated before shared and bootstrap
    credentials. This is the same health-verified decision used by preflight,
    setup, fresh-setup, and refresh; IAM access-key inventory is only reached
    later when no configured candidate works and a caller explicitly supplies
    freshly bootstrapped credentials.
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
        except ConfigError:
            project_storage = None
        if project_storage is not None:
            project_bucket = str(project_storage.checkpoint_bucket or "").strip()
            project_prefix = ""
            if project_bucket.startswith("s3://"):
                rest = project_bucket[len("s3://") :]
                project_bucket, _sep, project_prefix = rest.partition("/")
                project_prefix = project_prefix.strip("/")
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
                if emit_status:
                    typer.echo(
                        "  Using health-verified project artifact storage credentials."
                    )
                candidate["s3_bucket"] = project_bucket
                candidate["s3_prefix"] = project_prefix
                candidate["s3_endpoint"] = project_endpoint
                candidate["nebius_api_key"] = project_access_key
                candidate["nebius_secret_key"] = project_secret_key
                return candidate
    # With no selected project, resolve_project_storage(None) is the canonical
    # shared/default storage view and may combine the configured bucket with the
    # shared credential file. Never use this view for an explicit project: that
    # path above deliberately disables shared credential injection first.
    if not project_name:
        try:
            configured = resolve_project_storage(None)
        except ConfigError:
            configured = None
        configured_bucket = str(
            getattr(configured, "checkpoint_bucket", "") or ""
        ).strip()
        configured_prefix = ""
        if configured_bucket.startswith("s3://"):
            rest = configured_bucket[len("s3://") :]
            configured_bucket, _sep, configured_prefix = rest.partition("/")
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
        if configured_bucket and _storage_credentials_allow_writes(
            bucket=configured_bucket,
            endpoint=configured_endpoint,
            access_key=configured_access_key,
            secret_key=configured_secret_key,
            region=region,
            prefix=configured_prefix,
        ):
            if emit_status:
                typer.echo(
                    "  Using health-verified configured artifact storage credentials."
                )
            candidate["s3_bucket"] = configured_bucket
            candidate["s3_prefix"] = configured_prefix
            candidate["s3_endpoint"] = configured_endpoint
            candidate["nebius_api_key"] = configured_access_key
            candidate["nebius_secret_key"] = configured_secret_key
            return candidate
    # Never record a host-level shared bucket as an explicit project's remote
    # backend; keep immutable journal ownership exact.
    if not project_name:
        shared = load_credentials(environ={})
        shared_bucket = str(shared.s3_bucket or "").strip()
        shared_prefix = ""
        if shared_bucket.startswith("s3://"):
            rest = shared_bucket[len("s3://") :]
            shared_bucket, _sep, shared_prefix = rest.partition("/")
            shared_prefix = shared_prefix.strip("/")
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
            if emit_status:
                typer.echo(
                    "  Using health-verified shared artifact storage credentials."
                )
            candidate["s3_bucket"] = shared_bucket
            candidate["s3_prefix"] = shared_prefix
            candidate["s3_endpoint"] = shared_endpoint
            candidate["nebius_api_key"] = shared_access_key
            candidate["nebius_secret_key"] = shared_secret_key
            return candidate

    bucket = str(candidate.get("s3_bucket", "")).strip()
    endpoint = str(candidate.get("s3_endpoint", "")).strip()
    access_key = str(candidate.get("nebius_api_key", "")).strip()
    secret_key = str(candidate.get("nebius_secret_key", "")).strip()
    if _storage_credentials_allow_writes(
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
                if emit_status:
                    typer.echo(
                        "  Bootstrap S3 key has no data-plane access; falling back "
                        "to saved project terraform_state credentials."
                    )
                candidate["s3_bucket"] = saved_bucket
                candidate["s3_endpoint"] = saved_endpoint
                candidate["nebius_api_key"] = saved_access_key
                candidate["nebius_secret_key"] = saved_secret_key
                return candidate
    raise AgentStorageCredentialError(
        "unable to verify writable S3 credentials for deploy; "
        "configure object-storage credentials with data-plane access before deploying the agent"
    )


def _resolve_agent_service_account_id(
    project_alias: str,
    record: dict[str, Any],
) -> str:
    """Resolve service-account id for agent bootstrap and credential persistence."""
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
    if project_id:
        resolved = resolve_service_account_id(project_id)
        if resolved:
            return resolved
    return ""


def _persist_agent_service_account_id(
    service_account_id: str, project_id: str = ""
) -> None:
    _persist_project_agent_service_account_id(project_id, service_account_id)


def _creds_from_terraform_state(
    project_alias: str, record: dict[str, Any]
) -> dict[str, str] | None:
    """Build a bootstrap-shaped credential dict from saved terraform remote-state keys."""
    try:
        tf_state = resolve_terraform_state(project_alias)
    except ConfigError:
        return None
    access_key = str(getattr(tf_state, "access_key", "") or "").strip()
    secret_key = str(getattr(tf_state, "secret_key", "") or "").strip()
    bucket = str(getattr(tf_state, "bucket", "") or "").strip()
    endpoint = str(getattr(tf_state, "endpoint", "") or "").strip()
    if not (access_key and secret_key and bucket):
        return None
    region = str(record.get("region", "") or "eu-north1").strip()
    service_account_id = _resolve_agent_service_account_id(project_alias, record)
    return {
        "service_account_id": service_account_id,
        "nebius_api_key": access_key,
        "nebius_secret_key": secret_key,
        "s3_bucket": bucket,
        "s3_endpoint": endpoint,
        "nebius_project_id": str(record.get("project_id", "")).strip(),
        "nebius_region": region,
    }


def _resolve_agent_ssh_key(
    record: dict[str, Any],
    *,
    cli_ssh_key: str | None = None,
    default_key: str = "~/.ssh/id_ed25519",
) -> str:
    """Resolve SSH private key for agent bootstrap without requiring workbench SSH config."""
    if cli_ssh_key and cli_ssh_key.strip():
        return str(Path(cli_ssh_key).expanduser())
    stored = str(record.get("ssh_key_path", "")).strip()
    if stored:
        return str(Path(stored).expanduser())
    env_key = os.environ.get("NPA_SSH_KEY", "").strip()
    if env_key:
        return str(Path(env_key).expanduser())
    return str(Path(default_key).expanduser())


def _resolve_agent_storage_credentials(
    project_alias: str,
    record: dict[str, Any],
) -> tuple[str, str, str, str, str, str]:
    """Return bucket, prefix, endpoint, access key, secret key, and service account id."""
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
                service_account_id = _resolve_agent_service_account_id(
                    project_alias, record
                )
            return bucket, prefix, endpoint, access_key, secret_key, service_account_id
    try:
        tf_state = resolve_terraform_state(project_alias)
    except ConfigError:
        return (
            "",
            "",
            "",
            "",
            "",
            _resolve_agent_service_account_id(project_alias, record),
        )
    service_account_id = _resolve_agent_service_account_id(project_alias, record)
    return (
        str(getattr(tf_state, "bucket", "") or ""),
        "",
        str(getattr(tf_state, "endpoint", "") or ""),
        str(getattr(tf_state, "access_key", "") or ""),
        str(getattr(tf_state, "secret_key", "") or ""),
        service_account_id,
    )


def _write_agent_llm_env(
    ssh: SSHClient,
    *,
    tf_api_key: str,
    llm_provider: str,
    llm_model: str,
    llm_providers: list[str] | tuple[str, ...] = (DEFAULT_LLM_PROVIDER,),
    llm_models: list[str] | tuple[str, ...] = DEFAULT_LLM_MODELS,
) -> None:
    """Stage Token Factory credentials on the VM (chmod 600, not baked into image)."""
    if not tf_api_key.strip():
        return
    models_csv = ",".join(_normalize_llm_models(list(llm_models)))
    providers_csv = ",".join(
        _normalize_llm_models(
            [str(item) for item in llm_providers if str(item).strip()]
        )
        or [DEFAULT_LLM_PROVIDER]
    )
    env_content = (
        f"NEBIUS_TOKEN_FACTORY_KEY={tf_api_key.strip()}\n"
        f"NPA_AGENT_LLM_PROVIDER={llm_provider.strip() or DEFAULT_LLM_PROVIDER}\n"
        f"NPA_AGENT_LLM_PROVIDERS={providers_csv}\n"
        f"NPA_AGENT_LLM_MODEL={llm_model}\n"
        f"NPA_AGENT_LLM_MODELS={models_csv}\n"
    )
    env_b64 = base64.b64encode(env_content.encode("utf-8")).decode("ascii")
    ssh.run_or_raise(
        f"echo {shlex.quote(env_b64)} | base64 -d | sudo tee /opt/npa-agent/llm.env >/dev/null "
        "&& sudo chmod 600 /opt/npa-agent/llm.env"
    )


def _store_project_environment(
    *, project: str, project_id: str, tenant_id: str, region: str
) -> None:
    """Persist a project-scoped Nebius environment like a fresh configure step."""
    operation = current_operation()
    if operation is not None:
        operation.record_config_mutation(
            store="config.yaml",
            fields=[
                "default_project",
                f"projects.{project}.project_id",
                f"projects.{project}.tenant_id",
                f"projects.{project}.region",
            ],
        )
    write_config(
        {
            "default_project": project,
            "projects": {
                project: {
                    "project_id": project_id,
                    "tenant_id": tenant_id,
                    "region": region,
                }
            },
        }
    )


def _create_agent_source_archive() -> str:
    """Package the NPA source tree needed for agent-side workflow execution."""
    repo_root = Path(__file__).resolve().parents[4]
    include_roots = [
        repo_root / "npa",
        repo_root / "deploy" / "cluster",
    ]
    for path in include_roots:
        if not path.exists():
            raise ConfigError(f"Required agent source path is missing: {path}")

    exclude_names = {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".terraform",
        ".venv",
        "__pycache__",
        "node_modules",
    }

    tmp = tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False)
    tmp.close()

    def _filter(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        parts = set(Path(info.name).parts)
        if parts & exclude_names:
            return None
        if info.name.endswith((".pyc", ".pyo")):
            return None
        return info

    with tarfile.open(tmp.name, "w:gz") as archive:
        archive.add(repo_root / "npa", arcname="npa", filter=_filter)
        archive.add(
            repo_root / "deploy" / "cluster", arcname="deploy/cluster", filter=_filter
        )
        # Stage the repo-root docs/ + skills/ trees so the agent's retrieval
        # corpus (Blueprint Phase H) can ground on them at
        # /opt/npa-agent/npa-src/{docs,skills}. Text-only; excluded via _filter.
        for extra in ("docs", "skills"):
            extra_path = repo_root / extra
            if extra_path.exists():
                archive.add(extra_path, arcname=extra, filter=_filter)
    return tmp.name


def _stage_agent_npa_source(ssh: SSHClient) -> None:
    """Upload NPA package source and deploy assets to the agent VM."""
    archive_path = _create_agent_source_archive()
    remote_archive = f"/tmp/npa-agent-source-{secrets.token_hex(6)}.tar.gz"
    try:
        ssh.upload_file(archive_path, remote_archive)
        ssh.run_or_raise(
            " && ".join(
                [
                    f"sudo rm -rf {shlex.quote(AGENT_SOURCE_ROOT)}",
                    f"sudo mkdir -p {shlex.quote(AGENT_SOURCE_ROOT)}",
                    f"sudo tar -xzf {shlex.quote(remote_archive)} -C {shlex.quote(AGENT_SOURCE_ROOT)}",
                    f"sudo chown -R root:root {shlex.quote(AGENT_SOURCE_ROOT)}",
                    f"sudo chmod -R a+rX {shlex.quote(AGENT_SOURCE_ROOT)}",
                    f"rm -f {shlex.quote(remote_archive)}",
                ]
            )
        )
    finally:
        Path(archive_path).unlink(missing_ok=True)
        ssh.run(f"rm -f {shlex.quote(remote_archive)}")


def _is_routable_public_ip(value: str) -> bool:
    candidate = (value or "").strip()
    if not candidate:
        return False
    if candidate == "localhost":
        return False
    try:
        ip = ipaddress.ip_address(candidate)
    except ValueError:
        return False
    if ip.is_loopback or ip.is_private or ip.is_unspecified or ip.is_link_local:
        return False
    return True


def _agent_strip_url_credentials_js() -> str:
    """JS to strip user:pass@ from the URL bar while keeping HTTP Basic auth session."""
    return """    <script>
    (function stripUrlCredentials() {
      try {
        if (location.username || location.password) {
          const clean = location.protocol + "//" + location.host + location.pathname + location.search + location.hash;
          history.replaceState(null, "", clean);
        }
      } catch (_err) { /* best-effort */ }
    })();
    </script>"""


def _agent_mobile_login_help_html() -> str:
    """Mobile certificate + sign-in troubleshooting (public pages)."""
    return """    <details class="mobile-help" style="margin:20px 0;padding:12px 16px;border:1px solid #e0e0e0;border-radius:8px;background:#fffbeb;">
      <summary style="font-weight:600;cursor:pointer;">Phone / tablet login help</summary>
      <ol style="margin:12px 0 0;padding-left:20px;line-height:1.55;">
        <li><strong>Accept the certificate first.</strong> Open <a href="/healthz">/healthz</a> (no login). If Safari/Chrome warns the connection is not private, tap <em>Show Details</em> → <em>visit this website</em> / <em>Proceed</em>.</li>
        <li>Return here and use the sign-in form (mobile browsers block password-in-URL redirects).</li>
        <li>If sign-in still fails, try <strong>Chrome on Android</strong> or use a desktop browser.</li>
        <li>Username is prefilled; password is in your operator <code>auth.env</code> file.</li>
      </ol>
    </details>"""


def _bootstrap_agent_stack(
    *,
    host: str,
    ssh_user: str,
    ssh_key_path: str,
    project_alias: str,
    agent_name: str = DEFAULT_AGENT_NAME,
    project_id: str,
    tenant_id: str,
    region: str,
    auth_user: str,
    auth_password: str,
    agent_port: int,
    backend_port: int,
    rerun_port: int,
    llm_model: str = DEFAULT_LLM_MODEL,
    llm_models: list[str] | tuple[str, ...] = DEFAULT_LLM_MODELS,
    tf_api_key: str = "",
    nebius_ai_key: str = "",
    service_account_id: str = "",
    s3_bucket: str = "",
    s3_prefix: str = "",
    s3_endpoint: str = "",
    s3_access_key: str = "",
    s3_secret_key: str = "",
    s3_region: str = "eu-north1",
    nebius_project_id: str = "",
    nebius_tenant_id: str = "",
    public_https: bool = True,
    foxglove_embed_src: str = "",
    foxglove_viewer_backend: str = "",
    foxglove_org_slug: str = "",
    foxglove_live_url: str = "",
    foxglove_cloud_import_timeout_seconds: str = "",
    deployment: dict[str, str] | None = None,
    preload_stock_demo: bool = True,
) -> None:
    foxglove_env = agent_foxglove_config.bootstrap_env_values(
        embed_src=foxglove_embed_src,
        viewer_backend=foxglove_viewer_backend,
        org_slug=foxglove_org_slug,
        live_url=foxglove_live_url,
        cloud_import_timeout_seconds=foxglove_cloud_import_timeout_seconds,
    )
    ssh = SSHClient(
        config=resolve_ssh_config(
            ssh_host=host,
            ssh_user=ssh_user,
            ssh_key=ssh_key_path,
            project=None,
            name=None,
        ).ssh
    )
    catalog_json = json.dumps(_tool_catalog_payload())
    agent_chat_source = _embedded_agent_chat_source()
    agent_recordings_source = _embedded_agent_recordings_source()
    agent_backend_ship_script = render_shipped_backend_install()
    agent_workflow_source = _embedded_agent_workflow_source()
    agent_artifacts_source = _embedded_agent_artifacts_source()
    agent_access_source = _embedded_agent_access_source()
    agent_access_runtime_source = _embedded_agent_access_runtime_source()
    agent_artifact_content_source = _embedded_agent_artifact_content_source()
    agent_routing_source = _embedded_agent_routing_source()
    agent_visual_feedback_source = _embedded_agent_visual_feedback_source()
    agent_rrd_proxy_source = _embedded_agent_rrd_proxy_source()
    agent_state_source = _embedded_agent_state_source()
    agent_s3_guard_source = _embedded_agent_s3_guard_source()
    agent_stages_source = _embedded_agent_stages_source()
    agent_stage_runtime_source = _embedded_agent_stage_runtime_source()
    agent_viewer_runtime_source = _embedded_agent_viewer_runtime_source()
    agent_provenance_source = _embedded_agent_provenance_source()
    deployment = deployment or build_deployment_manifest(
        project_alias=project_alias,
        name=agent_name,
        require_clean=False,
    )
    deployment_json = json.dumps(deployment, sort_keys=True)
    deployment_b64 = base64.b64encode(deployment_json.encode("utf-8")).decode("ascii")
    # This check runs before staging source, writing manifests, or restarting
    # services. A stale/missing local record cannot authorize overwriting a VM
    # that is still advertising a different immutable owner.
    assert_remote_owner_if_present(ssh, deployment, backend_port=backend_port)
    preload_stock_demo_value = "1" if preload_stock_demo else "0"
    llm_models = _normalize_llm_models(list(llm_models))
    default_llm_models_json = json.dumps(llm_models)
    nginx_site_body = _nginx_agent_site_body(
        backend_port=backend_port, rerun_port=rerun_port
    )
    login_form_html = _agent_public_login_form_html(auth_user)
    mobile_login_help_html = _agent_mobile_login_help_html()
    strip_url_credentials_js = _agent_strip_url_credentials_js()
    https_ssl_setup = ""
    https_server_block = ""
    if public_https:
        https_ssl_setup = f"""
sudo mkdir -p /etc/nginx/ssl
if [ ! -s /etc/nginx/ssl/npa-agent.crt ] || [ ! -s /etc/nginx/ssl/npa-agent.key ]; then
  sudo openssl req -x509 -nodes -newkey rsa:2048 -days 825 \\
    -keyout /etc/nginx/ssl/npa-agent.key \\
    -out /etc/nginx/ssl/npa-agent.crt \\
    -subj "/CN=npa-agent/O=Nebius Physical AI" \\
    -addext "subjectAltName=IP:{host}"
  sudo chmod 600 /etc/nginx/ssl/npa-agent.key
fi
"""
        https_server_block = f"""
server {{
  listen {DEFAULT_HTTPS_PORT} ssl;
  server_name _;
  ssl_certificate /etc/nginx/ssl/npa-agent.crt;
  ssl_certificate_key /etc/nginx/ssl/npa-agent.key;
{nginx_site_body}
}}
"""
    nebius_profile = "cursor-sa"
    nebius_parent_id = shlex.quote((nebius_project_id or project_id).strip())
    expected_agent_service_account_id = shlex.quote(service_account_id.strip())
    expected_agent_tenant_id = shlex.quote((nebius_tenant_id or tenant_id).strip())
    lichtblick_port = DEFAULT_LICHTBLICK_PORT
    rerun_recording_arg = "/opt/npa-agent/sim2real.rrd " if preload_stock_demo else ""
    lichtblick_image = str(
        os.environ.get("NPA_AGENT_LICHTBLICK_IMAGE", "").strip()
        or "npa-lichtblick:1.26.0"
    )
    # Region-agnostic image acquisition uses the anonymous public release. An
    # explicit customer registry override remains available through deploy.images.
    lichtblick_pull_candidates = " ".join(
        shlex.quote(ref)
        for ref in container_image_candidates("lichtblick", preferred_region=region)
    )
    setup_script = f"""set -euo pipefail
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y nginx apache2-utils python3-venv python3-pip curl unzip ca-certificates coturn
if ! command -v nebius >/dev/null 2>&1; then
  curl -fsSL https://storage.eu-north1.nebius.cloud/cli/install.sh | bash
fi
if ! grep -q 'export PATH="$HOME/.nebius/bin:$PATH"' "$HOME/.profile" 2>/dev/null; then
  echo 'export PATH="$HOME/.nebius/bin:$PATH"' >> "$HOME/.profile"
fi
NEBIUS_BIN="$(command -v nebius || true)"
if [ -z "$NEBIUS_BIN" ] && [ -x "$HOME/.nebius/bin/nebius" ]; then
  NEBIUS_BIN="$HOME/.nebius/bin/nebius"
fi
if [ -z "$NEBIUS_BIN" ] || [ ! -x "$NEBIUS_BIN" ]; then
  echo "nebius CLI binary not found after install" >&2
  exit 1
fi
if ! command -v terraform >/dev/null 2>&1; then
  tmp_tf="$(mktemp -d)"
  curl -fsSL -o "$tmp_tf/terraform.zip" https://releases.hashicorp.com/terraform/1.13.3/terraform_1.13.3_linux_amd64.zip
  (cd "$tmp_tf" && unzip -q terraform.zip)
  sudo install -m 0755 "$tmp_tf/terraform" /usr/local/bin/terraform
  rm -rf "$tmp_tf"
fi
if ! command -v kubectl >/dev/null 2>&1; then
  kubectl_version="$(curl -fsSL https://dl.k8s.io/release/stable.txt)"
  curl -fsSL -o /tmp/kubectl "https://dl.k8s.io/release/$kubectl_version/bin/linux/amd64/kubectl"
  sudo install -m 0755 /tmp/kubectl /usr/local/bin/kubectl
  rm -f /tmp/kubectl
fi
if [ -s /mnt/cloud-metadata/token ]; then
  if ! "$NEBIUS_BIN" profile create --endpoint api.eu.nebius.cloud --token-file /mnt/cloud-metadata/token --profile {nebius_profile} --parent-id {nebius_parent_id} >/dev/null 2>&1; then
    "$NEBIUS_BIN" --profile {nebius_profile} iam get-access-token >/dev/null
  fi
  # The backend systemd service runs as root. Provision the same rotating
  # metadata profile under root's CLI home so tenant inventory does not fail
  # merely because bootstrap itself ran through the SSH user's home.
  if ! sudo -H "$NEBIUS_BIN" profile create --endpoint api.eu.nebius.cloud --token-file /mnt/cloud-metadata/token --profile {nebius_profile} --parent-id {nebius_parent_id} >/dev/null 2>&1; then
    sudo -H "$NEBIUS_BIN" --profile {nebius_profile} iam get-access-token >/dev/null
  fi
  # Inventory must use the exact attached identity and its rotating metadata
  # token. Scrub any operator/bootstrap token inherited by SSH before verifying.
  expected_sa={expected_agent_service_account_id}
  expected_tenant={expected_agent_tenant_id}
  expected_project={nebius_parent_id}
  inventory_env=(env -u NEBIUS_IAM_TOKEN -u NPA_NEBIUS_IAM_TOKEN -u TF_VAR_iam_token -u NPA_REUSE_IAM_TOKEN HOME=/root NEBIUS_PROFILE={nebius_profile})
  whoami_json="$(sudo "${{inventory_env[@]}}" "$NEBIUS_BIN" --config /root/.nebius/config.yaml --profile {nebius_profile} iam whoami --format json)"
  if [ -n "$expected_sa" ] && ! python3 -c 'import json, sys; expected = sys.argv[1]; matches = lambda value: any(map(matches, value.values())) if isinstance(value, dict) else any(map(matches, value)) if isinstance(value, list) else isinstance(value, str) and value == expected; raise SystemExit(0 if matches(json.load(sys.stdin)) else 1)' "$expected_sa" <<<"$whoami_json"; then
    echo "attached service-account verification failed" >&2
    exit 1
  fi
  if [ -n "$expected_tenant" ]; then
    # Tenant inventory is optional for a deliberately project-scoped agent.
    # If tenant-wide listing is denied, prove access to the exact deployment
    # project instead of forcing a broad tenant editors grant.
    if ! sudo "${{inventory_env[@]}}" "$NEBIUS_BIN" --config /root/.nebius/config.yaml --profile {nebius_profile} iam project list --parent-id "$expected_tenant" --all --format json >/dev/null 2>&1; then
      sudo "${{inventory_env[@]}}" "$NEBIUS_BIN" --config /root/.nebius/config.yaml --profile {nebius_profile} iam project get --id "$expected_project" --format json >/dev/null
    fi
  fi
fi
sudo mkdir -p /opt/npa-agent
printf '%s' {shlex.quote(deployment_b64)} | base64 -d | sudo tee /opt/npa-agent/deployment.json >/dev/null
sudo chmod 0644 /opt/npa-agent/deployment.json
cat <<'ENV' | sudo tee /opt/npa-agent/public.env >/dev/null
NPA_AGENT_PUBLIC_URL=https://{host}
NPA_AGENT_PUBLIC_HOST={host}
NPA_AGENT_PRELOAD_STOCK_DEMO={preload_stock_demo_value}
ENV
cat <<'ENV' | sudo tee /opt/npa-agent/foxglove.env >/dev/null
NPA_FOXGLOVE_ENABLED=1
NPA_FOXGLOVE_EMBED_SRC={foxglove_env["embed_src"]}
NPA_FOXGLOVE_VIEWER_BACKEND={foxglove_env["viewer_backend"]}
NPA_FOXGLOVE_ORG_SLUG={foxglove_env["org_slug"]}
NPA_FOXGLOVE_LIVE_URL={foxglove_env["live_url"]}
NPA_FOXGLOVE_CLOUD_IMPORT_TIMEOUT_SECONDS={foxglove_env["cloud_import_timeout_seconds"]}
NPA_FOXGLOVE_SDK_VERSION={foxglove_env["sdk_version"]}
ENV
sudo mkdir -p /opt/npa-agent/foxglove/sdk /opt/npa-agent/foxglove/app /opt/npa-agent/foxglove/data
# Install the pinned, sha512-verified @foxglove/embed browser SDK. Non-fatal: an
# agent VM without egress to the npm registry still deploys, and
# /api/foxglove/config reports exactly why the viewer is unavailable.
if sudo bash {AGENT_SOURCE_ROOT}/npa/docker/workbench/foxglove-embed/install-sdk.sh \\
    --dest /opt/npa-agent/foxglove/sdk \\
    --version {foxglove_env["sdk_version"]} \\
    --integrity {foxglove_env["sdk_integrity"]}; then
  sudo rm -f /opt/npa-agent/foxglove/INSTALL_FAILED
else
  echo "install-sdk.sh failed at $(date -u +%Y-%m-%dT%H:%M:%SZ)" | sudo tee /opt/npa-agent/foxglove/INSTALL_FAILED >/dev/null
fi
sudo cp {AGENT_SOURCE_ROOT}/npa/src/npa/cli/assets/foxglove/npa-foxglove-host.js /opt/npa-agent/foxglove/app/npa-foxglove-host.js
sudo chmod -R a+rX /opt/npa-agent/foxglove
{_AGENT_BACKEND_SHIP}
cat <<'PY' | sudo tee /opt/npa-agent/backend.py >/dev/null
import hashlib
import json
import os
import re
import secrets
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import httpx
import yaml
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

app = FastAPI(title="npa-agent")
DEPLOYMENT = {deployment_json}
TOOL_CATALOG = {catalog_json}
TOOL_REFS = sorted(TOOL_CATALOG.keys())
STATE_PATH = Path("/opt/npa-agent/session_state.json")
RRD_PATH = Path("/opt/npa-agent/sim2real.rrd")
PRELOAD_STOCK_DEMO = str(os.environ.get("NPA_AGENT_PRELOAD_STOCK_DEMO", "1")).strip().lower() not in {"0", "false", "no"}
RECORDING_PATH = Path("/opt/npa-agent/recordings/sim2real.rrd")
RECORDINGS_DIR = Path("/opt/npa-agent/recordings")
FOXGLOVE_ROOT = Path("/opt/npa-agent/foxglove")
FOXGLOVE_SDK_DIR = FOXGLOVE_ROOT / "sdk"
FOXGLOVE_DATA_DIR = FOXGLOVE_ROOT / "data"
# Keep a few published recordings so switching runs back and forth does not
# re-download, but do not let the public data path grow without bound.
FOXGLOVE_KEEP_PUBLISHED = 3

{_AGENT_STATE_EMBED}
from npa.cli.agent_resources import (
    assemble_k8s_backend_inventory,
    build_resource_inventory,
    discover_mk8s_accelerators,
    run_resource_discovery_command,
)
{_AGENT_S3_GUARD_EMBED}

{_AGENT_RRD_PROXY_EMBED}

# Foxglove viewer helpers + routes are SHIPPED modules (see agent_backend/).
from agent_backend.canonical_mcap import CANONICAL_MCAP_DEFAULT_STATE
from agent_backend.canonical_mcap import clear_cross_run_mcap_state
from agent_backend.canonical_mcap import has_rich_visualization_contract
from agent_backend.canonical_mcap import prepare_canonical_mcap
from agent_backend.foxglove import (
    artifact_source_fingerprint,
    convert_run_request,
    describe_foxglove_context,
    foxglove_status_payload,
    is_foxglove_artifact,
    publish_recording,
    resolve_foxglove_config,
    self_hosted_viewer_url,
)
from agent_backend.foxglove_cloud import (
    data_aware_layout_data,
    ensure_layout_from_credentials,
    ensure_recording_and_layout_from_credentials,
)
from agent_backend.foxglove_routes import FoxgloveDeps, register_foxglove_routes
from agent_backend.leisaac import load_manifest_artifact
from agent_backend.leisaac_routes import LeIsaacDeps, register_leisaac_routes


def _leisaac_websocket_connect(*args, **kwargs):
    # Imported only when a browser opens the gated LeIsaac tab. This keeps the
    # ordinary agent backend importable for offline/unit use; the deployed agent
    # venv installs websockets as part of the bootstrap below.
    from websockets.asyncio.client import connect

    return connect(*args, **kwargs)

RERUN_CAPABILITY_NAME_RE = re.compile(r"cap-[A-Za-z0-9_-]{{43}}\\.rrd")
MCAP_RECORDING_PATH = Path("/opt/npa-agent/recordings/sim2real.mcap")
LICHTBLICK_RECORDING_HTTP_PATH = "/lichtblick/recordings/sim2real.mcap"


def _agent_public_origin() -> str:
    # HTTPS origin for Rerun .rrd fetches (must be absolute; path-only URLs break).
    for key in ("NPA_AGENT_PUBLIC_URL", "NPA_AGENT_PUBLIC_ORIGIN"):
        raw = str(os.environ.get(key, "")).strip().rstrip("/")
        if raw.startswith("https://") or raw.startswith("http://"):
            return raw
    host = str(os.environ.get("NPA_AGENT_PUBLIC_HOST", "")).strip()
    if host:
        return f"https://{{host}}"
    return ""


def _rerun_recording_url(recording_path: str = "", *, cache_bust: bool = False) -> str:
    origin = _agent_public_origin()
    path = str(recording_path or "").strip()
    name = path.removeprefix("/rerun/recordings/")
    if not RERUN_CAPABILITY_NAME_RE.fullmatch(name): return ""
    path = f"/rerun/recordings/{{name}}"
    if origin:
        url = f"{{origin}}{{path}}"
    else:
        url = path
    if cache_bust:
        url = f"{{url}}?t={{int(time.time() * 1000)}}"
    return url


def _rerun_iframe_url(camera: str = "workspace", *, live_url: str = "", recording_path: str = "") -> str:
    cam = (camera or "workspace").strip() or "workspace"
    if live_url:
        return f"/rerun/?url={{quote(live_url, safe='')}}&hide_welcome_screen=1&theme=dark&camera={{cam}}"
    recording = _rerun_recording_url(recording_path)
    if not recording: return "/rerun/"
    # Rerun web viewer treats path-only values like `/rerun/...` as host `rerun`.
    return f"/rerun/?url={{quote(recording, safe='')}}&hide_welcome_screen=1&theme=dark&camera={{cam}}"


def _lichtblick_recording_url(*, cache_bust: bool = False) -> str:
    origin = _agent_public_origin()
    path = LICHTBLICK_RECORDING_HTTP_PATH
    url = f"{{origin}}{{path}}" if origin else path
    if cache_bust:
        url = f"{{url}}?t={{int(time.time() * 1000)}}"
    return url


def _lichtblick_iframe_url(
    *,
    mcap_url: str = "",
    mcap_size: int = 0,
    primary_camera: str = "",
    start_time_ns: int = 0,
    end_time_ns: int = 0,
) -> str:
    # Lichtblick opens a remote MCAP the same way the standalone tool does; the MCAP is
    # co-served same-origin under /lichtblick/recordings/ so the browser fetch needs no CORS.
    source = mcap_url or _lichtblick_recording_url()
    url = self_hosted_viewer_url(
        source,
        base="/lichtblick/",
        start_time_ns=start_time_ns,
        end_time_ns=end_time_ns,
    )
    size_hint = max(0, int(mcap_size or 0))
    size_query = f"&npa.size={{size_hint}}" if size_hint else ""
    camera = str(primary_camera or "").strip()
    camera_query = f"&npa.camera={{quote(camera, safe='')}}" if camera else ""
    return f"{{url}}{{size_query}}{{camera_query}}"


def _publish_mcap_recording(source: Path) -> Path:
    source = Path(source)
    if not source.is_file():
        return MCAP_RECORDING_PATH
    MCAP_RECORDING_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = MCAP_RECORDING_PATH.with_suffix(".mcap.tmp")
    shutil.copy2(source, tmp)
    tmp.replace(MCAP_RECORDING_PATH)
    return MCAP_RECORDING_PATH

RERUN_UNIT = "npa-rerun"
RERUN_WEB_PORT = {rerun_port}
LICHTBLICK_WEB_PORT = {lichtblick_port}
AGENT_PYTHON = Path("/opt/npa-agent/venv/bin/python")
DEFAULT_SCENE_SPEC = {{
    "schema": "npa.sim2real.manip_scene_spec.v1",
    "goal_pos": [0.5, 0.3, 0.04],
    "goal_threshold": 0.05,
    "objects": [{{"name": "cube", "asset_source": "primitive", "role": "manipuland", "primitive": "box"}}],
    "cameras": {{
        "workspace": {{
            "name": "workspace",
            "placement": "stock_workspace",
            "pos": [1.0, 0.0, 0.8],
            "look_at": [0.5, 0.0, 0.0],
            "fov": 60.0,
            "resolution": [640, 480],
        }},
        "wrist": {{
            "name": "wrist",
            "placement": "stock_ee_mounted",
            "pos": [0.4, 0.0, 0.4],
            "look_at": [0.5, 0.0, 0.0],
            "fov": 90.0,
            "resolution": [640, 480],
        }},
    }},
}}
DEFAULT_ROBOT_SPEC = {{
    "schema": "npa.sim2real.robot_spec.v1",
    "preset": "franka",
    "robot_source": "stock_franka",
    "name": "franka_panda",
}}
DEFAULT_ASSETS_MANIFEST = {{
    "schema": "npa.sim2real.assets_manifest.v1",
    "scene_status": "stock_tabletop",
    "robot_status": "stock_franka",
}}
DEFAULT_SELECTION = {{
    "scene_spec_uri": "",
    "assets_uri": "",
    "robot_spec_uri": "",
    "cameras_uri": "",
    "robot_preset": "franka",
    "sim_backend": "isaac",
    "props": [],
}}
DEFAULT_SIM_VIZ = {{
    "run_id": "",
    "source_type": "",
    "source_label": "",
    "stage": "idle",
    "rrd_uri": "",
    "rrd_updated_at": "",
    "artifact_uri": "",
    "artifact_key": "",
    "artifact_render": "",
    "artifact_preview_url": "",
    "artifact_download_url": "",
    "live_grpc_url": "",
    "mode": "static",
    "camera": "workspace",
    "rerun_ready": False,
    "rerun_iframe_url": "/rerun/",
    # MCAP recording for the embedded viewers. The same file is exposed twice:
    # mcap_uri is the same-origin path Lichtblick (OSS, in-page) streams, and
    # foxglove_url is the CORS-enabled copy the official Foxglove app fetches
    # cross-origin. mcap_updated_at timestamps both.
    "mcap_uri": "",
    "mcap_updated_at": "",
    "lichtblick_ready": False,
    "lichtblick_iframe_url": "/lichtblick/",
    "foxglove_ready": False,
    "foxglove_url": "",
    "canonical_mcap_s3_uri": "",
    "canonical_mcap_key": "",
    "canonical_mcap_sha256": "",
    "canonical_mcap_size_bytes": 0,
    "canonical_mcap_source": "",
    "canonical_mcap_provenance": {{}},
    "transport_state": "",
    "foxglove_cloud": {{}},
}}
SIM2REAL_STAGE_TEMPLATE = [
    ("submit", "Submit request"),
    ("stage_01_trigger", "1 Trigger"),
    ("stage_02_assets", "2 Assets"),
    ("stage_03_augment", "3 Augment"),
    ("stage_04_envs_raw", "4 Raw envs"),
    ("stage_05_envs_train", "5 Train split"),
    ("stage_06_tokens", "6 Tokens"),
    ("stage_07_actions_train", "7 Policy rollouts"),
    ("stage_08_vlm_eval_train", "8 VLM eval"),
    ("stage_09_training_signal", "9 Training signal"),
    ("stage_10_eval_heldout", "10 Held-out eval"),
    ("stage_11_outer_loop", "11 Threshold gate"),
    ("stage_12_external_validation_stub", "12 External validation"),
    ("stage_13_retrigger", "13 Retrigger"),
    ("stage_14_rerun_viz", "14 Rerun viz"),
]

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _slug(value: str, *, fallback: str = "default") -> str:
    token = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(value or "").strip()).strip("-.")
    return token or fallback

def _state_scope_parts() -> tuple[str, str, str]:
    project_alias = _slug(os.environ.get("NPA_AGENT_PROJECT_ALIAS", "default-project"))
    agent_name = _slug(os.environ.get("NPA_AGENT_NAME", "agent"))
    session_scope = _slug(os.environ.get("NPA_AGENT_SESSION_SCOPE", "default-session"))
    return project_alias, agent_name, session_scope

def _state_s3_settings() -> dict[str, str]:
    return {{
        "bucket": str(os.environ.get("NPA_AGENT_S3_BUCKET", "")).strip(),
        "endpoint": str(os.environ.get("NPA_AGENT_S3_ENDPOINT", "")).strip(),
        "access_key": str(os.environ.get("AWS_ACCESS_KEY_ID", "")).strip(),
        "secret_key": str(os.environ.get("AWS_SECRET_ACCESS_KEY", "")).strip(),
        "region": str(os.environ.get("AWS_REGION", "eu-north1")).strip() or "eu-north1",
        "prefix": str(os.environ.get("NPA_AGENT_STATE_S3_PREFIX", "npa-agent/session-state")).strip().strip("/"),
    }}

def _state_s3_key() -> str:
    settings = _state_s3_settings()
    project_alias, agent_name, session_scope = _state_scope_parts()
    deployment_id = _slug(
        str(DEPLOYMENT.get("deployment_id") or ""),
        fallback=f"{{project_alias}}-{{agent_name}}",
    )
    prefix = settings.get("prefix", "npa-agent/session-state")
    return (
        f"{{prefix}}/{{project_alias}}/{{agent_name}}/deployments/"
        f"{{deployment_id}}/{{session_scope}}.json"
    )

def _state_s3_client():
    settings = _state_s3_settings()
    if not (settings["bucket"] and settings["access_key"] and settings["secret_key"]):
        return None, settings
    try:
        client_kwargs = {{
            "endpoint_url": settings["endpoint"],
            "aws_access_key_id": settings["access_key"],
            "region_name": settings["region"],
        }}
        secret_param = "aws" + "_secret_access_key"
        client_kwargs[secret_param] = settings["secret_key"]
        return build_s3_client(**client_kwargs), settings
    except Exception:
        return None, settings

def _load_state_from_s3() -> dict | None:
    client, settings = _state_s3_client()
    if client is None:
        return None
    try:
        payload = client.get_object(Bucket=settings["bucket"], Key=_state_s3_key())
        body = payload.get("Body")
        if body is None:
            return None
        raw = body.read()
        if isinstance(raw, bytes):
            text = raw.decode("utf-8")
        else:
            text = str(raw)
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except Exception:
        return None

def _save_state_to_s3(state: dict) -> None:
    client, settings = _state_s3_client()
    if client is None:
        return
    try:
        client.put_object(
            Bucket=settings["bucket"],
            Key=_state_s3_key(),
            Body=(json.dumps(state, indent=2, sort_keys=True) + "\\n").encode("utf-8"),
            ContentType="application/json",
        )
    except Exception:
        return

def _default_state() -> dict:
    project_alias, agent_name, session_scope = _state_scope_parts()
    selection = dict(DEFAULT_SELECTION)
    if not PRELOAD_STOCK_DEMO:
        selection.update({{"robot_preset": "", "sim_backend": ""}})
    return {{
        "selection": selection,
        "camera_selection": ["workspace"],
        "sim_viz": dict(DEFAULT_SIM_VIZ),
        "sim_viz_runs": {{}},
        "sim2real_runs": {{}},
        "active_run_id": "",
        "latest_submit": {{}},
        "workflow_draft": {{"yaml": "", "name": "", "states": [], "updated_at": "", "plan": {{}}, "runnable": False}},
        "workflow_submit": {{}},
        "gpu_allocation_fallback": {{}},
        "chat_history": [],
        "active_chat_session_id": "default",
        "chat_sessions": {{}},
        "session_scope": session_scope,
        "agent_scope": {{"project_alias": project_alias, "name": agent_name}},
        "deployment_id": str(DEPLOYMENT.get("deployment_id") or ""),
        "state_version": 3,
    }}

_STATE_LOCK = threading.RLock()
_ARTIFACT_LOAD_LOCK = threading.RLock()
_STATE_STORE: StateStore | None = None


def _get_state_store() -> StateStore:
    global _STATE_STORE
    if _STATE_STORE is None:
        _STATE_STORE = StateStore(
            STATE_PATH,
            default_factory=_default_state,
            after_save=_save_state_to_s3,
        )
    return _STATE_STORE


def _normalize_loaded_state(data: dict | None) -> dict:
    if not isinstance(data, dict):
        return _default_state()
    expected_deployment_id = str(DEPLOYMENT.get("deployment_id") or "")
    if str(data.get("deployment_id") or "") != expected_deployment_id:
        # Local files and legacy S3 keys are mutable deployment state. Never
        # hydrate them into another deployment, even if alias/name were reused.
        return _default_state()
    merged = _default_state()
    merged.update(data)
    if not isinstance(merged.get("selection"), dict):
        merged["selection"] = dict(DEFAULT_SELECTION)
    if not isinstance(merged.get("camera_selection"), list):
        merged["camera_selection"] = ["workspace"]
    if not isinstance(merged.get("sim_viz"), dict):
        merged["sim_viz"] = dict(DEFAULT_SIM_VIZ)
    if not isinstance(merged.get("sim_viz_runs"), dict):
        merged["sim_viz_runs"] = {{}}
    if not isinstance(merged.get("sim2real_runs"), dict):
        merged["sim2real_runs"] = {{}}
    if not isinstance(merged.get("active_run_id"), str):
        merged["active_run_id"] = ""
    if not isinstance(merged.get("chat_history"), list):
        merged["chat_history"] = []
    if not isinstance(merged.get("chat_sessions"), dict):
        merged["chat_sessions"] = {{}}
    if not isinstance(merged.get("gpu_allocation_fallback"), dict):
        merged["gpu_allocation_fallback"] = {{}}
    if not isinstance(merged.get("active_chat_session_id"), str):
        merged["active_chat_session_id"] = "default"
    if not PRELOAD_STOCK_DEMO:
        # Artifact-only workspaces preserve source-qualified S3 selections and
        # discard stock/synthetic verifier state on every restart.
        merged["sim_viz_runs"] = {{
            key: value
            for key, value in merged["sim_viz_runs"].items()
            if isinstance(value, dict)
            and str(value.get("artifact_uri") or "").startswith("s3://")
        }}
        if not str(merged["sim_viz"].get("artifact_uri") or "").startswith("s3://"):
            merged["sim_viz"] = dict(DEFAULT_SIM_VIZ)
        if str(merged.get("active_run_id") or "") not in merged["sim_viz_runs"]:
            merged["active_run_id"] = ""
        clean = _default_state()
        for key in (
            "selection",
            "sim2real_runs",
            "latest_submit",
            "workflow_draft",
            "workflow_submit",
        ):
            merged[key] = clean[key]
    return merged


def _load_state_unlocked() -> dict:
    # Caller must hold _STATE_LOCK / store.lock for read-modify-write.
    store = _get_state_store()
    data = store.load()
    if not isinstance(data, dict) or not data:
        data = _load_state_from_s3()
    return _normalize_loaded_state(data)


def _save_state_unlocked(state: dict) -> None:
    # Caller must hold _STATE_LOCK / store.lock.
    state["updated_at"] = _now_iso()
    state["deployment_id"] = str(DEPLOYMENT.get("deployment_id") or "")
    state["state_version"] = int(state.get("state_version") or 3)
    _get_state_store().save(state)


def _load_state() -> dict:
    # Process-wide lock so concurrent Starlette threadpool handlers cannot
    # clobber confirm tokens / chat history / sim-viz selection.
    with _STATE_LOCK:
        return _load_state_unlocked()


def _save_state(state: dict) -> None:
    with _STATE_LOCK:
        # LeIsaac bundle/run selection is written through _mutate_state.  Older
        # handlers elsewhere in this backend still perform load -> work -> save;
        # preserve the latest atomic namespace when one of those handlers
        # finishes with a stale snapshot after a simulator restart.
        latest = _load_state_unlocked()
        preserved = preserve_latest_namespaces(state, latest, ("leisaac",))
        state.clear()
        state.update(preserved)
        _save_state_unlocked(state)


def _mutate_state(fn):
    # Atomic load → mutate → save under the process-wide lock.
    with _STATE_LOCK:
        state = _load_state_unlocked()
        result = fn(state)
        _save_state_unlocked(state)
        return result


def _record_sim_viz_run(state: dict, payload: dict | None) -> None:
    if not isinstance(payload, dict):
        return
    run_id = str(payload.get("run_id") or "").strip()
    if not run_id:
        return
    runs = state.get("sim_viz_runs")
    if not isinstance(runs, dict):
        runs = {{}}
    run_ref = str(payload.get("artifact_run_ref") or "").strip()
    history_key = run_ref or run_id
    existing = runs.get(history_key) if isinstance(runs.get(history_key), dict) else {{}}
    snapshot = dict(DEFAULT_SIM_VIZ)
    if isinstance(existing, dict):
        snapshot.update(existing)
    snapshot.update(payload)
    snapshot["run_id"] = run_id
    snapshot["source_type"], snapshot["source_label"] = resolve_run_source(payload, existing, run_id)
    incoming_rrd = bool(str(payload.get("rrd_uri") or "").strip())
    incoming_render = str(payload.get("artifact_render") or "").strip().lower()
    # A Rerun/demo update must not resurrect a prior video/image/json media preview.
    if incoming_rrd and incoming_render in {{"", "rerun"}}:
        if str(existing.get("artifact_render") or "").strip().lower() not in {{"", "rerun"}}:
            snapshot["artifact_render"] = "rerun"
            for key in (
                "artifact_key",
                "artifact_uri",
                "artifact_preview_url",
                "artifact_download_url",
                "visualization_note",
                "foxglove_url",
            ):
                if key not in payload or not str(payload.get(key) or "").strip():
                    snapshot[key] = ""
            snapshot["foxglove_ready"] = bool(payload.get("foxglove_ready"))
    else:
        no_preview = str(payload.get("preview_status") or "").strip() == "no_previewable_recording"
        # Never let a sparse update erase richer artifact fields from load-run,
        # except when an explicit run selection establishes an honest no-preview
        # state and therefore must clear a stale artifact from the prior run.
        if not no_preview:
            for key in (
                "artifact_render",
                "artifact_key",
                "artifact_uri",
                "artifact_preview_url",
                "artifact_download_url",
                "rrd_uri",
                "rerun_iframe_url",
                "visualization_note",
                "preview_entity",
                "foxglove_url",
                "mcap_updated_at",
            ):
                if not str(snapshot.get(key) or "").strip() and str(existing.get(key) or "").strip():
                    snapshot[key] = existing[key]
        if not payload.get("foxglove_ready") and existing.get("foxglove_ready") and str(snapshot.get("foxglove_url") or "").strip():
            snapshot["foxglove_ready"] = True
    runs[history_key] = snapshot
    state["sim_viz_runs"] = runs
    state["active_run_id"] = run_id
    state["active_run_ref"] = run_ref


def _default_sim2real_run_details(run_id: str, *, submitted_at: str = "", selection: dict | None = None) -> dict:
    stages = []
    for stage_id, label in SIM2REAL_STAGE_TEMPLATE:
        stages.append(
            stage_evidence_record(
                stage_id=stage_id,
                label=label,
                status="not_run",
                status_label="Not run",
                raw_status="not_run",
                evidence_type="workflow_status",
                evidence_source="agent_sim2real_submit_record",
                authority="authoritative",
                confidence="high",
                reason="The agent submit record explicitly says this stage was not launched.",
                summary="Not launched by the agent UI submit endpoint.",
            )
        )
    if stages:
        submit_stage_id, submit_stage_label = SIM2REAL_STAGE_TEMPLATE[0]
        stages[0] = stage_evidence_record(
            stage_id=submit_stage_id,
            label=submit_stage_label,
            status="succeeded",
            status_label="Succeeded",
            raw_status="accepted",
            evidence_type="event_log",
            evidence_source="agent_sim2real_submit_record",
            authority="authoritative",
            confidence="high",
            reason="The agent accepted and durably recorded this Sim2Real submit request.",
            started_at=submitted_at,
            finished_at=submitted_at,
            observed_at=submitted_at,
            summary="Agent accepted the Sim2Real submit request.",
        )
    return {{
        "run_id": run_id,
        "source_type": "workflow_history",
        "source_label": "Workflow history",
        "workflow_name": "sim2real-staged-loop",
        "status": "submitted",
        "result": "recorded_not_launched",
        "submitted_at": submitted_at,
        "updated_at": submitted_at or _now_iso(),
        "selection": selection if isinstance(selection, dict) else {{}},
        "stages": stages,
        "stage_summary": summarize_stage_evidence(stages),
        "logs": [
            {{
                "timestamp": submitted_at or _now_iso(),
                "level": "info",
                "message": "Sim2Real submit recorded by NPA agent.",
            }},
            {{
                "timestamp": submitted_at or _now_iso(),
                "level": "warn",
                "message": "The agent UI submit endpoint recorded the request but did not launch the full K8s Sim2Real pipeline; unexecuted stages are marked not_run.",
            }},
            {{
                "timestamp": submitted_at or _now_iso(),
                "level": "info",
                "message": "Use the operator workflow submit path for a real staged K8s run; this view remains truthful until run artifacts or a recording exist.",
            }},
        ],
        "artifacts": [],
    }}


def _merge_sim2real_run_details(base: dict, update: dict | None) -> dict:
    merged = dict(base)
    if isinstance(update, dict):
        for key, value in update.items():
            if key == "stages" and isinstance(value, list):
                merged[key] = value
            elif key == "logs" and isinstance(value, list):
                merged[key] = value
            elif key == "selection" and isinstance(value, dict):
                selection = dict(merged.get("selection", {{}}) if isinstance(merged.get("selection"), dict) else {{}})
                selection.update(value)
                merged[key] = selection
            else:
                merged[key] = value
    return merged


def _sim_viz_for_run(state: dict, run_id: str = "") -> dict:
    payload = dict(DEFAULT_SIM_VIZ)
    runs = state.get("sim_viz_runs")
    target = str(run_id or state.get("active_run_id") or "").strip()
    direct = runs.get(target) if isinstance(runs, dict) and target else None
    if isinstance(direct, dict):
        payload.update(direct)
    elif isinstance(runs, dict) and target:
        matches = [
            item
            for item in runs.values()
            if isinstance(item, dict)
            and str(item.get("run_id") or "").strip() == target
        ]
        # A plain basename is safe only when it names one historical source.
        if len(matches) == 1:
            payload.update(matches[0])
        elif run_id:
            payload["run_id"] = target
    elif run_id:
        payload["run_id"] = target
    else:
        current = state.get("sim_viz")
        if isinstance(current, dict):
            payload.update(current)
    return payload

def _stock_franka_selection() -> dict:
    return {{
        "scene_spec_uri": "stock://scene/default",
        "assets_uri": "",
        "robot_spec_uri": "stock://robot/franka",
        "cameras_uri": "stock://cameras/default",
        "robot_preset": "franka",
        "sim_backend": "isaac",
        "props": ["cube"],
    }}

def _camera_frustum_lines(pos: list[float], look_at: list[float], fov: float, *, depth: float = 0.35):
    import math

    px, py, pz = (float(pos[0]), float(pos[1]), float(pos[2]))
    lx, ly, lz = (float(look_at[0]), float(look_at[1]), float(look_at[2]))
    fx, fy, fz = (lx - px, ly - py, lz - pz)
    norm = math.sqrt(fx * fx + fy * fy + fz * fz) or 1.0
    fx, fy, fz = (fx / norm, fy / norm, fz / norm)
    upx, upy, upz = (0.0, 0.0, 1.0)
    rx = fy * upz - fz * upy
    ry = fz * upx - fx * upz
    rz = fx * upy - fy * upx
    rnorm = math.sqrt(rx * rx + ry * ry + rz * rz) or 1.0
    rx, ry, rz = (rx / rnorm, ry / rnorm, rz / rnorm)
    ux = ry * fz - rz * fy
    uy = rz * fx - rx * fz
    uz = rx * fy - ry * fx
    half_h = depth * math.tan(math.radians(float(fov) / 2.0))
    half_w = half_h * (4.0 / 3.0)
    cx = px + fx * depth
    cy = py + fy * depth
    cz = pz + fz * depth
    corners = []
    for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        corners.append(
            [
                cx + rx * half_w * sx + ux * half_h * sy,
                cy + ry * half_w * sx + uy * half_h * sy,
                cz + rz * half_w * sx + uz * half_h * sy,
            ]
        )
    origin = [px, py, pz]
    strips = [
        [origin, corners[0]],
        [origin, corners[1]],
        [origin, corners[2]],
        [origin, corners[3]],
        corners + [corners[0]],
    ]
    return origin, strips

_FRANKA_HOME_JOINTS = (0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785)

def _franka_joint_positions(joint_angles: tuple[float, ...]) -> list[list[float]]:
    import math

    dh = [
        (0.0, 0.0, 0.333),
        (0.0, -math.pi / 2.0, 0.0),
        (0.0, math.pi / 2.0, 0.316),
        (0.0825, math.pi / 2.0, 0.0),
        (-0.0825, -math.pi / 2.0, 0.384),
        (0.0, math.pi / 2.0, 0.0),
        (0.088, math.pi / 2.0, 0.0),
    ]

    def _matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
        return [
            [sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)
        ]

    transform = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    positions = [[0.0, 0.0, 0.0]]
    for index, (a, alpha, d) in enumerate(dh):
        theta = float(joint_angles[index])
        ct, st = math.cos(theta), math.sin(theta)
        ca, sa = math.cos(alpha), math.sin(alpha)
        step = [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ]
        transform = _matmul(transform, step)
        positions.append([transform[0][3], transform[1][3], transform[2][3]])
    ee = [transform[0][3], transform[1][3], transform[2][3] + 0.103]
    positions.append(ee)
    positions.append([ee[0], ee[1] + 0.04, ee[2]])
    positions.append([ee[0], ee[1] - 0.04, ee[2]])
    return positions

def _franka_demo_joint_angles(frame_index: int, frame_count: int) -> tuple[float, ...]:
    import math

    phase = (float(frame_index) / max(1.0, float(frame_count - 1))) * math.tau
    return (
        _FRANKA_HOME_JOINTS[0] + 0.22 * math.sin(phase),
        _FRANKA_HOME_JOINTS[1] + 0.16 * math.sin(phase + 0.5),
        _FRANKA_HOME_JOINTS[2] + 0.18 * math.sin(phase + 1.2),
        _FRANKA_HOME_JOINTS[3] + 0.12 * math.sin(phase + 1.7),
        _FRANKA_HOME_JOINTS[4] + 0.24 * math.sin(phase + 2.1),
        _FRANKA_HOME_JOINTS[5] + 0.10 * math.sin(phase + 2.7),
        _FRANKA_HOME_JOINTS[6] + 0.20 * math.sin(phase + 3.4),
    )


def _set_rerun_time(rr, seconds: float) -> None:
    if hasattr(rr, "set_time_seconds"):
        rr.set_time_seconds("log_time", seconds)
    else:
        rr.set_time("log_time", duration=seconds)


def _log_franka_robot_geometry(rr, joint_angles: tuple[float, ...] = _FRANKA_HOME_JOINTS) -> None:
    positions = _franka_joint_positions(joint_angles)
    arm_points = positions[:8]
    segments: list[list[list[float]]] = []
    for left, right in zip(arm_points, arm_points[1:]):
        dx = left[0] - right[0]
        dy = left[1] - right[1]
        dz = left[2] - right[2]
        if dx * dx + dy * dy + dz * dz < 1e-8:
            continue
        segments.append([left, right])
    link_color = [234, 88, 12]
    link_rgba = link_color + [255]
    rr.log(
        "robot/franka/base",
        rr.Boxes3D(
            centers=[[0.0, 0.0, 0.05]],
            half_sizes=[[0.085, 0.085, 0.05]],
            colors=[[100, 116, 139, 255]],
        ),
    )
    rr.log(
        "robot/franka/joints",
        rr.Points3D(
            arm_points,
            colors=[link_rgba] * len(arm_points),
            radii=[0.028] * len(arm_points),
        ),
    )
    if segments:
        rr.log(
            "robot/franka/links",
            rr.LineStrips3D(
                segments,
                colors=[link_color] * len(segments),
                radii=[0.018] * len(segments),
            ),
        )
    gripper_segments = [
        [positions[7], positions[8]],
        [positions[8], positions[9]],
        [positions[8], positions[10]],
    ]
    gripper_color = [59, 130, 246]
    rr.log(
        "robot/franka/gripper",
        rr.LineStrips3D(
            gripper_segments,
            colors=[gripper_color] * len(gripper_segments),
            radii=[0.012] * len(gripper_segments),
        ),
    )
    rr.log(
        "robot/franka",
        rr.TextDocument(
            "Franka Panda — stock tabletop pick-and-place demo (NPA agent preview)"
        ),
    )

def _generate_franka_demo_rrd(*, camera: str = "workspace") -> Path:
    import math

    import rerun as rr

    target = RRD_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    rr.init("npa-franka-tabletop-demo", spawn=False)
    rr.log(
        "agent/camera_inspector",
        rr.TextDocument(
            "Stock Franka tabletop demo with workspace and wrist camera frustums. "
            "Highlighted camera is selected for the next rollout."
        ),
    )
    rr.log(
        "world/table",
        rr.Boxes3D(
            centers=[[0.5, 0.0, 0.0]],
            half_sizes=[[0.4, 0.3, 0.02]],
            colors=[[180, 180, 180, 255]],
        ),
    )
    frame_count = 90
    for frame_index in range(frame_count):
        seconds = frame_index / 15.0
        _set_rerun_time(rr, seconds)
        phase = frame_index / max(1.0, float(frame_count - 1))
        cube_y = 0.3 - 0.42 * phase
        rr.log(
            "world/cube",
            rr.Boxes3D(
                centers=[[0.5, cube_y, 0.04]],
                half_sizes=[[0.025, 0.025, 0.025]],
                colors=[[59, 130, 246, 255]],
            ),
        )
        _log_franka_robot_geometry(rr, _franka_demo_joint_angles(frame_index, frame_count))
    cameras = DEFAULT_SCENE_SPEC.get("cameras", {{}})
    active = camera if camera in cameras else "workspace"
    for name, cam in cameras.items():
        if not isinstance(cam, dict):
            continue
        pos = list(cam.get("pos") or [0.0, 0.0, 0.0])
        look_at = list(cam.get("look_at") or [0.0, 1.0, 0.0])
        fov = float(cam.get("fov") or 60.0)
        res = cam.get("resolution") or [640, 480]
        width = int(res[0]) if len(res) > 0 else 640
        height = int(res[1]) if len(res) > 1 else 480
        entity = f"world/cameras/{{name}}"
        frustum_entity = f"world/camera_frustums/{{name}}"
        focal = width / (2.0 * math.tan(math.radians(fov / 2.0)))
        rr.log(entity, rr.Pinhole(focal_length=focal, width=width, height=height))
        rr.log(entity, rr.Transform3D(translation=pos))
        origin, strips = _camera_frustum_lines(pos, look_at, fov)
        color = [59, 130, 246] if name == active else [148, 163, 184]
        rr.log(
            f"{{frustum_entity}}/frustum",
            rr.LineStrips3D(strips, colors=[color] * len(strips)),
        )
        rr.log(f"{{frustum_entity}}/origin", rr.Points3D([origin], colors=[color], radii=[0.02]))
        label = (
            f"**{{name}}** (selected for next rollout)"
            if name == active
            else f"**{{name}}**"
        )
        rr.log(
            f"{{frustum_entity}}/label",
            rr.TextDocument(
                f"{{label}}\\n"
                f"pos={{pos}} look_at={{look_at}} fov={{fov}}° resolution={{width}}x{{height}}"
            ),
        )
        rr.log(
            f"rollouts/latest/{{name}}/camera",
            rr.TextDocument(
                f"Sim2Real rollout camera stream for `{{name}}` "
                f"(populated when a run writes frames to the recording)."
            ),
        )
    rr.log("demo/active_camera", rr.TextLog(active))
    rr.save(str(target))
    return target

def _publish_rrd_recording(source: Path) -> str:
    if not source.is_file(): return ""
    RECORDING_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = RECORDING_PATH.with_name(f"{{RECORDING_PATH.name}}.{{secrets.token_hex(6)}}.tmp")
    try:
        shutil.copy2(source, tmp)
        tmp.replace(RECORDING_PATH)
    finally:
        # replace() removes the temporary source on success. On any copy/replace
        # failure, remove only the randomized temporary path, never the published
        # destination.
        if tmp != RECORDING_PATH:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
    capability_name = f"cap-{{secrets.token_urlsafe(32)}}.rrd"
    if not RERUN_CAPABILITY_NAME_RE.fullmatch(capability_name): raise RuntimeError("generated Rerun capability filename is invalid")
    capability_path = RECORDINGS_DIR / capability_name
    capability_tmp = capability_path.with_name(
        f"{{capability_path.name}}.{{secrets.token_hex(6)}}.tmp"
    )
    try:
        shutil.copy2(RECORDING_PATH, capability_tmp)
        capability_tmp.replace(capability_path)
    finally:
        try:
            capability_tmp.unlink(missing_ok=True)
        except OSError:
            pass
    for stale in RECORDINGS_DIR.glob("cap-*.rrd"):
        if stale != capability_path and RERUN_CAPABILITY_NAME_RE.fullmatch(stale.name):
            try:
                stale.unlink(missing_ok=True)
            except OSError:
                pass
    return f"/rerun/recordings/{{capability_name}}"


def _safe_artifact_key(key: str) -> str:
    value = str(key or "").strip()
    if not value:
        raise HTTPException(status_code=400, detail="artifact key is required")
    variants = [value]
    for _attempt in range(2):
        decoded = unquote(variants[-1])
        if decoded == variants[-1]:
            break
        variants.append(decoded)
    for candidate in variants:
        if (
            candidate.startswith("/")
            or candidate.endswith("/")
            or "\\\\" in candidate
            or any(ord(ch) < 32 for ch in candidate)
            or any(part in {{"", ".", ".."}} for part in candidate.split("/"))
        ):
            raise HTTPException(status_code=400, detail="artifact key traversal is not allowed")
    return value


def _agent_s3_settings() -> dict[str, str]:
    return {{
        "bucket": str(os.environ.get("NPA_AGENT_S3_BUCKET", "")).strip(),
        "prefix": str(os.environ.get("NPA_AGENT_S3_PREFIX", "")).strip().strip("/"),
        "endpoint": str(os.environ.get("NPA_AGENT_S3_ENDPOINT", "")).strip(),
        "access_key": str(os.environ.get("AWS_ACCESS_KEY_ID", "")).strip(),
        "secret_key": str(os.environ.get("AWS_SECRET_ACCESS_KEY", "")).strip(),
        "region": str(os.environ.get("AWS_REGION", "eu-north1")).strip() or "eu-north1",
    }}


def _join_agent_s3_prefix(base_prefix: str, suffix: str = "") -> str:
    return "/".join(part.strip("/") for part in (base_prefix, suffix) if str(part or "").strip().strip("/"))


def _agent_insights_settings() -> dict[str, str]:
    # Resolve the insights backbone endpoint + append-only store URI from
    # config/env only (never hardcode endpoint, token, bucket, or store URI). A
    # configured NPA_INSIGHTS_ENDPOINT selects service mode; otherwise the tools
    # read the store directly. When no explicit store URI is set we derive one
    # from the agent's own S3 settings so a co-located store works out of the box.
    endpoint = str(os.environ.get("NPA_INSIGHTS_ENDPOINT", "")).strip().rstrip("/")
    store_uri = str(os.environ.get("NPA_INSIGHTS_STORE_URI", "")).strip()
    if not store_uri:
        s3 = _agent_s3_settings()
        bucket = str(s3.get("bucket") or "").strip()
        if bucket:
            prefix = _join_agent_s3_prefix(str(s3.get("prefix") or ""), "insights/store")
            store_uri = f"s3://{{bucket}}/{{prefix}}"
    return {{
        "endpoint": endpoint,
        "store_uri": store_uri,
        "token_env": str(os.environ.get("NPA_INSIGHTS_TOKEN_ENV", "INSIGHTS_TOKEN")).strip() or "INSIGHTS_TOKEN",
    }}


def _artifact_discovery_prefix(settings: dict[str, str], user_prefix: str = "") -> str:
    try:
        requested = _validate_source_prefix(str(user_prefix or ""))
        base = _validate_source_prefix(str(settings.get("prefix") or ""))
    except ArtifactDiscoveryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if requested:
        return _join_agent_s3_prefix(base, requested)
    return base

def _discovery_exclude_roots() -> set:
    # Exact prefix subtrees that hold agent state/chat memory, not runs. Preserve
    # nested prefixes so excluding one agent subtree does not hide unrelated data
    # under the same first path segment.
    roots = set()
    for prefix in (
        str(_state_s3_settings().get("prefix") or ""),
        _chat_memory_prefix(),
    ):
        normalized = str(prefix or "").strip().strip("/")
        if normalized:
            roots.add(normalized)
    return roots


def _agent_s3_client():
    settings = _agent_s3_settings()
    if not settings["bucket"] or not settings["access_key"] or not settings["secret_key"]:
        raise HTTPException(
            status_code=400,
            detail="S3 discovery is not configured on this agent (missing bucket or credentials).",
        )
    try:
        client_kwargs = {{
            "endpoint_url": settings["endpoint"],
            "aws_access_key_id": settings["access_key"],
            "region_name": settings["region"],
        }}
        secret_param = "aws" + "_secret_access_key"
        client_kwargs[secret_param] = settings["secret_key"]
        client = build_s3_client(**client_kwargs)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"failed to initialize S3 client: {{exc}}") from exc
    return client, settings


{_AGENT_ACCESS_RUNTIME_EMBED}


def _chat_memory_tenant() -> str:
    raw = (
        os.environ.get("NEBIUS_TENANT_ID", "")
        or os.environ.get("NEBIUS_PROJECT_ID", "")
        or "default-tenant"
    )
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(raw).strip()).strip("-")
    return value or "default-tenant"


def _chat_memory_prefix(settings: dict[str, str] | None = None) -> str:
    tenant = _chat_memory_tenant()
    project_alias, agent_name, _session_scope = _state_scope_parts()
    fallback = _slug(f"{{project_alias}}-{{agent_name}}", fallback="agent")
    deployment_id = _slug(str(DEPLOYMENT.get("deployment_id") or ""), fallback=fallback)
    return f"npa-agent/tenants/{{tenant}}/deployments/{{deployment_id}}/chat-sessions"


def _chat_memory_uri_matches_deployment(memory_uri: str) -> bool:
    value = str(memory_uri or "").strip()
    if not value:
        return True
    remainder = value.partition("://")[2]
    key = remainder.partition("/")[2].strip("/")
    expected = _chat_memory_prefix().strip("/") + "/"
    return bool(key) and key.startswith(expected)


def _chat_session_key(session_id: str, settings: dict[str, str] | None = None) -> str:
    safe = _sanitize_chat_session_id(session_id)
    return f"{{_chat_memory_prefix(settings)}}/{{safe}}.json"


def _chat_memory_uri(session_id: str, settings: dict[str, str] | None = None) -> str:
    resolved = settings or _agent_s3_settings()
    bucket = str(resolved.get("bucket") or "")
    if not bucket:
        return ""
    return f"s3://{{bucket}}/{{_chat_session_key(session_id, resolved)}}"


def _agent_s3_client_optional():
    try:
        return _agent_s3_client()
    except HTTPException:
        return None, _agent_s3_settings()
    except Exception:
        return None, _agent_s3_settings()


def _sanitize_chat_session_id(value: str) -> str:
    session_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "").strip()).strip("-")
    return session_id[:80] or "default"


def _chat_session_title(messages: list[dict] | None, fallback: str = "New chat") -> str:
    if isinstance(messages, list):
        for item in messages:
            if not isinstance(item, dict):
                continue
            if str(item.get("role") or "") != "user":
                continue
            content = str(item.get("content") or "").strip()
            if content:
                return content[:64]
    return fallback


def _normalize_chat_history(raw: object) -> list[dict]:
    # Persist text stubs only — never store screenshot data-URLs in session history.
    return normalize_messages_for_storage(raw)


_PLACEHOLDER_CHAT_TITLES = frozenset({{"", "new chat", "new chat session"}})


def _is_placeholder_chat_title(title: str) -> bool:
    return str(title or "").strip().lower() in _PLACEHOLDER_CHAT_TITLES


def _normalize_chat_session(session_id: str, payload: object | None = None) -> dict:
    now = _now_iso()
    data = payload if isinstance(payload, dict) else {{}}
    resolved_id = _sanitize_chat_session_id(str(data.get("id") or session_id or "default"))
    history = _normalize_chat_history(data.get("chat_history") or data.get("messages") or [])
    raw_title = str(data.get("title") or "").strip()
    if _is_placeholder_chat_title(raw_title):
        title = _chat_session_title(history, "New chat")
    else:
        title = raw_title or _chat_session_title(history, "New chat")
    created_at = str(data.get("created_at") or now)
    updated_at = str(data.get("updated_at") or now)
    return {{
        "id": resolved_id,
        "title": title[:96],
        "created_at": created_at,
        "updated_at": updated_at,
        "chat_history": history,
        "memory_uri": str(data.get("memory_uri") or _chat_memory_uri(resolved_id) or ""),
    }}


def _local_chat_sessions(state: dict) -> dict[str, dict]:
    sessions = state.get("chat_sessions")
    if not isinstance(sessions, dict):
        sessions = {{}}
    normalized: dict[str, dict] = {{}}
    discarded_foreign_memory = False
    for session_id, payload in sessions.items():
        memory_uri = str(payload.get("memory_uri") or "") if isinstance(payload, dict) else ""
        if memory_uri and not _chat_memory_uri_matches_deployment(memory_uri):
            # Legacy chat memory was tenant-wide, so one agent could hydrate a
            # different deployment's history. Never migrate a source-qualified
            # session across deployment namespaces.
            discarded_foreign_memory = True
            continue
        session = _normalize_chat_session(str(session_id), payload)
        normalized[session["id"]] = session
    if not normalized:
        migrated = _normalize_chat_session(
            "default",
            {{
                "id": "default",
                "title": "Default chat",
                "chat_history": [] if discarded_foreign_memory else state.get("chat_history", []),
            }},
        )
        normalized["default"] = migrated
    state["chat_sessions"] = normalized
    if str(state.get("active_chat_session_id") or "") not in normalized:
        state["active_chat_session_id"] = next(iter(normalized.keys()))
    state["chat_history"] = normalized[str(state["active_chat_session_id"])]["chat_history"]
    return normalized


def _load_chat_session_from_s3(session_id: str) -> dict | None:
    s3, settings = _agent_s3_client_optional()
    if s3 is None or not settings.get("bucket"):
        return None
    key = _chat_session_key(session_id, settings)
    try:
        obj = s3.get_object(Bucket=settings["bucket"], Key=key)
        body = obj.get("Body")
        raw = body.read() if hasattr(body, "read") else body
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        payload = json.loads(str(raw or "{{}}"))
        return _normalize_chat_session(session_id, payload)
    except Exception:
        return None


def _persist_chat_session_to_s3(session: dict) -> str:
    s3, settings = _agent_s3_client_optional()
    if s3 is None or not settings.get("bucket"):
        return ""
    session_id = _sanitize_chat_session_id(str(session.get("id") or "default"))
    key = _chat_session_key(session_id, settings)
    memory_uri = _chat_memory_uri(session_id, settings)
    payload = dict(session)
    payload["id"] = session_id
    payload["memory_uri"] = memory_uri
    payload["tenant_id"] = _chat_memory_tenant()
    try:
        s3.put_object(
            Bucket=settings["bucket"],
            Key=key,
            Body=(json.dumps(payload, indent=2, sort_keys=True) + "\\n").encode("utf-8"),
            ContentType="application/json",
        )
        return memory_uri
    except Exception:
        return ""


def _save_chat_session(state: dict, session: dict, *, active: bool = True) -> dict:
    sessions = _local_chat_sessions(state)
    normalized = _normalize_chat_session(str(session.get("id") or "default"), session)
    normalized["updated_at"] = _now_iso()
    memory_uri = _persist_chat_session_to_s3(normalized)
    if memory_uri:
        normalized["memory_uri"] = memory_uri
    sessions[normalized["id"]] = normalized
    state["chat_sessions"] = sessions
    if active:
        state["active_chat_session_id"] = normalized["id"]
        state["chat_history"] = normalized["chat_history"]
    _save_state(state)
    return normalized


def _lookup_chat_session(state: dict, session_id: str = "") -> dict | None:
    # Read-only lookup: never fabricate a placeholder session.
    sessions = _local_chat_sessions(state)
    target = _sanitize_chat_session_id(session_id or str(state.get("active_chat_session_id") or "default"))
    remote = _load_chat_session_from_s3(target)
    if remote is not None:
        sessions[target] = remote
        state["chat_sessions"] = sessions
        return remote
    if target in sessions:
        return sessions[target]
    return None


def _get_chat_session(state: dict, session_id: str = "") -> dict:
    sessions = _local_chat_sessions(state)
    target = _sanitize_chat_session_id(session_id or str(state.get("active_chat_session_id") or "default"))
    found = _lookup_chat_session(state, target)
    if found is not None:
        return found
    session = _normalize_chat_session(target, {{"id": target, "title": "New chat", "chat_history": []}})
    sessions[target] = session
    state["chat_sessions"] = sessions
    return session


def _append_chat_turn(session_id: str, history_base: list, assistant_msg: dict | None = None, *, title_hint: str = "") -> dict:
    # Re-read session under the state lock and append — never overwrite from a
    # stale snapshot taken before a long LLM/tool call (B2).
    with _STATE_LOCK:
        state = _load_state_unlocked()
        session = _get_chat_session(state, session_id)
        current = normalize_messages_for_storage(session.get("chat_history") or [])
        incoming = normalize_messages_for_storage(history_base or [])
        merged = list(current)
        if incoming:
            last_in = incoming[-1]
            tip_match = (
                merged
                and merged[-1].get("role") == last_in.get("role")
                and merged[-1].get("content") == last_in.get("content")
            )
            if not tip_match:
                if len(incoming) > len(merged):
                    merged = list(incoming)
                elif last_in not in merged:
                    merged.append(last_in)
        if assistant_msg and isinstance(assistant_msg, dict) and str(assistant_msg.get("content") or "").strip():
            merged.append({{"role": "assistant", "content": str(assistant_msg.get("content"))}})
        merged = merged[-80:]
        prior_title = str(session.get("title") or "")
        if _is_placeholder_chat_title(prior_title) or not prior_title:
            title = _chat_session_title(merged, title_hint or "New chat")
        else:
            title = prior_title
        session.update({{"id": session_id, "title": title, "chat_history": merged}})
        # _save_chat_session → _save_state re-acquires RLock (safe).
        return _save_chat_session(state, session, active=True)


def _list_chat_sessions(state: dict) -> list[dict]:
    sessions = _local_chat_sessions(state)
    s3, settings = _agent_s3_client_optional()
    if s3 is not None and settings.get("bucket"):
        prefix = _chat_memory_prefix(settings) + "/"
        try:
            resp = s3.list_objects_v2(Bucket=settings["bucket"], Prefix=prefix, MaxKeys=50)
            for item in resp.get("Contents", []) or []:
                key = str(item.get("Key") or "")
                if not key.endswith(".json"):
                    continue
                session_id = _sanitize_chat_session_id(Path(key).stem)
                remote = _load_chat_session_from_s3(session_id)
                if remote is not None:
                    sessions[session_id] = remote
        except Exception:
            pass
    state["chat_sessions"] = sessions
    rows = []
    for session in sessions.values():
        rows.append(public_chat_session_payload(session))
    return sorted(rows, key=lambda item: str(item.get("updated_at") or ""), reverse=True)


def _artifact_filename(key: str) -> str:
    import hashlib

    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
    leaf = Path(key).name or "artifact.bin"
    return f"{{digest}}-{{leaf}}"


def _artifact_preview_url(filename: str) -> str:
    return f"/api/artifacts/file/{{filename}}"


{_AGENT_VIEWER_RUNTIME_EMBED}


def _is_sim2real_pipeline_recording(key: str) -> bool:
    return str(key or "").endswith("/reports/sim2real.rrd")


DATA_FACTORY_APP_ID = "physical-ai-data-factory"


def _is_data_factory_recording(key: str) -> bool:
    # A Physical AI Data Factory run also writes reports/sim2real.rrd, but its
    # entities are input/ + augmented/ + captions/ (no held-out-sim camera), so
    # it needs a different viewer note than the Sim2Real pipeline recording.
    # Match the app id as a path segment (…/physical-ai-data-factory/…) rather
    # than a bare substring so an unrelated prefix that merely contains the
    # phrase is not misclassified.
    return _is_sim2real_pipeline_recording(key) and (DATA_FACTORY_APP_ID + "/") in str(key or "")


def _sim2real_pipeline_camera_label(requested: str = "") -> str:
    value = str(requested or "").strip()
    return value if value and value != "workspace" else "heldout-sim"


def _copy_artifact_preview(local_path: Path, key: str) -> str:
    filename = _artifact_filename(key)
    target = RECORDINGS_DIR / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    if local_path.resolve() != target.resolve():
        shutil.copy2(local_path, target)
    return _artifact_preview_url(filename)


def _publish_foxglove_recording(local_path: Path, key: str) -> str:
    return publish_recording(
        local_path, key, data_dir=FOXGLOVE_DATA_DIR, keep=FOXGLOVE_KEEP_PUBLISHED
    )


def _self_hosted_viewer_healthy() -> bool:
    # The OSS (Lichtblick) sidecar is best-effort on the VM; probe it rather than
    # assuming, so the Foxglove pane only offers a backend that can actually render.
    try:
        import urllib.request

        with urllib.request.urlopen(
            f"http://127.0.0.1:{{LICHTBLICK_WEB_PORT}}/", timeout=2
        ) as resp:
            return resp.status == 200
    except Exception:
        return False


def _foxglove_config(state: dict | None = None) -> dict:
    session = state if isinstance(state, dict) else _load_state()
    sim_viz = session.get("sim_viz") if isinstance(session.get("sim_viz"), dict) else {{}}
    env, origin = dict(os.environ), _agent_public_origin()
    payload = resolve_foxglove_config(
        env,
        assets_dir=FOXGLOVE_SDK_DIR,
        origin=origin,
        sim_viz=sim_viz,
        self_hosted_ready=_self_hosted_viewer_healthy(),
    )
    selected_foxglove = dict(sim_viz.get("foxglove_selected_artifact") or {{}})
    selected_key = str(selected_foxglove.get("key") or "")
    canonical_key = str(sim_viz.get("canonical_mcap_key") or "")
    provenance = (
        dict(sim_viz.get("canonical_mcap_provenance") or {{}})
        if not selected_key or selected_key == canonical_key
        else {{}}
    )
    rich_visualization = has_rich_visualization_contract(provenance)
    if provenance:
        payload["layout"] = (
            data_aware_layout_data(provenance) if rich_visualization else {{}}
        )
        if not rich_visualization:
            payload["layout_storage_key"] = (
                str(payload.get("layout_storage_key") or "npa-agent-foxglove")
                + "-source-default"
            )
        payload["visualization"] = {{
            "contract": str(provenance.get("visualization_contract") or ""),
            "fixed_frame": str(provenance.get("visualization_fixed_frame") or ""),
            "fidelity": str(provenance.get("visualization_fidelity") or ""),
            "topics": dict(provenance.get("schemas") or {{}}),
            "checked": rich_visualization,
        }}
    else:
        payload["layout"] = {{}}
        payload["layout_storage_key"] = (
            str(payload.get("layout_storage_key") or "npa-agent-foxglove")
            + "-source-default"
        )
        payload["visualization"] = {{"checked": False}}
    return payload


_RERUN_RESTART_MIN_INTERVAL_S = 8.0
_last_rerun_restart_monotonic = 0.0

def _rerun_service_active() -> bool:
    try:
        subprocess.run(
            ["systemctl", "is-active", "--quiet", RERUN_UNIT],
            check=True,
            capture_output=True,
            timeout=5,
        )
        return True
    except Exception:
        return False

def _rerun_web_viewer_healthy() -> bool:
    if not _rerun_service_active():
        return False
    try:
        import urllib.request

        with urllib.request.urlopen(
            f"http://127.0.0.1:{{RERUN_WEB_PORT}}/",
            timeout=2,
        ) as resp:
            return resp.status == 200
    except Exception:
        return False

def _wait_rerun_web_viewer_healthy(*, timeout_s: float = 12.0) -> bool:
    deadline = time.monotonic() + max(0.5, float(timeout_s))
    while time.monotonic() < deadline:
        if _rerun_web_viewer_healthy():
            return True
        time.sleep(0.4)
    return _rerun_web_viewer_healthy()


def _wait_for_rerun_web_viewer(*, timeout_s: float = 20.0) -> bool:
    return _wait_rerun_web_viewer_healthy(timeout_s=timeout_s)


def _rerun_ready_state(*, rrd_uri: str = "") -> bool:
    has_rrd = bool(str(rrd_uri or "").strip())
    if not has_rrd and RRD_PATH.is_file():
        has_rrd = True
    if has_rrd and not _rerun_service_active():
        _restart_rerun_serve()
    return has_rrd and _rerun_web_viewer_healthy()

def _restart_rerun_serve(*, force: bool = False) -> bool:
    global _last_rerun_restart_monotonic
    now = time.monotonic()
    if not force and _rerun_service_active():
        if now - _last_rerun_restart_monotonic < _RERUN_RESTART_MIN_INTERVAL_S:
            return True
    try:
        subprocess.run(
            ["sudo", "systemctl", "restart", RERUN_UNIT],
            check=True,
            capture_output=True,
            timeout=30,
        )
        _last_rerun_restart_monotonic = time.monotonic()
        return True
    except Exception:
        if _rerun_service_active():
            return True
        return False

def _wire_active_sim2real_recording(state: dict, *, camera: str = "workspace") -> dict | None:
    # Point the UI at an already-staged real Sim2Real recording, if present.
    current = state.get("sim_viz", {{}})
    if not isinstance(current, dict):
        current = {{}}
    latest = state.get("latest_submit", {{}})
    if not isinstance(latest, dict):
        latest = {{}}
    run_id = str(current.get("run_id") or latest.get("run_id") or "").strip()
    if not run_id or run_id == "franka-demo":
        return None
    if str(current.get("rrd_uri") or "").strip() and _served_recording_is_run_specific():
        return current
    # Reattach the run's OWN recording by content (run-specific entities), not by
    # a fragile size threshold — the stock demo is ~68KB and would pass a size
    # check. Recognize any run id (agent-run-*, sim2real-*, …) and fall back to
    # the run's local reports/ recording when the run-scoped copy is missing.
    def _has_run_entities(item):
        try:
            return item.is_file() and recording_has_run_entities(item.read_bytes())
        except Exception:
            return False

    run_rec = RECORDINGS_DIR / run_recording_basename(run_id)
    candidates = [
        run_rec,
        Path("/opt/npa-agent/runs") / run_id / "reports" / "sim2real.rrd",
    ]
    source = next((item for item in candidates if _has_run_entities(item)), None)
    if source is None:
        return None
    # Ensure a run-scoped copy exists, then publish it as the active recording.
    if source != run_rec:
        try:
            RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, run_rec)
        except Exception:
            pass
    capability_path = _publish_rrd_recording(run_rec if run_rec.is_file() else source)
    if RECORDING_PATH.is_file():
        RRD_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(RECORDING_PATH, RRD_PATH)
    _restart_rerun_serve(force=True)
    selection = _stock_franka_selection()
    state["selection"] = selection
    cam = (camera or "workspace").strip() or "workspace"
    state["camera_selection"] = [cam]
    updated_at = datetime.fromtimestamp(RRD_PATH.stat().st_mtime, tz=timezone.utc).isoformat()
    live_url = str(os.environ.get("NPA_AGENT_RERUN_LIVE_URL", "")).strip()
    iframe_url = (
        f"/rerun/?url={{quote(live_url, safe='')}}&hide_welcome_screen=1&theme=dark&camera={{cam}}"
        if live_url
        else _rerun_iframe_url(cam, recording_path=capability_path)
    )
    viz = {{
        "run_id": run_id,
        "stage": str(current.get("stage") or "completed"),
        "rrd_uri": f"file://{{RRD_PATH}}",
        "rrd_updated_at": updated_at,
        "artifact_uri": str(current.get("artifact_uri") or latest.get("rrd_uri") or ""),
        "artifact_key": str(current.get("artifact_key") or ""),
        "artifact_render": "rerun",
        "artifact_preview_url": capability_path,
        "artifact_download_url": "/api/sim-viz/rrd-blob",
        "live_grpc_url": live_url,
        "mode": "live" if live_url else "static",
        "camera": cam,
        "preview_camera": cam,
        "preview_entity": str(current.get("preview_entity") or "heldout/camera/env-00006/camera"),
        "rerun_ready": True,
        "rerun_iframe_url": iframe_url,
        "submit_mode": str(current.get("submit_mode") or latest.get("submit_mode") or "completed-k8s"),
        "workflow_name": "sim2real",
    }}
    if str(current.get("run_id") or "").strip() == run_id:
        # This helper repairs the Rerun recording after bootstrap or a shared
        # slot overwrite. It must not silently unpublish the same run's already
        # validated canonical MCAP while doing so; otherwise the next clean
        # Foxglove tab performs the entire S3 conversion again and briefly has
        # no source/layout. Cross-run selection is still cleared by
        # clear_cross_run_mcap_state in the artifact loader.
        for key in (
            *CANONICAL_MCAP_DEFAULT_STATE,
            "artifact_run_ref",
            "bucket",
            "project_id",
            "resolved_prefix",
            "mcap_uri",
            "mcap_updated_at",
            "lichtblick_ready",
            "lichtblick_iframe_url",
            "foxglove_ready",
            "foxglove_url",
        ):
            if key in current:
                viz[key] = current[key]
    for key in ("decision", "success_rate", "threshold"):
        if key in current:
            viz[key] = current[key]
        elif key in latest:
            viz[key] = latest[key]
    state["sim_viz"] = viz
    state["active_run_id"] = run_id
    runs = state.get("sim_viz_runs")
    if not isinstance(runs, dict):
        runs = {{}}
    runs[run_id] = {{**viz, "submitted_at": str(latest.get("submitted_at") or "")}}
    state["sim_viz_runs"] = runs
    _save_state(state)
    return viz

def _wire_franka_demo(
    state: dict, *, camera: str = "workspace", force_local_demo: bool = False
) -> dict:
    if not force_local_demo:
        active = _wire_active_sim2real_recording(state, camera=camera)
        if active is not None:
            return active
    # Preserve operator-posted custom URIs; only fill stock defaults when empty.
    current = state.get("selection") if isinstance(state.get("selection"), dict) else {{}}
    if not current:
        state["selection"] = _stock_franka_selection()
    cam = (camera or "workspace").strip() or "workspace"
    state["camera_selection"] = [cam]
    target = _generate_franka_demo_rrd(camera=cam)
    capability_path = _publish_rrd_recording(target)
    restarted = _restart_rerun_serve()
    viewer_ready = _wait_for_rerun_web_viewer() if restarted else False
    now = _now_iso()
    # Always use the stock demo run id and clear any prior media-artifact preview.
    viz = {{
        "run_id": "franka-demo",
        "source_type": "local_demo",
        "source_label": "Local demo",
        "stage": "demo",
        "rrd_uri": f"file://{{target}}",
        "rrd_updated_at": now,
        "live_grpc_url": "",
        "mode": "static",
        "camera": cam,
        "preview_camera": cam,
        "preview_entity": f"world/camera_frustums/{{cam}}/frustum",
        "rerun_ready": target.is_file() and viewer_ready,
        "rerun_iframe_url": _rerun_iframe_url(cam, recording_path=capability_path),
        "artifact_render": "rerun",
        "artifact_key": "",
        "artifact_uri": "",
        "artifact_preview_url": capability_path,
        "artifact_download_url": "/api/sim-viz/rrd-blob",
        "visualization_note": "",
    }}
    state["sim_viz"] = viz
    _record_sim_viz_run(state, viz)
    _save_state(state)
    return viz

def _wire_sim2real_run_preview(state: dict, *, run_id: str, camera: str = "workspace") -> dict:
    # Attach a concrete Rerun recording to a submitted Sim2Real run id.
    cam = (camera or "workspace").strip() or "workspace"
    state["camera_selection"] = [cam]
    target = _generate_franka_demo_rrd(camera=cam)
    capability_path = _publish_rrd_recording(target)
    restarted = _restart_rerun_serve()
    viewer_ready = _wait_for_rerun_web_viewer() if restarted else False
    now = _now_iso()
    viz = {{
        "run_id": str(run_id or "").strip() or f"agent-run-{{secrets.token_hex(6)}}",
        "source_type": "workflow_history",
        "source_label": "Workflow history",
        "stage": "stage_14_rerun_viz",
        "rrd_uri": f"file://{{target}}",
        "rrd_updated_at": now,
        "live_grpc_url": "",
        "mode": "static",
        "camera": cam,
        "preview_camera": cam,
        "preview_entity": f"world/camera_frustums/{{cam}}/frustum",
        "rerun_ready": target.is_file() and viewer_ready,
        "rerun_iframe_url": _rerun_iframe_url(cam, recording_path=capability_path),
        "artifact_preview_url": capability_path,
        "artifact_download_url": "/api/sim-viz/rrd-blob",
        "submit_mode": "sim2real",
        "workflow_name": "sim2real",
        "pipeline_visualization": True,
    }}
    state["sim_viz"] = viz
    _record_sim_viz_run(state, viz)
    _save_state(state)
    return viz

LLM_PROVIDER = os.environ.get("NPA_AGENT_LLM_PROVIDER", "{DEFAULT_LLM_PROVIDER}").strip() or "{DEFAULT_LLM_PROVIDER}"
LLM_PROVIDERS_ENV = os.environ.get("NPA_AGENT_LLM_PROVIDERS", "")
LLM_MODEL = os.environ.get("NPA_AGENT_LLM_MODEL", "{DEFAULT_LLM_MODEL}")
LLM_MODELS_ENV = os.environ.get("NPA_AGENT_LLM_MODELS", "")
DEFAULT_LLM_MODELS = {default_llm_models_json}
NPA_PROJECT_ALIAS = os.environ.get("NPA_AGENT_PROJECT_ALIAS", "").strip() or "default"
NPA_SOURCE_ROOT = Path("{AGENT_SOURCE_ROOT}")
NPA_CLI = Path("/opt/npa-agent/venv/bin/npa")
NPA_CLUSTER_TERRAFORM_DIR = NPA_SOURCE_ROOT / "deploy" / "cluster"
TF_BASE_URL = os.environ.get(
    "NEBIUS_TOKEN_FACTORY_BASE_URL", "https://api.tokenfactory.nebius.com/v1/"
).rstrip("/")
_THINK_RE = re.compile(
    r"\\A\\s*<think>(?P<reasoning>.*?)</think>\\s*", re.DOTALL
)
_MODELS_CACHE = {{"expires_at": 0.0, "models": []}}

def _normalize_llm_models(raw: str) -> list[str]:
    models: list[str] = []
    for part in str(raw or "").replace("\\n", ",").split(","):
        value = part.strip()
        if value and value not in models:
            models.append(value)
    return models

def _configured_llm_models() -> list[str]:
    configured = _normalize_llm_models(LLM_MODELS_ENV)
    if not configured:
        configured = [str(item) for item in DEFAULT_LLM_MODELS if str(item).strip()]
    if LLM_MODEL not in configured:
        configured.insert(0, LLM_MODEL)
    return configured

def _configured_llm_providers() -> list[str]:
    providers = _normalize_llm_models(LLM_PROVIDERS_ENV)
    if not providers:
        providers = [LLM_PROVIDER]
    if LLM_PROVIDER not in providers:
        providers.insert(0, LLM_PROVIDER)
    return providers

def _provider_base_url(provider: str) -> str:
    normalized = str(provider or "").strip().lower().replace("-", "_")
    if normalized in {"token_factory", "tokenfactory"}:
        return os.environ.get("NEBIUS_TOKEN_FACTORY_BASE_URL", "https://api.tokenfactory.nebius.com/v1/").rstrip("/")
    env_key = f"NPA_AGENT_{{normalized.upper()}}_BASE_URL"
    custom = str(os.environ.get(env_key, "")).strip()
    return custom.rstrip("/")

def _provider_api_key(provider: str) -> str:
    normalized = str(provider or "").strip().lower().replace("-", "_")
    if normalized in {"token_factory", "tokenfactory"}:
        return str(os.environ.get("NEBIUS_TOKEN_FACTORY_KEY", "")).strip()
    env_keys = [
        f"NPA_AGENT_{{normalized.upper()}}_API_KEY",
        f"NEBIUS_{{normalized.upper()}}_KEY",
    ]
    for key in env_keys:
        value = str(os.environ.get(key, "")).strip()
        if value:
            return value
    return ""

def _fetch_token_factory_models() -> list[str]:
    api_key = _provider_api_key("token_factory")
    if not api_key:
        return []
    base_url = _provider_base_url("token_factory")
    if not base_url:
        return []
    url = f"{{base_url}}/models"
    try:
        response = httpx.get(
            url,
            headers={{
                "Authorization": f"Bearer {{api_key}}",
                "Content-Type": "application/json",
            }},
            timeout=20.0,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return []
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return []
    models: list[str] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        value = str(item.get("id") or "").strip()
        if value and value not in models:
            models.append(value)
    return models

def _available_llm_models(*, refresh: bool = False) -> list[str]:
    configured = _configured_llm_models()
    now = time.monotonic()
    cache = _MODELS_CACHE
    if not refresh and cache.get("expires_at", 0.0) > now:
        cached = cache.get("models", [])
        if isinstance(cached, list) and cached:
            return cached
    live = _fetch_token_factory_models()
    if live:
        allowed = [model for model in configured if model in live]
        extras = [model for model in live if model not in allowed]
        resolved = (allowed + extras)[:32]
    else:
        resolved = configured
    cache["models"] = resolved
    cache["expires_at"] = now + 300.0
    return resolved

def _agent_system_prompt() -> str:
    lines = [
        "You are the NPA workbench assistant on a Nebius Physical AI agent VM.",
        "Help operators configure NPA: provision infrastructure, Cosmos3, S3 storage,",
        "workflows, sim assets, and Sim2Real runs. Be concise and actionable.",
        "",
        "Agent HTTP APIs on this VM (same-origin relative paths; nginx proxies /api/):",
        "- GET /api/access — tenant identity, project-by-project effective access, and searchable resources",
        "- GET /api/sim-assets, /api/sim-assets/selection, /api/sim-assets/cameras",
        "- GET /api/sim-viz/status — active run + .rrd URI for the Rerun iframe at /rerun/",
        "- GET /api/sim-viz/recordings — list available .rrd recording files for quick viewer switching",
        "- GET /api/sim-viz/runs — list run-scoped history (run_id, stage, camera, rrd_uri)",
        "- POST /api/sim-viz/load-run — switch active run context by run_id",
        "- GET /api/artifacts/runs?prefix=&limit= — discover run prefixes from object storage (no workflow allowlist)",
        "- GET /api/artifacts/run/{{run_id}} — list every object for a run with render hints",
        "- POST /api/sim-viz/load-artifact — load run_id+s3_uri or run_id+key into viewer/download",
        "- POST /api/sim-viz/load-franka-demo — load stock Franka tabletop demo into Rerun",
        "- GET /api/foxglove/config, /api/foxglove/status — embedded Foxglove viewer config + readiness",
        "- POST /api/foxglove/load-artifact | /api/foxglove/convert-run | /api/foxglove/live —"
        " open an .mcap/.bag recording, pack the active run's artifacts into MCAP, or attach a live ws:// source",
        "- POST /api/workflows/sim2real/submit — submit Sim2Real with current asset selection",
        "- GET/POST /api/workflows/draft — workflow YAML draft in session",
        "- POST /api/workflows/validate — validate npa.workflow/v0.0.1 or npa.workflow/v0.0.1-beta YAML",
        "- POST /api/workflows/plan — dry-run plan-spec for workflow YAML",
        "- POST /api/workflows/submit — validate workflow YAML, ensure agent-side Kubernetes infra when needed, and return scheduler plan",
        "- GET /api/models — list Token Factory chat models available to this VM key",
        "- GET /api/tools — workbench toolRef catalog",
        "- POST /api/agent/gpu-allocation/attempt — record typed GPU placement evidence and get a grounded fallback decision",
        "- POST /api/agent/gpu-allocation/consent — accept or decline the exact tracked fallback; acceptance requires its confirmation token",
        "",
        "To view Franka immediately, tell users to open the **Rerun** tab and click **Load Franka in Rerun**",
        "(or POST /api/sim-viz/load-franka-demo). The UI has two tabs: **Chat** and **Rerun**.",
        "Artifact-first browsing flow: call `/api/artifacts/runs`, inspect `/api/artifacts/run/{{id}}`,",
        "then `POST /api/sim-viz/load-artifact` with `run_id` + `s3_uri` or `run_id` + `key`.",
        "The **Rerun** tab embeds the viewer full-bleed beside a run-loading rail (mp4/video preview,",
        "artifact browser, and Load run data). There is no separate Cameras panel in the UI.",
        "Never suggest localhost, 127.0.0.1, or port 8080 — use relative /api/... paths or /rerun/.",
        "When asked about Sim2Real, workflow, or Rerun status, summarize run_id, stage, camera,",
        "rerun_ready, and latest_submit from session state — never reply with only a raw GET path.",
        "When generating workflow YAML, always emit canonical keys: apiVersion, kind, metadata, config,",
        "initial, and states. Never emit api_version, stages, or previous.outputs placeholders.",
        "",
        "Workbench toolRefs (invoke via npa workbench / npa.workflow on operator machine):",
    ]
    for key in TOOL_REFS:
        entry = TOOL_CATALOG.get(key, {{}})
        desc = entry.get("description", "")
        lines.append(f"- {{key}}: {{desc}}")
    lines.extend(
        [
            "",
            "Before Sim2Real submit, confirm scene/robot/camera selection.",
            "Always use real registry-qualified images: supported defaults resolve from",
            "public GHCR and ignore ambient or legacy private-registry configuration.",
            "Select custom/private bytes with an explicit image or workflow `--registry`;",
            "never keep registry placeholders in runnable workflows.",
            "For BYOF solution onboarding, use `npa workbench byof run`",
            "(or `npa/scripts/run_byof_repo.py`) to containerize an OSS repo,",
            "push to an explicitly selected customer registry, then launch a real Isaac-Lab run",
            "with `--image` override on RT-core GPUs (L40S / RTX PRO 6000).",
            "See docs/architecture/oss-onboarding-ladder.md for Tier 0→2 promotion.",
            "For live infra runs, verify GPU compatibility first (`sky check`, `sky gpus list`)",
            "and loop submit attempts in tmux until validation+plan+prechecks pass.",
            "After submit, point users to /rerun/ and poll /api/sim-viz/status until rrd_uri is set.",
        ]
    )
    return "\\n".join(lines)

def _split_reasoning(message: dict) -> tuple[str, str | None]:
    content = message.get("content")
    reasoning = message.get("reasoning") or message.get("reasoning_content")
    if reasoning is not None and not isinstance(reasoning, str):
        reasoning = str(reasoning)
    if isinstance(content, str):
        match = _THINK_RE.match(content)
        if match:
            visible = content[match.end() :].strip()
            trace = (match.group("reasoning").strip() or reasoning)
            return visible, trace
        return content.strip(), reasoning
    return "", (reasoning.strip() if reasoning else None)

def _provider_chat(*, provider: str, messages: list, model: str, extra: dict | None = None, max_tokens: int | None = None) -> dict:
    api_key = _provider_api_key(provider)
    if not api_key:
        raise RuntimeError(f"missing API key for provider '{{provider}}'")
    base_url = _provider_base_url(provider)
    if not base_url:
        raise RuntimeError(f"missing base URL for provider '{{provider}}'")
    url = f"{{base_url}}/chat/completions"
    payload = {{
        "model": model,
        "messages": messages,
        "temperature": 0.2,
    }}
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if isinstance(extra, dict):
        for _extra_key, _extra_value in extra.items():
            payload[_extra_key] = _extra_value
    for attempt in range(3):
        try:
            response = httpx.post(
                url,
                headers={{
                    "Authorization": f"Bearer {{api_key}}",
                    "Content-Type": "application/json",
                }},
                json=payload,
                timeout=120.0,
            )
            response.raise_for_status()
            data = response.json()
            break
        except httpx.HTTPError as exc:
            status_code = getattr(getattr(exc, "response", None), "status_code", 0)
            transient = bool(status_code in {{408, 409, 425, 429}} or status_code >= 500)
            if transient and attempt < 2:
                time.sleep(0.6 * (2 ** attempt))
                continue
            raise RuntimeError(f"provider '{{provider}}' request failed (status={{status_code}}): {{exc}}") from exc
    else:
        raise RuntimeError(f"provider '{{provider}}' did not return a response")
    if not isinstance(data, dict):
        raise RuntimeError(f"provider '{{provider}}' returned non-object response")
    return data

def _chat_with_resilience(
    *,
    messages: list,
    requested_model: str = "",
    tier: str = "standard",
    interactive: bool = True,
) -> tuple[dict, str, str]:
    providers = _configured_llm_providers()
    configured = _configured_llm_models()
    # Respect an explicit operator allowlist (NPA_AGENT_LLM_MODELS) by not
    # injecting tier-default models the operator did not opt into.
    allow_tier_defaults = not str(LLM_MODELS_ENV or "").strip()
    ladder = build_model_ladder(
        tier,
        configured,
        interactive=interactive,
        requested_model=requested_model,
        allow_tier_defaults=allow_tier_defaults,
    )
    # Drop flavors/models the key cannot serve (e.g. missing -fast variants) so
    # interactive turns do not burn a round-trip on a guaranteed 404.
    try:
        ladder = filter_available(ladder, _available_llm_models())
    except Exception:
        pass
    if not ladder:
        ladder = list(configured) or [requested_model] if requested_model else list(configured)
    extra = chat_extra(tier)
    errors: list[str] = []
    for provider in providers:
        for model in ladder:
            try:
                data = _provider_chat(provider=provider, messages=messages, model=model, extra=extra)
                return data, provider, model
            except Exception as exc:
                errors.append(str(exc))
                continue
    detail = "; ".join(errors[-4:]) if errors else "no providers configured"
    raise HTTPException(status_code=502, detail=f"LLM providers unavailable: {{detail}}")

{_AGENT_ROUTING_EMBED}

{_AGENT_VISUAL_FEEDBACK_EMBED}

{_AGENT_STAGES_EMBED}

{_AGENT_STAGE_RUNTIME_EMBED}

{_AGENT_PROVENANCE_EMBED}

import sys as _npa_sys
if "/opt/npa-agent" not in _npa_sys.path:
    _npa_sys.path.insert(0, "/opt/npa-agent")

{_AGENT_CHAT_EMBED}

from agent_backend.actions import (
    DEFAULT_MAX_STEPS,
    action_digest,
    allowlist_specs,
    confirmation_ok,
    normalize_group_by,
    normalize_threshold_op,
    run_action_loop,
    run_chat_action_loop,
)

{_AGENT_RECORDINGS_EMBED}

from agent_backend.sim2real_loop import (
    drive_sim2real_loop,
    gate_with_config_threshold,
    resolve_drive_config,
)

from agent_backend.semantic_router import classify_intent_semantic

# Phase G: run memory is a SHIPPED module (uploaded to /opt/npa-agent/agent_backend
# and imported here) rather than string-substituted into this f-string.
from agent_backend.memory import RunMemory, JsonFileStore
# Blueprint Phases H/I: retrieval + observability are also shipped modules.
from agent_backend import retrieval as _retrieval
from agent_backend import trace as _agent_tracing
from agent_backend import gpu_allocation_fallback as _gpu_fallback
from agent_backend import access_approval as _access_approval

{_AGENT_WORKFLOW_EMBED}

{_AGENT_ARTIFACTS_EMBED}

{_AGENT_ACCESS_EMBED}

def _workflow_draft_from_state(state: dict) -> dict:
    draft = state.get("workflow_draft", {{}})
    return draft if isinstance(draft, dict) else {{}}

def _save_workflow_draft(
    state: dict,
    yaml_text: str,
    validation: dict,
    *,
    plan: dict | None = None,
    runnable: bool | None = None,
) -> dict:
    resolved_plan = plan if isinstance(plan, dict) else {{}}
    resolved_runnable = bool(runnable) if runnable is not None else bool(validation.get("ok") and resolved_plan.get("ok"))
    draft = {{
        "yaml": yaml_text,
        "validation": validation if isinstance(validation, dict) else {{}},
        "plan": resolved_plan,
        "runnable": resolved_runnable,
        "updated_at": _now_iso(),
        "name": str((validation or {{}}).get("name") or ""),
        "status": str((validation or {{}}).get("status") or ""),
        "states": (validation or {{}}).get("states") or [],
    }}
    state["workflow_draft"] = draft
    _save_state(state)
    return draft

def _sim_viz_runs(state: dict) -> list[dict]:
    runs = state.get("sim_viz_runs")
    if not isinstance(runs, dict):
        return []
    snapshots: list[dict] = []
    for run_id, item in runs.items():
        if not isinstance(item, dict):
            continue
        snapshot = dict(DEFAULT_SIM_VIZ)
        snapshot.update(item)
        snapshot["run_id"] = str(item.get("run_id") or run_id or "").strip()
        if not snapshot["run_id"]:
            continue
        snapshots.append(snapshot)
    return sorted(
        snapshots,
        key=lambda item: (
            str(item.get("rrd_updated_at") or ""),
            str(item.get("run_id") or ""),
        ),
        reverse=True,
    )

def _resolve_workflow_yaml(payload: dict) -> str:
    # Empty/omitted YAML must 400 so validate/plan cannot leak another tab's draft.
    return str(payload.get("yaml") or "").strip()

def _agent_npa_ready() -> tuple[bool, str]:
    if not NPA_CLI.exists():
        return False, f"NPA CLI is not installed at {{NPA_CLI}}"
    if not (NPA_SOURCE_ROOT / "npa" / "pyproject.toml").is_file():
        return False, f"NPA source is not staged at {{NPA_SOURCE_ROOT}}"
    if not NPA_CLUSTER_TERRAFORM_DIR.is_dir():
        return False, f"Kubernetes Terraform assets are not staged at {{NPA_CLUSTER_TERRAFORM_DIR}}"
    return True, ""


def _load_agent_config_yaml() -> dict:
    path = Path.home() / ".npa" / "config.yaml"
    if not path.is_file():
        return {{}}
    try:
        import yaml

        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return {{}}
    return loaded if isinstance(loaded, dict) else {{}}


def _agent_project_alias(requested: str = "") -> str:
    requested = str(requested or "").strip()
    if requested:
        return requested
    config = _load_agent_config_yaml()
    configured = str(config.get("default_project") or "").strip()
    if configured:
        return configured
    return NPA_PROJECT_ALIAS


def _agent_k8s_backends(project: str = "") -> dict:
    config = _load_agent_config_yaml()
    alias = _agent_project_alias(project)
    ready, reason = _agent_npa_ready()
    cloud_clusters = _agent_cloud_mk8s_clusters(alias)
    inventory = assemble_k8s_backend_inventory(
        config=config, alias=alias, clusters_root=Path.home() / ".npa" / "clusters",
        cloud_clusters=cloud_clusters, npa_ready=ready,
        npa_error=reason, terraform_dir=NPA_CLUSTER_TERRAFORM_DIR,
    )
    inventory["agent_exists"] = _configured_healthy_agent_exists(alias, config)
    return inventory


def _configured_healthy_agent_exists(alias: str, config: dict | None = None) -> bool:
    # True only for this running agent's exact configured project record.
    payload = config if isinstance(config, dict) else _load_agent_config_yaml()
    projects = payload.get("projects") if isinstance(payload, dict) else {{}}
    project = projects.get(alias) if isinstance(projects, dict) else {{}}
    if not isinstance(project, dict):
        return False
    project_id = str(project.get("project_id") or "").strip()
    runtime_project_id = str(os.environ.get("NPA_PROJECT_ID") or "").strip()
    if not project_id or project_id != runtime_project_id:
        return False
    agents = project.get("agents")
    if not isinstance(agents, dict):
        return False
    runtime_agent_name = str(os.environ.get("NPA_AGENT_NAME") or "agent").strip()
    record = agents.get(runtime_agent_name)
    if not isinstance(record, dict):
        return False
    ready, _reason = _agent_npa_ready()
    return bool(
        ready
        and str(record.get("project_id") or project_id).strip() == project_id
        and str(record.get("public_ip") or "").strip()
    )


def _agent_command_env() -> dict:
    env = dict(os.environ)
    env["PATH"] = "/usr/local/bin:/usr/bin:/bin:" + env.get("PATH", "")
    env.setdefault("NPA_TERRAFORM_BIN", shutil.which("terraform") or "terraform")
    env.setdefault("NPA_KUBECTL_BIN", shutil.which("kubectl") or "kubectl")
    env.setdefault("NPA_NEBIUS_BIN", shutil.which("nebius") or "nebius")
    if Path("/mnt/cloud-metadata/token").is_file():
        env.setdefault("NEBIUS_PROFILE", "cursor-sa")
    if not env.get("TF_VAR_ssh_public_key"):
        for candidate in ("/home/ubuntu/.ssh/id_ed25519.pub", "/root/.ssh/id_ed25519.pub"):
            if os.path.isfile(candidate) and os.access(candidate, os.R_OK):
                env["TF_VAR_ssh_public_key"] = json.dumps({{"path": candidate}})
                break
        if not env.get("TF_VAR_ssh_public_key"):
            for candidate in ("/home/ubuntu/.ssh/authorized_keys", "/root/.ssh/authorized_keys"):
                path = Path(candidate)
                if not os.path.isfile(candidate) or not os.access(candidate, os.R_OK):
                    continue
                for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                    value = line.strip()
                    if value.startswith(("ssh-ed25519 ", "ssh-rsa ", "ecdsa-sha2-")):
                        env["TF_VAR_ssh_public_key"] = json.dumps({{"key": value}})
                        break
                if env.get("TF_VAR_ssh_public_key"):
                    break
    return env


def _agent_cloud_mk8s_clusters(project: str = "") -> list[dict]:
    config = _load_agent_config_yaml()
    projects = config.get("projects")
    if not isinstance(projects, dict):
        projects = {{}}
    project_block = projects.get(_agent_project_alias(project))
    if not isinstance(project_block, dict):
        project_block = {{}}
    parent_id = str(os.environ.get("NEBIUS_PROJECT_ID") or project_block.get("project_id") or "").strip()
    if not parent_id:
        return []
    nebius_bin = shutil.which("nebius") or "/usr/local/bin/nebius"
    if not Path(nebius_bin).exists() and shutil.which(nebius_bin) is None:
        return []
    command_env = _agent_command_env()
    command: list[str] = [nebius_bin]
    if Path("/mnt/cloud-metadata/token").is_file():
        for key in ("NEBIUS_IAM_TOKEN", "NPA_NEBIUS_IAM_TOKEN", "NEBIUS_IAM_TOKEN_FILE"):
            command_env.pop(key, None)
        command.extend(["--profile", "cursor-sa"])
    try:
        proc = subprocess.run(
            [*command, "mk8s", "cluster", "list", "--parent-id", parent_id, "--format", "json"],
            env=command_env,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        if proc.returncode != 0:
            return []
        payload = json.loads(proc.stdout or "{{}}")
    except Exception:
        return []
    items = payload.get("items") if isinstance(payload, dict) else []
    clusters: list[dict] = []
    if not isinstance(items, list):
        return clusters
    for item in items:
        if not isinstance(item, dict):
            continue
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {{}}
        status = item.get("status") if isinstance(item.get("status"), dict) else {{}}
        cluster_id = str(metadata.get("id") or "")
        raw = discover_mk8s_accelerators(cluster_id, command, command_env) if cluster_id else {{}}
        clusters.append({{
            "source": "nebius_mk8s",
            "id": cluster_id,
            "name": str(metadata.get("name") or ""),
            "status": str(status.get("state") or status.get("status") or ""),
            "raw": raw,
        }})
    return clusters


def _tenant_resource_inventory(*, force_refresh: bool = False) -> dict:
    return build_resource_inventory(
        config=_load_agent_config_yaml(), env=dict(os.environ), state=_load_state(),
        tool_refs=TOOL_REFS, generated_at=_now_iso(),
        runner=lambda command: run_resource_discovery_command(
            command, command_env=_agent_command_env()
        ),
        metadata_token_available=Path("/mnt/cloud-metadata/token").is_file(),
        force_refresh=force_refresh,
    )


def _run_agent_npa_json(args: list[str], *, timeout_s: int = 300) -> dict:
    ready, reason = _agent_npa_ready()
    if not ready:
        raise HTTPException(status_code=409, detail=reason)
    try:
        proc = subprocess.run(
            [str(NPA_CLI), *args],
            cwd=str(NPA_SOURCE_ROOT),
            env=_agent_command_env(),
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(
            status_code=502,
            detail=f"NPA command timed out after {{timeout_s}}s: {{args}}",
        ) from exc
    except OSError as exc:
        raise HTTPException(status_code=502, detail=f"NPA command failed to start: {{exc}}") from exc
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise HTTPException(status_code=502, detail=detail or f"NPA command failed: {{args}}")
    stdout = (proc.stdout or "").strip()
    try:
        return json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=502, detail=f"NPA command did not return JSON: {{stdout[-1000:]}}") from exc


_SIM2REAL_STAGE_BY_NUMBER = {{
    1: "stage_01_trigger",
    2: "stage_02_assets",
    3: "stage_03_augment",
    4: "stage_04_envs_raw",
    5: "stage_05_envs_train",
    6: "stage_06_tokens",
    7: "stage_07_actions_train",
    8: "stage_08_vlm_eval_train",
    9: "stage_09_training_signal",
    10: "stage_10_eval_heldout",
    11: "stage_11_outer_loop",
    12: "stage_12_external_validation_stub",
    13: "stage_13_retrigger",
    14: "stage_14_rerun_viz",
}}


def _update_sim2real_run(run_id: str, *, mutate) -> dict:
    state = _load_state()
    runs_detail = state.get("sim2real_runs")
    if not isinstance(runs_detail, dict):
        runs_detail = {{}}
    details = runs_detail.get(run_id)
    if not isinstance(details, dict):
        details = _default_sim2real_run_details(run_id, submitted_at=_now_iso(), selection={{}})
    details = mutate(details) or details
    details["updated_at"] = _now_iso()
    runs_detail[run_id] = details
    state["sim2real_runs"] = runs_detail
    sim_viz = state.get("sim_viz")
    if not isinstance(sim_viz, dict) or str(sim_viz.get("run_id") or "") == run_id:
        state["sim_viz"] = {{
            **(sim_viz if isinstance(sim_viz, dict) else {{}}),
            "run_id": run_id,
            "stage": str(details.get("status") or "running"),
            "rrd_updated_at": details["updated_at"],
            "camera": str((sim_viz or {{}}).get("camera") or "workspace") if isinstance(sim_viz, dict) else "workspace",
        }}
    _save_state(state)
    return details


def _append_run_log(details: dict, message: str, *, level: str = "info") -> None:
    logs = details.get("logs")
    if not isinstance(logs, list):
        logs = []
    logs.append({{"timestamp": _now_iso(), "level": level, "message": message}})
    details["logs"] = logs[-200:]


def _mark_stage(details: dict, stage_id: str, status: str, summary: str = "") -> None:
    stages = details.get("stages")
    if not isinstance(stages, list):
        stages = _default_sim2real_run_details(str(details.get("run_id") or ""), submitted_at=str(details.get("submitted_at") or "")).get("stages", [])
    normalized, status_label = normalize_explicit_stage_status(status)
    for item in stages:
        if isinstance(item, dict) and item.get("id") == stage_id:
            item["status"] = normalized
            item["status_label"] = status_label
            item["raw_status"] = str(status or "")
            item["evidence"] = {{
                "type": "event_log",
                "source": "agent_sim2real_runner",
                "authority": "authoritative",
                "confidence": "high",
                "reason": summary or f"The agent Sim2Real runner reported '{{status}}'.",
                "observed_at": _now_iso(),
            }}
            item["evidence_type"] = "event_log"
            item["evidence_source"] = "agent_sim2real_runner"
            item["authority"] = "authoritative"
            item["confidence"] = "high"
            item["diagnostic_reason"] = str(item["evidence"]["reason"])
            if normalized == "running" and not item.get("started_at"):
                item["started_at"] = _now_iso()
            if normalized in {{"succeeded", "failed", "skipped", "not_run"}}:
                item["finished_at"] = _now_iso()
            if summary:
                item["summary"] = summary
            break
    details["stages"] = stages
    details["stage_summary"] = summarize_stage_evidence(stages)


def _sim2real_agent_command(run_id: str, output_dir: Path) -> list[str]:
    settings = _agent_s3_settings()
    cmd = [
        str(AGENT_PYTHON),
        "-m",
        "npa.workflows.sim2real",
        "run",
        "--run-id",
        run_id,
        "--output-dir",
        str(output_dir),
        "--env-count",
        "6",
        "--train-fraction",
        "0.5",
        "--inner-iterations",
        "1",
        "--outer-iterations",
        "1",
        "--rollout-count",
        "1",
        "--steps-per-rollout",
        "2",
        "--heldout-env-count",
        "2",
        "--heldout-eval-limit",
        "2",
        "--sim-backend",
        "genesis",
        "--no-guardrails",
        "--rerun",
    ]
    return cmd


def _apply_sim2real_report_to_details(details: dict, report: dict) -> None:
    report_status = str(report.get("status") or "").lower()
    if report_status == "completed":
        for stage_id, label in SIM2REAL_STAGE_TEMPLATE:
            if stage_id == "submit":
                continue
            if stage_id == "stage_14_rerun_viz":
                continue
            _mark_stage(details, stage_id, "succeeded", f"Completed during local Sim2Real run: {{label}}.")
    records = report.get("component_records")
    if isinstance(records, list):
        for record in records:
            if not isinstance(record, dict):
                continue
            payload = record.get("payload") if isinstance(record.get("payload"), dict) else {{}}
            stage_num = payload.get("stage")
            try:
                stage_id = _SIM2REAL_STAGE_BY_NUMBER.get(int(stage_num))
            except Exception:
                stage_id = None
            path_text = str(record.get("path") or "").lower()
            component_text = str(record.get("component") or "").lower()
            if stage_id is None:
                if "stage_01_trigger" in path_text:
                    stage_id = "stage_01_trigger"
                elif "stage_02_assets" in path_text or "consumed_scene" in path_text:
                    stage_id = "stage_02_assets"
                elif "augment" in path_text or "cosmos2" in component_text:
                    stage_id = "stage_03_augment"
                elif "envs/raw" in path_text:
                    stage_id = "stage_04_envs_raw"
                elif "envs/train" in path_text:
                    stage_id = "stage_05_envs_train"
                elif "tokens" in path_text:
                    stage_id = "stage_06_tokens"
                elif "actions/train" in path_text or "policy" in component_text:
                    stage_id = "stage_07_actions_train"
                elif "vlm_eval" in path_text:
                    stage_id = "stage_08_vlm_eval_train"
                elif "training_signal" in path_text:
                    stage_id = "stage_09_training_signal"
                elif "eval/heldout" in path_text or "heldout" in component_text:
                    stage_id = "stage_10_eval_heldout"
                elif "outer_loop" in path_text or "decision" in path_text:
                    stage_id = "stage_11_outer_loop"
                elif "stage_12_external_validation" in path_text:
                    stage_id = "stage_12_external_validation_stub"
                elif "stage_13_retrigger" in path_text:
                    stage_id = "stage_13_retrigger"
            if stage_id:
                status = str(payload.get("status") or record.get("status") or "completed").lower()
                normalized = "succeeded" if status in {{"completed", "succeeded", "success", "written"}} else status
                _mark_stage(details, stage_id, normalized, str(record.get("component") or payload.get("schema") or stage_id))
    viz = report.get("visualization")
    if isinstance(viz, dict) and str(viz.get("status") or "").lower() in {{"written", "completed", "succeeded"}}:
        _mark_stage(details, "stage_14_rerun_viz", "succeeded", "Rerun recording written.")
    details["status"] = str(report.get("status") or "completed")
    details["result"] = "completed" if str(details["status"]).lower() == "completed" else str(details["status"])
    details["report"] = {{
        "status": report.get("status"),
        "run_id": report.get("run_id"),
        "latest_decision": ((report.get("outer_loop") or {{}}).get("latest_decision") or {{}}),
        "visualization": viz if isinstance(viz, dict) else {{}},
    }}


def _run_sim2real_pipeline_background(run_id: str, selection: dict) -> None:
    output_dir = Path("/opt/npa-agent/runs") / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    def _start(details: dict) -> dict:
        details["status"] = "running"
        details["result"] = "running"
        for stage_id, _label in SIM2REAL_STAGE_TEMPLATE:
            if stage_id == "submit":
                _mark_stage(details, stage_id, "succeeded", "Agent accepted the Sim2Real run request.")
            else:
                _mark_stage(details, stage_id, "pending", "Waiting for local Sim2Real runner.")
        _append_run_log(details, "Starting local Sim2Real runner on the agent VM.")
        return details

    _update_sim2real_run(run_id, mutate=_start)
    cmd = _sim2real_agent_command(run_id, output_dir)
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(NPA_SOURCE_ROOT),
            env=_agent_command_env(),
            text=True,
            capture_output=True,
            timeout=900,
            check=False,
        )
    except Exception as exc:
        def _fail_exc(details: dict) -> dict:
            details["status"] = "failed"
            details["result"] = "failed"
            _append_run_log(details, f"Sim2Real runner failed to start: {{exc}}", level="error")
            for stage_id, _label in SIM2REAL_STAGE_TEMPLATE:
                if stage_id != "submit":
                    _mark_stage(details, stage_id, "failed", "Runner failed before completing this stage.")
            return details

        _update_sim2real_run(run_id, mutate=_fail_exc)
        return

    report_path = output_dir / "reports" / "sim2real-report.json"
    rrd_path = output_dir / "reports" / "sim2real.rrd"

    def _upload_output_file(path: Path, relative_key: str) -> str:
        if not path.is_file():
            return ""
        settings = _agent_s3_settings()
        if not settings.get("bucket"):
            return ""
        s3, settings = _agent_s3_client()
        key = _join_agent_s3_prefix(
            _join_agent_s3_prefix(str(settings.get("prefix") or ""), "sim2real-b"),
            f"{{run_id}}/{{relative_key}}",
        )
        content_type = "application/octet-stream"
        if path.suffix.lower() == ".json":
            content_type = "application/json"
        s3.put_object(Bucket=settings["bucket"], Key=key, Body=path.read_bytes(), ContentType=content_type)
        return f"s3://{{settings['bucket']}}/{{key}}"

    def _upload_output_tree() -> list[str]:
        uploaded: list[str] = []
        for path in sorted(p for p in output_dir.rglob("*") if p.is_file()):
            rel = path.relative_to(output_dir).as_posix()
            uri = _upload_output_file(path, rel)
            if uri:
                uploaded.append(uri)
        return uploaded

    def _finish(details: dict) -> dict:
        stdout_tail = (proc.stdout or "")[-4000:].strip()
        stderr_tail = (proc.stderr or "")[-4000:].strip()
        if stdout_tail:
            _append_run_log(details, "runner stdout tail:\\n" + stdout_tail)
        if stderr_tail:
            _append_run_log(details, "runner stderr tail:\\n" + stderr_tail, level="warn" if proc.returncode == 0 else "error")
        if proc.returncode != 0:
            details["status"] = "failed"
            details["result"] = "failed"
            _append_run_log(details, f"Sim2Real runner exited with code {{proc.returncode}}.", level="error")
            for stage_id, _label in SIM2REAL_STAGE_TEMPLATE:
                if stage_id != "submit":
                    current = next((s for s in details.get("stages", []) if isinstance(s, dict) and s.get("id") == stage_id), {{}})
                    if current.get("status") not in {{"succeeded", "failed"}}:
                        _mark_stage(details, stage_id, "failed", "Runner exited before this stage completed.")
            return details
        if report_path.is_file():
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
                _apply_sim2real_report_to_details(details, report)
            except Exception as exc:
                details["status"] = "completed"
                details["result"] = "completed_with_report_parse_error"
                _append_run_log(details, f"Could not parse report: {{exc}}", level="warn")
        else:
            details["status"] = "completed"
            details["result"] = "completed_missing_report"
            _append_run_log(details, "Runner completed but report file was not found.", level="warn")
        if rrd_path.is_file():
            # Save a run-scoped copy so the run's recording is stable and cannot
            # be clobbered by a later franka-demo load / bootstrap.
            run_rec = RECORDINGS_DIR / run_recording_basename(run_id)
            try:
                RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
                shutil.copy2(rrd_path, run_rec)
            except Exception:
                pass
            # Only serve/claim the recording as run data when it actually holds
            # run-specific entities — never let the stock franka demo masquerade.
            try:
                _run_specific = recording_has_run_entities(rrd_path.read_bytes())
            except Exception:
                _run_specific = False
            if _run_specific:
                _publish_rrd_recording(rrd_path)
                try:
                    shutil.copy2(rrd_path, RRD_PATH)
                except Exception:
                    pass
                _restart_rerun_serve(force=True)
                _wait_rerun_web_viewer_healthy()
                _append_run_log(details, f"Published run-specific Rerun recording: {{run_rec}}")
            else:
                _append_run_log(
                    details,
                    "Run .rrd has no run-specific entities; not marking Rerun ready "
                    "(refusing to serve the stock demo as run data).",
                    level="warn",
                )
        uploaded = []
        try:
            uploaded = _upload_output_tree()
        except Exception as exc:
            _append_run_log(details, f"Failed to upload run tree to S3: {{exc}}", level="warn")
        if uploaded:
            details["artifact_uris"] = uploaded
            preview = ", ".join(uploaded[:5])
            suffix = " ..." if len(uploaded) > 5 else ""
            _append_run_log(details, f"Uploaded {{len(uploaded)}} run artifacts to S3: " + preview + suffix)
        return details

    _update_sim2real_run(run_id, mutate=_finish)
    state = _load_state()
    sim_viz = state.get("sim_viz")
    if not isinstance(sim_viz, dict):
        sim_viz = {{}}
    if rrd_path.is_file():
        try:
            _run_specific = recording_has_run_entities(rrd_path.read_bytes())
        except Exception:
            _run_specific = False
        run_rec = RECORDINGS_DIR / run_recording_basename(run_id)
        recording_uri = f"file://{{run_rec}}" if run_rec.is_file() else ""
        capability_path = _publish_rrd_recording(rrd_path) if _run_specific else ""
        # rerun_ready / rrd_uri only when the recording is genuinely run-specific.
        sim_viz.update(
            {{
                "run_id": run_id,
                "stage": "completed",
                "rrd_uri": recording_uri if _run_specific else "",
                "recording_uri": recording_uri,
                "rrd_updated_at": _now_iso(),
                "rerun_ready": bool(_run_specific and RECORDING_PATH.is_file() and _rerun_web_viewer_healthy()),
                "rerun_iframe_url": _rerun_iframe_url(str(sim_viz.get("camera") or "workspace"), recording_path=capability_path) if _run_specific else "",
                "artifact_preview_url": capability_path,
                "artifact_download_url": "/api/sim-viz/rrd-blob" if _run_specific else "",
                "camera": str(sim_viz.get("camera") or "workspace"),
            }}
        )
    else:
        sim_viz.update(
            {{"run_id": run_id, "stage": "completed", "rrd_updated_at": _now_iso(), "rerun_ready": False}}
        )
    state["sim_viz"] = sim_viz
    _record_sim_viz_run(state, sim_viz)
    _save_state(state)


def _write_workflow_temp_yaml(yaml_text: str) -> Path:
    tmp_dir = Path("/tmp/npa-agent-workflows")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    path = tmp_dir / f"workflow-{{secrets.token_hex(8)}}.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    return path


def _agent_mk8s_numeric(value, *, field: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"{{field}} must be an integer >= {{minimum}}")
    if isinstance(value, str) and not re.fullmatch(r"-?[0-9]+", value.strip()):
        raise ValueError(f"{{field}} must be an integer >= {{minimum}}")
    parsed = int(value)
    if parsed < minimum:
        raise ValueError(f"{{field}} must be an integer >= {{minimum}}")
    return parsed


def _normalize_agent_mk8s_desired(desired: dict | None) -> dict:
    requested = dict(desired) if isinstance(desired, dict) else {{}}
    for field, default, minimum in (
        ("gpu_nodes", -1, -1),
        ("cpu_nodes", -1, -1),
        ("gpu_health_stabilization_seconds", 120, 0),
        ("gpu_health_timeout_minutes", 60, 1),
    ):
        requested[field] = _agent_mk8s_numeric(
            requested.get(field, default), field=field, minimum=minimum
        )
    return requested


def _provision_agent_infra(
    project: str,
    cluster_name: str,
    *,
    dry_run: bool = False,
    validate: bool = True,
    skip_s3: bool = True,
    desired: dict | None = None,
    preemptible: bool | None = None,
) -> dict:
    ready, reason = _agent_npa_ready()
    if not ready:
        return {{"ok": False, "status": "blocked", "error": reason}}
    try:
        from npa.provisioning import provision_if_absent

        requested = _normalize_agent_mk8s_desired(desired)
        mig_value = requested.get("mig", False)
        mig_mapping = mig_value if isinstance(mig_value, dict) else {{}}
        result = provision_if_absent(
            project=project or None,
            cluster_name=cluster_name or "npa-cluster",
            terraform_dir=NPA_CLUSTER_TERRAFORM_DIR,
            skip_s3=skip_s3,
            validate=validate,
            sky_smoke=False,
            dry_run=dry_run,
            gpu_nodes=int(requested.get("gpu_nodes", -1)),
            cpu_nodes=int(requested.get("cpu_nodes", -1)),
            gpu_platform=str(requested.get("gpu_platform") or ""),
            gpu_preset=str(requested.get("gpu_preset") or ""),
            gpu_driver_mode=str(requested.get("gpu_driver_mode") or ""),
            gpu_workload_profile=str(requested.get("gpu_workload_profile") or ""),
            managed_driver_preset=str(requested.get("managed_driver_preset") or ""),
            gpu_health_stabilization_seconds=int(requested.get("gpu_health_stabilization_seconds", 120)),
            gpu_health_timeout_minutes=int(requested.get("gpu_health_timeout_minutes", 60)),
            gpu_cuda_smoke=bool(requested.get("gpu_cuda_smoke", True)),
            gpu_cuda_smoke_image=str(requested.get("gpu_cuda_smoke_image") or "nvcr.io/nvidia/k8s/cuda-sample:vectoradd-cuda12.5.0-ubuntu22.04"),
            mig_enabled=(bool(mig_mapping.get("enabled", True)) if mig_mapping else bool(mig_value)),
            mig_strategy=str(mig_mapping.get("strategy") or requested.get("mig_strategy") or "mixed"),
            mig_config=str(mig_mapping.get("config") or requested.get("mig_config") or "all-balanced"),
            capacity_block_group=str(requested.get("capacity_block_group") or ""),
            preemptible=preemptible,
        )
        payload = result.to_dict()
        payload["ok"] = True
        payload["dry_run"] = dry_run
        return payload
    except (TypeError, ValueError) as exc:
        return {{"ok": False, "status": "invalid", "error": str(exc), "dry_run": dry_run}}
    except Exception as exc:
        return {{"ok": False, "status": "error", "error": str(exc), "dry_run": dry_run}}


def _write_soperator_temp_spec(spec_text: str) -> Path:
    tmp_dir = Path("/tmp/npa-agent-soperator")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    path = tmp_dir / f"soperator-{{secrets.token_hex(8)}}.yaml"
    path.write_text(spec_text, encoding="utf-8")
    return path


def _soperator_spec_text_from_payload(body: dict) -> str:
    # Network-reachable: never accept local file paths (LFI). Specs must be
    # supplied inline as YAML text or a mapping.
    spec_text = str(body.get("spec_yaml") or body.get("yaml") or "").strip()
    if spec_text:
        return spec_text
    spec = body.get("spec")
    if isinstance(spec, dict):
        return yaml.safe_dump(spec, sort_keys=False)
    raise HTTPException(status_code=400, detail="Provide spec_yaml, yaml, or spec")


def _soperator_validate_payload(body: dict) -> dict:
    try:
        from npa.soperator.spec import SoperatorSpecError, spec_from_mapping

        spec_text = _soperator_spec_text_from_payload(body)
        loaded = yaml.safe_load(spec_text) or {{}}
        spec = spec_from_mapping(loaded)
        spec.validate()
        return {{
            "ok": True,
            "apiVersion": "npa.soperator/v0.0.1",
            "name": spec.name,
            "region": spec.region,
            "control_plane": {{
                "system_min_size": spec.system_min_size,
                "system_max_size": spec.effective_system_max_size(),
            }},
            "worker_pools": [pool.name for pool in spec.workers],
            "docker_cache_pools": [pool.name for pool in spec.workers if pool.docker_cache],
            "workers": [
                {{
                    "name": pool.name,
                    "platform": pool.platform,
                    "preset": pool.preset,
                    "size": pool.size,
                    "preemptible": pool.preemptible,
                    "capacity_mode": pool.capacity_mode(),
                    "reservation_selector": pool.reservation_selector_kind() or None,
                    "docker_cache": pool.docker_cache,
                }}
                for pool in spec.workers
            ],
        }}
    except HTTPException:
        raise
    except Exception as exc:
        return {{"ok": False, "error": str(exc), "apiVersion": "npa.soperator/v0.0.1"}}


def _soperator_deploy_from_payload(body: dict) -> dict:
    from npa.soperator.lifecycle import (
        SoperatorDeploymentValidationError,
        _validate_gpu_creation_check_timeout,
        _validate_immutable_solutions_library_ref,
    )
    from npa.soperator.spec import DEFAULT_SOLUTIONS_LIBRARY_REF

    ready, reason = _agent_npa_ready()
    if not ready:
        return {{"ok": False, "status": "blocked", "error": reason}}
    dry_run = bool(body.get("dry_run", False))
    validation = _soperator_validate_payload(body)
    if not validation.get("ok"):
        return {{"ok": False, "status": "invalid", "validation": validation}}
    try:
        ref = _validate_immutable_solutions_library_ref(
            str(
                body.get("ref")
                or body.get("solutions_library_ref")
                or DEFAULT_SOLUTIONS_LIBRARY_REF
            )
        )
    except ValueError as exc:
        return {{"ok": False, "status": "invalid", "error": str(exc), "validation": validation}}
    try:
        raw_gpu_timeout = body.get("gpu_creation_check_timeout_seconds")
        gpu_creation_check_timeout_seconds = (
            30 * 60 if raw_gpu_timeout is None else int(raw_gpu_timeout)
        )
        _validate_gpu_creation_check_timeout(gpu_creation_check_timeout_seconds)
    except (TypeError, ValueError) as exc:
        return {{"ok": False, "status": "invalid", "error": str(exc), "validation": validation}}
    if dry_run:
        return {{
            "ok": True,
            "status": "dry-run",
            "dry_run": True,
            "validation": validation,
            "command": "npa soperator deploy --spec <validated-spec> --output json",
            "solutions_library_ref": ref,
            "gpu_creation_check_timeout_seconds": gpu_creation_check_timeout_seconds,
        }}
    try:
        timeout_minutes = _agent_mk8s_numeric(
            body.get("timeout_minutes") or body.get("timeout") or 90,
            field="timeout_minutes",
            minimum=1,
        )
    except (TypeError, ValueError) as exc:
        return {{
            "ok": False,
            "status": "invalid",
            "error": str(exc),
            "validation": validation,
        }}
    project = _agent_project_alias(str(body.get("project") or ""))
    terraform_dir_text = str(body.get("terraform_dir") or "").strip()
    terraform_dir = Path(terraform_dir_text).expanduser() if terraform_dir_text else None
    apply_fixes = bool(body.get("apply_fixes", True))
    spec_path = _write_soperator_temp_spec(_soperator_spec_text_from_payload(body))
    try:
        from npa.soperator.lifecycle import deploy_cluster
        from npa.soperator.spec import load_spec

        spec = load_spec(spec_path)
        result = deploy_cluster(
            spec,
            terraform_dir=terraform_dir,
            solutions_library_ref=ref,
            project=project or None,
            timeout_minutes=timeout_minutes,
            gpu_creation_check_timeout_seconds=gpu_creation_check_timeout_seconds,
            apply_fixes=apply_fixes,
            stream_terraform_output=False,
            on_status=lambda msg: None,
        )
        return {{
            "ok": True,
            "status": "deployed",
            "dry_run": False,
            "validation": validation,
            "result": result,
        }}
    except SoperatorDeploymentValidationError as exc:
        return {{
            "ok": False,
            "status": "degraded-validation",
            "error": str(exc),
            "validation": validation,
            "result": exc.result,
        }}
    except Exception as exc:
        return {{"ok": False, "status": "error", "error": str(exc), "validation": validation}}
    finally:
        try:
            spec_path.unlink(missing_ok=True)
        except Exception:
            pass


def _soperator_status_payload(name: str) -> dict:
    cluster_name = str(name or "").strip()
    if not cluster_name:
        raise HTTPException(status_code=400, detail="name is required")
    return _run_agent_npa_json(["soperator", "status", "--name", cluster_name, "--output", "json"], timeout_s=60)


def _workflow_no_infra_response(*, validation: dict, plan: dict, run_id: str, infra: dict) -> dict:
    return {{
        "ok": False,
        "run_id": run_id,
        "submitted_at": _now_iso(),
        "name": str(validation.get("name") or ""),
        "validation": validation,
        "plan": plan,
        "infra": infra,
        "submit_mode": "blocked-no-infra",
        "reason": "no infra is specified or available",
        "message": (
            "No Kubernetes infra is specified or available for this workflow. "
            "Choose one option: let the agent deploy minimal Kubernetes infra, "
            "configure an existing backend in ~/.npa/config.yaml, or pass project/cluster_name in the submit payload."
        ),
        "options": infra.get("options", []),
    }}
_SKILL_CACHE = {{"loaded_at": 0.0, "index": {{}}, "root": Path("/")}}
_INTENT_SKILLS = {{
    "onboard_solution": ("byof-onboard", "oss-solution-registry-onboard"),
    "find_artifacts": ("find-artifacts",),
    "create_workflow": ("author-npa-workflow",),
    "create_vlm_rl_workflow": ("author-npa-workflow", "sim-to-real"),
    "create_gate_workflow": ("author-npa-workflow", "sim-to-real"),
    "create_data_factory_workflow": ("physical-ai-data-factory", "author-npa-workflow"),
    "live_infra_loop": ("submit-workflow", "gpu-selection"),
    "mk8s_provision": ("nebius-infra", "submit-workflow"),
    "soperator": ("soperator", "nebius-infra"),
    "cosmos3": ("cosmos3-setup", "cosmos3-npa-workflow"),
    "start_sim2real": ("sim2real-operate", "sim2real-engine"),
    "sim2real_status": ("sim2real-operate",),
    "watch_sim": ("sim2real-operate",),
}}

def _skill_index_candidates() -> list[Path]:
    return [
        Path("/opt/npa-agent/repo/skills/index.yaml"),
        Path("/workspace/skills/index.yaml"),
        Path.cwd() / "skills" / "index.yaml",
    ]

def _load_skill_index() -> tuple[dict[str, str], Path]:
    now = time.monotonic()
    cache = _SKILL_CACHE
    if cache.get("loaded_at", 0.0) > 0 and now - float(cache.get("loaded_at", 0.0)) < 60.0:
        return (
            dict(cache.get("index", {{}})) if isinstance(cache.get("index"), dict) else {{}},
            cache.get("root") if isinstance(cache.get("root"), Path) else Path("/"),
        )
    for candidate in _skill_index_candidates():
        if not candidate.is_file():
            continue
        try:
            payload = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {{}}
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        skills = payload.get("skills")
        if not isinstance(skills, list):
            continue
        # Paths are repo-root-relative: base on the dir CONTAINING skills/.
        root = candidate.parent.parent
        index: dict[str, str] = {{}}
        for entry in skills:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name") or "").strip()
            rel_path = str(entry.get("path") or "").strip()
            if not name or not rel_path:
                continue
            index[name] = rel_path
        cache["loaded_at"] = now
        cache["index"] = index
        cache["root"] = root
        return index, root
    cache["loaded_at"] = now
    cache["index"] = {{}}
    cache["root"] = Path("/")
    return {{}}, Path("/")

def _skill_excerpt(skill_name: str, *, max_chars: int = 900) -> str:
    index, root = _load_skill_index()
    rel_path = str(index.get(skill_name) or "").strip()
    if not rel_path:
        return ""
    path = root / rel_path
    if not path.is_file():
        return ""
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return ""
    excerpt = "\\n".join(line for line in text.splitlines() if line.strip())[:max_chars].strip()
    return excerpt

def _resolve_skill_context(*, user_text: str, intent: str | None) -> tuple[list[str], str]:
    names: list[str] = []
    if intent and intent in _INTENT_SKILLS:
        for name in _INTENT_SKILLS[intent]:
            if name not in names:
                names.append(name)
    lowered = str(user_text or "").lower()
    if "artifact" in lowered and "find-artifacts" not in names:
        names.append("find-artifacts")
    if ("workflow" in lowered or "yaml" in lowered) and "author-npa-workflow" not in names:
        names.append("author-npa-workflow")
    names[:0] = [n for n in skill_names_for_keywords(lowered) if n not in names]
    if (
        "npa-visual-feedback" in lowered
        or "describe this" in lowered
        or "visual feedback" in lowered
    ) and "agent-visual-feedback" not in names:
        names.insert(0, "agent-visual-feedback")
    snippets: list[str] = []
    for name in names[:4]:
        excerpt = _skill_excerpt(name)
        if excerpt:
            snippets.append(f"[skill:{{name}}]\\n{{excerpt}}")
    if not snippets:
        return names, ""
    return names, "Relevant NPA skill excerpts:\\n\\n" + "\\n\\n".join(snippets)

def _last_user_message(raw_messages: list) -> str:
    return text_from_messages(raw_messages)

def _dedupe(values: list[str]) -> list[str]:
    unique: list[str] = []
    for value in values:
        token = str(value or "").strip()
        if token and token not in unique:
            unique.append(token)
    return unique

def _maybe_toolground_chat_reply(
    user_text: str,
) -> tuple[str | None, list[str], list[str], str | None, dict | None, str | None]:
    intent = match_chat_intent(user_text)
    if not intent and re.search(r"\\bworkflow\\b.*\\b(?:yaml|spec)\\b", str(user_text or ""), re.IGNORECASE):
        intent = "create_workflow"
    if not intent:
        return None, [], [], None, None, None
    state = _load_state()
    suggested_apis = apis_for_intent(intent)
    apis_used: list[str] = []
    loaded_now = False
    rerun_ready = None
    default_cameras = list(DEFAULT_SCENE_SPEC.get("cameras", {{}}).values())
    if intent == "start_sim2real":
        submit = submit_sim2real({{}})
        apis_used.append("workflows/sim2real/submit")
        run_id = str(submit.get("run_id") or "")
        reply = (
            "**Started Sim2Real pipeline**\\n"
            f"- **run_id**: `{{run_id}}`\\n"
            f"- **submit_mode**: `{{submit.get('submit_mode') or submit.get('mode') or 'agent-local-sim2real'}}`\\n"
            "- Default agent submit is **local/demo** unless live K8s Sim2Real hooks succeed.\\n"
            "- The Stages panel will update stage timeline, result, and logs; Rerun will switch to the run recording when it is written.\\n"
            "- Full staged K8s Sim2Real still runs via operator skills / `npa workbench` on the operator machine."
        )
        return reply, _dedupe(apis_used), suggested_apis, None, submit, intent
    if intent == "find_artifacts":
        mentioned_run = ""
        match = re.search(r"\\b(agent-run-[A-Za-z0-9_-]+|sim2real-[A-Za-z0-9_.:-]+)\\b", str(user_text or ""))
        if match:
            mentioned_run = match.group(1)
        try:
            if mentioned_run:
                listed = artifacts_for_run(mentioned_run)
                apis_used.append("artifacts/run/{{run_id}}")
                if isinstance(listed, JSONResponse):
                    payload = json.loads(listed.body.decode("utf-8"))
                else:
                    payload = listed
                count = int(payload.get("count") or 0)
                preferred = payload.get("preferred") if isinstance(payload.get("preferred"), dict) else {{}}
                if count <= 0:
                    reply = (
                        "**No S3 artifacts found for that run.**\\n"
                        f"- **run_id**: `{{mentioned_run}}`\\n"
                        f"- **S3 prefix**: `{{payload.get('prefix', '')}}`\\n"
                        "- It may predate S3 upload support or belong to a destroyed agent VM."
                    )
                    return reply, _dedupe(apis_used), suggested_apis, None, payload, intent
                reply = (
                    "**Run artifacts found.**\\n"
                    f"- **run_id**: `{{mentioned_run}}`\\n"
                    f"- **artifact_count**: `{{count}}`\\n"
                    f"- **preferred**: `{{preferred.get('key', '')}}`\\n"
                    f"- **render**: `{{preferred.get('render', '')}}`"
                )
                return reply, _dedupe(apis_used), suggested_apis, None, payload, intent
            page = artifacts_runs(limit=5)
            apis_used.append("artifacts/runs")
            if isinstance(page, JSONResponse):
                payload = json.loads(page.body.decode("utf-8"))
            else:
                payload = page
            rows = payload.get("runs") if isinstance(payload, dict) else []
            latest = rows[0] if isinstance(rows, list) and rows else {{}}
            latest_run = str(latest.get("run_id") or "")
            if not latest_run:
                reply = (
                    "**No S3-backed Sim2Real runs are discoverable yet.**\\n"
                    f"- **S3 prefix**: `{{payload.get('prefix', '') if isinstance(payload, dict) else ''}}`"
                )
                return reply, _dedupe(apis_used), suggested_apis, None, payload if isinstance(payload, dict) else {{}}, intent
            details = artifacts_for_run(latest_run)
            apis_used.append("artifacts/run/{{run_id}}")
            if isinstance(details, JSONResponse):
                details_payload = json.loads(details.body.decode("utf-8"))
            else:
                details_payload = details
            preferred = details_payload.get("preferred") if isinstance(details_payload.get("preferred"), dict) else {{}}
            reply = (
                "**Use this S3-backed Sim2Real run.**\\n"
                f"- **run_id**: `{{latest_run}}`\\n"
                f"- **artifact_count**: `{{latest.get('artifact_count', '')}}`\\n"
                f"- **preferred_artifact**: `{{preferred.get('key', '')}}`\\n"
                f"- **render**: `{{preferred.get('render', '')}}`\\n"
                "- In the UI, paste this run id or select it from **Runs & artifacts** (latest first), then **List artifacts**."
            )
            return reply, _dedupe(apis_used), suggested_apis, None, details_payload, intent
        except Exception as exc:
            reply = f"**Artifact discovery failed.**\\n- **error**: `{{exc}}`"
            return reply, _dedupe(apis_used), suggested_apis, None, {{"ok": False, "error": str(exc)}}, intent
    if intent == "load_franka":
        sim_viz = state.get("sim_viz", {{}})
        if not isinstance(sim_viz, dict):
            sim_viz = {{}}
        rerun_ready = _rerun_ready_state(rrd_uri=str(sim_viz.get("rrd_uri") or ""))
        if not rerun_ready:
            selected = state.get("camera_selection", ["workspace"])
            cam = str(selected[0] if isinstance(selected, list) and selected else "workspace")
            _wire_franka_demo(state, camera=cam, force_local_demo=True)
            apis_used.append("sim-viz/load-franka-demo")
            state = _load_state()
            loaded_now = True
            sim_viz = state.get("sim_viz", {{}})
            if not isinstance(sim_viz, dict):
                sim_viz = {{}}
            rerun_ready = _rerun_ready_state(rrd_uri=str(sim_viz.get("rrd_uri") or ""))
    elif intent in {"sim2real_status", "watch_sim"}:
        # Ground watch/status replies on the same payload exposed by
        # GET /api/sim-viz/status so chat mirrors the live iframe panel.
        try:
            live_status = sim_viz_status()
            apis_used.append("sim-viz/status")
            if isinstance(live_status, dict):
                state["sim_viz"] = dict(live_status)
                _save_state(state)
        except Exception:
            live_status = None
        sim_viz = state.get("sim_viz", {{}})
        if not isinstance(sim_viz, dict) and isinstance(live_status, dict):
            sim_viz = dict(live_status)
        if not isinstance(sim_viz, dict):
            sim_viz = {{}}
        rerun_ready = _rerun_ready_state(rrd_uri=str(sim_viz.get("rrd_uri") or ""))
    elif intent == "foxglove_viewer":
        # Ground the reply on the same payload the viewer pane mounts from.
        try:
            state["foxglove"] = foxglove_status_payload(
                _foxglove_config(state),
                state.get("sim_viz") if isinstance(state.get("sim_viz"), dict) else {{}},
            )
            apis_used.append("foxglove/config")
            apis_used.append("foxglove/status")
        except Exception:
            state["foxglove"] = {{}}
    elif intent in {"infra_backends", "mk8s_provision"}:
        state["infra"] = _agent_k8s_backends()
        _save_state(state)
    elif intent == "tenant_resources":
        state["resources"] = _tenant_resource_inventory()
        apis_used.append("resources")
    elif intent == "list_recordings":
        try:
            runs_payload = sim_viz_runs()
            apis_used.append("sim-viz/runs")
            if isinstance(runs_payload, dict):
                state["sim_viz_runs"] = runs_payload.get("runs") or runs_payload.get("items") or []
            recordings_payload = sim_viz_recordings()
            apis_used.append("sim-viz/recordings")
            if isinstance(recordings_payload, dict):
                state["sim_viz_recordings"] = (
                    recordings_payload.get("recordings")
                    or recordings_payload.get("items")
                    or recordings_payload.get("files")
                    or []
                )
            live_status = sim_viz_status()
            apis_used.append("sim-viz/status")
            if isinstance(live_status, dict):
                state["sim_viz"] = dict(live_status)
            _save_state(state)
        except Exception:
            pass
    elif intent == "sim_assets":
        try:
            selection = get_sim_assets_selection()
            apis_used.append("sim-assets/selection")
            if isinstance(selection, dict):
                state["selection"] = dict(selection)
                _save_state(state)
            catalog = sim_assets()
            apis_used.append("sim-assets")
            if isinstance(catalog, dict):
                state["sim_assets_catalog"] = catalog
                _save_state(state)
        except Exception:
            pass
    elif intent == "cameras":
        try:
            cameras_payload = sim_assets_cameras()
            apis_used.append("sim-assets/cameras")
            if isinstance(cameras_payload, dict):
                cams = cameras_payload.get("cameras") or cameras_payload.get("items") or []
                if isinstance(cams, list) and cams:
                    default_cameras = cams
                state["cameras"] = cameras_payload
                _save_state(state)
        except Exception:
            pass
    elif intent in {{
        "create_workflow",
        "create_vlm_rl_workflow",
        "create_gate_workflow",
        "create_loop_gate_workflow",
        "create_rl_policy_workflow",
        "create_data_factory_workflow",
    }}:
        draft = None
        # Prefer catalog composition when a goal names a real tool; otherwise
        # fall back to the intent-specific templates.
        if goal_requests_catalog_composition(user_text):
            from npa.cli.agent_workflow import author_workflow_from_goal

            authored = author_workflow_from_goal(user_text, tool_refs=frozenset(TOOL_REFS))
            if authored.get("matched_tool_refs"):
                draft = authored
        if draft is None:
            infra_context = _agent_k8s_backends()
            s3_context = _agent_s3_settings()
            draft = generate_workflow_draft(
                user_text=user_text,
                intent=intent,
                bucket=str(s3_context.get("bucket") or ""),
                tool_refs=frozenset(TOOL_REFS),
                capabilities={{"tool_refs": list(TOOL_REFS)}},
                infrastructure=infra_context,
            )
        yaml_text = str(draft.get("yaml") or "").strip()
        validation = draft.get("validation") if isinstance(draft.get("validation"), dict) else {{}}
        plan = draft.get("plan") if isinstance(draft.get("plan"), dict) else {{}}
        runnable = bool(draft.get("runnable"))
        template = str(draft.get("template") or "two-step")
        _save_workflow_draft(state, yaml_text, validation, plan=plan, runnable=runnable)
        state["workflow_draft"]["template"] = template
        _save_state(state)
        apis_used.extend(["workflows/draft", "workflows/validate", "workflows/plan"])
        if not runnable:
            context_errors = draft.get("context_errors") if isinstance(draft.get("context_errors"), list) else []
            fail_reason = str(
                validation.get("error")
                or plan.get("error")
                or ("; ".join(str(item) for item in context_errors) if context_errors else "")
                or "validate+plan gate did not pass"
            )
            reply = (
                "**Could not generate runnable workflow YAML yet.**\\n"
                f"- **reason**: `{{fail_reason}}`\\n"
                "- Adjust your request or template details and retry;"
                " chat returns YAML only after both validation and planning succeed."
            )
            return reply, _dedupe(apis_used), suggested_apis, None, {{"ok": False, "validation": validation, "plan": plan}}, intent
        drop_note = str(draft.get("dropped_stages_note") or "").strip()
        reply = format_workflow_chat_reply(
            yaml_text,
            validation,
            template=template,
            plan=plan,
            runnable=runnable,
            dropped_stages_note=drop_note,
            warnings=draft.get("warnings") if isinstance(draft.get("warnings"), list) else [],
        )
        return reply, _dedupe(apis_used), suggested_apis, yaml_text, validation, intent
    if intent in {{
        "onboard_solution",
        "tools_catalog",
        "component_capabilities",
        "cosmos_capabilities",
        "lancedb_capabilities",
        "sonic_capabilities",
        "lerobot_capabilities",
        "groot_capabilities",
        "genesis_capabilities",
        "mjlab_capabilities",
        "isaac_lab_capabilities",
        "live_infra_loop",
        "workflow_execute_guidance",
        "soperator",
        "mk8s_provision",
        "cosmos3",
    }}:
        apis_used.append("tools")
    if intent in {{"soperator", "mk8s_provision"}}:
        apis_used.extend(suggested_apis)
    reply = build_grounded_reply(
        intent,
        state,
        TOOL_REFS,
        rerun_ready=rerun_ready,
        loaded_franka_now=loaded_now,
        default_cameras=default_cameras,
    )
    return reply, _dedupe(apis_used), suggested_apis, None, None, intent

def _sim2real_stage_count_from_report(state: dict[str, Any]) -> int:
    # Derive Sim2Real stage count from the active staged report when available.
    sim_viz = state.get("sim_viz", {{}})
    latest = state.get("latest_submit", {{}})
    run_id = ""
    if isinstance(sim_viz, dict):
        run_id = str(sim_viz.get("run_id") or "").strip()
    if not run_id and isinstance(latest, dict):
        run_id = str(latest.get("run_id") or "").strip()
    if not run_id:
        return 0
    report_path = Path("/opt/npa-agent/reports") / run_id / "sim2real-report.json"
    try:
        report = json.loads(report_path.read_text())
    except Exception:
        return 0
    artifacts = report.get("s3_artifacts") if isinstance(report, dict) else {{}}
    if isinstance(artifacts, dict):
        stage_keys = [str(key) for key in artifacts if str(key).startswith("stage_")]
        if stage_keys:
            return len(stage_keys)
    records = report.get("stage_records") if isinstance(report, dict) else []
    if isinstance(records, list) and records:
        return len(records)
    return 0


def _maybe_stage_count_numeric_reply(user_text: str, state: dict[str, Any]) -> str | None:
    lowered = str(user_text or "").lower()
    if not re.search(r"\\b(?:sim\\s*[- ]?2\\s*[- ]?real|sim2real|pipeline|workflow)\\b", lowered):
        return None
    if not re.search(r"\\b(?:stage|stages|step|steps)\\b", lowered):
        return None
    if not re.search(r"\\b(?:count|number|how many)\\b", lowered):
        return None
    value = _sim2real_stage_count_from_report(state)
    if value <= 0:
        return None
    match = re.search(r"(?:count|number|stages?|steps?)\\s*(?:-|minus)\\s*(\\d+)", lowered)
    if match:
        value -= int(match.group(1))
    return str(value)

# Plain-string origin-question detector (no regex: this code is embedded verbatim
# inside the backend f-string, where backslash escapes are unsafe).
_ORIGIN_NOUNS = (
    "image", "images", "frame", "frames", "input", "inputs", "scene",
    "footage", "clip", "clips", "photo", "photos", "picture", "pictures",
    "visual", "render", "data",
)
_ORIGIN_WORDS = (
    "original", "source", "raw", "initial", "underlying", "starting",
    "came from", "come from", "comes from", "where did", "where does",
)

def _origin_question(user_text: str) -> bool:
    t = str(user_text or "").lower()
    if not t:
        return False
    has_noun = any(n in t for n in _ORIGIN_NOUNS)
    has_origin = any(w in t for w in _ORIGIN_WORDS)
    if has_origin and has_noun:
        return True
    if "input" in t and ("what" in t or "which" in t):
        return True
    if "cosmos" in t and ("transform" in t or "input" in t or "start" in t):
        return True
    return False

def _maybe_origin_reply(user_text: str, *, visual_context=None, state=None):
    # Grounded answer to "what was the original input image?" — resolved from the
    # active run's REAL artifacts (source frames if stored; otherwise the earliest
    # stored visuals + augment engine + what the VLM labeled), never guessed.
    if not _origin_question(user_text):
        return None, []
    vc = visual_context if isinstance(visual_context, dict) else {{}}
    run_id = str(vc.get("run_id") or "").strip()
    if not run_id:
        st = state if isinstance(state, dict) else _load_state()
        sv = _sim_viz_for_run(st)
        run_id = str(sv.get("run_id") or "").strip()
    if not run_id or run_id == "franka-demo":
        return None, []
    try:
        normalized_run = validate_run_id(run_id)
    except Exception:
        return None, []
    try:
        s3, settings = _agent_s3_client()
        artifacts = find_run_artifacts(
            settings["bucket"],
            base_prefix=settings.get("prefix", ""),
            run_id=normalized_run,
            s3=s3,
        )
        keys = [str(a.key or "") for a in artifacts]
        if not keys:
            return None, []

        def _read_json(key: str):
            if not key:
                return None
            try:
                body = s3.get_object(Bucket=settings["bucket"], Key=key)["Body"].read()
                return json.loads(body)
            except Exception:
                return None

        origin = build_run_origin(keys, run_id=normalized_run, read_json=_read_json)
        summary = str(origin.get("summary") or "").strip()
        if not summary:
            return None, []
        return summary, ["artifacts/provenance/" + normalized_run]
    except Exception:
        return None, []

def _agent_chat_with_tools(*, raw_messages: list, model: str) -> dict | None:
    last_user = _last_user_message(raw_messages)
    if not last_user:
        return None
    numeric_reply = _maybe_stage_count_numeric_reply(last_user, _load_state())
    if numeric_reply is not None:
        return {{
            "ok": True,
            "model": "grounded",
            "reply": numeric_reply,
            "reasoning": None,
            "grounded": True,
            "apis_used": ["reports/sim2real-report.json"],
        }}
    tool_reply, apis_used, apis_suggested, workflow_yaml, workflow_validation, intent = _maybe_toolground_chat_reply(last_user)
    if not tool_reply:
        return None
    skill_names, _ = _resolve_skill_context(user_text=last_user, intent=intent)
    payload = {{
        "ok": True,
        "model": "grounded",
        "reply": tool_reply,
        "reasoning": None,
        "grounded": True,
        "apis_used": apis_used,
        "apis_suggested": apis_suggested,
        "skills_used": skill_names,
    }}
    if workflow_yaml:
        payload["workflow_yaml"] = workflow_yaml
    if isinstance(workflow_validation, dict):
        payload["workflow_validation"] = workflow_validation
        draft = _workflow_draft_from_state(_load_state())
        if isinstance(draft, dict) and draft.get("yaml"):
            payload["workflow_draft"] = draft
    return payload

# In-process cache for the semantic intent fallthrough so repeated paraphrases
# short-circuit to 0 tokens after the first classification.
_SEMANTIC_INTENT_CACHE = {{}}

def _semantic_route(user_text: str) -> dict:
    known = frozenset(INTENT_APIS.keys())

    def _model_call(messages, tier="cheap"):
        data, _provider, _model = _chat_with_resilience(
            messages=messages, tier=tier, interactive=True
        )
        return data

    try:
        return classify_intent_semantic(
            user_text,
            known_intents=known,
            model_call=_model_call,
            cache=_SEMANTIC_INTENT_CACHE,
        )
    except Exception:
        return {{"intent": None, "mode": "none", "confidence": 0.0, "tokens": 0, "source": "none"}}

@app.post("/chat")
def chat(payload: dict):
    raw_messages = payload.get("messages", [])
    if not isinstance(raw_messages, list) or not raw_messages:
        raise HTTPException(status_code=400, detail="messages must be a non-empty list")
    # Normalize UI "Auto" / explicit "auto" to empty so cost-tier routing runs.
    _raw_model = str(payload.get("model") or "").strip()
    if _raw_model.lower() in {{"", "auto"}}:
        explicit_model_override = ""
        model = ""
    else:
        explicit_model_override = _raw_model
        model = _raw_model
    visual_context = payload.get("visual_context") if isinstance(payload.get("visual_context"), dict) else {{}}
    visual_kind = normalize_visual_kind(
        str(visual_context.get("kind") or visual_context.get("visual_kind") or "")
    )
    # Preserve multimodal parts for Token Factory; storage uses text stubs only.
    llm_messages = normalize_messages_for_llm(raw_messages)
    last_content = text_from_messages(llm_messages) or _last_user_message(raw_messages)
    visual_turn = is_visual_feedback_turn(
        user_text=last_content,
        messages=llm_messages,
        visual_context=visual_context,
    )
    state = _load_state()
    session_id = _sanitize_chat_session_id(
        str(payload.get("session_id") or state.get("active_chat_session_id") or "default")
    )
    session = _get_chat_session(state, session_id)
    history = normalize_messages_for_storage(llm_messages, visual_kind=visual_kind)
    if len(history) <= 1 and isinstance(session.get("chat_history"), list):
        prior = normalize_messages_for_storage(session.get("chat_history", []))
        if history:
            history = [*prior, history[-1]]
            if llm_messages:
                llm_messages = normalize_messages_for_llm([*prior, llm_messages[-1]])
        else:
            history = prior
            llm_messages = normalize_messages_for_llm(prior)
    # Preserve merged session history across the LLM path (do not rebuild from a
    # short client payload and wipe prior turns after the model returns).
    merged_history = list(history)
    pending_access = state.get("access_approval")
    if not isinstance(pending_access, dict):
        pending_access = {{}}
    # Describe-this/multimodal turns must reach the visual path even when their
    # scene metadata happens to contain words such as model, dataset, catalog,
    # or approval.  Match the other grounded shortcuts: never classify a visual
    # turn as a deterministic access-approval conversation.
    access_action = "" if visual_turn else _access_approval.classify_followup(
        last_content, has_pending_plan=bool(pending_access)
    )
    if access_action:
        open_urls = []
        if access_action in {{"plan", "recheck"}}:
            access_plan = _access_approval.build_plan(
                capabilities=None,
                resume_command="npa configure --prepare-catalog-access",
                state_path=Path("/opt/npa-agent/access-approvals.json"),
                force=access_action == "recheck",
            )
            state["access_approval"] = access_plan
            reply = _access_approval.format_plan_reply(access_plan)
        elif access_action == "open":
            access_plan = pending_access
            open_urls = [
                str(url)
                for url in (access_plan.get("official_urls") or [])
                if str(url).startswith("https://")
            ]
            reply = _access_approval.format_open_reply(access_plan)
            state["access_approval"] = {{**access_plan, "pages_opened": True}}
        else:
            access_plan = pending_access
            reply = _access_approval.format_later_reply(access_plan)
            state["access_approval"] = access_plan
        history = [*merged_history, {{"role": "assistant", "content": reply}}][-80:]
        session.update(
            {{
                "id": session_id,
                "title": str(session.get("title") or _chat_session_title(history)),
                "chat_history": history,
            }}
        )
        session = _save_chat_session(state, session, active=True)
        _save_state(state)
        response = {{
            "ok": True,
            "model": "grounded",
            "reply": reply,
            "reasoning": None,
            "grounded": True,
            "tier": "grounded-access-approval",
            "apis_used": ["access-approvals"],
            "skills_used": ["access-approval"],
            "approval_plan": access_plan,
            "open_urls": open_urls,
            "safe_handoff": access_action in {{"open", "later"}},
            "resume_ready": str(access_plan.get("status") or "") == "ready",
            "session_id": session["id"],
            "session": public_chat_session_payload(session),
        }}
        return response
    # Grounded "where did this come from / what was the original input" answer.
    # Resolved from the active run's real artifacts. For a metadata/text turn we
    # return it directly (deterministic, 0 tokens); for a framed vision turn we
    # inject it so the model's "Where it comes from" is grounded, not guessed.
    origin_reply, origin_apis = _maybe_origin_reply(
        last_content, visual_context=visual_context, state=state
    )
    if origin_reply and not visual_turn and not has_image_content(llm_messages):
        history = [*merged_history, {{"role": "assistant", "content": origin_reply}}][-80:]
        session.update(
            {{
                "id": session_id,
                "title": str(session.get("title") or _chat_session_title(history)),
                "chat_history": history,
            }}
        )
        state = _load_state()
        session = _save_chat_session(state, session, active=True)
        _save_state(state)
        return {{
            "ok": True,
            "model": "grounded",
            "reply": origin_reply,
            "reasoning": None,
            "grounded": True,
            "tier": "grounded-provenance",
            "apis_used": origin_apis,
            "skills_used": ["agent-visual-feedback"],
            "session_id": session["id"],
            "session": public_chat_session_payload(session),
        }}
    # Small Sim2Real chat shortcut — persist the turn (do not return before session save).
    if (not visual_turn) and re.search(
        r"\\b(?:run|start|submit|launch)\\b.{{0,80}}\\b(?:small|simple|tiny|minimal)\\b.{{0,80}}\\bsim(?:\\s*[- ]?2\\s*[- ]?real|2real)\\b",
        last_content,
        re.IGNORECASE,
    ):
        run_id = f"agent-chat-small-{{secrets.token_hex(6)}}"
        submit = submit_sim2real({{"run_id": run_id}})
        live = submit.get("live_submit") if isinstance(submit, dict) else None
        if isinstance(live, dict) and live.get("ok"):
            reply = (
                f"Started small Sim2Real pipeline: **run_id** `{{run_id}}`. "
                f"Live submit session: `{{live.get('session')}}`; log: `{{live.get('log')}}`."
            )
        else:
            detail = str((live or {{}}).get("error") if isinstance(live, dict) else "recorded locally")
            reply = f"Recorded small Sim2Real submit **run_id** `{{run_id}}`; live launch detail: `{{detail}}`."
        history = [*merged_history, {{"role": "assistant", "content": reply}}][-80:]
        session.update(
            {{
                "id": session_id,
                "title": str(session.get("title") or _chat_session_title(history)),
                "chat_history": history,
            }}
        )
        state = _load_state()
        session = _save_chat_session(state, session, active=True)
        _save_state(state)
        return {{
            "ok": True,
            "model": "grounded",
            "reply": reply,
            "reasoning": None,
            "grounded": True,
            "apis_used": ["workflows/sim2real/submit"],
            "submit": submit,
            "session_id": session["id"],
            "session": public_chat_session_payload(session),
        }}
    # Metadata-only Describe-this: grounded reply (never invent pixels). Vision
    # turns with an attached frame fall through to Token Factory.
    if visual_turn and not has_image_content(llm_messages):
        meta_reply = build_metadata_only_visual_reply(visual_context)
        history = [*merged_history, {{"role": "assistant", "content": meta_reply}}][-80:]
        session.update(
            {{
                "id": session_id,
                "title": str(session.get("title") or _chat_session_title(history)),
                "chat_history": history,
            }}
        )
        state = _load_state()
        session = _save_chat_session(state, session, active=True)
        _save_state(state)
        return {{
            "ok": True,
            "model": "grounded",
            "reply": meta_reply,
            "reasoning": None,
            "grounded": True,
            "tier": "grounded-metadata",
            "visual_kind": visual_kind,
            "apis_used": ["sim-viz/status"],
            "skills_used": ["agent-visual-feedback"],
            "session_id": session["id"],
            "session": public_chat_session_payload(session),
        }}
    # Never short-circuit framed Describe-this / vision turns through intent tools.
    tool_result = None if visual_turn else _agent_chat_with_tools(raw_messages=history, model=model)
    if tool_result is not None:
        reply = str(tool_result.get("reply") or "").strip()
        if reply:
            history = [*history, {{"role": "assistant", "content": reply}}][-80:]
        session.update(
            {{
                "id": session_id,
                "title": str(session.get("title") or _chat_session_title(history)),
                "chat_history": history,
            }}
        )
        # Tool handlers may mutate session state (for example starting a Sim2Real
        # run). Reload before saving chat history so an older state snapshot does
        # not clobber the run monitor.
        state = _load_state()
        session = _save_chat_session(state, session, active=True)
        tool_result["session_id"] = session["id"]
        tool_result["session"] = public_chat_session_payload(session)
        _save_state(state)
        return tool_result
    live_ctx = format_live_context_block(_load_state())
    last_user = text_from_messages(llm_messages)
    intent = match_chat_intent(last_user) if not visual_turn else None
    # Semantic fallthrough (Phase D): only when the deterministic regex router
    # missed. Keyword/cache hits cost 0 tokens; a genuine miss spends one cheap
    # structured call. A mapped intent produces a side-effect-free grounded reply.
    semantic_tokens = 0
    if intent is None and not visual_turn:
        semantic = _semantic_route(last_user)
        semantic_tokens = int(semantic.get("tokens") or 0)
        sem_mode = str(semantic.get("mode") or "none")
        mapped = str(semantic.get("intent") or "") if sem_mode == "intent" else ""
        sem_reply = ""
        if mapped:
            sem_reply = build_grounded_reply(mapped, _load_state(), TOOL_REFS)
        elif sem_mode == "action":
            # The turn needs a multi-step tool loop. Drive it inline (the grounded
            # regex + semantic-intent tiers already missed, so this is a genuine
            # fallthrough) so the agent actually *uses* its read-only tools —
            # including the insights backbone — instead of describing an endpoint.
            def _action_model_call(messages, tier="cheap"):
                data, _provider, _model = _chat_with_resilience(
                    messages=messages, tier=tier, interactive=True
                )
                return data

            # Confirmation symmetry with POST /api/agent/act: a follow-up chat
            # turn that carries the minted confirm_token can actually execute the
            # gated action. Only consume the pending gate when a token is present
            # so an unrelated turn never burns it.
            chat_confirm_token = str(payload.get("confirm_token") or "").strip()
            if chat_confirm_token:
                chat_session_token, chat_confirm_digest, _chat_pending = _consume_agent_confirm_token()
            else:
                chat_session_token, chat_confirm_digest = "", ""
            action_result = run_chat_action_loop(
                last_user,
                tools=_agent_act_tools(),
                model_call=_action_model_call,
                tier=classify_tier(last_user),
                confirm_token=chat_confirm_token,
                session_token=chat_session_token,
                confirm_digest=chat_confirm_digest,
                live_context=format_live_context_block(_load_state()),
            )
            # Preserve the safety contract: a state-changing tool proposed from a
            # chat turn never auto-runs without a token — it stops here and we mint
            # a gate token bound to the exact action digest (same as /api/agent/act);
            # the operator re-sends it (with the goal) to execute.
            if action_result.get("needs_confirmation"):
                proposed = action_result.get("proposed_action") if isinstance(action_result.get("proposed_action"), dict) else {{}}
                digest = str(proposed.get("digest") or action_digest({{k: v for k, v in proposed.items() if k != "digest"}}))
                action_result["confirm_token"] = _issue_agent_confirm_token(proposed, digest)
            action_reply = str(action_result.get("reply") or "").strip()
            history = [*merged_history, {{"role": "assistant", "content": action_reply}}][-80:]
            session.update(
                {{
                    "id": session_id,
                    "title": str(session.get("title") or _chat_session_title(history)),
                    "chat_history": history,
                }}
            )
            state = _load_state()
            session = _save_chat_session(state, session, active=True)
            _save_state(state)
            response = {{
                "ok": bool(action_result.get("ok")),
                "model": "grounded",
                "reply": action_reply,
                "reasoning": None,
                "grounded": False,
                "tier": "semantic-action",
                "mode": action_result.get("mode"),
                # Include the tokens spent reaching MODE_ACTION, not just the loop's.
                "usage": {{"total_tokens": int((action_result.get("usage") or {{}}).get("total_tokens", 0)) + semantic_tokens}},
                "steps": action_result.get("steps") or [],
                "tools_used": action_result.get("tools_used") or [],
                "stopped_reason": action_result.get("stopped_reason"),
                "needs_confirmation": bool(action_result.get("needs_confirmation")),
                "proposed_action": action_result.get("proposed_action"),
                "semantic_mode": sem_mode,
                "apis_used": ["agent/act"],
                "session_id": session["id"],
                "session": public_chat_session_payload(session),
            }}
            if action_result.get("confirm_token"):
                response["confirm_token"] = action_result["confirm_token"]
            return response
        if sem_reply:
            grounded_zero = semantic_tokens == 0
            history = [*merged_history, {{"role": "assistant", "content": sem_reply}}][-80:]
            session.update(
                {{
                    "id": session_id,
                    "title": str(session.get("title") or _chat_session_title(history)),
                    "chat_history": history,
                }}
            )
            state = _load_state()
            session = _save_chat_session(state, session, active=True)
            _save_state(state)
            return {{
                "ok": True,
                "model": "grounded",
                "reply": sem_reply,
                "reasoning": None,
                "grounded": grounded_zero,
                "tier": "semantic-" + str(semantic.get("source") or sem_mode),
                "usage": {{"total_tokens": semantic_tokens}},
                "semantic_intent": mapped,
                "semantic_mode": sem_mode,
                "apis_used": [],
                "apis_suggested": apis_for_intent(mapped) if mapped else [],
                "session_id": session["id"],
                "session": public_chat_session_payload(session),
            }}
    # Retrieval grounded-first fallthrough (Blueprint Phase H): after the regex
    # AND semantic routers miss, answer from the indexed docs/skills corpus with
    # cited, extractive grounding (0 generation tokens). Only fires when a corpus
    # is indexed and the top match clears the confidence floor; otherwise /chat is
    # byte-for-byte unchanged.
    if intent is None and not visual_turn:
        retrieved = _maybe_retrieval_grounded(last_user)
        if retrieved is not None:
            reply = str(retrieved.get("answer") or "").strip()
            history = [*merged_history, {{"role": "assistant", "content": reply}}][-80:]
            session.update(
                {{
                    "id": session_id,
                    "title": str(session.get("title") or _chat_session_title(history)),
                    "chat_history": history,
                }}
            )
            state = _load_state()
            session = _save_chat_session(state, session, active=True)
            _save_state(state)
            return {{
                "ok": True,
                "model": "grounded",
                "reply": reply,
                "reasoning": None,
                "grounded": True,
                "tier": "retrieval-grounded",
                "usage": {{"total_tokens": 0}},
                "citations": retrieved.get("citations") or [],
                "apis_used": ["agent/retrieval/search"],
                "session_id": session["id"],
                "session": public_chat_session_payload(session),
            }}
    # Cost-tier routing: vision when an image is attached; otherwise escalate
    # Describe-this metadata-only turns to reasoning (not cheap caption fluff).
    tier = classify_tier(last_user, intent=intent, messages=llm_messages)
    if visual_turn and tier != TIER_VISION:
        tier = TIER_REASONING
    explicit_model = explicit_model_override
    budget_ok, _ = enforce_input_budget(last_user)
    skill_names, skill_ctx = _resolve_skill_context(user_text=last_user, intent=intent)
    if visual_turn and "agent-visual-feedback" not in skill_names:
        skill_names = ["agent-visual-feedback", *skill_names][:4]
        skill_excerpt = _skill_excerpt("agent-visual-feedback")
        if skill_excerpt:
            visual_skill_block = f"[skill:agent-visual-feedback]\\n{{skill_excerpt}}"
            if skill_ctx:
                skill_ctx = skill_ctx + "\\n\\n" + visual_skill_block
            else:
                skill_ctx = "Relevant NPA skill excerpts:\\n\\n" + visual_skill_block
    system_content = _agent_system_prompt() + "\\n\\n" + live_ctx
    visual_block = format_visual_context_block(visual_context)
    if visual_block:
        system_content += "\\n\\n" + visual_block
    if visual_turn:
        system_content += learning_visual_fact_block(visual_context)
    if origin_reply:
        # Ground the "Where it comes from" / original-input story with real facts.
        system_content += (
            "\\n\\nGrounded origin facts for this run (use verbatim for the original "
            "input / 'where it comes from'; do NOT contradict or hedge past these):\\n"
            + origin_reply
        )
    if visual_turn and not has_image_content(llm_messages):
        system_content += (
            "\\n\\nIMPORTANT: No viewer frame image is attached to this turn. "
            "Do not invent pixel content, RGB noise, or scenes. Answer from "
            "metadata/domain hints only and tell the operator how to capture a real frame."
        )
    if skill_ctx:
        system_content += "\\n\\n" + skill_ctx
    messages: list[dict] = [
        {{"role": "system", "content": system_content}}
    ]
    for item in llm_messages:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "user")).strip() or "user"
        content = item.get("content")
        if role == "user" and isinstance(content, str) and content:
            # Guardrail: cap oversized pastes so one turn cannot blow the budget.
            _within, content = enforce_input_budget(content)
        elif role == "user" and isinstance(content, list):
            # Trim text parts inside multimodal (vision) turns; keep image parts.
            trimmed_parts: list[dict] = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                if str(part.get("type") or "") == "text":
                    text_part = str(part.get("text") or "")
                    if text_part:
                        _within, text_part = enforce_input_budget(text_part)
                        trimmed_parts.append({{"type": "text", "text": text_part}})
                else:
                    trimmed_parts.append(part)
            content = trimmed_parts or content
        if content:
            messages.append({{"role": role, "content": content}})
    if len(messages) < 2:
        raise HTTPException(status_code=400, detail="at least one user message is required")
    data, selected_provider, selected_model = _chat_with_resilience(
        messages=messages,
        requested_model=explicit_model,
        tier=tier,
        interactive=True,
    )
    turn_usage = usage_summary(data)
    # Include any tokens spent on the semantic classifier so cost telemetry is
    # not under-reported when the fallthrough consulted the model then fell back.
    if semantic_tokens:
        turn_usage = dict(turn_usage)
        turn_usage["total_tokens"] = int(turn_usage.get("total_tokens", 0)) + semantic_tokens
        turn_usage["semantic_tokens"] = semantic_tokens
    try:
        message = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise HTTPException(status_code=502, detail="LLM response missing assistant message") from exc
    reply, reasoning = _split_reasoning(message)
    if visual_turn and learning_visual_reply_needs_correction(reply, visual_context):
        reply = truthful_learning_visual_reply(visual_context)
        reasoning = None
    if not reply and reasoning:
        reply = reasoning
        reasoning = None
    if not str(reply or "").strip():
        reply = "Model returned no content."
        reasoning = None
    # Re-read session history under the lock and append (do not clobber concurrent turns).
    session = _append_chat_turn(
        session_id,
        merged_history,
        {{"role": "assistant", "content": reply}},
    )
    return {{
        "ok": True,
        "model": selected_model,
        "provider": selected_provider,
        "reply": reply,
        "reasoning": reasoning,
        "tier": tier,
        "usage": turn_usage,
        "input_budget_ok": budget_ok,
        "visual_kind": visual_kind if visual_turn else "",
        "session_id": session["id"],
        "session": public_chat_session_payload(session),
        "skills_used": skill_names,
    }}

def _consume_agent_confirm_token():
    # Single-use consume: return the pending (token, digest, pending_action) and
    # clear the gate in state before any side effect so a replayed request cannot
    # re-authorize. Only call this when the request actually presents a
    # confirm_token, so an unrelated turn never burns a pending gate.
    def consume(state):
        act_state = state.get("agent_act")
        if not isinstance(act_state, dict):
            return "", "", None
        token = str(act_state.get("confirm_token") or "")
        digest = str(act_state.get("confirm_digest") or "")
        pending = act_state.get("pending_action")
        pending = pending if isinstance(pending, dict) else None
        if token:
            act_state["confirm_token"] = ""
            act_state["confirm_digest"] = ""
            act_state["pending_action"] = None
            state["agent_act"] = act_state
        return token, digest, pending

    return _mutate_state(consume)

def _peek_agent_confirm_token():
    state = _load_state()
    act_state = state.get("agent_act")
    if not isinstance(act_state, dict):
        return "", "", None
    token = str(act_state.get("confirm_token") or "")
    digest = str(act_state.get("confirm_digest") or "")
    pending = act_state.get("pending_action")
    return token, digest, pending if isinstance(pending, dict) else None

def _issue_agent_confirm_token(action, digest):
    # Issue a fresh token bound to a specific proposed action digest.
    token = secrets.token_hex(8)

    def issue(state):
        act_state = state.get("agent_act")
        if not isinstance(act_state, dict):
            act_state = {{}}
        act_state["confirm_token"] = token
        act_state["confirm_digest"] = str(digest or "")
        act_state["pending_action"] = action if isinstance(action, dict) else {{}}
        state["agent_act"] = act_state

    _mutate_state(issue)
    return token

def _act_response_to_dict(result) -> dict:
    # Route handlers may return either a plain dict or a JSONResponse; the action
    # loop needs a JSON-serializable observation either way.
    if isinstance(result, JSONResponse):
        try:
            return json.loads(result.body.decode("utf-8"))
        except Exception as exc:
            return {{"error": f"could not decode response: {{exc}}"}}
    if isinstance(result, dict):
        return result
    return {{"value": str(result)}}

def _agent_act_tools():
    def _tool_health(args):
        return {{"ok": True, "tool_refs": len(TOOL_REFS)}}

    def _tool_sim_viz_status(args):
        return _act_response_to_dict(sim_viz_status())

    def _tool_sim2real_status(args):
        return _act_response_to_dict(sim2real_status(run_id=str(args.get("run_id") or "")))

    def _tool_artifacts_runs(args):
        limit = args.get("limit")
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            limit = 10
        return _act_response_to_dict(
            artifacts_runs(
                prefix=str(args.get("prefix") or ""),
                limit=limit,
                q=str(args.get("q") or ""),
            )
        )

    def _tool_artifacts_run(args):
        run_id = str(args.get("run_id") or "").strip()
        if not run_id:
            return {{"error": "run_id is required"}}
        return _act_response_to_dict(artifacts_for_run(run_id))

    def _tool_validate(args):
        # Read-only: use the pure validator/planner so the loop never mutates the
        # persisted workflow draft as a side effect of "validating".
        spec = str(args.get("spec_yaml") or args.get("yaml") or "")
        if not spec.strip():
            return {{"error": "spec_yaml is required"}}
        validation = validate_workflow_yaml_text(spec, tool_refs=frozenset(TOOL_REFS))
        plan = (
            plan_workflow_yaml_text(spec, run_id="agent-act-validate", tool_refs=frozenset(TOOL_REFS))
            if validation.get("ok")
            else {{"ok": False}}
        )
        return {{
            "ok": bool(validation.get("ok")),
            "validation": validation,
            "runnable": bool(validation.get("ok") and plan.get("ok")),
        }}

    def _tool_plan(args):
        spec = str(args.get("spec_yaml") or args.get("yaml") or "")
        if not spec.strip():
            return {{"error": "spec_yaml is required"}}
        plan = plan_workflow_yaml_text(
            spec, run_id=str(args.get("run_id") or "agent-act"), tool_refs=frozenset(TOOL_REFS)
        )
        return {{"ok": bool(plan.get("ok")), "plan": plan}}

    def _insights_store_and_mode():
        settings = _agent_insights_settings()
        endpoint = str(settings.get("endpoint") or "")
        store_uri = str(settings.get("store_uri") or "")
        token_env = str(settings.get("token_env") or "INSIGHTS_TOKEN")
        return endpoint, store_uri, token_env

    def _tool_insights_query(args):
        endpoint, default_store, token_env = _insights_store_and_mode()
        store_uri = str(args.get("input_uri") or default_store or "").strip()
        if not store_uri:
            return {{"error": "insights store is not configured (set NPA_INSIGHTS_STORE_URI or NPA_INSIGHTS_ENDPOINT)"}}
        try:
            raw_value = args.get("threshold_value")
            threshold_value = float(raw_value) if raw_value not in (None, "") else None
        except (TypeError, ValueError):
            threshold_value = None
        try:
            limit = int(args.get("limit") or 100)
        except (TypeError, ValueError):
            limit = 100
        from npa.sdk.workbench import insights as _insights_sdk

        response = _insights_sdk.query(
            input_uri=store_uri,
            workflow=str(args.get("workflow") or ""),
            run_id=str(args.get("run_id") or ""),
            tool=str(args.get("tool") or ""),
            stage=str(args.get("stage") or ""),
            metric_name=str(args.get("metric_name") or ""),
            accelerator=str(args.get("accelerator") or ""),
            metric_kind=str(args.get("metric_kind") or ""),
            currency=str(args.get("currency") or ""),
            cost_basis=str(args.get("cost_basis") or ""),
            score_name=str(args.get("score_name") or ""),
            threshold_metric=str(args.get("threshold_metric") or ""),
            threshold_op=normalize_threshold_op(args.get("threshold_op")),
            threshold_value=threshold_value,
            limit=limit,
            service=bool(endpoint),
            endpoint=endpoint,
            token_env=token_env,
        )
        return response.model_dump(mode="json")

    def _tool_insights_compare(args):
        endpoint, default_store, token_env = _insights_store_and_mode()
        store_uri = str(args.get("input_uri") or default_store or "").strip()
        base_run = str(args.get("base_run") or "").strip()
        candidate_run = str(args.get("candidate_run") or "").strip()
        if not store_uri:
            return {{"error": "insights store is not configured (set NPA_INSIGHTS_STORE_URI or NPA_INSIGHTS_ENDPOINT)"}}
        if not base_run or not candidate_run:
            return {{"error": "base_run and candidate_run are required"}}
        metric_names = args.get("metric_names")
        if isinstance(metric_names, str):
            metric_names = [m.strip() for m in metric_names.split(",") if m.strip()]
        elif isinstance(metric_names, (list, tuple)):
            metric_names = [str(m).strip() for m in metric_names if str(m).strip()]
        else:
            metric_names = []
        from npa.sdk.workbench import insights as _insights_sdk

        response = _insights_sdk.compare(
            input_uri=store_uri,
            base_run=base_run,
            candidate_run=candidate_run,
            metric_names=metric_names,
            service=bool(endpoint),
            endpoint=endpoint,
            token_env=token_env,
        )
        return response.model_dump(mode="json")

    def _tool_insights_lineage(args):
        endpoint, default_store, token_env = _insights_store_and_mode()
        store_uri = str(args.get("input_uri") or default_store or "").strip()
        uri = str(args.get("uri") or "").strip()
        if not store_uri:
            return {{"error": "insights store is not configured (set NPA_INSIGHTS_STORE_URI or NPA_INSIGHTS_ENDPOINT)"}}
        if not uri:
            return {{"error": "uri is required"}}
        try:
            depth = int(args.get("depth") if args.get("depth") is not None else -1)
        except (TypeError, ValueError):
            depth = -1
        from npa.sdk.workbench import insights as _insights_sdk

        response = _insights_sdk.lineage(
            input_uri=store_uri,
            uri=uri,
            version=str(args.get("version") or ""),
            direction=str(args.get("direction") or "both"),
            depth=depth,
            service=bool(endpoint),
            endpoint=endpoint,
            token_env=token_env,
        )
        return response.model_dump(mode="json")

    def _tool_insights_dashboard(args):
        endpoint, default_store, token_env = _insights_store_and_mode()
        store_uri = str(args.get("input_uri") or default_store or "").strip()
        if not store_uri:
            return {{"error": "insights store is not configured (set NPA_INSIGHTS_STORE_URI or NPA_INSIGHTS_ENDPOINT)"}}
        from npa.sdk.workbench import insights as _insights_sdk

        response = _insights_sdk.dashboard(
            input_uri=store_uri,
            workflow=str(args.get("workflow") or ""),
            group_by=normalize_group_by(args.get("group_by")),
            latest_run=str(args.get("latest_run") or ""),
            service=bool(endpoint),
            endpoint=endpoint,
            token_env=token_env,
        )
        return response.model_dump(mode="json")

    def _tool_memory_explain_regression(args):
        baseline_run = str(
            args.get("baseline_run") or args.get("baseline") or args.get("run_a") or ""
        ).strip()
        candidate_run = str(
            args.get("candidate_run") or args.get("candidate") or args.get("run_b") or ""
        ).strip()
        if not baseline_run or not candidate_run:
            return {{"error": "baseline_run and candidate_run are required"}}
        return _agent_run_memory().explain_regression_data(candidate_run, baseline_run)

    def _tool_workflow_author(args):
        goal = str(args.get("goal") or args.get("description") or args.get("prompt") or "").strip()
        if not goal:
            return {{"error": "goal is required to author a workflow"}}
        from npa.cli.agent_workflow import author_workflow_from_goal

        result = author_workflow_from_goal(goal, tool_refs=frozenset(TOOL_REFS))
        # Surface a compact, JSON-serializable observation; yaml is only present
        # (and reported runnable) when validate + plan both pass on real toolRefs.
        return {{
            "ok": bool(result.get("ok")),
            "runnable": bool(result.get("runnable")),
            "yaml": str(result.get("yaml") or "") if result.get("runnable") else "",
            "tool_refs": result.get("tool_refs") or [],
            "padded_tool_refs": result.get("padded_tool_refs") or [],
            "states": result.get("states") or [],
            "validation": result.get("validation") or {{}},
            "error": result.get("error") or (None if result.get("runnable") else "authored spec did not pass validate+plan"),
        }}

    def _tool_submit(args):
        return _act_response_to_dict(submit_sim2real({{"run_id": str(args.get("run_id") or "")}}))

    def _tool_retrieval_search(args):
        query = str(args.get("query") or "").strip()
        if not query:
            return {{"error": "query is required"}}
        try:
            k = max(1, min(int(args.get("k") or RETRIEVAL_TOP_K), 20))
        except (TypeError, ValueError):
            k = RETRIEVAL_TOP_K
        store = _agent_retrieval_store_for_query(query)
        return _retrieval.retrieve(query, embed=_embed_texts, store=store, k=k)

    return {{
        "health": _tool_health,
        "sim_viz_status": _tool_sim_viz_status,
        "sim2real_status": _tool_sim2real_status,
        "artifacts_runs": _tool_artifacts_runs,
        "artifacts_run": _tool_artifacts_run,
        "workflow_validate_spec": _tool_validate,
        "workflow_plan_spec": _tool_plan,
        "retrieval_search": _tool_retrieval_search,
        "insights_query": _tool_insights_query,
        "insights_compare": _tool_insights_compare,
        "insights_lineage": _tool_insights_lineage,
        "insights_dashboard": _tool_insights_dashboard,
        "memory_explain_regression": _tool_memory_explain_regression,
        "workflow_author": _tool_workflow_author,
        "sim2real_submit": _tool_submit,
    }}

@app.post("/agent/act")
def agent_act(payload: dict):
    body = payload if isinstance(payload, dict) else {{}}
    raw_messages = body.get("messages", [])
    goal = str(body.get("goal") or "").strip()
    if not goal and isinstance(raw_messages, list):
        goal = _last_user_message(raw_messages)
    if not goal:
        raise HTTPException(status_code=400, detail="goal or messages is required")
    # Cap the goal so one oversized paste cannot blow the planner budget.
    _budget_ok, goal = enforce_input_budget(goal)
    confirm_token = str(body.get("confirm_token") or "").strip()
    # Only consume (and clear) the pending gate when the operator actually
    # presents a token — an unrelated turn must not burn a pending confirmation.
    if confirm_token:
        session_token, confirm_digest, pending = _consume_agent_confirm_token()
    else:
        session_token, confirm_digest, pending = "", "", None
    try:
        max_steps = int(body.get("max_steps"))
    except (TypeError, ValueError):
        max_steps = DEFAULT_MAX_STEPS
    max_steps = max(1, min(max_steps, 12))

    def _model_call(messages, tier="cheap"):
        data, _provider, _model = _chat_with_resilience(
            messages=messages, tier=tier, interactive=True
        )
        return data

    tier = classify_tier(goal)
    live_ctx = format_live_context_block(_load_state())
    result = run_action_loop(
        goal,
        tools=_agent_act_tools(),
        model_call=_model_call,
        confirm_token=confirm_token,
        session_token=session_token,
        confirm_digest=confirm_digest,
        confirmed_action=pending,
        tier=tier,
        max_steps=max_steps,
        live_context=live_ctx,
    )
    if result.get("needs_confirmation"):
        proposed = result.get("proposed_action") if isinstance(result.get("proposed_action"), dict) else {{}}
        digest = str(proposed.get("digest") or action_digest({{k: v for k, v in proposed.items() if k != "digest"}}))
        result["confirm_token"] = _issue_agent_confirm_token(proposed, digest)
    result["grounded"] = False
    result["mode"] = "agent-act"
    result["allowlist"] = allowlist_specs()
    result["input_budget_ok"] = _budget_ok
    # Phase I: record structured spans for the offline analyzer / injected tracer.
    _record_agent_trace(result)
    return result


from agent_backend.gpu_allocation_routes import (
    GpuAllocationDeps,
    register_gpu_allocation_routes,
)

register_gpu_allocation_routes(
    app,
    GpuAllocationDeps(
        mutate_state=_mutate_state,
        action_digest=action_digest,
    ),
    HTTPException,
)

def _sim2real_gate_metrics(run_id: str, iteration: int) -> dict:
    # Read gate metrics only from real run artifacts; never fabricate a score.
    run_id = str(run_id or "").strip()
    if not run_id:
        return {{}}
    # Real runner writes /opt/npa-agent/runs/<run_id>/reports/sim2real-report.json
    # with an outer_loop.latest_decision / latest_heldout_report schema.
    report_path = Path("/opt/npa-agent/runs") / run_id / "reports" / "sim2real-report.json"
    try:
        report = json.loads(report_path.read_text())
    except Exception:
        report = {{}}
    if not isinstance(report, dict):
        return {{}}
    outer_loop = report.get("outer_loop") if isinstance(report.get("outer_loop"), dict) else {{}}
    decision = outer_loop.get("latest_decision") if isinstance(outer_loop.get("latest_decision"), dict) else {{}}
    heldout = (
        outer_loop.get("latest_heldout_report")
        if isinstance(outer_loop.get("latest_heldout_report"), dict)
        else {{}}
    )
    success_rate = (
        decision.get("success_rate")
        if decision.get("success_rate") is not None
        else heldout.get("success_rate")
    )
    threshold = decision.get("threshold")
    metrics = {{}}
    if success_rate is not None:
        metrics["success_rate"] = success_rate
    if threshold is not None:
        metrics["threshold"] = threshold
    if decision.get("decision"):
        metrics["decision"] = decision.get("decision")
    return metrics

@app.post("/agent/sim2real/drive")
def agent_sim2real_drive(payload: dict):
    body = payload if isinstance(payload, dict) else {{}}
    config = body.get("config") if isinstance(body.get("config"), dict) else {{}}
    goal = str(body.get("goal") or "drive the sim2real outer loop").strip()
    state = _load_state()
    confirm_token = str(body.get("confirm_token") or "").strip()
    # Only consume the pending gate when a token is actually presented (so an
    # unrelated drive request cannot burn it).
    if confirm_token:
        session_token, confirm_digest, pending = _consume_agent_confirm_token()
    else:
        session_token, confirm_digest, pending = "", "", None
    default_run = ""
    sim_viz = state.get("sim_viz", {{}})
    if isinstance(sim_viz, dict):
        default_run = str(sim_viz.get("run_id") or "").strip()
    pending_cfg = pending.get("config") if isinstance(pending, dict) else None
    # On a confirming turn, reuse the exact proposed config so the confirmation
    # digest is stable; otherwise build fresh, minting a run_id at most once.
    cfg = resolve_drive_config(
        config,
        pending_config=pending_cfg,
        has_confirm_token=bool(confirm_token),
        active_run_id=default_run,
        run_id_factory=lambda: f"agent-drive-{{secrets.token_hex(4)}}",
    )
    try:
        max_iterations = int(body.get("max_iterations") or cfg.get("max_iterations") or 3)
    except (TypeError, ValueError):
        max_iterations = 3
    max_iterations = max(1, min(max_iterations, 5))

    def _launch(loop_cfg):
        return _act_response_to_dict(
            submit_sim2real({{"run_id": str(loop_cfg.get("run_id") or "")}})
        )

    def _status(run_id):
        return _act_response_to_dict(sim2real_status(run_id=str(run_id or "")))

    def _diagnose(gate_result, run_status):
        # Quantitative viewer eval (Phase F Gap 4) feeds the diagnosis.
        signals = extract_quantitative_signals(gate_result if isinstance(gate_result, dict) else {{}})
        if not signals.get("has_signal"):
            mode = "insufficient_signal"
        elif signals.get("policy_collapse") or signals.get("degenerate"):
            mode = "policy_collapse"
        else:
            mode = "low_success"
        diagnosis = {{
            "failure_mode": mode,
            "signals": signals,
            "notes": "; ".join(signals.get("notes", [])),
        }}
        baseline_run = str(cfg.get("baseline_run_id") or cfg.get("baseline_run") or "").strip()
        current_run = str(
            (run_status or {{}}).get("run_id") if isinstance(run_status, dict) else ""
        ).strip()
        if baseline_run and current_run:
            memory_evidence = _agent_run_memory().explain_regression_data(
                current_run, baseline_run
            )
            if memory_evidence.get("ok"):
                diagnosis["run_memory"] = memory_evidence
        return diagnosis

    # Honor the operator-configured threshold when the run report omits one.
    _gate = gate_with_config_threshold(_sim2real_gate_metrics, cfg.get("threshold"))
    result = drive_sim2real_loop(
        goal,
        config=cfg,
        launch=_launch,
        status=_status,
        gate=_gate,
        diagnose=_diagnose,
        confirm_token=confirm_token,
        session_token=session_token,
        confirm_digest=confirm_digest,
        max_iterations=max_iterations,
        confirmation_ok=confirmation_ok,
    )
    if result.get("needs_confirmation"):
        proposed = result.get("proposed_action") if isinstance(result.get("proposed_action"), dict) else {{}}
        digest = str(proposed.get("digest") or "")
        result["confirm_token"] = _issue_agent_confirm_token(proposed, digest)
    result["grounded"] = False
    result["mode"] = "sim2real-drive"
    result["apis_used"] = ["workflows/sim2real/submit", "workflows/sim2real/status"]
    # Persist the drive outcome to cross-run memory (Phase F Gap 5) so later
    # sessions can compare/explain regressions from stored metadata.
    if not result.get("needs_confirmation") and result.get("final_run_id"):
        try:
            last_gate = {{}}
            iters = result.get("iterations") or []
            if iters and isinstance(iters[-1], dict):
                last_gate = iters[-1].get("gate") or {{}}
            _agent_run_memory().record_run(
                result["final_run_id"],
                {{
                    "decision": result.get("decision"),
                    "stopped_reason": result.get("stopped_reason"),
                    "iterations": len(iters),
                    "success_rate": last_gate.get("success_rate"),
                    "threshold": last_gate.get("threshold"),
                    "recorded_at": _now_iso(),
                }},
                source="drive",
            )
        except Exception:
            pass
    # Phase I: record structured spans for the offline analyzer / injected tracer.
    _record_agent_trace(result)
    return result

def _agent_run_memory():
    return RunMemory(
        JsonFileStore("/opt/npa-agent/memory"),
        comparator=compare_rollouts,
    )

@app.post("/agent/memory/record")
def agent_memory_record(payload: dict):
    body = payload if isinstance(payload, dict) else {{}}
    run_id = str(body.get("run_id") or "").strip()
    if not run_id:
        raise HTTPException(status_code=400, detail="run_id is required")
    metadata = body.get("metadata") if isinstance(body.get("metadata"), dict) else {{}}
    record = _agent_run_memory().record_run(run_id, metadata)
    return {{"ok": True, "record": record}}

@app.get("/agent/memory/runs")
def agent_memory_runs(limit: int = 20):
    return {{"ok": True, "runs": _agent_run_memory().list_runs(limit=limit)}}

@app.get("/agent/memory/run/{{run_id:path}}")
def agent_memory_run(run_id: str):
    record = _agent_run_memory().get_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="run not found in memory")
    return {{"ok": True, "record": record}}

@app.get("/agent/memory/compare")
def agent_memory_compare(run_a: str, run_b: str):
    return _agent_run_memory().compare_runs(run_a, run_b)

@app.get("/agent/memory/explain")
def agent_memory_explain(baseline_run: str, candidate_run: str):
    return _agent_run_memory().explain_regression_data(candidate_run, baseline_run)

# ── Blueprint Phase H: retrieval / grounding ─────────────────────────────────
# LanceDB-backed vector store + Token Factory embeddings + provider-agnostic
# web_search. Endpoints/keys come from env/config (never hardcoded).
EMBED_MODEL_ENV = os.environ.get("NPA_AGENT_EMBED_MODEL", "").strip()
EMBED_MODEL_FALLBACK = "BAAI/bge-en-icl"
_EMBED_MODEL_CACHE = {{"model": "", "expires_at": 0.0}}

def _resolve_embed_model():
    # Explicit operator override wins; otherwise auto-discover an embedding model
    # the key actually serves (so we never hardcode a model this key lacks).
    if EMBED_MODEL_ENV:
        return EMBED_MODEL_ENV
    now = time.monotonic()
    cache = _EMBED_MODEL_CACHE
    if cache.get("model") and cache.get("expires_at", 0.0) > now:
        return cache["model"]
    model = EMBED_MODEL_FALLBACK
    try:
        for mid in _fetch_token_factory_models():
            low = str(mid).lower()
            if "embed" in low or "bge" in low or "e5-" in low or low.endswith("-e5"):
                model = mid
                break
    except Exception:
        pass
    cache["model"] = model
    cache["expires_at"] = now + 300.0
    return model

LANCEDB_URI = os.environ.get("NPA_AGENT_LANCEDB_URI", "").strip()
LANCEDB_TABLE = os.environ.get("NPA_AGENT_LANCEDB_TABLE", "").strip() or "npa_corpus"
SEARXNG_URL = os.environ.get("NPA_AGENT_SEARXNG_URL", "").strip().rstrip("/")
RETRIEVAL_DIR = Path("/opt/npa-agent/retrieval")
RETRIEVAL_TOP_K = _retrieval.DEFAULT_TOP_K
RETRIEVAL_CHAT_MIN_SCORE = 0.35
try:
    RETRIEVAL_CHAT_MIN_SCORE = float(os.environ.get("NPA_AGENT_RETRIEVAL_CHAT_MIN_SCORE", "0.35"))
except (TypeError, ValueError):
    RETRIEVAL_CHAT_MIN_SCORE = 0.35

def _embed_texts(texts):
    items = [str(t) for t in (texts or [])]
    if not items:
        return []
    api_key = _provider_api_key("token_factory")
    if not api_key:
        raise RuntimeError("missing Token Factory API key for embeddings")
    base_url = _provider_base_url("token_factory")
    if not base_url:
        raise RuntimeError("missing Token Factory base URL for embeddings")
    url = f"{{base_url}}/embeddings"
    response = httpx.post(
        url,
        headers={{"Authorization": f"Bearer {{api_key}}", "Content-Type": "application/json"}},
        json={{"model": _resolve_embed_model(), "input": items}},
        timeout=60.0,
    )
    response.raise_for_status()
    data = response.json()
    rows = data.get("data") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        raise RuntimeError("Token Factory embeddings response missing data")
    ordered = sorted(rows, key=lambda r: r.get("index", 0) if isinstance(r, dict) else 0)
    vectors = []
    for row in ordered:
        embedding = row.get("embedding") if isinstance(row, dict) else None
        vectors.append([float(x) for x in (embedding or [])])
    return vectors

def _agent_web_search(query):
    # Provider-agnostic live grounding: SearXNG self-hosted on AI Cloud (via npa)
    # exposes an OpenSearch-style JSON API. A hosted search can be swapped in by
    # matching this {{title, url, snippet}} shape — no Tavily dependency.
    if not SEARXNG_URL:
        return []
    try:
        response = httpx.get(
            f"{{SEARXNG_URL}}/search",
            params={{"q": str(query or ""), "format": "json"}},
            timeout=20.0,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return []
    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list):
        return []
    out = []
    for item in results[:10]:
        if not isinstance(item, dict):
            continue
        out.append({{
            "title": item.get("title") or "",
            "url": item.get("url") or "",
            "snippet": item.get("content") or "",
        }})
    return out

def _agent_retrieval_store(dim=None):
    if LANCEDB_URI and dim:
        try:
            return _retrieval.build_lance_store(LANCEDB_URI, LANCEDB_TABLE, dim=int(dim))
        except Exception:
            pass
    return _retrieval.JsonVectorStore(str(RETRIEVAL_DIR / "corpus.json"))

def _agent_retrieval_store_for_query(query):
    # Only probe the embedding dim (an extra embed call) when a Lance store is
    # configured; the pure-python Json store needs no schema dim.
    if not LANCEDB_URI:
        return _agent_retrieval_store(dim=None)
    try:
        qvec = _embed_texts([str(query or "")])
        dim = len(qvec[0]) if qvec and qvec[0] else None
    except Exception:
        dim = None
    return _agent_retrieval_store(dim=dim)

def _agent_retrieval_roots(body):
    roots = body.get("roots") if isinstance(body, dict) else None
    if isinstance(roots, list) and roots:
        return [str(r) for r in roots if str(r).strip()]
    return [
        str(NPA_SOURCE_ROOT / "docs"),
        str(NPA_SOURCE_ROOT / "skills"),
        str(NPA_SOURCE_ROOT / "npa" / "docs"),
        "/opt/npa-agent/repo/docs",
        "/opt/npa-agent/repo/skills",
    ]

@app.post("/agent/retrieval/index")
def agent_retrieval_index(payload: dict):
    body = payload if isinstance(payload, dict) else {{}}
    roots = _agent_retrieval_roots(body)
    documents = list(_retrieval.iter_corpus_documents(roots))
    if not documents:
        return {{"ok": False, "error": "no corpus documents found", "roots": roots}}
    dim = None
    try:
        probe = _embed_texts([(documents[0][2][:200] or "probe")])
        dim = len(probe[0]) if probe and probe[0] else None
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"embedding probe failed: {{exc}}")
    store = _agent_retrieval_store(dim=dim)
    try:
        result = _retrieval.index_corpus(documents, embed=_embed_texts, store=store, source="repo")
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"indexing failed: {{exc}}")
    result["roots"] = roots
    result["embed_model"] = _resolve_embed_model()
    result["backend"] = "lancedb" if (LANCEDB_URI and dim) else "json"
    return result

@app.get("/agent/retrieval/search")
def agent_retrieval_search(q: str = "", k: int = RETRIEVAL_TOP_K, web: bool = False):
    query = str(q or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="q is required")
    try:
        kk = max(1, min(int(k), 20))
    except (TypeError, ValueError):
        kk = RETRIEVAL_TOP_K
    store = _agent_retrieval_store_for_query(query)
    result = _retrieval.retrieve(
        query,
        embed=_embed_texts,
        store=store,
        k=kk,
        web_search=_agent_web_search if web else None,
        index_web=bool(web),
    )
    result["answer"] = _retrieval.format_grounded_answer(query, result.get("citations") or [])
    result["grounded"] = True
    return result

@app.get("/agent/retrieval/status")
def agent_retrieval_status():
    store = _agent_retrieval_store(dim=None)
    return {{
        "ok": True,
        "chunks": store.count(),
        "sources": store.sources(),
        "embed_model": _resolve_embed_model(),
        "backend": "lancedb" if LANCEDB_URI else "json",
        "web_search": bool(SEARXNG_URL),
    }}

def _maybe_retrieval_grounded(query):
    query = str(query or "").strip()
    if not query:
        return None
    store = _agent_retrieval_store(dim=None)
    # Only fire when a corpus is actually indexed; otherwise /chat is unchanged.
    if store.count() <= 0:
        return None
    try:
        result = _retrieval.retrieve(
            query, embed=_embed_texts, store=store, k=RETRIEVAL_TOP_K,
            min_score=RETRIEVAL_CHAT_MIN_SCORE,
        )
    except Exception:
        return None
    # Gating + formatting decision lives in the tested pure helper so this
    # embedded glue is thin (build store -> count guard -> retrieve -> delegate).
    return _retrieval.grounded_reply_from_result(
        query, result, min_score=RETRIEVAL_CHAT_MIN_SCORE
    )

# ── Blueprint Phase I: observability ─────────────────────────────────────────
# Structured spans via an injected tracer (Null by default; Langfuse/OTel when
# configured) + a persisted ring buffer feeding the offline analyzer.
TRACE_DIR = Path("/opt/npa-agent/trace")
TRACE_MAX = 200

def _agent_tracer():
    lf_public = os.environ.get("NPA_AGENT_LANGFUSE_PUBLIC_KEY", "").strip()
    lf_secret = os.environ.get("NPA_AGENT_LANGFUSE_SECRET_KEY", "").strip()
    lf_host = os.environ.get("NPA_AGENT_LANGFUSE_HOST", "").strip()
    if lf_public and lf_secret:
        try:
            return _agent_tracing.build_langfuse_tracer(
                public_key=lf_public, secret_key=lf_secret, host=lf_host
            )
        except Exception:
            pass
    return _agent_tracing.NullTracer()

def _spans_for_trace(trace):
    if isinstance(trace, dict) and "steps" in trace:
        return _agent_tracing.spans_from_action_loop(trace)
    return _agent_tracing.spans_from_drive(trace)

def _record_agent_trace(result):
    if not isinstance(result, dict):
        return
    try:
        TRACE_DIR.mkdir(parents=True, exist_ok=True)
        path = TRACE_DIR / "recent.json"
        try:
            existing = json.loads(path.read_text())
            if not isinstance(existing, list):
                existing = []
        except Exception:
            existing = []
        existing.append(result)
        existing = existing[-TRACE_MAX:]
        path.write_text(json.dumps(existing))
        _agent_tracing.record_spans(_agent_tracer(), _spans_for_trace(result))
    except Exception:
        pass

def _recent_agent_traces():
    try:
        data = json.loads((TRACE_DIR / "recent.json").read_text())
        return data if isinstance(data, list) else []
    except Exception:
        return []

@app.get("/agent/trace/spans")
def agent_trace_spans(limit: int = 20):
    try:
        lim = max(1, min(int(limit), 100))
    except (TypeError, ValueError):
        lim = 20
    spans = []
    for trace in _recent_agent_traces()[-lim:]:
        if not isinstance(trace, dict):
            continue
        spans.extend(s.to_dict() for s in _spans_for_trace(trace))
    return {{"ok": True, "spans": spans, "count": len(spans)}}

@app.post("/agent/trace/analyze")
def agent_trace_analyze(payload: dict):
    body = payload if isinstance(payload, dict) else {{}}
    traces = body.get("traces")
    if not isinstance(traces, list) or not traces:
        traces = _recent_agent_traces()
    return _agent_tracing.analyze_traces(traces)

@app.get("/health")
def health():
    state = _load_state()
    state_sha256 = hashlib.sha256(
        json.dumps(state, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {{
        "ok": True,
        "tool_refs": len(TOOL_REFS),
        "capabilities": {{
            "gpu_allocation_fallback": {{
                "status": "available",
                "grounded": True,
                "routes": [
                    "POST /api/agent/gpu-allocation/attempt",
                    "POST /api/agent/gpu-allocation/consent",
                ],
            }},
        }},
        "deployment": dict(DEPLOYMENT),
        "state_sha256": state_sha256,
    }}

@app.get("/deployment")
def deployment_identity():
    return dict(DEPLOYMENT)


@app.get("/access")
def agent_access(refresh: bool = False):
    return _agent_access_api_response(refresh)


@app.get("/models")
def models(refresh: bool = False):
    return {{
        "ok": True,
        "default": LLM_MODEL,
        "default_model": LLM_MODEL,
        "default_provider": LLM_PROVIDER,
        "providers": _configured_llm_providers(),
        "models": _available_llm_models(refresh=bool(refresh)),
    }}

@app.get("/session")
def session_bootstrap():
    state = _load_state()
    active_session = _get_chat_session(state, str(state.get("active_chat_session_id") or "default"))
    sim_viz = _sim_viz_for_run(state)
    selected = state.get("camera_selection", ["workspace"])
    camera = str(sim_viz.get("camera") or (selected[0] if isinstance(selected, list) and selected else "workspace"))
    sim_viz["camera"] = camera
    session_run_id = str(sim_viz.get("run_id") or "").strip()
    if PRELOAD_STOCK_DEMO and not sim_viz.get("rrd_uri") and session_run_id in {"", "franka-demo"} and RRD_PATH.is_file():
        sim_viz["rrd_uri"] = f"file://{{RRD_PATH}}"
    sim_viz["rerun_ready"] = _rerun_ready_state(rrd_uri=str(sim_viz.get("rrd_uri") or ""))
    history = active_session.get("chat_history", [])
    if not isinstance(history, list):
        history = []
    return {{
        "deployment": dict(DEPLOYMENT),
        "selection": state.get("selection", dict(DEFAULT_SELECTION)),
        "sim_viz": sim_viz,
        "latest_submit": state.get("latest_submit", {{}}),
        "sim_viz_runs": _sim_viz_runs(state),
        "infra": _agent_k8s_backends(),
        "workflow_draft": _workflow_draft_from_state(state),
        "workflow_submit": state.get("workflow_submit", {{}}),
        "camera_selection": state.get("camera_selection", ["workspace"]),
        "chat_history": history,
        "active_chat_session_id": active_session["id"],
        "chat_sessions": _list_chat_sessions(state),
        "chat_memory": {{
            "tenant": _chat_memory_tenant(),
            "s3_configured": bool(_agent_s3_settings().get("bucket") and _agent_s3_settings().get("access_key")),
            "prefix": _chat_memory_prefix(),
        }},
        "llm": {{
            "default": LLM_MODEL,
            "default_model": LLM_MODEL,
            "default_provider": LLM_PROVIDER,
            "provider": LLM_PROVIDER,
            "providers": _configured_llm_providers(),
            "model": LLM_MODEL,
            "models": _available_llm_models(),
        }},
    }}


@app.get("/chat/sessions")
def chat_sessions():
    state = _load_state()
    active_id = str(state.get("active_chat_session_id") or "default")
    settings = _agent_s3_settings()
    return {{
        "ok": True,
        "active_session_id": active_id,
        "sessions": _list_chat_sessions(state),
        "memory": {{
            "tenant": _chat_memory_tenant(),
            "s3_configured": bool(settings.get("bucket") and settings.get("access_key")),
            "prefix": _chat_memory_prefix(settings),
        }},
    }}


@app.post("/chat/sessions")
def create_chat_session(payload: dict | None = None):
    body = payload if isinstance(payload, dict) else {{}}
    state = _load_state()
    session_id = _sanitize_chat_session_id(str(body.get("id") or f"chat-{{secrets.token_urlsafe(8)}}"))
    title = str(body.get("title") or "New chat").strip() or "New chat"
    session = _normalize_chat_session(session_id, {{"id": session_id, "title": title, "chat_history": []}})
    saved = _save_chat_session(state, session, active=True)
    return {{"ok": True, "session": saved, "active_session_id": saved["id"], "sessions": _list_chat_sessions(state)}}


@app.get("/chat/sessions/{{session_id}}")
def get_chat_session(session_id: str):
    state = _load_state()
    session = _lookup_chat_session(state, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"chat session not found: {{session_id}}")
    return {{"ok": True, "session": session}}


@app.post("/chat/sessions/{{session_id}}/select")
def select_chat_session(session_id: str):
    state = _load_state()
    session = _get_chat_session(state, session_id)
    state["active_chat_session_id"] = session["id"]
    state["chat_history"] = session.get("chat_history", [])
    _save_state(state)
    return {{"ok": True, "session": session, "active_session_id": session["id"], "sessions": _list_chat_sessions(state)}}

@app.get("/tools")
def tools():
    return {{"tool_refs": TOOL_REFS}}

@app.get("/tools/{{tool_ref:path}}")
def tool(tool_ref: str):
    payload = TOOL_CATALOG.get(tool_ref)
    if payload is None:
        return {{"ok": False, "error": "unknown toolRef", "tool_ref": tool_ref}}
    return {{"ok": True, "tool_ref": tool_ref, **payload}}

def _served_recording_is_run_specific() -> bool:
    try:
        if not RECORDING_PATH.is_file(): return False
        sim_viz = _load_state().get("sim_viz")
        if isinstance(sim_viz, dict):
            bound_sha256 = str(sim_viz.get("served_recording_sha256") or "").strip().lower()
            if re.fullmatch(r"[0-9a-f]{{64}}", bound_sha256):
                recording_size = RECORDING_PATH.stat().st_size
                bound_size = int(sim_viz.get("served_recording_size_bytes") or 0)
                with RECORDING_PATH.open("rb") as stream:
                    has_rrd_header = stream.read(4) == b"RRF2"
                if bound_size > 0:
                    return has_rrd_header and recording_size == bound_size
                # Backward compatibility for state written before size binding.
                # A valid persisted hash plus an RRF2 file is sufficient until
                # the next artifact load records the exact byte count.
                return has_rrd_header and recording_size > 4
        # Compatibility for recordings wired by the legacy Sim2Real path.
        recording_bytes = RECORDING_PATH.read_bytes()
        return recording_has_run_entities(recording_bytes)
    except Exception:
        return False

@app.get("/sim-viz/status")
def sim_viz_status(run_id: str = ""):
    state = _load_state()
    # Self-heal: if the active run is a real run but the served recording is the
    # stock demo (clobbered by a later franka load / bootstrap), reattach the
    # run's own recording so we never serve/claim the demo as run data.
    _sv = state.get("sim_viz") if isinstance(state.get("sim_viz"), dict) else {{}}
    _active_run = str(_sv.get("run_id") or "").strip()
    _target_run = str(run_id or "").strip() or _active_run
    if _target_run and _target_run != "franka-demo" and not _served_recording_is_run_specific():
        _cam = str(_sv.get("camera") or "workspace") or "workspace"
        try:
            if _wire_active_sim2real_recording(state, camera=_cam) is not None:
                state = _load_state()
        except Exception:
            pass
    payload = _sim_viz_for_run(state, run_id=run_id)
    requested_run = str(run_id or "").strip()
    # Prefer the live sim_viz snapshot when it matches — history can lag behind
    # load-run under concurrent UI polls.
    current = state.get("sim_viz")
    if isinstance(current, dict):
        current_run = str(current.get("run_id") or "").strip()
        if current_run and (not requested_run or current_run == requested_run):
            merged = dict(payload)
            merged.update(current)
            # Live Rerun/demo snapshots must not keep a stale non-rerun media render
            # from history (that forces status to clear rrd_uri / rerun_ready).
            current_render = str(current.get("artifact_render") or "").strip().lower()
            if str(current.get("rrd_uri") or "").strip() and current_render in {{"", "rerun"}}:
                merged["artifact_render"] = current_render or "rerun"
                if not str(current.get("artifact_key") or "").strip():
                    merged["artifact_key"] = ""
                    merged["artifact_uri"] = ""
                    if "visualization_note" not in current:
                        merged["visualization_note"] = ""
            payload = merged
    selected = state.get("camera_selection", ["workspace"])
    camera = str(payload.get("camera") or (selected[0] if isinstance(selected, list) and selected else "workspace"))
    payload["camera"] = camera
    latest_submit = state.get("latest_submit", {{}})
    if not isinstance(latest_submit, dict):
        latest_submit = {{}}
    if not str(payload.get("run_id") or "").strip():
        payload["run_id"] = str(latest_submit.get("run_id") or "").strip()
    if str(payload.get("stage") or "idle").strip().lower() == "idle" and payload.get("run_id"):
        payload["stage"] = "submitted"
    # Read-only: do not _record/_save here. Concurrent GET status polls were
    # racing load-run and wiping artifact_render from sim_viz_runs.
    payload_run = str(payload.get("run_id") or "").strip()
    # Honest gate: a real run must not report rerun_ready / a run rrd_uri unless
    # the served recording actually holds run-specific entities (never the demo).
    if payload_run and payload_run != "franka-demo":
        if not _served_recording_is_run_specific():
            payload["rerun_ready"] = False
            payload["rrd_uri"] = ""
            payload["recording_status"] = "run_recording_unavailable"
        else:
            payload.pop("recording_status", None)
    run_has_specific_rrd = bool(str(payload.get("rrd_uri") or "").strip())
    live_url = str(payload.get("live_grpc_url") or "").strip()
    may_use_default_recording = payload_run in {"", "franka-demo"} and not requested_run
    if (
        str(payload.get("artifact_render") or "").strip().lower() in {"", "rerun"}
        and (live_url or run_has_specific_rrd or may_use_default_recording)
    ):
        if live_url:
            payload["rerun_iframe_url"] = (
                f"/rerun/?url={{quote(live_url, safe='')}}&hide_welcome_screen=1&theme=dark&camera={{camera}}"
            )
        else:
            payload["rerun_iframe_url"] = _rerun_iframe_url(camera, recording_path=str(payload.get("artifact_preview_url") or ""))
    else:
        payload["rerun_iframe_url"] = ""
    if not payload.get("rrd_uri") and may_use_default_recording and RRD_PATH.is_file():
        payload["rrd_uri"] = f"file://{{RRD_PATH}}"
    mode = str(payload.get("mode") or "static").strip().lower()
    payload["mode"] = "live" if mode == "live" else "static"
    artifact_render = str(payload.get("artifact_render") or "").strip().lower()
    preview_status = str(payload.get("preview_status") or "").strip().lower()
    if preview_status == "no_previewable_recording":
        # Do not let a shared/stale recording file turn a training run's honest
        # no-preview state back into rerun_ready=true on the next status poll.
        payload["rrd_uri"] = ""
        payload["rerun_ready"] = False
        payload["rerun_iframe_url"] = ""
    elif artifact_render and artifact_render != "rerun":
        payload["rrd_uri"] = ""
        payload["rerun_ready"] = False
        payload["rerun_iframe_url"] = ""
    else:
        payload["rerun_ready"] = _rerun_ready_state(rrd_uri=str(payload.get("rrd_uri") or ""))
    # Latest-first (rrd_updated_at), not alphabetical — keep UI choosers newest-on-top.
    payload["available_run_ids"] = [
        str(item.get("run_id") or "").strip()
        for item in _sim_viz_runs(state)
        if str(item.get("run_id") or "").strip()
    ]
    payload["available_runs"] = build_available_sim_viz_runs(_sim_viz_runs(state))
    payload["active_run_id"] = str(state.get("active_run_id") or payload.get("run_id") or "").strip()
    return payload

@app.get("/sim-viz/runs")
def sim_viz_runs():
    state = _load_state()
    active = sim_viz_status()
    runs = _sim_viz_runs(state)
    active_id = str(active.get("run_id") or "").strip()
    return {{
        "ok": True,
        "active_run_id": active_id,
        "runs": runs,
    }}

@app.post("/sim-viz/select-run")
def sim_viz_select_run(payload: dict | None = None):
    body = payload if isinstance(payload, dict) else {{}}
    requested_run = str(body.get("run_id") or "").strip()
    requested_ref = str(body.get("run_ref") or "").strip()
    if not requested_run:
        raise HTTPException(status_code=400, detail="run_id is required")
    state = _load_state()
    runs = _sim_viz_runs(state)
    matches = [
        item
        for item in runs
        if (requested_ref and str(item.get("artifact_run_ref") or "").strip() == requested_ref)
        or (not requested_ref and str(item.get("run_id") or "").strip() == requested_run)
    ]
    if len(matches) > 1:
        raise HTTPException(status_code=409, detail="run_id is ambiguous in viewer history; provide run_ref")
    selected = matches[0] if matches else None
    if not isinstance(selected, dict):
        raise HTTPException(status_code=404, detail=f"run_id not found: {{requested_run}}")
    selected_run = str(selected.get("run_id") or "").strip()
    if selected_run and selected_run != requested_run:
        raise HTTPException(status_code=400, detail="run_ref does not identify run_id")
    selected_ref = str(selected.get("artifact_run_ref") or "").strip()
    selected_render = str(selected.get("artifact_render") or "").strip().lower()
    if selected_ref and selected_render == "rerun":
        # Re-resolve source-qualified history because every load replaces the
        # shared active recording; metadata alone cannot restore exact bytes.
        selected_uri = str(selected.get("artifact_uri") or "").strip()
        if not selected_uri.startswith("s3://"):
            raise HTTPException(status_code=409, detail="source-qualified Rerun history is missing its S3 artifact URI")
        reload_request = {{"run_id": requested_run, "run_ref": selected_ref, "rrd_uri": selected_uri}}
        reload_request["camera"] = str(selected.get("camera") or "").strip()
        loaded = sim_viz_load_run(reload_request)
        return {{"ok": True, "sim_viz": loaded["sim_viz"], "selected": selected}}
    sim_viz = dict(DEFAULT_SIM_VIZ)
    if isinstance(state.get("sim_viz"), dict):
        sim_viz.update(state["sim_viz"])
    sim_viz.update(selected)
    sim_viz["run_id"] = requested_run
    state["sim_viz"] = sim_viz
    state["latest_submit"] = {{
        "run_id": requested_run,
        "submitted_at": str(selected.get("submitted_at") or _now_iso()),
        "submit_mode": str(selected.get("submit_mode") or "history-select"),
    }}
    _save_state(state)
    return {{"ok": True, "sim_viz": sim_viz_status(), "selected": selected}}
def _sim_viz_load_response(state: dict, sim_viz: dict, *, run_id: str) -> dict:
    # Echo the just-applied snapshot. Do not re-enter sim_viz_status here:
    # concurrent UI polls can rewrite state mid-load and return the wrong run.
    payload = dict(DEFAULT_SIM_VIZ)
    payload.update(sim_viz if isinstance(sim_viz, dict) else {{}})
    payload["run_id"] = str(run_id or payload.get("run_id") or "").strip()
    payload["active_run_id"] = str(state.get("active_run_id") or payload["run_id"] or "").strip()
    payload["available_run_ids"] = [
        str(item.get("run_id") or "").strip()
        for item in _sim_viz_runs(state)
        if str(item.get("run_id") or "").strip()
    ]
    payload["available_runs"] = build_available_sim_viz_runs(_sim_viz_runs(state))
    render = str(payload.get("artifact_render") or "").strip().lower()
    preview_status = str(payload.get("preview_status") or "").strip().lower()
    if preview_status == "no_previewable_recording":
        payload["rrd_uri"] = ""
        payload["rerun_ready"] = False
        payload["rerun_iframe_url"] = ""
    elif render and render != "rerun":
        payload["rrd_uri"] = ""
        payload["rerun_ready"] = False
        if not payload.get("rerun_iframe_url"):
            payload["rerun_iframe_url"] = ""
    else:
        payload["rerun_ready"] = _rerun_ready_state(rrd_uri=str(payload.get("rrd_uri") or ""))
        if not payload.get("rerun_iframe_url"):
            payload["rerun_iframe_url"] = _rerun_iframe_url(str(payload.get("camera") or "workspace"), recording_path=str(payload.get("artifact_preview_url") or ""))
    return payload

@app.post("/sim-viz/load-run")
def sim_viz_load_run(payload: dict | None = None):
    body = payload if isinstance(payload, dict) else {{}}
    run_id = str(body.get("run_id") or "").strip()
    if not run_id:
        raise HTTPException(status_code=400, detail="run_id is required")
    try: run_id = _validate_run_basename(run_id)
    except ArtifactDiscoveryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    requested_camera = str(body.get("camera") or "").strip()
    camera = requested_camera or "workspace"
    requested_rrd_uri = str(body.get("rrd_uri") or "").strip()
    requested_run_ref = str(body.get("run_ref") or "").strip()

    # Prefer a run-scoped Rerun recording over stale history entries. History can
    # contain JSON artifacts from prior clicks, which otherwise makes Load Run
    # show "Non-RRD artifact loaded" even when reports/sim2real.rrd exists.
    if requested_rrd_uri:
        s3, _settings = _agent_s3_client()
        bucket, key, _authorized_run = _authorize_agent_artifact_uri(
            s3=s3,
            settings=_settings,
            uri=requested_rrd_uri,
            run_id=run_id,
        )
        key = _safe_artifact_key(key)
        if render_hint_for_object(key=key) != "rerun":
            raise HTTPException(status_code=400, detail="rrd_uri must identify an RRD artifact")
        source_bucket, source_project, source_prefix = _artifact_source_metadata(
            _agent_access_report(), bucket, key, run_id
        )
        if requested_run_ref:
            resolution = resolve_run_artifacts(
                _agent_s3_buckets(s3, _settings),
                base_prefix=_settings.get("prefix", ""),
                run_ref_or_id=requested_run_ref,
                s3=s3,
            )
            if resolution is None or not any(
                item.key == key and item.s3_uri == requested_rrd_uri
                for item in resolution.artifacts
            ):
                raise HTTPException(status_code=400, detail="RRD URI is outside the selected run")
            run_id = resolution.run_id
            requested_run_ref = resolution.run_ref
            source_bucket = resolution.bucket
            source_prefix = resolution.source_prefix
            source_project = artifact_bucket_projects(_agent_access_report()).get(
                source_bucket, ""
            )
        elif source_bucket:
            requested_run_ref = encode_run_ref(source_bucket, source_prefix, run_id)
        local_name = _artifact_filename(key)
        local_path = RECORDINGS_DIR / local_name
        download_s3_uri(requested_rrd_uri, local_path, s3=s3)
        state = _load_state()
        sim_viz = _apply_loaded_artifact(
            state=state,
            run_id=validate_run_id(run_id),
            key=key,
            s3_uri=requested_rrd_uri,
            render="rerun",
            local_path=local_path,
            source_identity=(source_bucket, source_project, source_prefix),
            run_ref=requested_run_ref,
            requested_camera=requested_camera,
        )
        return {{"ok": True, "sim_viz": _sim_viz_load_response(state, sim_viz, run_id=run_id)}}

    session_response = _load_session_run_if_known(body=body, run_id=run_id, requested_camera=requested_camera)
    if session_response is not None:
        return session_response
    try:
        s3, settings = _agent_s3_client()
        requested_prefix = str(body.get("prefix") or "")
        requested_bucket, requested_project, requested_resolved_prefix, source_selected = _selected_run_request(body)
        artifacts = []
        resolution = None
        selected_prefix = ""
        selected_bucket = requested_bucket
        selected_project = requested_project
        resolved_run_id = run_id
        resolved_ref = requested_run_ref
        if requested_bucket:
            selected_bucket, selected_project, selected_prefix, artifacts = _load_selected_run_artifacts(
                s3=s3, settings=settings, run_id=run_id,
                resource_bucket=requested_bucket, project_id=requested_project,
                resolved_prefix=requested_resolved_prefix, source_selected=source_selected,
                exclude=_discovery_exclude_roots(),
            )
            resolved_ref = encode_run_ref(selected_bucket, selected_prefix, run_id)
        elif requested_prefix:
            effective_prefix = _artifact_discovery_prefix(settings, requested_prefix)
            artifacts = list_artifacts(settings["bucket"], validate_run_id(run_id), prefix=effective_prefix, s3=s3)
            selected_bucket = settings["bucket"]
            selected_prefix = effective_prefix
            if artifacts:
                resolved_ref = encode_run_ref(selected_bucket, selected_prefix, run_id)
        if not artifacts:
            resolution = resolve_run_artifacts(
                _agent_s3_buckets(s3, settings),
                base_prefix=settings.get("prefix", ""),
                run_ref_or_id=requested_run_ref or run_id,
                s3=s3,
            )
            if resolution is not None:
                artifacts = resolution.artifacts
                selected_bucket = resolution.bucket
                selected_prefix = resolution.source_prefix
                selected_project = artifact_bucket_projects(
                    _agent_access_report()
                ).get(selected_bucket, "")
                resolved_run_id = resolution.run_id
                resolved_ref = resolution.run_ref
        preferred = select_preferred_artifact(artifacts)
        if preferred and (preferred.render == "rerun" or (requested_bucket and source_selected)):
            local_name = _artifact_filename(preferred.key)
            local_path = RECORDINGS_DIR / local_name
            download_s3_uri(preferred.s3_uri, local_path, s3=s3)
            state = _load_state()
            sim_viz = _apply_loaded_artifact(
                state=state,
                run_id=resolved_run_id,
                key=preferred.key,
                s3_uri=preferred.s3_uri,
                render=preferred.render,
                local_path=local_path,
                source_identity=(selected_bucket, selected_project, selected_prefix),
                run_ref=resolved_ref,
                requested_camera=requested_camera,
            )
            return {{
                "ok": True,
                "contract": ARTIFACT_DISCOVERY_CONTRACT,
                "sim_viz": _sim_viz_load_response(state, sim_viz, run_id=resolved_run_id),
                "preferred": preferred.to_dict(),
                "run_ref": resolved_ref,
            }}
        if artifacts:
            # A selected run is still real and usable when its capability is a
            # service/session artifact rather than an RRD recording. Persist an
            # artifact-backed active context so conditional tabs (LeIsaac in
            # particular) survive periodic refresh, while keeping Rerun
            # truthfully unavailable instead of downloading JSON as a viewer.
            role_counts = artifact_inventory_counts(artifacts)
            state = _load_state()
            sim_viz = dict(DEFAULT_SIM_VIZ)
            current = state.get("sim_viz")
            if (
                isinstance(current, dict)
                and str(current.get("run_id") or "").strip() == resolved_run_id
            ):
                # An unqualified same-run load may legitimately find artifacts
                # but no preferred RRD. Keep a canonical MCAP that was already
                # validated/published for that exact run; selecting the View tab
                # must not make the next clean Foxglove profile reconvert S3.
                # Cross-run loads still start from DEFAULT_SIM_VIZ and therefore
                # cannot inherit another run's source, layout, or provenance.
                for key in (
                    *CANONICAL_MCAP_DEFAULT_STATE,
                    "mcap_uri",
                    "mcap_updated_at",
                    "lichtblick_ready",
                    "lichtblick_iframe_url",
                    "foxglove_ready",
                    "foxglove_url",
                ):
                    if key in current:
                        sim_viz[key] = current[key]
            sim_viz.update({{
                "run_id": resolved_run_id,
                "artifact_run_ref": resolved_ref,
                "stage": "artifacts_available",
                "camera": camera,
                "rrd_uri": "",
                "rerun_ready": False,
                "rerun_iframe_url": "",
                "artifact_key": str(preferred.key if preferred else ""),
                "artifact_uri": str(preferred.s3_uri if preferred else ""),
                "artifact_render": str(preferred.render if preferred else ""),
                "artifact_count": len(artifacts),
                "output_artifact_count": role_counts["output"],
                "input_artifact_count": role_counts["input"],
                "metadata_artifact_count": role_counts["metadata"],
                "preview_status": "no_previewable_recording",
                "visualization_note": "No previewable recording; artifacts available.",
                "rrd_updated_at": _now_iso(),
            }})
            state["active_run_id"] = resolved_run_id
            state["sim_viz"] = sim_viz
            _record_sim_viz_run(state, sim_viz)
            _save_state(state)
            return {{
                "ok": True,
                "artifacts_available": True,
                "artifact_count": len(artifacts),
                "output_artifact_count": role_counts["output"],
                "sim_viz": _sim_viz_load_response(state, sim_viz, run_id=resolved_run_id),
                "preferred": preferred.to_dict() if preferred else None,
                "run_ref": resolved_ref,
            }}
        if requested_bucket:
            raise HTTPException(status_code=404, detail="selected artifact source has no loadable artifacts")
    except AmbiguousRunError as exc:
        raise HTTPException(
            status_code=409,
            detail={{"error": "run_id is ambiguous; provide run_ref",
                     "run_id": exc.run_id, "run_refs": exc.references}},
        ) from exc
    except AmbiguousRunSourceError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ArtifactDiscoveryError as exc:
        raise HTTPException(status_code=400, detail="invalid run artifact request") from exc
    except HTTPException:
        raise
    except (ClientError, BotoCoreError, OSError, KeyError, TypeError, ValueError):
        # Fall back to the historical in-memory run selector below; callers still
        # get a useful 404 if the run has never been seen.
        pass

    state = _load_state()
    runs = state.get("sim_viz_runs")
    if not isinstance(runs, dict):
        runs = {{}}
    history_key = requested_run_ref or run_id
    selected = runs.get(history_key)
    if not isinstance(selected, dict) and not requested_run_ref:
        matches = [
            item
            for item in runs.values()
            if isinstance(item, dict)
            and str(item.get("run_id") or "").strip() == run_id
        ]
        if len(matches) > 1:
            raise HTTPException(
                status_code=409,
                detail="run_id is ambiguous in viewer history; provide run_ref",
            )
        selected = matches[0] if matches else None
    sim2real_runs = state.get("sim2real_runs") if isinstance(state.get("sim2real_runs"), dict) else {{}}
    # Never invent phantom run ids — require a known sim-viz or sim2real run.
    if not isinstance(selected, dict) or not selected:
        if run_id not in sim2real_runs:
            raise HTTPException(status_code=404, detail=f"run_id not found: {{run_id}}")
        selected = {{"run_id": run_id}}
    else:
        selected = dict(selected)
    rrd_uri = str(body.get("rrd_uri") or "").strip()
    if rrd_uri:
        if rrd_uri.startswith("file://") and not file_uri_path_allowed(
            rrd_uri, allowed_paths=(str(RECORDINGS_DIR), str(RRD_PATH))
        ):
            raise HTTPException(status_code=400, detail="file:// rrd_uri is outside the recordings allowlist")
        if rrd_uri.startswith("s3://"):
            _assert_s3_uri_in_agent_bucket(rrd_uri, _agent_s3_settings())
        selected["rrd_uri"] = rrd_uri
    if requested_camera:
        selected["camera"] = camera
    stage = str(body.get("stage") or "").strip()
    if stage:
        selected["stage"] = stage
    mode = str(body.get("mode") or "").strip().lower()
    if mode in {{"static", "live"}}:
        selected["mode"] = mode
    selected["run_id"] = run_id
    selected["rrd_updated_at"] = _now_iso()
    state["sim_viz"] = selected
    _record_sim_viz_run(state, selected)
    _save_state(state)
    return {{
        "ok": True,
        "sim_viz": _sim_viz_load_response(state, selected, run_id=run_id),
    }}

@app.get("/sim-viz/recordings")
def sim_viz_recordings():
    # List available .rrd recording files in /opt/npa-agent/recordings/ for quick viewer switching.
    recordings_dir = Path("/opt/npa-agent/recordings")
    result = []
    if recordings_dir.is_dir():
        for rrd_file in sorted(recordings_dir.glob("*.rrd"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                stat = rrd_file.stat()
                result.append({{
                    "name": rrd_file.name,
                    "path": f"/rerun/recordings/{{rrd_file.name}}",
                    "size_bytes": stat.st_size,
                    "updated_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                    "active": rrd_file.name == "sim2real.rrd",
                }})
            except OSError:
                continue
    return {{"recordings": result, "count": len(result)}}


@app.get("/artifacts/runs")
def artifacts_runs(
    prefix: str = "", limit: int = 50, q: str = "", cursor: str = "",
    resource_bucket: str = "", project_id: str = "",
):
    # q is a case-insensitive substring filter over a bounded, cached discovery
    # index. Response metadata says whether that source index was complete; a
    # bounded observed match count is never represented as a global total.
    try:
        s3, settings = _agent_s3_client()
        access_report = _agent_access_report()
        access_diagnostics = _agent_access_diagnostics(access_report)
        bucket_projects = artifact_bucket_projects(access_report)
        buckets, selected_scope = _agent_artifact_list_scope(
            access_report, resource_bucket, project_id
        )
        query = str(q or "").strip()
        page_size = max(1, min(int(limit), 500))
        offset = _artifact_run_cursor_offset(cursor)
        discovery_limit = 10_000

        def _page_response(page, *, effective_prefix: str):
            # The bounded discovery page is intentionally cached without the
            # search term.  A query only filters that already-discovered index;
            # making ``q`` part of the cache key caused every distinct browser
            # fragment to repeat the same full multi-bucket object walk.
            indexed = page.runs
            if query:
                needle = query.lower()
                indexed = [item for item in indexed if needle in item.run_id.lower()]
            source_complete = bool(not page.truncated and page.discovery_complete)
            observed_match_count = len(indexed)
            end = min(offset + page_size, len(indexed))
            visible = indexed[offset:end]
            has_more = end < len(indexed)
            next_cursor = _artifact_run_cursor(end) if has_more else ""
            return {{
                "ok": True,
                "contract": ARTIFACT_DISCOVERY_CONTRACT,
                "bucket": settings["bucket"],
                "buckets": buckets,
                "resource_scope": selected_scope,
                "prefix": effective_prefix,
                "base_prefix": settings.get("prefix", ""),
                "query": query,
                "summary_mode": "artifact_index",
                "namespace": "npa_workflow_artifact_run",
                "namespace_help": "Searches discovered NPA workflow/artifact runs; Codex maintenance job IDs are a separate operator-local namespace.",
                "access": access_diagnostics,
                "runs": [item.to_dict() for item in visible],
                "count": len(visible),
                "count_scope": "page",
                "total_runs": observed_match_count if source_complete else None,
                "total_runs_scope": (
                    "filtered_global" if query and source_complete
                    else "global" if source_complete
                    else "unavailable"
                ),
                "observed_run_count": int(page.total_runs),
                "observed_match_count": observed_match_count,
                "query_complete": source_complete,
                "limit": page_size,
                "cursor": cursor,
                "next_cursor": next_cursor,
                "truncated": bool(has_more or page.truncated or not page.discovery_complete),
                "pagination_complete": bool(
                    not has_more and not page.truncated and page.discovery_complete
                ),
                "source_errors": [dict(item) for item in page.source_errors],
            }}
        if prefix:
            effective_prefix = _artifact_discovery_prefix(settings, prefix)
            # Cached (TTL + stale-while-revalidate): the run list is polled on every
            # page load; walking a category's objects each time made the UI show
            # "no runs" for seconds. The cache serves a warm result instantly and
            # refreshes in the background, so only the first load pays the S3 walk.
            page = list_runs_cached_multi(
                buckets,
                prefix=effective_prefix,
                base_prefix=settings.get("prefix", ""),
                limit=discovery_limit,
                contains="",
                bucket_projects=bucket_projects,
                lightweight=True,
                s3=s3,
            )
            return _page_response(page, effective_prefix=effective_prefix)
        # No user prefix: discover runs generically across ALL bucket roots.
        # Runs live under <base>/<category>/<run_id>/... (base from config, e.g.
        # "checkpoints") AND directly at the bucket root <category>/<run_id>/...
        # (e.g. scenario-gen-smoke/..., physical-ai-data-factory/...). list_all_runs
        # enumerates category folders under both roots and merges them, so every
        # workflow's runs show without hardcoding any workflow path. Cached the same
        # way (the no-prefix walk is the slowest and the default UI view).
        base = settings.get("prefix", "")
        # Discover across every access-report bucket (not just the configured one).
        # Never enumerate every credential-readable bucket: the tenant access report
        # is the bounded, policy-qualified source of artifact storage locations.
        page = list_runs_cached_multi(
            buckets,
            base_prefix=base,
            limit=discovery_limit,
            exclude=_discovery_exclude_roots(),
            contains="",
            bucket_projects=bucket_projects,
            lightweight=True,
            s3=s3,
        )
        return _page_response(page, effective_prefix=base)
    except HTTPException:
        raise
    except Exception as exc:
        return JSONResponse(status_code=502, content={{"ok": False, "error": str(exc), "source": "s3"}})
def _resolved_run_artifacts(s3, settings, run_ref_or_id: str, *, prefix: str = ""):
    effective_prefix = _artifact_discovery_prefix(settings, prefix)
    resolution = None
    if prefix:
        normalized_run = _validate_run_basename(run_ref_or_id)
        artifacts = list_artifacts(
            settings["bucket"], normalized_run, prefix=effective_prefix, s3=s3
        )
        if artifacts:
            resolution = RunResolution(
                normalized_run, settings["bucket"], effective_prefix, artifacts
            )
    else:
        allowed_buckets, _scope = _agent_artifact_list_scope(
            _agent_access_report(), "", ""
        )
        resolution = resolve_run_artifacts(
            allowed_buckets,
            base_prefix=settings.get("prefix", ""),
            run_ref_or_id=run_ref_or_id,
            s3=s3,
        )
    if resolution is None:
        raise HTTPException(
            status_code=404,
            detail="run artifacts not found in configured S3 storage",
        )
    return (
        resolution.run_id,
        resolution.bucket,
        resolution.artifacts,
        resolution.source_prefix,
    )


{_AGENT_ARTIFACT_CONTENT_EMBED}


@app.get("/artifacts/run/{{run_id:path}}")
def artifacts_for_run(
    run_id: str,
    prefix: str = "",
    cursor: str = "",
    resolved_prefix: str = "",
    resource_bucket: str = "",
    project_id: str = "", source_selected: bool = False,
):
    try:
        requested_ref = str(run_id or "").strip()
        if requested_ref.startswith("npa1_"):
            ref_bucket, ref_prefix, normalized_run = decode_run_ref(requested_ref)
            if resource_bucket and resource_bucket != ref_bucket:
                raise ArtifactDiscoveryError("run_ref conflicts with resource_bucket")
            requested_selector = _validated_resolved_prefix(resolved_prefix or prefix)
            if requested_selector and requested_selector != ref_prefix:
                raise ArtifactDiscoveryError("run_ref conflicts with resolved_prefix")
            resource_bucket = ref_bucket
            resolved_prefix = ref_prefix
            source_selected = True
        else:
            normalized_run = validate_run_id(requested_ref)
        s3, settings = _agent_s3_client()
        exact_source_request = bool(
            requested_ref.startswith("npa1_") and resource_bucket and project_id
        )
        if exact_source_request:
            source_bucket, source_project, source_prefix = _authorize_exact_run_ref_source(
                s3=s3,
                settings=settings,
                run_id=normalized_run,
                run_ref=requested_ref,
                resource_bucket=resource_bucket,
                project_id=project_id,
                resolved_prefix=resolved_prefix,
            )
            resource_bucket = source_bucket
            project_id = source_project
            resolved_prefix = source_prefix
            allowed_buckets = [source_bucket]
            bucket_projects = {{source_bucket: source_project}}
            access_report = None
            access_diagnostics = {{
                "status": "available",
                "scope": "selected_source",
                "searched_projects": [{{"id": source_project, "name": source_project}}],
                "unavailable_projects": [],
            }}
        else:
            access_report = _agent_access_report()
            bucket_projects = artifact_bucket_projects(access_report)
            allowed_buckets, _selected_scope = _agent_artifact_list_scope(
                access_report, resource_bucket, project_id
            )
            access_diagnostics = _agent_access_diagnostics(access_report)
        search_buckets = [resource_bucket] if resource_bucket else allowed_buckets
        requested_prefix = _validated_resolved_prefix(resolved_prefix or prefix)
        matches, source_errors, discovery_complete = find_run_sources_across_buckets(
            search_buckets,
            base_prefix=settings.get("prefix", ""),
            run_id=normalized_run,
            exact_prefix=(requested_prefix if requested_prefix else "")
            if resource_bucket and (requested_prefix or source_selected)
            else None,
            exclude=_discovery_exclude_roots(),
            bucket_projects=bucket_projects,
            s3=s3,
        )
        if resource_bucket:
            matches = [item for item in matches if item.bucket == resource_bucket]
        if project_id:
            matches = [item for item in matches if item.project_id == project_id]
        if requested_prefix:
            matches = [item for item in matches if item.resolved_prefix == requested_prefix]
        elif source_selected:
            matches = [item for item in matches if not item.resolved_prefix]
        search_complete = bool(
            discovery_complete
            and not source_errors
            and (
                resource_bucket
                or (
                    access_report is not None
                    and _artifact_search_scope_complete(access_report)
                )
            )
        )
        if not matches:
            code = "run_not_discovered" if search_complete else "artifact_search_incomplete"
            status_code = 404 if search_complete else 503
            message = (
                "No discovered NPA workflow/artifact run has this identifier. "
                "Identifiers under /home/ubuntu/codex-runs are Codex maintenance job IDs, not NPA run IDs."
                if search_complete
                else "The run could not be resolved because one or more tenant artifact sources are inaccessible or incomplete."
            )
            return JSONResponse(
                status_code=status_code,
                content={{
                    "ok": False,
                    "error": {{"code": code, "message": message}},
                    "run_id": normalized_run,
                    "namespace": "npa_workflow_artifact_run",
                    "access": access_diagnostics,
                    "source_errors": [dict(item) for item in source_errors],
                }},
            )
        exact_selection = bool(resource_bucket and (requested_prefix or source_selected))
        if not exact_selection and not search_complete:
            return JSONResponse(
                status_code=503,
                content={{
                    "ok": False,
                    "error": {{
                        "code": "artifact_search_incomplete",
                        "message": "The run could not be selected uniquely because artifact discovery was incomplete.",
                    }},
                    "run_id": normalized_run,
                    "namespace": "npa_workflow_artifact_run",
                    "access": access_diagnostics,
                    "source_errors": [dict(item) for item in source_errors],
                }},
            )
        if len(matches) > 1:
            try:
                resolutions = [
                    RunResolution(
                        run_id=normalized_run,
                        bucket=item.bucket,
                        source_prefix=item.resolved_prefix,
                        artifacts=list_artifacts(
                            item.bucket,
                            normalized_run,
                            prefix=item.resolved_prefix,
                            s3=s3,
                        ),
                    )
                    for item in matches
                ]
                complete = prefer_complete_run_resolution(resolutions)
            except Exception:  # noqa: BLE001 - ambiguity remains the safe fallback
                complete = None
            if complete is None:
                return JSONResponse(
                    status_code=409,
                    content={{
                        "ok": False,
                        "error": {{
                            "code": "ambiguous_run_id",
                            "message": "This run ID exists in multiple artifact sources; select a project, bucket, and resolved prefix.",
                        }},
                        "run_id": normalized_run,
                        "sources": [item.to_dict() for item in matches],
                        "access": access_diagnostics,
                    }},
                )
            matches = [
                item
                for item in matches
                if item.bucket == complete.bucket
                and item.resolved_prefix == complete.source_prefix
            ]
        selected = matches[0]
        run_bucket = selected.bucket
        artifact_prefix = selected.resolved_prefix
        page = list_artifacts_page(
            run_bucket,
            normalized_run,
            prefix=artifact_prefix,
            cursor=cursor,
            s3=s3,
        )
        preferred = select_preferred_artifact(page.artifacts)
        role_counts = artifact_inventory_counts(page.artifacts)
        summary = build_run_summary(
            normalized_run,
            page.artifacts,
            _summary_documents_for_run(s3, run_bucket, page.artifacts),
        )
        if exact_source_request:
            # The card is rendered immediately before its playback action. Keep
            # the just-proven narrow authorization warm for the same 30-second
            # window as the effective-access cache so export need not repeat
            # project inventory and bucket probing. Export still re-HEADs the
            # exact object and validates its strong identity/provenance.
            _remember_exact_run_ref_source_authorization(
                run_id=normalized_run,
                run_ref=requested_ref,
                resource_bucket=run_bucket,
                project_id=str(bucket_projects.get(run_bucket) or ""),
                resolved_prefix=artifact_prefix,
            )
            _remember_foxglove_exact_artifact_inventory(
                run_id=normalized_run,
                run_ref=requested_ref,
                resource_bucket=run_bucket,
                project_id=str(bucket_projects.get(run_bucket) or ""),
                resolved_prefix=artifact_prefix,
                artifacts=page.artifacts,
            )
        return {{
            "ok": True,
            "contract": ARTIFACT_DISCOVERY_CONTRACT,
            "bucket": run_bucket,
            "project_id": str(bucket_projects.get(run_bucket) or ""),
            "prefix": artifact_prefix,
            "resolved_prefix": artifact_prefix,
            "base_prefix": settings.get("prefix", ""),
            "run_id": normalized_run,
            "run_ref": selected.run_ref,
            "pagination": {{
                "contract": "one_native_s3_page",
                "max_objects": 1000,
                "continue_with": ["next_cursor", "resolved_prefix", "resource_bucket", "source_selected"],
            }},
            "preferred": preferred.to_dict() if preferred else None,
            "access": access_diagnostics,
            **page.to_dict(),
            "output_artifact_count": role_counts["output"],
            "input_artifact_count": role_counts["input"],
            "metadata_artifact_count": role_counts["metadata"],
            "summary": summary,
            "no_recording": not bool(summary.get("has_recording")),
            "recording_state": str(summary.get("recording_state") or ""),
        }}
    except AmbiguousRunError as exc:
        raise HTTPException(
            status_code=409,
            detail={{"error": str(exc), "run_id": exc.run_id, "run_refs": exc.references}},
        ) from exc
    except ArtifactDiscoveryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        return JSONResponse(status_code=502, content={{"ok": False, "error": str(exc), "source": "s3"}})
_SENSITIVE_ARTIFACT_INFO_KEY = _SENSITIVE_PUBLIC_NAME
_SENSITIVE_ARTIFACT_INFO_VALUE = _SENSITIVE_PUBLIC_VALUE


def _public_artifact_info(value, depth: int = 0):
    # Artifact metadata may contain operator-authored config. Preserve its shape
    # for inspection while never reflecting credential-bearing fields or values.
    if depth > 8:
        return "<truncated>"
    if isinstance(value, dict):
        return {{
            str(key): (
                "<redacted>"
                if _SENSITIVE_ARTIFACT_INFO_KEY.search(str(key))
                else _public_artifact_info(item, depth + 1)
            )
            for key, item in value.items()
        }}
    if isinstance(value, list):
        return [_public_artifact_info(item, depth + 1) for item in value[:100]]
    if isinstance(value, str):
        public = _public_url_without_credentials(value)
        return "<redacted>" if _SENSITIVE_ARTIFACT_INFO_VALUE.search(public) else public
    return value


@app.get("/artifacts/stage/{{run_id:path}}")
def artifacts_stage(
    run_id: str,
    stage_key: str = "",
    prefix: str = "",
    resource_bucket: str = "",
    project_id: str = "",
    resolved_prefix: str = "", source_selected: bool = False,
):
    # Describe one pipeline stage and return its artifacts + inlined info/config
    # JSON so an operator can click a stage and manually inspect it (grounded in
    # the run's real S3 objects, no LLM call).
    try:
        normalized_run = validate_run_id(run_id)
        exact_prefix = _validated_resolved_prefix(resolved_prefix)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        s3, settings = _agent_s3_client()
        artifacts = []
        run_bucket = settings["bucket"]
        access_report = _agent_access_report()
        bucket_projects = artifact_bucket_projects(access_report)
        if resource_bucket:
            run_bucket, selected_project, exact_prefix, artifacts = (
                _load_selected_run_artifacts(
                    s3=s3,
                    settings=settings,
                    run_id=normalized_run,
                    resource_bucket=resource_bucket,
                    project_id=project_id,
                    resolved_prefix=exact_prefix,
                    source_selected=source_selected,
                    exclude=_discovery_exclude_roots(),
                )
            )
            if selected_project:
                bucket_projects[run_bucket] = selected_project
        elif prefix:
            artifacts = list_artifacts(
                settings["bucket"], normalized_run, prefix=_artifact_discovery_prefix(settings, prefix), s3=s3
            )
        if not artifacts and not resource_bucket:
            run_bucket, artifacts = find_run_artifacts_across_buckets(
                _agent_s3_buckets(s3, settings), base_prefix=settings.get("prefix", ""), run_id=normalized_run, s3=s3
            )
            if not run_bucket:
                run_bucket = settings["bucket"]
        wanted = str(stage_key or "").strip()
        keys = [str(item.key or "") for item in artifacts]
        marker = "/" + normalized_run + "/"
        effective_prefix = exact_prefix if resource_bucket else (
            keys[0].split(marker, 1)[0]
            if keys and marker in keys[0]
            else settings.get("prefix", "")
        )
        wrapper = run_stage_wrapper(keys, normalized_run, effective_prefix)
        stage_arts = [
            a for a in artifacts
            if not wanted
            or artifact_stage_key(
                str(a.key or ""), normalized_run, effective_prefix, wrapper
            ) == wanted
        ]
        label = _stage_label(wanted)
        description = _stage_description(wanted, label, len(stage_arts))
        # Inline small JSON artifacts (configs, grade, decision, reports, manifest)
        # so the stage's info/config is inspectable without a second round-trip.
        info: dict[str, Any] = {{}}
        for a in stage_arts:
            k = str(a.key or "")
            leaf = Path(k).name
            if not leaf.endswith(".json"):
                continue
            if int(getattr(a, "size", 0) or 0) > 65536 or len(info) >= 8:
                continue
            rel = k.split("/" + normalized_run + "/", 1)[-1]
            try:
                payload = _read_bounded_json_object(s3, run_bucket, k)
                if isinstance(payload, dict):
                    info[rel] = _public_artifact_info(payload)
            except Exception:
                continue
        return {{
            "ok": True,
            "run_id": normalized_run,
            "run_ref": encode_run_ref(run_bucket, effective_prefix, normalized_run),
            "project_id": str(bucket_projects.get(run_bucket) or project_id or ""),
            "bucket": run_bucket,
            "resolved_prefix": effective_prefix,
            "stage_key": wanted,
            "label": label,
            "description": description,
            "count": len(stage_arts),
            "artifacts": [item.to_dict() for item in stage_arts],
            "info": info,
        }}
    except HTTPException:
        raise
    except Exception as exc:
        return JSONResponse(status_code=502, content={{"ok": False, "error": str(exc), "source": "s3"}})


@app.get("/fiftyone/dataset/{{run_id:path}}")
def fiftyone_dataset(
    run_id: str,
    prefix: str = "",
    resource_bucket: str = "",
    project_id: str = "",
    resolved_prefix: str = "",
    source_selected: bool = False,
):
    # FiftyOne / Voxel51 view of a data-factory run: augmented scenario variants
    # (thumbnail + appearance tags + caption + video) and input frames as samples,
    # summarized with grade + curation. Grounded in the run's real S3 artifacts.
    try:
        normalized_run = validate_run_id(run_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        s3, settings = _agent_s3_client()
        artifacts = []
        bucket = settings["bucket"]
        selected_project = ""
        exact_prefix = _validated_resolved_prefix(resolved_prefix)
        if resource_bucket:
            bucket, selected_project, exact_prefix, artifacts = _load_selected_run_artifacts(
                s3=s3,
                settings=settings,
                run_id=normalized_run,
                resource_bucket=resource_bucket,
                project_id=project_id,
                resolved_prefix=exact_prefix,
                source_selected=source_selected,
                exclude=_discovery_exclude_roots(),
            )
        elif prefix:
            artifacts = list_artifacts(
                settings["bucket"], normalized_run, prefix=_artifact_discovery_prefix(settings, prefix), s3=s3
            )
        if not artifacts and not resource_bucket:
            bucket, artifacts = find_run_artifacts_across_buckets(
                _agent_s3_buckets(s3, settings), base_prefix=settings.get("prefix", ""), run_id=normalized_run, s3=s3
            )
            if not bucket:
                bucket = settings["bucket"]

        def _read_json(key: str):
            if not key:
                return None
            try:
                body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
                return json.loads(body)
            except Exception:
                return None

        keys = [str(a.key or "") for a in artifacts]
        dataset = build_fiftyone_dataset(keys, run_id=normalized_run, read_json=_read_json, bucket=bucket)
        return {{
            "ok": True,
            "run_id": normalized_run,
            "run_ref": encode_run_ref(bucket, exact_prefix, normalized_run),
            "bucket": bucket,
            "project_id": selected_project or project_id,
            "resolved_prefix": exact_prefix,
            **dataset,
        }}
    except HTTPException:
        raise
    except Exception as exc:
        return JSONResponse(status_code=502, content={{"ok": False, "error": str(exc), "source": "s3"}})


@app.get("/artifacts/provenance/{{run_id:path}}")
def artifacts_run_provenance(
    run_id: str,
    prefix: str = "",
    resource_bucket: str = "",
    project_id: str = "",
    resolved_prefix: str = "",
    source_selected: bool = False,
):
    # Where a run's data came from in the pipeline + which components produced it,
    # grounded in the run's real artifacts (and manifests) so Describe-this can
    # explain provenance instead of guessing.
    try:
        normalized_run = validate_run_id(run_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        s3, settings = _agent_s3_client()
        artifacts = []
        run_bucket = settings["bucket"]
        selected_project = ""
        exact_prefix = _validated_resolved_prefix(resolved_prefix)
        if resource_bucket:
            run_bucket, selected_project, exact_prefix, artifacts = _load_selected_run_artifacts(
                s3=s3,
                settings=settings,
                run_id=normalized_run,
                resource_bucket=resource_bucket,
                project_id=project_id,
                resolved_prefix=exact_prefix,
                source_selected=source_selected,
                exclude=_discovery_exclude_roots(),
            )
        elif prefix:
            artifacts = list_artifacts(settings["bucket"], normalized_run, prefix=_artifact_discovery_prefix(settings, prefix), s3=s3)
        if not artifacts and not resource_bucket:
            run_bucket, artifacts = find_run_artifacts_across_buckets(
                _agent_s3_buckets(s3, settings),
                base_prefix=settings.get("prefix", ""),
                run_id=normalized_run,
                s3=s3,
            )
        keys = [str(a.key or "") for a in artifacts]

        def _read_json(key: str):
            if not key:
                return None
            try:
                body = s3.get_object(Bucket=run_bucket, Key=key)["Body"].read()
                return json.loads(body)
            except Exception:
                return None

        prov = build_run_provenance(keys, run_id=normalized_run, read_json=_read_json)
        return {{
            "ok": True,
            "run_id": normalized_run,
            "bucket": run_bucket,
            "project_id": selected_project or project_id,
            "resolved_prefix": exact_prefix,
            **prov,
        }}
    except HTTPException:
        raise
    except Exception as exc:
        return JSONResponse(status_code=502, content={{"ok": False, "error": str(exc), "source": "s3"}})


@app.post("/sim-viz/load-artifact")
def sim_viz_load_artifact(payload: dict | None = None):
    body = payload if isinstance(payload, dict) else {{}}
    requested_uri = str(body.get("s3_uri") or "").strip()
    requested_run = str(body.get("run_id") or "").strip()
    requested_run_ref = str(body.get("run_ref") or "").strip()
    requested_key = str(body.get("key") or "").strip()
    requested_bucket, requested_project, requested_prefix, source_selected = (
        _selected_run_request(body)
    )
    exact_source_request = bool(
        requested_run_ref
        and requested_bucket
        and requested_project
        and "resolved_prefix" in body
        and source_selected
    )
    if not requested_uri and not (requested_run and requested_key):
        raise HTTPException(status_code=400, detail="Provide either s3_uri or run_id + key")
    try:
        s3, settings = _agent_s3_client()
        resolved_ref = ""
        resolution = None
        selected_source_identity = None
        if requested_uri:
            if not (requested_run_ref or requested_run):
                raise HTTPException(
                    status_code=400,
                    detail={{
                        "schema": "npa.agent.api_error/v1",
                        "contract_version": "npa.agent.load-artifact.v2",
                        "code": "run_id_required_for_s3_uri",
                        "message": "run_id or server-issued run_ref is required with s3_uri",
                        "migration": {{
                            "required_fields": ["run_id", "s3_uri"],
                            "preferred_fields": ["run_ref", "key"],
                            "discover_via": [
                                "GET /api/artifacts/runs",
                                "GET /api/artifacts/run/{{run_id_or_run_ref}}",
                            ],
                            "security_boundary": "only server-discovered inventory objects may be loaded",
                        }},
                    }},
                )
            bucket, key = parse_s3_uri(requested_uri)
            key = _safe_artifact_key(key)
            s3_uri = requested_uri
            if requested_run_ref:
                ref_bucket, ref_prefix, ref_run_id = decode_run_ref(
                    requested_run_ref
                )
                ref_scope = "/".join(
                    part for part in (ref_prefix, ref_run_id) if part
                ) + "/"
                if (
                    bucket != ref_bucket
                    or not key.startswith(ref_scope)
                    or (requested_run and requested_run != ref_run_id)
                ):
                    raise HTTPException(
                        status_code=400,
                        detail="artifact URI is outside the selected run",
                    )
                if exact_source_request:
                    source_bucket, source_project, source_prefix = (
                        _authorize_exact_run_ref_source(
                            s3=s3,
                            settings=settings,
                            run_id=ref_run_id,
                            run_ref=requested_run_ref,
                            resource_bucket=requested_bucket,
                            project_id=requested_project,
                            resolved_prefix=requested_prefix,
                        )
                    )
                    run_id, bucket, artifact = _resolved_artifact_for_content(
                        s3,
                        settings,
                        run_id=ref_run_id,
                        key=key,
                        requested_bucket=source_bucket,
                        exact_membership=True,
                        source_authorized=True,
                        resolved_prefix=source_prefix,
                    )
                    selected_source_identity = (
                        source_bucket,
                        source_project,
                        source_prefix,
                    )
                else:
                    # Compatibility callers that have only a run_ref still use
                    # bounded inventory membership; browser cards always send
                    # the complete source tuple and take the exact fast path.
                    run_id, bucket, artifact = _resolved_artifact_for_content(
                        s3,
                        settings,
                        run_id=ref_run_id,
                        key=key,
                        requested_bucket=ref_bucket,
                        exact_membership=True,
                    )
                key = str(artifact.key)
                s3_uri = str(artifact.s3_uri)
                resolved_ref = encode_run_ref(bucket, ref_prefix, run_id)
            elif requested_run:
                # A plain run basename may exist under several exact sources.
                # The caller already supplied an exact URI, so authorize that
                # bucket/key against the requested run instead of resolving the
                # ambiguous basename to an arbitrary sibling source.
                run_id, bucket, artifact = _resolved_artifact_for_content(
                    s3,
                    settings,
                    run_id=requested_run,
                    key=key,
                    requested_bucket=bucket,
                    exact_membership=True,
                )
                key = str(artifact.key)
                s3_uri = str(artifact.s3_uri)
        else:
            key = _safe_artifact_key(requested_key)
            if exact_source_request:
                source_bucket, source_project, source_prefix = (
                    _authorize_exact_run_ref_source(
                        s3=s3,
                        settings=settings,
                        run_id=requested_run,
                        run_ref=requested_run_ref,
                        resource_bucket=requested_bucket,
                        project_id=requested_project,
                        resolved_prefix=requested_prefix,
                    )
                )
                run_id, bucket, artifact = _resolved_artifact_for_content(
                    s3,
                    settings,
                    run_id=requested_run,
                    key=key,
                    requested_bucket=source_bucket,
                    exact_membership=True,
                    source_authorized=True,
                    resolved_prefix=source_prefix,
                )
                key = str(artifact.key)
                s3_uri = str(artifact.s3_uri)
                resolved_ref = requested_run_ref
                selected_source_identity = (
                    source_bucket,
                    source_project,
                    source_prefix,
                )
            else:
                resolution = resolve_run_artifacts(
                    _agent_s3_buckets(s3, settings),
                    base_prefix=settings.get("prefix", ""),
                    run_ref_or_id=requested_run_ref or requested_run,
                    s3=s3,
                )
                if resolution is None:
                    raise HTTPException(status_code=404, detail="run_id not found")
                if key not in {{item.key for item in resolution.artifacts}}:
                    raise HTTPException(status_code=400, detail="artifact key is outside the selected run")
                run_id = resolution.run_id
                bucket = resolution.bucket
                resolved_ref = resolution.run_ref
                s3_uri = f"s3://{{bucket}}/{{key}}"
        local_name = _artifact_filename(key)
        local_path = RECORDINGS_DIR / local_name
        download_s3_uri(s3_uri, local_path, s3=s3)
        render = render_hint_for_object(key=key)
        state = _load_state()
        if selected_source_identity is not None:
            source_bucket, source_project, source_prefix = selected_source_identity
        else:
            source_bucket, source_project, source_prefix = _artifact_source_metadata(
                _agent_access_report(), bucket, key, run_id
            )
        run_artifacts = resolution.artifacts if resolution is not None else []
        run_summary = build_run_summary(
            run_id,
            run_artifacts,
            _summary_documents_for_run(s3, bucket, run_artifacts),
        )
        learning_summary = run_summary.get("learning")
        learning_contract = (
            learning_summary.get("artifact_contract")
            if isinstance(learning_summary, dict)
            else None
        )
        sim_viz = _apply_loaded_artifact(
            state=state,
            run_id=run_id,
            key=key,
            s3_uri=s3_uri,
            render=render,
            local_path=local_path,
            source_identity=(source_bucket, source_project, source_prefix),
            run_ref=resolved_ref,
            artifact_contract=learning_contract,
        )
        return {{"ok": True, "contract": ARTIFACT_DISCOVERY_CONTRACT, "sim_viz": sim_viz, "render": render, "artifact_uri": s3_uri, "run_ref": resolved_ref}}
    except AmbiguousRunError as exc:
        raise HTTPException(
            status_code=409,
            detail={{"error": "run_id is ambiguous; provide run_ref",
                     "run_id": exc.run_id, "run_refs": exc.references}},
        ) from exc
    except ArtifactDiscoveryError as exc:
        raise HTTPException(status_code=400, detail="invalid run artifact request") from exc
    except HTTPException:
        raise
    except Exception:
        logging.getLogger("npa.agent.artifact_load").exception(
            "Artifact load storage request failed"
        )
        return JSONResponse(
            status_code=502,
            content={{
                "ok": False,
                "error": "artifact storage request failed",
                "error_code": "artifact_storage_error",
                "source": "s3",
            }},
        )


def _foxglove_convert_run(**kwargs):
    from npa.sdk.workbench.foxglove import convert_run

    return convert_run(**kwargs)


_FOXGLOVE_EXACT_INVENTORY_TTL_SECONDS = 30.0
_FOXGLOVE_EXACT_INVENTORY_CACHE: dict[tuple[str, ...], tuple[float, tuple]] = {{}}
_FOXGLOVE_EXACT_INVENTORY_LOCK = threading.Lock()


def _foxglove_exact_inventory_key(
    *, run_id: str, run_ref: str, resource_bucket: str, project_id: str, resolved_prefix: str
) -> tuple[str, ...]:
    return (
        str(os.environ.get("NEBIUS_TENANT_ID") or "").strip(),
        str(os.environ.get("NEBIUS_PROJECT_ID") or "").strip(),
        str(project_id or "").strip(),
        str(resource_bucket or "").strip(),
        str(resolved_prefix or "").strip(),
        str(run_id or "").strip(),
        str(run_ref or "").strip(),
    )


def _remember_foxglove_exact_artifact_inventory(
    *,
    run_id: str,
    run_ref: str,
    resource_bucket: str,
    project_id: str,
    resolved_prefix: str,
    artifacts,
) -> None:
    key = _foxglove_exact_inventory_key(
        run_id=run_id,
        run_ref=run_ref,
        resource_bucket=resource_bucket,
        project_id=project_id,
        resolved_prefix=resolved_prefix,
    )
    with _FOXGLOVE_EXACT_INVENTORY_LOCK:
        _FOXGLOVE_EXACT_INVENTORY_CACHE[key] = (
            time.monotonic() + _FOXGLOVE_EXACT_INVENTORY_TTL_SECONDS,
            tuple(artifacts or ()),
        )


def _cached_foxglove_exact_artifact_resolution(
    *,
    run_id: str,
    run_ref: str,
    resource_bucket: str,
    project_id: str,
    resolved_prefix: str,
):
    key = _foxglove_exact_inventory_key(
        run_id=run_id,
        run_ref=run_ref,
        resource_bucket=resource_bucket,
        project_id=project_id,
        resolved_prefix=resolved_prefix,
    )
    now_mono = time.monotonic()
    with _FOXGLOVE_EXACT_INVENTORY_LOCK:
        entry = _FOXGLOVE_EXACT_INVENTORY_CACHE.get(key)
        if entry is None or entry[0] <= now_mono:
            _FOXGLOVE_EXACT_INVENTORY_CACHE.pop(key, None)
            return None
        artifacts = list(entry[1])
    return RunResolution(run_id, resource_bucket, resolved_prefix, artifacts)


def _foxglove_artifact_fingerprint(s3, bucket: str, artifact) -> tuple[str, int, str]:
    key = str(artifact.key)
    try:
        head = s3.head_object(Bucket=bucket, Key=key)
    except Exception:
        logging.getLogger("npa.agent.foxglove").exception(
            "Could not read authoritative MCAP object identity"
        )
        return "", int(getattr(artifact, "size", 0) or 0), str(
            getattr(artifact, "last_modified", "") or ""
        )
    modified = head.get("LastModified") or getattr(artifact, "last_modified", "")
    if hasattr(modified, "isoformat"):
        modified = modified.isoformat()
    size = int(head.get("ContentLength") or getattr(artifact, "size", 0) or 0)
    fingerprint = artifact_source_fingerprint(
        bucket=bucket,
        key=key,
        size=size,
        last_modified=str(modified or ""),
        etag=str(head.get("ETag") or ""),
        version_id=str(head.get("VersionId") or ""),
    )
    return fingerprint, size, str(modified or "")


def _foxglove_resolve_artifact(payload: dict) -> dict:
    body = payload if isinstance(payload, dict) else {{}}
    run_id = str(body.get("run_id") or "").strip()
    run_ref = str(body.get("run_ref") or "").strip()
    key = _safe_artifact_key(str(body.get("key") or ""))
    if not run_id or not key:
        raise HTTPException(status_code=400, detail="run_id and key are required")
    s3, settings = _agent_s3_client()
    requested_bucket = str(body.get("resource_bucket") or body.get("bucket") or "").strip()
    requested_project = str(body.get("project_id") or "").strip()
    exact_source_request = bool(
        run_ref
        and requested_bucket
        and requested_project
        and "resolved_prefix" in body
    )
    if exact_source_request:
        source_bucket, source_project, source_prefix = _authorize_exact_run_ref_source(
            s3=s3,
            settings=settings,
            run_id=run_id,
            run_ref=run_ref,
            resource_bucket=requested_bucket,
            project_id=requested_project,
            resolved_prefix=str(body.get("resolved_prefix") or ""),
        )
        resolution_buckets = [source_bucket]
        resolution = _cached_foxglove_exact_artifact_resolution(
            run_id=run_id,
            run_ref=run_ref,
            resource_bucket=source_bucket,
            project_id=source_project,
            resolved_prefix=source_prefix,
        )
        if resolution is None:
            matches, source_errors, discovery_complete = find_run_sources_across_buckets(
                [source_bucket],
                base_prefix=settings.get("prefix", ""),
                run_id=run_id,
                exact_prefix=source_prefix,
                exclude=_discovery_exclude_roots(),
                bucket_projects={{source_bucket: source_project}},
                s3=s3,
            )
            exact_matches = [
                item
                for item in matches
                if item.bucket == source_bucket
                and item.project_id == source_project
                and item.resolved_prefix == source_prefix
            ]
            if source_errors or not discovery_complete:
                raise HTTPException(
                    status_code=503,
                    detail="the selected Foxglove artifact source could not be verified",
                )
            if not exact_matches:
                raise HTTPException(status_code=404, detail="run_id not found")
            if len(exact_matches) > 1:
                raise HTTPException(
                    status_code=409,
                    detail="the selected Foxglove artifact source is ambiguous",
                )
            namespaces = exact_matches[0].namespaces or (source_prefix,)
            resolution = RunResolution(
                run_id,
                source_bucket,
                source_prefix,
                [
                    artifact
                    for namespace in namespaces
                    for artifact in list_artifacts(
                        source_bucket,
                        run_id,
                        prefix=namespace,
                        s3=s3,
                    )
                ],
            )
    else:
        resolution_buckets = _agent_s3_buckets(s3, settings)
        resolution = None
    if resolution is None:
        resolution = resolve_run_artifacts(
            resolution_buckets,
            base_prefix=settings.get("prefix", ""),
            run_ref_or_id=run_ref or run_id,
            s3=s3,
        )
    if resolution is None:
        raise HTTPException(status_code=404, detail="run_id not found")
    if exact_source_request:
        _remember_foxglove_exact_artifact_inventory(
            run_id=resolution.run_id,
            run_ref=resolution.run_ref,
            resource_bucket=source_bucket,
            project_id=source_project,
            resolved_prefix=source_prefix,
            artifacts=resolution.artifacts,
        )
    artifact = next((item for item in resolution.artifacts if item.key == key), None)
    if artifact is None:
        raise HTTPException(status_code=400, detail="artifact key is outside the selected run")
    if not exact_source_request:
        source_bucket, source_project, source_prefix = _artifact_source_metadata(
            _agent_access_report(), resolution.bucket, key, resolution.run_id
        )
    selected = {{
        "run_id": resolution.run_id,
        "run_ref": resolution.run_ref,
        "key": key,
        "s3_uri": str(artifact.s3_uri),
        "bucket": source_bucket or resolution.bucket,
        "resource_bucket": source_bucket or resolution.bucket,
        "project_id": source_project,
        "resolved_prefix": source_prefix,
    }}
    for requested_field, actual_field in (
        ("bucket", "bucket"),
        ("project_id", "project_id"),
        ("resolved_prefix", "resolved_prefix"),
        ("s3_uri", "s3_uri"),
    ):
        request_key = (
            "resource_bucket"
            if requested_field == "bucket" and "resource_bucket" in body
            else requested_field
        )
        if request_key in body and str(body.get(request_key) or "") != str(
            selected.get(actual_field) or ""
        ):
            raise HTTPException(
                status_code=409,
                detail=f"the selected Foxglove artifact {{requested_field}} does not match discovery",
            )
    fingerprint, size, last_modified = _foxglove_artifact_fingerprint(
        s3, resolution.bucket, artifact
    )
    selected.update(
        {{
            "source_fingerprint": fingerprint,
            "source_size_bytes": size,
            "source_last_modified": last_modified,
        }}
    )
    return selected


def _foxglove_prepare_canonical_mcap(
    *, run_id: str, run_ref: str = "", fps: float, max_frames: int
):
    from npa.sdk.workbench.foxglove import inspect_mcap

    s3, settings = _agent_s3_client()
    resolution = resolve_run_artifacts(
        _agent_s3_buckets(s3, settings),
        base_prefix=settings.get("prefix", ""),
        run_ref_or_id=run_ref or run_id,
        s3=s3,
    )
    if resolution is None:
        raise RuntimeError(f"run_id not found: {{run_id}}")

    def _resolved_artifacts(*_args, **_kwargs):
        return resolution.bucket, list(resolution.artifacts)

    return prepare_canonical_mcap(
        run_id=run_id,
        source_bucket=resolution.bucket,
        source_prefix=resolution.source_prefix,
        fps=fps,
        max_frames=max_frames,
        validate_run_id=validate_run_id,
        s3_client=lambda: (s3, settings),
        list_buckets=_agent_s3_buckets,
        find_artifacts=_resolved_artifacts,
        safe_key=_safe_artifact_key,
        download=download_s3_uri,
        convert=_foxglove_convert_run,
        summarize=inspect_mcap,
        invalidate_cache=_run_list_cache_clear,
        now_iso=_now_iso,
        recordings_dir=RECORDINGS_DIR,
    )


def _foxglove_apply_prepared_canonical(
    *, canonical: dict, run_id: str, run_ref: str = ""
) -> dict:
    key = _safe_artifact_key(str(canonical.get("artifact_key") or ""))
    s3_uri = str(canonical.get("s3_uri") or "").strip()
    local_path = Path(str(canonical.get("local_path") or ""))
    if not key or not s3_uri or not local_path.is_file():
        raise HTTPException(
            status_code=502,
            detail="canonical MCAP persistence returned an incomplete local transport",
        )
    s3, _settings = _agent_s3_client()
    bucket, resolved_key = parse_s3_uri(s3_uri)
    if resolved_key != key:
        raise HTTPException(status_code=409, detail="canonical MCAP key changed before publication")
    head = s3.head_object(Bucket=bucket, Key=key)
    modified = head.get("LastModified") or ""
    if hasattr(modified, "isoformat"):
        modified = modified.isoformat()
    size = int(head.get("ContentLength") or local_path.stat().st_size)
    fingerprint = artifact_source_fingerprint(
        bucket=bucket,
        key=key,
        size=size,
        last_modified=str(modified or ""),
        etag=str(head.get("ETag") or ""),
        version_id=str(head.get("VersionId") or ""),
    )
    source_bucket, source_project, source_prefix = _artifact_source_metadata(
        _agent_access_report(), bucket, key, run_id
    )
    state = _load_state()
    sim_viz = _apply_loaded_artifact(
        state=state,
        run_id=run_id,
        key=key,
        s3_uri=s3_uri,
        render="mcap",
        local_path=local_path,
        source_identity=(source_bucket, source_project, source_prefix),
        run_ref=run_ref,
        source_fingerprint=fingerprint,
        source_size_bytes=size,
        source_last_modified=str(modified or ""),
    )
    return {{"ok": True, "render": "mcap", "sim_viz": sim_viz, "run_ref": run_ref}}


def _foxglove_ensure_cloud_recording(
    local_path: Path, run_id: str, *, provenance: dict
):
    return ensure_recording_and_layout_from_credentials(
        local_path,
        run_id,
        provenance,
        credentials_path="/root/.npa/credentials.yaml",
    )


def _foxglove_ensure_cloud_layout(*, provenance: dict):
    return ensure_layout_from_credentials(
        provenance,
        credentials_path="/root/.npa/credentials.yaml",
    )


register_foxglove_routes(
    app,
    FoxgloveDeps(
        load_state=_load_state,
        save_state=_save_state,
        record_run=_record_sim_viz_run,
        foxglove_config=_foxglove_config,
        load_artifact=sim_viz_load_artifact,
        convert_run=_foxglove_convert_run,
        now_iso=_now_iso,
        validate_run_id=validate_run_id,
        data_dir=FOXGLOVE_DATA_DIR,
        runs_dir=Path("/opt/npa-agent/runs"),
        keep_published=FOXGLOVE_KEEP_PUBLISHED,
        ensure_cloud_recording=_foxglove_ensure_cloud_recording,
        ensure_cloud_layout=_foxglove_ensure_cloud_layout,
        prepare_canonical_mcap=_foxglove_prepare_canonical_mcap,
        resolve_artifact=_foxglove_resolve_artifact,
        apply_prepared_canonical=_foxglove_apply_prepared_canonical,
    ),
    HTTPException,
)


def _leisaac_manifest_for_run(run_id: str) -> dict | None:
    # An active private relay is authoritative for which single LeIsaac run
    # this agent can serve. Reject unrelated viewer selections before generic
    # artifact discovery; otherwise a polling UI can repeatedly walk the full
    # bucket for an ordinary sim-viz run while teleoperation is launching.
    try:
        credential = json.loads(
            Path("/etc/npa/leisaac-relay.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        credential = None
    if (
        isinstance(credential, dict)
        and str(credential.get("run_id") or "").strip()
        and credential.get("run_id") != run_id
    ):
        return None
    manifest = load_manifest_artifact(
        run_id, validate_run_id=validate_run_id,
        s3_client=_agent_s3_client, s3_buckets=_agent_s3_buckets,
        find_artifacts=find_run_artifacts_across_buckets,
        exact_uri=(
            str(credential.get("manifest_uri") or "")
            if isinstance(credential, dict) and credential.get("run_id") == run_id
            else ""
        ),
    )
    if not isinstance(manifest, dict):
        return None
    # Dataset manifests remain nonsecret. Runtime authority is injected from
    # the short-lived root-owned relay credential, which teardown removes.
    if not isinstance(credential, dict):
        return manifest
    if credential.get("run_id") != run_id:
        return manifest
    raw_expiry = str(credential.get("expires_at") or "").strip()
    if raw_expiry:
        try:
            expiry = datetime.fromisoformat(raw_expiry.replace("Z", "+00:00"))
        except ValueError:
            return manifest
        if expiry <= datetime.now(timezone.utc):
            return manifest
    resolved = dict(manifest)
    resolved["session_nonce"] = str(credential.get("session_nonce") or "")
    return resolved


register_leisaac_routes(
    app,
    LeIsaacDeps(
        load_state=_load_state,
        save_state=_save_state,
        resolve_manifest=_leisaac_manifest_for_run,
        http_get=httpx.get,
        http_post=httpx.post,
        response=Response,
        websocket_connect=_leisaac_websocket_connect,
        mutate_state=_mutate_state,
        s3_client=_agent_s3_client,
        s3_buckets=_agent_s3_buckets,
    ),
)


@app.post("/sim-viz/load-franka-demo")
def load_franka_demo(payload: dict | None = None):
    body = payload if isinstance(payload, dict) else {{}}
    camera = str(body.get("camera") or "").strip()
    if not camera:
        state = _load_state()
        selected = state.get("camera_selection", ["workspace"])
        if isinstance(selected, list) and selected:
            camera = str(selected[0])
        else:
            camera = "workspace"
    state = _load_state()
    viz = _wire_franka_demo(state, camera=camera, force_local_demo=True)
    return {{"ok": True, "sim_viz": viz, "selection": state["selection"]}}

@app.post("/sim-viz/camera-preview")
def sim_viz_camera_preview(payload: dict | None = None):
    body = payload if isinstance(payload, dict) else {{}}
    camera = str(body.get("camera") or "").strip()
    if not camera:
        state = _load_state()
        selected = state.get("camera_selection", ["workspace"])
        if isinstance(selected, list) and selected:
            camera = str(selected[0])
        else:
            camera = "workspace"
    cameras = DEFAULT_SCENE_SPEC.get("cameras", {{}})
    if camera not in cameras:
        raise HTTPException(status_code=404, detail=f"unknown camera: {{camera}}")
    state = _load_state()
    viz = _wire_franka_demo(state, camera=camera)
    entity_path = f"world/camera_frustums/{{camera}}/frustum"
    return {{
        "ok": True,
        "camera": camera,
        "entity_path": entity_path,
        "rollout_entity_guess": f"rollouts/latest/{{camera}}/camera",
        "sim_viz": viz,
        "hint": "Open the Rerun panel and expand world/camera_frustums/<name>.",
    }}

def _sim_viz_rrd_file_response(run_id: str = ""):
    state = _load_state()
    sim_viz = _sim_viz_for_run(state, run_id=run_id)
    uri = str(sim_viz.get("rrd_uri") or "").strip()
    if uri.startswith("file://"):
        if not file_uri_path_allowed(uri, allowed_paths=(str(RECORDINGS_DIR), str(RRD_PATH))):
            raise HTTPException(status_code=400, detail="Refusing to serve file:// rrd_uri outside recordings allowlist")
        file_path = Path(uri[len("file://"):]).expanduser().resolve()
        if file_path.is_file():
            return FileResponse(str(file_path), media_type="application/octet-stream")
    if uri.startswith("http://") or uri.startswith("https://"):
        # Resolve DNS once and fetch the vetted IP (DNS-rebinding TOCTOU guard).
        allowed, fetch_url, host_header = resolve_rrd_proxy_target(uri)
        if not allowed:
            raise HTTPException(status_code=400, detail="Refusing to proxy disallowed rrd_uri host")
        try:
            chunks: list[bytes] = []
            total = 0
            headers = {{"Host": host_header}} if host_header else {{}}
            with httpx.stream("GET", fetch_url, timeout=20.0, headers=headers) as proxied:
                proxied.raise_for_status()
                for chunk in proxied.iter_bytes(1024 * 1024):
                    total += len(chunk)
                    if total > MAX_RRD_PROXY_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail=f"Proxied rrd_uri exceeds {{MAX_RRD_PROXY_BYTES}} byte cap",
                        )
                    chunks.append(chunk)
            return Response(content=b"".join(chunks), media_type="application/octet-stream")
        except HTTPException:
            raise
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"Unable to fetch remote sim2real.rrd: {{exc}}") from exc
    if PRELOAD_STOCK_DEMO and RRD_PATH.is_file():
        return FileResponse(str(RRD_PATH), media_type="application/octet-stream")
    raise HTTPException(status_code=404, detail="No sim2real.rrd file on disk yet")

@app.get("/sim-viz/rrd")
def sim_viz_rrd(run_id: str = ""):
    return _sim_viz_rrd_file_response(run_id=run_id)

# HEAD as well as GET: the UI probes this endpoint with HEAD before choosing the
# viewer URL, and a GET-only route answers 405, logging a console error on every
# page load. The probe failure is caught and ignored, so this is cosmetic -- but
# an error that fires every load trains operators to ignore the console.
@app.api_route("/sim-viz/rrd-blob", methods=["GET", "HEAD"])
def sim_viz_rrd_blob(run_id: str = ""):
    # Authenticated .rrd bytes for parent-page blob URL (Rerun wasm cannot send basic auth).
    return _sim_viz_rrd_file_response(run_id=run_id)

@app.on_event("startup")
def _boot_preload_sim_viz() -> None:
    if not PRELOAD_STOCK_DEMO or not RRD_PATH.is_file():
        return
    state = _load_state()
    sim_viz = state.get("sim_viz", {{}})
    if not isinstance(sim_viz, dict):
        sim_viz = {{}}
    if str(sim_viz.get("rrd_uri") or "").strip() and _served_recording_is_run_specific():
        return
    capability_path = _publish_rrd_recording(RRD_PATH)
    if str(sim_viz.get("rrd_uri") or "").strip():
        sim_viz["artifact_preview_url"] = capability_path
        sim_viz["artifact_download_url"] = "/api/sim-viz/rrd-blob"
        sim_viz["rerun_iframe_url"] = _rerun_iframe_url(str(sim_viz.get("camera") or "workspace"), recording_path=capability_path)
        state["sim_viz"] = sim_viz
        _record_sim_viz_run(state, sim_viz)
        _save_state(state)
        return
    selected = state.get("camera_selection", ["workspace"])
    cam = str(selected[0] if isinstance(selected, list) and selected else "workspace")
    run_id = str(sim_viz.get("run_id") or "").strip() or "franka-demo"
    now = _now_iso()
    state["sim_viz"] = {{
        "run_id": run_id,
        "stage": "demo",
        "rrd_uri": f"file://{{RRD_PATH}}",
        "rrd_updated_at": now,
        "live_grpc_url": "",
        "mode": "static",
        "camera": cam,
        "preview_camera": cam,
        "preview_entity": f"world/camera_frustums/{{cam}}/frustum",
        "rerun_ready": _rerun_ready_state(rrd_uri=f"file://{{RRD_PATH}}"),
        "rerun_iframe_url": _rerun_iframe_url(cam, recording_path=capability_path),
        "artifact_preview_url": capability_path,
        "artifact_download_url": "/api/sim-viz/rrd-blob",
    }}
    _record_sim_viz_run(state, state["sim_viz"])
    _save_state(state)

@app.get("/sim-assets")
def sim_assets():
    state = _load_state()
    selection = state.get("selection", {{}})
    if not isinstance(selection, dict):
        selection = dict(DEFAULT_SELECTION)
    return {{
        "scene_spec": DEFAULT_SCENE_SPEC,
        "robot_spec": DEFAULT_ROBOT_SPEC,
        "assets_manifest": DEFAULT_ASSETS_MANIFEST,
        "selection": selection,
        "resolved_uris": {{
            "scene_spec_uri": selection.get("scene_spec_uri", ""),
            "assets_uri": selection.get("assets_uri", ""),
            "robot_spec_uri": selection.get("robot_spec_uri", ""),
            "cameras_uri": selection.get("cameras_uri", ""),
        }},
    }}

@app.get("/sim-assets/catalog")
def sim_assets_catalog():
    return {{
        "entries": [
            {{"name": "stock_scene", "uri": "stock://scene/default"}},
            {{"name": "stock_robot_franka", "uri": "stock://robot/franka"}},
            {{"name": "customer_assets_root", "uri": "s3://customer-assets/"}},
        ]
    }}

@app.get("/sim-assets/cameras")
def sim_assets_cameras():
    state = _load_state()
    selected = state.get("camera_selection", ["workspace"])
    cameras = []
    for entry in list(DEFAULT_SCENE_SPEC["cameras"].values()):
        if not isinstance(entry, dict):
            continue
        camera_name = str(entry.get("name") or "").strip()
        camera_payload = dict(entry)
        if camera_name:
            camera_payload["preview_url"] = f"/api/sim-viz/camera-preview?camera={{camera_name}}"
        cameras.append(camera_payload)
    return {{"cameras": cameras, "selected": selected}}

@app.put("/sim-assets/cameras/selection")
def set_camera_selection(payload: dict):
    selected = payload.get("selected", [])
    if not isinstance(selected, list):
        raise HTTPException(status_code=400, detail="selected must be a list")
    state = _load_state()
    state["camera_selection"] = [str(item) for item in selected if str(item).strip()]
    cam = state["camera_selection"][0] if state["camera_selection"] else "workspace"
    preset = str((state.get("selection") or {{}}).get("robot_preset", "")).strip().lower()
    if preset == "franka":
        viz = _wire_franka_demo(state, camera=cam)
        return {{"selected": state["camera_selection"], "sim_viz": viz}}
    _save_state(state)
    return {{"selected": state["camera_selection"]}}

@app.post("/sim-assets/selection")
def set_sim_assets_selection(payload: dict):
    state = _load_state()
    selection = dict(DEFAULT_SELECTION)
    current = state.get("selection", {{}})
    if isinstance(current, dict):
        selection.update(current)
    for key in ("scene_spec_uri", "assets_uri", "robot_spec_uri", "cameras_uri", "robot_preset", "sim_backend"):
        if key in payload and payload[key] is not None:
            selection[key] = str(payload[key]).strip()
    if "props" in payload and isinstance(payload["props"], list):
        selection["props"] = [str(item) for item in payload["props"] if str(item).strip()]
    state["selection"] = selection
    preset = str(selection.get("robot_preset", "")).strip().lower()
    if preset == "franka":
        cam = str((state.get("camera_selection") or ["workspace"])[0])
        viz = _wire_franka_demo(state, camera=cam)
        # Return the persisted selection (post-wire) so response matches state.
        persisted = state.get("selection") if isinstance(state.get("selection"), dict) else selection
        return {{"ok": True, "selection": persisted, "sim_viz": viz}}
    _save_state(state)
    return {{"ok": True, "selection": selection}}
@app.get("/sim-assets/selection")
def get_sim_assets_selection():
    state = _load_state()
    selection = state.get("selection", {{}})
    if not isinstance(selection, dict):
        selection = dict(DEFAULT_SELECTION)
    return selection
@app.get("/workflows/sim2real/status")
def sim2real_status(
    run_id: str = "",
    prefix: str = "",
    resource_bucket: str = "",
    project_id: str = "",
    resolved_prefix: str = "", source_selected: bool = False,
):
    state = _load_state()
    latest = state.get("latest_submit", {{}})
    sim_viz = state.get("sim_viz", {{}})
    details = _sim2real_run_details(
        state,
        run_id=run_id,
        prefix=prefix,
        resource_bucket=resource_bucket,
        project_id=project_id,
        resolved_prefix=resolved_prefix,
        source_selected=source_selected,
    )
    return {{
        "ok": True,
        "latest_submit": latest if isinstance(latest, dict) else {{}},
        "sim_viz": sim_viz if isinstance(sim_viz, dict) else dict(DEFAULT_SIM_VIZ),
        "run": details,
        "stages": details.get("stages", []),
        "logs": details.get("logs", []),
    }}

@app.get("/workflows/sim2real/runs/{{run_id:path}}")
def sim2real_run_detail(
    run_id: str,
    prefix: str = "",
    resource_bucket: str = "",
    project_id: str = "",
    resolved_prefix: str = "", source_selected: bool = False,
):
    state = _load_state()
    details = _sim2real_run_details(
        state,
        run_id=run_id,
        prefix=prefix,
        resource_bucket=resource_bucket,
        project_id=project_id,
        resolved_prefix=resolved_prefix,
        source_selected=source_selected,
    )
    if not str(details.get("run_id") or "").strip():
        raise HTTPException(status_code=404, detail=f"run_id not found: {{run_id}}")
    return {{"ok": True, "run": details}}

@app.get("/workbench/actions")
def workbench_actions():
    return {{
        "actions": [
            {{
                "id": "configure_s3",
                "title": "Configure S3",
                "hint": "Run `npa configure` on operator machine to set storage credentials.",
            }},
            {{
                "id": "setup_cosmos",
                "title": "Setup Cosmos3",
                "hint": "Use `npa workbench cosmos check|fetch` before inference workflows.",
            }},
            {{
                "id": "submit_sim2real",
                "title": "Submit Sim2Real",
                "hint": "POST /api/workflows/sim2real/submit after confirming selection.",
            }},
            {{
                "id": "watch_sim",
                "title": "Watch sim",
                "hint": "GET /api/sim-viz/status and open /rerun/ iframe.",
            }},
        ]
    }}


@app.get("/workflows/draft")
@app.get("/workflows/npa/draft")
def get_workflow_draft():
    state = _load_state()
    draft = _workflow_draft_from_state(state)
    return {{"ok": True, "draft": draft}}


@app.get("/infra/k8s")
@app.get("/infra/backends")
@app.get("/infra/mk8s")
def list_k8s_infra(project: str = ""):
    return _agent_k8s_backends(project)


@app.get("/resources")
@app.get("/tenant-resources")
def tenant_resources(refresh: bool = False):
    return _tenant_resource_inventory(force_refresh=bool(refresh))


@app.post("/infra/provision")
@app.post("/infra/k8s/provision")
@app.post("/infra/mk8s/provision")
def provision_infra(payload: dict | None = None):
    body = payload if isinstance(payload, dict) else {{}}
    project = _agent_project_alias(str(body.get("project") or ""))
    cluster_name = str(body.get("cluster_name") or "npa-cluster").strip() or "npa-cluster"
    # Default dry_run=True — real Terraform apply requires an explicit confirm token.
    dry_run = bool(body.get("dry_run", True))
    validate = bool(body.get("validate", True))
    skip_s3 = bool(body.get("skip_s3", True))
    desired = {{
        key: body[key]
        for key in (
            "gpu_nodes", "cpu_nodes", "gpu_platform", "gpu_preset",
            "gpu_driver_mode", "gpu_workload_profile", "managed_driver_preset",
            "gpu_health_stabilization_seconds", "gpu_health_timeout_minutes",
            "gpu_cuda_smoke",
            "gpu_cuda_smoke_image", "mig", "mig_strategy", "mig_config",
            "capacity_block_group",
        )
        if key in body
    }}
    try:
        desired = _normalize_agent_mk8s_desired(desired)
    except (TypeError, ValueError) as exc:
        return JSONResponse(
            status_code=400,
            content={{"ok": False, "status": "invalid", "error": str(exc)}},
        )
    logical = str(body.get("logical_allocation") or "").strip()
    preemptible = bool(body.get("preemptible", False))
    if logical:
        fallback_records = _load_state().get("gpu_allocation_fallback")
        fallback_records = fallback_records if isinstance(fallback_records, dict) else {{}}
        fallback = fallback_records.get(_gpu_fallback.logical_allocation_ref(logical))
        if isinstance(fallback, dict) and fallback.get("selected_pool") == _gpu_fallback.PREEMPTIBLE:
            preemptible = True
    if not dry_run:
        confirm_token = str(body.get("confirm_token") or "").strip()
        proposed_action = {{
            "action": "provision_infra",
            "project": project,
            "cluster_name": cluster_name,
            "desired": desired,
            "preemptible": preemptible,
            "skip_s3": skip_s3,
            "validate": validate,
        }}
        digest = action_digest(proposed_action)
        if not confirm_token:
            token = _issue_agent_confirm_token(
                proposed_action,
                digest,
            )
            return {{
                "ok": False,
                "needs_confirmation": True,
                "confirm_token": token,
                "proposed_action": {{**proposed_action, "dry_run": False}},
                "error": "Real infra provision requires confirm_token",
                "project": project,
                "cluster_name": cluster_name,
            }}
        session_token, confirm_digest, _pending = _consume_agent_confirm_token()
        if not session_token or confirm_token != session_token or (confirm_digest and confirm_digest != digest):
            raise HTTPException(status_code=403, detail="invalid or expired confirm_token for provision")
    result = _provision_agent_infra(
        project,
        cluster_name,
        dry_run=dry_run,
        validate=validate,
        skip_s3=skip_s3,
        desired=desired,
        preemptible=preemptible,
    )
    if not result.get("ok") and result.get("status") == "invalid":
        return JSONResponse(status_code=400, content=result)
    status = _agent_k8s_backends(project)
    return {{"ok": bool(result.get("ok")), "project": project, "cluster_name": cluster_name, "result": result, "infra": status, "dry_run": dry_run, "capacity_pool": "preemptible" if preemptible else "on-demand"}}


@app.post("/infra/soperator/validate")
def validate_soperator(payload: dict | None = None):
    body = payload if isinstance(payload, dict) else {{}}
    result = _soperator_validate_payload(body)
    if not result.get("ok"):
        return JSONResponse(status_code=400, content=result)
    return result


@app.post("/infra/soperator/deploy")
def deploy_soperator(payload: dict | None = None):
    body = payload if isinstance(payload, dict) else {{}}
    result = _soperator_deploy_from_payload(body)
    if not result.get("ok") and result.get("status") in {{"invalid", "blocked"}}:
        return JSONResponse(status_code=409 if result.get("status") == "blocked" else 400, content=result)
    if not result.get("ok"):
        return JSONResponse(status_code=502, content=result)
    return result


@app.get("/infra/soperator/status/{{name}}")
def soperator_status(name: str):
    return _soperator_status_payload(name)


@app.post("/workflows/draft")
@app.put("/workflows/npa/draft")
def save_workflow_draft(payload: dict):
    body = payload if isinstance(payload, dict) else {{}}
    yaml_text = str(body.get("yaml") or "").strip()
    if not yaml_text:
        raise HTTPException(status_code=400, detail="yaml is required")
    validation = validate_workflow_yaml_text(yaml_text, tool_refs=frozenset(TOOL_REFS))
    plan = (
        plan_workflow_yaml_text(yaml_text, run_id="draft-save", tool_refs=frozenset(TOOL_REFS))
        if validation.get("ok")
        else {{"ok": False, "error": str(validation.get("error") or "validation failed")}}
    )
    runnable = bool(validation.get("ok") and plan.get("ok"))
    state = _load_state()
    draft = _save_workflow_draft(state, yaml_text, validation, plan=plan, runnable=runnable)
    # ok tracks YAML validation; runnable requires validation+plan.
    return {{
        "ok": bool(validation.get("ok")),
        "draft": draft,
        "validation": validation,
        "plan": plan,
        "runnable": runnable,
    }}

@app.post("/workflows/validate")
@app.post("/workflows/npa/validate")
def validate_workflow(payload: dict):
    body = payload if isinstance(payload, dict) else {{}}
    yaml_text = _resolve_workflow_yaml(body)
    if not yaml_text:
        raise HTTPException(status_code=400, detail="yaml is required")
    validation = validate_workflow_yaml_text(yaml_text, tool_refs=frozenset(TOOL_REFS))
    plan = (
        plan_workflow_yaml_text(yaml_text, run_id="validate-check", tool_refs=frozenset(TOOL_REFS))
        if validation.get("ok")
        else {{"ok": False, "error": str(validation.get("error") or "validation failed")}}
    )
    runnable = bool(validation.get("ok") and plan.get("ok"))
    state = _load_state()
    _save_workflow_draft(state, yaml_text, validation, plan=plan, runnable=runnable)
    # ok tracks YAML validation; runnable requires validation+plan.
    return {{
        "ok": bool(validation.get("ok")),
        "validation": validation,
        "plan": plan,
        "runnable": runnable,
    }}

@app.post("/workflows/plan")
@app.post("/workflows/npa/plan")
def plan_workflow(payload: dict):
    body = payload if isinstance(payload, dict) else {{}}
    yaml_text = _resolve_workflow_yaml(body)
    if not yaml_text:
        raise HTTPException(status_code=400, detail="yaml is required")
    run_id = str(body.get("run_id") or "").strip()
    assume_decision = str(body.get("assume_decision") or "").strip()
    plan = plan_workflow_yaml_text(
        yaml_text,
        run_id=run_id,
        assume_decision=assume_decision,
        tool_refs=frozenset(TOOL_REFS),
    )
    if not plan.get("ok"):
        raise HTTPException(status_code=400, detail=str(plan.get("error") or "plan failed"))
    return {{"ok": True, "plan": plan, "yaml": yaml_text}}

@app.post("/workflows/submit")
@app.post("/workflows/npa/submit")
def submit_npa_workflow(payload: dict):
    body = payload if isinstance(payload, dict) else {{}}
    yaml_text = _resolve_workflow_yaml(body)
    if not yaml_text:
        raise HTTPException(status_code=400, detail="yaml is required")
    validation = validate_workflow_yaml_text(yaml_text, tool_refs=frozenset(TOOL_REFS))
    if not validation.get("ok"):
        raise HTTPException(status_code=400, detail=str(validation.get("error") or "validation failed"))
    run_id = str(body.get("run_id") or f"agent-wf-{{secrets.token_hex(6)}}")
    assume_decision = str(body.get("assume_decision") or "").strip()
    plan = plan_workflow_yaml_text(
        yaml_text,
        run_id=run_id,
        assume_decision=assume_decision,
        tool_refs=frozenset(TOOL_REFS),
    )
    if not plan.get("ok"):
        raise HTTPException(status_code=400, detail=str(plan.get("error") or "plan failed"))
    project = _agent_project_alias(str(body.get("project") or ""))
    cluster_name = str(body.get("cluster_name") or "npa-cluster").strip() or "npa-cluster"
    # Submit is plan-only by default: never auto-provision real infra.
    allow_provision = bool(body.get("allow_provision", False))
    dry_run = bool(body.get("dry_run", False))
    validate_infra = bool(body.get("validate_infra", True))
    infra_before = _agent_k8s_backends(project)
    if not infra_before.get("has_infra") and not allow_provision:
        return _workflow_no_infra_response(validation=validation, plan=plan, run_id=run_id, infra=infra_before)
    provision = {{"ok": True, "status": "skipped", "actions": ["k8s:existing backend detected"]}}
    if allow_provision and (dry_run or not infra_before.get("has_infra")):
        # Real (non-dry-run) provision requires the confirm-token gate.
        if not dry_run and not infra_before.get("has_infra"):
            confirm_token = str(body.get("confirm_token") or "").strip()
            digest = "provision_infra:" + project + ":" + cluster_name
            if not confirm_token:
                token = _issue_agent_confirm_token(
                    {{"action": "provision_infra", "project": project, "cluster_name": cluster_name, "via": "workflows/submit"}},
                    digest,
                )
                blocked = _workflow_no_infra_response(validation=validation, plan=plan, run_id=run_id, infra=infra_before)
                blocked["needs_confirmation"] = True
                blocked["confirm_token"] = token
                blocked["proposed_action"] = {{
                    "action": "provision_infra",
                    "project": project,
                    "cluster_name": cluster_name,
                    "dry_run": False,
                }}
                return blocked
            session_token, confirm_digest, _pending = _consume_agent_confirm_token()
            if not session_token or confirm_token != session_token or (confirm_digest and confirm_digest != digest):
                raise HTTPException(status_code=403, detail="invalid or expired confirm_token for provision")
        provision = _provision_agent_infra(
            project,
            cluster_name,
            dry_run=dry_run,
            # provision-if-absent may validate a cached kubeconfig before its
            # own dry-run branch; validation launches real CUDA smoke pods.
            # Keep workflow dry-run strictly read-only.
            validate=False if dry_run else validate_infra,
            skip_s3=bool(body.get("skip_s3", True)),
        )
        if not provision.get("ok"):
            infra_error = dict(infra_before)
            infra_error["provision_error"] = provision.get("error") or provision
            blocked = _workflow_no_infra_response(validation=validation, plan=plan, run_id=run_id, infra=infra_error)
            blocked["provision"] = provision
            return blocked
    scheduler_plan = {{}}
    yaml_path = _write_workflow_temp_yaml(yaml_text)
    try:
        scheduler_plan = _run_agent_npa_json(
            [
                "workbench",
                "workflow",
                "run-spec",
                str(yaml_path),
                "--run-id",
                run_id,
                "--plan-only",
                "--scheduler-plan",
                "--json",
            ],
            timeout_s=180,
        )
    finally:
        try:
            yaml_path.unlink(missing_ok=True)
        except Exception:
            pass
    infra_after = _agent_k8s_backends(project)
    state = _load_state()
    _save_workflow_draft(state, yaml_text, validation, plan=plan, runnable=True)
    submit_record = {{
        "run_id": run_id,
        "submitted_at": _now_iso(),
        "name": str(validation.get("name") or ""),
        "validation": validation,
        "plan": plan,
        "scheduler_plan": scheduler_plan,
        "infra": infra_after,
        "provision": provision,
        "submit_mode": "agent-live-infra-plan" if not dry_run else "agent-live-infra-dry-run",
        "note": (
            "Agent validated the workflow, ensured Kubernetes infra with NPA when needed, "
            "and produced a scheduler plan. Workload execution uses the planned scheduler tasks."
        ),
    }}
    state["workflow_submit"] = submit_record
    state["latest_submit"] = {{
        "run_id": run_id,
        "submitted_at": submit_record["submitted_at"],
        "workflow_name": str(validation.get("name") or ""),
        "submit_mode": submit_record["submit_mode"],
        "cluster_name": cluster_name,
    }}
    _record_sim_viz_run(
        state,
        {{
            "run_id": run_id,
            "submitted_at": submit_record["submitted_at"],
            "stage": "submitted",
            "camera": str((state.get("sim_viz", {{}}) or {{}}).get("camera") or "workspace"),
            "rrd_uri": str((state.get("sim_viz", {{}}) or {{}}).get("rrd_uri") or ""),
            "rrd_updated_at": str((state.get("sim_viz", {{}}) or {{}}).get("rrd_updated_at") or ""),
            "submit_mode": submit_record["submit_mode"],
            "workflow_name": str(validation.get("name") or ""),
            "cluster_name": cluster_name,
        }},
    )
    _save_state(state)
    return {{"ok": True, **submit_record}}

@app.post("/workflows/sim2real/submit")
def submit_sim2real(payload: dict | None = None):
    body = payload if isinstance(payload, dict) else {{}}
    state = _load_state()
    selection = state.get("selection", {{}})
    if not isinstance(selection, dict):
        selection = dict(DEFAULT_SELECTION)
    run_id = str(body.get("run_id") or f"agent-run-{{secrets.token_hex(6)}}")
    env_block = {{
        "NPA_SIM2REAL_SCENE_SPEC_URI": selection.get("scene_spec_uri", ""),
        "NPA_SIM2REAL_ASSETS_URI": selection.get("assets_uri", ""),
        "NPA_SIM2REAL_CAMERAS_URI": selection.get("cameras_uri", ""),
        "NPA_SIM2REAL_ROBOT_SPEC_URI": selection.get("robot_spec_uri", ""),
        "NPA_SIM2REAL_ROBOT_PRESET": selection.get("robot_preset", "franka"),
        "NPA_SIM2REAL_SIM_BACKEND": selection.get("sim_backend", "isaac") or "isaac",
    }}
    state["latest_submit"] = {{
        "run_id": run_id,
        "submitted_at": _now_iso(),
        "selection": selection,
        "env": env_block,
        "submit_mode": "sim2real",
    }}
    submitted_at = str(state["latest_submit"]["submitted_at"])
    details = _default_sim2real_run_details(run_id, submitted_at=submitted_at, selection=selection)
    details["logs"].append(
        {{
            "timestamp": submitted_at,
            "level": "info",
            "message": "Selection: robot_preset={{}}, sim_backend={{}}".format(
                selection.get("robot_preset", "franka"),
                selection.get("sim_backend", "isaac"),
            ),
        }}
    )
    details["logs"].append(
        {{
            "timestamp": submitted_at,
            "level": "info",
            "message": "Launching local Sim2Real runner on the agent VM.",
        }}
    )
    details["result"] = "queued"
    runs_detail = state.get("sim2real_runs")
    if not isinstance(runs_detail, dict):
        runs_detail = {{}}
    runs_detail[run_id] = details
    state["sim2real_runs"] = runs_detail
    camera = str((state.get("sim_viz", {{}}) or {{}}).get("camera") or "workspace")
    sim_viz = _wire_sim2real_run_preview(state, run_id=run_id, camera=camera)
    script = Path("/opt/npa-agent/run-live-sim2real.sh")
    live_submit = None
    if script.is_file():
        try:
            proc = subprocess.run([str(script), run_id], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30, check=False)
        except subprocess.TimeoutExpired as exc:
            live_submit = {{"ok": False, "error": f"live sim2real submit timed out after 30s: {{exc}}"}}
            state["latest_submit"]["live_submit"] = live_submit
            details["result"] = "failed"
            details["logs"].append({{"timestamp": _now_iso(), "level": "error", "message": live_submit["error"]}})
            runs_detail[run_id] = details
            state["sim2real_runs"] = runs_detail
            _save_state(state)
            return JSONResponse(
                status_code=502,
                content={{
                    "ok": False,
                    "error": live_submit["error"],
                    "run_id": run_id,
                    "selection": selection,
                    "env": env_block,
                    "run": details,
                    "sim_viz": sim_viz,
                    "submit_mode": "live-k8s-timeout",
                    "live_submit": live_submit,
                }},
            )
        except OSError as exc:
            live_submit = {{"ok": False, "error": f"live sim2real submit failed to start: {{exc}}"}}
        else:
            if proc.returncode == 0:
                try:
                    live_submit = json.loads((proc.stdout or "{{}}").strip().splitlines()[-1])
                    state["latest_submit"]["submit_mode"] = "live-k8s"
                    state["latest_submit"]["live_submit"] = live_submit
                    _save_state(state)
                    return {{"ok": True, "run_id": run_id, "selection": selection, "env": env_block, "run": details, "sim_viz": sim_viz, "submit_mode": "live-k8s", "live_submit": live_submit}}
                except Exception:
                    live_submit = {{"ok": False, "error": proc.stdout[-500:]}}
            else:
                live_submit = {{"ok": False, "error": (proc.stderr or proc.stdout or f"exit {{proc.returncode}}").strip()}}
    _save_state(state)
    thread = threading.Thread(
        target=_run_sim2real_pipeline_background,
        args=(run_id, dict(selection)),
        daemon=True,
    )
    thread.start()
    response = {{"ok": True, "run_id": run_id, "selection": selection, "env": env_block, "run": details, "sim_viz": sim_viz, "submit_mode": "agent-local-sim2real"}}
    if live_submit is not None:
        response["live_submit"] = live_submit
    return response
PY
cat <<'PY' | sudo tee /opt/npa-agent/bootstrap_rrd.py >/dev/null
import math
from pathlib import Path

import rerun as rr

_FRANKA_HOME = (0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785)

def _franka_demo_joint_angles(frame_index, frame_count):
    phase = (float(frame_index) / max(1.0, float(frame_count - 1))) * math.tau
    return (
        _FRANKA_HOME[0] + 0.22 * math.sin(phase),
        _FRANKA_HOME[1] + 0.16 * math.sin(phase + 0.5),
        _FRANKA_HOME[2] + 0.18 * math.sin(phase + 1.2),
        _FRANKA_HOME[3] + 0.12 * math.sin(phase + 1.7),
        _FRANKA_HOME[4] + 0.24 * math.sin(phase + 2.1),
        _FRANKA_HOME[5] + 0.10 * math.sin(phase + 2.7),
        _FRANKA_HOME[6] + 0.20 * math.sin(phase + 3.4),
    )

def _set_rerun_time(seconds):
    if hasattr(rr, "set_time_seconds"):
        rr.set_time_seconds("log_time", seconds)
    else:
        rr.set_time("log_time", duration=seconds)

def _franka_joint_positions(joint_angles):
    dh = [
        (0.0, 0.0, 0.333),
        (0.0, -math.pi / 2.0, 0.0),
        (0.0, math.pi / 2.0, 0.316),
        (0.0825, math.pi / 2.0, 0.0),
        (-0.0825, -math.pi / 2.0, 0.384),
        (0.0, math.pi / 2.0, 0.0),
        (0.088, math.pi / 2.0, 0.0),
    ]

    def _matmul(a, b):
        return [
            [sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)
        ]

    transform = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    positions = [[0.0, 0.0, 0.0]]
    for index, (a, alpha, d) in enumerate(dh):
        theta = float(joint_angles[index])
        ct, st = math.cos(theta), math.sin(theta)
        ca, sa = math.cos(alpha), math.sin(alpha)
        step = [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ]
        transform = _matmul(transform, step)
        positions.append([transform[0][3], transform[1][3], transform[2][3]])
    ee = [transform[0][3], transform[1][3], transform[2][3] + 0.103]
    positions.append(ee)
    positions.append([ee[0], ee[1] + 0.04, ee[2]])
    positions.append([ee[0], ee[1] - 0.04, ee[2]])
    return positions

def _log_franka_robot_geometry(joint_angles=_FRANKA_HOME):
    positions = _franka_joint_positions(joint_angles)
    arm_points = positions[:8]
    segments = []
    for left, right in zip(arm_points, arm_points[1:]):
        dx = left[0] - right[0]
        dy = left[1] - right[1]
        dz = left[2] - right[2]
        if dx * dx + dy * dy + dz * dz < 1e-8:
            continue
        segments.append([left, right])
    link_color = [234, 88, 12]
    link_rgba = link_color + [255]
    rr.log(
        "robot/franka/base",
        rr.Boxes3D(
            centers=[[0.0, 0.0, 0.05]],
            half_sizes=[[0.085, 0.085, 0.05]],
            colors=[[100, 116, 139, 255]],
        ),
    )
    rr.log(
        "robot/franka/joints",
        rr.Points3D(
            arm_points,
            colors=[link_rgba] * len(arm_points),
            radii=[0.028] * len(arm_points),
        ),
    )
    if segments:
        rr.log(
            "robot/franka/links",
            rr.LineStrips3D(
                segments,
                colors=[link_color] * len(segments),
                radii=[0.018] * len(segments),
            ),
        )
    gripper_segments = [
        [positions[7], positions[8]],
        [positions[8], positions[9]],
        [positions[8], positions[10]],
    ]
    gripper_color = [59, 130, 246]
    rr.log(
        "robot/franka/gripper",
        rr.LineStrips3D(
            gripper_segments,
            colors=[gripper_color] * len(gripper_segments),
            radii=[0.012] * len(gripper_segments),
        ),
    )
    rr.log(
        "robot/franka",
        rr.TextDocument("Franka Panda — stock tabletop demo (bootstrap)"),
    )

target = Path("/opt/npa-agent/sim2real.rrd")
target.parent.mkdir(parents=True, exist_ok=True)
rr.init("npa-franka-tabletop-demo", spawn=False)
rr.log(
    "world/table",
    rr.Boxes3D(
        centers=[[0.5, 0.0, 0.0]],
        half_sizes=[[0.4, 0.3, 0.02]],
        colors=[[180, 180, 180, 255]],
    ),
)
frame_count = 90
for frame_index in range(frame_count):
    seconds = frame_index / 15.0
    _set_rerun_time(seconds)
    phase = frame_index / max(1.0, float(frame_count - 1))
    cube_y = 0.3 - 0.42 * phase
    rr.log(
        "world/cube",
        rr.Boxes3D(
            centers=[[0.5, cube_y, 0.04]],
            half_sizes=[[0.025, 0.025, 0.025]],
            colors=[[59, 130, 246, 255]],
        ),
    )
    _log_franka_robot_geometry(_franka_demo_joint_angles(frame_index, frame_count))
rr.log("cameras/workspace", rr.Pinhole(fov_y=60.0))
rr.log("cameras/wrist", rr.Pinhole(fov_y=90.0))
rr.save(str(target))
from pathlib import Path as _Path
import shutil as _shutil
_rec = _Path("/opt/npa-agent/recordings/sim2real.rrd")
_rec.parent.mkdir(parents=True, exist_ok=True)
_shutil.copy2(target, _rec)
PY
sudo mkdir -p /opt/npa-agent/recordings
if [ {preload_stock_demo_value} = 1 ]; then
  sudo cp -f /opt/npa-agent/sim2real.rrd /opt/npa-agent/recordings/sim2real.rrd || true
else
  sudo rm -f /opt/npa-agent/sim2real.rrd /opt/npa-agent/recordings/sim2real.rrd
fi
# Tiny ftyp sample so live media-type checks work even when S3 has no .mp4 runs.
sudo python3 - <<'PY'
from pathlib import Path
target = Path("/opt/npa-agent/recordings/sample-preview.mp4")
ftyp_data = b"isom" + bytes([0, 0, 0, 0]) + b"isomiso2mp41"
ftyp = (8 + len(ftyp_data)).to_bytes(4, "big") + b"ftyp" + ftyp_data
mdat = (8).to_bytes(4, "big") + b"mdat"
target.write_bytes(ftyp + mdat)
PY
cat <<'WELCOME' | sudo tee /opt/npa-agent/welcome.html >/dev/null
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>NPA Agent — welcome</title>
    <style>
      body {{ font-family: system-ui, sans-serif; max-width: 640px; margin: 48px auto; padding: 0 16px; line-height: 1.5; color: #1f2430; }}
      h1 {{ font-size: 1.4rem; }}
      h2 {{ font-size: 1.1rem; margin-bottom: 0.5rem; }}
      code, pre {{ background: #f0f2f5; padding: 2px 6px; border-radius: 4px; }}
      .ok {{ color: #18794e; }}
      .muted {{ color: #5f6573; font-size: 0.95rem; }}
      .sign-in-panel {{ margin: 24px 0; padding: 16px; border: 1px solid #e0e0e0; border-radius: 8px; background: #fafbfc; }}
      .sign-in {{ display: grid; gap: 10px; max-width: 360px; }}
      .sign-in label {{ font-weight: 600; font-size: 0.9rem; }}
      .sign-in input {{ padding: 8px 10px; border: 1px solid #c8ccd4; border-radius: 6px; font: inherit; }}
      .sign-in button {{ justify-self: start; padding: 8px 16px; border: 0; border-radius: 6px; background: #5e43f3; color: #fff; font: inherit; font-weight: 600; cursor: pointer; min-height: 44px; }}
      @media (max-width: 640px) {{
        .sign-in, .sign-in button {{ max-width: none; width: 100%; }}
      }}
      a {{ color: #5e43f3; }}
    </style>
  </head>
  <body>
    <h1>NPA Agent is running</h1>
    <p class="ok">This page is public (no login). The workbench UI at <code>/</code> is protected by HTTP Basic Auth.</p>
{strip_url_credentials_js}
{login_form_html}
{mobile_login_help_html}
    <ol>
      <li>Enter your password above and click <strong>Sign in</strong>, or open <a href="/">the workbench UI</a> if your browser shows the auth dialog.</li>
      <li>Username: <code>{auth_user}</code></li>
      <li>Password: from your operator&apos;s deploy output (<code>auth_password</code>) or <code>auth.env</code> on the machine that ran <code>npa agent deploy</code>.</li>
      <li>Customer URL: use <code>https://</code> on port <strong>443</strong> (no VPN or SSH tunnel). Your browser may warn about a self-signed certificate — choose to proceed.</li>
      <li>More help: <a href="/login-help.html">login help</a></li>
    </ol>
    <p>Health check (no auth): <a href="/healthz">/healthz</a></p>
    <p>UI version after login: check <code>&lt;meta name="npa-ui-version"&gt;</code> — expect <code>{AGENT_UI_VERSION}</code>.</p>
  </body>
</html>
WELCOME
cat <<'LOGINHELP' | sudo tee /opt/npa-agent/login-help.html >/dev/null
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Login required — NPA Agent</title>
    <style>
      body {{ font-family: system-ui, sans-serif; max-width: 640px; margin: 48px auto; padding: 0 16px; line-height: 1.5; color: #1f2430; }}
      h1 {{ font-size: 1.4rem; }}
      h2 {{ font-size: 1.1rem; margin-bottom: 0.5rem; }}
      code {{ background: #f0f2f5; padding: 2px 6px; border-radius: 4px; }}
      .muted {{ color: #5f6573; font-size: 0.95rem; }}
      .sign-in-panel {{ margin: 24px 0; padding: 16px; border: 1px solid #e0e0e0; border-radius: 8px; background: #fafbfc; }}
      .sign-in {{ display: grid; gap: 10px; max-width: 360px; }}
      .sign-in label {{ font-weight: 600; font-size: 0.9rem; }}
      .sign-in input {{ padding: 8px 10px; border: 1px solid #c8ccd4; border-radius: 6px; font: inherit; }}
      .sign-in button {{ justify-self: start; padding: 8px 16px; border: 0; border-radius: 6px; background: #5e43f3; color: #fff; font: inherit; font-weight: 600; cursor: pointer; min-height: 44px; }}
      @media (max-width: 640px) {{
        .sign-in, .sign-in button {{ max-width: none; width: 100%; }}
      }}
      a {{ color: #5e43f3; }}
    </style>
  </head>
  <body>
    <h1>HTTP Basic Auth required</h1>
    <p>The NPA Agent workbench did not receive valid credentials. Sign in below or use your browser&apos;s Basic-auth dialog for <code>/</code> and <code>/api/*</code>.</p>
{strip_url_credentials_js}
{login_form_html}
{mobile_login_help_html}
    <ul>
      <li>Username: <code>{auth_user}</code></li>
      <li>Password: from your operator&apos;s <code>auth.env</code> file (<code>AGENT_PASSWORD</code>).</li>
      <li>Try the public <a href="/welcome">welcome page</a> for step-by-step instructions.</li>
      <li>Health (no auth): <a href="/healthz">/healthz</a></li>
    </ul>
  </body>
</html>
LOGINHELP
cat <<'HTML' | sudo tee /opt/npa-agent/ui.html >/dev/null
{_AGENT_UI_HTML_EMBED}
HTML
sudo python3 -m venv /opt/npa-agent/venv
sudo /opt/npa-agent/venv/bin/pip install --upgrade pip
sudo /opt/npa-agent/venv/bin/pip install fastapi uvicorn httpx pyyaml boto3 websockets "rerun-sdk>=0.32"
sudo /opt/npa-agent/venv/bin/pip install -e "{AGENT_SOURCE_ROOT}/npa[server,foxglove]"
if [ {preload_stock_demo_value} = 1 ]; then
  sudo /opt/npa-agent/venv/bin/python /opt/npa-agent/bootstrap_rrd.py
else
  sudo rm -f /opt/npa-agent/sim2real.rrd /opt/npa-agent/recordings/sim2real.rrd
fi
sudo systemctl restart npa-rerun || true
cat <<'UNIT' | sudo tee /etc/systemd/system/npa-agent-backend.service >/dev/null
[Unit]
Description=NPA agent backend
After=network.target
[Service]
Type=simple
EnvironmentFile=-/opt/npa-agent/llm.env
EnvironmentFile=-/opt/npa-agent/nebius.env
EnvironmentFile=-/opt/npa-agent/s3.env
EnvironmentFile=-/opt/npa-agent/public.env
EnvironmentFile=-/opt/npa-agent/foxglove.env
ExecStart=/opt/npa-agent/venv/bin/uvicorn backend:app --host 127.0.0.1 --port {backend_port} --log-level warning --no-access-log --ws websockets --ws-max-size 4194304 --ws-max-queue 4 --ws-ping-interval 10 --ws-ping-timeout 10 --ws-per-message-deflate false
WorkingDirectory=/opt/npa-agent
Restart=always
[Install]
WantedBy=multi-user.target
UNIT
cat <<'UNIT' | sudo tee /etc/systemd/system/npa-rerun.service >/dev/null
[Unit]
Description=NPA rerun service
After=network.target
[Service]
Type=simple
ExecStart=/opt/npa-agent/venv/bin/rerun {rerun_recording_arg}--serve-web --web-viewer --bind 0.0.0.0 --web-viewer-port {rerun_port} --port 9876
WorkingDirectory=/opt/npa-agent
Restart=always
StartLimitIntervalSec=0
[Install]
WantedBy=multi-user.target
UNIT
# Lichtblick (Foxglove-compatible MCAP viewer) sidecar — best-effort: the agent UI
# embeds it at /lichtblick/ and co-serves the run MCAP at /lichtblick/recordings/.
# Fresh driverless agent images do not include Docker, so install the Ubuntu
# package before acquiring the sidecar image. If package or registry access is
# unavailable, the UI retains its explicit viewer-unavailable state.
if ! command -v docker >/dev/null 2>&1; then
  if sudo apt-get update -qq && sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y docker.io >/dev/null; then
    sudo systemctl enable --now docker
  else
    echo "docker install failed; Lichtblick self-hosted fallback will be unavailable"
  fi
fi
# The sidecar serves only the viewer bundle: the MCAP itself is served by nginx from
# the recordings alias below (so it needs no mount into the container), and the file
# is pre-created so that location returns an empty 200 rather than 404 before a run
# is loaded.
sudo mkdir -p /opt/npa-agent/recordings
sudo touch /opt/npa-agent/recordings/sim2real.mcap
cat <<'UNIT' | sudo tee /etc/systemd/system/npa-lichtblick.service >/dev/null
[Unit]
Description=NPA Lichtblick MCAP viewer sidecar
After=network.target docker.service
[Service]
Type=simple
ExecStartPre=-/usr/bin/docker rm -f npa-lichtblick
ExecStart=/usr/bin/docker run --rm --name npa-lichtblick -p 127.0.0.1:{lichtblick_port}:8080 {lichtblick_image}
ExecStop=-/usr/bin/docker rm -f npa-lichtblick
Restart=always
RestartSec=10
StartLimitIntervalSec=0
[Install]
WantedBy=multi-user.target
UNIT
sudo htpasswd -bc /etc/nginx/.npa-agent-htpasswd {shlex.quote(auth_user)} {shlex.quote(auth_password)}
{https_ssl_setup}
cat <<'NGINXLOG' | sudo tee /etc/nginx/conf.d/npa-agent-safe-log.conf >/dev/null
# Deliberately use $uri, never $request or $request_uri: those include the
# browser-controlled query string used by WebRTC signaling and artifact APIs.
log_format npa_agent_safe '$remote_addr [$time_local] "$request_method $uri $server_protocol" $status $body_bytes_sent';
NGINXLOG
cat <<'NGINX' | sudo tee /etc/nginx/sites-available/npa-agent >/dev/null
server {{
  listen {agent_port};
  server_name _;
{nginx_site_body}
}}
{https_server_block}
NGINX
sudo ln -sf /etc/nginx/sites-available/npa-agent /etc/nginx/sites-enabled/npa-agent
sudo rm -f /etc/nginx/sites-enabled/default
sudo systemctl daemon-reload
sudo systemctl reset-failed npa-agent-backend npa-rerun nginx || true
sudo systemctl enable --now npa-agent-backend npa-rerun nginx
sudo systemctl restart npa-rerun nginx
sudo systemctl restart npa-agent-backend
# Region-agnostic Lichtblick image acquisition: anonymously pull the public
# release (or an explicitly configured customer override) and retag it to the
# sidecar image. Best-effort — never blocks the deploy.
for lb_cand in {lichtblick_pull_candidates}; do
  lb_host="${{lb_cand%%/*}}"
  if sudo docker pull "$lb_cand" >/dev/null 2>&1; then
    sudo docker tag "$lb_cand" {lichtblick_image} >/dev/null 2>&1 || true
    echo "npa-lichtblick image acquired from $lb_host"
    break
  fi
done
# Best-effort Lichtblick sidecar (never blocks deploy if docker/image are absent).
sudo systemctl enable --now npa-lichtblick 2>/dev/null || echo "npa-lichtblick sidecar not started (docker/image unavailable; /lichtblick/ embed degrades gracefully)"
"""
    setup_script = (
        setup_script.replace(_AGENT_CHAT_EMBED, agent_chat_source)
        .replace(_AGENT_RECORDINGS_EMBED, agent_recordings_source)
        .replace(_AGENT_BACKEND_SHIP, agent_backend_ship_script)
        .replace(_AGENT_WORKFLOW_EMBED, agent_workflow_source)
        .replace(_AGENT_ARTIFACTS_EMBED, agent_artifacts_source)
        .replace(_AGENT_ACCESS_EMBED, agent_access_source)
        .replace(_AGENT_ACCESS_RUNTIME_EMBED, agent_access_runtime_source)
        .replace(_AGENT_ARTIFACT_CONTENT_EMBED, agent_artifact_content_source)
        .replace(_AGENT_ROUTING_EMBED, agent_routing_source)
        .replace(_AGENT_VISUAL_FEEDBACK_EMBED, agent_visual_feedback_source)
        .replace(_AGENT_RRD_PROXY_EMBED, agent_rrd_proxy_source)
        .replace(_AGENT_STATE_EMBED, agent_state_source)
        .replace(_AGENT_S3_GUARD_EMBED, agent_s3_guard_source)
        .replace(_AGENT_STAGES_EMBED, agent_stages_source)
        .replace(_AGENT_STAGE_RUNTIME_EMBED, agent_stage_runtime_source)
        .replace(_AGENT_VIEWER_RUNTIME_EMBED, agent_viewer_runtime_source)
        .replace(_AGENT_PROVENANCE_EMBED, agent_provenance_source)
        .replace(_AGENT_UI_HTML_EMBED, rendered_agent_ui_html())
    )
    # Use a unique remote path so concurrent bootstrap runs cannot clobber each other.
    remote_setup_script = f"/tmp/npa-agent-bootstrap-{secrets.token_hex(6)}.sh"
    try:
        _stage_agent_npa_source(ssh)
        ssh.upload_private_text(setup_script, remote_setup_script)
        ssh.run_or_raise(
            f"chmod 700 {shlex.quote(remote_setup_script)} && {shlex.quote(remote_setup_script)}",
            label="run agent bootstrap",
        )
    finally:
        ssh.run(f"rm -f {shlex.quote(remote_setup_script)}")
    _write_agent_llm_env(
        ssh,
        tf_api_key=tf_api_key,
        llm_provider=DEFAULT_LLM_PROVIDER,
        llm_providers=(DEFAULT_LLM_PROVIDER,),
        llm_model=llm_model,
        llm_models=llm_models,
    )
    _write_agent_s3_env(
        ssh,
        bucket=s3_bucket,
        prefix=s3_prefix,
        endpoint=s3_endpoint,
        access_key=s3_access_key,
        secret_key=s3_secret_key,
        region=s3_region,
    )
    _write_agent_operator_profile(
        ssh,
        ssh_user=ssh_user,
        project_alias=project_alias,
        project_id=project_id,
        tenant_id=tenant_id,
        region=region,
        tf_api_key=tf_api_key,
        nebius_ai_key=nebius_ai_key,
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix,
        s3_endpoint=s3_endpoint,
        s3_access_key=s3_access_key,
        s3_secret_key=s3_secret_key,
        service_account_id=service_account_id,
    )
    _write_agent_nebius_env(
        ssh,
        project_alias=project_alias,
        agent_name=agent_name,
        project_id=nebius_project_id or project_id,
        tenant_id=nebius_tenant_id or tenant_id,
        region=s3_region,
        service_account_id=service_account_id,
        bucket=s3_bucket,
        endpoint=s3_endpoint,
        access_key=s3_access_key,
        secret_key=s3_secret_key,
    )
    if (
        tf_api_key.strip()
        or (s3_bucket.strip() and s3_access_key.strip() and s3_secret_key.strip())
        or (
            (nebius_project_id or project_id).strip()
            and s3_access_key.strip()
            and s3_secret_key.strip()
        )
    ):
        ssh.run_or_raise(
            "sudo systemctl reset-failed npa-agent-backend || true; "
            "sudo systemctl restart npa-agent-backend"
        )
    verify_remote_deployment(ssh, deployment, backend_port=backend_port)
    _record_remote_setup_ready(
        ssh,
        project_alias=project_alias,
        agent_name=agent_name,
        project_id=project_id,
        endpoint=host,
    )


def _record_remote_setup_ready(
    ssh: SSHClient,
    *,
    project_alias: str,
    agent_name: str,
    project_id: str,
    endpoint: str,
) -> None:
    """Atomically record non-secret evidence required for restart adoption."""

    credential_paths = (
        "/opt/npa-agent/llm.env",
        "/opt/npa-agent/s3.env",
        "/opt/npa-agent/nebius.env",
    )
    service_paths = (
        "/opt/npa-agent/deployment.json",
        "/etc/systemd/system/npa-agent-backend.service",
    )
    result = ssh.run_or_raise(
        "sudo sha256sum "
        + " ".join(shlex.quote(path) for path in (*service_paths, *credential_paths)),
        label="fingerprint staged agent files",
    )
    if result is None:  # Lightweight rendered-backend test doubles.
        service_fingerprint = credential_fingerprint = "fixture-fingerprint"
    else:
        _code, stdout, _stderr = result
        rows = [line.split()[0] for line in stdout.splitlines() if line.split()]
        if len(rows) != len(service_paths) + len(credential_paths):
            raise SSHError("staged agent fingerprint inventory was incomplete")
        service_fingerprint = hashlib.sha256(
            "\n".join(rows[: len(service_paths)]).encode("ascii")
        ).hexdigest()
        credential_fingerprint = hashlib.sha256(
            "\n".join(rows[len(service_paths) :]).encode("ascii")
        ).hexdigest()
    state = {
        "phase": "remote_health_ready",
        "project_alias": project_alias,
        "agent_name": agent_name,
        "project_id": project_id,
        "endpoint": endpoint,
        "service_fingerprint": service_fingerprint,
        "credential_fingerprint": credential_fingerprint,
        "credential_fingerprint_files": ["llm.env", "s3.env", "nebius.env"],
    }
    _stage_private_text(
        ssh,
        content=json.dumps(state, sort_keys=True) + "\n",
        target="/opt/npa-agent/setup-state.json",
    )


def _health(
    url: str,
    *,
    user: str,
    password: str,
    timeout: float = 5.0,
    verify: bool = True,
) -> tuple[bool, int]:
    try:
        response = httpx.get(url, auth=(user, password), timeout=timeout, verify=verify)
    except httpx.HTTPError:
        return False, 0
    return response.status_code == 200, response.status_code


def _basic_auth_protects_endpoint(
    url: str,
    *,
    timeout: float = 5.0,
    verify: bool = True,
) -> tuple[bool, int]:
    """Prove that an unauthenticated request cannot reach the agent UI."""
    try:
        response = httpx.get(
            url,
            timeout=timeout,
            verify=verify,
            follow_redirects=False,
        )
    except httpx.HTTPError:
        return False, 0
    return response.status_code == 401, response.status_code


_artifact_only_http_probe = agent_resources.artifact_only_http_probe


ARTIFACT_ONLY_HTTP_TIMEOUT_SECONDS = 60.0


def _verify_artifact_only_live(
    *,
    record: dict[str, Any],
    auth_user: str,
    auth_password: str,
    tls_verify: bool,
    project: str,
    name: str,
) -> None:
    """Run the no-stock live gate without writing chat/workflow/demo state."""
    agent_base = str(record.get("agent_url", "")).rstrip("/")
    try:
        with httpx.Client(
            base_url=agent_base,
            auth=(auth_user, auth_password),
            verify=tls_verify,
            timeout=ARTIFACT_ONLY_HTTP_TIMEOUT_SECONDS,
        ) as client:
            result = _artifact_only_http_probe(client)
    except (httpx.HTTPError, DeploymentIdentityError) as exc:
        _fail(f"artifact-only read-only probe failed: {exc}")

    from npa.agent_rerun_bundle_check import (
        check_rerun_bundle_load_budget,
        format_bundle_budget_report,
    )

    bundle_result = check_rerun_bundle_load_budget(
        agent_base,
        auth=(auth_user, auth_password),
        verify=tls_verify,
    )
    typer.echo(format_bundle_budget_report(bundle_result))
    if not bundle_result.ok:
        _fail("rerun bundle load budget failed: " + "; ".join(bundle_result.errors[:4]))

    test_env = {
        **dict(os.environ),
        "NPA_INTEGRATION_E2E": "1",
        "NPA_AGENT_LIVE": "1",
        "NPA_AGENT_PROJECT": project,
        "NPA_AGENT_NAME": name,
        "NPA_AGENT_VERIFY_READ_ONLY": "1",
    }
    suites = (
        (
            "smoke",
            [
                "npa/tests/smoke/test_agent_smoke.py",
                "npa/tests/smoke/test_agent_chat_smoke.py",
            ],
        ),
        (
            "unit",
            ["npa/tests/cli/test_agent.py", "npa/tests/cli/test_agent_workflow.py"],
        ),
        (
            "read-only live e2e",
            [
                "npa/tests/e2e/test_agent_live.py",
                "-k",
                (
                    "agent_ui_html_smoke or agent_health_and_session or "
                    "agent_sim_assets_and_catalog or agent_tools_catalog or "
                    "agent_workbench_actions or agent_rerun_iframe_reachable"
                ),
            ],
        ),
    )
    for label, suite_args in suites:
        proc = subprocess.run(
            ["npa/.venv/bin/python", "-m", "pytest", *suite_args, "-q"],
            check=False,
            env=test_env,
        )
        if proc.returncode != 0:
            _fail(f"artifact-only {label} verification failed")

    # The test processes and bundle probe are read-only, too. Verify the exact
    # persisted state digest again after every gate has completed.
    try:
        with httpx.Client(
            base_url=agent_base,
            auth=(auth_user, auth_password),
            verify=tls_verify,
            timeout=ARTIFACT_ONLY_HTTP_TIMEOUT_SECONDS,
        ) as client:
            final = client.get("/api/health")
            final.raise_for_status()
            final_digest = str(final.json().get("state_sha256") or "")
    except (httpx.HTTPError, ValueError) as exc:
        _fail(f"artifact-only final state probe failed: {exc}")
    if final_digest != result["state_sha256"]:
        _fail("artifact-only verification changed durable state")
    typer.echo(
        "artifact-only read-only gate: "
        f"runs={result['run_count']} tool_refs={result['tool_ref_count']} "
        f"state_sha256={result['state_sha256']}"
    )


@app.command("preflight")
def preflight_cmd(
    project: str = typer.Option(
        "",
        "--project",
        help="Configured project alias whose writable S3 must be verified.",
    ),
    name: str = typer.Option(
        DEFAULT_AGENT_NAME,
        "--name",
        help="Agent deployment name this preflight is gating (capacity depends on it).",
    ),
    ssh_public_key_path: str = typer.Option(
        "~/.ssh/id_ed25519.pub",
        "--ssh-public-key-path",
        help="SSH public key path Terraform will read (its private key bootstraps the VM).",
    ),
    skip_nebius: bool = typer.Option(
        False, "--skip-nebius", help="Skip the live Nebius authentication check."
    ),
    agent_only: bool = typer.Option(False, "--agent-only"),
    output_json: bool = typer.Option(False, "--json", help="Print the report as JSON."),
) -> None:
    """Check Route C prerequisites before `npa agent deploy` / `fresh-setup`.

    Validates terraform, the SSH key pair, Nebius authentication, and the Token
    Factory key with no cloud side effects. The Terraform check includes the
    current-platform provider lock; the storage check executes the exact
    health-verified credential selection deploy will reuse, without listing or
    rotating IAM access keys. Exits non-zero on any FAIL.

    Capacity is resolved for ``--name`` so the gate matches the deploy it
    precedes: an existing agent of that name already holds its public IP, while a
    new name needs fresh headroom.
    """
    results = list(_agent_hard_prereq_results(ssh_public_key_path))
    if not skip_nebius:
        results.append(_agent_nebius_auth_result())
        try:
            saved = resolve_environment(project or None)
        except Exception:  # noqa: BLE001 - the shared result reports missing identity
            saved = None
        project_alias = _resolve_project_alias(project)
        results.append(
            _agent_whole_path_capacity_result(
                str(getattr(saved, "project_id", "") or ""),
                str(getattr(saved, "tenant_id", "") or ""),
                str(getattr(saved, "region", "") or ""),
                agent_exists=bool(
                    _agent_record(
                        project_alias, str(name or DEFAULT_AGENT_NAME).strip()
                    ).get("public_ip")
                ),
                include_paidf=not agent_only,
            )
        )
    results.append(_agent_ssh_egress_result())
    results.append(_agent_storage_result(project))
    tf_api_key, _default_llm_model = _resolve_deploy_llm_credentials()
    results.append(_agent_token_factory_result(tf_api_key))
    has_fail = _render_agent_checks(results, output_json=output_json)
    if has_fail:
        raise typer.Exit(code=1)


def _transactional_agent_command(command: str):
    """Run nested agent entrypoints in one durable operation context."""

    def decorate(function):
        signature = inspect.signature(function)

        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            parent = current_operation()
            if parent is not None:
                return function(*args, **kwargs)
            bound = signature.bind_partial(*args, **kwargs)
            bound.apply_defaults()
            project = _resolve_project_alias(str(bound.arguments.get("project") or ""))
            name = str(bound.arguments.get("name") or DEFAULT_AGENT_NAME).strip()
            project_id = str(bound.arguments.get("project_id") or "").strip()
            tenant_id = str(bound.arguments.get("tenant_id") or "").strip()
            region = str(bound.arguments.get("region") or "").strip()
            try:
                saved = resolve_environment(project)
            except ConfigError:
                saved = None
            if saved is not None:
                project_id = project_id or saved.project_id
                tenant_id = tenant_id or saved.tenant_id
                region = saved.region or region
            resume_argv = shlex.split(command) + ["--project", project, "--name", name]
            for argument, flag, effective in (
                ("project_id", "--project-id", project_id),
                ("tenant_id", "--tenant-id", tenant_id),
                ("region", "--region", region),
                ("ssh_user", "--ssh-user", bound.arguments.get("ssh_user")),
                (
                    "ssh_public_key_path",
                    "--ssh-public-key-path",
                    bound.arguments.get("ssh_public_key_path"),
                ),
                ("agent_port", "--agent-port", bound.arguments.get("agent_port")),
                ("backend_port", "--backend-port", bound.arguments.get("backend_port")),
                ("rerun_port", "--rerun-port", bound.arguments.get("rerun_port")),
                ("llm_model", "--llm-model", bound.arguments.get("llm_model")),
                (
                    "foxglove_embed_src",
                    "--foxglove-embed-src",
                    bound.arguments.get("foxglove_embed_src"),
                ),
                (
                    "foxglove_viewer_backend",
                    "--foxglove-viewer-backend",
                    bound.arguments.get("foxglove_viewer_backend"),
                ),
                (
                    "foxglove_org_slug",
                    "--foxglove-org-slug",
                    bound.arguments.get("foxglove_org_slug"),
                ),
                (
                    "foxglove_live_url",
                    "--foxglove-live-url",
                    bound.arguments.get("foxglove_live_url"),
                ),
            ):
                if argument in bound.arguments and effective not in (None, ""):
                    resume_argv.extend([flag, str(effective)])
            for model in _coerce_cli_list(bound.arguments.get("llm_models")):
                resume_argv.extend(["--llm-models", str(model)])
            from npa.provisioning_journal import operation_contains_secret

            for tf_value in _coerce_cli_list(bound.arguments.get("tf_var")):
                if operation_contains_secret([str(tf_value)]):
                    raise ValueError(
                        "Secret-shaped --tf-var values cannot be persisted in a crash-safe "
                        "recovery plan; use the supported credential store instead."
                    )
                resume_argv.extend(["--tf-var", str(tf_value)])
            if bool(bound.arguments.get("no_public_https")):
                resume_argv.append("--no-public-https")
            if bool(bound.arguments.get("agent_only")):
                resume_argv.append("--agent-only")
            if "wait_ssh" in bound.arguments:
                resume_argv.append(
                    "--wait-ssh"
                    if bool(bound.arguments.get("wait_ssh"))
                    else "--no-wait-ssh"
                )
            destroy_argv = [
                "npa",
                "agent",
                "destroy",
                "--project",
                project,
                "--name",
                name,
                "--yes",
            ]
            operation = ProvisioningOperation.prepare(
                command=command,
                project_alias=project,
                project_id=project_id,
                tenant_id=tenant_id,
                region=region,
                resource_type="agent",
                requested_name=name,
                ownership_source="agent-cli",
                resume_command="",
                destroy_command="",
                resume_argv=resume_argv,
                destroy_argv=destroy_argv,
            )
            exact_destroy_argv = [
                "npa",
                "agent",
                "destroy",
                "--operation-id",
                operation.operation_id,
                "--name",
                name,
                "--yes",
            ]
            if project_id:
                exact_destroy_argv.extend(["--project-id", project_id])
            if tenant_id:
                exact_destroy_argv.extend(["--tenant-id", tenant_id])
            if region:
                exact_destroy_argv.extend(["--region", region])
            operation.set_recovery_commands(
                resume_argv=resume_argv, destroy_argv=exact_destroy_argv
            )
            with operation_context(operation):
                try:
                    result = function(*args, **kwargs)
                except BaseException as exc:
                    tf_dir = provisioner.working_dir_path(project, name)
                    for candidate in (
                        tf_dir / "errored.tfstate",
                        tf_dir / "terraform.tfstate",
                    ):
                        if candidate.is_file():
                            operation.preserve_state_file(
                                candidate, name=candidate.stem
                            )
                    phase = str(operation.read().get("phase") or "")
                    if phase == "prepared":
                        operation.transition(
                            "rolled-back",
                            error=str(exc),
                            details={
                                "error_type": type(exc).__name__,
                                "mutation_started": False,
                            },
                        )
                    elif phase not in {
                        "recovery-required",
                        "rollback-incomplete",
                        "rolled-back",
                    }:
                        operation.transition(
                            "recovery-required",
                            error=str(exc),
                            details={"error_type": type(exc).__name__},
                        )
                    typer.echo(emit_recovery_summary(operation), err=True)
                    raise
                phase = str(operation.read().get("phase") or "")
                if phase == "resource-created":
                    operation.transition("state-durable", details={"verified": True})
                operation.commit()
                return result

        return wrapped

    return decorate


@app.command("deploy")
@resolve_typer_defaults
@_transactional_agent_command("npa agent deploy")
def deploy_cmd(
    project: str = typer.Option(
        "",
        "--project",
        help="NPA project alias to store config under (default: configured default_project).",
    ),
    name: str = typer.Option(
        DEFAULT_AGENT_NAME, "--name", help="Agent deployment name."
    ),
    project_id: str = typer.Option("", "--project-id", help="Nebius project ID."),
    tenant_id: str = typer.Option("", "--tenant-id", help="Nebius tenant ID."),
    region: str = typer.Option("eu-north1", "--region", help="Nebius region."),
    ssh_user: str = typer.Option("ubuntu", "--ssh-user", help="SSH username."),
    ssh_public_key_path: str = typer.Option(
        "~/.ssh/id_ed25519.pub",
        "--ssh-public-key-path",
        help="SSH public key path for Terraform.",
    ),
    tf_var: list[str] = typer.Option(
        [], "--tf-var", help="Additional Terraform var key=value."
    ),
    agent_only: bool = typer.Option(
        False, "--agent-only", help="Provision agent only."
    ),
    agent_port: int = typer.Option(
        DEFAULT_AGENT_PORT, "--agent-port", help="Public agent UI port."
    ),
    backend_port: int = typer.Option(
        DEFAULT_BACKEND_PORT, "--backend-port", help="Internal agent backend port."
    ),
    rerun_port: int = typer.Option(
        DEFAULT_RERUN_PORT, "--rerun-port", help="Rerun service port."
    ),
    llm_model: str = typer.Option(
        DEFAULT_LLM_MODEL,
        "--llm-model",
        help="Default Token Factory model for agent chat.",
    ),
    llm_models: list[str] = typer.Option(
        [],
        "--llm-models",
        help="Additional Token Factory model IDs (repeat flag or comma-separate values).",
    ),
    foxglove_embed_src: str = agent_foxglove_config.embed_src_option(),
    foxglove_viewer_backend: str = agent_foxglove_config.viewer_backend_option(),
    foxglove_org_slug: str = agent_foxglove_config.org_slug_option(),
    foxglove_live_url: str = agent_foxglove_config.live_url_option(),
    no_public_https: bool = typer.Option(
        False,
        "--no-public-https",
        help="Disable HTTPS on port 443 (customer access uses http://IP:agent-port only).",
    ),
    wait_ssh: bool = typer.Option(
        True,
        "--wait-ssh/--no-wait-ssh",
        help=(
            "Wait for the new VM's SSH and cloud-init before finishing (default). "
            "--no-wait-ssh keeps the VM when this machine cannot reach a fresh "
            "public IP on tcp/22 (VPN / split tunnel): it still bootstraps, but "
            "npa cannot verify it, so check https://<ip>/healthz yourself."
        ),
    ),
) -> None:
    """Provision VM + bootstrap the public NPA agent stack."""
    # Same resolution as status/destroy: an omitted --project must mean the
    # operator's configured default_project. Deploying under the static
    # us-central1 alias while status/destroy looked at the configured default left
    # the agent unreachable by name — and destroy then reported success on an
    # empty state while the real VM and its public IP kept running.
    project = _resolve_project_alias(project)
    # deploy_cmd is also called programmatically (fresh-setup, `agent setup`
    # wrappers). Coerce list-valued options so an unresolved Typer default
    # (OptionInfo) can never crash `for item in tf_var` / `list(llm_models)`.
    tf_var = _coerce_cli_list(tf_var)
    llm_models = _coerce_cli_list(llm_models)
    foxglove_settings = _resolve_foxglove_settings_or_fail(
        embed_src=foxglove_embed_src,
        viewer_backend=foxglove_viewer_backend,
        org_slug=foxglove_org_slug,
        live_url=foxglove_live_url,
    )
    # Expand ``~`` up front so the absolute path flows into Terraform vars and
    # outputs (e.g. ssh_key_path). Terraform reads the key with pathexpand, but
    # the raw var also lands in outputs consumed downstream, where an unexpanded
    # ``~`` breaks non-shell consumers.
    ssh_public_key_path = str(Path(ssh_public_key_path).expanduser())
    profile = os.environ.get("NPA_NEBIUS_PROFILE", "").strip()
    if profile and shutil.which("nebius"):
        subprocess.run(["nebius", "profile", "activate", profile], check=False)
    saved_env = resolve_environment(
        project,
        project_id=project_id or None,
        tenant_id=tenant_id or None,
        region=region or None,
    )
    env_project_id = project_id or (saved_env.project_id if saved_env else "")
    env_tenant_id = tenant_id or (saved_env.tenant_id if saved_env else "")
    env_region = region or (saved_env.region if saved_env else "")
    if not env_project_id or not env_tenant_id or not env_region:
        _fail("--project-id, --tenant-id, and --region are required")

    # Compute placement follows the project's own region, not the --region flag
    # (whose default can mismatch the project, e.g. alias us-central1 + default
    # eu-north1). Resolve the project's real region and prefer it so the quota
    # check, storage endpoint, and cloud-init all target where the VM lands.
    try:
        from npa.clients.nebius import get_project_region

        real_region = get_project_region(env_project_id)
    except Exception:  # noqa: BLE001
        real_region = ""
    if real_region and real_region != env_region:
        typer.echo(
            f"  Note: project {env_project_id} is in region {real_region!r}; using it "
            f"instead of {env_region!r} (compute placement follows the project).",
            err=True,
        )
        env_region = real_region

    # Capacity is the first provider-backed gate. A quota/RBAC failure must not
    # be preceded by the writable-storage probe (which briefly writes an S3
    # object) or by Terraform/bootstrap activity.
    _agent_exists = bool(_agent_record(project, name).get("public_ip"))
    try:
        _agent_check_whole_path_capacity(
            env_project_id,
            env_tenant_id,
            env_region,
            agent_exists=_agent_exists,
            include_paidf=not agent_only,
        )
    except Exception as exc:
        _fail(str(exc))

    operation = current_operation()
    if operation is not None:
        operation.update_identity(
            project_alias=project,
            project_id=env_project_id,
            tenant_id=env_tenant_id,
            region=env_region,
            allow_region_correction=True,
        )
        from npa.clients.config import CONFIG_PATH
        from npa.clients.credentials import (
            CREDENTIALS_PATH,
            preflight_private_yaml_store,
        )

        preflight_private_yaml_store(CONFIG_PATH)
        preflight_private_yaml_store(CREDENTIALS_PATH)

    # Fail before IAM/Terraform. Storage writes and removes one isolated probe;
    # missing binaries, SSH, or writable storage otherwise surface after cloud
    # changes. Surface the Token Factory warning before the VM exists too.
    # Resolve the deploy LLM creds once and thread them through to the VM
    # bootstrap below.
    tf_api_key, default_llm_model = _resolve_deploy_llm_credentials()
    prereq_results = _agent_hard_prereq_results(ssh_public_key_path)
    prereq_results.append(_agent_storage_result(project, env_region, name))
    tf_key_result = _agent_token_factory_result(tf_api_key)
    for result in prereq_results:
        if result.status == "FAIL":
            _fail(f"{result.summary} {result.remedy}".strip())
    # The deploy waits for the new VM's tcp/22 from this machine, so say up front
    # when this host cannot open outbound SSH at all — otherwise that shows up as a
    # five-minute wait and a rollback of a perfectly healthy VM.
    for warn_result in (tf_key_result, _agent_ssh_egress_result()):
        if warn_result.status == "WARN":
            typer.echo(f"  Warning: {warn_result.summary}", err=True)
            typer.echo(f"           {warn_result.remedy}", err=True)

    # Mark mutation only immediately before the first provider/storage action.
    if operation is not None:
        operation.transition("mutating")

    from npa.clients.nebius import (
        NebiusError,
        bootstrap_agent_environment,
        get_iam_token,
    )

    try:
        configured_storage = _resolve_deploy_storage_credentials(
            region=env_region,
            project_alias=project,
        )
        if operation is not None:
            from npa.clients.storage_validation import (
                terraform_backend_fingerprint,
                terraform_state_key,
            )

            backend_key = terraform_state_key(project, name)
            operation.update_identity(
                backend={
                    "bucket": str(configured_storage.get("s3_bucket", "")),
                    "endpoint": str(configured_storage.get("s3_endpoint", "")),
                    "region": env_region,
                    "state_key": backend_key,
                    "addressing_style": "path",
                    "credential_source": "project_resolver",
                    "config_fingerprint": terraform_backend_fingerprint(
                        bucket=str(configured_storage.get("s3_bucket", "")),
                        state_key=backend_key,
                        endpoint_url=str(configured_storage.get("s3_endpoint", "")),
                        access_key_id=str(configured_storage.get("nebius_api_key", "")),
                        secret_access_key=str(
                            configured_storage.get("nebius_secret_key", "")
                        ),
                        region=env_region,
                    ),
                }
            )
            operation.record_resource(
                resource_type="storage_bucket",
                requested_name=str(configured_storage.get("s3_bucket", "")),
                ownership="adopted",
                ownership_source="configured-project-storage-write-probe",
                project_id=env_project_id,
            )

        def _record_created_agent_resource(kind: str, metadata: dict[str, str]) -> None:
            if operation is None:
                return
            operation.record_resource(
                resource_type=f"agent_{kind}",
                requested_name=str(metadata.get("name") or metadata.get("id") or kind),
                provider_id=str(metadata.get("id") or ""),
                ownership="created_by_this_operation",
                ownership_source="agent-bootstrap-create-callback",
                project_id=env_project_id,
            )

        creds = bootstrap_agent_environment(
            env_project_id,
            env_tenant_id,
            env_region,
            on_status=lambda msg: typer.echo(f"  {msg}"),
            on_resource_created=_record_created_agent_resource,
            reuse_storage_credentials=configured_storage,
        )
        # Bootstrap already resolves the operator token and returns it with the
        # provider credentials.  Reuse that exact result so deploy does not make
        # a second, environment-dependent CLI/profile lookup after successful
        # bootstrap.  The fallback preserves compatibility with older/custom
        # bootstrap implementations that omitted ``iam_token``.
        iam_token = str(creds.get("iam_token") or "").strip() or get_iam_token()
    except (AgentStorageCredentialError, NebiusError) as exc:
        _fail(f"Nebius bootstrap failed: {exc}")

    public_https = not no_public_https
    extra_ingress_ports = _agent_extra_ingress_ports(
        agent_port=agent_port,
        rerun_port=rerun_port,
        public_https=public_https,
    )
    merged_vars: dict[str, str] = {
        "nebius_project_id": env_project_id,
        "nebius_region": env_region,
        "service_account_id": str(creds.get("service_account_id", "")),
        "iam_token": iam_token,
        "nebius_api_key": str(creds.get("nebius_api_key", "")),
        "nebius_secret_key": str(creds.get("nebius_secret_key", "")),
        "s3_bucket": str(creds.get("s3_bucket", "")),
        "s3_prefix": str(creds.get("s3_prefix", "")),
        "s3_endpoint": str(creds.get("s3_endpoint", "")),
        "instance_name": f"agent-{project}-{name}",
        "server_port": str(agent_port),
        "extra_ingress_ports": (
            "[" + ",".join(str(port) for port in extra_ingress_ports) + "]"
            if extra_ingress_ports
            else "[]"
        ),
        "workbench_type": "agent",
        "gpu_platform": "cpu-d3",
        "gpu_preset": "8vcpu-32gb",
        "image_family": DEFAULT_AGENT_IMAGE_FAMILY,
        "ssh_user": ssh_user,
        "ssh_public_key_path": ssh_public_key_path,
        "enable_preemptible": "false",
        "wait_for_ssh": "true" if wait_ssh else "false",
    }
    operation = current_operation()
    if operation is not None:
        operation.update_identity(
            project_alias=project,
            project_id=env_project_id,
            tenant_id=env_tenant_id,
            region=env_region,
            backend={
                "bucket": str(creds.get("s3_bucket", "")),
                "endpoint": str(creds.get("s3_endpoint", "")),
                "region": env_region,
            },
            allow_region_correction=True,
        )
        merged_vars["operation_id"] = operation.operation_id
    for item in tf_var:
        if "=" not in item:
            _fail(f"Invalid --tf-var value {item!r}; expected key=value")
        key, value = item.split("=", 1)
        merged_vars[key.strip()] = value.strip()
    ssh_source = str(merged_vars.get("ssh_cidr_block", "")).strip()
    application_source = str(merged_vars.get("application_cidr_block", "")).strip()
    if not ssh_source:
        _fail(
            "Agent deploy requires an explicit ssh_cidr_block so the verified "
            "post-create bootstrap can connect; pass it with --tf-var."
        )
    if not application_source:
        _fail(
            "Agent deploy requires an explicit application_cidr_block for its "
            "public HTTPS health boundary; pass it with --tf-var."
        )
    try:
        _ensure_terraform_state_bucket(
            project_id=env_project_id,
            bucket_name=str(merged_vars.get("s3_bucket", "")),
            endpoint=str(merged_vars.get("s3_endpoint", "")),
            access_key=str(merged_vars.get("nebius_api_key", "")),
            secret_key=str(merged_vars.get("nebius_secret_key", "")),
            region=env_region,
            project_alias=project,
            agent_name=name,
        )
    except NebiusError as exc:
        _fail(f"Unable to provision Terraform state bucket: {exc}")
    _persist_agent_project_config(
        project=project,
        project_id=env_project_id,
        tenant_id=env_tenant_id,
        region=env_region,
        merged_vars=merged_vars,
    )
    terraform_vars = dict(merged_vars)
    for credential_key in (
        "nebius_api_key",
        "nebius_secret_key",
        "s3_session_token",
    ):
        terraform_vars.pop(credential_key, None)

    tf_outputs: dict[str, Any] = {}
    typer.echo(
        "  Phase 1/4: applying agent VM infrastructure (Terraform streams its own progress)."
    )
    if operation is not None:
        operation.checkpoint(
            "vm_mutation_started",
            {
                "setup_phase": "vm_mutation_started",
                "project_id": env_project_id,
                "requested_name": name,
            },
        )
    try:
        with operation_heartbeats(
            operation,
            phase="terraform_apply",
            emit=lambda payload: typer.echo(
                "progress: " + json.dumps(payload, sort_keys=True), err=True
            ),
        ):
            tf_outputs = _apply_agent_terraform(
                project=project,
                name=name,
                merged_vars=terraform_vars,
                env_region=env_region,
                backend_credentials={
                    "access_key": str(creds.get("nebius_api_key", "")),
                    "secret_key": str(creds.get("nebius_secret_key", "")),
                    "session_token": str(creds.get("s3_session_token", "")),
                },
            )
    except ProvisionerError as exc:
        hint = _agent_deploy_failure_hint(str(exc))
        if hint:
            # Print the concise diagnosis last (the raw Terraform output already
            # streamed above), so it is the final thing the operator sees.
            _fail(hint)
        _fail(f"Terraform deploy failed: {exc}")

    operation = current_operation()
    if operation is not None and operation.read().get("phase") == "mutating":
        # Mocked/embedded provisioners may return already-verified outputs without
        # calling the Terraform wrapper that normally records this transition.
        operation.transition("resource-created", details={"terraform": "completed"})
        operation.transition("state-durable", details={"state": "provider-verified"})

    typer.echo("  Phase 2/4: Terraform complete; resolving the new VM endpoint.")

    public_ip = str(tf_outputs.get("vm_ip", ""))
    instance_id = str(tf_outputs.get("instance_id", ""))
    ssh_key_path = str(
        tf_outputs.get("ssh_key_path", "") or ssh_public_key_path.removesuffix(".pub")
    )
    if not _is_routable_public_ip(public_ip):
        try:
            _destroy_agent_terraform(
                project,
                name,
                record={
                    "instance_id": instance_id,
                    "project_id": env_project_id,
                    "region": env_region,
                },
                rollback_operation=True,
            )
        except ProvisionerError as cleanup_exc:
            typer.echo(f"  Warning: terraform rollback failed: {cleanup_exc}", err=True)
        _fail("Terraform output did not include a routable public IP")

    prior_record = _agent_record(project, name)
    prior_matches = bool(
        prior_record
        and str(prior_record.get("project_id") or "") == env_project_id
        and str(prior_record.get("instance_id") or "") == instance_id
        and str(prior_record.get("public_ip") or "") == public_ip
    )
    if prior_matches:
        try:
            _prior_user, auth_password = _load_auth_secret(
                str(prior_record.get("auth_secret_path") or "")
            )
            auth_path = Path(str(prior_record.get("auth_secret_path") or ""))
        except ValueError:
            prior_matches = False
    if not prior_matches:
        auth_password = secrets.token_urlsafe(18)
        auth_path = _write_auth_secret(
            project_alias=project,
            name=name,
            user=DEFAULT_AGENT_USER,
            password=auth_password,
        )
    # tf_api_key / default_llm_model were resolved once up front (before Terraform).
    configured_llm_model = str(llm_model or "").strip() or default_llm_model
    # With no explicit --llm-models, seed the cost-ordered default ladder so
    # per-turn routing can reach every tier (cheap/standard/reasoning/vision)
    # out of the box. An explicit --llm-models acts as a governance allowlist.
    extra_llm_models = list(llm_models) if llm_models else list(DEFAULT_LLM_MODELS)
    configured_llm_models = _normalize_llm_models(
        [configured_llm_model, *extra_llm_models]
    )
    # A missing Token Factory key is already surfaced up front (before Terraform)
    # by the deploy prerequisite check above.
    rollback_record = {
        "instance_id": instance_id,
        "project_id": env_project_id,
        "region": env_region,
        "service_account_id": str(creds.get("service_account_id", "")),
        "foxglove": foxglove_settings,
    }
    partial_urls = build_agent_urls(
        public_ip, agent_port=agent_port, public_https=public_https
    )
    partial_record = {
        **partial_urls,
        "project_alias": project,
        "name": name,
        "project_id": env_project_id,
        "tenant_id": env_tenant_id,
        "region": env_region,
        "public_ip": public_ip,
        "instance_id": instance_id,
        "auth_user": DEFAULT_AGENT_USER,
        "auth_secret_path": str(auth_path),
        "ssh_key_path": ssh_key_path,
        "public_https": public_https,
        "setup_state": "remote_bootstrap_pending",
        "service_account_id": str(creds.get("service_account_id", "")),
        "foxglove": foxglove_settings,
    }
    _store_agent_record(project, name, partial_record)
    if operation is not None:
        operation.checkpoint(
            "vm_identity_durable",
            {
                "setup_phase": "vm_identity_durable",
                "instance_id": instance_id,
                "endpoint": public_ip,
                "auth_identity": DEFAULT_AGENT_USER,
            },
        )
    prior_setup_state = str((prior_record or {}).get("setup_state") or "")
    typer.echo(
        "  Phase 3/4: reconciling or installing agent services over SSH; "
        "package and image setup can be quiet for several minutes; secret-free "
        "progress heartbeats continue during remote work. Diagnose remotely with "
        f"`ssh -i {ssh_key_path} {ssh_user}@{public_ip} sudo "
        "journalctl -u cloud-final -u npa-agent-backend -n 100`."
    )
    convergence = converge_remote_agent_setup(
        operation=operation,
        resuming=prior_matches
        and prior_setup_state
        in {"remote_bootstrap_pending", "reconciliation_indeterminate"},
        bootstrap=_bootstrap_agent_stack,
        reconcile=_reconcile_agent_setup,
        bootstrap_kwargs={
            "instance_id": instance_id,
            "host": public_ip,
            "ssh_user": ssh_user,
            "ssh_key_path": ssh_key_path,
            "project_alias": project,
            "agent_name": name,
            "project_id": env_project_id,
            "tenant_id": env_tenant_id,
            "region": env_region,
            "auth_user": DEFAULT_AGENT_USER,
            "auth_password": auth_password,
            "agent_port": agent_port,
            "backend_port": backend_port,
            "rerun_port": rerun_port,
            "llm_model": configured_llm_model,
            "llm_models": configured_llm_models,
            "tf_api_key": tf_api_key,
            "s3_bucket": str(merged_vars.get("s3_bucket", "")),
            "s3_prefix": str(merged_vars.get("s3_prefix", "")),
            "s3_endpoint": str(merged_vars.get("s3_endpoint", "")),
            "s3_access_key": str(merged_vars.get("nebius_api_key", "")),
            "s3_secret_key": str(merged_vars.get("nebius_secret_key", "")),
            "s3_region": env_region,
            "nebius_project_id": env_project_id,
            "nebius_tenant_id": env_tenant_id,
            "service_account_id": str(creds.get("service_account_id", "")),
            "public_https": public_https,
            "foxglove_embed_src": foxglove_settings["embed_src"],
            "foxglove_viewer_backend": foxglove_settings["viewer_backend"],
            "foxglove_org_slug": foxglove_settings["org_slug"],
            "foxglove_live_url": foxglove_settings["live_url"],
            "foxglove_cloud_import_timeout_seconds": foxglove_settings[
                "cloud_import_timeout_seconds"
            ],
        },
        reconcile_kwargs={
            "host": public_ip,
            "ssh_user": ssh_user,
            "ssh_key_path": ssh_key_path,
            "project_alias": project,
            "agent_name": name,
            "project_id": env_project_id,
            "auth_user": DEFAULT_AGENT_USER,
            "auth_password": auth_password,
            "agent_port": agent_port,
            "public_https": public_https,
        },
        persist_pending=lambda state: _store_agent_record(
            project, name, {**partial_record, "setup_state": state}
        ),
        status=lambda message: typer.echo(f"  {message}"),
        progress=lambda payload: typer.echo(
            "progress: " + json.dumps(payload, sort_keys=True), err=True
        ),
        fatal_errors=(DeploymentIdentityError,),
        transport_errors=(ConfigError, SSHError, ValueError),
    )
    reconciliation = convergence.evidence
    bootstrap_error = convergence.primary_error
    if reconciliation.get("state") == "healthy":
        if bootstrap_error is not None:
            typer.echo(
                "  Remote reconciliation adopted the healthy agent after the local "
                "transport ended without a success response.",
                err=True,
            )
    elif bootstrap_error is not None:
        # The VM identity is durable and may already have been mutated.  Preserve
        # it for exact resume; a transport error is not authority to destroy it.
        _fail(
            "VM bootstrap transport failed and exact remote reconciliation is "
            f"{reconciliation.get('state', 'indeterminate')}; rerun `npa agent "
            f"fresh-setup --project {project} --name {name} ...` to resume the "
            "first incomplete phase. The healthy VM will not be replaced. "
            f"Primary transport error: {type(bootstrap_error).__name__}"
        )
    else:
        _fail(
            "remote bootstrap returned success but health reconciliation is "
            f"{reconciliation.get('state', 'indeterminate')}; the operation remains resumable"
        )

    typer.echo(
        "  Phase 4/4 probe: remote installer completed; checking ingress and service health."
    )

    ingress_ports: list[int] = [agent_port, rerun_port]
    if public_https:
        ingress_ports.append(DEFAULT_HTTPS_PORT)
    try:
        ensure_ingress(
            vm_id=instance_id,
            ports=tuple(ingress_ports),
            source=application_source,
            allow_world_open=str(
                merged_vars.get("allow_world_open_application", "false")
            ).lower()
            == "true",
            tool="agent",
        )
        remove_npa_ingress_for_instance_ports(
            instance_id,
            ports=(backend_port,),
            on_status=lambda message: typer.echo(f"  {message}"),
        )
    except NetworkIngressError as exc:
        try:
            _destroy_agent_terraform(
                project,
                name,
                record=rollback_record,
                rollback_operation=True,
            )
        except ProvisionerError as cleanup_exc:
            typer.echo(f"  Warning: terraform rollback failed: {cleanup_exc}", err=True)
        _fail(f"npa network ensure-ingress failed: {exc}")

    urls = build_agent_urls(public_ip, agent_port=agent_port, public_https=public_https)
    record = AgentConfig(
        project_alias=project,
        name=name,
        project_id=env_project_id,
        tenant_id=env_tenant_id,
        region=env_region,
        public_ip=public_ip,
        instance_id=instance_id,
        agent_url=urls["agent_url"],
        rerun_url=urls["rerun_url"],
        sim_viz_url=urls["sim_viz_url"],
        sim_assets_url=urls["sim_assets_url"],
        cameras_api_url=urls["cameras_api_url"],
        auth_user=DEFAULT_AGENT_USER,
        auth_secret_path=str(auth_path),
        llm_provider=DEFAULT_LLM_PROVIDER,
        llm_model=configured_llm_model,
        llm_models=tuple(configured_llm_models),
        public_url=urls["public_url"],
        public_https=public_https,
        direct_url=urls["direct_url"],
        ssh_key_path=ssh_key_path,
        service_account_id=str(creds.get("service_account_id", "")),
    )
    final_record = record.to_dict()
    final_record["foxglove"] = foxglove_settings
    final_record["setup_state"] = "healthy"
    final_record["setup_evidence"] = {
        key: reconciliation.get(key)
        for key in (
            "state",
            "service_fingerprint",
            "credential_fingerprint",
            "models_healthy",
            "remote_phase",
        )
    }
    _store_agent_record(project, name, final_record)
    if operation is not None:
        operation.checkpoint(
            "health_verified",
            {
                **reconciliation,
                "setup_phase": "health_verified",
                "instance_id": instance_id,
            },
        )
    _persist_agent_project_config(
        project=project,
        project_id=env_project_id,
        tenant_id=env_tenant_id,
        region=env_region,
        merged_vars=merged_vars,
    )

    typer.echo(f"Customer URL: {urls['public_url']}")
    typer.echo(f"public_url: {urls['public_url']}")
    if public_https:
        typer.echo(
            "Note: HTTPS uses a self-signed certificate — browsers will warn once; "
            "choose to proceed or use curl with -k."
        )
        typer.echo(f"direct_url: {urls['direct_url']}")
    typer.echo(f"rerun_url: {urls['rerun_url']}")
    typer.echo(f"sim_viz_url: {urls['sim_viz_url']}")
    typer.echo(f"sim_assets_url: {urls['sim_assets_url']}")
    typer.echo(f"cameras_api_url: {urls['cameras_api_url']}")
    typer.echo(f"llm: {DEFAULT_LLM_PROVIDER}:{configured_llm_model}")
    typer.echo(f"llm_models: {', '.join(configured_llm_models)}")
    typer.echo(f"auth_user: {DEFAULT_AGENT_USER}")
    typer.echo(f"auth_secret_path: {auth_path}")
    typer.echo(f"auth_password: {redact_value(auth_password)}")


@app.command("fresh-setup")
@resolve_typer_defaults
@_transactional_agent_command("npa agent fresh-setup")
def fresh_setup_cmd(
    project: str = typer.Option(
        "",
        "--project",
        help="NPA project alias for this fresh environment (default: configured default_project).",
    ),
    name: str = typer.Option(
        DEFAULT_AGENT_NAME, "--name", help="Agent deployment name."
    ),
    project_id: str = typer.Option(..., "--project-id", help="Nebius project ID."),
    tenant_id: str = typer.Option(..., "--tenant-id", help="Nebius tenant ID."),
    region: str = typer.Option("eu-north1", "--region", help="Nebius region."),
    ssh_user: str = typer.Option("ubuntu", "--ssh-user", help="SSH username."),
    ssh_public_key_path: str = typer.Option(
        "~/.ssh/id_ed25519.pub",
        "--ssh-public-key-path",
        help="SSH public key path for Terraform.",
    ),
    tf_var: list[str] = typer.Option(
        [], "--tf-var", help="Additional Terraform var key=value."
    ),
    agent_only: bool = typer.Option(
        False,
        "--agent-only",
        help="Provision the fresh agent VM without reserving capacity for a follow-on cluster.",
    ),
    agent_port: int = typer.Option(
        DEFAULT_AGENT_PORT, "--agent-port", help="Public agent UI port."
    ),
    backend_port: int = typer.Option(
        DEFAULT_BACKEND_PORT, "--backend-port", help="Internal agent backend port."
    ),
    rerun_port: int = typer.Option(
        DEFAULT_RERUN_PORT, "--rerun-port", help="Rerun service port."
    ),
    llm_model: str = typer.Option(
        DEFAULT_LLM_MODEL,
        "--llm-model",
        help="Default Token Factory model for agent chat.",
    ),
    llm_models: list[str] = typer.Option(
        [],
        "--llm-models",
        help="Additional Token Factory model IDs (repeat flag or comma-separate values).",
    ),
    no_public_https: bool = typer.Option(
        False,
        "--no-public-https",
        help="Disable HTTPS on port 443 (customer access uses http://IP:agent-port only).",
    ),
    replace: bool = typer.Option(
        False,
        "--replace",
        help="Destroy an existing agent with the same project/name before fresh deploy.",
    ),
) -> None:
    """Initialize fresh project config and deploy a new agent from scratch."""
    project = _resolve_project_alias(project)
    existing = _agent_record(project, name)
    existing_setup = str(existing.get("setup_state") or "") if existing else ""
    resumable_existing = bool(
        existing
        and existing_setup
        in {"remote_bootstrap_pending", "reconciliation_indeterminate"}
        and str(existing.get("project_id") or "") == project_id.strip()
        and str(existing.get("tenant_id") or "") == tenant_id.strip()
    )
    if existing and not replace and not resumable_existing:
        _fail(
            f"Agent {project}/{name} already exists. Use --replace or choose a new --project/--name."
        )
    if existing and replace:
        typer.echo(f"Replacing existing agent {project}/{name} ...")
        destroy_cmd(project=project, name=name)
    elif resumable_existing:
        typer.echo(
            f"Resuming the exact incomplete agent setup for {project}/{name}; "
            "the existing VM will be reconciled and retained.",
            err=True,
        )
    _store_project_environment(
        project=project,
        project_id=project_id.strip(),
        tenant_id=tenant_id.strip(),
        region=region.strip(),
    )
    deploy_cmd(
        project=project,
        name=name,
        project_id=project_id,
        tenant_id=tenant_id,
        region=region,
        ssh_user=ssh_user,
        ssh_public_key_path=ssh_public_key_path,
        tf_var=tf_var,
        agent_only=agent_only,
        agent_port=agent_port,
        backend_port=backend_port,
        rerun_port=rerun_port,
        llm_model=llm_model,
        llm_models=llm_models,
        no_public_https=no_public_https,
    )


@app.command("setup")
@resolve_typer_defaults
def setup_cmd(
    name: str = typer.Option(
        DEFAULT_AGENT_NAME, "--name", help="Agent deployment name."
    ),
    project: str = typer.Option(
        "",
        "--project",
        help="NPA project alias to deploy into (default: interactive pick from `npa configure`).",
    ),
    ssh_public_key_path: str = typer.Option(
        "~/.ssh/id_ed25519.pub",
        "--ssh-public-key-path",
        help="SSH public key for the VM.",
    ),
    tf_var: list[str] = typer.Option(
        [],
        "--tf-var",
        help="Additional Terraform var key=value; use for explicit SSH/application CIDRs.",
    ),
    replace: bool = typer.Option(
        False,
        "--replace",
        help="Replace an existing agent with the same project/name.",
    ),
) -> None:
    """Interactively deploy an agent VM into a project you already configured.

    This is the simple path: run ``npa configure`` first to connect npa to Nebius
    AI Cloud (which saves your accessible projects), then ``npa agent setup`` picks
    one of those projects — reading tenant/project/region from ``~/.npa/config.yaml``
    instead of re-typing ids — and deploys.

    How the agent VM gets Nebius AI Cloud credentials: deploy provisions (or
    reuses) an ``npa-agent`` service account in the project, grants it the tenant
    ``editors`` role, and **attaches it to the VM**. Code on the VM then mints
    short-lived IAM access tokens from the Nebius VM metadata endpoint
    (``http://metadata.nebius.internal/v1/iam/sa/token/access_token``) on demand —
    an auto-rotating, key-less credential. No static "AI Cloud key" is stored on
    the VM. The same service account's access key provides S3 access.
    """
    from npa.clients.config import default_project_name, list_projects

    projects = list_projects()
    if not projects:
        _fail(
            "No Nebius projects configured. Run `npa configure` first to connect "
            "npa to Nebius AI Cloud, then re-run `npa agent setup`."
        )

    alias = project.strip()
    if alias and alias not in projects:
        _fail(
            f"Project alias {alias!r} is not configured. Available: {', '.join(projects)} (or run `npa configure`)."
        )
    if not alias:
        aliases = list(projects)
        if len(aliases) == 1:
            alias = aliases[0]
        else:
            default_alias = default_project_name()
            typer.echo("Configured Nebius projects:\n")
            for i, candidate in enumerate(aliases, start=1):
                stanza = projects[candidate] or {}
                marker = "  *" if candidate == default_alias else "   "
                typer.echo(
                    f"{marker}{i:>2}. {candidate}  ({stanza.get('region', '?')})  [{stanza.get('project_id', '')}]"
                )
            default_pick = (
                str(aliases.index(default_alias) + 1)
                if default_alias in aliases
                else "1"
            )
            raw = typer.prompt(
                "\nDeploy the agent into which project? (number)",
                default=default_pick,
            )
            choice = raw.strip()
            idx = int(choice) - 1 if choice.isdigit() else int(default_pick) - 1
            if not (0 <= idx < len(aliases)):
                idx = int(default_pick) - 1
            alias = aliases[idx]

    stanza = projects.get(alias) or {}
    project_id = str(stanza.get("project_id", "")).strip()
    tenant_id = str(stanza.get("tenant_id", "")).strip()
    region = str(stanza.get("region", "") or "eu-north1").strip()
    if not project_id or not tenant_id:
        _fail(
            f"Project {alias!r} is missing project_id/tenant_id in ~/.npa/config.yaml. Re-run `npa configure`."
        )

    if not Path(ssh_public_key_path).expanduser().exists():
        _fail(
            f"SSH public key not found at {ssh_public_key_path}. Create one with "
            "`ssh-keygen -t ed25519`, or pass --ssh-public-key-path."
        )

    typer.echo(
        f"\nDeploying agent '{name}' into project '{alias}' "
        f"({project_id}, {region}).\n"
        "The VM will get an attached `npa-agent` service account for key-less "
        "Nebius AI Cloud access.\n"
    )
    fresh_setup_cmd(
        project=alias,
        name=name,
        project_id=project_id,
        tenant_id=tenant_id,
        region=region,
        ssh_public_key_path=ssh_public_key_path,
        tf_var=tf_var,
        replace=replace,
    )


@app.command("bootstrap")
@resolve_typer_defaults
@_transactional_agent_command("npa agent bootstrap")
def bootstrap_cmd(
    project: str = typer.Option(
        "", "--project", help="NPA project alias (default: configured default_project)."
    ),
    name: str = typer.Option(
        DEFAULT_AGENT_NAME, "--name", help="Agent deployment name."
    ),
    ssh_user: str = typer.Option("ubuntu", "--ssh-user", help="SSH username."),
    ssh_key: str = typer.Option(
        "",
        "--ssh-key",
        help="SSH private key path (defaults to agent record or NPA_SSH_KEY).",
    ),
    agent_port: int = typer.Option(
        DEFAULT_AGENT_PORT, "--agent-port", help="Public agent UI port."
    ),
    backend_port: int = typer.Option(
        DEFAULT_BACKEND_PORT, "--backend-port", help="Internal agent backend port."
    ),
    rerun_port: int = typer.Option(
        DEFAULT_RERUN_PORT, "--rerun-port", help="Rerun service port."
    ),
    llm_model: str = typer.Option(
        "", "--llm-model", help="Override the active Token Factory model."
    ),
    llm_models: list[str] = typer.Option(
        [],
        "--llm-models",
        help="Override additional Token Factory model IDs (repeat flag or comma-separated values).",
    ),
    refresh_credentials: bool = typer.Option(
        False,
        "--refresh-credentials",
        help="Re-provision the long-lived npa-agent service account and restage VM credentials.",
    ),
    foxglove_embed_src: str = agent_foxglove_config.embed_src_option(),
    foxglove_viewer_backend: str = agent_foxglove_config.viewer_backend_option(),
    foxglove_org_slug: str = agent_foxglove_config.org_slug_option(),
    foxglove_live_url: str = agent_foxglove_config.live_url_option(),
    no_public_https: bool = typer.Option(
        False,
        "--no-public-https",
        help="Disable HTTPS on port 443 (customer access uses http://IP:agent-port only).",
    ),
) -> None:
    """Re-bootstrap agent UI/backend/nginx on an existing VM (refresh without Terraform)."""
    project = _resolve_project_alias(project)
    record = _agent_record(project, name)
    if not record:
        _fail(f"Agent config not found for {project}/{name}")
    foxglove_settings = _resolve_foxglove_settings_or_fail(
        embed_src=foxglove_embed_src,
        viewer_backend=foxglove_viewer_backend,
        org_slug=foxglove_org_slug,
        live_url=foxglove_live_url,
        saved=record.get("foxglove"),
    )
    try:
        public_ip = _resolve_record_public_ip(record)
    except NetworkIngressError as exc:
        _fail(str(exc))
    public_https = not no_public_https
    ssh_key_path = _resolve_agent_ssh_key(record, cli_ssh_key=ssh_key or None)
    if not Path(ssh_key_path).expanduser().exists():
        _fail(
            f"SSH private key not found at {ssh_key_path!r}. "
            "Pass --ssh-key, set NPA_SSH_KEY, or redeploy to persist ssh_key_path on the agent record."
        )
    try:
        auth_user, auth_password = _load_auth_secret(
            str(record.get("auth_secret_path", ""))
        )
    except ValueError as exc:
        _fail(str(exc))
    tf_api_key, default_llm_model = _resolve_deploy_llm_credentials()
    requested_llm_model = str(llm_model or "").strip()
    resolved_llm_model = requested_llm_model or default_llm_model
    # No explicit --llm-models => seed the cost-ordered default ladder (all
    # tiers). Existing record models are still merged below, so re-bootstrap
    # keeps any previously configured set.
    extra_llm_models = list(llm_models) if llm_models else list(DEFAULT_LLM_MODELS)
    resolved_llm_models = _normalize_llm_models([resolved_llm_model, *extra_llm_models])
    llm_block = record.get("llm", {}) if isinstance(record.get("llm"), dict) else {}
    if isinstance(llm_block.get("models"), list):
        resolved_llm_models = _normalize_llm_models(
            [*resolved_llm_models, *[str(item) for item in llm_block.get("models", [])]]
        )
    if (
        not requested_llm_model
        and isinstance(llm_block.get("model"), str)
        and llm_block["model"].strip()
    ):
        resolved_llm_model = llm_block["model"].strip()
    if resolved_llm_model not in resolved_llm_models:
        resolved_llm_models.insert(0, resolved_llm_model)
    if not tf_api_key:
        typer.echo(
            "Warning: Token Factory API key not found; chat endpoint will return 503.",
            err=True,
        )
    project_id = str(record.get("project_id", "")).strip()
    tenant_id = str(record.get("tenant_id", "")).strip()
    region = str(record.get("region", "") or "eu-north1")
    (
        s3_bucket,
        s3_prefix,
        s3_endpoint,
        s3_access_key,
        s3_secret_key,
        service_account_id,
    ) = _resolve_agent_storage_credentials(project, record)
    if not service_account_id:
        service_account_id = _resolve_agent_service_account_id(project, record)
    agent_credentials: dict[str, str] | None = None
    if refresh_credentials:
        if not (project_id and tenant_id and region):
            _fail(
                "agent record is missing project_id, tenant_id, or region for credential refresh"
            )
        from npa.clients.nebius import NebiusError, bootstrap_agent_environment

        creds: dict[str, str] | None = None
        try:
            try:
                configured_storage = _resolve_deploy_storage_credentials(
                    region=region,
                    project_alias=project,
                )
            except AgentStorageCredentialError:
                configured_storage = None
            creds = bootstrap_agent_environment(
                project_id,
                tenant_id,
                region,
                on_status=lambda msg: typer.echo(f"  {msg}"),
                **(
                    {"reuse_storage_credentials": configured_storage}
                    if configured_storage is not None
                    else {}
                ),
            )
        except NebiusError as exc:
            typer.echo(
                f"Warning: npa-agent provisioning failed ({exc}); reusing existing credentials.",
                err=True,
            )
        if creds is None:
            creds = _creds_from_terraform_state(project, record)
        if creds is None:
            _fail(
                "Nebius credential refresh failed and no terraform_state fallback is configured"
            )
        creds = _resolve_deploy_storage_credentials(
            region=region,
            bootstrap_creds=creds,
            project_alias=project,
        )
        agent_credentials = _agent_credentials_payload(creds)
        refreshed_service_account_id = str(
            agent_credentials.get("service_account_id") or ""
        ).strip()
        try:
            refreshed_service_account_id = consistent_agent_service_account_id(
                service_account_id, refreshed_service_account_id
            )
        except ValueError as exc:
            _fail(f"{exc}; refusing to replace the attached identity")
        s3_bucket = agent_credentials["s3_bucket"]
        s3_prefix = agent_credentials.get("s3_prefix", "")
        s3_endpoint = agent_credentials["s3_endpoint"]
        s3_access_key = agent_credentials["access_key"]
        s3_secret_key = agent_credentials["secret_key"]
        service_account_id = refreshed_service_account_id
        agent_credentials["service_account_id"] = service_account_id
        if not service_account_id:
            service_account_id = _resolve_agent_service_account_id(project, record)
            agent_credentials["service_account_id"] = service_account_id
        if s3_access_key and s3_secret_key:
            persist_agent_terraform_credentials(
                str(record.get("project_id") or ""),
                alias=project,
                bucket=s3_bucket,
                endpoint=s3_endpoint,
                access_key=s3_access_key,
                secret_key=s3_secret_key,
            )
            write_config(
                {
                    "projects": {
                        project: {
                            "terraform_state": {
                                "bucket": s3_bucket,
                                "endpoint": s3_endpoint,
                                "credential_source": "project_credentials_v2",
                            },
                        }
                    }
                }
            )
    operation = current_operation()
    resuming = str(record.get("setup_state") or "") in {
        "remote_bootstrap_pending",
        "reconciliation_indeterminate",
    }
    convergence = converge_remote_agent_setup(
        operation=operation,
        resuming=resuming,
        bootstrap=_bootstrap_agent_stack,
        reconcile=_reconcile_agent_setup,
        bootstrap_kwargs={
            "instance_id": str(record.get("instance_id") or ""),
            "host": public_ip,
            "ssh_user": ssh_user,
            "ssh_key_path": ssh_key_path,
            "project_alias": project,
            "agent_name": name,
            "project_id": str(record.get("project_id", "") or ""),
            "tenant_id": str(record.get("tenant_id", "") or ""),
            "region": str(record.get("region", "") or "eu-north1"),
            "auth_user": auth_user,
            "auth_password": auth_password,
            "agent_port": agent_port,
            "backend_port": backend_port,
            "rerun_port": rerun_port,
            "llm_model": resolved_llm_model,
            "llm_models": resolved_llm_models,
            "tf_api_key": tf_api_key,
            "s3_bucket": s3_bucket,
            "s3_prefix": s3_prefix,
            "s3_endpoint": s3_endpoint,
            "s3_access_key": s3_access_key,
            "s3_secret_key": s3_secret_key,
            "s3_region": region,
            "nebius_project_id": project_id,
            "nebius_tenant_id": tenant_id,
            "service_account_id": service_account_id,
            "public_https": public_https,
            "foxglove_embed_src": foxglove_settings["embed_src"],
            "foxglove_viewer_backend": foxglove_settings["viewer_backend"],
            "foxglove_org_slug": foxglove_settings["org_slug"],
            "foxglove_live_url": foxglove_settings["live_url"],
            "foxglove_cloud_import_timeout_seconds": foxglove_settings[
                "cloud_import_timeout_seconds"
            ],
        },
        reconcile_kwargs={
            "host": public_ip,
            "ssh_user": ssh_user,
            "ssh_key_path": ssh_key_path,
            "project_alias": project,
            "agent_name": name,
            "project_id": project_id,
            "auth_user": auth_user,
            "auth_password": auth_password,
            "agent_port": agent_port,
            "public_https": public_https,
        },
        persist_pending=lambda state: _store_agent_record(
            project, name, {**record, "setup_state": state}
        ),
        status=lambda message: typer.echo(f"  {message}"),
        progress=lambda payload: typer.echo(
            "progress: " + json.dumps(payload, sort_keys=True), err=True
        ),
        fatal_errors=(DeploymentIdentityError,),
        transport_errors=(ConfigError, SSHError, ValueError),
    )
    reconciliation = convergence.evidence
    bootstrap_error = convergence.primary_error
    if reconciliation.get("state") != "healthy":
        updated_incomplete = dict(record)
        updated_incomplete["setup_state"] = "reconciliation_indeterminate"
        _store_agent_record(project, name, updated_incomplete)
        if bootstrap_error is not None:
            _fail(
                "VM bootstrap transport failed; exact remote reconciliation is "
                f"{reconciliation.get('state', 'indeterminate')}. Rerun this same "
                "bootstrap command to resume without replacing the VM. Primary "
                f"transport error: {type(bootstrap_error).__name__}"
            )
        _fail("remote bootstrap returned but exact health evidence is incomplete")
    if bootstrap_error is not None:
        typer.echo(
            "  Remote reconciliation adopted the healthy agent after transport loss.",
            err=True,
        )
    instance_id = str(record.get("instance_id", "")).strip()
    if instance_id:
        try:
            remove_npa_ingress_for_instance_ports(
                instance_id,
                ports=(backend_port,),
                on_status=lambda message: typer.echo(f"  {message}"),
            )
        except NetworkIngressError as exc:
            _fail(f"npa network ensure-ingress failed: {exc}")
    urls = build_agent_urls(public_ip, agent_port=agent_port, public_https=public_https)
    updated = dict(record)
    updated.update(urls)
    updated["public_ip"] = public_ip
    updated["public_https"] = public_https
    llm_payload = dict(
        updated.get("llm", {}) if isinstance(updated.get("llm"), dict) else {}
    )
    llm_payload["provider"] = DEFAULT_LLM_PROVIDER
    llm_payload["model"] = resolved_llm_model
    llm_payload["models"] = list(resolved_llm_models)
    updated["llm"] = llm_payload
    updated["ssh_key_path"] = ssh_key_path
    updated["foxglove"] = foxglove_settings
    updated["setup_state"] = "healthy"
    updated["setup_evidence"] = {
        key: reconciliation.get(key)
        for key in (
            "state",
            "service_fingerprint",
            "credential_fingerprint",
            "models_healthy",
            "remote_phase",
        )
    }
    if service_account_id:
        updated["service_account_id"] = service_account_id
        _persist_agent_service_account_id(service_account_id, project_id)
    # Storage credentials remain in the owner-only project credential store and
    # are re-resolved on resume. Do not duplicate them into config-backed agent
    # records, operation journals, or teardown receipts.
    updated.pop("credentials", None)
    _store_agent_record(project, name, updated)
    if operation is not None:
        operation.checkpoint(
            "health_verified",
            {
                **reconciliation,
                "setup_phase": "health_verified",
                "instance_id": instance_id,
            },
        )
    typer.echo(f"Customer URL: {urls['public_url']}")
    typer.echo(f"bootstrapped: {project}/{name} at {urls['public_url']}")
    if public_https:
        typer.echo(f"direct_url: {urls['direct_url']}")


@app.command("status")
@intent_boundary(OperationIntent.OBSERVE)
def status_cmd(
    project: str = typer.Option(
        "", "--project", help="NPA project alias (default: configured default_project)."
    ),
    name: str = typer.Option(
        DEFAULT_AGENT_NAME, "--name", help="Agent deployment name."
    ),
    output_json: bool = typer.Option(False, "--json", help="Emit JSON."),
) -> None:
    """Show agent status, URLs, and health checks."""
    project = _resolve_project_alias(project)
    record = _agent_record(project, name)
    if not record:
        from npa.agent_status import partial_agent_status

        payload = partial_agent_status(project, name)
        if output_json:
            typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        else:
            for key, value in payload.items():
                typer.echo(f"{key}: {value}")
        if payload.get("classification") == "NOT_FOUND":
            raise typer.Exit(code=1)
        return
    try:
        auth_user, auth_password = _load_auth_secret(
            str(record.get("auth_secret_path", ""))
        )
    except ValueError as exc:
        _fail(str(exc))
    agent_url = str(record.get("agent_url", ""))
    rerun_url = str(record.get("rerun_url", ""))
    sim_viz_url = str(record.get("sim_viz_url", rerun_url))
    sim_assets_url = str(record.get("sim_assets_url", agent_url))
    cameras_api_url = str(
        record.get(
            "cameras_api_url", f"{agent_url.rstrip('/')}/assets/api/sim-assets/cameras"
        )
    )
    public_url = _record_customer_url(record)
    tls_verify = _record_tls_verify(record)
    ui_ok, ui_code = _health(
        agent_url, user=auth_user, password=auth_password, verify=tls_verify
    )
    basic_auth_enforced, unauthenticated_ui_code = _basic_auth_protects_endpoint(
        agent_url,
        verify=tls_verify,
    )
    rerun_ok, rerun_code = _health(
        sim_viz_url, user=auth_user, password=auth_password, verify=tls_verify
    )
    endpoint_disclosure_allowed = bool(ui_ok and basic_auth_enforced)
    payload = {
        "project": project,
        "name": name,
        "public_ip": record.get("public_ip", "") if endpoint_disclosure_allowed else "",
        "public_url": public_url if endpoint_disclosure_allowed else "",
        "public_https": _record_public_https(record),
        # The direct service URL is not covered by the public HTTPS Basic Auth
        # proof and is intentionally never part of status handoffs.
        "direct_url": "",
        "ui_url": agent_url if endpoint_disclosure_allowed else "",
        "rerun_url": rerun_url if endpoint_disclosure_allowed else "",
        "sim_viz_url": sim_viz_url if endpoint_disclosure_allowed else "",
        "sim_assets_url": sim_assets_url if endpoint_disclosure_allowed else "",
        "cameras_api_url": cameras_api_url if endpoint_disclosure_allowed else "",
        "health": bool(ui_ok and rerun_ok and basic_auth_enforced),
        "basic_auth_enforced": basic_auth_enforced,
        "unauthenticated_ui_status_code": unauthenticated_ui_code,
        "endpoint_disclosure_allowed": endpoint_disclosure_allowed,
        "ui_status_code": ui_code,
        "rerun_status_code": rerun_code,
        "llm": record.get("llm", {}),
    }
    if output_json:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    for key, value in payload.items():
        typer.echo(f"{key}: {value}")


@app.command("destroy")
@resolve_typer_defaults
@functools.wraps(_destroy_cmd_impl)
def destroy_cmd(*args: Any, **kwargs: Any) -> None:
    """Destroy agent VM/resources using exact live, receipt, or argument identity."""

    _destroy_cmd_impl(*args, **kwargs)


@app.command("verify-live")
def verify_live_cmd(
    project: str = typer.Option(
        "", "--project", help="NPA project alias (default: configured default_project)."
    ),
    name: str = typer.Option(
        DEFAULT_AGENT_NAME, "--name", help="Agent deployment name."
    ),
) -> None:
    """Exit 0 only when live infra checks and tests pass."""
    project = _resolve_project_alias(project)
    record = _agent_record(project, name)
    if not record:
        _fail(f"Agent config not found for {project}/{name}")
    public_ip = str(record.get("public_ip", "")).strip()
    region = str(record.get("region", "")).strip()
    if (
        not public_ip
        or public_ip in {"localhost", "127.0.0.1"}
        or public_ip.startswith("127.")
    ):
        _fail("agent VM does not have a non-localhost public IP")
    if not _is_routable_public_ip(public_ip):
        _fail("agent VM does not have a non-localhost public IP")
    if not region:
        _fail("agent record is missing its deploy region")
    try:
        auth_user, auth_password = _load_auth_secret(
            str(record.get("auth_secret_path", ""))
        )
    except ValueError as exc:
        _fail(str(exc))

    customer_url = _record_customer_url(record)
    tls_verify = _record_tls_verify(record)
    if customer_url:
        if _record_public_https(record) and customer_url != f"https://{public_ip}/":
            _fail("public customer URL is not the canonical HTTPS public-IP endpoint")
        try:
            welcome_resp = httpx.get(
                f"{customer_url.rstrip('/')}/welcome",
                timeout=5.0,
                verify=tls_verify,
            )
            if welcome_resp.status_code != 200:
                _fail(
                    f"public welcome page unhealthy (status={welcome_resp.status_code})"
                )
            healthz_resp = httpx.get(
                f"{customer_url.rstrip('/')}/healthz",
                timeout=5.0,
                verify=tls_verify,
            )
            if healthz_resp.status_code != 200:
                _fail(f"public healthz unhealthy (status={healthz_resp.status_code})")
            unauthenticated_ui = httpx.get(
                customer_url,
                timeout=5.0,
                verify=tls_verify,
                follow_redirects=False,
            )
            if unauthenticated_ui.status_code != 401:
                _fail(
                    "public UI did not enforce basic authentication "
                    f"(status={unauthenticated_ui.status_code})"
                )
            api_health = httpx.get(
                f"{customer_url.rstrip('/')}/api/health",
                auth=(auth_user, auth_password),
                timeout=5.0,
                verify=tls_verify,
            )
            if api_health.status_code != 200:
                _fail(
                    "authenticated public API unhealthy "
                    f"(status={api_health.status_code})"
                )
        except httpx.HTTPError as exc:
            _fail(f"public customer URL unreachable: {exc}")

    ui_ok, ui_code = _health(
        str(record.get("agent_url", "")),
        user=auth_user,
        password=auth_password,
        verify=tls_verify,
    )
    if not ui_ok:
        _fail(f"UI health failed behind basic auth (status={ui_code})")
    sim_viz_url = str(record.get("sim_viz_url", record.get("rerun_url", "")))
    rerun_ok, rerun_code = _health(
        sim_viz_url,
        user=auth_user,
        password=auth_password,
        verify=tls_verify,
    )
    if not rerun_ok:
        _fail(f"embedded rerun iframe endpoint unhealthy (status={rerun_code})")
    try:
        sim_viz_status_resp = httpx.get(
            f"{str(record.get('agent_url', '')).rstrip('/')}/api/sim-viz/status",
            auth=(auth_user, auth_password),
            timeout=5.0,
            verify=tls_verify,
        )
        sim_viz_status_resp.raise_for_status()
        sim_viz_status = sim_viz_status_resp.json()
    except Exception as exc:  # noqa: BLE001
        _fail(f"sim viz status endpoint unhealthy: {exc}")
    if not isinstance(sim_viz_status, dict):
        _fail("sim viz status endpoint did not return JSON object")

    sim_assets_base = str(
        record.get("sim_assets_url", record.get("agent_url", ""))
    ).rstrip("/")
    if not sim_assets_base:
        _fail("sim_assets_url missing from agent config")
    try:
        sim_assets_resp = httpx.get(
            f"{sim_assets_base}/api/sim-assets",
            auth=(auth_user, auth_password),
            timeout=5.0,
            verify=tls_verify,
        )
        sim_assets_resp.raise_for_status()
        sim_assets_payload = sim_assets_resp.json()
    except Exception as exc:  # noqa: BLE001
        _fail(f"sim assets endpoint unhealthy: {exc}")
    if (
        not isinstance(sim_assets_payload, dict)
        or "scene_spec" not in sim_assets_payload
        or "robot_spec" not in sim_assets_payload
    ):
        _fail("sim assets endpoint missing scene_spec/robot_spec payload")

    try:
        cameras_resp = httpx.get(
            f"{sim_assets_base}/api/sim-assets/cameras",
            auth=(auth_user, auth_password),
            timeout=5.0,
            verify=tls_verify,
        )
        cameras_resp.raise_for_status()
        cameras_payload = cameras_resp.json()
    except Exception as exc:  # noqa: BLE001
        _fail(f"cameras endpoint unhealthy: {exc}")
    cameras = (
        cameras_payload.get("cameras", []) if isinstance(cameras_payload, dict) else []
    )
    if not isinstance(cameras, list) or not cameras:
        _fail("cameras endpoint returned no cameras")

    selection_body = {
        "robot_preset": "franka",
        "sim_backend": "isaac",
        "scene_spec_uri": "stock://scene/default",
        "assets_uri": "",
        "robot_spec_uri": "stock://robot/franka",
        "cameras_uri": "stock://cameras/default",
        "props": ["cube"],
    }
    try:
        selection_set = httpx.post(
            f"{sim_assets_base}/api/sim-assets/selection",
            auth=(auth_user, auth_password),
            json=selection_body,
            timeout=5.0,
            verify=tls_verify,
        )
        selection_set.raise_for_status()
        selection_get = httpx.get(
            f"{sim_assets_base}/api/sim-assets/selection",
            auth=(auth_user, auth_password),
            timeout=5.0,
            verify=tls_verify,
        )
        selection_get.raise_for_status()
        selected_payload = selection_get.json()
    except Exception as exc:  # noqa: BLE001
        _fail(f"sim asset selection round-trip failed: {exc}")
    if not isinstance(selected_payload, dict):
        _fail("sim asset selection GET did not return JSON object")
    for key in (
        "scene_spec_uri",
        "assets_uri",
        "robot_spec_uri",
        "cameras_uri",
        "robot_preset",
        "sim_backend",
    ):
        if selected_payload.get(key) != selection_body[key]:
            _fail(f"sim asset selection round-trip did not persist {key}")

    try:
        submit_resp = httpx.post(
            f"{str(record.get('agent_url', '')).rstrip('/')}/api/workflows/sim2real/submit",
            auth=(auth_user, auth_password),
            json={},
            timeout=5.0,
            verify=tls_verify,
        )
        submit_resp.raise_for_status()
        submit_payload = submit_resp.json()
    except Exception as exc:  # noqa: BLE001
        _fail(f"workflow submit endpoint failed: {exc}")
    if not isinstance(submit_payload, dict) or not submit_payload.get("run_id"):
        _fail("workflow submit endpoint did not return run_id")
    submit_run_id = str(submit_payload.get("run_id") or "").strip()
    submit_viz = submit_payload.get("sim_viz", {})
    if not isinstance(submit_viz, dict) or submit_viz.get("run_id") != submit_run_id:
        _fail("workflow submit endpoint did not return run-scoped sim_viz")
    if not (submit_viz.get("rrd_uri") or submit_viz.get("rerun_ready")):
        _fail("workflow submit endpoint did not attach a visualizable .rrd to the run")
    try:
        submitted_status_resp = httpx.get(
            f"{str(record.get('agent_url', '')).rstrip('/')}/api/sim-viz/status",
            auth=(auth_user, auth_password),
            params={"run_id": submit_run_id},
            timeout=15.0,
            verify=tls_verify,
        )
        submitted_status_resp.raise_for_status()
        submitted_status = submitted_status_resp.json()
    except Exception as exc:  # noqa: BLE001
        _fail(f"submitted sim2real run status endpoint failed: {exc}")
    if (
        not isinstance(submitted_status, dict)
        or submitted_status.get("run_id") != submit_run_id
    ):
        _fail("submitted sim2real run status did not preserve run_id")
    if not submitted_status.get("rrd_uri"):
        _fail("submitted sim2real run status did not include rrd_uri")
    try:
        submitted_rrd_blob = httpx.get(
            f"{str(record.get('agent_url', '')).rstrip('/')}/api/sim-viz/rrd-blob",
            auth=(auth_user, auth_password),
            params={"run_id": submit_run_id},
            timeout=15.0,
            verify=tls_verify,
        )
        submitted_rrd_blob.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        _fail(f"submitted sim2real run rrd-blob endpoint failed: {exc}")
    if len(submitted_rrd_blob.content) < 64:
        _fail(
            "submitted sim2real run rrd-blob endpoint returned unexpectedly small payload"
        )

    try:
        load_demo_resp = httpx.post(
            f"{str(record.get('agent_url', '')).rstrip('/')}/api/sim-viz/load-franka-demo",
            auth=(auth_user, auth_password),
            json={"camera": "workspace"},
            timeout=30.0,
            verify=tls_verify,
        )
        load_demo_resp.raise_for_status()
        load_demo_payload = load_demo_resp.json()
    except Exception as exc:  # noqa: BLE001
        _fail(f"load-franka-demo endpoint failed: {exc}")
    if not isinstance(load_demo_payload, dict) or not load_demo_payload.get("ok"):
        _fail("load-franka-demo endpoint did not return ok=true")
    sim_viz_after_demo = load_demo_payload.get("sim_viz", {})
    if not isinstance(sim_viz_after_demo, dict) or not (
        sim_viz_after_demo.get("rerun_ready") or sim_viz_after_demo.get("rrd_uri")
    ):
        _fail("load-franka-demo did not mark rerun_ready/rrd_uri")

    try:
        preview_resp = httpx.post(
            f"{str(record.get('agent_url', '')).rstrip('/')}/api/sim-viz/camera-preview",
            auth=(auth_user, auth_password),
            json={"camera": "workspace"},
            timeout=15.0,
            verify=tls_verify,
        )
        preview_resp.raise_for_status()
        preview_payload = preview_resp.json()
    except Exception as exc:  # noqa: BLE001
        _fail(f"camera-preview endpoint failed: {exc}")
    if not isinstance(preview_payload, dict) or not preview_payload.get("ok"):
        _fail("camera-preview endpoint did not return ok=true")

    agent_base = str(record.get("agent_url", "")).rstrip("/")
    try:
        rrd_resp = httpx.get(
            f"{agent_base}/api/sim-viz/rrd",
            auth=(auth_user, auth_password),
            timeout=15.0,
            verify=tls_verify,
        )
        rrd_resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        _fail(f"sim-viz rrd endpoint failed after load-franka-demo: {exc}")
    rrd_ct = str(rrd_resp.headers.get("content-type", ""))
    if "application/json" in rrd_ct:
        if not isinstance(rrd_resp.json(), dict):
            _fail("sim-viz rrd JSON response was not an object")
    elif len(rrd_resp.content) < 64:
        _fail("sim-viz rrd endpoint returned unexpectedly small payload")
    try:
        rrd_blob_resp = httpx.get(
            f"{agent_base}/api/sim-viz/rrd-blob",
            auth=(auth_user, auth_password),
            timeout=15.0,
            verify=tls_verify,
        )
        rrd_blob_resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        _fail(f"sim-viz rrd-blob endpoint failed after load-franka-demo: {exc}")
    rrd_blob_ct = str(rrd_blob_resp.headers.get("content-type", ""))
    if "application/json" in rrd_blob_ct:
        if not isinstance(rrd_blob_resp.json(), dict):
            _fail("sim-viz rrd-blob JSON response was not an object")
    elif len(rrd_blob_resp.content) < 64:
        _fail("sim-viz rrd-blob endpoint returned unexpectedly small payload")

    rerun_static_ok = False
    for static_path in (
        "/rerun/index.js",
        "/rerun/re_viewer.js",
        "/rerun/favicon.ico",
        "/rerun/version",
    ):
        try:
            static_resp = httpx.get(
                f"{agent_base}{static_path}",
                auth=(auth_user, auth_password),
                timeout=15.0,
                verify=tls_verify,
            )
            if static_resp.status_code == 200 and static_resp.content:
                rerun_static_ok = True
                break
        except httpx.HTTPError:
            continue
    if not rerun_static_ok:
        _fail(
            "rerun static asset probe failed (no /rerun/*.js|ico|version responded 200)"
        )

    # Lichtblick embed probe (informational): the recordings alias serves the co-served
    # MCAP same-origin, and /lichtblick/ proxies the viewer sidecar. The sidecar is
    # best-effort (docker/image), so this never fails the run — it reports embed status.
    for lichtblick_path in ("/lichtblick/recordings/sim2real.mcap", "/lichtblick/"):
        try:
            lb_resp = httpx.get(
                f"{agent_base}{lichtblick_path}",
                auth=(auth_user, auth_password),
                timeout=15.0,
                verify=tls_verify,
            )
            typer.echo(
                f"lichtblick embed probe {lichtblick_path} -> {lb_resp.status_code}"
            )
        except httpx.HTTPError as exc:
            typer.echo(f"lichtblick embed probe {lichtblick_path} -> error: {exc}")

    from npa.agent_rerun_bundle_check import (
        check_rerun_bundle_load_budget,
        format_bundle_budget_report,
    )

    bundle_result = check_rerun_bundle_load_budget(
        agent_base,
        auth=(auth_user, auth_password),
        verify=tls_verify,
    )
    typer.echo(format_bundle_budget_report(bundle_result))
    if not bundle_result.ok:
        _fail("rerun bundle load budget failed: " + "; ".join(bundle_result.errors[:4]))

    try:
        health_resp = httpx.get(
            f"{agent_base}/api/health",
            auth=(auth_user, auth_password),
            timeout=5.0,
            verify=tls_verify,
        )
        health_resp.raise_for_status()
        health_payload = health_resp.json()
    except Exception as exc:  # noqa: BLE001
        _fail(f"health endpoint failed: {exc}")
    if not isinstance(health_payload, dict) or not health_payload.get("ok"):
        _fail("health endpoint did not return ok=true")

    try:
        leisaac_status_resp = httpx.get(
            f"{agent_base}/api/leisaac/status",
            auth=(auth_user, auth_password),
            timeout=10.0,
            verify=tls_verify,
        )
        leisaac_status_resp.raise_for_status()
        leisaac_status_payload = leisaac_status_resp.json()
    except Exception as exc:  # noqa: BLE001
        _fail(f"LeIsaac status endpoint failed: {exc}")
    if not isinstance(leisaac_status_payload, dict) or not isinstance(
        leisaac_status_payload.get("available"), bool
    ):
        _fail("LeIsaac status endpoint did not return an availability object")

    try:
        workflow_status_resp = httpx.get(
            f"{agent_base}/api/workflows/sim2real/status",
            auth=(auth_user, auth_password),
            timeout=30.0,
            verify=tls_verify,
        )
        workflow_status_resp.raise_for_status()
        workflow_status_payload = workflow_status_resp.json()
    except Exception as exc:  # noqa: BLE001
        _fail(f"workflow status endpoint failed: {exc}")
    if not isinstance(workflow_status_payload, dict):
        _fail("workflow status endpoint did not return JSON object")

    ui_resp = httpx.get(
        str(record.get("agent_url", "")),
        auth=(auth_user, auth_password),
        timeout=10.0,
        verify=tls_verify,
    )
    if ui_resp.status_code != 200:
        _fail(f"UI html fetch failed (status={ui_resp.status_code})")
    ui_html = ui_resp.text
    for marker in (
        'name="viewport" content="width=device-width',
        'id="chatForm"',
        'id="mobileChatAuth"',
        'id="tabMain"',
        'id="tabRerun"',
        'id="stagesPanel"',
        'id="agentAccessPanel"',
        'id="agentAccessProjectSelect"',
        "/api/access",
        "<h3>Stages</h3>",
        'id="stagesRunSelect"',
        'id="stagesLoadRun"',
        "loadSelectedRun",
        "stages-run-picker",
        "filterStagesRunSelect",
        "Search NPA workflow/artifact runs",
        "function sendChat(",
        "function wireUi(",
        "activateMainTab",
        "initNpaAgentUi",
        "mobile-agent",
        "history.replaceState",
        "location.username",
        f'name="npa-ui-version" content="{AGENT_UI_VERSION}"',
        # Media preview contract — keep in sync with AGENT_MEDIA_PREVIEW_CONTRACT
        # (HTML-visible subset; backend route markers are source-tested separately).
        "authenticatedPreviewObjectUrl",
        "Loading video preview…",
        'id="renderModeVideo"',
        'id="artifactPreviewHost"',
        'id="viewerPaneMedia"',
        "URL.createObjectURL(blob)",
        # No user-visible Rerun "Loading application bundle" splash.
        'id="rerunBundleCover"',
        "waitUntilRerunPastBundleSplash",
        "Preparing viewer…",
        "Warm Rerun assets before revealing the iframe",
        "Uncover without blocking mount latency",
        "scheduleRerunBundleUncover",
        "safeHideRerunBundleCover",
        "non-blank canvas",
        "swapRerunRecordingInPlace",
        "add_receiver",
        # Describe-this visual feedback (vision tier).
        'id="describeVisual"',
        "captureVisualContext",
        "describeVisual",
        "[npa-visual-feedback]",
        "visual_context",
        "enqueueChatJob",
        "processChatQueue",
        "queueChatText",
        "viewer-focus",
        'id="chatDrawerToggle"',
        "thinking-ellipsis",
        "waitForQualityRerunFrame",
        "captureCanvasDataUrl",
        "ensureRerunCaptureBridge",
        "pickBestIframeCanvas",
        "sampleFrameStats",
        "skipUserAppend",
        "Describe this — capturing",
        "do not prefetch .rrd bytes",
        'id="openFullChatTab"',
        "openFullChatTab",
        'id="chatDrawerClose"',
        "chat-fab",
        "transform-origin: bottom right",
        # Embedded Foxglove viewer pane.
        'id="renderModeFoxglove"',
        'id="viewerPaneFoxglove"',
        'id="foxgloveHost"',
        "ensureFoxgloveViewer",
        "mountFoxgloveViewer",
        "/api/foxglove/config",
        'id="tenantResourcesPanel"',
        "<h3>Tenant resources</h3>",
        'id="tenantResourcesRefresh"',
        "refreshTenantResources",
        "/api/resources",
        "Accessible / discovered",
        "Configured references",
        # Capability-gated LeIsaac tab and authenticated WebRTC bridge.
        LEISAAC_CONTROL_READINESS_CONTRACT,
        "ensureLeIsaacTab",
        "ensureLeIsaacTab(leisaacCapability)",
        "unavailableLeIsaacStatus",
        "removeLeIsaacTab",
        "refreshLeIsaacCapability",
        "connectLeIsaac",
        "/api/leisaac/status",
        "/api/leisaac/select",
        "/api/leisaac/bundles/reset",
    ):
        if marker not in ui_html:
            _fail(f"UI html missing wiring marker: {marker}")
    if 'loading="lazy"' in ui_html:
        _fail("UI html must not use lazy-loading on the Rerun iframe")
    if ".tab-panel[hidden]" in ui_html:
        _fail("UI html must not hide tab panels with display:none via hidden attribute")
    if (
        'Mount the viewer immediately so "Loading application bundle" starts early'
        in ui_html
    ):
        _fail(
            "UI must not mount Rerun before bundle warm (exposes Loading application bundle)"
        )
    if "await waitUntilRerunPastBundleSplash(iframe, 45000)" in ui_html:
        _fail("UI must not block mount on long splash wait (latency)")
    if "await waitUntilRerunPastBundleSplash(iframe, 120000)" in ui_html:
        _fail("UI must not block mount on long splash wait (latency)")
    load_art_src = ui_html.split("async function loadArtifact(payload)")[1].split(
        "async function refresh()"
    )[0]
    if "swapRerunRecordingInPlace" not in load_art_src:
        _fail(
            "loadArtifact must soft-swap Rerun recordings instead of always remounting wasm"
        )
    # Guard against regressions that put bare authenticated URLs on <video src>
    # (browsers omit Authorization headers for media elements under basic auth).
    if (
        '`<video controls src="${previewUrl}">`' in ui_html
        or '<video controls src="${previewUrl}">' in ui_html
    ):
        _fail("UI html must not assign artifact previewUrl directly to <video src>")
    if '`<img alt="artifact image" src="${previewUrl}"' in ui_html:
        _fail("UI html must not assign artifact previewUrl directly to <img src>")

    try:
        session_resp = httpx.get(
            f"{str(record.get('agent_url', '')).rstrip('/')}/api/session",
            auth=(auth_user, auth_password),
            timeout=5.0,
            verify=tls_verify,
        )
        session_resp.raise_for_status()
        session_payload = session_resp.json()
    except Exception as exc:  # noqa: BLE001
        _fail(f"session endpoint failed: {exc}")
    if not isinstance(session_payload, dict):
        _fail("session endpoint did not return JSON object")

    try:
        access_resp = httpx.get(
            f"{str(record.get('agent_url', '')).rstrip('/')}/api/access",
            auth=(auth_user, auth_password),
            timeout=5.0,
            verify=tls_verify,
        )
        access_resp.raise_for_status()
        access_payload = access_resp.json()
    except Exception as exc:  # noqa: BLE001
        _fail(f"agent access endpoint failed: {exc}")
    if (
        not isinstance(access_payload, dict)
        or access_payload.get("apiVersion") != ACCESS_SCHEMA
    ):
        _fail("agent access endpoint returned an invalid schema")
    if access_payload.get("status") not in ACCESS_STATES:
        _fail("agent access endpoint returned an invalid status")
    if not isinstance(access_payload.get("projects"), list):
        _fail("agent access endpoint did not return a projects list")

    try:
        tools_resp = httpx.get(
            f"{str(record.get('agent_url', '')).rstrip('/')}/api/tools",
            auth=(auth_user, auth_password),
            timeout=5.0,
            verify=tls_verify,
        )
        tools_resp.raise_for_status()
        tool_refs = tools_resp.json().get("tool_refs", [])
    except Exception as exc:  # noqa: BLE001
        _fail(f"agent toolRef catalog request failed: {exc}")
    if len(tool_refs) < 19:
        _fail(f"toolRef catalog too small: expected >=19, got {len(tool_refs)}")
    try:
        resolve_resp = httpx.get(
            f"{str(record.get('agent_url', '')).rstrip('/')}/api/tools/{tool_refs[0]}",
            auth=(auth_user, auth_password),
            timeout=5.0,
            verify=tls_verify,
        )
        resolve_resp.raise_for_status()
        resolved = resolve_resp.json()
    except Exception as exc:  # noqa: BLE001
        _fail(f"agent toolRef resolve failed: {exc}")
    if not resolved.get("ok"):
        _fail("agent failed to resolve toolRef catalog entry")
    if not isinstance(resolved.get("argv_template"), list):
        _fail("resolved toolRef entry missing argv_template list")
    try:
        chat_smoke = httpx.post(
            f"{str(record.get('agent_url', '')).rstrip('/')}/api/chat",
            auth=(auth_user, auth_password),
            json={
                "messages": [
                    {"role": "user", "content": "what is the current sim2real status"}
                ]
            },
            timeout=30.0,
            verify=tls_verify,
        )
        chat_smoke.raise_for_status()
        chat_payload = chat_smoke.json()
    except Exception as exc:  # noqa: BLE001
        _fail(f"chat endpoint smoke failed: {exc}")
    if not isinstance(chat_payload, dict) or not chat_payload.get("ok"):
        _fail("chat endpoint did not return ok=true")
    reply = str(chat_payload.get("reply") or "")
    if "run_id" not in reply and "stage" not in reply:
        _fail("chat status reply missing run_id/stage fields")
    if (
        reply.strip().startswith("GET /api")
        or reply.strip() == "GET /api/sim-viz/status"
    ):
        _fail("chat status reply returned raw GET path instead of unpacked status")
    if not chat_payload.get("grounded"):
        _fail("chat status reply expected grounded=true from intent router")
    apis_used = chat_payload.get("apis_used")
    if not isinstance(apis_used, list) or not apis_used:
        _fail("chat status reply expected non-empty apis_used list")

    try:
        wf_chat = httpx.post(
            f"{str(record.get('agent_url', '')).rstrip('/')}/api/chat",
            auth=(auth_user, auth_password),
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": "create 2-step sim2real workflow with 5000 environments, seed 9, an RTX PRO 6000 accelerator, and 1 GPU",
                    }
                ]
            },
            timeout=30.0,
            verify=tls_verify,
        )
        wf_chat.raise_for_status()
        wf_payload = wf_chat.json()
    except Exception as exc:  # noqa: BLE001
        _fail(f"create-workflow chat smoke failed: {exc}")
    if not isinstance(wf_payload, dict) or not wf_payload.get("workflow_yaml"):
        _fail("create-workflow chat did not return workflow_yaml")
    wf_yaml = str(wf_payload.get("workflow_yaml") or "")
    # Assert the chat authored a real npa.workflow spec, not specific stage names:
    # the author's stage naming is not a stable contract (a valid 2-step sim2real
    # spec may use finalize/heldout-eval rather than augment/envgen), and the
    # validate + submit dry-run below already prove the spec is runnable.
    if "apiVersion" not in wf_yaml or "npa.workflow" not in wf_yaml:
        _fail("create-workflow chat yaml is not an npa.workflow spec")
    try:
        wf_validate = httpx.post(
            f"{agent_base}/api/workflows/validate",
            auth=(auth_user, auth_password),
            json={"yaml": wf_yaml},
            timeout=15.0,
            verify=tls_verify,
        )
        wf_validate.raise_for_status()
        wf_val_payload = wf_validate.json()
    except Exception as exc:  # noqa: BLE001
        _fail(f"workflow validate endpoint failed: {exc}")
    if not isinstance(wf_val_payload, dict) or not wf_val_payload.get("ok"):
        _fail("workflow validate endpoint did not return ok=true")
    try:
        infra_resp = httpx.get(
            f"{agent_base}/api/infra/k8s",
            auth=(auth_user, auth_password),
            timeout=15.0,
            verify=tls_verify,
        )
        infra_resp.raise_for_status()
        infra_payload = infra_resp.json()
    except Exception as exc:  # noqa: BLE001
        _fail(f"infra discovery endpoint failed: {exc}")
    if not isinstance(infra_payload, dict) or not infra_payload.get("ok"):
        _fail("infra discovery endpoint did not return ok=true")
    if not infra_payload.get("agent_npa_ready"):
        _fail(f"agent NPA runtime is not ready: {infra_payload.get('agent_npa_error')}")
    try:
        resources_resp = httpx.get(
            f"{agent_base}/api/resources?refresh=true",
            auth=(auth_user, auth_password),
            timeout=60.0,
            verify=tls_verify,
        )
        resources_resp.raise_for_status()
        resources_payload = resources_resp.json()
    except Exception as exc:  # noqa: BLE001
        _fail(f"tenant resource inventory endpoint failed: {exc}")
    try:
        agent_resources.validate_resource_inventory(resources_payload)
    except ValueError as exc:
        _fail(str(exc))
    try:
        wf_submit = httpx.post(
            f"{agent_base}/api/workflows/submit",
            auth=(auth_user, auth_password),
            json={
                "yaml": wf_yaml,
                "run_id": "verify-live-agent-infra",
                "dry_run": True,
                "allow_provision": True,
                "validate_infra": False,
            },
            timeout=120.0,
            verify=tls_verify,
        )
        wf_submit.raise_for_status()
        wf_submit_payload = wf_submit.json()
    except Exception as exc:  # noqa: BLE001
        _fail(f"workflow submit dry-run endpoint failed: {exc}")
    if not isinstance(wf_submit_payload, dict) or not wf_submit_payload.get("ok"):
        _fail("workflow submit dry-run endpoint did not return ok=true")
    if "scheduler_plan" not in wf_submit_payload:
        _fail("workflow submit dry-run missing scheduler_plan")
    if str(wf_submit_payload.get("submit_mode") or "") != "agent-live-infra-dry-run":
        _fail("workflow submit dry-run did not report agent-live-infra-dry-run")

    try:
        onboard_chat = httpx.post(
            f"{str(record.get('agent_url', '')).rstrip('/')}/api/chat",
            auth=(auth_user, auth_password),
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "add an open source repo, containerize, push to registry, and run LeIsaac on live infra"
                        ),
                    }
                ]
            },
            timeout=30.0,
            verify=tls_verify,
        )
        onboard_chat.raise_for_status()
        onboard_payload = onboard_chat.json()
    except Exception as exc:  # noqa: BLE001
        _fail(f"onboard_solution chat smoke failed: {exc}")
    if not isinstance(onboard_payload, dict) or not onboard_payload.get("ok"):
        _fail("onboard_solution chat did not return ok=true")
    onboard_reply = str(onboard_payload.get("reply") or "")
    if (
        "npa workbench byof run" not in onboard_reply
        and "run_byof_repo.py" not in onboard_reply
    ):
        _fail(
            "onboard_solution chat reply missing byof CLI or run_byof_repo.py command"
        )
    if (
        "byof-onboard" not in onboard_reply
        and "skills/workflows/byof-onboard" not in onboard_reply
    ):
        _fail("onboard_solution chat reply missing byof-onboard skill reference")
    if "--base-profile" not in onboard_reply and "--base-image" not in onboard_reply:
        _fail("onboard_solution chat reply missing base image guidance")
    if "<repo-url>" not in onboard_reply:
        _fail("onboard_solution chat reply missing runnable placeholders")
    if onboard_reply.strip().startswith("GET /api"):
        _fail("onboard_solution chat returned raw GET path instead of guidance")
    if not onboard_payload.get("grounded"):
        _fail("onboard_solution chat expected grounded=true")
    onboard_apis = onboard_payload.get("apis_used")
    if not isinstance(onboard_apis, list) or "tools" not in onboard_apis:
        _fail("onboard_solution chat expected tools in apis_used")

    test_env = {
        **dict(os.environ),
        "NPA_INTEGRATION_E2E": "1",
        "NPA_AGENT_LIVE": "1",
        "NPA_AGENT_PROJECT": project,
        "NPA_AGENT_NAME": name,
    }
    if os.environ.get("NPA_AGENT_CHAT_LIVE") == "1":
        test_env["NPA_AGENT_CHAT_LIVE"] = "1"
    # Run the local test gate with the *current* interpreter and the repo root
    # (resolved from this file's location) as cwd, so the command works from any
    # cwd within the source checkout. Previously it shelled out to a relative
    # "npa/.venv/bin/python" and relative test paths, which raised
    # FileNotFoundError unless invoked from the repo root. This gate is a
    # source-tree dev/operator command (the npa/tests/ tree it runs is not
    # shipped in a wheel), so the parents[4] repo-root resolution is expected.
    import sys as _sys

    repo_root = Path(__file__).resolve().parents[4]
    py = _sys.executable or "python3"
    smoke = subprocess.run(
        [
            py,
            "-m",
            "pytest",
            "npa/tests/smoke/test_agent_smoke.py",
            "npa/tests/smoke/test_agent_chat_smoke.py",
            "-q",
        ],
        check=False,
        env=test_env,
        cwd=str(repo_root),
    )
    if smoke.returncode != 0:
        _fail(
            "pytest npa/tests/smoke/test_agent_smoke.py test_agent_chat_smoke.py failed"
        )
    unit = subprocess.run(
        [
            py,
            "-m",
            "pytest",
            "npa/tests/cli/test_agent.py",
            "npa/tests/cli/test_agent_workflow.py",
            "-q",
        ],
        check=False,
        env=test_env,
        cwd=str(repo_root),
    )
    if unit.returncode != 0:
        _fail("pytest npa/tests/cli/test_agent.py failed")
    e2e = subprocess.run(
        [py, "-m", "pytest", "npa/tests/e2e/test_agent_live.py", "-q"],
        check=False,
        env=test_env,
        cwd=str(repo_root),
    )
    if e2e.returncode != 0:
        _fail("pytest npa/tests/e2e/test_agent_live.py failed")
    typer.echo("verify-live: ok")
