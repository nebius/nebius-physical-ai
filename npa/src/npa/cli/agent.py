"""Top-level CLI for deploying and operating the NPA agent VM."""

from __future__ import annotations

import base64
import json
import os
import secrets
import shlex
import shutil
import subprocess
import ipaddress
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import typer

from npa.clients.config import (
    ConfigError,
    resolve_environment,
    resolve_ssh_config,
    resolve_terraform_state,
    write_config,
)
from npa.clients.env import redact_value
from npa.clients.network import (
    NetworkIngressError,
    ensure_ingress,
    remove_ingress_for_instance,
)
from npa.clients.ssh import SSHClient, SSHError
from npa.deploy import provisioner
from npa.deploy.provisioner import ProvisionerError
from npa.orchestration.npa_workflow.catalog import TOOL_CATALOG

app = typer.Typer(
    name="agent",
    help="Deploy and operate a public NPA chat agent VM.",
    no_args_is_help=True,
)

DEFAULT_AGENT_PORT = 8088
DEFAULT_BACKEND_PORT = 8787
DEFAULT_RERUN_PORT = 9090
DEFAULT_PROJECT_ALIAS = "us-central1"
DEFAULT_AGENT_NAME = "agent"
DEFAULT_AGENT_USER = "npa"
DEFAULT_LLM_PROVIDER = "token_factory"
DEFAULT_LLM_MODEL = "nvidia/Cosmos3-Super-Reasoner"
DEFAULT_LLM_MODELS = (
    DEFAULT_LLM_MODEL,
    "meta-llama/Llama-3.3-70B-Instruct",
    "Qwen/Qwen2.5-VL-72B-Instruct",
)
AGENT_UI_VERSION = "2026070403"
DEFAULT_HTTPS_PORT = 443
AGENT_SOURCE_ROOT = "/opt/npa-agent/npa-src"


def _embedded_agent_workflow_source() -> str:
    """Return agent_workflow.py source embedded into the remote agent backend."""
    import re

    path = Path(__file__).with_name("agent_workflow.py")
    raw = path.read_text(encoding="utf-8")
    raw = re.sub(r'^""".*?"""\s*\n', "", raw, count=1, flags=re.DOTALL)
    raw = re.sub(r"^from __future__ import annotations\s*\n", "", raw)
    return raw


def _embedded_agent_chat_source() -> str:
    """Return agent_chat.py source embedded into the remote agent backend."""
    import re

    path = Path(__file__).with_name("agent_chat.py")
    raw = path.read_text(encoding="utf-8")
    raw = re.sub(r'^""".*?"""\s*\n', "", raw, count=1, flags=re.DOTALL)
    raw = re.sub(r"^from __future__ import annotations\s*\n", "", raw)
    return raw


_AGENT_CHAT_EMBED = "__NPA_AGENT_CHAT_EMBED__"
_AGENT_WORKFLOW_EMBED = "__NPA_AGENT_WORKFLOW_EMBED__"
_AGENT_ARTIFACTS_EMBED = "__NPA_AGENT_ARTIFACTS_EMBED__"


def _embedded_agent_artifacts_source() -> str:
    """Return workflows/artifacts.py source embedded into the remote agent backend."""
    import re

    path = Path(__file__).resolve().parents[1] / "workflows" / "artifacts.py"
    raw = path.read_text(encoding="utf-8")
    raw = re.sub(r'^""".*?"""\s*\n', "", raw, count=1, flags=re.DOTALL)
    raw = re.sub(r"^from __future__ import annotations\s*\n", "", raw)
    return raw


@dataclass(frozen=True)
class AgentConfig:
    project_alias: str
    name: str
    project_id: str
    tenant_id: str
    region: str
    public_ip: str
    instance_id: str
    agent_url: str
    rerun_url: str
    sim_viz_url: str
    sim_assets_url: str
    cameras_api_url: str
    auth_user: str
    auth_secret_path: str
    llm_provider: str
    llm_model: str
    service_account_id: str = ""
    llm_models: tuple[str, ...] = ()
    public_url: str = ""
    public_https: bool = True
    direct_url: str = ""
    ssh_key_path: str = ""
    credentials: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "project_id": self.project_id,
            "tenant_id": self.tenant_id,
            "region": self.region,
            "public_ip": self.public_ip,
            "instance_id": self.instance_id,
            "service_account_id": self.service_account_id,
            "agent_url": self.agent_url,
            "rerun_url": self.rerun_url,
            "sim_viz_url": self.sim_viz_url,
            "sim_assets_url": self.sim_assets_url,
            "cameras_api_url": self.cameras_api_url,
            "auth_user": self.auth_user,
            "auth_secret_path": self.auth_secret_path,
            "llm": {
                "provider": self.llm_provider,
                "model": self.llm_model,
                "models": list(self.llm_models or (self.llm_model,)),
            },
        }
        if self.public_url:
            payload["public_url"] = self.public_url
        if self.public_https:
            payload["public_https"] = True
        if self.direct_url:
            payload["direct_url"] = self.direct_url
        if self.ssh_key_path:
            payload["ssh_key_path"] = self.ssh_key_path
        if self.service_account_id:
            payload["service_account_id"] = self.service_account_id
        if self.credentials:
            payload["credentials"] = dict(self.credentials)
        return payload


def build_agent_urls(
    public_ip: str,
    *,
    agent_port: int = DEFAULT_AGENT_PORT,
    public_https: bool = True,
) -> dict[str, str]:
    """Return customer-facing and operator-direct URLs for an agent VM."""
    direct = f"http://{public_ip}:{agent_port}/"
    if public_https:
        base = f"https://{public_ip}/"
    else:
        base = direct
    root = base.rstrip("/")
    return {
        "public_url": base,
        "agent_url": base,
        "rerun_url": f"{root}/rerun/",
        "sim_viz_url": f"{root}/rerun/",
        "sim_assets_url": f"{root}/assets/",
        "cameras_api_url": f"{root}/assets/api/sim-assets/cameras",
        "direct_url": direct,
    }


def _record_public_https(record: dict[str, Any]) -> bool:
    if "public_https" in record:
        return bool(record.get("public_https"))
    public_url = str(record.get("public_url", "")).strip()
    if public_url.startswith("https://"):
        return True
    agent_url = str(record.get("agent_url", "")).strip()
    return agent_url.startswith("https://")


def _record_tls_verify(record: dict[str, Any]) -> bool:
    """Self-signed HTTPS on the VM public IP is expected; skip CA verification."""
    return not _record_public_https(record)


def _record_customer_url(record: dict[str, Any]) -> str:
    public_url = str(record.get("public_url", "")).strip()
    if public_url:
        return public_url
    return str(record.get("agent_url", "")).strip()


def _fail(message: str) -> None:
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(code=1)


def _looks_like_compute_permission_denied(message: str) -> bool:
    lowered = str(message or "").lower()
    return "permissiondenied" in lowered and "service compute" in lowered


def _ensure_terraform_state_bucket(
    *,
    project_id: str,
    bucket_name: str,
) -> None:
    """Ensure the Terraform backend bucket exists before terraform init.

    Fresh deploys may receive reused credentials that point at a bucket deleted
    out-of-band. In that case, recreate the bucket to keep fresh-setup
    provisioning self-healing.
    """
    project = str(project_id or "").strip()
    bucket = str(bucket_name or "").strip()
    if not project or not bucket:
        return
    from npa.clients.nebius import (
        NebiusError,
        bucket_exists,
        ensure_bucket,
    )

    try:
        exists = bucket_exists(project, bucket)
    except NebiusError:
        # Let terraform init surface detailed auth/endpoint errors.
        return
    if exists:
        return
    typer.echo(f"  Terraform state bucket {bucket!r} missing; creating it ...")
    ensure_bucket(project, bucket)


def _apply_agent_terraform(
    *,
    project: str,
    name: str,
    merged_vars: dict[str, str],
    env_region: str,
) -> dict[str, Any]:
    """Apply agent Terraform, retrying without VM SA attachment on compute IAM denial."""
    tf_dir = provisioner.prepare_working_dir(
        project,
        name,
        bucket=merged_vars.get("s3_bucket", ""),
        region=env_region,
        endpoint=merged_vars.get("s3_endpoint", ""),
    )
    provisioner.init(
        tf_dir=tf_dir,
        backend_config={
            "access_key": merged_vars.get("nebius_api_key", ""),
            "secret_key": merged_vars.get("nebius_secret_key", ""),
        },
    )
    try:
        return provisioner.apply(tf_dir=tf_dir, tf_vars=merged_vars)
    except ProvisionerError as exc:
        sa_id = str(merged_vars.get("service_account_id", "")).strip()
        if sa_id and _looks_like_compute_permission_denied(str(exc)):
            typer.echo(
                "  Compute create denied with VM service-account attachment; "
                "retrying without attached service_account_id ..."
            )
            retry_vars = dict(merged_vars)
            retry_vars["service_account_id"] = ""
            return provisioner.apply(tf_dir=tf_dir, tf_vars=retry_vars)
        raise


def _agent_record(project_alias: str, name: str) -> dict[str, Any]:
    cfg = resolve_project_agents(project_alias)
    record = cfg.get(name, {})
    return record if isinstance(record, dict) else {}


def resolve_project_agents(project_alias: str) -> dict[str, Any]:
    from npa.clients.config import list_projects

    projects = list_projects()
    project = projects.get(project_alias, {})
    agents = project.get("agents", {}) if isinstance(project, dict) else {}
    return agents if isinstance(agents, dict) else {}


def _store_agent_record(project_alias: str, name: str, payload: dict[str, Any]) -> None:
    write_config({"projects": {project_alias: {"agents": {name: payload}}}})


def _remove_agent_record(project_alias: str, name: str) -> None:
    from npa.clients.config import CONFIG_PATH, _load_yaml
    import yaml

    data = _load_yaml()
    projects = data.get("projects", {})
    if not isinstance(projects, dict):
        return
    project = projects.get(project_alias, {})
    if not isinstance(project, dict):
        return
    agents = project.get("agents", {})
    if not isinstance(agents, dict) or name not in agents:
        return
    del agents[name]
    project["agents"] = agents
    projects[project_alias] = project
    data["projects"] = projects
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CONFIG_PATH.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, default_flow_style=False, sort_keys=False)
    CONFIG_PATH.chmod(0o600)


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


def _agent_terraform_state_exists(project: str, name: str) -> bool:
    tf_dir = provisioner.working_dir_path(project, name)
    return (tf_dir / ".terraform").is_dir()


def _resolve_destroy_tf_vars(
    project: str,
    name: str,
    record: dict[str, Any] | None,
) -> dict[str, str]:
    state = resolve_terraform_state(project)
    saved_env = resolve_environment(project)
    region = str((record or {}).get("region", "") or (saved_env.region if saved_env else "") or "us-central1")
    project_id = str((record or {}).get("project_id", "") or (saved_env.project_id if saved_env else ""))
    service_account_id = str((record or {}).get("service_account_id", "")).strip()
    if not service_account_id:
        creds = (record or {}).get("credentials", {})
        if isinstance(creds, dict):
            service_account_id = str(creds.get("service_account_id", "")).strip()
    if not service_account_id:
        service_account_id = _resolve_agent_service_account_id(project, record or {})
    from npa.clients.nebius import get_iam_token

    iam_token = get_iam_token()
    return {
        "nebius_project_id": project_id,
        "nebius_region": region,
        "service_account_id": service_account_id,
        "iam_token": iam_token,
        "instance_name": f"agent-{project}-{name}",
        "server_port": str(DEFAULT_AGENT_PORT),
        "workbench_type": "lerobot",
        "gpu_platform": "cpu-d3",
        "gpu_preset": "8vcpu-32gb",
        "enable_preemptible": "false",
        "nebius_api_key": state.access_key,
        "nebius_secret_key": state.secret_key,
        "s3_bucket": state.bucket,
        "s3_endpoint": state.endpoint,
        "extra_ingress_ports": "[]",
    }


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


_AGENT_INSTANCE_DESTROY_TARGETS = (
    "null_resource.wait_for_cloud_init",
    "nebius_compute_v1_instance.workbench",
)


def _cleanup_orphan_agent_instances(project_id: str, instance_name: str) -> None:
    """Delete cloud VM instances matching the agent name but missing from TF state."""
    project_id = str(project_id or "").strip()
    instance_name = str(instance_name or "").strip()
    if not project_id or not instance_name:
        return
    from npa.clients.nebius import NebiusError, _run, _run_json

    try:
        payload = _run_json(["compute", "instance", "list", "--parent-id", project_id])
    except NebiusError:
        return
    items = payload.get("items", [])
    if not isinstance(items, list):
        return
    for item in items:
        if not isinstance(item, dict):
            continue
        meta = item.get("metadata", {})
        if not isinstance(meta, dict):
            continue
        if str(meta.get("name", "")).strip() != instance_name:
            continue
        instance_id = str(meta.get("id", "")).strip()
        if not instance_id:
            continue
        try:
            _run(["compute", "instance", "delete", instance_id], check=False)
            typer.echo(f"  Deleted orphan agent instance {instance_id}")
        except NebiusError:
            continue


def _destroy_agent_terraform(
    project: str,
    name: str,
    *,
    record: dict[str, Any] | None = None,
) -> None:
    """Destroy the agent Terraform stack and optional npa-managed ingress rules."""
    if not _agent_terraform_state_exists(project, name):
        return
    state = resolve_terraform_state(project)
    if not state.bucket or not state.access_key or not state.secret_key:
        _fail(
            f"Terraform state backend is not configured for project {project!r}. "
            "Run `npa configure` or redeploy once to persist terraform_state."
        )
    tf_vars = _resolve_destroy_tf_vars(project, name, record)
    region = tf_vars["nebius_region"]
    instance_id = str((record or {}).get("instance_id", "")).strip()
    instance_name = tf_vars["instance_name"]
    project_id = tf_vars["nebius_project_id"]
    _cleanup_agent_ingress(instance_id)
    _cleanup_orphan_agent_instances(project_id, instance_name)
    tf_dir = provisioner.prepare_working_dir(
        project,
        name,
        bucket=state.bucket,
        region=region,
        endpoint=state.endpoint,
    )
    provisioner.init(
        tf_dir=tf_dir,
        backend_config={"access_key": state.access_key, "secret_key": state.secret_key},
    )

    def _run_destroy() -> None:
        provisioner.destroy(tf_dir=tf_dir, tf_vars=tf_vars)

    def _destroy_compute_first() -> None:
        managed = set(provisioner.state_list(tf_dir))
        targets = [t for t in _AGENT_INSTANCE_DESTROY_TARGETS if t in managed]
        if targets:
            provisioner.destroy(tf_dir=tf_dir, tf_vars=tf_vars, targets=targets)

    try:
        _run_destroy()
    except ProvisionerError as first_exc:
        _cleanup_agent_ingress(instance_id)
        try:
            _destroy_compute_first()
            _run_destroy()
        except ProvisionerError:
            raise first_exc from None


def _auth_secret_path(project_alias: str, name: str) -> Path:
    return Path.home() / ".npa" / "agents" / project_alias / name / "auth.env"


def _write_auth_secret(*, project_alias: str, name: str, user: str, password: str) -> Path:
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
        raise ValueError(f"auth secret missing AGENT_USER/AGENT_PASSWORD: {secret_path}")
    return user, password


def _tool_catalog_keys() -> list[str]:
    return sorted(TOOL_CATALOG.keys())


def _tool_catalog_payload() -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {}
    for key in _tool_catalog_keys():
        entry = TOOL_CATALOG[key]
        payload[key] = {
            "description": entry.description,
            "argv_template": list(entry.argv_template),
        }
    return payload


def _resolve_deploy_llm_credentials() -> tuple[str, str]:
    """Return Token Factory API key and default model for agent VM bootstrap."""
    from npa.clients.credentials import load_credentials

    creds = load_credentials()
    return creds.token_factory_api_key, DEFAULT_LLM_MODEL


def _resolve_operator_credentials() -> tuple[str, str]:
    """Return Nebius AI Cloud key and Token Factory API key from operator credentials."""
    from npa.clients.credentials import load_credentials

    creds = load_credentials()
    return creds.ai_cloud_api_key, creds.token_factory_api_key


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


def _agent_credentials_payload(creds: dict[str, str]) -> dict[str, str]:
    """Normalize Nebius bootstrap output for persistence on the agent record."""
    return {
        "service_account_id": str(creds.get("service_account_id", "")).strip(),
        "s3_bucket": str(creds.get("s3_bucket", "")).strip(),
        "s3_prefix": str(creds.get("s3_prefix", "")).strip().strip("/"),
        "s3_endpoint": str(creds.get("s3_endpoint", "")).strip(),
        "access_key": str(creds.get("nebius_api_key", "")).strip(),
        "secret_key": str(creds.get("nebius_secret_key", "")).strip(),
    }


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
        endpoint_url = f"https://storage.{str(region or '').strip() or 'eu-north1'}.nebius.cloud"
    try:
        import boto3
    except Exception:
        return False
    client_kwargs = {
        "endpoint_url": endpoint_url,
        "aws_access_key_id": str(access_key or "").strip(),
        "region_name": str(region or "").strip() or None,
        "aws_" "secret_access_key": str(secret_key or "").strip(),
    }
    client = boto3.client("s3", **client_kwargs)
    normalized_prefix = str(prefix or "").strip().strip("/")
    probe_base = "/".join(part for part in (normalized_prefix, "npa-agent/probe") if part)
    probe_key = f"{probe_base}/{secrets.token_hex(8)}.txt"
    try:
        client.list_objects_v2(Bucket=bucket_name, Prefix=(probe_base + "/") if probe_base else "", MaxKeys=1)
        client.put_object(Bucket=bucket_name, Key=probe_key, Body=b"ok")
        client.delete_object(Bucket=bucket_name, Key=probe_key)
        return True
    except Exception:
        return False


def _resolve_deploy_storage_credentials(
    *,
    region: str,
    bootstrap_creds: dict[str, str],
) -> dict[str, str]:
    """Prefer configured artifact storage keys; fall back to bootstrap keys when needed."""
    candidate = dict(bootstrap_creds)
    from npa.clients.credentials import load_credentials

    shared = load_credentials(environ={})
    shared_bucket = str(shared.s3_bucket or "").strip()
    shared_prefix = ""
    if shared_bucket.startswith("s3://"):
        rest = shared_bucket[len("s3://"):]
        shared_bucket, _sep, shared_prefix = rest.partition("/")
        shared_prefix = shared_prefix.strip("/")
    shared_endpoint = str(shared.s3_endpoint or f"https://storage.{region}.nebius.cloud").strip()
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
        typer.echo("  Using shared configured artifact storage credentials for the agent.")
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
    typer.echo(
        "  Warning: unable to verify writable S3 credentials for deploy; "
        "continuing with bootstrap-provided keys.",
        err=True,
    )
    return candidate


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


def _persist_agent_service_account_id(service_account_id: str) -> None:
    """Write discovered SA id into ~/.npa/credentials.yaml when missing."""
    sa_id = str(service_account_id or "").strip()
    if not sa_id:
        return
    from npa.clients.credentials import write_credentials_file
    from npa.clients.nebius import _saved_service_account_id

    if _saved_service_account_id() == sa_id:
        return
    write_credentials_file({"nebius": {"service_account_id": sa_id}})


def _creds_from_terraform_state(project_alias: str, record: dict[str, Any]) -> dict[str, str] | None:
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


def _credentials_block_from_storage(
    *,
    service_account_id: str,
    s3_bucket: str,
    s3_prefix: str = "",
    s3_endpoint: str,
    s3_access_key: str,
    s3_secret_key: str,
) -> dict[str, str]:
    return {
        "service_account_id": service_account_id.strip(),
        "s3_bucket": s3_bucket.strip(),
        "s3_prefix": s3_prefix.strip().strip("/"),
        "s3_endpoint": s3_endpoint.strip(),
        "access_key": s3_access_key.strip(),
        "secret_key": s3_secret_key.strip(),
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
                service_account_id = _resolve_agent_service_account_id(project_alias, record)
            return bucket, prefix, endpoint, access_key, secret_key, service_account_id
    try:
        tf_state = resolve_terraform_state(project_alias)
    except ConfigError:
        return "", "", "", "", "", _resolve_agent_service_account_id(project_alias, record)
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
        _normalize_llm_models([str(item) for item in llm_providers if str(item).strip()])
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


def _write_agent_s3_env(
    ssh: SSHClient,
    *,
    bucket: str,
    prefix: str = "",
    endpoint: str,
    access_key: str,
    secret_key: str,
    region: str,
) -> None:
    """Stage S3 discovery credentials on the VM (read-only operator scope preferred)."""
    if not (bucket.strip() and access_key.strip() and secret_key.strip()):
        return
    env_lines = [
        f"NPA_AGENT_S3_BUCKET={bucket.strip()}",
        f"NPA_AGENT_S3_PREFIX={prefix.strip().strip('/')}",
        f"NPA_AGENT_S3_ENDPOINT={endpoint.strip()}",
        f"AWS_ACCESS_KEY_ID={access_key.strip()}",
        f"AWS_SECRET_ACCESS_KEY={secret_key.strip()}",
        f"AWS_REGION={region.strip() or 'eu-north1'}",
        "",
    ]
    env_b64 = base64.b64encode("\n".join(env_lines).encode("utf-8")).decode("ascii")
    ssh.run_or_raise(
        f"echo {shlex.quote(env_b64)} | base64 -d | sudo tee /opt/npa-agent/s3.env >/dev/null "
        "&& sudo chmod 600 /opt/npa-agent/s3.env"
    )


def _write_agent_operator_profile(
    ssh: SSHClient,
    *,
    ssh_user: str,
    project_alias: str,
    project_id: str,
    tenant_id: str,
    region: str,
    tf_api_key: str,
    nebius_ai_key: str,
    s3_bucket: str,
    s3_prefix: str = "",
    s3_endpoint: str,
    s3_access_key: str,
    s3_secret_key: str,
    service_account_id: str = "",
) -> None:
    """Write ~/.npa/config.yaml + credentials.yaml on the agent VM for operator workflows."""
    if not (project_alias and project_id and tenant_id and region):
        return
    config_payload: dict[str, Any] = {
        "default_project": project_alias,
        "projects": {
            project_alias: {
                "project_id": project_id,
                "tenant_id": tenant_id,
                "region": region,
            }
        },
    }
    credentials_payload: dict[str, Any] = {"tokens": {}}
    tokens = credentials_payload["tokens"]
    if isinstance(tokens, dict):
        if nebius_ai_key.strip():
            tokens["NEBIUS_AI_CLOUD_KEY"] = nebius_ai_key.strip()
        if tf_api_key.strip():
            tokens["NEBIUS_TOKEN_FACTORY_KEY"] = tf_api_key.strip()
    storage_payload = {
        "access_key_id": s3_access_key.strip(),
        "secret_access_key": s3_secret_key.strip(),
        "endpoint": s3_endpoint.strip(),
        "bucket": "s3://" + s3_bucket.strip() + (("/" + s3_prefix.strip().strip("/") + "/") if s3_prefix.strip().strip("/") else ""),
    }
    if any(storage_payload.values()):
        credentials_payload["storage"] = storage_payload
    if service_account_id.strip():
        credentials_payload["nebius"] = {"service_account_id": service_account_id.strip()}
    config_b64 = base64.b64encode(json.dumps(config_payload, indent=2).encode("utf-8")).decode("ascii")
    creds_b64 = base64.b64encode(json.dumps(credentials_payload, indent=2).encode("utf-8")).decode("ascii")
    user_home = f"/home/{ssh_user}"
    targets = [
        (f"{user_home}/.npa", f"{ssh_user}:{ssh_user}"),
        ("/root/.npa", "root:root"),
    ]
    commands: list[str] = []
    for npa_dir, owner in targets:
        config_path = f"{npa_dir}/config.yaml"
        creds_path = f"{npa_dir}/credentials.yaml"
        commands.extend(
            [
                f"sudo mkdir -p {shlex.quote(npa_dir)}",
                f"echo {shlex.quote(config_b64)} | base64 -d | sudo tee {shlex.quote(config_path)} >/dev/null",
                f"echo {shlex.quote(creds_b64)} | base64 -d | sudo tee {shlex.quote(creds_path)} >/dev/null",
                f"sudo chown -R {shlex.quote(owner)} {shlex.quote(npa_dir)}",
                f"sudo chmod 700 {shlex.quote(npa_dir)}",
                f"sudo chmod 600 {shlex.quote(config_path)} {shlex.quote(creds_path)}",
            ]
        )
    ssh.run_or_raise(" && ".join(commands))


def _store_project_environment(*, project: str, project_id: str, tenant_id: str, region: str) -> None:
    """Persist a project-scoped Nebius environment like a fresh configure step."""
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


def _write_agent_nebius_env(
    ssh: SSHClient,
    *,
    project_alias: str,
    agent_name: str,
    project_id: str,
    tenant_id: str,
    region: str,
    service_account_id: str,
    bucket: str,
    endpoint: str,
    access_key: str,
    secret_key: str,
    iam_token: str = "",
) -> None:
    """Stage long-lived Nebius project credentials on the agent VM."""
    if not (project_id.strip() and access_key.strip() and secret_key.strip()):
        return
    env_lines = [
        f"NPA_AGENT_PROJECT_ALIAS={project_alias.strip()}",
        f"NPA_AGENT_NAME={agent_name.strip()}",
        f"NEBIUS_PROJECT_ID={project_id.strip()}",
        f"NEBIUS_TENANT_ID={tenant_id.strip()}",
        f"NEBIUS_REGION={region.strip() or 'eu-north1'}",
        f"NEBIUS_SERVICE_ACCOUNT_ID={service_account_id.strip()}",
        f"NEBIUS_S3_BUCKET={bucket.strip()}",
        f"NEBIUS_S3_ENDPOINT={endpoint.strip()}",
        f"AWS_ACCESS_KEY_ID={access_key.strip()}",
        f"AWS_SECRET_ACCESS_KEY={secret_key.strip()}",
        f"AWS_REGION={region.strip() or 'eu-north1'}",
    ]
    if iam_token.strip():
        env_lines.extend(
            [
                "NEBIUS_PROFILE=agent-bootstrap",
                f"NEBIUS_IAM_TOKEN={iam_token.strip()}",
                f"NPA_NEBIUS_IAM_TOKEN={iam_token.strip()}",
                f"TF_VAR_iam_token={iam_token.strip()}",
                "NPA_REUSE_IAM_TOKEN=1",
            ]
        )
    env_lines.append("")
    env_b64 = base64.b64encode("\n".join(env_lines).encode("utf-8")).decode("ascii")
    ssh.run_or_raise(
        f"echo {shlex.quote(env_b64)} | base64 -d | sudo tee /opt/npa-agent/nebius.env >/dev/null "
        "&& sudo chmod 600 /opt/npa-agent/nebius.env"
    )
    if iam_token.strip():
        ssh.run_or_raise(
            "sudo bash -lc "
            + shlex.quote(
                "\n".join(
                    [
                        "set -euo pipefail",
                        "set -a",
                        ". /opt/npa-agent/nebius.env",
                        "set +a",
                        "mkdir -p /root/.npa",
                        "printf '%s' \"$NEBIUS_IAM_TOKEN\" > /root/.npa/nebius-token",
                        "chmod 600 /root/.npa/nebius-token",
                        "NEBIUS_BIN=\"$(command -v nebius || true)\"",
                        "if [ -z \"$NEBIUS_BIN\" ] && [ -x /usr/local/bin/nebius ]; then NEBIUS_BIN=/usr/local/bin/nebius; fi",
                        "if [ -n \"$NEBIUS_BIN\" ]; then",
                        "  \"$NEBIUS_BIN\" profile create --endpoint api.eu.nebius.cloud --token-file /root/.npa/nebius-token --profile agent-bootstrap --parent-id \"$NEBIUS_PROJECT_ID\" >/dev/null 2>&1 || true",
                        "  NEBIUS_PROFILE=agent-bootstrap \"$NEBIUS_BIN\" iam get-access-token >/dev/null",
                        "fi",
                    ]
                )
            )
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
        archive.add(repo_root / "deploy" / "cluster", arcname="deploy/cluster", filter=_filter)
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


def _agent_public_login_form_html(auth_user: str) -> str:
    """Shared Sign in form for public welcome/login-help pages (mobile-safe basic auth)."""
    return f"""    <section class="sign-in-panel" aria-labelledby="sign-in-heading">
      <h2 id="sign-in-heading">Sign in</h2>
      <p class="muted">Use the form if your browser does not show an HTTP Basic Auth dialog.</p>
      <form id="npa-sign-in" class="sign-in" autocomplete="on">
        <label for="npa-user">Username</label>
        <input id="npa-user" name="username" type="text" value="{auth_user}" autocomplete="username" required>
        <label for="npa-pass">Password</label>
        <input id="npa-pass" name="password" type="password" autocomplete="current-password" required>
        <button type="submit" id="npa-sign-in-btn">Sign in</button>
        <p id="npa-sign-in-status" class="muted" role="status" aria-live="polite"></p>
      </form>
      <p class="muted note">Credentials are not left in the address bar after sign-in.</p>
    </section>
    <script>
    (function () {{
      try {{
        if (location.username || location.password) {{
          const clean = location.protocol + "//" + location.host + location.pathname + location.search + location.hash;
          history.replaceState(null, "", clean);
        }}
      }} catch (_err) {{ /* best-effort */ }}
      var form = document.getElementById("npa-sign-in");
      var statusEl = document.getElementById("npa-sign-in-status");
      var btn = document.getElementById("npa-sign-in-btn");
      if (!form) return;

      function setStatus(msg, isError) {{
        if (!statusEl) return;
        statusEl.textContent = msg || "";
        statusEl.style.color = isError ? "#991b1b" : "#5f6573";
      }}

      function isMobileUa() {{
        return /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent || "");
      }}

      function destPath() {{
        var rawPath = String(location.pathname || "/");
        var normalizedPath = rawPath.length > 1 && rawPath.endsWith("/") ? rawPath.slice(0, -1) : rawPath;
        return (normalizedPath === "/login-help.html" || normalizedPath === "/welcome") ? "/" : normalizedPath;
      }}

      function basicAuthHeader(user, pass) {{
        return "Basic " + btoa(unescape(encodeURIComponent(user + ":" + pass)));
      }}

      function persistBasicAuth(user, pass) {{
        try {{
          sessionStorage.setItem("npa_agent_basic_auth", basicAuthHeader(user, pass));
        }} catch (_err) {{ /* sessionStorage may be unavailable */ }}
      }}

      function xhrSignIn(user, pass, dest) {{
        return new Promise(function (resolve, reject) {{
          var xhr = new XMLHttpRequest();
          xhr.open("GET", dest, true, user, pass);
          xhr.onload = function () {{
            if (xhr.status >= 200 && xhr.status < 400) {{
              resolve();
              return;
            }}
            if (xhr.status === 401) {{
              reject(new Error("Invalid username or password."));
              return;
            }}
            reject(new Error("Sign-in failed (HTTP " + xhr.status + ")."));
          }};
          xhr.onerror = function () {{
            reject(new Error("Network error — open /healthz first and accept the certificate warning."));
          }};
          xhr.send();
        }});
      }}

      function fetchSignIn(user, pass, dest) {{
        return fetch(dest, {{
          method: "GET",
          headers: {{ "Authorization": basicAuthHeader(user, pass) }},
          credentials: "omit",
          cache: "no-store",
        }}).then(function (resp) {{
          if (!resp.ok) {{
            throw new Error(resp.status === 401 ? "Invalid username or password." : "Sign-in failed (HTTP " + resp.status + ").");
          }}
        }});
      }}

      function urlEmbedSignIn(user, pass, dest) {{
        var u = encodeURIComponent(user);
        var p = encodeURIComponent(pass);
        location.href = location.protocol + "//" + u + ":" + p + "@" + location.host + dest;
      }}

      form.addEventListener("submit", function (ev) {{
        ev.preventDefault();
        var user = document.getElementById("npa-user").value;
        var pass = document.getElementById("npa-pass").value;
        var dest = destPath();
        setStatus("Signing in…", false);
        if (btn) btn.disabled = true;

        xhrSignIn(user, pass, dest)
          .catch(function () {{ return fetchSignIn(user, pass, dest); }})
          .then(function () {{
            persistBasicAuth(user, pass);
            window.location.href = dest;
          }})
          .catch(function (err) {{
            if (!isMobileUa()) {{
              persistBasicAuth(user, pass);
              urlEmbedSignIn(user, pass, dest);
              return;
            }}
            setStatus((err && err.message) ? err.message : "Sign-in failed on this device.", true);
            if (btn) btn.disabled = false;
          }});
      }});
    }})();
    </script>"""


def _nginx_agent_site_body(
    *,
    backend_port: int,
    rerun_port: int,
) -> str:
    """Shared nginx locations for the agent UI (HTTP and HTTPS server blocks)."""
    return f"""  auth_basic "NPA Agent";
  auth_basic_user_file /etc/nginx/.npa-agent-htpasswd;
  location = /healthz {{
    auth_basic off;
    default_type application/json;
    return 200 '{{"ok":true,"service":"npa-agent","welcome":"/welcome","ui":"/","ui_version":"{AGENT_UI_VERSION}"}}';
  }}
  location = /welcome {{
    auth_basic off;
    alias /opt/npa-agent/welcome.html;
    default_type text/html;
    add_header Cache-Control "no-store" always;
  }}
  location = /login-help.html {{
    auth_basic off;
    alias /opt/npa-agent/login-help.html;
    default_type text/html;
    add_header Cache-Control "no-store" always;
  }}
  location /api/ {{
    proxy_pass http://127.0.0.1:{backend_port}/;
  }}
  location /assets/api/ {{
    rewrite ^/assets/api/(.*)$ /$1 break;
    proxy_pass http://127.0.0.1:{backend_port}/;
  }}
  location /rerun/recordings/ {{
    auth_basic off;
    alias /opt/npa-agent/recordings/;
    default_type application/octet-stream;
    add_header Cache-Control "no-cache" always;
  }}
  location ~* ^/rerun/.+\\.(wasm|js|ico|svg)$ {{
    rewrite ^/rerun/(.*)$ /$1 break;
    proxy_pass http://127.0.0.1:{rerun_port};
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_connect_timeout 30s;
    proxy_read_timeout 300s;
    proxy_send_timeout 300s;
    gzip on;
    gzip_types application/wasm application/javascript text/javascript image/svg+xml;
    gzip_min_length 256;
    add_header Cache-Control "public, max-age=31536000, immutable" always;
  }}
  location /rerun/ {{
    proxy_pass http://127.0.0.1:{rerun_port}/;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_connect_timeout 30s;
    proxy_read_timeout 300s;
    proxy_send_timeout 300s;
    add_header Cache-Control "public, max-age=3600" always;
  }}
  location / {{
    root /opt/npa-agent;
    index ui.html;
    try_files /ui.html =404;
    add_header Cache-Control "no-store, no-cache, must-revalidate" always;
    add_header Pragma "no-cache" always;
  }}"""


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
) -> None:
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
    agent_workflow_source = _embedded_agent_workflow_source()
    agent_artifacts_source = _embedded_agent_artifacts_source()
    llm_models = _normalize_llm_models(list(llm_models))
    default_llm_models_json = json.dumps(llm_models)
    nginx_site_body = _nginx_agent_site_body(backend_port=backend_port, rerun_port=rerun_port)
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
    setup_script = f"""set -euo pipefail
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y nginx apache2-utils python3-venv python3-pip curl unzip ca-certificates
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
fi
sudo mkdir -p /opt/npa-agent
cat <<'PY' | sudo tee /opt/npa-agent/backend.py >/dev/null
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

import httpx
import yaml
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse, JSONResponse

app = FastAPI(title="npa-agent")
TOOL_CATALOG = {catalog_json}
TOOL_REFS = sorted(TOOL_CATALOG.keys())
STATE_PATH = Path("/opt/npa-agent/session_state.json")
RRD_PATH = Path("/opt/npa-agent/sim2real.rrd")
RECORDING_PATH = Path("/opt/npa-agent/recordings/sim2real.rrd")
RECORDINGS_DIR = Path("/opt/npa-agent/recordings")
RERUN_UNIT = "npa-rerun"
RERUN_WEB_PORT = {rerun_port}
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
    prefix = settings.get("prefix", "npa-agent/session-state")
    return f"{{prefix}}/{{project_alias}}/{{agent_name}}/{{session_scope}}.json"

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
    return {{
        "selection": dict(DEFAULT_SELECTION),
        "camera_selection": ["workspace"],
        "sim_viz": dict(DEFAULT_SIM_VIZ),
        "sim_viz_runs": {{}},
        "sim2real_runs": {{}},
        "active_run_id": "",
        "latest_submit": {{}},
        "workflow_draft": {{"yaml": "", "name": "", "states": [], "updated_at": "", "plan": {{}}, "runnable": False}},
        "workflow_submit": {{}},
        "chat_history": [],
        "active_chat_session_id": "default",
        "chat_sessions": {{}},
        "session_scope": session_scope,
        "agent_scope": {{"project_alias": project_alias, "name": agent_name}},
        "state_version": 2,
    }}

def _load_state() -> dict:
    data = None
    if STATE_PATH.exists():
        try:
            payload = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                data = payload
        except Exception:
            data = None
    if data is None:
        data = _load_state_from_s3()
    if not isinstance(data, dict):
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
    if not isinstance(merged.get("active_chat_session_id"), str):
        merged["active_chat_session_id"] = "default"
    return merged

def _save_state(state: dict) -> None:
    state["updated_at"] = _now_iso()
    state["state_version"] = int(state.get("state_version") or 2)
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
    _save_state_to_s3(state)


def _record_sim_viz_run(state: dict, payload: dict | None) -> None:
    if not isinstance(payload, dict):
        return
    run_id = str(payload.get("run_id") or "").strip()
    if not run_id:
        return
    runs = state.get("sim_viz_runs")
    if not isinstance(runs, dict):
        runs = {{}}
    snapshot = dict(DEFAULT_SIM_VIZ)
    snapshot.update(payload)
    snapshot["run_id"] = run_id
    runs[run_id] = snapshot
    state["sim_viz_runs"] = runs
    state["active_run_id"] = run_id


def _default_sim2real_run_details(run_id: str, *, submitted_at: str = "", selection: dict | None = None) -> dict:
    stages = []
    for index, (stage_id, label) in enumerate(SIM2REAL_STAGE_TEMPLATE):
        stages.append(
            {{
                "id": stage_id,
                "label": label,
                "status": "not_run",
                "started_at": "",
                "finished_at": "",
                "summary": "Not launched by the agent UI submit endpoint.",
            }}
        )
    if stages:
        stages[0]["status"] = "succeeded"
        stages[0]["started_at"] = submitted_at
        stages[0]["finished_at"] = submitted_at
        stages[0]["summary"] = "Agent accepted the Sim2Real submit request."
    return {{
        "run_id": run_id,
        "status": "submitted",
        "result": "recorded_not_launched",
        "submitted_at": submitted_at,
        "updated_at": submitted_at or _now_iso(),
        "selection": selection if isinstance(selection, dict) else {{}},
        "stages": stages,
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


def _sim2real_run_details(state: dict, run_id: str = "") -> dict:
    latest = state.get("latest_submit", {{}})
    if not isinstance(latest, dict):
        latest = {{}}
    sim_viz = state.get("sim_viz", {{}})
    if not isinstance(sim_viz, dict):
        sim_viz = {{}}
    resolved_run_id = str(run_id or latest.get("run_id") or sim_viz.get("run_id") or state.get("active_run_id") or "").strip()
    details_map = state.get("sim2real_runs")
    if not isinstance(details_map, dict):
        details_map = {{}}
    existing = details_map.get(resolved_run_id, {{}}) if resolved_run_id else {{}}
    submitted_at = str(latest.get("submitted_at") or sim_viz.get("rrd_updated_at") or "")
    selection = latest.get("selection") if isinstance(latest.get("selection"), dict) else {{}}
    details = _default_sim2real_run_details(resolved_run_id, submitted_at=submitted_at, selection=selection)
    details = _merge_sim2real_run_details(details, existing if isinstance(existing, dict) else {{}})
    stage = str(sim_viz.get("stage") or details.get("status") or "submitted").strip()
    if stage:
        details["status"] = stage
    if sim_viz.get("rrd_uri"):
        if str(details.get("result") or "") not in {"completed", "failed", "running"}:
            details["result"] = "recording_available"
        for item in details.get("stages", []):
            if isinstance(item, dict) and item.get("id") == "stage_14_rerun_viz":
                item["status"] = "succeeded"
                item["summary"] = "Rerun recording is available."
    elif resolved_run_id:
        details["result"] = "recorded_not_launched"
    return details


def _sim_viz_for_run(state: dict, run_id: str = "") -> dict:
    payload = dict(DEFAULT_SIM_VIZ)
    current = state.get("sim_viz")
    if isinstance(current, dict):
        payload.update(current)
    runs = state.get("sim_viz_runs")
    target = str(run_id or state.get("active_run_id") or "").strip()
    if isinstance(runs, dict) and target and isinstance(runs.get(target), dict):
        payload.update(runs[target])
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
    _publish_rrd_recording(target)
    return target

def _publish_rrd_recording(source: Path) -> Path:
    if not source.is_file():
        return RECORDING_PATH
    RECORDING_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = RECORDING_PATH.with_suffix(".rrd.tmp")
    shutil.copy2(source, tmp)
    tmp.replace(RECORDING_PATH)
    return RECORDING_PATH


def _safe_artifact_key(key: str) -> str:
    value = str(key or "").strip().lstrip("/")
    if not value:
        raise HTTPException(status_code=400, detail="artifact key is required")
    parts = value.split("/")
    if any(part in {{"", ".", ".."}} for part in parts):
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


def _chat_memory_tenant() -> str:
    raw = (
        os.environ.get("NEBIUS_TENANT_ID", "")
        or os.environ.get("NEBIUS_PROJECT_ID", "")
        or "default-tenant"
    )
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(raw).strip()).strip("-")
    return value or "default-tenant"


def _chat_memory_prefix(settings: dict[str, str] | None = None) -> str:
    bucket = (settings or {{}}).get("bucket", "")
    tenant = _chat_memory_tenant()
    return f"npa-agent/tenants/{{tenant}}/chat-sessions"


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
    history: list[dict] = []
    if not isinstance(raw, list):
        return history
    for item in raw:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip()
        content = str(item.get("content") or "").strip()
        if role in {{"user", "assistant"}} and content:
            history.append({{"role": role, "content": content}})
    return history[-80:]


def _normalize_chat_session(session_id: str, payload: object | None = None) -> dict:
    now = _now_iso()
    data = payload if isinstance(payload, dict) else {{}}
    resolved_id = _sanitize_chat_session_id(str(data.get("id") or session_id or "default"))
    history = _normalize_chat_history(data.get("chat_history") or data.get("messages") or [])
    title = str(data.get("title") or "").strip() or _chat_session_title(history, "New chat")
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
    for session_id, payload in sessions.items():
        session = _normalize_chat_session(str(session_id), payload)
        normalized[session["id"]] = session
    if not normalized:
        migrated = _normalize_chat_session(
            "default",
            {{
                "id": "default",
                "title": "Default chat",
                "chat_history": state.get("chat_history", []),
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


def _get_chat_session(state: dict, session_id: str = "") -> dict:
    sessions = _local_chat_sessions(state)
    target = _sanitize_chat_session_id(session_id or str(state.get("active_chat_session_id") or "default"))
    remote = _load_chat_session_from_s3(target)
    if remote is not None:
        sessions[target] = remote
        state["chat_sessions"] = sessions
        return remote
    if target in sessions:
        return sessions[target]
    session = _normalize_chat_session(target, {{"id": target, "title": "New chat", "chat_history": []}})
    sessions[target] = session
    state["chat_sessions"] = sessions
    return session


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
    _save_state(state)
    rows = []
    for session in sessions.values():
        history = session.get("chat_history") if isinstance(session, dict) else []
        rows.append({{
            "id": str(session.get("id") or ""),
            "title": str(session.get("title") or "New chat"),
            "created_at": str(session.get("created_at") or ""),
            "updated_at": str(session.get("updated_at") or ""),
            "message_count": len(history) if isinstance(history, list) else 0,
            "memory_uri": str(session.get("memory_uri") or ""),
        }})
    return sorted(rows, key=lambda item: str(item.get("updated_at") or ""), reverse=True)


def _artifact_filename(key: str) -> str:
    import hashlib

    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
    leaf = Path(key).name or "artifact.bin"
    return f"{{digest}}-{{leaf}}"


def _artifact_preview_url(filename: str) -> str:
    return f"/api/artifacts/file/{{filename}}"


def _apply_loaded_artifact(
    *,
    state: dict,
    run_id: str,
    key: str,
    s3_uri: str,
    render: str,
    local_path: Path,
) -> dict:
    now = _now_iso()
    sim_viz = dict(DEFAULT_SIM_VIZ)
    current = state.get("sim_viz")
    if isinstance(current, dict):
        sim_viz.update(current)
    sim_viz.update(
        {{
            "run_id": run_id,
            "stage": "artifact-loaded",
            "rrd_updated_at": now,
            "artifact_uri": s3_uri,
            "artifact_key": key,
            "artifact_render": render,
            "mode": "static",
            "camera": str(sim_viz.get("camera") or "workspace"),
        }}
    )
    if render == "rerun":
        _publish_rrd_recording(local_path)
        _restart_rerun_serve(force=True)
        sim_viz["rrd_uri"] = f"file://{{RECORDING_PATH}}"
        sim_viz["artifact_preview_url"] = "/rerun/recordings/sim2real.rrd"
        sim_viz["artifact_download_url"] = "/rerun/recordings/sim2real.rrd"
        sim_viz["rerun_iframe_url"] = f"/rerun/?url=/rerun/recordings/sim2real.rrd&camera={{sim_viz['camera']}}"
        sim_viz["rerun_ready"] = RECORDING_PATH.is_file() and _rerun_web_viewer_healthy()
    else:
        filename = _artifact_filename(key)
        target = RECORDINGS_DIR / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, target)
        preview_url = _artifact_preview_url(filename)
        sim_viz["artifact_preview_url"] = preview_url
        sim_viz["artifact_download_url"] = preview_url
        sim_viz["rrd_uri"] = ""
        sim_viz["rerun_iframe_url"] = "/rerun/"
        sim_viz["rerun_ready"] = False
    state["sim_viz"] = sim_viz
    _record_sim_viz_run(state, sim_viz)
    _save_state(state)
    return sim_viz

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

def _rerun_ready_state(*, rrd_uri: str = "") -> bool:
    has_rrd = bool(str(rrd_uri or "").strip())
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

def _wire_franka_demo(state: dict, *, camera: str = "workspace") -> dict:
    selection = _stock_franka_selection()
    state["selection"] = selection
    cam = (camera or "workspace").strip() or "workspace"
    state["camera_selection"] = [cam]
    target = _generate_franka_demo_rrd(camera=cam)
    restarted = _restart_rerun_serve()
    now = _now_iso()
    prior = state.get("sim_viz", {{}})
    run_id = str(prior.get("run_id") or "").strip() or "franka-demo"
    viz = {{
        "run_id": run_id,
        "stage": "demo",
        "rrd_uri": f"file://{{target}}",
        "rrd_updated_at": now,
        "live_grpc_url": "",
        "mode": "static",
        "camera": cam,
        "preview_camera": cam,
        "preview_entity": f"world/cameras/{{cam}}",
        "rerun_ready": target.is_file() and _rerun_web_viewer_healthy(),
        "rerun_iframe_url": f"/rerun/?url=/rerun/recordings/sim2real.rrd&camera={{cam}}",
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
        "- GET /api/sim-assets, /api/sim-assets/selection, /api/sim-assets/cameras",
        "- GET /api/sim-viz/status — active run + .rrd URI for the Rerun iframe at /rerun/",
        "- GET /api/sim-viz/recordings — list available .rrd recording files for quick viewer switching",
        "- GET /api/sim-viz/runs — list run-scoped history (run_id, stage, camera, rrd_uri)",
        "- POST /api/sim-viz/load-run — switch active run context by run_id",
        "- GET /api/artifacts/runs?prefix=&limit= — discover run prefixes from object storage (no workflow allowlist)",
        "- GET /api/artifacts/run/{{run_id}} — list every object for a run with render hints",
        "- POST /api/sim-viz/load-artifact — load explicit s3_uri (or run_id+key) into viewer/download",
        "- POST /api/sim-viz/load-franka-demo — load stock Franka tabletop demo into Rerun",
        "- POST /api/workflows/sim2real/submit — submit Sim2Real with current asset selection",
        "- GET/POST /api/workflows/draft — workflow YAML draft in session",
        "- POST /api/workflows/validate — validate npa.workflow/v0.0.1 or npa.workflow/v0.0.1-beta YAML",
        "- POST /api/workflows/plan — dry-run plan-spec for workflow YAML",
        "- POST /api/workflows/submit — validate workflow YAML, ensure agent-side Kubernetes infra when needed, and return scheduler plan",
        "- GET /api/models — list Token Factory chat models available to this VM key",
        "- GET /api/tools — workbench toolRef catalog",
        "",
        "To view Franka immediately, tell users to click **Load Franka in Rerun** in the Sim Assets panel",
        "(or POST /api/sim-viz/load-franka-demo). Open the embedded viewer at /rerun/.",
        "Artifact-first browsing flow: call `/api/artifacts/runs`, inspect `/api/artifacts/run/{{id}}`,",
        "then `POST /api/sim-viz/load-artifact` with explicit `s3_uri` or `run_id` + `key`.",
        "The **Cameras** panel is the center column below chat: stock workspace and wrist cameras",
        "with 2D frustum schematics, selection, and **Preview in Rerun**.",
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
            "Always use real registry-qualified images from your Nebius container registry",
            "(or `NPA_REGISTRY` / `container_registry` in ~/.npa/config.yaml); never keep",
            "`<your-registry-id>` placeholders in runnable workflows.",
            "For BYOF solution onboarding, use the",
            "`npa/scripts/run_byof_repo.py` flow to containerize an OSS repo,",
            "push to the configured Nebius registry, then launch a real Isaac-Lab run",
            "with `--image` override on RT-core GPUs (L40S / RTX PRO 6000).",
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

def _provider_chat(*, provider: str, messages: list, model: str) -> dict:
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

def _chat_with_resilience(*, messages: list, requested_model: str) -> tuple[dict, str, str]:
    providers = _configured_llm_providers()
    models = _configured_llm_models()
    if requested_model and requested_model not in models:
        models.insert(0, requested_model)
    errors: list[str] = []
    for provider in providers:
        for model in models:
            try:
                data = _provider_chat(provider=provider, messages=messages, model=model)
                return data, provider, model
            except Exception as exc:
                errors.append(str(exc))
                continue
    detail = "; ".join(errors[-4:]) if errors else "no providers configured"
    raise HTTPException(status_code=502, detail=f"LLM providers unavailable: {{detail}}")

{_AGENT_CHAT_EMBED}

{_AGENT_WORKFLOW_EMBED}

{_AGENT_ARTIFACTS_EMBED}

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

def _record_sim_viz_run(state: dict, record: dict) -> None:
    if not isinstance(record, dict):
        return
    run_id = str(record.get("run_id") or "").strip()
    if not run_id:
        return
    entries = state.get("sim_viz_runs")
    if not isinstance(entries, dict):
        entries = {{}}
    snapshot = dict(DEFAULT_SIM_VIZ)
    snapshot.update(record)
    snapshot["run_id"] = run_id
    entries[run_id] = snapshot
    state["sim_viz_runs"] = entries
    state["active_run_id"] = run_id

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
    yaml_text = str(payload.get("yaml") or "").strip()
    if yaml_text:
        return yaml_text
    draft = _workflow_draft_from_state(_load_state())
    return str(draft.get("yaml") or "").strip()

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
    projects = config.get("projects")
    if not isinstance(projects, dict):
        projects = {{}}
    project_block = projects.get(alias)
    if not isinstance(project_block, dict):
        project_block = {{}}
    configured: list[dict] = []
    kube_block = project_block.get("kubernetes")
    if isinstance(kube_block, dict) and kube_block:
        configured.append({{
            "source": "project_config",
            "project": alias,
            "cluster_name": str(kube_block.get("cluster_name") or kube_block.get("name") or ""),
            "context": str(kube_block.get("context") or kube_block.get("context_name") or ""),
            "kubeconfig": str(kube_block.get("kubeconfig") or kube_block.get("kubeconfig_path") or ""),
            "gpu_profile": str(kube_block.get("gpu_profile") or ""),
            "raw": {{k: v for k, v in kube_block.items() if k not in {{"token", "secret", "password"}}}},
        }})
    clusters_root = Path.home() / ".npa" / "clusters"
    local_clusters: list[dict] = []
    if clusters_root.is_dir():
        for item in sorted(clusters_root.iterdir()):
            if not item.is_dir():
                continue
            kubeconfig = item / "kubeconfig"
            state_path = item / "state.json"
            local_clusters.append({{
                "source": "local_state",
                "cluster_name": item.name,
                "context": item.name,
                "kubeconfig": str(kubeconfig),
                "kubeconfig_exists": kubeconfig.is_file(),
                "state_exists": state_path.is_file(),
            }})
    ready, reason = _agent_npa_ready()
    cloud_clusters = _agent_cloud_mk8s_clusters(alias)
    return {{
        "ok": True,
        "project": alias,
        "configured": configured,
        "local_clusters": local_clusters,
        "cloud_clusters": cloud_clusters,
        "has_infra": bool(
            configured
            or any(item.get("kubeconfig_exists") for item in local_clusters)
            or cloud_clusters
        ),
        "agent_npa_ready": ready,
        "agent_npa_error": reason,
        "terraform_dir": str(NPA_CLUSTER_TERRAFORM_DIR),
        "options": [
            "POST /api/infra/provision to let the agent create the minimal Kubernetes backend.",
            "Add projects.<alias>.kubernetes to ~/.npa/config.yaml on the agent to use an existing backend.",
            "Pass project/cluster_name in the workflow submit payload to target a known backend.",
        ],
    }}


def _agent_command_env() -> dict:
    env = dict(os.environ)
    env["PATH"] = "/usr/local/bin:/usr/bin:/bin:" + env.get("PATH", "")
    env.setdefault("NPA_TERRAFORM_BIN", shutil.which("terraform") or "terraform")
    env.setdefault("NPA_KUBECTL_BIN", shutil.which("kubectl") or "kubectl")
    env.setdefault("NPA_NEBIUS_BIN", shutil.which("nebius") or "nebius")
    if not env.get("TF_VAR_ssh_public_key"):
        for candidate in ("/home/ubuntu/.ssh/id_ed25519.pub", "/root/.ssh/id_ed25519.pub"):
            if Path(candidate).is_file():
                env["TF_VAR_ssh_public_key"] = json.dumps({{"path": candidate}})
                break
        if not env.get("TF_VAR_ssh_public_key"):
            for candidate in ("/home/ubuntu/.ssh/authorized_keys", "/root/.ssh/authorized_keys"):
                path = Path(candidate)
                if not path.is_file():
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
    try:
        proc = subprocess.run(
            [nebius_bin, "mk8s", "cluster", "list", "--parent-id", parent_id, "--format", "json"],
            env=_agent_command_env(),
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
        clusters.append({{
            "source": "nebius_mk8s",
            "id": str(metadata.get("id") or ""),
            "name": str(metadata.get("name") or ""),
            "status": str(status.get("state") or status.get("status") or ""),
            "raw_status": {{k: v for k, v in status.items() if k not in {{"token", "secret", "password"}}}},
        }})
    return clusters


def _run_agent_npa_json(args: list[str], *, timeout_s: int = 300) -> dict:
    ready, reason = _agent_npa_ready()
    if not ready:
        raise HTTPException(status_code=409, detail=reason)
    proc = subprocess.run(
        [str(NPA_CLI), *args],
        cwd=str(NPA_SOURCE_ROOT),
        env=_agent_command_env(),
        text=True,
        capture_output=True,
        timeout=timeout_s,
        check=False,
    )
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
    for item in stages:
        if isinstance(item, dict) and item.get("id") == stage_id:
            item["status"] = status
            if status == "running" and not item.get("started_at"):
                item["started_at"] = _now_iso()
            if status in {{"succeeded", "failed"}}:
                item["finished_at"] = _now_iso()
            if summary:
                item["summary"] = summary
            break
    details["stages"] = stages


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
    if settings.get("bucket"):
        cmd.extend([
            "--s3-bucket",
            str(settings["bucket"]),
            "--s3-prefix",
            _join_agent_s3_prefix(str(settings.get("prefix") or ""), "sim2real-b"),
            "--upload-artifacts",
        ])
    if settings.get("endpoint"):
        cmd.extend(["--s3-endpoint", str(settings["endpoint"])])
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
            _publish_rrd_recording(rrd_path)
            try:
                shutil.copy2(rrd_path, RRD_PATH)
            except Exception:
                pass
            _restart_rerun_serve(force=True)
            _append_run_log(details, f"Published Rerun recording: {{rrd_path}}")
        return details

    _update_sim2real_run(run_id, mutate=_finish)
    state = _load_state()
    sim_viz = state.get("sim_viz")
    if not isinstance(sim_viz, dict):
        sim_viz = {{}}
    if rrd_path.is_file():
        sim_viz.update(
            {{
                "run_id": run_id,
                "stage": "completed",
                "rrd_uri": f"file://{{RECORDING_PATH}}",
                "rrd_updated_at": _now_iso(),
                "rerun_ready": RECORDING_PATH.is_file() and _rerun_web_viewer_healthy(),
                "rerun_iframe_url": f"/rerun/?url=/rerun/recordings/sim2real.rrd&camera={{sim_viz.get('camera') or 'workspace'}}",
                "camera": str(sim_viz.get("camera") or "workspace"),
            }}
        )
    else:
        sim_viz.update({{"run_id": run_id, "stage": "completed", "rrd_updated_at": _now_iso()}})
    state["sim_viz"] = sim_viz
    _record_sim_viz_run(state, sim_viz)
    _save_state(state)


def _write_workflow_temp_yaml(yaml_text: str) -> Path:
    tmp_dir = Path("/tmp/npa-agent-workflows")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    path = tmp_dir / f"workflow-{{secrets.token_hex(8)}}.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    return path


def _provision_agent_infra(
    project: str,
    cluster_name: str,
    *,
    dry_run: bool = False,
    validate: bool = True,
    skip_s3: bool = True,
) -> dict:
    ready, reason = _agent_npa_ready()
    if not ready:
        return {{"ok": False, "status": "blocked", "error": reason}}
    try:
        from npa.provisioning import provision_if_absent

        result = provision_if_absent(
            project=project or None,
            cluster_name=cluster_name or "npa-cluster",
            terraform_dir=NPA_CLUSTER_TERRAFORM_DIR,
            skip_s3=skip_s3,
            validate=validate,
            sky_smoke=False,
            dry_run=dry_run,
        )
        payload = result.to_dict()
        payload["ok"] = True
        payload["dry_run"] = dry_run
        return payload
    except Exception as exc:
        return {{"ok": False, "status": "error", "error": str(exc), "dry_run": dry_run}}


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
    "onboard_solution": ("byof-onboard",),
    "find_artifacts": ("find-artifacts",),
    "create_workflow": ("author-npa-workflow",),
    "create_vlm_rl_workflow": ("author-npa-workflow", "sim-to-real"),
    "create_gate_workflow": ("author-npa-workflow", "sim-to-real"),
    "live_infra_loop": ("submit-workflow", "gpu-selection"),
    "cosmos3": ("cosmos3-setup",),
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
        root_name = str(payload.get("root") or "skills").strip() or "skills"
        root = candidate.parent / root_name
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
    snippets: list[str] = []
    for name in names[:4]:
        excerpt = _skill_excerpt(name)
        if excerpt:
            snippets.append(f"[skill:{{name}}]\\n{{excerpt}}")
    if not snippets:
        return names, ""
    return names, "Relevant NPA skill excerpts:\\n\\n" + "\\n\\n".join(snippets)

def _last_user_message(raw_messages: list) -> str:
    for item in reversed(raw_messages):
        if not isinstance(item, dict):
            continue
        if str(item.get("role", "")).strip() == "user":
            return str(item.get("content", "")).strip()
    return ""

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
            "- **mode**: `agent-local-sim2real`\\n"
            "- The Run Monitor will update stages, result, and logs; Rerun will switch to the run recording when it is written."
        )
        return reply, _dedupe(apis_used), suggested_apis, None, submit, intent
    if intent == "load_franka":
        sim_viz = state.get("sim_viz", {{}})
        if not isinstance(sim_viz, dict):
            sim_viz = {{}}
        rerun_ready = _rerun_ready_state(rrd_uri=str(sim_viz.get("rrd_uri") or ""))
        if not rerun_ready:
            selected = state.get("camera_selection", ["workspace"])
            cam = str(selected[0] if isinstance(selected, list) and selected else "workspace")
            _wire_franka_demo(state, camera=cam)
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
    elif intent == "infra_backends":
        state["infra"] = _agent_k8s_backends()
        _save_state(state)
    elif intent in {{"create_workflow", "create_vlm_rl_workflow", "create_gate_workflow"}}:
        draft = generate_workflow_draft(
            user_text=user_text,
            intent=intent,
            tool_refs=frozenset(TOOL_REFS),
            capabilities={{"tool_refs": list(TOOL_REFS)}},
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
            fail_reason = str(validation.get("error") or plan.get("error") or "validate+plan gate did not pass")
            reply = (
                "**Could not generate runnable workflow YAML yet.**\\n"
                f"- **reason**: `{{fail_reason}}`\\n"
                "- Adjust your request or template details and retry;"
                " chat returns YAML only after both validation and planning succeed."
            )
            return reply, _dedupe(apis_used), suggested_apis, None, {{"ok": False, "validation": validation, "plan": plan}}, intent
        reply = format_workflow_chat_reply(yaml_text, validation, template=template, plan=plan, runnable=runnable)
        return reply, _dedupe(apis_used), suggested_apis, yaml_text, validation, intent
    if intent in {{"onboard_solution", "tools_catalog", "component_capabilities", "cosmos_capabilities", "lancedb_capabilities", "live_infra_loop"}}:
        apis_used.append("tools")
    reply = build_grounded_reply(
        intent,
        state,
        TOOL_REFS,
        rerun_ready=rerun_ready,
        loaded_franka_now=loaded_now,
        default_cameras=default_cameras,
    )
    return reply, _dedupe(apis_used), suggested_apis, None, None, intent

def _agent_chat_with_tools(*, raw_messages: list, model: str) -> dict | None:
    last_user = _last_user_message(raw_messages)
    if not last_user:
        return None
    tool_reply, apis_used, apis_suggested, workflow_yaml, workflow_validation, intent = _maybe_toolground_chat_reply(last_user)
    if not tool_reply:
        return None
    skill_names, _ = _resolve_skill_context(user_text=last_user, intent=intent)
    payload = {{
        "ok": True,
        "model": model,
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

@app.post("/chat")
def chat(payload: dict):
    raw_messages = payload.get("messages", [])
    if not isinstance(raw_messages, list) or not raw_messages:
        raise HTTPException(status_code=400, detail="messages must be a non-empty list")
    model = str(payload.get("model") or LLM_MODEL).strip() or LLM_MODEL
    state = _load_state()
    session_id = _sanitize_chat_session_id(
        str(payload.get("session_id") or state.get("active_chat_session_id") or "default")
    )
    session = _get_chat_session(state, session_id)
    history = _normalize_chat_history(raw_messages)
    if len(history) <= 1 and isinstance(session.get("chat_history"), list):
        prior = _normalize_chat_history(session.get("chat_history", []))
        if history:
            history = [*prior, history[-1]]
        else:
            history = prior
    raw_messages = history
    tool_result = _agent_chat_with_tools(raw_messages=raw_messages, model=model)
    if tool_result is not None:
        reply = str(tool_result.get("reply") or "").strip()
        history: list[dict] = []
        for item in raw_messages:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "user")).strip() or "user"
            content = str(item.get("content", "")).strip()
            if role in {{"user", "assistant"}} and content:
                history.append({{"role": role, "content": content}})
        if reply:
            history.append({{"role": "assistant", "content": reply}})
        session.update(
            {{
                "id": session_id,
                "title": str(session.get("title") or _chat_session_title(history)),
                "chat_history": history[-80:],
            }}
        )
        # Tool handlers may mutate session state (for example starting a Sim2Real
        # run). Reload before saving chat history so an older state snapshot does
        # not clobber the run monitor.
        state = _load_state()
        session = _save_chat_session(state, session, active=True)
        tool_result["session_id"] = session["id"]
        tool_result["session"] = {{
            "id": session["id"],
            "title": session["title"],
            "memory_uri": session.get("memory_uri", ""),
            "message_count": len(session.get("chat_history", [])),
        }}
        _save_state(state)
        return tool_result
    live_ctx = format_live_context_block(_load_state())
    last_user = _last_user_message(raw_messages)
    intent = match_chat_intent(last_user)
    skill_names, skill_ctx = _resolve_skill_context(user_text=last_user, intent=intent)
    system_content = _agent_system_prompt() + "\\n\\n" + live_ctx
    if skill_ctx:
        system_content += "\\n\\n" + skill_ctx
    messages: list[dict] = [
        {{"role": "system", "content": system_content}}
    ]
    for item in raw_messages:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "user")).strip() or "user"
        content = str(item.get("content", "")).strip()
        if content:
            messages.append({{"role": role, "content": content}})
    if len(messages) < 2:
        raise HTTPException(status_code=400, detail="at least one user message is required")
    data, selected_provider, selected_model = _chat_with_resilience(messages=messages, requested_model=model)
    try:
        message = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise HTTPException(status_code=502, detail="LLM response missing assistant message") from exc
    reply, reasoning = _split_reasoning(message)
    if not reply and reasoning:
        reply = reasoning
        reasoning = None
    state = _load_state()
    history: list[dict] = []
    for item in raw_messages:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "user")).strip() or "user"
        content = str(item.get("content", "")).strip()
        if role in {{"user", "assistant"}} and content:
            history.append({{"role": role, "content": content}})
    if reply:
        history.append({{"role": "assistant", "content": reply}})
    session.update(
        {{
            "id": session_id,
            "title": str(session.get("title") or _chat_session_title(history)),
            "chat_history": history[-80:],
        }}
    )
    session = _save_chat_session(state, session, active=True)
    return {{
        "ok": True,
        "model": selected_model,
        "provider": selected_provider,
        "reply": reply,
        "reasoning": reasoning,
        "session_id": session["id"],
        "session": {{
            "id": session["id"],
            "title": session["title"],
            "memory_uri": session.get("memory_uri", ""),
            "message_count": len(session.get("chat_history", [])),
        }},
        "skills_used": skill_names,
    }}

@app.get("/health")
def health():
    return {{"ok": True, "tool_refs": len(TOOL_REFS)}}

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
    sim_viz = dict(DEFAULT_SIM_VIZ)
    if isinstance(state.get("sim_viz"), dict):
        sim_viz.update(state["sim_viz"])
    selected = state.get("camera_selection", ["workspace"])
    camera = str(sim_viz.get("camera") or (selected[0] if isinstance(selected, list) and selected else "workspace"))
    sim_viz["camera"] = camera
    session_run_id = str(sim_viz.get("run_id") or "").strip()
    if not sim_viz.get("rrd_uri") and session_run_id in {"", "franka-demo"} and RRD_PATH.is_file():
        sim_viz["rrd_uri"] = f"file://{{RRD_PATH}}"
    sim_viz["rerun_ready"] = _rerun_ready_state(rrd_uri=str(sim_viz.get("rrd_uri") or ""))
    history = active_session.get("chat_history", [])
    if not isinstance(history, list):
        history = []
    return {{
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
    session = _get_chat_session(state, session_id)
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

@app.get("/sim-viz/status")
def sim_viz_status(run_id: str = ""):
    state = _load_state()
    payload = _sim_viz_for_run(state, run_id=run_id)
    requested_run = str(run_id or "").strip()
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
    _record_sim_viz_run(state, payload)
    payload_run = str(payload.get("run_id") or "").strip()
    run_has_specific_rrd = bool(str(payload.get("rrd_uri") or "").strip())
    may_use_default_recording = payload_run in {"", "franka-demo"} and not requested_run
    if (
        str(payload.get("artifact_render") or "").strip().lower() in {"", "rerun"}
        and (run_has_specific_rrd or may_use_default_recording)
    ):
        payload["rerun_iframe_url"] = f"/rerun/?url=/rerun/recordings/sim2real.rrd&camera={{camera}}"
    else:
        payload["rerun_iframe_url"] = ""
    if not payload.get("rrd_uri") and may_use_default_recording and RRD_PATH.is_file():
        payload["rrd_uri"] = f"file://{{RRD_PATH}}"
    mode = str(payload.get("mode") or "static").strip().lower()
    payload["mode"] = "live" if mode == "live" else "static"
    payload["rerun_ready"] = _rerun_ready_state(rrd_uri=str(payload.get("rrd_uri") or ""))
    runs = state.get("sim_viz_runs")
    if isinstance(runs, dict):
        payload["available_run_ids"] = sorted(str(key) for key in runs.keys() if str(key).strip())
    else:
        payload["available_run_ids"] = []
    payload["active_run_id"] = str(state.get("active_run_id") or payload.get("run_id") or "").strip()
    _save_state(state)
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
    if not requested_run:
        raise HTTPException(status_code=400, detail="run_id is required")
    state = _load_state()
    runs = _sim_viz_runs(state)
    selected = next((item for item in runs if str(item.get("run_id") or "").strip() == requested_run), None)
    if not isinstance(selected, dict):
        raise HTTPException(status_code=404, detail=f"run_id not found: {{requested_run}}")
    sim_viz = dict(DEFAULT_SIM_VIZ)
    if isinstance(state.get("sim_viz"), dict):
        sim_viz.update(state["sim_viz"])
    sim_viz.update(
        {{
            "run_id": requested_run,
            "stage": str(selected.get("stage") or sim_viz.get("stage") or "submitted"),
            "rrd_uri": str(selected.get("rrd_uri") or sim_viz.get("rrd_uri") or ""),
            "rrd_updated_at": str(selected.get("rrd_updated_at") or sim_viz.get("rrd_updated_at") or ""),
            "camera": str(selected.get("camera") or sim_viz.get("camera") or "workspace"),
        }}
    )
    state["sim_viz"] = sim_viz
    state["latest_submit"] = {{
        "run_id": requested_run,
        "submitted_at": str(selected.get("submitted_at") or _now_iso()),
        "submit_mode": str(selected.get("submit_mode") or "history-select"),
    }}
    _save_state(state)
    return {{"ok": True, "sim_viz": sim_viz_status(), "selected": selected}}

@app.post("/sim-viz/load-run")
def sim_viz_load_run(payload: dict | None = None):
    body = payload if isinstance(payload, dict) else {{}}
    run_id = str(body.get("run_id") or "").strip()
    if not run_id:
        raise HTTPException(status_code=400, detail="run_id is required")
    state = _load_state()
    runs = state.get("sim_viz_runs")
    if not isinstance(runs, dict):
        runs = {{}}
    selected = runs.get(run_id)
    if not isinstance(selected, dict):
        selected = {{}}
    rrd_uri = str(body.get("rrd_uri") or "").strip()
    if rrd_uri:
        selected["rrd_uri"] = rrd_uri
    camera = str(body.get("camera") or "").strip()
    if camera:
        selected["camera"] = camera
    stage = str(body.get("stage") or "").strip()
    if stage:
        selected["stage"] = stage
    mode = str(body.get("mode") or "").strip().lower()
    if mode in {{"static", "live"}}:
        selected["mode"] = mode
    if not selected:
        raise HTTPException(status_code=404, detail=f"run_id not found: {{run_id}}")
    selected["run_id"] = run_id
    selected["rrd_updated_at"] = _now_iso()
    state["sim_viz"] = selected
    _record_sim_viz_run(state, selected)
    _save_state(state)
    return {{
        "ok": True,
        "sim_viz": sim_viz_status(run_id=run_id),
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
def artifacts_runs(prefix: str = "", limit: int = 50):
    try:
        s3, settings = _agent_s3_client()
        effective_prefix = _join_agent_s3_prefix(settings.get("prefix", ""), prefix)
        page = list_runs(
            settings["bucket"],
            prefix=effective_prefix,
            limit=limit,
            s3=s3,
        )
        return {{"ok": True, "bucket": settings["bucket"], "prefix": effective_prefix, "base_prefix": settings.get("prefix", ""), **page.to_dict()}}
    except HTTPException:
        raise
    except Exception as exc:
        return JSONResponse(status_code=502, content={{"ok": False, "error": str(exc), "source": "s3"}})


@app.get("/artifacts/run/{{run_id:path}}")
def artifacts_for_run(run_id: str, prefix: str = ""):
    try:
        normalized_run = validate_run_id(run_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        s3, settings = _agent_s3_client()
        effective_prefix = _join_agent_s3_prefix(settings.get("prefix", ""), prefix)
        artifacts = list_artifacts(
            settings["bucket"],
            normalized_run,
            prefix=effective_prefix,
            s3=s3,
        )
        preferred = select_preferred_artifact(artifacts)
        return {{
            "ok": True,
            "bucket": settings["bucket"],
            "prefix": effective_prefix,
            "base_prefix": settings.get("prefix", ""),
            "run_id": normalized_run,
            "count": len(artifacts),
            "artifacts": [item.to_dict() for item in artifacts],
            "preferred": preferred.to_dict() if preferred else None,
        }}
    except HTTPException:
        raise
    except Exception as exc:
        return JSONResponse(status_code=502, content={{"ok": False, "error": str(exc), "source": "s3"}})


@app.get("/artifacts/file/{{filename}}")
def artifact_file(filename: str):
    safe_name = Path(str(filename)).name
    if safe_name != filename:
        raise HTTPException(status_code=400, detail="invalid artifact filename")
    target = RECORDINGS_DIR / safe_name
    if not target.is_file():
        raise HTTPException(status_code=404, detail=f"artifact file not found: {{filename}}")
    return FileResponse(str(target), media_type="application/octet-stream")


@app.post("/sim-viz/load-artifact")
def sim_viz_load_artifact(payload: dict | None = None):
    body = payload if isinstance(payload, dict) else {{}}
    requested_uri = str(body.get("s3_uri") or "").strip()
    requested_run = str(body.get("run_id") or "").strip()
    requested_key = str(body.get("key") or "").strip()
    if not requested_uri and not (requested_run and requested_key):
        raise HTTPException(status_code=400, detail="Provide either s3_uri or run_id + key")
    try:
        s3, settings = _agent_s3_client()
        if requested_uri:
            bucket, key = parse_s3_uri(requested_uri)
            run_guess = str(body.get("run_id") or _run_id_for_key(key, ""))
            run_id = validate_run_id(run_guess) if run_guess else "artifact"
            s3_uri = requested_uri
        else:
            run_id = validate_run_id(requested_run)
            key = _safe_artifact_key(requested_key)
            bucket = settings["bucket"]
            s3_uri = f"s3://{{bucket}}/{{key}}"
        local_name = _artifact_filename(key)
        local_path = RECORDINGS_DIR / local_name
        download_s3_uri(s3_uri, local_path, s3=s3)
        render = render_hint_for_object(key=key)
        state = _load_state()
        sim_viz = _apply_loaded_artifact(
            state=state,
            run_id=run_id,
            key=key,
            s3_uri=s3_uri,
            render=render,
            local_path=local_path,
        )
        return {{"ok": True, "sim_viz": sim_viz, "render": render, "artifact_uri": s3_uri}}
    except HTTPException:
        raise
    except Exception as exc:
        return JSONResponse(status_code=502, content={{"ok": False, "error": str(exc), "source": "s3"}})


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
    viz = _wire_franka_demo(state, camera=camera)
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
    entity_path = f"world/cameras/{{camera}}"
    return {{
        "ok": True,
        "camera": camera,
        "entity_path": entity_path,
        "rollout_entity_guess": f"rollouts/latest/{{camera}}/camera",
        "sim_viz": viz,
        "hint": "Open the Rerun panel and expand world/cameras/<name>.",
    }}

def _sim_viz_rrd_file_response(run_id: str = ""):
    state = _load_state()
    sim_viz = _sim_viz_for_run(state, run_id=run_id)
    uri = str(sim_viz.get("rrd_uri") or "").strip()
    if uri.startswith("file://"):
        file_path = Path(uri[len("file://"):])
        if file_path.is_file():
            return FileResponse(str(file_path), media_type="application/octet-stream")
    if uri.startswith("http://") or uri.startswith("https://"):
        try:
            proxied = httpx.get(uri, timeout=20.0)
            proxied.raise_for_status()
            return Response(content=proxied.content, media_type="application/octet-stream")
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"Unable to fetch remote sim2real.rrd: {{exc}}") from exc
    if RRD_PATH.is_file():
        return FileResponse(str(RRD_PATH), media_type="application/octet-stream")
    raise HTTPException(status_code=404, detail="No sim2real.rrd file on disk yet")

@app.get("/sim-viz/rrd")
def sim_viz_rrd(run_id: str = ""):
    return _sim_viz_rrd_file_response(run_id=run_id)

@app.get("/sim-viz/rrd-blob")
def sim_viz_rrd_blob(run_id: str = ""):
    # Authenticated .rrd bytes for parent-page blob URL (Rerun wasm cannot send basic auth).
    return _sim_viz_rrd_file_response(run_id=run_id)

@app.on_event("startup")
def _boot_preload_sim_viz() -> None:
    if not RRD_PATH.is_file():
        return
    _publish_rrd_recording(RRD_PATH)
    state = _load_state()
    sim_viz = state.get("sim_viz", {{}})
    if not isinstance(sim_viz, dict):
        sim_viz = {{}}
    if str(sim_viz.get("rrd_uri") or "").strip():
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
        "preview_entity": f"world/cameras/{{cam}}",
        "rerun_ready": _rerun_ready_state(rrd_uri=f"file://{{RRD_PATH}}"),
        "rerun_iframe_url": f"/rerun/?url=/rerun/recordings/sim2real.rrd&camera={{cam}}",
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
        return {{"ok": True, "selection": selection, "sim_viz": viz}}
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
def sim2real_status(run_id: str = ""):
    state = _load_state()
    latest = state.get("latest_submit", {{}})
    sim_viz = state.get("sim_viz", {{}})
    details = _sim2real_run_details(state, run_id=run_id)
    return {{
        "ok": True,
        "latest_submit": latest if isinstance(latest, dict) else {{}},
        "sim_viz": sim_viz if isinstance(sim_viz, dict) else dict(DEFAULT_SIM_VIZ),
        "run": details,
        "stages": details.get("stages", []),
        "logs": details.get("logs", []),
    }}

@app.get("/workflows/sim2real/runs/{{run_id:path}}")
def sim2real_run_detail(run_id: str):
    state = _load_state()
    details = _sim2real_run_details(state, run_id=run_id)
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
def list_k8s_infra(project: str = ""):
    return _agent_k8s_backends(project)


@app.post("/infra/provision")
def provision_infra(payload: dict | None = None):
    body = payload if isinstance(payload, dict) else {{}}
    project = _agent_project_alias(str(body.get("project") or ""))
    cluster_name = str(body.get("cluster_name") or "npa-cluster").strip() or "npa-cluster"
    dry_run = bool(body.get("dry_run", False))
    validate = bool(body.get("validate", True))
    skip_s3 = bool(body.get("skip_s3", True))
    result = _provision_agent_infra(project, cluster_name, dry_run=dry_run, validate=validate, skip_s3=skip_s3)
    status = _agent_k8s_backends(project)
    return {{"ok": bool(result.get("ok")), "project": project, "cluster_name": cluster_name, "result": result, "infra": status}}


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
    return {{"ok": runnable, "draft": draft, "validation": validation, "plan": plan, "runnable": runnable}}

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
    return {{"ok": runnable, "validation": validation, "plan": plan, "runnable": runnable}}

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
    allow_provision = bool(body.get("allow_provision", True))
    dry_run = bool(body.get("dry_run", False))
    validate_infra = bool(body.get("validate_infra", True))
    infra_before = _agent_k8s_backends(project)
    if not infra_before.get("has_infra") and not allow_provision:
        return _workflow_no_infra_response(validation=validation, plan=plan, run_id=run_id, infra=infra_before)
    provision = {{"ok": True, "status": "skipped", "actions": ["k8s:existing backend detected"]}}
    if allow_provision and (dry_run or not infra_before.get("has_infra")):
        provision = _provision_agent_infra(
            project,
            cluster_name,
            dry_run=dry_run,
            validate=validate_infra,
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
def submit_sim2real(payload: dict):
    state = _load_state()
    selection = state.get("selection", {{}})
    if not isinstance(selection, dict):
        selection = dict(DEFAULT_SELECTION)
    run_id = f"agent-run-{{secrets.token_hex(6)}}"
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
    }}
    submitted_at = str(state["latest_submit"]["submitted_at"])
    state["sim_viz"] = {{
        "run_id": run_id,
        "stage": "submitted",
        "rrd_uri": "",
        "rrd_updated_at": submitted_at,
        "live_grpc_url": "",
        "mode": "static",
        "rerun_ready": False,
        "rerun_iframe_url": "",
        "camera": "workspace",
    }}
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
    _record_sim_viz_run(
        state,
        {{
            "run_id": run_id,
            "submitted_at": submitted_at,
            "stage": "submitted",
            "camera": str((state.get("sim_viz", {{}}) or {{}}).get("camera") or "workspace"),
            "rrd_uri": "",
            "rrd_updated_at": str((state.get("sim_viz", {{}}) or {{}}).get("rrd_updated_at") or ""),
            "submit_mode": "sim2real",
            "workflow_name": "sim2real",
            "rerun_ready": False,
            "rerun_iframe_url": "",
        }},
    )
    _save_state(state)
    thread = threading.Thread(
        target=_run_sim2real_pipeline_background,
        args=(run_id, dict(selection)),
        daemon=True,
    )
    thread.start()
    return {{"ok": True, "run_id": run_id, "selection": selection, "env": env_block, "run": details, "submit_mode": "agent-local-sim2real"}}
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
sudo cp -f /opt/npa-agent/sim2real.rrd /opt/npa-agent/recordings/sim2real.rrd || true
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
<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
    <meta http-equiv="Cache-Control" content="no-store, no-cache, must-revalidate">
    <meta name="npa-ui-version" content="{AGENT_UI_VERSION}">
    <title>NPA Agent</title>
    <link rel="preload" href="/rerun/re_viewer.js" as="script" crossorigin>
    <link rel="preload" href="/rerun/re_viewer_bg.wasm" as="fetch" type="application/wasm" crossorigin>
    <link rel="prefetch" href="/rerun/recordings/sim2real.rrd" as="fetch">
    <style>
      :root {{
        --bg: #f5f6f8;
        --surface: #ffffff;
        --text: #1f2430;
        --muted: #5f6573;
        --border: #e0e0e0;
        --brand: #5e43f3;
        --brand-strong: #4d35d4;
        --sidebar: #1e1f22;
        --ok-bg: #e8f7ee;
        --ok-text: #18794e;
        --shadow: 0 8px 22px rgba(30, 31, 34, 0.08);
      }}
      * {{ box-sizing: border-box; }}
      html {{
        overflow-x: hidden;
        width: 100%;
        max-width: 100%;
        -webkit-text-size-adjust: 100%;
      }}
      body {{
        margin: 0;
        padding-bottom: 36px;
        font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        background: var(--bg);
        color: var(--text);
        overflow-x: hidden;
        width: 100%;
        max-width: 100%;
      }}
      img, video, iframe, pre, textarea, select, input, .panel, .page, .chrome {{
        max-width: 100%;
      }}
      .chrome {{
        max-width: 1640px;
        margin: 0 auto;
        min-height: 100vh;
      }}
      .topbar {{
        background: var(--sidebar);
        color: #eef0f3;
        padding: 14px 18px;
        border-bottom: 1px solid #2a2c31;
        display: flex;
        align-items: center;
        justify-content: space-between;
      }}
      .brand {{
        font-weight: 700;
        letter-spacing: 0.02em;
        font-size: 13px;
      }}
      .brand-sub {{
        color: #abb2bf;
        font-size: 12px;
      }}
      .page {{
        padding: 16px;
        display: grid;
        gap: 14px;
      }}
      .layout {{ display: grid; gap: 14px; }}
      .layout-3 {{ grid-template-columns: 1fr 0.95fr 1.35fr; }}
      .panel {{
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 14px;
        background: var(--surface);
        box-shadow: var(--shadow);
      }}
      .panel h3 {{ margin: 0 0 10px 0; font-size: 17px; }}
      .panel p {{ margin: 0; color: var(--muted); }}
      .subsection {{
        border: 1px solid #e7e8ee;
        border-radius: 10px;
        background: #fafbff;
        padding: 10px;
        margin-top: 10px;
      }}
      .subsection h4 {{ margin: 0 0 8px 0; font-size: 13px; color: #303649; }}
      .field-row {{ display: grid; gap: 8px; grid-template-columns: 1fr 1fr; }}
      .field label {{ display: block; font-size: 12px; color: #4f5668; margin-bottom: 4px; }}
      .field select, .field input {{
        width: 100%;
        border: 1px solid #d4d8e2;
        border-radius: 9px;
        padding: 8px;
        font-family: inherit;
        background: #fff;
      }}
      .pill-list {{ display: flex; gap: 6px; flex-wrap: wrap; }}
      .pill {{
        display: inline-flex;
        align-items: center;
        border-radius: 999px;
        border: 1px solid #d8dbeb;
        color: #394056;
        background: #fff;
        padding: 4px 9px;
        font-size: 12px;
      }}
      .cameras-panel {{ border-color: #d7dbf6; }}
      .camera-card {{
        border: 1px solid #e2e8f0; border-radius: 10px; padding: 10px; margin-bottom: 10px;
        background: #fff;
      }}
      .camera-card.selected {{ border: 2px solid #5e43f3; box-shadow: 0 0 0 1px rgba(94, 67, 243, 0.18); }}
      .camera-card h4 {{ margin: 0 0 6px 0; }}
      .camera-meta {{ font-size: 12px; color: #4b5568; margin-bottom: 6px; }}
      .camera-actions {{ display: flex; gap: 8px; flex-wrap: wrap; margin-top: 8px; }}
      .camera-frustum {{ display: flex; justify-content: center; }}
      .rollout-hint {{ font-size: 13px; color: #39465c; margin: 0 0 10px 0; }}
      .chat-panel {{ margin-bottom: 12px; }}
      .chat-panel-head {{
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 10px;
        margin-bottom: 4px;
      }}
      .chat-panel-head h3 {{ margin: 0; }}
      .mobile-only-toggle {{ display: none; }}
      .chat-composer {{
        background: var(--surface);
      }}
      .mobile-chat-auth {{
        display: none;
        margin: 0 0 10px;
        padding: 12px;
        border: 1px solid #fcd34d;
        border-radius: 10px;
        background: #fffbeb;
      }}
      .mobile-chat-auth-row {{
        display: flex;
        flex-direction: column;
        gap: 8px;
      }}
      .mobile-chat-auth-row input {{
        width: 100%;
        padding: 10px 12px;
        border: 1px solid #d4d8e2;
        border-radius: 10px;
        font: inherit;
        font-size: 16px;
      }}
      body.mobile-agent.mobile-needs-auth .mobile-chat-auth {{
        display: block;
      }}
      body.mobile-agent.mobile-needs-auth #chatForm,
      body.mobile-agent.mobile-needs-auth .actions-inline {{
        opacity: 0.55;
        pointer-events: none;
      }}
      body.mobile-agent .mobile-chat-auth {{
        order: -1;
      }}
      .chat-toolbar {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 10px;
        margin: 8px 0 10px;
        flex-wrap: wrap;
      }}
      .chat-model {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 12px;
        color: #4f5668;
      }}
      .chat-session {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 12px;
        color: #4f5668;
      }}
      .chat-session select {{
        max-width: 220px;
      }}
      .chat-model select {{
        border: 1px solid #d4d8e2;
        border-radius: 999px;
        padding: 6px 10px;
        background: #fff;
        color: #1f2430;
        font: inherit;
      }}
      .chat-session select {{
        border: 1px solid #d4d8e2;
        border-radius: 999px;
        padding: 6px 10px;
        background: #fff;
        color: #1f2430;
        font: inherit;
      }}
      .chat-log {{
        height: 320px; overflow-y: auto; background: #f9fafc; border: 1px solid var(--border);
        border-radius: 10px; padding: 10px; margin-bottom: 10px;
        font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      }}
      .msg-row {{
        display: flex;
        margin: 8px 0;
      }}
      .msg-row.user {{ justify-content: flex-end; }}
      .msg-row.assistant, .msg-row.error, .msg-row.thinking {{ justify-content: flex-start; }}
      .msg-card {{
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        gap: 6px;
        max-width: 78%;
      }}
      .msg-row.user .msg-card {{ align-items: flex-end; }}
      .bubble {{
        max-width: 100%;
        border-radius: 12px;
        border: 1px solid var(--border);
        background: #fff;
        color: var(--text);
        padding: 10px 12px;
        font-size: 14px;
        line-height: 1.45;
      }}
      .bubble pre {{
        margin: 8px 0;
        background: #f4f6fc;
        border: 1px solid #dde2f0;
        border-radius: 8px;
        padding: 10px;
        overflow-x: auto;
      }}
      .bubble pre code {{
        white-space: pre;
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
        font-size: 12px;
        line-height: 1.4;
        border: none;
        background: transparent;
        padding: 0;
      }}
      .bubble p {{ margin: 0 0 6px 0; color: inherit; }}
      .bubble p:last-child {{ margin-bottom: 0; }}
      .bubble ul {{ margin: 4px 0 6px 18px; padding: 0; }}
      .bubble li {{ margin: 2px 0; }}
      .bubble code {{
        background: #f0f1f6;
        border: 1px solid #e4e6f0;
        border-radius: 6px;
        padding: 1px 5px;
        font-size: 12px;
      }}
      .msg-row.user .bubble {{
        background: var(--brand);
        border-color: var(--brand);
        color: #fff;
      }}
      .msg-row.error .bubble {{
        border-color: #e9b8b8;
        background: #fff8f8;
      }}
      .msg-actions {{
        display: inline-flex;
        gap: 6px;
      }}
      .msg-copy-btn {{
        border: 1px solid #d5d9e6;
        border-radius: 999px;
        background: #fff;
        color: #3c4458;
        font-size: 11px;
        padding: 4px 10px;
        cursor: pointer;
      }}
      .msg-copy-btn:hover {{ background: #f5f7ff; }}
      .msg-row.thinking .bubble {{
        min-width: 90px;
        background: #f2f3f8;
      }}
      .thinking-dots {{
        display: inline-flex;
        gap: 4px;
        align-items: center;
      }}
      .thinking-dots span {{
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #6f7785;
        display: inline-block;
        animation: pulse 1s infinite ease-in-out;
      }}
      .thinking-dots span:nth-child(2) {{ animation-delay: 0.18s; }}
      .thinking-dots span:nth-child(3) {{ animation-delay: 0.36s; }}
      .sparkle {{
        display: inline-block;
        color: #5e43f3;
        margin-right: 6px;
        font-size: 13px;
      }}
      @keyframes pulse {{
        0%, 80%, 100% {{ opacity: 0.35; transform: translateY(0); }}
        40% {{ opacity: 1; transform: translateY(-1px); }}
      }}
      .chat-input {{ display: flex; gap: 10px; align-items: flex-end; }}
      .chat-input textarea {{
        flex: 1; min-height: 58px; resize: vertical; font-family: inherit; padding: 10px 12px;
        border: 1px solid var(--border);
        border-radius: 12px;
        font-size: 16px;
        outline: none;
      }}
      .chat-input textarea:focus {{ border-color: #c2bae7; box-shadow: 0 0 0 3px rgba(94, 67, 243, 0.12); }}
      iframe {{ width: 100%; height: 380px; border: 1px solid var(--border); border-radius: 10px; }}
      .rerun-placeholder {{
        width: 100%;
        min-height: 220px;
        border: 1px dashed var(--border);
        border-radius: 10px;
        padding: 24px 20px;
        text-align: center;
        color: var(--muted);
        background: #fafbff;
      }}
      .rerun-placeholder strong {{ color: var(--text); }}
      .status-row {{ display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 8px; font-size: 14px; }}
      .btn-row {{ display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 8px; }}
      .btn {{
        border: 1px solid #d6d9e4;
        background: #fff;
        border-radius: 999px;
        color: #2d3342;
        padding: 8px 12px;
        font-size: 13px;
        cursor: pointer;
        touch-action: manipulation;
        -webkit-tap-highlight-color: transparent;
      }}
      .btn:hover {{ background: #f5f6fb; }}
      .btn-primary {{
        background: var(--brand);
        border-color: var(--brand);
        color: #fff;
      }}
      .btn-primary:hover {{ background: var(--brand-strong); }}
      .btn[disabled], .btn:disabled {{
        opacity: 0.65;
        cursor: not-allowed;
      }}
      .cta {{ color: #92400e; background: #fffbeb; border: 1px solid #fcd34d; border-radius: 8px; padding: 8px 10px; }}
      .badge {{ display: inline-block; padding: 3px 9px; border-radius: 999px; background: #ece9ff; color: #33207d; font-size: 12px; }}
      .badge-ok {{ background: var(--ok-bg); color: var(--ok-text); }}
      .run-details {{
        margin-top: 10px;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        background: #f8fafc;
        padding: 10px;
      }}
      .run-details h4 {{ margin: 0 0 8px 0; font-size: 13px; color: #263247; }}
      .run-summary {{
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin-bottom: 8px;
      }}
      .stage-list {{
        display: grid;
        gap: 6px;
        margin: 8px 0;
      }}
      .stage-item {{
        display: grid;
        grid-template-columns: 92px 1fr;
        gap: 8px;
        align-items: start;
        border: 1px solid #e5e7eb;
        background: #fff;
        border-radius: 8px;
        padding: 7px 8px;
        font-size: 12px;
      }}
      .stage-status {{
        border-radius: 999px;
        padding: 3px 7px;
        text-align: center;
        font-weight: 700;
        text-transform: uppercase;
        font-size: 10px;
        background: #eef2ff;
        color: #3730a3;
      }}
      .stage-status.succeeded {{ background: #dcfce7; color: #166534; }}
      .stage-status.failed {{ background: #fee2e2; color: #991b1b; }}
      .stage-status.running {{ background: #fef3c7; color: #92400e; }}
      .stage-status.pending {{ background: #f1f5f9; color: #475569; }}
      .stage-status.not_run {{ background: #f8fafc; color: #64748b; border: 1px solid #cbd5e1; }}
      .stage-label {{ font-weight: 700; color: #263247; }}
      .stage-summary {{ color: #64748b; margin-top: 2px; }}
      .run-log {{
        margin: 8px 0 0 0;
        max-height: 180px;
        overflow: auto;
        white-space: pre-wrap;
        background: #0f172a;
        color: #dbeafe;
        border-radius: 8px;
        padding: 10px;
        font-size: 12px;
      }}
      .actions-inline {{ margin-top: 10px; display:flex; gap:8px; flex-wrap:wrap; }}
      .quick-pill {{
        border-radius: 999px;
        border: 1px solid #c8c0f5;
        background: #f6f4ff;
        color: #3d2f9c;
        font-size: 12px;
        padding: 7px 12px;
      }}
      .quick-pill:hover {{ background: #ede9ff; }}
      .hint {{ font-size: 13px; color: var(--muted); }}
      .status-bar {{
        position: fixed;
        left: 0;
        right: 0;
        bottom: 0;
        z-index: 900;
        padding: 8px 14px;
        font-size: 12px;
        color: #334155;
        background: rgba(255, 255, 255, 0.96);
        border-top: 1px solid var(--border);
        box-shadow: 0 -4px 16px rgba(30, 31, 34, 0.06);
      }}
      .toast-host {{
        position: fixed;
        top: 14px;
        right: 14px;
        z-index: 1000;
        display: flex;
        flex-direction: column;
        gap: 8px;
        max-width: min(420px, calc(100vw - 28px));
        pointer-events: none;
      }}
      .toast {{
        pointer-events: auto;
        padding: 10px 12px;
        border-radius: 10px;
        font-size: 13px;
        border: 1px solid #d6d9e4;
        background: #fff;
        color: #1f2430;
        box-shadow: var(--shadow);
        animation: toast-in 0.18s ease-out;
      }}
      .toast-info {{ border-color: #c8c0f5; background: #f6f4ff; color: #3d2f9c; }}
      .toast-success {{ border-color: #86efac; background: var(--ok-bg); color: var(--ok-text); }}
      .toast-error {{ border-color: #fca5a5; background: #fef2f2; color: #991b1b; }}
      @keyframes toast-in {{
        from {{ opacity: 0; transform: translateY(-6px); }}
        to {{ opacity: 1; transform: translateY(0); }}
      }}
      @media (max-width: 1280px) {{
        .layout-3 {{ grid-template-columns: 1fr; }}
      }}
      @media (max-width: 900px) {{
        .topbar {{
          flex-direction: column;
          align-items: flex-start;
          gap: 8px;
        }}
        .page {{ padding: 12px; gap: 12px; }}
        .panel {{ padding: 12px; }}
        .field-row {{ grid-template-columns: 1fr; }}
        .msg-card {{ max-width: 92%; }}
        iframe {{ height: 300px; }}
      }}
      @media (max-width: 640px) {{
        body {{ padding-bottom: calc(52px + env(safe-area-inset-bottom)); }}
        .chrome {{ min-height: auto; }}
        .brand {{ font-size: 12px; }}
        .brand-sub {{ font-size: 11px; line-height: 1.35; }}
        .chat-input {{
          flex-direction: column;
          align-items: stretch;
        }}
        .chat-input .btn {{ width: 100%; }}
        .btn, .quick-pill {{
          min-height: 44px;
          font-size: 14px;
        }}
        .status-bar {{
          padding-bottom: calc(8px + env(safe-area-inset-bottom));
        }}
      }}
      body.mobile-agent {{
        padding-bottom: calc(56px + env(safe-area-inset-bottom));
        overflow-x: hidden;
      }}
      body.mobile-agent .chrome {{
        max-width: 100%;
        width: 100%;
        overflow-x: hidden;
      }}
      body.mobile-agent .page {{
        display: flex;
        flex-direction: column;
        gap: 10px;
        padding: 10px 10px calc(68px + env(safe-area-inset-bottom));
        width: 100%;
        max-width: 100%;
        overflow-x: hidden;
      }}
      body.mobile-agent .panel {{
        width: 100%;
        max-width: 100%;
        overflow-x: hidden;
      }}
      body.mobile-agent .chat-panel {{
        display: flex;
        flex-direction: column;
        flex: 1 1 auto;
        min-height: calc(100dvh - 112px);
        margin-bottom: 0;
      }}
      body.mobile-agent .chat-panel .hint:first-of-type {{
        display: none;
      }}
      body.mobile-agent .chat-toolbar {{
        flex-direction: column;
        align-items: stretch;
      }}
      body.mobile-agent .chat-model {{
        width: 100%;
        justify-content: space-between;
      }}
      body.mobile-agent .chat-session {{
        width: 100%;
      }}
      body.mobile-agent .chat-model select,
      body.mobile-agent .chat-session select {{
        flex: 1 1 auto;
        max-width: 100%;
      }}
      body.mobile-agent .chat-log {{
        flex: 1 1 auto;
        min-height: 38dvh;
        max-height: 52dvh;
        height: auto;
      }}
      body.mobile-agent .chat-composer {{
        position: sticky;
        bottom: calc(52px + env(safe-area-inset-bottom));
        z-index: 850;
        margin-top: auto;
        padding-top: 10px;
        border-top: 1px solid var(--border);
      }}
      body.mobile-agent .actions-inline {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px;
      }}
      body.mobile-agent .actions-inline .quick-pill {{
        width: 100%;
        justify-content: center;
      }}
      body.mobile-agent .workflow-panel,
      body.mobile-agent .layout-3 {{
        display: none;
      }}
      body.mobile-agent.mobile-show-panels .workflow-panel,
      body.mobile-agent.mobile-show-panels .layout-3 {{
        display: grid;
      }}
      body.mobile-agent .mobile-only-toggle {{
        display: inline-flex;
        flex-shrink: 0;
        min-height: 36px;
      }}
      body.mobile-agent .topbar .badge {{
        display: none;
      }}
      body.mobile-agent .brand-sub {{
        display: none;
      }}
      body.mobile-agent .brand {{
        font-size: 11px;
        line-height: 1.35;
        word-break: break-word;
      }}
      body.mobile-agent .chat-toolbar .hint {{
        display: none;
      }}
      .workflow-panel textarea {{
        width: 100%;
        min-height: 220px;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 12px;
        line-height: 1.45;
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 10px 12px;
        resize: vertical;
        background: #fafbff;
      }}
      .workflow-meta {{
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin: 8px 0 10px;
        font-size: 12px;
      }}
      .workflow-meta .pill {{
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 4px 8px;
        border-radius: 999px;
        background: #f1f5f9;
        border: 1px solid #e2e8f0;
      }}
      .yaml-block {{
        margin: 8px 0;
        padding: 10px 12px;
        border-radius: 8px;
        background: #0f172a;
        color: #e2e8f0;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 12px;
        line-height: 1.45;
        overflow-x: auto;
        white-space: pre-wrap;
      }}
      #workflowYaml {{
        width: 100%;
        min-height: 220px;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 12px;
        border: 1px solid #d4d8e2;
        border-radius: 10px;
        padding: 10px 12px;
        resize: vertical;
        background: #fafbff;
      }}
    </style>
  </head>
  <body>
    <div class="chrome">
      <header class="topbar">
        <div>
          <div class="brand">NEBIUS | NPA WORKBENCH AGENT</div>
          <div class="brand-sub">Sim2Real operations, assets, cameras, and Rerun visualization</div>
        </div>
        <span class="badge badge-ok">Secure basic-auth session</span>
      </header>
      <main class="page">
        <section class="panel chat-panel">
          <div class="chat-panel-head">
            <div>
              <h3>Workbench Chat</h3>
              <p class="hint">Ask about configure, provision, Cosmos3, S3, workflows, sim assets, and Rerun visualization.</p>
            </div>
            <button id="mobilePanelsToggle" class="btn mobile-only-toggle" type="button" aria-expanded="false">Panels</button>
          </div>
          <div class="chat-toolbar">
            <span class="hint">Grounded responses use live `/api/*` context from this VM.</span>
            <label for="chatSessionSelect" class="chat-session">
              Session
              <select id="chatSessionSelect">
                <option value="default" selected>Default chat</option>
              </select>
            </label>
            <button id="newChatSession" class="btn" type="button">New chat</button>
            <label for="chatModel" class="chat-model">
              Token Factory model
              <select id="chatModel">
                <option value="{DEFAULT_LLM_MODEL}" selected>{DEFAULT_LLM_MODEL}</option>
              </select>
            </label>
          </div>
          <div id="mobileChatAuth" class="mobile-chat-auth" aria-live="polite">
            <p class="hint">Mobile chat needs your agent password once on this device (iOS Safari does not send saved login on chat requests).</p>
            <div class="mobile-chat-auth-row">
              <input id="mobileChatPassword" type="password" placeholder="Agent password" autocomplete="current-password">
              <button id="mobileChatAuthBtn" class="btn btn-primary" type="button">Unlock chat</button>
            </div>
          </div>
          <div id="chatLog" class="chat-log"></div>
          <form id="chatForm" class="chat-composer chat-input" autocomplete="off">
            <textarea id="chatInput" placeholder="How do I configure S3 for Sim2Real?" rows="2" enterkeyhint="send"></textarea>
            <button id="chatSend" class="btn btn-primary" type="submit">Send</button>
          </form>
          <div class="actions-inline">
            <button id="chatActionS3" class="btn quick-pill" type="button">Configure S3</button>
            <button id="chatActionCosmos" class="btn quick-pill" type="button">Setup Cosmos3</button>
            <button id="chatActionWatch" class="btn quick-pill" type="button">Watch sim</button>
            <button id="chatActionWorkflow" class="btn quick-pill" type="button">2-step Sim2Real YAML</button>
          </div>
        </section>
        <section class="panel workflow-panel">
          <h3>Workflow YAML</h3>
          <p class="hint">npa.workflow/v0.0.1-beta specs — generate via chat, upload, validate, plan, or submit.</p>
          <div class="workflow-meta">
            <span class="pill">name: <strong id="workflowName">—</strong></span>
            <span class="pill">validation: <strong id="workflowValidation">pending</strong></span>
            <span class="pill">states: <strong id="workflowStates">—</strong></span>
          </div>
          <textarea id="workflowYaml" spellcheck="false" placeholder="Ask chat to create a 2-step Sim2Real workflow, or paste YAML here…"></textarea>
          <div class="btn-row" style="margin-top:8px;">
            <button id="workflowUpload" class="btn" type="button">Upload YAML</button>
            <button id="workflowValidate" class="btn" type="button">Validate</button>
            <button id="workflowPlan" class="btn" type="button">Plan</button>
            <button id="workflowSubmitYaml" class="btn btn-primary" type="button">Submit YAML</button>
          </div>
          <pre id="workflowPlanOutput" class="hint" style="margin-top:8px; white-space:pre-wrap;"></pre>
        </section>
        <section class="panel run-monitor-panel">
          <h3>Sim2Real Run Monitor</h3>
          <p class="hint">Stage timeline, result, and logs for the active run. This panel is independent from Rerun visualization.</p>
          <div id="runDetails" class="run-details">
            <h4>Run status, result, and logs</h4>
            <div id="runSummary" class="run-summary"></div>
            <div id="stageList" class="stage-list"></div>
            <pre id="runLog" class="run-log">No run selected.</pre>
          </div>
        </section>
        <div class="layout layout-3">
          <section class="panel">
            <h3>Sim Assets</h3>
            <div class="subsection">
              <h4>Selection</h4>
              <div class="field-row">
                <div class="field">
                  <label for="sceneMode">Scene mode</label>
                  <select id="sceneMode">
                    <option value="stock" selected>stock</option>
                    <option value="byo_mesh">byo_mesh</option>
                    <option value="scene_spec">scene_spec</option>
                  </select>
                </div>
                <div class="field">
                  <label for="robotPreset">Robot preset</label>
                  <select id="robotPreset">
                    <option value="franka" selected>stock_franka</option>
                    <option value="ur5e">preset:ur5e</option>
                  </select>
                </div>
              </div>
              <div class="field-row" style="margin-top:8px;">
                <div class="field">
                  <label for="cameraMode">Camera mode</label>
                  <select id="cameraMode">
                    <option value="stock" selected>stock</option>
                    <option value="custom">custom</option>
                  </select>
                </div>
                <div class="field">
                  <label for="simBackend">Sim backend</label>
                  <select id="simBackend">
                    <option value="isaac" selected>isaac</option>
                    <option value="genesis">genesis</option>
                  </select>
                </div>
              </div>
            </div>
            <div class="subsection">
              <h4>Props</h4>
              <div class="pill-list">
                <label class="pill"><input id="propCube" type="checkbox" checked> cube</label>
              </div>
            </div>
            <div class="subsection">
              <h4>Resolved assets</h4>
              <div id="assetsSummary" class="hint"></div>
            </div>
            <div class="btn-row">
              <button id="applySelection" class="btn" type="button">Apply stock selection</button>
              <button id="loadFrankaRerun" class="btn" type="button">Load Franka in Rerun (fallback)</button>
              <button id="submitWorkflow" class="btn btn-primary" type="button">Submit Sim2Real</button>
              <button id="workflowStatus" class="btn" type="button">Workflow status</button>
            </div>
            <div class="field-row" style="margin-top:8px;">
              <div class="field">
                <label for="runIdInput">Run ID</label>
                <input id="runIdInput" type="text" placeholder="agent-run-..." />
              </div>
              <div class="field">
                <label for="runIdSelect">Known runs</label>
                <select id="runIdSelect">
                  <option value="">(select run)</option>
                </select>
              </div>
            </div>
            <div class="btn-row" style="margin-top:8px;">
              <button id="loadRunData" class="btn" type="button">Load run data</button>
            </div>
            <div class="subsection" style="margin-top:10px;">
              <h4>Artifact browser</h4>
              <div class="field-row">
                <div class="field">
                  <label for="artifactPrefix">Prefix</label>
                  <input id="artifactPrefix" type="text" placeholder="optional/path/prefix" />
                </div>
                <div class="field">
                  <label for="artifactRunSelect">Discovered runs</label>
                  <select id="artifactRunSelect">
                    <option value="">(select discovered run)</option>
                  </select>
                </div>
              </div>
              <div class="btn-row" style="margin-top:8px;">
                <button id="artifactRefreshRuns" class="btn" type="button">Discover runs</button>
                <button id="artifactLoadRunArtifacts" class="btn" type="button">List artifacts</button>
              </div>
              <div id="artifactList" class="hint" style="margin-top:8px;"></div>
            </div>
          </section>
          <section class="panel cameras-panel">
            <h3>Cameras</h3>
            <p id="cameraRolloutHint" class="rollout-hint">Active for next rollout: <strong id="activeCameraLabel">workspace</strong></p>
            <p class="rollout-hint">Stock workspace and wrist cameras from the Sim2Real default scene spec.</p>
            <div id="cameraCards"></div>
            <select id="cameraSelect" hidden aria-hidden="true"></select>
            <div id="rerunEntityHint" class="rollout-hint"></div>
          </section>
          <section class="panel">
            <h3>Rerun (embedded)</h3>
            <div id="simviz">
              <div class="status-row">
                <span>Run: <strong id="simRunId">—</strong></span>
                <span>Stage: <span id="simStage" class="badge">idle</span></span>
                <span>Camera: <strong id="simCamera">workspace</strong></span>
              </div>
              <div class="btn-row">
                <button id="openRerun" class="btn" type="button">Open in Rerun</button>
              </div>
              <p id="simvizCta" class="cta">Discover artifacts first; use Franka demo fallback when no S3 artifacts are available.</p>
            </div>
            <div id="rerunPlaceholder" class="rerun-placeholder">
              <p>Loading Rerun viewer…</p>
              <p class="hint">Preloading WASM and stock Franka recording from bootstrap cache.</p>
              <button id="loadRerunViewer" class="btn btn-primary" type="button">Load Rerun viewer</button>
            </div>
            <iframe id="rerunFrame" title="rerun" hidden></iframe>
            <div id="artifactPreviewHost" class="hint" hidden style="margin-top:10px;"></div>
          </section>
        </div>
      </main>
    </div>
    <noscript><p style="padding:16px;background:#fff3cd;">JavaScript is required for the NPA Agent workbench. Enable JS and reload.</p></noscript>
    <div id="statusBar" class="status-bar" aria-live="polite">Ready</div>
    <div id="toastHost" class="toast-host" aria-live="polite"></div>
    <script>
      (function initNpaAgentUi() {{
      try {{
      "use strict";
      if (location.username || location.password) {{
        const clean = location.protocol + "//" + location.host + location.pathname + location.search + location.hash;
        history.replaceState(null, "", clean);
      }}
      function detectMobileLayout() {{
        const narrow = window.matchMedia("(max-width: 900px)").matches;
        const coarse = window.matchMedia("(pointer: coarse)").matches;
        const mobileUa = /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent || "");
        return narrow || (coarse && mobileUa);
      }}
      function applyMobileLayout() {{
        if (!detectMobileLayout()) return;
        document.body.classList.add("mobile-agent");
      }}
      let mobileAuthTokenCache = "";
      function mobileAuthHeader() {{
        if (mobileAuthTokenCache) {{
          return mobileAuthTokenCache;
        }}
        try {{
          return String(sessionStorage.getItem("npa_agent_basic_auth") || "").trim();
        }} catch (_err) {{
          return "";
        }}
      }}
      function hasMobileChatAuth() {{
        return Boolean(mobileAuthHeader());
      }}
      function persistMobileBasicAuth(user, pass) {{
        const token = "Basic " + btoa(unescape(encodeURIComponent(String(user || "") + ":" + String(pass || ""))));
        mobileAuthTokenCache = token;
        try {{
          sessionStorage.setItem("npa_agent_basic_auth", token);
        }} catch (_err) {{ /* sessionStorage may be blocked in private browsing */ }}
        return token;
      }}
      function clearMobileBasicAuth() {{
        mobileAuthTokenCache = "";
        try {{
          sessionStorage.removeItem("npa_agent_basic_auth");
        }} catch (_err) {{ /* ignore */ }}
      }}
      function setMobileAuthNeeded(needed) {{
        if (!document.body.classList.contains("mobile-agent")) return;
        document.body.classList.toggle("mobile-needs-auth", Boolean(needed));
        if (!needed) {{
          document.body.classList.add("mobile-auth-ready");
        }}
      }}
      async function verifyMobileChatAuth() {{
        const auth = mobileAuthHeader();
        if (!auth) {{
          return false;
        }}
        try {{
          const resp = await fetch("/api/health", {{
            credentials: "omit",
            cache: "no-store",
            headers: {{ Authorization: auth }},
          }});
          if (resp.status === 401) {{
            clearMobileBasicAuth();
            return false;
          }}
          return resp.ok;
        }} catch (_err) {{
          return false;
        }}
      }}
      async function probeMobileChatAuth() {{
        if (!document.body.classList.contains("mobile-agent")) return true;
        if (!hasMobileChatAuth()) {{
          setMobileAuthNeeded(true);
          return false;
        }}
        const ok = await verifyMobileChatAuth();
        setMobileAuthNeeded(!ok);
        return ok;
      }}
      async function unlockMobileChatAuth(password) {{
        const pass = String(password || "").trim();
        if (!pass) {{
          throw new Error("Enter your agent password.");
        }}
        persistMobileBasicAuth("{DEFAULT_AGENT_USER}", pass);
        const ok = await verifyMobileChatAuth();
        if (!ok) {{
          clearMobileBasicAuth();
          throw new Error("Invalid password — try again or reopen /login-help.html.");
        }}
        setMobileAuthNeeded(false);
        showToast("Chat unlocked", "success");
        return true;
      }}
      function withMobileAuth(headers) {{
        const merged = {{ ...(headers || {{}}) }};
        const auth = mobileAuthHeader();
        if (auth && !merged.Authorization) {{
          merged.Authorization = auth;
        }}
        return merged;
      }}
      applyMobileLayout();
      window.addEventListener("resize", applyMobileLayout);
      const chatHistory = [];
      let activeChatSessionId = "default";
      let chatSendInFlight = false;
      let thinkingNode = null;
      function setStatus(text) {{
        const bar = document.getElementById("statusBar");
        if (bar) bar.textContent = String(text || "");
      }}
      function showToast(message, kind) {{
        const host = document.getElementById("toastHost");
        if (!host) return;
        const toast = document.createElement("div");
        const tone = kind === "error" ? "toast-error" : kind === "success" ? "toast-success" : "toast-info";
        toast.className = "toast " + tone;
        toast.textContent = String(message || "");
        host.appendChild(toast);
        window.setTimeout(() => {{
          if (toast.parentNode) toast.parentNode.removeChild(toast);
        }}, 4200);
      }}
      function bindClick(id, fn, label) {{
        const el = document.getElementById(id);
        if (!el) {{
          console.error("Missing UI control:", id);
          showToast("Missing control: " + id, "error");
          return;
        }}
        el.addEventListener("click", async (event) => {{
          event.preventDefault();
          const actionLabel = String(label || id);
          setStatus(actionLabel + "...");
          showToast(actionLabel, "info");
        try {{
          const result = await fn(event);
          if (result === false) {{
            setStatus("Ready");
            return;
          }}
          setStatus(actionLabel + " done");
          showToast(actionLabel + " done", "success");
        }} catch (err) {{
            const msg = String(err && err.message ? err.message : err);
            setStatus(actionLabel + " failed");
            showToast(msg, "error");
            console.error(actionLabel, err);
          }}
        }});
      }}
      function wireUi() {{
        const chatForm = document.getElementById("chatForm");
        if (chatForm) {{
          chatForm.addEventListener("submit", (event) => {{
            event.preventDefault();
            sendChat().catch((err) => showToast(String(err), "error"));
          }});
        }}
        const mobileToggle = document.getElementById("mobilePanelsToggle");
        if (mobileToggle) {{
          mobileToggle.addEventListener("click", () => {{
            const open = document.body.classList.toggle("mobile-show-panels");
            mobileToggle.setAttribute("aria-expanded", open ? "true" : "false");
            mobileToggle.textContent = open ? "Hide panels" : "Panels";
          }});
        }}
        const mobileAuthBtn = document.getElementById("mobileChatAuthBtn");
        const mobileAuthPass = document.getElementById("mobileChatPassword");
        if (mobileAuthBtn && mobileAuthPass) {{
          mobileAuthBtn.addEventListener("click", async () => {{
            try {{
              mobileAuthBtn.disabled = true;
              await unlockMobileChatAuth(mobileAuthPass.value);
              mobileAuthPass.value = "";
            }} catch (err) {{
              showToast(String(err && err.message ? err.message : err), "error");
            }} finally {{
              mobileAuthBtn.disabled = false;
            }}
          }});
          mobileAuthPass.addEventListener("keydown", async (event) => {{
            if (event.key !== "Enter") return;
            event.preventDefault();
            mobileAuthBtn.click();
          }});
        }}
        bindClick("chatActionS3", () => {{
          setChatInput("Help me configure S3 credentials and bucket for NPA workflows.");
        }}, "Insert S3 prompt");
        bindClick("chatActionCosmos", () => {{
          setChatInput("How do I set up Cosmos3 in the NPA workbench?");
        }}, "Insert Cosmos3 prompt");
        bindClick("chatActionWatch", () => {{
          setChatInput("Watch the sim in Rerun and keep retrying recording+iframe mount until SUCCESS using /api/sim-viz/status.");
        }}, "Insert watch-sim prompt");
        bindClick("chatActionWorkflow", () => {{
          setChatInput("Create a 2-step sim2real workflow YAML with real toolRefs from the catalog.");
        }}, "Insert workflow YAML prompt");
        bindClick("newChatSession", createNewChatSession, "New chat session");
        bindClick("workflowUpload", uploadWorkflowYaml, "Upload workflow YAML");
        bindClick("workflowValidate", validateWorkflowYaml, "Validate workflow YAML");
        bindClick("workflowPlan", planWorkflowYaml, "Plan workflow YAML");
        bindClick("workflowSubmitYaml", submitWorkflowYaml, "Submit workflow YAML");
        bindClick("loadFrankaRerun", loadFrankaDemo, "Load Franka in Rerun");
        bindClick("artifactRefreshRuns", refreshArtifactRuns, "Discover artifact runs");
        bindClick("artifactLoadRunArtifacts", loadArtifactsForSelectedRun, "List run artifacts");
        bindClick("loadRerunViewer", () => loadRerunViewer(), "Load Rerun viewer");
        bindClick("openRerun", openRerunTab, "Open Rerun");
        bindClick("applySelection", applySelection, "Apply stock selection");
        bindClick("submitWorkflow", submitWorkflow, "Submit Sim2Real");
        bindClick("workflowStatus", showWorkflowStatus, "Workflow status");
        bindClick("loadRunData", loadRunData, "Load run data");
        const chatInput = document.getElementById("chatInput");
        if (chatInput) {{
          chatInput.addEventListener("keydown", (e) => {{
            if (e.key === "Enter" && !e.shiftKey) {{
              e.preventDefault();
              sendChat().catch((err) => showToast(String(err), "error"));
            }}
          }});
        }}
        const chatSessionSelect = document.getElementById("chatSessionSelect");
        if (chatSessionSelect) {{
          chatSessionSelect.addEventListener("change", async () => {{
            const sessionId = String(chatSessionSelect.value || "default");
            try {{
              await selectChatSession(sessionId);
            }} catch (err) {{
              showToast(String(err && err.message ? err.message : err), "error");
            }}
          }});
        }}
        const cameraSelect = document.getElementById("cameraSelect");
        if (cameraSelect) {{
          cameraSelect.addEventListener("change", async (e) => {{
            try {{
              await selectCamera(String(e.target.value || ""));
              showToast("Camera selected", "success");
            }} catch (err) {{
              showToast(String(err), "error");
            }}
          }});
        }}
        const runIdSelect = document.getElementById("runIdSelect");
        if (runIdSelect) {{
          runIdSelect.addEventListener("change", () => {{
            const chosen = String(runIdSelect.value || "").trim();
            const input = document.getElementById("runIdInput");
            if (input && chosen) input.value = chosen;
          }});
        }}
        const artifactRunSelect = document.getElementById("artifactRunSelect");
        if (artifactRunSelect) {{
          artifactRunSelect.addEventListener("change", async () => {{
            const selectedRun = String(artifactRunSelect.value || "").trim();
            if (!selectedRun) return;
            await loadArtifactsForSelectedRun();
          }});
        }}
        const robotPreset = document.getElementById("robotPreset");
        if (robotPreset) {{
          robotPreset.addEventListener("change", async (e) => {{
            try {{
              const data = await apiJson("/api/sim-assets/selection", {{
                method: "POST",
                headers: {{ "content-type": "application/json" }},
                body: JSON.stringify(selectionPayloadFromUi()),
              }});
              if (String(e.target.value || "franka") === "franka" && data.sim_viz) {{
                reloadRerunIframe(data.sim_viz.camera || "workspace");
              }}
              await refresh();
              showToast("Robot preset updated", "success");
            }} catch (err) {{
              showToast(String(err), "error");
            }}
          }});
        }}
      }}
      function escapeHtml(text) {{
        return String(text || "")
          .replace(/&/g, "&amp;")
          .replace(/</g, "&lt;")
          .replace(/>/g, "&gt;");
      }}
      function renderInlineMarkdownLite(text) {{
        let value = escapeHtml(text);
        value = value.replace(/`([^`]+)`/g, "<code>$1</code>");
        value = value.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
        return value;
      }}
      function normalizeAssistantReply(text) {{
        const raw = String(text || "").trim();
        if (!raw) return raw;
        if (!/^[\[\{{]/.test(raw)) return raw;
        try {{
          const parsed = JSON.parse(raw);
          if (parsed && typeof parsed === "object") {{
            const lines = [];
            for (const [key, value] of Object.entries(parsed)) {{
              const rendered = (value !== null && typeof value === "object")
                ? "`" + JSON.stringify(value) + "`"
                : "`" + String(value) + "`";
              lines.push("- **" + key + "**: " + rendered);
            }}
            return lines.join("\\n");
          }}
        }} catch (_err) {{
          // Keep original non-JSON text.
        }}
        return raw;
      }}
      function markdownLiteHtml(text) {{
        const lines = String(text || "").split(/\\r?\\n/);
        let html = "";
        let inList = false;
        let listKind = "";
        let inCode = false;
        let codeLang = "";
        let codeLines = [];
        const closeList = () => {{
          if (!inList) return;
          html += listKind === "ol" ? "</ol>" : "</ul>";
          inList = false;
          listKind = "";
        }};
        const closeCode = () => {{
          if (!inCode) return;
          const langAttr = codeLang ? ' data-lang="' + escapeHtml(codeLang) + '"' : "";
          html += "<pre><code" + langAttr + ">" + escapeHtml(codeLines.join("\\n")) + "</code></pre>";
          inCode = false;
          codeLang = "";
          codeLines = [];
        }};
        for (const raw of lines) {{
          const line = String(raw || "");
          const fenceMatch = line.match(/^```([a-zA-Z0-9_-]+)?\s*$/);
          if (fenceMatch) {{
            if (inCode) {{
              closeCode();
            }} else {{
              closeList();
              inCode = true;
              codeLang = String(fenceMatch[1] || "").toLowerCase();
              codeLines = [];
            }}
            continue;
          }}
          if (inCode) {{
            codeLines.push(line);
            continue;
          }}
          if (/^\s*[-*]\s+/.test(line)) {{
            if (!inList || listKind !== "ul") {{
              closeList();
              html += "<ul>";
              inList = true;
              listKind = "ul";
            }}
            html += "<li>" + renderInlineMarkdownLite(line.replace(/^\s*[-*]\s+/, "")) + "</li>";
            continue;
          }}
          if (/^\s*\d+\.\s+/.test(line)) {{
            if (!inList || listKind !== "ol") {{
              closeList();
              html += "<ol>";
              inList = true;
              listKind = "ol";
            }}
            html += "<li>" + renderInlineMarkdownLite(line.replace(/^\s*\d+\.\s+/, "")) + "</li>";
            continue;
          }}
          closeList();
          if (!line.trim()) {{
            continue;
          }}
          html += "<p>" + renderInlineMarkdownLite(line) + "</p>";
        }}
        closeList();
        closeCode();
        return html || "<p></p>";
      }}
      function extractFencedCode(text, preferredLang) {{
        const raw = String(text || "");
        const blocks = [];
        const re = /```([a-zA-Z0-9_-]+)?\s*\\n([\\s\\S]*?)```/g;
        let match;
        while ((match = re.exec(raw)) !== null) {{
          blocks.push({{
            lang: String(match[1] || "").toLowerCase(),
            body: String(match[2] || "").replace(/\s+$/, ""),
          }});
        }}
        if (!blocks.length) return "";
        if (preferredLang) {{
          const target = blocks.find((item) => item.lang === String(preferredLang).toLowerCase());
          if (target) return target.body;
        }}
        return blocks[0].body;
      }}
      async function copyTextToClipboard(text) {{
        const value = String(text || "");
        if (!value) return false;
        if (typeof navigator !== "undefined" && navigator.clipboard && navigator.clipboard.writeText) {{
          await navigator.clipboard.writeText(value);
          return true;
        }}
        const ta = document.createElement("textarea");
        ta.value = value;
        ta.setAttribute("readonly", "readonly");
        ta.style.position = "absolute";
        ta.style.left = "-9999px";
        document.body.appendChild(ta);
        ta.select();
        const ok = document.execCommand("copy");
        document.body.removeChild(ta);
        return Boolean(ok);
      }}
      function appendChat(role, text, opts) {{
        const options = opts || {{}};
        const rawText = String(text || "");
        const log = document.getElementById("chatLog");
        const row = document.createElement("div");
        row.className = "msg-row " + role;
        const card = document.createElement("div");
        card.className = "msg-card";
        const bubble = document.createElement("div");
        bubble.className = "bubble";
        if (options.thinking) {{
          bubble.innerHTML =
            '<span class="sparkle">✦</span><span class="thinking-dots"><span></span><span></span><span></span></span>';
        }} else {{
          bubble.innerHTML = markdownLiteHtml(rawText);
        }}
        card.appendChild(bubble);
        if (!options.thinking && role === "assistant") {{
          const actions = document.createElement("div");
          actions.className = "msg-actions";
          const copyBtn = document.createElement("button");
          copyBtn.type = "button";
          copyBtn.className = "msg-copy-btn";
          const yamlBlock = extractFencedCode(rawText, "yaml");
          const payload = yamlBlock || rawText;
          copyBtn.textContent = yamlBlock ? "Copy YAML" : "Copy";
          copyBtn.addEventListener("click", async () => {{
            try {{
              const ok = await copyTextToClipboard(payload);
              if (!ok) throw new Error("copy failed");
              showToast(yamlBlock ? "YAML copied" : "Message copied", "success");
            }} catch (err) {{
              showToast(String(err && err.message ? err.message : err), "error");
            }}
          }});
          actions.appendChild(copyBtn);
          card.appendChild(actions);
        }}
        row.appendChild(card);
        log.appendChild(row);
        log.scrollTop = log.scrollHeight;
        return row;
      }}
      function showThinkingBubble() {{
        if (thinkingNode) return;
        thinkingNode = appendChat("thinking", "", {{ thinking: true }});
      }}
      function clearThinkingBubble() {{
        if (thinkingNode && thinkingNode.parentNode) {{
          thinkingNode.parentNode.removeChild(thinkingNode);
        }}
        thinkingNode = null;
      }}
      function setChatBusy(isBusy) {{
        const btn = document.getElementById("chatSend");
        const input = document.getElementById("chatInput");
        const model = document.getElementById("chatModel");
        const busy = Boolean(isBusy);
        if (btn) btn.disabled = busy;
        if (input) input.disabled = busy;
        if (model) model.disabled = busy;
      }}
      function setChatModels(models, selectedModel) {{
        const select = document.getElementById("chatModel");
        if (!select) return;
        const values = Array.isArray(models)
          ? [...new Set(models.map((item) => String(item || "").trim()).filter(Boolean))]
          : [];
        if (!values.length) {{
          values.push("{DEFAULT_LLM_MODEL}");
        }}
        const preferred = String(selectedModel || select.value || values[0] || "").trim();
        const chosen = values.includes(preferred) ? preferred : values[0];
        select.innerHTML = "";
        for (const model of values) {{
          const opt = document.createElement("option");
          opt.value = model;
          opt.textContent = model;
          if (model === chosen) opt.selected = true;
          select.appendChild(opt);
        }}
      }}
      function selectedChatModel() {{
        const select = document.getElementById("chatModel");
        return String((select && select.value) || "").trim() || "{DEFAULT_LLM_MODEL}";
      }}
      function clearChatLog() {{
        const log = document.getElementById("chatLog");
        if (log) log.innerHTML = "";
        chatHistory.splice(0, chatHistory.length);
      }}
      function renderChatHistory(history) {{
        clearChatLog();
        const hist = Array.isArray(history) ? history : [];
        for (const msg of hist) {{
          const role = String(msg.role || "");
          const content = String(msg.content || "").trim();
          if (!content || (role !== "user" && role !== "assistant")) continue;
          appendChat(role, content);
          chatHistory.push({{ role, content }});
        }}
      }}
      function updateChatSessionSelector(sessions, activeId) {{
        const select = document.getElementById("chatSessionSelect");
        if (!select) return;
        const rows = Array.isArray(sessions) ? sessions : [];
        const active = String(activeId || activeChatSessionId || "default");
        select.innerHTML = "";
        if (!rows.length) {{
          const opt = document.createElement("option");
          opt.value = active;
          opt.textContent = "Default chat";
          opt.selected = true;
          select.appendChild(opt);
          return;
        }}
        for (const row of rows) {{
          const id = String(row.id || "").trim();
          if (!id) continue;
          const opt = document.createElement("option");
          opt.value = id;
          const count = Number(row.message_count || 0);
          opt.textContent = String(row.title || id) + (count ? " (" + String(count) + ")" : "");
          if (id === active) opt.selected = true;
          select.appendChild(opt);
        }}
      }}
      async function refreshChatSessions(activeId) {{
        const data = await apiJson("/api/chat/sessions");
        activeChatSessionId = String(data.active_session_id || activeId || activeChatSessionId || "default");
        updateChatSessionSelector(data.sessions, activeChatSessionId);
        return data;
      }}
      async function selectChatSession(sessionId) {{
        const safeId = String(sessionId || "default").trim() || "default";
        const data = await apiJson("/api/chat/sessions/" + encodeURIComponent(safeId) + "/select", {{
          method: "POST",
          headers: {{ "content-type": "application/json" }},
          body: JSON.stringify({{}}),
        }});
        const session = data.session || {{}};
        activeChatSessionId = String(data.active_session_id || session.id || safeId);
        updateChatSessionSelector(data.sessions, activeChatSessionId);
        renderChatHistory(session.chat_history || []);
        showToast("Loaded chat session", "success");
        return session;
      }}
      async function createNewChatSession() {{
        const data = await apiJson("/api/chat/sessions", {{
          method: "POST",
          headers: {{ "content-type": "application/json" }},
          body: JSON.stringify({{ title: "New chat" }}),
        }});
        const session = data.session || {{}};
        activeChatSessionId = String(data.active_session_id || session.id || "default");
        updateChatSessionSelector(data.sessions, activeChatSessionId);
        renderChatHistory([]);
        showToast("New chat session ready", "success");
        return true;
      }}
      function setWorkflowYaml(text, validation) {{
        const area = document.getElementById("workflowYaml");
        if (area) area.value = String(text || "");
        updateWorkflowMeta(validation || {{}});
      }}
      function updateWorkflowMeta(validation) {{
        const nameEl = document.getElementById("workflowName");
        const valEl = document.getElementById("workflowValidation");
        const statesEl = document.getElementById("workflowStates");
        const name = String((validation && validation.name) || "—");
        const status = String((validation && validation.status) || (validation && validation.ok ? "valid" : "pending"));
        const states = Array.isArray(validation && validation.states)
          ? validation.states.join(", ")
          : String((validation && validation.states) || "—");
        if (nameEl) nameEl.textContent = name;
        if (valEl) valEl.textContent = status;
        if (statesEl) statesEl.textContent = states || "—";
      }}
      function currentWorkflowYaml() {{
        const area = document.getElementById("workflowYaml");
        return String((area && area.value) || "").trim();
      }}
      async function uploadWorkflowYaml() {{
        const yaml = currentWorkflowYaml();
        if (!yaml) throw new Error("Paste or generate workflow YAML first");
        const data = await apiJson("/api/workflows/draft", {{
          method: "POST",
          headers: {{ "content-type": "application/json" }},
          body: JSON.stringify({{ yaml }}),
        }});
        updateWorkflowMeta((data.validation) || {{}});
        appendChat("assistant", "Uploaded workflow YAML to the **Workflow YAML** panel (`" + String((data.validation && data.validation.name) || "draft") + "`).");
        return true;
      }}
      async function validateWorkflowYaml() {{
        const yaml = currentWorkflowYaml();
        if (!yaml) throw new Error("Paste or generate workflow YAML first");
        const data = await apiJson("/api/workflows/validate", {{
          method: "POST",
          headers: {{ "content-type": "application/json" }},
          body: JSON.stringify({{ yaml }}),
        }});
        updateWorkflowMeta((data.validation) || {{}});
        if (!data.ok) throw new Error(String((data.validation && data.validation.error) || "validation failed"));
        return true;
      }}
      async function planWorkflowYaml() {{
        const yaml = currentWorkflowYaml();
        if (!yaml) throw new Error("Paste or generate workflow YAML first");
        const data = await apiJson("/api/workflows/plan", {{
          method: "POST",
          headers: {{ "content-type": "application/json" }},
          body: JSON.stringify({{ yaml, run_id: "agent-plan" }}),
        }});
        const out = document.getElementById("workflowPlanOutput");
        const steps = Array.isArray(data.plan && data.plan.steps) ? data.plan.steps : [];
        const lines = steps.map((step, idx) =>
          String(idx + 1).padStart(2, "0") + ". " + String(step.state || "?") +
          " toolRef=" + String(step.tool_ref || step.toolRef || "")
        );
        if (out) out.textContent = lines.length ? lines.join("\\n") : JSON.stringify(data.plan || {{}}, null, 2);
        updateWorkflowMeta((data.plan && data.plan.workflow) ? {{ name: data.plan.workflow, status: "planned", states: steps.map((s) => s.state) }} : {{}});
        return true;
      }}
      async function submitWorkflowYaml() {{
        const yaml = currentWorkflowYaml();
        if (!yaml) throw new Error("Paste or generate workflow YAML first");
        const data = await apiJson("/api/workflows/submit", {{
          method: "POST",
          headers: {{ "content-type": "application/json" }},
          body: JSON.stringify({{ yaml }}),
        }});
        updateWorkflowMeta((data.validation) || {{}});
        appendChat(
          "assistant",
          "Submitted npa.workflow YAML — **run_id**: `" + String(data.run_id || "") +
            "`, **mode**: `" + String(data.submit_mode || "") + "`."
        );
        return true;
      }}
      async function sendChat() {{
        const input = document.getElementById("chatInput");
        if (!input) {{
          throw new Error("Chat input missing");
        }}
        if (chatSendInFlight) {{
          return false;
        }}
        if (document.body.classList.contains("mobile-agent") && !hasMobileChatAuth()) {{
          setMobileAuthNeeded(true);
          showToast("Unlock chat with your agent password first.", "error");
          return false;
        }}
        const text = String(input.value || "").trim();
        const model = selectedChatModel();
        if (!text) {{
          showToast("Enter a message first", "info");
          return false;
        }}
        chatSendInFlight = true;
        appendChat("user", text);
        chatHistory.push({{ role: "user", content: text }});
        setChatBusy(true);
        showThinkingBubble();
        try {{
          const data = await apiJson("/api/chat", {{
            method: "POST",
            headers: {{ "content-type": "application/json" }},
            body: JSON.stringify({{ messages: chatHistory, model, session_id: activeChatSessionId }}),
          }});
          clearThinkingBubble();
          input.value = "";
          if (data && data.model) {{
            const select = document.getElementById("chatModel");
            if (select) select.value = String(data.model);
          }}
          if (data && data.session_id) {{
            activeChatSessionId = String(data.session_id);
          }}
          const reply = normalizeAssistantReply(data.reply || "");
          if (reply) {{
            appendChat("assistant", reply);
            chatHistory.push({{ role: "assistant", content: reply }});
            refreshChatSessions(activeChatSessionId).catch(() => {{ /* best-effort session list refresh */ }});
          }} else {{
            appendChat("error", "empty reply from model");
          }}
          if (data.workflow_yaml) {{
            setWorkflowYaml(data.workflow_yaml, data.workflow_validation || {{}});
          }}
          const draft = data.workflow_draft;
          if (!data.workflow_yaml && draft && draft.yaml) {{
            setWorkflowYaml(draft.yaml, draft.validation || draft);
          }}
        }} catch (err) {{
          clearThinkingBubble();
          const message = String(err && err.message ? err.message : err);
          input.value = text;
          const tail = chatHistory[chatHistory.length - 1];
          if (tail && tail.role === "user" && tail.content === text) {{
            chatHistory.pop();
          }}
          appendChat("error", "Send failed; your draft was restored. " + message);
          throw err;
        }} finally {{
          chatSendInFlight = false;
          setChatBusy(false);
          input.focus();
        }}
      }}
      function setChatInput(text) {{
        const input = document.getElementById("chatInput");
        input.value = text;
        input.focus();
      }}
      let lastRrdUpdatedAt = "";
      let rerunIframeLoaded = false;
      let rerunBootInProgress = false;
      let lastRrdBlobUrl = "";
      let activeRunId = "";
      let activeArtifactRender = "";
      let lastRerunBlobStatus = "pending";
      let lastRerunMountStatus = "pending";
      const RERUN_RECORDING_PATH = "/rerun/recordings/sim2real.rrd";
      const RERUN_BUNDLE_ASSETS = ["/rerun/re_viewer.js", "/rerun/re_viewer_bg.wasm"];
      let rerunBundleWarmPromise = null;
      const RERUN_BLOB_SUCCESS = "SUCCESS";
      const RERUN_MOUNT_SUCCESS = "SUCCESS";
      function setRerunBlobStatus(status, detail) {{
        const text = String(status || "").trim() || "pending";
        lastRerunBlobStatus = text;
        const extra = detail ? " (" + String(detail) + ")" : "";
        setStatus("Rerun recording: " + text + extra);
      }}
      function setRerunMountStatus(status, detail) {{
        const text = String(status || "").trim() || "pending";
        lastRerunMountStatus = text;
        const extra = detail ? " (" + String(detail) + ")" : "";
        setStatus("Rerun mount: " + text + extra);
      }}
      async function warmRerunBundle() {{
        if (rerunBundleWarmPromise) {{
          return rerunBundleWarmPromise;
        }}
        rerunBundleWarmPromise = Promise.all(
          RERUN_BUNDLE_ASSETS.map((path) =>
            fetchWithTimeout(
              path,
              {{ credentials: "include", cache: "force-cache" }},
              120000
            ).then((resp) => {{
              if (!resp.ok) {{
                throw new Error("Rerun bundle asset failed: " + path);
              }}
              return resp.blob();
            }})
          )
        ).catch((err) => {{
          rerunBundleWarmPromise = null;
          throw err;
        }});
        return rerunBundleWarmPromise;
      }}
      async function resolveRerunRecordingUrl() {{
        const cacheBust = "?t=" + String(Date.now());
        const recordingUrl = location.origin + RERUN_RECORDING_PATH + cacheBust;
        const resp = await fetchWithTimeout(RERUN_RECORDING_PATH, {{ credentials: "include", method: "HEAD" }}, 8000);
        if (!resp.ok) {{
          throw new Error("Rerun recording not published yet");
        }}
        setRerunBlobStatus(RERUN_BLOB_SUCCESS, "recording=public");
        return recordingUrl;
      }}
      async function resolveRerunRrdUrl(maxAttempts, runId) {{
        const attempts = Math.max(1, Number(maxAttempts || 18));
        const targetRunId = String(runId || activeRunId || "").trim();
        const query = targetRunId ? ("?run_id=" + encodeURIComponent(targetRunId)) : "";
        let lastErr = null;
        for (let i = 0; i < attempts; i += 1) {{
          try {{
            const resp = await fetchWithTimeout("/api/sim-viz/rrd-blob" + query, {{ credentials: "include" }}, 12000);
            if (!resp.ok) {{
              if (resp.status === 401) {{
                window.location.href = "/login-help.html";
              }}
              throw new Error("Failed to fetch .rrd for Rerun viewer");
            }}
            const blob = await resp.blob();
            if (blob.size < 64) {{
              throw new Error("Rerun .rrd payload is too small");
            }}
            if (lastRrdBlobUrl) {{
              URL.revokeObjectURL(lastRrdBlobUrl);
            }}
            lastRrdBlobUrl = URL.createObjectURL(blob);
            setRerunBlobStatus(RERUN_BLOB_SUCCESS, "bytes=" + String(blob.size));
            return lastRrdBlobUrl;
          }} catch (err) {{
            lastErr = err;
            setRerunBlobStatus("retrying", "attempt " + String(i + 1) + "/" + String(attempts));
            if (i + 1 >= attempts) {{
              break;
            }}
            const backoffMs = Math.min(2500, 700 + i * 175);
            await new Promise((resolve) => window.setTimeout(resolve, backoffMs));
          }}
        }}
        setRerunBlobStatus("failed");
        throw lastErr || new Error("Failed to fetch .rrd for Rerun viewer");
      }}
      async function rerunIframeSrc(camera, runId) {{
        const cam = String(camera || "workspace");
        // Rerun's wasm viewer fetches the URL itself and cannot reliably use
        // parent-page basic-auth/blob state. nginx exposes this recording path
        // without auth specifically so the iframe can load visuals directly.
        const rrdUrl = await resolveRerunRecordingUrl();
        return (
          "/rerun/?url=" +
          encodeURIComponent(rrdUrl) +
          "&renderer=webgl&hide_welcome_screen=1&camera=" +
          encodeURIComponent(cam)
        );
      }}
      function showRerunPlaceholder(message) {{
        const placeholder = document.getElementById("rerunPlaceholder");
        const iframe = document.getElementById("rerunFrame");
        if (placeholder) {{
          placeholder.hidden = false;
          if (message) {{
            const hint = placeholder.querySelector(".hint");
            if (hint) hint.textContent = String(message);
          }}
        }}
        if (iframe) {{
          iframe.hidden = true;
          iframe.removeAttribute("src");
        }}
        rerunIframeLoaded = false;
      }}
      function hideRerunPlaceholder() {{
        const placeholder = document.getElementById("rerunPlaceholder");
        const iframe = document.getElementById("rerunFrame");
        if (placeholder) placeholder.hidden = true;
        if (iframe) iframe.hidden = false;
      }}
      function hideArtifactPreview() {{
        const host = document.getElementById("artifactPreviewHost");
        if (!host) return;
        host.hidden = true;
        host.innerHTML = "";
      }}
      async function showArtifactPreview(simViz, render) {{
        const host = document.getElementById("artifactPreviewHost");
        if (!host) return;
        const previewUrl = String((simViz && simViz.artifact_preview_url) || "");
        const downloadUrl = String((simViz && simViz.artifact_download_url) || previewUrl || "");
        const safeRender = String(render || "");
        if (!previewUrl) {{
          host.hidden = false;
          host.innerHTML = "<p>No preview URL available. Use download.</p>";
          return;
        }}
        if (safeRender === "image") {{
          host.hidden = false;
          host.innerHTML = `<img alt="artifact image" src="${{previewUrl}}" style="max-width:100%;border-radius:8px;border:1px solid #dbe2ef;" />`;
          return;
        }}
        if (safeRender === "video") {{
          host.hidden = false;
          host.innerHTML = `<video controls style="max-width:100%;border-radius:8px;border:1px solid #dbe2ef;" src="${{previewUrl}}"></video>`;
          return;
        }}
        if (safeRender === "json" || safeRender === "text") {{
          try {{
            const resp = await fetchWithTimeout(previewUrl, {{ credentials: "include" }}, 12000);
            if (!resp.ok) throw new Error("preview fetch failed");
            const text = await resp.text();
            host.hidden = false;
            host.innerHTML = `<pre style="white-space:pre-wrap;background:#0f172a;color:#e2e8f0;padding:10px;border-radius:8px;max-height:280px;overflow:auto;">${{escapeHtml(text.slice(0, 20000))}}</pre>`;
            return;
          }} catch (_err) {{
            // fall through to download link
          }}
        }}
        host.hidden = false;
        host.innerHTML = `<p>Artifact render: <strong>${{escapeHtml(safeRender || "download")}}</strong>. <a href="${{downloadUrl}}" target="_blank" rel="noopener">Download artifact</a></p>`;
      }}
      async function waitForRerunReady(maxAttempts) {{
        const attempts = Number(maxAttempts || 12);
        for (let i = 0; i < attempts; i += 1) {{
          const simViz = await loadJson("/api/sim-viz/status");
          if (simViz && simViz.rerun_ready) {{
            return simViz;
          }}
          await new Promise((resolve) => window.setTimeout(resolve, 500));
        }}
        throw new Error("Rerun viewer is not ready yet (service or .rrd missing)");
      }}
      async function waitForIframeLoad(iframe, timeoutMs) {{
        const timeout = Math.max(500, Number(timeoutMs || 8000));
        return await new Promise((resolve, reject) => {{
          let done = false;
          const timer = window.setTimeout(() => {{
            if (done) return;
            done = true;
            reject(new Error("Rerun iframe load timed out"));
          }}, timeout);
          function finish(ok, err) {{
            if (done) return;
            done = true;
            window.clearTimeout(timer);
            iframe.removeEventListener("load", onLoad);
            iframe.removeEventListener("error", onError);
            if (ok) {{
              resolve(true);
            }} else {{
              reject(err || new Error("Rerun iframe failed to load"));
            }}
          }}
          function onLoad() {{
            finish(true);
          }}
          function onError() {{
            finish(false, new Error("Rerun iframe error event"));
          }}
          iframe.addEventListener("load", onLoad, {{ once: true }});
          iframe.addEventListener("error", onError, {{ once: true }});
        }});
      }}
      async function mountRerunIframe(camera, runId) {{
        const iframe = document.getElementById("rerunFrame");
        if (!iframe) return true;
        const src = await rerunIframeSrc(camera, runId);
        setRerunMountStatus("retrying", "navigating");
        iframe.src = src;
        hideRerunPlaceholder();
        await waitForIframeLoad(iframe, 12000);
        rerunIframeLoaded = true;
        setRerunMountStatus(RERUN_MOUNT_SUCCESS, "loaded");
        return true;
      }}
      async function mountRerunIframeUntilSuccess(camera, maxAttempts, runId) {{
        const attempts = Math.max(1, Number(maxAttempts || 8));
        let lastErr = null;
        for (let i = 0; i < attempts; i += 1) {{
          try {{
            setRerunBlobStatus("retrying", "mount " + String(i + 1) + "/" + String(attempts));
            setRerunMountStatus("retrying", "mount " + String(i + 1) + "/" + String(attempts));
            await mountRerunIframe(camera, runId);
            if (lastRerunBlobStatus !== RERUN_BLOB_SUCCESS || lastRerunMountStatus !== RERUN_MOUNT_SUCCESS) {{
              throw new Error("Rerun iframe mount missing SUCCESS blob/mount state");
            }}
            return true;
          }} catch (err) {{
            lastErr = err;
            setRerunBlobStatus("retrying", "mount " + String(i + 1) + "/" + String(attempts));
            setRerunMountStatus("retrying", "mount " + String(i + 1) + "/" + String(attempts));
            if (i + 1 >= attempts) {{
              break;
            }}
            await new Promise((resolve) => window.setTimeout(resolve, 1000));
          }}
        }}
        throw lastErr || new Error("Rerun iframe mount did not reach SUCCESS");
      }}
      async function waitForRerunSuccess(camera, options) {{
        const opts = options || {{}};
        const deadlineMs = Math.max(5000, Number(opts.deadlineMs || 120000));
        const sleepMs = Math.max(500, Number(opts.sleepMs || 1200));
        const mountAttemptsPerLoop = Math.max(1, Number(opts.mountAttemptsPerLoop || 4));
        const successStreakTarget = Math.max(1, Number(opts.successStreakTarget || 2));
        const targetRunId = String(opts.runId || "").trim();
        const baselineUpdatedAt = String(opts.baselineRrdUpdatedAt || "").trim();
        const start = Date.now();
        let lastErr = null;
        let successStreak = 0;
        while (Date.now() - start < deadlineMs) {{
          try {{
            const statusPath = targetRunId
              ? "/api/sim-viz/status?run_id=" + encodeURIComponent(targetRunId)
              : "/api/sim-viz/status";
            const status = await loadJson(statusPath);
            const selectedCamera = String((status && status.camera) || camera || "workspace");
            const activeRunId = String((status && status.run_id) || "").trim();
            const activeStage = String((status && status.stage) || "idle").trim().toLowerCase();
            const activeUpdatedAt = String((status && status.rrd_updated_at) || "").trim();
            const runMatches = !targetRunId || (activeRunId && activeRunId === targetRunId);
            const stageAdvanced = !["", "idle", "submitted", "queued"].includes(activeStage);
            const rrdFresh = Boolean(activeUpdatedAt && (!baselineUpdatedAt || activeUpdatedAt !== baselineUpdatedAt));
            if (status && status.rrd_uri && runMatches && (stageAdvanced || rrdFresh)) {{
              await mountRerunIframeUntilSuccess(selectedCamera, mountAttemptsPerLoop, targetRunId);
              if (lastRerunBlobStatus === RERUN_BLOB_SUCCESS && lastRerunMountStatus === RERUN_MOUNT_SUCCESS) {{
                successStreak += 1;
                if (successStreak >= successStreakTarget) {{
                  return status;
                }}
              }} else {{
                successStreak = 0;
              }}
            }} else {{
              successStreak = 0;
            }}
          }} catch (err) {{
            lastErr = err;
            successStreak = 0;
          }}
          await new Promise((resolve) => window.setTimeout(resolve, sleepMs));
        }}
        if (lastErr) {{
          throw lastErr;
        }}
        throw new Error("Timed out waiting for rerun recording/iframe SUCCESS");
      }}
      function reloadRerunIframe(camera) {{
        if (!rerunIframeLoaded) return Promise.resolve();
        return mountRerunIframeUntilSuccess(camera, 6);
      }}
      async function loadRerunViewer(camera) {{
        const cam = String(camera || document.getElementById("cameraSelect").value || "workspace");
        let simViz = await loadJson(activeRunId ? "/api/sim-viz/status?run_id=" + encodeURIComponent(activeRunId) : "/api/sim-viz/status");
        if (!(simViz && (simViz.rrd_uri || simViz.rerun_ready))) {{
          showToast("No run recording yet; loading stock Franka visual fallback in Rerun.", "info");
          await apiJson("/api/sim-viz/load-franka-demo", {{
            method: "POST",
            headers: {{ "content-type": "application/json" }},
            body: JSON.stringify({{ camera: cam }}),
          }});
          activeRunId = "franka-demo";
          simViz = await waitForRerunReady();
        }} else {{
          simViz = await waitForRerunReady();
        }}
        const runId = String((simViz && simViz.run_id) || activeRunId || "").trim();
        if (runId) activeRunId = runId;
        await waitForRerunSuccess(String(simViz.camera || cam), {{ deadlineMs: 90000, mountAttemptsPerLoop: 4, runId }});
        return true;
      }}
      async function pollSimVizUntilRrd(maxAttempts, delayMs, targetRunId) {{
        const attempts = Math.max(1, Number(maxAttempts || 36));
        const sleepMs = Math.max(250, Number(delayMs || 1500));
        const runId = String(targetRunId || "").trim();
        let last = null;
        for (let i = 0; i < attempts; i += 1) {{
          const statusPath = runId
            ? "/api/sim-viz/status?run_id=" + encodeURIComponent(runId)
            : "/api/sim-viz/status";
          const simViz = await loadJson(statusPath);
          last = simViz;
          const activeRunId = String((simViz && simViz.run_id) || "").trim();
          const runMatches = !runId || (activeRunId && activeRunId === runId);
          if (simViz && simViz.rrd_uri && runMatches) {{
            return simViz;
          }}
          await new Promise((resolve) => window.setTimeout(resolve, sleepMs));
        }}
        return last || {{}};
      }}
      async function loadFrankaDemo() {{
        const camera = String(document.getElementById("cameraSelect").value || "workspace");
        const result = await apiJson("/api/sim-viz/load-franka-demo", {{
          method: "POST",
          headers: {{ "content-type": "application/json" }},
          body: JSON.stringify({{ camera }}),
        }});
        await waitForRerunReady();
        await waitForRerunSuccess(camera, {{ deadlineMs: 90000, mountAttemptsPerLoop: 4 }});
        activeArtifactRender = "rerun";
        hideArtifactPreview();
        const simViz = await loadJson("/api/sim-viz/status");
        if (simViz && (simViz.rerun_ready || simViz.rrd_uri)) {{
          const stage = String(simViz.stage || "demo");
          appendChat(
            "assistant",
            "Loaded **stock Franka** demo — **stage**: `" + stage + "`, **camera**: `" + camera + "`, **rerun_ready**: `" +
              String(Boolean(simViz.rerun_ready)) + "`."
          );
        }} else if (result && result.ok === false) {{
          appendChat("error", "Franka demo load did not complete — check Rerun service status.");
        }}
        await refresh();
        return true;
      }}
      async function fetchWithTimeout(path, init, timeoutMs) {{
        const ms = Number(timeoutMs || 12000);
        const ctrl = new AbortController();
        const timer = window.setTimeout(() => ctrl.abort(), ms);
        try {{
          return await fetch(path, {{ ...(init || {{}}), signal: ctrl.signal }});
        }} finally {{
          window.clearTimeout(timer);
        }}
      }}
      async function apiJson(path, init) {{
        const req = init || {{}};
        const isMobile = document.body.classList.contains("mobile-agent");
        const headers = withMobileAuth({{
          ...(req.headers || {{}}),
        }});
        if (isMobile && String(path || "").startsWith("/api/") && !headers.Authorization) {{
          setMobileAuthNeeded(true);
          throw new Error("Unlock chat with your agent password.");
        }}
        const useExplicitAuth = isMobile && Boolean(headers.Authorization);
        const opts = {{
          ...req,
          credentials: useExplicitAuth ? "omit" : "include",
          headers,
        }};
        const pathText = String(path || "");
        const timeoutMs = (
          pathText === "/api/chat" ||
          pathText === "/api/workflows/submit" ||
          pathText === "/api/infra/provision"
        ) ? 900000 : 12000;
        let resp;
        try {{
          resp = await fetchWithTimeout(path, opts, timeoutMs);
        }} catch (err) {{
          if (err && err.name === "AbortError") {{
            throw new Error("Request timed out: " + String(path));
          }}
          throw err;
        }}
        let data = null;
        try {{
          data = await resp.json();
        }} catch (_err) {{
          data = null;
        }}
        if (!resp.ok) {{
          if (resp.status === 401) {{
            if (document.body.classList.contains("mobile-agent")) {{
              setMobileAuthNeeded(true);
              throw new Error("Unlock chat with your agent password.");
            }}
            window.location.href = "/login-help.html";
            throw new Error("Authentication required. Open / and sign in with HTTP Basic Auth.");
          }}
          const detail =
            (data && (data.detail || data.error || data.message)) ||
            resp.statusText ||
            "request failed";
          throw new Error(String(detail));
        }}
        return data;
      }}
      async function loadJson(path) {{
        return await apiJson(path);
      }}
      function artifactPrefixValue() {{
        const input = document.getElementById("artifactPrefix");
        return String((input && input.value) || "").trim();
      }}
      async function refreshArtifactRuns() {{
        const prefix = artifactPrefixValue();
        const query = prefix ? ("?prefix=" + encodeURIComponent(prefix) + "&limit=100") : "?limit=100";
        const data = await apiJson("/api/artifacts/runs" + query);
        const select = document.getElementById("artifactRunSelect");
        if (select) {{
          select.innerHTML = '<option value="">(select discovered run)</option>';
          const runs = Array.isArray(data.runs) ? data.runs : [];
          for (const run of runs) {{
            const runId = String(run.run_id || "").trim();
            if (!runId) continue;
            const opt = document.createElement("option");
            opt.value = runId;
            opt.textContent = runId + (run.has_viewable ? " [viewable]" : " [download]");
            select.appendChild(opt);
          }}
        }}
        const list = document.getElementById("artifactList");
        if (list) {{
          const trunc = data.truncated ? " (truncated)" : "";
          list.textContent = "Runs discovered: " + String(data.total_runs || 0) + trunc;
        }}
        return true;
      }}
      async function loadArtifactsForSelectedRun() {{
        const select = document.getElementById("artifactRunSelect");
        const runId = String((select && select.value) || "").trim();
        if (!runId) throw new Error("Select a discovered run first");
        const prefix = artifactPrefixValue();
        const query = prefix ? ("?prefix=" + encodeURIComponent(prefix)) : "";
        const data = await apiJson("/api/artifacts/run/" + encodeURIComponent(runId) + query);
        const list = document.getElementById("artifactList");
        if (!list) return false;
        const artifacts = Array.isArray(data.artifacts) ? data.artifacts : [];
        if (!artifacts.length) {{
          list.innerHTML = "<p>No artifacts found for this run.</p>";
          return true;
        }}
        list.innerHTML = artifacts.map((item, idx) => {{
          const key = String(item.key || "");
          const render = String(item.render || "download");
          const s3uri = String(item.s3_uri || "");
          return (
            '<div style="padding:8px 0;border-top:1px solid #e2e8f0;">' +
            '<div><code>' + escapeHtml(key) + '</code></div>' +
            '<div style="font-size:12px;color:#64748b;">render=' + escapeHtml(render) + ' size=' + escapeHtml(String(item.size || 0)) + '</div>' +
            '<div style="margin-top:6px;"><button class="btn" type="button" data-action="load-artifact" data-run-id="' + escapeHtml(runId) + '" data-key="' + escapeHtml(key) + '" data-s3-uri="' + escapeHtml(s3uri) + '" data-render="' + escapeHtml(render) + '">Load</button></div>' +
            '</div>'
          );
        }}).join("");
        list.querySelectorAll("button[data-action='load-artifact']").forEach((btn) => {{
          btn.addEventListener("click", async (event) => {{
            event.preventDefault();
            const payload = {{
              run_id: String(btn.getAttribute("data-run-id") || "").trim(),
              key: String(btn.getAttribute("data-key") || "").trim(),
              s3_uri: String(btn.getAttribute("data-s3-uri") || "").trim(),
            }};
            await loadArtifact(payload);
          }});
        }});
        return true;
      }}
      async function loadArtifact(payload) {{
        const body = {{
          run_id: String((payload && payload.run_id) || "").trim(),
          key: String((payload && payload.key) || "").trim(),
          s3_uri: String((payload && payload.s3_uri) || "").trim(),
        }};
        const data = await apiJson("/api/sim-viz/load-artifact", {{
          method: "POST",
          headers: {{ "content-type": "application/json" }},
          body: JSON.stringify(body),
        }});
        const simViz = data.sim_viz || {{}};
        const render = String(data.render || simViz.artifact_render || "");
        activeArtifactRender = render;
        if (render === "rerun") {{
          hideArtifactPreview();
          await waitForRerunReady();
          await waitForRerunSuccess(String(simViz.camera || "workspace"), {{ deadlineMs: 90000, mountAttemptsPerLoop: 4 }});
        }} else {{
          showRerunPlaceholder("Artifact loaded. Use download/preview below.");
          await showArtifactPreview(simViz, render);
        }}
        appendChat(
          "assistant",
          "Loaded artifact `" + String(simViz.artifact_key || body.key || "") + "` with render `" + render + "`."
        );
        await refresh();
        return true;
      }}
      async function refresh() {{
        try {{
          const session = await loadJson("/api/session");
          if (session && session.llm) {{
            const currentModel = String(
              (session.llm.model || session.llm.default_model || session.llm.default || "")
            );
            setChatModels(session.llm.models, currentModel);
          }}
          if (session && session.chat_sessions) {{
            activeChatSessionId = String(session.active_chat_session_id || activeChatSessionId || "default");
            updateChatSessionSelector(session.chat_sessions, activeChatSessionId);
          }}
          if (session && session.workflow_draft && session.workflow_draft.yaml) {{
            setWorkflowYaml(session.workflow_draft.yaml, session.workflow_draft.validation || {{}});
          }}
          const assets = await loadJson("/api/sim-assets");
          const cameras = await loadJson("/api/sim-assets/cameras");
          const statusPath = activeRunId
            ? "/api/sim-viz/status?run_id=" + encodeURIComponent(activeRunId)
            : "/api/sim-viz/status";
          const simViz = await loadJson(statusPath);
          activeRunId = String((simViz && (simViz.active_run_id || simViz.run_id)) || activeRunId || "").trim();
          updateRunSelector(simViz);
          await loadRunDetails(activeRunId);
          renderAssetsSummary(assets);
          document.getElementById("simRunId").textContent = String(simViz.run_id || "-");
          document.getElementById("simStage").textContent = String(simViz.stage || "idle");
          document.getElementById("simCamera").textContent = String(simViz.camera || "workspace");
          activeArtifactRender = String((simViz && simViz.artifact_render) || activeArtifactRender || "");
          const cta = document.getElementById("simvizCta");
          const ready = Boolean(simViz.rerun_ready || simViz.rrd_uri);
          if (cta) {{
            cta.hidden = ready && rerunIframeLoaded;
            if (!ready) {{
              cta.textContent = "No run-specific Rerun recording yet. Use the Run status/logs panel below for stage progress and result.";
            }}
          }}
          if (activeArtifactRender && activeArtifactRender !== "rerun") {{
            showRerunPlaceholder("Non-RRD artifact loaded. Use preview/download below.");
            await showArtifactPreview(simViz, activeArtifactRender);
          }} else {{
            hideArtifactPreview();
          }}
          if (!ready && (!activeArtifactRender || activeArtifactRender === "rerun")) {{
            showRerunPlaceholder("Waiting for .rrd data. Click Load Franka in Rerun or submit Sim2Real.");
          }} else if (!rerunIframeLoaded && !rerunBootInProgress && (!activeArtifactRender || activeArtifactRender === "rerun")) {{
            showRerunPlaceholder("Rerun is ready. Click Load Rerun viewer or Load Franka in Rerun.");
          }}
          const robotPreset = document.getElementById("robotPreset");
          if (assets.selection && assets.selection.robot_preset) {{
            robotPreset.value = String(assets.selection.robot_preset);
          }}
          document.getElementById("simBackend").value = String((assets.selection && assets.selection.sim_backend) || "isaac");
          const props = Array.isArray(assets.selection && assets.selection.props) ? assets.selection.props.map(String) : [];
          document.getElementById("propCube").checked = props.includes("cube");
          const select = document.getElementById("cameraSelect");
          const selected = new Set((cameras.selected || []).map(String));
          const list = Array.isArray(cameras.cameras) ? cameras.cameras : [];
          const activeName = String(
            (selected.size ? [...selected][0] : null) || simViz.camera || "workspace"
          );
          document.getElementById("activeCameraLabel").textContent = activeName;
          select.innerHTML = "";
          for (const cam of list) {{
            const opt = document.createElement("option");
            opt.value = String(cam.name || "");
            opt.textContent = String(cam.name || "");
            if (selected.has(opt.value) || opt.value === activeName) opt.selected = true;
            select.appendChild(opt);
          }}
          renderCameraCards(list, activeName, simViz);
          const updatedAt = String(simViz.rrd_updated_at || "");
          if (rerunIframeLoaded && updatedAt && updatedAt !== lastRrdUpdatedAt) {{
            lastRrdUpdatedAt = updatedAt;
            reloadRerunIframe(simViz.camera || activeName);
          }} else if (updatedAt) {{
            lastRrdUpdatedAt = updatedAt;
          }}
        }} catch (err) {{
          document.getElementById("simvizCta").hidden = false;
          document.getElementById("simvizCta").textContent = "Failed to fetch sim viz status";
        }}
      }}
      function frustumSvg(camera, selected) {{
        const pos = Array.isArray(camera.pos) ? camera.pos : [0, 0, 0];
        const lookAt = Array.isArray(camera.look_at) ? camera.look_at : [0, 1, 0];
        const fov = Number(camera.fov || 60);
        const dx = lookAt[0] - pos[0];
        const dy = lookAt[1] - pos[1];
        const angle = Math.atan2(dy, dx);
        const spread = (fov * Math.PI / 180) / 2;
        const r = 50;
        const cx = 80;
        const cy = 80;
        const p1x = cx + r * Math.cos(angle - spread);
        const p1y = cy + r * Math.sin(angle - spread);
        const p2x = cx + r * Math.cos(angle + spread);
        const p2y = cy + r * Math.sin(angle + spread);
        const stroke = selected ? "#5e43f3" : "#94a3b8";
        const fill = selected ? "rgba(94,67,243,0.18)" : "rgba(148,163,184,0.15)";
        return `<svg width="160" height="160" viewBox="0 0 160 160" role="img" aria-label="camera frustum">
          <circle cx="${{cx}}" cy="${{cy}}" r="4" fill="${{stroke}}"/>
          <polygon points="${{cx}},${{cy}} ${{p1x}},${{p1y}} ${{p2x}},${{p2y}}" fill="${{fill}}" stroke="${{stroke}}"/>
          <text x="8" y="152" font-size="11" fill="#334155">${{String(camera.name || "")}}</text>
        </svg>`;
      }}
      function renderCameraCards(list, activeName, simViz) {{
        const holder = document.getElementById("cameraCards");
        holder.innerHTML = "";
        for (const cam of list) {{
          const name = String(cam.name || "");
          const selected = name === activeName;
          const card = document.createElement("div");
          card.className = "camera-card" + (selected ? " selected" : "");
          const pos = Array.isArray(cam.pos) ? cam.pos.map((v) => Number(v).toFixed(2)).join(", ") : "-";
          const look = Array.isArray(cam.look_at) ? cam.look_at.map((v) => Number(v).toFixed(2)).join(", ") : "-";
          const res = Array.isArray(cam.resolution) ? cam.resolution.join("x") : "640x480";
          card.innerHTML = `
            <h4>` + name + (selected ? ' <span class="badge">selected</span>' : '') + `</h4>
            <div class="camera-meta">placement: ${{String(cam.placement || "custom")}} - fov ${{Number(cam.fov || 60)}}deg - ${{res}}</div>
            <div class="camera-meta">pos [${{pos}}] - look_at [${{look}}]</div>
            <div class="camera-frustum">${{frustumSvg(cam, selected)}}</div>
            <div class="camera-actions">
              <button class="btn" type="button" data-action="select" data-camera="${{name}}">Select</button>
              <button class="btn" type="button" data-action="preview" data-camera="${{name}}">Preview in Rerun</button>
            </div>`;
          holder.appendChild(card);
        }}
        const entity = String(simViz.preview_entity || ("world/cameras/" + activeName));
        const rollout = "rollouts/latest/" + activeName + "/camera";
        document.getElementById("rerunEntityHint").textContent =
          (simViz.rerun_ready || simViz.rrd_uri)
            ? "Rerun entities: " + entity + " (frustum) - " + rollout + " (rollout frames when available)"
            : "Preview in Rerun to log camera frustums; rollout frames appear after Sim2Real runs.";
        holder.querySelectorAll("button[data-action]").forEach((btn) => {{
          btn.addEventListener("click", async (event) => {{
            event.preventDefault();
            const camera = String(btn.getAttribute("data-camera") || "");
            const action = String(btn.getAttribute("data-action") || "");
            const label = action === "select" ? "Select " + camera : "Preview " + camera;
            setStatus(label + "...");
            showToast(label, "info");
            try {{
              if (action === "select") {{
                await selectCamera(camera);
              }} else {{
                await previewCamera(camera);
              }}
              setStatus(label + " done");
              showToast(label + " done", "success");
            }} catch (err) {{
              setStatus(label + " failed");
              showToast(String(err), "error");
            }}
          }});
        }});
      }}
      function renderAssetsSummary(assets) {{
        const selection = (assets && assets.selection) || {{}};
        const resolved = (assets && assets.resolved_uris) || {{}};
        const scene = String(selection.scene_spec_uri || "stock://scene/default");
        const robot = String(selection.robot_spec_uri || "stock://robot/franka");
        const cameras = String(selection.cameras_uri || "stock://cameras/default");
        const backend = String(selection.sim_backend || "isaac");
        const props = Array.isArray(selection.props) ? selection.props : [];
        const propsText = props.length ? props.join(", ") : "none";
        document.getElementById("assetsSummary").innerHTML =
          "<div><span class='pill'>scene: " + escapeHtml(scene) + "</span></div>" +
          "<div style='margin-top:6px;'><span class='pill'>robot: " + escapeHtml(robot) + "</span></div>" +
          "<div style='margin-top:6px;'><span class='pill'>cameras: " + escapeHtml(cameras) + "</span></div>" +
          "<div style='margin-top:6px;'><span class='pill'>backend: " + escapeHtml(backend) + "</span></div>" +
          "<div style='margin-top:6px;'>props: " + escapeHtml(propsText) + "</div>" +
          "<div style='margin-top:6px;'>resolved scene URI: <code>" + escapeHtml(String(resolved.scene_spec_uri || scene)) + "</code></div>";
      }}
      function selectionPayloadFromUi() {{
        const robotPreset = String(document.getElementById("robotPreset").value || "franka");
        const sceneMode = String(document.getElementById("sceneMode").value || "stock");
        const cameraMode = String(document.getElementById("cameraMode").value || "stock");
        const backend = String(document.getElementById("simBackend").value || "isaac");
        return {{
          scene_spec_uri: sceneMode === "stock" ? "stock://scene/default" : "",
          robot_spec_uri: robotPreset === "franka" ? "stock://robot/franka" : "",
          cameras_uri: cameraMode === "stock" ? "stock://cameras/default" : "",
          robot_preset: robotPreset,
          sim_backend: backend,
          props: document.getElementById("propCube").checked ? ["cube"] : []
        }};
      }}
      function updateRunSelector(simViz) {{
        const select = document.getElementById("runIdSelect");
        if (!select) return;
        const current = String((simViz && simViz.run_id) || "").trim();
        const runs = Array.isArray(simViz && simViz.available_run_ids) ? simViz.available_run_ids.map(String) : [];
        select.innerHTML = '<option value="">(select run)</option>';
        for (const runId of runs) {{
          const opt = document.createElement("option");
          opt.value = runId;
          opt.textContent = runId;
          if (runId === current) opt.selected = true;
          select.appendChild(opt);
        }}
        const input = document.getElementById("runIdInput");
        if (input && current) input.value = current;
      }}
      function normalizeStageStatus(value) {{
        const raw = String(value || "pending").trim().toLowerCase();
        if (["succeeded", "success", "done", "complete", "completed"].includes(raw)) return "succeeded";
        if (["failed", "error", "blocked"].includes(raw)) return "failed";
        if (["running", "active", "submitted", "queued"].includes(raw)) return raw === "submitted" ? "running" : raw;
        if (["not_run", "not-run", "skipped", "not launched", "not_launched"].includes(raw)) return "not_run";
        return "pending";
      }}
      function renderRunDetails(details) {{
        const run = (details && details.run) || details || {{}};
        const summary = document.getElementById("runSummary");
        const stagesHost = document.getElementById("stageList");
        const logHost = document.getElementById("runLog");
        if (!summary || !stagesHost || !logHost) return;
        const runId = String(run.run_id || "");
        const result = String(run.result || "pending");
        const status = String(run.status || "idle");
        const updatedAt = String(run.updated_at || run.submitted_at || "");
        summary.innerHTML =
          '<span class="pill">run: <strong>' + escapeHtml(runId || "none") + '</strong></span>' +
          '<span class="pill">status: <strong>' + escapeHtml(status) + '</strong></span>' +
          '<span class="pill">result: <strong>' + escapeHtml(result) + '</strong></span>' +
          '<span class="pill">updated: <strong>' + escapeHtml(updatedAt || "-") + '</strong></span>';
        const stages = Array.isArray(run.stages) ? run.stages : [];
        stagesHost.innerHTML = stages.map((stage) => {{
          const statusClass = normalizeStageStatus(stage.status);
          const label = String(stage.label || stage.id || "");
          const stageSummary = String(stage.summary || "");
          return (
            '<div class="stage-item">' +
            '<span class="stage-status ' + escapeHtml(statusClass) + '">' + escapeHtml(statusClass) + '</span>' +
            '<div><div class="stage-label">' + escapeHtml(label) + '</div>' +
            '<div class="stage-summary">' + escapeHtml(stageSummary || String(stage.id || "")) + '</div></div>' +
            '</div>'
          );
        }}).join("") || '<div class="hint">No stage data available yet.</div>';
        const logs = Array.isArray(run.logs) ? run.logs : [];
        logHost.textContent = logs.length
          ? logs.map((entry) => {{
              const ts = String(entry.timestamp || "");
              const level = String(entry.level || "info").toUpperCase();
              const message = String(entry.message || "");
              return "[" + ts + "] " + level + " " + message;
            }}).join("\\n")
          : "No log entries yet.";
      }}
      async function loadRunDetails(runId) {{
        const target = String(runId || activeRunId || "").trim();
        const path = target
          ? "/api/workflows/sim2real/runs/" + encodeURIComponent(target)
          : "/api/workflows/sim2real/status";
        try {{
          const data = await loadJson(path);
          renderRunDetails(data);
          return data.run || data;
        }} catch (err) {{
          renderRunDetails({{ run: {{ run_id: target, status: "unknown", result: "unavailable", logs: [{{ timestamp: new Date().toISOString(), level: "error", message: String(err && err.message ? err.message : err) }}], stages: [] }} }});
          return null;
        }}
      }}
      async function loadRunData() {{
        const input = document.getElementById("runIdInput");
        const runId = String((input && input.value) || "").trim();
        if (!runId) {{
          throw new Error("Enter or select a run_id first");
        }}
        const data = await apiJson("/api/sim-viz/load-run", {{
          method: "POST",
          headers: {{ "content-type": "application/json" }},
          body: JSON.stringify({{ run_id: runId }}),
        }});
        activeRunId = runId;
        appendChat("assistant", "Loaded run context — **run_id**: `" + runId + "`.");
        await loadRunDetails(runId);
        if (data && data.sim_viz && (data.sim_viz.rrd_uri || data.sim_viz.rerun_ready)) {{
          await waitForRerunSuccess(String(data.sim_viz.camera || "workspace"), {{ runId }});
        }} else {{
          showRerunPlaceholder("No .rrd recording for this run yet. See run stages and logs below.");
        }}
        await refresh();
      }}
      async function selectCamera(camera) {{
        const selected = String(camera || "");
        await apiJson("/api/sim-assets/cameras/selection", {{
          method: "PUT",
          headers: {{ "content-type": "application/json" }},
          body: JSON.stringify({{ selected: selected ? [selected] : [] }}),
        }});
        await apiJson("/api/sim-assets/selection", {{
          method: "POST",
          headers: {{ "content-type": "application/json" }},
          body: JSON.stringify(selectionPayloadFromUi()),
        }});
        await refresh();
      }}
      async function previewCamera(camera) {{
        const data = await apiJson("/api/sim-viz/camera-preview", {{
          method: "POST",
          headers: {{ "content-type": "application/json" }},
          body: JSON.stringify({{ camera }}),
        }});
        await waitForRerunReady();
        await waitForRerunSuccess(camera, {{ deadlineMs: 90000, mountAttemptsPerLoop: 4 }});
        const entity = String(data.entity_path || ("world/cameras/" + camera));
        appendChat("assistant", "Previewing `" + camera + "` in Rerun at `" + entity + "`.");
        await refresh();
      }}
      async function applySelection() {{
        const data = await apiJson("/api/sim-assets/selection", {{
          method: "POST",
          headers: {{ "content-type": "application/json" }},
          body: JSON.stringify(selectionPayloadFromUi()),
        }});
        if (data.sim_viz && rerunIframeLoaded) {{
          reloadRerunIframe(data.sim_viz.camera || "workspace");
        }}
        await refresh();
      }}
      async function submitWorkflow() {{
        const baseline = await loadJson("/api/sim-viz/status");
        const baselineUpdatedAt = String((baseline && baseline.rrd_updated_at) || "").trim();
        const data = await apiJson("/api/workflows/sim2real/submit", {{
          method: "POST",
          headers: {{ "content-type": "application/json" }},
          body: JSON.stringify({{}}),
        }});
        appendChat("assistant", `Submitted Sim2Real run: **${{data.run_id || "unknown"}}**`);
        appendChat("assistant", "Watching sim progress: rendering stage/result/logs immediately; Rerun opens only after a run-specific `.rrd` is available.");
        const submittedRunId = String(data.run_id || "").trim();
        if (submittedRunId) activeRunId = submittedRunId;
        renderRunDetails(data);
        const simViz = await pollSimVizUntilRrd(8, 1500, submittedRunId);
        if (simViz && simViz.rrd_uri) {{
          await waitForRerunSuccess(
            simViz.camera || "workspace",
            {{
              deadlineMs: 180000,
              mountAttemptsPerLoop: 5,
              runId: submittedRunId,
              baselineRrdUpdatedAt: baselineUpdatedAt,
            }}
          );
          if (lastRerunBlobStatus !== RERUN_BLOB_SUCCESS || lastRerunMountStatus !== RERUN_MOUNT_SUCCESS) {{
            throw new Error("Rerun recording/iframe did not reach SUCCESS after workflow submit");
          }}
          appendChat(
            "assistant",
            "Rerun update: **run_id** `" +
              String(simViz.run_id || data.run_id || "unknown") +
              "`, **stage** `" +
              String(simViz.stage || "running") +
              "`, **iframe** `/rerun/`, **blob_mount** `" + RERUN_BLOB_SUCCESS + "`"
          );
        }} else {{
          await loadRunDetails(submittedRunId);
          showRerunPlaceholder("No run-specific Rerun recording yet. Stage timeline, result, and logs are shown below.");
          appendChat(
            "assistant",
            "Run `" + submittedRunId + "` is recorded with **stage** `" +
              String((simViz && simViz.stage) || "submitted") +
              "`. No `.rrd` recording is available yet, so the Run status/logs panel is the source of truth."
          );
        }}
        await refresh();
      }}
      async function showWorkflowStatus() {{
        const status = await loadJson("/api/workflows/sim2real/status");
        renderRunDetails(status);
        appendChat(
          "assistant",
          "Latest workflow status:\\n- run_id: `" +
            String((status.latest_submit || {{}}).run_id || "none") +
            "`\\n- stage: `" +
            String((status.sim_viz || {{}}).stage || "idle") +
            "`"
        );
      }}
      async function openRerunTab() {{
        const statusPath = activeRunId
          ? "/api/sim-viz/status?run_id=" + encodeURIComponent(activeRunId)
          : "/api/sim-viz/status";
        const simViz = await loadJson(statusPath);
        if (!(simViz && (simViz.rrd_uri || simViz.rerun_ready))) {{
          await loadRunDetails(activeRunId);
          showRerunPlaceholder("No run-specific Rerun recording yet. Stage timeline, result, and logs are shown below.");
          showToast("No Rerun recording for this run yet; showing run logs instead.", "info");
          return;
        }}
        const camera = String(simViz.camera || document.getElementById("cameraSelect").value || "workspace");
        const src = await rerunIframeSrc(camera, String((simViz && simViz.run_id) || activeRunId || "").trim());
        window.open(src, "_blank", "noopener");
      }}
      async function restoreSession() {{
        try {{
          const session = await loadJson("/api/session");
          activeChatSessionId = String(session.active_chat_session_id || activeChatSessionId || "default");
          updateChatSessionSelector(session.chat_sessions, activeChatSessionId);
          if (session && session.llm) {{
            const currentModel = String(
              (session.llm.model || session.llm.default_model || session.llm.default || "")
            );
            setChatModels(session.llm.models, currentModel);
          }}
          renderChatHistory(session.chat_history || []);
          const draft = session.workflow_draft;
          if (draft && draft.yaml) {{
            setWorkflowYaml(draft.yaml, draft.validation || draft);
          }}
        }} catch (_err) {{
          // Session restore is best-effort on first load.
        }}
      }}
      async function ensureFrankaRerunLoaded() {{
        const simViz = await loadJson("/api/sim-viz/status");
        const artifactRender = String((simViz && simViz.artifact_render) || "");
        if (artifactRender && artifactRender !== "rerun") {{
          activeArtifactRender = artifactRender;
          await showArtifactPreview(simViz, artifactRender);
          return;
        }}
        const camera = String(simViz.camera || "workspace");
        if (!simViz.rerun_ready && !simViz.rrd_uri) {{
          setStatus("Loading Franka demo...");
          showToast("Loading stock Franka in Rerun", "info");
          await loadFrankaDemo();
          return;
        }}
        if (!rerunIframeLoaded) {{
          setStatus("Opening Rerun viewer...");
          await loadRerunViewer(camera);
        }}
      }}
      async function bootPage() {{
        rerunBootInProgress = true;
        showRerunPlaceholder("Loading stock Franka preview...");
        setStatus("Preloading Rerun…");
        try {{
          await warmRerunBundle();
        }} catch (warmErr) {{
          console.warn("rerun bundle warm failed", warmErr);
        }}
        try {{
          await restoreSession();
        }} catch (_err) {{
          // Session restore is best-effort on first paint.
        }}
        try {{
          await refresh();
          try {{
            await refreshArtifactRuns();
          }} catch (_artifactErr) {{
            // artifact discovery is optional when S3 credentials are missing.
          }}
          if (document.body.classList.contains("mobile-agent")) {{
            setStatus("Ready");
            showToast("Mobile chat ready", "success");
          }} else {{
            await ensureFrankaRerunLoaded();
            setStatus("Ready");
            showToast("Franka demo ready in Rerun", "success");
          }}
        }} catch (err) {{
          console.warn("franka auto-load failed", err);
          if (document.body.classList.contains("mobile-agent")) {{
            setStatus("Ready");
          }} else {{
            showRerunPlaceholder("Could not auto-load Franka. Click Load Franka in Rerun.");
            showToast(String(err && err.message ? err.message : err), "error");
            setStatus("Ready");
          }}
        }} finally {{
          rerunBootInProgress = false;
        }}
      }}
      let refreshTimer = null;
      function startPeriodicRefresh() {{
        if (refreshTimer !== null) return;
        refreshTimer = window.setInterval(() => {{
          refresh().catch(() => {{ /* periodic refresh is best-effort */ }});
        }}, 10000);
      }}
      function startApp() {{
        try {{
          wireUi();
          setStatus("UI wired");
        }} catch (err) {{
          showToast("UI wiring failed: " + String(err), "error");
          console.error(err);
        }}
        probeMobileChatAuth()
          .catch(() => false)
          .finally(() => {{
            bootPage().catch((err) => {{
              showToast("Boot failed: " + String(err), "error");
              console.error(err);
            }});
          }});
        const armPeriodic = () => startPeriodicRefresh();
        document.addEventListener("click", armPeriodic, {{ once: true }});
        document.addEventListener("keydown", armPeriodic, {{ once: true }});
      }}
      if (document.readyState === "loading") {{
        document.addEventListener("DOMContentLoaded", startApp);
      }} else {{
        startApp();
      }}
      }} catch (bootErr) {{
        console.error("NPA Agent UI failed to boot", bootErr);
        const bar = document.getElementById("statusBar");
        if (bar) bar.textContent = "UI boot error — reload or check console";
      }}
      }})();
    </script>
  </body>
</html>
HTML
sudo python3 -m venv /opt/npa-agent/venv
sudo /opt/npa-agent/venv/bin/pip install --upgrade pip
sudo /opt/npa-agent/venv/bin/pip install fastapi uvicorn httpx pyyaml boto3 "rerun-sdk>=0.32"
sudo /opt/npa-agent/venv/bin/pip install -e "{AGENT_SOURCE_ROOT}/npa[server]"
sudo /opt/npa-agent/venv/bin/python /opt/npa-agent/bootstrap_rrd.py
sudo systemctl restart npa-rerun || true
cat <<'UNIT' | sudo tee /etc/systemd/system/npa-agent-backend.service >/dev/null
[Unit]
Description=NPA agent backend
After=network.target
[Service]
Type=simple
EnvironmentFile=-/opt/npa-agent/llm.env
EnvironmentFile=-/opt/npa-agent/s3.env
EnvironmentFile=-/opt/npa-agent/nebius.env
ExecStart=/opt/npa-agent/venv/bin/uvicorn backend:app --host 0.0.0.0 --port {backend_port}
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
ExecStart=/opt/npa-agent/venv/bin/rerun /opt/npa-agent/sim2real.rrd --serve-web --web-viewer --bind 0.0.0.0 --web-viewer-port {rerun_port} --port 9876
WorkingDirectory=/opt/npa-agent
Restart=always
StartLimitIntervalSec=0
[Install]
WantedBy=multi-user.target
UNIT
sudo htpasswd -bc /etc/nginx/.npa-agent-htpasswd {shlex.quote(auth_user)} {shlex.quote(auth_password)}
{https_ssl_setup}
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
"""
    setup_script = (
        setup_script.replace(_AGENT_CHAT_EMBED, agent_chat_source)
        .replace(_AGENT_WORKFLOW_EMBED, agent_workflow_source)
        .replace(_AGENT_ARTIFACTS_EMBED, agent_artifacts_source)
    )
    local_setup_script = ""
    # Use a unique remote path so concurrent bootstrap runs cannot clobber each other.
    remote_setup_script = f"/tmp/npa-agent-bootstrap-{secrets.token_hex(6)}.sh"
    try:
        _stage_agent_npa_source(ssh)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
            handle.write(setup_script)
            local_setup_script = handle.name
        ssh.upload_file(local_setup_script, remote_setup_script)
        ssh.run_or_raise(f"chmod 700 {shlex.quote(remote_setup_script)} && {shlex.quote(remote_setup_script)}")
    finally:
        if local_setup_script:
            Path(local_setup_script).unlink(missing_ok=True)
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
    agent_iam_token = ""
    try:
        from npa.clients.nebius import get_iam_token

        agent_iam_token = get_iam_token()
    except Exception:
        agent_iam_token = ""
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
        iam_token=agent_iam_token,
    )
    if (
        tf_api_key.strip()
        or (s3_bucket.strip() and s3_access_key.strip() and s3_secret_key.strip())
        or ((nebius_project_id or project_id).strip() and s3_access_key.strip() and s3_secret_key.strip())
    ):
        ssh.run_or_raise(
            "sudo systemctl reset-failed npa-agent-backend || true; "
            "sudo systemctl restart npa-agent-backend"
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


@app.command("deploy")
def deploy_cmd(
    project: str = typer.Option(DEFAULT_PROJECT_ALIAS, "--project", help="NPA project alias to store config under."),
    name: str = typer.Option(DEFAULT_AGENT_NAME, "--name", help="Agent deployment name."),
    project_id: str = typer.Option("", "--project-id", help="Nebius project ID."),
    tenant_id: str = typer.Option("", "--tenant-id", help="Nebius tenant ID."),
    region: str = typer.Option("us-central1", "--region", help="Nebius region."),
    ssh_user: str = typer.Option("ubuntu", "--ssh-user", help="SSH username."),
    ssh_public_key_path: str = typer.Option("~/.ssh/id_ed25519.pub", "--ssh-public-key-path", help="SSH public key path for Terraform."),
    tf_var: list[str] = typer.Option([], "--tf-var", help="Additional Terraform var key=value."),
    agent_port: int = typer.Option(DEFAULT_AGENT_PORT, "--agent-port", help="Public agent UI port."),
    backend_port: int = typer.Option(DEFAULT_BACKEND_PORT, "--backend-port", help="Internal agent backend port."),
    rerun_port: int = typer.Option(DEFAULT_RERUN_PORT, "--rerun-port", help="Rerun service port."),
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
) -> None:
    """Provision VM + bootstrap the public NPA agent stack."""
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

    from npa.clients.nebius import NebiusError, bootstrap_agent_environment, get_iam_token

    try:
        creds = bootstrap_agent_environment(
            env_project_id,
            env_tenant_id,
            env_region,
            on_status=lambda msg: typer.echo(f"  {msg}"),
        )
        creds = _resolve_deploy_storage_credentials(
            region=env_region,
            bootstrap_creds=creds,
        )
        iam_token = get_iam_token()
    except NebiusError as exc:
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
        "workbench_type": "lerobot",
        "gpu_platform": "cpu-d3",
        "gpu_preset": "8vcpu-32gb",
        "ssh_user": ssh_user,
        "ssh_public_key_path": ssh_public_key_path,
        "enable_preemptible": "false",
    }
    for item in tf_var:
        if "=" not in item:
            _fail(f"Invalid --tf-var value {item!r}; expected key=value")
        key, value = item.split("=", 1)
        merged_vars[key.strip()] = value.strip()
    try:
        _ensure_terraform_state_bucket(
            project_id=env_project_id,
            bucket_name=str(merged_vars.get("s3_bucket", "")),
        )
    except NebiusError as exc:
        _fail(f"Unable to provision Terraform state bucket: {exc}")

    tf_outputs: dict[str, Any] = {}
    try:
        tf_outputs = _apply_agent_terraform(
            project=project,
            name=name,
            merged_vars=merged_vars,
            env_region=env_region,
        )
    except ProvisionerError as exc:
        try:
            _destroy_agent_terraform(project, name)
        except ProvisionerError as cleanup_exc:
            typer.echo(f"  Warning: terraform rollback failed: {cleanup_exc}", err=True)
        _fail(f"Terraform deploy failed: {exc}")

    public_ip = str(tf_outputs.get("vm_ip", ""))
    instance_id = str(tf_outputs.get("instance_id", ""))
    ssh_key_path = str(tf_outputs.get("ssh_key_path", "") or ssh_public_key_path.removesuffix(".pub"))
    if not _is_routable_public_ip(public_ip):
        try:
            _destroy_agent_terraform(
                project,
                name,
                record={"instance_id": instance_id, "project_id": env_project_id, "region": env_region},
            )
        except ProvisionerError as cleanup_exc:
            typer.echo(f"  Warning: terraform rollback failed: {cleanup_exc}", err=True)
        _fail("Terraform output did not include a routable public IP")

    auth_password = secrets.token_urlsafe(18)
    auth_path = _write_auth_secret(
        project_alias=project,
        name=name,
        user=DEFAULT_AGENT_USER,
        password=auth_password,
    )
    tf_api_key, default_llm_model = _resolve_deploy_llm_credentials()
    configured_llm_model = str(llm_model or "").strip() or default_llm_model
    configured_llm_models = _normalize_llm_models([configured_llm_model, *llm_models])
    nebius_ai_key, _ = _resolve_operator_credentials()
    if not tf_api_key:
        typer.echo(
            "Warning: Token Factory API key not found in credentials; "
            "agent chat will return 503 until `npa agent bootstrap` with a configured key.",
            err=True,
        )
    rollback_record = {
        "instance_id": instance_id,
        "project_id": env_project_id,
        "region": env_region,
        "service_account_id": str(creds.get("service_account_id", "")),
    }
    try:
        _bootstrap_agent_stack(
            host=public_ip,
            ssh_user=ssh_user,
            ssh_key_path=ssh_key_path,
            project_alias=project,
            agent_name=name,
            project_id=env_project_id,
            tenant_id=env_tenant_id,
            region=env_region,
            auth_user=DEFAULT_AGENT_USER,
            auth_password=auth_password,
            agent_port=agent_port,
            backend_port=backend_port,
            rerun_port=rerun_port,
            llm_model=configured_llm_model,
            llm_models=configured_llm_models,
            tf_api_key=tf_api_key,
            nebius_ai_key=nebius_ai_key,
            s3_bucket=str(merged_vars.get("s3_bucket", "")),
            s3_prefix=str(merged_vars.get("s3_prefix", "")),
            s3_endpoint=str(merged_vars.get("s3_endpoint", "")),
            s3_access_key=str(merged_vars.get("nebius_api_key", "")),
            s3_secret_key=str(merged_vars.get("nebius_secret_key", "")),
            s3_region=env_region,
            nebius_project_id=env_project_id,
            nebius_tenant_id=env_tenant_id,
            service_account_id=str(creds.get("service_account_id", "")),
            public_https=public_https,
        )
    except (ConfigError, SSHError, ValueError) as exc:
        try:
            _destroy_agent_terraform(project, name, record=rollback_record)
        except ProvisionerError as cleanup_exc:
            typer.echo(f"  Warning: terraform rollback failed: {cleanup_exc}", err=True)
        _fail(f"VM bootstrap failed: {exc}")

    ingress_ports: list[int] = [agent_port, rerun_port]
    if public_https:
        ingress_ports.append(DEFAULT_HTTPS_PORT)
    try:
        ensure_ingress(vm_id=instance_id, ports=tuple(ingress_ports), tool="agent")
    except NetworkIngressError as exc:
        try:
            _destroy_agent_terraform(project, name, record=rollback_record)
        except ProvisionerError as cleanup_exc:
            typer.echo(f"  Warning: terraform rollback failed: {cleanup_exc}", err=True)
        _fail(f"npa network ensure-ingress failed: {exc}")

    urls = build_agent_urls(public_ip, agent_port=agent_port, public_https=public_https)
    agent_credentials = _agent_credentials_payload(creds)
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
        credentials=agent_credentials,
    )
    _store_agent_record(project, name, record.to_dict())
    write_config(
        {
            "projects": {
                project: {
                    "project_id": env_project_id,
                    "tenant_id": env_tenant_id,
                    "region": env_region,
                    "terraform_state": {
                        "bucket": merged_vars.get("s3_bucket", ""),
                        "endpoint": merged_vars.get("s3_endpoint", ""),
                        "access_key": merged_vars.get("nebius_api_key", ""),
                        "secret_key": merged_vars.get("nebius_secret_key", ""),
                    },
                }
            }
        }
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
def fresh_setup_cmd(
    project: str = typer.Option(DEFAULT_PROJECT_ALIAS, "--project", help="NPA project alias for this fresh environment."),
    name: str = typer.Option(DEFAULT_AGENT_NAME, "--name", help="Agent deployment name."),
    project_id: str = typer.Option(..., "--project-id", help="Nebius project ID."),
    tenant_id: str = typer.Option(..., "--tenant-id", help="Nebius tenant ID."),
    region: str = typer.Option("us-central1", "--region", help="Nebius region."),
    ssh_user: str = typer.Option("ubuntu", "--ssh-user", help="SSH username."),
    ssh_public_key_path: str = typer.Option("~/.ssh/id_ed25519.pub", "--ssh-public-key-path", help="SSH public key path for Terraform."),
    tf_var: list[str] = typer.Option([], "--tf-var", help="Additional Terraform var key=value."),
    agent_port: int = typer.Option(DEFAULT_AGENT_PORT, "--agent-port", help="Public agent UI port."),
    backend_port: int = typer.Option(DEFAULT_BACKEND_PORT, "--backend-port", help="Internal agent backend port."),
    rerun_port: int = typer.Option(DEFAULT_RERUN_PORT, "--rerun-port", help="Rerun service port."),
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
    existing = _agent_record(project, name)
    if existing and not replace:
        _fail(
            f"Agent {project}/{name} already exists. Use --replace or choose a new --project/--name."
        )
    if existing and replace:
        typer.echo(f"Replacing existing agent {project}/{name} ...")
        destroy_cmd(project=project, name=name)
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
        agent_port=agent_port,
        backend_port=backend_port,
        rerun_port=rerun_port,
        llm_model=llm_model,
        llm_models=llm_models,
        no_public_https=no_public_https,
    )


@app.command("bootstrap")
def bootstrap_cmd(
    project: str = typer.Option(DEFAULT_PROJECT_ALIAS, "--project", help="NPA project alias."),
    name: str = typer.Option(DEFAULT_AGENT_NAME, "--name", help="Agent deployment name."),
    ssh_user: str = typer.Option("ubuntu", "--ssh-user", help="SSH username."),
    ssh_key: str = typer.Option("", "--ssh-key", help="SSH private key path (defaults to agent record or NPA_SSH_KEY)."),
    agent_port: int = typer.Option(DEFAULT_AGENT_PORT, "--agent-port", help="Public agent UI port."),
    backend_port: int = typer.Option(DEFAULT_BACKEND_PORT, "--backend-port", help="Internal agent backend port."),
    rerun_port: int = typer.Option(DEFAULT_RERUN_PORT, "--rerun-port", help="Rerun service port."),
    llm_model: str = typer.Option("", "--llm-model", help="Override the active Token Factory model."),
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
    no_public_https: bool = typer.Option(
        False,
        "--no-public-https",
        help="Disable HTTPS on port 443 (customer access uses http://IP:agent-port only).",
    ),
) -> None:
    """Re-bootstrap agent UI/backend/nginx on an existing VM (refresh without Terraform)."""
    record = _agent_record(project, name)
    if not record:
        _fail(f"Agent config not found for {project}/{name}")
    public_ip = str(record.get("public_ip", "")).strip()
    if not _is_routable_public_ip(public_ip):
        _fail("agent VM does not have a routable public IP")
    public_https = not no_public_https
    ssh_key_path = _resolve_agent_ssh_key(record, cli_ssh_key=ssh_key or None)
    if not Path(ssh_key_path).expanduser().exists():
        _fail(
            f"SSH private key not found at {ssh_key_path!r}. "
            "Pass --ssh-key, set NPA_SSH_KEY, or redeploy to persist ssh_key_path on the agent record."
        )
    try:
        auth_user, auth_password = _load_auth_secret(str(record.get("auth_secret_path", "")))
    except ValueError as exc:
        _fail(str(exc))
    tf_api_key, default_llm_model = _resolve_deploy_llm_credentials()
    requested_llm_model = str(llm_model or "").strip()
    resolved_llm_model = requested_llm_model or default_llm_model
    resolved_llm_models = _normalize_llm_models([resolved_llm_model, *llm_models])
    nebius_ai_key, _ = _resolve_operator_credentials()
    llm_block = record.get("llm", {}) if isinstance(record.get("llm"), dict) else {}
    if isinstance(llm_block.get("models"), list):
        resolved_llm_models = _normalize_llm_models(
            [*resolved_llm_models, *[str(item) for item in llm_block.get("models", [])]]
        )
    if not requested_llm_model and isinstance(llm_block.get("model"), str) and llm_block["model"].strip():
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
    s3_bucket, s3_prefix, s3_endpoint, s3_access_key, s3_secret_key, service_account_id = (
        _resolve_agent_storage_credentials(project, record)
    )
    if not service_account_id:
        service_account_id = _resolve_agent_service_account_id(project, record)
    agent_credentials: dict[str, str] | None = None
    if refresh_credentials:
        if not (project_id and tenant_id and region):
            _fail("agent record is missing project_id, tenant_id, or region for credential refresh")
        from npa.clients.nebius import NebiusError, bootstrap_agent_environment

        creds: dict[str, str] | None = None
        try:
            creds = bootstrap_agent_environment(
                project_id,
                tenant_id,
                region,
                on_status=lambda msg: typer.echo(f"  {msg}"),
            )
        except NebiusError as exc:
            typer.echo(
                f"Warning: npa-agent provisioning failed ({exc}); reusing existing credentials.",
                err=True,
            )
        if creds is None:
            creds = _creds_from_terraform_state(project, record)
        if creds is None:
            _fail("Nebius credential refresh failed and no terraform_state fallback is configured")
        creds = _resolve_deploy_storage_credentials(region=region, bootstrap_creds=creds)
        agent_credentials = _agent_credentials_payload(creds)
        s3_bucket = agent_credentials["s3_bucket"]
        s3_prefix = agent_credentials.get("s3_prefix", "")
        s3_endpoint = agent_credentials["s3_endpoint"]
        s3_access_key = agent_credentials["access_key"]
        s3_secret_key = agent_credentials["secret_key"]
        service_account_id = agent_credentials["service_account_id"]
        if not service_account_id:
            service_account_id = _resolve_agent_service_account_id(project, record)
            agent_credentials["service_account_id"] = service_account_id
        if s3_access_key and s3_secret_key:
            write_config(
                {
                    "projects": {
                        project: {
                            "terraform_state": {
                                "bucket": s3_bucket,
                                "endpoint": s3_endpoint,
                                "access_key": s3_access_key,
                                "secret_key": s3_secret_key,
                            },
                        }
                    }
                }
            )
    try:
        _bootstrap_agent_stack(
            host=public_ip,
            ssh_user=ssh_user,
            ssh_key_path=ssh_key_path,
            project_alias=project,
            agent_name=name,
            project_id=str(record.get("project_id", "") or ""),
            tenant_id=str(record.get("tenant_id", "") or ""),
            region=str(record.get("region", "") or "eu-north1"),
            auth_user=auth_user,
            auth_password=auth_password,
            agent_port=agent_port,
            backend_port=backend_port,
            rerun_port=rerun_port,
            llm_model=resolved_llm_model,
            llm_models=resolved_llm_models,
            tf_api_key=tf_api_key,
            nebius_ai_key=nebius_ai_key,
            s3_bucket=s3_bucket,
            s3_prefix=s3_prefix,
            s3_endpoint=s3_endpoint,
            s3_access_key=s3_access_key,
            s3_secret_key=s3_secret_key,
            s3_region=region,
            nebius_project_id=project_id,
            nebius_tenant_id=tenant_id,
            service_account_id=service_account_id,
            public_https=public_https,
        )
    except (ConfigError, SSHError, ValueError) as exc:
        _fail(f"VM bootstrap failed: {exc}")
    instance_id = str(record.get("instance_id", "")).strip()
    if instance_id:
        ingress_ports: list[int] = [agent_port, rerun_port]
        if public_https:
            ingress_ports.append(DEFAULT_HTTPS_PORT)
        try:
            ensure_ingress(vm_id=instance_id, ports=tuple(ingress_ports), tool="agent")
        except NetworkIngressError as exc:
            typer.echo(
                f"Warning: npa network ensure-ingress failed ({exc}). "
                "Customer HTTPS on port 443 may be unreachable until ingress is opened.",
                err=True,
            )
    urls = build_agent_urls(public_ip, agent_port=agent_port, public_https=public_https)
    updated = dict(record)
    updated.update(urls)
    updated["public_https"] = public_https
    llm_payload = dict(updated.get("llm", {}) if isinstance(updated.get("llm"), dict) else {})
    llm_payload["provider"] = DEFAULT_LLM_PROVIDER
    llm_payload["model"] = resolved_llm_model
    llm_payload["models"] = list(resolved_llm_models)
    updated["llm"] = llm_payload
    updated["ssh_key_path"] = ssh_key_path
    if service_account_id:
        updated["service_account_id"] = service_account_id
        _persist_agent_service_account_id(service_account_id)
    if s3_bucket and s3_access_key and s3_secret_key:
        updated["credentials"] = _credentials_block_from_storage(
            service_account_id=service_account_id,
            s3_bucket=s3_bucket,
            s3_prefix=s3_prefix,
            s3_endpoint=s3_endpoint,
            s3_access_key=s3_access_key,
            s3_secret_key=s3_secret_key,
        )
    elif refresh_credentials and agent_credentials is not None:
        updated["credentials"] = agent_credentials
    _store_agent_record(project, name, updated)
    typer.echo(f"Customer URL: {urls['public_url']}")
    typer.echo(f"bootstrapped: {project}/{name} at {urls['public_url']}")
    if public_https:
        typer.echo(f"direct_url: {urls['direct_url']}")


@app.command("status")
def status_cmd(
    project: str = typer.Option(DEFAULT_PROJECT_ALIAS, "--project", help="NPA project alias."),
    name: str = typer.Option(DEFAULT_AGENT_NAME, "--name", help="Agent deployment name."),
    output_json: bool = typer.Option(False, "--json", help="Emit JSON."),
) -> None:
    """Show agent status, URLs, and health checks."""
    record = _agent_record(project, name)
    if not record:
        _fail(f"Agent config not found for {project}/{name}")
    try:
        auth_user, auth_password = _load_auth_secret(str(record.get("auth_secret_path", "")))
    except ValueError as exc:
        _fail(str(exc))
    agent_url = str(record.get("agent_url", ""))
    rerun_url = str(record.get("rerun_url", ""))
    sim_viz_url = str(record.get("sim_viz_url", rerun_url))
    sim_assets_url = str(record.get("sim_assets_url", agent_url))
    cameras_api_url = str(
        record.get("cameras_api_url", f"{agent_url.rstrip('/')}/assets/api/sim-assets/cameras")
    )
    public_url = _record_customer_url(record)
    tls_verify = _record_tls_verify(record)
    ui_ok, ui_code = _health(agent_url, user=auth_user, password=auth_password, verify=tls_verify)
    rerun_ok, rerun_code = _health(sim_viz_url, user=auth_user, password=auth_password, verify=tls_verify)
    payload = {
        "project": project,
        "name": name,
        "public_ip": record.get("public_ip", ""),
        "public_url": public_url,
        "public_https": _record_public_https(record),
        "direct_url": record.get("direct_url", ""),
        "ui_url": agent_url,
        "rerun_url": rerun_url,
        "sim_viz_url": sim_viz_url,
        "sim_assets_url": sim_assets_url,
        "cameras_api_url": cameras_api_url,
        "health": bool(ui_ok and rerun_ok),
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
def destroy_cmd(
    project: str = typer.Option(DEFAULT_PROJECT_ALIAS, "--project", help="NPA project alias."),
    name: str = typer.Option(DEFAULT_AGENT_NAME, "--name", help="Agent deployment name."),
) -> None:
    """Destroy agent VM/resources and remove saved config entry."""
    record = _agent_record(project, name)
    if not record and not _agent_terraform_state_exists(project, name):
        _fail(f"Agent config not found for {project}/{name}")
    try:
        _destroy_agent_terraform(project, name, record=record or None)
    except ProvisionerError as exc:
        _fail(f"Terraform destroy failed: {exc}")
    if record:
        _remove_agent_record(project, name)
    typer.echo(f"destroyed: {project}/{name}")


@app.command("verify-live")
def verify_live_cmd(
    project: str = typer.Option(DEFAULT_PROJECT_ALIAS, "--project", help="NPA project alias."),
    name: str = typer.Option(DEFAULT_AGENT_NAME, "--name", help="Agent deployment name."),
) -> None:
    """Exit 0 only when live infra checks and tests pass."""
    record = _agent_record(project, name)
    if not record:
        _fail(f"Agent config not found for {project}/{name}")
    public_ip = str(record.get("public_ip", "")).strip()
    region = str(record.get("region", "")).strip()
    if not public_ip or public_ip in {"localhost", "127.0.0.1"} or public_ip.startswith("127."):
        _fail("agent VM does not have a non-localhost public IP")
    if not _is_routable_public_ip(public_ip):
        _fail("agent VM does not have a non-localhost public IP")
    if region != "us-central1":
        _fail(f"agent region mismatch: expected us-central1, got {region!r}")
    try:
        auth_user, auth_password = _load_auth_secret(str(record.get("auth_secret_path", "")))
    except ValueError as exc:
        _fail(str(exc))

    customer_url = _record_customer_url(record)
    tls_verify = _record_tls_verify(record)
    if customer_url:
        try:
            welcome_resp = httpx.get(
                f"{customer_url.rstrip('/')}/welcome",
                timeout=5.0,
                verify=tls_verify,
            )
            if welcome_resp.status_code != 200:
                _fail(f"public welcome page unhealthy (status={welcome_resp.status_code})")
            healthz_resp = httpx.get(
                f"{customer_url.rstrip('/')}/healthz",
                timeout=5.0,
                verify=tls_verify,
            )
            if healthz_resp.status_code != 200:
                _fail(f"public healthz unhealthy (status={healthz_resp.status_code})")
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

    sim_assets_base = str(record.get("sim_assets_url", record.get("agent_url", ""))).rstrip("/")
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
    if not isinstance(sim_assets_payload, dict) or "scene_spec" not in sim_assets_payload or "robot_spec" not in sim_assets_payload:
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
    cameras = cameras_payload.get("cameras", []) if isinstance(cameras_payload, dict) else []
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
        _fail("rerun static asset probe failed (no /rerun/*.js|ico|version responded 200)")

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
        workflow_status_resp = httpx.get(
            f"{agent_base}/api/workflows/sim2real/status",
            auth=(auth_user, auth_password),
            timeout=5.0,
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
        "function sendChat(",
        "function wireUi(",
        "initNpaAgentUi",
        "mobile-agent",
        "history.replaceState",
        "location.username",
        f'name="npa-ui-version" content="{AGENT_UI_VERSION}"',
    ):
        if marker not in ui_html:
            _fail(f"UI html missing wiring marker: {marker}")

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
            json={"messages": [{"role": "user", "content": "what is the current sim2real status"}]},
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
    if reply.strip().startswith("GET /api") or reply.strip() == "GET /api/sim-viz/status":
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
            json={"messages": [{"role": "user", "content": "create 2-step sim2real workflow"}]},
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
    if "augment" not in wf_yaml or "envgen" not in wf_yaml:
        _fail("create-workflow chat yaml missing sim2real stages")
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
                            "add an open source repo, containerize, push to registry, "
                            "and run LeIsaac on live infra"
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
    if "run_byof_repo.py" not in onboard_reply:
        _fail("onboard_solution chat reply missing run_byof_repo.py command")
    if "byof-onboard" not in onboard_reply and "skills/workflows/byof-onboard" not in onboard_reply:
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
    smoke = subprocess.run(
        [
            "npa/.venv/bin/python",
            "-m",
            "pytest",
            "npa/tests/smoke/test_agent_smoke.py",
            "npa/tests/smoke/test_agent_chat_smoke.py",
            "-q",
        ],
        check=False,
        env=test_env,
    )
    if smoke.returncode != 0:
        _fail("pytest npa/tests/smoke/test_agent_smoke.py test_agent_chat_smoke.py failed")
    unit = subprocess.run(
        [
            "npa/.venv/bin/python",
            "-m",
            "pytest",
            "npa/tests/cli/test_agent.py",
            "npa/tests/cli/test_agent_workflow.py",
            "-q",
        ],
        check=False,
        env=test_env,
    )
    if unit.returncode != 0:
        _fail("pytest npa/tests/cli/test_agent.py failed")
    e2e = subprocess.run(
        ["npa/.venv/bin/python", "-m", "pytest", "npa/tests/e2e/test_agent_live.py", "-q"],
        check=False,
        env=test_env,
    )
    if e2e.returncode != 0:
        _fail("pytest npa/tests/e2e/test_agent_live.py failed")
    typer.echo("verify-live: ok")
