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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from npa.workflows.sim2real_health import CheckResult

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
# Cost-ordered ladder (cheapest-capable first). Per-turn cost-tier routing
# (see agent_routing.build_model_ladder) reorders this for each request, but the
# default configured order still surfaces the cheap workhorse ahead of the
# branded reasoner so no-routing paths and the /models picker default cheap.
DEFAULT_LLM_MODELS = (
    "Qwen/Qwen3-32B",
    "meta-llama/Llama-3.3-70B-Instruct",
    DEFAULT_LLM_MODEL,
    "Qwen/Qwen2.5-VL-72B-Instruct",
)
AGENT_UI_VERSION = "2026071925"
DEFAULT_HTTPS_PORT = 443
AGENT_SOURCE_ROOT = "/opt/npa-agent/npa-src"
_AGENT_TERRAFORM_RUNTIME_ONLY_VARS = frozenset({"s3_prefix"})

# Contract markers that must stay in the embedded agent UI/backend. verify-live,
# smoke, and unit tests share this list so media-preview regressions cannot
# silently disappear after a bootstrap drift or template edit.
AGENT_MEDIA_PREVIEW_CONTRACT = (
    "authenticatedPreviewObjectUrl",
    "Loading video preview…",
    'data-preview-url="',
    "Keep the Rerun iframe mounted under the media pane",
    'id="renderModeVideo"',
    'id="artifactPreviewHost"',
    'id="viewerPaneMedia"',
    "URL.createObjectURL(blob)",
    '@app.api_route("/artifacts/file/{{filename}}", methods=["GET", "HEAD"])',
    "artifact_media_type(",
)

# Rerun wasm splash must never be user-visible. Cover the iframe until past
# "Loading application bundle", and fully warm assets before first reveal.
AGENT_RERUN_NO_BUNDLE_SPLASH_CONTRACT = (
    'id="rerunBundleCover"',
    "waitUntilRerunPastBundleSplash",
    "showRerunBundleCover",
    "hideRerunBundleCover",
    "safeHideRerunBundleCover",
    "Warm Rerun assets before revealing the iframe",
    "Preparing viewer…",
    # Cover may stay up, but mount/boot must not await long splash polls (latency).
    "Uncover without blocking mount latency",
    # Canvas-painted splash is not DOM text — require non-blank pixels before uncover.
    "non-blank canvas",
    # Run switches must soft-swap recordings without remounting wasm.
    "swapRerunRecordingInPlace",
    "add_receiver",
)

# Describe-this visual feedback: capture current viewer frame → vision tier chat.
AGENT_VISUAL_FEEDBACK_CONTRACT = (
    'id="describeVisual"',
    "captureVisualContext",
    "describeVisual",
    "[npa-visual-feedback]",
    "visual_context",
    "normalize_messages_for_llm",
    "infer_visual_domain_hints",
    "frameLooksBlank",
    "sampleFrameStats",
    "captureCanvasDataUrl",
    "ensureRerunCaptureBridge",
    "grabFromRerunCaptureBridge",
    "pickBestIframeCanvas",
    "probeRerunCanvasContent",
    "waitForQualityRerunFrame",
    "skipUserAppend",
    "Describe this — capturing",
    "client_max_body_size 32m",
    "maxChars = 700000",
)

AGENT_CHAT_QUEUE_CONTRACT = (
    "chatQueue",
    "enqueueChatJob",
    "processChatQueue",
    "queueChatText",
)

AGENT_VIEWER_CHAT_DRAWER_CONTRACT = (
    "viewer-focus",
    "chat-drawer-open",
    'id="chatDrawerToggle"',
    "openChatDrawer",
    "openFullChatTab",
    "setChatDrawerOpen",
    'id="openFullChatTab"',
    'id="chatDrawerClose"',
    "chat-fab",
    "transform-origin: bottom right",
)

AGENT_STAGES_RUN_PICKER_CONTRACT = (
    'id="stagesRunSelect"',
    'id="stagesRunInput"',
    'id="stagesLoadRun"',
    "stages-run-picker",
    "loadSelectedRun",
    "syncRunChooserFields",
    "filterStagesRunSelect",
    "Search or paste run ID",
)

AGENT_READABLE_COLOR_CONTRACT = (
    "--ink-strong",
    "thinking-ellipsis",
    "Color contrast rules",
)


def _embedded_agent_workflow_source() -> str:
    """Return agent_workflow.py source embedded into the remote agent backend."""
    import re

    path = Path(__file__).with_name("agent_workflow.py")
    raw = path.read_text(encoding="utf-8")
    raw = re.sub(r'^""".*?"""\s*\n', "", raw, count=1, flags=re.DOTALL)
    raw = re.sub(r"^from __future__ import annotations\s*\n", "", raw)
    return raw


def _embedded_agent_routing_source() -> str:
    """Return agent_routing.py source embedded into the remote agent backend."""
    import re

    path = Path(__file__).with_name("agent_routing.py")
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


def _embedded_agent_actions_source() -> str:
    """Return agent_actions.py source embedded into the remote agent backend."""
    import re

    path = Path(__file__).with_name("agent_actions.py")
    raw = path.read_text(encoding="utf-8")
    raw = re.sub(r'^""".*?"""\s*\n', "", raw, count=1, flags=re.DOTALL)
    raw = re.sub(r"^from __future__ import annotations\s*\n", "", raw)
    return raw


def _embedded_agent_sim2real_loop_source() -> str:
    """Return agent_sim2real_loop.py source embedded into the remote agent backend."""
    import re

    path = Path(__file__).with_name("agent_sim2real_loop.py")
    raw = path.read_text(encoding="utf-8")
    raw = re.sub(r'^""".*?"""\s*\n', "", raw, count=1, flags=re.DOTALL)
    raw = re.sub(r"^from __future__ import annotations\s*\n", "", raw)
    return raw


def _embedded_agent_semantic_router_source() -> str:
    """Return agent_semantic_router.py source embedded into the remote agent backend."""
    import re

    path = Path(__file__).with_name("agent_semantic_router.py")
    raw = path.read_text(encoding="utf-8")
    raw = re.sub(r'^""".*?"""\s*\n', "", raw, count=1, flags=re.DOTALL)
    raw = re.sub(r"^from __future__ import annotations\s*\n", "", raw)
    return raw


_AGENT_CHAT_EMBED = "__NPA_AGENT_CHAT_EMBED__"
_AGENT_ACTIONS_EMBED = "__NPA_AGENT_ACTIONS_EMBED__"
_AGENT_SIM2REAL_LOOP_EMBED = "__NPA_AGENT_SIM2REAL_LOOP_EMBED__"
_AGENT_SEMANTIC_ROUTER_EMBED = "__NPA_AGENT_SEMANTIC_ROUTER_EMBED__"
_AGENT_WORKFLOW_EMBED = "__NPA_AGENT_WORKFLOW_EMBED__"
_AGENT_ARTIFACTS_EMBED = "__NPA_AGENT_ARTIFACTS_EMBED__"
_AGENT_ROUTING_EMBED = "__NPA_AGENT_ROUTING_EMBED__"
_AGENT_VISUAL_FEEDBACK_EMBED = "__NPA_AGENT_VISUAL_FEEDBACK_EMBED__"
_AGENT_RRD_PROXY_EMBED = "__NPA_AGENT_RRD_PROXY_EMBED__"
_AGENT_STAGES_EMBED = "__NPA_AGENT_STAGES_EMBED__"
_AGENT_UI_HTML_EMBED = "__NPA_AGENT_UI_HTML__"


def _embedded_agent_stages_source() -> str:
    """Return agent_stages.py source embedded into the remote agent backend."""
    import re

    path = Path(__file__).with_name("agent_stages.py")
    raw = path.read_text(encoding="utf-8")
    raw = re.sub(r'^""".*?"""\s*\n', "", raw, count=1, flags=re.DOTALL)
    raw = re.sub(r"^from __future__ import annotations\s*\n", "", raw)
    return raw


def rendered_agent_ui_html() -> str:
    """Return the agent UI HTML with bootstrap placeholders substituted.

    The UI lives in ``agent_ui.html`` (outside the bootstrap f-string) so JS can
    use normal braces and ``agent.py`` stays under the monolith size ratchet.
    """
    path = Path(__file__).with_name("agent_ui.html")
    raw = path.read_text(encoding="utf-8")
    return raw.replace("{AGENT_UI_VERSION}", AGENT_UI_VERSION).replace(
        "{DEFAULT_AGENT_USER}", DEFAULT_AGENT_USER
    )


def _embedded_agent_visual_feedback_source() -> str:
    """Return agent_visual_feedback.py source embedded into the remote agent backend."""
    import re

    path = Path(__file__).with_name("agent_visual_feedback.py")
    raw = path.read_text(encoding="utf-8")
    raw = re.sub(r'^""".*?"""\s*\n', "", raw, count=1, flags=re.DOTALL)
    raw = re.sub(r"^from __future__ import annotations\s*\n", "", raw)
    return raw


def _embedded_agent_rrd_proxy_source() -> str:
    """Return agent_rrd_proxy.py source embedded into the remote agent backend."""
    import re

    path = Path(__file__).with_name("agent_rrd_proxy.py")
    raw = path.read_text(encoding="utf-8")
    raw = re.sub(r'^""".*?"""\s*\n', "", raw, count=1, flags=re.DOTALL)
    raw = re.sub(r"^from __future__ import annotations\s*\n", "", raw)
    return raw


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


def _persist_agent_project_config(
    *,
    project: str,
    project_id: str,
    tenant_id: str,
    region: str,
    merged_vars: dict[str, str],
) -> None:
    write_config(
        {
            "projects": {
                project: {
                    "project_id": project_id,
                    "tenant_id": tenant_id,
                    "region": region,
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
    tf_vars = {key: value for key, value in merged_vars.items() if key not in _AGENT_TERRAFORM_RUNTIME_ONLY_VARS}
    try:
        return provisioner.apply(tf_dir=tf_dir, tf_vars=tf_vars)
    except ProvisionerError as exc:
        sa_id = str(merged_vars.get("service_account_id", "")).strip()
        if sa_id and _looks_like_compute_permission_denied(str(exc)):
            typer.echo(
                "  Compute create denied with VM service-account attachment; "
                "retrying without attached service_account_id ..."
            )
            retry_vars = dict(tf_vars)
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
    region = str((record or {}).get("region", "") or (saved_env.region if saved_env else "") or "eu-north1")
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


def _terraform_binary() -> str:
    """Return the terraform binary path/name, honoring NPA_TERRAFORM_BIN."""
    return (os.environ.get("NPA_TERRAFORM_BIN") or shutil.which("terraform") or "").strip()


def _agent_hard_prereq_results(ssh_public_key_path: str) -> list[CheckResult]:
    """Cheap, side-effect-free Route C prerequisites (terraform + SSH keys).

    These are checked before any cloud IAM side effects or Terraform apply so a
    missing binary or key surfaces up front instead of mid-run (after which a
    transient SSH failure would auto-roll-back a freshly provisioned VM).
    """
    from npa.workflows.sim2real_health import CheckResult, FAIL, PASS

    results: list[Any] = []

    terraform = _terraform_binary()
    if terraform:
        results.append(
            CheckResult(name="terraform", status=PASS, summary=f"terraform found ({terraform}).")
        )
    else:
        results.append(
            CheckResult(
                name="terraform",
                status=FAIL,
                summary="terraform binary not found on PATH.",
                remedy="Install it: https://developer.hashicorp.com/terraform/install",
            )
        )

    pub_path = Path(ssh_public_key_path).expanduser()
    if pub_path.is_file():
        results.append(
            CheckResult(name="ssh_public_key", status=PASS, summary=f"SSH public key present ({pub_path}).")
        )
    else:
        priv_hint = str(pub_path)[:-4] if str(pub_path).endswith(".pub") else str(pub_path)
        results.append(
            CheckResult(
                name="ssh_public_key",
                status=FAIL,
                summary=f"SSH public key not found: {pub_path}",
                remedy=(
                    f"Generate a keypair (`ssh-keygen -t ed25519 -f {priv_hint}`) "
                    "or pass --ssh-public-key-path to an existing key."
                ),
            )
        )

    # The deploy flow uses the private key alongside the public key (pub path
    # minus the .pub suffix) to bootstrap the VM over SSH. If --ssh-public-key-path
    # is given without a .pub suffix, this resolves to the same path as the public
    # key check above, which at worst yields a slightly redundant message.
    priv_str = str(pub_path)[:-4] if str(pub_path).endswith(".pub") else str(pub_path)
    priv_path = Path(priv_str)
    if priv_path.is_file():
        results.append(
            CheckResult(name="ssh_private_key", status=PASS, summary=f"SSH private key present ({priv_path}).")
        )
    else:
        results.append(
            CheckResult(
                name="ssh_private_key",
                status=FAIL,
                summary=f"SSH private key not found: {priv_path}",
                remedy="The private key next to the public key is required to bootstrap the VM over SSH.",
            )
        )

    return results


def _agent_token_factory_result(tf_key: str | None = None) -> CheckResult:
    """Token Factory key check (WARN): the headline chat feature needs it.

    Pass a pre-resolved ``tf_key`` to avoid re-reading credentials when the
    caller already has them.
    """
    from npa.workflows.sim2real_health import CheckResult, PASS, WARN

    if tf_key is None:
        tf_key, _ = _resolve_deploy_llm_credentials()
    if tf_key:
        return CheckResult(
            name="token_factory", status=PASS, summary="Token Factory API key is configured."
        )
    return CheckResult(
        name="token_factory",
        status=WARN,
        summary="Token Factory API key not found; agent chat will return 503 until it is set.",
        remedy=(
            "Get a key (starts with 'v1.') at https://tokenfactory.nebius.com/ and run "
            "`npa configure --token-factory-key <key>`, then re-run `npa agent bootstrap`."
        ),
    )


def _agent_nebius_auth_result() -> CheckResult:
    """Live Nebius auth check (FAIL): deploy needs an authenticated profile to provision."""
    from npa.workflows.sim2real_health import CheckResult, FAIL, PASS

    try:
        from npa.clients.nebius import get_iam_token

        token = get_iam_token()
    except Exception as exc:  # noqa: BLE001 - any auth/CLI error means "not ready"
        return CheckResult(
            name="nebius_profile",
            status=FAIL,
            summary="No authenticated Nebius CLI profile.",
            remedy="Install/authenticate the Nebius CLI and run `npa configure`.",
            details=(str(exc),),
        )
    if token:
        return CheckResult(
            name="nebius_profile", status=PASS, summary="Nebius CLI profile is authenticated."
        )
    return CheckResult(
        name="nebius_profile",
        status=FAIL,
        summary="Nebius IAM token unavailable.",
        remedy="Run `npa configure` / `nebius profile create` to authenticate.",
    )


def _render_agent_checks(results: list[CheckResult], *, output_json: bool) -> bool:
    """Render agent preflight CheckResults; return True when any FAIL is present.

    Uses the shared report renderer so agent and workbench-health preflight
    output stay aligned.
    """
    from npa.workflows.sim2real_health import format_check_report, has_failure

    typer.echo(format_check_report(results, output_json=output_json))
    return has_failure(results)


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
    project_alias: str = "",
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
    project_name = str(project_alias or "").strip()
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
                candidate["s3_bucket"] = saved_bucket
                candidate["s3_endpoint"] = saved_endpoint
                candidate["nebius_api_key"] = saved_access_key
                candidate["nebius_secret_key"] = saved_secret_key
                return candidate
    _fail(
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
  # Describe-this / multimodal chat posts JPEG data-URLs; default 1m rejects them (413 → browser Failed to fetch).
  client_max_body_size 32m;
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
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_connect_timeout 30s;
    proxy_read_timeout 900s;
    proxy_send_timeout 900s;
    client_max_body_size 32m;
  }}
  location /assets/api/ {{
    rewrite ^/assets/api/(.*)$ /$1 break;
    proxy_pass http://127.0.0.1:{backend_port}/;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_connect_timeout 30s;
    proxy_read_timeout 900s;
    proxy_send_timeout 900s;
    client_max_body_size 32m;
  }}
  location /rerun/recordings/ {{
    auth_basic off;
    alias /opt/npa-agent/recordings/;
    default_type application/octet-stream;
    add_header Cache-Control "no-cache" always;
    add_header Access-Control-Allow-Origin * always;
    add_header Access-Control-Allow-Methods "GET, HEAD, OPTIONS" always;
    add_header Cross-Origin-Resource-Policy "cross-origin" always;
  }}
  location ~* ^/rerun/.+\\.(wasm|js|ico|svg)$ {{
    auth_basic off;
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
    auth_basic off;
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
    agent_actions_source = _embedded_agent_actions_source()
    agent_sim2real_loop_source = _embedded_agent_sim2real_loop_source()
    agent_semantic_router_source = _embedded_agent_semantic_router_source()
    agent_workflow_source = _embedded_agent_workflow_source()
    agent_artifacts_source = _embedded_agent_artifacts_source()
    agent_routing_source = _embedded_agent_routing_source()
    agent_visual_feedback_source = _embedded_agent_visual_feedback_source()
    agent_rrd_proxy_source = _embedded_agent_rrd_proxy_source()
    agent_stages_source = _embedded_agent_stages_source()
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
cat <<'ENV' | sudo tee /opt/npa-agent/public.env >/dev/null
NPA_AGENT_PUBLIC_URL=https://{host}
NPA_AGENT_PUBLIC_HOST={host}
ENV
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
from urllib.parse import quote

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

RERUN_RECORDING_HTTP_PATH = "/rerun/recordings/sim2real.rrd"


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


def _rerun_recording_url(*, cache_bust: bool = False) -> str:
    origin = _agent_public_origin()
    path = RERUN_RECORDING_HTTP_PATH
    if origin:
        url = f"{{origin}}{{path}}"
    else:
        url = path
    if cache_bust:
        url = f"{{url}}?t={{int(time.time() * 1000)}}"
    return url


def _rerun_iframe_url(camera: str = "workspace", *, live_url: str = "") -> str:
    cam = (camera or "workspace").strip() or "workspace"
    if live_url:
        return f"/rerun/?url={{quote(live_url, safe='')}}&hide_welcome_screen=1&theme=dark&camera={{cam}}"
    recording = _rerun_recording_url()
    # Rerun web viewer treats path-only values like `/rerun/...` as host `rerun`.
    return f"/rerun/?url={{quote(recording, safe='')}}&hide_welcome_screen=1&theme=dark&camera={{cam}}"

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
    # Single-tenant operator-VM model: lock-free read-modify-write on STATE_PATH
    # (+ best-effort S3 mirror). Concurrent writers are last-writer-wins — fine
    # for one operator UI, not safe if this ever becomes a multi-client service.
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
    # See _load_state: no file lock — last writer wins under concurrent requests.
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
    existing = runs.get(run_id) if isinstance(runs.get(run_id), dict) else {{}}
    snapshot = dict(DEFAULT_SIM_VIZ)
    if isinstance(existing, dict):
        snapshot.update(existing)
    snapshot.update(payload)
    snapshot["run_id"] = run_id
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
            ):
                if key not in payload or not str(payload.get(key) or "").strip():
                    snapshot[key] = ""
    else:
        # Never let a sparse update erase richer artifact fields from load-run.
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
        ):
            if not str(snapshot.get(key) or "").strip() and str(existing.get(key) or "").strip():
                snapshot[key] = existing[key]
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


def _workflow_stage_defs_from_state(state: dict) -> list[tuple[str, str, list[str]]]:
    draft = _workflow_draft_from_state(state)
    stages: list[tuple[str, str, list[str]]] = []
    plan = draft.get("plan") if isinstance(draft.get("plan"), dict) else {{}}
    for source in (plan.get("steps"), plan.get("states"), draft.get("states")):
        if not isinstance(source, list):
            continue
        for item in source:
            if isinstance(item, dict):
                raw_id = str(item.get("state") or item.get("id") or item.get("name") or "").strip()
                label = str(item.get("label") or item.get("description") or raw_id).strip() or raw_id
            else:
                raw_id = str(item or "").strip()
                label = raw_id
            if not raw_id:
                continue
            stage_id = _slug(raw_id, fallback="stage")
            patterns = [raw_id, raw_id.replace("_", "-"), raw_id.replace("-", "_")]
            if (stage_id, label, patterns) not in stages:
                stages.append((stage_id, label, patterns))
        if stages:
            break
    return stages


def _artifact_backed_run_details(state: dict, run_id: str) -> dict | None:
    if not run_id:
        return None
    try:
        s3, settings = _agent_s3_client()
        effective_prefix = _artifact_discovery_prefix(settings, "")
        artifacts = list_artifacts(settings["bucket"], validate_run_id(run_id), prefix=effective_prefix, s3=s3)
    except Exception:
        return None
    if not artifacts:
        return None
    keys = [str(item.key or "") for item in artifacts]
    workflow_stage_defs = _workflow_stage_defs_from_state(state)
    overlay_unmatched = run_owns_workflow_stage_overlay(state, run_id)
    stages = build_artifact_backed_stages(
        keys,
        run_id=run_id,
        prefix=effective_prefix,
        workflow_stage_defs=workflow_stage_defs,
        overlay_unmatched=overlay_unmatched,
    )
    report_ready = any(key.endswith("/reports/sim2real-report.json") or key.endswith("/reports/report.json") for key in keys)
    rrd_ready = any(key.endswith(".rrd") for key in keys)
    preferred = select_preferred_artifact(artifacts)
    report_note = ""
    report_artifact = next((item for item in artifacts if item.key.endswith("/reports/sim2real-report.json")), None)
    if report_artifact:
        local_report = RECORDINGS_DIR / (_artifact_filename(report_artifact.key) + ".json")
        try:
            download_s3_uri(report_artifact.s3_uri, local_report, s3=s3)
            report = json.loads(local_report.read_text(encoding="utf-8"))
            viz = report.get("visualization") if isinstance(report.get("visualization"), dict) else {{}}
            decision = report.get("outer_loop", {{}}).get("latest_decision", {{}}) if isinstance(report.get("outer_loop"), dict) else {{}}
            source = str(viz.get("source") or "").strip()
            success_rate = decision.get("success_rate")
            if source or success_rate is not None:
                report_note = (
                    "Report summary: visualization source="
                    + (source or "unknown")
                    + (f", success_rate={{success_rate}}" if success_rate is not None else "")
                    + "."
                )
        except Exception:
            report_note = ""
    return {{
        "run_id": run_id,
        "status": "completed" if report_ready else "artifact-backed",
        "result": "rerun_ready" if rrd_ready else "artifacts_available",
        "submitted_at": "",
        "updated_at": max((str(item.last_modified or "") for item in artifacts), default=_now_iso()),
        "selection": {{}},
        "stages": stages,
        "logs": [
            {{
                "timestamp": _now_iso(),
                "level": "info",
                "message": f"Derived stage timeline from {{len(artifacts)}} S3 artifacts.",
            }},
            {{
                "timestamp": _now_iso(),
                "level": "info" if rrd_ready else "warn",
                "message": (
                    f"Preferred viewable artifact: {{preferred.key}}"
                    if preferred
                    else "No preferred viewable artifact found."
                ),
            }},
            {{
                "timestamp": _now_iso(),
                "level": "info",
                "message": report_note or "No structured run report summary was available.",
            }},
        ],
        "artifacts": [item.to_dict() for item in artifacts[:25]],
    }}


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
    artifact_details = _artifact_backed_run_details(state, resolved_run_id)
    if artifact_details:
        details = _merge_sim2real_run_details(details, artifact_details)
    stage = str(sim_viz.get("stage") or details.get("status") or "submitted").strip()
    if stage and not artifact_details:
        details["status"] = stage
    if sim_viz.get("rrd_uri"):
        if str(details.get("result") or "") not in {"completed", "failed", "running", "rerun_ready"}:
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
    runs = state.get("sim_viz_runs")
    target = str(run_id or state.get("active_run_id") or "").strip()
    if isinstance(runs, dict) and target and isinstance(runs.get(target), dict):
        payload.update(runs[target])
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


def _artifact_discovery_prefix(settings: dict[str, str], user_prefix: str = "") -> str:
    requested = str(user_prefix or "").strip().strip("/")
    base = str(settings.get("prefix") or "").strip().strip("/")
    if requested:
        return _join_agent_s3_prefix(base, requested)
    return _join_agent_s3_prefix(base, "sim2real-b")


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
    # Persist text stubs only — never store screenshot data-URLs in session history.
    return normalize_messages_for_storage(raw)


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


def _is_sim2real_pipeline_recording(key: str) -> bool:
    return str(key or "").endswith("/reports/sim2real.rrd")


def _sim2real_pipeline_camera_label(requested: str = "") -> str:
    value = str(requested or "").strip()
    return value if value and value != "workspace" else "heldout-sim"


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
    camera = str(sim_viz.get("camera") or "workspace")
    if render == "rerun" and _is_sim2real_pipeline_recording(key):
        camera = _sim2real_pipeline_camera_label(camera)
    sim_viz.update(
        {{
            "run_id": run_id,
            "stage": "artifact-loaded",
            "rrd_updated_at": now,
            "artifact_uri": s3_uri,
            "artifact_key": key,
            "artifact_render": render,
            "mode": "static",
            "camera": camera,
        }}
    )
    if render == "rerun":
        _publish_rrd_recording(local_path)
        restarted = _restart_rerun_serve(force=True)
        rerun_ready = _wait_rerun_web_viewer_healthy() if restarted else False
        sim_viz["rrd_uri"] = f"file://{{RECORDING_PATH}}"
        sim_viz["artifact_preview_url"] = "/rerun/recordings/sim2real.rrd"
        sim_viz["artifact_download_url"] = "/rerun/recordings/sim2real.rrd"
        sim_viz["rerun_iframe_url"] = _rerun_iframe_url(str(sim_viz.get("camera") or "workspace"))
        sim_viz["rerun_ready"] = RECORDING_PATH.is_file() and rerun_ready
        if _is_sim2real_pipeline_recording(key):
            sim_viz["preview_entity"] = "camera"
            sim_viz["visualization_note"] = (
                "Pipeline Sim2Real recording loaded. The primary Rerun view is the "
                "held-out simulation camera stream; any 3D Franka/world entities are "
                "reference proxy context, not custom hardware footage."
            )
    else:
        filename = _artifact_filename(key)
        target = RECORDINGS_DIR / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        if local_path.resolve() != target.resolve():
            shutil.copy2(local_path, target)
        preview_url = _artifact_preview_url(filename)
        sim_viz["artifact_preview_url"] = preview_url
        sim_viz["artifact_download_url"] = preview_url
        sim_viz["rrd_uri"] = ""
        sim_viz["rerun_iframe_url"] = "/rerun/"
        sim_viz["rerun_ready"] = False
        sim_viz["preview_entity"] = ""
        sim_viz["visualization_note"] = (
            f"Loaded {{render}} artifact preview. Use the Video/Image/Data viewer tabs."
        )
    sim_viz["active_run_id"] = run_id
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
    if not run_id.startswith("sim2real-"):
        return None
    candidates = [RECORDINGS_DIR / f"{{run_id}}.rrd", RECORDING_PATH, RRD_PATH]
    source = next((item for item in candidates if item.is_file() and item.stat().st_size > 65536), None)
    if source is None:
        return None
    if source != RECORDING_PATH:
        _publish_rrd_recording(source)
    if source != RRD_PATH and RECORDING_PATH.is_file():
        RRD_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(RECORDING_PATH, RRD_PATH)
    _restart_rerun_serve(force=False)
    selection = _stock_franka_selection()
    state["selection"] = selection
    cam = (camera or "workspace").strip() or "workspace"
    state["camera_selection"] = [cam]
    updated_at = datetime.fromtimestamp(RRD_PATH.stat().st_mtime, tz=timezone.utc).isoformat()
    live_url = str(os.environ.get("NPA_AGENT_RERUN_LIVE_URL", "")).strip()
    iframe_url = (
        f"/rerun/?url={{quote(live_url, safe='')}}&hide_welcome_screen=1&theme=dark&camera={{cam}}"
        if live_url
        else _rerun_iframe_url(cam)
    )
    viz = {{
        "run_id": run_id,
        "stage": str(current.get("stage") or "completed"),
        "rrd_uri": f"file://{{RRD_PATH}}",
        "rrd_updated_at": updated_at,
        "artifact_uri": str(current.get("artifact_uri") or latest.get("rrd_uri") or ""),
        "artifact_key": str(current.get("artifact_key") or ""),
        "artifact_render": "rerun",
        "artifact_preview_url": "/rerun/recordings/sim2real.rrd",
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

def _wire_franka_demo(state: dict, *, camera: str = "workspace") -> dict:
    active = _wire_active_sim2real_recording(state, camera=camera)
    if active is not None:
        return active
    selection = _stock_franka_selection()
    state["selection"] = selection
    cam = (camera or "workspace").strip() or "workspace"
    state["camera_selection"] = [cam]
    target = _generate_franka_demo_rrd(camera=cam)
    restarted = _restart_rerun_serve()
    viewer_ready = _wait_for_rerun_web_viewer() if restarted else False
    now = _now_iso()
    # Always use the stock demo run id and clear any prior media-artifact preview.
    viz = {{
        "run_id": "franka-demo",
        "stage": "demo",
        "rrd_uri": f"file://{{target}}",
        "rrd_updated_at": now,
        "live_grpc_url": "",
        "mode": "static",
        "camera": cam,
        "preview_camera": cam,
        "preview_entity": f"world/camera_frustums/{{cam}}/frustum",
        "rerun_ready": target.is_file() and viewer_ready,
        "rerun_iframe_url": _rerun_iframe_url(cam),
        "artifact_render": "rerun",
        "artifact_key": "",
        "artifact_uri": "",
        "artifact_preview_url": "/rerun/recordings/sim2real.rrd",
        "artifact_download_url": "/rerun/recordings/sim2real.rrd",
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
    restarted = _restart_rerun_serve()
    viewer_ready = _wait_for_rerun_web_viewer() if restarted else False
    now = _now_iso()
    viz = {{
        "run_id": str(run_id or "").strip() or f"agent-run-{{secrets.token_hex(6)}}",
        "stage": "stage_14_rerun_viz",
        "rrd_uri": f"file://{{target}}",
        "rrd_updated_at": now,
        "live_grpc_url": "",
        "mode": "static",
        "camera": cam,
        "preview_camera": cam,
        "preview_entity": f"world/camera_frustums/{{cam}}/frustum",
        "rerun_ready": target.is_file() and viewer_ready,
        "rerun_iframe_url": _rerun_iframe_url(cam),
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
        "To view Franka immediately, tell users to open the **Rerun** tab and click **Load Franka in Rerun**",
        "(or POST /api/sim-viz/load-franka-demo). The UI has two tabs: **Chat** and **Rerun**.",
        "Artifact-first browsing flow: call `/api/artifacts/runs`, inspect `/api/artifacts/run/{{id}}`,",
        "then `POST /api/sim-viz/load-artifact` with explicit `s3_uri` or `run_id` + `key`.",
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
            "Always use real registry-qualified images from your Nebius container registry",
            "(or `NPA_REGISTRY` / `container_registry` in ~/.npa/config.yaml); never keep",
            "`<your-registry-id>` placeholders in runnable workflows.",
            "For BYOF solution onboarding, use `npa workbench byof run`",
            "(or `npa/scripts/run_byof_repo.py`) to containerize an OSS repo,",
            "push to the configured Nebius registry, then launch a real Isaac-Lab run",
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

{_AGENT_RRD_PROXY_EMBED}

{_AGENT_STAGES_EMBED}

{_AGENT_CHAT_EMBED}

{_AGENT_ACTIONS_EMBED}

{_AGENT_SIM2REAL_LOOP_EMBED}

{_AGENT_SEMANTIC_ROUTER_EMBED}

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
            _publish_rrd_recording(rrd_path)
            try:
                shutil.copy2(rrd_path, RRD_PATH)
            except Exception:
                pass
            _restart_rerun_serve(force=True)
            _wait_rerun_web_viewer_healthy()
            _append_run_log(details, f"Published Rerun recording: {{rrd_path}}")
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
        sim_viz.update(
            {{
                "run_id": run_id,
                "stage": "completed",
                "rrd_uri": f"file://{{RECORDING_PATH}}",
                "rrd_updated_at": _now_iso(),
                "rerun_ready": RECORDING_PATH.is_file() and _rerun_web_viewer_healthy(),
                "rerun_iframe_url": _rerun_iframe_url(str(sim_viz.get("camera") or "workspace")),
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


def _write_soperator_temp_spec(spec_text: str) -> Path:
    tmp_dir = Path("/tmp/npa-agent-soperator")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    path = tmp_dir / f"soperator-{{secrets.token_hex(8)}}.yaml"
    path.write_text(spec_text, encoding="utf-8")
    return path


def _soperator_spec_text_from_payload(body: dict) -> str:
    spec_text = str(body.get("spec_yaml") or body.get("yaml") or "").strip()
    if spec_text:
        return spec_text
    spec = body.get("spec")
    if isinstance(spec, dict):
        return yaml.safe_dump(spec, sort_keys=False)
    spec_path = str(body.get("spec_path") or "").strip()
    if spec_path:
        path = Path(spec_path).expanduser()
        if not path.is_file():
            raise HTTPException(status_code=400, detail=f"soperator spec file not found: {{spec_path}}")
        return path.read_text(encoding="utf-8")
    raise HTTPException(status_code=400, detail="Provide spec_yaml, yaml, spec, or spec_path")


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
            "worker_pools": [pool.name for pool in spec.workers],
            "docker_cache_pools": [pool.name for pool in spec.workers if pool.docker_cache],
            "workers": [
                {{
                    "name": pool.name,
                    "platform": pool.platform,
                    "preset": pool.preset,
                    "size": pool.size,
                    "preemptible": pool.preemptible,
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
    ready, reason = _agent_npa_ready()
    if not ready:
        return {{"ok": False, "status": "blocked", "error": reason}}
    dry_run = bool(body.get("dry_run", False))
    validation = _soperator_validate_payload(body)
    if not validation.get("ok"):
        return {{"ok": False, "status": "invalid", "validation": validation}}
    if dry_run:
        return {{
            "ok": True,
            "status": "dry-run",
            "dry_run": True,
            "validation": validation,
            "command": "npa soperator deploy --spec <validated-spec> --output json",
        }}
    timeout_minutes = int(body.get("timeout_minutes") or body.get("timeout") or 90)
    project = _agent_project_alias(str(body.get("project") or ""))
    terraform_dir_text = str(body.get("terraform_dir") or "").strip()
    terraform_dir = Path(terraform_dir_text).expanduser() if terraform_dir_text else None
    ref = str(body.get("ref") or body.get("solutions_library_ref") or "main")
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
            apply_fixes=apply_fixes,
            on_status=lambda msg: None,
        )
        return {{
            "ok": True,
            "status": "deployed",
            "dry_run": False,
            "validation": validation,
            "result": result,
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
    "live_infra_loop": ("submit-workflow", "gpu-selection"),
    "mk8s_provision": ("nebius-infra", "submit-workflow"),
    "soperator": ("soperator", "nebius-infra"),
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
        match = re.search(r"\b(agent-run-[A-Za-z0-9_-]+|sim2real-[A-Za-z0-9_.:-]+)\b", str(user_text or ""))
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
    elif intent in {"infra_backends", "mk8s_provision"}:
        state["infra"] = _agent_k8s_backends()
        _save_state(state)
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
    }}:
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
    if not re.search(r"\b(?:sim\s*[- ]?2\s*[- ]?real|sim2real|pipeline|workflow)\b", lowered):
        return None
    if not re.search(r"\b(?:stage|stages|step|steps)\b", lowered):
        return None
    if not re.search(r"\b(?:count|number|how many)\b", lowered):
        return None
    value = _sim2real_stage_count_from_report(state)
    if value <= 0:
        return None
    match = re.search(r"(?:count|number|stages?|steps?)\s*(?:-|minus)\s*(\d+)", lowered)
    if match:
        value -= int(match.group(1))
    return str(value)

def _agent_chat_with_tools(*, raw_messages: list, model: str) -> dict | None:
    last_user = _last_user_message(raw_messages)
    if not last_user:
        return None
    numeric_reply = _maybe_stage_count_numeric_reply(last_user, _load_state())
    if numeric_reply is not None:
        return {{
            "ok": True,
            "model": model,
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
    model = str(payload.get("model") or LLM_MODEL).strip() or LLM_MODEL
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
    # Small Sim2Real chat shortcut — persist the turn (do not return before session save).
    if (not visual_turn) and re.search(
        r"\b(?:run|start|submit|launch)\b.{{0,80}}\b(?:small|simple|tiny|minimal)\b.{{0,80}}\bsim(?:\s*[- ]?2\s*[- ]?real|2real)\b",
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
            "model": model,
            "reply": reply,
            "reasoning": None,
            "grounded": True,
            "apis_used": ["workflows/sim2real/submit"],
            "submit": submit,
            "session_id": session["id"],
            "session": {{
                "id": session["id"],
                "title": session["title"],
                "memory_uri": session.get("memory_uri", ""),
                "message_count": len(session.get("chat_history", [])),
            }},
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
            "model": model,
            "reply": meta_reply,
            "reasoning": None,
            "grounded": True,
            "tier": "grounded-metadata",
            "visual_kind": visual_kind,
            "apis_used": ["sim-viz/status"],
            "skills_used": ["agent-visual-feedback"],
            "session_id": session["id"],
            "session": {{
                "id": session["id"],
                "title": session["title"],
                "memory_uri": session.get("memory_uri", ""),
                "message_count": len(session.get("chat_history", [])),
            }},
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
        tool_result["session"] = {{
            "id": session["id"],
            "title": session["title"],
            "memory_uri": session.get("memory_uri", ""),
            "message_count": len(session.get("chat_history", [])),
        }}
        _save_state(state)
        return tool_result
    live_ctx = format_live_context_block(_load_state())
    last_user = text_from_messages(llm_messages)
    intent = match_chat_intent(last_user) if not visual_turn else None
    # Semantic fallthrough (Phase D): only when the deterministic regex router
    # missed. Keyword/cache hits cost 0 tokens; a genuine miss spends one cheap
    # structured call. A mapped intent produces a side-effect-free grounded reply.
    if intent is None and not visual_turn:
        semantic = _semantic_route(last_user)
        mapped = str(semantic.get("intent") or "") if semantic.get("mode") == "intent" else ""
        if mapped:
            grounded_zero = int(semantic.get("tokens") or 0) == 0
            sem_reply = build_grounded_reply(mapped, _load_state(), TOOL_REFS)
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
                "model": model,
                "reply": sem_reply,
                "reasoning": None,
                "grounded": grounded_zero,
                "tier": "semantic-" + str(semantic.get("source") or "model"),
                "usage": {{"total_tokens": int(semantic.get("tokens") or 0)}},
                "semantic_intent": mapped,
                "apis_used": apis_for_intent(mapped),
                "session_id": session["id"],
                "session": {{
                    "id": session["id"],
                    "title": session["title"],
                    "memory_uri": session.get("memory_uri", ""),
                    "message_count": len(session.get("chat_history", [])),
                }},
            }}
    # Cost-tier routing: vision when an image is attached; otherwise escalate
    # Describe-this metadata-only turns to reasoning (not cheap caption fluff).
    tier = classify_tier(last_user, intent=intent, messages=llm_messages)
    if visual_turn and tier != TIER_VISION:
        tier = TIER_REASONING
    explicit_model = str(payload.get("model") or "").strip()
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
    try:
        message = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise HTTPException(status_code=502, detail="LLM response missing assistant message") from exc
    reply, reasoning = _split_reasoning(message)
    if not reply and reasoning:
        reply = reasoning
        reasoning = None
    state = _load_state()
    session = _get_chat_session(state, session_id)
    history = list(merged_history)
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
        "tier": tier,
        "usage": turn_usage,
        "input_budget_ok": budget_ok,
        "visual_kind": visual_kind if visual_turn else "",
        "session_id": session["id"],
        "session": {{
            "id": session["id"],
            "title": session["title"],
            "memory_uri": session.get("memory_uri", ""),
            "message_count": len(session.get("chat_history", [])),
        }},
        "skills_used": skill_names,
    }}

def _consume_agent_confirm_token():
    # Single-use consume: return the pending (token, digest) and clear the gate
    # in state before any side effect so a replayed request cannot re-authorize.
    state = _load_state()
    act_state = state.get("agent_act")
    if not isinstance(act_state, dict):
        return "", ""
    token = str(act_state.get("confirm_token") or "")
    digest = str(act_state.get("confirm_digest") or "")
    if token:
        act_state["confirm_token"] = ""
        act_state["confirm_digest"] = ""
        act_state["pending_action"] = None
        state["agent_act"] = act_state
        _save_state(state)
    return token, digest

def _issue_agent_confirm_token(action, digest):
    # Issue a fresh token bound to a specific proposed action digest.
    token = secrets.token_hex(8)
    state = _load_state()
    act_state = state.get("agent_act")
    if not isinstance(act_state, dict):
        act_state = {{}}
    act_state["confirm_token"] = token
    act_state["confirm_digest"] = str(digest or "")
    act_state["pending_action"] = action if isinstance(action, dict) else {{}}
    state["agent_act"] = act_state
    _save_state(state)
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
        return _act_response_to_dict(artifacts_runs(prefix=str(args.get("prefix") or ""), limit=limit))

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

    def _tool_submit(args):
        return _act_response_to_dict(submit_sim2real({{"run_id": str(args.get("run_id") or "")}}))

    return {{
        "health": _tool_health,
        "sim_viz_status": _tool_sim_viz_status,
        "sim2real_status": _tool_sim2real_status,
        "artifacts_runs": _tool_artifacts_runs,
        "artifacts_run": _tool_artifacts_run,
        "workflow_validate_spec": _tool_validate,
        "workflow_plan_spec": _tool_plan,
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
    # Consume the single-use confirmation token: read it, then clear the pending
    # gate in state before running the loop so a replay cannot re-authorize.
    session_token, confirm_digest = _consume_agent_confirm_token()
    confirm_token = str(body.get("confirm_token") or "").strip()
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
    return result

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
    session_token, confirm_digest = _consume_agent_confirm_token()
    confirm_token = str(body.get("confirm_token") or "").strip()
    default_run = ""
    sim_viz = state.get("sim_viz", {{}})
    if isinstance(sim_viz, dict):
        default_run = str(sim_viz.get("run_id") or "").strip()
    cfg = dict(config)
    if not str(cfg.get("run_id") or "").strip():
        cfg["run_id"] = default_run or f"agent-drive-{{secrets.token_hex(4)}}"
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
        notes = []
        sr = gate_result.get("success_rate") if isinstance(gate_result, dict) else None
        if sr is None:
            notes.append("no gate success_rate available on this run yet")
            mode = "insufficient_signal"
        elif float(sr) <= 0.0:
            notes.append("degenerate rollout: success_rate at floor")
            mode = "policy_collapse"
        else:
            notes.append("below-threshold success_rate")
            mode = "low_success"
        return {{"failure_mode": mode, "notes": "; ".join(notes)}}

    result = drive_sim2real_loop(
        goal,
        config=cfg,
        launch=_launch,
        status=_status,
        gate=_sim2real_gate_metrics,
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
    return result

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
    sim_viz = _sim_viz_for_run(state)
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
            payload["rerun_iframe_url"] = _rerun_iframe_url(camera)
    else:
        payload["rerun_iframe_url"] = ""
    if not payload.get("rrd_uri") and may_use_default_recording and RRD_PATH.is_file():
        payload["rrd_uri"] = f"file://{{RRD_PATH}}"
    mode = str(payload.get("mode") or "static").strip().lower()
    payload["mode"] = "live" if mode == "live" else "static"
    artifact_render = str(payload.get("artifact_render") or "").strip().lower()
    if artifact_render and artifact_render != "rerun":
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
    payload["available_runs"] = [
        {{
            "run_id": str(item.get("run_id") or "").strip(),
            "last_modified": str(
                item.get("rrd_updated_at")
                or item.get("updated_at")
                or item.get("submitted_at")
                or ""
            ).strip(),
            "stage": str(item.get("stage") or "").strip(),
        }}
        for item in _sim_viz_runs(state)
        if str(item.get("run_id") or "").strip()
    ]
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
    payload["available_runs"] = [
        {{
            "run_id": str(item.get("run_id") or "").strip(),
            "last_modified": str(
                item.get("rrd_updated_at")
                or item.get("updated_at")
                or item.get("submitted_at")
                or ""
            ).strip(),
            "stage": str(item.get("stage") or "").strip(),
        }}
        for item in _sim_viz_runs(state)
        if str(item.get("run_id") or "").strip()
    ]
    render = str(payload.get("artifact_render") or "").strip().lower()
    if render and render != "rerun":
        payload["rrd_uri"] = ""
        payload["rerun_ready"] = False
        if not payload.get("rerun_iframe_url"):
            payload["rerun_iframe_url"] = ""
    else:
        payload["rerun_ready"] = _rerun_ready_state(rrd_uri=str(payload.get("rrd_uri") or ""))
        if not payload.get("rerun_iframe_url"):
            payload["rerun_iframe_url"] = _rerun_iframe_url(str(payload.get("camera") or "workspace"))
    return payload


@app.post("/sim-viz/load-run")
def sim_viz_load_run(payload: dict | None = None):
    body = payload if isinstance(payload, dict) else {{}}
    run_id = str(body.get("run_id") or "").strip()
    if not run_id:
        raise HTTPException(status_code=400, detail="run_id is required")
    requested_camera = str(body.get("camera") or "").strip()
    camera = requested_camera or "workspace"
    requested_rrd_uri = str(body.get("rrd_uri") or "").strip()

    # Prefer a run-scoped Rerun recording over stale history entries. History can
    # contain JSON artifacts from prior clicks, which otherwise makes Load Run
    # show "Non-RRD artifact loaded" even when reports/sim2real.rrd exists.
    if requested_rrd_uri:
        s3, _settings = _agent_s3_client()
        bucket, key = parse_s3_uri(requested_rrd_uri)
        local_name = _artifact_filename(key)
        local_path = RECORDINGS_DIR / local_name
        download_s3_uri(requested_rrd_uri, local_path, s3=s3)
        state = _load_state()
        sim_viz = _apply_loaded_artifact(
            state=state,
            run_id=validate_run_id(run_id),
            key=key,
            s3_uri=requested_rrd_uri,
            render=render_hint_for_object(key=key),
            local_path=local_path,
        )
        if requested_camera:
            sim_viz["camera"] = _sim2real_pipeline_camera_label(camera) if _is_sim2real_pipeline_recording(key) else camera
            state["sim_viz"] = sim_viz
            _record_sim_viz_run(state, sim_viz)
            _save_state(state)
        return {{"ok": True, "sim_viz": _sim_viz_load_response(state, sim_viz, run_id=run_id)}}

    try:
        s3, settings = _agent_s3_client()
        effective_prefix = _artifact_discovery_prefix(settings, str(body.get("prefix") or ""))
        artifacts = list_artifacts(settings["bucket"], validate_run_id(run_id), prefix=effective_prefix, s3=s3)
        preferred = select_preferred_artifact(artifacts)
        if preferred and preferred.render == "rerun":
            local_name = _artifact_filename(preferred.key)
            local_path = RECORDINGS_DIR / local_name
            download_s3_uri(preferred.s3_uri, local_path, s3=s3)
            state = _load_state()
            sim_viz = _apply_loaded_artifact(
                state=state,
                run_id=run_id,
                key=preferred.key,
                s3_uri=preferred.s3_uri,
                render=preferred.render,
                local_path=local_path,
            )
            if requested_camera:
                sim_viz["camera"] = _sim2real_pipeline_camera_label(camera) if _is_sim2real_pipeline_recording(preferred.key) else camera
                state["sim_viz"] = sim_viz
                _record_sim_viz_run(state, sim_viz)
                _save_state(state)
            return {{
                "ok": True,
                "sim_viz": _sim_viz_load_response(state, sim_viz, run_id=run_id),
                "preferred": preferred.to_dict(),
            }}
    except Exception:
        # Fall back to the historical in-memory run selector below; callers still
        # get a useful 404 if the run has never been seen.
        pass

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
def artifacts_runs(prefix: str = "", limit: int = 50):
    try:
        s3, settings = _agent_s3_client()
        effective_prefix = _artifact_discovery_prefix(settings, prefix)
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
        effective_prefix = _artifact_discovery_prefix(settings, prefix)
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


@app.api_route("/artifacts/file/{{filename}}", methods=["GET", "HEAD"])
def artifact_file(filename: str):
    safe_name = Path(str(filename)).name
    if safe_name != filename:
        raise HTTPException(status_code=400, detail="invalid artifact filename")
    target = RECORDINGS_DIR / safe_name
    if not target.is_file():
        raise HTTPException(status_code=404, detail=f"artifact file not found: {{filename}}")
    # artifact_media_type comes from the embedded workflows/artifacts.py module.
    return FileResponse(str(target), media_type=artifact_media_type(safe_name))


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
        file_path = Path(uri[len("file://"):])
        if file_path.is_file():
            return FileResponse(str(file_path), media_type="application/octet-stream")
    if uri.startswith("http://") or uri.startswith("https://"):
        # Server-side fetch of session rrd_uri — hardened allowlist + size cap
        # (see embedded agent_rrd_proxy.rrd_proxy_uri_allowed / MAX_RRD_PROXY_BYTES).
        if not rrd_proxy_uri_allowed(uri):
            raise HTTPException(status_code=400, detail="Refusing to proxy disallowed rrd_uri host")
        try:
            chunks: list[bytes] = []
            total = 0
            with httpx.stream("GET", uri, timeout=20.0) as proxied:
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
        "preview_entity": f"world/camera_frustums/{{cam}}/frustum",
        "rerun_ready": _rerun_ready_state(rrd_uri=f"file://{{RRD_PATH}}"),
        "rerun_iframe_url": _rerun_iframe_url(cam),
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
@app.get("/infra/mk8s")
def list_k8s_infra(project: str = ""):
    return _agent_k8s_backends(project)


@app.post("/infra/provision")
@app.post("/infra/k8s/provision")
@app.post("/infra/mk8s/provision")
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
        proc = subprocess.run([str(script), run_id], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30, check=False)
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
sudo cp -f /opt/npa-agent/sim2real.rrd /opt/npa-agent/recordings/sim2real.rrd || true
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
EnvironmentFile=-/opt/npa-agent/nebius.env
EnvironmentFile=-/opt/npa-agent/s3.env
EnvironmentFile=-/opt/npa-agent/public.env
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
        .replace(_AGENT_ACTIONS_EMBED, agent_actions_source)
        .replace(_AGENT_SIM2REAL_LOOP_EMBED, agent_sim2real_loop_source)
        .replace(_AGENT_SEMANTIC_ROUTER_EMBED, agent_semantic_router_source)
        .replace(_AGENT_WORKFLOW_EMBED, agent_workflow_source)
        .replace(_AGENT_ARTIFACTS_EMBED, agent_artifacts_source)
        .replace(_AGENT_ROUTING_EMBED, agent_routing_source)
        .replace(_AGENT_VISUAL_FEEDBACK_EMBED, agent_visual_feedback_source)
        .replace(_AGENT_RRD_PROXY_EMBED, agent_rrd_proxy_source)
        .replace(_AGENT_STAGES_EMBED, agent_stages_source)
        .replace(_AGENT_UI_HTML_EMBED, rendered_agent_ui_html())
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


@app.command("preflight")
def preflight_cmd(
    ssh_public_key_path: str = typer.Option(
        "~/.ssh/id_ed25519.pub",
        "--ssh-public-key-path",
        help="SSH public key path Terraform will read (its private key bootstraps the VM).",
    ),
    skip_nebius: bool = typer.Option(
        False, "--skip-nebius", help="Skip the live Nebius authentication check."
    ),
    output_json: bool = typer.Option(False, "--json", help="Print the report as JSON."),
) -> None:
    """Check Route C prerequisites before `npa agent deploy` / `fresh-setup`.

    Validates terraform, the SSH key pair, Nebius authentication, and the Token
    Factory key with no cloud side effects, so late failures (which auto-roll-back
    a freshly provisioned VM) surface up front. Exits non-zero on any FAIL.
    """
    results = list(_agent_hard_prereq_results(ssh_public_key_path))
    if not skip_nebius:
        results.append(_agent_nebius_auth_result())
    results.append(_agent_token_factory_result())
    has_fail = _render_agent_checks(results, output_json=output_json)
    if has_fail:
        raise typer.Exit(code=1)


@app.command("deploy")
def deploy_cmd(
    project: str = typer.Option(DEFAULT_PROJECT_ALIAS, "--project", help="NPA project alias to store config under."),
    name: str = typer.Option(DEFAULT_AGENT_NAME, "--name", help="Agent deployment name."),
    project_id: str = typer.Option("", "--project-id", help="Nebius project ID."),
    tenant_id: str = typer.Option("", "--tenant-id", help="Nebius tenant ID."),
    region: str = typer.Option("eu-north1", "--region", help="Nebius region."),
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

    # Fail fast on cheap, side-effect-free prerequisites BEFORE any cloud IAM
    # side effects or Terraform apply: a missing terraform binary or SSH key
    # otherwise surfaces mid-run (as a raw Terraform file() error or a late
    # provisioner failure) after infrastructure has already been touched. Surface
    # the Token Factory warning here too, rather than only after the VM exists.
    # Resolve the deploy LLM creds once and thread them through to the VM
    # bootstrap below.
    tf_api_key, default_llm_model = _resolve_deploy_llm_credentials()
    prereq_results = _agent_hard_prereq_results(ssh_public_key_path)
    tf_key_result = _agent_token_factory_result(tf_api_key)
    for result in prereq_results:
        if result.status == "FAIL":
            _fail(f"{result.summary} {result.remedy}".strip())
    if tf_key_result.status == "WARN":
        typer.echo(f"  Warning: {tf_key_result.summary}", err=True)
        typer.echo(f"           {tf_key_result.remedy}", err=True)

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
            project_alias=project,
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
    _persist_agent_project_config(
        project=project,
        project_id=env_project_id,
        tenant_id=env_tenant_id,
        region=env_region,
        merged_vars=merged_vars,
    )

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
    # tf_api_key / default_llm_model were resolved once up front (before Terraform).
    configured_llm_model = str(llm_model or "").strip() or default_llm_model
    # With no explicit --llm-models, seed the cost-ordered default ladder so
    # per-turn routing can reach every tier (cheap/standard/reasoning/vision)
    # out of the box. An explicit --llm-models acts as a governance allowlist.
    extra_llm_models = list(llm_models) if llm_models else list(DEFAULT_LLM_MODELS)
    configured_llm_models = _normalize_llm_models([configured_llm_model, *extra_llm_models])
    nebius_ai_key, _ = _resolve_operator_credentials()
    # A missing Token Factory key is already surfaced up front (before Terraform)
    # by the deploy prerequisite check above.
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
def fresh_setup_cmd(
    project: str = typer.Option(DEFAULT_PROJECT_ALIAS, "--project", help="NPA project alias for this fresh environment."),
    name: str = typer.Option(DEFAULT_AGENT_NAME, "--name", help="Agent deployment name."),
    project_id: str = typer.Option(..., "--project-id", help="Nebius project ID."),
    tenant_id: str = typer.Option(..., "--tenant-id", help="Nebius tenant ID."),
    region: str = typer.Option("eu-north1", "--region", help="Nebius region."),
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
    # No explicit --llm-models => seed the cost-ordered default ladder (all
    # tiers). Existing record models are still merged below, so re-bootstrap
    # keeps any previously configured set.
    extra_llm_models = list(llm_models) if llm_models else list(DEFAULT_LLM_MODELS)
    resolved_llm_models = _normalize_llm_models([resolved_llm_model, *extra_llm_models])
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
    if not region:
        _fail("agent record is missing its deploy region")
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
    if not isinstance(submitted_status, dict) or submitted_status.get("run_id") != submit_run_id:
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
        _fail("submitted sim2real run rrd-blob endpoint returned unexpectedly small payload")

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
        _fail(
            "rerun bundle load budget failed: "
            + "; ".join(bundle_result.errors[:4])
        )

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
        'id="tabChat"',
        'id="tabRerun"',
        'id="stagesPanel"',
        "<h3>Stages</h3>",
        'id="stagesRunSelect"',
        'id="stagesLoadRun"',
        "loadSelectedRun",
        "stages-run-picker",
        "filterStagesRunSelect",
        "Search or paste run ID",
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
    ):
        if marker not in ui_html:
            _fail(f"UI html missing wiring marker: {marker}")
    if 'loading="lazy"' in ui_html:
        _fail("UI html must not use lazy-loading on the Rerun iframe")
    if ".tab-panel[hidden]" in ui_html:
        _fail("UI html must not hide tab panels with display:none via hidden attribute")
    if 'Mount the viewer immediately so "Loading application bundle" starts early' in ui_html:
        _fail("UI must not mount Rerun before bundle warm (exposes Loading application bundle)")
    if "await waitUntilRerunPastBundleSplash(iframe, 45000)" in ui_html:
        _fail("UI must not block mount on long splash wait (latency)")
    if "await waitUntilRerunPastBundleSplash(iframe, 120000)" in ui_html:
        _fail("UI must not block mount on long splash wait (latency)")
    load_art_src = ui_html.split("async function loadArtifact(payload)")[1].split("async function refresh()")[0]
    if "swapRerunRecordingInPlace" not in load_art_src:
        _fail("loadArtifact must soft-swap Rerun recordings instead of always remounting wasm")
    # Guard against regressions that put bare authenticated URLs on <video src>
    # (browsers omit Authorization headers for media elements under basic auth).
    if '`<video controls src="${previewUrl}">`' in ui_html or '<video controls src="${previewUrl}">' in ui_html:
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
    if "npa workbench byof run" not in onboard_reply and "run_byof_repo.py" not in onboard_reply:
        _fail("onboard_solution chat reply missing byof CLI or run_byof_repo.py command")
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
