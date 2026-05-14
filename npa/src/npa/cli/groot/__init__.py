"""npa workbench groot - NVIDIA Isaac GR00T runtime and policy tools."""

from __future__ import annotations

import json
import os
import re
import shlex
import tempfile
import time
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import typer
from rich.console import Console

from npa.cli.ingress import (
    ensure_alias_ingress,
    ensure_deploy_ingress,
    ingress_summary,
    register_byovm_alias,
    resolve_deploy_instance_id,
)
from npa.cli.path_contract import (
    PathContractError,
    validate_read_path,
    validate_write_path,
)
from npa.clients.config import (
    APP_STATUS_HEALTHY,
    APP_STATUS_INSTALL_FAILED,
    APP_STATUS_INSTALLING,
    APP_STATUS_PROVISIONED,
    ConfigError,
    SSHConfig,
    alias_has_terraform_state,
    default_project_name,
    default_workbench_name,
    list_projects,
    remove_workbench_config,
    resolve_config,
    resolve_container_registry,
    resolve_credentials,
    resolve_environment,
    resolve_project_storage,
    resolve_ssh_config,
    resolve_terraform_state,
    update_workbench_app_status,
    workbench_is_byovm,
    write_config,
)
from npa.clients.credentials import (
    apply_shared_credential_env,
    load_credentials,
    shared_credential_env,
    warn_if_hf_token_missing,
)
from npa.clients.env import (
    merge_env_file_content,
    render_redacted_env_diff,
    render_redacted_env_file,
)
from npa.clients.endpoint import EndpointError, service_endpoint
from npa.clients.huggingface import validate_hf_access
from npa.clients.http import HTTPClient, ServerError
from npa.clients.network import NetworkIngressError
from npa.clients.project_credentials import storage_env_for_project
from npa.clients.serverless import EndpointNotFoundError, ServerlessClient, ServerlessClientError
from npa.clients.ssh import SSHClient, SSHError
from npa.deploy import provisioner
from npa.deploy.configurator import (
    HealthCheckMode,
    audit_remote_env,
    docker_exec_cmd,
    health_check_auto,
    write_manifest,
    write_remote_docker_env_file,
)
from npa.deploy.cleanup import (
    CleanupPartialError,
    classify_alias_state,
    list_terraform_managed_resources,
    remove_partial_config_entry,
    terraform_destroy_partial,
)
from npa.deploy.byovm import (
    RUNTIME_HELP,
    apply_project_storage_vars,
    detect_gpu_info,
    gpu_config_fields,
    gpu_env_fields,
    is_byovm_runtime,
    resolve_byovm_target,
    select_visible_devices,
    workbench_storage_outputs,
)
from npa.deploy.images import container_image_for_tool
from npa.deploy.provisioner import ProvisionerError
from npa.serverless_common import (
    SubnetResolutionError,
    build_serverless_job_env,
    build_serverless_output_upload_cmd,
    resolve_gpu_platform,
    resolve_subnet,
    split_serverless_env,
    validate_output_path,
)

app = typer.Typer(
    name="groot",
    help="NVIDIA Isaac GR00T humanoid foundation-model workbench.",
    no_args_is_help=True,
)

console = Console(stderr=True)

_project_alias: str = ""
_workbench_name: str = ""

GROOT_RELEASE = "n1.7"
GROOT_RUNTIME_VERSION = "0.1.0"
GROOT_PYPI_PACKAGE = f"nvidia-gr00t-sdk=={GROOT_RUNTIME_VERSION}"
GROOT_REPO_URL = "https://github.com/NVIDIA/Isaac-GR00T.git"
GROOT_REPO_REF = "3df8b3825d67f755e69141446f4315f281b9b7e6"
COSMOS_REASON_MODEL = "nvidia/Cosmos-Reason2-2B"
COSMOS_REASON_REVISION = "9ce19a195e423419c349abfc86fd07178b230561"
GROOT_HOME = "/opt/groot"
GROOT_DATA_MOUNT = "/opt/groot-data"
GROOT_REPO = f"{GROOT_HOME}/Isaac-GR00T"
GROOT_VENV = f"{GROOT_REPO}/.venv"
GROOT_MODEL_DIR = f"{GROOT_DATA_MOUNT}/models"
GROOT_DATA_CACHE = f"{GROOT_DATA_MOUNT}/data_cache"
GROOT_CHECKPOINT_CACHE = f"{GROOT_DATA_MOUNT}/checkpoint_cache"
GROOT_OUTPUT_DIR = f"{GROOT_DATA_MOUNT}/outputs"
GROOT_BASE_MODEL_CACHE = f"{GROOT_DATA_MOUNT}/base_model_cache"
GROOT_EVAL_DATA_CACHE = f"{GROOT_DATA_MOUNT}/eval_data_cache"
GROOT_CONFIG_CACHE = f"{GROOT_DATA_MOUNT}/config_cache"
GROOT_SERVICE = "npa-groot-server"
GROOT_CONTAINER_NAME = "npa-groot"
GROOT_ENV_FILE = "/etc/npa-groot-server/env"
GROOT_CONTAINER_ENV_FILE = "/etc/npa-groot/env"
GROOT_CONTAINER_WORKBENCH_TYPE = "groot-container"
GROOT_REMOTE_ENV_NAMES = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_ENDPOINT_URL",
    "NEBIUS_S3_ENDPOINT",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "NGC_API_KEY",
    "NGC_ORG",
    "NGC_TEAM",
)
GROOT_CREDENTIAL_ENV_NAMES = (
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "NGC_API_KEY",
    "NGC_ORG",
    "NGC_TEAM",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_ENDPOINT_URL",
    "NEBIUS_S3_ENDPOINT",
    "NEBIUS_S3_BUCKET",
)
DEFAULT_MODEL = "nvidia/GR00T-N1.7-3B"
HF_MODEL_REVISIONS = {
    DEFAULT_MODEL: "2fc962b973bccdd5d8ce4f67cc63b264d6886495",
    COSMOS_REASON_MODEL: COSMOS_REASON_REVISION,
}
DEFAULT_EMBODIMENT_TAG = "NEW_EMBODIMENT"
DEFAULT_SERVER_PORT = 8080
GROOT_POLICY_PORT = 5555
ISAAC_LAB_VERSION = "2.3.2.post1"
ISAAC_LAB_HOME = "/opt/isaac-lab"
ISAAC_LAB_VENV = f"{ISAAC_LAB_HOME}/venv"
PIP_EXTRA_INDEX_URL = "https://pypi.nvidia.com"

SUPPORTED_EMBODIMENT_TAGS = (
    "OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT",
    "XDOF",
    "XDOF_SUBTASK",
    "REAL_G1",
    "REAL_R1_PRO_SHARPA",
    "REAL_R1_PRO_SHARPA_HUMAN",
    "REAL_R1_PRO_SHARPA_MAXINSIGHTS",
    "REAL_R1_PRO_SHARPA_MECKA",
    "UNITREE_G1",
    "SIMPLER_ENV_GOOGLE",
    "SIMPLER_ENV_WIDOWX",
    "LIBERO_PANDA",
    "NEW_EMBODIMENT",
)


def _groot_gated_models(model: str = DEFAULT_MODEL) -> list[str]:
    repos = [model or DEFAULT_MODEL]
    if COSMOS_REASON_MODEL not in repos:
        repos.append(COSMOS_REASON_MODEL)
    return repos


def _model_check_or_fail(
    *,
    credentials: Any,
    model: str,
    skip_model_check: bool,
    dry_run: bool,
    no_shared_creds: bool,
) -> None:
    if skip_model_check:
        for repo in _groot_gated_models(model):
            console.print(f"  HF access check skipped for {repo}")
        if dry_run:
            console.print("  [dry-run] HF gated-model validation skipped")
        return
    token = "" if no_shared_creds else credentials.hf_token
    if not token:
        warn_if_hf_token_missing(credentials, warn=console.print)
        for repo in _groot_gated_models(model):
            console.print(f"  HF access check skipped for {repo}")
        if dry_run:
            raise typer.Exit(1)
        return
    for repo in _groot_gated_models(model):
        result = validate_hf_access(token, repo)
        if not result.ok:
            _fail(result.error or f"Unable to validate Hugging Face access to {repo}")
        prefix = "[dry-run] " if dry_run else ""
        console.print(f"  {prefix}HF access ok: {repo}")


def _groot_service_env(
    *,
    credentials: Any,
    merged_vars: dict[str, str],
    storage_ep: str,
    bucket: str,
    env_region: str,
    server_port: int,
    service_env: dict[str, str],
    include_shared_creds: bool,
) -> dict[str, str]:
    env = {
        "GROOT_MODEL_PATH": DEFAULT_MODEL,
        "GROOT_EMBODIMENT_TAG": DEFAULT_EMBODIMENT_TAG,
        "GROOT_MODEL_DIR": GROOT_MODEL_DIR,
        "GROOT_OUTPUT_DIR": GROOT_OUTPUT_DIR,
        "GROOT_SERVER_PORT": str(server_port),
        "HF_HOME": f"{GROOT_DATA_MOUNT}/hf_cache",
        "HUGGINGFACE_HUB_CACHE": f"{GROOT_DATA_MOUNT}/hf_cache",
        "AWS_ACCESS_KEY_ID": merged_vars.get("nebius_api_key", ""),
        "AWS_SECRET_ACCESS_KEY": merged_vars.get("nebius_secret_key", ""),
        "AWS_ENDPOINT_URL": storage_ep,
        "NEBIUS_S3_ENDPOINT": storage_ep,
        "NEBIUS_S3_BUCKET": bucket,
        "NEBIUS_REGION": env_region,
        "OMNI_KIT_ACCEPT_EULA": "YES",
        "ACCEPT_EULA": "Y",
        "ISAACSIM_ACCEPT_EULA": "YES",
        "PYTHONUNBUFFERED": "1",
        **service_env,
    }
    return apply_shared_credential_env(env, credentials, include=include_shared_creds)


def _groot_audit_env(env: dict[str, str]) -> dict[str, str]:
    return {key: env[key] for key in GROOT_CREDENTIAL_ENV_NAMES if env.get(key)}


def _print_ngc_env_audit(
    *,
    credentials: Any,
    service_env: dict[str, str],
    remote_path: str,
) -> None:
    tokens = getattr(credentials, "tokens", {}) or {}
    ngc_api_key = getattr(credentials, "ngc_api_key", "") or tokens.get(
        "NGC_API_KEY", ""
    )
    if ngc_api_key and service_env.get("NGC_API_KEY"):
        console.print("    Credential audit: NGC credentials merged and written.")
    elif ngc_api_key:
        console.print(
            f"    Warning: NGC credentials configured but not written to {remote_path}"
        )
    else:
        console.print(
            "    Warning: NGC credentials not configured; continuing without NGC service env."
        )


EMBODIMENT_ALIASES = {
    "g1": "UNITREE_G1",
    "unitree_g1": "UNITREE_G1",
    "real_g1": "REAL_G1",
    "droid": "OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT",
    "libero": "LIBERO_PANDA",
    "panda": "LIBERO_PANDA",
    "widowx": "SIMPLER_ENV_WIDOWX",
    "google_robot": "SIMPLER_ENV_GOOGLE",
    "new": "NEW_EMBODIMENT",
    "custom": "NEW_EMBODIMENT",
}


class OutputFormat(str, Enum):
    text = "text"
    json = "json"


class WorkbenchRuntime(str, Enum):
    vm = "vm"
    container = "container"
    byovm = "byovm"
    serverless = "serverless"


class InferenceMode(str, Enum):
    pytorch = "pytorch"
    tensorrt = "tensorrt"


class ConvertDirection(str, Enum):
    lerobot_to_groot = "lerobot-to-groot"
    groot_to_lerobot = "groot-to-lerobot"


@app.callback()
def main(
    project: str = typer.Option(
        "",
        "--project",
        "-p",
        help="Project alias from ~/.npa/config.yaml.",
    ),
    name: str = typer.Option(
        "",
        "--name",
        "-n",
        help="Workbench instance name within the project.",
    ),
) -> None:
    """NVIDIA Isaac GR00T runtime and policy tools."""
    global _project_alias, _workbench_name
    _project_alias = project
    _workbench_name = name


def _fail(msg: str, code: int = 1) -> None:
    console.print(f"[red]Error:[/red] {msg}")
    raise typer.Exit(code)


def _confirm_or_exit(prompt: str) -> None:
    if not typer.confirm(prompt, default=False):
        typer.echo("Aborted.")
        raise typer.Exit(code=1)


def _output(data: dict[str, Any], fmt: OutputFormat) -> None:
    if fmt == OutputFormat.json:
        typer.echo(json.dumps(data, indent=2))
    else:
        for key, val in data.items():
            typer.echo(f"  {key}: {val}")


def _get_config(**overrides: str):
    try:
        return resolve_config(
            project=_project_alias or None,
            name=_workbench_name or None,
            **{k: v for k, v in overrides.items() if v is not None},
        )
    except ConfigError as exc:
        _fail(str(exc))


def _get_ssh_config(**overrides: str):
    try:
        return resolve_ssh_config(
            project=_project_alias or None,
            name=_workbench_name or None,
            **{k: v for k, v in overrides.items() if v is not None},
        )
    except ConfigError as exc:
        _fail(str(exc))


def _remote_bash(script: str) -> str:
    return f"bash -lc {shlex.quote(script)}"


def _remote_python(script: str) -> str:
    return f"{GROOT_VENV}/bin/python -c {shlex.quote(script)}"


def _is_s3_uri(path: str) -> bool:
    return path.startswith("s3://")


def _path_name(path: str, default: str = "artifact") -> str:
    if _is_s3_uri(path):
        parsed = urlparse(path)
        return Path(parsed.path.rstrip("/")).name or parsed.netloc or default
    return Path(path.rstrip("/")).name or default


def _model_slug(model: str) -> str:
    return (
        model.removeprefix("ngc://")
        .split("@", 1)[0]
        .replace("/", "--")
        .replace(":", "--")
    )


def _hf_revision_flag(model: str) -> str:
    revision = HF_MODEL_REVISIONS.get(model.split("@", 1)[0])
    if not revision:
        return ""
    return f" --revision {shlex.quote(revision)}"


def _is_hf_groot_model_ref(ref: str) -> bool:
    return ref.startswith("nvidia/GR00T-") or ref == DEFAULT_MODEL


@app.command("ensure-ingress")
def ensure_ingress_cmd(
    name: str = typer.Option(
        "",
        "--name",
        "-n",
        help="Workbench alias to repair. Defaults to the active workbench alias.",
    ),
    source: str = typer.Option(
        "0.0.0.0/0",
        "--source",
        help="Source CIDR allowed to reach the GR00T server.",
    ),
) -> None:
    """Ensure public ingress for the saved GR00T BYOVM alias."""
    try:
        result = ensure_alias_ingress(
            tool="groot",
            port=8082,
            project_alias=_project_alias or None,
            name=name or _workbench_name or None,
            source=source,
        )
    except (ConfigError, NetworkIngressError) as exc:
        _fail(str(exc))
    typer.echo(ingress_summary(result, 8082))


@app.command("register-byovm")
def register_byovm_cmd(
    alias: str = typer.Option(
        ..., "--alias", help="Workbench alias to create or update."
    ),
    instance_id: str = typer.Option(
        ..., "--instance-id", help="Nebius compute instance ID."
    ),
    port: int = typer.Option(8082, "--port", help="GR00T HTTP service port."),
) -> None:
    """Register an existing VM as a GR00T BYOVM alias and ensure ingress."""
    try:
        register_byovm_alias(
            tool="groot",
            alias=alias,
            instance_id=instance_id,
            port=port,
            project_alias=_project_alias or None,
            warn=console.print,
        )
    except (ConfigError, NetworkIngressError) as exc:
        _fail(str(exc))


def _cache_dir(kind: str, uri: str) -> str:
    parsed = urlparse(uri)
    cache_key = f"{parsed.netloc}_{parsed.path.strip('/').replace('/', '_')}"
    return f"{GROOT_DATA_MOUNT}/{kind}_cache/{cache_key}"


def _normalize_embodiment_tag(tag: str) -> str:
    value = (tag or DEFAULT_EMBODIMENT_TAG).strip()
    if not value:
        return DEFAULT_EMBODIMENT_TAG
    key = value.lower().replace("-", "_")
    if key in EMBODIMENT_ALIASES:
        return EMBODIMENT_ALIASES[key]
    return value


def _is_ngc_model_ref(model: str) -> bool:
    ref = model.removeprefix("ngc://")
    if model.startswith("ngc://"):
        return True
    if ":" in ref and "/" in ref:
        return True
    return ref.count("/") >= 2 and not ref.startswith("nvidia/GR00T-")


def _s3_path_name(path: str, default: str = "result.json") -> str:
    name = (
        Path(urlparse(path).path.rstrip("/")).name
        if _is_s3_uri(path)
        else Path(path).name
    )
    return name or default


def _remote_download_dir_cmd(uri: str, local_dir: str, endpoint_url: str = "") -> str:
    parsed = urlparse(uri)
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/").rstrip("/")
    prefix_with_slash = prefix + "/" if prefix else ""
    script = f"""
import os
import pathlib
import boto3

s3 = boto3.client(
    "s3",
    endpoint_url=os.environ.get("NEBIUS_S3_ENDPOINT") or os.environ.get("AWS_ENDPOINT_URL") or {endpoint_url!r} or None,
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID") or None,
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY") or None,
)
dest = pathlib.Path({local_dir!r})
dest.mkdir(parents=True, exist_ok=True)
for page in s3.get_paginator("list_objects_v2").paginate(Bucket={bucket!r}, Prefix={prefix_with_slash!r}):
    for obj in page.get("Contents", []):
        key = obj["Key"]
        rel = key[len({prefix_with_slash!r}):]
        if not rel:
            continue
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file({bucket!r}, key, str(target))
print("npa_s3_download_done")
"""
    return _remote_python(script)


def _remote_upload_dir_cmd(local_dir: str, uri: str, endpoint_url: str = "") -> str:
    parsed = urlparse(uri)
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/").rstrip("/")
    prefix_with_slash = prefix + "/" if prefix else ""
    script = f"""
import os
import pathlib
import boto3

s3 = boto3.client(
    "s3",
    endpoint_url=os.environ.get("NEBIUS_S3_ENDPOINT") or os.environ.get("AWS_ENDPOINT_URL") or {endpoint_url!r} or None,
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID") or None,
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY") or None,
)
base = pathlib.Path({local_dir!r})
for file_path in base.rglob("*"):
    if file_path.is_file():
        s3.upload_file(str(file_path), {bucket!r}, {prefix_with_slash!r} + str(file_path.relative_to(base)))
print("npa_s3_upload_done")
"""
    return _remote_python(script)


def _remote_upload_file_cmd(local_file: str, uri: str, endpoint_url: str = "") -> str:
    parsed = urlparse(uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    script = f"""
import os
import boto3

s3 = boto3.client(
    "s3",
    endpoint_url=os.environ.get("NEBIUS_S3_ENDPOINT") or os.environ.get("AWS_ENDPOINT_URL") or {endpoint_url!r} or None,
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID") or None,
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY") or None,
)
s3.upload_file({local_file!r}, {bucket!r}, {key!r})
print("npa_s3_upload_done")
"""
    return _remote_python(script)


def _storage_client_for_config(cfg: Any):
    from npa.clients.storage import StorageClient

    return StorageClient.from_environment(
        endpoint_url=cfg.storage.endpoint_url,
        aws_access_key_id=cfg.storage.aws_access_key_id,
        aws_secret_access_key=cfg.storage.aws_secret_access_key,
    )


def _storage_client_for_project_or_environment():
    from npa.clients.storage import StorageClient, StorageError

    try:
        state = resolve_terraform_state(_project_alias or None)
    except ConfigError:
        state = None

    try:
        return StorageClient.from_environment(
            endpoint_url=getattr(state, "endpoint", "") if state else "",
            aws_access_key_id=getattr(state, "access_key", "") if state else "",
            aws_secret_access_key=getattr(state, "secret_key", "") if state else "",
        )
    except StorageError as exc:
        _fail(str(exc))


def _storage_env_tokens(cfg: Any) -> dict[str, str]:
    storage = getattr(cfg, "storage", None)
    if storage is None:
        return {}
    tokens: dict[str, str] = {}
    endpoint_url = getattr(storage, "endpoint_url", "")
    if endpoint_url:
        tokens["AWS_ENDPOINT_URL"] = endpoint_url
        tokens["NEBIUS_S3_ENDPOINT"] = endpoint_url
    aws_access_key_id = getattr(storage, "aws_access_key_id", "")
    if aws_access_key_id:
        tokens["AWS_ACCESS_KEY_ID"] = aws_access_key_id
    aws_secret_access_key = getattr(storage, "aws_secret_access_key", "")
    if aws_secret_access_key:
        tokens["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key
    return tokens


def _ssh_client(cfg: Any, *, extra_tokens: dict[str, str] | None = None) -> SSHClient:
    tokens = dict(getattr(cfg.ssh, "tokens", {}) or {})
    tokens.update(_storage_env_tokens(cfg))
    for key, value in (extra_tokens or {}).items():
        if value:
            tokens[key] = value
    ssh_cfg = SSHConfig(
        host=cfg.ssh.host,
        user=cfg.ssh.user,
        key_path=cfg.ssh.key_path,
        tokens=tokens,
    )
    return SSHClient(ssh_cfg)


def _is_container_config(cfg: Any) -> bool:
    return str(getattr(cfg, "runtime", "vm")) == WorkbenchRuntime.container.value


def _container_exec_env(names: tuple[str, ...]) -> str:
    if not names:
        return ""
    env_assignments = " ".join(f'{name}="${{{name}:-}}"' for name in names)
    env_flags = " ".join(f"-e {name}" for name in names)
    return f"sudo env {env_assignments} docker exec {env_flags} {shlex.quote(GROOT_CONTAINER_NAME)} bash -lc "


def _container_exec(command: str, *, pass_env: tuple[str, ...] = ()) -> str:
    if not pass_env:
        return docker_exec_cmd(GROOT_CONTAINER_NAME, command)
    return _container_exec_env(pass_env) + shlex.quote(command)


def _runtime_command(
    cfg: Any,
    command: str,
    *,
    pass_env: tuple[str, ...] = (),
) -> str:
    if _is_container_config(cfg):
        return _container_exec(command, pass_env=pass_env)
    return command


def _is_serverless_runtime(runtime: Any) -> bool:
    return str(getattr(runtime, "value", runtime)) == WorkbenchRuntime.serverless.value


def _remote_bash(script: str) -> str:
    return f"bash -lc {shlex.quote(script)}"


def _serverless_job_name(project: str, name: str, tool: str) -> str:
    raw = f"npa-{tool}-jobs-{project}-{name}".lower()
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9-]+", "-", raw)).strip("-")[:63]


def _serverless_job_env(
    project: str,
    output_path: str,
    extra_env: dict[str, str] | None = None,
) -> tuple[dict[str, str], dict[str, str]]:
    storage = resolve_project_storage(project)
    shared_env = shared_credential_env(load_credentials(environ={}))
    env = build_serverless_job_env(
        output_path=output_path,
        hf_token=shared_env.get("HF_TOKEN") or shared_env.get("HUGGING_FACE_HUB_TOKEN") or None,
        s3_credentials={
            "aws_access_key_id": storage.aws_access_key_id or shared_env.get("AWS_ACCESS_KEY_ID", ""),
            "aws_secret_access_key": storage.aws_secret_access_key or shared_env.get("AWS_SECRET_ACCESS_KEY", ""),
            "endpoint_url": storage.endpoint_url or shared_env.get("AWS_ENDPOINT_URL", ""),
        },
        extra_env=extra_env,
    )
    return split_serverless_env(env)


def _groot_serverless_infer_command(
    *,
    input_path: str,
    dataset_path: str,
    embodiment_tag: str,
    inference_mode: str,
    steps: int,
    action_horizon: int,
    model_variant: str,
) -> str:
    local_dir = "/tmp/npa-groot-infer"
    script = f"""
import json, os, pathlib, time

out = pathlib.Path("{local_dir}")
out.mkdir(parents=True, exist_ok=True)
started = time.time()
try:
    import gr00t  # noqa: F401
    import_status = "available"
except Exception as exc:
    import_status = f"unavailable: {{type(exc).__name__}}: {{exc}}"
manifest = {{
    "status": "success",
    "tool": "groot",
    "input_path": {input_path!r},
    "dataset_path": {dataset_path!r},
    "embodiment_tag": {embodiment_tag!r},
    "inference_mode": {inference_mode!r},
    "steps": {steps},
    "action_horizon": {action_horizon},
    "model_variant": {model_variant!r},
    "groot_import": import_status,
    "job": os.environ.get("NPA_JOB_NAME", ""),
    "duration_seconds": round(time.time() - started, 3),
}}
(out / "npa_groot_infer_results.json").write_text(json.dumps(manifest, indent=2))
(out / "predicted_actions.json").write_text(json.dumps({{"actions": [], "manifest": manifest}}, indent=2))
print("NPA_GROOT_SERVERLESS_INFER_DONE", os.environ.get("NPA_OUTPUT_PATH", ""), flush=True)
""".strip()
    upload = build_serverless_output_upload_cmd(local_dir, "")
    body = (
        'NPA_PYTHON_BIN="${NPA_PYTHON_BIN:-python3}"\n'
        'if ! command -v "$NPA_PYTHON_BIN" >/dev/null 2>&1; then NPA_PYTHON_BIN=python; fi\n'
        f'"$NPA_PYTHON_BIN" <<\'PY\'\n{script}\nPY\n{upload}'
    )
    return _remote_bash(body)


def _groot_serverless_infer(
    *,
    input_path: str,
    dataset_path: str,
    output_path: str,
    project_id: str,
    image: str,
    gpu_type: str,
    gpu_count: int,
    gpu_preset: str,
    subnet_id: str,
    job_name: str,
    submit_only: bool,
    poll_interval: float,
    timeout: float,
    embodiment_tag: str,
    inference_mode: str,
    steps: int,
    action_horizon: int,
    model_variant: str,
    output: OutputFormat,
) -> None:
    try:
        validate_output_path(output_path)
        platform, preset, resolved_gpu_count = resolve_gpu_platform(gpu_type, gpu_count)
    except ValueError as exc:
        _fail(str(exc))
    if gpu_preset:
        preset = gpu_preset
    proj_alias = _project_alias or default_project_name()
    wb_name = _workbench_name or default_workbench_name()
    env_cfg = resolve_environment(proj_alias)
    resolved_project_id = project_id or (env_cfg.project_id if env_cfg else "")
    if not resolved_project_id:
        _fail("GR00T infer --runtime serverless requires --project-id or a configured project.")
    name = job_name or _serverless_job_name(proj_alias, wb_name, "groot")
    out = output_path.rstrip("/") + "/"
    try:
        subnet = resolve_subnet(
            project_id=resolved_project_id,
            explicit_subnet_id=subnet_id,
        )
    except SubnetResolutionError as exc:
        _fail(str(exc))
    env, extra_env = _serverless_job_env(
        proj_alias,
        out,
        {
            "NPA_JOB_NAME": name,
            "GROOT_SERVERLESS_SMOKE": "1",
            "GROOT_MODEL_VARIANT": model_variant,
        },
    )
    client = ServerlessClient()
    try:
        existing = client.get_job(name, resolved_project_id)
    except EndpointNotFoundError:
        existing = None
    try:
        if existing is not None:
            info = existing if submit_only or existing.status in {"succeeded", "failed", "cancelled"} else client.poll_job(existing.id, resolved_project_id, interval_s=poll_interval, ceiling_s=timeout)
            _output({"status": "existing", "job_id": info.id, "job_name": info.name, "job_status": info.status, "output_path": out}, output)
            return
        info = client.create_job(
            project_id=resolved_project_id,
            name=name,
            image=image
            or container_image_for_tool(
                "groot",
                registry=resolve_container_registry(proj_alias),
                tag=GROOT_RUNTIME_VERSION,
            ),
            command=_groot_serverless_infer_command(
                input_path=input_path,
                dataset_path=dataset_path,
                embodiment_tag=embodiment_tag,
                inference_mode=inference_mode,
                steps=steps,
                action_horizon=action_horizon,
                model_variant=model_variant,
            ),
            gpu_type=platform,
            gpu_count=resolved_gpu_count,
            preset=preset,
            subnet_id=subnet,
            output_path=out,
            env=env,
            extra_env=extra_env,
        )
        if not submit_only:
            info = client.poll_job(info.id, resolved_project_id, interval_s=poll_interval, ceiling_s=timeout)
    except ValueError as exc:
        _fail(str(exc))
    except ServerlessClientError as exc:
        _fail(f"Serverless Job failed: {exc}")
    except TimeoutError as exc:
        _fail(str(exc))
    _output({"status": "submitted" if submit_only else info.status, "job_id": info.id, "job_name": info.name, "output_path": out}, output)


def _service_env_lines(fields: dict[str, str] | None) -> str:
    if not fields:
        return ""
    return "\n".join(
        f"{key}={shlex.quote(value)}" for key, value in fields.items() if value
    )


def _gpu_env_from_config(cfg: Any) -> dict[str, str]:
    visible = str(getattr(cfg, "cuda_visible_devices", "") or "")
    if not visible:
        return {}
    env = {
        "CUDA_VISIBLE_DEVICES": visible,
        "NPA_GPU_COUNT": str(getattr(cfg, "gpu_count", "") or ""),
        "NPA_DETECTED_GPU_COUNT": str(getattr(cfg, "detected_gpu_count", "") or ""),
        "NPA_GPU_TYPE": str(getattr(cfg, "gpu_platform", "") or ""),
    }
    return {key: value for key, value in env.items() if value}


def _apply_saved_terraform_state(
    merged_vars: dict[str, str],
    *,
    project: str | None,
    explicit_vars: dict[str, str],
) -> None:
    state = resolve_terraform_state(project)
    mapping = {
        "s3_bucket": state.bucket,
        "s3_endpoint": state.endpoint,
        "nebius_api_key": state.access_key,
        "nebius_secret_key": state.secret_key,
    }
    for key, value in mapping.items():
        if value and key not in explicit_vars:
            merged_vars[key] = value


def _terraform_state_config(merged_vars: dict[str, str]) -> dict[str, str]:
    state = {
        "bucket": merged_vars.get("s3_bucket", ""),
        "endpoint": merged_vars.get("s3_endpoint", ""),
        "access_key": merged_vars.get("nebius_api_key", ""),
        "secret_key": merged_vars.get("nebius_secret_key", ""),
    }
    return {key: value for key, value in state.items() if value}


def _is_groot_workbench(name: str, wb_cfg: dict[str, Any]) -> bool:
    wtype = wb_cfg.get("workbench_type")
    if wtype:
        return wtype == "groot"

    normalized = name.replace("_", "-").lower()
    if "groot" in normalized or "gr00t" in normalized:
        return bool(wb_cfg.get("endpoint") or wb_cfg.get("ssh", {}).get("host"))
    return False


def _build_server_py(default_model: str, default_embodiment_tag: str) -> str:
    """Return the remote FastAPI server source for synchronous GR00T policy inference."""
    return f'''\
from __future__ import annotations

import base64
import json
import os
from importlib import metadata
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_DIR = Path(os.environ.get("GROOT_MODEL_DIR", "{GROOT_MODEL_DIR}"))
DEFAULT_MODEL = os.environ.get("GROOT_MODEL_PATH", "{default_model}")
DEFAULT_EMBODIMENT_TAG = os.environ.get("GROOT_EMBODIMENT_TAG", "{default_embodiment_tag}")

app = FastAPI(title="NPA GR00T Server")
_policy: Any | None = None
_loaded_model = ""
_loaded_embodiment_tag = ""


class ServeRequest(BaseModel):
    model_path: str | None = None
    embodiment_tag: str | None = None
    device: str = "cuda"


class InputFile(BaseModel):
    filename: str | None = None
    content_base64: str


class InferRequest(BaseModel):
    observation: dict[str, Any] | None = None
    input: InputFile | None = None


def _version(dist: str) -> str:
    try:
        return metadata.version(dist)
    except Exception:
        return "unknown"


def _model_slug(model: str) -> str:
    return model.removeprefix("ngc://").replace("/", "--").replace(":", "--")


def _local_model_path(model: str) -> str:
    candidate = MODEL_DIR / _model_slug(model)
    return str(candidate) if candidate.exists() else model


def _ngc_credentials_configured() -> bool:
    if os.environ.get("NGC_API_KEY"):
        return True
    cfg = Path.home() / ".ngc" / "config"
    return cfg.exists() and "apikey" in cfg.read_text(errors="ignore")


def _jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {{str(k): _jsonable(v) for k, v in value.items()}}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _decode_observation(req: InferRequest) -> dict[str, Any]:
    if req.observation is not None:
        return req.observation
    if req.input is None:
        raise HTTPException(status_code=400, detail="Provide observation or JSON input file")
    data = base64.b64decode(req.input.content_base64)
    try:
        payload = json.loads(data.decode("utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Input file must be JSON: {{exc}}") from exc
    if isinstance(payload, dict) and "observation" in payload and isinstance(payload["observation"], dict):
        return payload["observation"]
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Observation JSON must be an object")
    return payload


def _load(model_path: str | None = None, embodiment_tag: str | None = None, device: str = "cuda"):
    global _policy, _loaded_model, _loaded_embodiment_tag
    requested = model_path or DEFAULT_MODEL
    tag = embodiment_tag or DEFAULT_EMBODIMENT_TAG
    if _policy is not None and _loaded_model == requested and _loaded_embodiment_tag == tag:
        return _policy

    try:
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.policy.gr00t_policy import Gr00tPolicy
    except Exception as exc:
        raise RuntimeError(f"GR00T is not importable: {{exc}}") from exc

    resolved_tag = EmbodimentTag.resolve(tag)
    source = _local_model_path(requested)
    _policy = Gr00tPolicy(
        model_path=source,
        embodiment_tag=resolved_tag,
        device=device,
        strict=True,
    )
    _loaded_model = requested
    _loaded_embodiment_tag = tag
    return _policy


@app.get("/health")
def health() -> dict[str, Any]:
    return {{
        "status": "ok",
        "model": DEFAULT_MODEL,
        "loaded": _policy is not None,
        "loaded_model": _loaded_model,
        "embodiment_tag": _loaded_embodiment_tag or DEFAULT_EMBODIMENT_TAG,
        "groot_version": _version("gr00t"),
        "ngc_credentials_configured": _ngc_credentials_configured(),
    }}


@app.post("/serve")
def serve(req: ServeRequest) -> dict[str, Any]:
    model = req.model_path or DEFAULT_MODEL
    tag = req.embodiment_tag or DEFAULT_EMBODIMENT_TAG
    _load(model, tag, req.device)
    return {{
        "status": "serving",
        "model": model,
        "embodiment_tag": tag,
        "device": req.device,
    }}


@app.post("/infer")
def infer(req: InferRequest) -> dict[str, Any]:
    policy = _load()
    observation = _decode_observation(req)
    try:
        result = policy.get_action(observation)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"GR00T inference failed: {{exc}}") from exc

    if isinstance(result, tuple) and len(result) == 2:
        action, info = result
    else:
        action, info = result, {{}}
    return {{"actions": _jsonable(action), "info": _jsonable(info), "model": _loaded_model}}
'''


def _build_runtime_pin_patch_command() -> str:
    """Patch the pinned GR00T checkout so gated backbone loads are revision-stable."""
    return f"""\
python3.10 - <<'PY'
from pathlib import Path

repo = Path({GROOT_REPO!r})
cosmos_model = {COSMOS_REASON_MODEL!r}
cosmos_revision = {COSMOS_REASON_REVISION!r}


def replace_once(rel: str, old: str, new: str) -> None:
    path = repo / rel
    text = path.read_text()
    if old in text:
        path.write_text(text.replace(old, new, 1))
        return
    if new in text:
        return
    raise RuntimeError("Could not apply GR00T runtime pin patch to " + rel)


replace_once(
    "gr00t/configs/model/gr00t_n1d7.py",
    "    model_revision: str | None = None\\n",
    "    model_revision: str | None = \\"" + cosmos_revision + "\\"\\n",
)
replace_once(
    "gr00t/experiment/launch_finetune.py",
    "    config.model.model_name = \\"" + cosmos_model + "\\"\\n",
    "    config.model.model_name = \\"" + cosmos_model + "\\"\\n"
    "    config.model.model_revision = \\"" + cosmos_revision + "\\"\\n",
)
replace_once(
    "gr00t/model/gr00t_n1d7/gr00t_n1d7.py",
    "        super().__init__(config)\\n"
    "        self.config = config\\n\\n"
    "        backbone_cls = get_backbone_cls(config)\\n",
    "        super().__init__(config)\\n"
    "        self.config = config\\n"
    "        if (\\n"
    "            getattr(config, \\"model_name\\", \\"\\") == \\"" + cosmos_model + "\\"\\n"
    "            and not getattr(config, \\"model_revision\\", None)\\n"
    "        ):\\n"
    "            config.model_revision = \\"" + cosmos_revision + "\\"\\n"
    "        if getattr(config, \\"model_revision\\", None) and \\"revision\\" not in transformers_loading_kwargs:\\n"
    "            transformers_loading_kwargs = {{\\n"
    "                **transformers_loading_kwargs,\\n"
    "                \\"revision\\": config.model_revision,\\n"
    "            }}\\n\\n"
    "        backbone_cls = get_backbone_cls(config)\\n",
)
replace_once(
    "gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py",
    "        model_name: str = \\"" + cosmos_model + "\\",\\n"
    "        model_type: str = \\"qwen\\",\\n",
    "        model_name: str = \\"" + cosmos_model + "\\",\\n"
    "        model_revision: str | None = \\"" + cosmos_revision + "\\",\\n"
    "        model_type: str = \\"qwen\\",\\n",
)
replace_once(
    "gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py",
    "        self.model_name = model_name\\n"
    "        self.model_type = model_type\\n\\n",
    "        self.model_name = model_name\\n"
    "        self.model_revision = model_revision\\n"
    "        self.model_type = model_type\\n"
    "        if model_revision and \\"revision\\" not in transformers_loading_kwargs:\\n"
    "            transformers_loading_kwargs = {{\\n"
    "                **transformers_loading_kwargs,\\n"
    "                \\"revision\\": model_revision,\\n"
    "            }}\\n\\n",
)
replace_once(
    "gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py",
    "        processor_kwargs.setdefault(\\"model_name\\", \\"" + cosmos_model + "\\")\\n",
    "        processor_kwargs.setdefault(\\"model_name\\", \\"" + cosmos_model + "\\")\\n"
    "        processor_kwargs.setdefault(\\"model_revision\\", \\"" + cosmos_revision + "\\")\\n",
)
print("GROOT_RUNTIME_PIN_PATCH_OK " + cosmos_revision)
PY
"""


def _build_install_command(
    port: int = DEFAULT_SERVER_PORT,
    *,
    env_fields: dict[str, str] | None = None,
) -> str:
    server_py = _build_server_py(DEFAULT_MODEL, DEFAULT_EMBODIMENT_TAG)
    runtime_pin_patch = _build_runtime_pin_patch_command()
    extra_env = _service_env_lines(env_fields)
    script = f"""\
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
export OMNI_KIT_ACCEPT_EULA="${{OMNI_KIT_ACCEPT_EULA:-YES}}"
sudo apt-get update
sudo apt-get install -y software-properties-common build-essential git git-lfs curl unzip ffmpeg libsm6 libxext6 libglu1-mesa
git lfs install --system || true
if ! command -v python3.10 >/dev/null 2>&1; then
  sudo add-apt-repository -y ppa:deadsnakes/ppa || true
  sudo apt-get update
fi
sudo apt-get install -y python3.10 python3.10-venv python3.10-dev python3-pip
if ! command -v python3.11 >/dev/null 2>&1; then
  sudo add-apt-repository -y ppa:deadsnakes/ppa || true
  sudo apt-get update
fi
sudo apt-get install -y python3.11 python3.11-venv python3.11-dev
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
sudo mkdir -p {GROOT_HOME} {GROOT_DATA_MOUNT} {GROOT_MODEL_DIR} {GROOT_DATA_CACHE} {GROOT_CHECKPOINT_CACHE} {GROOT_OUTPUT_DIR} {GROOT_DATA_MOUNT}/checkpoints {GROOT_DATA_MOUNT}/hf_cache {ISAAC_LAB_HOME}
sudo chown -R "$USER:$USER" {GROOT_HOME}/ {GROOT_DATA_MOUNT}/ {ISAAC_LAB_HOME}
if ! command -v ngc >/dev/null 2>&1; then
  curl -L -o /tmp/ngccli_linux.zip https://ngc.nvidia.com/downloads/ngccli_linux.zip
  rm -rf /tmp/ngccli
  unzip -q /tmp/ngccli_linux.zip -d /tmp/ngccli
  sudo install /tmp/ngccli/ngc-cli/ngc /usr/local/bin/ngc
fi
if [ ! -d {GROOT_REPO}/.git ]; then
  git clone --recurse-submodules {shlex.quote(GROOT_REPO_URL)} {GROOT_REPO}
else
  git -C {GROOT_REPO} fetch --tags --recurse-submodules
fi
git -C {GROOT_REPO} checkout {shlex.quote(GROOT_REPO_REF)}
actual_groot_ref="$(git -C {GROOT_REPO} rev-parse HEAD)"
if [ "$actual_groot_ref" != {shlex.quote(GROOT_REPO_REF)} ]; then
  echo "ERROR: expected Isaac-GR00T ref {GROOT_REPO_REF}, got $actual_groot_ref" >&2
  exit 1
fi
git -C {GROOT_REPO} submodule update --init --recursive
{runtime_pin_patch}
cd {GROOT_REPO}
uv sync --python 3.10
uv pip install --python {GROOT_VENV}/bin/python boto3 fastapi "uvicorn[standard]"
python3.11 -m venv {ISAAC_LAB_VENV}
{ISAAC_LAB_VENV}/bin/python -m pip install --upgrade pip setuptools wheel
{ISAAC_LAB_VENV}/bin/python -m pip install "isaaclab[isaacsim,all]=={ISAAC_LAB_VERSION}" --extra-index-url {PIP_EXTRA_INDEX_URL}
cat > {GROOT_HOME}/server.py <<'PY'
{server_py}
PY
sudo mkdir -p /etc/npa-groot-server
sudo tee /etc/npa-groot-server/env >/dev/null <<'ENV'
GROOT_MODEL_PATH={DEFAULT_MODEL}
GROOT_EMBODIMENT_TAG={DEFAULT_EMBODIMENT_TAG}
GROOT_MODEL_DIR={GROOT_MODEL_DIR}
GROOT_OUTPUT_DIR={GROOT_OUTPUT_DIR}
GROOT_SERVER_PORT={port}
HF_HOME={GROOT_DATA_MOUNT}/hf_cache
HUGGINGFACE_HUB_CACHE={GROOT_DATA_MOUNT}/hf_cache
OMNI_KIT_ACCEPT_EULA=YES
{extra_env}
ENV
sudo tee /etc/systemd/system/{GROOT_SERVICE}.service >/dev/null <<'UNIT'
[Unit]
Description=NPA GR00T policy server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory={GROOT_HOME}
EnvironmentFile=/etc/npa-groot-server/env
ExecStart={GROOT_VENV}/bin/python -m uvicorn server:app --host 0.0.0.0 --port $GROOT_SERVER_PORT
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
UNIT
sudo systemctl daemon-reload
sudo systemctl enable {GROOT_SERVICE}
sudo systemctl restart {GROOT_SERVICE}
{GROOT_VENV}/bin/python - <<'PY'
import importlib
import importlib.metadata as metadata

importlib.import_module("gr00t")
version = metadata.version("gr00t")
if version != "{GROOT_RUNTIME_VERSION}":
    raise RuntimeError(f"expected gr00t {GROOT_RUNTIME_VERSION}, found {{version}}")
print("GR00T_ENV_SMOKE_OK")
print("gr00t_version=" + version)
PY
{ISAAC_LAB_VENV}/bin/python - <<'PY'
from importlib import metadata

import isaaclab

version = metadata.version("isaaclab")
if version != "{ISAAC_LAB_VERSION}":
    raise RuntimeError(f"expected isaaclab {ISAAC_LAB_VERSION}, found {{version}}")
print("ISAAC_LAB_ENV_SMOKE_OK")
print("isaaclab_version=" + version)
PY
"""
    return _remote_bash(script)


def _build_serve_command(
    model_path: str,
    embodiment_tag: str,
    port: int,
    *,
    env_fields: dict[str, str] | None = None,
) -> str:
    server_py = _build_server_py(model_path, embodiment_tag)
    payload = json.dumps(
        {
            "model_path": model_path,
            "embodiment_tag": embodiment_tag,
            "device": "cuda",
        }
    )
    extra_env = _service_env_lines(env_fields)
    script = f"""\
set -euo pipefail
cat > {GROOT_HOME}/server.py <<'PY'
{server_py}
PY
sudo mkdir -p /etc/npa-groot-server
sudo tee /etc/npa-groot-server/env >/dev/null <<'ENV'
GROOT_MODEL_PATH={model_path}
GROOT_EMBODIMENT_TAG={embodiment_tag}
GROOT_MODEL_DIR={GROOT_MODEL_DIR}
GROOT_OUTPUT_DIR={GROOT_OUTPUT_DIR}
GROOT_SERVER_PORT={port}
HF_HOME={GROOT_DATA_MOUNT}/hf_cache
HUGGINGFACE_HUB_CACHE={GROOT_DATA_MOUNT}/hf_cache
{extra_env}
ENV
sudo systemctl daemon-reload
sudo systemctl enable {GROOT_SERVICE}
sudo systemctl restart {GROOT_SERVICE}
for i in $(seq 1 120); do
  if curl -fsS -X POST "http://127.0.0.1:{port}/serve" \
    -H "Content-Type: application/json" \
    -d {shlex.quote(payload)}; then
    echo
    echo GROOT_SERVE_READY
    exit 0
  fi
  sleep 2
done
sudo systemctl --no-pager status {GROOT_SERVICE} || true
exit 1
"""
    return _remote_bash(script)


def _build_container_serve_command(
    model_path: str, embodiment_tag: str, port: int
) -> str:
    payload = json.dumps(
        {
            "model_path": model_path,
            "embodiment_tag": embodiment_tag,
            "device": "cuda",
        }
    )
    script = f"""\
set -euo pipefail
for i in $(seq 1 120); do
  if curl -fsS -X POST "http://127.0.0.1:{port}/serve" \
    -H "Content-Type: application/json" \
    -d {shlex.quote(payload)}; then
    echo
    echo GROOT_SERVE_READY
    exit 0
  fi
  sleep 2
done
echo "GR00T container server did not become ready on port {port}" >&2
exit 1
"""
    return _remote_bash(script)


def _build_reload_env_command(
    env_names: tuple[str, ...],
    *,
    port: int,
    restart: bool = True,
) -> str:
    names_json = json.dumps(list(env_names))
    env_assignments = " ".join(f'{name}="${{{name}:-}}"' for name in env_names)
    restart_block = ""
    if restart:
        restart_block = f"""
if [ "$mode" = "systemd" ]; then
  sudo systemctl restart {GROOT_SERVICE}
elif [ "$mode" = "container" ]; then
  sudo docker restart {GROOT_CONTAINER_NAME} >/dev/null
fi
for i in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:{port}/health" >/dev/null 2>/dev/null; then
    break
  fi
  sleep 1
done
curl -fsS "http://127.0.0.1:{port}/health" >/dev/null
"""
    script = f"""\
set -euo pipefail
server_env={GROOT_ENV_FILE}
container_env={GROOT_CONTAINER_ENV_FILE}
env_path=""
mode=""
if sudo test -f "$server_env"; then
  env_path="$server_env"
  mode="systemd"
elif sudo test -f "$container_env"; then
  env_path="$container_env"
  mode="container"
else
  echo "No GR00T service env file found" >&2
  exit 2
fi
sudo env {env_assignments} python3 - "$env_path" {shlex.quote(names_json)} <<'PY'
from pathlib import Path
import json
import os
import sys

path = Path(sys.argv[1])
env_names = json.loads(sys.argv[2])
updates = {{name: os.environ.get(name, "") for name in env_names if os.environ.get(name, "")}}
if not updates:
    raise SystemExit("No credential values were supplied")

lines = path.read_text().splitlines() if path.exists() else []
seen = set()
out = []
for line in lines:
    key = line.split("=", 1)[0] if "=" in line else ""
    if key in updates:
        if key not in seen:
            out.append(f"{{key}}={{updates[key]}}")
            seen.add(key)
        continue
    out.append(line)
for key, value in updates.items():
    if key not in seen:
        out.append(f"{{key}}={{value}}")
path.write_text("\\n".join(out).rstrip() + "\\n")
path.chmod(0o600)
print("updated_keys=" + ",".join(sorted(updates)))
PY
{restart_block}
echo "NPA_GROOT_RELOAD_ENV_COMPLETE env_path=$env_path mode=$mode"
    """
    return _remote_bash(script)


def _build_read_env_command() -> str:
    script = f"""\
set -euo pipefail
server_env={GROOT_ENV_FILE}
container_env={GROOT_CONTAINER_ENV_FILE}
env_path=""
mode=""
if sudo test -f "$server_env"; then
  env_path="$server_env"
  mode="systemd"
elif sudo test -f "$container_env"; then
  env_path="$container_env"
  mode="container"
else
  echo "NPA_GROOT_ENV_READ env_path= mode=missing"
  exit 0
fi
echo "NPA_GROOT_ENV_READ env_path=$env_path mode=$mode"
sudo cat "$env_path" || true
"""
    return _remote_bash(script)


def _parse_env_read(stdout: str) -> tuple[str, str, str]:
    env_path = ""
    mode = ""
    body: list[str] = []
    for line in stdout.splitlines():
        if line.startswith("NPA_GROOT_ENV_READ "):
            parts = dict(
                item.split("=", 1)
                for item in line.removeprefix("NPA_GROOT_ENV_READ ").split()
                if "=" in item
            )
            env_path = parts.get("env_path", "")
            mode = parts.get("mode", "")
            continue
        body.append(line)
    return env_path, mode, "\n".join(body) + ("\n" if body else "")


def _shared_groot_env_or_fail(credentials: Any) -> dict[str, str]:
    credential_env = {
        key: value
        for key, value in shared_credential_env(credentials).items()
        if key in GROOT_CREDENTIAL_ENV_NAMES and value
    }
    if not credential_env:
        _fail("No shared credentials found in environment or ~/.npa/credentials.yaml.")
    return credential_env


def _read_current_env_for_dry_run(cfg: Any, credential_env: dict[str, str]) -> tuple[str, str, str]:
    ssh = _ssh_client(cfg, extra_tokens=credential_env)
    try:
        _, stdout, _ = ssh.run_or_raise(_build_read_env_command(), stream=False)
    except SSHError:
        return "", "missing", ""
    return _parse_env_read(stdout)


def _apply_env_update(
    cfg: Any,
    credential_env: dict[str, str],
    *,
    service_port: int,
    restart: bool,
    output: OutputFormat,
) -> dict[str, Any]:
    ssh = _ssh_client(cfg, extra_tokens=credential_env)
    env_names = tuple(sorted(credential_env))
    try:
        _, stdout, stderr = ssh.run_or_raise(
            _build_reload_env_command(env_names, port=service_port, restart=restart),
            stream=output != OutputFormat.json,
        )
    except SSHError as exc:
        _fail(f"GR00T env reload failed: {exc}")

    env_path = ""
    env_mode = ""
    for line in stdout.splitlines():
        if line.startswith("NPA_GROOT_RELOAD_ENV_COMPLETE "):
            parts = dict(
                item.split("=", 1)
                for item in line.removeprefix("NPA_GROOT_RELOAD_ENV_COMPLETE ").split()
                if "=" in item
            )
            env_path = parts.get("env_path", "")
            env_mode = parts.get("mode", "")

    result: dict[str, Any] = {
        "updated_keys": list(env_names),
        "env_path": env_path,
        "mode": env_mode,
        "restarted": restart,
        "port": service_port,
    }
    if stderr.strip():
        result["stderr_tail"] = stderr.strip()[-1000:]
    return result


def _build_download_command(
    model: str, output_path: str, endpoint_url: str = ""
) -> str:
    model_ref = model.removeprefix("ngc://")
    slug = _model_slug(model)
    output_is_s3 = _is_s3_uri(output_path)
    local_dir = (
        f"{GROOT_MODEL_DIR}/{slug}" if not output_path or output_is_s3 else output_path
    )
    source_kind = "ngc" if _is_ngc_model_ref(model) else "hf"
    revision_flag = _hf_revision_flag(model)
    upload_cmd = (
        _remote_upload_dir_cmd(local_dir, output_path, endpoint_url)
        if output_is_s3
        else "true"
    )
    script = f"""\
set -euo pipefail
mkdir -p {shlex.quote(local_dir)}
cd {GROOT_REPO}
if [ {shlex.quote(source_kind)} = "ngc" ]; then
  if [ -n "${{NGC_API_KEY:-}}" ]; then
    mkdir -p "$HOME/.ngc"
    cat > "$HOME/.ngc/config" <<NGC
[CURRENT]
apikey = $NGC_API_KEY
format_type = ascii
NGC
    if [ -n "${{NGC_ORG:-}}" ]; then
      printf 'org = %s\\n' "$NGC_ORG" >> "$HOME/.ngc/config"
    fi
    if [ -n "${{NGC_TEAM:-}}" ]; then
      printf 'team = %s\\n' "$NGC_TEAM" >> "$HOME/.ngc/config"
    fi
    chmod 600 "$HOME/.ngc/config"
  fi
  ngc registry model download-version {shlex.quote(model_ref)} -d {shlex.quote(local_dir)}
else
  if [ -n "${{HF_TOKEN:-}}" ]; then
    uv run huggingface-cli download {shlex.quote(model)}{revision_flag} --local-dir {shlex.quote(local_dir)} --token "$HF_TOKEN"
  else
    uv run huggingface-cli download {shlex.quote(model)}{revision_flag} --local-dir {shlex.quote(local_dir)}
  fi
fi
{upload_cmd}
echo NPA_GROOT_DOWNLOAD_COMPLETE
echo {shlex.quote(local_dir)}
"""
    return _remote_bash(script)


def _resolve_remote_path_setup(
    ref: str, local_dir: str, endpoint_url: str
) -> tuple[str, str]:
    if _is_s3_uri(ref):
        return local_dir, _remote_download_dir_cmd(
            ref, local_dir, endpoint_url
        ) + " && "
    return ref, ""


def _resolve_remote_file_setup(
    ref: str, local_file: str, endpoint_url: str
) -> tuple[str, str]:
    if not _is_s3_uri(ref):
        return ref, ""
    parsed = urlparse(ref)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    script = f"""
import os
import pathlib
import boto3

s3 = boto3.client(
    "s3",
    endpoint_url=os.environ.get("NEBIUS_S3_ENDPOINT") or os.environ.get("AWS_ENDPOINT_URL") or {endpoint_url!r} or None,
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID") or None,
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY") or None,
)
target = pathlib.Path({local_file!r})
target.parent.mkdir(parents=True, exist_ok=True)
s3.download_file({bucket!r}, {key!r}, str(target))
print("npa_s3_download_done")
"""
    return local_file, _remote_python(script) + " && "


def _build_finetune_command(
    *,
    input_path: str,
    output_path: str,
    base_model: str,
    robot_embodiment: str,
    num_gpus: int,
    config: str,
    endpoint_url: str,
    max_steps: int | None = None,
    global_batch_size: int | None = None,
    dataloader_num_workers: int | None = None,
    save_steps: int | None = None,
    save_total_limit: int | None = None,
    save_only_model: bool = False,
) -> str:
    dataset_dir, dataset_setup = _resolve_remote_path_setup(
        input_path,
        _cache_dir("data", input_path),
        endpoint_url,
    )
    if _is_hf_groot_model_ref(base_model):
        resolved_base, base_setup = _resolve_infer_checkpoint_setup(
            base_model, endpoint_url
        )
    else:
        resolved_base, base_setup = _resolve_remote_path_setup(
            base_model,
            _cache_dir("base_model", base_model),
            endpoint_url,
        )
    resolved_config = ""
    config_setup = ""
    if config:
        resolved_config, config_setup = _resolve_remote_file_setup(
            config,
            f"{GROOT_CONFIG_CACHE}/{_path_name(config, 'config.yaml')}",
            endpoint_url,
        )

    output_is_s3 = _is_s3_uri(output_path)
    output_dir = (
        f"{GROOT_DATA_MOUNT}/checkpoints/finetune-{int(time.time())}"
        if output_is_s3
        else output_path
    )
    upload_cmd = (
        _remote_upload_dir_cmd(output_dir, output_path, endpoint_url)
        if output_is_s3
        else "true"
    )
    tag = _normalize_embodiment_tag(robot_embodiment)
    if num_gpus > 1:
        launcher = f"uv run torchrun --nproc_per_node={num_gpus} --master_port=29500 gr00t/experiment/launch_finetune.py"
    else:
        launcher = "uv run python gr00t/experiment/launch_finetune.py"
    train_args = ""
    if max_steps is not None:
        train_args += f" \\\n  --max-steps {max_steps}"
    if global_batch_size is not None:
        train_args += f" \\\n  --global-batch-size {global_batch_size}"
    if dataloader_num_workers is not None:
        train_args += f" \\\n  --dataloader-num-workers {dataloader_num_workers}"
    if save_steps is not None:
        train_args += f" \\\n  --save-steps {save_steps}"
    if save_total_limit is not None:
        train_args += f" \\\n  --save-total-limit {save_total_limit}"
    if save_only_model:
        train_args += " \\\n  --save-only-model"
    script = f"""\
set -euo pipefail
cd {GROOT_REPO}
mkdir -p {GROOT_DATA_CACHE} {GROOT_CHECKPOINT_CACHE} {GROOT_DATA_MOUNT}/checkpoints {GROOT_CONFIG_CACHE}
{dataset_setup}{base_setup}{config_setup}modality_config_path={shlex.quote(resolved_config)}
if [ -z "$modality_config_path" ] && [ -f {shlex.quote(dataset_dir)}/meta/npa_groot_modality_config.py ]; then
  modality_config_path={shlex.quote(dataset_dir)}/meta/npa_groot_modality_config.py
fi
modality_config_arg=()
if [ -n "$modality_config_path" ]; then
  modality_config_arg=(--modality-config-path "$modality_config_path")
fi
{launcher} \
  --base-model-path {shlex.quote(resolved_base)} \
  --dataset-path {shlex.quote(dataset_dir)} \
  --embodiment-tag {shlex.quote(tag)} \
  --num-gpus {num_gpus} \
  --output-dir {shlex.quote(output_dir)} \
  "${{modality_config_arg[@]}}"{train_args}
{upload_cmd}
echo NPA_GROOT_FINETUNE_COMPLETE
echo {shlex.quote(output_dir)}
"""
    return _remote_bash(script)


def _build_offline_eval_command(
    *,
    checkpoint_path: str,
    dataset_path: str,
    output_path: str,
    robot_embodiment: str,
    endpoint_url: str,
) -> str:
    checkpoint_dir, checkpoint_setup = _resolve_remote_path_setup(
        checkpoint_path,
        _cache_dir("checkpoint", checkpoint_path),
        endpoint_url,
    )
    dataset_dir, dataset_setup = _resolve_remote_path_setup(
        dataset_path,
        _cache_dir("eval_data", dataset_path),
        endpoint_url,
    )
    output_is_s3 = _is_s3_uri(output_path)
    output_dir = (
        f"{GROOT_OUTPUT_DIR}/offline-eval-{int(time.time())}"
        if output_is_s3
        else output_path
    )
    upload_cmd = (
        _remote_upload_dir_cmd(output_dir, output_path, endpoint_url)
        if output_is_s3
        else "true"
    )
    tag = _normalize_embodiment_tag(robot_embodiment)
    script = f"""\
set -euo pipefail
cd {GROOT_REPO}
mkdir -p {shlex.quote(output_dir)}
eval_plot_path={shlex.quote(f"{output_dir}/traj_0.jpeg")}
eval_log_path={shlex.quote(f"{output_dir}/open_loop_eval.log")}
{checkpoint_setup}{dataset_setup}{GROOT_VENV}/bin/python gr00t/eval/open_loop_eval.py \
  --dataset-path {shlex.quote(dataset_dir)} \
  --embodiment-tag {shlex.quote(tag)} \
  --model-path {shlex.quote(checkpoint_dir)} \
  --traj-ids 0 \
  --action-horizon 16 \
  --save-plot-path "$eval_plot_path" 2>&1 | tee "$eval_log_path"
{GROOT_VENV}/bin/python - <<'PY'
import json
import re
import time
from pathlib import Path

output_dir = Path({output_dir!r})
log_path = output_dir / "open_loop_eval.log"
log_text = log_path.read_text(errors="replace") if log_path.exists() else ""


def _last_float(pattern: str) -> float | None:
    matches = re.findall(pattern, log_text)
    return float(matches[-1]) if matches else None


metrics = {{
    "mse": _last_float(r"Average MSE across all trajs:\\s*([0-9.eE+-]+)")
           or _last_float(r"MSE for trajectory \\d+:\\s*([0-9.eE+-]+)"),
    "mae": _last_float(r"Average MAE across all trajs:\\s*([0-9.eE+-]+)")
           or _last_float(r"MAE:\\s*([0-9.eE+-]+)"),
}}
metrics = {{key: value for key, value in metrics.items() if value is not None}}
artifacts = sorted(str(path.relative_to(output_dir)) for path in output_dir.rglob("*") if path.is_file())
result = {{
    "status": "success",
    "metrics": metrics,
    "episode_count": 1,
    "checkpoint_path": {checkpoint_path!r},
    "dataset_path": {dataset_path!r},
    "robot_embodiment": {tag!r},
    "artifacts": artifacts,
    "created_unix": round(time.time(), 3),
}}
(output_dir / "npa_groot_eval_results.json").write_text(json.dumps(result, indent=2))
PY
{upload_cmd}
echo NPA_GROOT_OFFLINE_EVAL_COMPLETE
echo {shlex.quote(output_dir)}
"""
    return _remote_bash(script)


def _resolve_infer_checkpoint_setup(
    ref: str,
    endpoint_url: str,
) -> tuple[str, str]:
    if _is_s3_uri(ref):
        local_dir = _cache_dir("checkpoint", ref)
        return local_dir, _remote_download_dir_cmd(
            ref, local_dir, endpoint_url
        ) + " && "
    if _is_hf_groot_model_ref(ref):
        local_dir = f"{GROOT_MODEL_DIR}/{_model_slug(ref)}"
        revision_flag = _hf_revision_flag(ref)
        script = f"""\
if [ ! -f {shlex.quote(local_dir)}/config.json ]; then
  mkdir -p {shlex.quote(local_dir)}
  if [ -n "${{HF_TOKEN:-}}" ]; then
    uv run huggingface-cli download {shlex.quote(ref)}{revision_flag} --local-dir {shlex.quote(local_dir)} --token "$HF_TOKEN"
  else
    uv run huggingface-cli download {shlex.quote(ref)}{revision_flag} --local-dir {shlex.quote(local_dir)}
  fi
fi
"""
        return local_dir, script
    return ref, ""


def _build_infer_command(
    *,
    checkpoint_path: str,
    dataset_path: str,
    output_path: str,
    embodiment_tag: str,
    inference_mode: str,
    endpoint_url: str,
    steps: int,
    action_horizon: int,
    trt_engine_path: str,
) -> str:
    checkpoint_dir, checkpoint_setup = _resolve_infer_checkpoint_setup(
        checkpoint_path,
        endpoint_url,
    )
    dataset_dir, dataset_setup = _resolve_remote_path_setup(
        dataset_path,
        _cache_dir("infer_data", dataset_path),
        endpoint_url,
    )
    output_is_s3 = _is_s3_uri(output_path)
    output_dir = (
        f"{GROOT_OUTPUT_DIR}/infer-{int(time.time())}" if output_is_s3 else output_path
    )
    upload_cmd = (
        _remote_upload_dir_cmd(output_dir, output_path, endpoint_url)
        if output_is_s3
        else "true"
    )
    tag = _normalize_embodiment_tag(embodiment_tag)
    script = f"""\
set -euo pipefail
cd {GROOT_REPO}
mkdir -p {shlex.quote(output_dir)}
infer_plot_path={shlex.quote(f"{output_dir}/traj_0.jpeg")}
infer_log_path={shlex.quote(f"{output_dir}/standalone_inference.log")}
{checkpoint_setup}{dataset_setup}{GROOT_VENV}/bin/python - <<'PY' 2>&1 | tee "$infer_log_path"
import importlib.util
import json
from pathlib import Path

import numpy as np

from gr00t.data.embodiment_tags import EmbodimentTag

output_dir = Path({output_dir!r})
output_dir.mkdir(parents=True, exist_ok=True)

script_path = Path("scripts/deployment/standalone_inference_script.py")
spec = importlib.util.spec_from_file_location("npa_groot_standalone_infer", script_path)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {{script_path}}")
standalone = importlib.util.module_from_spec(spec)
spec.loader.exec_module(standalone)

args = standalone.ArgsConfig(
    model_path={checkpoint_dir!r},
    dataset_path={dataset_dir!r},
    embodiment_tag=EmbodimentTag.resolve({tag!r}),
    traj_ids=[0],
    inference_mode={inference_mode!r},
    trt_engine_path={trt_engine_path!r},
    steps={steps},
    action_horizon={action_horizon},
    save_plot_path=str(output_dir / "traj_0.jpeg"),
    get_performance_stats=True,
)
pred_actions, observation_vector = standalone.main(args)
arrays = {{
    f"trajectory_{{idx}}": np.asarray(actions)
    for idx, actions in enumerate(pred_actions)
}}
np.savez_compressed(output_dir / "predicted_actions.npz", **arrays)
preview = []
for idx, actions in enumerate(pred_actions):
    arr = np.asarray(actions)
    preview.append({{
        "trajectory_id": idx,
        "shape": list(arr.shape),
        "first_action": arr[0].tolist() if arr.size else [],
    }})
manifest = {{
    "status": "success",
    "checkpoint_path": {checkpoint_path!r},
    "dataset_path": {dataset_path!r},
    "embodiment_tag": {tag!r},
    "inference_mode": {inference_mode!r},
    "trajectory_count": len(pred_actions),
    "action_horizon": {action_horizon},
    "steps": {steps},
    "predicted_actions": preview,
    "observation_shape": list(np.asarray(observation_vector).shape)
    if observation_vector is not None else [],
    "artifacts": [],
}}
(output_dir / "npa_groot_infer_results.json").write_text(json.dumps(manifest, indent=2))
PY
{GROOT_VENV}/bin/python - <<'PY'
import json
from pathlib import Path

output_dir = Path({output_dir!r})
manifest_path = output_dir / "npa_groot_infer_results.json"
manifest = json.loads(manifest_path.read_text())
manifest["artifacts"] = sorted(
    str(path.relative_to(output_dir))
    for path in output_dir.rglob("*")
    if path.is_file()
)
manifest_path.write_text(json.dumps(manifest, indent=2))
PY
{upload_cmd}
echo NPA_GROOT_INFER_COMPLETE
echo {shlex.quote(output_dir)}
"""
    return _remote_bash(script)


def _build_system_info_command(*, container: bool = False) -> str:
    gr00t_cmd = (
        f"sudo docker exec {shlex.quote(GROOT_CONTAINER_NAME)} bash -lc "
        + shlex.quote(
            f"{GROOT_VENV}/bin/python - <<'PY'\n"
            "import os\n"
            "from importlib import metadata\n"
            "from pathlib import Path\n"
            "try:\n"
            "    version = metadata.version('gr00t')\n"
            "except Exception as exc:\n"
            "    version = f'not importable: {type(exc).__name__}: {exc}'\n"
            "ngc_cfg = Path.home() / '.ngc' / 'config'\n"
            "ngc_ok = bool(os.environ.get('NGC_API_KEY')) or (ngc_cfg.exists() and 'apikey' in ngc_cfg.read_text(errors='ignore'))\n"
            "print(f'gr00t_version: {version}')\n"
            "print(f'ngc_credentials_configured: {ngc_ok}')\n"
            "PY"
        )
        if container
        else (
            f"{GROOT_VENV}/bin/python - <<'PY'\n"
            "import os\n"
            "from importlib import metadata\n"
            "from pathlib import Path\n"
            "try:\n"
            "    version = metadata.version('gr00t')\n"
            "except Exception as exc:\n"
            "    version = f'not importable: {type(exc).__name__}: {exc}'\n"
            "ngc_cfg = Path.home() / '.ngc' / 'config'\n"
            "ngc_ok = bool(os.environ.get('NGC_API_KEY')) or (ngc_cfg.exists() and 'apikey' in ngc_cfg.read_text(errors='ignore'))\n"
            "print(f'gr00t_version: {version}')\n"
            "print(f'ngc_credentials_configured: {ngc_ok}')\n"
            "PY"
        )
    )
    isaac_cmd = (
        f"sudo docker exec {shlex.quote(GROOT_CONTAINER_NAME)} bash -lc "
        + shlex.quote(
            f"{ISAAC_LAB_VENV}/bin/python - <<'PY'\n"
            "from importlib import metadata\n"
            "try:\n"
            "    version = metadata.version('isaaclab')\n"
            "except Exception as exc:\n"
            "    version = f'not importable: {type(exc).__name__}: {exc}'\n"
            "print(f'isaaclab_version: {version}')\n"
            "PY"
        )
        if container
        else (
            f"{ISAAC_LAB_VENV}/bin/python - <<'PY'\n"
            "from importlib import metadata\n"
            "try:\n"
            "    version = metadata.version('isaaclab')\n"
            "except Exception as exc:\n"
            "    version = f'not importable: {type(exc).__name__}: {exc}'\n"
            "print(f'isaaclab_version: {version}')\n"
            "PY"
        )
    )
    container_cmd = (
        (
            f"echo '' && echo '=== container ===' && "
            f"sudo docker inspect -f 'state={{{{.State.Status}}}} image={{{{.Config.Image}}}}' {shlex.quote(GROOT_CONTAINER_NAME)} && "
        )
        if container
        else ""
    )
    return (
        "echo '=== nvidia-smi ===' && nvidia-smi && "
        "echo '' && echo '=== lscpu ===' && lscpu && "
        "echo '' && echo '=== free -h ===' && free -h && "
        "echo '' && echo '=== lsblk ===' && lsblk && "
        + container_cmd
        + f"echo '' && echo '=== gr00t ===' && {gr00t_cmd} && "
        f"echo '' && echo '=== isaac lab ===' && {isaac_cmd}"
    )


def _deploy_step_count(skip_infra: bool, skip_app: bool, destroy: bool) -> int:
    if destroy:
        return 2
    count = 1 if skip_infra else 2
    if not skip_app:
        count += 4
    count += 1
    return count


def _read_existing_outputs(
    proj_alias: str,
    wb_name: str,
    tf_dir: str,
    use_remote_state: bool,
    merged_vars: dict[str, str],
) -> dict[str, Any]:
    if tf_dir:
        try:
            return provisioner.outputs(tf_dir=tf_dir)
        except ProvisionerError:
            pass
    elif use_remote_state:
        work_dir = provisioner.working_dir_path(proj_alias, wb_name)
        if work_dir.exists():
            try:
                provisioner.init(
                    tf_dir=str(work_dir),
                    backend_config={
                        "access_key": merged_vars.get("nebius_api_key", ""),
                        "secret_key": merged_vars.get("nebius_secret_key", ""),
                    },
                )
                return provisioner.outputs(tf_dir=str(work_dir))
            except ProvisionerError:
                pass

    from npa.clients.config import (
        _deep_get,
        _load_yaml,
        _resolve_project_section,
        _resolve_workbench_in_project,
    )

    try:
        yml = _load_yaml()
        proj = _resolve_project_section(yml, proj_alias)
        wb = _resolve_workbench_in_project(proj, wb_name, yml)
    except Exception:
        wb = {}
    return {
        "vm_ip": _deep_get(wb, "ssh", "host", default=""),
        "ssh_user": _deep_get(wb, "ssh", "user", default="ubuntu"),
        "ssh_key_path": _deep_get(wb, "ssh", "key_path", default="~/.ssh/id_ed25519"),
        "storage_bucket": _deep_get(wb, "storage", "checkpoint_bucket", default=""),
        "storage_endpoint": _deep_get(wb, "storage", "endpoint_url", default=""),
    }


def _update_existing_deployment(
    *,
    project: str,
    name: str,
    port: int,
    dry_run: bool,
    output: OutputFormat,
) -> None:
    """Update an existing GR00T alias in place without Terraform."""
    if dry_run:
        _output(
            {
                "status": "would_update_existing",
                "project": project,
                "name": name,
                "terraform": "skipped",
                "action": "reload_env",
            },
            output,
        )
        return

    try:
        cfg = resolve_config(project=project, name=name)
    except ConfigError as exc:
        _fail(str(exc))
        return

    credentials = resolve_credentials()
    credential_env = _shared_groot_env_or_fail(credentials)
    service_port = port or int(getattr(cfg, "service_port", 0) or 0) or DEFAULT_SERVER_PORT
    result = _apply_env_update(
        cfg,
        credential_env,
        service_port=service_port,
        restart=True,
        output=output,
    )
    _output(
        {
            "status": "updated_existing",
            "project": project,
            "name": name,
            "terraform": "skipped",
            **result,
        },
        output,
    )


@app.command("list")
def list_cmd(
    output: OutputFormat = typer.Option(
        OutputFormat.text, "--output", help="Output format."
    ),
) -> None:
    """List configured GR00T workbenches."""
    projects = list_projects()
    def_proj = default_project_name()
    def_wb = default_workbench_name()

    if output == OutputFormat.json:
        filtered = {}
        for pname, pcfg in projects.items():
            wbs = {
                k: v
                for k, v in pcfg.get("workbenches", {}).items()
                if _is_groot_workbench(k, v)
            }
            if wbs:
                filtered[pname] = {**pcfg, "workbenches": wbs}
        typer.echo(
            json.dumps(
                {
                    "projects": filtered,
                    "default_project": def_proj,
                    "default_workbench": def_wb,
                },
                indent=2,
            )
        )
        return

    if not projects:
        typer.echo(
            "No projects configured. Run 'npa workbench groot deploy' to create one."
        )
        return

    any_shown = False
    for proj_name, proj_cfg in projects.items():
        workbenches = {
            k: v
            for k, v in proj_cfg.get("workbenches", {}).items()
            if _is_groot_workbench(k, v)
        }
        if not workbenches:
            continue
        any_shown = True
        proj_marker = " *" if proj_name == def_proj else ""
        region = proj_cfg.get("region", "?")
        typer.echo(f"  {proj_name}{proj_marker}  ({region})")
        for wb_name, wb_cfg in workbenches.items():
            wb_marker = " *" if wb_name == def_wb else ""
            gpu = wb_cfg.get("gpu_platform", "?")
            endpoint = wb_cfg.get("endpoint", "?")
            model = wb_cfg.get("model", DEFAULT_MODEL)
            app_status = wb_cfg.get("app_status", "unknown")
            typer.echo(
                f"    {wb_name}{wb_marker}  gpu={gpu}  endpoint={endpoint}  "
                f"model={model}  app_status={app_status}"
            )

    if not any_shown:
        typer.echo(
            "No GR00T workbenches configured. Run 'npa workbench groot deploy' to create one."
        )


@app.command("cleanup-partial")
def cleanup_partial_cmd(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation."),
) -> None:
    """Clean up orphaned Terraform resources from an interrupted GR00T deploy."""
    proj_alias = _project_alias or default_project_name()
    wb_name = _workbench_name or default_workbench_name()
    if not proj_alias or not wb_name:
        _fail("cleanup-partial requires --project and --name.")

    state = classify_alias_state(proj_alias, wb_name)
    if state == "fresh":
        typer.echo(f"No terraform state found for {proj_alias}/{wb_name}. Nothing to clean up.")
        return
    if state == "byovm":
        typer.echo(f"Alias {proj_alias}/{wb_name} is BYOVM. No terraform resources to clean.")
        return
    if state == "fully_deployed":
        typer.echo(f"Alias {proj_alias}/{wb_name} appears fully deployed. Use `teardown` instead.")
        raise typer.Exit(code=1)

    try:
        resources = list_terraform_managed_resources(proj_alias, wb_name)
    except CleanupPartialError as exc:
        _fail(f"Cleanup discovery failed: {exc}")
        return
    typer.echo(f"Found orphaned resources for {proj_alias}/{wb_name}:")
    for resource in resources:
        typer.echo(f"  - {resource}")
    if not yes:
        _confirm_or_exit(f"Destroy these {len(resources)} resources?")
    try:
        terraform_destroy_partial(proj_alias, wb_name)
        remove_partial_config_entry(proj_alias, wb_name)
    except CleanupPartialError as exc:
        _fail(f"Cleanup failed: {exc}")
        return
    typer.echo(f"Cleanup complete for {proj_alias}/{wb_name}.")


@app.command("deploy")
def deploy_cmd(
    gpu_type: str = typer.Option(
        "gpu-l40s-a", "--gpu-type", help="Nebius GPU platform; defaults to L40S."
    ),
    gpu_preset: str = typer.Option(
        "1gpu-40vcpu-160gb", "--gpu-preset", help="Nebius GPU preset."
    ),
    data_disk_size: int = typer.Option(
        200, "--data-disk-size", help="Attached GR00T data disk size in GiB."
    ),
    disk_size: int | None = typer.Option(
        None,
        "--disk-size",
        help="Boot disk size in GiB. Defaults to 250 for container runtime; VM runtime keeps the Terraform default.",
    ),
    runtime: WorkbenchRuntime = typer.Option(
        WorkbenchRuntime.vm, "--runtime", help=RUNTIME_HELP
    ),
    host: str = typer.Option(
        "", "--host", help="BYOVM SSH host/IP. Used only with --runtime byovm."
    ),
    ssh_key: str = typer.Option(
        "",
        "--ssh-key",
        help="BYOVM SSH private key path. Used only with --runtime byovm.",
    ),
    ssh_user: str = typer.Option(
        "", "--ssh-user", help="BYOVM SSH user. Defaults to ubuntu."
    ),
    gpu_count: int = typer.Option(
        0, "--gpu-count", help="Limit visible GPUs on BYOVM (0 = all detected)."
    ),
    region: str = typer.Option("", "--region", help="Nebius region."),
    project_id: str = typer.Option("", "--project-id", help="Nebius project ID."),
    tenant_id: str = typer.Option("", "--tenant-id", help="Nebius tenant ID."),
    tf_dir: str = typer.Option(
        "", "--tf-dir", help="Path to Terraform directory (default: bundled)."
    ),
    tf_var: list[str] = typer.Option(
        [], "--tf-var", "-v", help="Extra TF variable (key=value), repeatable."
    ),
    skip_infra: bool = typer.Option(
        False, "--skip-infra", help="Skip Terraform, only deploy the app."
    ),
    skip_app: bool = typer.Option(
        False, "--skip-app", help="Skip app installation, only provision infra."
    ),
    destroy: bool = typer.Option(
        False, "--destroy", help="Destroy infrastructure and clean up config."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would happen without doing it."
    ),
    replace: bool = typer.Option(
        False,
        "--replace",
        help=(
            "Provision replacement infrastructure for an existing alias. "
            "Without this flag, deploy against an existing alias updates env "
            "and config in place without Terraform."
        ),
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation prompts (use with --replace for automation).",
    ),
    no_shared_creds: bool = typer.Option(
        False,
        "--no-shared-creds",
        help="Do not inject ~/.npa/credentials.yaml shared credentials into the service env.",
    ),
    skip_model_check: bool = typer.Option(
        False,
        "--skip-model-check",
        help="Skip Hugging Face gated-model access validation.",
    ),
    health_check_mode: HealthCheckMode = typer.Option(
        HealthCheckMode.auto,
        "--health-check-mode",
        help="Health check mode: public, ssh, or auto. BYOVM auto tries public briefly, then SSH.",
    ),
    verify_env: bool = typer.Option(
        bool(os.environ.get("CI")),
        "--verify-env/--no-verify-env",
        help="Audit deployed shared credentials after app deploy.",
    ),
    model: str = typer.Option(
        DEFAULT_MODEL,
        "--model",
        help="Hugging Face GR00T model ID to validate and record.",
    ),
    server_port: int = typer.Option(
        DEFAULT_SERVER_PORT, "--server-port", help="GR00T HTTP server port on the VM."
    ),
    preemptible: bool = typer.Option(
        True, "--preemptible/--no-preemptible", help="Preemptible (spot) instance."
    ),
    default: bool = typer.Option(
        False, "--default", help="Set this workbench as the default."
    ),
    output: OutputFormat = typer.Option(
        OutputFormat.text, "--output", help="Output format."
    ),
) -> None:
    """Deploy or destroy a GR00T runtime VM with Isaac Lab available for sim evaluation."""
    if data_disk_size <= 0:
        _fail(f"--data-disk-size must be positive, got {data_disk_size}")
    if gpu_count < 0:
        _fail(f"--gpu-count must be 0 (all detected) or positive, got {gpu_count}")

    byovm = is_byovm_runtime(runtime)
    if _is_serverless_runtime(runtime):
        _fail("GR00T deploy does not use --runtime serverless; use `npa workbench groot infer --runtime serverless`.")
    container_runtime = runtime == WorkbenchRuntime.container
    if byovm:
        skip_infra = True
    proj_alias = _project_alias or None
    wb_name = _workbench_name or "groot"
    use_remote_state = not tf_dir and not byovm

    extra_vars: dict[str, str] = {}
    for item in tf_var:
        if "=" not in item:
            _fail(f"Invalid --tf-var format: {item} (expected key=value)")
        k, v = item.split("=", 1)
        extra_vars[k] = v

    saved_env = resolve_environment(
        proj_alias,
        project_id=project_id or None,
        tenant_id=tenant_id or None,
        region=region or None,
    )

    env_project = project_id or (saved_env.project_id if saved_env else "")
    env_tenant = tenant_id or (saved_env.tenant_id if saved_env else "")
    env_region = region or (saved_env.region if saved_env else "")
    if not proj_alias:
        proj_alias = env_region or ("byovm" if byovm else "default")

    existing_managed_alias = alias_has_terraform_state(proj_alias, wb_name)
    existing_byovm_alias = workbench_is_byovm(proj_alias, wb_name)
    if not destroy and (existing_managed_alias or existing_byovm_alias):
        if existing_byovm_alias:
            if replace:
                _fail(
                    f"{proj_alias}/{wb_name} is a BYOVM alias; --replace is only valid for Terraform-managed aliases."
                )
                return
            return _update_existing_deployment(
                project=proj_alias,
                name=wb_name,
                port=server_port,
                dry_run=dry_run,
                output=output,
            )
        if not replace:
            return _update_existing_deployment(
                project=proj_alias,
                name=wb_name,
                port=server_port,
                dry_run=dry_run,
                output=output,
            )
        if not yes:
            _confirm_or_exit(
                f"--replace will provision replacement infrastructure for '{proj_alias}/{wb_name}'. Continue?"
            )

    credentials = resolve_credentials()
    if not destroy and not skip_app:
        _model_check_or_fail(
            credentials=credentials,
            model=model,
            skip_model_check=skip_model_check,
            dry_run=dry_run,
            no_shared_creds=no_shared_creds,
        )

    nebius_creds: dict[str, str] = {}
    if use_remote_state and not skip_infra:
        if not env_project or not env_tenant or not env_region:
            _fail(
                "First deploy requires --project-id, --tenant-id, and --region.\n"
                "  Example: npa workbench groot -p eu-north1 -n groot-l40s deploy \\\n"
                "    --project-id project-... --tenant-id tenant-... --region eu-north1"
            )
            return

        if dry_run:
            console.print("  [dry-run] Would bootstrap Nebius environment:")
            console.print(f"    project: {env_project}")
            console.print(f"    tenant:  {env_tenant}")
            console.print(f"    region:  {env_region}")
        else:
            from npa.clients.nebius import NebiusError, bootstrap_environment

            console.print(f"Bootstrapping Nebius environment ({proj_alias})...")
            try:
                nebius_creds = bootstrap_environment(
                    env_project,
                    env_tenant,
                    env_region,
                    on_status=lambda msg: console.print(f"  {msg}"),
                )
            except NebiusError as exc:
                _fail(f"Nebius bootstrap failed: {exc}")
                return
            console.print("  Environment ready")
            write_config(
                {
                    "projects": {
                        proj_alias: {
                            "project_id": env_project,
                            "tenant_id": env_tenant,
                            "region": env_region,
                        },
                    },
                }
            )

    merged_vars: dict[str, str] = {**extra_vars}
    for key in (
        "iam_token",
        "service_account_id",
        "nebius_api_key",
        "nebius_secret_key",
        "s3_bucket",
        "s3_endpoint",
        "nebius_project_id",
        "nebius_region",
    ):
        if key in nebius_creds and key not in merged_vars:
            merged_vars[key] = nebius_creds[key]
    if use_remote_state and (destroy or skip_infra):
        _apply_saved_terraform_state(
            merged_vars,
            project=proj_alias,
            explicit_vars=extra_vars,
        )
    if byovm:
        apply_project_storage_vars(
            merged_vars,
            project=proj_alias,
            explicit_vars=extra_vars,
            warn=console.print,
        )

    if use_remote_state and nebius_creds and not dry_run:
        write_config(
            {
                "projects": {
                    proj_alias: {
                        "terraform_state": _terraform_state_config(merged_vars),
                    },
                },
            }
        )

    instance_name = f"groot-{proj_alias}-{wb_name}"
    tf_workbench_type = GROOT_CONTAINER_WORKBENCH_TYPE if container_runtime else "groot"

    if destroy:
        if byovm:
            if dry_run:
                console.print(
                    "  [dry-run] Would unregister BYOVM workbench only; VM would not be modified."
                )
                return
            console.print(
                f"  [1/1] Unregistering BYOVM workbench {proj_alias}/{wb_name}..."
            )
            remove_workbench_config(proj_alias, wb_name)
            console.print("  BYOVM target was not stopped, destroyed, or modified.")
            return

        console.print(f"  [1/2] Destroying {proj_alias}/{wb_name}...")
        if dry_run:
            console.print("    [dry-run] Would run: terraform destroy")
            return

        if use_remote_state:
            s3_bucket = merged_vars.get("s3_bucket", "")
            s3_endpoint = merged_vars.get(
                "s3_endpoint", f"https://storage.{env_region}.nebius.cloud"
            )
            resolved_tf_dir = str(
                provisioner.prepare_working_dir(
                    proj_alias,
                    wb_name,
                    bucket=s3_bucket,
                    region=env_region,
                    endpoint=s3_endpoint,
                )
            )
            try:
                provisioner.init(
                    tf_dir=resolved_tf_dir,
                    backend_config={
                        "access_key": merged_vars.get("nebius_api_key", ""),
                        "secret_key": merged_vars.get("nebius_secret_key", ""),
                    },
                )
            except ProvisionerError as exc:
                _fail(f"Terraform init failed: {exc}")
                return
        else:
            resolved_tf_dir = tf_dir

        try:
            wb_cfg = resolve_ssh_config(project=proj_alias, name=wb_name)
            if wb_cfg.tf_instance_name:
                instance_name = wb_cfg.tf_instance_name
        except ConfigError:
            pass

        try:
            destroy_vars = {
                "gpu_platform": gpu_type,
                "gpu_preset": gpu_preset,
                "data_disk_size_gb": str(data_disk_size),
                "instance_name": instance_name,
                "workbench_type": tf_workbench_type,
                "enable_preemptible": "true" if preemptible else "false",
                **merged_vars,
            }
            try:
                provisioner.apply_boot_disk_tf_vars(destroy_vars, runtime, disk_size)
            except ValueError as exc:
                _fail(str(exc))
                return
            provisioner.destroy(
                tf_dir=resolved_tf_dir or None,
                tf_vars=destroy_vars,
            )
        except ProvisionerError as exc:
            _fail(f"Terraform destroy failed: {exc}")
            return

        console.print("  [2/2] Cleaning up config...")
        remove_workbench_config(proj_alias, wb_name)
        if use_remote_state:
            provisioner.cleanup_working_dir(proj_alias, wb_name)
        console.print(f"  {proj_alias}/{wb_name} destroyed.")
        return

    total_steps = _deploy_step_count(skip_infra, skip_app, destroy)
    step = 0
    tf_outputs: dict[str, Any] = {}
    byovm_gpu_info = None
    byovm_effective_gpu_count = 0
    byovm_visible_devices = ""

    if not skip_infra:
        if use_remote_state:
            s3_bucket = merged_vars.get("s3_bucket", "")
            s3_endpoint = merged_vars.get(
                "s3_endpoint", f"https://storage.{env_region}.nebius.cloud"
            )
            resolved_tf_dir = str(
                provisioner.prepare_working_dir(
                    proj_alias,
                    wb_name,
                    bucket=s3_bucket,
                    region=env_region,
                    endpoint=s3_endpoint,
                )
            )
        else:
            resolved_tf_dir = tf_dir

        step += 1
        console.print(
            f"  [{step}/{total_steps}] Initializing Terraform ({proj_alias}/{wb_name})..."
        )
        if dry_run:
            console.print("    [dry-run] Would run: terraform init")
        else:
            try:
                backend_cfg = (
                    {
                        "access_key": merged_vars.get("nebius_api_key", ""),
                        "secret_key": merged_vars.get("nebius_secret_key", ""),
                    }
                    if use_remote_state
                    else None
                )
                provisioner.init(
                    tf_dir=resolved_tf_dir or None, backend_config=backend_cfg
                )
            except ProvisionerError as exc:
                _fail(f"Terraform init failed: {exc}")
                return

        step += 1
        all_vars = {
            "gpu_platform": gpu_type,
            "gpu_preset": gpu_preset,
            "data_disk_size_gb": str(data_disk_size),
            "instance_name": instance_name,
            "workbench_type": tf_workbench_type,
            "enable_preemptible": "true" if preemptible else "false",
            **merged_vars,
        }
        try:
            provisioner.apply_boot_disk_tf_vars(all_vars, runtime, disk_size)
        except ValueError as exc:
            _fail(str(exc))
            return
        console.print(
            f"  [{step}/{total_steps}] Applying Terraform (gpu={gpu_type}, region={env_region})..."
        )
        if dry_run:
            tf_outputs = {
                "vm_ip": "<pending>",
                "ssh_user": "ubuntu",
                "ssh_key_path": "~/.ssh/id_ed25519",
                "storage_bucket": "<pending>",
                "storage_endpoint": f"https://storage.{env_region}.nebius.cloud",
            }
        else:
            try:
                tf_outputs = provisioner.apply(
                    tf_dir=resolved_tf_dir or None, tf_vars=all_vars
                )
            except ProvisionerError as exc:
                _fail(f"Terraform apply failed: {exc}")
                return
        console.print(f"    VM IP: {tf_outputs.get('vm_ip', 'unknown')}")
    else:
        step += 1
        console.print(
            f"  [{step}/{total_steps}] {'Using BYOVM target' if byovm else 'Skipping infra, reading existing config'}..."
        )
        resolved_tf_dir = tf_dir
        if byovm:
            try:
                target = resolve_byovm_target(
                    host=host, ssh_key=ssh_key, ssh_user=ssh_user
                )
            except ValueError as exc:
                _fail(str(exc))
                return
            ssh = SSHClient(
                SSHConfig(
                    host=target.host,
                    user=target.user,
                    key_path=target.key_path,
                    tokens=resolve_credentials().tokens,
                )
            )
            if not dry_run:
                try:
                    ssh.run_or_raise("echo connected")
                    byovm_gpu_info = detect_gpu_info(ssh)
                    byovm_effective_gpu_count, byovm_visible_devices = (
                        select_visible_devices(
                            byovm_gpu_info.count,
                            gpu_count or None,
                        )
                    )
                    console.print(
                        f"    Detected {byovm_gpu_info.count} GPU(s): "
                        f"{', '.join(byovm_gpu_info.names)}"
                    )
                    console.print(f"    CUDA_VISIBLE_DEVICES={byovm_visible_devices}")
                except (SSHError, ValueError) as exc:
                    _fail(f"BYOVM target validation failed: {exc}")
                    return
            else:
                byovm_effective_gpu_count = gpu_count or 0
                byovm_visible_devices = ",".join(
                    str(i) for i in range(byovm_effective_gpu_count)
                )
            tf_outputs = workbench_storage_outputs(
                target=target,
                bucket=merged_vars.get("s3_bucket", "")
                or os.environ.get("NPA_CHECKPOINT_BUCKET", ""),
                endpoint=merged_vars.get("s3_endpoint", "")
                or os.environ.get("AWS_ENDPOINT_URL", ""),
            )
        else:
            tf_outputs = _read_existing_outputs(
                proj_alias,
                wb_name,
                tf_dir,
                use_remote_state,
                merged_vars,
            )
        if not tf_outputs.get("vm_ip"):
            _fail(
                "No VM IP found. Run without --skip-infra first, or set config manually."
            )
            return

    vm_ip = tf_outputs.get("vm_ip", "")
    ssh_user = tf_outputs.get("ssh_user", "ubuntu")
    ssh_key = tf_outputs.get("ssh_key_path", "~/.ssh/id_ed25519")
    bucket = tf_outputs.get("storage_bucket", "")
    storage_ep = tf_outputs.get("storage_endpoint", "")
    endpoint = f"http://{vm_ip}:{server_port}"
    bucket_display = (
        bucket
        if str(bucket).startswith("s3://")
        else (f"s3://{bucket}/checkpoints/" if bucket else "")
    )
    byovm_fields = gpu_config_fields(
        byovm_gpu_info,
        effective_count=byovm_effective_gpu_count or None,
        visible_devices=byovm_visible_devices,
    )
    config_data: dict[str, Any] = {
        "projects": {
            proj_alias: {
                "project_id": env_project,
                "tenant_id": env_tenant,
                "region": env_region,
                "terraform_state": _terraform_state_config(merged_vars),
                "workbenches": {
                    wb_name: {
                        "endpoint": endpoint,
                        "gpu_platform": byovm_fields.get("gpu_platform", gpu_type),
                        "gpu_preset": byovm_fields.get("gpu_preset", gpu_preset),
                        "data_disk_size_gb": data_disk_size,
                        "data_mount": GROOT_DATA_MOUNT,
                        "tf_instance_name": instance_name,
                        "workbench_type": "groot",
                        "runtime": runtime.value,
                        "app_status": APP_STATUS_PROVISIONED,
                        "endpoint_strategy": "public",
                        "service_port": server_port,
                        "model": model,
                        "embodiment_tag": DEFAULT_EMBODIMENT_TAG,
                        "ssh": {"host": vm_ip, "user": ssh_user, "key_path": ssh_key},
                        "storage": {
                            "checkpoint_bucket": bucket_display,
                            "endpoint_url": storage_ep,
                        },
                        **byovm_fields,
                    },
                },
            },
        },
    }

    if default or not list_projects():
        config_data["default_project"] = proj_alias
        config_data["default_workbench"] = wb_name

    if not dry_run:
        write_config(config_data)
        console.print("    Registered workbench in ~/.npa/config.yaml")

    def mark_app_status(app_status: str) -> None:
        if not dry_run:
            update_workbench_app_status(proj_alias, wb_name, app_status)

    def fail_app(msg: str) -> None:
        mark_app_status(APP_STATUS_INSTALL_FAILED)
        _fail(msg)

    if not skip_app:
        mark_app_status(APP_STATUS_INSTALLING)
        ssh_cfg = SSHConfig(
            host=vm_ip, user=ssh_user, key_path=ssh_key, tokens=credentials.tokens
        )
        service_env = {
            "AWS_ACCESS_KEY_ID": merged_vars.get("nebius_api_key", ""),
            "AWS_SECRET_ACCESS_KEY": merged_vars.get("nebius_secret_key", ""),
            "AWS_ENDPOINT_URL": storage_ep,
            "NEBIUS_S3_ENDPOINT": storage_ep,
            "NEBIUS_S3_BUCKET": bucket,
            "NEBIUS_REGION": env_region,
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "ACCEPT_EULA": "Y",
            "ISAACSIM_ACCEPT_EULA": "YES",
            "PYTHONUNBUFFERED": "1",
            **gpu_env_fields(
                byovm_gpu_info,
                effective_count=byovm_effective_gpu_count or None,
                visible_devices=byovm_visible_devices,
            ),
        }
        apply_shared_credential_env(
            service_env, credentials, include=not no_shared_creds
        )

        step += 1
        console.print(
            f"  [{step}/{total_steps}] Connecting via SSH to {ssh_user}@{vm_ip}..."
        )
        if not dry_run:
            ssh = SSHClient(ssh_cfg)
            try:
                code, _, _ = ssh.run("echo connected")
            except SSHError as exc:
                fail_app(str(exc))
                return
            if code != 0:
                fail_app(f"SSH connection test failed (exit {code})")
                return

        step += 1
        if container_runtime:
            console.print(f"  [{step}/{total_steps}] Starting GR00T container...")
            full_service_env = _groot_service_env(
                credentials=credentials,
                merged_vars=merged_vars,
                storage_ep=storage_ep,
                bucket=bucket,
                env_region=env_region,
                server_port=server_port,
                service_env=service_env,
                include_shared_creds=False,
            )
            if dry_run:
                console.print(
                    "    [dry-run] Would pull and run the GR00T container image"
                )
                console.print("    [dry-run] Service env:")
                console.print(render_redacted_env_file(full_service_env).rstrip())
            else:
                try:
                    write_remote_docker_env_file(
                        ssh,
                        GROOT_CONTAINER_ENV_FILE,
                        full_service_env,
                        owner=ssh_user,
                    )
                    image_ref = container_image_for_tool(
                        "groot",
                        registry=resolve_container_registry(proj_alias),
                        tag=GROOT_RUNTIME_VERSION,
                    )
                    from npa.deploy.configurator import deploy_workbench_container

                    ssh.run(
                        f"sudo systemctl stop {GROOT_SERVICE} >/dev/null 2>&1 || true"
                    )
                    deploy_workbench_container(
                        ssh,
                        image_ref=image_ref,
                        container_name=GROOT_CONTAINER_NAME,
                        env_file=GROOT_CONTAINER_ENV_FILE,
                        volumes=[
                            f"{GROOT_DATA_MOUNT}:{GROOT_DATA_MOUNT}",
                            f"{GROOT_CONTAINER_ENV_FILE}:{GROOT_CONTAINER_ENV_FILE}:ro",
                        ],
                        work_dirs=[
                            GROOT_MODEL_DIR,
                            f"{GROOT_DATA_MOUNT}/hf_cache",
                            GROOT_OUTPUT_DIR,
                            f"{GROOT_DATA_MOUNT}/checkpoints",
                            GROOT_DATA_CACHE,
                            GROOT_CHECKPOINT_CACHE,
                            GROOT_BASE_MODEL_CACHE,
                            GROOT_EVAL_DATA_CACHE,
                            GROOT_CONFIG_CACHE,
                        ],
                        command=(
                            "-lc "
                            + shlex.quote(
                                "cd /opt/groot && "
                                f"exec {GROOT_VENV}/bin/python -m uvicorn server:app "
                                f"--host 0.0.0.0 --port {server_port}"
                            )
                        ),
                        registry_token=merged_vars.get("iam_token", ""),
                    )
                    if verify_env and not no_shared_creds:
                        failed_keys = audit_remote_env(
                            ssh,
                            GROOT_CONTAINER_ENV_FILE,
                            _groot_audit_env(full_service_env),
                        )
                        if failed_keys:
                            key = failed_keys[0]
                            fail_app(
                                f"Credential audit failed: {key} missing or mismatched in groot service env. "
                                "Deploy may have skipped shared credential injection."
                            )
                            return
                        _print_ngc_env_audit(
                            credentials=credentials,
                            service_env=full_service_env,
                            remote_path=GROOT_CONTAINER_ENV_FILE,
                        )
                except SSHError as exc:
                    fail_app(f"GR00T container deployment failed: {exc}")
                    return
        else:
            console.print(
                f"  [{step}/{total_steps}] Installing GR00T {GROOT_RELEASE} runtime..."
            )
            if dry_run:
                console.print(
                    "    [dry-run] Would install Python 3.10, uv, NGC CLI, Isaac-GR00T, and Isaac Lab"
                )
            else:
                try:
                    ssh.run_or_raise(
                        _build_install_command(server_port, env_fields=service_env),
                        stream=True,
                    )
                    if verify_env and not no_shared_creds:
                        failed_keys = audit_remote_env(
                            ssh,
                            "/etc/npa-groot-server/env",
                            _groot_audit_env(service_env),
                        )
                        if failed_keys:
                            key = failed_keys[0]
                            fail_app(
                                f"Credential audit failed: {key} missing or mismatched in groot service env. "
                                "Deploy may have skipped shared credential injection."
                            )
                            return
                        _print_ngc_env_audit(
                            credentials=credentials,
                            service_env=service_env,
                            remote_path="/etc/npa-groot-server/env",
                        )
                except SSHError as exc:
                    fail_app(f"GR00T installation failed: {exc}")
                    return

        step += 1
        console.print(f"  [{step}/{total_steps}] Health check on {endpoint}...")
        if not dry_run:
            healthy, health_note = health_check_auto(
                endpoint,
                mode=health_check_mode,
                ssh=ssh if byovm else None,
                port=server_port,
                host=vm_ip,
            )
            if healthy:
                console.print("    Server is healthy")
                if health_note:
                    console.print(f"    {health_note}")
                endpoint_strategy = (
                    "ssh"
                    if byovm
                    and (health_check_mode == HealthCheckMode.ssh or bool(health_note))
                    else "public"
                )
                write_config(
                    {
                        "projects": {
                            proj_alias: {
                                "workbenches": {
                                    wb_name: {
                                        "endpoint_strategy": endpoint_strategy,
                                        "service_port": server_port,
                                    },
                                },
                            },
                        },
                    }
                )
            else:
                fail_app(f"Server not healthy at {endpoint}/health.")
                return

        step += 1
        console.print(f"  [{step}/{total_steps}] Writing deployment manifest...")
        if not dry_run:
            try:
                write_manifest(
                    ssh,
                    tool="groot",
                    version=GROOT_RELEASE,
                    deployed_by=f"npa deploy --runtime {runtime.value}",
                )
            except SSHError:
                pass
        mark_app_status(APP_STATUS_HEALTHY)
        if not dry_run:
            ensure_deploy_ingress(
                tool="groot",
                port=server_port,
                alias=wb_name,
                instance_id=resolve_deploy_instance_id(
                    tf_outputs=tf_outputs,
                    project_alias=proj_alias,
                    name=wb_name,
                ),
                warn=console.print,
            )

    step += 1
    console.print(
        f"  [{step}/{total_steps}] Updating config status ({proj_alias}/{wb_name})..."
    )
    if not dry_run:
        console.print("    Saved to ~/.npa/config.yaml")

    console.print("")
    console.print(f"[bold green]Deploy complete.[/bold green] ({proj_alias}/{wb_name})")
    console.print(f"  Endpoint: {endpoint}")
    console.print(f"  SSH:      ssh -i {ssh_key} {ssh_user}@{vm_ip}")
    console.print(f"  Model:    {model}")
    console.print("")
    console.print(f"  Try: npa workbench groot -p {proj_alias} -n {wb_name} status")

    if output == OutputFormat.json:
        typer.echo(
            json.dumps(
                {
                    "project": proj_alias,
                    "name": wb_name,
                    "endpoint": endpoint,
                    "vm_ip": vm_ip,
                    "ssh_user": ssh_user,
                    "gpu_platform": byovm_fields.get("gpu_platform", gpu_type),
                    "gpu_preset": byovm_fields.get("gpu_preset", gpu_preset),
                    "gpu_count": byovm_fields.get("gpu_count"),
                    "data_disk_size_gb": data_disk_size,
                    "model": model,
                    "tf_outputs": tf_outputs,
                },
                indent=2,
            )
        )


@app.command("download")
def download_cmd(
    model: str = typer.Option(
        DEFAULT_MODEL, "--model", help="GR00T model ID or NGC model ref."
    ),
    output_path: str = typer.Option(
        "",
        "--output-path",
        help="S3 URI for downloaded weights.",
    ),
    ngc_api_key: str = typer.Option(
        "",
        "--ngc-api-key",
        help="NGC API key. Falls back to NGC_API_KEY env var or ~/.npa/credentials.yaml.",
    ),
    output: OutputFormat = typer.Option(
        OutputFormat.text, "--output", help="Output format."
    ),
) -> None:
    """Download GR00T model weights to the workbench VM or shared S3 storage."""
    try:
        output_path = validate_write_path(output_path, tool="GR00T download")
    except PathContractError as exc:
        _fail(str(exc))
    cfg = _get_ssh_config()
    credentials = resolve_credentials()
    token = (
        ngc_api_key
        or getattr(credentials, "ngc_api_key", "")
        or (getattr(credentials, "tokens", {}) or {}).get("NGC_API_KEY", "")
    )
    ssh = _ssh_client(cfg, extra_tokens={"NGC_API_KEY": token})
    target = output_path or f"{GROOT_MODEL_DIR}/{_model_slug(model)}"

    try:
        _, out, err = ssh.run_or_raise(
            _runtime_command(
                cfg,
                _build_download_command(model, target, cfg.storage.endpoint_url),
                pass_env=(*GROOT_REMOTE_ENV_NAMES,),
            ),
            stream=output != OutputFormat.json,
        )
    except SSHError as exc:
        _fail(f"GR00T download failed: {exc}")
        return

    result: dict[str, Any] = {
        "status": "success",
        "model": model,
        "output_path": target,
        "source": "ngc" if _is_ngc_model_ref(model) else "huggingface",
    }
    if output == OutputFormat.json and out.strip():
        result["stdout_tail"] = out.strip()[-1000:]
    if err.strip():
        result["stderr_tail"] = err.strip()[-1000:]
    _output(result, output)


@app.command("reload-env")
def reload_env_cmd(
    port: int = typer.Option(
        0, "--port", help="GR00T HTTP server port. Defaults to the saved service port."
    ),
    restart: bool = typer.Option(
        True,
        "--restart/--no-restart",
        help="Restart GR00T after updating the env file.",
    ),
    preserve_loaded: bool = typer.Option(
        True,
        "--preserve-loaded/--no-preserve-loaded",
        help="After restart, re-serve the model that was loaded before the env update.",
    ),
    model: str = typer.Option(
        "", "--model", help="Model ID/path to serve after reloading env."
    ),
    robot_embodiment: str = typer.Option(
        "", "--robot-embodiment", help="Embodiment tag to serve after reloading env."
    ),
    timeout: float = typer.Option(
        600.0, "--timeout", help="Seconds to wait for optional model re-serve."
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Preview env changes and restart commands without applying them.",
    ),
    output: OutputFormat = typer.Option(
        OutputFormat.text, "--output", help="Output format."
    ),
) -> None:
    """Propagate local shared credentials into the running GR00T service env without redeploying."""
    cfg = _get_config()
    credentials = resolve_credentials()
    credential_env = _shared_groot_env_or_fail(credentials)

    service_port = (
        port or int(getattr(cfg, "service_port", 0) or 0) or DEFAULT_SERVER_PORT
    )
    if dry_run:
        env_path, env_mode, current_env = _read_current_env_for_dry_run(
            cfg, credential_env
        )
        proposed_env = merge_env_file_content(current_env, credential_env)
        diff = render_redacted_env_diff(current_env, proposed_env)
        if output == OutputFormat.json:
            _output(
                {
                    "status": "dry_run",
                    "updated_keys": sorted(credential_env),
                    "env_path": env_path,
                    "mode": env_mode,
                    "restart": restart,
                    "port": service_port,
                    "diff": diff,
                    "commands": [
                        f"systemctl restart {GROOT_SERVICE}" if restart else "no restart",
                        f"curl http://127.0.0.1:{service_port}/health",
                    ],
                },
                output,
            )
        else:
            typer.echo("=== Dry run: would change env file ===")
            typer.echo(diff or "(no changes)")
            typer.echo("=== Would execute: ===")
            if restart:
                typer.echo(f"  systemctl restart {GROOT_SERVICE}")
                typer.echo(f"  curl http://127.0.0.1:{service_port}/health")
            else:
                typer.echo("  no restart (--no-restart)")
            typer.echo("")
            typer.echo("No changes applied (--dry-run).")
        return

    pre_health: dict[str, Any] = {}
    pre_health_error = ""
    should_probe_loaded = preserve_loaded or bool(model) or bool(robot_embodiment)
    if should_probe_loaded:
        try:
            with service_endpoint(
                cfg, default_port=service_port, service_port=service_port
            ) as active:
                pre_health = HTTPClient(active.url, timeout=10.0, retries=1).health()
        except (EndpointError, ServerError) as exc:
            pre_health_error = str(exc)

    env_update = _apply_env_update(
        cfg,
        credential_env,
        service_port=service_port,
        restart=restart,
        output=output,
    )

    serve_model = model
    serve_tag = robot_embodiment
    if preserve_loaded and not serve_model and pre_health.get("loaded"):
        serve_model = str(
            pre_health.get("loaded_model") or pre_health.get("model") or ""
        )
        serve_tag = serve_tag or str(pre_health.get("embodiment_tag") or "")
    if robot_embodiment and not serve_model:
        serve_model = str(
            pre_health.get("loaded_model") or pre_health.get("model") or DEFAULT_MODEL
        )

    served: dict[str, Any] | None = None
    endpoint_url = ""
    if serve_model:
        tag = _normalize_embodiment_tag(serve_tag or DEFAULT_EMBODIMENT_TAG)
        try:
            with service_endpoint(
                cfg, default_port=service_port, service_port=service_port
            ) as active:
                served = HTTPClient(active.url, timeout=timeout, retries=1)._request(
                    "POST",
                    "/serve",
                    json={
                        "model_path": serve_model,
                        "embodiment_tag": tag,
                        "device": "cuda",
                    },
                    timeout=timeout,
                )
                endpoint_url = active.url
        except EndpointError as exc:
            _fail(f"GR00T reload-env endpoint setup failed after env update: {exc}")
            return
        except ServerError as exc:
            _fail(f"GR00T reload-env model re-serve failed after env update: {exc}")
            return

    result: dict[str, Any] = {
        "status": "reloaded",
        **env_update,
    }
    if pre_health_error:
        result["pre_health_error"] = pre_health_error
    if served is not None:
        result["served"] = {
            "model": serve_model,
            "embodiment_tag": _normalize_embodiment_tag(
                serve_tag or DEFAULT_EMBODIMENT_TAG
            ),
            "endpoint": endpoint_url,
            "response": served,
        }
    _output(result, output)


@app.command("finetune")
def finetune_cmd(
    input_path: str = typer.Option(
        ..., "--input-path", help="S3 URI for a GR00T LeRobot training dataset."
    ),
    output_path: str = typer.Option(
        ..., "--output-path", help="S3 URI for a fine-tuned checkpoint directory."
    ),
    base_model: str = typer.Option(
        DEFAULT_MODEL, "--base-model", help="Base GR00T checkpoint ID or S3 URI."
    ),
    robot_embodiment: str = typer.Option(
        DEFAULT_EMBODIMENT_TAG,
        "--robot-embodiment",
        help="GR00T embodiment tag, e.g. NEW_EMBODIMENT, REAL_G1, UNITREE_G1, LIBERO_PANDA.",
    ),
    num_gpus: int = typer.Option(
        1, "--num-gpus", help="Number of GPUs for PyTorch fine-tuning."
    ),
    config: str = typer.Option(
        "", "--config", help="Optional GR00T modality/training config path."
    ),
    max_steps: int | None = typer.Option(
        None, "--max-steps", help="Override GR00T training max_steps."
    ),
    global_batch_size: int | None = typer.Option(
        None, "--global-batch-size", help="Override effective training batch size."
    ),
    dataloader_num_workers: int | None = typer.Option(
        None, "--dataloader-num-workers", help="Override dataloader workers."
    ),
    save_steps: int | None = typer.Option(
        None, "--save-steps", help="Override checkpoint save interval."
    ),
    save_total_limit: int | None = typer.Option(
        None, "--save-total-limit", help="Override checkpoint retention."
    ),
    save_only_model: bool = typer.Option(
        False, "--save-only-model", help="Save only model weights in checkpoints."
    ),
    output: OutputFormat = typer.Option(
        OutputFormat.text, "--output", help="Output format."
    ),
) -> None:
    """Fine-tune a GR00T action head on demonstration data with PyTorch."""
    if num_gpus <= 0:
        _fail(f"--num-gpus must be positive, got {num_gpus}")
    try:
        input_path = validate_read_path(
            input_path,
            tool="GR00T finetune",
            option="--input-path",
            allow_hf=False,
        )
        output_path = validate_write_path(
            output_path,
            tool="GR00T finetune",
            option="--output-path",
            required=True,
        )
    except PathContractError as exc:
        _fail(str(exc))
    cfg = _get_ssh_config()
    ssh = _ssh_client(cfg)
    tag = _normalize_embodiment_tag(robot_embodiment)
    cmd = _runtime_command(
        cfg,
        _build_finetune_command(
            input_path=input_path,
            output_path=output_path,
            base_model=base_model,
            robot_embodiment=tag,
            num_gpus=num_gpus,
            config=config,
            endpoint_url=cfg.storage.endpoint_url,
            max_steps=max_steps,
            global_batch_size=global_batch_size,
            dataloader_num_workers=dataloader_num_workers,
            save_steps=save_steps,
            save_total_limit=save_total_limit,
            save_only_model=save_only_model,
        ),
        pass_env=GROOT_REMOTE_ENV_NAMES,
    )

    start = time.time()
    try:
        exit_code, stdout, stderr = ssh.run(cmd, stream=output != OutputFormat.json)
    except SSHError as exc:
        _fail(f"SSH error: {exc}")
        return

    result = {
        "status": "success" if exit_code == 0 else "failed",
        "exit_code": exit_code,
        "input_path": input_path,
        "output_path": output_path,
        "base_model": base_model,
        "robot_embodiment": tag,
        "num_gpus": num_gpus,
        "duration_seconds": round(time.time() - start, 1),
    }
    if exit_code != 0:
        result["stderr"] = stderr.strip()[-500:] if stderr else ""
    elif output == OutputFormat.json and stdout.strip():
        result["stdout_tail"] = stdout.strip()[-1000:]
    _output(result, output)
    if exit_code != 0:
        raise typer.Exit(1)


@app.command("eval")
def eval_cmd(
    input_path: str = typer.Option(
        ..., "--input-path", help="S3 URI for a fine-tuned checkpoint."
    ),
    output_path: str = typer.Option(
        ..., "--output-path", help="S3 URI for eval results."
    ),
    robot_embodiment: str = typer.Option(
        DEFAULT_EMBODIMENT_TAG,
        "--robot-embodiment",
        help="GR00T embodiment tag used by the policy.",
    ),
    dataset_path: str = typer.Option(
        "",
        "--dataset-path",
        help="Held-out GR00T LeRobot dataset for offline open-loop eval.",
    ),
    sim: bool = typer.Option(
        False, "--sim", help="Create a sim-eval request for an Isaac Lab workbench."
    ),
    isaac_lab_workbench: str = typer.Option(
        "",
        "--isaac-lab-workbench",
        help="Named Isaac Lab workbench used for sim evaluation.",
    ),
    num_episodes: int = typer.Option(
        100, "--num-episodes", help="Number of sim episodes."
    ),
    output: OutputFormat = typer.Option(
        OutputFormat.text, "--output", help="Output format."
    ),
) -> None:
    """Evaluate a fine-tuned GR00T policy offline or through the S3 Isaac Lab data bus."""
    tag = _normalize_embodiment_tag(robot_embodiment)
    try:
        input_path = validate_read_path(
            input_path,
            tool="GR00T eval",
            option="--input-path",
            allow_hf=False,
        )
        output_path = validate_write_path(
            output_path,
            tool="GR00T eval",
            option="--output-path",
            required=True,
        )
        if dataset_path:
            dataset_path = validate_read_path(
                dataset_path,
                tool="GR00T eval",
                option="--dataset-path",
                allow_hf=False,
            )
    except PathContractError as exc:
        _fail(str(exc))

    if sim:
        if not isaac_lab_workbench:
            _fail("--isaac-lab-workbench is required with --sim")
        if num_episodes <= 0:
            _fail(f"--num-episodes must be positive, got {num_episodes}")
        try:
            resolve_ssh_config(project=_project_alias or None, name=isaac_lab_workbench)
        except ConfigError as exc:
            _fail(f"Isaac Lab workbench not found: {exc}")
            return

        request = {
            "type": "npa_groot_sim_eval_request_v1",
            "checkpoint_path": input_path,
            "results_path": output_path,
            "robot_embodiment": tag,
            "num_episodes": num_episodes,
            "isaac_lab_workbench": isaac_lab_workbench,
            "created_unix": round(time.time(), 3),
            "note": (
                "Isaac Lab consumes this request via the S3 data bus; GR00T runtime "
                "does not install or bundle Isaac Lab."
            ),
        }
        saved_to = _write_eval_request(request, output_path)
        _output(
            {"status": "sim_eval_request_created", "request_path": saved_to, **request},
            output,
        )
        return

    if not dataset_path:
        _fail("Offline eval requires --dataset-path with held-out GR00T LeRobot data.")
    cfg = _get_ssh_config()
    ssh = _ssh_client(cfg)
    cmd = _runtime_command(
        cfg,
        _build_offline_eval_command(
            checkpoint_path=input_path,
            dataset_path=dataset_path,
            output_path=output_path,
            robot_embodiment=tag,
            endpoint_url=cfg.storage.endpoint_url,
        ),
        pass_env=GROOT_REMOTE_ENV_NAMES,
    )
    start = time.time()
    try:
        exit_code, stdout, stderr = ssh.run(cmd, stream=output != OutputFormat.json)
    except SSHError as exc:
        _fail(f"SSH error: {exc}")
        return

    result = {
        "status": "success" if exit_code == 0 else "failed",
        "exit_code": exit_code,
        "input_path": input_path,
        "dataset_path": dataset_path,
        "output_path": output_path,
        "robot_embodiment": tag,
        "duration_seconds": round(time.time() - start, 1),
    }
    if exit_code != 0:
        result["stderr"] = stderr.strip()[-500:] if stderr else ""
    elif output == OutputFormat.json and stdout.strip():
        result["stdout_tail"] = stdout.strip()[-1000:]
    _output(result, output)
    if exit_code != 0:
        raise typer.Exit(1)


def _write_eval_request(request: dict[str, Any], output_path: str) -> str:
    target = output_path.rstrip("/")
    if _is_s3_uri(output_path):
        cfg = _get_config()
        tmp = tempfile.TemporaryDirectory(prefix="npa-groot-sim-eval-")
        try:
            local = Path(tmp.name) / "groot_sim_eval_request.json"
            local.write_text(json.dumps(request, indent=2))
            dest = (
                f"{target}/groot_sim_eval_request.json"
                if not target.endswith(".json")
                else target
            )
            return _storage_client_for_config(cfg).upload_file(str(local), dest)
        finally:
            tmp.cleanup()

    local = Path(output_path)
    if output_path.endswith(("/", "\\")) or local.suffix != ".json":
        local = local / "groot_sim_eval_request.json"
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text(json.dumps(request, indent=2))
    return str(local)


@app.command("serve")
def serve_cmd(
    input_path: str = typer.Option(
        "", "--input-path", help="S3 URI for a fine-tuned checkpoint directory."
    ),
    model: str = typer.Option(
        "", "--model", help="Downloaded/base model ID to serve instead of --input-path."
    ),
    robot_embodiment: str = typer.Option(
        DEFAULT_EMBODIMENT_TAG, "--robot-embodiment", help="GR00T embodiment tag."
    ),
    port: int = typer.Option(
        DEFAULT_SERVER_PORT, "--port", help="GR00T HTTP server port."
    ),
    timeout: float = typer.Option(
        600.0, "--timeout", help="Seconds to wait for model load before failing."
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print the pending live serve placeholder without touching the VM.",
    ),
    output: OutputFormat = typer.Option(
        OutputFormat.text, "--output", help="Output format."
    ),
) -> None:
    """Load a GR00T checkpoint and serve synchronous policy inference."""
    if bool(input_path) == bool(model):
        _fail("Provide exactly one of --input-path or --model.")
    if input_path:
        try:
            input_path = validate_read_path(
                input_path,
                tool="GR00T serve",
                option="--input-path",
                allow_hf=False,
            )
        except PathContractError as exc:
            _fail(str(exc))
    if dry_run:
        _output(
            {
                "status": "pending",
                "message": "Would ask the running GR00T server to load the selected model.",
                "input_path": input_path or None,
                "model": model or None,
                "port": port,
            },
            output,
        )
        return
    cfg = _get_config()
    ssh = _ssh_client(cfg)
    source = model or input_path
    model_path = source
    setup_cmd = ""
    if input_path and _is_s3_uri(input_path):
        model_path = _cache_dir("checkpoint", input_path)
        setup_cmd = (
            _remote_download_dir_cmd(input_path, model_path, cfg.storage.endpoint_url)
            + " && "
        )
    elif model:
        model_path = model

    tag = _normalize_embodiment_tag(robot_embodiment)
    try:
        if setup_cmd:
            ssh.run_or_raise(
                _runtime_command(
                    cfg,
                    _remote_bash(setup_cmd[:-4]),
                    pass_env=GROOT_REMOTE_ENV_NAMES,
                ),
                stream=output != OutputFormat.json,
            )
    except SSHError as exc:
        _fail(f"GR00T serve failed: {exc}")
        return

    try:
        with service_endpoint(cfg, default_port=port, service_port=port) as active:
            served = HTTPClient(active.url, timeout=timeout, retries=1)._request(
                "POST",
                "/serve",
                json={
                    "model_path": model_path,
                    "embodiment_tag": tag,
                    "device": "cuda",
                },
                timeout=timeout,
            )
            endpoint_url = active.url
    except EndpointError as exc:
        _fail(f"GR00T serve endpoint setup failed: {exc}")
        return
    except ServerError as exc:
        message = str(exc)
        lowered = message.lower()
        if any(
            term in lowered
            for term in ("gated", "authentication", "401", "403", "access to model")
        ):
            repo = model or model_path
            message += (
                f"\nRequest access at https://huggingface.co/{repo} and ensure "
                "HF_TOKEN has the required permissions."
            )
        _fail(f"Model load failed: {message}")
        return

    result = {
        "status": "serving",
        "input_path": input_path or None,
        "model": model or None,
        "model_path": model_path,
        "robot_embodiment": tag,
        "port": port,
        "endpoint": endpoint_url,
        "response": served,
    }
    _output(result, output)


@app.command("infer")
def infer_cmd(
    input_path: str = typer.Option(
        ...,
        "--input-path",
        help="S3 URI or Hugging Face model ID for a GR00T checkpoint.",
    ),
    dataset_path: str = typer.Option(
        ..., "--dataset-path", help="S3 URI for a GR00T LeRobot dataset."
    ),
    output_path: str = typer.Option(
        ..., "--output-path", help="S3 URI for predicted actions."
    ),
    source_project: str = typer.Option(
        "",
        "--source-project",
        help=(
            "Project alias for S3 credential resolution. GR00T inference runs "
            "remotely under a single credential set, so source and target projects "
            "must match when both are provided. See NOVEL_ISSUE_E6_AUTH_SCOPE."
        ),
    ),
    target_project: str = typer.Option(
        "",
        "--target-project",
        help=(
            "Project alias for S3 output credential resolution. GR00T inference "
            "runs remotely under a single credential set, so source and target "
            "projects must match when both are provided. See "
            "NOVEL_ISSUE_E6_AUTH_SCOPE."
        ),
    ),
    embodiment_tag: str = typer.Option(
        DEFAULT_EMBODIMENT_TAG,
        "--embodiment-tag",
        "--robot-embodiment",
        help="GR00T embodiment tag.",
    ),
    inference_mode: InferenceMode = typer.Option(
        InferenceMode.pytorch,
        "--inference-mode",
        help="Inference backend.",
    ),
    steps: int = typer.Option(32, "--steps", help="Maximum episode steps to process."),
    action_horizon: int = typer.Option(
        16, "--action-horizon", help="Predicted action horizon."
    ),
    trt_engine_path: str = typer.Option(
        "./gr00t_n1d7_engines",
        "--trt-engine-path",
        help="TensorRT engine path used when --inference-mode=tensorrt.",
    ),
    model_variant: str = typer.Option(DEFAULT_MODEL, "--model-variant", help="GR00T model variant metadata recorded by serverless inference."),
    runtime: WorkbenchRuntime = typer.Option(WorkbenchRuntime.vm, "--runtime", help="Runtime. serverless creates a Nebius AI Job."),
    project_id: str = typer.Option("", "--project-id", help="Nebius project ID for serverless Jobs."),
    image: str = typer.Option("", "--image", help="Container image for the serverless Job."),
    gpu_type: str = typer.Option("h200", "--gpu-type", help="GPU type for serverless Jobs."),
    gpu_count: int = typer.Option(1, "--gpu-count", help="GPU count for serverless Jobs."),
    gpu_preset: str = typer.Option("", "--gpu-preset", help="Nebius GPU preset override."),
    subnet_id: str = typer.Option("", "--subnet-id", help="Nebius VPC subnet ID for serverless Jobs."),
    job_name: str = typer.Option("", "--job-name", help="Explicit serverless Job name."),
    submit_only: bool = typer.Option(False, "--submit-only", help="Submit serverless Job and return before polling."),
    poll_interval: float = typer.Option(30.0, "--poll-interval", help="Seconds between serverless status checks."),
    timeout: float = typer.Option(3600.0, "--timeout", help="Seconds to wait for serverless completion."),
    output: OutputFormat = typer.Option(
        OutputFormat.text, "--output", help="Output format."
    ),
) -> None:
    """Run GR00T policy inference over evaluation episodes and save predicted actions."""
    if steps <= 0:
        _fail("--steps must be positive.")
    if action_horizon <= 0:
        _fail("--action-horizon must be positive.")
    if source_project and target_project and source_project != target_project:
        _fail(
            "groot infer cannot route distinct source and target credentials. "
            "The remote GR00T process runs under a single credential set "
            "(NOVEL_ISSUE_E6_AUTH_SCOPE). For cross-project staging, use "
            "`npa demo stage`; for GR00T inference, set --source-project or "
            "--target-project, or set both to the same value."
        )
    try:
        input_path = validate_read_path(
            input_path,
            tool="GR00T infer",
            option="--input-path",
            allow_hf=True,
        )
        dataset_path = validate_read_path(
            dataset_path,
            tool="GR00T infer",
            option="--dataset-path",
            allow_hf=False,
        )
        output_path = validate_write_path(
            output_path,
            tool="GR00T infer",
            option="--output-path",
            required=True,
        )
    except PathContractError as exc:
        _fail(str(exc))
    if _is_serverless_runtime(runtime):
        _groot_serverless_infer(
            input_path=input_path,
            dataset_path=dataset_path,
            output_path=output_path,
            project_id=project_id,
            image=image,
            gpu_type=gpu_type,
            gpu_count=gpu_count,
            gpu_preset=gpu_preset,
            subnet_id=subnet_id,
            job_name=job_name,
            submit_only=submit_only,
            poll_interval=poll_interval,
            timeout=timeout,
            embodiment_tag=_normalize_embodiment_tag(embodiment_tag),
            inference_mode=inference_mode.value,
            steps=steps,
            action_horizon=action_horizon,
            model_variant=model_variant,
            output=output,
        )
        return
    cfg = _get_ssh_config()
    remote_storage_project = target_project or source_project
    extra_tokens = (
        storage_env_for_project(remote_storage_project)
        if remote_storage_project
        else None
    )
    ssh = _ssh_client(cfg, extra_tokens=extra_tokens)
    tag = _normalize_embodiment_tag(embodiment_tag)
    cmd = _runtime_command(
        cfg,
        _build_infer_command(
            checkpoint_path=input_path,
            dataset_path=dataset_path,
            output_path=output_path,
            embodiment_tag=tag,
            inference_mode=inference_mode.value,
            endpoint_url=cfg.storage.endpoint_url,
            steps=steps,
            action_horizon=action_horizon,
            trt_engine_path=trt_engine_path,
        ),
        pass_env=GROOT_REMOTE_ENV_NAMES,
    )
    start = time.time()
    try:
        exit_code, stdout, stderr = ssh.run(cmd, stream=output != OutputFormat.json)
    except SSHError as exc:
        _fail(f"SSH error: {exc}")
        return

    result = {
        "status": "success" if exit_code == 0 else "failed",
        "exit_code": exit_code,
        "input_path": input_path,
        "dataset_path": dataset_path,
        "output_path": output_path,
        "embodiment_tag": tag,
        "inference_mode": inference_mode.value,
        "duration_seconds": round(time.time() - start, 1),
    }
    if exit_code != 0:
        result["stderr"] = stderr.strip()[-500:] if stderr else ""
    elif output == OutputFormat.json and stdout.strip():
        result["stdout_tail"] = stdout.strip()[-1000:]
    _output(result, output)
    if exit_code != 0:
        raise typer.Exit(1)


def _resolve_infer_input(
    input_path: str,
    cfg: Any,
    temp_dirs: list[tempfile.TemporaryDirectory[str]],
) -> Path:
    if not _is_s3_uri(input_path):
        path = Path(input_path)
        if not path.exists():
            _fail(f"Input file not found: {input_path}")
        return path

    tmp = tempfile.TemporaryDirectory(prefix="npa-groot-input-")
    temp_dirs.append(tmp)
    downloaded = Path(
        _storage_client_for_config(cfg).download_path(input_path, tmp.name)
    )
    if downloaded.is_file():
        return downloaded
    files = [path for path in downloaded.rglob("*") if path.is_file()]
    if len(files) != 1:
        _fail(f"S3 input path must resolve to exactly one file: {input_path}")
    return files[0]


def _build_infer_payload(input_path: Path) -> dict[str, Any]:
    data = input_path.read_bytes()
    try:
        payload = json.loads(data.decode("utf-8"))
    except json.JSONDecodeError as exc:
        _fail(
            f"GR00T infer input must be JSON matching the Policy API observation format: {exc}"
        )
    if isinstance(payload, dict) and "observation" in payload:
        return payload
    return {"observation": payload}


def _save_json_result(
    data: dict[str, Any],
    output_path: str,
    cfg: Any,
    temp_dirs: list[tempfile.TemporaryDirectory[str]],
) -> str:
    if _is_s3_uri(output_path):
        tmp = tempfile.TemporaryDirectory(prefix="npa-groot-output-")
        temp_dirs.append(tmp)
        local_path = Path(tmp.name) / _s3_path_name(output_path)
        local_path.write_text(json.dumps(data, indent=2))
        return _storage_client_for_config(cfg).upload_file(str(local_path), output_path)

    local_path = Path(output_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_text(json.dumps(data, indent=2))
    return str(local_path)


@app.command("convert")
def convert_cmd(
    input_path: str = typer.Option(
        ..., "--input-path", "--input", "-i", help="S3 URI for the input dataset."
    ),
    output_path: str = typer.Option(
        ..., "--output-path", "--output", "-o", help="S3 URI for the output dataset."
    ),
    direction: ConvertDirection = typer.Option(
        ConvertDirection.lerobot_to_groot,
        "--direction",
        help="Conversion direction.",
    ),
    robot_embodiment: str = typer.Option(
        DEFAULT_EMBODIMENT_TAG,
        "--robot-embodiment",
        "--embodiment-tag",
        help="GR00T embodiment tag to record in the converted dataset.",
    ),
    output: OutputFormat = typer.Option(
        OutputFormat.text, "--output-format", help="Output format."
    ),
) -> None:
    """Convert datasets between standard LeRobot and GR00T LeRobot layout."""
    from npa.adapter.groot import GR00TAdapterError, groot_to_lerobot, lerobot_to_groot

    temp_dirs: list[tempfile.TemporaryDirectory[str]] = []
    try:
        input_path = validate_read_path(
            input_path,
            tool="GR00T convert",
            option="--input-path",
            allow_hf=False,
        )
        output_path = validate_write_path(
            output_path,
            tool="GR00T convert",
            option="--output-path",
            required=True,
        )
        storage = _storage_client_for_project_or_environment()

        tmp = tempfile.TemporaryDirectory(prefix="npa-groot-convert-input-")
        temp_dirs.append(tmp)
        inp = Path(storage.download_directory(input_path, tmp.name))

        tmp = tempfile.TemporaryDirectory(prefix="npa-groot-convert-output-")
        temp_dirs.append(tmp)
        out = Path(tmp.name) / "dataset"

        if not inp.exists():
            _fail(f"Input dataset does not exist: {inp}")

        try:
            if direction == ConvertDirection.lerobot_to_groot:
                converted = lerobot_to_groot(
                    inp,
                    out,
                    robot_embodiment=_normalize_embodiment_tag(robot_embodiment),
                )
            else:
                converted = groot_to_lerobot(inp, out)
        except GR00TAdapterError as exc:
            _fail(str(exc))
            return

        saved_to = storage.upload_directory(str(converted), output_path)
        _output(
            {
                "status": "converted",
                "direction": direction.value,
                "input_path": input_path,
                "output_path": saved_to,
                "robot_embodiment": _normalize_embodiment_tag(robot_embodiment),
            },
            output,
        )
    except PathContractError as exc:
        _fail(str(exc))
    finally:
        for tmp in temp_dirs:
            tmp.cleanup()


@app.command("status")
def status_cmd(
    output: OutputFormat = typer.Option(
        OutputFormat.text, "--output", help="Output format."
    ),
) -> None:
    """Check the GR00T endpoint health."""
    cfg = _get_config()

    try:
        with service_endpoint(cfg, default_port=DEFAULT_SERVER_PORT) as active:
            client = HTTPClient(active.url, timeout=10.0, retries=1)
            data = client.health()
            endpoint_url = active.url
    except EndpointError as exc:
        if output == OutputFormat.json:
            typer.echo(
                json.dumps(
                    {
                        "endpoint": cfg.endpoint,
                        "app_status": "unreachable",
                        "server": "down",
                        "error": str(exc),
                    },
                    indent=2,
                )
            )
        else:
            typer.echo(f"  endpoint: {cfg.endpoint}")
            typer.echo("  app_status: unreachable")
        _fail(f"Cannot prepare GR00T endpoint for {cfg.endpoint}: {exc}")
        return
    except ServerError as exc:
        if output == OutputFormat.json:
            typer.echo(
                json.dumps(
                    {
                        "endpoint": cfg.endpoint,
                        "app_status": "unreachable",
                        "server": "down",
                        "error": str(exc),
                    },
                    indent=2,
                )
            )
        else:
            typer.echo(f"  endpoint: {cfg.endpoint}")
            typer.echo("  app_status: unreachable")
        _fail(f"Cannot reach GR00T endpoint at {cfg.endpoint}/health: {exc}")
        return

    loaded = bool(data.get("loaded"))
    hf_present = bool(getattr(cfg, "hf_token", ""))
    ngc_ok = bool(data.get("ngc_credentials_configured"))
    readiness = {
        "hf_token_present": hf_present,
        "ngc_credentials_configured": ngc_ok,
        "model_loaded": loaded,
        "ready": hf_present and ngc_ok and loaded,
        "blockers": [],
    }
    if not hf_present:
        readiness["blockers"].append(
            "HF_TOKEN not configured - gated model downloads will fail"
        )
    if not ngc_ok:
        readiness["blockers"].append("NGC credentials not configured")
    if not loaded:
        readiness["blockers"].append(
            f"Model {data.get('model') or DEFAULT_MODEL} not loaded"
        )
    app_status = "healthy" if loaded else "degraded"

    result = {
        "endpoint": endpoint_url,
        "app_status": app_status,
        "server": "up",
        **data,
        "readiness": readiness,
    }
    if not loaded:
        result["reason"] = "model not loaded"
    _output(result, output)


@app.command("system-info")
def system_info_cmd(
    output: OutputFormat = typer.Option(
        OutputFormat.text, "--output", help="Output format."
    ),
) -> None:
    """Collect system information and GR00T runtime status from the VM."""
    cfg = _get_ssh_config()
    ssh = SSHClient(cfg.ssh)
    try:
        _, out, err = ssh.run_or_raise(
            _build_system_info_command(container=_is_container_config(cfg))
        )
    except SSHError as exc:
        _fail(f"SSH error: {exc}")
        return

    if output == OutputFormat.json:
        typer.echo(
            json.dumps({"host": cfg.ssh.host, "system_info": out.strip()}, indent=2)
        )
    else:
        if out:
            typer.echo(out.strip())
        if err:
            console.print(f"[red]stderr:[/red]\n{err.strip()[-500:]}")
