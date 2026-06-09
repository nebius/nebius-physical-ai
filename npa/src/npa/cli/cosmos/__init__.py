"""npa workbench cosmos - NVIDIA Cosmos model serving endpoints."""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
import os
import re
import shlex
import tempfile
import time
import uuid
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import typer
from boto3.exceptions import S3UploadFailedError
from rich.console import Console

from npa.cli._error_formatting import format_error_for_user
from npa.cli.ingress import (
    ensure_alias_ingress,
    ensure_deploy_ingress,
    ingress_summary,
    register_byovm_alias,
    resolve_deploy_instance_id,
)
from npa.cli.path_contract import PathContractError, validate_write_path
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
    update_workbench_serverless_endpoint,
    update_workbench_app_status,
    workbench_entry,
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
from npa.clients.project_credentials import storage_client_for_project
from npa.clients.scoped_credentials import (
    bucket_from_s3_uri,
    run_with_host_credential_fallback,
)
from npa.clients.ssh import SSHClient, SSHError
from npa.errors import ScopedCredentialError
from npa.clients.serverless import (
    EndpointInfo,
    EndpointSpec,
    EndpointStatus,
    EndpointNotFoundError,
    JobInfo,
    ServerlessClient,
    ServerlessClientError,
)
from npa.deploy import provisioner
from npa.deploy.configurator import (
    HealthCheckMode,
    audit_remote_env,
    docker_exec_cmd,
    health_check_auto,
    write_manifest,
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
    apply_storage_env_vars,
    detect_gpu_info,
    gpu_config_fields,
    gpu_env_fields,
    is_byovm_runtime,
    resolve_byovm_target,
    runtime_uses_container,
    select_visible_devices,
    ssh_config_for_target,
    workbench_storage_outputs,
)
from npa.deploy.images import container_image_for_tool
from npa.deploy.provisioner import ProvisionerError
from npa.deploy.safety import (
    PlanDecision,
    analyze_terraform_plan,
    format_replacement_required_error,
)
from npa.serverless_common import (
    SubnetResolutionError,
    build_serverless_job_env,
    build_serverless_output_upload_cmd,
    resolve_gpu_platform,
    resolve_subnet,
    split_serverless_env,
    validate_output_path,
)
from npa.workbench.cosmos.cosmos3 import (
    DEFAULT_COSMOS3_MODEL_ID,
    DEFAULT_COSMOS3_SOURCE_REPO,
    DEFAULT_GITHUB_TOKEN_ENV,
    DEFAULT_HF_TOKEN_ENV,
    DEFAULT_NGC_API_KEY_ENV,
    DEFAULT_REASONING_PARSER,
    DEFAULT_TOOL_CALL_PARSER,
    Cosmos3AccessConfig,
    check_cosmos3_access,
    fetch_cosmos3_artifacts,
)

app = typer.Typer(
    name="cosmos",
    help="NVIDIA Cosmos world model serving and inference endpoints.",
    no_args_is_help=True,
)

console = Console(stderr=True)
logger = logging.getLogger(__name__)

_project_alias: str = ""
_workbench_name: str = ""

COSMOS_VERSION = "1.0.9"
COSMOS_HOME = "/opt/cosmos"
COSMOS_VENV = f"{COSMOS_HOME}/venv"
COSMOS_DATA_HOME = "/opt/cosmos-data"
COSMOS_MODEL_DIR = f"{COSMOS_DATA_HOME}/models"
COSMOS_HF_CACHE = f"{COSMOS_DATA_HOME}/hf_cache"
COSMOS_OUTPUT_DIR = f"{COSMOS_DATA_HOME}/outputs"
COSMOS_SERVICE = "npa-cosmos-server"
COSMOS_PIP_EXTRA_INDEX_URL = (
    "https://nvidia-cosmos.github.io/cosmos-dependencies/cu126_torch260/simple"
)
COSMOS_TORCH_VERSION = "2.6.0"
COSMOS_TORCHVISION_VERSION = "0.21.0"
COSMOS_FLASH_ATTN_VERSION = "2.6.3"
COSMOS_FLASH_ATTN_WHEEL_URL = (
    "https://github.com/nvidia-cosmos/cosmos-dependencies/releases/download/v1.1.0/"
    "flash_attn-2.6.3%2Bcu126.torch260-cp310-cp310-linux_x86_64.whl"
)
COSMOS_NATTEN_VERSION = "0.21.0"
COSMOS_NATTEN_WHEEL_URL = (
    "https://github.com/nvidia-cosmos/cosmos-dependencies/releases/download/v1.1.0/"
    "natten-0.21.0%2Bcu126.torch260-cp310-cp310-linux_x86_64.whl"
)
COSMOS_PEFT_MIN_VERSION = "0.17.0"
DEFAULT_MODEL = "nvidia/Cosmos-1.0-Diffusion-7B-Text2World"
COSMOS_INFER_HTTP_TIMEOUT = 30.0
COSMOS_INFER_POLL_INTERVAL = 10.0
COSMOS_ENV_FILE = "/etc/npa-cosmos-server/env"
COSMOS_CREDENTIAL_ENV_NAMES = {
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_ENDPOINT_URL",
    "NEBIUS_S3_ENDPOINT",
    "NEBIUS_S3_BUCKET",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
}

UPLOAD_FAILURE_ERRORS = (
    ScopedCredentialError,
    S3UploadFailedError,
)

def _cosmos_gated_models(model: str) -> list[str]:
    return [model or DEFAULT_MODEL]


def _model_check_or_fail(
    *,
    credentials: Any,
    model: str,
    skip_model_check: bool,
    dry_run: bool,
    no_shared_creds: bool,
) -> None:
    if skip_model_check:
        for repo in _cosmos_gated_models(model):
            console.print(f"  HF access check skipped for {repo}")
        if dry_run:
            console.print("  [dry-run] HF gated-model validation skipped")
        return
    token = "" if no_shared_creds else credentials.hf_token
    if not token:
        warn_if_hf_token_missing(credentials, warn=console.print)
        for repo in _cosmos_gated_models(model):
            console.print(f"  HF access check skipped for {repo}")
        if dry_run:
            raise typer.Exit(1)
        return
    for repo in _cosmos_gated_models(model):
        result = validate_hf_access(token, repo)
        if not result.ok:
            _fail(result.error or f"Unable to validate Hugging Face access to {repo}")
        prefix = "[dry-run] " if dry_run else ""
        console.print(f"  {prefix}HF access ok: {repo}")


def _cosmos_service_env(
    *,
    model: str,
    server_port: int,
    credentials: Any,
    merged_vars: dict[str, str],
    storage_ep: str,
    bucket: str,
    env_region: str,
    byovm_gpu_info: Any,
    byovm_effective_gpu_count: int,
    byovm_visible_devices: str,
    include_shared_creds: bool,
    no_guardrails: bool = False,
) -> dict[str, str]:
    env = {
        "COSMOS_MODEL_ID": model,
        "COSMOS_MODEL_DIR": COSMOS_MODEL_DIR,
        "COSMOS_OUTPUT_DIR": COSMOS_OUTPUT_DIR,
        "COSMOS_SERVER_PORT": str(server_port),
        "COSMOS_DISABLE_SAFETY": "1" if no_guardrails else "0",
        "HF_HOME": COSMOS_HF_CACHE,
        "HUGGINGFACE_HUB_CACHE": COSMOS_HF_CACHE,
        "HF_TOKEN": credentials.hf_token,
        "AWS_ACCESS_KEY_ID": merged_vars.get("nebius_api_key", ""),
        "AWS_SECRET_ACCESS_KEY": merged_vars.get("nebius_secret_key", ""),
        "AWS_ENDPOINT_URL": storage_ep,
        "NEBIUS_S3_ENDPOINT": storage_ep,
        "NEBIUS_S3_BUCKET": bucket,
        "NEBIUS_REGION": env_region,
        "COSMOS_TENSOR_PARALLEL_SIZE": str(byovm_effective_gpu_count or 1),
        "PYTHONUNBUFFERED": "1",
        **gpu_env_fields(
            byovm_gpu_info,
            effective_count=byovm_effective_gpu_count or None,
            visible_devices=byovm_visible_devices,
        ),
    }
    return apply_shared_credential_env(env, credentials, include=include_shared_creds)


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


class OutputFormat(str, Enum):
    text = "text"
    json = "json"


class Backend(str, Enum):
    basic = "basic"
    nim = "nim"
    triton = "triton"


class WorkbenchRuntime(str, Enum):
    vm = "vm"
    container = "container"
    byovm = "byovm"
    serverless = "serverless"


def is_serverless_runtime(runtime: Any) -> bool:
    return str(getattr(runtime, "value", runtime)) == WorkbenchRuntime.serverless.value


COSMOS_RUNTIME_HELP = (
    f"{RUNTIME_HELP} serverless creates a Nebius Serverless AI Endpoint "
    "for the Cosmos container and stores its endpoint URL in the workbench alias."
)


COSMOS_CONTAINER_NAME = "npa-cosmos"


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
    """NVIDIA Cosmos world model serving and inference endpoints."""
    global _project_alias, _workbench_name
    _project_alias = project
    _workbench_name = name


def _fail(msg: str, code: int = 1) -> None:
    console.print(f"[red]Error:[/red] {msg}")
    raise typer.Exit(code)


def _fail_serverless(exc: ServerlessClientError, output: OutputFormat = OutputFormat.text) -> None:
    typer.echo(format_error_for_user(exc, output_format=output.value), err=True)
    raise typer.Exit(1)


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


def _cosmos3_access_config(
    *,
    model_id: str,
    source_repo_url: str,
    cache_dir: Path | None,
    github_token_env: str,
    hf_token_env: str,
    ngc_api_key_env: str,
    require_ngc: bool,
    reasoning_parser: str,
    tool_call_parser: str,
) -> Cosmos3AccessConfig:
    return Cosmos3AccessConfig.from_env(
        model_id=model_id,
        source_repo_url=source_repo_url,
        cache_dir=cache_dir,
        github_token_env=github_token_env,
        hf_token_env=hf_token_env,
        ngc_api_key_env=ngc_api_key_env,
        require_ngc=require_ngc,
        reasoning_parser=reasoning_parser,
        tool_call_parser=tool_call_parser,
    )


def _finish_cosmos3_result(data: dict[str, Any], output: OutputFormat) -> None:
    _output(data, output)
    if data.get("status") != "ok":
        raise typer.Exit(1)


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
    checkpoint_bucket = getattr(storage, "checkpoint_bucket", "")
    if checkpoint_bucket:
        tokens["NEBIUS_S3_BUCKET"] = checkpoint_bucket
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


def _shared_cosmos_env_or_fail(cfg: Any, credentials: Any) -> dict[str, str]:
    merged = {**_storage_env_tokens(cfg), **shared_credential_env(credentials)}
    credential_env = {
        key: value
        for key, value in merged.items()
        if key in COSMOS_CREDENTIAL_ENV_NAMES and value
    }
    if not credential_env:
        _fail("No shared credentials found in environment, ~/.npa/credentials.yaml, or project config.")
    return credential_env


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
  sudo systemctl restart {COSMOS_SERVICE}
elif [ "$mode" = "container" ]; then
  sudo docker restart {COSMOS_CONTAINER_NAME} >/dev/null
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
env_path={COSMOS_ENV_FILE}
mode=""
if sudo systemctl cat {COSMOS_SERVICE} >/dev/null 2>&1; then
  mode="systemd"
elif sudo docker inspect {COSMOS_CONTAINER_NAME} >/dev/null 2>&1; then
  mode="container"
elif sudo test -f "$env_path"; then
  mode="env-only"
else
  echo "No Cosmos service env file found" >&2
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

path.parent.mkdir(parents=True, exist_ok=True)
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
echo "NPA_COSMOS_RELOAD_ENV_COMPLETE env_path=$env_path mode=$mode"
    """
    return _remote_bash(script)


def _build_read_env_command() -> str:
    script = f"""\
set -euo pipefail
env_path={COSMOS_ENV_FILE}
mode=""
if sudo systemctl cat {COSMOS_SERVICE} >/dev/null 2>&1; then
  mode="systemd"
elif sudo docker inspect {COSMOS_CONTAINER_NAME} >/dev/null 2>&1; then
  mode="container"
elif sudo test -f "$env_path"; then
  mode="env-only"
else
  echo "NPA_COSMOS_ENV_READ env_path= mode=missing"
  exit 0
fi
echo "NPA_COSMOS_ENV_READ env_path=$env_path mode=$mode"
sudo cat "$env_path" || true
"""
    return _remote_bash(script)


def _parse_env_read(stdout: str) -> tuple[str, str, str]:
    env_path = ""
    mode = ""
    body: list[str] = []
    for line in stdout.splitlines():
        if line.startswith("NPA_COSMOS_ENV_READ "):
            parts = dict(
                item.split("=", 1)
                for item in line.removeprefix("NPA_COSMOS_ENV_READ ").split()
                if "=" in item
            )
            env_path = parts.get("env_path", "")
            mode = parts.get("mode", "")
            continue
        body.append(line)
    return env_path, mode, "\n".join(body) + ("\n" if body else "")


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
        _fail(f"Cosmos env reload failed: {exc}")

    env_path = ""
    env_mode = ""
    for line in stdout.splitlines():
        if line.startswith("NPA_COSMOS_RELOAD_ENV_COMPLETE "):
            parts = dict(
                item.split("=", 1)
                for item in line.removeprefix("NPA_COSMOS_RELOAD_ENV_COMPLETE ").split()
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


def _model_slug(model: str) -> str:
    return model.replace("/", "--").replace(":", "--")


def _is_s3_uri(path: str) -> bool:
    return path.startswith("s3://")


def _s3_path_name(path: str, default: str = "result.json") -> str:
    name = (
        Path(urlparse(path).path.rstrip("/")).name
        if _is_s3_uri(path)
        else Path(path).name
    )
    return name or default


def _gpu_selection_error() -> str:
    return (
        "GPU selection is required for Cosmos deploy. Provide --gpu-type and --gpu-preset.\n"
        "  Common starting points:\n"
        "    7B Text2World: --gpu-type gpu-l40s-a --gpu-preset 1gpu-40vcpu-160gb\n"
        "    7B Text2World faster serving: --gpu-type gpu-h100-sxm --gpu-preset 1gpu-16vcpu-200gb\n"
        "    14B or larger models: use gpu-h200-sxm, or a multi-GPU preset where available."
    )


def _validate_gpu_selection(gpu_type: str, gpu_preset: str) -> None:
    if not gpu_type and not gpu_preset:
        _fail(_gpu_selection_error())
    if not gpu_type:
        _fail("Missing --gpu-type. Cosmos deploy does not provide a default GPU type.")
    if not gpu_preset:
        _fail(
            "Missing --gpu-preset. Provide the Nebius GPU preset that matches the selected GPU type."
        )


@app.command("check")
def check_cmd(
    model_id: str = typer.Option(
        "",
        "--model-id",
        envvar="NPA_COSMOS3_MODEL_ID",
        help=f"HF model repo ID. Defaults to NPA_COSMOS3_MODEL_ID or {DEFAULT_COSMOS3_MODEL_ID}.",
    ),
    source_repo_url: str = typer.Option(
        "",
        "--source-repo-url",
        envvar="NPA_COSMOS3_SOURCE_REPO",
        help=f"Source repository URL. Defaults to NPA_COSMOS3_SOURCE_REPO or {DEFAULT_COSMOS3_SOURCE_REPO}.",
    ),
    cache_dir: Path | None = typer.Option(
        None,
        "--cache-dir",
        envvar="NPA_COSMOS3_CACHE",
        help="Ephemeral runtime cache directory.",
    ),
    github_token_env: str = typer.Option(
        DEFAULT_GITHUB_TOKEN_ENV,
        "--github-token-env",
        help="Environment variable that holds the GitHub token.",
    ),
    hf_token_env: str = typer.Option(
        DEFAULT_HF_TOKEN_ENV,
        "--hf-token-env",
        help="Environment variable that holds the Hugging Face token.",
    ),
    ngc_api_key_env: str = typer.Option(
        DEFAULT_NGC_API_KEY_ENV,
        "--ngc-api-key-env",
        help="Environment variable that holds the NGC API key.",
    ),
    require_ngc: bool = typer.Option(
        False,
        "--require-ngc/--no-require-ngc",
        help="Require NGC auth for workflows that also need an NGC base container.",
    ),
    reasoning_parser: str = typer.Option(
        DEFAULT_REASONING_PARSER,
        "--reasoning-parser",
        help="vLLM reasoning parser setting carried into serve config.",
    ),
    tool_call_parser: str = typer.Option(
        DEFAULT_TOOL_CALL_PARSER,
        "--tool-call-parser",
        help="vLLM tool-call parser setting carried into serve config.",
    ),
    output: OutputFormat = typer.Option(
        OutputFormat.text,
        "--output",
        help="Output format.",
    ),
) -> None:
    """Check Cosmos3 source and HF checkpoint access without downloading weights."""
    cfg = _cosmos3_access_config(
        model_id=model_id,
        source_repo_url=source_repo_url,
        cache_dir=cache_dir,
        github_token_env=github_token_env,
        hf_token_env=hf_token_env,
        ngc_api_key_env=ngc_api_key_env,
        require_ngc=require_ngc,
        reasoning_parser=reasoning_parser,
        tool_call_parser=tool_call_parser,
    )
    _finish_cosmos3_result(check_cosmos3_access(cfg).as_dict(), output)


@app.command("fetch")
def fetch_cmd(
    model_id: str = typer.Option(
        "",
        "--model-id",
        envvar="NPA_COSMOS3_MODEL_ID",
        help=f"HF model repo ID. Defaults to NPA_COSMOS3_MODEL_ID or {DEFAULT_COSMOS3_MODEL_ID}.",
    ),
    source_repo_url: str = typer.Option(
        "",
        "--source-repo-url",
        envvar="NPA_COSMOS3_SOURCE_REPO",
        help=f"Source repository URL. Defaults to NPA_COSMOS3_SOURCE_REPO or {DEFAULT_COSMOS3_SOURCE_REPO}.",
    ),
    cache_dir: Path | None = typer.Option(
        None,
        "--cache-dir",
        envvar="NPA_COSMOS3_CACHE",
        help="Ephemeral runtime cache directory.",
    ),
    github_token_env: str = typer.Option(
        DEFAULT_GITHUB_TOKEN_ENV,
        "--github-token-env",
        help="Environment variable that holds the GitHub token.",
    ),
    hf_token_env: str = typer.Option(
        DEFAULT_HF_TOKEN_ENV,
        "--hf-token-env",
        help="Environment variable that holds the Hugging Face token.",
    ),
    ngc_api_key_env: str = typer.Option(
        DEFAULT_NGC_API_KEY_ENV,
        "--ngc-api-key-env",
        help="Environment variable that holds the NGC API key.",
    ),
    require_ngc: bool = typer.Option(
        False,
        "--require-ngc/--no-require-ngc",
        help="Require NGC auth for workflows that also need an NGC base container.",
    ),
    reasoning_parser: str = typer.Option(
        DEFAULT_REASONING_PARSER,
        "--reasoning-parser",
        help="vLLM reasoning parser setting carried into serve config.",
    ),
    tool_call_parser: str = typer.Option(
        DEFAULT_TOOL_CALL_PARSER,
        "--tool-call-parser",
        help="vLLM tool-call parser setting carried into serve config.",
    ),
    skip_checkpoint: bool = typer.Option(
        False,
        "--skip-checkpoint",
        help="Clone the source repo but do not download the HF checkpoint.",
    ),
    hf_include: list[str] = typer.Option(
        [],
        "--hf-include",
        help="Optional Hugging Face download include pattern; repeatable.",
    ),
    hf_exclude: list[str] = typer.Option(
        [],
        "--hf-exclude",
        help="Optional Hugging Face download exclude pattern; repeatable.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Replace existing runtime cache subdirectories before fetching.",
    ),
    output: OutputFormat = typer.Option(
        OutputFormat.text,
        "--output",
        help="Output format.",
    ),
) -> None:
    """Clone source and download the HF checkpoint into ephemeral runtime cache."""
    cfg = _cosmos3_access_config(
        model_id=model_id,
        source_repo_url=source_repo_url,
        cache_dir=cache_dir,
        github_token_env=github_token_env,
        hf_token_env=hf_token_env,
        ngc_api_key_env=ngc_api_key_env,
        require_ngc=require_ngc,
        reasoning_parser=reasoning_parser,
        tool_call_parser=tool_call_parser,
    )
    result = fetch_cosmos3_artifacts(
        cfg,
        download_checkpoint=not skip_checkpoint,
        hf_include_patterns=hf_include,
        hf_exclude_patterns=hf_exclude,
        force=force,
    )
    _finish_cosmos3_result(result.as_dict(), output)


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
        help="Source CIDR allowed to reach the Cosmos server.",
    ),
) -> None:
    """Ensure public ingress for the saved Cosmos BYOVM alias."""
    try:
        result = ensure_alias_ingress(
            tool="cosmos",
            port=8081,
            project_alias=_project_alias or None,
            name=name or _workbench_name or None,
            source=source,
        )
    except (ConfigError, NetworkIngressError) as exc:
        _fail(str(exc))
    typer.echo(ingress_summary(result, 8081))


@app.command("register-byovm")
def register_byovm_cmd(
    alias: str = typer.Option(
        ..., "--alias", help="Workbench alias to create or update."
    ),
    instance_id: str = typer.Option(
        ..., "--instance-id", help="Nebius compute instance ID."
    ),
    port: int = typer.Option(8081, "--port", help="Cosmos HTTP service port."),
) -> None:
    """Register an existing VM as a Cosmos BYOVM alias and ensure ingress."""
    try:
        register_byovm_alias(
            tool="cosmos",
            alias=alias,
            instance_id=instance_id,
            port=port,
            project_alias=_project_alias or None,
            warn=console.print,
        )
    except (ConfigError, NetworkIngressError) as exc:
        _fail(str(exc))


def _ensure_basic_backend(backend: Backend) -> None:
    if backend != Backend.basic:
        typer.echo("NIM/Triton backend is not yet implemented")
        raise typer.Exit(1)


def _is_cosmos_workbench(name: str, wb_cfg: dict[str, Any]) -> bool:
    """True when the workbench entry is a Cosmos serving VM."""
    wtype = wb_cfg.get("workbench_type")
    if wtype:
        return wtype == "cosmos"

    normalized = name.replace("_", "-").lower()
    if "cosmos" in normalized:
        return bool(wb_cfg.get("endpoint") or wb_cfg.get("ssh", {}).get("host"))
    return False


def _build_server_py(default_model: str) -> str:
    """Return the remote FastAPI server source for Cosmos inference."""
    return f'''\
from __future__ import annotations

import base64
import os
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

try:
    from diffusers import CosmosTextToWorldPipeline
except Exception:
    CosmosTextToWorldPipeline = None
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video

MODEL_DIR = Path(os.environ.get("COSMOS_MODEL_DIR", "{COSMOS_MODEL_DIR}"))
OUTPUT_DIR = Path(os.environ.get("COSMOS_OUTPUT_DIR", "{COSMOS_HOME}/outputs"))
DEFAULT_MODEL = "{default_model}"
DISABLE_SAFETY = os.environ.get("COSMOS_DISABLE_SAFETY", "0").strip().lower() in {{"1", "true", "yes", "on"}}

app = FastAPI(title="NPA Cosmos Server")
_pipe: Any | None = None
_loaded_model = ""
_jobs: dict[str, dict[str, Any]] = {{}}
_jobs_lock = threading.Lock()
_generation_lock = threading.Lock()


class ServeRequest(BaseModel):
    model: str | None = None


class InputFile(BaseModel):
    filename: str | None = None
    content_base64: str | None = None
    mime_type: str | None = None


class InferRequest(BaseModel):
    prompt: str | None = None
    input: InputFile | None = None


class _NoOpSafetyChecker:
    def to(self, *_args, **_kwargs):
        return self

    def check_text_safety(self, _prompt: str) -> bool:
        return True

    def check_video_safety(self, video):
        return video


def _model_id() -> str:
    return os.environ.get("COSMOS_MODEL_ID", DEFAULT_MODEL)


def _local_model_path(model: str) -> str:
    candidate = MODEL_DIR / model.replace("/", "--").replace(":", "--")
    return str(candidate) if candidate.exists() else model


def _load(model: str | None = None):
    global _pipe, _loaded_model
    requested = model or _model_id()
    if _pipe is not None and _loaded_model == requested:
        return _pipe

    source = _local_model_path(requested)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    if CosmosTextToWorldPipeline is not None and "Text2World" in requested:
        load_kwargs = {{"torch_dtype": dtype}}
        if DISABLE_SAFETY:
            load_kwargs["safety_checker"] = _NoOpSafetyChecker()
        pipe = CosmosTextToWorldPipeline.from_pretrained(source, **load_kwargs)
    else:
        pipe = DiffusionPipeline.from_pretrained(source, torch_dtype=dtype)
    if torch.cuda.is_available():
        pipe.to("cuda")
    _pipe = pipe
    _loaded_model = requested
    return pipe


@app.get("/health")
def health() -> dict[str, Any]:
    return {{"status": "ok", "model": _model_id(), "loaded": _pipe is not None}}


@app.post("/serve")
def serve(req: ServeRequest) -> dict[str, Any]:
    model = req.model or _model_id()
    _load(model)
    return {{"status": "serving", "model": model}}


def _decode_input(input_file: InputFile | None) -> str | None:
    if input_file is None or not input_file.content_base64:
        return None
    suffix = Path(input_file.filename or "input").suffix
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    handle.write(base64.b64decode(input_file.content_base64))
    handle.close()
    return handle.name


def _input_argument(input_path: str) -> Any:
    suffix = Path(input_path).suffix.lower()
    if suffix in {{".jpg", ".jpeg", ".png", ".webp", ".bmp"}}:
        return Image.open(input_path).convert("RGB")
    return input_path


def _export_result(result: Any) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = getattr(result, "frames", None)
    images = getattr(result, "images", None)

    if frames:
        out = OUTPUT_DIR / f"cosmos-{{uuid.uuid4().hex}}.mp4"
        export_to_video(frames[0], str(out), fps=30)
        return out
    if images:
        out = OUTPUT_DIR / f"cosmos-{{uuid.uuid4().hex}}.png"
        images[0].save(out)
        return out

    out = OUTPUT_DIR / f"cosmos-{{uuid.uuid4().hex}}.txt"
    out.write_text(str(result))
    return out


@app.post("/infer")
def infer(req: InferRequest) -> dict[str, Any]:
    job_id = uuid.uuid4().hex
    now = time.time()
    with _jobs_lock:
        _jobs[job_id] = {{
            "job_id": job_id,
            "status": "running",
            "model": _model_id(),
            "submitted_at": now,
            "updated_at": now,
        }}

    thread = threading.Thread(target=_run_job, args=(job_id, req), daemon=True)
    thread.start()
    return {{"job_id": job_id, "status": "running", "model": _model_id()}}


@app.get("/jobs/{{job_id}}")
def job_status(job_id: str) -> dict[str, Any]:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Unknown job_id: {{job_id}}")
        return dict(job)


def _set_job(job_id: str, **updates: Any) -> None:
    with _jobs_lock:
        job = _jobs.setdefault(job_id, {{"job_id": job_id}})
        job.update(updates)
        job["updated_at"] = time.time()


def _run_job(job_id: str, req: InferRequest) -> None:
    try:
        with _generation_lock:
            data = _run_inference(req)
        _set_job(job_id, status="completed", **data)
    except Exception as exc:
        _set_job(job_id, status="failed", error=str(exc))


def _run_inference(req: InferRequest) -> dict[str, Any]:
    pipe = _load()
    prompt = req.prompt or ""
    input_path = _decode_input(req.input)

    if input_path:
        input_arg = _input_argument(input_path)
        attempts = [
            {{"prompt": prompt, "image": input_arg}},
            {{"prompt": prompt, "video": input_arg}},
            {{"prompt": prompt, "input_image_or_video_path": input_path}},
            {{"prompt": prompt, "input_path": input_path}},
        ]
        last_error: TypeError | None = None
        for kwargs in attempts:
            try:
                result = pipe(**kwargs)
                break
            except TypeError as exc:
                last_error = exc
        else:
            raise last_error or TypeError("Cosmos pipeline did not accept the input file")
    else:
        result = pipe(prompt=prompt)

    output_path = _export_result(result)
    return {{"model": _loaded_model, "output_path": str(output_path)}}
'''


def _build_install_command(model: str, port: int, *, no_guardrails: bool = False) -> str:
    server_py = _build_server_py(model)
    model_slug = _model_slug(model)
    disable_safety = "1" if no_guardrails else "0"
    script = f"""\
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update
sudo apt-get install -y software-properties-common git curl ffmpeg
if ! command -v python3.10 >/dev/null 2>&1; then
  sudo add-apt-repository -y ppa:deadsnakes/ppa || true
  sudo apt-get update
fi
sudo apt-get install -y python3.10 python3.10-venv python3.10-dev python3-pip
if [ ! -d {COSMOS_DATA_HOME} ]; then
  sudo mkdir -p {COSMOS_DATA_HOME}
fi
sudo mkdir -p {COSMOS_HOME} {COSMOS_MODEL_DIR} {COSMOS_HF_CACHE} {COSMOS_OUTPUT_DIR}
sudo chown -R "$USER:$USER" {COSMOS_HOME} {COSMOS_DATA_HOME}
python3.10 -m venv {COSMOS_VENV}
{COSMOS_VENV}/bin/python -m pip install --upgrade pip setuptools wheel
{COSMOS_VENV}/bin/python -m pip install "torch=={COSMOS_TORCH_VERSION}" "torchvision=={COSMOS_TORCHVISION_VERSION}" --extra-index-url {COSMOS_PIP_EXTRA_INDEX_URL}
flash_attn_wheel="/tmp/flash_attn-{COSMOS_FLASH_ATTN_VERSION}-cp310-cp310-linux_x86_64.whl"
curl -L -o "$flash_attn_wheel" "{COSMOS_FLASH_ATTN_WHEEL_URL}"
{COSMOS_VENV}/bin/python -m pip install --no-deps "$flash_attn_wheel"
natten_wheel="/tmp/natten-{COSMOS_NATTEN_VERSION}-cp310-cp310-linux_x86_64.whl"
curl -L -o "$natten_wheel" "{COSMOS_NATTEN_WHEEL_URL}"
{COSMOS_VENV}/bin/python -m pip install --no-deps "$natten_wheel"
{COSMOS_VENV}/bin/python -m pip install "cosmos-predict2[cu126]=={COSMOS_VERSION}" --extra-index-url {COSMOS_PIP_EXTRA_INDEX_URL}
{COSMOS_VENV}/bin/python -m pip install "diffusers>=0.34.0" "peft>={COSMOS_PEFT_MIN_VERSION}" transformers accelerate fastapi "uvicorn[standard]" huggingface_hub pillow "imageio[ffmpeg]" pydantic python-multipart
{COSMOS_VENV}/bin/python -m pip install --no-deps cosmos_guardrail
cat > {COSMOS_HOME}/server.py <<'PY'
{server_py}
PY
if [ -f /opt/lerobot/.env ]; then
  set -a
  . /opt/lerobot/.env
  set +a
fi
export HF_HOME={COSMOS_HF_CACHE}
export HUGGINGFACE_HUB_CACHE={COSMOS_HF_CACHE}
if [ -n "${{HF_TOKEN:-}}" ]; then
  {COSMOS_VENV}/bin/huggingface-cli download {shlex.quote(model)} --local-dir {COSMOS_MODEL_DIR}/{model_slug} --token "$HF_TOKEN"
else
  {COSMOS_VENV}/bin/huggingface-cli download {shlex.quote(model)} --local-dir {COSMOS_MODEL_DIR}/{model_slug}
fi
sudo mkdir -p /etc/npa-cosmos-server
sudo tee /etc/npa-cosmos-server/env >/dev/null <<'ENV'
COSMOS_MODEL_ID={model}
COSMOS_MODEL_DIR={COSMOS_MODEL_DIR}
COSMOS_OUTPUT_DIR={COSMOS_OUTPUT_DIR}
COSMOS_SERVER_PORT={port}
COSMOS_DISABLE_SAFETY={disable_safety}
HF_HOME={COSMOS_HF_CACHE}
HUGGINGFACE_HUB_CACHE={COSMOS_HF_CACHE}
ENV
if [ -n "${{HF_TOKEN:-}}" ]; then
  printf 'HF_TOKEN=%s\n' "$HF_TOKEN" | sudo tee -a /etc/npa-cosmos-server/env >/dev/null
fi
sudo tee /etc/systemd/system/{COSMOS_SERVICE}.service >/dev/null <<'UNIT'
[Unit]
Description=NPA Cosmos model server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory={COSMOS_HOME}
EnvironmentFile=/etc/npa-cosmos-server/env
ExecStart={COSMOS_VENV}/bin/uvicorn server:app --host 0.0.0.0 --port $COSMOS_SERVER_PORT
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
UNIT
sudo systemctl daemon-reload
sudo systemctl enable {COSMOS_SERVICE}
sudo systemctl restart {COSMOS_SERVICE}
{COSMOS_VENV}/bin/python - <<'PY'
from importlib import metadata
version = metadata.version("cosmos-predict2")
if version != "{COSMOS_VERSION}":
    raise RuntimeError(f"expected cosmos-predict2 {COSMOS_VERSION}, found {{version}}")
print("COSMOS_ENV_SMOKE_OK")
PY
"""
    return _remote_bash(script)


def _build_serve_command(model: str, port: int, *, no_guardrails: bool = False) -> str:
    server_py = _build_server_py(model)
    disable_safety = "1" if no_guardrails else "0"
    script = f"""\
set -euo pipefail
cat > {COSMOS_HOME}/server.py <<'PY'
{server_py}
PY
sudo mkdir -p /etc/npa-cosmos-server
sudo tee /etc/npa-cosmos-server/env >/dev/null <<'ENV'
COSMOS_MODEL_ID={model}
COSMOS_MODEL_DIR={COSMOS_MODEL_DIR}
COSMOS_OUTPUT_DIR={COSMOS_OUTPUT_DIR}
COSMOS_SERVER_PORT={port}
COSMOS_DISABLE_SAFETY={disable_safety}
HF_HOME={COSMOS_HF_CACHE}
HUGGINGFACE_HUB_CACHE={COSMOS_HF_CACHE}
ENV
if [ -n "${{HF_TOKEN:-}}" ]; then
  printf 'HF_TOKEN=%s\n' "$HF_TOKEN" | sudo tee -a /etc/npa-cosmos-server/env >/dev/null
fi
sudo systemctl daemon-reload
sudo systemctl enable {COSMOS_SERVICE}
sudo systemctl restart {COSMOS_SERVICE}
sudo systemctl --no-pager status {COSMOS_SERVICE} || true
"""
    return _remote_bash(script)


def _deploy_step_count(skip_infra: bool, skip_app: bool, destroy: bool) -> int:
    if destroy:
        return 2
    count = 1 if skip_infra else 2
    if not skip_app:
        count += 4
    count += 1
    return count


def _parse_key_value_options(items: list[str], *, flag: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            _fail(f"Invalid {flag} format: {item} (expected KEY=VALUE)")
        key, value = item.split("=", 1)
        if not key:
            _fail(f"Invalid {flag} format: {item} (empty key)")
        parsed[key] = value
    return parsed


def _serverless_endpoint_name(project: str, name: str) -> str:
    raw = f"npa-cosmos-{project}-{name}".lower()
    normalized = re.sub(r"[^a-z0-9-]+", "-", raw).strip("-")
    normalized = re.sub(r"-+", "-", normalized)
    return normalized[:63].rstrip("-") or "npa-cosmos"


def _serverless_endpoint_ref(cfg: Any) -> str:
    serverless = getattr(cfg, "serverless", None)
    return (
        str(getattr(serverless, "endpoint_id", "") or "")
        or str(getattr(serverless, "endpoint_name", "") or "")
        or str(getattr(cfg, "name", "") or "")
    )


def _serverless_hf_env() -> dict[str, str]:
    file_credentials = load_credentials(environ={})
    token = (
        file_credentials.hf_token
        or os.environ.get("HF_TOKEN", "")
        or os.environ.get("HUGGINGFACE_TOKEN", "")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN", "")
    )
    if not token:
        return {}
    return {
        "HF_TOKEN": token,
        "HUGGINGFACE_HUB_TOKEN": token,
    }


def _serverless_job_name(project: str, name: str) -> str:
    raw = f"npa-cosmos-jobs-{project}-{name}".lower()
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9-]+", "-", raw)).strip("-")[:63]


def _serverless_train_output_path(project: str, job_name: str, output_path: str) -> str:
    if output_path:
        try:
            validate_output_path(output_path)
        except ValueError:
            _fail("Cosmos train --output-path expects an S3 URI for serverless jobs.")
        return output_path.rstrip("/") + "/"
    storage = resolve_project_storage(project)
    bucket = getattr(storage, "checkpoint_bucket", "")
    if not bucket:
        _fail("Cosmos train --runtime serverless requires storage.checkpoint_bucket.")
    bucket = bucket.rstrip("/")
    if not bucket.startswith("s3://"):
        bucket = f"s3://{bucket}"
    return f"{bucket}/jobs/{job_name}/"


def _serverless_job_env(
    project: str,
    *,
    require_hf: bool,
    output_path: str = "",
) -> tuple[dict[str, str], dict[str, str]]:
    storage = resolve_project_storage(project)
    hf_env = _serverless_hf_env()
    shared_env = shared_credential_env(load_credentials(environ={}))
    hf_token = (
        hf_env.get("HF_TOKEN")
        or hf_env.get("HUGGING_FACE_HUB_TOKEN")
        or hf_env.get("HUGGINGFACE_HUB_TOKEN")
        or shared_env.get("HF_TOKEN")
        or shared_env.get("HUGGING_FACE_HUB_TOKEN")
    )
    env = build_serverless_job_env(
        output_path=output_path,
        hf_token=hf_token or None,
        s3_credentials={
            "aws_access_key_id": storage.aws_access_key_id or shared_env.get("AWS_ACCESS_KEY_ID", ""),
            "aws_secret_access_key": storage.aws_secret_access_key or shared_env.get("AWS_SECRET_ACCESS_KEY", ""),
            "endpoint_url": storage.endpoint_url or shared_env.get("AWS_ENDPOINT_URL", ""),
        },
        extra_env={"NPA_REQUIRE_HF": "1" if require_hf else "0"},
    )
    return split_serverless_env(env)


def _cosmos_train_smoke_command(seconds: int) -> str:
    local_dir = "/tmp/npa-cosmos-train-smoke"
    script = f"""
import json, os, pathlib, time
from urllib.parse import urlparse
if os.environ.get("NPA_REQUIRE_HF") == "1" and not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")):
    raise SystemExit("HF auth missing")
time.sleep({max(0, seconds)})
uri = os.environ["NPA_OUTPUT_PATH"]
p = urlparse(uri)
out = pathlib.Path("{local_dir}")
out.mkdir(parents=True, exist_ok=True)
(out / "checkpoint.json").write_text(json.dumps({{"status": "succeeded", "job": os.environ.get("NPA_JOB_NAME", ""), "smoke": True}}, indent=2))
print("NPA_COSMOS_TRAIN_SMOKE_DONE", uri.rstrip("/") + "/checkpoint.json", flush=True)
""".strip()
    upload = build_serverless_output_upload_cmd(local_dir, "")
    return _remote_bash(f"python3 - <<'PY'\n{script}\nPY\n{upload}")


def _serverless_project_id(cfg: Any) -> str:
    serverless = getattr(cfg, "serverless", None)
    return (
        str(getattr(serverless, "project_id", "") or "")
        or str(getattr(cfg, "project_id", "") or "")
    )


def _delete_serverless_endpoint_for_config(
    cfg: Any,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    project_id = _serverless_project_id(cfg)
    endpoint_ref = _serverless_endpoint_ref(cfg)
    if not project_id:
        raise ServerlessClientError("Serverless project_id is not saved for this alias")
    if not endpoint_ref:
        raise ServerlessClientError("Serverless endpoint ID/name is not saved for this alias")
    if dry_run:
        return {
            "status": "dry_run",
            "project_id": project_id,
            "endpoint": endpoint_ref,
        }
    client = ServerlessClient()
    client.delete_endpoint(project_id, endpoint_ref)
    return {
        "status": "deleted",
        "project_id": project_id,
        "endpoint": endpoint_ref,
    }


def _serverless_endpoint_status(cfg: Any) -> EndpointInfo:
    project_id = _serverless_project_id(cfg)
    endpoint_ref = _serverless_endpoint_ref(cfg)
    if not project_id:
        raise ServerlessClientError("Serverless project_id is not saved for this alias")
    if not endpoint_ref:
        raise EndpointNotFoundError("Serverless endpoint ID/name is not saved for this alias")
    return ServerlessClient().get_endpoint(project_id, endpoint_ref)


def _serverless_job_status_payload(
    client: ServerlessClient,
    info: JobInfo,
    *,
    platform: str = "",
    gpu_count: int = 0,
) -> dict[str, Any]:
    status = client.classify_queue_state(info)
    payload: dict[str, Any] = {
        "job_id": info.id,
        "job_name": info.name,
        "status": status,
        "raw_status": info.status,
        "output_uris": list(info.output_uris),
    }
    if info.status == "queued":
        payload["queue_state_classification"] = (
            "capacity" if status == "waiting_for_capacity" else "scheduled"
        )
        payload["queued_for_seconds"] = info.queued_for_seconds
        payload["platform"] = platform or info.platform
        payload["gpu_count"] = gpu_count or info.gpu_count
        payload["hint"] = (
            "Platform may be at capacity. Retry status in a few minutes."
            if status == "waiting_for_capacity"
            else "Job is scheduled and waiting to start."
        )
    return payload


def _serverless_job_status_for_config(cfg: Any) -> dict[str, Any] | None:
    job_cfg = getattr(cfg, "serverless_job", None)
    job_ref = str(getattr(job_cfg, "job_id", "") or getattr(job_cfg, "job_name", ""))
    project_id = str(getattr(job_cfg, "project_id", "") or _serverless_project_id(cfg))
    if not job_ref or not project_id:
        return None
    client = ServerlessClient()
    info = client.get_job(job_ref, project_id)
    result = _serverless_job_status_payload(
        client,
        info,
        platform=str(getattr(job_cfg, "gpu_type", "")),
        gpu_count=int(getattr(job_cfg, "gpu_count", 0) or 0),
    )
    result.update({"runtime": "serverless", "workbench": getattr(cfg, "name", "")})
    return result


def _deploy_serverless_endpoint(
    *,
    proj_alias: str,
    wb_name: str,
    project_id: str,
    env_region: str,
    image: str,
    platform: str,
    preset: str,
    container_port: int,
    model: str,
    auth: str,
    subnet_id: str,
    env_vars: dict[str, str],
    volumes: list[str],
    no_guardrails: bool,
    replace: bool,
    default: bool,
    wait: bool,
    dry_run: bool,
    output: OutputFormat,
) -> None:
    if not project_id:
        _fail(
            "Serverless deploy requires a Nebius project ID. Configure the project "
            "in ~/.npa/config.yaml or pass --project-id."
        )
    if not platform or not preset:
        _validate_gpu_selection(platform, preset)

    endpoint_name = _serverless_endpoint_name(proj_alias, wb_name)
    image_ref = image or container_image_for_tool(
        "cosmos", registry=resolve_container_registry(proj_alias)
    )
    serverless_env = {
        "COSMOS_MODEL_ID": model,
        "COSMOS_SERVER_PORT": str(container_port),
        "COSMOS_DISABLE_SAFETY": "1" if no_guardrails else "0",
        **env_vars,
    }
    extra_env = _serverless_hf_env()

    existing = workbench_entry(proj_alias, wb_name)
    if existing and str(existing.get("runtime", "")).lower() == "serverless" and not replace:
        _fail(
            f"Serverless alias {proj_alias}/{wb_name} already exists. "
            "Use --replace to delete and recreate the endpoint."
        )
    try:
        resolved_subnet_id = resolve_subnet(
            project_id=project_id,
            explicit_subnet_id=subnet_id,
        )
    except SubnetResolutionError as exc:
        _fail(str(exc))

    if dry_run:
        result = {
            "status": "dry_run",
            "project": proj_alias,
            "name": wb_name,
            "runtime": "serverless",
            "serverless_project_id": project_id,
            "endpoint_name": endpoint_name,
            "image": image_ref,
            "platform": platform,
            "preset": preset,
            "container_port": container_port,
            "auth": auth,
            "subnet_id": resolved_subnet_id,
            "model": model,
            "volumes": volumes,
            "env_keys": sorted(set(serverless_env) | set(extra_env)),
            "replace": replace,
        }
        _output(result, output)
        return

    if replace and existing:
        try:
            cfg = resolve_config(project=proj_alias, name=wb_name)
            _delete_serverless_endpoint_for_config(cfg)
        except (ConfigError, EndpointNotFoundError):
            pass
        except ServerlessClientError as exc:
            _fail_serverless(exc, output)

    set_default = default or not list_projects()
    client = ServerlessClient()
    spec = EndpointSpec(
        name=endpoint_name,
        project_id=project_id,
        image=image_ref,
        platform=platform,
        preset=preset,
        container_ports=[container_port],
        auth=auth,
        subnet_id=resolved_subnet_id,
        env=serverless_env,
        volumes=volumes,
    )

    try:
        info = client.create_endpoint(spec, extra_env=extra_env)
        if wait:
            info = client.wait_for_running(project_id, info.id or endpoint_name)
    except ValueError as exc:
        _fail(str(exc))
    except ServerlessClientError as exc:
        _fail_serverless(exc, output)
    except TimeoutError as exc:
        _fail(str(exc))

    endpoint_url = info.url
    update_workbench_serverless_endpoint(
        proj_alias,
        wb_name,
        endpoint_id=info.id,
        endpoint_name=info.name or endpoint_name,
        project_id=project_id,
        url=endpoint_url,
        image=image_ref,
        platform=platform,
        preset=preset,
        container_port=container_port,
        auth=auth,
    )
    write_config(
        {
            "projects": {
                proj_alias: {
                    "project_id": project_id,
                    "region": env_region,
                    "workbenches": {
                        wb_name: {
                            "model": model,
                            "backend": Backend.basic.value,
                        },
                    },
                },
            },
        }
    )
    if wait and info.status is EndpointStatus.RUNNING:
        update_workbench_app_status(proj_alias, wb_name, APP_STATUS_HEALTHY)
    if set_default:
        write_config({"default_project": proj_alias, "default_workbench": wb_name})

    result = {
        "status": "deployed",
        "project": proj_alias,
        "name": wb_name,
        "runtime": "serverless",
        "serverless_project_id": project_id,
        "endpoint_id": info.id,
        "endpoint_name": info.name or endpoint_name,
        "endpoint": endpoint_url,
        "serverless_status": info.status.value,
        "image": image_ref,
        "platform": platform,
        "preset": preset,
        "container_port": container_port,
        "model": model,
    }
    if output == OutputFormat.text:
        console.print("")
        console.print(f"[bold green]Deploy complete.[/bold green] ({proj_alias}/{wb_name})")
    _output(result, output)


@app.command("autoscale")
def autoscale_cmd(
    min_replicas: int = typer.Option(
        1,
        "--min-replicas",
        help="Minimum serverless endpoint replicas.",
    ),
    max_replicas: int = typer.Option(
        4,
        "--max-replicas",
        help="Maximum serverless endpoint replicas for parallel Cosmos inference.",
    ),
    target_concurrency: int = typer.Option(
        0,
        "--target-concurrency",
        help="Optional target concurrent requests per replica.",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show the autoscale plan only."),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Configure Cosmos serverless endpoint autoscaling."""
    if min_replicas < 0:
        _fail(f"--min-replicas must be non-negative, got {min_replicas}")
    if max_replicas < 1:
        _fail(f"--max-replicas must be positive, got {max_replicas}")
    if max_replicas < min_replicas:
        _fail("--max-replicas must be greater than or equal to --min-replicas")
    if target_concurrency < 0:
        _fail(f"--target-concurrency must be non-negative, got {target_concurrency}")

    cfg = _get_config()
    if not is_serverless_runtime(getattr(cfg, "runtime", "")):
        _fail("Cosmos autoscale requires a --runtime serverless endpoint alias.")
    project_id = _serverless_project_id(cfg)
    endpoint_ref = _serverless_endpoint_ref(cfg)
    if not project_id or not endpoint_ref:
        _fail("Cosmos autoscale requires saved serverless project and endpoint metadata.")

    plan = {
        "project": getattr(cfg, "project", ""),
        "name": getattr(cfg, "name", ""),
        "runtime": "serverless",
        "serverless_project_id": project_id,
        "endpoint": endpoint_ref,
        "min_replicas": min_replicas,
        "max_replicas": max_replicas,
        "target_concurrency": target_concurrency,
    }
    if dry_run or _env_dry_run():
        _output({"status": "dry_run", **plan}, output)
        return

    try:
        info = ServerlessClient().set_endpoint_autoscale(
            project_id,
            endpoint_ref,
            min_replicas=min_replicas,
            max_replicas=max_replicas,
            target_concurrency=target_concurrency,
        )
    except ServerlessClientError as exc:
        _fail_serverless(exc, output)
        return

    write_config(
        {
            "projects": {
                getattr(cfg, "project", ""): {
                    "workbenches": {
                        getattr(cfg, "name", ""): {
                            "serverless": {
                                "autoscale": {
                                    "min_replicas": min_replicas,
                                    "max_replicas": max_replicas,
                                    "target_concurrency": target_concurrency,
                                }
                            }
                        }
                    }
                }
            }
        }
    )
    _output(
        {
            "status": "configured",
            **plan,
            "endpoint_id": info.id,
            "endpoint_name": info.name,
            "serverless_status": info.status.value,
        },
        output,
    )


def _env_dry_run() -> bool:
    return os.environ.get("NPA_DRY_RUN", "").lower() in {"1", "true", "yes"} or os.environ.get(
        "DRY_RUN", ""
    ).lower() in {"1", "true", "yes"}


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


@app.command("list")
def list_cmd(
    output: OutputFormat = typer.Option(
        OutputFormat.text, "--output", help="Output format."
    ),
) -> None:
    """List configured Cosmos workbenches."""
    projects = list_projects()
    def_proj = default_project_name()
    def_wb = default_workbench_name()

    if output == OutputFormat.json:
        filtered = {}
        for pname, pcfg in projects.items():
            wbs = {
                k: v
                for k, v in pcfg.get("workbenches", {}).items()
                if _is_cosmos_workbench(k, v)
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
            "No projects configured. Run 'npa workbench cosmos deploy' to create one."
        )
        return

    any_shown = False
    for proj_name, proj_cfg in projects.items():
        workbenches = {
            k: v
            for k, v in proj_cfg.get("workbenches", {}).items()
            if _is_cosmos_workbench(k, v)
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
            "No Cosmos workbenches configured. Run 'npa workbench cosmos deploy' to create one."
        )


@app.command("cleanup-partial")
def cleanup_partial_cmd(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation."),
) -> None:
    """Clean up orphaned Terraform resources from an interrupted Cosmos deploy."""
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
    gpu_type: str = typer.Option("", "--gpu-type", help="Nebius GPU platform."),
    gpu_preset: str = typer.Option("", "--gpu-preset", help="Nebius GPU preset."),
    platform: str = typer.Option(
        "",
        "--platform",
        help="Serverless compute platform. Defaults to --gpu-type for serverless deploys.",
    ),
    preset: str = typer.Option(
        "",
        "--preset",
        help="Serverless compute preset. Defaults to --gpu-preset for serverless deploys.",
    ),
    image: str = typer.Option(
        "",
        "--image",
        help="Container image for serverless endpoint deploys. Defaults to the configured npa Cosmos image.",
    ),
    container_port: int = typer.Option(
        0,
        "--container-port",
        help="Container port exposed by serverless endpoint deploys. Defaults to --server-port.",
    ),
    auth: str = typer.Option(
        "none",
        "--auth",
        help="Serverless endpoint auth mode: none or token.",
    ),
    subnet_id: str = typer.Option(
        "",
        "--subnet-id",
        help="Serverless subnet ID. Required by Nebius when a project has multiple subnets.",
    ),
    env: list[str] = typer.Option(
        [],
        "--env",
        help="Non-secret serverless container env var in KEY=VALUE form. Repeatable.",
    ),
    volume: list[str] = typer.Option(
        [],
        "--volume",
        help="Serverless endpoint volume mount. Repeatable.",
    ),
    wait: bool = typer.Option(
        False,
        "--wait",
        help="Wait for the serverless endpoint to reach RUNNING.",
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
            "Without this flag, deploy against an existing alias updates in place without Terraform."
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
    no_guardrails: bool = typer.Option(
        False,
        "--no-guardrails",
        help="Opt out of Cosmos safety guardrails for generated outputs.",
    ),
    health_check_mode: HealthCheckMode = typer.Option(
        HealthCheckMode.auto,
        "--health-check-mode",
        help="Health check mode: public, ssh, or auto. BYOVM auto tries public briefly, then SSH.",
    ),
    verify_env: bool = typer.Option(
        False,
        "--verify-env/--no-verify-env",
        help="Audit deployed shared credentials after app deploy.",
    ),
    model: str = typer.Option(
        DEFAULT_MODEL,
        "--model",
        help="Hugging Face Cosmos model ID to download and serve.",
    ),
    backend: Backend = typer.Option(
        Backend.basic,
        "--backend",
        help=(
            "Serving backend: basic uses the built-in FastAPI/Diffusers server; "
            "nim will use NVIDIA NIM containers; triton will use Triton/TensorRT model serving."
        ),
    ),
    server_port: int = typer.Option(
        8080,
        "--server-port",
        help="Cosmos server port on the VM. For serverless deploys this is the default container port.",
    ),
    preemptible: bool = typer.Option(
        True, "--preemptible/--no-preemptible", help="Preemptible (spot) instance."
    ),
    runtime: WorkbenchRuntime = typer.Option(
        WorkbenchRuntime.vm, "--runtime", help=COSMOS_RUNTIME_HELP
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
        "", "--ssh-user", help="BYOVM SSH username. Defaults to ubuntu."
    ),
    gpu_count: int = typer.Option(
        0, "--gpu-count", help="Limit visible GPUs on BYOVM (0 = all detected)."
    ),
    disk_size: int | None = typer.Option(
        None,
        "--disk-size",
        help="Boot disk size in GiB. Defaults to 250 for container runtime; VM runtime keeps the Terraform default.",
    ),
    default: bool = typer.Option(
        False, "--default", help="Set this workbench as the default."
    ),
    output: OutputFormat = typer.Option(
        OutputFormat.text, "--output", help="Output format."
    ),
) -> None:
    """Deploy or destroy a Cosmos model serving backend."""
    _ensure_basic_backend(backend)
    byovm = is_byovm_runtime(runtime)
    serverless = is_serverless_runtime(runtime)
    if not destroy and not byovm and not serverless:
        _validate_gpu_selection(gpu_type, gpu_preset)

    proj_alias = _project_alias or None
    wb_name = _workbench_name or "cosmos"
    use_remote_state = not tf_dir and not byovm
    if byovm:
        skip_infra = True

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
        proj_alias = env_region or ("serverless" if serverless else ("byovm" if byovm else "default"))

    if serverless:
        if destroy:
            try:
                cfg = resolve_config(project=proj_alias, name=wb_name)
                result = _delete_serverless_endpoint_for_config(cfg, dry_run=dry_run)
                if not dry_run:
                    remove_workbench_config(proj_alias, wb_name)
                _output({**result, "project": proj_alias, "name": wb_name}, output)
            except ConfigError as exc:
                _fail(f"Serverless endpoint destroy failed: {exc}")
            except ServerlessClientError as exc:
                _fail_serverless(exc, output)
            return

        try:
            serverless_env = _parse_key_value_options(env, flag="--env")
        except typer.Exit:
            raise
        _deploy_serverless_endpoint(
            proj_alias=proj_alias,
            wb_name=wb_name,
            project_id=env_project,
            env_region=env_region,
            image=image,
            platform=platform or gpu_type,
            preset=preset or gpu_preset,
            container_port=container_port or server_port,
            model=model,
            auth=auth,
            subnet_id=subnet_id,
            env_vars=serverless_env,
            volumes=volume,
            no_guardrails=no_guardrails,
            replace=replace,
            default=default,
            wait=wait,
            dry_run=dry_run,
            output=output,
        )
        return

    existing_managed_alias = alias_has_terraform_state(proj_alias, wb_name)
    existing_byovm_alias = workbench_is_byovm(proj_alias, wb_name)
    if not destroy and (existing_managed_alias or existing_byovm_alias):
        if replace and existing_byovm_alias:
            _fail(
                f"{proj_alias}/{wb_name} is a BYOVM alias; --replace is only valid for Terraform-managed aliases."
            )
            return
        if replace:
            if not yes:
                _confirm_or_exit(
                    f"--replace will provision replacement infrastructure for '{proj_alias}/{wb_name}'. Continue?"
                )
        else:
            console.print(
                f"Existing alias {proj_alias}/{wb_name} found; updating in place without Terraform."
            )
            skip_infra = True
            use_remote_state = False

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
                "  Example: npa workbench cosmos -p eu-north1 -n cosmos-7b deploy \\\n"
                "    --project-id project-... --tenant-id tenant-... --region eu-north1 \\\n"
                "    --gpu-type gpu-l40s-a --gpu-preset 1gpu-40vcpu-160gb"
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
        apply_storage_env_vars(merged_vars, explicit_vars=extra_vars)
    if byovm:
        apply_project_storage_vars(
            merged_vars,
            project=proj_alias,
            explicit_vars=extra_vars,
            warn=console.print,
        )
        apply_storage_env_vars(merged_vars, explicit_vars=extra_vars)
    if not byovm:
        try:
            provisioner.apply_boot_disk_tf_vars(merged_vars, runtime, disk_size)
        except ValueError as exc:
            _fail(str(exc))
            return

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

    instance_name = f"cosmos-{proj_alias}-{wb_name}"

    if destroy:
        if byovm:
            console.print(
                f"  [1/1] Unregistering BYOVM workbench {proj_alias}/{wb_name}..."
            )
            if not dry_run:
                remove_workbench_config(proj_alias, wb_name)
            console.print(
                f"  {proj_alias}/{wb_name} unregistered. BYOVM host was not modified."
            )
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
            provisioner.destroy(
                tf_dir=resolved_tf_dir or None,
                tf_vars={
                    "gpu_platform": gpu_type,
                    "gpu_preset": gpu_preset,
                    "instance_name": instance_name,
                    "workbench_type": "cosmos",
                    "enable_preemptible": "true" if preemptible else "false",
                    **merged_vars,
                },
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
            "instance_name": instance_name,
            "workbench_type": "cosmos",
            "enable_preemptible": "true" if preemptible else "false",
            **merged_vars,
        }
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
                plan_output = provisioner.plan(
                    tf_dir=resolved_tf_dir or None, tf_vars=all_vars
                )
                plan_analysis = analyze_terraform_plan(
                    plan_output, existing_state=existing_managed_alias
                )
                if plan_analysis.decision == PlanDecision.REPLACEMENT_REQUIRED:
                    if not replace:
                        _fail(format_replacement_required_error(plan_analysis))
                        return
                    console.print(
                        "    Replacement allowed by --replace: "
                        + ", ".join(
                            item.address for item in plan_analysis.replacement_resources
                        )
                    )
                if plan_analysis.decision == PlanDecision.NO_CHANGES:
                    console.print("    Terraform plan has no changes; deploy is a no-op.")
                    tf_outputs = provisioner.outputs(tf_dir=resolved_tf_dir or None)
                else:
                    tf_outputs = provisioner.apply(
                        tf_dir=resolved_tf_dir or None, tf_vars=all_vars
                    )
            except ProvisionerError as exc:
                _fail(f"Terraform plan/apply failed: {exc}")
                return
        console.print(f"    VM IP: {tf_outputs.get('vm_ip', 'unknown')}")
    else:
        step += 1
        console.print(
            f"  [{step}/{total_steps}] "
            + (
                "Using BYOVM target..."
                if byovm
                else "Skipping infra, reading existing config..."
            )
        )
        resolved_tf_dir = tf_dir
        if byovm:
            try:
                target = resolve_byovm_target(
                    host=host, ssh_key=ssh_key, ssh_user=ssh_user
                )
                bucket = merged_vars.get("s3_bucket", "") or os.environ.get(
                    "NPA_CHECKPOINT_BUCKET", ""
                )
                storage_ep = merged_vars.get("s3_endpoint", "") or os.environ.get(
                    "AWS_ENDPOINT_URL", ""
                )
                tf_outputs = workbench_storage_outputs(
                    target=target, bucket=bucket, endpoint=storage_ep
                )
                if not dry_run:
                    ssh = SSHClient(
                        ssh_config_for_target(target, tokens=credentials.tokens)
                    )
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
            except (ValueError, SSHError) as exc:
                _fail(str(exc))
                return
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
                        "tf_instance_name": instance_name,
                        "workbench_type": "cosmos",
                        "runtime": runtime.value,
                        "app_status": APP_STATUS_PROVISIONED,
                        "endpoint_strategy": "public",
                        "service_port": server_port,
                        "model": model,
                        "backend": backend.value,
                        **byovm_fields,
                        "ssh": {"host": vm_ip, "user": ssh_user, "key_path": ssh_key},
                        "storage": {
                            "checkpoint_bucket": bucket_display,
                            "endpoint_url": storage_ep,
                        },
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

        if runtime_uses_container(runtime):
            step += 1
            console.print(
                f"  [{step}/{total_steps}] Starting Cosmos container and preparing {model}..."
            )
            service_env = _cosmos_service_env(
                model=model,
                server_port=server_port,
                credentials=credentials,
                merged_vars=merged_vars,
                storage_ep=storage_ep,
                bucket=bucket,
                env_region=env_region,
                byovm_gpu_info=byovm_gpu_info,
                byovm_effective_gpu_count=byovm_effective_gpu_count,
                byovm_visible_devices=byovm_visible_devices,
                include_shared_creds=not no_shared_creds,
                no_guardrails=no_guardrails,
            )
            if dry_run:
                console.print(
                    "    [dry-run] Would pull and run the Cosmos container image"
                )
                console.print("    [dry-run] Service env:")
                console.print(render_redacted_env_file(service_env).rstrip())
            else:
                from npa.deploy.configurator import (
                    deploy_workbench_container,
                    write_remote_docker_env_file,
                )

                try:
                    write_remote_docker_env_file(
                        ssh,
                        "/etc/npa-cosmos-server/env",
                        service_env,
                        owner=ssh_user,
                    )
                    image_ref = container_image_for_tool(
                        "cosmos",
                        registry=resolve_container_registry(proj_alias),
                    )
                    ssh.run(
                        "sudo systemctl stop npa-cosmos-server >/dev/null 2>&1 || true"
                    )
                    deploy_workbench_container(
                        ssh,
                        image_ref=image_ref,
                        container_name=COSMOS_CONTAINER_NAME,
                        env_file="/etc/npa-cosmos-server/env",
                        volumes=[
                            f"{COSMOS_DATA_HOME}:{COSMOS_DATA_HOME}",
                            "/etc/npa-cosmos-server/env:/etc/npa-cosmos-server/env:ro",
                        ],
                        work_dirs=[
                            COSMOS_MODEL_DIR,
                            COSMOS_HF_CACHE,
                            COSMOS_OUTPUT_DIR,
                        ],
                        command=(
                            "-lc "
                            + shlex.quote(
                                "cd /opt/cosmos && "
                                f"exec /opt/cosmos/venv/bin/python -m uvicorn server:app --host 0.0.0.0 --port {server_port}"
                            )
                        ),
                        registry_token=merged_vars.get("iam_token", ""),
                    )
                    model_slug = _model_slug(model)
                    download_cmd = (
                        f'if [ -n "${{HF_TOKEN:-}}" ]; then '
                        f'huggingface-cli download {shlex.quote(model)} --local-dir {COSMOS_MODEL_DIR}/{model_slug} --token "$HF_TOKEN"; '
                        f"else huggingface-cli download {shlex.quote(model)} --local-dir {COSMOS_MODEL_DIR}/{model_slug}; fi"
                    )
                    ssh.run_or_raise(
                        docker_exec_cmd(COSMOS_CONTAINER_NAME, download_cmd),
                        stream=True,
                    )
                    ssh.run_or_raise(f"sudo docker restart {COSMOS_CONTAINER_NAME}")
                    if verify_env and not no_shared_creds:
                        failed_keys = audit_remote_env(
                            ssh,
                            "/etc/npa-cosmos-server/env",
                            shared_credential_env(credentials),
                        )
                        if failed_keys:
                            key = failed_keys[0]
                            fail_app(
                                f"Credential audit failed: {key} missing or mismatched in cosmos service env. "
                                "Deploy may have skipped shared credential injection."
                            )
                            return
                        _print_ngc_env_audit(
                            credentials=credentials,
                            service_env=service_env,
                            remote_path="/etc/npa-cosmos-server/env",
                        )
                except SSHError as exc:
                    fail_app(f"Cosmos container deployment failed: {exc}")
                    return
        else:
            step += 1
            console.print(
                f"  [{step}/{total_steps}] Installing Cosmos serving stack and downloading {model}..."
            )
            if dry_run:
                console.print(
                    "    [dry-run] Would create /opt/cosmos/venv, install Cosmos dependencies, and download model weights"
                )
            else:
                try:
                    ssh.run_or_raise(
                        _build_install_command(
                            model, server_port, no_guardrails=no_guardrails
                        ),
                        stream=True,
                    )
                except SSHError as exc:
                    fail_app(f"Cosmos installation failed: {exc}")
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
                    "ssh_fallback"
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
                    tool="cosmos",
                    version=COSMOS_VERSION,
                    deployed_by=f"npa deploy --runtime {runtime.value}",
                )
            except SSHError:
                pass
        mark_app_status(APP_STATUS_HEALTHY)
        if not dry_run:
            ensure_deploy_ingress(
                tool="cosmos",
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
    console.print(f"  Try: npa workbench cosmos -p {proj_alias} -n {wb_name} status")

    if output == OutputFormat.json:
        typer.echo(
            json.dumps(
                {
                    "project": proj_alias,
                    "name": wb_name,
                    "endpoint": endpoint,
                    "vm_ip": vm_ip,
                    "ssh_user": ssh_user,
                    "gpu_platform": gpu_type,
                    "gpu_preset": gpu_preset,
                    "runtime": runtime.value,
                    "model": model,
                    "backend": backend.value,
                    "tf_outputs": tf_outputs,
                },
                indent=2,
            )
        )


@app.command("teardown")
def teardown_cmd(
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation prompts.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be deleted without changing Nebius or local config.",
    ),
    output: OutputFormat = typer.Option(
        OutputFormat.text, "--output", help="Output format."
    ),
) -> None:
    """Delete a Cosmos serverless endpoint and remove its local alias."""
    cfg = _get_config()
    if not is_serverless_runtime(getattr(cfg, "runtime", "")):
        _fail("Cosmos teardown currently supports --runtime serverless aliases. Use deploy --destroy for VM aliases.")

    if not yes and not dry_run:
        _confirm_or_exit(f"Delete serverless endpoint for '{cfg.project}/{cfg.name}'?")

    try:
        result = _delete_serverless_endpoint_for_config(cfg, dry_run=dry_run)
    except ServerlessClientError as exc:
        _fail_serverless(exc, output)
        return

    if not dry_run:
        remove_workbench_config(cfg.project, cfg.name)

    _output(
        {
            **result,
            "project": cfg.project,
            "name": cfg.name,
            "runtime": "serverless",
            "config_removed": not dry_run,
        },
        output,
    )


@app.command("reload-env")
def reload_env_cmd(
    port: int = typer.Option(
        0, "--port", help="Cosmos HTTP server port. Defaults to the saved service port."
    ),
    restart: bool = typer.Option(
        True,
        "--restart/--no-restart",
        help="Restart Cosmos after updating the env file.",
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
    """Propagate local shared credentials into the running Cosmos service env without redeploying."""
    cfg = _get_config()
    credentials = resolve_credentials()
    credential_env = _shared_cosmos_env_or_fail(cfg, credentials)
    service_port = port or int(getattr(cfg, "service_port", 0) or 0) or 8080
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
                        f"systemctl restart {COSMOS_SERVICE}" if restart else "no restart",
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
                typer.echo(f"  systemctl restart {COSMOS_SERVICE}")
                typer.echo(f"  curl http://127.0.0.1:{service_port}/health")
            else:
                typer.echo("  no restart (--no-restart)")
            typer.echo("")
            typer.echo("No changes applied (--dry-run).")
        return

    result = _apply_env_update(
        cfg,
        credential_env,
        service_port=service_port,
        restart=restart,
        output=output,
    )
    _output({"status": "reloaded", **result}, output)


@app.command("serve")
def serve_cmd(
    model: str = typer.Option(
        DEFAULT_MODEL, "--model", help="Hugging Face Cosmos model ID to serve."
    ),
    backend: Backend = typer.Option(
        Backend.basic,
        "--backend",
        help=(
            "Serving backend: basic restarts the built-in FastAPI/Diffusers server; "
            "nim will run NVIDIA NIM; triton will run Triton/TensorRT serving."
        ),
    ),
    port: int = typer.Option(8080, "--port", help="Server port."),
    no_guardrails: bool = typer.Option(
        False,
        "--no-guardrails",
        help="Opt out of Cosmos safety guardrails for generated outputs.",
    ),
    output: OutputFormat = typer.Option(
        OutputFormat.text, "--output", help="Output format."
    ),
) -> None:
    """Start or pre-warm the saved Cosmos model server."""
    _ensure_basic_backend(backend)
    cfg = _get_config()

    if is_serverless_runtime(getattr(cfg, "runtime", "")):
        try:
            with service_endpoint(cfg, default_port=port, service_port=port) as active:
                data = HTTPClient(active.url, timeout=120.0, retries=1).health()
        except EndpointError as exc:
            _fail(f"Cosmos serverless endpoint setup failed: {exc}")
            return
        except ServerError as exc:
            _fail(f"Cosmos serverless pre-warm failed: {exc}")
            return
        _output(
            {
                "status": "prewarmed",
                "runtime": "serverless",
                "endpoint": active.url,
                "model": data.get("model") or model,
                "server": data.get("status", "up"),
            },
            output,
        )
        return

    if output != OutputFormat.json:
        console.print(f"[bold]Restarting Cosmos server[/bold]: {model}")

    out = ""
    err = ""
    if runtime_uses_container(getattr(cfg, "runtime", "vm")):
        try:
            with service_endpoint(cfg, default_port=port, service_port=port) as active:
                served = HTTPClient(active.url, timeout=120.0, retries=1).serve_model(
                    model, timeout=120.0
                )
            out = json.dumps(served)
        except EndpointError as exc:
            _fail(f"Cosmos serve endpoint setup failed: {exc}")
            return
        except ServerError as exc:
            _fail(f"Cosmos serve request failed: {exc}")
            return
    else:
        ssh = SSHClient(cfg.ssh)
        try:
            _, out, err = ssh.run_or_raise(
                _build_serve_command(model, port, no_guardrails=no_guardrails)
            )
        except SSHError as exc:
            _fail(f"SSH error: {exc}")
            return

    result: dict[str, Any] = {
        "status": "serving",
        "model": model,
        "port": port,
        "endpoint": cfg.endpoint,
        "guardrails": "off" if no_guardrails else "on",
    }
    if output == OutputFormat.json and out.strip():
        result["stdout_tail"] = out.strip()[-1000:]
    if err.strip():
        result["stderr_tail"] = err.strip()[-1000:]
    _output(result, output)


@app.command(
    "finetune",
    help="Roadmap placeholder for LoRA or full fine-tuning of Cosmos models on custom datasets.",
)
def finetune_cmd() -> None:
    """LoRA or full fine-tuning of Cosmos models on custom data."""
    typer.echo("not yet implemented")
    raise typer.Exit(1)


@app.command("train", help="Submit a Cosmos training job.")
def train_cmd(
    action: str = typer.Argument("submit", help="submit, status, or cancel."),
    job_id: str = typer.Argument("", help="Job ID or name for status/cancel."),
    runtime: WorkbenchRuntime = typer.Option(WorkbenchRuntime.vm, "--runtime", help="Application runtime. serverless creates a Nebius AI Job for Cosmos training."),
    project_id: str = typer.Option("", "--project-id", help="Nebius project ID for serverless Jobs."),
    image: str = typer.Option("", "--image", help="Container image for the training job."),
    gpu_type: str = typer.Option("gpu-h200-sxm", "--gpu-type", help="Nebius GPU platform."),
    gpu_count: int = typer.Option(1, "--gpu-count", help="GPU count."),
    gpu_preset: str = typer.Option("1gpu-16vcpu-200gb", "--gpu-preset", help="Nebius GPU preset."),
    subnet_id: str = typer.Option("", "--subnet-id", help="Nebius VPC subnet ID for serverless Jobs."),
    output_path: str = typer.Option("", "--output-path", "--output", help="S3 checkpoint output URI."),
    job_name: str = typer.Option("", "--job-name", help="Explicit serverless Job name."),
    smoke: bool = typer.Option(False, "--smoke", help="Run the minimal e2e smoke workload."),
    smoke_seconds: int = typer.Option(0, "--smoke-seconds", help="Seconds the smoke job should run."),
    require_hf: bool = typer.Option(False, "--require-hf", help="Require HF token inside the job."),
    submit_only: bool = typer.Option(False, "--submit-only", help="Submit and return before polling."),
    poll_interval: float = typer.Option(30.0, "--poll-interval", help="Seconds between status checks."),
    timeout: float = typer.Option(2400.0, "--timeout", help="Seconds to wait for completion."),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output-format", help="CLI output format."),
) -> None:
    if not is_serverless_runtime(runtime):
        _fail("Cosmos train currently supports --runtime serverless only.")
    proj_alias = _project_alias or default_project_name()
    wb_name = _workbench_name or default_workbench_name()
    env_cfg = resolve_environment(proj_alias)
    resolved_project_id = project_id or (env_cfg.project_id if env_cfg else "")
    if not resolved_project_id:
        _fail("Cosmos train --runtime serverless requires a Nebius project ID.")
    client = ServerlessClient()
    ref = job_id or job_name
    if action == "status":
        if not ref:
            _fail("Provide a job ID or name for train status.")
        info = client.get_job(ref, resolved_project_id)
        _output(
            _serverless_job_status_payload(
                client,
                info,
                platform=gpu_type,
                gpu_count=gpu_count,
            ),
            output,
        )
        return
    if action == "cancel":
        if not ref:
            _fail("Provide a job ID or name for train cancel.")
        info = client.cancel_job(ref, resolved_project_id)
        _output({"job_id": info.id, "job_name": info.name, "status": info.status}, output)
        return
    if action != "submit":
        _fail("Cosmos train action must be submit, status, or cancel.")
    if not smoke:
        _fail("Cosmos train requires --smoke until full Cosmos training is implemented.")
    try:
        resolved_gpu_type, resolved_gpu_preset, resolved_gpu_count = resolve_gpu_platform(
            gpu_type,
            gpu_count,
        )
    except ValueError as exc:
        _fail(str(exc))
    if gpu_preset and not (
        gpu_preset == "1gpu-16vcpu-200gb"
        and resolved_gpu_preset != gpu_preset
        and gpu_type.lower() not in {"h200", "h100", "gpu-h200-sxm", "gpu-h100-sxm"}
    ):
        resolved_gpu_preset = gpu_preset
    name = job_name or _serverless_job_name(proj_alias, wb_name)
    out = _serverless_train_output_path(proj_alias, name, output_path)
    try:
        existing = client.get_job(name, resolved_project_id)
    except EndpointNotFoundError:
        existing = None
    if existing is not None:
        info = existing if submit_only or existing.status in {"succeeded", "failed", "cancelled"} else client.poll_job(existing.id, resolved_project_id, interval_s=poll_interval, ceiling_s=timeout)
        _output({"status": "existing", "job_id": info.id, "job_name": info.name, "job_status": info.status, "output_path": out}, output)
        return
    try:
        subnet_id = resolve_subnet(
            project_id=resolved_project_id,
            explicit_subnet_id=subnet_id,
        )
    except SubnetResolutionError as exc:
        _fail(str(exc))
    env, extra_env = _serverless_job_env(proj_alias, require_hf=require_hf, output_path=out)
    env.update({"COSMOS_TRAIN_SMOKE": "1", "NPA_JOB_NAME": name})
    try:
        info = client.create_job(
            project_id=resolved_project_id,
            name=name,
            image=image or container_image_for_tool("cosmos", registry=resolve_container_registry(proj_alias)),
            command=_cosmos_train_smoke_command(smoke_seconds),
            gpu_type=resolved_gpu_type,
            gpu_count=resolved_gpu_count,
            preset=resolved_gpu_preset,
            subnet_id=subnet_id,
            output_path=out,
            env=env,
            extra_env=extra_env,
        )
        if not submit_only:
            info = client.poll_job(info.id, resolved_project_id, interval_s=poll_interval, ceiling_s=timeout)
    except ValueError as exc:
        _fail(str(exc))
    except ServerlessClientError as exc:
        _fail_serverless(exc, output)
    except TimeoutError as exc:
        _fail(str(exc))
    _output({"status": "submitted" if submit_only else info.status, "job_id": info.id, "job_name": info.name, "output_path": out}, output)


@app.command(
    "optimize",
    help="Roadmap placeholder for TensorRT compilation and quantization of Cosmos models.",
)
def optimize_cmd() -> None:
    """TensorRT compilation and quantization for Cosmos model serving."""
    typer.echo("not yet implemented")
    raise typer.Exit(1)


def _storage_client_for_config(
    cfg: Any,
    *,
    project: str | None = None,
    allow_host_creds: bool = False,
):
    from npa.clients.storage import StorageClient

    if project:
        return storage_client_for_project(project, allow_host_creds=allow_host_creds)
    return StorageClient.from_environment(
        endpoint_url=cfg.storage.endpoint_url,
        aws_access_key_id=cfg.storage.aws_access_key_id,
        aws_secret_access_key=cfg.storage.aws_secret_access_key,
    )


def _resolve_infer_input(
    input_path: str,
    cfg: Any,
    temp_dirs: list[tempfile.TemporaryDirectory[str]],
    *,
    source_project: str | None = None,
) -> Path | None:
    if not input_path:
        return None
    if not _is_s3_uri(input_path):
        return Path(input_path)

    tmp = tempfile.TemporaryDirectory(prefix="npa-cosmos-input-")
    temp_dirs.append(tmp)
    downloaded = Path(
        _storage_client_for_config(cfg, project=source_project).download_path(
            input_path, tmp.name
        )
    )
    if downloaded.is_file():
        return downloaded

    files = [path for path in downloaded.rglob("*") if path.is_file()]
    if len(files) != 1:
        _fail(f"S3 input path must resolve to exactly one file: {input_path}")
    return files[0]


def _build_infer_payload(prompt: str, input_path: Path | None) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if prompt:
        payload["prompt"] = prompt
    if input_path is not None:
        if not input_path.exists():
            _fail(f"Input file not found: {input_path}")
        payload["input"] = {
            "filename": input_path.name,
            "mime_type": mimetypes.guess_type(str(input_path))[0]
            or "application/octet-stream",
            "content_base64": base64.b64encode(input_path.read_bytes()).decode("ascii"),
        }
    if not payload:
        _fail("Provide --prompt, --input, or both for Cosmos inference.")
    return payload


def _write_inference_output(data: dict[str, Any], output_path: Path) -> None:
    for key in ("video_base64", "image_base64", "output_base64", "result_base64"):
        value = data.get(key)
        if isinstance(value, str) and value:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(base64.b64decode(value))
            return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2))


def _save_inference_output(
    data: dict[str, Any],
    output_path: str,
    cfg: Any,
    temp_dirs: list[tempfile.TemporaryDirectory[str]],
    *,
    allow_host_creds: bool = False,
    target_project: str | None = None,
) -> str:
    if not output_path:
        return ""

    if not _is_s3_uri(output_path):
        local_path = Path(output_path)
        _write_inference_output(data, local_path)
        return str(local_path)

    tmp = tempfile.TemporaryDirectory(prefix="npa-cosmos-output-")
    temp_dirs.append(tmp)
    local_path = Path(tmp.name) / _s3_path_name(output_path)
    _write_inference_output(data, local_path)

    def scoped_upload() -> str:
        saved_to = _storage_client_for_config(
            cfg,
            project=target_project,
            allow_host_creds=allow_host_creds,
        ).upload_file(str(local_path), output_path)
        data["upload_mode"] = "local"
        return saved_to

    def remote_upload() -> str:
        saved_to = _upload_local_file_via_remote_env(
            SSHClient(cfg.ssh),
            local_path,
            output_path,
            temp_dirs,
        )
        data["upload_mode"] = "remote"
        return saved_to

    def record_fallback(local_exc: BaseException) -> None:
        data["local_upload_error"] = str(local_exc)

    return run_with_host_credential_fallback(
        scoped_upload,
        remote_upload,
        bucket=bucket_from_s3_uri(output_path),
        operation="Cosmos infer output upload",
        allow_host_creds=allow_host_creds,
        logger=logger,
        on_fallback=record_fallback,
    )


def _upload_remote_file_via_env(
    ssh: SSHClient,
    remote_file: str,
    output_path: str,
    *,
    env_file: str = "/etc/npa-cosmos-server/env",
) -> str:
    parsed = urlparse(output_path)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    if parsed.scheme != "s3" or not bucket or not key:
        raise SSHError(f"Remote upload expects an s3:// output path: {output_path}")

    script = f"""\
set -euo pipefail
if [ ! -f {shlex.quote(env_file)} ]; then
  echo "missing env file: {shlex.quote(env_file)}" >&2
  exit 1
fi
set -a
. {shlex.quote(env_file)}
set +a
python3 - <<'PY'
import os

import boto3

local = {remote_file!r}
bucket = {bucket!r}
key = {key!r}
endpoint = os.environ.get("AWS_ENDPOINT_URL") or os.environ.get("NEBIUS_S3_ENDPOINT")
access_key = os.environ.get("AWS_ACCESS_KEY_ID")
secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
if not endpoint:
    raise RuntimeError("AWS_ENDPOINT_URL/NEBIUS_S3_ENDPOINT is not configured")
if not access_key or not secret_key:
    raise RuntimeError("AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY are not configured")
s3 = boto3.client(
    "s3",
    endpoint_url=endpoint,
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key,
)
s3.upload_file(local, bucket, key)
print("npa_remote_s3_upload_done")
PY
"""
    ssh.run_or_raise(f"sudo bash -lc {shlex.quote(script)}")
    return f"s3://{bucket}/{key}"


def _upload_local_file_via_remote_env(
    ssh: SSHClient,
    local_path: Path,
    output_path: str,
    temp_dirs: list[tempfile.TemporaryDirectory[str]],
) -> str:
    remote_file = f"/tmp/npa-cosmos-output-{uuid.uuid4().hex}-{local_path.name}"
    ssh.upload_file(str(local_path), remote_file)
    try:
        return _upload_remote_file_via_env(ssh, remote_file, output_path)
    finally:
        ssh.run(f"rm -f {shlex.quote(remote_file)}")


def _local_output_path(remote_path: str, output_path: str) -> Path:
    remote_name = Path(remote_path).name or f"cosmos-output-{uuid.uuid4().hex}"
    if not output_path:
        return Path(remote_name)

    local = Path(output_path)
    if output_path.endswith(("/", "\\")) or (local.exists() and local.is_dir()):
        return local / remote_name
    return local


def _download_remote_output(
    remote_path: str,
    output_path: str,
    cfg: Any,
    temp_dirs: list[tempfile.TemporaryDirectory[str]],
    *,
    result: dict[str, Any] | None = None,
    allow_host_creds: bool = False,
    target_project: str | None = None,
) -> str:
    if _is_s3_uri(output_path):
        tmp = tempfile.TemporaryDirectory(prefix="npa-cosmos-output-")
        temp_dirs.append(tmp)
        local_path = Path(tmp.name) / (
            Path(remote_path).name or f"cosmos-output-{uuid.uuid4().hex}"
        )
        ssh = SSHClient(cfg.ssh)
        ssh.download_file(remote_path, str(local_path))

        def scoped_upload() -> str:
            saved_to = _storage_client_for_config(
                cfg,
                project=target_project,
                allow_host_creds=allow_host_creds,
            ).upload_file(str(local_path), output_path)
            if result is not None:
                result["upload_mode"] = "local"
            return saved_to

        def remote_upload() -> str:
            saved_to = _upload_remote_file_via_env(ssh, remote_path, output_path)
            if result is not None:
                result["upload_mode"] = "remote"
            return saved_to

        def record_fallback(local_exc: BaseException) -> None:
            if result is not None:
                result["local_upload_error"] = str(local_exc)

        return run_with_host_credential_fallback(
            scoped_upload,
            remote_upload,
            bucket=bucket_from_s3_uri(output_path),
            operation="Cosmos infer output upload",
            allow_host_creds=allow_host_creds,
            logger=logger,
            on_fallback=record_fallback,
        )

    local_path = _local_output_path(remote_path, output_path)
    return SSHClient(cfg.ssh).download_file(remote_path, str(local_path))


def _poll_inference_job(
    client: HTTPClient,
    job_id: str,
    *,
    deadline: float,
    poll_interval: float,
    output_format: OutputFormat,
    quiet: bool = False,
    initial_status: str = "",
) -> dict[str, Any]:
    started_at = time.monotonic()
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _fail(f"Inference timed out waiting for job {job_id}")

        try:
            data = client.job_status(
                job_id, timeout=min(COSMOS_INFER_HTTP_TIMEOUT, max(1.0, remaining))
            )
        except ServerError as exc:
            _fail(f"Inference status check failed: {exc}")

        status = str(data.get("status", "")).lower()
        if data.get("error") and status not in {"completed"}:
            _fail(f"Inference job failed: {data.get('error')}")
        if output_format != OutputFormat.json and not quiet:
            elapsed = int(time.monotonic() - started_at)
            progress = ""
            if data.get("progress") is not None:
                progress = f" {data.get('progress')}%"
            elif data.get("percent") is not None:
                progress = f" {data.get('percent')}%"
            step = ""
            if data.get("step") is not None and data.get("total_steps") is not None:
                step = f" (step {data.get('step')}/{data.get('total_steps')})"
            typer.echo(
                f"[{elapsed}s] Generating...{progress} (status: {status or 'unknown'}){step}"
            )

        if status == "completed":
            return data
        if status in {"failed", "error"}:
            _fail(f"Inference job failed: {data.get('error', 'unknown error')}")
        if status not in {"running", "queued", "pending"}:
            _fail(
                f"Inference job {job_id} returned unknown status: {status or '<missing>'}"
            )

        sleep_for = min(poll_interval, max(0.0, deadline - time.monotonic()))
        if sleep_for > 0:
            time.sleep(sleep_for)


def _fail_upload(
    exc: BaseException,
    output_path: str,
    remote_output_path: str,
) -> None:
    lines = [
        f"Generation succeeded but uploading the output to {output_path} failed: {exc}"
    ]
    if remote_output_path:
        lines.append(
            f"The generated file is still on the Cosmos VM at {remote_output_path}; "
            "copy it off the VM before tearing the cluster down to avoid re-running "
            "generation."
        )
    lines.append(
        "This is an upload problem, not a generation failure. Verify the destination "
        "bucket exists and the Service Account has write access to it."
    )
    console.print("[red]Error:[/red] " + "\n".join(lines), soft_wrap=True)
    raise typer.Exit(1)


@app.command("infer")
def infer_cmd(
    prompt: str = typer.Option(
        "", "--prompt", help="Text prompt for text-to-world generation."
    ),
    input_path: str = typer.Option(
        "",
        "--input-path",
        "--input",
        help="S3 URI or local input image/video file for image/video-to-world generation.",
    ),
    output_path: str = typer.Option(
        "",
        "--output-path",
        "--output",
        help="S3 URI where the generated output file is saved.",
    ),
    source_project: str = typer.Option(
        "",
        "--source-project",
        help="Project alias whose scoped principal reads S3 inference inputs.",
    ),
    target_project: str = typer.Option(
        "",
        "--target-project",
        help="Project alias whose scoped principal writes S3 inference outputs.",
    ),
    timeout: float = typer.Option(
        1200.0,
        "--timeout",
        help="Wall-clock seconds for submit, poll, and output download.",
    ),
    poll_interval: float = typer.Option(
        COSMOS_INFER_POLL_INTERVAL,
        "--poll-interval",
        help="Seconds between Cosmos job status checks.",
    ),
    output_format: OutputFormat = typer.Option(
        OutputFormat.text, "--output-format", help="CLI output format."
    ),
    quiet: bool = typer.Option(
        False, "--quiet", help="Suppress progress output while polling."
    ),
    submit_only: bool = typer.Option(
        False,
        "--submit-only",
        help="Submit the inference job and return before polling for completion.",
    ),
    allow_host_creds: bool = typer.Option(
        False,
        "--allow-host-creds",
        help="Allow fallback to VM host credentials when scoped S3 upload credentials are denied.",
    ),
) -> None:
    """Submit a Cosmos inference job, poll until completion, then download the output."""
    try:
        output_path = validate_write_path(output_path, tool="Cosmos infer")
    except PathContractError as exc:
        _fail(str(exc))

    cfg = _get_config()
    resolved_source_project = source_project or None
    resolved_target_project = target_project or None
    temp_dirs: list[tempfile.TemporaryDirectory[str]] = []
    try:
        payload = _build_infer_payload(
            prompt,
            _resolve_infer_input(
                input_path,
                cfg,
                temp_dirs,
                source_project=resolved_source_project,
            ),
        )
        deadline = time.monotonic() + timeout

        try:
            with service_endpoint(cfg, default_port=8080) as active:
                client = HTTPClient(
                    active.url, timeout=COSMOS_INFER_HTTP_TIMEOUT, retries=1
                )
                generation_started = time.monotonic()
                submitted = client.infer(
                    payload, timeout=min(COSMOS_INFER_HTTP_TIMEOUT, max(1.0, timeout))
                )

                job_id = str(submitted.get("job_id") or "")
                if not job_id:
                    _fail(
                        f"Inference submit response did not include job_id: {submitted}"
                    )
                if output_format != OutputFormat.json:
                    typer.echo(f"  job_id: {job_id}")
                    typer.echo(f"  job_status: {submitted.get('status', 'unknown')}")
                if submit_only:
                    _output({**submitted, "job_id": job_id}, output_format)
                    return

                data = _poll_inference_job(
                    client,
                    job_id,
                    deadline=deadline,
                    poll_interval=poll_interval,
                    output_format=output_format,
                    quiet=quiet,
                    initial_status=str(submitted.get("status", "")),
                )
        except ServerError as exc:
            _fail(f"Inference submit failed: {exc}")
            return
        except EndpointError as exc:
            _fail(f"Inference endpoint setup failed: {exc}")
            return
        if output_format != OutputFormat.json and not quiet:
            typer.echo(
                f"Generation complete in {time.monotonic() - generation_started:.1f}s"
            )

        result = {**data, "job_id": job_id}
        remote_output_path = str(data.get("output_path") or "")
        try:
            if remote_output_path:
                downloaded_to = _download_remote_output(
                    remote_output_path,
                    output_path,
                    cfg,
                    temp_dirs,
                    result=result,
                    allow_host_creds=allow_host_creds,
                    target_project=resolved_target_project,
                )
                result["downloaded_to"] = downloaded_to
                if _is_s3_uri(downloaded_to):
                    result["saved_to"] = downloaded_to
            elif output_path:
                saved_to = _save_inference_output(
                    result,
                    output_path,
                    cfg,
                    temp_dirs,
                    allow_host_creds=allow_host_creds,
                    target_project=resolved_target_project,
                )
                if saved_to:
                    result["saved_to"] = saved_to
        except UPLOAD_FAILURE_ERRORS as exc:
            _fail_upload(exc, output_path, remote_output_path)
        _output(result, output_format)
    finally:
        for tmp in temp_dirs:
            tmp.cleanup()


@app.command("status")
def status_cmd(
    output: OutputFormat = typer.Option(
        OutputFormat.text, "--output", help="Output format."
    ),
) -> None:
    """Check the Cosmos endpoint health."""
    cfg = _get_config()

    if is_serverless_runtime(getattr(cfg, "runtime", "")):
        try:
            job_status = _serverless_job_status_for_config(cfg)
            if job_status is not None:
                _output(job_status, output)
                return
            info = _serverless_endpoint_status(cfg)
        except ServerlessClientError as exc:
            _fail_serverless(exc, output)
            return

        endpoint_url = info.url or cfg.endpoint
        data: dict[str, Any] = {}
        server = "unknown"
        health_error = ""
        if endpoint_url:
            try:
                data = HTTPClient(endpoint_url, timeout=10.0, retries=1).health()
                server = "up"
            except ServerError as exc:
                server = "down"
                health_error = str(exc)
        result = {
            "endpoint": endpoint_url,
            "app_status": "healthy" if server == "up" else "provisioned",
            "runtime": "serverless",
            "serverless_status": info.status.value,
            "server": server,
            "endpoint_id": info.id,
            "endpoint_name": info.name,
            **data,
        }
        if health_error:
            result["health_error"] = health_error
        _output(result, output)
        return

    try:
        with service_endpoint(cfg, default_port=8080) as active:
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
        _fail(f"Cannot prepare Cosmos endpoint for {cfg.endpoint}: {exc}")
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
        _fail(f"Cannot reach Cosmos endpoint at {cfg.endpoint}/health: {exc}")
        return

    loaded = bool(data.get("loaded", True))
    readiness = {
        "hf_token_present": bool(getattr(cfg, "hf_token", "")),
        "model_loaded": loaded,
        "ready": bool(getattr(cfg, "hf_token", "")) and loaded,
        "blockers": [],
    }
    if not readiness["hf_token_present"]:
        readiness["blockers"].append(
            "HF_TOKEN not configured - gated model downloads will fail"
        )
    if not loaded:
        readiness["blockers"].append(
            f"Model {data.get('model') or DEFAULT_MODEL} not loaded"
        )
    app_status = "healthy" if loaded else "degraded"

    result = {
        "endpoint": endpoint_url,
        "app_status": app_status,
        "runtime": getattr(cfg, "runtime", "vm"),
        "server": "up",
        **data,
        "readiness": readiness,
    }
    if not loaded:
        result["reason"] = "model not loaded"
    if runtime_uses_container(getattr(cfg, "runtime", "vm")):
        ssh = SSHClient(cfg.ssh)
        code, out, _ = ssh.run(
            "sudo docker inspect -f 'state={{.State.Status}} image={{.Config.Image}}' npa-cosmos 2>/dev/null || true"
        )
        if code == 0 and out.strip():
            result["container"] = out.strip()
    _output(result, output)


@app.command("system-info")
def system_info_cmd(
    output: OutputFormat = typer.Option(
        OutputFormat.text, "--output", help="Output format."
    ),
) -> None:
    """Collect and display system hardware information from the Cosmos VM."""
    cfg = _get_ssh_config()
    ssh = SSHClient(cfg.ssh)
    info_cmd = (
        "echo '=== nvidia-smi ===' && nvidia-smi && "
        "echo '' && echo '=== lscpu ===' && lscpu && "
        "echo '' && echo '=== free -h ===' && free -h && "
        "echo '' && echo '=== lsblk ===' && lsblk"
    )
    if runtime_uses_container(getattr(cfg, "runtime", "vm")):
        info_cmd += (
            " && echo '' && echo '=== container ===' && "
            "sudo docker inspect -f 'state={{.State.Status}} image={{.Config.Image}}' npa-cosmos"
        )

    try:
        _, out, err = ssh.run_or_raise(info_cmd)
    except SSHError as exc:
        _fail(f"SSH error: {exc}")
        return

    if output == OutputFormat.json:
        typer.echo(
            json.dumps(
                {
                    "host": cfg.ssh.host,
                    "runtime": getattr(cfg, "runtime", "vm"),
                    "system_info": out.strip(),
                },
                indent=2,
            )
        )
    else:
        if out:
            typer.echo(out.strip())
        if err:
            console.print(f"[red]stderr:[/red]\n{err.strip()[-500:]}")
