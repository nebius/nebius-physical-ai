"""npa workbench lerobot — LeRobot training, evaluation, serving, and inference."""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import typer
from rich.console import Console

from npa.cli._error_formatting import format_error_for_user
from npa.cli.path_contract import PathContractError, validate_read_path, validate_write_path
from npa.clients.config import (
    APP_STATUS_HEALTHY,
    APP_STATUS_INSTALL_FAILED,
    APP_STATUS_INSTALLING,
    APP_STATUS_PROVISIONED,
    ConfigError,
    default_project_name,
    default_workbench_name,
    list_projects,
    resolve_config,
    resolve_container_registry,
    resolve_credentials,
    resolve_environment,
    resolve_project_storage,
    resolve_terraform_state,
    update_workbench_app_status,
    update_workbench_serverless_job,
)
from npa.clients.credentials import apply_shared_credential_env, shared_credential_env
from npa.clients.endpoint import EndpointError, service_endpoint
from npa.clients.serverless import EndpointNotFoundError, JobInfo, ServerlessClient, ServerlessClientError
from npa.deploy.images import container_image_for_tool, supported_tool_version
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

app = typer.Typer(
    name="lerobot",
    help="LeRobot policy training, evaluation, serving, and inference.",
    no_args_is_help=True,
)

console = Console(stderr=True)

# Set by the Typer callback so every subcommand can read it.
_project_alias: str = ""
_workbench_name: str = ""


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


class WorkbenchRuntime(str, Enum):
    vm = "vm"
    container = "container"
    byovm = "byovm"
    serverless = "serverless"


def is_serverless_runtime(runtime: WorkbenchRuntime | str) -> bool:
    value = runtime.value if isinstance(runtime, WorkbenchRuntime) else runtime
    return value == WorkbenchRuntime.serverless.value


class OutputFormat(str, Enum):
    text = "text"
    json = "json"


@app.callback()
def main(
    project: str = typer.Option(
        "",
        "--project",
        "-p",
        help="Project alias (as configured in ~/.npa/config.yaml).",
    ),
    name: str = typer.Option(
        "",
        "--name",
        "-n",
        help="Workbench instance name within the project.",
    ),
) -> None:
    """LeRobot policy training, evaluation, serving, and inference."""
    global _project_alias, _workbench_name
    _project_alias = project
    _workbench_name = name


def _output(data: dict, fmt: OutputFormat) -> None:
    if fmt == OutputFormat.json:
        typer.echo(json.dumps(data, indent=2))
    else:
        for key, val in data.items():
            typer.echo(f"  {key}: {val}")


def _fail(msg: str, code: int = 1) -> None:
    console.print(f"[red]Error:[/red] {msg}")
    raise typer.Exit(code)


def _fail_serverless(exc: ServerlessClientError, output: OutputFormat = OutputFormat.text) -> None:
    typer.echo(format_error_for_user(exc, output_format=output.value), err=True)
    raise typer.Exit(1)


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


def _get_config(**overrides):
    try:
        return resolve_config(
            project=_project_alias or None,
            name=_workbench_name or None,
            **{k: v for k, v in overrides.items() if v is not None},
        )
    except ConfigError as exc:
        _fail(str(exc))


def _is_s3_uri(path: str) -> bool:
    return path.startswith("s3://")


def _path_name(path: str) -> str:
    if _is_s3_uri(path):
        parsed = urlparse(path)
        return parsed.path.rstrip("/").rsplit("/", 1)[-1] or parsed.netloc or "dataset"
    return Path(path.rstrip("/")).name or "dataset"


def _remote_cache_dir(kind: str, uri: str) -> str:
    parsed = urlparse(uri)
    cache_key = f"{parsed.netloc}_{parsed.path.strip('/').replace('/', '_')}"
    return f"/opt/lerobot/{kind}_cache/{cache_key}"


def _remote_python(script: str) -> str:
    return f"python3 -c {shlex.quote(script)}"


def _remote_bash(script: str) -> str:
    return f"bash -lc {shlex.quote(script)}"


def _runtime_exec_cmd(cfg: Any, command: str) -> str:
    if runtime_uses_container(getattr(cfg, "runtime", "vm")):
        return f"sudo docker exec npa-lerobot bash -lc {shlex.quote(command)}"
    return command


def _effective_gpu_count(cfg: Any, requested: int | None = None) -> tuple[int, str]:
    configured = int(getattr(cfg, "gpu_count", 0) or 0)
    count = requested if requested and requested > 0 else configured or 1
    visible = ",".join(str(i) for i in range(count))
    configured_visible = str(getattr(cfg, "cuda_visible_devices", "") or "")
    if not requested and configured_visible:
        visible = configured_visible
        count = len([item for item in configured_visible.split(",") if item.strip()])
    return count, visible


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


# ── list ─────────────────────────────────────────────────────────────────


def _is_lerobot_workbench(name: str, wb_cfg: dict) -> bool:
    """True when the workbench entry is a LeRobot VM, not a Genesis sim VM.

    Checks ``workbench_type`` first (authoritative when present), then
    falls back to name matching and the legacy endpoint heuristic for
    configs written before the type field existed.  Unprovisioned
    placeholders (no endpoint AND no SSH host) are excluded.
    """
    wtype = wb_cfg.get("workbench_type")
    if wtype:
        return wtype == "lerobot"
    if "genesis" in name and "lerobot" not in name:
        return False
    # Must have an endpoint or at least a reachable SSH host to be
    # considered provisioned.  Empty placeholders belong to neither list.
    if not wb_cfg.get("endpoint") and not wb_cfg.get("ssh", {}).get("host"):
        return False
    return bool(wb_cfg.get("endpoint"))


@app.command("list")
def list_cmd(
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """List configured LeRobot workbenches (excludes Genesis VMs)."""
    from npa.clients.config import default_project_name, default_workbench_name, list_projects

    projects = list_projects()
    def_proj = default_project_name()
    def_wb = default_workbench_name()

    if output == OutputFormat.json:
        # Filter to lerobot-only workbenches in JSON output too.
        filtered = {}
        for pname, pcfg in projects.items():
            wbs = {k: v for k, v in pcfg.get("workbenches", {}).items()
                   if _is_lerobot_workbench(k, v)}
            if wbs:
                filtered[pname] = {**pcfg, "workbenches": wbs}
        typer.echo(json.dumps({
            "projects": filtered,
            "default_project": def_proj,
            "default_workbench": def_wb,
        }, indent=2))
        return

    if not projects:
        typer.echo("No projects configured. Run 'npa workbench lerobot deploy' to create one.")
        return

    any_shown = False
    for proj_name, proj_cfg in projects.items():
        workbenches = {k: v for k, v in proj_cfg.get("workbenches", {}).items()
                       if _is_lerobot_workbench(k, v)}
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
            app_status = wb_cfg.get("app_status", "unknown")
            typer.echo(
                f"    {wb_name}{wb_marker}  gpu={gpu}  endpoint={endpoint}  "
                f"app_status={app_status}"
            )

    if not any_shown:
        typer.echo("No LeRobot workbenches configured. Run 'npa workbench lerobot deploy' to create one.")


# ── status ───────────────────────────────────────────────────────────────


@app.command()
def status(
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Check what's running on the VM."""
    cfg = _get_config()

    if is_serverless_runtime(getattr(cfg, "runtime", "")):
        job_cfg = getattr(cfg, "serverless_job", None)
        job_ref = str(getattr(job_cfg, "job_id", "") or getattr(job_cfg, "job_name", ""))
        project_id = str(getattr(job_cfg, "project_id", "") or getattr(cfg, "project_id", ""))
        if job_ref and project_id:
            client = ServerlessClient()
            try:
                info = client.get_job(job_ref, project_id)
            except ServerlessClientError as exc:
                _fail_serverless(exc, output)
            result = _serverless_job_status_payload(
                client,
                info,
                platform=str(getattr(job_cfg, "gpu_type", "")),
                gpu_count=int(getattr(job_cfg, "gpu_count", 0) or 0),
            )
            result.update({"runtime": "serverless", "workbench": getattr(cfg, "name", "")})
            _output(result, output)
            return

    from npa.clients.http import HTTPClient, ServerError

    try:
        with service_endpoint(cfg, default_port=8080) as active:
            client = HTTPClient(active.url)
            data = client.status()
            endpoint_url = active.url
    except EndpointError as exc:
        if output == OutputFormat.json:
            typer.echo(json.dumps({
                "endpoint": cfg.endpoint,
                "app_status": cfg.app_status or "unknown",
                "server": "down",
                "error": str(exc),
            }, indent=2))
        else:
            typer.echo(f"  endpoint: {cfg.endpoint}")
            typer.echo(f"  app_status: {cfg.app_status or 'unknown'}")
        _fail(f"Cannot prepare server endpoint for {cfg.endpoint}: {exc}")
        return
    except ServerError as exc:
        if output == OutputFormat.json:
            typer.echo(json.dumps({
                "endpoint": cfg.endpoint,
                "app_status": cfg.app_status or "unknown",
                "server": "down",
                "error": str(exc),
            }, indent=2))
        else:
            typer.echo(f"  endpoint: {cfg.endpoint}")
            typer.echo(f"  app_status: {cfg.app_status or 'unknown'}")
        _fail(f"Cannot reach server at {cfg.endpoint}: {exc}")
        return  # unreachable, keeps type checker happy
    container_info: dict[str, str] = {}
    if runtime_uses_container(getattr(cfg, "runtime", "vm")):
        from npa.clients.ssh import SSHClient

        ssh = SSHClient(cfg.ssh)
        code, out, _ = ssh.run(
            "sudo docker inspect -f 'state={{.State.Status}} image={{.Config.Image}}' npa-lerobot 2>/dev/null || true"
        )
        if code == 0 and out.strip():
            for part in out.strip().split():
                if "=" in part:
                    key, value = part.split("=", 1)
                    container_info[key] = value

    if output == OutputFormat.json:
        typer.echo(json.dumps({
            "app_status": cfg.app_status or "unknown",
            "runtime": getattr(cfg, "runtime", "vm"),
            "container": container_info,
            **data,
        }, indent=2))
    else:
        typer.echo(f"  endpoint: {endpoint_url}")
        typer.echo(f"  app_status: {cfg.app_status or 'unknown'}")
        typer.echo(f"  runtime: {getattr(cfg, 'runtime', 'vm')}")
        if container_info:
            typer.echo(f"  container: {container_info.get('state', 'unknown')} ({container_info.get('image', 'unknown')})")
        typer.echo(f"  server: up")
        ps = data.get("policy_server", {})
        if ps.get("running"):
            typer.echo(f"  policy_server: running (checkpoint: {ps.get('checkpoint', 'unknown')})")
        else:
            typer.echo(f"  policy_server: stopped")
        for job in data.get("jobs", []):
            typer.echo(f"  job: {job.get('name', '?')} [{job.get('status', '?')}]")
        if not data.get("jobs"):
            typer.echo(f"  jobs: none")


def _lerobot_serverless_job_name(wb_name: str, suffix: str | None = None) -> str:
    timestamp = suffix or datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    raw = f"npa-lerobot-{wb_name}-{timestamp}".lower()
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9-]+", "-", raw)).strip("-")[:63]


def _lerobot_serverless_train_output_path(bucket: str, wb_name: str, job_name: str) -> str:
    if not bucket:
        _fail("LeRobot train --runtime serverless requires storage.checkpoint_bucket.")
    normalized = bucket.rstrip("/")
    if not normalized.startswith("s3://"):
        normalized = f"s3://{normalized.lstrip('/')}"
    return f"{normalized}/lerobot/{wb_name}/{job_name}/"


def _lerobot_serverless_output_path(
    project: str,
    wb_name: str,
    job_name: str,
    output_path: str,
) -> str:
    if output_path:
        if not output_path.startswith("s3://"):
            _fail("LeRobot train --output-path expects an S3 URI for serverless jobs.")
        return output_path.rstrip("/") + "/"
    storage = resolve_project_storage(project)
    return _lerobot_serverless_train_output_path(storage.checkpoint_bucket, wb_name, job_name)


def _configured_lerobot_subnet(project_id: str, project: str = "", name: str = "") -> str:
    projects = list_projects()
    candidates: list[dict[str, Any]] = []
    if project and isinstance(projects.get(project), dict):
        candidates.append(projects[project])
    for cfg in projects.values():
        if isinstance(cfg, dict) and str(cfg.get("project_id") or "") == project_id:
            candidates.append(cfg)

    for project_cfg in candidates:
        workbenches = project_cfg.get("workbenches") if isinstance(project_cfg, dict) else {}
        wb_cfg = workbenches.get(name, {}) if isinstance(workbenches, dict) and name else {}
        sources = (
            wb_cfg.get("serverless_job", {}) if isinstance(wb_cfg, dict) else {},
            wb_cfg,
            project_cfg.get("serverless_job", {}),
            project_cfg,
        )
        for source in sources:
            if isinstance(source, dict):
                configured = source.get("subnet_id") or source.get("vpc_subnet_id") or source.get("subnet")
                if configured:
                    return str(configured)
    return ""


def _lerobot_serverless_train_subnet_id(project_id: str, project: str = "", name: str = "") -> str:
    configured = _configured_lerobot_subnet(project_id, project, name)
    if configured:
        return configured
    result = subprocess.run(
        ["nebius", "vpc", "subnet", "list", "--parent-id", project_id, "--format", "json"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        console.print(f"[yellow]Warning:[/yellow] Unable to discover Jobs subnet: {result.stderr.strip()}")
        return ""
    items = (json.loads(result.stdout or "{}").get("items") or [])
    ready = [
        item for item in items
        if str(((item.get("status") or {}).get("state") or "")).upper() in {"READY", ""}
    ]
    ranked = sorted(
        ready,
        key=lambda item: (
            "lerobot" not in str((item.get("metadata") or {}).get("name", "")).lower(),
            "default" not in str((item.get("metadata") or {}).get("name", "")).lower(),
        ),
    )
    subnet = str(((ranked[0].get("metadata") or {}).get("id") or "")) if ranked else ""
    if subnet and len(ready) > 1:
        console.print(f"[yellow]Warning:[/yellow] Using discovered Jobs subnet {subnet}. Pass --subnet-id to override.")
    return subnet


def _lerobot_serverless_job_env(
    hf_token: str,
    s3_access_key: str,
    s3_secret_key: str,
    output_path: str,
    *,
    s3_endpoint: str = "",
) -> dict[str, str]:
    env = {
        "NPA_OUTPUT_PATH": output_path,
        "PYTHONUNBUFFERED": "1",
        "HF_HOME": "/tmp/hf_home",
        "LEROBOT_HF_HOME": "/tmp/hf_home",
    }
    if hf_token:
        env["HF_TOKEN"] = hf_token
        env["HUGGING_FACE_HUB_TOKEN"] = hf_token
    if s3_access_key:
        env["AWS_ACCESS_KEY_ID"] = s3_access_key
        env["S3_ACCESS_KEY"] = s3_access_key
    if s3_secret_key:
        env["AWS_SECRET_ACCESS_KEY"] = s3_secret_key
        env["S3_SECRET_KEY"] = s3_secret_key
    if s3_endpoint:
        env["AWS_ENDPOINT_URL"] = s3_endpoint
        env["NEBIUS_S3_ENDPOINT"] = s3_endpoint
    return {key: value for key, value in env.items() if value}


def _s3_bucket_name(uri: str) -> str:
    if not uri:
        return ""
    normalized = uri if uri.startswith("s3://") else f"s3://{uri.lstrip('/')}"
    return urlparse(normalized).netloc


def _serverless_storage_env_values(storage: Any, credentials: Any, output_path: str) -> tuple[str, str, str]:
    storage_bucket = _s3_bucket_name(getattr(storage, "checkpoint_bucket", ""))
    output_bucket = _s3_bucket_name(output_path)
    use_credentials_storage = bool(
        output_bucket
        and storage_bucket
        and output_bucket != storage_bucket
        and getattr(credentials, "s3_access_key_id", "")
        and getattr(credentials, "s3_secret_access_key", "")
    )
    if use_credentials_storage:
        return (
            str(getattr(credentials, "s3_access_key_id", "")),
            str(getattr(credentials, "s3_secret_access_key", "")),
            str(getattr(credentials, "s3_endpoint", "") or getattr(storage, "endpoint_url", "")),
        )
    return (
        str(getattr(storage, "aws_access_key_id", "") or getattr(credentials, "s3_access_key_id", "")),
        str(getattr(storage, "aws_secret_access_key", "") or getattr(credentials, "s3_secret_access_key", "")),
        str(getattr(storage, "endpoint_url", "") or getattr(credentials, "s3_endpoint", "")),
    )


def _split_serverless_env(env: dict[str, str]) -> tuple[dict[str, str], dict[str, str]]:
    secret_names = {
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "S3_ACCESS_KEY",
        "S3_SECRET_KEY",
    }
    safe = {key: value for key, value in env.items() if key not in secret_names}
    extra = {key: value for key, value in env.items() if key in secret_names}
    return safe, extra


def _lerobot_gpu_platform(gpu_type: str) -> str:
    normalized = gpu_type.strip().lower()
    aliases = {
        "h200": "gpu-h200-sxm",
        "b300": "gpu-b300-sxm",
        "l40s": "gpu-l40s-a",
        "gpu-rtx-pro-6000": "gpu-rtx6000",
        "rtx-pro-6000": "gpu-rtx6000",
        "rtx6000": "gpu-rtx6000",
    }
    return aliases.get(normalized, gpu_type)


def _lerobot_serverless_gpu_preset(platform: str, gpu_count: int) -> str:
    presets = {
        "gpu-b300-sxm": {
            1: "1gpu-24vcpu-346gb",
            8: "8gpu-192vcpu-2768gb",
        },
        "gpu-l40s-a": {
            1: "1gpu-40vcpu-160gb",
        },
        "gpu-l40s-d": {
            1: "1gpu-48vcpu-288gb",
            2: "2gpu-96vcpu-576gb",
            4: "4gpu-192vcpu-1152gb",
        },
        "gpu-rtx6000": {
            1: "1gpu-24vcpu-218gb",
            8: "8gpu-192vcpu-1744gb",
        },
    }
    return presets.get(platform, {}).get(gpu_count, "")


def _warn_for_lerobot_gpu_policy(policy_type: str, gpu_type: str) -> None:
    normalized_policy = policy_type.strip().lower()
    normalized_gpu = gpu_type.strip().lower()
    if normalized_policy == "diffusion" and "b300" in normalized_gpu:
        console.print(
            "[yellow]Warning:[/yellow] B300 is ~2.5x slower than H200 on Diffusion Policy "
            "due to PTX JIT compilation. Consider --gpu-type h200 unless you specifically "
            "need B300 for memory or availability."
        )


def _default_lerobot_profile_script_path() -> Path:
    relative = Path("research/lerobot-deploy/training/profile_train.py")
    for parent in Path(__file__).resolve().parents:
        candidate = parent / relative
        if candidate.exists():
            return candidate
    return Path.cwd() / relative


def _serverless_upload_output_cmd(local_dir: str) -> str:
    script = f"""
import os
import pathlib
from urllib.parse import urlparse
import boto3

uri = os.environ["NPA_OUTPUT_PATH"]
parsed = urlparse(uri)
if parsed.scheme != "s3" or not parsed.netloc:
    raise SystemExit(f"NPA_OUTPUT_PATH must be s3:// URI, got: {{uri}}")
base = pathlib.Path({local_dir!r})
if not base.exists():
    raise SystemExit(f"output directory missing: {{base}}")
prefix = parsed.path.strip("/")
prefix_with_slash = prefix + "/" if prefix else ""
s3 = boto3.client("s3", endpoint_url=os.environ.get("AWS_ENDPOINT_URL") or os.environ.get("NEBIUS_S3_ENDPOINT") or None)
for file_path in base.rglob("*"):
    if file_path.is_file():
        s3.upload_file(str(file_path), parsed.netloc, prefix_with_slash + str(file_path.relative_to(base)))
print("NPA_LEROBOT_OUTPUT_UPLOADED", uri.rstrip("/") + "/", flush=True)
"""
    return _remote_python(script)


def _lerobot_train_container_command(
    policy_type: str,
    dataset: str,
    input_path: str,
    steps: int,
    batch_size: int,
    num_workers: int,
    *,
    env_type: str = "",
    env_task: str = "",
    device: str = "cuda",
    smoke: bool = False,
) -> str:
    effective_steps = 50 if smoke else steps
    effective_batch = 4 if smoke else batch_size
    output_dir = "/tmp/lerobot_output"
    dataset_setup_cmd = ""
    if input_path:
        if _is_s3_uri(input_path):
            resolved_dataset = f"/tmp/lerobot_dataset/{_path_name(input_path)}"
            dataset_setup_cmd = _remote_download_dir_cmd(input_path, resolved_dataset) + " && "
        else:
            resolved_dataset = input_path
        dataset_arg = (
            f"--dataset.repo_id={shlex.quote(_path_name(input_path))} "
            f"--dataset.root={shlex.quote(resolved_dataset)} "
        )
    else:
        dataset_arg = f"--dataset.repo_id={shlex.quote(dataset)} "
    env_type_arg = f"--env.type={shlex.quote(env_type)} " if env_type else ""
    env_task_arg = f"--env.task={shlex.quote(env_task)} " if env_task else ""
    num_workers_arg = f"--num_workers={num_workers} " if num_workers >= 0 else ""
    command = (
        "set -euo pipefail && "
        "cd /opt/lerobot && "
        "source /opt/lerobot/venv/bin/activate && "
        "if [ -f /opt/lerobot/.env ]; then set -a && source /opt/lerobot/.env && set +a; fi && "
        "mkdir -p /tmp/hf_home && "
        f"{dataset_setup_cmd}"
        f"lerobot-train "
        f"--policy.type={shlex.quote(policy_type)} "
        f"--policy.push_to_hub=false "
        f"{dataset_arg}"
        f"{env_type_arg}"
        f"{env_task_arg}"
        f"{num_workers_arg}"
        f"--output_dir={output_dir} "
        f"--steps={effective_steps} "
        f"--save_freq={effective_steps} "
        f"--eval_freq=1000000 "
        f"--batch_size={effective_batch} "
        f"--policy.device={shlex.quote(device)} "
        f"--eval.batch_size=1 "
        f"--eval.n_episodes=1 && "
        f"{_serverless_upload_output_cmd(output_dir)} && "
        "echo NPA_TRAIN_COMPLETE"
    )
    return _remote_bash(command)


def _lerobot_profile_train_container_command(
    *,
    script_path: Path,
    mode: str,
    policy_type: str,
    dataset_repo_id: str,
    steps: int,
    batch_size: int,
    num_workers: int,
    warmup_steps: int,
    output_path: str,
    compile_model: bool = False,
    skip_first: int = 10,
    warmup: int = 5,
    active: int = 50,
) -> str:
    """Build the container command for serverless LeRobot profile runs."""
    import base64
    import gzip

    script_path = script_path.expanduser().resolve()
    script_size = script_path.stat().st_size
    if script_size > 100_000:
        raise ValueError(
            f"Script {script_path} is {script_size} bytes; inline embed supports up to 100KB. "
            "Future: implement S3 transient upload for larger scripts."
        )
    if not output_path.startswith("s3://"):
        raise ValueError(f"--output-path must start with s3:// for serverless profile-train; got {output_path}")
    if num_workers < 0:
        raise ValueError(f"--num-workers must be >= 0 for profile-train, got {num_workers}")

    script_b64 = base64.b64encode(gzip.compress(script_path.read_bytes())).decode("ascii")
    compile_arg = " --compile" if compile_model else ""
    command = f"""
set -euo pipefail
cd /opt/lerobot
source /opt/lerobot/venv/bin/activate
if [ -f /opt/lerobot/.env ]; then set -a && source /opt/lerobot/.env && set +a; fi
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
mkdir -p /tmp/hf_home
export HF_HOME=/tmp/hf_home
export LEROBOT_HF_HOME=/tmp/hf_home

OUTPUT_DIR=/tmp/lerobot_profile_$(date +%s)_$$
mkdir -p "$OUTPUT_DIR"
export OUTPUT_DIR

NUM_WORKERS={num_workers}
if [ "$NUM_WORKERS" = "0" ]; then
    NUM_WORKERS=$(nproc)
fi

printf '%s' {shlex.quote(script_b64)} | base64 -d | gzip -dc > /tmp/profile_train.py
chmod +x /tmp/profile_train.py

python /tmp/profile_train.py \\
    --mode={shlex.quote(mode)} \\
    --policy_type={shlex.quote(policy_type)} \\
    --dataset_repo_id={shlex.quote(dataset_repo_id)} \\
    --steps={steps} \\
    --batch_size={batch_size} \\
    --num_workers="$NUM_WORKERS" \\
    --warmup_steps={warmup_steps} \\
    --skip_first={skip_first} \\
    --warmup={warmup} \\
    --active={active} \\
    --output_dir="$OUTPUT_DIR" \\
    --device=cuda{compile_arg}

python3 <<'PYUPLOAD'
import os
import pathlib
from urllib.parse import urlparse

import boto3

output_path = os.environ["NPA_OUTPUT_PATH"]
parsed = urlparse(output_path)
if parsed.scheme != "s3" or not parsed.netloc:
    raise SystemExit(f"NPA_OUTPUT_PATH must be s3:// URI, got: {{output_path}}")
prefix = parsed.path.strip("/")
prefix_with_slash = prefix + "/" if prefix else ""
output_dir = pathlib.Path(os.environ["OUTPUT_DIR"])
s3 = boto3.client("s3", endpoint_url=os.environ.get("AWS_ENDPOINT_URL") or os.environ.get("NEBIUS_S3_ENDPOINT") or None)
count = 0
for path in output_dir.rglob("*"):
    if path.is_file():
        key = prefix_with_slash + str(path.relative_to(output_dir))
        s3.upload_file(str(path), parsed.netloc, key)
        print(f"uploaded s3://{{parsed.netloc}}/{{key}}", flush=True)
        count += 1
print(f"NPA_PROFILE_UPLOAD_COMPLETE files={{count}}", flush=True)
PYUPLOAD

echo NPA_PROFILE_COMPLETE
"""
    wrapped = _remote_bash(command)
    if len(wrapped) > 32_000:
        raise ValueError(
            f"Embedded profile-train command is {len(wrapped)} characters after compression; "
            "Nebius Serverless currently accepts at most about 32768 characters. "
            "Use a smaller script or add transient object storage upload support."
        )
    return wrapped


def _train_serverless(
    *,
    proj_alias: str,
    wb_name: str,
    project_id: str,
    policy_type: str,
    dataset: str,
    input_path: str,
    job_name: str,
    steps: int,
    batch_size: int,
    num_workers: int,
    gpu_count: int,
    device: str,
    env_type: str,
    env_task: str,
    output_path: str,
    image: str,
    gpu_type: str,
    subnet_id: str,
    submit_only: bool,
    smoke: bool,
    poll_interval: float,
    wait_timeout: int,
    output: OutputFormat,
) -> None:
    if num_workers < -1:
        _fail(f"--num-workers must be -1 (omit) or >= 0, got {num_workers}")
    if gpu_count < 0:
        _fail(f"--gpu-count must be 0 (default 1 for serverless) or positive, got {gpu_count}")
    dataset_ref = input_path or dataset
    if not dataset_ref:
        _fail("Provide --dataset or --input-path.")
    resolved_project_id = project_id
    if not resolved_project_id:
        env_cfg = resolve_environment(proj_alias)
        resolved_project_id = env_cfg.project_id if env_cfg else ""
    if not resolved_project_id:
        _fail("LeRobot train --runtime serverless requires a Nebius project ID.")

    _warn_for_lerobot_gpu_policy(policy_type, gpu_type)
    name = job_name or _lerobot_serverless_job_name(wb_name)
    out = _lerobot_serverless_output_path(proj_alias, wb_name, name, output_path)
    client = ServerlessClient()
    try:
        existing = client.get_job(name, resolved_project_id)
    except EndpointNotFoundError:
        existing = None
    platform = _lerobot_gpu_platform(gpu_type)
    resolved_image = image or container_image_for_tool("lerobot", registry=resolve_container_registry(proj_alias))
    if existing is not None:
        info = existing
        if not submit_only and existing.status not in {"succeeded", "failed", "cancelled"}:
            info = client.poll_job(
                existing.id,
                resolved_project_id,
                interval_s=poll_interval,
                ceiling_s=wait_timeout,
            )
        update_workbench_serverless_job(
            proj_alias,
            wb_name,
            job_id=info.id,
            job_name=info.name,
            project_id=resolved_project_id,
            image=resolved_image,
            gpu_type=platform,
            gpu_count=gpu_count or 1,
            subnet_id=subnet_id,
            output_path=out,
            last_status=info.status,
            last_submitted_at=datetime.now(timezone.utc).isoformat(),
        )
        _output({"status": "existing", "job_id": info.id, "job_name": info.name, "job_status": info.status, "output_path": out}, output)
        return

    storage = resolve_project_storage(proj_alias)
    credentials = resolve_credentials()
    s3_access_key, s3_secret_key, s3_endpoint = _serverless_storage_env_values(storage, credentials, out)
    env = _lerobot_serverless_job_env(
        credentials.hf_token,
        s3_access_key,
        s3_secret_key,
        out,
        s3_endpoint=s3_endpoint,
    )
    env["NPA_JOB_NAME"] = name
    safe_env, extra_env = _split_serverless_env(env)
    submitted_at = datetime.now(timezone.utc).isoformat()
    subnet = subnet_id or _lerobot_serverless_train_subnet_id(resolved_project_id, proj_alias, wb_name)
    try:
        info = client.create_job(
            project_id=resolved_project_id,
            name=name,
            image=resolved_image,
            command=_lerobot_train_container_command(
                policy_type,
                dataset,
                input_path,
                steps,
                batch_size,
                num_workers,
                env_type=env_type,
                env_task=env_task,
                device=device,
                smoke=smoke,
            ),
            gpu_type=platform,
            gpu_count=gpu_count or 1,
            subnet_id=subnet,
            output_path=out,
            env=safe_env,
            extra_env=extra_env,
            preset=_lerobot_serverless_gpu_preset(platform, gpu_count or 1),
        )
        if not submit_only:
            info = client.poll_job(
                info.id,
                resolved_project_id,
                interval_s=poll_interval,
                ceiling_s=wait_timeout,
            )
    except ValueError as exc:
        _fail(str(exc))
    except ServerlessClientError as exc:
        _fail_serverless(exc, output)
    except TimeoutError as exc:
        _fail(str(exc))

    update_workbench_serverless_job(
        proj_alias,
        wb_name,
        job_id=info.id,
        job_name=info.name,
        project_id=resolved_project_id,
        image=resolved_image,
        gpu_type=platform,
        gpu_count=gpu_count or 1,
        subnet_id=subnet,
        output_path=out,
        last_status=info.status,
        last_submitted_at=submitted_at,
    )
    _output({"status": "submitted" if submit_only else info.status, "job_id": info.id, "job_name": info.name, "output_path": out}, output)


def _profile_train_serverless(
    *,
    proj_alias: str,
    wb_name: str,
    project_id: str,
    image: str,
    script: Path,
    mode: str,
    policy_type: str,
    dataset_repo_id: str,
    steps: int,
    batch_size: int,
    num_workers: int,
    warmup_steps: int,
    skip_first: int,
    warmup: int,
    active: int,
    compile_model: bool,
    gpu_type: str,
    gpu_count: int,
    subnet_id: str,
    job_name: str,
    output_path: str,
    submit_only: bool,
    poll_interval: float,
    wait_timeout: int,
    output: OutputFormat,
) -> None:
    if mode not in ("wallclock", "profiler", "inference"):
        _fail(f"Invalid --mode: {mode} (choose from: wallclock, profiler, inference)")
    if steps <= 0:
        _fail(f"--steps must be positive, got {steps}")
    if warmup_steps < 0:
        _fail(f"--warmup-steps must be >= 0, got {warmup_steps}")
    if steps <= warmup_steps and mode in {"wallclock", "inference"}:
        _fail(f"--steps ({steps}) must be greater than --warmup-steps ({warmup_steps})")
    if gpu_count < 1:
        _fail(f"--gpu-count must be positive for serverless profile-train, got {gpu_count}")
    if not dataset_repo_id:
        _fail("LeRobot profile-train --runtime serverless requires --dataset-repo-id.")
    if not policy_type:
        _fail("LeRobot profile-train --runtime serverless requires --policy-type.")
    if not output_path:
        _fail("LeRobot profile-train --runtime serverless requires --output-path.")
    if not output_path.startswith("s3://"):
        _fail("LeRobot profile-train --output-path expects an S3 URI for serverless jobs.")
    if not script.exists() or not script.is_file():
        _fail(f"LeRobot profile-train script not found: {script}")

    resolved_project_id = project_id
    if not resolved_project_id:
        env_cfg = resolve_environment(proj_alias)
        resolved_project_id = env_cfg.project_id if env_cfg else ""
    if not resolved_project_id:
        _fail("LeRobot profile-train --runtime serverless requires a Nebius project ID.")

    _warn_for_lerobot_gpu_policy(policy_type, gpu_type)
    name = job_name or _lerobot_serverless_job_name(wb_name)
    out = output_path.rstrip("/") + "/"
    client = ServerlessClient()
    platform = _lerobot_gpu_platform(gpu_type)
    resolved_image = image or container_image_for_tool("lerobot", registry=resolve_container_registry(proj_alias))

    try:
        existing = client.get_job(name, resolved_project_id)
    except EndpointNotFoundError:
        existing = None

    submitted_at = datetime.now(timezone.utc).isoformat()
    subnet = subnet_id or _lerobot_serverless_train_subnet_id(resolved_project_id, proj_alias, wb_name)

    if existing is not None:
        info = existing
        if not submit_only and existing.status not in {"succeeded", "failed", "cancelled"}:
            info = client.poll_job(
                existing.id,
                resolved_project_id,
                interval_s=poll_interval,
                ceiling_s=wait_timeout,
            )
        update_workbench_serverless_job(
            proj_alias,
            wb_name,
            job_id=info.id,
            job_name=info.name,
            project_id=resolved_project_id,
            image=resolved_image,
            gpu_type=platform,
            gpu_count=gpu_count,
            subnet_id=subnet,
            output_path=out,
            last_status=info.status,
            last_submitted_at=submitted_at,
        )
        _output({"status": "existing", "job_id": info.id, "job_name": info.name, "job_status": info.status, "output_path": out}, output)
        return

    storage = resolve_project_storage(proj_alias)
    credentials = resolve_credentials()
    s3_access_key, s3_secret_key, s3_endpoint = _serverless_storage_env_values(storage, credentials, out)
    env = _lerobot_serverless_job_env(
        credentials.hf_token,
        s3_access_key,
        s3_secret_key,
        out,
        s3_endpoint=s3_endpoint,
    )
    env["NPA_JOB_NAME"] = name
    safe_env, extra_env = _split_serverless_env(env)
    job_timeout = f"{max(1, int((wait_timeout + 3599) // 3600))}h"
    try:
        info = client.create_job(
            project_id=resolved_project_id,
            name=name,
            image=resolved_image,
            command=_lerobot_profile_train_container_command(
                script_path=script,
                mode=mode,
                policy_type=policy_type,
                dataset_repo_id=dataset_repo_id,
                steps=steps,
                batch_size=batch_size,
                num_workers=num_workers,
                warmup_steps=warmup_steps,
                output_path=out,
                compile_model=compile_model,
                skip_first=skip_first,
                warmup=warmup,
                active=active,
            ),
            gpu_type=platform,
            gpu_count=gpu_count,
            subnet_id=subnet,
            output_path=out,
            env=safe_env,
            extra_env=extra_env,
            timeout=job_timeout,
            preset=_lerobot_serverless_gpu_preset(platform, gpu_count),
        )
        if not submit_only:
            info = client.poll_job(
                info.id,
                resolved_project_id,
                interval_s=poll_interval,
                ceiling_s=wait_timeout,
            )
    except ValueError as exc:
        _fail(str(exc))
    except ServerlessClientError as exc:
        _fail_serverless(exc, output)
    except TimeoutError as exc:
        _fail(str(exc))

    update_workbench_serverless_job(
        proj_alias,
        wb_name,
        job_id=info.id,
        job_name=info.name,
        project_id=resolved_project_id,
        image=resolved_image,
        gpu_type=platform,
        gpu_count=gpu_count,
        subnet_id=subnet,
        output_path=out,
        last_status=info.status,
        last_submitted_at=submitted_at,
    )
    _output({"status": "submitted" if submit_only else info.status, "job_id": info.id, "job_name": info.name, "output_path": out}, output)


# ── train ────────────────────────────────────────────────────────────────


@app.command()
def train(
    policy_type: str = typer.Option(..., "--policy-type", help="Policy type (act, diffusion, smolvla)."),
    dataset: str = typer.Option("", "--dataset", help="HF dataset repo ID."),
    input_path: str = typer.Option(
        "",
        "--input-path",
        help="S3 URI for a LeRobotDataset. Overrides --dataset.",
    ),
    job_name: str = typer.Option(..., "--job-name", help="Unique name for this training run."),
    steps: int = typer.Option(5000, "--steps", help="Training steps."),
    batch_size: int = typer.Option(8, "--batch-size", help="Batch size."),
    env_type: str = typer.Option("", "--env-type", help="Environment type (omit to use lerobot default)."),
    env_task: str = typer.Option("", "--env-task", help="Environment task."),
    num_workers: int = typer.Option(-1, "--num-workers", help="Dataloader num_workers (-1 = omit, 0+ = explicit)."),
    gpu_count: int = typer.Option(0, "--gpu-count", help="Number of GPUs (0 = workbench config; uses accelerate launch for >1)."),
    device: str = typer.Option("cuda", "--device", help="Device."),
    output_path: str = typer.Option(
        "",
        "--output-path",
        help="S3 URI where the checkpoint output is written.",
    ),
    runtime: WorkbenchRuntime = typer.Option(WorkbenchRuntime.vm, "--runtime", help="Runtime backend: vm, container, byovm, or serverless."),
    project_id: str = typer.Option("", "--project-id", help="Nebius project ID override for serverless Jobs."),
    image: str = typer.Option("", "--image", help="Container image override for serverless Jobs."),
    gpu_type: str = typer.Option("h200", "--gpu-type", help="GPU type for serverless Jobs (h200, b300, l40s, or Nebius platform)."),
    subnet_id: str = typer.Option("", "--subnet-id", help="Subnet ID for serverless Jobs (auto-discovered if omitted)."),
    submit_only: bool = typer.Option(False, "--submit-only", help="Submit Job and return immediately without polling."),
    smoke: bool = typer.Option(False, "--smoke", help="Use smoke training settings for serverless Jobs."),
    poll_interval: float = typer.Option(30.0, "--poll-interval", help="Seconds between serverless Job status checks."),
    wait_timeout: int = typer.Option(3600, "--wait-timeout", help="Max seconds to wait for Job completion when not --submit-only."),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Run lerobot-train on the VM via SSH, stream logs."""
    try:
        if input_path:
            input_path = validate_read_path(
                input_path,
                tool="LeRobot train",
                option="--input-path",
                allow_hf=False,
            )
        output_path = validate_write_path(output_path, tool="LeRobot train")
    except PathContractError as exc:
        _fail(str(exc))
        return

    dataset_ref = input_path or dataset
    if not dataset_ref:
        _fail("Provide --dataset or --input-path.")
        return

    if is_serverless_runtime(runtime):
        _train_serverless(
            proj_alias=_project_alias or default_project_name(),
            wb_name=_workbench_name or default_workbench_name(),
            project_id=project_id,
            policy_type=policy_type,
            dataset=dataset,
            input_path=input_path,
            job_name=job_name,
            steps=steps,
            batch_size=batch_size,
            num_workers=num_workers,
            gpu_count=gpu_count,
            device=device,
            env_type=env_type,
            env_task=env_task,
            output_path=output_path,
            image=image,
            gpu_type=gpu_type,
            subnet_id=subnet_id,
            submit_only=submit_only,
            smoke=smoke,
            poll_interval=poll_interval,
            wait_timeout=wait_timeout,
            output=output,
        )
        return

    cfg = _get_config()

    from npa.clients.ssh import SSHClient, SSHError

    ssh = SSHClient(cfg.ssh)
    stream_logs = output != OutputFormat.json

    output_is_s3 = _is_s3_uri(output_path)
    checkpoint_dir = (
        output_path if output_path and not output_is_s3 else f"/opt/lerobot/checkpoints/{job_name}"
    )
    status_dir = "/opt/lerobot/job_status"

    if num_workers < -1:
        _fail(f"--num-workers must be -1 (omit) or >= 0, got {num_workers}")
        return

    dataset_setup_cmd = ""
    if input_path:
        if _is_s3_uri(input_path):
            resolved_dataset = _remote_cache_dir("dataset", input_path)
            dataset_setup_cmd = (
                _remote_download_dir_cmd(input_path, resolved_dataset, cfg.storage.endpoint_url)
                + " && "
            )
        else:
            resolved_dataset = input_path
        dataset_arg = (
            f"--dataset.repo_id={_path_name(input_path)} "
            f"--dataset.root={resolved_dataset} "
        )
    else:
        dataset_arg = f"--dataset.repo_id={dataset_ref} "

    env_type_arg = f"--env.type={env_type} " if env_type else ""
    env_task_arg = f"--env.task={env_task} " if env_task else ""
    num_workers_arg = f"--num_workers={num_workers} " if num_workers >= 0 else ""
    if gpu_count < 0:
        _fail(f"--gpu-count must be 0 (workbench config) or positive, got {gpu_count}")
        return

    effective_gpu_count, visible_devices = _effective_gpu_count(cfg, gpu_count)
    # Multi-GPU: use accelerate launch and restrict visible devices.
    if effective_gpu_count > 1:
        train_launcher = (
            f"accelerate launch --multi_gpu --num_processes={effective_gpu_count} "
            f"$(which lerobot-train)"
        )
    else:
        train_launcher = "lerobot-train"

    cmd = (
        f"source /opt/lerobot/venv/bin/activate && "
        f"set -a && source /opt/lerobot/.env && set +a && "
        f"export CUDA_VISIBLE_DEVICES={visible_devices} && "
        f"mkdir -p {status_dir} && "
        f"{dataset_setup_cmd}"
        f"START_TS=$(date +%s) && "
        f"{train_launcher} "
        f"--policy.type={policy_type} "
        f"--policy.push_to_hub=false "
        f"{dataset_arg}"
        f"{env_type_arg}"
        f"{env_task_arg}"
        f"{num_workers_arg}"
        f"--output_dir={checkpoint_dir} "
        f"--steps={steps} "
        f"--save_freq={steps} "
        f"--eval_freq=1000000 "
        f"--batch_size={batch_size} "
        f"--policy.device={device} "
        f"--eval.batch_size=1 "
        f"--eval.n_episodes=1 && "
        f"END_TS=$(date +%s) && "
        f"DURATION=$((END_TS - START_TS)) && "
        f"echo '{{\"status\": \"success\", \"job_name\": \"{job_name}\", "
        f"\"checkpoint_path\": \"{checkpoint_dir}/checkpoints/last/pretrained_model\", "
        f"\"duration_seconds\": '\"$DURATION\"'}}' > {status_dir}/{job_name}.json && "
        f"echo NPA_TRAIN_COMPLETE"
    )

    if stream_logs:
        console.print(f"[bold]Training {policy_type} on {dataset_ref}[/bold] (job: {job_name})")

    start = time.time()
    try:
        exit_code, stdout, stderr = ssh.run(_runtime_exec_cmd(cfg, cmd), stream=stream_logs)
    except SSHError as exc:
        _fail(str(exc))
        return

    duration = round(time.time() - start, 1)
    ckpt_path = f"{checkpoint_dir}/checkpoints/last/pretrained_model"

    result = {
        "status": "success" if exit_code == 0 else "failed",
        "exit_code": exit_code,
        "job_name": job_name,
        "checkpoint_path": ckpt_path if exit_code == 0 else None,
        "duration_seconds": duration,
    }

    if exit_code != 0:
        result["stderr"] = stderr.strip()[-500:] if stderr else ""

    if exit_code == 0 and output_path:
        if output_is_s3:
            if stream_logs:
                console.print(f"[bold]Uploading checkpoint to {output_path}...[/bold]")
            upload_cmd = (
                f"source /opt/lerobot/venv/bin/activate && "
                f"set -a && source /opt/lerobot/.env && set +a && "
                f"{_remote_upload_dir_cmd(ckpt_path, output_path, cfg.storage.endpoint_url)}"
            )
            up_code, up_out, up_err = ssh.run(_runtime_exec_cmd(cfg, upload_cmd))
            if up_code == 0:
                result["output_path"] = output_path.rstrip("/") + "/"
            else:
                result["status"] = "failed"
                result["exit_code"] = up_code or 1
                result["output_upload_error"] = up_err.strip()[-500:] if up_err else up_out.strip()[-500:]
                exit_code = up_code or 1
        else:
            result["output_path"] = ckpt_path

    # Upload checkpoint to object storage if configured
    if exit_code == 0 and not output_is_s3 and cfg.storage.checkpoint_bucket and cfg.storage.endpoint_url:
        if stream_logs:
            console.print("[bold]Uploading checkpoint to object storage...[/bold]")
        try:
            from urllib.parse import urlparse as _urlparse

            _parsed = _urlparse(cfg.storage.checkpoint_bucket)
            bucket_name = _parsed.netloc
            bucket_prefix = _parsed.path.lstrip("/").rstrip("/")
            s3_dest_prefix = f"{bucket_prefix}/{job_name}" if bucket_prefix else f"checkpoints/{job_name}"

            upload_cmd = (
                f"source /opt/lerobot/venv/bin/activate && "
                f"set -a && source /opt/lerobot/.env && set +a && "
                f"python3 -c \""
                f"import boto3, os, pathlib; "
                f"s3 = boto3.client('s3', "
                f"endpoint_url=os.environ.get('NEBIUS_S3_ENDPOINT', '{cfg.storage.endpoint_url}'), "
                f"aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID', ''), "
                f"aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY', '')); "
                f"ckpt = pathlib.Path('{ckpt_path}'); "
                f"[s3.upload_file(str(f), '{bucket_name}', '{s3_dest_prefix}/' + str(f.relative_to(ckpt))) for f in ckpt.rglob('*') if f.is_file()]; "
                f"print('uploaded')\""
            )
            _, up_out, _ = ssh.run(_runtime_exec_cmd(cfg, upload_cmd))
            if "uploaded" in up_out:
                result["storage_uri"] = f"s3://{bucket_name}/{s3_dest_prefix}/"
        except Exception:
            pass  # non-fatal; checkpoint is still on the VM

    _output(result, output)
    if exit_code != 0:
        raise typer.Exit(1)


# ── eval ─────────────────────────────────────────────────────────────────


@app.command("eval")
def eval_cmd(
    checkpoint: str = typer.Option("", "--checkpoint", help="Checkpoint path, HF repo, or s3:// URI."),
    input_path: str = typer.Option(
        "",
        "--input-path",
        help="S3 URI or Hugging Face Hub checkpoint ID. Overrides --checkpoint.",
    ),
    env: str = typer.Option(..., "--env", help="Environment type."),
    env_task: str = typer.Option("", "--env-task", help="Environment task."),
    episodes: int = typer.Option(10, "--episodes", help="Number of eval episodes."),
    output_path: str = typer.Option(
        "",
        "--output-path",
        help="S3 URI where eval results are written.",
    ),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Run lerobot-eval on the VM, return metrics."""
    try:
        if input_path:
            input_path = validate_read_path(
                input_path,
                tool="LeRobot eval",
                option="--input-path",
                allow_hf=True,
            )
        output_path = validate_write_path(output_path, tool="LeRobot eval")
    except PathContractError as exc:
        _fail(str(exc))
        return

    checkpoint_ref = input_path or checkpoint
    if not checkpoint_ref:
        _fail("Provide --checkpoint or --input-path.")
        return

    cfg = _get_config()

    from npa.clients.ssh import SSHClient, SSHError

    ssh = SSHClient(cfg.ssh)
    stream_logs = output != OutputFormat.json

    # Resolve checkpoint: if s3://, download on the VM first
    resolved_checkpoint = checkpoint_ref
    if _is_s3_uri(checkpoint_ref):
        local_cache = _remote_cache_dir("checkpoint", checkpoint_ref)
        resolve_cmd = (
            f"source /opt/lerobot/venv/bin/activate && "
            f"set -a && source /opt/lerobot/.env && set +a && "
            f"{_remote_download_dir_cmd(checkpoint_ref, local_cache, cfg.storage.endpoint_url)}"
        )
        try:
            ssh.run_or_raise(_runtime_exec_cmd(cfg, resolve_cmd))
            resolved_checkpoint = local_cache
        except Exception:
            pass  # try using the URI directly

    output_is_s3 = _is_s3_uri(output_path)
    eval_output_dir = (
        output_path if output_path and not output_is_s3 else f"/tmp/npa-eval-{int(time.time())}"
    )
    env_task_arg = f"--env.task={env_task}" if env_task else ""

    cmd = (
        f"source /opt/lerobot/venv/bin/activate && "
        f"set -a && source /opt/lerobot/.env && set +a && "
        f"lerobot-eval "
        f"--policy.path={resolved_checkpoint} "
        f"--env.type={env} "
        f"{env_task_arg} "
        f"--eval.n_episodes={episodes} "
        f"--eval.batch_size=1 "
        f"--output_dir={eval_output_dir} && "
        f"cat {eval_output_dir}/eval_info.json"
    )

    if stream_logs:
        console.print(f"[bold]Evaluating checkpoint[/bold]: {checkpoint_ref}")

    start = time.time()
    try:
        exit_code, stdout, stderr = ssh.run(_runtime_exec_cmd(cfg, cmd), stream=stream_logs)
    except SSHError as exc:
        _fail(str(exc))
        return

    duration = round(time.time() - start, 1)

    # Parse eval_info.json from the tail of stdout
    eval_metrics: dict = {}
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                eval_metrics = json.loads(line)
                break
            except json.JSONDecodeError:
                # Try collecting from this line to end
                json_start = stdout.rfind("{\n")
                if json_start >= 0:
                    try:
                        eval_metrics = json.loads(stdout[json_start:])
                    except json.JSONDecodeError:
                        pass
                break

    overall = eval_metrics.get("overall", {})
    result = {
        "status": "success" if exit_code == 0 else "failed",
        "exit_code": exit_code,
        "checkpoint": checkpoint_ref,
        "duration_seconds": duration,
        "pc_success": overall.get("pc_success"),
        "avg_sum_reward": overall.get("avg_sum_reward"),
        "n_episodes": overall.get("n_episodes"),
        "eval_seconds": overall.get("eval_s"),
    }

    if exit_code != 0:
        result["stderr"] = stderr.strip()[-500:] if stderr else ""

    if exit_code == 0 and output_path:
        if output_is_s3:
            if stream_logs:
                console.print(f"[bold]Uploading eval results to {output_path}...[/bold]")
            upload_cmd = (
                f"source /opt/lerobot/venv/bin/activate && "
                f"set -a && source /opt/lerobot/.env && set +a && "
                f"{_remote_upload_dir_cmd(eval_output_dir, output_path, cfg.storage.endpoint_url)}"
            )
            up_code, up_out, up_err = ssh.run(_runtime_exec_cmd(cfg, upload_cmd))
            if up_code == 0:
                result["output_path"] = output_path.rstrip("/") + "/"
            else:
                result["status"] = "failed"
                result["exit_code"] = up_code or 1
                result["output_upload_error"] = up_err.strip()[-500:] if up_err else up_out.strip()[-500:]
                exit_code = up_code or 1
        else:
            result["output_path"] = eval_output_dir

    _output(result, output)
    if exit_code != 0:
        raise typer.Exit(1)


# ── serve ────────────────────────────────────────────────────────────────


@app.command()
def serve(
    input_path: str = typer.Option(
        "",
        "--input-path",
        "-i",
        help="S3 URI or Hugging Face Hub checkpoint ID to serve.",
    ),
    # Deprecated path alias: keep --checkpoint working for existing scripts.
    checkpoint: str = typer.Option("", "--checkpoint", hidden=True),
    env_type: str = typer.Option("", "--env-type", help="Environment type (needed for shape resolution)."),
    env_task: str = typer.Option("", "--env-task", help="Environment task."),
    port: int = typer.Option(8080, "--port", help="Server port."),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Start or restart the PolicyServer with a given checkpoint."""
    checkpoint_ref = input_path or checkpoint
    if not checkpoint_ref:
        _fail("Provide --input-path.")
        return
    if input_path:
        try:
            input_path = validate_read_path(
                input_path,
                tool="LeRobot serve",
                option="--input-path",
                allow_hf=True,
            )
            checkpoint_ref = input_path
        except PathContractError as exc:
            _fail(str(exc))
            return

    cfg = _get_config()

    from npa.clients.http import HTTPClient, ServerError

    if output != OutputFormat.json:
        console.print(f"[bold]Loading checkpoint:[/bold] {checkpoint_ref}")

    try:
        with service_endpoint(cfg, default_port=port) as active:
            client = HTTPClient(active.url)
            data = client.serve(checkpoint_ref, env_type=env_type or None, env_task=env_task or None)

            # Wait for healthy
            if output != OutputFormat.json:
                console.print("Waiting for PolicyServer to be ready...")
            if not client.wait_healthy(timeout=120.0):
                _fail("PolicyServer did not become healthy within 120s")
                return
    except EndpointError as exc:
        _fail(f"PolicyServer endpoint setup failed: {exc}")
        return
    except ServerError as exc:
        _fail(f"Failed to start PolicyServer: {exc}")
        return

    result = {
        "status": "serving",
        "checkpoint": checkpoint_ref,
        **data,
    }
    _output(result, output)


# ── infer ────────────────────────────────────────────────────────────────


@app.command()
def infer(
    observation: Path = typer.Option(..., "--observation", help="Path to observation JSON file."),
    output_path: str = typer.Option(
        "",
        "--output-path",
        "-o",
        help="S3 URI where the inference response JSON is written.",
    ),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """POST an observation to the running PolicyServer, return predicted actions."""
    if not observation.exists():
        _fail(f"Observation file not found: {observation}")
        return
    try:
        output_path = validate_write_path(output_path, tool="LeRobot infer")
    except PathContractError as exc:
        _fail(str(exc))
        return

    obs_data = json.loads(observation.read_text())

    cfg = _get_config()

    from npa.clients.http import HTTPClient, ServerError

    try:
        with service_endpoint(cfg, default_port=8080) as active:
            client = HTTPClient(active.url)
            data = client.infer(obs_data)
    except EndpointError as exc:
        _fail(f"Inference endpoint setup failed: {exc}")
        return
    except ServerError as exc:
        _fail(f"Inference failed: {exc}")
        return

    if output_path:
        if _is_s3_uri(output_path):
            from npa.clients.storage import StorageClient

            with tempfile.TemporaryDirectory(prefix="npa-lerobot-infer-") as tmp:
                local_file = Path(tmp) / "infer-response.json"
                local_file.write_text(json.dumps(data, indent=2))
                saved_to = StorageClient.from_environment(
                    endpoint_url=cfg.storage.endpoint_url,
                    aws_access_key_id=cfg.storage.aws_access_key_id,
                    aws_secret_access_key=cfg.storage.aws_secret_access_key,
                ).upload_file(str(local_file), output_path)
        else:
            local_path = Path(output_path)
            if output_path.endswith("/") or (local_path.exists() and local_path.is_dir()):
                local_path = local_path / "infer-response.json"
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_text(json.dumps(data, indent=2))
            saved_to = str(local_path)
        data = {"status": "success", "output_path": saved_to, **data}

    _output(data, output)


# ── list-checkpoints ─────────────────────────────────────────────────────


@app.command("list-checkpoints")
def list_checkpoints(
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """List available checkpoints on the VM and in object storage."""
    cfg = _get_config()

    from npa.clients.ssh import SSHClient, SSHError

    results: dict = {"vm_checkpoints": [], "storage_checkpoints": []}

    # List local checkpoints on VM
    ssh = SSHClient(cfg.ssh)
    try:
        _, out, _ = ssh.run(
            _runtime_exec_cmd(
                cfg,
                "find /opt/lerobot/checkpoints -maxdepth 4 -name 'pretrained_model' -type d 2>/dev/null "
                "| sort || true",
            )
        )
        for line in out.strip().splitlines():
            line = line.strip()
            if line:
                # Extract job name from path
                parts = line.split("/")
                name_idx = parts.index("checkpoints") + 1 if "checkpoints" in parts else -1
                name = parts[name_idx] if 0 <= name_idx < len(parts) else line
                results["vm_checkpoints"].append({"name": name, "path": line})
    except SSHError:
        results["vm_checkpoints_error"] = "Could not connect to VM"

    # List storage checkpoints
    if cfg.storage.checkpoint_bucket and cfg.storage.endpoint_url:
        try:
            from npa.clients.storage import StorageClient

            store = StorageClient(
                endpoint_url=cfg.storage.endpoint_url,
                aws_access_key_id=cfg.storage.aws_access_key_id,
                aws_secret_access_key=cfg.storage.aws_secret_access_key,
            )
            results["storage_checkpoints"] = store.list_checkpoints(
                cfg.storage.checkpoint_bucket
            )
        except Exception as exc:
            results["storage_checkpoints_error"] = str(exc)

    if output == OutputFormat.json:
        typer.echo(json.dumps(results, indent=2))
    else:
        typer.echo("VM checkpoints:")
        if results["vm_checkpoints"]:
            for ckpt in results["vm_checkpoints"]:
                typer.echo(f"  {ckpt['name']}  →  {ckpt['path']}")
        else:
            typer.echo("  (none)")
        typer.echo("Storage checkpoints:")
        if results["storage_checkpoints"]:
            for ckpt in results["storage_checkpoints"]:
                typer.echo(f"  {ckpt['name']}  →  {ckpt['uri']}")
        else:
            typer.echo("  (none)")


# ── deploy ───────────────────────────────────────────────────────────────


@app.command()
def deploy(
    gpu_type: str = typer.Option("gpu-h200-sxm", "--gpu-type", help="Nebius GPU platform."),
    gpu_preset: str = typer.Option("1gpu-16vcpu-200gb", "--gpu-preset", help="GPU preset."),
    region: str = typer.Option("", "--region", help="Nebius region (saved per project)."),
    project_id: str = typer.Option("", "--project-id", help="Nebius project ID (saved per project)."),
    tenant_id: str = typer.Option("", "--tenant-id", help="Nebius tenant ID (saved per project)."),
    tf_dir: str = typer.Option("", "--tf-dir", help="Path to Terraform directory (default: bundled)."),
    tf_var: list[str] = typer.Option([], "--tf-var", "-v", help="Extra TF variable (key=value), repeatable."),
    skip_infra: bool = typer.Option(False, "--skip-infra", help="Skip Terraform, only redeploy app."),
    skip_app: bool = typer.Option(False, "--skip-app", help="Skip app deployment, only provision infra."),
    destroy: bool = typer.Option(False, "--destroy", help="Destroy infrastructure and clean up config."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would happen without doing it."),
    no_shared_creds: bool = typer.Option(False, "--no-shared-creds", help="Do not inject ~/.npa/credentials.yaml shared credentials into the service env."),
    checkpoint: str = typer.Option("", "--checkpoint", help="Pre-load a checkpoint after deploy."),
    server_port: int = typer.Option(8080, "--server-port", help="Server port on the VM."),
    runtime: WorkbenchRuntime = typer.Option(WorkbenchRuntime.vm, "--runtime", help=RUNTIME_HELP),
    host: str = typer.Option("", "--host", help="BYOVM SSH host/IP. Used only with --runtime byovm."),
    ssh_key: str = typer.Option("", "--ssh-key", help="BYOVM SSH private key path. Used only with --runtime byovm."),
    ssh_user: str = typer.Option("", "--ssh-user", help="BYOVM SSH username. Defaults to ubuntu."),
    gpu_count: int = typer.Option(0, "--gpu-count", help="Limit visible GPUs on BYOVM (0 = all detected)."),
    disk_size: int | None = typer.Option(None, "--disk-size", help="Boot disk size in GiB. Defaults to 250 for container runtime; VM runtime keeps the Terraform default."),
    preemptible: bool = typer.Option(True, "--preemptible/--no-preemptible", help="Preemptible (spot) instance. Pass --no-preemptible for regular VMs."),
    default: bool = typer.Option(False, "--default", help="Set this workbench as the default."),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Deploy or update LeRobot infrastructure and application.

    On first deploy, pass --project-id and --tenant-id.  These are saved in
    the workbench config and reused automatically on subsequent deploys.
    Auth is handled via the ``nebius`` CLI (must be logged in).
    """
    from npa.deploy.provisioner import ProvisionerError, apply_boot_disk_tf_vars

    proj_alias = _project_alias or None
    wb_name = _workbench_name or "b200"
    byovm = is_byovm_runtime(runtime)
    use_remote_state = not tf_dir and not byovm
    lerobot_version = supported_tool_version("lerobot")
    if byovm:
        skip_infra = True

    # Parse extra TF vars
    extra_vars: dict[str, str] = {}
    for item in tf_var:
        if "=" not in item:
            _fail(f"Invalid --tf-var format: {item} (expected key=value)")
        k, v = item.split("=", 1)
        extra_vars[k] = v

    # ── Resolve environment from project config ──────────────────────
    from npa.clients.config import resolve_environment

    saved_env = resolve_environment(
        proj_alias,
        project_id=project_id or None,
        tenant_id=tenant_id or None,
        region=region or None,
    )

    env_project = project_id or (saved_env.project_id if saved_env else "")
    env_tenant = tenant_id or (saved_env.tenant_id if saved_env else "")
    env_region = region or (saved_env.region if saved_env else "")

    # Derive a project alias if the user didn't provide one.
    if not proj_alias:
        proj_alias = env_region or ("byovm" if byovm else "default")

    container_registry = resolve_container_registry(proj_alias)
    container_image = container_image_for_tool(
        "lerobot",
        registry=container_registry,
        tag=lerobot_version,
    )
    cloud_init_workbench_type = (
        "lerobot-container"
        if runtime_uses_container(runtime)
        else "lerobot"
    )

    # ── Bootstrap Nebius environment ─────────────────────────────────
    nebius_creds: dict[str, str] = {}

    if use_remote_state and not skip_infra:
        if not env_project or not env_tenant or not env_region:
            _fail(
                "First deploy requires --project-id, --tenant-id, and --region.\n"
                "  Example: npa workbench lerobot -p me-west1 -n b200 deploy \\\n"
                "    --project-id project-... --tenant-id tenant-... \\\n"
                "    --region me-west1 --gpu-type gpu-b200-sxm"
            )
            return

        if dry_run:
            console.print(f"  [dry-run] Would bootstrap Nebius environment:")
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

            # Persist project early so retries don't need IDs.
            from npa.clients.config import write_config as _early_write

            _early_write({
                "projects": {
                    proj_alias: {
                        "project_id": env_project,
                        "tenant_id": env_tenant,
                        "region": env_region,
                    },
                },
            })

    # Merge bootstrapped credentials into TF vars.
    merged_vars: dict[str, str] = {**extra_vars}
    for key in (
        "iam_token", "service_account_id",
        "nebius_api_key", "nebius_secret_key",
        "s3_bucket", "s3_endpoint",
        "nebius_project_id", "nebius_region",
    ):
        if key in nebius_creds and key not in merged_vars:
            merged_vars[key] = nebius_creds[key]
    merged_vars.setdefault("lerobot_version", lerobot_version)
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
            apply_boot_disk_tf_vars(merged_vars, runtime, disk_size)
        except ValueError as exc:
            _fail(str(exc))
            return

    if use_remote_state and nebius_creds and not dry_run:
        from npa.clients.config import write_config as _state_write

        _state_write({
            "projects": {
                proj_alias: {
                    "terraform_state": _terraform_state_config(merged_vars),
                },
            },
        })

    total_steps = _deploy_step_count(skip_infra, skip_app, destroy, checkpoint)
    step = 0

    # ── Destroy flow ─────────────────────────────────────────────────
    if destroy:
        if byovm:
            step += 1
            console.print(f"  [{step}/{total_steps}] Unregistering BYOVM workbench {proj_alias}/{wb_name}...")
            if not dry_run:
                from npa.clients.config import remove_workbench_config

                remove_workbench_config(proj_alias, wb_name)
            console.print(f"  {proj_alias}/{wb_name} unregistered. BYOVM host was not modified.")
            return

        step += 1
        console.print(f"  [{step}/{total_steps}] Destroying {proj_alias}/{wb_name}...")
        if dry_run:
            console.print("    [dry-run] Would run: terraform destroy")
            return

        from npa.deploy import provisioner

        if use_remote_state:
            s3_bucket = merged_vars.get("s3_bucket", "")
            s3_endpoint = merged_vars.get("s3_endpoint", f"https://storage.{env_region}.nebius.cloud")
            resolved_tf_dir = str(provisioner.prepare_working_dir(
                proj_alias, wb_name,
                bucket=s3_bucket, region=env_region, endpoint=s3_endpoint,
            ))
            try:
                provisioner.init(tf_dir=resolved_tf_dir, backend_config={
                    "access_key": merged_vars.get("nebius_api_key", ""),
                    "secret_key": merged_vars.get("nebius_secret_key", ""),
                })
            except ProvisionerError as exc:
                _fail(f"Terraform init failed: {exc}")
                return
        else:
            resolved_tf_dir = tf_dir

        # Read the stored tf_instance_name if available so the destroy
        # matches the name used during the original apply.
        destroy_instance_name = f"lerobot-{proj_alias}-{wb_name}"
        try:
            wb_cfg = resolve_config(project=proj_alias, name=wb_name)
            if wb_cfg.tf_instance_name:
                destroy_instance_name = wb_cfg.tf_instance_name
        except ConfigError:
            pass

        try:
            provisioner.destroy(
                tf_dir=resolved_tf_dir or None,
                tf_vars={"gpu_platform": gpu_type, "gpu_preset": gpu_preset,
                         "instance_name": destroy_instance_name,
                         "workbench_type": cloud_init_workbench_type,
                         "enable_preemptible": "true" if preemptible else "false",
                         **merged_vars},
            )
        except ProvisionerError as exc:
            _fail(f"Terraform destroy failed: {exc}")
            return

        step += 1
        console.print(f"  [{step}/{total_steps}] Cleaning up config...")
        from npa.clients.config import remove_workbench_config
        remove_workbench_config(proj_alias, wb_name)
        if use_remote_state:
            provisioner.cleanup_working_dir(proj_alias, wb_name)
        console.print(f"  {proj_alias}/{wb_name} destroyed.")
        return

    # ── Phase 1: Infrastructure ──────────────────────────────────────
    tf_outputs: dict = {}
    byovm_gpu_info = None
    byovm_effective_gpu_count = 0
    byovm_visible_devices = ""

    if not skip_infra:
        from npa.deploy import provisioner

        if use_remote_state:
            s3_bucket = merged_vars.get("s3_bucket", "")
            s3_endpoint = merged_vars.get("s3_endpoint", f"https://storage.{env_region}.nebius.cloud")
            resolved_tf_dir = str(provisioner.prepare_working_dir(
                proj_alias, wb_name,
                bucket=s3_bucket, region=env_region, endpoint=s3_endpoint,
            ))
        else:
            resolved_tf_dir = tf_dir

        step += 1
        console.print(f"  [{step}/{total_steps}] Initializing Terraform ({proj_alias}/{wb_name})...")
        if dry_run:
            console.print(f"    [dry-run] Would run: terraform init")
        else:
            try:
                backend_cfg = (
                    {"access_key": merged_vars.get("nebius_api_key", ""),
                     "secret_key": merged_vars.get("nebius_secret_key", "")}
                    if use_remote_state else None
                )
                provisioner.init(tf_dir=resolved_tf_dir or None, backend_config=backend_cfg)
            except ProvisionerError as exc:
                _fail(f"Terraform init failed: {exc}")
                return

        step += 1
        apply_instance_name = f"lerobot-{proj_alias}-{wb_name}"
        all_vars = {
            "gpu_platform": gpu_type, "gpu_preset": gpu_preset,
            "instance_name": apply_instance_name,
            "workbench_type": cloud_init_workbench_type,
            "enable_preemptible": "true" if preemptible else "false",
            **merged_vars,
        }
        console.print(f"  [{step}/{total_steps}] Applying Terraform (gpu={gpu_type}, region={env_region})...")
        if dry_run:
            console.print(f"    [dry-run] terraform apply")
            tf_outputs = {"vm_ip": "<pending>", "ssh_user": "ubuntu",
                          "ssh_key_path": "~/.ssh/id_ed25519",
                          "storage_bucket": "<pending>",
                          "storage_endpoint": f"https://storage.{env_region}.nebius.cloud"}
        else:
            try:
                tf_outputs = provisioner.apply(tf_dir=resolved_tf_dir or None, tf_vars=all_vars)
            except ProvisionerError as exc:
                _fail(f"Terraform apply failed: {exc}")
                return
        console.print(f"    VM IP: {tf_outputs.get('vm_ip', 'unknown')}")
    else:
        step += 1
        console.print(
            f"  [{step}/{total_steps}] "
            + ("Using BYOVM target..." if byovm else "Skipping infra, reading existing config...")
        )
        resolved_tf_dir = tf_dir
        if byovm:
            try:
                from npa.clients.config import resolve_credentials
                from npa.clients.ssh import SSHClient, SSHError

                target = resolve_byovm_target(host=host, ssh_key=ssh_key, ssh_user=ssh_user)
                bucket = (
                    merged_vars.get("s3_bucket", "")
                    or os.environ.get("NPA_CHECKPOINT_BUCKET", "")
                )
                storage_ep = (
                    merged_vars.get("s3_endpoint", "")
                    or os.environ.get("AWS_ENDPOINT_URL", "")
                )
                tf_outputs = workbench_storage_outputs(
                    target=target,
                    bucket=bucket,
                    endpoint=storage_ep,
                )
                if not dry_run:
                    ssh = SSHClient(ssh_config_for_target(target, tokens=resolve_credentials().tokens))
                    ssh.run_or_raise("echo connected")
                    byovm_gpu_info = detect_gpu_info(ssh)
                    byovm_effective_gpu_count, byovm_visible_devices = select_visible_devices(
                        byovm_gpu_info.count,
                        gpu_count or None,
                    )
                    console.print(
                        f"    Detected {byovm_gpu_info.count} GPU(s): "
                        f"{', '.join(byovm_gpu_info.names)}"
                    )
                    console.print(f"    CUDA_VISIBLE_DEVICES={byovm_visible_devices}")
                else:
                    byovm_effective_gpu_count = gpu_count or 0
            except (ValueError, SSHError) as exc:
                _fail(str(exc))
                return
        elif resolved_tf_dir:
            from npa.deploy import provisioner
            try:
                tf_outputs = provisioner.outputs(tf_dir=resolved_tf_dir)
            except ProvisionerError:
                pass
        elif use_remote_state:
            from npa.deploy import provisioner
            work_dir = provisioner.working_dir_path(proj_alias, wb_name)
            if work_dir.exists():
                try:
                    provisioner.init(tf_dir=str(work_dir), backend_config={
                        "access_key": merged_vars.get("nebius_api_key", ""),
                        "secret_key": merged_vars.get("nebius_secret_key", ""),
                    })
                    tf_outputs = provisioner.outputs(tf_dir=str(work_dir))
                except ProvisionerError:
                    pass

        if not tf_outputs:
            from npa.clients.config import _load_yaml, _deep_get, _resolve_project_section, _resolve_workbench_in_project
            yml = _load_yaml()
            proj = _resolve_project_section(yml, proj_alias)
            wb = _resolve_workbench_in_project(proj, wb_name, yml)
            tf_outputs = {
                "vm_ip": _deep_get(wb, "ssh", "host", default=""),
                "ssh_user": _deep_get(wb, "ssh", "user", default="ubuntu"),
                "ssh_key_path": _deep_get(wb, "ssh", "key_path", default="~/.ssh/id_ed25519"),
                "storage_bucket": _deep_get(wb, "storage", "checkpoint_bucket", default=""),
                "storage_endpoint": _deep_get(wb, "storage", "endpoint_url", default=""),
            }

        if not tf_outputs.get("vm_ip"):
            _fail("No VM IP found. Run without --skip-infra first, or set config manually.")
            return

    # ── Phase 2: Application ─────────────────────────────────────────
    vm_ip = tf_outputs.get("vm_ip", "")
    ssh_user = tf_outputs.get("ssh_user", "ubuntu")
    ssh_key = tf_outputs.get("ssh_key_path", "~/.ssh/id_ed25519")
    bucket = tf_outputs.get("storage_bucket", "")
    storage_ep = tf_outputs.get("storage_endpoint", "")
    endpoint = f"http://{vm_ip}:{server_port}"
    bucket_display = bucket if str(bucket).startswith("s3://") else (f"s3://{bucket}/checkpoints/" if bucket else "")
    instance_name = f"lerobot-{proj_alias}-{wb_name}"
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
                        "workbench_type": "lerobot",
                        "runtime": runtime.value,
                        "app_status": APP_STATUS_PROVISIONED,
                        **byovm_fields,
                        "ssh": {"host": vm_ip, "user": ssh_user, "key_path": ssh_key},
                        "storage": {"checkpoint_bucket": bucket_display, "endpoint_url": storage_ep},
                    },
                },
            },
        },
    }

    from npa.clients.config import list_projects
    if default or not list_projects():
        config_data["default_project"] = proj_alias
        config_data["default_workbench"] = wb_name

    if not dry_run:
        from npa.clients.config import write_config
        write_config(config_data)
        console.print(f"    Registered workbench in ~/.npa/config.yaml")

    def mark_app_status(app_status: str) -> None:
        if not dry_run:
            update_workbench_app_status(proj_alias, wb_name, app_status)

    def fail_app(msg: str) -> None:
        mark_app_status(APP_STATUS_INSTALL_FAILED)
        _fail(msg)

    if not skip_app:
        mark_app_status(APP_STATUS_INSTALLING)
        from npa.clients.config import SSHConfig, resolve_credentials
        from npa.clients.ssh import SSHClient, SSHError
        from npa.deploy.configurator import (
            ConfiguratorError, deploy_lerobot_container, deploy_server, health_check,
            install_lerobot, write_manifest, write_remote_docker_env_file,
        )

        credentials = resolve_credentials()
        ssh_cfg = SSHConfig(
            host=vm_ip,
            user=ssh_user,
            key_path=ssh_key,
            tokens=credentials.tokens,
        )

        step += 1
        console.print(f"  [{step}/{total_steps}] Connecting via SSH to {ssh_user}@{vm_ip}...")
        if not dry_run:
            ssh = SSHClient(ssh_cfg)
            try:
                code, out, _ = ssh.run("echo connected")
                if code != 0:
                    fail_app(f"SSH connection test failed (exit {code})")
                    return
            except SSHError as exc:
                fail_app(str(exc))
                return

        server_config = {
            "server_port": server_port,
            "checkpoint_dir": "/opt/lerobot/checkpoints",
            "checkpoint_bucket": bucket if str(bucket).startswith("s3://") else (f"s3://{bucket}/checkpoints/" if bucket else ""),
            "storage_endpoint": storage_ep,
            "job_status_dir": "/opt/lerobot/job_status",
            "log_dir": "/var/log/npa-lerobot",
            "hf_cache_dir": "/opt/lerobot/hf_cache",
            "training_output_dir": "/opt/lerobot/checkpoints",
            "cuda_visible_devices": byovm_visible_devices,
            "gpu_count": byovm_effective_gpu_count,
            "shared_env": shared_credential_env(credentials) if not no_shared_creds else {},
        }
        if runtime == WorkbenchRuntime.vm:
            step += 1
            console.print(f"  [{step}/{total_steps}] Checking LeRobot installation...")
            if not dry_run:
                if install_lerobot(ssh):
                    console.print("    LeRobot already installed")
                else:
                    fail_app("LeRobot not installed. cloud-init may still be running - wait and retry.")
                    return

            step += 1
            console.print(f"  [{step}/{total_steps}] Deploying npa-lerobot-server...")
            if not dry_run:
                try:
                    deploy_server(ssh, server_config)
                except (SSHError, ConfiguratorError) as exc:
                    fail_app(f"Server deployment failed: {exc}")
                    return
        else:
            step += 1
            console.print(f"  [{step}/{total_steps}] Deploying LeRobot container ({container_image})...")
            if not dry_run:
                try:
                    if byovm:
                        byovm_env = {
                            "AWS_ACCESS_KEY_ID": merged_vars.get("nebius_api_key", "") or os.environ.get("AWS_ACCESS_KEY_ID", ""),
                            "AWS_SECRET_ACCESS_KEY": merged_vars.get("nebius_secret_key", "") or os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
                            "AWS_ENDPOINT_URL": storage_ep,
                            "NEBIUS_S3_ENDPOINT": storage_ep,
                            "NEBIUS_S3_BUCKET": bucket,
                            "NEBIUS_REGION": env_region,
                            "MUJOCO_GL": "egl",
                            "PYOPENGL_PLATFORM": "egl",
                            "PYTHONUNBUFFERED": "1",
                            **gpu_env_fields(
                                byovm_gpu_info,
                                effective_count=byovm_effective_gpu_count or None,
                                visible_devices=byovm_visible_devices,
                            ),
                        }
                        apply_shared_credential_env(byovm_env, credentials, include=not no_shared_creds)
                        write_remote_docker_env_file(
                            ssh,
                            "/opt/lerobot/.env",
                            byovm_env,
                            owner=ssh_user,
                        )
                    deploy_lerobot_container(
                        ssh,
                        image_ref=container_image,
                        server_config=server_config,
                        ssh_user=ssh_user,
                        registry_token=merged_vars.get("iam_token", ""),
                    )
                except (SSHError, ConfiguratorError) as exc:
                    fail_app(f"Container deployment failed: {exc}")
                    return

            step += 1
            console.print(f"  [{step}/{total_steps}] Checking container status...")
            if not dry_run:
                try:
                    _, running, _ = ssh.run_or_raise(
                        "sudo docker inspect -f '{{.State.Running}}' npa-lerobot"
                    )
                except SSHError as exc:
                    fail_app(f"Container status check failed: {exc}")
                    return
                if running.strip() != "true":
                    fail_app("LeRobot container is not running.")
                    return

        step += 1
        console.print(f"  [{step}/{total_steps}] Health check on {endpoint}...")
        if not dry_run:
            if health_check(endpoint):
                console.print("    Server is healthy")
            else:
                fail_app(f"Server not healthy at {endpoint}/health.")
                return

        step += 1
        console.print(f"  [{step}/{total_steps}] Writing deployment manifest...")
        if not dry_run:
            try:
                write_manifest(ssh, tool="lerobot", version=lerobot_version, deployed_by=f"npa deploy --runtime {runtime.value}")
            except SSHError:
                pass
        mark_app_status(APP_STATUS_HEALTHY)

    # ── Write config ─────────────────────────────────────────────────
    step += 1
    console.print(f"  [{step}/{total_steps}] Updating config status ({proj_alias}/{wb_name})...")
    if not dry_run:
        console.print(f"    Saved to ~/.npa/config.yaml")

    # ── Optional: pre-load checkpoint ────────────────────────────────
    if checkpoint and not dry_run and not skip_app:
        step += 1
        console.print(f"  [{step}/{total_steps}] Pre-loading checkpoint: {checkpoint}...")
        from npa.clients.http import HTTPClient, ServerError
        client = HTTPClient(endpoint)
        try:
            client.serve(checkpoint)
        except ServerError as exc:
            console.print(f"    [warn] Could not pre-load checkpoint: {exc}")

    # ── Summary ──────────────────────────────────────────────────────
    console.print("")
    console.print(f"[bold green]Deploy complete.[/bold green] ({proj_alias}/{wb_name})")
    console.print(f"  Endpoint:  {endpoint}")
    console.print(f"  SSH:       ssh -i {ssh_key} {ssh_user}@{vm_ip}")
    if bucket_display:
        console.print(f"  Storage:   {bucket_display}")
    console.print("")
    console.print(f"  Try: npa workbench lerobot -p {proj_alias} -n {wb_name} status")

    if output == OutputFormat.json:
        typer.echo(json.dumps({
            "project": proj_alias, "name": wb_name,
            "endpoint": endpoint, "vm_ip": vm_ip, "ssh_user": ssh_user,
            "gpu_platform": byovm_fields.get("gpu_platform", gpu_type),
            "gpu_preset": byovm_fields.get("gpu_preset", gpu_preset),
            "gpu_count": byovm_fields.get("gpu_count"),
            "environment": {"project_id": env_project, "tenant_id": env_tenant, "region": env_region},
            "tf_outputs": tf_outputs,
        }, indent=2))


def _deploy_step_count(skip_infra: bool, skip_app: bool, destroy: bool, checkpoint: str) -> int:
    if destroy:
        return 2
    count = 0
    if not skip_infra:
        count += 2  # init + apply
    else:
        count += 1  # read existing
    if not skip_app:
        count += 5  # ssh + lerobot check + deploy + health + manifest
    count += 1  # write config
    if checkpoint:
        count += 1
    return count


# ── system-info ─────────────────────────────────────────────────────────


@app.command("system-info")
def system_info_cmd(
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Collect and display system hardware information from the VM."""
    cfg = _get_config()
    from npa.clients.ssh import SSHClient, SSHError

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
            "sudo docker inspect -f 'state={{.State.Status}} image={{.Config.Image}}' npa-lerobot"
        )
    try:
        _, out, _ = ssh.run_or_raise(info_cmd)
    except SSHError as exc:
        _fail(str(exc))
        return

    if output == OutputFormat.json:
        typer.echo(json.dumps({"system_info": out.strip()}, indent=2))
    else:
        typer.echo(out)


# ── benchmark ───────────────────────────────────────────────────────────


@app.command("benchmark")
def benchmark_cmd(
    run: list[str] = typer.Option(
        ..., "--run", "-r",
        help="Training spec as POLICY:DATASET:STEPS (repeatable).",
    ),
    num_workers: list[int] = typer.Option(
        ..., "--num-workers", "-w",
        help="Dataloader num_workers to test (0 = max CPUs). Repeatable.",
    ),
    batch_size: int = typer.Option(8, "--batch-size", help="Batch size for all runs."),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Run a benchmark suite: collect system info, train each model at each num_workers value, upload results to S3."""
    import base64

    cfg = _get_config()
    from npa.clients.ssh import SSHClient, SSHError

    ssh = SSHClient(cfg.ssh)
    stream = output != OutputFormat.json

    # Parse run specs
    specs: list[dict[str, Any]] = []
    for r in run:
        parts = r.split(":")
        if len(parts) != 3:
            _fail(f"Invalid --run format: '{r}' (expected POLICY:DATASET:STEPS)")
        try:
            steps_val = int(parts[2])
        except ValueError:
            _fail(f"Invalid STEPS value in --run '{r}': '{parts[2]}' is not an integer")
            return  # unreachable, keeps type checker happy
        if steps_val <= 0:
            _fail(f"STEPS must be positive in --run '{r}', got {steps_val}")
            return
        specs.append({"policy": parts[0], "dataset": parts[1], "steps": steps_val})

    # Validate num_workers (must be >= 0; 0 means max CPUs)
    for nw in num_workers:
        if nw < 0:
            _fail(f"--num-workers must be >= 0 (0 = max CPUs), got {nw}")
            return

    # Detect max CPUs on the VM
    try:
        _, nproc_out, _ = ssh.run_or_raise(_runtime_exec_cmd(cfg, "nproc"))
    except SSHError as exc:
        _fail(str(exc))
        return
    max_cpus = int(nproc_out.strip())

    # Resolve num_workers (0 → max CPUs).  Deduplicate on the resolved
    # value so that e.g. --num-workers 0 --num-workers 24 on a 24-CPU
    # host produces one run, not two writing to the same directory.
    resolved_workers: list[int] = []
    _seen_resolved: set[int] = set()
    for nw in num_workers:
        resolved = max_cpus if nw == 0 else nw
        if resolved in _seen_resolved:
            continue
        _seen_resolved.add(resolved)
        resolved_workers.append(resolved)

    # Create benchmark directory
    ts = time.strftime("%Y%m%d-%H%M%S")
    wb = _workbench_name or "default"
    bench_dir = f"/opt/lerobot/benchmarks/benchmark-{ts}"
    s3_prefix = f"benchmarks/{wb}/benchmark-{ts}"

    total_runs = len(specs) * len(resolved_workers)
    if stream:
        console.print(f"[bold]Benchmark: {total_runs} runs on {wb}[/bold]")
        console.print(f"  CPUs detected: {max_cpus}")
        console.print(f"  num_workers: {resolved_workers}")
        console.print(f"  batch_size: {batch_size}")

    try:
        ssh.run_or_raise(_runtime_exec_cmd(cfg, f"mkdir -p {bench_dir}"))
    except SSHError as exc:
        _fail(str(exc))
        return

    # ── Collect system info ──────────────────────────────────────────
    if stream:
        console.print("\n[bold]Collecting system info...[/bold]")
    try:
        ssh.run_or_raise(
            _runtime_exec_cmd(
                cfg,
                f"{{ nvidia-smi; echo '---'; lscpu; echo '---'; free -h; echo '---'; lsblk; }}"
                f" > {bench_dir}/system_info.txt 2>&1",
            )
        )
        if stream:
            console.print("  system_info.txt saved")
    except SSHError as exc:
        if stream:
            console.print(f"  [yellow]Warning: system info collection failed: {exc}[/yellow]")

    # ── Training runs ────────────────────────────────────────────────
    results: list[dict[str, Any]] = []
    run_idx = 0

    for spec in specs:
        for nw in resolved_workers:
            run_idx += 1
            policy = spec["policy"]
            dataset = spec["dataset"]
            steps = spec["steps"]
            dataset_slug = dataset.replace("/", "_")
            run_name = f"{policy}_{dataset_slug}_s{steps}_w{nw}"
            run_dir = f"{bench_dir}/{run_name}"
            output_dir = f"{run_dir}/train_output"

            if stream:
                console.print(
                    f"\n[bold][{run_idx}/{total_runs}] {policy} | {dataset} | "
                    f"workers={nw} | batch={batch_size}[/bold]"
                )

            # Drop OS page cache between runs so later num_workers settings
            # don't benefit from warmed storage/cache state of earlier runs.
            try:
                cache_code, _, _ = ssh.run(
                    "sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null"
                )
                if cache_code != 0 and stream:
                    console.print(
                        "  [yellow]Warning: could not drop page cache (no sudo);"
                        " later runs may benefit from warm cache[/yellow]"
                    )
            except SSHError:
                if stream:
                    console.print(
                        "  [yellow]Warning: could not drop page cache;"
                        " later runs may benefit from warm cache[/yellow]"
                    )

            try:
                ssh.run(f"mkdir -p {run_dir}")
            except SSHError as exc:
                if stream:
                    console.print(f"  [red]SSH failed (mkdir): {exc}[/red]")
                results.append({
                    "run_name": run_name, "policy": policy, "dataset": dataset,
                    "steps": steps, "num_workers": nw, "batch_size": batch_size,
                    "status": "failed", "exit_code": -1,
                    "duration_seconds": 0, "error": f"SSH: {exc}",
                })
                continue

            train_cmd = (
                f"source /opt/lerobot/venv/bin/activate && "
                f"set -a && source /opt/lerobot/.env && set +a && "
                f"export MUJOCO_GL=egl && export PYOPENGL_PLATFORM=egl && "
                f"set -o pipefail && "
                f"lerobot-train "
                f"--policy.type={policy} "
                f"--policy.push_to_hub=false "
                f"--dataset.repo_id={dataset} "
                f"--output_dir={output_dir} "
                f"--steps={steps} "
                f"--save_freq={steps} "
                f"--eval_freq=1000000 "
                f"--batch_size={batch_size} "
                f"--num_workers={nw} "
                f"--policy.device=cuda "
                f"--eval.batch_size=1 "
                f"--eval.n_episodes=1 "
                f"2>&1 | tee {run_dir}/train.log"
            )

            start = time.time()
            try:
                exit_code, stdout, stderr = ssh.run(_runtime_exec_cmd(cfg, train_cmd), stream=stream)
            except SSHError as exc:
                if stream:
                    console.print(f"  [red]SSH failed (train): {exc}[/red]")
                results.append({
                    "run_name": run_name, "policy": policy, "dataset": dataset,
                    "steps": steps, "num_workers": nw, "batch_size": batch_size,
                    "status": "failed", "exit_code": -1,
                    "duration_seconds": round(time.time() - start, 1),
                    "error": f"SSH: {exc}",
                })
                continue
            duration = round(time.time() - start, 1)

            summary: dict[str, Any] = {
                "run_name": run_name,
                "policy": policy,
                "dataset": dataset,
                "steps": steps,
                "num_workers": nw,
                "batch_size": batch_size,
                "status": "success" if exit_code == 0 else "failed",
                "exit_code": exit_code,
                "duration_seconds": duration,
            }
            results.append(summary)

            # Write run summary JSON on the VM (base64 for shell safety)
            summary_json = json.dumps(summary, indent=2)
            b64 = base64.b64encode(summary_json.encode()).decode()
            try:
                ssh.run(f"echo {b64} | base64 -d > {run_dir}/summary.json")
            except SSHError:
                pass  # best-effort; summary is already in `results`

            if stream:
                status_str = "[green]OK[/green]" if exit_code == 0 else "[red]FAILED[/red]"
                console.print(f"  {status_str} ({duration}s)")

    # ── Write overall benchmark summary ──────────────────────────────
    overall: dict[str, Any] = {
        "workbench": wb,
        "benchmark_dir": bench_dir,
        "max_cpus": max_cpus,
        "batch_size": batch_size,
        "num_workers_tested": resolved_workers,
        "total_runs": total_runs,
        "passed": sum(1 for r in results if r["status"] == "success"),
        "failed": sum(1 for r in results if r["status"] != "success"),
        "runs": results,
    }
    overall_json = json.dumps(overall, indent=2)
    b64 = base64.b64encode(overall_json.encode()).decode()
    try:
        ssh.run(f"echo {b64} | base64 -d > {bench_dir}/benchmark_summary.json")
    except SSHError:
        if stream:
            console.print("  [yellow]Warning: could not write benchmark_summary.json to VM[/yellow]")

    # ── Upload results to S3 ─────────────────────────────────────────
    if cfg.storage.checkpoint_bucket and cfg.storage.endpoint_url:
        if stream:
            console.print("\n[bold]Uploading results to S3...[/bold]")

        upload_script = (
            "import boto3, os, pathlib\n"
            "s3 = boto3.client('s3',\n"
            "    endpoint_url=os.environ.get('NEBIUS_S3_ENDPOINT', ''),\n"
            "    aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID', ''),\n"
            "    aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY', ''))\n"
            "bucket = os.environ.get('NEBIUS_S3_BUCKET', '')\n"
            f"base = pathlib.Path('{bench_dir}')\n"
            "exts = ('.txt', '.log', '.json')\n"
            "count = 0\n"
            "for f in sorted(base.rglob('*')):\n"
            "    if f.is_file() and f.suffix in exts:\n"
            f"        key = '{s3_prefix}/' + str(f.relative_to(base))\n"
            "        s3.upload_file(str(f), bucket, key)\n"
            "        count += 1\n"
            "print('uploaded ' + str(count) + ' files')\n"
        )
        script_b64 = base64.b64encode(upload_script.encode()).decode()
        upload_cmd = (
            f"source /opt/lerobot/venv/bin/activate && "
            f"set -a && source /opt/lerobot/.env && set +a && "
            f"echo {script_b64} | base64 -d | python3"
        )
        try:
            up_code, up_out, up_err = ssh.run(_runtime_exec_cmd(cfg, upload_cmd))
        except SSHError as exc:
            up_code, up_out, up_err = -1, "", str(exc)
        if up_code != 0:
            msg = up_err.strip()[-300:] if up_err else up_out.strip()[-300:]
            if stream:
                console.print(f"  [red]S3 upload failed (exit {up_code}):[/red] {msg}")
            overall["upload_status"] = "failed"
            overall["upload_error"] = msg
        elif "uploaded" in up_out:
            if stream:
                console.print(f"  {up_out.strip()}")
                console.print(f"  S3 prefix: {s3_prefix}/")
            overall["upload_status"] = "success"
        else:
            if stream:
                console.print(f"  [yellow]S3 upload returned no confirmation[/yellow]")
            overall["upload_status"] = "unknown"

    # ── Output ───────────────────────────────────────────────────────
    any_failure = overall["failed"] > 0 or overall.get("upload_status") == "failed"

    if output == OutputFormat.json:
        typer.echo(json.dumps(overall, indent=2))
    elif stream:
        tag = "[bold green]Benchmark complete.[/bold green]" if not any_failure else "[bold red]Benchmark finished with failures.[/bold red]"
        console.print(f"\n{tag}")
        console.print(f"  VM results:  {bench_dir}")
        console.print(f"  Runs: {total_runs} ({overall['passed']} passed, {overall['failed']} failed)")
        if cfg.storage.checkpoint_bucket:
            console.print(f"  S3 results:  {s3_prefix}/")

    if any_failure:
        raise typer.Exit(1)


# ── profile-train ──────────────────────────────────────────────────────


@app.command("profile-train")
def profile_train_cmd(
    run: list[str] | None = typer.Option(
        None, "--run", "-r",
        help="Training spec as POLICY:DATASET:STEPS (repeatable).",
    ),
    mode: str = typer.Option("wallclock", "--mode", "-m", help="Measurement mode: wallclock, profiler, or inference."),
    policy_type: str = typer.Option("", "--policy-type", help="Policy type for --runtime serverless."),
    dataset_repo_id: str = typer.Option("", "--dataset-repo-id", help="HF dataset repo ID for --runtime serverless."),
    steps: int = typer.Option(100, "--steps", help="Training steps for --runtime serverless."),
    compile_model: bool = typer.Option(False, "--compile", help="Apply torch.compile to the policy model."),
    num_workers: int = typer.Option(0, "--num-workers", "-w", help="Dataloader num_workers (0 = max CPUs)."),
    batch_size: int = typer.Option(8, "--batch-size", help="Batch size."),
    warmup_steps: int = typer.Option(10, "--warmup-steps", help="Warmup steps before measurement (both modes)."),
    skip_first: int = typer.Option(10, "--skip-first", help="(profiler mode) Profiler schedule skip_first."),
    warmup: int = typer.Option(5, "--warmup", help="(profiler mode) Profiler schedule warmup."),
    active: int = typer.Option(50, "--active", help="(profiler mode) Profiler schedule active."),
    runtime: WorkbenchRuntime = typer.Option(WorkbenchRuntime.vm, "--runtime", help="Runtime backend: vm, container, byovm, or serverless."),
    script: Path | None = typer.Option(
        None,
        "--script",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help=(
            "Path to local profile script to execute. Required for --runtime serverless. "
            "Defaults to research/lerobot-deploy/training/profile_train.py if not specified."
        ),
    ),
    project_id: str = typer.Option("", "--project-id", help="Nebius project ID override for serverless Jobs."),
    image: str = typer.Option("", "--image", help="Container image override for serverless Jobs."),
    gpu_type: str = typer.Option("h200", "--gpu-type", help="GPU type for serverless Jobs (h200, b300, l40s, gpu-rtx-pro-6000, or Nebius platform)."),
    gpu_count: int = typer.Option(1, "--gpu-count", help="Number of GPUs for serverless Jobs."),
    subnet_id: str = typer.Option("", "--subnet-id", help="Subnet ID for serverless Jobs (auto-discovered if omitted)."),
    job_name: str = typer.Option("", "--job-name", help="Unique serverless Job name."),
    output_path: str = typer.Option("", "--output-path", help="S3 URI where profile artifacts are written for serverless Jobs."),
    submit_only: bool = typer.Option(False, "--submit-only", help="Submit Job and return immediately without polling."),
    poll_interval: float = typer.Option(30.0, "--poll-interval", help="Seconds between serverless Job status checks."),
    wait_timeout: int = typer.Option(5400, "--wait-timeout", help="Max seconds to wait for Job completion when not --submit-only."),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Profile training. Modes: wallclock (throughput), profiler (torch.profiler), or inference."""
    import base64

    stream = output != OutputFormat.json

    # Inference mode measures single-sample latency — force batch_size=1.
    if mode == "inference" and batch_size != 1:
        if stream:
            console.print(f"  [yellow]Note: inference mode overrides batch_size={batch_size} → 1[/yellow]")
        batch_size = 1

    run_specs = run or []

    if is_serverless_runtime(runtime):
        if run_specs:
            if len(run_specs) != 1:
                _fail("LeRobot profile-train --runtime serverless accepts exactly one --run spec.")
            parts = run_specs[0].split(":")
            if len(parts) != 3:
                _fail(f"Invalid --run format: '{run_specs[0]}' (expected POLICY:DATASET:STEPS)")
            if not policy_type:
                policy_type = parts[0]
            if not dataset_repo_id:
                dataset_repo_id = parts[1]
            try:
                steps = int(parts[2])
            except ValueError:
                _fail(f"Invalid STEPS value in --run '{run_specs[0]}': '{parts[2]}' is not an integer")
                return
        script_path = script or _default_lerobot_profile_script_path()
        _profile_train_serverless(
            proj_alias=_project_alias or default_project_name(),
            wb_name=_workbench_name or default_workbench_name(),
            project_id=project_id,
            image=image,
            script=script_path,
            mode=mode,
            policy_type=policy_type,
            dataset_repo_id=dataset_repo_id,
            steps=steps,
            batch_size=batch_size,
            num_workers=num_workers,
            warmup_steps=warmup_steps,
            skip_first=skip_first,
            warmup=warmup,
            active=active,
            compile_model=compile_model,
            gpu_type=gpu_type,
            gpu_count=gpu_count,
            subnet_id=subnet_id,
            job_name=job_name,
            output_path=output_path,
            submit_only=submit_only,
            poll_interval=poll_interval,
            wait_timeout=wait_timeout,
            output=output,
        )
        return

    cfg = _get_config(runtime=runtime.value) if runtime != WorkbenchRuntime.vm else _get_config()
    from npa.clients.ssh import SSHClient, SSHError

    ssh = SSHClient(cfg.ssh)

    # Parse run specs
    specs: list[dict[str, Any]] = []
    if not run_specs:
        _fail("LeRobot profile-train requires at least one --run spec unless --runtime serverless is used.")
    for r in run_specs:
        parts = r.split(":")
        if len(parts) != 3:
            _fail(f"Invalid --run format: '{r}' (expected POLICY:DATASET:STEPS)")
        try:
            steps_val = int(parts[2])
        except ValueError:
            _fail(f"Invalid STEPS value in --run '{r}': '{parts[2]}' is not an integer")
            return
        if steps_val <= 0:
            _fail(f"STEPS must be positive in --run '{r}', got {steps_val}")
            return
        specs.append({"policy": parts[0], "dataset": parts[1], "steps": steps_val})

    # Resolve num_workers (0 → max CPUs)
    try:
        _, nproc_out, _ = ssh.run_or_raise(_runtime_exec_cmd(cfg, "nproc"))
    except SSHError as exc:
        _fail(str(exc))
        return
    max_cpus = int(nproc_out.strip())
    resolved_workers = max_cpus if num_workers == 0 else num_workers

    ts = time.strftime("%Y%m%d-%H%M%S")
    wb = _workbench_name or "default"
    profile_dir = f"/opt/lerobot/benchmarks/{mode}-{ts}"
    s3_prefix = f"profiles/{wb}/{mode}-{ts}"

    if mode not in ("wallclock", "profiler", "inference"):
        _fail(f"Invalid --mode: {mode} (choose from: wallclock, profiler, inference)")
        return

    if stream:
        console.print(f"[bold]profile-train ({mode}): {len(specs)} run(s) on {wb}[/bold]")
        console.print(f"  CPUs: {max_cpus}, num_workers: {resolved_workers}, batch_size: {batch_size}")

    # Pre-cache datasets
    unique_datasets = {s["dataset"] for s in specs}
    if stream:
        console.print(f"\n[bold]Pre-caching {len(unique_datasets)} dataset(s)...[/bold]")
    for ds in unique_datasets:
        try:
            ssh.run_or_raise(
                _runtime_exec_cmd(
                    cfg,
                    f"source /opt/lerobot/venv/bin/activate && "
                    f"set -a && source /opt/lerobot/.env && set +a && "
                    f"python3 -c \"from lerobot.datasets.lerobot_dataset import LeRobotDataset; "
                    f"LeRobotDataset('{ds}'); print('cached: {ds}')\"",
                ),
            )
            if stream:
                console.print(f"  {ds} cached")
        except SSHError as exc:
            _fail(f"Dataset cache failed for {ds}: {exc}")
            return

    results: list[dict[str, Any]] = []
    for idx, spec in enumerate(specs, 1):
        policy = spec["policy"]
        dataset = spec["dataset"]
        steps = spec["steps"]
        run_name = f"{policy}_{dataset.replace('/', '_')}"
        run_dir = f"{profile_dir}/{run_name}"

        if stream:
            console.print(f"\n[bold][{idx}/{len(specs)}] {policy} | {dataset} | workers={resolved_workers}[/bold]")

        cmd = (
            f"source /opt/lerobot/venv/bin/activate && "
            f"set -a && source /opt/lerobot/.env && set +a && "
            f"export MUJOCO_GL=egl && export PYOPENGL_PLATFORM=egl && "
            f"python3 /opt/lerobot/profile_train.py "
            f"--mode={mode} "
            f"--policy_type={policy} "
            f"--dataset_repo_id={dataset} "
            f"--steps={steps} "
            f"--batch_size={batch_size} "
            f"--num_workers={resolved_workers} "
            f"--output_dir={run_dir} "
            f"--warmup_steps={warmup_steps} "
            f"--skip_first={skip_first} "
            f"--warmup={warmup} "
            f"--active={active}"
            f"{' --compile' if compile_model else ''}"
        )

        start = time.time()
        try:
            exit_code, stdout, stderr = ssh.run(_runtime_exec_cmd(cfg, cmd), stream=stream)
        except SSHError as exc:
            if stream:
                console.print(f"  [red]SSH failed: {exc}[/red]")
            results.append({"run_name": run_name, "status": "failed", "error": str(exc)})
            continue
        duration = round(time.time() - start, 1)

        result: dict[str, Any] = {
            "run_name": run_name, "policy": policy, "dataset": dataset,
            "steps": steps, "num_workers": resolved_workers, "batch_size": batch_size,
            "status": "success" if exit_code == 0 else "failed",
            "duration_seconds": duration,
        }
        if exit_code != 0:
            result["stderr"] = stderr.strip()[-500:] if stderr else ""
        results.append(result)

        if stream:
            tag = "[green]OK[/green]" if exit_code == 0 else "[red]FAILED[/red]"
            console.print(f"  {tag} ({duration}s)")

    # Upload to S3
    if cfg.storage.checkpoint_bucket and cfg.storage.endpoint_url:
        if stream:
            console.print(f"\n[bold]Uploading to S3...[/bold]")
        upload_script = (
            "import boto3, os, pathlib\n"
            "s3 = boto3.client('s3',\n"
            "    endpoint_url=os.environ.get('NEBIUS_S3_ENDPOINT', ''),\n"
            "    aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID', ''),\n"
            "    aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY', ''))\n"
            "bucket = os.environ.get('NEBIUS_S3_BUCKET', '')\n"
            f"base = pathlib.Path('{profile_dir}')\n"
            "count = 0\n"
            "for f in sorted(base.rglob('*')):\n"
            "    if f.is_file():\n"
            f"        key = '{s3_prefix}/' + str(f.relative_to(base))\n"
            "        s3.upload_file(str(f), bucket, key)\n"
            "        count += 1\n"
            "print('uploaded ' + str(count) + ' files')\n"
        )
        script_b64 = base64.b64encode(upload_script.encode()).decode()
        upload_cmd = (
            f"source /opt/lerobot/venv/bin/activate && "
            f"set -a && source /opt/lerobot/.env && set +a && "
            f"echo {script_b64} | base64 -d | python3"
        )
        try:
            _, up_out, _ = ssh.run(_runtime_exec_cmd(cfg, upload_cmd))
            if "uploaded" in up_out and stream:
                console.print(f"  {up_out.strip()}")
                console.print(f"  S3 prefix: {s3_prefix}/")
        except SSHError:
            if stream:
                console.print("  [yellow]S3 upload failed[/yellow]")

    overall = {
        "profile_dir": profile_dir, "total_runs": len(specs),
        "passed": sum(1 for r in results if r["status"] == "success"),
        "failed": sum(1 for r in results if r["status"] != "success"),
        "runs": results,
    }

    if output == OutputFormat.json:
        typer.echo(json.dumps(overall, indent=2))
    elif stream:
        tag = "[bold green]Profile complete.[/bold green]" if overall["failed"] == 0 else "[bold red]Profile finished with failures.[/bold red]"
        console.print(f"\n{tag}")
        console.print(f"  VM results: {profile_dir}")
        console.print(f"  Runs: {len(specs)} ({overall['passed']} passed, {overall['failed']} failed)")

    if overall["failed"] > 0:
        raise typer.Exit(1)


# ── train-student ───────────────────────────────────────────────────────


@app.command("train-student")
def train_student_cmd(
    dataset: str = typer.Option(
        "", "--dataset", help="Path to local LeRobotDataset v3 directory."
    ),
    input_path: str = typer.Option(
        "",
        "--input-path",
        help="S3 URI for a LeRobotDataset v3 directory. Overrides --dataset.",
    ),
    policy: str = typer.Option("act", "--policy", help="Policy type (act, diffusion)."),
    epochs: int = typer.Option(100, "--epochs", help="Number of training epochs."),
    batch_size: int = typer.Option(64, "--batch-size", help="Batch size."),
    num_workers: int = typer.Option(4, "--num-workers", help="Dataloader num_workers (>= 0)."),
    device: str = typer.Option("cuda", "--device", help="Torch device."),
    output_dir: str = typer.Option(
        "./checkpoints/student/", "--output-dir", help="Checkpoint output directory."
    ),
    output_path: str = typer.Option(
        "",
        "--output-path",
        help="S3 URI where the student checkpoint is written.",
    ),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Train a vision-only student policy via LeRobot imitation learning.

    Uses a local LeRobotDataset v3 directory produced by 'npa adapter convert'.
    The student sees only camera observations and joint state — never privileged
    simulator state.
    """
    dataset_ref = input_path or dataset
    if not dataset_ref:
        _fail("Provide --dataset or --input-path.")
        return
    try:
        if input_path:
            input_path = validate_read_path(
                input_path,
                tool="LeRobot train-student",
                option="--input-path",
                allow_hf=False,
            )
            dataset_ref = input_path
        output_path = validate_write_path(output_path, tool="LeRobot train-student")
    except PathContractError as exc:
        _fail(str(exc))
        return

    temp_dirs: list[tempfile.TemporaryDirectory[str]] = []

    try:
        if _is_s3_uri(dataset_ref):
            from npa.clients.storage import StorageClient

            tmp = tempfile.TemporaryDirectory(prefix="npa-train-student-input-")
            temp_dirs.append(tmp)
            ds_path = Path(
                StorageClient.from_environment().download_directory(dataset_ref, tmp.name)
            )
        else:
            ds_path = Path(dataset_ref)

        output_is_s3 = _is_s3_uri(output_path)
        if output_path and output_is_s3:
            tmp = tempfile.TemporaryDirectory(prefix="npa-train-student-output-")
            temp_dirs.append(tmp)
            local_output_dir = Path(tmp.name)
        else:
            local_output_dir = Path(output_path or output_dir)

        if not ds_path.exists():
            _fail(f"Dataset not found: {ds_path}")
        if not (ds_path / "meta" / "info.json").exists():
            _fail(f"Not a valid LeRobotDataset v3 directory: {ds_path} (missing meta/info.json)")
        if epochs <= 0:
            _fail(f"--epochs must be positive, got {epochs}")
        if batch_size <= 0:
            _fail(f"--batch-size must be positive, got {batch_size}")
        if num_workers < 0:
            _fail(f"--num-workers must be >= 0, got {num_workers}")

        stream_logs = output != OutputFormat.json

        if stream_logs:
            console.print(f"[bold]Training student ({policy})[/bold]")
            console.print(f"  dataset: {dataset_ref}")
            console.print(f"  epochs={epochs}  batch_size={batch_size}  device={device}")
            console.print(f"  output: {output_path or output_dir}")

        from npa.lerobot.train_student import StudentTrainingError, train_student

        try:
            result = train_student(
                dataset_path=ds_path,
                output_dir=local_output_dir,
                policy_type=policy,
                num_epochs=epochs,
                batch_size=batch_size,
                device=device,
                num_workers=num_workers,
                stream=stream_logs,
            )
        except StudentTrainingError as exc:
            _fail(str(exc))
            return

        if output_path:
            if output_is_s3:
                from npa.clients.storage import StorageClient

                checkpoint_path = Path(
                    result.get("checkpoint_path")
                    or local_output_dir / "checkpoints" / "last" / "pretrained_model"
                )
                uploaded = StorageClient.from_environment().upload_directory(
                    str(checkpoint_path), output_path
                )
                result["output_path"] = uploaded
            else:
                result["output_path"] = result.get("checkpoint_path")

        if output == OutputFormat.json:
            typer.echo(json.dumps(result, indent=2))
        else:
            console.print(f"[green]Student training complete.[/green]")
            for k, v in result.items():
                console.print(f"  {k}: {v}")
    finally:
        for tmp in temp_dirs:
            tmp.cleanup()
