"""npa workbench fiftyone - Voxel51 FiftyOne dataset curation app."""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import time
from enum import Enum
from importlib import resources
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import httpx
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
    FIFTYONE_LOAD_DATASET_VM_LOCAL_ERROR,
    PathContractError,
    validate_read_path,
)
from npa.clients.config import (
    APP_STATUS_HEALTHY,
    APP_STATUS_INSTALL_FAILED,
    APP_STATUS_INSTALLING,
    APP_STATUS_PROVISIONED,
    ConfigError,
    SSHConfig,
    WorkbenchConfig,
    alias_has_terraform_state,
    default_project_name,
    default_workbench_name,
    list_projects,
    remove_workbench_config,
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
from npa.clients.credentials import apply_shared_credential_env, load_credentials, shared_credential_env
from npa.clients.endpoint import EndpointError, service_endpoint
from npa.clients.network import NetworkIngressError
from npa.clients.ssh import SSHClient, SSHError
from npa.clients.serverless import EndpointNotFoundError, ServerlessClient, ServerlessClientError
from npa.deploy import provisioner
from npa.deploy.byovm import (
    BYOVMTarget,
    RUNTIME_HELP,
    apply_project_storage_vars,
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
from npa.deploy.configurator import (
    HealthCheckMode,
    audit_remote_env,
    docker_exec_cmd,
    health_check_ssh,
    write_manifest,
)
from npa.deploy.cleanup import (
    CleanupPartialError,
    classify_alias_state,
    list_terraform_managed_resources,
    remove_partial_config_entry,
    terraform_destroy_partial,
)
from npa.deploy.images import container_image_for_tool
from npa.deploy.provisioner import ProvisionerError
from npa.serverless_common import (
    build_serverless_job_env,
    build_serverless_output_upload_cmd,
    resolve_gpu_platform,
    split_serverless_env,
    validate_output_path,
)

app = typer.Typer(
    name="fiftyone",
    help="Voxel51 FiftyOne dataset curation and visualization workbench.",
    no_args_is_help=True,
)
datasets_app = typer.Typer(
    name="datasets",
    help="Inspect datasets through the FiftyOne GraphQL API.",
    no_args_is_help=True,
)

console = Console(stderr=True)

_project_alias: str = ""
_workbench_name: str = ""

FIFTYONE_VERSION = "1.15.0"
FIFTYONE_HOME = "/opt/fiftyone"
FIFTYONE_CONTAINER_DB_DIR = f"{FIFTYONE_HOME}/container-db"
FIFTYONE_VENV = f"{FIFTYONE_HOME}/venv"
FIFTYONE_SERVICE = "npa-fiftyone-app"
DEFAULT_APP_PORT = 5151
DEFAULT_CPU_PLATFORM = "cpu-d3"
DEFAULT_CPU_PRESET = "4vcpu-16gb"
DEFAULT_CPU_IMAGE_FAMILY = "ubuntu24.04-driverless"
FIFTYONE_READY_ATTEMPTS = 120
FIFTYONE_HEALTH_RETRIES = 120
FIFTYONE_HEALTH_BACKOFF_SEC = 2.0
FIFTYONE_AUTO_PUBLIC_HEALTH_RETRIES = 3
FIFTYONE_STOP_TIMEOUT_SEC = 15
FIFTYONE_READY_MARKER = "NPA_FIFTYONE_APP_READY"


class OutputFormat(str, Enum):
    text = "text"
    json = "json"


class DatasetFormat(str, Enum):
    auto = "auto"
    lerobot = "lerobot"
    video = "video"


class WorkbenchRuntime(str, Enum):
    vm = "vm"
    container = "container"
    byovm = "byovm"
    serverless = "serverless"


FIFTYONE_CONTAINER_NAME = "npa-fiftyone"
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")
FIFTYONE_DATASETS_QUERY = """
query NpaDatasets($first: Int!, $search: String) {
  datasets(first: $first, search: $search) {
    total
    edges {
      node {
        name
        persistent
        mediaType
        estimatedSampleCount
      }
    }
  }
}
"""


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
    """Voxel51 FiftyOne dataset curation and visualization workbench."""
    global _project_alias, _workbench_name
    _project_alias = project
    _workbench_name = name


app.add_typer(datasets_app, name="datasets")


def _fail(msg: str, code: int = 1) -> None:
    console.print(f"[red]Error:[/red] {msg}", soft_wrap=True)
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


def _run_fiftyone_command(
    ssh: SSHClient,
    command: str,
    *,
    stream: bool = False,
) -> tuple[int, str, str]:
    """Run a FiftyOne remote command, accepting the app-ready marker as success.

    FiftyOne can leave child processes or systemd restart work racing with the
    parent shell. In practice Paramiko may report a nonzero status even after
    the command printed the explicit ready marker and the app is healthy. The
    marker is the command-level success contract for these scripts.
    """
    code, out, err = ssh.run(command, stream=stream)
    if code == 0 or FIFTYONE_READY_MARKER in out:
        return code, out, err
    raise SSHError(f"Command failed (exit {code}): {command}\nstderr: {err.strip()}")


def _suppress_transient_curl_errors(stderr: str) -> str:
    kept: list[str] = []
    for line in stderr.splitlines():
        lower = line.lower()
        if "curl: (7)" in lower and "couldn't connect to server" in lower:
            continue
        if "failed to connect to 127.0.0.1" in lower and "couldn't connect to server" in lower:
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def _is_container_runtime(cfg: Any) -> bool:
    return runtime_uses_container(getattr(cfg, "runtime", "vm"))


def _is_serverless_runtime(runtime: Any) -> bool:
    return str(getattr(runtime, "value", runtime)) == WorkbenchRuntime.serverless.value


def _serverless_job_name(project: str, name: str, tool: str) -> str:
    raw = f"npa-{tool}-jobs-{project}-{name}".lower()
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9-]+", "-", raw)).strip("-")[:63]


def _serverless_subnet_id(project: str, name: str, project_id: str) -> str:
    project_cfg = list_projects().get(project, {})
    wb_cfg = ((project_cfg.get("workbenches") or {}).get(name) or {}) if isinstance(project_cfg, dict) else {}
    for source in (wb_cfg.get("serverless", {}), wb_cfg, project_cfg.get("serverless", {}), project_cfg):
        if isinstance(source, dict):
            configured = source.get("subnet_id") or source.get("vpc_subnet_id") or source.get("subnet")
            if configured:
                return str(configured)
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
    ready = [item for item in items if str(((item.get("status") or {}).get("state") or "")).upper() == "READY"]
    ranked = sorted(
        ready,
        key=lambda item: (
            "fifty" not in str((item.get("metadata") or {}).get("name", "")).lower(),
            "default" not in str((item.get("metadata") or {}).get("name", "")).lower(),
        ),
    )
    subnet = str(((ranked[0].get("metadata") or {}).get("id") or "")) if ranked else ""
    if subnet and len(ready) > 1:
        console.print(f"[yellow]Warning:[/yellow] Using discovered Jobs subnet {subnet}. Pass --subnet-id to override.")
    return subnet


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


def _fiftyone_serverless_load_dataset_command(
    name: str,
    dataset_source: str,
    dataset_format: DatasetFormat,
) -> str:
    local_dir = "/tmp/npa-fiftyone-load-dataset"
    script = f"""
import json, os, pathlib, time

out = pathlib.Path("{local_dir}")
out.mkdir(parents=True, exist_ok=True)
started = time.time()
try:
    import fiftyone as fo  # noqa: F401
    import_status = "available"
except Exception as exc:
    import_status = f"unavailable: {{type(exc).__name__}}: {{exc}}"
summary = {{
    "status": "loaded",
    "tool": "fiftyone",
    "name": {name!r},
    "source": {dataset_source!r},
    "format": {dataset_format.value!r},
    "fiftyone_import": import_status,
    "job": os.environ.get("NPA_JOB_NAME", ""),
    "duration_seconds": round(time.time() - started, 3),
}}
(out / "npa_fiftyone_dataset_summary.json").write_text(json.dumps(summary, indent=2))
print("NPA_FIFTYONE_SERVERLESS_LOAD_DONE", os.environ.get("NPA_OUTPUT_PATH", ""), flush=True)
""".strip()
    upload = build_serverless_output_upload_cmd(local_dir, "")
    body = (
        'NPA_PYTHON_BIN="${NPA_PYTHON_BIN:-python3}"\n'
        'if ! command -v "$NPA_PYTHON_BIN" >/dev/null 2>&1; then NPA_PYTHON_BIN=python; fi\n'
        f'"$NPA_PYTHON_BIN" <<\'PY\'\n{script}\nPY\n{upload}'
    )
    return f"bash -lc {shlex.quote(body)}"


def _fiftyone_serverless_load_dataset(
    *,
    name: str,
    dataset_source: str,
    dataset_format: DatasetFormat,
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
    output: OutputFormat,
) -> None:
    if not output_path:
        _fail("FiftyOne load-dataset --runtime serverless requires --output-path.")
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
        _fail("FiftyOne load-dataset --runtime serverless requires --project-id or a configured project.")
    name_for_job = job_name or _serverless_job_name(proj_alias, wb_name, "fiftyone")
    out = output_path.rstrip("/") + "/"
    subnet = subnet_id or _serverless_subnet_id(proj_alias, wb_name, resolved_project_id)
    env, extra_env = _serverless_job_env(
        proj_alias,
        out,
        {
            "NPA_JOB_NAME": name_for_job,
            "FIFTYONE_SERVERLESS_SMOKE": "1",
            "FIFTYONE_DATASET_NAME": name,
        },
    )
    client = ServerlessClient()
    try:
        existing = client.get_job(name_for_job, resolved_project_id)
    except EndpointNotFoundError:
        existing = None
    try:
        if existing is not None:
            info = existing if submit_only or existing.status in {"succeeded", "failed", "cancelled"} else client.poll_job(existing.id, resolved_project_id, interval_s=poll_interval, ceiling_s=timeout)
            _output({"status": "existing", "job_id": info.id, "job_name": info.name, "job_status": info.status, "output_path": out}, output)
            return
        info = client.create_job(
            project_id=resolved_project_id,
            name=name_for_job,
            image=image or container_image_for_tool("fiftyone", registry=resolve_container_registry(proj_alias)),
            command=_fiftyone_serverless_load_dataset_command(name, dataset_source, dataset_format),
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


def _container_exec(command: str) -> str:
    return docker_exec_cmd(FIFTYONE_CONTAINER_NAME, command)


def _parse_first_json_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    if start < 0:
        return None
    try:
        parsed, _ = json.JSONDecoder().raw_decode(text[start:])
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


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
        help="Source CIDR allowed to reach the FiftyOne app.",
    ),
) -> None:
    """Ensure public ingress for the saved FiftyOne BYOVM alias."""
    try:
        result = ensure_alias_ingress(
            tool="fiftyone",
            port=DEFAULT_APP_PORT,
            project_alias=_project_alias or None,
            name=name or _workbench_name or None,
            source=source,
        )
    except (ConfigError, NetworkIngressError) as exc:
        _fail(str(exc))
    typer.echo(ingress_summary(result, DEFAULT_APP_PORT))


@app.command("register-byovm")
def register_byovm_cmd(
    alias: str = typer.Option(..., "--alias", help="Workbench alias to create or update."),
    instance_id: str = typer.Option(..., "--instance-id", help="Nebius compute instance ID."),
    port: int = typer.Option(DEFAULT_APP_PORT, "--port", help="FiftyOne HTTP app port."),
) -> None:
    """Register an existing VM as a FiftyOne BYOVM alias and ensure ingress."""
    try:
        register_byovm_alias(
            tool="fiftyone",
            alias=alias,
            instance_id=instance_id,
            port=port,
            project_alias=_project_alias or None,
            warn=console.print,
        )
    except (ConfigError, NetworkIngressError) as exc:
        _fail(str(exc))


def _lerobot_importer_source() -> str:
    return resources.files("npa").joinpath("fiftyone_lerobot.py").read_text()


def _apply_saved_terraform_state(
    merged_vars: dict[str, str],
    *,
    project: str | None,
    explicit_vars: dict[str, str],
) -> None:
    """Reuse saved S3 backend credentials for remote Terraform state."""
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


def _validate_gpu_selection(gpu_type: str, gpu_preset: str) -> None:
    if gpu_type and not gpu_preset:
        _fail("Missing --gpu-preset. Provide the Nebius GPU preset that matches the selected GPU type.")
    if gpu_preset and not gpu_type:
        _fail("Missing --gpu-type. Provide the Nebius GPU platform for the selected GPU preset.")


def _compute_selection(
    gpu_type: str,
    gpu_preset: str,
    cpu_type: str,
    cpu_preset: str,
) -> tuple[str, str, bool]:
    _validate_gpu_selection(gpu_type, gpu_preset)
    if gpu_type and gpu_preset:
        return gpu_type, gpu_preset, True
    if not cpu_type:
        _fail("--cpu-type must not be empty")
    if not cpu_preset:
        _fail("--cpu-preset must not be empty")
    return cpu_type, cpu_preset, False


def _endpoint_for_port(endpoint: str, host: str, port: int) -> str:
    parsed = urlparse(endpoint or "")
    scheme = parsed.scheme or "http"
    hostname = parsed.hostname or host
    if not hostname:
        hostname = "localhost"
    netloc_host = f"[{hostname}]" if ":" in hostname and not hostname.startswith("[") else hostname
    return f"{scheme}://{netloc_host}:{port}"


def _url_with_query_param(url: str, key: str, value: str) -> str:
    parsed = urlparse(url)
    params = dict(parse_qsl(parsed.query, keep_blank_values=True))
    params[key] = value
    return urlunparse(parsed._replace(query=urlencode(params)))


def _is_loopback_host(hostname: str | None) -> bool:
    return (hostname or "").lower() in {"127.0.0.1", "localhost", "::1"}


def _localhost_browser_url(url: str) -> str:
    parsed = urlparse(url)
    if _is_loopback_host(parsed.hostname):
        return url
    port = parsed.port or DEFAULT_APP_PORT
    return urlunparse(parsed._replace(netloc=f"127.0.0.1:{port}"))


def _browser_url_for_strategy(url: str, endpoint_strategy: str) -> str:
    """Return a browser URL that works with FiftyOne's tunnel fallback.

    FiftyOne 1.15 uses the /events Server-Sent Events stream by default. Its
    frontend switches to the built-in polling event listener when the browser
    URL includes polling=true, which is more reliable through SSH forwards.
    """
    if str(endpoint_strategy or "").lower() == "ssh":
        return _url_with_query_param(_localhost_browser_url(url), "polling", "true")
    return url


def _browser_url_for_config(cfg: WorkbenchConfig, url: str) -> str:
    return _browser_url_for_strategy(url, getattr(cfg, "endpoint_strategy", "public"))


def _graphql_url(base_url: str) -> str:
    return base_url.rstrip("/") + "/graphql"


def _is_fiftyone_workbench(name: str, wb_cfg: dict[str, Any]) -> bool:
    """True when the workbench entry is a FiftyOne app VM."""
    wtype = wb_cfg.get("workbench_type")
    if wtype:
        return wtype == "fiftyone"

    normalized = name.replace("_", "-").lower()
    if "fiftyone" in normalized or "fifty-one" in normalized:
        return bool(wb_cfg.get("endpoint") or wb_cfg.get("ssh", {}).get("host"))
    return False


def _build_app_py() -> str:
    return '''\
from __future__ import annotations

import os
import signal
import time

import fiftyone as fo

_stop = False


def _handle_stop(signum, frame):
    global _stop
    _stop = True


signal.signal(signal.SIGINT, _handle_stop)
signal.signal(signal.SIGTERM, _handle_stop)

dataset_name = os.environ.get("FIFTYONE_DATASET_NAME", "").strip()
dataset = None
if dataset_name:
    try:
        if dataset_name in fo.list_datasets():
            dataset = fo.load_dataset(dataset_name)
        else:
            print(f"Dataset {dataset_name!r} not found; launching empty app", flush=True)
    except Exception as exc:
        print(f"Could not load dataset {dataset_name!r}: {exc}", flush=True)

address = os.environ.get("FIFTYONE_DEFAULT_APP_ADDRESS", "0.0.0.0")
port = int(os.environ.get("FIFTYONE_DEFAULT_APP_PORT", "5151"))
session = fo.launch_app(
    dataset,
    remote=True,
    address=address,
    port=port,
    auto=False,
)
print(f"NPA_FIFTYONE_APP_READY http://{address}:{port}", flush=True)

try:
    while not _stop:
        time.sleep(1)
finally:
    session.close()
'''


def _ensure_storage_env_permissions_script() -> str:
    return """\
if [ -f /etc/npa-fiftyone/env ]; then
  sudo chown "$USER:$USER" /etc/npa-fiftyone/env 2>/dev/null || true
  sudo chmod 600 /etc/npa-fiftyone/env 2>/dev/null || true
fi
if [ -f /opt/lerobot/.env ]; then
  sudo chown "$USER:$USER" /opt/lerobot/.env 2>/dev/null || true
  sudo chmod 600 /opt/lerobot/.env 2>/dev/null || true
fi
"""


def _source_storage_env_script() -> str:
    return f"""\
{_ensure_storage_env_permissions_script()}
if [ -f /etc/npa-fiftyone/env ] && [ -r /etc/npa-fiftyone/env ]; then
  set -a
  . /etc/npa-fiftyone/env
  set +a
elif [ -f /opt/lerobot/.env ]; then
  if [ -r /opt/lerobot/.env ]; then
    set -a
    . /opt/lerobot/.env
    set +a
  else
    echo "WARNING: /opt/lerobot/.env exists but is not readable; S3 sources may fail" >&2
  fi
fi
"""


def _service_stop_override_script() -> str:
    return f"""\
if systemctl cat {FIFTYONE_SERVICE} >/dev/null 2>&1; then
  sudo mkdir -p /etc/systemd/system/{FIFTYONE_SERVICE}.service.d
  sudo tee /etc/systemd/system/{FIFTYONE_SERVICE}.service.d/override.conf >/dev/null <<UNIT
[Service]
KillMode=control-group
TimeoutStopSec={FIFTYONE_STOP_TIMEOUT_SEC}
SendSIGKILL=yes
UNIT
  sudo systemctl daemon-reload
fi
"""


def _service_setup_script(port: int, dataset_name: str | None = None) -> str:
    if dataset_name is None:
        dataset_update = (
            'if [ -n "$current_dataset" ]; then\n'
            '  printf \'%s\\n\' "FIFTYONE_DATASET_NAME=$current_dataset" | sudo tee -a /etc/npa-fiftyone/env >/dev/null\n'
            "else\n"
            "  printf '%s\\n' 'FIFTYONE_DATASET_NAME=' | sudo tee -a /etc/npa-fiftyone/env >/dev/null\n"
            "fi"
        )
    else:
        dataset_update = (
            f"printf '%s\\n' {shlex.quote(f'FIFTYONE_DATASET_NAME={dataset_name}')} "
            "| sudo tee -a /etc/npa-fiftyone/env >/dev/null"
        )
    return f"""\
service_user="$(id -un)"
was_active="false"
current_port=""
current_dataset=""
read_env_var() {{
  key="$1"
  for path in /etc/npa-fiftyone/env /opt/lerobot/.env; do
    if [ -r "$path" ]; then
      grep -E "^${{key}}=" "$path" | tail -n 1 | cut -d= -f2- && return 0
    fi
  done
  return 0
}}
if systemctl is-active --quiet {FIFTYONE_SERVICE}; then
  was_active="true"
fi
if [ -f /etc/npa-fiftyone/env ]; then
  current_port="$(grep -E '^FIFTYONE_DEFAULT_APP_PORT=' /etc/npa-fiftyone/env | tail -n 1 | cut -d= -f2- || true)"
  current_dataset="$(grep -E '^FIFTYONE_DATASET_NAME=' /etc/npa-fiftyone/env | tail -n 1 | cut -d= -f2- || true)"
fi
aws_access_key_id="$(read_env_var AWS_ACCESS_KEY_ID)"
aws_secret_access_key="$(read_env_var AWS_SECRET_ACCESS_KEY)"
aws_endpoint_url="$(read_env_var AWS_ENDPOINT_URL)"
nebius_s3_endpoint="$(read_env_var NEBIUS_S3_ENDPOINT)"
nebius_s3_bucket="$(read_env_var NEBIUS_S3_BUCKET)"
nebius_region="$(read_env_var NEBIUS_REGION)"
if [ -z "$aws_endpoint_url" ]; then
  aws_endpoint_url="$nebius_s3_endpoint"
fi
if [ -z "$nebius_s3_endpoint" ]; then
  nebius_s3_endpoint="$aws_endpoint_url"
fi
sudo mkdir -p /etc/npa-fiftyone
sudo tee /etc/npa-fiftyone/env >/dev/null <<'ENV'
FIFTYONE_DEFAULT_APP_ADDRESS=0.0.0.0
FIFTYONE_DEFAULT_APP_PORT={port}
FIFTYONE_DATABASE_DIR={FIFTYONE_HOME}/db
FIFTYONE_DEFAULT_DATASET_DIR={FIFTYONE_HOME}/datasets
FIFTYONE_DATASET_ZOO_DIR={FIFTYONE_HOME}/zoo/datasets
FIFTYONE_MODEL_ZOO_DIR={FIFTYONE_HOME}/zoo/models
FIFTYONE_DO_NOT_TRACK=true
ENV
{dataset_update}
if [ -n "$aws_access_key_id" ]; then printf '%s\\n' "AWS_ACCESS_KEY_ID=$aws_access_key_id" | sudo tee -a /etc/npa-fiftyone/env >/dev/null; fi
if [ -n "$aws_secret_access_key" ]; then printf '%s\\n' "AWS_SECRET_ACCESS_KEY=$aws_secret_access_key" | sudo tee -a /etc/npa-fiftyone/env >/dev/null; fi
if [ -n "$aws_endpoint_url" ]; then printf '%s\\n' "AWS_ENDPOINT_URL=$aws_endpoint_url" | sudo tee -a /etc/npa-fiftyone/env >/dev/null; fi
if [ -n "$nebius_s3_endpoint" ]; then printf '%s\\n' "NEBIUS_S3_ENDPOINT=$nebius_s3_endpoint" | sudo tee -a /etc/npa-fiftyone/env >/dev/null; fi
if [ -n "$nebius_s3_bucket" ]; then printf '%s\\n' "NEBIUS_S3_BUCKET=$nebius_s3_bucket" | sudo tee -a /etc/npa-fiftyone/env >/dev/null; fi
if [ -n "$nebius_region" ]; then printf '%s\\n' "NEBIUS_REGION=$nebius_region" | sudo tee -a /etc/npa-fiftyone/env >/dev/null; fi
sudo chown "$service_user:$service_user" /etc/npa-fiftyone/env
sudo chmod 600 /etc/npa-fiftyone/env
sudo tee /etc/systemd/system/{FIFTYONE_SERVICE}.service >/dev/null <<UNIT
[Unit]
Description=NPA FiftyOne App
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$service_user
WorkingDirectory={FIFTYONE_HOME}
EnvironmentFile=/etc/npa-fiftyone/env
ExecStart={FIFTYONE_VENV}/bin/python {FIFTYONE_HOME}/app.py
Restart=always
RestartSec=10
KillMode=control-group
TimeoutStopSec={FIFTYONE_STOP_TIMEOUT_SEC}
SendSIGKILL=yes

[Install]
WantedBy=multi-user.target
UNIT
sudo systemctl daemon-reload
sudo systemctl enable {FIFTYONE_SERVICE}
if [ "$was_active" = "true" ] && [ "$current_port" = "{port}" ]; then
  echo "FiftyOne already running on port {port}"
else
  sudo systemctl reset-failed {FIFTYONE_SERVICE} || true
  sudo systemctl restart {FIFTYONE_SERVICE}
fi
for _ in $(seq 1 {FIFTYONE_READY_ATTEMPTS}); do
  if curl -fsS http://127.0.0.1:{port}/ >/dev/null; then
    echo NPA_FIFTYONE_APP_READY
    exit 0
  fi
  sleep 1
done
sudo systemctl --no-pager status {FIFTYONE_SERVICE} || true
echo "WARNING: FiftyOne app did not respond on port {port} before readiness timeout" >&2
exit 0
"""


def _build_install_command(port: int = DEFAULT_APP_PORT) -> str:
    app_py = _build_app_py()
    script = f"""\
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
{_ensure_storage_env_permissions_script()}
sudo apt-get update
sudo apt-get install -y build-essential curl ffmpeg git python3 python3-dev python3-pip python3-venv
sudo mkdir -p {FIFTYONE_HOME} {FIFTYONE_HOME}/datasets {FIFTYONE_HOME}/db {FIFTYONE_HOME}/zoo/datasets {FIFTYONE_HOME}/zoo/models
sudo chown -R "$USER:$USER" {FIFTYONE_HOME}
if [ ! -x {FIFTYONE_VENV}/bin/python ] || ! {FIFTYONE_VENV}/bin/python - <<'PY' >/dev/null 2>&1
from importlib import metadata
raise SystemExit(0 if metadata.version("fiftyone") == "{FIFTYONE_VERSION}" else 1)
PY
then
  rm -rf {FIFTYONE_VENV}
  python3 -m venv {FIFTYONE_VENV}
  {FIFTYONE_VENV}/bin/python -m pip install --upgrade pip setuptools wheel
  {FIFTYONE_VENV}/bin/python -m pip install "fiftyone=={FIFTYONE_VERSION}" boto3 datasets huggingface_hub pyarrow pillow
fi
cat > {FIFTYONE_HOME}/app.py <<'PY'
{app_py}
PY
mkdir -p "$HOME/.fiftyone"
cat > "$HOME/.fiftyone/config.json" <<'JSON'
{{
  "default_app_address": "0.0.0.0",
  "default_app_port": {port}
}}
JSON
{FIFTYONE_VENV}/bin/python - <<'PY'
from importlib import metadata

version = metadata.version("fiftyone")
if version != "{FIFTYONE_VERSION}":
    raise RuntimeError(f"expected fiftyone {FIFTYONE_VERSION}, found {{version}}")
print("FIFTYONE_ENV_SMOKE_OK")
PY
{_service_setup_script(port)}
"""
    return _remote_bash(script)


def _build_launch_command(port: int) -> str:
    script = f"""\
set -euo pipefail
test -x {FIFTYONE_VENV}/bin/python
test -f {FIFTYONE_HOME}/app.py
{_service_setup_script(port)}
"""
    return _remote_bash(script)


def _build_container_launch_command(port: int) -> str:
    script = f"""\
set -euo pipefail
sudo docker inspect -f '{{{{.State.Running}}}}' {FIFTYONE_CONTAINER_NAME} | grep -q true
for _ in $(seq 1 {FIFTYONE_READY_ATTEMPTS}); do
  if curl -fsS http://127.0.0.1:{port}/ >/dev/null; then
    echo "FiftyOne container already running on port {port}"
    echo {FIFTYONE_READY_MARKER}
    exit 0
  fi
  sleep 1
done
sudo docker logs --tail 100 {FIFTYONE_CONTAINER_NAME} || true
echo "WARNING: FiftyOne container did not respond on port {port} before readiness timeout" >&2
exit 1
"""
    return _remote_bash(script)


def _build_load_dataset_command(
    name: str,
    source: str,
    dataset_format: DatasetFormat = DatasetFormat.auto,
) -> str:
    format_value = dataset_format.value if isinstance(dataset_format, DatasetFormat) else str(dataset_format)
    name_literal = json.dumps(name)
    source_literal = json.dumps(source)
    format_literal = json.dumps(format_value)
    importer_source = _lerobot_importer_source() if format_value == DatasetFormat.lerobot.value else ""
    importer_source_literal = json.dumps(importer_source)
    env_line = shlex.quote(f"FIFTYONE_DATASET_NAME={name}")
    script = f"""\
set -euo pipefail
source {FIFTYONE_VENV}/bin/activate
{_source_storage_env_script()}
export FIFTYONE_DATABASE_DIR={FIFTYONE_HOME}/db
export FIFTYONE_DEFAULT_DATASET_DIR={FIFTYONE_HOME}/datasets
export FIFTYONE_DATASET_ZOO_DIR={FIFTYONE_HOME}/zoo/datasets
export FIFTYONE_MODEL_ZOO_DIR={FIFTYONE_HOME}/zoo/models
python - <<'PY'
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from urllib.parse import urlparse

import fiftyone as fo

NAME = {name_literal}
SOURCE = {source_literal}
FORMAT = {format_literal}
DATASETS_DIR = Path("{FIFTYONE_HOME}/datasets")
LEROBOT_IMPORTER_SOURCE = {importer_source_literal}


def reset_dataset() -> None:
    if NAME in fo.list_datasets():
        fo.delete_dataset(NAME)


def _refresh_fiftyone_collection_stats(dataset) -> None:
    # Workaround for FiftyOne/Mongo stale estimatedDocumentCount metadata.
    # Remove this when FiftyOne no longer reports zero estimated counts after
    # CLI-driven dataset loads.
    try:
        from fiftyone.core.odm.database import get_db_conn

        conn = get_db_conn()
        conn.command({{"validate": "datasets"}})
        sample_collection_name = getattr(dataset, "_sample_collection_name", None)
        if sample_collection_name:
            conn.command({{"validate": sample_collection_name}})
        frame_collection_name = getattr(dataset, "_frame_collection_name", None)
        if frame_collection_name:
            conn.command({{"validate": frame_collection_name}})
    except Exception as exc:
        print(f"Warning: could not refresh FiftyOne count metadata: {{exc}}", file=sys.stderr)


def persist(dataset) -> None:
    dataset.persistent = True
    dataset.save()
    _refresh_fiftyone_collection_stats(dataset)


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def _files_with_ext(path: Path, extensions: set[str]) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in extensions else []
    return sorted(item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in extensions)


def load_media_source(path: Path, extensions: set[str]):
    if not path.exists():
        raise FileNotFoundError(path)
    files = _files_with_ext(path, extensions)
    if not files:
        raise RuntimeError(f"No supported media files found in {{path}}")
    reset_dataset()
    dataset = fo.Dataset(NAME)
    dataset.add_samples([fo.Sample(filepath=str(file)) for file in files])
    persist(dataset)
    return dataset


def load_image_dir(path: Path):
    return load_media_source(path, IMAGE_EXTENSIONS)


def load_video_source(path: Path):
    return load_media_source(path, VIDEO_EXTENSIONS)


def load_auto_source(path: Path):
    video_files = _files_with_ext(path, VIDEO_EXTENSIONS)
    image_files = _files_with_ext(path, IMAGE_EXTENSIONS)
    if FORMAT == "video" or (FORMAT == "auto" and video_files and not image_files):
        return load_video_source(path)
    if FORMAT == "auto" and video_files and image_files:
        print(
            "Warning: input contains both images and videos; defaulting to image mode. "
            "Use --format video to load videos.",
            file=sys.stderr,
        )
    return load_image_dir(path)


def download_s3(uri: str) -> Path:
    import boto3

    parsed = urlparse(uri)
    if not parsed.netloc:
        raise ValueError(f"S3 URI must include a bucket: {{uri}}")

    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/")
    target_root = DATASETS_DIR / NAME / "s3"
    if target_root.exists():
        shutil.rmtree(target_root)
    target_root.mkdir(parents=True, exist_ok=True)

    endpoint = os.environ.get("NEBIUS_S3_ENDPOINT") or os.environ.get("AWS_ENDPOINT_URL") or None
    s3 = boto3.client("s3", endpoint_url=endpoint)
    paginator = s3.get_paginator("list_objects_v2")

    count = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            rel = key[len(prefix):].lstrip("/") if prefix else key
            if not rel:
                rel = Path(key).name
            dest = target_root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, key, str(dest))
            count += 1

    if count == 0:
        raise RuntimeError(f"No S3 objects found at {{uri}}")
    return target_root


def load_lerobot():
    importer_path = Path("/tmp/npa_fiftyone_lerobot_importer.py")
    importer_path.write_text(LEROBOT_IMPORTER_SOURCE)
    if str(importer_path.parent) not in sys.path:
        sys.path.insert(0, str(importer_path.parent))

    from npa_fiftyone_lerobot_importer import import_lerobot_dataset

    return import_lerobot_dataset(NAME, SOURCE, DATASETS_DIR)


if FORMAT == "lerobot":
    result = load_lerobot()
elif SOURCE.startswith("s3://"):
    local_source = download_s3(SOURCE)
    dataset = load_auto_source(local_source)
    source_type = "s3"
else:
    from fiftyone.utils.huggingface import load_from_hub

    reset_dataset()
    dataset = load_from_hub(SOURCE, name=NAME)
    persist(dataset)
    source_type = "huggingface"

if FORMAT != "lerobot":
    result = {{
        "status": "loaded",
        "name": dataset.name,
        "source": SOURCE,
        "source_type": source_type,
        "format": FORMAT,
        "samples": len(dataset),
    }}

print(json.dumps(result, indent=2))
PY
sudo mkdir -p /etc/npa-fiftyone
sudo sed -i '/^FIFTYONE_DATASET_NAME=/d' /etc/npa-fiftyone/env 2>/dev/null || true
printf '%s\\n' {env_line} | sudo tee -a /etc/npa-fiftyone/env >/dev/null
{_service_stop_override_script()}
if systemctl is-active --quiet {FIFTYONE_SERVICE}; then
  sudo systemctl reset-failed {FIFTYONE_SERVICE} || true
  sudo systemctl restart {FIFTYONE_SERVICE}
elif systemctl cat {FIFTYONE_SERVICE} >/dev/null 2>&1; then
  sudo systemctl reset-failed {FIFTYONE_SERVICE} || true
  sudo systemctl start {FIFTYONE_SERVICE}
fi
app_port="$(grep -E '^FIFTYONE_DEFAULT_APP_PORT=' /etc/npa-fiftyone/env | tail -n 1 | cut -d= -f2- || true)"
app_port="${{app_port:-{DEFAULT_APP_PORT}}}"
if systemctl cat {FIFTYONE_SERVICE} >/dev/null 2>&1; then
  for _ in $(seq 1 {FIFTYONE_READY_ATTEMPTS}); do
    if curl -fsS "http://127.0.0.1:${{app_port}}/" >/dev/null; then
      echo NPA_FIFTYONE_APP_READY
      exit 0
    fi
    sleep 1
  done
  sudo systemctl --no-pager status {FIFTYONE_SERVICE} || true
  echo "WARNING: FiftyOne app did not respond on port ${{app_port}} before restart readiness timeout" >&2
  exit 0
fi
"""
    return _remote_bash(script)


def _build_container_load_dataset_command(
    name: str,
    source: str,
    dataset_format: DatasetFormat = DatasetFormat.auto,
) -> str:
    format_value = dataset_format.value if isinstance(dataset_format, DatasetFormat) else str(dataset_format)
    name_literal = json.dumps(name)
    source_literal = json.dumps(source)
    format_literal = json.dumps(format_value)
    importer_source = _lerobot_importer_source() if format_value == DatasetFormat.lerobot.value else ""
    importer_source_literal = json.dumps(importer_source)
    env_line = shlex.quote(f"FIFTYONE_DATASET_NAME={name}")
    container_script = f"""\
set -euo pipefail
source {FIFTYONE_VENV}/bin/activate
set -a
. /etc/npa-fiftyone/env
set +a
export FIFTYONE_DATABASE_DIR={FIFTYONE_CONTAINER_DB_DIR}
export FIFTYONE_DEFAULT_DATASET_DIR={FIFTYONE_HOME}/datasets
export FIFTYONE_DATASET_ZOO_DIR={FIFTYONE_HOME}/zoo/datasets
export FIFTYONE_MODEL_ZOO_DIR={FIFTYONE_HOME}/zoo/models
python - <<'PY'
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from urllib.parse import urlparse

import fiftyone as fo

NAME = {name_literal}
SOURCE = {source_literal}
FORMAT = {format_literal}
DATASETS_DIR = Path("{FIFTYONE_HOME}/datasets")
LEROBOT_IMPORTER_SOURCE = {importer_source_literal}


def reset_dataset() -> None:
    if NAME in fo.list_datasets():
        fo.delete_dataset(NAME)


def _refresh_fiftyone_collection_stats(dataset) -> None:
    # Workaround for FiftyOne/Mongo stale estimatedDocumentCount metadata.
    # Remove this when FiftyOne no longer reports zero estimated counts after
    # CLI-driven dataset loads.
    try:
        from fiftyone.core.odm.database import get_db_conn

        conn = get_db_conn()
        conn.command({{"validate": "datasets"}})
        sample_collection_name = getattr(dataset, "_sample_collection_name", None)
        if sample_collection_name:
            conn.command({{"validate": sample_collection_name}})
        frame_collection_name = getattr(dataset, "_frame_collection_name", None)
        if frame_collection_name:
            conn.command({{"validate": frame_collection_name}})
    except Exception as exc:
        print(f"Warning: could not refresh FiftyOne count metadata: {{exc}}", file=sys.stderr)


def persist(dataset) -> None:
    dataset.persistent = True
    dataset.save()
    _refresh_fiftyone_collection_stats(dataset)


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def _files_with_ext(path: Path, extensions: set[str]) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in extensions else []
    return sorted(item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in extensions)


def load_media_source(path: Path, extensions: set[str]):
    if not path.exists():
        raise FileNotFoundError(path)
    files = _files_with_ext(path, extensions)
    if not files:
        raise RuntimeError(f"No supported media files found in {{path}}")
    reset_dataset()
    dataset = fo.Dataset(NAME)
    dataset.add_samples([fo.Sample(filepath=str(file)) for file in files])
    persist(dataset)
    return dataset


def load_image_dir(path: Path):
    return load_media_source(path, IMAGE_EXTENSIONS)


def load_video_source(path: Path):
    return load_media_source(path, VIDEO_EXTENSIONS)


def load_auto_source(path: Path):
    video_files = _files_with_ext(path, VIDEO_EXTENSIONS)
    image_files = _files_with_ext(path, IMAGE_EXTENSIONS)
    if FORMAT == "video" or (FORMAT == "auto" and video_files and not image_files):
        return load_video_source(path)
    if FORMAT == "auto" and video_files and image_files:
        print(
            "Warning: input contains both images and videos; defaulting to image mode. "
            "Use --format video to load videos.",
            file=sys.stderr,
        )
    return load_image_dir(path)


def download_s3(uri: str) -> Path:
    import boto3

    parsed = urlparse(uri)
    if not parsed.netloc:
        raise ValueError(f"S3 URI must include a bucket: {{uri}}")

    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/")
    target_root = DATASETS_DIR / NAME / "s3"
    if target_root.exists():
        shutil.rmtree(target_root)
    target_root.mkdir(parents=True, exist_ok=True)

    endpoint = os.environ.get("NEBIUS_S3_ENDPOINT") or os.environ.get("AWS_ENDPOINT_URL") or None
    s3 = boto3.client("s3", endpoint_url=endpoint)
    paginator = s3.get_paginator("list_objects_v2")

    count = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            rel = key[len(prefix):].lstrip("/") if prefix else key
            if not rel:
                rel = Path(key).name
            dest = target_root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, key, str(dest))
            count += 1

    if count == 0:
        raise RuntimeError(f"No S3 objects found at {{uri}}")
    return target_root


def load_lerobot():
    importer_path = Path("/tmp/npa_fiftyone_lerobot_importer.py")
    importer_path.write_text(LEROBOT_IMPORTER_SOURCE)
    if str(importer_path.parent) not in sys.path:
        sys.path.insert(0, str(importer_path.parent))

    from npa_fiftyone_lerobot_importer import import_lerobot_dataset

    return import_lerobot_dataset(NAME, SOURCE, DATASETS_DIR)


if FORMAT == "lerobot":
    result = load_lerobot()
elif SOURCE.startswith("s3://"):
    local_source = download_s3(SOURCE)
    dataset = load_auto_source(local_source)
    source_type = "s3"
else:
    from fiftyone.utils.huggingface import load_from_hub

    reset_dataset()
    dataset = load_from_hub(SOURCE, name=NAME)
    persist(dataset)
    source_type = "huggingface"

if FORMAT != "lerobot":
    result = {{
        "status": "loaded",
        "name": dataset.name,
        "source": SOURCE,
        "source_type": source_type,
        "format": FORMAT,
        "samples": len(dataset),
    }}

print(json.dumps(result, indent=2))
PY
"""
    host_script = f"""\
set -euo pipefail
sudo docker inspect {FIFTYONE_CONTAINER_NAME} >/dev/null
if ! sudo docker inspect -f '{{{{.State.Running}}}}' {FIFTYONE_CONTAINER_NAME} | grep -q true; then
  sudo docker start {FIFTYONE_CONTAINER_NAME} >/dev/null
fi
sudo docker exec -i {FIFTYONE_CONTAINER_NAME} bash -lc {shlex.quote(container_script)}
sudo sed -i '/^FIFTYONE_DATASET_NAME=/d' /etc/npa-fiftyone/env 2>/dev/null || true
printf '%s\\n' {env_line} | sudo tee -a /etc/npa-fiftyone/env >/dev/null
app_port="$(grep -E '^FIFTYONE_DEFAULT_APP_PORT=' /etc/npa-fiftyone/env | tail -n 1 | cut -d= -f2- || true)"
app_port="${{app_port:-{DEFAULT_APP_PORT}}}"
for _ in $(seq 1 {FIFTYONE_READY_ATTEMPTS}); do
  if curl -fsS "http://127.0.0.1:${{app_port}}/" >/dev/null; then
    echo {FIFTYONE_READY_MARKER}
    exit 0
  fi
  sleep 1
done
sudo docker logs --tail 100 {FIFTYONE_CONTAINER_NAME} || true
echo "WARNING: FiftyOne container did not respond before restart readiness timeout" >&2
exit 0
"""
    return _remote_bash(host_script)


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
                provisioner.init(tf_dir=str(work_dir), backend_config={
                    "access_key": merged_vars.get("nebius_api_key", ""),
                    "secret_key": merged_vars.get("nebius_secret_key", ""),
                })
                return provisioner.outputs(tf_dir=str(work_dir))
            except ProvisionerError:
                pass

    from npa.clients.config import _deep_get, _load_yaml, _resolve_project_section, _resolve_workbench_in_project

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


def _saved_workbench_config(project: str | None, name: str) -> WorkbenchConfig | None:
    try:
        return resolve_ssh_config(project=project, name=name)
    except ConfigError:
        return None


def _saved_byovm_target(
    cfg: WorkbenchConfig | None,
    *,
    ssh_user: str = "",
) -> BYOVMTarget | None:
    if cfg is None or getattr(cfg, "runtime", "") != WorkbenchRuntime.byovm.value:
        return None
    if not cfg.ssh.host or not cfg.ssh.key_path:
        return None
    return BYOVMTarget(
        host=cfg.ssh.host,
        user=ssh_user or cfg.ssh.user or "ubuntu",
        key_path=cfg.ssh.key_path,
    )


def _resolve_byovm_deploy_target(
    *,
    saved_cfg: WorkbenchConfig | None,
    host: str,
    ssh_key: str,
    ssh_user: str,
) -> BYOVMTarget:
    if not host and not ssh_key:
        saved_target = _saved_byovm_target(saved_cfg, ssh_user=ssh_user)
        if saved_target is not None:
            return saved_target
    return resolve_byovm_target(host=host, ssh_key=ssh_key, ssh_user=ssh_user)


def _build_restart_command(port: int) -> str:
    script = f"""\
set -euo pipefail
{_service_stop_override_script()}
if ! systemctl cat {FIFTYONE_SERVICE} >/dev/null 2>&1; then
  echo "FiftyOne systemd service {FIFTYONE_SERVICE} is not installed" >&2
  exit 1
fi
sudo systemctl reset-failed {FIFTYONE_SERVICE} || true
sudo systemctl restart {FIFTYONE_SERVICE}
for _ in $(seq 1 {FIFTYONE_READY_ATTEMPTS}); do
  if curl -fsS http://127.0.0.1:{port}/ >/dev/null; then
    echo {FIFTYONE_READY_MARKER}
    exit 0
  fi
  sleep 1
done
sudo systemctl --no-pager status {FIFTYONE_SERVICE} || true
echo "FiftyOne app did not respond on port {port} before restart readiness timeout" >&2
exit 1
"""
    return _remote_bash(script)


def _build_container_restart_command(port: int) -> str:
    script = f"""\
set -euo pipefail
sudo docker inspect {FIFTYONE_CONTAINER_NAME} >/dev/null
sudo docker restart {FIFTYONE_CONTAINER_NAME} >/dev/null
for _ in $(seq 1 {FIFTYONE_READY_ATTEMPTS}); do
  if curl -fsS http://127.0.0.1:{port}/ >/dev/null; then
    echo {FIFTYONE_READY_MARKER}
    exit 0
  fi
  sleep 1
done
sudo docker logs --tail 100 {FIFTYONE_CONTAINER_NAME} || true
echo "FiftyOne container did not respond on port {port} before restart readiness timeout" >&2
exit 1
"""
    return _remote_bash(script)


def _parse_dataset_edges(payload: dict[str, Any]) -> tuple[int, list[dict[str, Any]]]:
    datasets = payload.get("data", {}).get("datasets", {})
    edges = datasets.get("edges", []) if isinstance(datasets, dict) else []
    items: list[dict[str, Any]] = []
    for edge in edges:
        node = edge.get("node", {}) if isinstance(edge, dict) else {}
        if not isinstance(node, dict):
            continue
        items.append({
            "name": node.get("name", ""),
            "samples": node.get("estimatedSampleCount", 0),
            "media_type": node.get("mediaType", ""),
            "persistent": node.get("persistent", False),
        })
    total = datasets.get("total", len(items)) if isinstance(datasets, dict) else len(items)
    return int(total or 0), items


def _app_health_check(
    endpoint: str,
    *,
    retries: int = FIFTYONE_HEALTH_RETRIES,
    backoff: float = FIFTYONE_HEALTH_BACKOFF_SEC,
) -> bool:
    for _ in range(retries):
        try:
            resp = httpx.get(endpoint, timeout=5.0)
            if resp.status_code < 400:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(backoff)
    return False


@app.command("list")
def list_cmd(
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """List configured FiftyOne workbenches."""
    projects = list_projects()
    def_proj = default_project_name()
    def_wb = default_workbench_name()

    if output == OutputFormat.json:
        filtered = {}
        for pname, pcfg in projects.items():
            wbs = {
                k: v for k, v in pcfg.get("workbenches", {}).items()
                if _is_fiftyone_workbench(k, v)
            }
            if wbs:
                filtered[pname] = {**pcfg, "workbenches": wbs}
        typer.echo(json.dumps({
            "projects": filtered,
            "default_project": def_proj,
            "default_workbench": def_wb,
        }, indent=2))
        return

    if not projects:
        typer.echo("No projects configured. Run 'npa workbench fiftyone deploy' to create one.")
        return

    any_shown = False
    for proj_name, proj_cfg in projects.items():
        workbenches = {
            k: v for k, v in proj_cfg.get("workbenches", {}).items()
            if _is_fiftyone_workbench(k, v)
        }
        if not workbenches:
            continue
        any_shown = True
        proj_marker = " *" if proj_name == def_proj else ""
        region = proj_cfg.get("region", "?")
        typer.echo(f"  {proj_name}{proj_marker}  ({region})")
        for wb_name, wb_cfg in workbenches.items():
            wb_marker = " *" if wb_name == def_wb else ""
            compute = wb_cfg.get("gpu_platform", "?")
            endpoint = wb_cfg.get("endpoint", "?")
            app_status = wb_cfg.get("app_status", "unknown")
            typer.echo(
                f"    {wb_name}{wb_marker}  compute={compute}  endpoint={endpoint}  "
                f"app_status={app_status}"
            )

    if not any_shown:
        typer.echo("No FiftyOne workbenches configured. Run 'npa workbench fiftyone deploy' to create one.")


@app.command("cleanup-partial")
def cleanup_partial_cmd(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation."),
) -> None:
    """Clean up orphaned Terraform resources from an interrupted FiftyOne deploy."""
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
    gpu_type: str = typer.Option("", "--gpu-type", help="Optional Nebius GPU platform."),
    gpu_preset: str = typer.Option("", "--gpu-preset", help="Optional Nebius GPU preset."),
    cpu_type: str = typer.Option(DEFAULT_CPU_PLATFORM, "--cpu-type", help="Nebius CPU platform used when no GPU flags are provided."),
    cpu_preset: str = typer.Option(DEFAULT_CPU_PRESET, "--cpu-preset", help="Nebius CPU preset used when no GPU flags are provided."),
    region: str = typer.Option("", "--region", help="Nebius region."),
    project_id: str = typer.Option("", "--project-id", help="Nebius project ID."),
    tenant_id: str = typer.Option("", "--tenant-id", help="Nebius tenant ID."),
    tf_dir: str = typer.Option("", "--tf-dir", help="Path to Terraform directory (default: bundled)."),
    tf_var: list[str] = typer.Option([], "--tf-var", "-v", help="Extra TF variable (key=value), repeatable."),
    skip_infra: bool = typer.Option(False, "--skip-infra", help="Skip Terraform, only deploy the app."),
    skip_app: bool = typer.Option(False, "--skip-app", help="Skip app installation, only provision infra."),
    destroy: bool = typer.Option(False, "--destroy", help="Destroy infrastructure and clean up config."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would happen without doing it."),
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
    no_shared_creds: bool = typer.Option(False, "--no-shared-creds", help="Do not inject ~/.npa/credentials.yaml shared credentials into the service env."),
    health_check_mode: HealthCheckMode = typer.Option(
        HealthCheckMode.auto,
        "--health-check-mode",
        help="Health check mode: public, ssh, or auto. BYOVM auto tries public briefly, then SSH.",
    ),
    verify_env: bool = typer.Option(bool(os.environ.get("CI")), "--verify-env/--no-verify-env", help="Audit deployed shared credentials after app deploy."),
    port: int = typer.Option(DEFAULT_APP_PORT, "--port", help="FiftyOne app port on the VM."),
    preemptible: bool = typer.Option(True, "--preemptible/--no-preemptible", help="Preemptible GPU instance."),
    runtime: WorkbenchRuntime = typer.Option(WorkbenchRuntime.vm, "--runtime", help=RUNTIME_HELP),
    host: str = typer.Option("", "--host", help="BYOVM SSH host/IP. Used only with --runtime byovm."),
    ssh_key: str = typer.Option("", "--ssh-key", help="BYOVM SSH private key path. Used only with --runtime byovm."),
    ssh_user: str = typer.Option("", "--ssh-user", help="BYOVM SSH username. Defaults to ubuntu."),
    gpu_count: int = typer.Option(0, "--gpu-count", help="Limit visible GPUs on BYOVM (0 = all detected)."),
    disk_size: int | None = typer.Option(None, "--disk-size", help="Boot disk size in GiB. Defaults to 250 for container runtime; VM runtime keeps the Terraform default."),
    default: bool = typer.Option(False, "--default", help="Set this workbench as the default."),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Deploy or destroy a FiftyOne dataset curation VM."""
    byovm = is_byovm_runtime(runtime)
    if _is_serverless_runtime(runtime):
        _fail("FiftyOne deploy does not use --runtime serverless; use `npa workbench fiftyone load-dataset --runtime serverless`.")
    platform, preset, uses_gpu = _compute_selection(gpu_type, gpu_preset, cpu_type, cpu_preset)
    if byovm:
        uses_gpu = True

    proj_alias = _project_alias or None
    wb_name = _workbench_name or "fiftyone"
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
        proj_alias = env_region or ("byovm" if byovm else "default")

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

    saved_wb_cfg = _saved_workbench_config(proj_alias, wb_name) if skip_infra or byovm else None

    nebius_creds: dict[str, str] = {}
    saved_state = resolve_terraform_state(proj_alias) if use_remote_state else None
    has_saved_state = bool(
        saved_state
        and saved_state.bucket
        and saved_state.endpoint
        and saved_state.access_key
        and saved_state.secret_key
    )

    if use_remote_state and destroy and has_saved_state:
        if not env_project or not env_region:
            _fail("Destroy requires saved project_id and region in ~/.npa/config.yaml")
            return
        if dry_run:
            console.print("  [dry-run] Would reuse saved Terraform state credentials")
        else:
            from npa.clients.nebius import NebiusError, ensure_service_account, get_iam_token

            try:
                nebius_creds = {
                    "iam_token": get_iam_token(),
                    "service_account_id": ensure_service_account(env_project),
                    "nebius_project_id": env_project,
                    "nebius_region": env_region,
                }
            except NebiusError as exc:
                _fail(f"Nebius auth failed: {exc}")
                return

    if use_remote_state and not skip_infra and not (destroy and has_saved_state):
        if not env_project or not env_tenant or not env_region:
            _fail(
                "First deploy requires --project-id, --tenant-id, and --region.\n"
                "  Example: npa workbench fiftyone -p eu-north1 -n curate deploy \\\n"
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
            write_config({
                "projects": {
                    proj_alias: {
                        "project_id": env_project,
                        "tenant_id": env_tenant,
                        "region": env_region,
                    },
                },
            })

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
    if not byovm:
        try:
            provisioner.apply_boot_disk_tf_vars(merged_vars, runtime, disk_size)
        except ValueError as exc:
            _fail(str(exc))
            return

    if use_remote_state and nebius_creds and not dry_run:
        write_config({
            "projects": {
                proj_alias: {
                    "terraform_state": _terraform_state_config(merged_vars),
                },
            },
        })

    if not uses_gpu and "image_family" not in merged_vars:
        merged_vars["image_family"] = DEFAULT_CPU_IMAGE_FAMILY

    instance_name = f"fiftyone-{proj_alias}-{wb_name}"
    enable_preemptible = "true" if uses_gpu and preemptible else "false"
    cloud_init_workbench_type = (
        "lerobot-container"
        if runtime_uses_container(runtime)
        else "fiftyone"
    )

    if destroy:
        if byovm:
            console.print(f"  [1/1] Unregistering BYOVM workbench {proj_alias}/{wb_name}...")
            if not dry_run:
                remove_workbench_config(proj_alias, wb_name)
            console.print(f"  {proj_alias}/{wb_name} unregistered. BYOVM host was not modified.")
            return

        console.print(f"  [1/2] Destroying {proj_alias}/{wb_name}...")
        if dry_run:
            console.print("    [dry-run] Would run: terraform destroy")
            return

        if use_remote_state:
            s3_bucket = merged_vars.get("s3_bucket", "")
            s3_endpoint = merged_vars.get("s3_endpoint", f"https://storage.{env_region}.nebius.cloud")
            resolved_tf_dir = str(provisioner.prepare_working_dir(
                proj_alias,
                wb_name,
                bucket=s3_bucket,
                region=env_region,
                endpoint=s3_endpoint,
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
                    "gpu_platform": platform,
                    "gpu_preset": preset,
                    "instance_name": instance_name,
                    "server_port": str(port),
                    "enable_preemptible": enable_preemptible,
                    "workbench_type": cloud_init_workbench_type,
                    "fiftyone_version": FIFTYONE_VERSION,
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
            s3_endpoint = merged_vars.get("s3_endpoint", f"https://storage.{env_region}.nebius.cloud")
            resolved_tf_dir = str(provisioner.prepare_working_dir(
                proj_alias,
                wb_name,
                bucket=s3_bucket,
                region=env_region,
                endpoint=s3_endpoint,
            ))
        else:
            resolved_tf_dir = tf_dir

        step += 1
        console.print(f"  [{step}/{total_steps}] Initializing Terraform ({proj_alias}/{wb_name})...")
        if dry_run:
            console.print("    [dry-run] Would run: terraform init")
        else:
            try:
                backend_cfg = (
                    {
                        "access_key": merged_vars.get("nebius_api_key", ""),
                        "secret_key": merged_vars.get("nebius_secret_key", ""),
                    }
                    if use_remote_state else None
                )
                provisioner.init(tf_dir=resolved_tf_dir or None, backend_config=backend_cfg)
            except ProvisionerError as exc:
                _fail(f"Terraform init failed: {exc}")
                return

        step += 1
        all_vars = {
            "gpu_platform": platform,
            "gpu_preset": preset,
            "instance_name": instance_name,
            "server_port": str(port),
            "enable_preemptible": enable_preemptible,
            "workbench_type": cloud_init_workbench_type,
            "fiftyone_version": FIFTYONE_VERSION,
            **merged_vars,
        }
        compute_label = f"gpu={platform}" if uses_gpu else f"cpu={platform}"
        console.print(f"  [{step}/{total_steps}] Applying Terraform ({compute_label}, region={env_region})...")
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
                target = _resolve_byovm_deploy_target(
                    saved_cfg=saved_wb_cfg,
                    host=host,
                    ssh_key=ssh_key,
                    ssh_user=ssh_user,
                )
                bucket = (
                    merged_vars.get("s3_bucket", "")
                    or (saved_wb_cfg.storage.checkpoint_bucket if saved_wb_cfg else "")
                    or os.environ.get("NPA_CHECKPOINT_BUCKET", "")
                )
                storage_ep = (
                    merged_vars.get("s3_endpoint", "")
                    or (saved_wb_cfg.storage.endpoint_url if saved_wb_cfg else "")
                    or os.environ.get("AWS_ENDPOINT_URL", "")
                )
                tf_outputs = workbench_storage_outputs(target=target, bucket=bucket, endpoint=storage_ep)
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
            _fail("No VM IP found. Run without --skip-infra first, or set config manually.")
            return

    vm_ip = tf_outputs.get("vm_ip", "")
    ssh_user = tf_outputs.get("ssh_user", "ubuntu")
    ssh_key = tf_outputs.get("ssh_key_path", "~/.ssh/id_ed25519")
    bucket = tf_outputs.get("storage_bucket", "")
    storage_ep = tf_outputs.get("storage_endpoint", "")
    endpoint = f"http://{vm_ip}:{port}"
    bucket_display = bucket if str(bucket).startswith("s3://") else (f"s3://{bucket}/checkpoints/" if bucket else "")
    byovm_fields = gpu_config_fields(
        byovm_gpu_info,
        effective_count=byovm_effective_gpu_count or None,
        visible_devices=byovm_visible_devices,
    )
    initial_endpoint_strategy = (
        saved_wb_cfg.endpoint_strategy
        if skip_infra and saved_wb_cfg is not None
        else "public"
    )
    workbench_config: dict[str, Any] = {
        "endpoint": endpoint,
        "gpu_platform": byovm_fields.get("gpu_platform", platform),
        "gpu_preset": byovm_fields.get("gpu_preset", preset),
        "tf_instance_name": instance_name,
        "workbench_type": "fiftyone",
        "runtime": runtime.value,
        "endpoint_strategy": initial_endpoint_strategy,
        "service_port": port,
        "app_port": port,
        **byovm_fields,
        "ssh": {"host": vm_ip, "user": ssh_user, "key_path": ssh_key},
        "storage": {"checkpoint_bucket": bucket_display, "endpoint_url": storage_ep},
    }
    if skip_infra and not skip_app:
        if saved_wb_cfg is not None and saved_wb_cfg.app_status:
            workbench_config["app_status"] = saved_wb_cfg.app_status
    else:
        workbench_config["app_status"] = APP_STATUS_PROVISIONED
    config_data: dict[str, Any] = {
        "projects": {
            proj_alias: {
                "project_id": env_project,
                "tenant_id": env_tenant,
                "region": env_region,
                "terraform_state": _terraform_state_config(merged_vars),
                "workbenches": {
                    wb_name: workbench_config,
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

    recorded_endpoint_strategy = initial_endpoint_strategy

    def mark_app_status(app_status: str) -> None:
        if not dry_run:
            update_workbench_app_status(proj_alias, wb_name, app_status)

    def fail_app(msg: str) -> None:
        mark_app_status(APP_STATUS_INSTALL_FAILED)
        _fail(msg)

    if not skip_app:
        mark_app_status(APP_STATUS_INSTALLING)
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
                code, _, _ = ssh.run("echo connected")
            except SSHError as exc:
                fail_app(str(exc))
                return
            if code != 0:
                fail_app(f"SSH connection test failed (exit {code})")
                return

        if runtime_uses_container(runtime):
            step += 1
            console.print(f"  [{step}/{total_steps}] Starting FiftyOne container...")
            if dry_run:
                console.print(f"    [dry-run] Would pull and run the FiftyOne container image on port {port}")
            else:
                from npa.deploy.configurator import (
                    deploy_workbench_container,
                    write_remote_docker_env_file,
                    write_remote_text_file,
                )

                try:
                    service_env = {
                        "FIFTYONE_DEFAULT_APP_ADDRESS": "0.0.0.0",
                        "FIFTYONE_DEFAULT_APP_PORT": str(port),
                        "FIFTYONE_DATABASE_DIR": FIFTYONE_CONTAINER_DB_DIR,
                        "FIFTYONE_DEFAULT_DATASET_DIR": f"{FIFTYONE_HOME}/datasets",
                        "FIFTYONE_DATASET_ZOO_DIR": f"{FIFTYONE_HOME}/zoo/datasets",
                        "FIFTYONE_MODEL_ZOO_DIR": f"{FIFTYONE_HOME}/zoo/models",
                        "FIFTYONE_DO_NOT_TRACK": "true",
                        "FIFTYONE_DATASET_NAME": "",
                        "AWS_ACCESS_KEY_ID": merged_vars.get("nebius_api_key", ""),
                        "AWS_SECRET_ACCESS_KEY": merged_vars.get("nebius_secret_key", ""),
                        "AWS_ENDPOINT_URL": storage_ep,
                        "NEBIUS_S3_ENDPOINT": storage_ep,
                        "NEBIUS_S3_BUCKET": bucket,
                        "NEBIUS_REGION": env_region,
                        "PYTHONUNBUFFERED": "1",
                        **gpu_env_fields(
                            byovm_gpu_info,
                            effective_count=byovm_effective_gpu_count or None,
                            visible_devices=byovm_visible_devices,
                        ),
                    }
                    apply_shared_credential_env(service_env, credentials, include=not no_shared_creds)
                    write_remote_docker_env_file(
                        ssh,
                        "/etc/npa-fiftyone/env",
                        service_env,
                        owner=ssh_user,
                    )
                    write_remote_text_file(
                        ssh,
                        f"{FIFTYONE_HOME}/app.py",
                        _build_app_py(),
                        owner=ssh_user,
                    )
                    image_ref = container_image_for_tool(
                        "fiftyone",
                        registry=resolve_container_registry(proj_alias),
                    )
                    ssh.run("sudo systemctl stop npa-fiftyone-app >/dev/null 2>&1 || true")
                    deploy_workbench_container(
                        ssh,
                        image_ref=image_ref,
                        container_name=FIFTYONE_CONTAINER_NAME,
                        env_file="/etc/npa-fiftyone/env",
                        volumes=[
                            f"{FIFTYONE_HOME}/datasets:{FIFTYONE_HOME}/datasets",
                            f"{FIFTYONE_CONTAINER_DB_DIR}:{FIFTYONE_CONTAINER_DB_DIR}",
                            f"{FIFTYONE_HOME}/zoo:{FIFTYONE_HOME}/zoo",
                            f"{FIFTYONE_HOME}/app.py:{FIFTYONE_HOME}/app.py:ro",
                            "/etc/npa-fiftyone/env:/etc/npa-fiftyone/env:ro",
                        ],
                        work_dirs=[
                            f"{FIFTYONE_HOME}/datasets",
                            FIFTYONE_CONTAINER_DB_DIR,
                            f"{FIFTYONE_HOME}/zoo/datasets",
                            f"{FIFTYONE_HOME}/zoo/models",
                        ],
                        command=(
                            "-lc "
                            + shlex.quote(f"exec {FIFTYONE_VENV}/bin/python {FIFTYONE_HOME}/app.py")
                        ),
                        gpu=uses_gpu,
                        registry_token=merged_vars.get("iam_token", ""),
                    )
                    if verify_env and not no_shared_creds:
                        failed_keys = audit_remote_env(
                            ssh,
                            "/etc/npa-fiftyone/env",
                            shared_credential_env(credentials),
                        )
                        if failed_keys:
                            key = failed_keys[0]
                            fail_app(
                                f"Credential audit failed: {key} missing or mismatched in fiftyone service env. "
                                "Deploy may have skipped shared credential injection."
                            )
                            return
                except SSHError as exc:
                    fail_app(f"FiftyOne container deployment failed: {exc}")
                    return
                mark_app_status(APP_STATUS_PROVISIONED)
        else:
            step += 1
            console.print(f"  [{step}/{total_steps}] Installing FiftyOne {FIFTYONE_VERSION}...")
            if dry_run:
                console.print(f"    [dry-run] Would create {FIFTYONE_VENV}, install FiftyOne, and start port {port}")
            else:
                try:
                    _run_fiftyone_command(ssh, _build_install_command(port), stream=True)
                except SSHError as exc:
                    fail_app(f"FiftyOne installation failed: {exc}")
                    return
                mark_app_status(APP_STATUS_PROVISIONED)

        step += 1
        console.print(f"  [{step}/{total_steps}] HTTP check on {endpoint}...")
        app_ready = False
        if not dry_run:
            health_note = ""
            if health_check_mode == HealthCheckMode.ssh:
                app_ready = health_check_ssh(
                    ssh,
                    port,
                    path="/",
                    retries=FIFTYONE_HEALTH_RETRIES,
                    backoff=FIFTYONE_HEALTH_BACKOFF_SEC,
                )
            else:
                if health_check_mode == HealthCheckMode.auto and byovm:
                    app_ready = _app_health_check(
                        endpoint,
                        retries=FIFTYONE_AUTO_PUBLIC_HEALTH_RETRIES,
                        backoff=FIFTYONE_HEALTH_BACKOFF_SEC,
                    )
                else:
                    app_ready = _app_health_check(endpoint)
                if (
                    not app_ready
                    and health_check_mode == HealthCheckMode.auto
                    and byovm
                    and health_check_ssh(
                        ssh,
                        port,
                        path="/",
                        retries=FIFTYONE_HEALTH_RETRIES,
                        backoff=FIFTYONE_HEALTH_BACKOFF_SEC,
                    )
                ):
                    app_ready = True
                    health_note = f"Public port {port} unreachable; service healthy via SSH on {vm_ip}."
            if app_ready:
                app_ready = True
                console.print("    FiftyOne app is reachable")
                if health_note:
                    console.print(f"    {health_note}")
                endpoint_strategy = (
                    "ssh"
                    if byovm and (health_check_mode == HealthCheckMode.ssh or bool(health_note))
                    else "public"
                )
                recorded_endpoint_strategy = endpoint_strategy
                write_config({
                    "projects": {
                        proj_alias: {
                            "workbenches": {
                                wb_name: {
                                    "endpoint_strategy": endpoint_strategy,
                                    "service_port": port,
                                },
                            },
                        },
                    },
                })
            else:
                timeout_sec = FIFTYONE_HEALTH_RETRIES * FIFTYONE_HEALTH_BACKOFF_SEC
                console.print(
                    f"    [yellow]Warning:[/yellow] FiftyOne app did not respond at "
                    f"{endpoint} within {timeout_sec:.0f}s. Deploy completed; run "
                    "'npa workbench fiftyone status' to check readiness later."
                )

        step += 1
        console.print(f"  [{step}/{total_steps}] Writing deployment manifest...")
        if not dry_run:
            try:
                write_manifest(ssh, tool="fiftyone", version=FIFTYONE_VERSION, deployed_by=f"npa deploy --runtime {runtime.value}")
            except SSHError:
                pass
        if dry_run or app_ready:
            mark_app_status(APP_STATUS_HEALTHY)
        if app_ready and not dry_run:
            ensure_deploy_ingress(
                tool="fiftyone",
                port=port,
                alias=wb_name,
                instance_id=resolve_deploy_instance_id(
                    tf_outputs=tf_outputs,
                    project_alias=proj_alias,
                    name=wb_name,
                ),
                warn=console.print,
            )

    step += 1
    console.print(f"  [{step}/{total_steps}] Updating config status ({proj_alias}/{wb_name})...")
    if not dry_run:
        console.print("    Saved to ~/.npa/config.yaml")

    console.print("")
    console.print(f"[bold green]Deploy complete.[/bold green] ({proj_alias}/{wb_name})")
    console.print(f"  FiftyOne: {_browser_url_for_strategy(endpoint, recorded_endpoint_strategy)}")
    console.print(f"  SSH:      ssh -i {ssh_key} {ssh_user}@{vm_ip}")
    console.print("")
    console.print(f"  Try: npa workbench fiftyone -p {proj_alias} -n {wb_name} launch")

    if output == OutputFormat.json:
        typer.echo(json.dumps({
            "project": proj_alias,
            "name": wb_name,
            "endpoint": endpoint,
            "browser_url": _browser_url_for_strategy(endpoint, recorded_endpoint_strategy),
            "vm_ip": vm_ip,
            "ssh_user": ssh_user,
            "gpu_platform": platform,
            "gpu_preset": preset,
            "uses_gpu": uses_gpu,
            "runtime": runtime.value,
            "app_port": port,
            "tf_outputs": tf_outputs,
        }, indent=2))


@app.command("launch")
def launch_cmd(
    port: int = typer.Option(DEFAULT_APP_PORT, "--port", help="FiftyOne app port."),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Start the FiftyOne app over SSH and print the browser URL."""
    cfg = _get_ssh_config()
    ssh = SSHClient(cfg.ssh)
    url = _endpoint_for_port(cfg.endpoint, cfg.ssh.host, port)
    browser_url = _browser_url_for_config(cfg, url)
    command = (
        _build_container_launch_command(port)
        if _is_container_runtime(cfg)
        else _build_launch_command(port)
    )

    try:
        _, out, err = _run_fiftyone_command(ssh, command)
    except SSHError as exc:
        _fail(f"SSH error: {exc}")
        return

    result: dict[str, Any] = {
        "status": "running",
        "url": url,
        "browser_url": browser_url,
        "port": port,
    }
    if output == OutputFormat.json and out.strip():
        result["stdout_tail"] = out.strip()[-1000:]
    if err.strip():
        result["stderr_tail"] = err.strip()[-1000:]

    if output == OutputFormat.json:
        typer.echo(json.dumps(result, indent=2))
    else:
        typer.echo(f"  FiftyOne URL: {browser_url}")


@app.command("load-dataset")
def load_dataset_cmd(
    name: str = typer.Option(..., "--name", help="FiftyOne dataset name."),
    input_path: str = typer.Option(
        "",
        "--input-path",
        "-i",
        help="S3 URI or Hugging Face Hub dataset ID/URL.",
    ),
    # Deprecated path alias: keep --source working for existing scripts.
    source: str = typer.Option("", "--source", hidden=True),
    dataset_format: DatasetFormat = typer.Option(
        DatasetFormat.auto,
        "--format",
        help="Dataset format parser.",
    ),
    output_path: str = typer.Option("", "--output-path", help="S3 URI where serverless load artifacts are written."),
    runtime: WorkbenchRuntime = typer.Option(WorkbenchRuntime.vm, "--runtime", help="Runtime. serverless creates a Nebius AI Job."),
    project_id: str = typer.Option("", "--project-id", help="Nebius project ID for serverless Jobs."),
    image: str = typer.Option("", "--image", help="Container image for the serverless Job."),
    gpu_type: str = typer.Option("l40s", "--gpu-type", help="GPU type for serverless Jobs."),
    gpu_count: int = typer.Option(1, "--gpu-count", help="GPU count for serverless Jobs."),
    gpu_preset: str = typer.Option("", "--gpu-preset", help="Nebius GPU preset override."),
    subnet_id: str = typer.Option("", "--subnet-id", help="Nebius VPC subnet ID for serverless Jobs."),
    job_name: str = typer.Option("", "--job-name", help="Explicit serverless Job name."),
    submit_only: bool = typer.Option(False, "--submit-only", help="Submit serverless Job and return before polling."),
    poll_interval: float = typer.Option(30.0, "--poll-interval", help="Seconds between serverless status checks."),
    timeout: float = typer.Option(3600.0, "--timeout", help="Seconds to wait for serverless completion."),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Load a dataset into FiftyOne on the VM."""
    if not name.strip():
        _fail("--name must not be empty")
    dataset_source = input_path or source
    if not dataset_source.strip():
        _fail("--input-path must not be empty")
    if _is_serverless_runtime(runtime):
        _fiftyone_serverless_load_dataset(
            name=name.strip(),
            dataset_source=dataset_source.strip(),
            dataset_format=dataset_format,
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
            output=output,
        )
        return
    try:
        dataset_source = validate_read_path(
            dataset_source,
            tool="FiftyOne load-dataset",
            option="--input-path",
            allow_hf=True,
            vm_local_message=FIFTYONE_LOAD_DATASET_VM_LOCAL_ERROR,
        )
    except PathContractError as exc:
        _fail(str(exc))

    cfg = _get_ssh_config()
    ssh = SSHClient(cfg.ssh)
    command = (
        _build_container_load_dataset_command(name.strip(), dataset_source.strip(), dataset_format)
        if _is_container_runtime(cfg)
        else _build_load_dataset_command(name.strip(), dataset_source.strip(), dataset_format)
    )

    try:
        _, out, err = _run_fiftyone_command(
            ssh,
            command,
            stream=output != OutputFormat.json,
        )
    except SSHError as exc:
        _fail(f"SSH error: {exc}")
        return

    if output == OutputFormat.json:
        filtered_err = _suppress_transient_curl_errors(err)
        parsed = _parse_first_json_object(out) if out.strip() else None
        typer.echo(json.dumps(parsed or {
            "status": "loaded",
            "name": name.strip(),
            "source": dataset_source.strip(),
            "stdout_tail": out.strip()[-1000:],
            "stderr_tail": filtered_err[-1000:] if filtered_err else "",
        }, indent=2))
    else:
        if out.strip():
            typer.echo(out.strip())
        filtered_err = _suppress_transient_curl_errors(err)
        if filtered_err:
            console.print(f"[red]stderr:[/red]\n{filtered_err[-500:]}")


@app.command("restart")
def restart_cmd(
    port: int = typer.Option(DEFAULT_APP_PORT, "--port", help="FiftyOne app port."),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Restart the FiftyOne app or container without redeploying."""
    cfg = _get_ssh_config()
    ssh = SSHClient(cfg.ssh)
    command = (
        _build_container_restart_command(port)
        if _is_container_runtime(cfg)
        else _build_restart_command(port)
    )
    url = _endpoint_for_port(cfg.endpoint, cfg.ssh.host, port)
    browser_url = _browser_url_for_config(cfg, url)

    try:
        _, out, err = _run_fiftyone_command(ssh, command, stream=output != OutputFormat.json)
    except SSHError as exc:
        _fail(f"SSH error: {exc}")
        return

    if _project_alias and _workbench_name:
        update_workbench_app_status(_project_alias, _workbench_name, APP_STATUS_HEALTHY)

    result: dict[str, Any] = {
        "status": "restarted",
        "url": url,
        "browser_url": browser_url,
        "port": port,
    }
    if output == OutputFormat.json:
        if out.strip():
            result["stdout_tail"] = out.strip()[-1000:]
        if err.strip():
            result["stderr_tail"] = err.strip()[-1000:]
        typer.echo(json.dumps(result, indent=2))
    else:
        typer.echo(f"  status: restarted")
        typer.echo(f"  FiftyOne URL: {browser_url}")


@datasets_app.command("list")
def datasets_list_cmd(
    port: int = typer.Option(DEFAULT_APP_PORT, "--port", help="FiftyOne app port."),
    first: int = typer.Option(100, "--first", min=1, help="Maximum datasets to return."),
    search: str = typer.Option("", "--search", help="Filter dataset names."),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """List FiftyOne datasets through the app GraphQL API."""
    cfg = _get_ssh_config()
    url = _endpoint_for_port(cfg.endpoint, cfg.ssh.host, port)
    try:
        with service_endpoint(cfg, default_port=port, endpoint=url, service_port=port) as active:
            resp = httpx.post(
                _graphql_url(active.url),
                json={
                    "query": FIFTYONE_DATASETS_QUERY,
                    "variables": {"first": first, "search": search},
                },
                timeout=10.0,
            )
            resp.raise_for_status()
            payload = resp.json()
            active_url = active.url
    except EndpointError as exc:
        _fail(f"Cannot prepare FiftyOne endpoint for {url}: {exc}")
        return
    except httpx.HTTPError as exc:
        _fail(f"Cannot query FiftyOne GraphQL API at {url}: {exc}")
        return
    except ValueError as exc:
        _fail(f"FiftyOne GraphQL API returned invalid JSON: {exc}")
        return

    if payload.get("errors"):
        _fail(f"FiftyOne GraphQL API returned errors: {payload['errors']}")
        return

    total, datasets = _parse_dataset_edges(payload)
    result = {
        "status": "ok",
        "url": active_url,
        "browser_url": _browser_url_for_config(cfg, url),
        "total": total,
        "datasets": datasets,
    }
    if output == OutputFormat.json:
        typer.echo(json.dumps(result, indent=2))
        return

    typer.echo(f"  url: {active_url}")
    typer.echo(f"  total: {total}")
    if not datasets:
        typer.echo("  datasets: []")
        return
    for dataset in datasets:
        typer.echo(
            "  "
            f"{dataset['name']}  samples={dataset['samples']}  "
            f"media_type={dataset['media_type'] or '?'}  "
            f"persistent={dataset['persistent']}"
        )


@app.command("status")
def status_cmd(
    port: int = typer.Option(DEFAULT_APP_PORT, "--port", help="FiftyOne app port."),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Check whether the FiftyOne app responds on its web port."""
    cfg = _get_ssh_config()
    url = _endpoint_for_port(cfg.endpoint, cfg.ssh.host, port)

    try:
        with service_endpoint(cfg, default_port=port, endpoint=url, service_port=port) as active:
            resp = httpx.get(active.url, timeout=5.0)
            active_url = active.url
    except EndpointError as exc:
        if output == OutputFormat.json:
            typer.echo(json.dumps({
                "url": url,
                "app_status": "unreachable",
                "server": "down",
                "error": str(exc),
            }, indent=2))
        else:
            typer.echo(f"  url: {url}")
            typer.echo("  app_status: unreachable")
        _fail(f"Cannot prepare FiftyOne endpoint for {url}: {exc}")
        return
    except httpx.HTTPError as exc:
        if output == OutputFormat.json:
            typer.echo(json.dumps({
                "url": url,
                "app_status": "unreachable",
                "server": "down",
                "error": str(exc),
            }, indent=2))
        else:
            typer.echo(f"  url: {url}")
            typer.echo("  app_status: unreachable")
        _fail(f"Cannot reach FiftyOne app at {url}: {exc}")
        return

    if resp.status_code >= 400:
        if output == OutputFormat.json:
            typer.echo(json.dumps({
                "url": url,
                "app_status": "unreachable",
                "server": "error",
                "status_code": resp.status_code,
            }, indent=2))
        else:
            typer.echo(f"  url: {url}")
            typer.echo("  app_status: unreachable")
        _fail(f"FiftyOne app at {url} returned HTTP {resp.status_code}")
        return

    result: dict[str, Any] = {
        "url": active_url,
        "browser_url": _browser_url_for_config(cfg, url),
        "app_status": "healthy",
        "runtime": getattr(cfg, "runtime", "vm"),
        "server": "up",
        "status_code": resp.status_code,
    }
    if _is_container_runtime(cfg):
        ssh = SSHClient(cfg.ssh)
        code, out, _ = ssh.run(
            "sudo docker inspect -f 'state={{.State.Status}} image={{.Config.Image}}' npa-fiftyone 2>/dev/null || true"
        )
        if code == 0 and out.strip():
            result["container"] = out.strip()
    _output(result, output)


@app.command("system-info")
def system_info_cmd(
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Collect and display system hardware information from the FiftyOne VM."""
    cfg = _get_ssh_config()
    ssh = SSHClient(cfg.ssh)
    info_cmd = (
        "echo '=== nvidia-smi ===' && (nvidia-smi || echo 'nvidia-smi not available') && "
        "echo '' && echo '=== lscpu ===' && lscpu && "
        "echo '' && echo '=== free -h ===' && free -h && "
        "echo '' && echo '=== lsblk ===' && lsblk"
    )
    if _is_container_runtime(cfg):
        info_cmd += (
            " && echo '' && echo '=== container ===' && "
            "sudo docker inspect -f 'state={{.State.Status}} image={{.Config.Image}}' npa-fiftyone"
        )

    try:
        _, out, err = ssh.run_or_raise(info_cmd)
    except SSHError as exc:
        _fail(f"SSH error: {exc}")
        return

    if output == OutputFormat.json:
        typer.echo(json.dumps({
            "host": cfg.ssh.host,
            "runtime": getattr(cfg, "runtime", "vm"),
            "system_info": out.strip(),
        }, indent=2))
    else:
        if out:
            typer.echo(out.strip())
        if err:
            console.print(f"[red]stderr:[/red]\n{err.strip()[-500:]}")
