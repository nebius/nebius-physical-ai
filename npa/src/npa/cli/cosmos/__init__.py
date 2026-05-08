"""npa workbench cosmos - NVIDIA Cosmos model serving endpoints."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import shlex
import tempfile
import time
import uuid
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import typer
from rich.console import Console

from npa.clients.config import (
    APP_STATUS_HEALTHY,
    APP_STATUS_INSTALL_FAILED,
    APP_STATUS_INSTALLING,
    APP_STATUS_PROVISIONED,
    ConfigError,
    SSHConfig,
    default_project_name,
    default_workbench_name,
    list_projects,
    remove_workbench_config,
    resolve_config,
    resolve_container_registry,
    resolve_credentials,
    resolve_environment,
    resolve_ssh_config,
    resolve_terraform_state,
    update_workbench_app_status,
    write_config,
)
from npa.clients.http import HTTPClient, ServerError
from npa.clients.ssh import SSHClient, SSHError
from npa.deploy import provisioner
from npa.deploy.configurator import docker_exec_cmd, health_check, write_manifest
from npa.deploy.byovm import (
    RUNTIME_HELP,
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

app = typer.Typer(
    name="cosmos",
    help="NVIDIA Cosmos world model serving and inference endpoints.",
    no_args_is_help=True,
)

console = Console(stderr=True)

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
COSMOS_PIP_EXTRA_INDEX_URL = "https://nvidia-cosmos.github.io/cosmos-dependencies/cu126_torch260/simple"
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


def _model_slug(model: str) -> str:
    return model.replace("/", "--").replace(":", "--")


def _is_s3_uri(path: str) -> bool:
    return path.startswith("s3://")


def _s3_path_name(path: str, default: str = "result.json") -> str:
    name = Path(urlparse(path).path.rstrip("/")).name if _is_s3_uri(path) else Path(path).name
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
        _fail("Missing --gpu-preset. Provide the Nebius GPU preset that matches the selected GPU type.")


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
DISABLE_SAFETY = os.environ.get("COSMOS_DISABLE_SAFETY", "1").strip().lower() not in {{"0", "false", "no"}}

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


def _build_install_command(model: str, port: int) -> str:
    server_py = _build_server_py(model)
    model_slug = _model_slug(model)
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
COSMOS_DISABLE_SAFETY=1
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


def _build_serve_command(model: str, port: int) -> str:
    server_py = _build_server_py(model)
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
COSMOS_DISABLE_SAFETY=1
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


@app.command("list")
def list_cmd(
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """List configured Cosmos workbenches."""
    projects = list_projects()
    def_proj = default_project_name()
    def_wb = default_workbench_name()

    if output == OutputFormat.json:
        filtered = {}
        for pname, pcfg in projects.items():
            wbs = {
                k: v for k, v in pcfg.get("workbenches", {}).items()
                if _is_cosmos_workbench(k, v)
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
        typer.echo("No projects configured. Run 'npa workbench cosmos deploy' to create one.")
        return

    any_shown = False
    for proj_name, proj_cfg in projects.items():
        workbenches = {
            k: v for k, v in proj_cfg.get("workbenches", {}).items()
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
        typer.echo("No Cosmos workbenches configured. Run 'npa workbench cosmos deploy' to create one.")


@app.command("deploy")
def deploy_cmd(
    gpu_type: str = typer.Option("", "--gpu-type", help="Nebius GPU platform."),
    gpu_preset: str = typer.Option("", "--gpu-preset", help="Nebius GPU preset."),
    region: str = typer.Option("", "--region", help="Nebius region."),
    project_id: str = typer.Option("", "--project-id", help="Nebius project ID."),
    tenant_id: str = typer.Option("", "--tenant-id", help="Nebius tenant ID."),
    tf_dir: str = typer.Option("", "--tf-dir", help="Path to Terraform directory (default: bundled)."),
    tf_var: list[str] = typer.Option([], "--tf-var", "-v", help="Extra TF variable (key=value), repeatable."),
    skip_infra: bool = typer.Option(False, "--skip-infra", help="Skip Terraform, only deploy the app."),
    skip_app: bool = typer.Option(False, "--skip-app", help="Skip app installation, only provision infra."),
    destroy: bool = typer.Option(False, "--destroy", help="Destroy infrastructure and clean up config."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would happen without doing it."),
    model: str = typer.Option(DEFAULT_MODEL, "--model", help="Hugging Face Cosmos model ID to download and serve."),
    backend: Backend = typer.Option(
        Backend.basic,
        "--backend",
        help=(
            "Serving backend: basic uses the built-in FastAPI/Diffusers server; "
            "nim will use NVIDIA NIM containers; triton will use Triton/TensorRT model serving."
        ),
    ),
    server_port: int = typer.Option(8080, "--server-port", help="Cosmos server port on the VM."),
    preemptible: bool = typer.Option(True, "--preemptible/--no-preemptible", help="Preemptible (spot) instance."),
    runtime: WorkbenchRuntime = typer.Option(WorkbenchRuntime.vm, "--runtime", help=RUNTIME_HELP),
    host: str = typer.Option("", "--host", help="BYOVM SSH host/IP. Used only with --runtime byovm."),
    ssh_key: str = typer.Option("", "--ssh-key", help="BYOVM SSH private key path. Used only with --runtime byovm."),
    ssh_user: str = typer.Option("", "--ssh-user", help="BYOVM SSH username. Defaults to ubuntu."),
    gpu_count: int = typer.Option(0, "--gpu-count", help="Limit visible GPUs on BYOVM (0 = all detected)."),
    disk_size: int | None = typer.Option(None, "--disk-size", help="Boot disk size in GiB. Defaults to 250 for container runtime; VM runtime keeps the Terraform default."),
    default: bool = typer.Option(False, "--default", help="Set this workbench as the default."),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Deploy or destroy a Cosmos model serving VM."""
    _ensure_basic_backend(backend)
    byovm = is_byovm_runtime(runtime)
    if not destroy and not byovm:
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
        proj_alias = env_region or ("byovm" if byovm else "default")

    credentials = resolve_credentials()
    if not destroy and not skip_app and not credentials.hf_token:
        _fail(
            "HF_TOKEN required for Cosmos deployment. Add it to ~/.npa/credentials.yaml "
            "under tokens: or set HF_TOKEN as an environment variable."
        )
        return

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

    instance_name = f"cosmos-{proj_alias}-{wb_name}"

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
            "gpu_platform": gpu_type,
            "gpu_preset": gpu_preset,
            "instance_name": instance_name,
            "workbench_type": "cosmos",
            "enable_preemptible": "true" if preemptible else "false",
            **merged_vars,
        }
        console.print(f"  [{step}/{total_steps}] Applying Terraform (gpu={gpu_type}, region={env_region})...")
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
                target = resolve_byovm_target(host=host, ssh_key=ssh_key, ssh_user=ssh_user)
                bucket = merged_vars.get("s3_bucket", "") or os.environ.get("NPA_CHECKPOINT_BUCKET", "")
                storage_ep = merged_vars.get("s3_endpoint", "") or os.environ.get("AWS_ENDPOINT_URL", "")
                tf_outputs = workbench_storage_outputs(target=target, bucket=bucket, endpoint=storage_ep)
                if not dry_run:
                    ssh = SSHClient(ssh_config_for_target(target, tokens=credentials.tokens))
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
    endpoint = f"http://{vm_ip}:{server_port}"
    bucket_display = bucket if str(bucket).startswith("s3://") else (f"s3://{bucket}/checkpoints/" if bucket else "")
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
                        "model": model,
                        "backend": backend.value,
                        **byovm_fields,
                        "ssh": {"host": vm_ip, "user": ssh_user, "key_path": ssh_key},
                        "storage": {"checkpoint_bucket": bucket_display, "endpoint_url": storage_ep},
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
        ssh_cfg = SSHConfig(host=vm_ip, user=ssh_user, key_path=ssh_key, tokens=credentials.tokens)

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
            console.print(f"  [{step}/{total_steps}] Starting Cosmos container and preparing {model}...")
            if dry_run:
                console.print("    [dry-run] Would pull and run the Cosmos container image")
            else:
                from npa.deploy.configurator import deploy_workbench_container, write_remote_env_file

                try:
                    write_remote_env_file(
                        ssh,
                        "/etc/npa-cosmos-server/env",
                        {
                            "COSMOS_MODEL_ID": model,
                            "COSMOS_MODEL_DIR": COSMOS_MODEL_DIR,
                            "COSMOS_OUTPUT_DIR": COSMOS_OUTPUT_DIR,
                            "COSMOS_SERVER_PORT": str(server_port),
                            "COSMOS_DISABLE_SAFETY": "1",
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
                        },
                        owner=ssh_user,
                    )
                    image_ref = container_image_for_tool(
                        "cosmos",
                        registry=resolve_container_registry(proj_alias),
                    )
                    ssh.run("sudo systemctl stop npa-cosmos-server >/dev/null 2>&1 || true")
                    deploy_workbench_container(
                        ssh,
                        image_ref=image_ref,
                        container_name=COSMOS_CONTAINER_NAME,
                        env_file="/etc/npa-cosmos-server/env",
                        volumes=[
                            f"{COSMOS_DATA_HOME}:{COSMOS_DATA_HOME}",
                            "/etc/npa-cosmos-server/env:/etc/npa-cosmos-server/env:ro",
                        ],
                        work_dirs=[COSMOS_MODEL_DIR, COSMOS_HF_CACHE, COSMOS_OUTPUT_DIR],
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
                        "set -a; . /etc/npa-cosmos-server/env; set +a; "
                        f"if [ -n \"${{HF_TOKEN:-}}\" ]; then "
                        f"huggingface-cli download {shlex.quote(model)} --local-dir {COSMOS_MODEL_DIR}/{model_slug} --token \"$HF_TOKEN\"; "
                        f"else huggingface-cli download {shlex.quote(model)} --local-dir {COSMOS_MODEL_DIR}/{model_slug}; fi"
                    )
                    ssh.run_or_raise(docker_exec_cmd(COSMOS_CONTAINER_NAME, download_cmd), stream=True)
                    ssh.run_or_raise(f"sudo docker restart {COSMOS_CONTAINER_NAME}")
                except SSHError as exc:
                    fail_app(f"Cosmos container deployment failed: {exc}")
                    return
        else:
            step += 1
            console.print(f"  [{step}/{total_steps}] Installing Cosmos serving stack and downloading {model}...")
            if dry_run:
                console.print("    [dry-run] Would create /opt/cosmos/venv, install Cosmos dependencies, and download model weights")
            else:
                try:
                    ssh.run_or_raise(_build_install_command(model, server_port), stream=True)
                except SSHError as exc:
                    fail_app(f"Cosmos installation failed: {exc}")
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
                write_manifest(ssh, tool="cosmos", version=COSMOS_VERSION, deployed_by=f"npa deploy --runtime {runtime.value}")
            except SSHError:
                pass
        mark_app_status(APP_STATUS_HEALTHY)

    step += 1
    console.print(f"  [{step}/{total_steps}] Updating config status ({proj_alias}/{wb_name})...")
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
        typer.echo(json.dumps({
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
        }, indent=2))


@app.command("serve")
def serve_cmd(
    model: str = typer.Option(DEFAULT_MODEL, "--model", help="Hugging Face Cosmos model ID to serve."),
    backend: Backend = typer.Option(
        Backend.basic,
        "--backend",
        help=(
            "Serving backend: basic restarts the built-in FastAPI/Diffusers server; "
            "nim will run NVIDIA NIM; triton will run Triton/TensorRT serving."
        ),
    ),
    port: int = typer.Option(8080, "--port", help="Server port."),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Start or restart the Cosmos model server over SSH."""
    _ensure_basic_backend(backend)
    cfg = _get_config()

    if output != OutputFormat.json:
        console.print(f"[bold]Restarting Cosmos server[/bold]: {model}")

    out = ""
    err = ""
    if runtime_uses_container(getattr(cfg, "runtime", "vm")):
        try:
            served = HTTPClient(cfg.endpoint, timeout=120.0, retries=1).serve_model(model, timeout=120.0)
            out = json.dumps(served)
        except ServerError as exc:
            _fail(f"Cosmos serve request failed: {exc}")
            return
    else:
        ssh = SSHClient(cfg.ssh)
        try:
            _, out, err = ssh.run_or_raise(_build_serve_command(model, port))
        except SSHError as exc:
            _fail(f"SSH error: {exc}")
            return

    result: dict[str, Any] = {
        "status": "serving",
        "model": model,
        "port": port,
        "endpoint": cfg.endpoint,
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


@app.command(
    "optimize",
    help="Roadmap placeholder for TensorRT compilation and quantization of Cosmos models.",
)
def optimize_cmd() -> None:
    """TensorRT compilation and quantization for Cosmos model serving."""
    typer.echo("not yet implemented")
    raise typer.Exit(1)


def _storage_client_for_config(cfg: Any):
    from npa.clients.storage import StorageClient

    return StorageClient.from_environment(
        endpoint_url=cfg.storage.endpoint_url,
        aws_access_key_id=cfg.storage.aws_access_key_id,
        aws_secret_access_key=cfg.storage.aws_secret_access_key,
    )


def _resolve_infer_input(
    input_path: str,
    cfg: Any,
    temp_dirs: list[tempfile.TemporaryDirectory[str]],
) -> Path | None:
    if not input_path:
        return None
    if not _is_s3_uri(input_path):
        return Path(input_path)

    tmp = tempfile.TemporaryDirectory(prefix="npa-cosmos-input-")
    temp_dirs.append(tmp)
    downloaded = Path(_storage_client_for_config(cfg).download_path(input_path, tmp.name))
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
            "mime_type": mimetypes.guess_type(str(input_path))[0] or "application/octet-stream",
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
    return _storage_client_for_config(cfg).upload_file(str(local_path), output_path)


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
) -> str:
    if _is_s3_uri(output_path):
        tmp = tempfile.TemporaryDirectory(prefix="npa-cosmos-output-")
        temp_dirs.append(tmp)
        local_path = Path(tmp.name) / (Path(remote_path).name or f"cosmos-output-{uuid.uuid4().hex}")
        SSHClient(cfg.ssh).download_file(remote_path, str(local_path))
        return _storage_client_for_config(cfg).upload_file(str(local_path), output_path)

    local_path = _local_output_path(remote_path, output_path)
    return SSHClient(cfg.ssh).download_file(remote_path, str(local_path))


def _poll_inference_job(
    client: HTTPClient,
    job_id: str,
    *,
    deadline: float,
    poll_interval: float,
    output_format: OutputFormat,
    initial_status: str = "",
) -> dict[str, Any]:
    last_status = initial_status.lower()
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _fail(f"Inference timed out waiting for job {job_id}")

        try:
            data = client.job_status(job_id, timeout=min(COSMOS_INFER_HTTP_TIMEOUT, max(1.0, remaining)))
        except ServerError as exc:
            _fail(f"Inference status check failed: {exc}")

        status = str(data.get("status", "")).lower()
        if output_format != OutputFormat.json and status != last_status:
            typer.echo(f"  job_status: {status}")
            last_status = status

        if status == "completed":
            return data
        if status == "failed":
            _fail(f"Inference job failed: {data.get('error', 'unknown error')}")
        if status not in {"running", "queued", "pending"}:
            _fail(f"Inference job {job_id} returned unknown status: {status or '<missing>'}")

        sleep_for = min(poll_interval, max(0.0, deadline - time.monotonic()))
        if sleep_for > 0:
            time.sleep(sleep_for)


@app.command("infer")
def infer_cmd(
    prompt: str = typer.Option("", "--prompt", help="Text prompt for text-to-world generation."),
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
        help="S3 URI or local path where the generated output file is saved.",
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
    output_format: OutputFormat = typer.Option(OutputFormat.text, "--output-format", help="CLI output format."),
) -> None:
    """Submit a Cosmos inference job, poll until completion, then download the output."""
    cfg = _get_config()
    temp_dirs: list[tempfile.TemporaryDirectory[str]] = []
    try:
        payload = _build_infer_payload(prompt, _resolve_infer_input(input_path, cfg, temp_dirs))
        client = HTTPClient(cfg.endpoint, timeout=COSMOS_INFER_HTTP_TIMEOUT, retries=1)
        deadline = time.monotonic() + timeout

        try:
            submitted = client.infer(payload, timeout=min(COSMOS_INFER_HTTP_TIMEOUT, max(1.0, timeout)))
        except ServerError as exc:
            _fail(f"Inference submit failed: {exc}")
            return

        job_id = str(submitted.get("job_id") or "")
        if not job_id:
            _fail(f"Inference submit response did not include job_id: {submitted}")
        if output_format != OutputFormat.json:
            typer.echo(f"  job_id: {job_id}")
            typer.echo(f"  job_status: {submitted.get('status', 'unknown')}")

        data = _poll_inference_job(
            client,
            job_id,
            deadline=deadline,
            poll_interval=poll_interval,
            output_format=output_format,
            initial_status=str(submitted.get("status", "")),
        )

        result = {**data, "job_id": job_id}
        remote_output_path = str(data.get("output_path") or "")
        if remote_output_path:
            downloaded_to = _download_remote_output(remote_output_path, output_path, cfg, temp_dirs)
            result["downloaded_to"] = downloaded_to
            if _is_s3_uri(downloaded_to):
                result["saved_to"] = downloaded_to
        elif output_path:
            saved_to = _save_inference_output(result, output_path, cfg, temp_dirs)
            if saved_to:
                result["saved_to"] = saved_to
        _output(result, output_format)
    finally:
        for tmp in temp_dirs:
            tmp.cleanup()


@app.command("status")
def status_cmd(
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Check the Cosmos endpoint health."""
    cfg = _get_config()
    client = HTTPClient(cfg.endpoint, timeout=10.0, retries=1)

    try:
        data = client.health()
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
        _fail(f"Cannot reach Cosmos endpoint at {cfg.endpoint}/health: {exc}")
        return

    result = {
        "endpoint": cfg.endpoint,
        "app_status": cfg.app_status or "unknown",
        "runtime": getattr(cfg, "runtime", "vm"),
        "server": "up",
        **data,
    }
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
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
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
