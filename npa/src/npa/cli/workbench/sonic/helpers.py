"""Shared helpers for the SONIC Workbench CLI."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import re
import shlex
from typing import Any

import typer
from rich.console import Console

from npa.clients.config import (
    ConfigError,
    default_project_name,
    default_workbench_name,
    list_projects,
    resolve_container_registry,
    resolve_environment,
    resolve_project_storage,
)
from npa.clients.credentials import load_credentials, shared_credential_env
from npa.deploy.images import container_image_for_tool
from npa.serverless_common import build_serverless_job_env, split_serverless_env

console = Console(stderr=True)

DEFAULT_MODEL_REPO = "nvidia/GEAR-SONIC"
DEFAULT_CHECKPOINT = "nvidia/GEAR-SONIC:sonic_release/last.pt"
DEFAULT_EMBODIMENT = "unitree-g1"
DEFAULT_EMBODIMENT_TAG = "UNITREE_G1_SONIC"
SONIC_VERSION = "0.1.0"
SONIC_CONTAINER_NAME = "npa-sonic"
SONIC_REPO_URL = "https://github.com/NVlabs/GR00T-WholeBodyControl.git"

EXPECTED_HF_ARTIFACTS = (
    "model_encoder.onnx",
    "model_decoder.onnx",
    "observation_config.yaml",
    "planner_sonic.onnx",
)

_project_alias = ""
_workbench_name = ""


class OutputFormat(str, Enum):
    text = "text"
    json = "json"


class WorkbenchRuntime(str, Enum):
    vm = "vm"
    container = "container"
    byovm = "byovm"
    serverless = "serverless"


class TrainRuntime(str, Enum):
    vm = "vm"
    container = "container"
    serverless = "serverless"


class DeployMode(str, Enum):
    sim = "sim"
    real = "real"
    zmq = "zmq"
    vr = "vr"
    keyboard = "keyboard"
    gamepad = "gamepad"


class ServeMode(str, Enum):
    sim = "sim"
    real = "real"


class InputType(str, Enum):
    keyboard = "keyboard"
    gamepad = "gamepad"
    zmq = "zmq"
    vr = "vr"
    zmq_manager = "zmq_manager"


class CheckpointSource(str, Enum):
    hf = "hf"
    local = "local"
    s3 = "s3"


@dataclass(frozen=True)
class SonicContext:
    project: str
    name: str


def set_context(project: str, name: str) -> None:
    global _project_alias, _workbench_name
    _project_alias = project
    _workbench_name = name


def context() -> SonicContext:
    return SonicContext(
        project=_project_alias or default_project_name(),
        name=_workbench_name or default_workbench_name(),
    )


def fail(msg: str, code: int = 1) -> None:
    console.print(f"[red]Error:[/red] {msg}")
    raise typer.Exit(code)


def output(data: dict[str, Any], fmt: OutputFormat) -> None:
    if fmt == OutputFormat.json:
        typer.echo(json.dumps(data, indent=2))
        return
    for key, value in data.items():
        typer.echo(f"  {key}: {value}")


def enum_value(value: Any) -> str:
    return str(getattr(value, "value", value))


def validate_port(value: int, name: str) -> int:
    if value < 1024 or value > 65535:
        fail(f"{name} must be between 1024 and 65535, got {value}")
    return value


def validate_checkpoint_args(source: CheckpointSource, checkpoint_path: str) -> None:
    if source in {CheckpointSource.local, CheckpointSource.s3} and not checkpoint_path:
        fail(f"--checkpoint-path is required when --checkpoint-source is {source.value}.")


def validate_tensorrt_version(version: str) -> None:
    if not version:
        return
    if not (version.startswith("10.13") or version.startswith("10.7")):
        fail("--tensorrt-version must be TensorRT 10.13.x for x86_64 or 10.7.x for Jetson.")


def require_real_confirmation(mode: str, confirm_real: bool) -> None:
    if mode == "real" and not confirm_real:
        fail("real robot mode requires --confirm-real; no automated real-robot launch is allowed.")


def normalize_embodiment(value: str) -> str:
    raw = (value or DEFAULT_EMBODIMENT).strip()
    normalized = raw.lower().replace("_", "-")
    if normalized in {"unitree-g1", "g1", "unitree-g1-sonic"}:
        return DEFAULT_EMBODIMENT_TAG
    if not re.fullmatch(r"[A-Za-z0-9_.:-]+", raw):
        fail(f"Invalid SONIC embodiment tag: {value}")
    return raw.upper().replace("-", "_")


def is_sonic_workbench(name: str, wb_cfg: dict[str, Any]) -> bool:
    wtype = str(wb_cfg.get("workbench_type", "")).lower()
    if wtype in {"sonic", "sonic-container"}:
        return True
    return "sonic" in name.lower()


def sonic_workbenches() -> dict[str, dict[str, Any]]:
    filtered: dict[str, dict[str, Any]] = {}
    for project, project_cfg in list_projects().items():
        workbenches = project_cfg.get("workbenches", {}) if isinstance(project_cfg, dict) else {}
        if not isinstance(workbenches, dict):
            continue
        sonic_entries = {
            name: cfg
            for name, cfg in workbenches.items()
            if isinstance(cfg, dict) and is_sonic_workbench(name, cfg)
        }
        if sonic_entries:
            filtered[project] = {"workbenches": sonic_entries}
    return filtered


def serverless_job_name(project: str, name: str, tool: str = "sonic") -> str:
    raw = f"npa-{tool}-jobs-{project}-{name}".lower()
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9-]+", "-", raw)).strip("-")[:63]


def remote_bash(script: str) -> str:
    return f"bash -lc {shlex.quote(script)}"


def serverless_job_env(
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


def resolve_project_id(explicit_project_id: str) -> str:
    ctx = context()
    env_cfg = resolve_environment(ctx.project)
    project_id = explicit_project_id or (env_cfg.project_id if env_cfg else "")
    if not project_id:
        fail("SONIC --runtime serverless requires --project-id or a configured project.")
    return project_id


def sonic_image(project: str, image: str = "") -> str:
    if image:
        return image
    try:
        registry = resolve_container_registry(project)
    except ConfigError:
        registry = ""
    return container_image_for_tool("sonic", registry=registry or None) if registry else container_image_for_tool("sonic")
