"""npa workbench isaac-lab - Isaac Lab deployment and remote execution."""

from __future__ import annotations

import json
import logging
import os
import shlex
import tarfile
import tempfile
import time
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import typer
from rich.console import Console

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
    resolve_credentials,
    resolve_environment,
    resolve_container_registry,
    resolve_ssh_config,
    update_workbench_app_status,
    workbench_is_byovm,
    write_config,
)
from npa.clients.credentials import apply_shared_credential_env
from npa.clients.project_credentials import storage_client_for_project
from npa.clients.scoped_credentials import (
    bucket_from_s3_uri,
    run_with_host_credential_fallback,
)
from npa.clients.ssh import SSHClient, SSHError
from npa.errors import ScopedCredentialError
from npa.deploy import provisioner
from npa.deploy.byovm import (
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
from npa.deploy.configurator import docker_exec_cmd, write_manifest
from npa.deploy.images import container_image_for_tool
from npa.deploy.provisioner import ProvisionerError

app = typer.Typer(
    name="isaac-lab",
    help="Isaac Lab simulation workbench deployment, training, and evaluation.",
    no_args_is_help=True,
)

console = Console(stderr=True)
logger = logging.getLogger(__name__)

_project_alias: str = ""
_workbench_name: str = ""

ISAAC_LAB_VERSION = "2.3.2.post1"
ISAAC_LAB_HOME = "/opt/isaac-lab"
ISAAC_LAB_VENV = f"{ISAAC_LAB_HOME}/venv"
ISAAC_LAB_SITE_PACKAGES = f"{ISAAC_LAB_VENV}/lib/python3.11/site-packages"
ISAAC_LAB_PKG = f"{ISAAC_LAB_SITE_PACKAGES}/isaaclab"
PIP_EXTRA_INDEX_URL = "https://pypi.nvidia.com"


class OutputFormat(str, Enum):
    text = "text"
    json = "json"


class WorkbenchRuntime(str, Enum):
    vm = "vm"
    container = "container"
    byovm = "byovm"


ISAAC_CONTAINER_NAME = "npa-isaac-lab"


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
    """Isaac Lab deployment, training, and evaluation."""
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


def _get_ssh_config(**overrides):
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


def _is_container_runtime(cfg: Any) -> bool:
    return runtime_uses_container(getattr(cfg, "runtime", "vm"))


def _runtime_bash(cfg: Any, script: str) -> str:
    if _is_container_runtime(cfg):
        return docker_exec_cmd(ISAAC_CONTAINER_NAME, script)
    return _remote_bash(script)


def _is_s3_uri(path: str) -> bool:
    return path.startswith("s3://")


def _storage_client(
    cfg,
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


def _upload_remote_directory_to_s3(
    ssh: SSHClient,
    cfg,
    remote_dir: str,
    output_path: str,
    *,
    target_project: str | None = None,
) -> str:
    archive_remote = f"/tmp/npa-isaac-lab-output-{int(time.time() * 1000)}.tgz"
    with tempfile.TemporaryDirectory(prefix="npa-isaac-lab-output-") as tmp:
        archive_local = Path(tmp) / "output.tgz"
        extract_dir = Path(tmp) / "output"
        ssh.run_or_raise(
            _remote_bash(
                f"tar -C {shlex.quote(remote_dir)} -czf {shlex.quote(archive_remote)} ."
            )
        )
        try:
            ssh.download_file(archive_remote, str(archive_local))
        finally:
            ssh.run(f"rm -f {shlex.quote(archive_remote)}")
        extract_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_local, "r:gz") as archive:
            archive.extractall(extract_dir, filter="data")
        return _storage_client(cfg, project=target_project).upload_directory(
            str(extract_dir), output_path
        )


def _download_remote_directory(
    ssh: SSHClient, remote_dir: str, local_dir: Path
) -> Path:
    archive_remote = f"/tmp/npa-isaac-lab-download-{int(time.time() * 1000)}.tgz"
    archive_local = local_dir.parent / "raw.tgz"
    ssh.run_or_raise(
        _remote_bash(
            f"test -d {shlex.quote(remote_dir)} && "
            f"tar -C {shlex.quote(remote_dir)} -czf {shlex.quote(archive_remote)} ."
        )
    )
    try:
        ssh.download_file(archive_remote, str(archive_local))
    finally:
        ssh.run(f"rm -f {shlex.quote(archive_remote)}")
    local_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_local, "r:gz") as archive:
        archive.extractall(local_dir, filter="data")
    return local_dir


def _upload_local_directory_via_remote_env(
    ssh: SSHClient,
    local_dir: Path,
    remote_dir: str,
    output_path: str,
    *,
    env_file: str = "/etc/npa-isaac-lab/env",
) -> str:
    parsed = urlparse(output_path)
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/").rstrip("/") + "/"
    if parsed.scheme != "s3" or not bucket or not prefix.strip("/"):
        raise SSHError(f"Remote upload expects an s3:// output path: {output_path}")

    ssh.run_or_raise(
        _remote_bash(
            f"rm -rf {shlex.quote(remote_dir)} && mkdir -p {shlex.quote(remote_dir)}"
        )
    )
    try:
        ssh.upload_directory(str(local_dir), remote_dir)
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
from pathlib import Path

import boto3

base = Path({remote_dir!r})
bucket = {bucket!r}
prefix = {prefix!r}
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
count = 0
for path in base.rglob("*"):
    if path.is_file():
        s3.upload_file(str(path), bucket, prefix + str(path.relative_to(base)))
        count += 1
print(f"npa_remote_s3_upload_done files={{count}}")
PY
"""
        ssh.run_or_raise(f"sudo bash -lc {shlex.quote(script)}")
    finally:
        ssh.run(f"rm -rf {shlex.quote(remote_dir)}")
    return f"s3://{bucket}/{prefix}"


def _prepare_remote_input_path(ssh: SSHClient, cfg, input_path: str) -> str:
    if not _is_s3_uri(input_path):
        return input_path

    remote_dir = (
        f"{ISAAC_LAB_HOME}/inputs/npa-input-{int(time.time() * 1000)}"
        if _is_container_runtime(cfg)
        else f"/tmp/npa-isaac-lab-input-{int(time.time() * 1000)}"
    )
    with tempfile.TemporaryDirectory(prefix="npa-isaac-lab-input-") as tmp:
        local_dir = Path(tmp) / "input"
        _storage_client(cfg).download_path(input_path, str(local_dir))
        checkpoint_candidates = sorted(
            path
            for path in local_dir.rglob("*")
            if path.is_file() and path.name.endswith((".json", ".pt", ".pth"))
        )
        local_checkpoint = (
            checkpoint_candidates[0] if checkpoint_candidates else local_dir
        )
        if local_checkpoint.is_dir():
            ssh.upload_directory(str(local_checkpoint), remote_dir)
            return remote_dir
        remote_checkpoint = f"{remote_dir}/{local_checkpoint.name}"
        ssh.upload_file(str(local_checkpoint), remote_checkpoint)
        return remote_checkpoint


def _gpu_selection_error() -> str:
    return (
        "GPU selection is required for Isaac Lab deploy. Provide --gpu-type and --gpu-preset.\n"
        "  Suggested starting points:\n"
        "    Simulation workloads (L40S): --gpu-type gpu-l40s-a --gpu-preset 1gpu-40vcpu-160gb\n"
        "    Heavier Isaac Lab training: use H100/H200, e.g. gpu-h100-sxm or gpu-h200-sxm with a matching Nebius GPU preset."
    )


def _validate_gpu_selection(gpu_type: str, gpu_preset: str) -> None:
    if not gpu_type and not gpu_preset:
        _fail(_gpu_selection_error())
    if not gpu_type:
        _fail(
            "Missing --gpu-type. Isaac Lab deploy does not provide a default GPU type."
        )
    if not gpu_preset:
        _fail(
            "Missing --gpu-preset. Provide the Nebius GPU preset that matches the selected GPU type."
        )


def _build_install_command() -> str:
    script = f"""\
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update
sudo apt-get install -y software-properties-common build-essential git curl libglu1-mesa
if ! command -v python3.11 >/dev/null 2>&1; then
  sudo add-apt-repository -y ppa:deadsnakes/ppa || true
  sudo apt-get update
fi
sudo apt-get install -y python3.11 python3.11-venv python3.11-dev
sudo mkdir -p {ISAAC_LAB_HOME}
sudo chown -R "$USER:$USER" {ISAAC_LAB_HOME}
python3.11 -m venv {ISAAC_LAB_VENV}
{ISAAC_LAB_VENV}/bin/python -m pip install --upgrade pip setuptools wheel
{ISAAC_LAB_VENV}/bin/python -m pip install "isaaclab[isaacsim,all]=={ISAAC_LAB_VERSION}" --extra-index-url {PIP_EXTRA_INDEX_URL}
source {ISAAC_LAB_VENV}/bin/activate
export OMNI_KIT_ACCEPT_EULA="${{OMNI_KIT_ACCEPT_EULA:-YES}}"
python - <<'PY'
from importlib import metadata

from isaaclab.app import AppLauncher

version = metadata.version("isaaclab")
if version != "{ISAAC_LAB_VERSION}":
    raise RuntimeError(f"expected isaaclab {ISAAC_LAB_VERSION}, found {{version}}")

simulation_app = None
try:
    simulation_app = AppLauncher(headless=True).app
    if simulation_app is None:
        raise RuntimeError("AppLauncher.app is None")
    update = getattr(simulation_app, "update", None)
    if callable(update):
        update()
finally:
    if simulation_app is not None:
        close = getattr(simulation_app, "close", None)
        if callable(close):
            close()

print("ISAAC_LAB_ENV_SMOKE_OK")
PY
"""
    return _remote_bash(script)


def _activate_prefix() -> str:
    return (
        f"set -euo pipefail\n"
        f"source {ISAAC_LAB_VENV}/bin/activate\n"
        f'export OMNI_KIT_ACCEPT_EULA="${{OMNI_KIT_ACCEPT_EULA:-YES}}"\n'
        f'export ACCEPT_EULA="${{ACCEPT_EULA:-Y}}"\n'
        f'export ISAACSIM_ACCEPT_EULA="${{ISAACSIM_ACCEPT_EULA:-YES}}"\n'
        f"export ISAACLAB_PKG={ISAAC_LAB_PKG}\n"
        'export PYTHONPATH="$ISAACLAB_PKG/source/isaaclab:'
        "$ISAACLAB_PKG/source/isaaclab_tasks:"
        "$ISAACLAB_PKG/source/isaaclab_rl:"
        "$ISAACLAB_PKG/source/isaaclab_assets:"
        "$ISAACLAB_PKG/source/isaaclab_mimic:"
        "$ISAACLAB_PKG/source/isaaclab_contrib:"
        '${PYTHONPATH:-}"\n'
    )


def _container_prefix() -> str:
    return (
        "set -euo pipefail\n"
        'export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"\n'
        'export ACCEPT_EULA="${ACCEPT_EULA:-Y}"\n'
        'export ISAACSIM_ACCEPT_EULA="${ISAACSIM_ACCEPT_EULA:-YES}"\n'
        "export PYTHONUNBUFFERED=1\n"
    )


def _build_train_script(task: str, num_envs: int, steps: int, output_dir: str) -> str:
    return f"""\
import json
import time
from pathlib import Path

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import gymnasium as gym
import math
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

task = {task!r}
num_envs = {num_envs}
steps = {steps}
output_dir = Path({output_dir!r})
output_dir.mkdir(parents=True, exist_ok=True)
gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
devices = [f"cuda:{{idx}}" for idx in range(gpu_count)] or ["cpu"]
device = devices[0]
envs_per_device = {{
    dev: math.ceil(num_envs / max(1, gpu_count)) if idx < (num_envs % max(1, gpu_count) or max(1, gpu_count)) else num_envs // max(1, gpu_count)
    for idx, dev in enumerate(devices)
}}
started = time.time()
env = None

try:
    print(f"ISAAC_LAB_TRAIN_START task={{task}} num_envs={{num_envs}} steps={{steps}} device={{device}}", flush=True)
    print(f"ISAAC_LAB_MULTI_GPU_DEVICES devices={{devices}} envs_per_device={{envs_per_device}}", flush=True)
    env_cfg = parse_env_cfg(task, device=device, num_envs=num_envs)
    print("ISAAC_LAB_ENV_CREATE_START", flush=True)
    env = gym.make(task, cfg=env_cfg)
    print("ISAAC_LAB_ENV_CREATE_COMPLETE", flush=True)
    print("ISAAC_LAB_ENV_RESET_START", flush=True)
    env.reset()
    print("ISAAC_LAB_ENV_RESET_COMPLETE", flush=True)
    reward_total = 0.0

    for step in range(steps):
        actions = torch.as_tensor(env.action_space.sample(), device=device, dtype=torch.float32)
        _, rewards, _, _, _ = env.step(actions)
        reward_total += float(torch.as_tensor(rewards).mean().item())
        if (step + 1) == steps or (step + 1) % max(1, min(10, steps)) == 0:
            print(f"ISAAC_LAB_TRAIN_STEP step={{step + 1}}/{{steps}}", flush=True)

    summary = {{
        "status": "success",
        "task": task,
        "num_envs": num_envs,
        "steps": steps,
        "device": device,
        "devices": devices,
        "envs_per_device": envs_per_device,
        "mean_reward": reward_total / steps,
        "checkpoint_path": str(output_dir / "npa_isaac_lab_random_policy_checkpoint.json"),
        "duration_seconds": round(time.time() - started, 3),
    }}
    checkpoint = {{
        "format": "npa_isaac_lab_random_policy_v1",
        "task": task,
        "policy": "action_space_sample",
        "num_envs": num_envs,
        "steps": steps,
        "device": device,
        "created_unix": round(time.time(), 3),
    }}
    (output_dir / "npa_isaac_lab_random_policy_checkpoint.json").write_text(json.dumps(checkpoint, indent=2))
    (output_dir / "npa_isaac_lab_train_summary.json").write_text(json.dumps(summary, indent=2))
    print("ISAAC_LAB_TRAIN_COMPLETE")
    print(json.dumps(summary, indent=2), flush=True)
finally:
    if env is not None:
        env.close()
    simulation_app.close()
"""


def _build_eval_script(
    task: str, checkpoint: str, num_episodes: int, output_dir: str
) -> str:
    return f"""\
import json
import time
from pathlib import Path

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

task = {task!r}
checkpoint_path = Path({checkpoint!r})
num_episodes = {num_episodes}
output_dir = Path({output_dir!r})
max_steps_per_episode = 50
output_dir.mkdir(parents=True, exist_ok=True)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
started = time.time()
env = None

try:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {{checkpoint_path}}")

    try:
        checkpoint_info = json.loads(checkpoint_path.read_text())
    except json.JSONDecodeError:
        checkpoint_info = {{"format": "unknown"}}

    print(
        f"ISAAC_LAB_EVAL_START task={{task}} checkpoint={{checkpoint_path}} "
        f"episodes={{num_episodes}} device={{device}}",
        flush=True,
    )
    env_cfg = parse_env_cfg(task, device=device, num_envs=1)
    print("ISAAC_LAB_ENV_CREATE_START", flush=True)
    env = gym.make(task, cfg=env_cfg)
    print("ISAAC_LAB_ENV_CREATE_COMPLETE", flush=True)

    episode_results = []
    for episode in range(num_episodes):
        env.reset()
        episode_reward = 0.0
        steps_ran = 0
        for step in range(max_steps_per_episode):
            actions = torch.as_tensor(env.action_space.sample(), device=device, dtype=torch.float32)
            _, rewards, terminated, truncated, _ = env.step(actions)
            episode_reward += float(torch.as_tensor(rewards).mean().item())
            steps_ran = step + 1

            terminated_tensor = torch.as_tensor(terminated)
            truncated_tensor = torch.as_tensor(truncated)
            done = bool(terminated_tensor.any().item()) or bool(truncated_tensor.any().item())
            if done:
                break

        result = {{
            "episode": episode + 1,
            "steps": steps_ran,
            "reward": episode_reward,
        }}
        episode_results.append(result)
        print(
            f"ISAAC_LAB_EVAL_EPISODE episode={{episode + 1}}/{{num_episodes}} "
            f"steps={{steps_ran}} reward={{episode_reward:.6f}}",
            flush=True,
        )

    mean_reward = sum(item["reward"] for item in episode_results) / num_episodes
    summary = {{
        "status": "success",
        "task": task,
        "checkpoint": str(checkpoint_path),
        "checkpoint_format": checkpoint_info.get("format", "unknown"),
        "num_episodes": num_episodes,
        "max_steps_per_episode": max_steps_per_episode,
        "device": device,
        "mean_reward": mean_reward,
        "episodes": episode_results,
        "duration_seconds": round(time.time() - started, 3),
    }}
    summary_path = output_dir / "npa_isaac_lab_eval_summary.json"
    summary["output_path"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2))
    print("ISAAC_LAB_EVAL_COMPLETE")
    print(json.dumps(summary, indent=2), flush=True)
finally:
    if env is not None:
        env.close()
    simulation_app.close()
"""


def _build_export_lerobot_script(
    task: str,
    num_episodes: int,
    steps_per_episode: int,
    output_dir: str,
) -> str:
    from npa.adapter.isaac_lab_lerobot import G1_STATE_NAMES_43

    state_names_json = json.dumps(G1_STATE_NAMES_43)
    return f"""\
import json
import time
from pathlib import Path

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

task = {task!r}
num_episodes = {num_episodes}
steps_per_episode = {steps_per_episode}
output_dir = Path({output_dir!r})
state_names = {state_names_json}
output_dir.mkdir(parents=True, exist_ok=True)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
started = time.time()
env = None

source_aliases = {{
    "waist_yaw_joint": "torso_joint",
    "left_hand_pinky_joint": "left_five_joint",
    "left_hand_ring_joint": "left_three_joint",
    "left_hand_middle_joint": "left_zero_joint",
    "left_hand_index_joint": "left_six_joint",
    "left_hand_thumb_bend_joint": "left_four_joint",
    "left_hand_thumb_rotation_joint": "left_one_joint",
    "left_hand_aux_joint": "left_two_joint",
    "right_hand_pinky_joint": "right_five_joint",
    "right_hand_ring_joint": "right_three_joint",
    "right_hand_middle_joint": "right_zero_joint",
    "right_hand_index_joint": "right_six_joint",
    "right_hand_thumb_bend_joint": "right_four_joint",
    "right_hand_thumb_rotation_joint": "right_one_joint",
    "right_hand_aux_joint": "right_two_joint",
}}


def _robot_from_env(env):
    scene = getattr(getattr(env, "unwrapped", env), "scene", None)
    if scene is None:
        raise RuntimeError("Isaac Lab scene is not available")
    try:
        return scene["robot"]
    except Exception:
        pass
    keys = scene.keys() if hasattr(scene, "keys") else []
    for key in keys:
        try:
            candidate = scene[key]
        except Exception:
            continue
        if hasattr(getattr(candidate, "data", None), "joint_pos"):
            return candidate
    raise RuntimeError("Could not locate robot articulation with joint_pos data")


def _to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value, dtype=np.float32)


def _joint_map(joint_names, values):
    values = _to_numpy(values).reshape(-1)
    return {{
        name: float(values[idx])
        for idx, name in enumerate(joint_names[: len(values)])
    }}


def _canonicalize(values_by_joint):
    out = np.zeros(len(state_names), dtype=np.float32)
    for idx, target_name in enumerate(state_names):
        source_name = source_aliases.get(target_name, target_name)
        out[idx] = float(values_by_joint.get(source_name, 0.0))
    return out


try:
    print(
        f"ISAAC_LAB_EXPORT_LEROBOT_START task={{task}} "
        f"episodes={{num_episodes}} steps_per_episode={{steps_per_episode}} device={{device}}",
        flush=True,
    )
    env_cfg = parse_env_cfg(task, device=device, num_envs=1)
    env = gym.make(task, cfg=env_cfg)
    robot = _robot_from_env(env)
    joint_names = list(getattr(getattr(robot, "data", None), "joint_names", []) or [])
    if not joint_names:
        raise RuntimeError("Robot joint_names are empty")
    print(f"ISAAC_LAB_EXPORT_JOINTS count={{len(joint_names)}}", flush=True)

    total_frames = 0
    episode_lengths = []
    for episode_index in range(num_episodes):
        env.reset()
        states = []
        actions_out = []

        for step in range(steps_per_episode):
            robot = _robot_from_env(env)
            state_values = _to_numpy(robot.data.joint_pos)[0]
            sample = torch.as_tensor(env.action_space.sample(), device=device, dtype=torch.float32)
            sample_np = _to_numpy(sample)
            action_values = sample_np[0] if sample_np.ndim > 1 else sample_np

            states.append(_canonicalize(_joint_map(joint_names, state_values)))
            actions_out.append(_canonicalize(_joint_map(joint_names, action_values)))

            _, _rewards, terminated, truncated, _info = env.step(sample)
            done = bool(torch.as_tensor(terminated).any().item()) or bool(torch.as_tensor(truncated).any().item())
            if done:
                break

        if not states:
            raise RuntimeError(f"episode {{episode_index}} produced no frames")
        episode_dir = output_dir / f"episode_{{episode_index:06d}}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        np.save(episode_dir / "state.npy", np.stack(states).astype(np.float32))
        np.save(episode_dir / "actions.npy", np.stack(actions_out).astype(np.float32))
        (episode_dir / "episode_meta.json").write_text(json.dumps({{
            "episode_index": episode_index,
            "length": len(states),
            "task": task,
        }}, indent=2))
        total_frames += len(states)
        episode_lengths.append(len(states))
        print(
            f"ISAAC_LAB_EXPORT_EPISODE episode={{episode_index + 1}}/{{num_episodes}} "
            f"frames={{len(states)}}",
            flush=True,
        )

    meta = {{
        "format": "npa_isaac_lab_g1_rollout_v1",
        "task": task,
        "robot_type": "unitree_g1",
        "fps": 50,
        "state_names": state_names,
        "action_names": state_names,
        "source_joint_names": joint_names,
        "num_episodes": num_episodes,
        "steps_per_episode": steps_per_episode,
        "episode_lengths": episode_lengths,
        "total_frames": total_frames,
        "created_unix": round(time.time(), 3),
        "duration_seconds": round(time.time() - started, 3),
    }}
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print("ISAAC_LAB_EXPORT_LEROBOT_COMPLETE")
    print(json.dumps(meta, indent=2), flush=True)
finally:
    if env is not None:
        env.close()
    simulation_app.close()
"""


def _is_isaac_lab_workbench(name: str, wb_cfg: dict) -> bool:
    """True when the workbench is an Isaac Lab VM."""
    wtype = wb_cfg.get("workbench_type")
    if wtype:
        return wtype == "isaac-lab"

    normalized = name.replace("_", "-").lower()
    if "isaac-lab" in normalized or "isaaclab" in normalized:
        return bool(wb_cfg.get("ssh", {}).get("host"))
    return False


@app.command("list")
def list_cmd(
    output_format: OutputFormat = typer.Option(
        OutputFormat.text,
        "--output-format",
        help="Output format.",
    ),
) -> None:
    """List configured Isaac Lab workbenches."""
    projects = list_projects()
    def_proj = default_project_name()
    def_wb = default_workbench_name()

    if output_format == OutputFormat.json:
        filtered = {}
        for pname, pcfg in projects.items():
            wbs = {
                k: v
                for k, v in pcfg.get("workbenches", {}).items()
                if _is_isaac_lab_workbench(k, v)
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
            "No projects configured. Run 'npa workbench isaac-lab deploy' to create one."
        )
        return

    any_shown = False
    for proj_name, proj_cfg in projects.items():
        workbenches = {
            k: v
            for k, v in proj_cfg.get("workbenches", {}).items()
            if _is_isaac_lab_workbench(k, v)
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
            host = wb_cfg.get("ssh", {}).get("host", "?")
            app_status = wb_cfg.get("app_status", "unknown")
            typer.echo(
                f"    {wb_name}{wb_marker}  gpu={gpu}  ssh={host}  app_status={app_status}"
            )

    if not any_shown:
        typer.echo(
            "No Isaac Lab workbenches configured. Run 'npa workbench isaac-lab deploy' to create one."
        )


@app.command("deploy")
def deploy_cmd(
    gpu_type: str = typer.Option("", "--gpu-type", help="Nebius GPU platform."),
    gpu_preset: str = typer.Option("", "--gpu-preset", help="Nebius GPU preset."),
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
    preemptible: bool = typer.Option(
        True, "--preemptible/--no-preemptible", help="Preemptible (spot) instance."
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
    output_format: OutputFormat = typer.Option(
        OutputFormat.text, "--output-format", help="Output format."
    ),
) -> None:
    """Deploy or destroy an Isaac Lab workbench."""
    byovm = is_byovm_runtime(runtime)
    if not destroy and not byovm:
        _validate_gpu_selection(gpu_type, gpu_preset)

    proj_alias = _project_alias or None
    wb_name = _workbench_name or "isaac-lab"
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

    nebius_creds: dict[str, str] = {}
    if use_remote_state and not skip_infra:
        if not env_project or not env_tenant or not env_region:
            _fail(
                "First deploy requires --project-id, --tenant-id, and --region.\n"
                "  Example: npa workbench isaac-lab -p eu-north1 -n isaac-lab deploy \\\n"
                "    --project-id project-... --tenant-id tenant-... \\\n"
                "    --region eu-north1"
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

    instance_name = f"isaac-lab-{proj_alias}-{wb_name}"
    cloud_init_workbench_type = (
        "lerobot-container" if runtime_uses_container(runtime) else "isaac-lab"
    )

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

    total_steps = _deploy_step_count(skip_infra, skip_app)
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
            "workbench_type": cloud_init_workbench_type,
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
                        ssh_config_for_target(
                            target, tokens=resolve_credentials().tokens
                        )
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
        elif resolved_tf_dir:
            try:
                tf_outputs = provisioner.outputs(tf_dir=resolved_tf_dir)
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
                    tf_outputs = provisioner.outputs(tf_dir=str(work_dir))
                except ProvisionerError:
                    pass

        if not tf_outputs:
            from npa.clients.config import (
                _deep_get,
                _load_yaml,
                _resolve_project_section,
                _resolve_workbench_in_project,
            )

            yml = _load_yaml()
            proj = _resolve_project_section(yml, proj_alias)
            wb = _resolve_workbench_in_project(proj, wb_name, yml)
            tf_outputs = {
                "vm_ip": _deep_get(wb, "ssh", "host", default=""),
                "ssh_user": _deep_get(wb, "ssh", "user", default="ubuntu"),
                "ssh_key_path": _deep_get(
                    wb, "ssh", "key_path", default="~/.ssh/id_ed25519"
                ),
                "storage_bucket": _deep_get(
                    wb, "storage", "checkpoint_bucket", default=""
                ),
                "storage_endpoint": _deep_get(
                    wb, "storage", "endpoint_url", default=""
                ),
            }

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
                "workbenches": {
                    wb_name: {
                        "gpu_platform": byovm_fields.get("gpu_platform", gpu_type),
                        "gpu_preset": byovm_fields.get("gpu_preset", gpu_preset),
                        "tf_instance_name": instance_name,
                        "workbench_type": "isaac-lab",
                        "runtime": runtime.value,
                        "app_status": APP_STATUS_PROVISIONED,
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
        credentials = resolve_credentials()
        ssh_cfg = SSHConfig(
            host=vm_ip,
            user=ssh_user,
            key_path=ssh_key,
            tokens=credentials.tokens,
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
            console.print(f"  [{step}/{total_steps}] Starting Isaac Lab container...")
            if dry_run:
                console.print(
                    "    [dry-run] Would pull and run the Isaac Lab container image"
                )
            else:
                from npa.deploy.configurator import (
                    deploy_workbench_container,
                    write_remote_docker_env_file,
                )

                try:
                    service_env = {
                        "ACCEPT_EULA": "Y",
                        "ISAACSIM_ACCEPT_EULA": "YES",
                        "OMNI_KIT_ACCEPT_EULA": "YES",
                        "PRIVACY_CONSENT": "Y",
                        "AWS_ACCESS_KEY_ID": merged_vars.get("nebius_api_key", ""),
                        "AWS_SECRET_ACCESS_KEY": merged_vars.get(
                            "nebius_secret_key", ""
                        ),
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
                    apply_shared_credential_env(
                        service_env, credentials, include=not no_shared_creds
                    )
                    write_remote_docker_env_file(
                        ssh,
                        "/etc/npa-isaac-lab/env",
                        service_env,
                        owner=ssh_user,
                    )
                    image_ref = container_image_for_tool(
                        "isaac-lab",
                        registry=resolve_container_registry(proj_alias),
                    )
                    deploy_workbench_container(
                        ssh,
                        image_ref=image_ref,
                        container_name=ISAAC_CONTAINER_NAME,
                        env_file="/etc/npa-isaac-lab/env",
                        volumes=[
                            f"{ISAAC_LAB_HOME}/runs:{ISAAC_LAB_HOME}/runs",
                            f"{ISAAC_LAB_HOME}/evals:{ISAAC_LAB_HOME}/evals",
                            f"{ISAAC_LAB_HOME}/inputs:{ISAAC_LAB_HOME}/inputs",
                        ],
                        work_dirs=[
                            f"{ISAAC_LAB_HOME}/runs",
                            f"{ISAAC_LAB_HOME}/evals",
                            f"{ISAAC_LAB_HOME}/inputs",
                        ],
                        registry_token=merged_vars.get("iam_token", ""),
                    )
                except SSHError as exc:
                    fail_app(f"Isaac Lab container deployment failed: {exc}")
                    return
        else:
            step += 1
            console.print(
                f"  [{step}/{total_steps}] Installing Isaac Lab {ISAAC_LAB_VERSION}..."
            )
            if dry_run:
                console.print(
                    "    [dry-run] Would install Python 3.11, Isaac Lab, and Isaac Sim"
                )
            else:
                try:
                    ssh.run_or_raise(_build_install_command(), stream=True)
                except SSHError as exc:
                    fail_app(f"Isaac Lab installation failed: {exc}")
                    return

        step += 1
        console.print(f"  [{step}/{total_steps}] Writing deployment manifest...")
        if not dry_run:
            try:
                write_manifest(
                    ssh,
                    tool="isaac-lab",
                    version=ISAAC_LAB_VERSION,
                    deployed_by=f"npa deploy --runtime {runtime.value}",
                )
            except SSHError:
                pass
        mark_app_status(APP_STATUS_HEALTHY)

    step += 1
    console.print(
        f"  [{step}/{total_steps}] Updating config status ({proj_alias}/{wb_name})..."
    )
    if not dry_run:
        console.print("    Saved to ~/.npa/config.yaml")

    console.print("")
    console.print(f"[bold green]Deploy complete.[/bold green] ({proj_alias}/{wb_name})")
    console.print(f"  SSH:  ssh -i {ssh_key} {ssh_user}@{vm_ip}")
    console.print("")
    console.print(f"  Try: npa workbench isaac-lab -p {proj_alias} -n {wb_name} status")

    if output_format == OutputFormat.json:
        typer.echo(
            json.dumps(
                {
                    "project": proj_alias,
                    "name": wb_name,
                    "vm_ip": vm_ip,
                    "ssh_user": ssh_user,
                    "gpu_platform": byovm_fields.get("gpu_platform", gpu_type),
                    "gpu_preset": byovm_fields.get("gpu_preset", gpu_preset),
                    "gpu_count": byovm_fields.get("gpu_count"),
                    "runtime": runtime.value,
                    "tf_outputs": tf_outputs,
                },
                indent=2,
            )
        )


def _deploy_step_count(skip_infra: bool, skip_app: bool) -> int:
    count = 1 if skip_infra else 2
    if not skip_app:
        count += 3
    count += 1
    return count


@app.command("status")
def status_cmd(
    output_format: OutputFormat = typer.Option(
        OutputFormat.text, "--output-format", help="Output format."
    ),
) -> None:
    """Check Isaac Lab VM status via SSH."""
    cfg = _get_ssh_config()
    ssh = SSHClient(cfg.ssh)

    if _is_container_runtime(cfg):
        status_cmd_str = (
            "echo '=== hostname ===' && hostname && "
            "echo '' && echo '=== uptime ===' && uptime && "
            "echo '' && echo '=== container ===' && sudo docker inspect -f 'state={{.State.Status}} image={{.Config.Image}}' npa-isaac-lab && "
            "echo '' && echo '=== isaac lab version ===' && "
            'sudo docker exec npa-isaac-lab bash -lc \'/isaac-sim/python.sh -c "import importlib.metadata as m; print(m.version(\\"isaaclab\\"))"\''
        )
    else:
        status_cmd_str = (
            "echo '=== hostname ===' && hostname && "
            "echo '' && echo '=== uptime ===' && uptime && "
            f"echo '' && echo '=== isaac lab venv ===' && test -x {ISAAC_LAB_VENV}/bin/python && echo 'venv: present' || echo 'venv: missing'; "
            f"echo '' && echo '=== isaac lab version ===' && {ISAAC_LAB_VENV}/bin/python -c 'import importlib.metadata as m; print(m.version(\"isaaclab\"))' 2>/dev/null || echo 'isaaclab not importable'; "
            "echo '' && echo '=== isaac lab processes ===' && "
            "ps -eo pid=,comm=,args= | "
            "awk '$2 !~ /^(bash|sh|zsh|ps|awk)$/ && $0 ~ /(isaaclab|isaacsim|isaac-sim|python.*isaac)/ {print}' | "
            "sed '/^$/d' || true"
        )

    try:
        code, out, err = ssh.run_or_raise(status_cmd_str)
    except SSHError as exc:
        if output_format == OutputFormat.json:
            typer.echo(
                json.dumps(
                    {
                        "host": cfg.ssh.host,
                        "app_status": cfg.app_status or "unknown",
                        "status": "unreachable",
                        "error": str(exc),
                    },
                    indent=2,
                )
            )
        else:
            typer.echo(f"app_status: {cfg.app_status or 'unknown'}")
        _fail(f"SSH error: {exc}")
        return

    if output_format == OutputFormat.json:
        typer.echo(
            json.dumps(
                {
                    "host": cfg.ssh.host,
                    "app_status": cfg.app_status or "unknown",
                    "runtime": getattr(cfg, "runtime", "vm"),
                    "status": "reachable" if code == 0 else "error",
                    "output": out.strip() if out else "",
                },
                indent=2,
            )
        )
    else:
        console.print(f"[bold]Isaac Lab VM: {cfg.ssh.host}[/bold]")
        typer.echo(f"app_status: {cfg.app_status or 'unknown'}")
        typer.echo(f"runtime: {getattr(cfg, 'runtime', 'vm')}")
        if out:
            typer.echo(out.strip())
        if code != 0 and err:
            console.print(f"[red]stderr:[/red]\n{err.strip()[-500:]}")


@app.command("system-info")
def system_info_cmd(
    output_format: OutputFormat = typer.Option(
        OutputFormat.text, "--output-format", help="Output format."
    ),
) -> None:
    """Collect and display system hardware information from the Isaac Lab VM."""
    cfg = _get_ssh_config()
    ssh = SSHClient(cfg.ssh)
    info_cmd = (
        "echo '=== nvidia-smi ===' && nvidia-smi && "
        "echo '' && echo '=== lscpu ===' && lscpu && "
        "echo '' && echo '=== free -h ===' && free -h && "
        "echo '' && echo '=== lsblk ===' && lsblk"
    )
    if _is_container_runtime(cfg):
        info_cmd += (
            " && echo '' && echo '=== container ===' && "
            "sudo docker inspect -f 'state={{.State.Status}} image={{.Config.Image}}' npa-isaac-lab"
        )

    try:
        _, out, err = ssh.run_or_raise(info_cmd)
    except SSHError as exc:
        _fail(f"SSH error: {exc}")
        return

    if output_format == OutputFormat.json:
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


@app.command("train")
def train_cmd(
    task: str = typer.Option(
        ..., "--task", help="Isaac Lab task, e.g. Isaac-Reach-Franka-v0."
    ),
    num_envs: int = typer.Option(
        64, "--num-envs", help="Number of parallel environments."
    ),
    steps: int = typer.Option(1000, "--steps", help="Training iterations to run."),
    output_path: str = typer.Option(
        "",
        "--output-path",
        "-o",
        help="S3 URI where training artifacts are written.",
    ),
    # Deprecated path alias: keep --output-dir working for existing scripts.
    output_dir: str = typer.Option("", "--output-dir", hidden=True),
    output_format: OutputFormat = typer.Option(
        OutputFormat.text, "--output-format", help="Output format."
    ),
) -> None:
    """Run Isaac Lab training on the VM via SSH."""
    if num_envs <= 0:
        _fail(f"--num-envs must be positive, got {num_envs}")
    if steps <= 0:
        _fail(f"--steps must be positive, got {steps}")

    cfg = _get_ssh_config()
    try:
        if output_path:
            output_path = validate_write_path(output_path, tool="Isaac Lab train")
    except PathContractError as exc:
        _fail(str(exc))
        return
    ssh = SSHClient(cfg.ssh)
    target_output = output_path or output_dir or f"{ISAAC_LAB_HOME}/runs"
    output_is_s3 = _is_s3_uri(target_output)
    remote_output_dir = (
        f"{ISAAC_LAB_HOME}/runs/npa-train-{int(time.time())}"
        if output_is_s3
        else target_output
    )
    prefix = _container_prefix() if _is_container_runtime(cfg) else _activate_prefix()
    python_bin = "/isaac-sim/python.sh" if _is_container_runtime(cfg) else "python"

    cmd = _runtime_bash(
        cfg,
        prefix
        + f"mkdir -p {shlex.quote(remote_output_dir)}\n"
        + f"{python_bin} - <<'PY'\n{_build_train_script(task, num_envs, steps, remote_output_dir)}PY\n",
    )
    stream_logs = output_format != OutputFormat.json

    if stream_logs:
        console.print(f"[bold]Training Isaac Lab task[/bold]: {task}")

    start = time.time()
    try:
        exit_code, stdout, stderr = ssh.run(cmd, stream=stream_logs)
    except SSHError as exc:
        _fail(f"SSH error: {exc}")
        return

    result = {
        "status": "success" if exit_code == 0 else "failed",
        "exit_code": exit_code,
        "task": task,
        "num_envs": num_envs,
        "steps": steps,
        "output_path": target_output,
        "output_dir": remote_output_dir,
        "duration_seconds": round(time.time() - start, 1),
    }
    if exit_code != 0:
        result["stderr"] = stderr.strip()[-500:] if stderr else ""
    else:
        if output_is_s3:
            try:
                result["output_path"] = _upload_remote_directory_to_s3(
                    ssh, cfg, remote_output_dir, target_output
                )
            except Exception as exc:
                result["status"] = "failed"
                result["exit_code"] = 1
                result["output_upload_error"] = str(exc)
                exit_code = 1
        if output_format == OutputFormat.json and stdout.strip():
            result["stdout_tail"] = stdout.strip()[-1000:]

    _output(result, output_format)
    if exit_code != 0:
        raise typer.Exit(1)


@app.command("eval")
def eval_cmd(
    task: str = typer.Option(
        ..., "--task", help="Isaac Lab task, e.g. Isaac-Reach-Franka-v0."
    ),
    input_path: str = typer.Option(
        "",
        "--input-path",
        "-i",
        help="S3 URI for a checkpoint.",
    ),
    # Deprecated path alias: keep --checkpoint working for existing scripts.
    checkpoint: str = typer.Option("", "--checkpoint", hidden=True),
    num_episodes: int = typer.Option(
        10, "--num-episodes", help="Number of evaluation episodes."
    ),
    output_path: str = typer.Option(
        "",
        "--output-path",
        "-o",
        help="S3 URI where eval artifacts are written.",
    ),
    # Deprecated path alias: keep --output-dir working for existing scripts.
    output_dir: str = typer.Option("", "--output-dir", hidden=True),
    output_format: OutputFormat = typer.Option(
        OutputFormat.text, "--output-format", help="Output format."
    ),
) -> None:
    """Run Isaac Lab evaluation on the VM via SSH."""
    if num_episodes <= 0:
        _fail(f"--num-episodes must be positive, got {num_episodes}")

    cfg = _get_ssh_config()
    try:
        if input_path:
            input_path = validate_read_path(
                input_path,
                tool="Isaac Lab eval",
                option="--input-path",
                allow_hf=False,
            )
        if output_path:
            output_path = validate_write_path(output_path, tool="Isaac Lab eval")
    except PathContractError as exc:
        _fail(str(exc))
        return
    ssh = SSHClient(cfg.ssh)
    checkpoint_ref = input_path or checkpoint
    if not checkpoint_ref:
        _fail("Provide --input-path.")
        return

    target_output = output_path or output_dir or f"{ISAAC_LAB_HOME}/evals"
    output_is_s3 = _is_s3_uri(target_output)
    remote_output_dir = (
        f"{ISAAC_LAB_HOME}/evals/npa-eval-{int(time.time())}"
        if output_is_s3
        else target_output
    )
    prefix = _container_prefix() if _is_container_runtime(cfg) else _activate_prefix()
    python_bin = "/isaac-sim/python.sh" if _is_container_runtime(cfg) else "python"

    try:
        remote_checkpoint = _prepare_remote_input_path(ssh, cfg, checkpoint_ref)
    except Exception as exc:
        _fail(f"Failed to prepare --input-path: {exc}")
        return

    cmd = _runtime_bash(
        cfg,
        prefix
        + f"mkdir -p {shlex.quote(remote_output_dir)}\n"
        + f"{python_bin} - <<'PY'\n{_build_eval_script(task, remote_checkpoint, num_episodes, remote_output_dir)}PY\n",
    )
    stream_logs = output_format != OutputFormat.json

    if stream_logs:
        console.print(f"[bold]Evaluating Isaac Lab checkpoint[/bold]: {checkpoint_ref}")

    start = time.time()
    try:
        exit_code, stdout, stderr = ssh.run(cmd, stream=stream_logs)
    except SSHError as exc:
        _fail(f"SSH error: {exc}")
        return

    result = {
        "status": "success" if exit_code == 0 else "failed",
        "exit_code": exit_code,
        "task": task,
        "input_path": checkpoint_ref,
        "checkpoint": remote_checkpoint,
        "num_episodes": num_episodes,
        "output_path": target_output,
        "output_dir": remote_output_dir,
        "duration_seconds": round(time.time() - start, 1),
    }
    if exit_code != 0:
        result["stderr"] = stderr.strip()[-500:] if stderr else ""
    else:
        if output_is_s3:
            try:
                result["output_path"] = _upload_remote_directory_to_s3(
                    ssh, cfg, remote_output_dir, target_output
                )
            except Exception as exc:
                result["status"] = "failed"
                result["exit_code"] = 1
                result["output_upload_error"] = str(exc)
                exit_code = 1
        if output_format == OutputFormat.json and stdout.strip():
            result["stdout_tail"] = stdout.strip()[-1000:]

    _output(result, output_format)
    if exit_code != 0:
        raise typer.Exit(1)


@app.command("export-lerobot")
def export_lerobot_cmd(
    task: str = typer.Option(
        ..., "--task", help="Isaac Lab humanoid/G1 task to roll out."
    ),
    num_episodes: int = typer.Option(
        10, "--num-episodes", help="Number of episodes to export."
    ),
    steps_per_episode: int = typer.Option(
        50, "--steps-per-episode", help="Maximum steps recorded per episode."
    ),
    output_path: str = typer.Option(
        ..., "--output-path", "-o", help="S3 URI for the LeRobotDataset output."
    ),
    target_project: str = typer.Option(
        "",
        "--target-project",
        help="Project alias whose scoped principal writes the LeRobotDataset output.",
    ),
    fps: int = typer.Option(
        50, "--fps", help="Frame rate to record in LeRobot metadata."
    ),
    placeholder_video: bool = typer.Option(
        True,
        "--placeholder-video/--no-placeholder-video",
        help="Include a small synthetic ego-view video so visual GR00T loaders have an image modality.",
    ),
    output_format: OutputFormat = typer.Option(
        OutputFormat.text, "--output-format", help="Output format."
    ),
    allow_host_creds: bool = typer.Option(
        False,
        "--allow-host-creds",
        help=(
            "Use --allow-host-creds to allow fallback to VM host credentials "
            "when scoped S3 upload credentials are denied."
        ),
    ),
) -> None:
    """Generate Isaac Lab G1 rollouts and export them as a standard LeRobotDataset.

    Use --allow-host-creds only for intentional VM host credential fallback.
    """
    if num_episodes <= 0:
        _fail(f"--num-episodes must be positive, got {num_episodes}")
    if steps_per_episode <= 0:
        _fail(f"--steps-per-episode must be positive, got {steps_per_episode}")
    if fps <= 0:
        _fail(f"--fps must be positive, got {fps}")

    try:
        output_path = validate_write_path(
            output_path,
            tool="Isaac Lab export-lerobot",
            option="--output-path",
            required=True,
        )
    except PathContractError as exc:
        _fail(str(exc))
        return

    cfg = _get_ssh_config()
    resolved_target_project = target_project or None
    ssh = SSHClient(cfg.ssh)
    remote_raw_dir = f"{ISAAC_LAB_HOME}/runs/npa-export-lerobot-{int(time.time())}/raw"
    prefix = _container_prefix() if _is_container_runtime(cfg) else _activate_prefix()
    python_bin = "/isaac-sim/python.sh" if _is_container_runtime(cfg) else "python"
    cmd = _runtime_bash(
        cfg,
        prefix
        + f"rm -rf {shlex.quote(remote_raw_dir)} && mkdir -p {shlex.quote(remote_raw_dir)}\n"
        + f"{python_bin} - <<'PY'\n"
        + _build_export_lerobot_script(
            task, num_episodes, steps_per_episode, remote_raw_dir
        )
        + "PY\n",
    )
    stream_logs = output_format != OutputFormat.json
    if stream_logs:
        console.print(f"[bold]Exporting Isaac Lab task to LeRobot[/bold]: {task}")

    start = time.time()
    try:
        exit_code, stdout, stderr = ssh.run(cmd, stream=stream_logs)
    except SSHError as exc:
        _fail(f"SSH error: {exc}")
        return

    result: dict[str, Any] = {
        "status": "success" if exit_code == 0 else "failed",
        "exit_code": exit_code,
        "task": task,
        "num_episodes": num_episodes,
        "steps_per_episode": steps_per_episode,
        "remote_raw_dir": remote_raw_dir,
        "output_path": output_path,
        "duration_seconds": round(time.time() - start, 1),
    }
    if exit_code != 0:
        result["stderr"] = stderr.strip()[-500:] if stderr else ""
        _output(result, output_format)
        raise typer.Exit(1)

    with tempfile.TemporaryDirectory(prefix="npa-isaac-lab-lerobot-") as tmp:
        tmp_path = Path(tmp)
        raw_dir = tmp_path / "raw"
        lerobot_dir = tmp_path / "lerobot"
        try:
            _download_remote_directory(ssh, remote_raw_dir, raw_dir)
            from npa.adapter.isaac_lab_lerobot import IsaacLabLeRobotError, convert

            converted = convert(
                raw_dir,
                lerobot_dir,
                fps=fps,
                robot_type="unitree_g1",
                task=task,
                include_placeholder_video=placeholder_video,
            )

            def scoped_upload() -> str:
                saved_to = _storage_client(
                    cfg,
                    project=resolved_target_project,
                    allow_host_creds=allow_host_creds,
                ).upload_directory(str(converted), output_path)
                result["upload_mode"] = "local"
                return saved_to

            def remote_upload() -> str:
                remote_converted_dir = f"{ISAAC_LAB_HOME}/runs/npa-export-lerobot-{int(time.time())}/converted"
                saved_to = _upload_local_directory_via_remote_env(
                    ssh,
                    converted,
                    remote_converted_dir,
                    output_path,
                )
                result["upload_mode"] = "remote-env"
                return saved_to

            def record_fallback(upload_exc: BaseException) -> None:
                result["local_upload_error"] = str(upload_exc)

            saved_to = run_with_host_credential_fallback(
                scoped_upload,
                remote_upload,
                bucket=bucket_from_s3_uri(output_path),
                operation="Isaac Lab export-lerobot upload",
                allow_host_creds=allow_host_creds,
                logger=logger,
                on_fallback=record_fallback,
            )
        except (IsaacLabLeRobotError, SSHError, OSError, tarfile.TarError) as exc:
            result["status"] = "failed"
            result["exit_code"] = 1
            result["export_error"] = str(exc)
            _output(result, output_format)
            raise typer.Exit(1) from exc
        except ScopedCredentialError:
            raise
        except Exception as exc:
            result["status"] = "failed"
            result["exit_code"] = 1
            result["upload_error"] = str(exc)
            _output(result, output_format)
            raise typer.Exit(1) from exc

    result["output_path"] = saved_to
    if output_format == OutputFormat.json and stdout.strip():
        result["stdout_tail"] = stdout.strip()[-1000:]
    _output(result, output_format)
