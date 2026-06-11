"""npa genesis — Genesis simulation training and evaluation commands.

Runs locally by default.  When ``-p``/``-n`` are provided, the command
is forwarded to the workbench VM via SSH (same conda env the distill
workflow creates).
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import queue
import re
import shlex
import shutil
import sys
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from npa.clients.config import (
    default_project_name,
    default_workbench_name,
    resolve_container_registry,
    resolve_environment,
    resolve_project_storage,
)
from npa.clients.credentials import load_credentials, shared_credential_env
from npa.clients.credentials import apply_shared_credential_env
from npa.clients.serverless import EndpointNotFoundError, ServerlessClient, ServerlessClientError
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
from npa.serverless_common import (
    SubnetResolutionError,
    build_serverless_job_env,
    build_serverless_output_upload_cmd,
    resolve_gpu_platform,
    resolve_subnet,
    split_serverless_env,
    validate_output_path,
)
from npa.workbench.training_config import (
    TrainingConfig,
    TrainingConfigError,
    build_training_config,
    checkpoint_s3_uri as resolve_checkpoint_s3_uri,
    overrides_to_mapping,
    shell_env_exports,
    upload_checkpoint_path,
)

app = typer.Typer(
    name="genesis",
    help="Genesis simulation: teacher training, demo generation, evaluation.",
    no_args_is_help=True,
)

console = Console(stderr=True)

# Set by the Typer callback; empty means local execution.
_project_alias: str = ""
_workbench_name: str = ""

# Conda paths matching the VM layout created by distill_two_vm._setup_vm.
_CONDA_BIN = "/opt/conda/bin/conda"
_DEFAULT_CONDA_ENV = "genesis"

# Known subcommands — used to locate the subcommand boundary in sys.argv
# when reconstructing the remote command.
_SUBCOMMANDS = frozenset({
    "train-teacher", "generate-demos", "simulate", "eval-teacher",
    "eval-student", "diagnose", "tune",
})

# Infrastructure subcommands run locally (they manage the VM itself).
_INFRA_SUBCOMMANDS = frozenset({
    "list", "deploy", "status", "system-info",
})


class OutputFormat(str, Enum):
    text = "text"
    json = "json"


class WorkbenchRuntime(str, Enum):
    vm = "vm"
    container = "container"
    byovm = "byovm"
    serverless = "serverless"


class ActionSpace(str, Enum):
    cartesian = "cartesian"
    joint = "joint"


def _fail(msg: str, code: int = 1) -> None:
    console.print(f"[red]Error:[/red] {msg}")
    raise typer.Exit(code)


def _output(data: dict[str, Any], fmt: OutputFormat) -> None:
    if fmt == OutputFormat.json:
        typer.echo(json.dumps(data, indent=2))
    else:
        for key, val in data.items():
            typer.echo(f"  {key}: {val}")


def _is_s3_uri(path: str) -> bool:
    return path.startswith("s3://")


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


def _genesis_warn_if_non_hopper_gpu(platform: str) -> None:
    if platform not in {"gpu-h200-sxm", "gpu-h100-sxm", "gpu-b300-sxm", "gpu-b200-sxm-a"}:
        console.print(
            "[yellow]Warning:[/yellow] Genesis GPU-parallel simulation performs best on Hopper-class GPUs; "
            "this serverless smoke will still run on the selected platform."
        )


def _genesis_serverless_train_teacher_command(
    *,
    n_envs: int,
    max_iterations: int,
    action_space: str,
    seed: int,
    training_config: TrainingConfig | None = None,
    env_overrides: dict[str, Any] | None = None,
    ppo_overrides: dict[str, Any] | None = None,
) -> str:
    config = training_config or TrainingConfig()
    local_dir = "/tmp/npa-genesis-train-teacher"
    training_env = shell_env_exports(config.env())
    script = f"""
import json, os, pathlib, time

out = pathlib.Path("{local_dir}")
out.mkdir(parents=True, exist_ok=True)
log_dir = out / "logs"
started = time.time()
from npa.genesis.train_teacher import PPOConfig, train_teacher

ppo_cfg = PPOConfig()
for key, value in {json.dumps(ppo_overrides or {}, sort_keys=True)}.items():
    setattr(ppo_cfg, key, value)
result = train_teacher(
    n_envs={n_envs},
    max_iterations={max_iterations},
    output_dir=out,
    device=os.environ.get("NPA_TRAINING_DEVICE", "cuda"),
    log_dir=log_dir,
    seed={seed},
    ppo_cfg=ppo_cfg,
    action_space={action_space!r},
    env_overrides={json.dumps(env_overrides or {}, sort_keys=True)} or None,
)
summary = {{
    **result,
    "tool": "genesis",
    "job": os.environ.get("NPA_JOB_NAME", ""),
    "duration_seconds": round(time.time() - started, 3),
    "data_path": os.environ.get("NPA_TRAINING_DATA_PATH", ""),
    "overrides": json.loads(os.environ.get("NPA_TRAINING_OVERRIDES_JSON", "[]") or "[]"),
    "wandb": {{
        "enabled": os.environ.get("NPA_TRAINING_WANDB_ENABLED", "0") == "1",
        "project": os.environ.get("NPA_TRAINING_WANDB_PROJECT", ""),
        "run_name": os.environ.get("NPA_TRAINING_WANDB_RUN_NAME", ""),
        "mode": os.environ.get("WANDB_MODE", ""),
    }},
    "checkpoint_s3_uri": os.environ.get("NPA_CHECKPOINT_S3_URI", ""),
}}
(out / "train_teacher_summary.json").write_text(json.dumps(summary, indent=2))
(out / "npa_genesis_checkpoint_manifest.json").write_text(json.dumps({{
    "format": "npa_genesis_ppo_checkpoint_v1",
    **summary,
}}, indent=2))
print("NPA_GENESIS_SERVERLESS_TRAIN_TEACHER_DONE", os.environ.get("NPA_OUTPUT_PATH", ""), flush=True)
""".strip()
    upload = build_serverless_output_upload_cmd(local_dir, "")
    body = (
        'NPA_PYTHON_BIN="${NPA_PYTHON_BIN:-python3}"\n'
        'if ! command -v "$NPA_PYTHON_BIN" >/dev/null 2>&1; then NPA_PYTHON_BIN=python; fi\n'
        f'{training_env}\n'
        f'"$NPA_PYTHON_BIN" <<\'PY\'\n{script}\nPY\n{upload}'
    )
    return _remote_bash(body)


def _genesis_serverless_train_teacher(
    *,
    n_envs: int,
    max_iterations: int,
    output_path: str,
    training_config: TrainingConfig,
    env_overrides: dict[str, Any],
    ppo_overrides: dict[str, Any],
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
    seed: int,
    action_space: str,
    output_format: OutputFormat,
) -> None:
    if not output_path:
        _fail("Genesis train-teacher --runtime serverless requires --output-path.")
    try:
        validate_output_path(output_path)
        platform, preset, resolved_gpu_count = resolve_gpu_platform(gpu_type, gpu_count)
    except ValueError as exc:
        _fail(str(exc))
    if gpu_preset:
        preset = gpu_preset
    _genesis_warn_if_non_hopper_gpu(platform)
    proj_alias = _project_alias or default_project_name()
    wb_name = _workbench_name or default_workbench_name()
    env_cfg = resolve_environment(proj_alias)
    resolved_project_id = project_id or (env_cfg.project_id if env_cfg else "")
    if not resolved_project_id:
        _fail("Genesis train-teacher --runtime serverless requires --project-id or a configured project.")
    name = job_name or _serverless_job_name(proj_alias, wb_name, "genesis")
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
            "GENESIS_SERVERLESS_REAL_TRAIN": "1",
            **training_config.env(),
        },
    )
    merged_env = dict(env)
    merged_env.update(extra_env)
    env, extra_env = split_serverless_env(merged_env)
    client = ServerlessClient()
    try:
        existing = client.get_job(name, resolved_project_id)
    except EndpointNotFoundError:
        existing = None
    try:
        if existing is not None:
            info = existing if submit_only or existing.status in {"succeeded", "failed", "cancelled"} else client.poll_job(existing.id, resolved_project_id, interval_s=poll_interval, ceiling_s=timeout)
            _output({"status": "existing", "job_id": info.id, "job_name": info.name, "job_status": info.status, "output_path": out}, output_format)
            return
        info = client.create_job(
            project_id=resolved_project_id,
            name=name,
            image=image or container_image_for_tool("genesis", registry=resolve_container_registry(proj_alias)),
            command=_genesis_serverless_train_teacher_command(
                n_envs=n_envs,
                max_iterations=max_iterations,
                action_space=action_space,
                seed=seed,
                training_config=training_config,
                env_overrides=env_overrides,
                ppo_overrides=ppo_overrides,
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
    _output(
        {
            "status": "submitted" if submit_only else info.status,
            "job_id": info.id,
            "job_name": info.name,
            "output_path": out,
            "training_config": training_config.public_dict(),
        },
        output_format,
    )


def _parse_positive_int(value: str | None) -> int:
    if value is None or value.strip() == "":
        return 0
    try:
        parsed = int(value)
    except ValueError:
        return 0
    return parsed if parsed > 0 else 0


def _configured_generate_gpu_count(cli_gpu_count: int) -> int:
    if cli_gpu_count < 0:
        _fail(f"--gpu-count must be non-negative, got {cli_gpu_count}")
    if cli_gpu_count > 0:
        return cli_gpu_count
    return _parse_positive_int(os.environ.get("NPA_GPU_COUNT")) or 1


def _visible_gpu_ids(gpu_count: int) -> list[str]:
    visible = [
        part.strip()
        for part in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if part.strip()
    ]
    if visible:
        return visible[:gpu_count]
    return [str(idx) for idx in range(gpu_count)]


def _split_total(total: int, parts: int) -> list[int]:
    if parts <= 0:
        _fail("Internal error: parts must be positive")
    if total <= 0:
        return [0] * parts
    base, remainder = divmod(total, parts)
    return [base + (1 if idx < remainder else 0) for idx in range(parts)]


@dataclass(frozen=True)
class _GenesisGenerateShard:
    rank: int
    gpu_id: str
    n_envs: int
    n_episodes: int
    output_dir: str
    checkpoint_path: str
    domain_randomize: bool
    fps: int
    seed: int
    allow_failure_demos: bool
    action_space: str
    set_egl_device: bool = True


def _generate_demos_shard(queue: Any, shard: _GenesisGenerateShard) -> None:
    """Run one Genesis demo process pinned to exactly one GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = shard.gpu_id
    os.environ["QD_VISIBLE_DEVICE"] = shard.gpu_id
    if shard.set_egl_device:
        os.environ["EGL_DEVICE_ID"] = shard.gpu_id
    else:
        os.environ.pop("EGL_DEVICE_ID", None)

    try:
        from npa.genesis.generate_demos import generate_demos

        result = generate_demos(
            checkpoint_path=Path(shard.checkpoint_path),
            n_envs=shard.n_envs,
            n_episodes=shard.n_episodes,
            output_dir=Path(shard.output_dir),
            domain_randomize=shard.domain_randomize,
            fps=shard.fps,
            seed=shard.seed,
            allow_failure_demos=shard.allow_failure_demos,
            action_space=shard.action_space,
        )
    except Exception as exc:
        queue.put({
            "rank": shard.rank,
            "gpu_id": shard.gpu_id,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        })
        raise

    queue.put({
        "rank": shard.rank,
        "gpu_id": shard.gpu_id,
        "ok": True,
        "output_dir": shard.output_dir,
        "result": result,
    })


def _drain_shard_queue(result_queue: Any) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    while True:
        try:
            messages.append(result_queue.get_nowait())
        except queue.Empty:
            return messages


def _copy_shard_episodes(shard_output: Path, final_output: Path, start_idx: int) -> int:
    next_idx = start_idx
    for episode_dir in sorted(shard_output.glob("episode_*")):
        if not episode_dir.is_dir():
            continue
        dest = final_output / f"episode_{next_idx:04d}"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(episode_dir, dest)
        next_idx += 1
    return next_idx


def _run_multi_gpu_generate_demos(
    *,
    checkpoint_path: Path,
    n_envs: int,
    n_episodes: int,
    output_dir: Path,
    domain_randomize: bool,
    fps: int,
    seed: int,
    allow_failure_demos: bool,
    action_space: str,
    gpu_count: int,
    set_egl_device: bool = True,
) -> dict[str, Any]:
    from npa.genesis.generate_demos import DemoGenerationError

    effective_gpu_count = min(gpu_count, n_envs)
    if n_episodes > 0:
        effective_gpu_count = min(effective_gpu_count, n_episodes)
    gpu_ids = _visible_gpu_ids(effective_gpu_count)
    if len(gpu_ids) < effective_gpu_count:
        effective_gpu_count = len(gpu_ids)
    if effective_gpu_count <= 1:
        from npa.genesis.generate_demos import generate_demos

        return generate_demos(
            checkpoint_path=checkpoint_path,
            n_envs=n_envs,
            n_episodes=n_episodes,
            output_dir=output_dir,
            domain_randomize=domain_randomize,
            fps=fps,
            seed=seed,
            allow_failure_demos=allow_failure_demos,
            action_space=action_space,
        )

    env_splits = _split_total(n_envs, effective_gpu_count)
    episode_splits = _split_total(n_episodes, effective_gpu_count)
    output_dir.mkdir(parents=True, exist_ok=True)

    ctx = mp.get_context("fork" if hasattr(os, "fork") else "spawn")
    queue = ctx.Queue()
    processes: list[tuple[mp.Process, _GenesisGenerateShard]] = []
    with tempfile.TemporaryDirectory(prefix="npa-genesis-shards-") as shard_root:
        for rank, gpu_id in enumerate(gpu_ids[:effective_gpu_count]):
            shard = _GenesisGenerateShard(
                rank=rank,
                gpu_id=gpu_id,
                n_envs=env_splits[rank],
                n_episodes=episode_splits[rank],
                output_dir=str(Path(shard_root) / f"gpu_{rank}"),
                checkpoint_path=str(checkpoint_path),
                domain_randomize=domain_randomize,
                fps=fps,
                seed=seed + rank,
                allow_failure_demos=allow_failure_demos,
                action_space=action_space,
                set_egl_device=set_egl_device,
            )
            process = ctx.Process(target=_generate_demos_shard, args=(queue, shard))
            process.start()
            processes.append((process, shard))

        for process, _shard in processes:
            process.join()

        messages = _drain_shard_queue(queue)
        by_rank = {int(message["rank"]): message for message in messages}
        successful: list[dict[str, Any]] = []
        failed: list[dict[str, Any]] = []

        for process, shard in processes:
            message = by_rank.get(shard.rank)
            if process.exitcode == 0 and message and message.get("ok"):
                successful.append(message)
                continue

            failed.append({
                "rank": shard.rank,
                "gpu_id": shard.gpu_id,
                "exit_code": process.exitcode,
                "error": message.get("error") if message else "process exited before reporting status",
            })

        if not successful:
            if set_egl_device and failed and all(
                "EGL" in str(item.get("error", "")) for item in failed
            ):
                result = _run_multi_gpu_generate_demos(
                    checkpoint_path=checkpoint_path,
                    n_envs=n_envs,
                    n_episodes=n_episodes,
                    output_dir=output_dir,
                    domain_randomize=domain_randomize,
                    fps=fps,
                    seed=seed,
                    allow_failure_demos=allow_failure_demos,
                    action_space=action_space,
                    gpu_count=gpu_count,
                    set_egl_device=False,
                )
                result["egl_device_fallback"] = True
                return result
            failed_gpus = ", ".join(str(item["gpu_id"]) for item in failed)
            raise DemoGenerationError(
                f"Genesis multi-GPU generation failed on all shards. Failed GPUs: {failed_gpus}"
            )

        next_episode_idx = 0
        for message in sorted(successful, key=lambda item: int(item["rank"])):
            next_episode_idx = _copy_shard_episodes(
                Path(str(message["output_dir"])),
                output_dir,
                next_episode_idx,
            )

        shard_results = [message["result"] for message in successful]
        total_attempted = sum(int(result.get("total_attempted", 0)) for result in shard_results)
        total_successes = sum(int(result.get("total_successes", 0)) for result in shard_results)
        total_episodes = sum(int(result.get("total_episodes", 0)) for result in shard_results)
        fps_values = [
            float(result["fps"])
            for result in shard_results
            if result.get("fps") is not None
        ]
        teacher_success_rate = (
            total_successes / total_attempted if total_attempted > 0 else 0.0
        )

        return {
            "status": "partial_failure" if failed else "success",
            "output_dir": str(output_dir),
            "gpu_count": effective_gpu_count,
            "gpu_ids": gpu_ids[:effective_gpu_count],
            "total_episodes": total_episodes or next_episode_idx,
            "total_successes": total_successes,
            "total_attempted": total_attempted,
            "teacher_success_rate": round(teacher_success_rate, 4),
            "includes_failures": any(
                bool(result.get("includes_failures", False)) for result in shard_results
            ),
            "domain_randomize": domain_randomize,
            "fps": sum(fps_values) if fps_values else None,
            "egl_device_id_enabled": set_egl_device,
            "shards": [
                {
                    "rank": int(message["rank"]),
                    "gpu_id": str(message["gpu_id"]),
                    "n_envs": env_splits[int(message["rank"])],
                    "n_episodes": episode_splits[int(message["rank"])],
                    "status": "success",
                    "total_episodes": int(message["result"].get("total_episodes", 0)),
                }
                for message in sorted(successful, key=lambda item: int(item["rank"]))
            ],
            "failed_shards": failed,
        }


def _print_result(result: dict[str, Any]) -> None:
    for k, v in result.items():
        console.print(f"  {k}: {v}")


def _forward_remote(project: str, name: str) -> None:
    """Forward the current genesis subcommand to a workbench VM via SSH.

    Reconstructs the CLI invocation from ``sys.argv``, strips ``-p``/``-n``,
    and runs it inside the remote conda environment.  Always raises
    ``typer.Exit``.
    """
    from npa.clients.config import ConfigError, resolve_ssh_config
    from npa.clients.ssh import SSHClient, SSHError

    try:
        cfg = resolve_ssh_config(project=project or None, name=name or None)
    except ConfigError as exc:
        _fail(str(exc))

    # Find the subcommand in sys.argv so we can forward everything from
    # that point onward (the callback options -p/-n stay local).
    subcmd_idx = None
    for i, arg in enumerate(sys.argv):
        if arg in _SUBCOMMANDS:
            subcmd_idx = i
            break

    if subcmd_idx is None:
        _fail("Could not determine genesis subcommand from arguments.")

    subcmd_args = sys.argv[subcmd_idx:]
    if subcmd_args and subcmd_args[0] == "simulate":
        subcmd_args = ["generate-demos", *subcmd_args[1:]]
    remote_cmd = "npa workbench genesis " + " ".join(
        shlex.quote(a) for a in subcmd_args
    )

    if runtime_uses_container(getattr(cfg, "runtime", "vm")):
        from npa.deploy.configurator import docker_exec_cmd

        full_cmd = docker_exec_cmd(
            "npa-genesis",
            "set -a; test -f /opt/lerobot/.env && . /opt/lerobot/.env; set +a; "
            f"{remote_cmd}",
        )
    else:
        activate = (
            "set -a && test -f /opt/lerobot/.env && . /opt/lerobot/.env; set +a && "
            f'eval "$({_CONDA_BIN} shell.bash hook)" && '
            f"conda activate {_DEFAULT_CONDA_ENV} && "
        )
        full_cmd = f"{activate}{remote_cmd}"

    console.print(f"[bold]Forwarding to {cfg.ssh.host}[/bold]")
    console.print(f"  {remote_cmd}")

    try:
        ssh = SSHClient(cfg.ssh)
        code, _, stderr = ssh.run(full_cmd, stream=True)
    except SSHError as exc:
        _fail(f"SSH error: {exc}")

    if code != 0 and stderr:
        console.print(f"[red]Remote stderr:[/red]\n{stderr.strip()[-1000:]}")

    raise typer.Exit(code)


@app.callback()
def main(
    ctx: typer.Context,
    project: str = typer.Option(
        "",
        "--project",
        "-p",
        help="Project alias from ~/.npa/config.yaml. "
             "When set, the command runs on the workbench VM via SSH "
             "instead of locally.",
    ),
    name: str = typer.Option(
        "",
        "--name",
        "-n",
        help="Workbench instance name within the project.",
    ),
) -> None:
    """Genesis simulation: teacher training, demo generation, evaluation."""
    global _project_alias, _workbench_name
    _project_alias = project
    _workbench_name = name

    if not (project or name):
        return  # Local mode — proceed to subcommand.

    if ctx.invoked_subcommand is None:
        return  # No subcommand — let Typer show help.

    # Infrastructure subcommands (deploy, status, system-info) run locally
    # and use the project/name to target the right config — don't forward.
    if ctx.invoked_subcommand in _INFRA_SUBCOMMANDS:
        return
    if "--runtime" in sys.argv:
        runtime_idx = sys.argv.index("--runtime")
        if len(sys.argv) > runtime_idx + 1 and sys.argv[runtime_idx + 1] == WorkbenchRuntime.serverless.value:
            return

    _forward_remote(project, name)


def _parse_env_overrides(raw: list[str]) -> dict[str, object]:
    """Parse repeatable --env-override KEY=VALUE strings into a dict.

    Handles booleans (true/false), ints, floats, and falls back to str.
    Also supports shorthand keys for tuple fields:
        friction_min=0.6  →  friction_range = (0.6, <existing_max>)
        friction_max=1.5  →  friction_range = (<existing_min>, 1.5)
    """
    out: dict[str, object] = {}
    for item in raw:
        if "=" not in item:
            _fail(f"--env-override must be KEY=VALUE, got '{item}'")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            _fail(f"--env-override has empty key in '{item}'")
        # Bool
        if value.lower() in ("true", "false"):
            out[key] = value.lower() == "true"
        else:
            # Try int, then float, then keep as str
            try:
                out[key] = int(value)
            except ValueError:
                try:
                    out[key] = float(value)
                except ValueError:
                    out[key] = value
    return out


# Shorthand keys that expand into tuple-field updates.
_RANGE_SHORTHANDS: dict[str, tuple[str, int]] = {
    "friction_min": ("friction_range", 0),
    "friction_max": ("friction_range", 1),
}

# Default tuple values used when only min or max is specified.
_RANGE_DEFAULTS: dict[str, tuple] = {
    "friction_range": (0.3, 1.2),
}
_GENESIS_THRESHOLD_KEYS = {
    "approach_threshold",
    "lift_threshold",
    "place_threshold",
}


def _expand_env_overrides(overrides: dict[str, object]) -> dict[str, object]:
    """Expand shorthand keys (friction_min, friction_max) into tuple fields.

    Also strips keys that are not EnvConfig fields (e.g. approach_threshold)
    so callers that feed overrides directly to EnvConfig don't blow up.
    Returns a new dict.
    """
    out: dict[str, object] = {}
    range_updates: dict[str, list] = {}  # field_name → [min, max]

    for key, value in overrides.items():
        if key in _RANGE_SHORTHANDS:
            field, idx = _RANGE_SHORTHANDS[key]
            if field not in range_updates:
                default = _RANGE_DEFAULTS.get(field, (0.0, 1.0))
                range_updates[field] = list(default)
            range_updates[field][idx] = float(value)
        elif key in _GENESIS_THRESHOLD_KEYS:
            # Keep as-is — callers that care (diagnose, tune) handle these.
            out[key] = value
        else:
            out[key] = value

    for field, vals in range_updates.items():
        out[field] = tuple(vals)

    return out


_GENESIS_RUNNER_OVERRIDE_KEYS = {"n_envs", "max_iterations", "seed", "action_space"}
_GENESIS_PPO_OVERRIDE_KEYS = {
    "actor_hidden_dims",
    "critic_hidden_dims",
    "activation",
    "init_noise_std",
    "learning_rate",
    "num_learning_epochs",
    "num_mini_batches",
    "gamma",
    "lam",
    "clip_param",
    "value_loss_coef",
    "entropy_coef",
    "max_grad_norm",
    "use_clipped_value_loss",
    "schedule",
    "desired_kl",
    "num_steps_per_env",
    "save_interval",
    "empirical_normalization",
}


def _split_genesis_training_overrides(
    overrides: tuple[str, ...],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Route canonical overrides to runner, PPO, or EnvConfig settings."""

    if not overrides:
        return {}, {}, {}
    runner: dict[str, Any] = {}
    ppo: dict[str, Any] = {}
    env: dict[str, Any] = {}
    for raw_key, value in overrides_to_mapping(overrides).items():
        key = raw_key.removeprefix("genesis.")
        leaf = key.rsplit(".", 1)[-1]
        if key in _GENESIS_RUNNER_OVERRIDE_KEYS:
            runner[key] = value
        elif leaf in _GENESIS_PPO_OVERRIDE_KEYS:
            ppo[leaf] = value
        elif key.startswith(("env.", "environment.")):
            env[key.split(".", 1)[1]] = value
        else:
            env[key] = value
    return runner, ppo, _expand_env_overrides(env)


def _ppo_config_from_overrides(overrides: dict[str, Any]):
    if not overrides:
        return None
    from npa.genesis.train_teacher import PPOConfig

    config = PPOConfig()
    for key, value in overrides.items():
        if key not in _GENESIS_PPO_OVERRIDE_KEYS:
            _fail(f"Unsupported Genesis PPO override: {key}")
        setattr(config, key, value)
    return config


# ── train-teacher ───────────────────────────────────────────────────────


@app.command("train-teacher")
def train_teacher_cmd(
    n_envs: int = typer.Option(4096, "--n-envs", help="Number of parallel environments."),
    max_iterations: int = typer.Option(500, "--max-iterations", help="PPO training iterations."),
    output: str = typer.Option(
        "./checkpoints/teacher/", "--output", "-o", help="Checkpoint output directory."
    ),
    output_path: str = typer.Option("", "--output-path", help="S3 URI where serverless training artifacts are written."),
    data_path: str = typer.Option("", "--data-path", help="Optional custom training data path recorded with the run."),
    override: list[str] = typer.Option(
        [],
        "--override",
        help="Generic training override as KEY=VALUE. Repeat for PPO, EnvConfig, or runner keys.",
    ),
    wandb_enabled: bool = typer.Option(False, "--wandb/--no-wandb", help="Enable W&B logging metadata for the training run."),
    wandb_project: str = typer.Option("", "--wandb-project", help="W&B project name."),
    wandb_run_name: str = typer.Option("", "--wandb-run-name", help="W&B run name."),
    wandb_mode: str = typer.Option("offline", "--wandb-mode", help="W&B mode such as online, offline, or disabled."),
    checkpoint_s3_uri: str = typer.Option("", "--checkpoint-s3-uri", help="S3 URI for checkpoint upload."),
    checkpoint_s3_endpoint_url: str = typer.Option("", "--checkpoint-s3-endpoint-url", help="S3-compatible endpoint URL."),
    checkpoint_s3_access_key_id: str = typer.Option("", "--checkpoint-s3-access-key-id", help="S3 access key ID."),
    checkpoint_s3_secret_access_key: str = typer.Option("", "--checkpoint-s3-secret-access-key", help="S3 secret access key."),
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
    device: str = typer.Option("cuda", "--device", help="Torch device."),
    log_dir: str = typer.Option("./logs/teacher/", "--log-dir", help="Tensorboard log directory."),
    seed: int = typer.Option(42, "--seed", help="Random seed."),
    action_space: ActionSpace = typer.Option(
        ActionSpace.cartesian, "--action-space",
        help="Action space: 'cartesian' (4D: delta xyz + gripper, uses IK) "
             "or 'joint' (8D: delta joint positions + gripper).",
    ),
    env_override: list[str] = typer.Option(
        [], "--env-override",
        help="EnvConfig override as KEY=VALUE (repeatable). "
             "e.g. --env-override approach_scale=0 --env-override domain_randomize=true",
    ),
    output_format: OutputFormat = typer.Option(
        OutputFormat.text, "--output-format", help="Output format."
    ),
) -> None:
    """Train an RL teacher policy with PPO using privileged state in Genesis."""
    try:
        training_config = build_training_config(
            data_path=data_path,
            overrides=override,
            wandb_enabled=wandb_enabled,
            wandb_project=wandb_project,
            wandb_run_name=wandb_run_name,
            wandb_mode=wandb_mode,
            checkpoint_s3_uri=checkpoint_s3_uri,
            checkpoint_s3_endpoint_url=checkpoint_s3_endpoint_url,
            checkpoint_s3_access_key_id=checkpoint_s3_access_key_id,
            checkpoint_s3_secret_access_key=checkpoint_s3_secret_access_key,
        )
        runner_overrides, ppo_overrides, shared_env_overrides = _split_genesis_training_overrides(
            training_config.overrides
        )
    except TrainingConfigError as exc:
        _fail(str(exc))
        return
    if runner_overrides:
        n_envs = int(runner_overrides.get("n_envs", n_envs))
        max_iterations = int(runner_overrides.get("max_iterations", max_iterations))
        seed = int(runner_overrides.get("seed", seed))
        if runner_overrides.get("action_space"):
            action_space = ActionSpace(str(runner_overrides["action_space"]))
    if n_envs <= 0:
        _fail(f"--n-envs must be positive, got {n_envs}")
    if max_iterations <= 0:
        _fail(f"--max-iterations must be positive, got {max_iterations}")
    env_overrides = _expand_env_overrides(_parse_env_overrides(env_override))
    env_overrides.update(shared_env_overrides)
    checkpoint_output_path = resolve_checkpoint_s3_uri(training_config, output_path)
    train_overrides = {k: v for k, v in env_overrides.items() if k not in _GENESIS_THRESHOLD_KEYS}
    if _is_serverless_runtime(runtime):
        _genesis_serverless_train_teacher(
            n_envs=n_envs,
            max_iterations=max_iterations,
            output_path=checkpoint_output_path,
            training_config=training_config,
            env_overrides=train_overrides,
            ppo_overrides=ppo_overrides,
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
            seed=seed,
            action_space=action_space.value,
            output_format=output_format,
        )
        return

    # Threshold keys are for diagnose, not training — strip them.
    ppo_cfg = _ppo_config_from_overrides(ppo_overrides)

    console.print("[bold]Training teacher (PPO)[/bold]")
    console.print(f"  n_envs={n_envs}  max_iterations={max_iterations}")
    console.print(f"  action_space={action_space.value}")
    if train_overrides:
        console.print(f"  env_overrides: {train_overrides}")
    console.print(f"  output: {output}")
    console.print(f"  device: {device}")

    from npa.genesis.train_teacher import TrainingError, train_teacher

    try:
        result = train_teacher(
            n_envs=n_envs,
            max_iterations=max_iterations,
            output_dir=Path(output),
            device=device,
            log_dir=Path(log_dir),
            seed=seed,
            ppo_cfg=ppo_cfg,
            action_space=action_space.value,
            env_overrides=train_overrides if train_overrides else None,
        )
    except TrainingError as exc:
        _fail(str(exc))
        return
    uploaded_checkpoint = upload_checkpoint_path(Path(output), training_config)
    if uploaded_checkpoint:
        result["checkpoint_s3_uri"] = uploaded_checkpoint
    result["training_config"] = training_config.public_dict()

    if output_format == OutputFormat.json:
        typer.echo(json.dumps(result, indent=2))
    else:
        console.print("[green]Teacher training complete.[/green]")
        for k, v in result.items():
            console.print(f"  {k}: {v}")


# ── generate-demos ──────────────────────────────────────────────────────


@app.command("simulate")
@app.command("generate-demos")
def generate_demos_cmd(
    checkpoint: str = typer.Option(
        ..., "--checkpoint", help="Path to trained teacher checkpoint."
    ),
    n_envs: int = typer.Option(4096, "--n-envs", help="Number of parallel environments."),
    n_episodes: int = typer.Option(0, "--n-episodes", help="Episodes to collect (0 = one batch)."),
    gpu_count: int = typer.Option(
        0,
        "--gpu-count",
        help="Genesis process count for demo generation. 0 uses NPA_GPU_COUNT or single-GPU.",
    ),
    output_path: str = typer.Option(
        "",
        "--output-path",
        "-o",
        help="S3 URI or local directory where demo numpy arrays are saved. Overrides --output.",
    ),
    # Deprecated path alias: keep --output working for existing scripts.
    output: str = typer.Option("", "--output", hidden=True),
    domain_randomize: bool = typer.Option(
        True, "--domain-randomize/--no-domain-randomize",
        help="Enable domain randomization during recording.",
    ),
    fps: int = typer.Option(20, "--fps", help="Camera frame rate for recording."),
    seed: int = typer.Option(42, "--seed", help="Random seed."),
    allow_failure_demos: bool = typer.Option(
        False, "--allow-failure-demos/--no-failure-demos",
        help="Save all episodes even when 0 teacher successes (for development).",
    ),
    action_space: ActionSpace = typer.Option(
        ActionSpace.cartesian, "--action-space",
        help="Action space: 'cartesian' (4D) or 'joint' (8D). "
             "Must match the action space used during teacher training.",
    ),
    output_format: OutputFormat = typer.Option(
        OutputFormat.text, "--output-format", help="Output format."
    ),
) -> None:
    """Generate camera-only demonstrations using a trained teacher policy."""
    if n_envs <= 0:
        _fail(f"--n-envs must be positive, got {n_envs}")
    if n_episodes < 0:
        _fail(f"--n-episodes must be non-negative, got {n_episodes}")

    ckpt = Path(checkpoint)
    if not ckpt.exists():
        _fail(f"Checkpoint not found: {ckpt}")

    target_output = output_path or output or "./data/demos/"
    configured_gpu_count = _configured_generate_gpu_count(gpu_count)

    console.print("[bold]Generating demonstrations[/bold]")
    console.print(f"  checkpoint: {ckpt}")
    console.print(f"  n_envs={n_envs}  domain_randomize={domain_randomize}")
    if configured_gpu_count > 1:
        console.print(f"  gpu_count={configured_gpu_count}  mode=multiprocess")
    console.print(f"  output: {target_output}")

    from npa.genesis.generate_demos import DemoGenerationError, generate_demos

    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    local_output = Path(target_output)
    if _is_s3_uri(target_output):
        temp_dir = tempfile.TemporaryDirectory(prefix="npa-genesis-demos-")
        local_output = Path(temp_dir.name)

    try:
        try:
            if configured_gpu_count > 1:
                result = _run_multi_gpu_generate_demos(
                    checkpoint_path=ckpt,
                    n_envs=n_envs,
                    n_episodes=n_episodes,
                    output_dir=local_output,
                    domain_randomize=domain_randomize,
                    fps=fps,
                    seed=seed,
                    allow_failure_demos=allow_failure_demos,
                    action_space=action_space.value,
                    gpu_count=configured_gpu_count,
                )
            else:
                result = generate_demos(
                    checkpoint_path=ckpt,
                    n_envs=n_envs,
                    n_episodes=n_episodes,
                    output_dir=local_output,
                    domain_randomize=domain_randomize,
                    fps=fps,
                    seed=seed,
                    allow_failure_demos=allow_failure_demos,
                    action_space=action_space.value,
                )
        except DemoGenerationError as exc:
            _fail(str(exc))
            return

        if _is_s3_uri(target_output):
            from npa.clients.storage import StorageClient

            uploaded = StorageClient.from_environment().upload_directory(
                str(local_output), target_output
            )
            result["output_path"] = uploaded
        else:
            result["output_path"] = str(local_output)

        failed_shards = result.get("failed_shards") or []
        if failed_shards:
            failed_gpus = ", ".join(str(item.get("gpu_id")) for item in failed_shards)
            if output_format == OutputFormat.json:
                typer.echo(json.dumps(result, indent=2))
            else:
                console.print("[red]Demo generation partially failed.[/red]")
                _print_result(result)
                console.print(
                    f"[red]Failed Genesis GPU shard(s):[/red] {failed_gpus}. "
                    f"Partial output: {result.get('output_path')}"
                )
            raise typer.Exit(1)
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()

    if output_format == OutputFormat.json:
        typer.echo(json.dumps(result, indent=2))
    else:
        console.print("[green]Demo generation complete.[/green]")
        _print_result(result)


# ── eval-teacher ───────────────────────────────────────────────────────


@app.command("eval-teacher")
def eval_teacher_cmd(
    checkpoint: str = typer.Option(
        ..., "--checkpoint", help="Path to trained teacher checkpoint (model.pt)."
    ),
    n_envs: int = typer.Option(1024, "--n-envs", help="Number of parallel environments."),
    seed: int = typer.Option(7777, "--seed", help="Random seed (held-out from demo generation)."),
    action_space: ActionSpace = typer.Option(
        ActionSpace.cartesian, "--action-space",
        help="Action space: 'cartesian' (4D) or 'joint' (8D). "
             "Must match the action space used during teacher training.",
    ),
    output_format: OutputFormat = typer.Option(
        OutputFormat.text, "--output-format", help="Output format."
    ),
) -> None:
    """Evaluate the teacher under held-out conditions (no cameras, privileged state)."""
    ckpt = Path(checkpoint)
    if not ckpt.exists():
        _fail(f"Checkpoint not found: {ckpt}")

    console.print("[bold]Evaluating teacher (held-out)[/bold]")
    console.print(f"  checkpoint: {ckpt}")
    console.print(f"  n_envs={n_envs}  seed={seed}")

    from npa.genesis.generate_demos import DemoGenerationError, eval_teacher

    try:
        rate = eval_teacher(
            checkpoint_path=ckpt,
            n_envs=n_envs,
            seed=seed,
            action_space=action_space.value,
        )
    except DemoGenerationError as exc:
        _fail(str(exc))
        return

    if output_format == OutputFormat.json:
        typer.echo(json.dumps({"teacher_success_rate": round(rate, 4)}, indent=2))
    else:
        console.print("[green]Teacher eval complete.[/green]")
        console.print(f"  success_rate: {rate:.2%}")


# ── eval-student ────────────────────────────────────────────────────────


@app.command("eval-student")
def eval_student_cmd(
    checkpoint: str = typer.Option(
        "", "--checkpoint", help="Path to trained student policy checkpoint."
    ),
    input_path: str = typer.Option(
        "",
        "--input-path",
        help="S3 URI or local path to trained student policy checkpoint. Overrides --checkpoint.",
    ),
    n_envs: int = typer.Option(1024, "--n-envs", help="Number of parallel environments."),
    n_episodes: int = typer.Option(1024, "--n-episodes", help="Total evaluation episodes."),
    output: str = typer.Option(
        "./eval/", "--output", "-o", help="Output directory for eval metrics."
    ),
    domain_randomize: bool = typer.Option(
        True, "--domain-randomize/--no-domain-randomize",
        help="Enable domain randomization during eval.",
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed (held-out from training)."),
    teacher_success_rate: float = typer.Option(
        -1.0, "--teacher-success-rate",
        help="User-provided teacher success rate used as baseline for computing "
             "the distillation gap. -1 means unknown (gap calculation is skipped).",
    ),
    action_space: ActionSpace = typer.Option(
        ActionSpace.cartesian, "--action-space",
        help="Action space: 'cartesian' (4D) or 'joint' (8D). "
             "Must match the action space used for demo generation / student training.",
    ),
    output_format: OutputFormat = typer.Option(
        OutputFormat.text, "--output-format", help="Output format."
    ),
) -> None:
    """Evaluate a student vision policy in Genesis simulation."""
    checkpoint_ref = input_path or checkpoint
    if not checkpoint_ref:
        _fail("Provide --checkpoint or --input-path.")
        return

    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    if _is_s3_uri(checkpoint_ref):
        from npa.clients.storage import StorageClient

        temp_dir = tempfile.TemporaryDirectory(prefix="npa-genesis-student-checkpoint-")
        ckpt = Path(
            StorageClient.from_environment().download_directory(checkpoint_ref, temp_dir.name)
        )
    else:
        ckpt = Path(checkpoint_ref)

    if not ckpt.exists():
        if temp_dir is not None:
            temp_dir.cleanup()
        _fail(f"Checkpoint not found: {ckpt}")

    console.print("[bold]Evaluating student policy[/bold]")
    console.print(f"  checkpoint: {ckpt}")
    console.print(f"  n_envs={n_envs}  n_episodes={n_episodes}")
    console.print(f"  output: {output}")

    from npa.genesis.eval_student import EvalError, eval_student

    tsr = teacher_success_rate if teacher_success_rate >= 0.0 else None
    try:
        try:
            result = eval_student(
                checkpoint_path=ckpt,
                n_envs=n_envs,
                n_episodes=n_episodes,
                output_dir=Path(output),
                domain_randomize=domain_randomize,
                seed=seed,
                teacher_success_rate=tsr,
                action_space=action_space.value,
            )
        except EvalError as exc:
            _fail(str(exc))
            return
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()

    if output_format == OutputFormat.json:
        typer.echo(json.dumps(result, indent=2))
    else:
        console.print("[green]Evaluation complete.[/green]")
        console.print(f"  success_rate: {result.get('success_rate', 'N/A')}")
        console.print(f"  n_episodes: {result.get('n_episodes', 'N/A')}")
        for k, v in result.items():
            if k not in ("success_rate", "n_episodes"):
                console.print(f"  {k}: {v}")


# ── diagnose ───────────────────────────────────────────────────────────


@app.command("diagnose")
def diagnose_cmd(
    checkpoint: str = typer.Option(
        ..., "--checkpoint", help="Path to trained teacher checkpoint (model.pt)."
    ),
    n_envs: int = typer.Option(1024, "--n-envs", help="Number of parallel environments."),
    n_episodes: int = typer.Option(
        0, "--n-episodes", help="Total episodes to evaluate (0 = one batch)."
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed."),
    output: str = typer.Option(
        "", "--output", "-o",
        help="Path to save diagnosis JSON (empty = don't save).",
    ),
    action_space: ActionSpace = typer.Option(
        ActionSpace.cartesian, "--action-space",
        help="Action space: 'cartesian' (4D) or 'joint' (8D). "
             "Must match the action space used during teacher training. "
             "When approach is the bottleneck and action_space is joint, "
             "diagnose will suggest switching to cartesian.",
    ),
    env_override: list[str] = typer.Option(
        [], "--env-override",
        help="Override as KEY=VALUE (repeatable). EnvConfig fields go to the "
             "simulation; threshold keys (approach_threshold, lift_threshold, "
             "place_threshold) override diagnosis classification thresholds.",
    ),
    output_format: OutputFormat = typer.Option(
        OutputFormat.text, "--output-format", help="Output format."
    ),
) -> None:
    """Diagnose teacher policy failures: run rollouts, classify failure phases, suggest fixes."""
    ckpt = Path(checkpoint)
    if not ckpt.exists():
        _fail(f"Checkpoint not found: {ckpt}")

    if n_envs <= 0:
        _fail(f"--n-envs must be positive, got {n_envs}")

    env_overrides = _expand_env_overrides(_parse_env_overrides(env_override))

    console.print("[bold]Diagnosing teacher policy[/bold]")
    console.print(f"  checkpoint: {ckpt}")
    console.print(f"  n_envs={n_envs}  n_episodes={n_episodes or n_envs}")
    if env_overrides:
        console.print(f"  overrides: {env_overrides}")

    from npa.genesis.diagnose import DiagnoseError, diagnose_teacher, save_diagnosis

    try:
        result = diagnose_teacher(
            checkpoint_path=ckpt,
            n_envs=n_envs,
            n_episodes=n_episodes,
            seed=seed,
            action_space=action_space.value,
            env_overrides=env_overrides if env_overrides else None,
        )
    except DiagnoseError as exc:
        _fail(str(exc))
        return

    if output:
        save_diagnosis(result, Path(output))
        console.print(f"  diagnosis saved to: {output}")

    if output_format == OutputFormat.json:
        typer.echo(json.dumps(result, indent=2))
    else:
        _print_diagnosis(result)


def _print_diagnosis(result: dict) -> None:
    """Pretty-print a diagnosis result to stderr."""
    console.print("\n[green]Diagnosis complete.[/green]")
    console.print(
        f"  success rate: {result['success_rate']:.2%} "
        f"({result['success_count']}/{result['n_episodes']})"
    )

    console.print("\n  [bold]Phase breakdown:[/bold]")
    for phase, count in result["phase_counts"].items():
        bar = "█" * min(count, 40)
        console.print(f"    {phase:10s}  {count:4d}  {bar}")

    bottleneck = result["bottleneck"]
    if bottleneck == "none":
        console.print("\n  [green]No failure bottleneck — policy has some success.[/green]")
        return

    console.print(f"\n  [bold red]Bottleneck: {bottleneck}[/bold red]")

    suggestion = result.get("suggestion", {})
    if suggestion:
        console.print(f"  {suggestion['description']}")
        console.print(f"\n  [bold]Suggested fix:[/bold] {suggestion['human_hint']}")
        console.print(f"  Config changes: {suggestion['config_changes']}")


# ── tune ───────────────────────────────────────────────────────────────


@app.command("tune")
def tune_cmd(
    checkpoint: str = typer.Option(
        ..., "--checkpoint", help="Path to initial teacher checkpoint (model.pt)."
    ),
    max_rounds: int = typer.Option(
        5, "--max-rounds", help="Maximum diagnose→retrain iterations."
    ),
    retrain_iterations: int = typer.Option(
        100, "--retrain-iterations",
        help="PPO iterations per retrain round (short run).",
    ),
    n_envs: int = typer.Option(
        4096, "--n-envs", help="Parallel environments for retraining."
    ),
    diagnose_n_envs: int = typer.Option(
        1024, "--diagnose-n-envs", help="Parallel environments for diagnosis."
    ),
    seed: int = typer.Option(42, "--seed", help="Base random seed."),
    output: str = typer.Option(
        "./checkpoints/tune/", "--output", "-o",
        help="Output directory for per-round checkpoints.",
    ),
    data_path: str = typer.Option("", "--data-path", help="Optional custom training data path recorded with the run."),
    override: list[str] = typer.Option(
        [],
        "--override",
        help="Generic training override as KEY=VALUE. Repeat for PPO, EnvConfig, or tune runner keys.",
    ),
    wandb_enabled: bool = typer.Option(False, "--wandb/--no-wandb", help="Enable W&B logging metadata for the training run."),
    wandb_project: str = typer.Option("", "--wandb-project", help="W&B project name."),
    wandb_run_name: str = typer.Option("", "--wandb-run-name", help="W&B run name."),
    wandb_mode: str = typer.Option("offline", "--wandb-mode", help="W&B mode such as online, offline, or disabled."),
    checkpoint_s3_uri: str = typer.Option("", "--checkpoint-s3-uri", help="S3 URI for checkpoint upload."),
    checkpoint_s3_endpoint_url: str = typer.Option("", "--checkpoint-s3-endpoint-url", help="S3-compatible endpoint URL."),
    checkpoint_s3_access_key_id: str = typer.Option("", "--checkpoint-s3-access-key-id", help="S3 access key ID."),
    checkpoint_s3_secret_access_key: str = typer.Option("", "--checkpoint-s3-secret-access-key", help="S3 secret access key."),
    log_dir: str = typer.Option(
        "./logs/tune/", "--log-dir", help="Tensorboard log directory.",
    ),
    device: str = typer.Option("cuda", "--device", help="Torch device."),
    action_space: ActionSpace = typer.Option(
        ActionSpace.cartesian, "--action-space",
        help="Action space: 'cartesian' (4D) or 'joint' (8D). "
             "If diagnose suggests switching to cartesian, subsequent "
             "rounds will use the new action space automatically.",
    ),
    env_override: list[str] = typer.Option(
        [], "--env-override",
        help="Override as KEY=VALUE (repeatable). EnvConfig fields are "
             "applied to both retrain and diagnose envs; threshold keys "
             "(approach_threshold, lift_threshold, place_threshold) override "
             "diagnosis classification thresholds.",
    ),
    min_success_rate: float = typer.Option(
        0.0, "--min-success-rate",
        help="Stop tuning when success rate exceeds this value. "
             "0.0 (default) stops as soon as any episode succeeds. "
             "e.g. 0.20 requires 20%% success before stopping.",
    ),
    output_format: OutputFormat = typer.Option(
        OutputFormat.text, "--output-format", help="Output format."
    ),
) -> None:
    """Auto-tune loop: diagnose → adjust config → retrain → re-diagnose."""
    ckpt = Path(checkpoint)
    if not ckpt.exists():
        _fail(f"Checkpoint not found: {ckpt}")

    if max_rounds <= 0:
        _fail(f"--max-rounds must be positive, got {max_rounds}")
    if retrain_iterations <= 0:
        _fail(f"--retrain-iterations must be positive, got {retrain_iterations}")
    if n_envs <= 0:
        _fail(f"--n-envs must be positive, got {n_envs}")
    if diagnose_n_envs <= 0:
        _fail(f"--diagnose-n-envs must be positive, got {diagnose_n_envs}")
    if not 0.0 <= min_success_rate <= 1.0:
        _fail(f"--min-success-rate must be in [0.0, 1.0], got {min_success_rate}")

    try:
        training_config = build_training_config(
            data_path=data_path,
            overrides=override,
            wandb_enabled=wandb_enabled,
            wandb_project=wandb_project,
            wandb_run_name=wandb_run_name,
            wandb_mode=wandb_mode,
            checkpoint_s3_uri=checkpoint_s3_uri,
            checkpoint_s3_endpoint_url=checkpoint_s3_endpoint_url,
            checkpoint_s3_access_key_id=checkpoint_s3_access_key_id,
            checkpoint_s3_secret_access_key=checkpoint_s3_secret_access_key,
        )
        runner_overrides, ppo_overrides, shared_env_overrides = _split_genesis_training_overrides(
            training_config.overrides
        )
    except TrainingConfigError as exc:
        _fail(str(exc))
        return
    if runner_overrides:
        n_envs = int(runner_overrides.get("n_envs", n_envs))
        seed = int(runner_overrides.get("seed", seed))
        if runner_overrides.get("action_space"):
            action_space = ActionSpace(str(runner_overrides["action_space"]))
        if "max_iterations" in runner_overrides:
            retrain_iterations = int(runner_overrides["max_iterations"])
    if retrain_iterations <= 0:
        _fail(f"--retrain-iterations must be positive, got {retrain_iterations}")
    if n_envs <= 0:
        _fail(f"--n-envs must be positive, got {n_envs}")
    env_overrides = _expand_env_overrides(_parse_env_overrides(env_override))
    env_overrides.update(shared_env_overrides)

    console.print("[bold]Auto-tuning teacher policy[/bold]")
    console.print(f"  checkpoint: {ckpt}")
    console.print(f"  max_rounds={max_rounds}  retrain_iterations={retrain_iterations}")
    console.print(f"  n_envs={n_envs}  diagnose_n_envs={diagnose_n_envs}")
    console.print(f"  min_success_rate={min_success_rate:.0%}")
    if env_overrides:
        console.print(f"  overrides: {env_overrides}")
    console.print(f"  output: {output}")

    from npa.genesis.tune import TuneError, tune_teacher

    try:
        result = tune_teacher(
            checkpoint_path=ckpt,
            max_rounds=max_rounds,
            retrain_iterations=retrain_iterations,
            n_envs=n_envs,
            diagnose_n_envs=diagnose_n_envs,
            seed=seed,
            output_dir=Path(output),
            log_dir=Path(log_dir),
            device=device,
            action_space=action_space.value,
            env_overrides=env_overrides if env_overrides else None,
            ppo_overrides=ppo_overrides,
            min_success_rate=min_success_rate,
        )
    except TuneError as exc:
        _fail(str(exc))
        return
    uploaded_checkpoint = upload_checkpoint_path(Path(output), training_config)
    if uploaded_checkpoint:
        result["checkpoint_s3_uri"] = uploaded_checkpoint
    result["training_config"] = training_config.public_dict()

    if output_format == OutputFormat.json:
        typer.echo(json.dumps(result, indent=2))
    else:
        _print_tune_result(result)

    if result["status"] != "success":
        raise typer.Exit(1)


def _print_tune_result(result: dict) -> None:
    """Pretty-print the tune loop result to stderr."""
    status = result["status"]
    if status == "success":
        console.print("\n[green bold]Tuning succeeded![/green bold]")
    else:
        console.print("\n[red bold]Tuning did not achieve success > 0%.[/red bold]")

    console.print(f"  rounds completed: {result['rounds_completed']}")
    console.print(f"  final success rate: {result['final_success_rate']:.2%}")
    console.print(f"  final checkpoint: {result['final_checkpoint']}")

    if result.get("env_overrides_applied"):
        console.print(f"  config overrides applied: {result['env_overrides_applied']}")

    console.print("\n  [bold]Round history:[/bold]")
    for r in result.get("rounds", []):
        action = r.get("action", "?")
        fix = r.get("fix_applied", "—")
        console.print(
            f"    round {r['round']}: "
            f"success={r['success_rate']:.2%}  "
            f"bottleneck={r['bottleneck']}  "
            f"action={action}  fix={fix}"
        )


# ── Infrastructure management ──────────────────────────────────────────


def _get_ssh_config(**overrides):
    """Resolve workbench config via SSH-only resolution (no endpoint required)."""
    from npa.clients.config import ConfigError, resolve_ssh_config

    try:
        return resolve_ssh_config(
            project=_project_alias or None,
            name=_workbench_name or None,
            **{k: v for k, v in overrides.items() if v is not None},
        )
    except ConfigError as exc:
        _fail(str(exc))


def _is_genesis_workbench(name: str, wb_cfg: dict) -> bool:
    """True when the workbench is a Genesis sim VM.

    Checks ``workbench_type`` first (authoritative when present), then
    falls back to name matching and the legacy endpoint heuristic for
    configs written before the type field existed.  Unprovisioned
    placeholders (no endpoint AND no SSH host) are excluded.
    """
    wtype = wb_cfg.get("workbench_type")
    if wtype:
        return wtype == "genesis"
    if "genesis" in name:
        return True
    # No endpoint could mean genesis OR unprovisioned placeholder.
    # Require an SSH host to distinguish real genesis VMs from placeholders.
    if not wb_cfg.get("endpoint"):
        return bool(wb_cfg.get("ssh", {}).get("host"))
    return False


@app.command("list")
def list_cmd(
    output_format: OutputFormat = typer.Option(OutputFormat.text, "--output-format", help="Output format."),
) -> None:
    """List configured Genesis workbenches (excludes LeRobot VMs)."""
    from npa.clients import config as client_config

    projects = client_config.list_projects()
    def_proj = client_config.default_project_name()
    def_wb = client_config.default_workbench_name()

    if output_format == OutputFormat.json:
        filtered = {}
        for pname, pcfg in projects.items():
            wbs = {k: v for k, v in pcfg.get("workbenches", {}).items()
                   if _is_genesis_workbench(k, v)}
            if wbs:
                filtered[pname] = {**pcfg, "workbenches": wbs}
        typer.echo(json.dumps({
            "projects": filtered,
            "default_project": def_proj,
            "default_workbench": def_wb,
        }, indent=2))
        return

    if not projects:
        typer.echo("No projects configured. Run 'npa workbench genesis deploy' to create one.")
        return

    any_shown = False
    for proj_name, proj_cfg in projects.items():
        workbenches = {k: v for k, v in proj_cfg.get("workbenches", {}).items()
                       if _is_genesis_workbench(k, v)}
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
            typer.echo(f"    {wb_name}{wb_marker}  gpu={gpu}  ssh={host}  app_status={app_status}")

    if not any_shown:
        typer.echo("No Genesis workbenches configured. Run 'npa workbench genesis deploy' to create one.")


@app.command("deploy")
def deploy_cmd(
    gpu_type: str = typer.Option("gpu-l40s-a", "--gpu-type", help="Nebius GPU platform."),
    gpu_preset: str = typer.Option("1gpu-40vcpu-160gb", "--gpu-preset", help="GPU preset."),
    region: str = typer.Option("", "--region", help="Nebius region."),
    project_id: str = typer.Option("", "--project-id", help="Nebius project ID."),
    tenant_id: str = typer.Option("", "--tenant-id", help="Nebius tenant ID."),
    tf_dir: str = typer.Option("", "--tf-dir", help="Path to Terraform directory (default: bundled)."),
    tf_var: list[str] = typer.Option([], "--tf-var", "-v", help="Extra TF variable (key=value), repeatable."),
    skip_infra: bool = typer.Option(False, "--skip-infra", help="Skip Terraform, only verify connectivity."),
    destroy: bool = typer.Option(False, "--destroy", help="Destroy infrastructure and clean up config."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would happen without doing it."),
    no_shared_creds: bool = typer.Option(False, "--no-shared-creds", help="Do not inject ~/.npa/credentials.yaml shared credentials into the service env."),
    preemptible: bool = typer.Option(True, "--preemptible/--no-preemptible", help="Preemptible (spot) instance."),
    runtime: WorkbenchRuntime = typer.Option(WorkbenchRuntime.vm, "--runtime", help=RUNTIME_HELP),
    host: str = typer.Option("", "--host", help="BYOVM SSH host/IP. Used only with --runtime byovm."),
    ssh_key: str = typer.Option("", "--ssh-key", help="BYOVM SSH private key path. Used only with --runtime byovm."),
    ssh_user: str = typer.Option("", "--ssh-user", help="BYOVM SSH username. Defaults to ubuntu."),
    gpu_count: int = typer.Option(0, "--gpu-count", help="Limit visible GPUs on BYOVM (0 = all detected)."),
    disk_size: int | None = typer.Option(None, "--disk-size", help="Boot disk size in GiB. Defaults to 250 for container runtime; VM runtime keeps the Terraform default."),
    default: bool = typer.Option(False, "--default", help="Set this workbench as the default."),
    output_format: OutputFormat = typer.Option(OutputFormat.text, "--output-format", help="Output format."),
) -> None:
    """Deploy or destroy a Genesis simulation VM.

    On first deploy, pass --project-id and --tenant-id.  These are saved
    and reused automatically on subsequent deploys.
    """
    from npa.deploy.provisioner import ProvisionerError, apply_boot_disk_tf_vars

    proj_alias = _project_alias or None
    wb_name = _workbench_name or "genesis"
    byovm = is_byovm_runtime(runtime)
    if _is_serverless_runtime(runtime):
        _fail("Genesis deploy does not use --runtime serverless; use `npa workbench genesis train-teacher --runtime serverless`.")
    use_remote_state = not tf_dir and not byovm
    if byovm:
        skip_infra = True

    # Parse extra TF vars.
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

    if not proj_alias:
        proj_alias = env_region or ("byovm" if byovm else "default")

    # ── Bootstrap Nebius environment ─────────────────────────────────
    nebius_creds: dict[str, str] = {}

    if use_remote_state and not skip_infra:
        if not env_project or not env_tenant or not env_region:
            _fail(
                "First deploy requires --project-id, --tenant-id, and --region.\n"
                "  Example: npa workbench genesis -p eu-north1 -n genesis deploy \\\n"
                "    --project-id project-... --tenant-id tenant-... \\\n"
                "    --region eu-north1 --gpu-type gpu-l40s-a"
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
    if use_remote_state and (destroy or skip_infra):
        from npa.clients.config import resolve_terraform_state

        state = resolve_terraform_state(proj_alias)
        saved = {
            "s3_bucket": state.bucket,
            "s3_endpoint": state.endpoint,
            "nebius_api_key": state.access_key,
            "nebius_secret_key": state.secret_key,
        }
        for key, value in saved.items():
            if value and key not in extra_vars:
                merged_vars[key] = value
        apply_storage_env_vars(merged_vars, explicit_vars=extra_vars)
    if byovm:
        apply_project_storage_vars(
            merged_vars,
            project=proj_alias,
            explicit_vars=extra_vars,
            warn=console.print,
        )
        try:
            from npa.clients.config import ConfigError, resolve_terraform_state

            state = resolve_terraform_state(proj_alias)
        except ConfigError:
            state = None
        if state is not None:
            saved = {
                "s3_bucket": state.bucket,
                "s3_endpoint": state.endpoint,
                "nebius_api_key": state.access_key,
                "nebius_secret_key": state.secret_key,
            }
            for key, value in saved.items():
                if value and key not in extra_vars and not merged_vars.get(key):
                    merged_vars[key] = value
        apply_storage_env_vars(merged_vars, explicit_vars=extra_vars)
    if not byovm:
        try:
            apply_boot_disk_tf_vars(merged_vars, runtime, disk_size)
        except ValueError as exc:
            _fail(str(exc))
            return

    instance_name = f"genesis-{proj_alias}-{wb_name}"
    cloud_init_workbench_type = (
        "lerobot-container"
        if runtime_uses_container(runtime)
        else "genesis"
    )

    # ── Destroy flow ─────────────────────────────────────────────────
    if destroy:
        if byovm:
            console.print(f"  [1/1] Unregistering BYOVM workbench {proj_alias}/{wb_name}...")
            if not dry_run:
                from npa.clients.config import remove_workbench_config

                remove_workbench_config(proj_alias, wb_name)
            console.print(f"  {proj_alias}/{wb_name} unregistered. BYOVM host was not modified.")
            return

        console.print(f"  [1/2] Destroying {proj_alias}/{wb_name}...")
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

        # Read the stored tf_instance_name if available.
        from npa.clients.config import ConfigError, resolve_ssh_config
        try:
            wb_cfg = resolve_ssh_config(project=proj_alias, name=wb_name)
            if wb_cfg.tf_instance_name:
                instance_name = wb_cfg.tf_instance_name
        except ConfigError:
            pass

        try:
            provisioner.destroy(
                tf_dir=resolved_tf_dir or None,
                tf_vars={"gpu_platform": gpu_type, "gpu_preset": gpu_preset,
                         "instance_name": instance_name,
                         "enable_preemptible": "true" if preemptible else "false",
                         **merged_vars},
            )
        except ProvisionerError as exc:
            _fail(f"Terraform destroy failed: {exc}")
            return

        console.print("  [2/2] Cleaning up config...")
        from npa.clients.config import remove_workbench_config
        remove_workbench_config(proj_alias, wb_name)
        if use_remote_state:
            provisioner.cleanup_working_dir(proj_alias, wb_name)
        console.print(f"  {proj_alias}/{wb_name} destroyed.")
        return

    # ── Provision flow ───────────────────────────────────────────────
    tf_outputs: dict[str, Any] = {}
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

        console.print(f"  [1/3] Initializing Terraform ({proj_alias}/{wb_name})...")
        if not dry_run:
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

        all_vars = {
            "gpu_platform": gpu_type, "gpu_preset": gpu_preset,
            "instance_name": instance_name,
            "workbench_type": cloud_init_workbench_type,
            "enable_preemptible": "true" if preemptible else "false",
            **merged_vars,
        }
        console.print(f"  [2/3] Applying Terraform (gpu={gpu_type}, region={env_region})...")
        if dry_run:
            tf_outputs = {"vm_ip": "<pending>", "ssh_user": "ubuntu",
                          "ssh_key_path": "~/.ssh/id_ed25519"}
        else:
            try:
                tf_outputs = provisioner.apply(tf_dir=resolved_tf_dir or None, tf_vars=all_vars)
            except ProvisionerError as exc:
                _fail(f"Terraform apply failed: {exc}")
                return
        console.print(f"    VM IP: {tf_outputs.get('vm_ip', 'unknown')}")
    else:
        console.print(f"  [1/2] {'Using BYOVM target' if byovm else 'Skipping infra, reading existing config'}...")
        resolved_tf_dir = tf_dir

        if byovm:
            try:
                from npa.clients.config import resolve_credentials
                from npa.clients.ssh import SSHClient, SSHError

                target = resolve_byovm_target(host=host, ssh_key=ssh_key, ssh_user=ssh_user)
                bucket = merged_vars.get("s3_bucket", "") or os.environ.get("NPA_CHECKPOINT_BUCKET", "")
                storage_ep = merged_vars.get("s3_endpoint", "") or os.environ.get("AWS_ENDPOINT_URL", "")
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
        elif not tf_outputs:
            from npa.clients.config import _load_yaml, _deep_get, _resolve_project_section, _resolve_workbench_in_project
            yml = _load_yaml()
            try:
                proj = _resolve_project_section(yml, proj_alias)
                wb = _resolve_workbench_in_project(proj, wb_name, yml)
            except Exception:
                wb = {}
            tf_outputs = {
                "vm_ip": _deep_get(wb, "ssh", "host", default=""),
                "ssh_user": _deep_get(wb, "ssh", "user", default="ubuntu"),
                "ssh_key_path": _deep_get(wb, "ssh", "key_path", default="~/.ssh/id_ed25519"),
            }

        if not tf_outputs.get("vm_ip"):
            _fail("No VM IP found. Run without --skip-infra first, or set config manually.")
            return

    # ── Verify SSH connectivity ──────────────────────────────────────
    vm_ip = tf_outputs.get("vm_ip", "")
    ssh_user = tf_outputs.get("ssh_user", "ubuntu")
    ssh_key = tf_outputs.get("ssh_key_path", "~/.ssh/id_ed25519")

    step_label = "[3/3]" if not skip_infra else "[2/2]"
    console.print(f"  {step_label} Writing config ({proj_alias}/{wb_name})...")

    bucket = tf_outputs.get("storage_bucket", merged_vars.get("s3_bucket", ""))
    bucket_display = bucket if str(bucket).startswith("s3://") else (f"s3://{bucket}/checkpoints/" if bucket else "")
    storage_ep = tf_outputs.get("storage_endpoint", merged_vars.get("s3_endpoint", ""))
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
                        "workbench_type": "genesis",
                        "runtime": runtime.value,
                        "app_status": "provisioned",
                        **byovm_fields,
                        "ssh": {"host": vm_ip, "user": ssh_user, "key_path": ssh_key},
                        "storage": {"checkpoint_bucket": bucket_display, "endpoint_url": storage_ep},
                    },
                },
            },
        },
    }

    from npa.clients import config as client_config

    if default or not client_config.list_projects():
        config_data["default_project"] = proj_alias
        config_data["default_workbench"] = wb_name

    if not dry_run:
        from npa.clients.config import update_workbench_app_status, write_config
        write_config(config_data)
        if runtime == WorkbenchRuntime.vm:
            update_workbench_app_status(proj_alias, wb_name, "healthy")
        console.print("    Saved to ~/.npa/config.yaml")

    if runtime_uses_container(runtime):
        step_label = "[container]"
        console.print(f"  {step_label} Starting Genesis container...")
        if not dry_run:
            from npa.clients.config import SSHConfig, resolve_container_registry, resolve_credentials, update_workbench_app_status
            from npa.clients.ssh import SSHClient, SSHError
            from npa.deploy.configurator import (
                deploy_workbench_container,
                write_manifest,
                write_remote_docker_env_file,
            )
            from npa.deploy.images import container_image_for_tool

            credentials = resolve_credentials()
            ssh_cfg = SSHConfig(host=vm_ip, user=ssh_user, key_path=ssh_key, tokens=credentials.tokens)
            ssh = SSHClient(ssh_cfg)
            try:
                code, _, _ = ssh.run("echo connected")
                if code != 0:
                    update_workbench_app_status(proj_alias, wb_name, "install_failed")
                    _fail(f"SSH connection test failed (exit {code})")
                    return
                update_workbench_app_status(proj_alias, wb_name, "installing")
                service_env = {
                    "AWS_ACCESS_KEY_ID": merged_vars.get("nebius_api_key", ""),
                    "AWS_SECRET_ACCESS_KEY": merged_vars.get("nebius_secret_key", ""),
                    "AWS_ENDPOINT_URL": storage_ep,
                    "NEBIUS_S3_ENDPOINT": storage_ep,
                    "NEBIUS_S3_BUCKET": bucket,
                    "NEBIUS_REGION": env_region,
                    "NVIDIA_DRIVER_CAPABILITIES": "all",
                    "MUJOCO_GL": "egl",
                    "PYOPENGL_PLATFORM": "egl",
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
                    "/opt/lerobot/.env",
                    service_env,
                    owner=ssh_user,
                )
                image_ref = container_image_for_tool(
                    "genesis",
                    registry=resolve_container_registry(proj_alias),
                )
                deploy_workbench_container(
                    ssh,
                    image_ref=image_ref,
                    container_name="npa-genesis",
                    env_file="/opt/lerobot/.env",
                    group_add=["0", "video", "render"],
                    devices=["/dev/dri"],
                    volumes=[
                        "/opt/lerobot/.env:/opt/lerobot/.env:ro",
                        "/opt/genesis/outputs:/opt/genesis/outputs",
                    ],
                    work_dirs=["/opt/genesis/outputs"],
                    registry_token=merged_vars.get("iam_token", ""),
                )
                write_manifest(ssh, tool="genesis", version=image_ref.rsplit(":", 1)[-1], deployed_by=f"npa deploy --runtime {runtime.value}")
                update_workbench_app_status(proj_alias, wb_name, "healthy")
            except SSHError as exc:
                update_workbench_app_status(proj_alias, wb_name, "install_failed")
                _fail(f"Genesis container deployment failed: {exc}")
                return

    console.print("")
    console.print(f"[bold green]Deploy complete.[/bold green] ({proj_alias}/{wb_name})")
    console.print(f"  SSH:  ssh -i {ssh_key} {ssh_user}@{vm_ip}")
    console.print("")
    console.print(f"  Try: npa workbench genesis -p {proj_alias} -n {wb_name} status")

    if output_format == OutputFormat.json:
        typer.echo(json.dumps({
            "project": proj_alias, "name": wb_name,
            "vm_ip": vm_ip, "ssh_user": ssh_user,
            "gpu_platform": byovm_fields.get("gpu_platform", gpu_type),
            "gpu_preset": byovm_fields.get("gpu_preset", gpu_preset),
            "gpu_count": byovm_fields.get("gpu_count"),
            "runtime": runtime.value,
        }, indent=2))


@app.command("status")
def status_cmd(
    output_format: OutputFormat = typer.Option(OutputFormat.text, "--output-format", help="Output format."),
) -> None:
    """Check Genesis VM status via SSH (processes, GPU, conda env)."""
    cfg = _get_ssh_config()

    from npa.clients.ssh import SSHClient, SSHError

    ssh = SSHClient(cfg.ssh)

    if runtime_uses_container(getattr(cfg, "runtime", "vm")):
        status_cmd_str = (
            "echo '=== hostname ===' && hostname && "
            "echo '' && echo '=== uptime ===' && uptime && "
            "echo '' && echo '=== nvidia-smi ===' && nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo 'nvidia-smi not available'; "
            "echo '' && echo '=== container ===' && sudo docker inspect -f 'state={{.State.Status}} image={{.Config.Image}}' npa-genesis && "
            "echo '' && echo '=== genesis version ===' && sudo docker exec npa-genesis bash -lc 'python -c \"import genesis; print(getattr(genesis, \\\"__version__\\\", \\\"unknown\\\"))\"'"
        )
    else:
        status_cmd_str = (
            "echo '=== hostname ===' && hostname && "
            "echo '' && echo '=== uptime ===' && uptime && "
            "echo '' && echo '=== nvidia-smi ===' && nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo 'nvidia-smi not available' && "
            "echo '' && echo '=== conda envs ===' && /opt/conda/bin/conda env list 2>/dev/null || echo 'conda not installed' && "
            "echo '' && echo '=== genesis processes ===' && pgrep -af 'genesis\\|npa.*genesis' 2>/dev/null || echo 'no genesis processes'"
        )

    try:
        code, out, err = ssh.run_or_raise(status_cmd_str)
    except SSHError as exc:
        if output_format == OutputFormat.json:
            typer.echo(json.dumps({
                "host": cfg.ssh.host,
                "app_status": cfg.app_status or "unknown",
                "status": "unreachable",
                "error": str(exc),
            }, indent=2))
        else:
            typer.echo(f"app_status: {cfg.app_status or 'unknown'}")
        _fail(f"SSH error: {exc}")
        return

    if output_format == OutputFormat.json:
        typer.echo(json.dumps({
            "host": cfg.ssh.host,
            "app_status": cfg.app_status or "unknown",
            "runtime": getattr(cfg, "runtime", "vm"),
            "status": "reachable" if code == 0 else "error",
            "output": out.strip() if out else "",
        }, indent=2))
    else:
        console.print(f"[bold]Genesis VM: {cfg.ssh.host}[/bold]")
        typer.echo(f"app_status: {cfg.app_status or 'unknown'}")
        typer.echo(f"runtime: {getattr(cfg, 'runtime', 'vm')}")
        if out:
            typer.echo(out.strip())
        if code != 0 and err:
            console.print(f"[red]stderr:[/red]\n{err.strip()[-500:]}")


@app.command("system-info")
def system_info_cmd(
    output_format: OutputFormat = typer.Option(OutputFormat.text, "--output-format", help="Output format."),
) -> None:
    """Collect and display system hardware information from the Genesis VM."""
    cfg = _get_ssh_config()

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
            "sudo docker inspect -f 'state={{.State.Status}} image={{.Config.Image}}' npa-genesis"
        )

    try:
        code, out, err = ssh.run_or_raise(info_cmd)
    except SSHError as exc:
        _fail(f"SSH error: {exc}")
        return

    if output_format == OutputFormat.json:
        typer.echo(json.dumps({
            "host": cfg.ssh.host,
            "runtime": getattr(cfg, "runtime", "vm"),
            "system_info": out.strip(),
        }, indent=2))
    else:
        if out:
            typer.echo(out.strip())
        if code != 0 and err:
            console.print(f"[red]stderr:[/red]\n{err.strip()[-500:]}")
