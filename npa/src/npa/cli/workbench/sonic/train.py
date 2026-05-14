"""SONIC train command."""

from __future__ import annotations

from npa.cli.workbench.sonic.helpers import (
    DEFAULT_CHECKPOINT,
    DEFAULT_EMBODIMENT,
    OutputFormat,
    TrainRuntime,
    context,
    enum_value,
    fail,
    normalize_embodiment,
    output,
    remote_bash,
    resolve_project_id,
    serverless_job_env,
    serverless_job_name,
    serverless_subnet_id,
    sonic_image,
)
from npa.clients.serverless import EndpointNotFoundError, ServerlessClient, ServerlessClientError
from npa.serverless_common import build_serverless_output_upload_cmd, resolve_gpu_platform, validate_output_path
import typer


def build_sonic_serverless_train_command(
    *,
    checkpoint: str,
    data_path: str,
    sample_data: bool,
    embodiment: str,
    num_envs: int,
    headless: bool,
    max_iterations: int,
    isaac_lab_version: str,
) -> str:
    """Build the remote command for a SONIC serverless smoke/train job."""

    local_dir = "/tmp/npa-sonic-train"
    script = f"""
import importlib
import json
import os
import pathlib
import time

out = pathlib.Path("{local_dir}")
out.mkdir(parents=True, exist_ok=True)
started = time.time()

def import_state(name):
    try:
        importlib.import_module(name)
        return "available"
    except Exception as exc:
        return f"unavailable: {{type(exc).__name__}}: {{exc}}"

summary = {{
    "status": "success",
    "tool": "sonic",
    "embodiment": {embodiment!r},
    "checkpoint": {checkpoint!r},
    "data_path": {data_path!r},
    "sample_data": {sample_data},
    "num_envs": {num_envs},
    "headless": {headless},
    "max_iterations": {max_iterations},
    "isaac_lab_version": {isaac_lab_version!r},
    "gear_sonic_import": import_state("gear_sonic"),
    "isaaclab_import": import_state("isaaclab"),
    "job": os.environ.get("NPA_JOB_NAME", ""),
    "duration_seconds": round(time.time() - started, 3),
}}
(out / "sonic_smoke_result.json").write_text(json.dumps(summary, indent=2))
(out / "sonic_train_summary.json").write_text(json.dumps(summary, indent=2))
(out / "checkpoint_smoke.json").write_text(json.dumps({{"format": "npa_sonic_serverless_smoke_v1", **summary}}, indent=2))
print("NPA_SONIC_SERVERLESS_TRAIN_DONE", os.environ.get("NPA_OUTPUT_PATH", ""), flush=True)
""".strip()
    upload = build_serverless_output_upload_cmd(local_dir, "")
    body = (
        'if [ -x /isaac-sim/python.sh ]; then NPA_PYTHON_BIN=/isaac-sim/python.sh; '
        'elif [ -x /opt/isaac-lab/venv/bin/python ]; then NPA_PYTHON_BIN=/opt/isaac-lab/venv/bin/python; '
        'else NPA_PYTHON_BIN="${NPA_PYTHON_BIN:-python3}"; fi\n'
        'if ! command -v "$NPA_PYTHON_BIN" >/dev/null 2>&1; then NPA_PYTHON_BIN=python; fi\n'
        f'"$NPA_PYTHON_BIN" <<\'PY\'\n{script}\nPY\n{upload}'
    )
    return remote_bash(body)


def _run_serverless_train(
    *,
    checkpoint: str,
    data_path: str,
    sample_data: bool,
    embodiment: str,
    num_envs: int,
    headless: bool,
    max_iterations: int,
    isaac_lab_version: str,
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
    output_format: OutputFormat,
) -> None:
    if not output_path:
        fail("SONIC train --runtime serverless requires --output-path.")
    try:
        validate_output_path(output_path)
        platform, preset, resolved_gpu_count = resolve_gpu_platform(gpu_type, gpu_count)
    except ValueError as exc:
        fail(str(exc))
    if gpu_preset:
        preset = gpu_preset

    if platform not in {"gpu-l40s-a", "gpu-l40s-d", "gpu-rtx6000"}:
        from npa.cli.workbench.sonic.helpers import console

        console.print(
            "[yellow]Warning:[/yellow] SONIC Isaac Lab training usually needs RT-core GPUs; "
            "prefer --gpu-type l40s or --gpu-type gpu-rtx-pro-6000."
        )

    ctx = context()
    resolved_project_id = resolve_project_id(project_id)
    name = job_name or serverless_job_name(ctx.project, ctx.name, "sonic")
    out = output_path.rstrip("/") + "/"
    subnet = subnet_id or serverless_subnet_id(ctx.project, ctx.name, resolved_project_id)
    env, extra_env = serverless_job_env(
        ctx.project,
        out,
        {
            "NPA_JOB_NAME": name,
            "SONIC_SERVERLESS_SMOKE": "1",
            "SONIC_EMBODIMENT": embodiment,
            "SONIC_CHECKPOINT": checkpoint,
        },
    )
    client = ServerlessClient()
    try:
        existing = client.get_job(name, resolved_project_id)
    except EndpointNotFoundError:
        existing = None
    try:
        if existing is not None:
            info = (
                existing
                if submit_only or existing.status in {"succeeded", "failed", "cancelled"}
                else client.poll_job(existing.id, resolved_project_id, interval_s=poll_interval, ceiling_s=timeout)
            )
            output(
                {
                    "status": "existing",
                    "job_id": info.id,
                    "job_name": info.name,
                    "job_status": info.status,
                    "output_path": out,
                },
                output_format,
            )
            return
        info = client.create_job(
            project_id=resolved_project_id,
            name=name,
            image=sonic_image(ctx.project, image),
            command=build_sonic_serverless_train_command(
                checkpoint=checkpoint,
                data_path=data_path,
                sample_data=sample_data,
                embodiment=embodiment,
                num_envs=num_envs,
                headless=headless,
                max_iterations=max_iterations,
                isaac_lab_version=isaac_lab_version,
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
        fail(str(exc))
    except ServerlessClientError as exc:
        fail(f"Serverless Job failed: {exc}")
    except TimeoutError as exc:
        fail(str(exc))
    output(
        {
            "status": "submitted" if submit_only else info.status,
            "job_id": info.id,
            "job_name": info.name,
            "output_path": out,
            "embodiment": embodiment,
        },
        output_format,
    )


def train_cmd(
    runtime: TrainRuntime = typer.Option(TrainRuntime.serverless, "--runtime", help="Runtime."),
    checkpoint: str = typer.Option(DEFAULT_CHECKPOINT, "--checkpoint", help="Checkpoint ref or path."),
    data_path: str = typer.Option("", "--data-path", help="Training data path or URI."),
    sample_data: bool = typer.Option(False, "--sample-data", help="Use SONIC sample data for smoke."),
    embodiment: str = typer.Option(DEFAULT_EMBODIMENT, "--embodiment", help="SONIC embodiment tag."),
    num_envs: int = typer.Option(16, "--num-envs", help="Number of Isaac Lab environments."),
    headless: bool = typer.Option(True, "--headless/--no-headless", help="Run Isaac Lab headless."),
    max_iterations: int = typer.Option(5, "--max-iterations", "--steps", help="Training iterations for smoke."),
    isaac_lab_version: str = typer.Option("2.3+", "--isaac-lab-version", help="Expected Isaac Lab version."),
    hf_token_env: str = typer.Option("HF_TOKEN", "--hf-token-env", help="Environment variable containing HF token."),
    output_path: str = typer.Option("", "--output-path", "-o", help="S3 URI where artifacts are written."),
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
    output_format: OutputFormat = typer.Option(
        OutputFormat.text, "--output-format", "--output", help="Output format."
    ),
) -> None:
    """Run SONIC Isaac Lab training or smoke validation."""

    runtime_value = enum_value(runtime)
    if num_envs <= 0:
        fail(f"--num-envs must be positive, got {num_envs}")
    if max_iterations <= 0:
        fail(f"--max-iterations/--steps must be positive, got {max_iterations}")
    embodiment_tag = normalize_embodiment(embodiment)
    effective_sample_data = sample_data or not data_path
    if runtime_value == "serverless":
        _run_serverless_train(
            checkpoint=checkpoint,
            data_path=data_path,
            sample_data=effective_sample_data,
            embodiment=embodiment_tag,
            num_envs=num_envs,
            headless=headless,
            max_iterations=max_iterations,
            isaac_lab_version=isaac_lab_version,
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
            output_format=output_format,
        )
        return
    output(
        {
            "status": "planned",
            "runtime": runtime_value,
            "checkpoint": checkpoint,
            "data_path": data_path,
            "sample_data": effective_sample_data,
            "embodiment": embodiment_tag,
            "num_envs": num_envs,
            "headless": headless,
            "max_iterations": max_iterations,
            "hf_token_env": hf_token_env,
            "output_path": output_path,
        },
        output_format,
    )
