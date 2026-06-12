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
    sonic_image,
)
from npa.clients.serverless import EndpointNotFoundError, ServerlessClient, ServerlessClientError
from npa.serverless_common import (
    SubnetResolutionError,
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
)
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
    training_config: TrainingConfig | None = None,
) -> str:
    """Build the remote command for a real SONIC serverless train job."""

    config = training_config or TrainingConfig()
    local_dir = "/tmp/npa-sonic-train"
    upload = build_serverless_output_upload_cmd(local_dir, "")
    training_env = config.env()
    env_lines = "\n".join(f"export {key}={value!r}" for key, value in training_env.items())
    body = (
        'if [ -x /isaac-sim/python.sh ]; then NPA_PYTHON_BIN=/isaac-sim/python.sh; '
        'elif [ -x /opt/isaac-lab/venv/bin/python ]; then NPA_PYTHON_BIN=/opt/isaac-lab/venv/bin/python; '
        'else NPA_PYTHON_BIN="${NPA_PYTHON_BIN:-python3}"; fi\n'
        'if ! command -v "$NPA_PYTHON_BIN" >/dev/null 2>&1; then NPA_PYTHON_BIN=python; fi\n'
        f"{env_lines}\n"
        f"export NPA_LOCAL_OUTPUT_DIR={local_dir!r}\n"
        "export SONIC_RUN_REAL_TRAIN=1\n"
        f"export SONIC_CHECKPOINT={checkpoint!r}\n"
        f"export SONIC_CHECKPOINT_PATH={checkpoint!r}\n"
        f"export SONIC_DATA_PATH={(config.data_path or data_path)!r}\n"
        f"export SONIC_SAMPLE_DATA={'1' if sample_data else '0'}\n"
        f"export SONIC_EMBODIMENT={embodiment!r}\n"
        f"export SONIC_NUM_ENVS={str(num_envs)!r}\n"
        f"export SONIC_HEADLESS={'True' if headless else 'False'}\n"
        f"export SONIC_MAX_ITERATIONS={str(max_iterations)!r}\n"
        f"export SONIC_ISAAC_LAB_VERSION={isaac_lab_version!r}\n"
        'if [ -x /entrypoint.sh ]; then /entrypoint.sh train; '
        'else echo "/entrypoint.sh not found in SONIC image" >&2; exit 127; fi\n'
        "sonic_rc=$?\n"
        f"{upload}\n"
        'exit "$sonic_rc"'
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
    image_variant: str,
    gpu_type: str,
    gpu_count: int,
    gpu_preset: str,
    subnet_id: str,
    job_name: str,
    submit_only: bool,
    poll_interval: float,
    timeout: float,
    output_format: OutputFormat,
    training_config: TrainingConfig,
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

    ctx = context()
    resolved_project_id = resolve_project_id(project_id)
    name = job_name or serverless_job_name(ctx.project, ctx.name, "sonic")
    out = output_path.rstrip("/") + "/"
    try:
        subnet = resolve_subnet(
            project_id=resolved_project_id,
            explicit_subnet_id=subnet_id,
        )
    except SubnetResolutionError as exc:
        fail(str(exc))
    try:
        resolved_image = sonic_image(
            ctx.project,
            image,
            gpu_target=platform,
            image_variant=image_variant,
        )
    except ValueError as exc:
        fail(str(exc))
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
    env.update(training_config.env())
    safe_env, secret_env = split_serverless_env(env)
    extra_env.update(secret_env)
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
                    "training_config": training_config.public_dict(),
                },
                output_format,
            )
            return
        info = client.create_job(
            project_id=resolved_project_id,
            name=name,
            image=resolved_image,
            command=build_sonic_serverless_train_command(
                checkpoint=checkpoint,
                data_path=data_path,
                sample_data=sample_data,
                embodiment=embodiment,
                num_envs=num_envs,
                headless=headless,
                max_iterations=max_iterations,
                isaac_lab_version=isaac_lab_version,
                training_config=training_config,
            ),
            gpu_type=platform,
            gpu_count=resolved_gpu_count,
            preset=preset,
            subnet_id=subnet,
            output_path=out,
            env=safe_env,
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
            "training_config": training_config.public_dict(),
        },
        output_format,
    )


def train_cmd(
    runtime: TrainRuntime = typer.Option(TrainRuntime.serverless, "--runtime", help="Runtime."),
    checkpoint: str = typer.Option(DEFAULT_CHECKPOINT, "--checkpoint", help="Checkpoint ref or path."),
    data_path: str = typer.Option("", "--data-path", help="Training data path or URI."),
    sample_data: bool = typer.Option(False, "--sample-data", help="Use SONIC sample data for smoke."),
    override: list[str] = typer.Option(
        [],
        "--override",
        help="Generic Hydra override as KEY=VALUE. Repeat for learning rate, clip params, terminations, or any trainer key.",
    ),
    wandb_enabled: bool = typer.Option(False, "--wandb/--no-wandb", help="Enable W&B logging for the training run."),
    wandb_project: str = typer.Option("", "--wandb-project", help="W&B project name."),
    wandb_run_name: str = typer.Option("", "--wandb-run-name", help="W&B run name."),
    wandb_mode: str = typer.Option("offline", "--wandb-mode", help="W&B mode such as online, offline, or disabled."),
    checkpoint_s3_uri: str = typer.Option("", "--checkpoint-s3-uri", help="S3 URI for checkpoint upload."),
    checkpoint_s3_endpoint_url: str = typer.Option("", "--checkpoint-s3-endpoint-url", help="S3-compatible endpoint URL."),
    checkpoint_s3_access_key_id: str = typer.Option("", "--checkpoint-s3-access-key-id", help="S3 access key ID."),
    checkpoint_s3_secret_access_key: str = typer.Option("", "--checkpoint-s3-secret-access-key", help="S3 secret access key."),
    embodiment: str = typer.Option(DEFAULT_EMBODIMENT, "--embodiment", help="SONIC embodiment tag."),
    num_envs: int = typer.Option(16, "--num-envs", help="Number of Isaac Lab environments."),
    headless: bool = typer.Option(True, "--headless/--no-headless", help="Run Isaac Lab headless."),
    max_iterations: int = typer.Option(5, "--max-iterations", "--steps", help="Training iterations for smoke."),
    isaac_lab_version: str = typer.Option("2.3+", "--isaac-lab-version", help="Expected Isaac Lab version."),
    hf_token_env: str = typer.Option("HF_TOKEN", "--hf-token-env", help="Environment variable containing HF token."),
    output_path: str = typer.Option("", "--output-path", "-o", help="S3 URI where artifacts are written."),
    project_id: str = typer.Option("", "--project-id", help="Nebius project ID for serverless Jobs."),
    image: str = typer.Option("", "--image", help="Container image for the serverless Job."),
    image_variant: str = typer.Option(
        "",
        "--image-variant",
        help="SONIC image manifest variant. Defaults from --gpu-type.",
    ),
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
    except TrainingConfigError as exc:
        fail(str(exc))
    runtime_value = enum_value(runtime)
    if num_envs <= 0:
        fail(f"--num-envs must be positive, got {num_envs}")
    if max_iterations <= 0:
        fail(f"--max-iterations/--steps must be positive, got {max_iterations}")
    embodiment_tag = normalize_embodiment(embodiment)
    data_path = training_config.data_path
    effective_sample_data = sample_data or not data_path
    checkpoint_output_path = resolve_checkpoint_s3_uri(training_config, output_path)
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
            output_path=checkpoint_output_path,
            project_id=project_id,
            image=image,
            image_variant=image_variant,
            gpu_type=gpu_type,
            gpu_count=gpu_count,
            gpu_preset=gpu_preset,
            subnet_id=subnet_id,
            job_name=job_name,
            submit_only=submit_only,
            poll_interval=poll_interval,
            timeout=timeout,
            output_format=output_format,
            training_config=training_config,
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
            "output_path": checkpoint_output_path,
            "training_config": training_config.public_dict(),
        },
        output_format,
    )
