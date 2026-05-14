"""SONIC deploy command."""

from __future__ import annotations

from npa.cli.workbench.sonic.helpers import (
    CheckpointSource,
    DeployMode,
    OutputFormat,
    WorkbenchRuntime,
    context,
    enum_value,
    output,
    require_real_confirmation,
    validate_checkpoint_args,
    validate_port,
    validate_tensorrt_version,
)
from npa.serverless_common import validate_output_path
import typer


def deploy_cmd(
    runtime: WorkbenchRuntime = typer.Option(WorkbenchRuntime.vm, "--runtime", help="Runtime."),
    mode: DeployMode = typer.Option(DeployMode.sim, "--mode", help="SONIC deploy mode."),
    checkpoint_source: CheckpointSource = typer.Option(
        CheckpointSource.hf, "--checkpoint-source", help="Checkpoint source."
    ),
    model_repo: str = typer.Option("nvidia/GEAR-SONIC", "--model-repo", help="Hugging Face model repo."),
    checkpoint_path: str = typer.Option("", "--checkpoint-path", help="Local, S3, or HF checkpoint path."),
    hf_token_env: str = typer.Option("HF_TOKEN", "--hf-token-env", help="Environment variable containing the HF token."),
    tensorrt_version: str = typer.Option("", "--tensorrt-version", help="TensorRT version override."),
    port: int = typer.Option(5557, "--port", help="Realtime debug/visualization port."),
    zmq_port: int = typer.Option(5556, "--zmq-port", help="ZMQ input port."),
    project_id: str = typer.Option("", "--project-id", help="Nebius project ID."),
    tenant_id: str = typer.Option("", "--tenant-id", help="Nebius tenant ID."),
    region: str = typer.Option("", "--region", help="Nebius region."),
    gpu_type: str = typer.Option("", "--gpu-type", help="GPU type for GPU deploy paths."),
    output_path: str = typer.Option("", "--output-path", help="S3 URI for serverless deploy smoke output."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Plan only."),
    default: bool = typer.Option(False, "--default", help="Mark this SONIC runtime as default when persisted."),
    confirm_real: bool = typer.Option(False, "--confirm-real", help="Required to acknowledge real robot mode."),
    output_format: OutputFormat = typer.Option(
        OutputFormat.text, "--output-format", "--output", help="Output format."
    ),
) -> None:
    """Prepare or plan a SONIC runtime."""

    runtime_value = enum_value(runtime)
    mode_value = enum_value(mode)
    validate_port(port, "--port")
    validate_port(zmq_port, "--zmq-port")
    validate_checkpoint_args(checkpoint_source, checkpoint_path)
    validate_tensorrt_version(tensorrt_version)
    require_real_confirmation(mode_value, confirm_real)
    if runtime_value == "serverless":
        if not output_path:
            from npa.cli.workbench.sonic.helpers import fail

            fail("SONIC deploy --runtime serverless requires --output-path.")
        try:
            validate_output_path(output_path)
        except ValueError as exc:
            from npa.cli.workbench.sonic.helpers import fail

            fail(str(exc))

    ctx = context()
    plan = {
        "status": "planned" if dry_run or runtime_value in {"vm", "byovm"} else "ready",
        "project": ctx.project,
        "workbench": ctx.name,
        "runtime": runtime_value,
        "mode": mode_value,
        "checkpoint_source": checkpoint_source.value,
        "model_repo": model_repo,
        "checkpoint_path": checkpoint_path,
        "hf_token_env": hf_token_env,
        "tensorrt_version": tensorrt_version or "auto",
        "port": port,
        "zmq_port": zmq_port,
        "project_id": project_id,
        "tenant_id": tenant_id,
        "region": region,
        "gpu_type": gpu_type,
        "output_path": output_path,
        "default": default,
        "next": "npa workbench sonic serve --runtime container --mode sim --input-type keyboard",
    }
    output(plan, output_format)
