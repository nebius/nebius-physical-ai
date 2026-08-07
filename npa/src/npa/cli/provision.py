"""Runtime provisioning CLI hooks."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path

import typer

from npa.provisioning import provision_if_absent

app = typer.Typer(
    name="provision-if-absent",
    help="Ensure configured Kubernetes and S3 runtime resources exist.",
    no_args_is_help=False,
)


class OutputFormat(str, Enum):
    text = "text"
    json = "json"


@app.callback(invoke_without_command=True)
def provision_if_absent_cmd(
    project: str = typer.Option(
        "", "--project", help="Project alias from ~/.npa/config.yaml."
    ),
    cluster_name: str = typer.Option(
        "npa-cluster", "--cluster-name", help="Cluster profile/context name."
    ),
    terraform_dir: Path | None = typer.Option(
        None, "--terraform-dir", help="Terraform cluster directory."
    ),
    kubeconfig: Path | None = typer.Option(
        None, "--kubeconfig", help="Dedicated kubeconfig path."
    ),
    context_name: str = typer.Option("", "--context", help="Kubeconfig context name."),
    skip_k8s: bool = typer.Option(
        False, "--skip-k8s", help="Do not ensure Kubernetes."
    ),
    skip_s3: bool = typer.Option(False, "--skip-s3", help="Do not ensure S3."),
    validate: bool = typer.Option(
        True, "--validate/--skip-validate", help="Run post-apply Kubernetes validation."
    ),
    sky_smoke: bool = typer.Option(
        False, "--sky-smoke/--skip-sky-smoke", help="Run a SkyPilot GPU smoke task."
    ),
    gpu_nodes: int = typer.Option(
        -1,
        "--gpu-nodes",
        help="Number of GPU nodes, matching `npa cluster up`. -1 keeps the configured value.",
    ),
    cpu_nodes: int = typer.Option(
        -1,
        "--cpu-nodes",
        help="Number of CPU nodes, matching `npa cluster up`. -1 keeps the configured value.",
    ),
    cpu_platform: str = typer.Option(
        "", "--cpu-platform", help="CPU node platform, matching `npa cluster up`."
    ),
    cpu_preset: str = typer.Option(
        "", "--cpu-preset", help="CPU node preset, matching `npa cluster up`."
    ),
    gpu_platform: str = typer.Option(
        "", "--gpu-platform", help="GPU node platform, matching `npa cluster up`."
    ),
    gpu_preset: str = typer.Option(
        "", "--gpu-preset", help="GPU node preset, matching `npa cluster up`."
    ),
    preemptible: bool | None = typer.Option(
        None,
        "--preemptible/--on-demand",
        help=(
            "Run the GPU node group as preemptible, matching `npa cluster up`. "
            "This changes the capacity pool but not hard instance/disk/IP quotas; "
            "a reclaim stops the node mid-run."
        ),
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Resolve settings and print intended actions only."
    ),
    timeout: int = typer.Option(
        120, "--timeout", help="Terraform apply timeout in minutes."
    ),
    accelerator: str = typer.Option(
        "",
        "--accelerator",
        help="Requested SkyPilot accelerator (for example RTXPRO6000:1) to gate readiness.",
    ),
    gpu_readiness_timeout: float = typer.Option(
        600.0,
        "--gpu-readiness-timeout",
        help="Seconds to wait for SkyPilot GPU discovery without deleting capacity.",
    ),
    gpu_readiness_poll_interval: float = typer.Option(
        10.0,
        "--gpu-readiness-poll-interval",
        help="Seconds between SkyPilot GPU discovery checks.",
    ),
    sky_bin: str = typer.Option("", "--sky-bin", help="Pinned SkyPilot executable."),
    output_format: OutputFormat = typer.Option(
        OutputFormat.text, "--output-format", help="Output format."
    ),
) -> None:
    """Provision S3 and Kubernetes only when they are absent."""
    result = provision_if_absent(
        project=project or None,
        cluster_name=cluster_name,
        terraform_dir=terraform_dir,
        kubeconfig=kubeconfig,
        context_name=context_name,
        skip_k8s=skip_k8s,
        skip_s3=skip_s3,
        validate=validate,
        sky_smoke=sky_smoke,
        dry_run=dry_run,
        timeout=timeout,
        gpu_nodes=gpu_nodes,
        cpu_nodes=cpu_nodes,
        cpu_platform=cpu_platform,
        cpu_preset=cpu_preset,
        gpu_platform=gpu_platform,
        gpu_preset=gpu_preset,
        preemptible=preemptible,
        accelerator=accelerator,
        gpu_readiness_timeout=gpu_readiness_timeout,
        gpu_readiness_poll_interval=gpu_readiness_poll_interval,
        sky_bin=sky_bin,
        output_format=output_format.value,
    )
    payload = result.to_dict()
    if output_format == OutputFormat.json:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(f"status: {result.status}")
        typer.echo(f"project: {result.project}")
        typer.echo(f"cluster: {result.cluster_name}")
        typer.echo(f"kubeconfig: {result.kubeconfig_path}")
        typer.echo(f"storage: {result.storage_bucket}")
        typer.echo(f"gpu_readiness: {result.gpu_readiness}")
        if result.operation_id:
            typer.echo(f"operation_id: {result.operation_id}")
            typer.echo(f"operation_journal: {result.operation_journal}")
            typer.echo(f"recovery_command: {result.recovery_command}")
        for action in result.actions:
            typer.echo(f"action: {action}")
        for warning in result.warnings:
            typer.echo(f"warning: {warning}")
        if result.status == "ok" and result.kubeconfig_path and not dry_run:
            # The kubeconfig is written outside ~/.kube/config. `npa workbench
            # workflow submit --infra k8s/<context>` finds it on its own; kubectl
            # and a bare `sky` need this export.
            typer.echo(
                f"For kubectl / sky in this shell: export KUBECONFIG={result.kubeconfig_path}"
            )
    if not dry_run and result.status not in {"ok", "ready"}:
        # Exiting 0 on a partial run made the follow-up submit the place where the
        # missing cluster surfaced, long after the command that was supposed to
        # create it "succeeded". A read-only plan is different: blocked/unknown
        # is truthful output and still means plan rendering itself succeeded.
        raise typer.Exit(code=1)
