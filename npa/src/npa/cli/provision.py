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
    project: str = typer.Option("", "--project", help="Project alias from ~/.npa/config.yaml."),
    cluster_name: str = typer.Option("npa-cluster", "--cluster-name", help="Cluster profile/context name."),
    terraform_dir: Path | None = typer.Option(None, "--terraform-dir", help="Terraform cluster directory."),
    kubeconfig: Path | None = typer.Option(None, "--kubeconfig", help="Dedicated kubeconfig path."),
    context_name: str = typer.Option("", "--context", help="Kubeconfig context name."),
    skip_k8s: bool = typer.Option(False, "--skip-k8s", help="Do not ensure Kubernetes."),
    skip_s3: bool = typer.Option(False, "--skip-s3", help="Do not ensure S3."),
    validate: bool = typer.Option(True, "--validate/--skip-validate", help="Run post-apply Kubernetes validation."),
    sky_smoke: bool = typer.Option(False, "--sky-smoke/--skip-sky-smoke", help="Run a SkyPilot GPU smoke task."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Resolve settings and print intended actions only."),
    timeout: int = typer.Option(120, "--timeout", help="Terraform apply timeout in minutes."),
    output_format: OutputFormat = typer.Option(OutputFormat.text, "--output-format", help="Output format."),
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
    if result.status != "ok":
        # Exiting 0 on a partial run made the follow-up submit the place where the
        # missing cluster surfaced, long after the command that was supposed to
        # create it "succeeded".
        raise typer.Exit(code=1)
