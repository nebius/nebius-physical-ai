"""``npa soperator`` -- deploy Slurm-on-Kubernetes (soperator) clusters.

Wraps the public nebius-solutions-library soperator Terraform recipe from a
compact ``npa.soperator/v0.0.1`` spec that supports multiple worker presets and
a per-pool Docker/Enroot image cache disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

app = typer.Typer(
    name="soperator",
    help="Deploy and manage Nebius soperator (Slurm-on-Kubernetes) clusters.",
    no_args_is_help=True,
)


def deploy_cmd(
    spec_path: Path = typer.Option(
        ...,
        "--spec",
        "-f",
        help="Path to an npa.soperator/v0.0.1 cluster spec YAML.",
    ),
    project: str = typer.Option(
        "", "--project", help="Config project alias to resolve region/tenant/project from ~/.npa."
    ),
    terraform_dir: Path | None = typer.Option(
        None,
        "--terraform-dir",
        help="Path to a checked-out solutions-library 'soperator' recipe dir. "
        "If omitted, the library is cloned under ~/.npa/soperator.",
    ),
    solutions_library_ref: str = typer.Option(
        "main", "--ref", help="Git ref of nebius-solutions-library to clone when needed."
    ),
    timeout: int = typer.Option(90, "--timeout", help="Terraform apply timeout in minutes."),
    apply_fixes: bool = typer.Option(
        True,
        "--apply-fixes/--skip-fixes",
        help="Apply the post-deploy fixes (monitoring CRDs, CRD patch, scripts configmap) "
        "the 4.1.0-stable recipe needs to reach a working Slurm.",
    ),
    output: str = typer.Option("text", "--output", help="Output format: text or json."),
) -> None:
    """Deploy a soperator cluster from a spec (multiple presets + optional docker cache)."""

    from npa.soperator.lifecycle import deploy_cluster
    from npa.soperator.spec import SoperatorSpecError, load_spec

    try:
        spec = load_spec(spec_path)
    except (SoperatorSpecError, FileNotFoundError, OSError) as exc:
        raise typer.BadParameter(f"Invalid soperator spec: {exc}") from exc

    result = deploy_cluster(
        spec,
        terraform_dir=terraform_dir,
        solutions_library_ref=solutions_library_ref,
        project=project or None,
        timeout_minutes=timeout,
        apply_fixes=apply_fixes,
        on_status=lambda msg: typer.echo(f"  - {msg}"),
    )
    if output == "json":
        typer.echo(json.dumps(result, indent=2))
    else:
        typer.echo(f"Deployed soperator cluster '{result['name']}' in {result['region']}.")
        typer.echo(f"  kube context: {result['kube_context']}")
        typer.echo(f"  worker pools: {', '.join(result['worker_pools'])}")
        if result.get("docker_cache_pools"):
            typer.echo(
                f"  docker-cache pools (IO_M3): {', '.join(result['docker_cache_pools'])}"
            )
        typer.echo(f"  install dir: {result['install_dir']}")


def destroy_cmd(
    name: str = typer.Option(..., "--name", help="Cluster name (company_name in the spec)."),
    terraform_dir: Path | None = typer.Option(
        None, "--terraform-dir", help="solutions-library 'soperator' recipe dir (if not the default)."
    ),
    solutions_library_ref: str = typer.Option("main", "--ref"),
    project: str = typer.Option(
        "",
        "--project",
        help="Config project alias to resolve region/tenant/project from ~/.npa "
        "(only used for installs predating the env sidecar).",
    ),
    timeout: int = typer.Option(90, "--timeout", help="Terraform destroy timeout in minutes."),
    force: bool = typer.Option(False, "--force", help="Skip confirmation."),
) -> None:
    """Destroy an npa-managed soperator cluster by name."""

    from npa.soperator.lifecycle import destroy_cluster

    if not force and not typer.confirm(f"Destroy soperator cluster '{name}'?"):
        raise typer.Exit(1)
    destroy_cluster(
        name,
        terraform_dir=terraform_dir,
        solutions_library_ref=solutions_library_ref,
        project=project or None,
        timeout_minutes=timeout,
        on_status=lambda msg: typer.echo(f"  - {msg}"),
    )
    typer.echo(f"Destroyed soperator cluster '{name}'.")


def status_cmd(
    name: str = typer.Option(..., "--name", help="Cluster name."),
    output: str = typer.Option("text", "--output", help="Output format: text or json."),
) -> None:
    """Show a soperator cluster's Slurm partitions/nodes via kubectl."""

    import os
    import shutil

    context = f"nebius-{name}-slurm"
    kubectl = os.environ.get("NPA_KUBECTL_BIN") or "kubectl"
    if not shutil.which(kubectl):
        raise typer.BadParameter(f"kubectl not found: {kubectl}")
    import subprocess

    proc = subprocess.run(
        [kubectl, "--context", context, "exec", "-n", "soperator", "controller-0",
         "-c", "slurmctld", "--", "sinfo"],
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip()
        raise typer.BadParameter(f"Could not query Slurm on '{name}': {detail}")
    if output == "json":
        typer.echo(json.dumps({"name": name, "context": context, "sinfo": proc.stdout}))
    else:
        typer.echo(proc.stdout)


app.command("deploy")(deploy_cmd)
app.command("destroy")(destroy_cmd)
app.command("status")(status_cmd)
