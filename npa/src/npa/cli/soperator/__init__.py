"""``npa soperator`` -- deploy Slurm-on-Kubernetes (soperator) clusters.

Wraps the public nebius-solutions-library soperator Terraform recipe from a
compact ``npa.soperator/v0.0.1`` spec that supports multiple worker presets and
a per-pool Docker/Enroot image cache disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from npa.soperator.lifecycle import DEFAULT_GPU_CREATION_CHECK_TIMEOUT_SECONDS
from npa.soperator.spec import DEFAULT_SOLUTIONS_LIBRARY_REF

app = typer.Typer(
    name="soperator",
    help="Deploy and manage Nebius soperator (Slurm-on-Kubernetes) clusters.",
    no_args_is_help=True,
)


def plan_cmd(
    spec_path: Path = typer.Option(
        ...,
        "--spec",
        "-f",
        help="Path to an npa.soperator/v0.0.1 cluster spec YAML.",
    ),
    output: str = typer.Option("text", "--output", help="Output format: text or json."),
) -> None:
    """Show a public-safe, provider-free Soperator capacity plan."""

    from npa.cluster_backends import get_backend
    from npa.soperator.spec import SoperatorSpecError, load_spec

    try:
        result = get_backend("soperator").plan(load_spec(spec_path))
    except (SoperatorSpecError, FileNotFoundError, OSError) as exc:
        raise typer.BadParameter(f"Invalid soperator spec: {exc}") from exc
    if output not in {"text", "json"}:
        raise typer.BadParameter("--output must be text or json")
    if output == "json":
        typer.echo(json.dumps(result, indent=2))
        return
    typer.echo(f"Soperator plan for '{result['name']}' ({result['region']}):")
    typer.echo(
        "  system autoscaling: "
        f"{result['control_plane']['system_min_size']}.."
        f"{result['control_plane']['system_max_size']}"
    )
    for worker in result["workers"]:
        typer.echo(
            f"  worker {worker['name']}: size={worker['size']} "
            f"preset={worker['preset']} capacity={worker['capacity_mode']}"
        )
    if result["reservation_preflight"] == "required":
        typer.echo("  reserved-capacity provider preflight: required before apply")


def deploy_cmd(
    spec_path: Path = typer.Option(
        ...,
        "--spec",
        "-f",
        help="Path to an npa.soperator/v0.0.1 cluster spec YAML.",
    ),
    project: str = typer.Option(
        "",
        "--project",
        help="Config project alias to resolve region/tenant/project from ~/.npa.",
    ),
    terraform_dir: Path | None = typer.Option(
        None,
        "--terraform-dir",
        help="Path to a checked-out solutions-library 'soperator' recipe dir. "
        "If omitted, the library is cloned under ~/.npa/soperator.",
    ),
    solutions_library_ref: str = typer.Option(
        DEFAULT_SOLUTIONS_LIBRARY_REF,
        "--ref",
        help="Immutable 40-character nebius-solutions-library commit SHA.",
    ),
    root_login_ssh_public_key_file: Path | None = typer.Option(
        None,
        "--root-login-ssh-public-key-file",
        help="Public-key file granting root SSH access on the public login node; "
        "overrides the spec, environment, and operator-home discovery.",
    ),
    timeout: int = typer.Option(
        90, "--timeout", help="Terraform apply timeout in minutes."
    ),
    gpu_creation_check_timeout: int = typer.Option(
        DEFAULT_GPU_CREATION_CHECK_TIMEOUT_SECONDS,
        "--gpu-creation-check-timeout",
        min=1,
        help="Independent end-to-end mandatory GPU gate timeout in seconds. Bounds "
        "Slurm queueing, job wall time, and the local kubectl process; --timeout "
        "continues to apply only to Terraform.",
    ),
    apply_fixes: bool = typer.Option(
        True,
        "--apply-fixes/--skip-fixes",
        help="Apply monitoring prerequisites/repair, CRD and scripts compatibility "
        "fixes, Ubuntu userns setup, and best-effort worker recovery. Mandatory "
        "direct CUDA creation checks for GPU pools run with either setting.",
    ),
    source_preflight_only: bool = typer.Option(
        False,
        "--source-preflight-only",
        help="Reconcile/verify the pinned source and planned installation path, "
        "then stop before Terraform initialization or provider mutation.",
    ),
    output: str = typer.Option("text", "--output", help="Output format: text or json."),
) -> None:
    """Deploy or reconcile a pinned-contract Soperator cluster spec."""

    from npa.cluster_backends import get_backend
    from npa.cluster_backends.soperator import SoperatorApplyRequest
    from npa.soperator.lifecycle import (
        SoperatorDeploymentValidationError,
        SoperatorStateCaptureError,
    )
    from npa.soperator.spec import SoperatorSpecError, load_spec

    try:
        spec = load_spec(spec_path)
    except (SoperatorSpecError, FileNotFoundError, OSError) as exc:
        raise typer.BadParameter(f"Invalid soperator spec: {exc}") from exc

    if output not in {"text", "json"}:
        raise typer.BadParameter("--output must be text or json")
    json_mode = output == "json"
    try:
        result = get_backend("soperator").apply(
            spec,
            SoperatorApplyRequest(
                terraform_dir=terraform_dir,
                solutions_library_ref=solutions_library_ref,
                root_login_ssh_public_key_file=root_login_ssh_public_key_file,
                project=project or None,
                timeout_minutes=timeout,
                gpu_creation_check_timeout_seconds=gpu_creation_check_timeout,
                apply_fixes=apply_fixes,
                source_preflight_only=source_preflight_only,
                stream_terraform_output=not json_mode,
                on_status=lambda msg: typer.echo(f"  - {msg}", err=json_mode),
            ),
        )
    except SoperatorDeploymentValidationError as exc:
        if json_mode:
            typer.echo(json.dumps(exc.result, indent=2))
        else:
            result = exc.result
            typer.echo(
                f"Soperator cluster '{result['name']}' was applied, but mandatory "
                "post-apply validation failed.",
                err=True,
            )
            typer.echo(f"  validation: {result['validation']['message']}", err=True)
            typer.echo(f"  kube context: {result['kube_context']}", err=True)
            typer.echo(f"  worker pools: {', '.join(result['worker_pools'])}", err=True)
            typer.echo(f"  install dir: {result['install_dir']}", err=True)
        raise typer.Exit(1) from exc
    except SoperatorStateCaptureError as exc:
        if json_mode:
            typer.echo(json.dumps(exc.result, indent=2))
        else:
            typer.echo(
                f"Soperator cluster '{exc.result['name']}' was applied, but "
                "authoritative ownership state could not be captured.",
                err=True,
            )
            typer.echo(f"  error: {exc.result['error']}", err=True)
            typer.echo(f"  recovery: {exc.result['recovery']}", err=True)
        raise typer.Exit(1) from exc
    except (ValueError, OSError, RuntimeError) as exc:
        raise typer.BadParameter(f"Soperator deploy failed: {exc}") from exc
    if output == "json":
        typer.echo(json.dumps(result, indent=2))
    elif source_preflight_only:
        typer.echo(
            f"Deploy source preflight passed for soperator cluster '{result['name']}'; "
            "no provider mutation was performed."
        )
        for worker in result.get("workers", []):
            verification = (
                " (reservation not provider-verified in source-only mode)"
                if worker["capacity_mode"] == "reserved"
                else ""
            )
            typer.echo(
                f"  worker {worker['name']} capacity: "
                f"{worker['capacity_mode']}{verification}"
            )
        typer.echo(f"  install dir: {result['install_dir']}")
    else:
        typer.echo(
            f"Deployed soperator cluster '{result['name']}' in {result['region']}."
        )
        typer.echo(f"  kube context: {result['kube_context']}")
        typer.echo(f"  worker pools: {', '.join(result['worker_pools'])}")
        if result.get("docker_cache_pools"):
            typer.echo(
                f"  docker-cache pools (IO_M3): {', '.join(result['docker_cache_pools'])}"
            )
        for worker in result.get("workers", []):
            typer.echo(f"  worker {worker['name']} capacity: {worker['capacity_mode']}")
        typer.echo(f"  install dir: {result['install_dir']}")


def destroy_cmd(
    name: str = typer.Option(
        ..., "--name", help="Cluster name (company_name in the spec)."
    ),
    terraform_dir: Path | None = typer.Option(
        None,
        "--terraform-dir",
        help="solutions-library 'soperator' recipe dir (if not the default).",
    ),
    solutions_library_ref: str = typer.Option(
        DEFAULT_SOLUTIONS_LIBRARY_REF,
        "--ref",
        help="Immutable 40-character nebius-solutions-library commit SHA.",
    ),
    project: str = typer.Option(
        "",
        "--project",
        help="Config project alias to resolve region/tenant/project from ~/.npa "
        "(only used for installs predating the env sidecar).",
    ),
    timeout: int = typer.Option(
        90, "--timeout", help="Terraform destroy timeout in minutes."
    ),
    source_preflight_only: bool = typer.Option(
        False,
        "--source-preflight-only",
        help="Reconcile/verify the pinned source and installation path, then stop "
        "before Terraform initialization or any provider deletion.",
    ),
    force: bool = typer.Option(False, "--force", help="Skip confirmation."),
) -> None:
    """Destroy an npa-managed soperator cluster by name."""

    from npa.cluster_backends import get_backend
    from npa.cluster_backends.soperator import SoperatorDestroyRequest
    from npa.soperator.spec import SoperatorSpec

    if (
        not source_preflight_only
        and not force
        and not typer.confirm(f"Destroy soperator cluster '{name}'?")
    ):
        raise typer.Exit(1)
    try:
        result = get_backend("soperator").destroy(
            SoperatorSpec(name=name),
            SoperatorDestroyRequest(
                terraform_dir=terraform_dir,
                solutions_library_ref=solutions_library_ref,
                project=project or None,
                timeout_minutes=timeout,
                source_preflight_only=source_preflight_only,
                on_status=lambda msg: typer.echo(f"  - {msg}"),
            ),
        )
    except (ValueError, OSError, RuntimeError) as exc:
        raise typer.BadParameter(f"Soperator destroy failed: {exc}") from exc
    if source_preflight_only:
        assert result is not None
        typer.echo(
            f"Destroy source preflight passed for soperator cluster '{name}'; "
            "no provider mutation was performed."
        )
        return
    typer.echo(f"Destroyed soperator cluster '{name}'.")


def status_cmd(
    name: str = typer.Option(..., "--name", help="Cluster name."),
    terraform_dir: Path | None = typer.Option(
        None,
        "--terraform-dir",
        help="Optional authoritative solutions-library soperator recipe directory.",
    ),
    output: str = typer.Option("text", "--output", help="Output format: text or json."),
) -> None:
    """Show a soperator cluster's Slurm partitions/nodes via kubectl."""

    from npa.cluster_backends import get_backend
    from npa.cluster_backends.soperator import SoperatorStatusRequest
    from npa.soperator.spec import SoperatorSpec

    try:
        backend_status = get_backend("soperator").status(
            SoperatorSpec(name=name),
            SoperatorStatusRequest(terraform_dir=terraform_dir),
        )
    except (ValueError, OSError, RuntimeError) as exc:
        raise typer.BadParameter(f"Soperator status failed: {exc}") from exc
    workers = backend_status["workers"]
    if output not in {"text", "json"}:
        raise typer.BadParameter("--output must be text or json")
    if output == "json":
        typer.echo(
            json.dumps(
                {
                    "name": name,
                    "context": backend_status["context"],
                    "sinfo": backend_status["sinfo"],
                    "workers": workers,
                    "capacity_status": "applied" if workers else "unknown",
                }
            )
        )
    else:
        typer.echo(backend_status["sinfo"])
        if workers:
            for worker in workers:
                typer.echo(
                    f"worker {worker['name']} capacity: {worker['capacity_mode']} "
                    f"({worker['nodes']} node(s))"
                )
        else:
            typer.echo("worker capacity: unknown (local Terraform state not found)")


app.command("plan")(plan_cmd)
app.command("deploy")(deploy_cmd)
app.command("destroy")(destroy_cmd)
app.command("status")(status_cmd)
