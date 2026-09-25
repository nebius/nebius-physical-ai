"""Expose native namespace creation and private Kubernetes context selection."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from npa.lifecycle_intent import OperationIntent, intent_boundary, json_stdout_contract
from npa.workbench.namespaces import apply_namespace, write_namespace_context


app = typer.Typer(help="Create or select Kubernetes namespaces.", no_args_is_help=True)


@app.command("apply")
@intent_boundary(OperationIntent.ENSURE_PRESENT)
@json_stdout_contract
def apply_cmd(
    name: str = typer.Argument(..., help="Kubernetes namespace."),
    context: str = typer.Option(..., "--context", help="Exact authenticated context."),
    kubeconfig: str = typer.Option("", "--kubeconfig", help="Source kubeconfig."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print the manifest without contacting Kubernetes."
    ),
    output_format: str = typer.Option(
        "json", "--output-format", help="Output format (json)."
    ),
) -> None:
    """Create a missing namespace; reuse existing namespaces without changing access.

    Args:
        name, context, kubeconfig: Namespace and explicit authenticated target.
        dry_run: Render only.
        output_format: JSON output.
    Returns:
        None.
    Raises:
        typer.BadParameter: Input or cluster access is invalid.
    """
    if output_format != "json":
        raise typer.BadParameter("--output-format must be json")
    try:
        result = apply_namespace(
            name, context=context, kubeconfig=kubeconfig, dry_run=dry_run
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(json.dumps(result))


@app.command("context")
@json_stdout_contract
def context_cmd(
    name: str = typer.Argument(..., help="Existing namespace; default is supported."),
    context: str = typer.Option(..., "--context", help="Exact source context."),
    output_dir: Path = typer.Option(..., "--output-dir", help="New private directory."),
    kubeconfig: str = typer.Option("", "--kubeconfig", help="Your source kubeconfig."),
    sky_config: Path | None = typer.Option(
        None, "--sky-config", help="Existing SkyPilot config to preserve."
    ),
    output_format: str = typer.Option(
        "json", "--output-format", help="Output format (json)."
    ),
) -> None:
    """Select a namespace privately, preserving existing credentials and access.

    Args:
        name, context, kubeconfig: Namespace and caller's authenticated context.
        output_dir: New private destination.
        sky_config: Existing SkyPilot settings, or normal environment/default path.
        output_format: JSON output.
    Returns:
        None.
    Raises:
        typer.BadParameter: Configuration cannot be prepared.
    """
    if output_format != "json":
        raise typer.BadParameter("--output-format must be json")
    try:
        result = write_namespace_context(
            name,
            context=context,
            output_dir=output_dir,
            kubeconfig=kubeconfig,
            sky_config=sky_config,
        )
    except (ValueError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(json.dumps(result))
