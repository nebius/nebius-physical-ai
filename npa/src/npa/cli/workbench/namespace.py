"""Expose native team namespace setup and private Kubernetes contexts."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from npa.lifecycle_intent import json_stdout_contract
from npa.workbench.namespaces import apply_namespace, write_namespace_context


app = typer.Typer(
    help="Manage team namespaces and researcher access.", no_args_is_help=True
)

_ADMIN_CONTEXT = typer.Option(..., "--context", help="Exact administrator context.")
_ADMIN_KUBECONFIG = typer.Option("", "--kubeconfig", help="Administrator kubeconfig.")
_USERS = typer.Option([], "--user", help="Complete member list; repeat per username.")
_JSON_FORMAT = typer.Option("json", "--output-format", help="Output format (json).")


@app.command("apply")
@json_stdout_contract
def apply_cmd(
    name: str = typer.Argument(..., help="Dedicated Kubernetes namespace."),
    context: str = _ADMIN_CONTEXT,
    kubeconfig: str = _ADMIN_KUBECONFIG,
    user: list[str] = _USERS,
    group: list[str] = typer.Option(
        [], "--group", help="Complete group list; repeat for each group."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print manifests without contacting Kubernetes."
    ),
    output_format: str = _JSON_FORMAT,
) -> None:
    """Apply team access; omitted members lose this command's previous grants.

    Args:
        name, context, kubeconfig: Namespace and explicit administrator target.
        user, group: Complete researcher identity lists, not additions.
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
            name,
            context=context,
            kubeconfig=kubeconfig,
            users=tuple(user),
            groups=tuple(group),
            dry_run=dry_run,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(json.dumps(result))


@app.command("context")
@json_stdout_contract
def context_cmd(
    name: str = typer.Argument(..., help="Existing team namespace."),
    context: str = typer.Option(
        ..., "--context", help="Exact source context with your own credentials."
    ),
    output_dir: Path = typer.Option(
        ..., "--output-dir", help="New private directory for namespace configuration."
    ),
    kubeconfig: str = typer.Option("", "--kubeconfig", help="Your source kubeconfig."),
    output_format: str = typer.Option(
        "json", "--output-format", help="Output format (json)."
    ),
) -> None:
    """Prepare private client configuration without changing the source context.

    Args:
        name, context, kubeconfig: Namespace and caller's authenticated context.
        output_dir: New private destination.
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
            name, context=context, output_dir=output_dir, kubeconfig=kubeconfig
        )
    except (ValueError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(json.dumps(result))
