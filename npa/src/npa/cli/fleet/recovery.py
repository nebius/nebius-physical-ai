"""Expose guarded recovery for one interrupted Fleet apply."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import typer
import yaml

from npa.cli._typer_defaults import resolve_typer_defaults
from npa.lifecycle_intent import OperationIntent, intent_boundary, json_stdout_contract

_SPEC = typer.Option(..., "--spec", "-f", help="Path to the owner-private Fleet spec.")
_PROJECTS = typer.Option(
    "", "--only-projects", help="Comma-separated project keys or display names."
)
_CLUSTERS = typer.Option(
    "",
    "--only-clusters",
    help="Comma-separated cluster names within selected projects.",
)
_PREFIX = typer.Option(
    "", "--project-prefix", help="Override the spec's project display-name prefix."
)
_PROFILE = typer.Option(
    "", "--profile", help="Override the spec's Nebius authentication profile."
)
_YES = typer.Option(False, "--yes", "-y", help="Skip the recovery confirmation.")
_OUTPUT = typer.Option(
    "text", "--output", "--output-format", help="Output format: text or json."
)


@resolve_typer_defaults
@json_stdout_contract
@intent_boundary(OperationIntent.MUTATE)
def recover_fleet_cmd(
    spec_path: Path = _SPEC,
    only_projects: str = _PROJECTS,
    only_clusters: str = _CLUSTERS,
    project_prefix: str = _PREFIX,
    profile: str = _PROFILE,
    yes: bool = _YES,
    output_format: Literal["text", "json"] = _OUTPUT,
) -> None:
    """Adopt one verified terminal MK8s apply without Terraform mutation.

    Args:
        spec_path: Existing Fleet declaration.
        only_projects: Project keys or display names selecting one target.
        only_clusters: Cluster names selecting one target.
        project_prefix: Optional project display-name prefix override.
        profile: Optional Nebius authentication profile override.
        yes: Skip the explicit recovery confirmation.
        output_format: Sanitized text or one JSON document.
    Raises:
        typer.Exit: Identity, provider, or registration proof failed.
    """
    if not _confirm(yes, output_format):
        return
    options = {
        "only_projects": _selectors(only_projects),
        "only_clusters": _selectors(only_clusters),
        "project_prefix": project_prefix or None,
        "profile": profile or None,
    }
    _recover_and_emit(spec_path, output_format, options)


def _confirm(yes: bool, output_format: str) -> bool:
    if yes:
        return True
    error_stream = output_format == "json"
    typer.echo(
        "Recovery verifies and registers one existing target; it never runs Terraform.",
        err=error_stream,
    )
    if typer.confirm("Proceed to recover one Fleet target?", err=error_stream):
        return True
    typer.echo("Aborted.", err=error_stream)
    return False


def _selectors(value: str) -> list[str] | None:
    selected = [name.strip() for name in value.split(",") if name.strip()]
    return selected or None


def _recover_and_emit(spec_path: Path, output_format: str, options: dict) -> None:
    from npa.fleet.recovery import FleetRecoveryError, recover_interrupted_fleet_apply
    from npa.fleet.spec import FleetSpecError, load_spec

    try:
        result = recover_interrupted_fleet_apply(load_spec(spec_path), **options)
    except (
        FleetRecoveryError,
        FleetSpecError,
        OSError,
        ValueError,
        yaml.YAMLError,
    ) as exc:
        _emit({"status": "failed", "failure": "recovery_unavailable"}, output_format)
        if output_format != "json":
            typer.echo(
                "Fleet recovery failed; exact provider and local receipts remain owner-private.",
                err=True,
            )
        raise typer.Exit(1) from exc
    _emit(result, output_format)


def _emit(result: dict, output_format: str) -> None:
    if output_format == "json":
        typer.echo(json.dumps(result, indent=2, sort_keys=True))
        return
    typer.echo(
        "Fleet target recovered pending validation: "
        f"{result.get('project_key', '')}/{result.get('cluster_name', '')}."
    )
