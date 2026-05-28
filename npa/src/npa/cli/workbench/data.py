"""npa workbench data - S3 data bridge commands."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from enum import Enum
from typing import Any

import typer
from rich.console import Console

from npa.clients.project_credentials import s3_client_for_project
from npa.errors import ScopedCredentialError
from npa.workbench.data import (
    DataBridgeError,
    list_s3_objects,
    status_s3_prefix,
    sync_s3_prefix,
)

app = typer.Typer(
    name="data",
    help="S3 data import bridge for Workbench pipelines.",
    no_args_is_help=True,
)
console = Console(stderr=True)

_project_alias = ""


class OutputFormat(str, Enum):
    text = "text"
    json = "json"


@app.callback()
def main(
    project: str = typer.Option(
        "",
        "--project",
        "-p",
        help="Default project alias for S3 credentials.",
    ),
) -> None:
    """S3 data import bridge for Workbench pipelines."""
    global _project_alias
    _project_alias = project


@app.command("sync")
def sync_cmd(
    input_path: str = typer.Option(
        ...,
        "--input-path",
        "--source-uri",
        "--source",
        help="Source S3 URI or prefix.",
    ),
    output_path: str = typer.Option(
        ...,
        "--output-path",
        "--destination-uri",
        "--destination",
        help="Destination S3 URI or prefix.",
    ),
    source_project: str = typer.Option(
        "",
        "--source-project",
        help="Project alias for source S3 credentials. Defaults to --project.",
    ),
    target_project: str = typer.Option(
        "",
        "--target-project",
        help="Project alias for target S3 credentials. Defaults to --project.",
    ),
    allow_host_creds: bool = typer.Option(
        False,
        "--allow-host-creds",
        help="Allow host S3 credentials when scoped project credentials are absent.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Plan the sync without writing destination objects.",
    ),
    limit: int = typer.Option(0, "--limit", help="Maximum objects to process; 0 means all."),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Copy S3 objects between pipeline prefixes."""
    effective_dry_run = dry_run or _env_dry_run()
    src_project = source_project or _project_alias or None
    dst_project = target_project or _project_alias or None
    try:
        source_s3 = s3_client_for_project(src_project, allow_host_creds=allow_host_creds)
        target_s3 = s3_client_for_project(dst_project, allow_host_creds=allow_host_creds)
        result = sync_s3_prefix(
            input_path,
            output_path,
            source_s3_client=source_s3,
            target_s3_client=target_s3,
            dry_run=effective_dry_run,
            limit=limit,
        )
    except (DataBridgeError, ScopedCredentialError) as exc:
        _fail(str(exc))
        return
    _emit(asdict(result), output)


@app.command("status")
def status_cmd(
    input_path: str = typer.Option(
        ...,
        "--input-path",
        "--uri",
        help="S3 URI or prefix to inspect.",
    ),
    project: str = typer.Option(
        "",
        "--source-project",
        help="Project alias for S3 credentials. Defaults to --project.",
    ),
    allow_host_creds: bool = typer.Option(
        False,
        "--allow-host-creds",
        help="Allow host S3 credentials when scoped project credentials are absent.",
    ),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Show object count and bytes for an S3 prefix."""
    try:
        s3 = s3_client_for_project(project or _project_alias or None, allow_host_creds=allow_host_creds)
        payload = status_s3_prefix(input_path, s3_client=s3)
    except (DataBridgeError, ScopedCredentialError) as exc:
        _fail(str(exc))
        return
    _emit(payload, output)


@app.command("list")
def list_cmd(
    input_path: str = typer.Option(
        ...,
        "--input-path",
        "--uri",
        help="S3 URI or prefix to list.",
    ),
    project: str = typer.Option(
        "",
        "--source-project",
        help="Project alias for S3 credentials. Defaults to --project.",
    ),
    allow_host_creds: bool = typer.Option(
        False,
        "--allow-host-creds",
        help="Allow host S3 credentials when scoped project credentials are absent.",
    ),
    limit: int = typer.Option(100, "--limit", help="Maximum objects to print; 0 means all."),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """List S3 objects under a Workbench data prefix."""
    try:
        s3 = s3_client_for_project(project or _project_alias or None, allow_host_creds=allow_host_creds)
        objects = list_s3_objects(input_path, s3_client=s3, limit=limit)
    except (DataBridgeError, ScopedCredentialError) as exc:
        _fail(str(exc))
        return
    _emit({"uri": input_path, "objects": [asdict(item) for item in objects]}, output)


def _env_dry_run() -> bool:
    return os.environ.get("NPA_DRY_RUN", "").lower() in {"1", "true", "yes"} or os.environ.get(
        "DRY_RUN", ""
    ).lower() in {"1", "true", "yes"}


def _emit(payload: dict[str, Any], output: OutputFormat) -> None:
    if output == OutputFormat.json:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    for key, value in payload.items():
        if key in {"copied", "objects", "sample"}:
            typer.echo(f"  {key}: {len(value)}")
        else:
            typer.echo(f"  {key}: {value}")


def _fail(message: str) -> None:
    console.print(f"[red]Error:[/red] {message}")
    raise typer.Exit(1)
