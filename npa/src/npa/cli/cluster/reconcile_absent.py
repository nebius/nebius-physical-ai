"""Explicit, read-only cloud verification and local terminal reconciliation."""

import json
import math
from pathlib import Path
import sqlite3
import subprocess
import tarfile

import yaml

import typer

from npa.cluster.absence_process import DEFAULT_VERIFICATION_TIMEOUT_SECONDS
from npa.cluster.reconcile_absent import reconcile_absent
from npa.lifecycle_intent import OperationIntent, intent_boundary, json_stdout_contract


@json_stdout_contract
@intent_boundary(OperationIntent.MUTATE)
def reconcile_absent_cmd(
    evidence_file: Path = typer.Option(
        ...,
        "--evidence-file",
        help="Private pinned original lifecycle evidence manifest.",
    ),
    verification_timeout_seconds: float = typer.Option(
        DEFAULT_VERIFICATION_TIMEOUT_SECONDS,
        "--verification-timeout-seconds",
        min=0,
        help="Deadline per verification subprocess in seconds; 0 disables the deadline.",
    ),
    output_format: str = typer.Option(
        "json", "--output-format", help="Output format: json."
    ),
) -> None:
    """Reconcile an absent legacy cluster operation; never delete or relaunch."""
    if not math.isfinite(verification_timeout_seconds):
        raise typer.BadParameter(
            "Verification timeout must be finite",
            param_hint="--verification-timeout-seconds",
        )
    if output_format != "json":
        raise typer.BadParameter("Only JSON output is supported")
    try:
        result = reconcile_absent(
            evidence_file, verification_timeout_seconds=verification_timeout_seconds
        )
    except (
        OSError,
        ValueError,
        RuntimeError,
        KeyError,
        TypeError,
        IndexError,
        AttributeError,
        sqlite3.Error,
        tarfile.TarError,
        yaml.YAMLError,
        subprocess.SubprocessError,
    ):
        typer.echo(
            json.dumps({"status": "verification-unavailable", "reconciled": False})
        )
        raise typer.Exit(1) from None
    typer.echo(json.dumps(result))
