"""npa CLI entry point."""

from __future__ import annotations

import typer

from npa.cli.workbench import app as workbench_app
from npa.cli.adapter import app as adapter_app
from npa.cli.workflow import app as workflow_app

app = typer.Typer(
    name="npa",
    help="Nebius Physical AI workbench CLI.",
    no_args_is_help=True,
)
app.add_typer(workbench_app, name="workbench")
app.add_typer(adapter_app, name="adapter")
app.add_typer(workflow_app, name="workflow")


def app_entry() -> None:
    app()
