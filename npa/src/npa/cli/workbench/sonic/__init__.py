"""npa workbench sonic - NVIDIA GEAR-SONIC whole-body-control tool."""

from __future__ import annotations

import typer

app = typer.Typer(
    name="sonic",
    help="NVIDIA GEAR-SONIC whole-body-control workbench.",
    no_args_is_help=True,
)


@app.callback()
def main(
    project: str = typer.Option("", "--project", "-p", help="NPA project alias."),
    name: str = typer.Option("", "--name", "-n", help="Workbench name."),
) -> None:
    """NVIDIA GEAR-SONIC whole-body-control workbench."""
    _ = (project, name)
