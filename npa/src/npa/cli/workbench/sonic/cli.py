"""Typer entry point for `npa workbench sonic`."""

from __future__ import annotations

import typer

from npa.cli.workbench.sonic import (
    deploy,
    export as export_mod,
    list as list_mod,
    serve,
    status,
    train,
)
from npa.cli.workbench.sonic.helpers import set_context

app = typer.Typer(
    name="sonic",
    help="NVIDIA GEAR-SONIC whole-body-control workbench.",
    no_args_is_help=True,
)


@app.callback()
def main(
    project: str = typer.Option("", "--project", "-p", help="Project alias from ~/.npa/config.yaml."),
    name: str = typer.Option("", "--name", "-n", help="Workbench instance name within the project."),
) -> None:
    """NVIDIA GEAR-SONIC whole-body-control workbench."""
    set_context(project, name)


app.command("deploy")(deploy.deploy_cmd)
app.command("train")(train.train_cmd)
app.command("export")(export_mod.export_cmd)
app.command("serve")(serve.serve_cmd)
app.command("status")(status.status_cmd)
app.command("list")(list_mod.list_cmd)
