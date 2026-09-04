"""Typer entry point for `npa workbench robocasa`."""

from __future__ import annotations

import typer

from npa.cli.workbench.robocasa import deploy, list as list_mod, run, status, system_info

app = typer.Typer(
    name="robocasa",
    help="RoboCasa kitchen-task simulation workbench.",
    no_args_is_help=True,
)


app.command("deploy")(deploy.deploy_cmd)
app.command("run")(run.run_cmd)
app.command("status")(status.status_cmd)
app.command("system-info")(system_info.system_info_cmd)
app.command("list")(list_mod.list_cmd)
