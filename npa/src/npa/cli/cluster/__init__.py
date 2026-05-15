"""NPA Workbench cluster target/profile CLI."""

from __future__ import annotations

import typer

from npa.cli.cluster.deploy import deploy_cmd
from npa.cli.cluster.destroy import destroy_cmd
from npa.cli.cluster.scope import CLUSTER_SCOPE_EPILOG
from npa.cli.cluster.node_group import app as node_group_app
from npa.cli.cluster.status import list_cmd, status_cmd

app = typer.Typer(
    name="cluster",
    help="Manage NPA Workbench cluster targets and profiles.",
    epilog=CLUSTER_SCOPE_EPILOG,
    no_args_is_help=True,
)

app.command("deploy")(deploy_cmd)
app.command("destroy")(destroy_cmd)
app.command("status")(status_cmd)
app.command("list")(list_cmd)
app.add_typer(node_group_app, name="node-group")
