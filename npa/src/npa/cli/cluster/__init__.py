"""Cluster lifecycle CLI."""

from __future__ import annotations

import typer

from npa.cli.cluster.deploy import deploy_cmd

app = typer.Typer(
    name="cluster",
    help="Manage Nebius Managed Kubernetes clusters for NPA workflows.",
    no_args_is_help=True,
)

app.command("deploy")(deploy_cmd)
