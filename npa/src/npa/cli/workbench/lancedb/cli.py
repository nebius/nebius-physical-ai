"""Typer app for `npa workbench lancedb`."""

from __future__ import annotations

import typer

from .create_table import create_table_cmd
from .deploy import deploy_cmd
from .import_bdd100k import import_bdd100k_cmd
from .import_lerobot import import_lerobot_cmd
from .list import list_cmd
from .query import query_cmd
from .status import status_cmd

app = typer.Typer(
    name="lancedb",
    help="Deploy and query LanceDB vector-search workbenches.",
    no_args_is_help=True,
)

app.command("deploy")(deploy_cmd)
app.command("status")(status_cmd)
app.command("list")(list_cmd)
app.command("create-table")(create_table_cmd)
app.command("query")(query_cmd)
app.command("import-lerobot")(import_lerobot_cmd)
app.command("import-bdd100k")(import_bdd100k_cmd)
