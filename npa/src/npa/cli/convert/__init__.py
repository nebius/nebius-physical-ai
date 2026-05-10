"""npa convert - standalone artifact conversion commands."""

from __future__ import annotations

import typer

from npa.cli.convert.lerobot_to_rrd import lerobot_to_rrd_cmd

app = typer.Typer(
    name="convert",
    help="Convert datasets and prediction artifacts between standalone formats.",
    no_args_is_help=True,
)

app.command("lerobot-to-rrd", help="Convert a LeRobotDataset to a Rerun .rrd recording.")(
    lerobot_to_rrd_cmd
)
