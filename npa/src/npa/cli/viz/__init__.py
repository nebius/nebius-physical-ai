"""npa viz — visualization primitives."""

from __future__ import annotations

import typer

from npa.cli.viz.lerobot import lerobot_cmd

app = typer.Typer(
    name="viz",
    help="Render Physical AI dataset and prediction visualizations.",
    no_args_is_help=True,
)

app.command(
    "lerobot",
    help="[DEPRECATED] Use `npa convert lerobot-to-mp4` instead.",
)(lerobot_cmd)
