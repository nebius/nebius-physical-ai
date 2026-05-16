"""npa workbench — parent group for all workbench tools."""

from __future__ import annotations

import typer

from npa.clients.credentials import load_credentials
from npa.cli.workbench.lerobot import app as lerobot_app
from npa.cli.cosmos import app as cosmos_app
from npa.cli.fiftyone import app as fiftyone_app
from npa.cli.genesis import app as genesis_app
from npa.cli.groot import app as groot_app
from npa.cli.isaac_lab import app as isaac_lab_app
from npa.cli.workbench.sonic import app as sonic_app
from npa.cli.workbench.lancedb import app as lancedb_app
from npa.cli.workbench.detection_training import app as detection_training_app

app = typer.Typer(
    name="workbench",
    help="Physical AI workbench tools.",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """Physical AI workbench tools."""
    load_credentials(
        warn=lambda msg: typer.echo(msg, err=True),
        export_to_environment=True,
    )


app.add_typer(lerobot_app, name="lerobot")
app.add_typer(cosmos_app, name="cosmos")
app.add_typer(fiftyone_app, name="fiftyone")
app.add_typer(genesis_app, name="genesis")
app.add_typer(groot_app, name="groot")
app.add_typer(isaac_lab_app, name="isaac-lab")
app.add_typer(sonic_app, name="sonic")
app.add_typer(lancedb_app, name="lancedb")
app.add_typer(detection_training_app, name="detection-training")
