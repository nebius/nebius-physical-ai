"""npa workbench — parent group for all workbench tools."""

from __future__ import annotations

import typer

from npa.clients.credentials import load_credentials
from npa.cli.workbench.byof import app as byof_app
from npa.cli.workbench.cosmos2 import app as cosmos2_app
from npa.cli.workbench.cosmos3 import app as cosmos3_app
from npa.cli.workbench.data import app as data_app
from npa.cli.workbench.lerobot import app as lerobot_app
from npa.cli.workbench.mjlab import app as mjlab_app
from npa.cli.cosmos import app as cosmos_app
from npa.cli.fiftyone import app as fiftyone_app
from npa.cli.genesis import app as genesis_app
from npa.cli.groot import app as groot_app
from npa.cli.isaac_lab import app as isaac_lab_app
from npa.cli.workbench.sonic import app as sonic_app
from npa.cli.workbench.lancedb import app as lancedb_app
from npa.cli.workbench.detection_training import app as detection_training_app
from npa.cli.workbench.golden_eval import app as golden_eval_app
from npa.cli.workbench.token_factory import app as token_factory_app
from npa.cli.workbench.vlm_eval import app as vlm_eval_app
from npa.cli.workbench.workflow import app as workflow_app
from npa.cli.workbench.health import app as health_app
from npa.cli.workbench.sim2real import app as sim2real_app

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
app.add_typer(cosmos2_app, name="cosmos2")
app.add_typer(cosmos3_app, name="cosmos3")
app.add_typer(fiftyone_app, name="fiftyone")
app.add_typer(genesis_app, name="genesis")
app.add_typer(groot_app, name="groot")
app.add_typer(isaac_lab_app, name="isaac-lab")
app.add_typer(sonic_app, name="sonic")
app.add_typer(mjlab_app, name="mjlab")
app.add_typer(lancedb_app, name="lancedb")
app.add_typer(detection_training_app, name="detection-training")
app.add_typer(vlm_eval_app, name="vlm-eval")
app.add_typer(token_factory_app, name="token-factory")
app.add_typer(byof_app, name="byof")
app.add_typer(workflow_app, name="workflow")
app.add_typer(health_app, name="health")
app.add_typer(sim2real_app, name="sim2real", hidden=True)
app.add_typer(golden_eval_app, name="golden-eval")
# Backward-compatible S3 bridge; not advertised in workbench --help.
app.add_typer(data_app, name="data", hidden=True)
