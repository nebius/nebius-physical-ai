"""npa CLI entry point."""

from __future__ import annotations

import typer

from npa.cli.workbench import app as workbench_app
from npa.cli.adapter import app as adapter_app
from npa.cli.workflow import app as workflow_app

app = typer.Typer(
    name="npa",
    help="Nebius Physical AI workbench CLI.",
    no_args_is_help=True,
)
app.add_typer(workbench_app, name="workbench")
app.add_typer(adapter_app, name="adapter")
app.add_typer(workflow_app, name="workflow")


_SETUP_GUIDANCE = """Credential setup

Create ~/.npa/credentials.yaml for user-level tokens and BYOVM SSH defaults:

tokens:
  HF_TOKEN: hf_REPLACE_ME
ngc:
  api_key: nvapi_REPLACE_ME
  # org: optional-ngc-org
  # team: optional-ngc-team
ssh:
  host: 203.0.113.10
  user: ubuntu
  key_path: ~/.ssh/id_ed25519

Then secure it:

chmod 600 ~/.npa/credentials.yaml

Deploy commands create and update ~/.npa/config.yaml for projects, workbenches,
SSH targets, container registry settings, and Terraform state.
"""


@app.command("configure", help="Show credential and config setup guidance.")
def configure() -> None:
    """Show credential and config setup guidance."""
    typer.echo(_SETUP_GUIDANCE)


@app.command("init", help="Show credential and config setup guidance.")
def init() -> None:
    """Show credential and config setup guidance."""
    configure()


def app_entry() -> None:
    app()
