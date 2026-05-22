"""npa CLI entry point."""

from __future__ import annotations

import os
import sys
import traceback
from importlib.metadata import version as package_version

import typer

from npa.cli._error_formatting import format_error_for_user
from npa.cli.workbench import app as workbench_app
from npa.cli.adapter import app as adapter_app
from npa.cli.cluster import app as cluster_app
from npa.cli.convert import app as convert_app
from npa.cli.demo import app as demo_app
from npa.cli.network import app as network_app
from npa.cli.rerun import app as rerun_app
from npa.cli.skypilot import app as skypilot_app
from npa.cli.viz import app as viz_app
from npa.cli.workflow import app as workflow_app
from npa.clients.serverless import ServerlessClientError

app = typer.Typer(
    name="npa",
    help="Nebius Physical AI workbench CLI.",
    no_args_is_help=True,
)
app.add_typer(workbench_app, name="workbench")
app.add_typer(adapter_app, name="adapter")
app.add_typer(cluster_app, name="cluster")
app.add_typer(convert_app, name="convert")
app.add_typer(demo_app, name="demo")
app.add_typer(network_app, name="network")
app.add_typer(rerun_app, name="rerun")
app.add_typer(skypilot_app, name="skypilot")
app.add_typer(viz_app, name="viz")
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


def _version_callback(value: bool) -> None:
    if not value:
        return
    typer.echo(f"npa {package_version('npa')}")
    raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Show the installed npa version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """Nebius Physical AI workbench CLI."""


@app.command("configure", help="Show credential and config setup guidance.")
def configure() -> None:
    """Show credential and config setup guidance."""
    typer.echo(_SETUP_GUIDANCE)


@app.command("init", help="Show credential and config setup guidance.")
def init() -> None:
    """Show credential and config setup guidance."""
    configure()


def app_entry() -> None:
    try:
        app()
    except KeyboardInterrupt:
        sys.exit(130)
    except ServerlessClientError as exc:
        print(format_error_for_user(exc, output_format=_detect_error_format()), file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(format_error_for_user(exc, output_format=_detect_error_format()), file=sys.stderr)
        if os.environ.get("NPA_DEBUG"):
            traceback.print_exc()
        else:
            print("  Run with NPA_DEBUG=1 for full traceback.", file=sys.stderr)
        sys.exit(2)


def _detect_error_format() -> str:
    env_format = os.environ.get("NPA_ERROR_FORMAT", "").lower()
    if env_format in {"json", "text"}:
        return env_format
    args = sys.argv[1:]
    for index, value in enumerate(args):
        if value in {"--output", "--output-format", "--format"} and index + 1 < len(args):
            if args[index + 1].lower() == "json":
                return "json"
        if value in {"--output=json", "--output-format=json", "--format=json"}:
            return "json"
    return "text"
