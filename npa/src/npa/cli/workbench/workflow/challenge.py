"""Expose local challenge preparation through the standard workflow command group."""

from enum import Enum
import json
from pathlib import Path

from botocore.exceptions import BotoCoreError, ClientError
import typer

from npa.lifecycle_intent import json_stdout_contract
from npa.clients.storage import StorageError
from npa.orchestration.npa_workflow.challenges import check_setup, initialize, prepare

app = typer.Typer(
    help="Prepare a BEHAVIOR DEV evaluation from one setup file.", no_args_is_help=True
)


class OutputFormat(str, Enum):
    """Select machine-readable output.

    Args:
        value: The supported JSON format.
    Returns:
        Output format member.
    Raises:
        ValueError: The format is unsupported.
    """

    json = "json"


def _emit(operation, *args, **kwargs):
    try:
        result = operation(*args, **kwargs)
    except ValueError as exc:
        result = {"status": "blocked", "issues": str(exc).splitlines()}
    except OSError:
        result = {
            "status": "blocked",
            "issues": [
                "Use readable inputs and a new writable kit directory; existing files are preserved."
            ],
        }
    except (BotoCoreError, ClientError, RuntimeError, StorageError):
        result = {
            "status": "blocked",
            "issues": [
                "Input publication failed. Check the selected project's storage access; retain the local kit and run ID."
            ],
        }
    typer.echo(json.dumps(result))
    if result.get("status") == "blocked":
        raise typer.Exit(1)


@app.command("init")
@json_stdout_contract
def init_cmd(
    directory: Path = typer.Option(
        ..., "--directory", help="New private local setup directory."
    ),
    output_format: OutputFormat = typer.Option(OutputFormat.json, "--output-format"),
) -> None:
    """Create an editable BEHAVIOR setup file without provisioning resources.

    Args:
        directory: New setup directory; its parent must exist.
        output_format: JSON output format.
    Returns:
        None.
    Raises:
        typer.Exit: The directory cannot be created.
    """
    _emit(initialize, directory)


@app.command("check")
@json_stdout_contract
def check_cmd(
    config: Path = typer.Option(..., "--config", help="Private setup YAML."),
    output_format: OutputFormat = typer.Option(OutputFormat.json, "--output-format"),
) -> None:
    """Check local prerequisites; GPU readiness requires runtime preflight.

    Args:
        config: Local setup YAML.
        output_format: JSON output format.
    Returns:
        None.
    Raises:
        typer.Exit: Local configuration or source verification fails.
    """
    _emit(check_setup, config.expanduser())


@app.command("prepare")
@json_stdout_contract
def prepare_cmd(
    config: Path = typer.Option(..., "--config", help="Private setup YAML."),
    directory: Path = typer.Option(
        ..., "--directory", help="New private local evaluation kit."
    ),
    run_id: str = typer.Option(
        ..., "--run-id", help="Explicit unique workflow run identity."
    ),
    publish: bool = typer.Option(
        False, "--publish", help="Verify project storage and publish immutable inputs."
    ),
    output_format: OutputFormat = typer.Option(OutputFormat.json, "--output-format"),
) -> None:
    """Freeze ten prescribed DEV cases and generate standard Workbench commands.

    Args:
        config: Local setup YAML.
        directory: New kit directory; its parent must exist.
        run_id: Explicit workflow identity.
        publish: Upload and verify recipe and runbook in project storage.
        output_format: JSON output format.
    Returns:
        None.
    Raises:
        typer.Exit: Preparation or publication fails; existing evidence is retained.
    """
    _emit(prepare, config, directory, run_id, publish=publish)
