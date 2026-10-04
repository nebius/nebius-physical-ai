"""Expose FLUX Action fine-tuning through the standard S3 workbench interface."""

from __future__ import annotations

import json
from enum import Enum

import typer

from npa.lifecycle_intent import json_stdout_contract
from npa.workbench.flux_action.schemas import FinetuneRequest

app = typer.Typer(
    no_args_is_help=True, help="Fine-tune FLUX 3 Action on a declared robot embodiment."
)


class OutputFormat(str, Enum):
    """Supported command renderings.

    Args: text or json.
    Returns: A format member.
    Raises: ValueError for unsupported formats.
    """

    text = "text"
    json = "json"


@app.command("finetune")
@json_stdout_contract
def finetune_cmd(
    input_path: str = typer.Option(
        ..., "--input-path", help="S3 prefix containing LeRobot meta/data/videos."
    ),
    recipe_uri: str = typer.Option(
        ..., "--recipe-uri", help="S3 JSON robot contract and training schedule."
    ),
    output_path: str = typer.Option(
        ..., "--output-path", help="Fresh, run-specific S3 artifact prefix."
    ),
    processes: int = typer.Option(
        8, "--processes", min=1, help="Local torchrun GPU processes; match allocation."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Read recipe/metadata and validate without training."
    ),
    output_format: OutputFormat = typer.Option(OutputFormat.json, "--output-format"),
) -> None:
    """Index demonstrations, train the native policy, and export a verified checkpoint.

    Args: S3 dataset, recipe, output paths, GPU count, and rendering settings.
    Returns: None; emits one result document.
    Raises: typer.Exit if validation or training fails.
    """
    from npa.workbench.flux_action.runner import finetune

    try:
        request = FinetuneRequest(
            input_path=input_path,
            recipe_uri=recipe_uri,
            output_path=output_path,
            processes=processes,
        )
        result = finetune(request, dry_run=dry_run)
    except (ValueError, RuntimeError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    typer.echo(
        json.dumps(result) if output_format == OutputFormat.json else result["status"]
    )
