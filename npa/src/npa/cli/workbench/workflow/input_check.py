"""Expose read-only PAIDF input diagnostics before workflow submission."""

import json
from pathlib import Path

import typer

from npa.cli.workbench.workflow.batch import OutputFormat
from npa.lifecycle_intent import json_stdout_contract
from npa.workflows.data_factory_input_diagnostics import check_paidf_input


@json_stdout_contract
def check_input_cmd(
    input_video: Path | None = typer.Option(None, "--input-video"),
    input_uri: str = typer.Option("", "--input-uri", "--input-path"),
    lerobot_uri: str = typer.Option("", "--lerobot-uri"),
    lerobot_camera: str = typer.Option("", "--lerobot-camera"),
    lerobot_episode: int | None = typer.Option(None, "--lerobot-episode", min=0),
    project: str = typer.Option("", "--project", "-p"),
    s3_endpoint: str = typer.Option("", "--s3-endpoint"),
    output_format: OutputFormat = typer.Option(OutputFormat.json, "--output-format"),
) -> None:
    """Check PAIDF MP4 or LeRobot selection and decoding without uploading.

    Args:
        input_video: Local MP4.
        input_uri: One S3 MP4 object.
        lerobot_uri: S3 LeRobot dataset root.
        lerobot_camera: Explicit full camera feature.
        lerobot_episode: Explicit episode index.
        project: Saved storage project.
        s3_endpoint: Storage endpoint override.
        output_format: JSON or text.
    Returns:
        None.
    Raises:
        typer.Exit: Input validation failed.
    """
    result = check_paidf_input(
        input_video=input_video,
        input_uri=input_uri,
        lerobot_uri=lerobot_uri,
        camera=lerobot_camera,
        episode=lerobot_episode,
        project=project,
        endpoint=s3_endpoint,
    )
    _emit_input_check(result, output_format)


def _emit_input_check(result, output_format):
    typer.echo(
        json.dumps(result, indent=2)
        if output_format == OutputFormat.json
        else str(result)
    )
    if result["status"] != "passed":
        raise typer.Exit(1)
