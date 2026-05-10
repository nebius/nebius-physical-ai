"""Standalone LeRobotDataset to Rerun recording conversion command."""

from __future__ import annotations

import typer
from rich.console import Console

from npa.clients.storage import StorageError
from npa.viz.adapters import groot_predictions_to_rerun, lerobot_to_rerun
from npa.viz.adapters.lerobot_to_rerun import RerunAdapterError
from npa.viz.lerobot import VizDataError


console = Console(stderr=True)


def lerobot_to_rrd_cmd(
    input_path: str = typer.Option(
        ..., "--input-path", "--input", "-i", help="Local or s3:// LeRobotDataset directory."
    ),
    output_path: str = typer.Option(
        ..., "--output-path", "--output", "-o", help="Local or s3:// .rrd output path."
    ),
    duration: float | None = typer.Option(
        None, "--duration", help="Maximum recording duration in seconds. Defaults to adapter cap."
    ),
    predictions_path: str = typer.Option(
        "", "--predictions-path", help="Optional local or s3:// GR00T prediction artifact path."
    ),
) -> None:
    """Convert a LeRobotDataset, optionally with GR00T predictions, to a Rerun `.rrd`."""
    try:
        if predictions_path:
            groot_predictions_to_rerun(
                predictions_path,
                input_path,
                output_path,
                duration_s=duration,
            )
        else:
            lerobot_to_rerun(
                input_path,
                output_path,
                duration_s=duration,
            )
    except (RerunAdapterError, StorageError, VizDataError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc

    console.print("[green]Conversion complete.[/green]")
    console.print(f"  output: {output_path}")
    console.print("  format: Rerun .rrd")
