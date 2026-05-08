"""npa adapter — convert simulation data to training formats."""

from __future__ import annotations

import typer

app = typer.Typer(
    name="adapter",
    help="Convert simulation data to training dataset formats.",
    no_args_is_help=True,
)


def _is_s3_uri(path: str) -> bool:
    return path.startswith("s3://")


@app.command("convert")
def convert_cmd(
    input_dir: str = typer.Option(
        ..., "--input-path", "--input", "-i", help="Directory of episode numpy arrays."
    ),
    output_dir: str = typer.Option(
        ..., "--output-path", "--output", "-o", help="Output LeRobotDataset v3 directory."
    ),
    fps: int = typer.Option(20, "--fps", help="Frame rate for video encoding."),
    robot: str = typer.Option(
        "franka_panda", "--robot", help="Robot type identifier."
    ),
    task: str = typer.Option(
        "Pick and place cube to target",
        "--task",
        help="Task description for the dataset.",
    ),
) -> None:
    """Convert Genesis/sim demo numpy arrays to LeRobotDataset v3 format."""
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from rich.console import Console

    from npa.adapter.sim_to_lerobot import AdapterError, convert

    console = Console(stderr=True)
    temp_dirs: list[TemporaryDirectory[str]] = []

    try:
        if _is_s3_uri(input_dir):
            from npa.clients.storage import StorageClient

            tmp = TemporaryDirectory(prefix="npa-adapter-input-")
            temp_dirs.append(tmp)
            inp = Path(
                StorageClient.from_environment().download_directory(input_dir, tmp.name)
            )
        else:
            inp = Path(input_dir)

        if _is_s3_uri(output_dir):
            tmp = TemporaryDirectory(prefix="npa-adapter-output-")
            temp_dirs.append(tmp)
            out = Path(tmp.name)
        else:
            out = Path(output_dir)

        if not inp.exists():
            console.print(f"[red]Error:[/red] Input directory does not exist: {inp}")
            raise typer.Exit(1)

        console.print(f"[bold]Converting demos → LeRobotDataset v3[/bold]")
        console.print(f"  input:  {inp}")
        console.print(f"  output: {output_dir if _is_s3_uri(output_dir) else out}")
        console.print(f"  fps={fps}  robot={robot}")

        try:
            convert(inp, out, fps=fps, robot_type=robot, task=task)
        except AdapterError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(1)

        if _is_s3_uri(output_dir):
            from npa.clients.storage import StorageClient

            uploaded = StorageClient.from_environment().upload_directory(str(out), output_dir)
            console.print(f"[green]Conversion complete.[/green] Dataset at: {uploaded}")
        else:
            console.print(f"[green]Conversion complete.[/green] Dataset at: {out}")
    finally:
        for tmp in temp_dirs:
            tmp.cleanup()
