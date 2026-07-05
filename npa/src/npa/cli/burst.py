"""CLI entrypoint for burst SkyPilot jobs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from npa import burst


app = typer.Typer(
    name="burst",
    help="Submit and inspect cold-start multi-node SkyPilot GPU jobs.",
    no_args_is_help=True,
)


@app.command("submit")
def submit_cmd(
    image: str = typer.Option(
        ...,
        "--image",
        help="Container image to run. Plain image refs are rendered as docker:<image>.",
    ),
    nodes: int = typer.Option(
        ...,
        "--nodes",
        min=1,
        help="Number of SkyPilot nodes to gang-schedule.",
    ),
    gpu_per_node: str = typer.Option(
        ...,
        "--gpu-per-node",
        help="SkyPilot accelerator spec per node, e.g. L40S:1 or H100:8.",
    ),
    entrypoint: str = typer.Option(
        ...,
        "--entrypoint",
        help="Shell command executed by torchrun on each worker process.",
    ),
    name: str = typer.Option(
        "npa-burst",
        "--name",
        help="SkyPilot managed job name.",
    ),
    output_json: bool = typer.Option(
        False,
        "--json",
        help="Print only the serialized job handle.",
    ),
) -> None:
    """Submit one coupled multi-node burst job."""

    handle = burst.submit(
        image=image,
        num_nodes=nodes,
        gpu_per_node=gpu_per_node,
        entrypoint=entrypoint,
        name=name,
    )
    if output_json:
        typer.echo(handle.to_json())
        return
    typer.echo(f"submitted burst job {handle.job_id} ({handle.name})")
    typer.echo(f"handle: {handle.to_json()}")


@app.command("submit-yaml")
def submit_yaml_cmd(
    yaml_path: Path = typer.Argument(..., help="Rendered single-task SkyPilot/workbench YAML to submit."),
    name: Optional[str] = typer.Option(
        None,
        "--name",
        help="SkyPilot managed job name. Defaults to the task name in the YAML.",
    ),
    var: list[str] = typer.Option(
        [],
        "--var",
        "-v",
        help="Template/env override as KEY=VALUE. May be repeated.",
    ),
    output_json: bool = typer.Option(
        False,
        "--json",
        help="Print only the serialized job handle.",
    ),
) -> None:
    """Submit one rendered workbench YAML task through the burst path."""

    handle = burst.submit_yaml(
        yaml_path,
        name=name,
        env_overrides=_parse_vars(var),
    )
    if output_json:
        typer.echo(handle.to_json())
        return
    typer.echo(f"submitted burst yaml job {handle.job_id} ({handle.name})")
    typer.echo(f"handle: {handle.to_json()}")


@app.command("status")
def status_cmd(
    job: str = typer.Argument(..., help="Job ID or serialized burst handle JSON."),
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        help="SkyPilot global config path to use for this query.",
    ),
) -> None:
    """Query a burst job status."""

    result = burst.status(job, config_path=config_path)
    typer.echo(json.dumps(result.__dict__, sort_keys=True))


@app.command("logs")
def logs_cmd(
    job: str = typer.Argument(..., help="Job ID or serialized burst handle JSON."),
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow logs until completion."),
    tail: Optional[int] = typer.Option(None, "--tail", min=1, help="Return only the last N lines."),
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        help="SkyPilot global config path to use for this query.",
    ),
) -> None:
    """Stream or fetch burst job logs."""

    result = burst.logs(job, follow=follow, tail=tail, config_path=config_path)
    typer.echo(result.text, nl=not result.text.endswith("\n"))


def _parse_vars(values: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise typer.BadParameter(f"--var must be KEY=VALUE, got {value!r}")
        key, item = value.split("=", 1)
        key = key.strip()
        if not key:
            raise typer.BadParameter("--var key must be non-empty")
        parsed[key] = item
    return parsed
