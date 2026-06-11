"""npa workbench golden-eval — per-container golden-eval / hello-world reruns.

Each Workbench container declares one golden eval in
``npa/src/npa/smoke/golden_evals.yaml``. These commands list, inspect, validate,
and run those evals. ``validate`` is offline (used by nightly CI); ``run``
executes the eval command and is meant to run on a host/GPU that has the
container's runtime available.
"""

from __future__ import annotations

import json
import shlex
import subprocess
from enum import Enum

import typer
from rich.console import Console
from rich.table import Table

from npa.deploy.images import CONTAINER_IMAGE_NAMES
from npa.smoke.manifest import container, load_manifest, validate_manifest

app = typer.Typer(
    name="golden-eval",
    help="Per-container golden-eval / hello-world reruns.",
    no_args_is_help=True,
)
console = Console()
err_console = Console(stderr=True)


class OutputFormat(str, Enum):
    text = "text"
    json = "json"


@app.command("list")
def list_evals(
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", "-o"),
) -> None:
    """List every container and its golden eval."""

    specs = load_manifest()
    if output is OutputFormat.json:
        payload = {
            name: {
                "image": spec.image,
                "physical_ai_useful": spec.physical_ai.get("useful"),
                "kind": spec.golden_eval.kind,
                "gpu": spec.golden_eval.gpu,
                "status": spec.golden_eval.status,
                "command": spec.golden_eval.command,
            }
            for name, spec in specs.items()
        }
        console.print_json(json.dumps(payload))
        return

    table = Table(title="Container golden evals")
    table.add_column("container", style="cyan", no_wrap=True)
    table.add_column("kind")
    table.add_column("gpu")
    table.add_column("status")
    table.add_column("command", overflow="fold")
    for name, spec in specs.items():
        ge = spec.golden_eval
        table.add_row(name, ge.kind, ge.gpu, ge.status, ge.command)
    console.print(table)


@app.command("show")
def show(name: str = typer.Argument(..., help="Container key, e.g. 'lerobot'.")) -> None:
    """Show the full safety + Physical AI + golden-eval record for a container."""

    try:
        spec = container(name)
    except KeyError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    payload = {
        "name": spec.name,
        "image": spec.image,
        "dockerfile": spec.dockerfile,
        "foundation": spec.foundation,
        "physical_ai": spec.physical_ai,
        "safety": spec.safety,
        "golden_eval": {
            "kind": spec.golden_eval.kind,
            "command": spec.golden_eval.command,
            "gpu": spec.golden_eval.gpu,
            "timeout_seconds": spec.golden_eval.timeout_seconds,
            "status": spec.golden_eval.status,
            "module": spec.golden_eval.module,
            "env_module": spec.golden_eval.env_module,
            "artifact": spec.golden_eval.artifact,
        },
    }
    console.print_json(json.dumps(payload))


@app.command("validate")
def validate() -> None:
    """Validate manifest completeness and consistency (offline; nightly CI gate)."""

    report = validate_manifest(expected_tools=set(CONTAINER_IMAGE_NAMES))
    if report.ok:
        count = len(load_manifest())
        console.print(f"[green]OK[/green]: {count} containers have valid golden-eval entries")
        return
    err_console.print("[red]Golden-eval manifest validation failed:[/red]")
    for issue in report.issues:
        err_console.print(f"  - {issue}")
    raise typer.Exit(code=1)


@app.command("run")
def run(
    name: str = typer.Argument(..., help="Container key, e.g. 'lerobot'."),
    execute: bool = typer.Option(
        False,
        "--execute/--dry-run",
        help="Execute the eval command locally (requires the container runtime).",
    ),
    serverless: bool = typer.Option(
        False,
        "--serverless",
        help="Run the golden eval in its container image on a Nebius Serverless GPU.",
    ),
    gpu: str = typer.Option(
        "", "--gpu", help="Serverless GPU type override (e.g. h200, h100, l40s, b300)."
    ),
    timeout: str = typer.Option("40m", "--timeout", help="Serverless job timeout."),
) -> None:
    """Print, execute locally, or run on serverless a container's golden eval.

    - default: print the command (dry run).
    - ``--execute``: run locally (only works where the runtime is present).
    - ``--serverless``: submit the eval to a Nebius Serverless Job in the real
      container image on a GPU, and wait for the PASS/FAIL result.
    """

    try:
        spec = container(name)
    except KeyError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    ge = spec.golden_eval
    console.print(f"[cyan]{spec.name}[/cyan] ({spec.image}) golden eval: {ge.kind}, gpu={ge.gpu}")
    console.print(f"  $ {ge.command}")

    if serverless:
        from npa.smoke.serverless_runner import submit_golden_eval

        def _on_change(job: object) -> None:
            err_console.print(f"  -> {getattr(job, 'status', '?')}")

        result = submit_golden_eval(
            name,
            gpu_type=gpu or None,
            timeout=timeout,
            on_state_change=_on_change,
        )
        console.print_json(json.dumps(result))
        if not result.get("ok"):
            raise typer.Exit(code=1)
        return

    if not execute:
        return

    try:
        completed = subprocess.run(
            shlex.split(ge.command),
            timeout=ge.timeout_seconds,
            check=False,
        )
    except FileNotFoundError as exc:
        err_console.print(f"[red]command not runnable here: {exc}[/red]")
        raise typer.Exit(code=2) from exc
    except subprocess.TimeoutExpired as exc:
        err_console.print(f"[red]golden eval timed out after {ge.timeout_seconds}s[/red]")
        raise typer.Exit(code=124) from exc
    if completed.returncode != 0:
        raise typer.Exit(code=completed.returncode)
