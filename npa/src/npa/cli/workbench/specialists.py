"""Operate the optional local specialist runtime through its shared coordinator."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path

import typer

from npa.agent_backend.specialists.config import load_config
from npa.agent_backend.specialists.team import SpecialistTeam
from npa.lifecycle_intent import json_stdout_contract

app = typer.Typer(
    help="Self-hosted specialist agents and durable task monitoring.",
    no_args_is_help=True,
)


class OutputFormat(str, Enum):
    """Declare the machine-readable output contract for finite management commands.

    Args: None.
    Returns: The supported JSON output selection.
    Raises: None.
    """

    json = "json"


@app.callback()
def configure(
    ctx: typer.Context,
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
):
    """Select an operator-owned team configuration.

    Args: ctx: CLI context. config: Local JSON configuration, not a data handoff path.
    Returns: None; configures subsequent management commands.
    Raises: ValueError, OSError: Configuration is invalid.
    """
    ctx.obj = {"path": str(config), "team": SpecialistTeam(load_config(config))}


@app.command("serve")
def serve(
    ctx: typer.Context, host: str = "127.0.0.1", port: int = 8790, workers: bool = True
):
    """Serve the browser monitor and optionally supervise independent workers.

    Args: ctx: CLI context. host, port: HTTP bind address. workers: Start profile processes.
    Returns: None until service shutdown.
    Raises: ValueError, OSError: Authentication or startup fails.
    """
    import uvicorn
    from npa.agent_backend.specialists.service import create_app

    uvicorn.run(create_app(ctx.obj["path"], with_workers=workers), host=host, port=port)


@app.command("worker")
def worker(ctx: typer.Context, specialist: str):
    """Run one specialist independently of the monitor.

    Args: ctx: CLI context. specialist: Configured role.
    Returns: None until SIGINT or SIGTERM.
    Raises: ValueError, OSError: Configuration or storage fails.
    """
    from npa.agent_backend.specialists.worker import run_worker

    run_worker(ctx.obj["path"], specialist)


@app.command("submit")
@json_stdout_contract(fail_closed_on_exception=True)
def submit(
    ctx: typer.Context,
    goal: str,
    specialist: str = "auto",
    task_id: str = "",
    parent_id: str = "",
    output_format: OutputFormat = OutputFormat.json,
):
    """Enqueue a goal and emit its durable JSON receipt.

    Args: ctx: CLI context. goal: Request. specialist: Role or auto. task_id: Retry identity.
        parent_id: Completed task to follow up. output_format: JSON receipt format.
    Returns: None; writes one JSON task receipt.
    Raises: ValueError, KeyError: Request is invalid.
    """
    _emit(
        ctx.obj["team"].submit(
            goal, specialist=specialist, task_id=task_id, parent_id=parent_id
        )
    )


@app.command("status")
@json_stdout_contract(fail_closed_on_exception=True)
def status(
    ctx: typer.Context,
    task_id: str = "",
    output_format: OutputFormat = OutputFormat.json,
):
    """Emit observed team or task state as JSON.

    Args: ctx: CLI context. task_id: Optional exact task. output_format: JSON result format.
    Returns: None; writes one JSON result.
    Raises: KeyError: Task does not exist.
    """
    _emit(ctx.obj["team"].status(task_id))


@app.command("pause")
@json_stdout_contract(fail_closed_on_exception=True)
def pause(
    ctx: typer.Context,
    task_id: str = "",
    specialist: str = "",
    resume: bool = False,
    output_format: OutputFormat = OutputFormat.json,
):
    """Pause a task/profile, or resume it with --resume.

    Args: ctx: CLI context. task_id, specialist: Exactly one target. resume: Resume instead.
        output_format: JSON state format.
    Returns: None; writes the resulting JSON state.
    Raises: ValueError, KeyError: Target cannot be controlled.
    """
    _emit(
        ctx.obj["team"].pause(task_id=task_id, specialist=specialist, paused=not resume)
    )


@app.command("reconcile")
@json_stdout_contract(fail_closed_on_exception=True)
def reconcile(
    ctx: typer.Context,
    task_id: str,
    call_id: str = "",
    result: str = "",
    retry: bool = False,
    output_format: OutputFormat = OutputFormat.json,
):
    """Record an inspected operation result or explicitly authorize retry.

    Args: ctx: CLI context. task_id: Blocked task. call_id: Interrupted call.
        result: Verified JSON object. retry: Explicit retry authorization.
        output_format: JSON receipt format.
    Returns: None; emits the requeued task receipt.
    Raises: ValueError, KeyError: Reconciliation is invalid.
    """
    receipt = json.loads(result) if result else None
    if receipt is not None and not isinstance(receipt, dict):
        raise typer.BadParameter("result must be a JSON object")
    _emit(
        ctx.obj["team"].reconcile(task_id, call_id=call_id, result=receipt, retry=retry)
    )


def _emit(value):
    typer.echo(json.dumps(value, indent=2, sort_keys=True))


@app.command("cancel")
@json_stdout_contract(fail_closed_on_exception=True)
def cancel(
    ctx: typer.Context, task_id: str, output_format: OutputFormat = OutputFormat.json
):
    """Stop a task at its next durable boundary; external workloads remain independent.

    Args: ctx: CLI context. task_id: Existing task. output_format: JSON receipt format.
    Returns: None; emits the cancellation receipt.
    Raises: ValueError, KeyError: Task cannot be cancelled.
    """
    _emit(ctx.obj["team"].cancel(task_id))
