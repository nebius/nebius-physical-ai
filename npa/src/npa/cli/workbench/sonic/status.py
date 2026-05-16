"""SONIC status command."""

from __future__ import annotations

from typing import Any

import typer

from npa.cli.workbench.sonic.helpers import (
    OutputFormat,
    WorkbenchRuntime,
    context,
    enum_value,
    fail,
    is_sonic_workbench,
    output,
)
from npa.clients.config import list_projects
from npa.clients.serverless import EndpointNotFoundError, ServerlessClient, ServerlessClientError


def _configured_status(project: str, name: str) -> dict[str, Any]:
    project_cfg = list_projects().get(project, {})
    workbenches = project_cfg.get("workbenches", {}) if isinstance(project_cfg, dict) else {}
    wb_cfg = workbenches.get(name, {}) if isinstance(workbenches, dict) else {}
    if not isinstance(wb_cfg, dict) or not is_sonic_workbench(name, wb_cfg):
        fail("--name must reference a configured SONIC workbench for vm/container/byovm status.")
    return {
        "project": project,
        "workbench": name,
        "runtime": wb_cfg.get("runtime", "vm"),
        "mode": wb_cfg.get("mode", "unknown"),
        "checkpoint_source": wb_cfg.get("checkpoint_source", ""),
        "checkpoint_path": wb_cfg.get("checkpoint_path", ""),
        "ports": {
            "zmq": wb_cfg.get("zmq_port", 5556),
            "debug": wb_cfg.get("port", 5557),
        },
        "build_state": wb_cfg.get("build_state", "unknown"),
        "last_smoke_status": wb_cfg.get("last_smoke_status", "unknown"),
        "app_status": wb_cfg.get("app_status", "unknown"),
    }


def status_cmd(
    runtime: WorkbenchRuntime = typer.Option(WorkbenchRuntime.vm, "--runtime", help="Runtime to inspect."),
    name: str = typer.Option("", "--name", help="Workbench or serverless job name."),
    project_id: str = typer.Option("", "--project-id", help="Nebius project ID for serverless job lookup."),
    job_id: str = typer.Option("", "--job-id", help="Serverless Job ID or name."),
    output_format: OutputFormat = typer.Option(
        OutputFormat.text, "--output-format", "--output", help="Output format."
    ),
) -> None:
    """Inspect SONIC runtime state."""

    runtime_value = enum_value(runtime)
    ctx = context()
    target_name = name or ctx.name
    if runtime_value == "serverless":
        lookup = job_id or target_name
        if not lookup:
            fail("SONIC status --runtime serverless requires --job-id or --name.")
        if not project_id:
            fail("SONIC status --runtime serverless requires --project-id.")
        client = ServerlessClient()
        try:
            info = client.get_job(lookup, project_id)
        except EndpointNotFoundError:
            output({"status": "not_found", "job": lookup, "project_id": project_id}, output_format)
            return
        except ServerlessClientError as exc:
            fail(f"Serverless Job lookup failed: {exc}")
        output(
            {
                "status": info.status,
                "job_id": info.id,
                "job_name": info.name,
                "project_id": project_id,
                "runtime": "serverless",
            },
            output_format,
        )
        return
    output(_configured_status(ctx.project, target_name), output_format)
