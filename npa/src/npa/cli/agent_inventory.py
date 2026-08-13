"""`npa agent list` — inventory of agent deployments npa knows about.

There was no way to ask what agents exist: operators had to guess a
`--project`/`--name` pair and run `npa agent status` (or `destroy`) blind. The
records live in ``~/.npa/config.yaml`` under ``projects.<alias>.agents``, which is
what `deploy` writes and `destroy` removes, so this reads them directly and needs
no cloud call.
"""

from __future__ import annotations

import json
from typing import Any

import typer


def agent_list_cmd(
    project: str = typer.Option(
        "", "--project", help="Only list agents recorded under this project alias."
    ),
    output_json: bool = typer.Option(False, "--json", help="Emit JSON instead of a table."),
) -> None:
    """List agent deployments recorded in ~/.npa/config.yaml."""
    rows = agent_rows(project)
    if output_json:
        typer.echo(json.dumps(rows, indent=2, sort_keys=True))
        return
    if not rows:
        scope = f" under project {project!r}" if project else ""
        typer.echo(
            f"No agents recorded{scope}. `npa agent setup` deploys one; "
            "`npa configure` sets the project up first."
        )
        return
    typer.echo(_format_table(rows))
    typer.echo("")
    typer.echo(
        "Inspect one with `npa agent status --project <project> --name <name>`, "
        "or remove it with `npa agent destroy --project <project> --name <name>`."
    )


def agent_rows(project: str = "") -> list[dict[str, Any]]:
    """Return one row per recorded agent, across projects unless *project* is given."""
    from npa.clients.config import list_projects

    try:
        projects = list_projects()
    except Exception:  # noqa: BLE001 - an unreadable config lists nothing
        return []
    wanted = str(project or "").strip()
    rows: list[dict[str, Any]] = []
    for alias, stanza in sorted((projects or {}).items()):
        if wanted and alias != wanted:
            continue
        agents = (stanza or {}).get("agents") if isinstance(stanza, dict) else None
        if not isinstance(agents, dict):
            continue
        for name, record in sorted(agents.items()):
            record = record if isinstance(record, dict) else {}
            rows.append(
                {
                    "project": alias,
                    "name": name,
                    "public_ip": str(record.get("public_ip", "") or ""),
                    "region": str(record.get("region", "") or ""),
                    "instance_id": str(record.get("instance_id", "") or ""),
                    "agent_url": str(record.get("agent_url", "") or ""),
                    "created_at": str(record.get("created_at", "") or ""),
                }
            )
    return rows


def _format_table(rows: list[dict[str, Any]]) -> str:
    headers = ["PROJECT", "NAME", "PUBLIC_IP", "REGION", "INSTANCE_ID", "URL"]
    keys = ["project", "name", "public_ip", "region", "instance_id", "agent_url"]
    values = [[str(row.get(key, "") or "-") for key in keys] for row in rows]
    widths = [
        max(len(headers[index]), *(len(value[index]) for value in values))
        for index in range(len(headers))
    ]
    lines = ["  ".join(header.ljust(widths[index]) for index, header in enumerate(headers))]
    lines.append("  ".join("-" * width for width in widths))
    lines.extend(
        "  ".join(value[index].ljust(widths[index]) for index in range(len(headers)))
        for value in values
    )
    return "\n".join(lines)
