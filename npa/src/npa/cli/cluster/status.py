"""`npa cluster status` and `npa cluster list` commands."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

import typer

from npa.cluster.api import ClusterInfo, MK8sClient
from npa.cluster.config import DEFAULT_REGION, resolve_project_id
from npa.cluster.exceptions import ClusterConfigError, ClusterError, ClusterNotFoundError
from npa.cluster.state import ClusterState, list_local_clusters, load_cluster_state, save_cluster_state


def status_cmd(
    name: str = typer.Option("", "--name", help="Cluster name. Lists all known clusters when omitted."),
    output_format: str = typer.Option("table", "--format", help="Output format: table or json."),
    project_id: str = typer.Option("", "--project-id", help="Nebius project ID. Defaults from local state or NPA config."),
) -> None:
    """Show cluster state from Nebius and the local cache."""

    _emit_status(name=name, output_format=output_format, project_id=project_id)


def list_cmd(
    output_format: str = typer.Option("table", "--format", help="Output format: table or json."),
    project_id: str = typer.Option("", "--project-id", help="Nebius project ID. Defaults from NPA config."),
) -> None:
    """List known NPA clusters."""

    _emit_status(name="", output_format=output_format, project_id=project_id)


def _emit_status(*, name: str, output_format: str, project_id: str) -> None:
    fmt = output_format.lower()
    if fmt not in {"table", "json"}:
        raise typer.BadParameter("--format must be table or json")
    try:
        resolved_project_id = _resolve_project_for_status(project_id)
        rows = _collect_rows(name=name, project_id=resolved_project_id)
        if fmt == "json":
            typer.echo(json.dumps(rows, indent=2, sort_keys=True))
        else:
            typer.echo(_format_table(rows))
    except ClusterError as exc:
        typer.echo(f"Cluster status failed: {exc}", err=True)
        raise typer.Exit(1) from exc


def _resolve_project_for_status(explicit_project_id: str) -> str:
    if explicit_project_id.strip():
        return explicit_project_id.strip()
    try:
        return resolve_project_id()
    except ClusterConfigError:
        return ""


def _collect_rows(*, name: str, project_id: str) -> list[dict[str, Any]]:
    client = MK8sClient(timeout=120, poll_interval=30.0)
    local_by_name = {state.name: state for state in list_local_clusters()}
    remote_by_name: dict[str, ClusterInfo] = {}

    if name:
        local_state = load_cluster_state(name)
        if local_state is not None:
            local_by_name[name] = local_state
        lookup_project_id = (local_state.project_id if local_state else "") or project_id
        if lookup_project_id:
            try:
                remote = client.get_cluster(
                    local_state.cluster_id if local_state else name,
                    project_id=lookup_project_id,
                )
                remote_by_name[remote.name or name] = remote
            except ClusterNotFoundError:
                pass
        target_names = [name]
    else:
        if project_id:
            for remote in client.list_clusters(project_id):
                remote_by_name[remote.name] = remote
        target_names = sorted(set(local_by_name) | set(remote_by_name))

    rows: list[dict[str, Any]] = []
    for target_name in target_names:
        local_state = local_by_name.get(target_name)
        remote = remote_by_name.get(target_name)
        if remote is None and local_state is not None:
            try:
                remote = client.get_cluster(local_state.cluster_id, project_id=local_state.project_id or project_id)
            except ClusterNotFoundError:
                pass
        rows.append(_row_for_cluster(client, target_name, local_state, remote))
    return rows


def _row_for_cluster(
    client: MK8sClient,
    name: str,
    local_state: ClusterState | None,
    remote: ClusterInfo | None,
) -> dict[str, Any]:
    node_count = local_state.node_count if local_state else 0
    node_group_id = local_state.node_group_id if local_state else ""
    if remote is not None and remote.id:
        groups = client.list_node_groups(remote.id)
        if groups:
            node_count = sum(group.node_count for group in groups)
            node_group_id = groups[0].id
    state = remote.status if remote is not None else "UNKNOWN"
    endpoint = (remote.endpoint if remote is not None else "") or (local_state.endpoint if local_state else "")
    created_at = (remote.created_at if remote is not None else "") or (local_state.created_at if local_state else "")
    row = {
        "name": (remote.name if remote is not None and remote.name else name),
        "cluster_id": (remote.id if remote is not None else "") or (local_state.cluster_id if local_state else ""),
        "state": state,
        "region": local_state.region if local_state else DEFAULT_REGION,
        "node_count": node_count,
        "node_group_id": node_group_id,
        "endpoint": endpoint,
        "age": _age(created_at),
        "created_at": created_at,
        "project_id": (remote.project_id if remote is not None else "") or (local_state.project_id if local_state else ""),
    }
    if local_state is not None and remote is not None:
        save_cluster_state(
            replace(
                local_state,
                last_seen_state=state,
                node_group_id=node_group_id or local_state.node_group_id,
                node_count=node_count or local_state.node_count,
                endpoint=endpoint,
            )
        )
    return row


def _format_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No clusters found."
    headers = ["NAME", "CLUSTER_ID", "STATE", "REGION", "NODES", "ENDPOINT", "AGE"]
    values = [
        [
            str(row["name"]),
            str(row["cluster_id"]),
            str(row["state"]),
            str(row["region"]),
            str(row["node_count"]),
            str(row["endpoint"]),
            str(row["age"]),
        ]
        for row in rows
    ]
    widths = [
        max(len(headers[index]), *(len(value[index]) for value in values))
        for index in range(len(headers))
    ]
    lines = ["  ".join(header.ljust(widths[index]) for index, header in enumerate(headers))]
    lines.append("  ".join("-" * width for width in widths))
    lines.extend("  ".join(value[index].ljust(widths[index]) for index in range(len(headers))) for value in values)
    return "\n".join(lines)


def _age(created_at: str) -> str:
    if not created_at:
        return ""
    try:
        timestamp = created_at.replace("Z", "+00:00")
        created = datetime.fromisoformat(timestamp)
    except ValueError:
        return ""
    delta = datetime.now(timezone.utc) - created.astimezone(timezone.utc)
    seconds = max(0, int(delta.total_seconds()))
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    if hours < 48:
        return f"{hours}h"
    return f"{hours // 24}d"
