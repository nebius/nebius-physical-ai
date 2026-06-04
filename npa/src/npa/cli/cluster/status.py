"""`npa cluster status` and `npa cluster list` commands."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import typer

from npa.cli.cluster.terraform_lifecycle import _read_tfvars, terraform_status
from npa.cluster.api import ClusterInfo, MK8sClient
from npa.cluster.config import DEFAULT_REGION, resolve_project_id
from npa.cluster.exceptions import ClusterConfigError, ClusterError, ClusterNotFoundError
from npa.cluster.state import (
    ClusterState,
    kubeconfig_file,
    list_local_clusters,
    load_cluster_state,
    save_cluster_state,
)


def status_cmd(
    name: str = typer.Option("", "--name", help="NPA cluster target name. Lists all known targets when omitted."),
    output_format: str = typer.Option("table", "--format", help="Output format: table or json."),
    project_id: str = typer.Option("", "--project-id", help="Nebius project ID. Defaults from local state or NPA config."),
    terraform_dir: Path | None = typer.Option(
        None,
        "--terraform-dir",
        help="Terraform cluster directory to include outputs from.",
    ),
) -> None:
    """Show NPA cluster target state from Nebius and the local cache."""

    _emit_status(name=name, output_format=output_format, project_id=project_id, terraform_dir=terraform_dir)


def list_cmd(
    output_format: str = typer.Option("table", "--format", help="Output format: table or json."),
    project_id: str = typer.Option("", "--project-id", help="Nebius project ID. Defaults from NPA config."),
) -> None:
    """List NPA Workbench cluster targets known locally or in the configured project."""

    _emit_status(name="", output_format=output_format, project_id=project_id)


def _emit_status(*, name: str, output_format: str, project_id: str, terraform_dir: Path | None = None) -> None:
    fmt = output_format.lower()
    if fmt not in {"table", "json"}:
        raise typer.BadParameter("--format must be table or json")
    try:
        resolved_project_id = _resolve_project_for_status(project_id)
        rows = _collect_rows(name=name, project_id=resolved_project_id, terraform_dir=terraform_dir)
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


def _collect_rows(*, name: str, project_id: str, terraform_dir: Path | None = None) -> list[dict[str, Any]]:
    client = MK8sClient(timeout=120, poll_interval=30.0)
    local_by_name = {state.name: state for state in list_local_clusters()}
    remote_by_name: dict[str, ClusterInfo] = {}
    terraform_row = _terraform_row(terraform_dir)

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
        if terraform_row and terraform_row["name"] == name:
            local_by_name.setdefault(name, _state_from_terraform_row(terraform_row))
    else:
        if project_id:
            for remote in client.list_clusters(project_id):
                remote_by_name[remote.name] = remote
        target_names = sorted(set(local_by_name) | set(remote_by_name) | ({terraform_row["name"]} if terraform_row else set()))

    rows: list[dict[str, Any]] = []
    for target_name in target_names:
        local_state = local_by_name.get(target_name)
        remote = remote_by_name.get(target_name)
        if remote is None and local_state is not None:
            try:
                remote = client.get_cluster(local_state.cluster_id, project_id=local_state.project_id or project_id)
            except ClusterNotFoundError:
                pass
        row = _row_for_cluster(client, target_name, local_state, remote)
        if terraform_row is not None and terraform_row["name"] == target_name:
            row = _merge_terraform_row(row, terraform_row)
        rows.append(row)
    return rows


def _terraform_row(terraform_dir: Path | None) -> dict[str, Any] | None:
    if terraform_dir is None:
        return None
    outputs = terraform_status(terraform_dir)
    if not outputs:
        return None
    cluster = outputs.get("kube_cluster", {}).get("value") or {}
    if not isinstance(cluster, dict) or not cluster.get("name"):
        return None
    endpoints = cluster.get("endpoints") if isinstance(cluster.get("endpoints"), dict) else {}
    filesystem = outputs.get("shared_filesystem", {}).get("value") or {}
    shared_filesystem_id = str(filesystem.get("id") or "") if isinstance(filesystem, dict) else ""
    filesystem_csi = outputs.get("filesystem_csi", {}).get("value") or {}
    if not shared_filesystem_id:
        filesystem_csi = {}
    tfvars = _read_tfvars(terraform_dir)
    name = str(cluster.get("name"))
    return {
        "name": name,
        "cluster_id": str(cluster.get("id") or ""),
        "region": str(tfvars.get("region") or DEFAULT_REGION),
        "endpoint": str(endpoints.get("public_endpoint") or ""),
        "kubeconfig_path": str(kubeconfig_file(name)),
        "terraform_dir": str(terraform_dir),
        "k8s_training_ref": str(outputs.get("k8s_training_ref", {}).get("value") or ""),
        "shared_filesystem_id": shared_filesystem_id,
        "filesystem_csi_storage_class": (
            str(filesystem_csi.get("storage_class_name") or "") if isinstance(filesystem_csi, dict) else ""
        ),
        "filesystem_csi_status": str(filesystem_csi.get("status") or "") if isinstance(filesystem_csi, dict) else "",
    }


def _state_from_terraform_row(row: dict[str, Any]) -> ClusterState:
    return ClusterState(
        name=str(row["name"]),
        cluster_id=str(row.get("cluster_id") or ""),
        project_id="",
        region=str(row.get("region") or DEFAULT_REGION),
        node_count=0,
        node_platform="",
        node_preset="",
        k8s_version="",
        subnet_id="",
        created_at="",
        endpoint=str(row.get("endpoint") or ""),
        kubeconfig_path=str(row.get("kubeconfig_path") or ""),
    )


def _merge_terraform_row(row: dict[str, Any], terraform_row: dict[str, Any]) -> dict[str, Any]:
    merged = dict(row)
    for key, value in terraform_row.items():
        if key in {"name", "cluster_id", "endpoint", "kubeconfig_path"}:
            merged[key] = merged.get(key) or value
        else:
            merged[key] = value
    return merged


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
        "kubeconfig_path": local_state.kubeconfig_path if local_state else "",
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
    headers = ["NAME", "CLUSTER_ID", "STATE", "REGION", "NODES", "ENDPOINT", "KUBECONFIG", "AGE"]
    values = [
        [
            str(row["name"]),
            str(row["cluster_id"]),
            str(row["state"]),
            str(row["region"]),
            str(row["node_count"]),
            str(row["endpoint"]),
            str(row["kubeconfig_path"]),
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
