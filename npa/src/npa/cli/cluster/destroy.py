"""`npa cluster destroy` command."""

from __future__ import annotations

import typer

from npa.cluster.api import ClusterInfo, MK8sClient
from npa.cluster.config import resolve_project_id
from npa.cluster.exceptions import ClusterConfigError, ClusterError, ClusterNotFoundError
from npa.cluster.state import (
    ClusterState,
    delete_cluster_state,
    list_local_clusters,
    load_cluster_state,
)


def destroy_cmd(
    name: str = typer.Option(..., "--name", help="NPA cluster target/profile name to clean up."),
    force: bool = typer.Option(False, "--force", help="Skip confirmation for NPA target cleanup."),
    timeout: int = typer.Option(30, "--timeout", help="Target cleanup wait timeout in minutes."),
    project_id: str = typer.Option("", "--project-id", help="Nebius project ID. Defaults from local state or NPA config."),
) -> None:
    """Clean up an NPA Workbench cluster target and remove its local profile state.

    Wraps `nebius mk8s` cluster delete for target cleanup.
    """

    try:
        local_state = load_cluster_state(name)
        resolved_project_id = _resolve_project_for_destroy(local_state, project_id)
        client = MK8sClient(timeout=timeout * 60, poll_interval=30.0)
        target = _find_destroy_target(client, name, local_state, resolved_project_id)

        if target is None:
            if local_state is not None:
                delete_cluster_state(name)
                typer.echo(f"Cluster {name} no longer exists remotely; local state removed.")
                return
            available = ", ".join(cluster.name for cluster in list_local_clusters()) or "(none)"
            typer.echo(f"Cluster {name} not found. Local clusters: {available}")
            return

        if not force and not typer.confirm(f"Destroy cluster {target.name or name} ({target.id})?"):
            raise typer.Exit(1)

        typer.echo(f"Destroying cluster {target.name or name} ({target.id})...")
        client.delete_cluster(target.id, project_id=resolved_project_id)
        client.wait_for_deleted(target.id, project_id=resolved_project_id, timeout_minutes=timeout)
        delete_cluster_state(name)
        typer.echo(f"Cluster {name} destroyed and local state removed.")
    except ClusterError as exc:
        typer.echo(f"Cluster destroy failed: {exc}", err=True)
        raise typer.Exit(1) from exc


def _resolve_project_for_destroy(local_state: ClusterState | None, explicit_project_id: str) -> str:
    if explicit_project_id.strip():
        return explicit_project_id.strip()
    if local_state is not None and local_state.project_id:
        return local_state.project_id
    try:
        return resolve_project_id()
    except ClusterConfigError:
        return ""


def _find_destroy_target(
    client: MK8sClient,
    name: str,
    local_state: ClusterState | None,
    project_id: str,
) -> ClusterInfo | None:
    if local_state is not None:
        try:
            return client.get_cluster(local_state.cluster_id, project_id=local_state.project_id or project_id)
        except ClusterNotFoundError:
            return None
    if not project_id:
        return None
    try:
        return client.get_cluster(name, project_id=project_id)
    except ClusterNotFoundError:
        return None
