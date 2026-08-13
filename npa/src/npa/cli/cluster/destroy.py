"""`npa cluster destroy` command."""

from __future__ import annotations

from pathlib import Path

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
from npa.lifecycle_intent import OperationIntent, intent_boundary


@intent_boundary(OperationIntent.DESTROY)
def destroy_cmd(
    name: str = typer.Option(..., "--name", help="NPA cluster target/profile name to clean up."),
    force: bool = typer.Option(False, "--force", help="Skip confirmation for NPA target cleanup."),
    timeout: int = typer.Option(30, "--timeout", help="Target cleanup wait timeout in minutes."),
    project_id: str = typer.Option("", "--project-id", help="Nebius project ID. Defaults from local state or NPA config."),
) -> None:
    """Delete a Managed Kubernetes cluster through the API and drop its local state.

    For a cluster created by `npa cluster up` / `npa provision-if-absent`, use
    `npa cluster down` instead: Terraform owns the VPC network and subnet as well,
    and `down` removes those, the cluster, and the local state in one step. This
    command is the API-only path — for a cluster Terraform does not manage, or to
    clear local state for a cluster that is already gone.
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
        _warn_terraform_leftovers(name)
    except ClusterError as exc:
        typer.echo(f"Cluster destroy failed: {exc}", err=True)
        raise typer.Exit(1) from exc


def _terraform_state_dir() -> Path | None:
    """Return a deploy/cluster directory that holds Terraform state, or None."""
    for candidate in (Path.cwd() / "deploy" / "cluster",):
        if any(candidate.glob("*.tfstate")) or (candidate / ".terraform").is_dir():
            return candidate
    return None


def _warn_terraform_leftovers(name: str) -> None:
    """Say what this command does *not* delete.

    `destroy` removes the Managed Kubernetes cluster through the API, but a
    cluster created by `npa cluster up` / `npa provision-if-absent` also owns a
    Terraform-managed VPC network and subnet. Deleting only the cluster leaves
    those running and the Terraform state describing a cluster that is gone —
    reported as an indefinitely orphaned `<cluster>-network`.
    """
    tf_dir = _terraform_state_dir()
    if tf_dir is None:
        return
    typer.echo("")
    typer.echo(
        f"Note: Terraform state in {tf_dir} still describes this cluster and owns its "
        f"network/subnet ({name}-network). The API delete above does not touch them.",
        err=True,
    )
    typer.echo(
        f"  Finish the teardown with `npa cluster down --terraform-dir {tf_dir} --force` "
        "(it reads project/tenant/region from ~/.npa/config.yaml when tfvars omit them).",
        err=True,
    )


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
