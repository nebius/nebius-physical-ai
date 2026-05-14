"""`npa cluster deploy` command."""

from __future__ import annotations

from dataclasses import replace

import typer

from npa.cluster.api import MK8sClient
from npa.cluster.config import (
    DEFAULT_K8S_VERSION,
    DEFAULT_NODE_PLATFORM,
    DEFAULT_NODE_PRESET,
    DEFAULT_REGION,
    ClusterConfig,
    resolve_project_id,
)
from npa.cluster.exceptions import ClusterError
from npa.cluster.state import ClusterState, kubeconfig_file, save_cluster_state, utc_now_iso
from npa.serverless_common.subnet import resolve_subnet


def deploy_cmd(
    name: str = typer.Option(..., "--name", help="Human-readable cluster name."),
    region: str = typer.Option(DEFAULT_REGION, "--region", help="Nebius region."),
    node_count: int = typer.Option(1, "--node-count", help="Number of CPU worker nodes."),
    node_preset: str = typer.Option(DEFAULT_NODE_PRESET, "--node-preset", help="CPU node preset."),
    k8s_version: str = typer.Option(DEFAULT_K8S_VERSION, "--k8s-version", help="Kubernetes major.minor version."),
    wait: bool = typer.Option(True, "--wait/--no-wait", help="Wait until cluster and node group are READY."),
    timeout: int = typer.Option(30, "--timeout", help="Deploy wait timeout in minutes."),
    project_id: str = typer.Option("", "--project-id", help="Nebius project ID. Defaults from NPA config or env."),
    subnet_id: str = typer.Option("", "--subnet-id", help="VPC subnet ID. Defaults through NPA subnet resolution."),
    node_platform: str = typer.Option(DEFAULT_NODE_PLATFORM, "--node-platform", help="CPU node platform."),
) -> None:
    """Provision a CPU-only Nebius Managed Kubernetes cluster."""

    try:
        resolved_project_id = resolve_project_id(project_id)
        resolved_subnet_id = subnet_id.strip() or resolve_subnet(resolved_project_id)
        config = ClusterConfig(
            name=name,
            project_id=resolved_project_id,
            region=region,
            node_count=node_count,
            node_platform=node_platform,
            node_preset=node_preset,
            k8s_version=k8s_version,
            subnet_id=resolved_subnet_id,
            wait=wait,
            timeout_minutes=timeout,
        )
        client = MK8sClient(timeout=timeout * 60, poll_interval=30.0)
        typer.echo(f"Creating cluster {config.name} in {config.region}...")
        cluster = client.create_cluster(config)
        state = _state_from_cluster(config, cluster.id, cluster.status, cluster.node_group_id, cluster.endpoint)
        save_cluster_state(state, metadata=_metadata("created", cluster.status))

        if wait:
            def on_state_change(current, groups) -> None:
                group_state = ", ".join(f"{group.name}:{group.status}" for group in groups) or "no-node-groups"
                typer.echo(f"State: cluster={current.status} node_groups={group_state}")

            cluster = client.wait_for_ready(
                cluster.id,
                project_id=config.project_id,
                expected_node_count=config.node_count,
                timeout_minutes=config.timeout_minutes,
                on_state_change=on_state_change,
            )
            state = replace(
                state,
                last_seen_state=cluster.status,
                node_group_id=cluster.node_group_id or state.node_group_id,
                node_count=cluster.node_count or state.node_count,
                endpoint=cluster.endpoint or state.endpoint,
            )
            save_cluster_state(state, metadata=_metadata("ready", cluster.status))

        kubeconfig_path = kubeconfig_file(config.name)
        client.get_kubeconfig(cluster.id, kubeconfig_path, context_name=config.name, external=True)
        state = replace(state, kubeconfig_path=str(kubeconfig_path))
        save_cluster_state(state, metadata=_metadata("kubeconfig_written", state.last_seen_state))

        typer.echo(f"Cluster ID: {cluster.id}")
        if state.node_group_id:
            typer.echo(f"Node group ID: {state.node_group_id}")
        typer.echo(f"State: {state.last_seen_state}")
        typer.echo(f"Kubeconfig: {kubeconfig_path}")
    except ClusterError as exc:
        typer.echo(f"Cluster deploy failed: {exc}", err=True)
        raise typer.Exit(1) from exc


def _state_from_cluster(
    config: ClusterConfig,
    cluster_id: str,
    state: str,
    node_group_id: str,
    endpoint: str,
) -> ClusterState:
    return ClusterState(
        name=config.name,
        cluster_id=cluster_id,
        project_id=config.project_id,
        region=config.region,
        node_count=config.node_count,
        node_platform=config.node_platform,
        node_preset=config.node_preset,
        k8s_version=config.k8s_version,
        subnet_id=config.subnet_id,
        created_at=utc_now_iso(),
        last_seen_state=state,
        node_group_id=node_group_id,
        endpoint=endpoint,
    )


def _metadata(event: str, state: str) -> dict[str, str]:
    return {
        "managed_by": "npa cluster",
        "event": event,
        "last_seen_state": state,
        "updated_at": utc_now_iso(),
        "teardown": "Run `npa cluster destroy --name <name> --force` when finished.",
    }
