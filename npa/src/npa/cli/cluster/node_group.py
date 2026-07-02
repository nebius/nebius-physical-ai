"""`npa cluster node-group` commands."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

import typer

from npa.cli.cluster.scope import CLUSTER_SCOPE_EPILOG
from npa.cluster.api import ClusterInfo, MK8sClient, NodeGroupInfo, cluster_subnet_id
from npa.cluster.config import (
    DEFAULT_CPU_NODE_GROUP_PRESET,
    DEFAULT_K8S_VERSION,
    DEFAULT_NODE_PLATFORM,
    CpuNodeGroupConfig,
    NodeGroupConfig,
    resolve_project_id,
)
from npa.cluster.exceptions import ClusterConfigError, ClusterError, ClusterNotFoundError, NodeGroupNotFoundError
from npa.cluster.node_group import L40S_WARNING
from npa.cluster.state import (
    ClusterState,
    NodeGroupState,
    delete_node_group_state,
    list_local_clusters,
    list_node_group_states,
    load_cluster_state,
    load_node_group_state,
    save_node_group_state,
    utc_now_iso,
)

app = typer.Typer(
    name="node-group",
    help="Manage GPU node groups attached to NPA Workbench cluster targets.",
    epilog=CLUSTER_SCOPE_EPILOG,
    no_args_is_help=True,
)


def add_cmd(
    cluster_name: str = typer.Option(..., "--cluster-name", help="Parent NPA cluster target/profile name."),
    name: str = typer.Option(
        "",
        "--name",
        help="NPA node-group profile name. Defaults from cluster target and GPU type.",
    ),
    gpu_type: str = typer.Option(..., "--gpu-type", help="GPU type: h100, h200, l40s, or rtx6000."),
    node_count: int = typer.Option(1, "--node-count", help="GPU worker count for this NPA node-group profile."),
    node_preset: str = typer.Option(
        "",
        "--node-preset",
        help="Override the NPA default GPU node preset for this profile.",
    ),
    public_ip: bool = typer.Option(
        False,
        "--public-ip",
        help="Assign public IPs to GPU nodes in this NPA target profile.",
    ),
    capacity_block_group: str = typer.Option(
        "",
        "--capacity-block-group",
        help="Optional private capacity block group ID for strict reservation selection.",
    ),
    autoscaling_min: int | None = typer.Option(
        None,
        "--autoscaling-min",
        help=(
            "Autoscaling minimum for this NPA GPU node-group profile. "
            "Use nebius mk8s for broader node-group administration."
        ),
    ),
    autoscaling_max: int | None = typer.Option(
        None,
        "--autoscaling-max",
        help=(
            "Autoscaling maximum for this NPA GPU node-group profile. "
            "Use nebius mk8s for broader node-group administration."
        ),
    ),
    wait: bool = typer.Option(
        True,
        "--wait/--no-wait",
        help="Wait until the NPA GPU node-group target is READY/RUNNING.",
    ),
    timeout: int = typer.Option(30, "--timeout", help="GPU target readiness timeout in minutes."),
    project_id: str = typer.Option("", "--project-id", help="Nebius project ID. Defaults from local state or config."),
    subnet_id: str = typer.Option(
        "",
        "--subnet-id",
        help="VPC subnet ID. Defaults from local cluster state or the cluster control plane.",
    ),
) -> None:
    """Attach a GPU node-group profile to an NPA Workbench cluster target.

    Wraps `nebius mk8s` node-group create with NPA GPU aliases and local cache.
    """

    try:
        client = MK8sClient(timeout=timeout * 60, poll_interval=30.0)
        cluster, local_state = _resolve_cluster(client, cluster_name, project_id)
        config = NodeGroupConfig(
            cluster_name=cluster_name,
            name=name,
            gpu_type=gpu_type,
            project_id=cluster.project_id,
            cluster_id=cluster.id,
            node_count=node_count,
            node_preset=node_preset,
            public_ip=public_ip,
            autoscaling_min=autoscaling_min,
            autoscaling_max=autoscaling_max,
            wait=wait,
            timeout_minutes=timeout,
            k8s_version=(local_state.k8s_version if local_state else DEFAULT_K8S_VERSION),
            subnet_id=(
                subnet_id.strip()
                or (local_state.subnet_id if local_state else "")
                or cluster_subnet_id(cluster)
            ),
            capacity_block_group=capacity_block_group,
        )
        if not config.subnet_id:
            raise ClusterConfigError(
                "subnet ID is required for GPU node groups. Pass --subnet-id or deploy the "
                "cluster with `npa cluster deploy` so local state captures it."
            )
        if config.gpu_type == "l40s":
            typer.echo(L40S_WARNING, err=True)

        typer.echo(
            f"Creating GPU node group {config.name} on {cluster.name or cluster_name} "
            f"({config.platform}/{config.node_preset})..."
        )
        node_group = client.create_gpu_node_group(config, cluster.id)
        state = _state_from_config(config, node_group.id, node_group.status)
        save_node_group_state(state)

        if wait:
            def on_state_change(current: NodeGroupInfo) -> None:
                typer.echo(f"State: node_group={current.name}:{current.status}")

            node_group = client.wait_for_node_group_ready(
                cluster.id,
                node_group.id or config.name,
                timeout_minutes=config.timeout_minutes,
                on_state_change=on_state_change,
            )
            state = replace(
                state,
                node_group_id=node_group.id or state.node_group_id,
                last_seen_state=node_group.status,
                node_count=node_group.node_count or state.node_count,
            )
            save_node_group_state(state)

        typer.echo(f"Node group: {config.name}")
        typer.echo(f"Node group ID: {state.node_group_id}")
        typer.echo(f"State: {state.last_seen_state}")
        typer.echo(f"GPU type: {state.gpu_type}")
        typer.echo(f"Platform/preset: {state.platform}/{state.preset}")
        typer.echo(f"Nodes: {state.node_count}")
    except ClusterError as exc:
        typer.echo(f"Node-group add failed: {exc}", err=True)
        raise typer.Exit(1) from exc


def add_cpu_cmd(
    cluster_name: str = typer.Option(..., "--cluster-name", help="Parent NPA cluster target/profile name."),
    name: str = typer.Option(
        "",
        "--name",
        help="NPA node-group profile name. Defaults from cluster target and preset.",
    ),
    platform: str = typer.Option(
        DEFAULT_NODE_PLATFORM,
        "--platform",
        help="CPU node platform: cpu-e2 or cpu-d3.",
    ),
    preset: str = typer.Option(
        DEFAULT_CPU_NODE_GROUP_PRESET,
        "--preset",
        help="CPU preset, e.g. 8vcpu-32gb, 16vcpu-64gb, 32vcpu-128gb.",
    ),
    node_count: int = typer.Option(1, "--node-count", help="Fixed CPU worker count for this profile."),
    autoscaling_min: int | None = typer.Option(
        None,
        "--autoscaling-min",
        help="Autoscaling minimum (use 0 to scale CPU capacity to zero when idle).",
    ),
    autoscaling_max: int | None = typer.Option(
        None,
        "--autoscaling-max",
        help="Autoscaling maximum for batched CPU workloads.",
    ),
    public_ip: bool = typer.Option(
        False,
        "--public-ip",
        help="Assign public IPs to CPU nodes in this profile.",
    ),
    boot_disk_size_gib: int = typer.Option(
        128,
        "--boot-disk-size-gib",
        help="Boot disk size per CPU node, in GiB.",
    ),
    subnet_id: str = typer.Option(
        "",
        "--subnet-id",
        help="VPC subnet ID. Defaults from local cluster state when available.",
    ),
    wait: bool = typer.Option(
        True,
        "--wait/--no-wait",
        help="Wait until the CPU node-group target is READY/RUNNING.",
    ),
    timeout: int = typer.Option(30, "--timeout", help="CPU target readiness timeout in minutes."),
    project_id: str = typer.Option("", "--project-id", help="Nebius project ID. Defaults from local state or config."),
) -> None:
    """Attach a CPU node-group profile to an NPA Workbench cluster target.

    Adds GPU-free, batchable capacity for CPU-only workloads (motion
    retargeting, batched inference) so they do not consume GPU nodes. Wraps
    `nebius mk8s` node-group create with NPA CPU presets and local cache.
    """

    try:
        client = MK8sClient(timeout=timeout * 60, poll_interval=30.0)
        cluster, local_state = _resolve_cluster(client, cluster_name, project_id)
        config = CpuNodeGroupConfig(
            cluster_name=cluster_name,
            name=name,
            project_id=cluster.project_id,
            cluster_id=cluster.id,
            platform=platform,
            node_preset=preset,
            node_count=node_count,
            public_ip=public_ip,
            autoscaling_min=autoscaling_min,
            autoscaling_max=autoscaling_max,
            wait=wait,
            timeout_minutes=timeout,
            boot_disk_size_gib=boot_disk_size_gib,
            k8s_version=(local_state.k8s_version if local_state else DEFAULT_K8S_VERSION),
            subnet_id=(subnet_id.strip() or (local_state.subnet_id if local_state else "")),
        )

        typer.echo(
            f"Creating CPU node group {config.name} on {cluster.name or cluster_name} "
            f"({config.platform}/{config.node_preset})..."
        )
        node_group = client.create_node_group(
            cluster_id=cluster.id,
            name=config.name,
            platform=config.platform,
            preset=config.node_preset,
            node_count=config.node_count,
            public_ip=config.public_ip,
            autoscaling_min=config.autoscaling_min,
            autoscaling_max=config.autoscaling_max,
            subnet_id=config.subnet_id,
            k8s_version=config.k8s_version,
            boot_disk_type=config.boot_disk_type,
            boot_disk_size_gib=config.boot_disk_size_gib,
        )
        state = _cpu_state_from_config(config, node_group.id, node_group.status)
        save_node_group_state(state)

        if wait:
            def on_state_change(current: NodeGroupInfo) -> None:
                typer.echo(f"State: node_group={current.name}:{current.status}")

            node_group = client.wait_for_node_group_ready(
                cluster.id,
                node_group.id or config.name,
                timeout_minutes=config.timeout_minutes,
                on_state_change=on_state_change,
            )
            state = replace(
                state,
                node_group_id=node_group.id or state.node_group_id,
                last_seen_state=node_group.status,
                node_count=node_group.node_count or state.node_count,
            )
            save_node_group_state(state)

        typer.echo(f"Node group: {config.name}")
        typer.echo(f"Node group ID: {state.node_group_id}")
        typer.echo(f"State: {state.last_seen_state}")
        typer.echo(f"Platform/preset: {state.platform}/{state.preset}")
        typer.echo(f"Nodes: {state.node_count}")
        if config.autoscaling_min is not None:
            typer.echo(f"Autoscaling: {config.autoscaling_min}-{config.autoscaling_max}")
    except ClusterError as exc:
        typer.echo(f"CPU node-group add failed: {exc}", err=True)
        raise typer.Exit(1) from exc


def remove_cmd(
    cluster_name: str = typer.Option(..., "--cluster-name", help="Parent NPA cluster target/profile name."),
    name: str = typer.Option(..., "--name", help="NPA node-group profile name to remove."),
    force: bool = typer.Option(False, "--force", help="Skip confirmation for NPA node-group cleanup."),
    timeout: int = typer.Option(30, "--timeout", help="Node-group target cleanup wait timeout in minutes."),
    project_id: str = typer.Option("", "--project-id", help="Nebius project ID. Defaults from local state or config."),
) -> None:
    """Remove an NPA GPU node-group profile and clean up its local cache.

    Wraps `nebius mk8s` node-group delete for target cleanup.
    """

    try:
        client = MK8sClient(timeout=timeout * 60, poll_interval=30.0)
        local_state = load_node_group_state(cluster_name, name)
        try:
            cluster, _ = _resolve_cluster(client, cluster_name, project_id)
        except ClusterNotFoundError:
            if local_state is not None:
                delete_node_group_state(cluster_name, name)
                typer.echo(f"Node group {name} no longer has a remote cluster; local state removed.")
                return
            typer.echo(f"Node group {name} not found.")
            return

        remote = _find_node_group(client, cluster.id, name, local_state)
        if remote is None:
            if local_state is not None:
                delete_node_group_state(cluster_name, name)
                typer.echo(f"Node group {name} no longer exists remotely; local state removed.")
                return
            typer.echo(f"Node group {name} not found.")
            return

        if not force and not typer.confirm(f"Remove node group {remote.name or name} ({remote.id})?"):
            raise typer.Exit(1)

        typer.echo(f"Removing node group {remote.name or name} ({remote.id})...")
        client.delete_node_group(cluster.id, remote.id)
        client.wait_for_node_group_deleted(cluster.id, remote.id, timeout_minutes=timeout)
        delete_node_group_state(cluster_name, name)
        typer.echo(f"Node group {name} removed and local state cleaned.")
    except ClusterError as exc:
        typer.echo(f"Node-group remove failed: {exc}", err=True)
        raise typer.Exit(1) from exc


def status_cmd(
    cluster_name: str = typer.Option(..., "--cluster-name", help="Parent NPA cluster target/profile name."),
    name: str = typer.Option("", "--name", help="NPA node-group profile name. Lists all node groups when omitted."),
    output_format: str = typer.Option("table", "--format", help="Output format: table or json."),
    project_id: str = typer.Option("", "--project-id", help="Nebius project ID. Defaults from local state or config."),
) -> None:
    """Show NPA GPU node-group target state from Nebius and the local cache."""

    _emit_status(cluster_name=cluster_name, name=name, output_format=output_format, project_id=project_id)


def list_cmd(
    cluster_name: str = typer.Option(
        "",
        "--cluster-name",
        help="Parent NPA cluster target/profile name. Lists all known targets when omitted.",
    ),
    output_format: str = typer.Option("table", "--format", help="Output format: table or json."),
    project_id: str = typer.Option("", "--project-id", help="Nebius project ID. Defaults from local state or config."),
) -> None:
    """List GPU node-group profiles attached to NPA Workbench cluster targets."""

    _emit_status(cluster_name=cluster_name, name="", output_format=output_format, project_id=project_id)


def _emit_status(*, cluster_name: str, name: str, output_format: str, project_id: str) -> None:
    fmt = output_format.lower()
    if fmt not in {"table", "json"}:
        raise typer.BadParameter("--format must be table or json")
    try:
        client = MK8sClient(timeout=120, poll_interval=30.0)
        rows = _collect_rows(client=client, cluster_name=cluster_name, name=name, project_id=project_id)
        if fmt == "json":
            typer.echo(json.dumps(rows, indent=2, sort_keys=True))
        else:
            typer.echo(_format_table(rows))
    except ClusterError as exc:
        typer.echo(f"Node-group status failed: {exc}", err=True)
        raise typer.Exit(1) from exc


def _collect_rows(
    *,
    client: MK8sClient,
    cluster_name: str,
    name: str,
    project_id: str,
) -> list[dict[str, Any]]:
    clusters = _target_clusters(client, cluster_name, project_id)
    rows: list[dict[str, Any]] = []
    for cluster, local_cluster in clusters:
        local_states = {state.name: state for state in list_node_group_states(cluster.name or local_cluster.name)}
        remote_groups = {group.name: group for group in client.list_node_groups(cluster.id)}
        if name:
            target_names = [name]
        else:
            target_names = sorted(set(local_states) | set(remote_groups))
        for target_name in target_names:
            local_state = local_states.get(target_name)
            remote = remote_groups.get(target_name)
            if remote is None and local_state is not None:
                try:
                    remote = client.get_node_group(cluster.id, local_state.node_group_id or target_name)
                except NodeGroupNotFoundError:
                    pass
            if remote is None and local_state is None:
                continue
            rows.append(_row_for_node_group(cluster, local_state, remote, target_name))
    return rows


def _target_clusters(
    client: MK8sClient,
    cluster_name: str,
    project_id: str,
) -> list[tuple[ClusterInfo, ClusterState]]:
    if cluster_name:
        cluster, local_state = _resolve_cluster(client, cluster_name, project_id)
        if local_state is None:
            local_state = _local_state_from_remote(cluster)
        return [(cluster, local_state)]

    local_by_name = {state.name: state for state in list_local_clusters()}
    remote_by_name: dict[str, ClusterInfo] = {}
    resolved_project_id = _resolve_project(project_id)
    if resolved_project_id:
        for cluster in client.list_clusters(resolved_project_id):
            remote_by_name[cluster.name] = cluster
    target_names = sorted(set(local_by_name) | set(remote_by_name))
    targets: list[tuple[ClusterInfo, ClusterState]] = []
    for target_name in target_names:
        local_state = local_by_name.get(target_name)
        remote = remote_by_name.get(target_name)
        if remote is None and local_state is not None:
            try:
                remote = client.get_cluster(local_state.cluster_id, project_id=local_state.project_id or resolved_project_id)
            except ClusterNotFoundError:
                continue
        if remote is not None:
            targets.append((remote, local_state or _local_state_from_remote(remote)))
    return targets


def _resolve_cluster(
    client: MK8sClient,
    cluster_name: str,
    project_id: str,
) -> tuple[ClusterInfo, ClusterState | None]:
    local_state = load_cluster_state(cluster_name)
    lookup_project_id = (local_state.project_id if local_state else "") or _resolve_project(project_id)
    if local_state is not None:
        return client.get_cluster(local_state.cluster_id, project_id=lookup_project_id), local_state
    if not lookup_project_id:
        raise ClusterConfigError("Nebius project ID is required when local cluster state is missing.")
    return client.get_cluster(cluster_name, project_id=lookup_project_id), None


def _resolve_project(explicit_project_id: str) -> str:
    if explicit_project_id.strip():
        return explicit_project_id.strip()
    try:
        return resolve_project_id()
    except ClusterConfigError:
        return ""


def _find_node_group(
    client: MK8sClient,
    cluster_id: str,
    name: str,
    local_state: NodeGroupState | None,
) -> NodeGroupInfo | None:
    targets = []
    if local_state is not None and local_state.node_group_id:
        targets.append(local_state.node_group_id)
    targets.append(name)
    for target in targets:
        try:
            return client.get_node_group(cluster_id, target)
        except NodeGroupNotFoundError:
            continue
    return None


def _state_from_config(config: NodeGroupConfig, node_group_id: str, state: str) -> NodeGroupState:
    return NodeGroupState(
        cluster_name=config.cluster_name,
        name=config.name,
        node_group_id=node_group_id,
        gpu_type=config.gpu_type,
        platform=config.platform,
        preset=config.node_preset,
        node_count=config.node_count,
        created_at=utc_now_iso(),
        last_seen_state=state,
        public_ip=config.public_ip,
        autoscaling_min=config.autoscaling_min,
        autoscaling_max=config.autoscaling_max,
    )


def _cpu_state_from_config(config: CpuNodeGroupConfig, node_group_id: str, state: str) -> NodeGroupState:
    return NodeGroupState(
        cluster_name=config.cluster_name,
        name=config.name,
        node_group_id=node_group_id,
        gpu_type="cpu",
        platform=config.platform,
        preset=config.node_preset,
        node_count=config.node_count,
        created_at=utc_now_iso(),
        last_seen_state=state,
        public_ip=config.public_ip,
        autoscaling_min=config.autoscaling_min,
        autoscaling_max=config.autoscaling_max,
    )


def _row_for_node_group(
    cluster: ClusterInfo,
    local_state: NodeGroupState | None,
    remote: NodeGroupInfo | None,
    fallback_name: str,
) -> dict[str, Any]:
    state = remote.status if remote is not None else "UNKNOWN"
    created_at = (remote.created_at if remote is not None else "") or (local_state.created_at if local_state else "")
    row = {
        "cluster_name": cluster.name,
        "name": (remote.name if remote is not None and remote.name else fallback_name),
        "node_group_id": (remote.id if remote is not None else "") or (local_state.node_group_id if local_state else ""),
        "state": state,
        "gpu_type": (remote.gpu_type if remote is not None else "") or (local_state.gpu_type if local_state else ""),
        "node_count": (remote.node_count if remote is not None and remote.node_count else 0)
        or (local_state.node_count if local_state else 0),
        "age": _age(created_at),
        "created_at": created_at,
        "platform": (remote.platform if remote is not None else "") or (local_state.platform if local_state else ""),
        "preset": (remote.preset if remote is not None else "") or (local_state.preset if local_state else ""),
        "public_ip": (remote.public_ip if remote is not None else False) or (local_state.public_ip if local_state else False),
        "autoscaling_min": (remote.autoscaling_min if remote is not None else None)
        if remote is not None and remote.autoscaling_min is not None
        else (local_state.autoscaling_min if local_state else None),
        "autoscaling_max": (remote.autoscaling_max if remote is not None else None)
        if remote is not None and remote.autoscaling_max is not None
        else (local_state.autoscaling_max if local_state else None),
    }
    if local_state is not None and remote is not None:
        save_node_group_state(
            replace(
                local_state,
                node_group_id=row["node_group_id"],
                last_seen_state=state,
                node_count=row["node_count"],
                platform=row["platform"],
                preset=row["preset"],
                gpu_type=row["gpu_type"],
                public_ip=row["public_ip"],
                autoscaling_min=row["autoscaling_min"],
                autoscaling_max=row["autoscaling_max"],
            )
        )
    return row


def _local_state_from_remote(cluster: ClusterInfo) -> ClusterState:
    return ClusterState(
        name=cluster.name,
        cluster_id=cluster.id,
        project_id=cluster.project_id,
        region="eu-north1",
        node_count=cluster.node_count,
        node_platform="",
        node_preset="",
        k8s_version=DEFAULT_K8S_VERSION,
        subnet_id="",
        created_at=cluster.created_at or utc_now_iso(),
        last_seen_state=cluster.status,
        endpoint=cluster.endpoint,
    )


def _format_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No node groups found."
    headers = ["CLUSTER", "NAME", "NODE_GROUP_ID", "STATE", "GPU", "NODES", "PLATFORM", "PRESET", "AGE"]
    values = [
        [
            str(row["cluster_name"]),
            str(row["name"]),
            str(row["node_group_id"]),
            str(row["state"]),
            str(row["gpu_type"]),
            str(row["node_count"]),
            str(row["platform"]),
            str(row["preset"]),
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


app.command("add")(add_cmd)
app.command("add-cpu")(add_cpu_cmd)
app.command("remove")(remove_cmd)
app.command("status")(status_cmd)
app.command("list")(list_cmd)
