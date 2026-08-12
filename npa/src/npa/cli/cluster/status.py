"""`npa cluster status` and `npa cluster list` commands."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any

import typer

from npa.lifecycle_intent import OperationIntent, intent_boundary

from npa.cli.cluster.terraform_lifecycle import _read_tfvars, terraform_status
from npa.cluster.api import ClusterInfo, MK8sClient
from npa.cluster.config import resolve_project_id
from npa.cluster.exceptions import (
    ClusterConfigError,
    ClusterError,
    ClusterNotFoundError,
)
from npa.cluster.state import (
    ClusterState,
    kubeconfig_file,
    list_local_clusters,
    load_cluster_state,
    save_cluster_state,
)

logger = logging.getLogger(__name__)


@intent_boundary(OperationIntent.OBSERVE)
def status_cmd(
    name: str = typer.Option(
        "",
        "--name",
        help="NPA cluster target name. Lists all known targets when omitted.",
    ),
    output_format: str = typer.Option(
        "table", "--format", help="Output format: table or json."
    ),
    project_id: str = typer.Option(
        "",
        "--project-id",
        help="Nebius project ID. Defaults from local state or NPA config.",
    ),
    project: str = typer.Option(
        "",
        "--project",
        help="NPA project alias whose saved project_id to use (like `npa cluster up`).",
    ),
    terraform_dir: Path | None = typer.Option(
        None,
        "--terraform-dir",
        help="Terraform cluster directory to include outputs from.",
    ),
    cached: bool = typer.Option(
        False,
        "--cached",
        help="Show explicitly marked last-known local state without live provider verification.",
    ),
) -> None:
    """Show NPA cluster target state from Nebius and the local cache."""

    _emit_status(
        name=name,
        output_format=output_format,
        project_id=project_id,
        project=project,
        terraform_dir=terraform_dir,
        cached=cached,
    )


@intent_boundary(OperationIntent.OBSERVE)
def list_cmd(
    output_format: str = typer.Option(
        "table", "--format", help="Output format: table or json."
    ),
    project_id: str = typer.Option(
        "", "--project-id", help="Nebius project ID. Defaults from NPA config."
    ),
    project: str = typer.Option(
        "",
        "--project",
        help="NPA project alias whose saved project_id to use (like `npa cluster up`).",
    ),
    cached: bool = typer.Option(
        False, "--cached", help="Show last-known local state only."
    ),
) -> None:
    """List NPA Workbench cluster targets known locally or in the configured project."""

    _emit_status(
        name="",
        output_format=output_format,
        project_id=project_id,
        project=project,
        cached=cached,
    )


def _emit_status(
    *,
    name: str,
    output_format: str,
    project_id: str,
    project: str = "",
    terraform_dir: Path | None = None,
    cached: bool = False,
) -> None:
    fmt = output_format.lower()
    if fmt not in {"table", "json"}:
        raise typer.BadParameter("--format must be table or json")
    try:
        resolved_project_id, configured_region = _resolve_environment_for_status(
            project_id, project
        )
        rows = _collect_rows(
            name=name,
            project_id=resolved_project_id,
            configured_region=configured_region,
            terraform_dir=terraform_dir,
            cached=cached,
            project_scoped=bool(project.strip() or project_id.strip()),
        )
        if fmt == "json":
            typer.echo(json.dumps(rows, indent=2, sort_keys=True))
        else:
            typer.echo(_format_table(rows))
        if any(
            str(row.get("verification_status") or "") == "VERIFICATION_UNAVAILABLE"
            for row in rows
        ):
            raise typer.Exit(2)
        states = {str(row.get("state") or "").upper() for row in rows}
        if states & {"PARTIAL", "DEGRADED"}:
            raise typer.Exit(3)
        if any(state.startswith("FAILED") for state in states):
            raise typer.Exit(1)
    except ClusterError as exc:
        typer.echo(f"Cluster status failed: {exc}", err=True)
        raise typer.Exit(1) from exc


def _resolve_project_for_status(
    explicit_project_id: str, project_alias: str = ""
) -> str:
    return _resolve_environment_for_status(explicit_project_id, project_alias)[0]


def _resolve_environment_for_status(
    explicit_project_id: str, project_alias: str = ""
) -> tuple[str, str]:
    configured_region = ""
    if explicit_project_id.strip():
        try:
            from npa.clients.config import resolve_environment

            saved = resolve_environment(project_alias or None)
            configured_region = str(getattr(saved, "region", "") or "")
        except Exception:  # noqa: BLE001 - explicit project id is sufficient
            logger.debug(
                "Could not resolve configured region for explicit project id",
                exc_info=True,
            )
        return explicit_project_id.strip(), configured_region
    alias = (project_alias or "").strip()
    if alias:
        # Accept the same `--project <alias>` other cluster commands (up/down)
        # take, resolving it to the saved project id before the config default.
        from npa.clients.config import resolve_environment

        try:
            saved = resolve_environment(alias)
        except Exception as exc:  # noqa: BLE001 - never inherit an unrelated default profile
            raise ClusterConfigError(
                f"configured project alias {alias!r} could not be resolved: {exc}"
            ) from exc
        resolved = str(getattr(saved, "project_id", "") or "")
        configured_region = str(getattr(saved, "region", "") or "")
        if resolved:
            return resolved, configured_region
        raise ClusterConfigError(
            f"configured project alias {alias!r} has no immutable project_id; "
            "refusing to fall back to a different default project"
        )
    try:
        resolved = resolve_project_id()
    except ClusterConfigError:
        resolved = ""
    try:
        from npa.clients.config import resolve_environment

        saved = resolve_environment(None)
        configured_region = configured_region or str(getattr(saved, "region", "") or "")
    except Exception:  # noqa: BLE001 - explicit unknown fallback below
        logger.debug("Could not resolve default configured region", exc_info=True)
    return resolved, configured_region


def _collect_rows(
    *,
    name: str,
    project_id: str,
    configured_region: str = "",
    terraform_dir: Path | None = None,
    cached: bool = False,
    project_scoped: bool = False,
) -> list[dict[str, Any]]:
    client = MK8sClient(timeout=120, poll_interval=30.0)
    local_by_name = {
        state.name: state
        for state in list_local_clusters()
        if not project_scoped or not project_id or state.project_id == project_id
    }
    remote_by_name: dict[str, ClusterInfo] = {}
    verification_errors: dict[str, str] = {}
    verified_absent: set[str] = set()
    terraform_row = _terraform_row(terraform_dir)
    if terraform_row is not None and project_scoped and project_id:
        terraform_project = str(terraform_row.get("project_id") or "")
        if terraform_project != project_id:
            # Terraform outputs/tfvars without the exact selected project are not
            # evidence about that project and must never be merged into its row.
            terraform_row = None

    if name:
        local_state = load_cluster_state(name)
        if local_state is not None and (
            not project_scoped
            or not project_id
            or local_state.project_id == project_id
        ):
            local_by_name[name] = local_state
        elif local_state is not None:
            local_state = None
        lookup_project_id = (
            local_state.project_id if local_state else ""
        ) or project_id
        if lookup_project_id and not cached:
            try:
                target_remote = client.get_cluster(
                    local_state.cluster_id if local_state else name,
                    project_id=lookup_project_id,
                )
                remote_by_name[target_remote.name or name] = target_remote
            except ClusterNotFoundError:
                verified_absent.add(name)
            except Exception as exc:  # noqa: BLE001 - preserve exact local evidence
                verification_errors[name] = f"{type(exc).__name__}: {exc}"
        target_names = [name]
        if terraform_row and terraform_row["name"] == name:
            local_by_name.setdefault(name, _state_from_terraform_row(terraform_row))
    else:
        if project_id and not cached:
            try:
                for remote_item in client.list_clusters(project_id):
                    remote_by_name[remote_item.name] = remote_item
            except Exception as exc:  # noqa: BLE001
                local_names = list(local_by_name) or ["<configured-project>"]
                for local_name in local_names:
                    verification_errors[str(local_name)] = (
                        f"{type(exc).__name__}: {exc}"
                    )
        target_names = sorted(
            set(local_by_name)
            | set(remote_by_name)
            | ({terraform_row["name"]} if terraform_row else set())
        )
        if not target_names and not project_id and terraform_row is None:
            target_names = ["<not-configured>"]
        if not target_names and verification_errors:
            target_names = sorted(verification_errors)

    rows: list[dict[str, Any]] = []
    for target_name in target_names:
        local_state = local_by_name.get(target_name)
        remote: ClusterInfo | None = remote_by_name.get(target_name)
        if (
            remote is None
            and local_state is not None
            and not cached
            and target_name not in verified_absent
        ):
            try:
                remote = client.get_cluster(
                    local_state.cluster_id,
                    project_id=local_state.project_id or project_id,
                )
            except ClusterNotFoundError:
                verified_absent.add(target_name)
            except Exception as exc:  # noqa: BLE001
                verification_errors[target_name] = f"{type(exc).__name__}: {exc}"
        row = _row_for_cluster(
            client,
            target_name,
            local_state,
            remote,
            configured_region=configured_region,
            verification_error=verification_errors.get(target_name, ""),
            cached=cached,
            verified_absent=target_name in verified_absent,
        )
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
    raw_endpoints = cluster.get("endpoints")
    endpoints: dict[str, Any] = raw_endpoints if isinstance(raw_endpoints, dict) else {}
    filesystem = outputs.get("shared_filesystem", {}).get("value") or {}
    shared_filesystem_id = (
        str(filesystem.get("id") or "") if isinstance(filesystem, dict) else ""
    )
    filesystem_csi = outputs.get("filesystem_csi", {}).get("value") or {}
    if not shared_filesystem_id:
        filesystem_csi = {}
    tfvars = _read_tfvars(terraform_dir)
    name = str(cluster.get("name"))
    region = str(tfvars.get("region") or "").strip()
    training_ref = outputs.get("k8s_training_ref")
    training_value = training_ref.get("value") if isinstance(training_ref, dict) else ""
    return {
        "name": name,
        "cluster_id": str(cluster.get("id") or ""),
        "project_id": str(tfvars.get("parent_id") or ""),
        "region": region or None,
        "region_source": "terraform_tfvars" if region else "unknown",
        "endpoint": str(endpoints.get("public_endpoint") or ""),
        "kubeconfig_path": str(kubeconfig_file(name)),
        "terraform_dir": str(terraform_dir),
        "k8s_training_ref": str(training_value or ""),
        "shared_filesystem_id": shared_filesystem_id,
        "filesystem_csi_storage_class": (
            str(filesystem_csi.get("storage_class_name") or "")
            if isinstance(filesystem_csi, dict)
            else ""
        ),
        "filesystem_csi_status": str(filesystem_csi.get("status") or "")
        if isinstance(filesystem_csi, dict)
        else "",
    }


def _state_from_terraform_row(row: dict[str, Any]) -> ClusterState:
    return ClusterState(
        name=str(row["name"]),
        cluster_id=str(row.get("cluster_id") or ""),
        project_id="",
        region=str(row.get("region") or ""),
        node_count=0,
        node_platform="",
        node_preset="",
        k8s_version="",
        subnet_id="",
        created_at="",
        endpoint=str(row.get("endpoint") or ""),
        kubeconfig_path=str(row.get("kubeconfig_path") or ""),
    )


def _merge_terraform_row(
    row: dict[str, Any], terraform_row: dict[str, Any]
) -> dict[str, Any]:
    merged = dict(row)
    for key, value in terraform_row.items():
        if key in {"region", "region_source"}:
            continue
        if key in {"name", "cluster_id", "endpoint", "kubeconfig_path"}:
            merged[key] = merged.get(key) or value
        else:
            merged[key] = value
    if not merged.get("region") and terraform_row.get("region"):
        merged["region"] = terraform_row["region"]
        merged["region_source"] = (
            terraform_row.get("region_source") or "terraform_tfvars"
        )
    return merged


def _row_for_cluster(
    client: MK8sClient,
    name: str,
    local_state: ClusterState | None,
    remote: ClusterInfo | None,
    *,
    configured_region: str = "",
    verification_error: str = "",
    cached: bool = False,
    verified_absent: bool = False,
) -> dict[str, Any]:
    node_count = local_state.node_count if local_state else 0
    expected_node_count = node_count
    node_group_id = local_state.node_group_id if local_state else ""
    node_groups: list[dict[str, Any]] = []
    provider_state = remote.status if remote is not None else ""
    if remote is not None and remote.id:
        try:
            groups = client.list_node_groups(remote.id)
        except Exception as exc:  # noqa: BLE001 - partial provider reads are unverified
            groups = []
            verification_error = verification_error or f"{type(exc).__name__}: {exc}"
        if groups:
            node_count = sum(group.node_count for group in groups)
            node_group_id = groups[0].id
            # A cluster whose CPU group is up while its GPU group never provisions
            # reports RUNNING with a quietly smaller node_count. Keep each group's
            # own state so a half-created cluster is visible.
            node_groups = [
                {
                    "name": group.name,
                    "id": group.id,
                    "state": group.status,
                    "nodes": group.node_count,
                    "platform": group.platform,
                    "preset": group.preset,
                    "reason": _node_group_reason(group.raw),
                }
                for group in groups
            ]
    state = (
        provider_state
        if remote is not None
        else "ABSENT"
        if verified_absent
        else local_state.last_seen_state
        if local_state is not None
        else "NOT_CONFIGURED"
    )
    kubeconfig_path = local_state.kubeconfig_path if local_state else ""
    kubeconfig_available = bool(
        kubeconfig_path and Path(kubeconfig_path).expanduser().is_file()
    )
    if state.upper() in {"READY", "RUNNING"}:
        if node_groups and (
            any(
                str(group.get("state") or "").upper() not in {"READY", "RUNNING"}
                for group in node_groups
            )
            or (expected_node_count > 0 and node_count < expected_node_count)
        ):
            state = "DEGRADED"
        elif not node_groups or node_count <= 0 or not kubeconfig_available:
            state = "PARTIAL"
    endpoint = (remote.endpoint if remote is not None else "") or (
        local_state.endpoint if local_state else ""
    )
    created_at = (remote.created_at if remote is not None else "") or (
        local_state.created_at if local_state else ""
    )
    remote_region = _remote_region(remote)
    local_region = local_state.region if local_state else ""
    resolved_region = remote_region or local_region or configured_region
    region_source = (
        "provider_inventory"
        if remote_region
        else "local_cluster_state"
        if local_region
        else "project_configuration"
        if configured_region
        else "unknown"
    )
    row = {
        "name": (remote.name if remote is not None and remote.name else name),
        "cluster_id": (remote.id if remote is not None else "")
        or (local_state.cluster_id if local_state else ""),
        "state": state,
        "region": resolved_region or None,
        "region_source": region_source,
        "node_count": node_count,
        "node_group_id": node_group_id,
        "endpoint": endpoint,
        "kubeconfig_path": kubeconfig_path,
        "kubeconfig_available": kubeconfig_available,
        "age": _age(created_at),
        "created_at": created_at,
        "project_id": (remote.project_id if remote is not None else "")
        or (local_state.project_id if local_state else ""),
        "node_groups": node_groups,
        "provider_state": provider_state,
        "health_state": state,
        "failure_reasons": [
            str(group.get("reason") or group.get("state") or "unknown")
            for group in node_groups
            if str(group.get("state") or "").upper() not in {"READY", "RUNNING"}
        ]
        + (["kubeconfig is unavailable"] if not kubeconfig_available and remote else [])
        + (
            ["provider reports no worker node groups"]
            if remote and not node_groups
            else []
        )
        + (
            [
                f"expected {expected_node_count} worker nodes, provider reports {node_count}"
            ]
            if remote and expected_node_count > 0 and node_count < expected_node_count
            else []
        ),
    }
    if local_state is not None and remote is not None:
        save_cluster_state(
            replace(
                local_state,
                last_seen_state=state,
                last_seen_at=datetime.now(timezone.utc)
                .replace(microsecond=0)
                .isoformat()
                .replace("+00:00", "Z"),
                node_group_id=node_group_id or local_state.node_group_id,
                node_count=node_count or local_state.node_count,
                endpoint=endpoint,
            )
        )
    from npa.verification import (
        CACHED,
        VERIFIED,
        VERIFICATION_UNAVAILABLE,
        apply_verification,
    )

    retry = f"npa cluster status --name {name}"
    if local_state and local_state.project_id:
        retry += f" --project-id {local_state.project_id}"
    verification_status = (
        CACHED
        if cached
        else VERIFICATION_UNAVAILABLE
        if verification_error
        else VERIFIED
    )
    return apply_verification(
        row,
        status=verification_status,
        target=(local_state.cluster_id if local_state else name),
        last_known_state=state,
        last_known_at=(local_state.last_seen_at if local_state else created_at),
        last_known_source=(
            "provider_api"
            if remote is not None or verified_absent
            else "local_cluster_state"
            if local_state is not None
            else "configuration"
        ),
        reason=(
            verification_error
            if verification_error
            else "live provider query intentionally skipped (--cached)"
            if cached
            else ""
        ),
        retry_command=retry,
        state_key="state",
    )


def _node_group_reason(raw: object) -> str:
    """Extract a quota/capacity failure message without assuming one API shape."""

    if not isinstance(raw, dict):
        return ""
    for container in (raw.get("status"), raw):
        if not isinstance(container, dict):
            continue
        for key in ("error_message", "status_message", "message", "reason"):
            value = str(container.get(key) or "").strip()
            if value:
                return value
    return ""


def _remote_region(remote: ClusterInfo | None) -> str:
    """Read the provider inventory's region without assuming a default."""

    if remote is None or not isinstance(remote.raw, dict):
        return ""
    raw = remote.raw
    candidates = [
        raw.get("region"),
        raw.get("region_id"),
        (raw.get("metadata") or {}).get("region")
        if isinstance(raw.get("metadata"), dict)
        else "",
        (raw.get("metadata") or {}).get("region_id")
        if isinstance(raw.get("metadata"), dict)
        else "",
        (raw.get("spec") or {}).get("region")
        if isinstance(raw.get("spec"), dict)
        else "",
        (raw.get("spec") or {}).get("region_id")
        if isinstance(raw.get("spec"), dict)
        else "",
    ]
    return next(
        (str(value).strip() for value in candidates if str(value or "").strip()), ""
    )


def _format_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No clusters found."
    preamble: list[str] = []
    for row in rows:
        if row.get("verification_status") not in {"VERIFICATION_UNAVAILABLE", "CACHED"}:
            continue
        verification = row.get("live_verification") or {}
        last_known = row.get("last_known") or {}
        preamble.append(str(row.get("verification_status")))
        preamble.append(
            f"last-known state: {last_known.get('state', 'UNKNOWN')} "
            f"(observed_at={last_known.get('observed_at') or 'unknown'}, "
            f"source={last_known.get('source') or 'unknown'})"
        )
        if verification.get("reason"):
            preamble.append(
                f"cause [{verification.get('error_code') or 'CACHED'}]: {verification.get('reason')}"
            )
        if verification.get("retry_command"):
            preamble.append(f"retry: {verification.get('retry_command')}")
    headers = [
        "NAME",
        "CLUSTER_ID",
        "STATE",
        "REGION",
        "NODES",
        "ENDPOINT",
        "KUBECONFIG",
        "AGE",
    ]
    values = [
        [
            str(row["name"]),
            str(row["cluster_id"]),
            str(row["state"]),
            str(row.get("region") or "unknown"),
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
    lines = [
        "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers))
    ]
    lines.append("  ".join("-" * width for width in widths))
    lines.extend(
        "  ".join(value[index].ljust(widths[index]) for index in range(len(headers)))
        for value in values
    )
    lines.extend(_node_group_lines(rows))
    return "\n".join([*preamble, *([""] if preamble else []), *lines])


def _node_group_lines(rows: list[dict[str, Any]]) -> list[str]:
    """Return per-node-group lines for clusters whose groups are not all running.

    A cluster is `RUNNING` as soon as its control plane is; a GPU node group that
    never provisioned (no capacity, no quota) is invisible in the cluster row.
    """
    lines: list[str] = []
    for row in rows:
        groups = row.get("node_groups") or []
        if not groups or all(
            str(group.get("state", "")).upper() == "RUNNING" for group in groups
        ):
            continue
        lines.append("")
        lines.append(f"Node groups for {row.get('name', '')} that are not RUNNING:")
        for group in groups:
            state = str(group.get("state", "") or "UNKNOWN")
            if state.upper() == "RUNNING":
                continue
            shape = " ".join(
                part
                for part in (
                    str(group.get("platform", "")),
                    str(group.get("preset", "")),
                )
                if part
            )
            lines.append(
                f"  {group.get('name', '')}: {state} ({group.get('nodes', 0)} node(s))"
                + (f" [{shape}]" if shape else "")
            )
        lines.append(
            "  A node group stuck out of RUNNING usually means the platform refused it "
            "(GPU quota or capacity). `npa cluster up` checks GPU quota before apply; "
            "`npa cluster down --force` removes a half-created cluster."
        )
    return lines


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
