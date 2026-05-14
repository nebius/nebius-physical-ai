"""Nebius Managed Kubernetes CLI wrapper."""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from npa.cluster.config import ClusterConfig
from npa.cluster.exceptions import ClusterError, ClusterNotFoundError, ClusterTimeoutError

SubprocessRunner = Callable[..., subprocess.CompletedProcess[str]]

READY_STATES = {"READY"}
ERROR_STATES = {"ERROR", "FAILED"}


@dataclass
class NodeGroupInfo:
    id: str
    name: str
    cluster_id: str
    status: str = "UNKNOWN"
    node_count: int = 0
    raw: dict[str, Any] | None = None


@dataclass
class ClusterInfo:
    id: str
    name: str
    project_id: str
    status: str = "UNKNOWN"
    created_at: str = ""
    endpoint: str = ""
    node_count: int = 0
    node_group_id: str = ""
    raw: dict[str, Any] | None = None


class MK8sClient:
    """Small subprocess client for ``nebius mk8s`` commands."""

    def __init__(
        self,
        *,
        nebius_bin: str | None = None,
        subprocess_runner: SubprocessRunner | None = None,
        timeout: int = 900,
        retries: int = 3,
        poll_interval: float = 30.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._nebius_bin = nebius_bin or shutil.which("nebius") or "nebius"
        self._runner = subprocess_runner or subprocess.run
        self._timeout = timeout
        self._retries = max(1, retries)
        self._poll_interval = poll_interval
        self._sleep = sleep

    def create_cluster(self, config: ClusterConfig) -> ClusterInfo:
        if not config.project_id:
            raise ValueError("project_id is required")
        if not config.subnet_id:
            raise ValueError("subnet_id is required")

        created_cluster_id = ""
        try:
            result = self._run(
                [
                    "mk8s",
                    "cluster",
                    "create",
                    "--parent-id",
                    config.project_id,
                    "--name",
                    config.name,
                    "--control-plane-subnet-id",
                    config.subnet_id,
                    "--control-plane-endpoints-public-endpoint",
                    "true",
                    "--control-plane-etcd-cluster-size",
                    "1",
                    "--control-plane-version",
                    config.k8s_version,
                    "--format",
                    "json",
                ],
                timeout=self._timeout,
            )
            if result.returncode != 0:
                self._raise_for_error(result, f"create cluster failed for {config.name}")
            cluster = self._parse_cluster(result.stdout, fallback_project_id=config.project_id)
            if not cluster.id:
                cluster = self.get_cluster(config.name, project_id=config.project_id)
            created_cluster_id = cluster.id
            node_group = self.create_node_group(config, cluster.id)
            cluster.node_group_id = node_group.id
            cluster.node_count = node_group.node_count
            return cluster
        except Exception:
            if created_cluster_id:
                try:
                    self.delete_cluster(created_cluster_id, project_id=config.project_id)
                except ClusterError:
                    pass
            raise

    def create_node_group(self, config: ClusterConfig, cluster_id: str) -> NodeGroupInfo:
        network_interface: dict[str, Any] = {"subnet_id": config.subnet_id}
        if config.public_node_ip:
            network_interface["public_ip_address"] = {}
        result = self._run(
            [
                "mk8s",
                "node-group",
                "create",
                "--parent-id",
                cluster_id,
                "--name",
                f"{config.name}-cpu",
                "--version",
                config.k8s_version,
                "--fixed-node-count",
                str(config.node_count),
                "--template-resources-platform",
                config.node_platform,
                "--template-resources-preset",
                config.node_preset,
                "--template-boot-disk-type",
                config.boot_disk_type,
                "--template-boot-disk-size-gibibytes",
                str(config.boot_disk_size_gib),
                "--template-network-interfaces",
                json.dumps([network_interface], separators=(",", ":")),
                "--format",
                "json",
            ],
            timeout=self._timeout,
        )
        if result.returncode != 0:
            self._raise_for_error(result, f"create node group failed for {config.name}")
        node_group = self._parse_node_group(result.stdout, cluster_id=cluster_id)
        if not node_group.id:
            groups = self.list_node_groups(cluster_id)
            for group in groups:
                if group.name == f"{config.name}-cpu":
                    return group
        return node_group

    def list_clusters(self, project_id: str) -> list[ClusterInfo]:
        result = self._run(
            ["mk8s", "cluster", "list", "--parent-id", project_id, "--format", "json"],
            timeout=120,
        )
        if result.returncode != 0:
            self._raise_for_error(result, f"list clusters failed for project {project_id}")
        return [
            self._cluster_from_dict(item, fallback_project_id=project_id)
            for item in _as_items(_json_loads(result.stdout))
        ]

    def get_cluster(self, cluster_id_or_name: str, *, project_id: str = "") -> ClusterInfo:
        result = self._run(
            ["mk8s", "cluster", "get", "--id", cluster_id_or_name, "--format", "json"],
            timeout=120,
        )
        if result.returncode == 0:
            return self._parse_cluster(result.stdout, fallback_project_id=project_id)
        if project_id:
            by_name = self._run(
                [
                    "mk8s",
                    "cluster",
                    "get-by-name",
                    "--parent-id",
                    project_id,
                    "--name",
                    cluster_id_or_name,
                    "--format",
                    "json",
                ],
                timeout=120,
            )
            if by_name.returncode == 0:
                return self._parse_cluster(by_name.stdout, fallback_project_id=project_id)
            if self._is_not_found(by_name):
                raise ClusterNotFoundError(f"Cluster {cluster_id_or_name} not found in project {project_id}")
            self._raise_for_error(by_name, f"get cluster failed for {cluster_id_or_name}")
        if self._is_not_found(result):
            raise ClusterNotFoundError(f"Cluster {cluster_id_or_name} not found")
        self._raise_for_error(result, f"get cluster failed for {cluster_id_or_name}")
        raise AssertionError("unreachable")

    def delete_cluster(self, cluster_id_or_name: str, *, project_id: str = "") -> None:
        try:
            cluster = self.get_cluster(cluster_id_or_name, project_id=project_id)
        except ClusterNotFoundError:
            return
        result = self._run(["mk8s", "cluster", "delete", "--id", cluster.id], timeout=self._timeout)
        if result.returncode != 0 and not self._is_not_found(result):
            self._raise_for_error(result, f"delete cluster failed for {cluster_id_or_name}")

    def list_node_groups(self, cluster_id: str) -> list[NodeGroupInfo]:
        result = self._run(
            ["mk8s", "node-group", "list", "--parent-id", cluster_id, "--format", "json"],
            timeout=120,
        )
        if result.returncode != 0:
            if self._is_not_found(result):
                return []
            self._raise_for_error(result, f"list node groups failed for cluster {cluster_id}")
        return [
            self._node_group_from_dict(item, cluster_id=cluster_id)
            for item in _as_items(_json_loads(result.stdout))
        ]

    def get_kubeconfig(
        self,
        cluster_id: str,
        kubeconfig_path: Path,
        *,
        context_name: str = "",
        external: bool = True,
    ) -> Path:
        kubeconfig_path.parent.mkdir(parents=True, exist_ok=True)
        args = [
            "mk8s",
            "cluster",
            "get-credentials",
            "--id",
            cluster_id,
            "--force",
            "--kubeconfig",
            str(kubeconfig_path),
        ]
        args.append("--external" if external else "--internal")
        if context_name:
            args.extend(["--context-name", context_name])
        result = self._run(args, timeout=120)
        if result.returncode != 0:
            self._raise_for_error(result, f"get kubeconfig failed for cluster {cluster_id}")
        return kubeconfig_path

    def wait_for_ready(
        self,
        cluster_id: str,
        *,
        project_id: str = "",
        expected_node_count: int = 1,
        timeout_minutes: int = 30,
        on_state_change: Callable[[ClusterInfo, list[NodeGroupInfo]], None] | None = None,
    ) -> ClusterInfo:
        deadline = time.monotonic() + timeout_minutes * 60
        last_state = ""
        last_cluster: ClusterInfo | None = None
        last_groups: list[NodeGroupInfo] = []
        while time.monotonic() <= deadline:
            cluster = self.get_cluster(cluster_id, project_id=project_id)
            groups = self.list_node_groups(cluster.id)
            state_key = f"{cluster.status}:{','.join(group.status for group in groups)}"
            if state_key != last_state and on_state_change is not None:
                on_state_change(cluster, groups)
            last_state = state_key
            last_cluster = cluster
            last_groups = groups
            if cluster.status in ERROR_STATES:
                raise ClusterError(f"Cluster {cluster.id} entered terminal state {cluster.status}")
            if any(group.status in ERROR_STATES for group in groups):
                failed = ", ".join(f"{group.name}:{group.status}" for group in groups)
                raise ClusterError(f"Cluster {cluster.id} has failed node group state: {failed}")
            ready_nodes = sum(group.node_count for group in groups if group.status in READY_STATES)
            if cluster.status in READY_STATES and groups and ready_nodes >= expected_node_count:
                cluster.node_count = ready_nodes
                cluster.node_group_id = groups[0].id
                return cluster
            self._sleep(self._poll_interval)
        status = last_cluster.status if last_cluster else "UNKNOWN"
        group_status = ", ".join(group.status for group in last_groups) or "none"
        raise ClusterTimeoutError(
            f"Cluster {cluster_id} did not become READY within {timeout_minutes} minutes "
            f"(cluster={status}, node_groups={group_status})"
        )

    def wait_for_deleted(
        self,
        cluster_id_or_name: str,
        *,
        project_id: str = "",
        timeout_minutes: int = 30,
    ) -> None:
        deadline = time.monotonic() + timeout_minutes * 60
        while time.monotonic() <= deadline:
            try:
                self.get_cluster(cluster_id_or_name, project_id=project_id)
            except ClusterNotFoundError:
                return
            self._sleep(self._poll_interval)
        raise ClusterTimeoutError(
            f"Cluster {cluster_id_or_name} was not deleted within {timeout_minutes} minutes"
        )

    def _parse_cluster(self, raw: str, *, fallback_project_id: str = "") -> ClusterInfo:
        items = _as_items(_json_loads(raw))
        if not items:
            return ClusterInfo(id="", name="", project_id=fallback_project_id, raw={})
        return self._cluster_from_dict(items[0], fallback_project_id=fallback_project_id)

    def _parse_node_group(self, raw: str, *, cluster_id: str) -> NodeGroupInfo:
        items = _as_items(_json_loads(raw))
        if not items:
            return NodeGroupInfo(id="", name="", cluster_id=cluster_id, raw={})
        return self._node_group_from_dict(items[0], cluster_id=cluster_id)

    def _cluster_from_dict(self, data: dict[str, Any], *, fallback_project_id: str = "") -> ClusterInfo:
        metadata = data.get("metadata") if isinstance(data, dict) else {}
        return ClusterInfo(
            id=str((metadata or {}).get("id") or data.get("id") or ""),
            name=str((metadata or {}).get("name") or data.get("name") or ""),
            project_id=str((metadata or {}).get("parent_id") or data.get("parent_id") or fallback_project_id),
            status=_normalize_state(_deep_get(data, ("status", "state"), ("state",))),
            created_at=str((metadata or {}).get("created_at") or data.get("created_at") or ""),
            endpoint=_endpoint_from_cluster(data),
            raw=data,
        )

    def _node_group_from_dict(self, data: dict[str, Any], *, cluster_id: str) -> NodeGroupInfo:
        metadata = data.get("metadata") if isinstance(data, dict) else {}
        node_count = _int_value(
            _deep_get(
                data,
                ("spec", "fixed_node_count"),
                ("spec", "fixedNodeCount"),
                ("status", "node_count"),
                ("status", "nodeCount"),
            )
        )
        return NodeGroupInfo(
            id=str((metadata or {}).get("id") or data.get("id") or ""),
            name=str((metadata or {}).get("name") or data.get("name") or ""),
            cluster_id=cluster_id,
            status=_normalize_state(_deep_get(data, ("status", "state"), ("state",))),
            node_count=node_count,
            raw=data,
        )

    def _run(
        self,
        args: Sequence[str],
        *,
        timeout: int | None = None,
    ) -> subprocess.CompletedProcess[str]:
        full_args = [self._nebius_bin, *args]
        last_result: subprocess.CompletedProcess[str] | None = None
        for attempt in range(1, self._retries + 1):
            try:
                result = self._runner(
                    full_args,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                    timeout=timeout or self._timeout,
                )
            except subprocess.TimeoutExpired as exc:
                if attempt >= self._retries:
                    raise ClusterError(
                        f"Nebius CLI timed out: {shlex.join(full_args)}"
                    ) from exc
                self._sleep(min(2 ** attempt, 10))
                continue
            last_result = result
            if result.returncode == 0 or not self._is_transient(result) or attempt >= self._retries:
                return result
            self._sleep(min(2 ** attempt, 10))
        if last_result is not None:
            return last_result
        raise ClusterError(f"Unable to run Nebius CLI: {shlex.join(full_args)}")

    def _raise_for_error(self, result: subprocess.CompletedProcess[str], prefix: str) -> None:
        if self._is_not_found(result):
            raise ClusterNotFoundError(prefix)
        detail = (result.stderr or result.stdout or "").strip()
        suffix = f": {detail}" if detail else f" (exit code {result.returncode})"
        raise ClusterError(f"{prefix}{suffix}")

    @staticmethod
    def _is_not_found(result: subprocess.CompletedProcess[str]) -> bool:
        detail = f"{result.stderr}\n{result.stdout}".lower()
        return "not found" in detail or "notfound" in detail

    @staticmethod
    def _is_transient(result: subprocess.CompletedProcess[str]) -> bool:
        detail = f"{result.stderr}\n{result.stdout}".lower()
        transient_markers = (
            "temporarily",
            "timeout",
            "deadline",
            "unavailable",
            "too many requests",
            "connection reset",
            "try again",
            "429",
            "500",
            "502",
            "503",
            "504",
        )
        return any(marker in detail for marker in transient_markers)


def _json_loads(raw: str) -> Any:
    if not raw.strip():
        return {}
    return json.loads(raw)


def _as_items(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict):
        items = data.get("items")
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
        if any(key in data for key in ("metadata", "status", "spec", "id")):
            return [data]
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    return []


def _deep_get(data: dict[str, Any], *paths: tuple[str, ...]) -> Any:
    for path in paths:
        current: Any = data
        for key in path:
            if not isinstance(current, dict) or key not in current:
                current = None
                break
            current = current[key]
        if current not in (None, ""):
            return current
    return ""


def _normalize_state(value: Any) -> str:
    text = str(value or "").strip()
    return text.upper() if text else "UNKNOWN"


def _int_value(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _endpoint_from_cluster(data: dict[str, Any]) -> str:
    endpoint = _deep_get(
        data,
        ("status", "control_plane", "endpoints", "public_endpoint"),
        ("status", "controlPlane", "endpoints", "publicEndpoint"),
        ("status", "endpoint"),
        ("spec", "control_plane", "endpoints", "public_endpoint"),
        ("spec", "controlPlane", "endpoints", "publicEndpoint"),
    )
    if isinstance(endpoint, dict):
        for key in ("address", "url", "endpoint", "ip_address", "ipAddress"):
            value = endpoint.get(key)
            if value:
                return str(value)
        return ""
    return str(endpoint or "")
