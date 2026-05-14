"""Local state files for ``npa cluster``."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from npa.cluster.exceptions import ClusterStateError

CLUSTERS_DIR = Path.home() / ".npa" / "clusters"


@dataclass
class ClusterState:
    name: str
    cluster_id: str
    project_id: str
    region: str
    node_count: int
    node_platform: str
    node_preset: str
    k8s_version: str
    subnet_id: str
    created_at: str
    last_seen_state: str = "UNKNOWN"
    node_group_id: str = ""
    endpoint: str = ""
    kubeconfig_path: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ClusterState":
        try:
            return cls(
                name=str(data["name"]),
                cluster_id=str(data["cluster_id"]),
                project_id=str(data["project_id"]),
                region=str(data["region"]),
                node_count=int(data["node_count"]),
                node_platform=str(data["node_platform"]),
                node_preset=str(data["node_preset"]),
                k8s_version=str(data["k8s_version"]),
                subnet_id=str(data.get("subnet_id", "")),
                created_at=str(data["created_at"]),
                last_seen_state=str(data.get("last_seen_state", "UNKNOWN")),
                node_group_id=str(data.get("node_group_id", "")),
                endpoint=str(data.get("endpoint", "")),
                kubeconfig_path=str(data.get("kubeconfig_path", "")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ClusterStateError(f"Malformed cluster state: {exc}") from exc


@dataclass
class NodeGroupState:
    cluster_name: str
    name: str
    node_group_id: str
    gpu_type: str
    platform: str
    preset: str
    node_count: int
    created_at: str
    last_seen_state: str = "UNKNOWN"
    public_ip: bool = False
    autoscaling_min: int | None = None
    autoscaling_max: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NodeGroupState":
        try:
            return cls(
                cluster_name=str(data["cluster_name"]),
                name=str(data["name"]),
                node_group_id=str(data["node_group_id"]),
                gpu_type=str(data["gpu_type"]),
                platform=str(data["platform"]),
                preset=str(data["preset"]),
                node_count=int(data["node_count"]),
                created_at=str(data["created_at"]),
                last_seen_state=str(data.get("last_seen_state", "UNKNOWN")),
                public_ip=bool(data.get("public_ip", False)),
                autoscaling_min=_optional_int(data.get("autoscaling_min")),
                autoscaling_max=_optional_int(data.get("autoscaling_max")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ClusterStateError(f"Malformed node-group state: {exc}") from exc


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def cluster_dir(name: str, *, base_dir: Path | None = None) -> Path:
    return (base_dir or CLUSTERS_DIR) / name


def state_file(name: str, *, base_dir: Path | None = None) -> Path:
    return cluster_dir(name, base_dir=base_dir) / "cluster.json"


def metadata_file(name: str, *, base_dir: Path | None = None) -> Path:
    return cluster_dir(name, base_dir=base_dir) / "metadata.json"


def kubeconfig_file(name: str, *, base_dir: Path | None = None) -> Path:
    return cluster_dir(name, base_dir=base_dir) / "kubeconfig"


def node_groups_dir(cluster_name: str, *, base_dir: Path | None = None) -> Path:
    return cluster_dir(cluster_name, base_dir=base_dir) / "node-groups"


def node_group_state_file(
    cluster_name: str,
    node_group_name: str,
    *,
    base_dir: Path | None = None,
) -> Path:
    return node_groups_dir(cluster_name, base_dir=base_dir) / f"{node_group_name}.json"


def save_cluster_state(
    cluster_state: ClusterState,
    *,
    base_dir: Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    directory = cluster_dir(cluster_state.name, base_dir=base_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = state_file(cluster_state.name, base_dir=base_dir)
    path.write_text(json.dumps(asdict(cluster_state), indent=2, sort_keys=True) + "\n")
    if metadata is not None:
        metadata_file(cluster_state.name, base_dir=base_dir).write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
    return path


def load_cluster_state(name: str, *, base_dir: Path | None = None) -> ClusterState | None:
    path = state_file(name, base_dir=base_dir)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ClusterStateError(f"Malformed cluster state {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ClusterStateError(f"Malformed cluster state {path}: expected object")
    return ClusterState.from_dict(data)


def list_local_clusters(*, base_dir: Path | None = None) -> list[ClusterState]:
    root = base_dir or CLUSTERS_DIR
    if not root.exists():
        return []
    clusters: list[ClusterState] = []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        state = load_cluster_state(path.name, base_dir=root)
        if state is not None:
            clusters.append(state)
    return clusters


def delete_cluster_state(name: str, *, base_dir: Path | None = None) -> None:
    directory = cluster_dir(name, base_dir=base_dir)
    if directory.exists():
        shutil.rmtree(directory)


def save_node_group_state(
    node_group_state: NodeGroupState,
    *,
    base_dir: Path | None = None,
) -> Path:
    directory = node_groups_dir(node_group_state.cluster_name, base_dir=base_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = node_group_state_file(
        node_group_state.cluster_name,
        node_group_state.name,
        base_dir=base_dir,
    )
    path.write_text(json.dumps(asdict(node_group_state), indent=2, sort_keys=True) + "\n")
    return path


def load_node_group_state(
    cluster_name: str,
    node_group_name: str,
    *,
    base_dir: Path | None = None,
) -> NodeGroupState | None:
    path = node_group_state_file(cluster_name, node_group_name, base_dir=base_dir)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ClusterStateError(f"Malformed node-group state {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ClusterStateError(f"Malformed node-group state {path}: expected object")
    return NodeGroupState.from_dict(data)


def list_node_group_states(
    cluster_name: str,
    *,
    base_dir: Path | None = None,
) -> list[NodeGroupState]:
    directory = node_groups_dir(cluster_name, base_dir=base_dir)
    if not directory.exists():
        return []
    states: list[NodeGroupState] = []
    for path in sorted(directory.glob("*.json")):
        state = load_node_group_state(cluster_name, path.stem, base_dir=base_dir)
        if state is not None:
            states.append(state)
    return states


def delete_node_group_state(
    cluster_name: str,
    node_group_name: str,
    *,
    base_dir: Path | None = None,
) -> None:
    path = node_group_state_file(cluster_name, node_group_name, base_dir=base_dir)
    if path.exists():
        path.unlink()
    directory = node_groups_dir(cluster_name, base_dir=base_dir)
    if directory.exists() and not any(directory.iterdir()):
        directory.rmdir()


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)
