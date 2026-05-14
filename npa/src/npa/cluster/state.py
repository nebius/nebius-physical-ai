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
