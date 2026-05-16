from __future__ import annotations

import json

import pytest

from npa.cluster.exceptions import ClusterStateError
from npa.cluster.state import (
    NodeGroupState,
    delete_node_group_state,
    list_node_group_states,
    load_node_group_state,
    save_node_group_state,
)


def _state(name: str = "cluster-a-h100-gpu") -> NodeGroupState:
    return NodeGroupState(
        cluster_name="cluster-a",
        name=name,
        node_group_id="mk8snodegroup-a",
        gpu_type="h100",
        platform="gpu-h100-sxm",
        preset="1gpu-16vcpu-200gb",
        node_count=1,
        created_at="2026-05-14T22:46:30Z",
        last_seen_state="RUNNING",
        public_ip=False,
    )


def test_node_group_state_roundtrip(tmp_path) -> None:
    saved = save_node_group_state(_state(), base_dir=tmp_path)

    assert saved == tmp_path / "cluster-a" / "node-groups" / "cluster-a-h100-gpu.json"
    assert load_node_group_state("cluster-a", "cluster-a-h100-gpu", base_dir=tmp_path) == _state()


def test_missing_node_group_state_returns_none(tmp_path) -> None:
    assert load_node_group_state("cluster-a", "missing", base_dir=tmp_path) is None


def test_list_node_group_states(tmp_path) -> None:
    save_node_group_state(_state("cluster-a-h200-gpu"), base_dir=tmp_path)
    save_node_group_state(_state("cluster-a-h100-gpu"), base_dir=tmp_path)

    assert [state.name for state in list_node_group_states("cluster-a", base_dir=tmp_path)] == [
        "cluster-a-h100-gpu",
        "cluster-a-h200-gpu",
    ]


def test_delete_node_group_state_removes_empty_directory(tmp_path) -> None:
    save_node_group_state(_state(), base_dir=tmp_path)

    delete_node_group_state("cluster-a", "cluster-a-h100-gpu", base_dir=tmp_path)

    assert load_node_group_state("cluster-a", "cluster-a-h100-gpu", base_dir=tmp_path) is None
    assert not (tmp_path / "cluster-a" / "node-groups").exists()


def test_corrupted_node_group_state_raises(tmp_path) -> None:
    path = tmp_path / "cluster-a" / "node-groups"
    path.mkdir(parents=True)
    (path / "cluster-a-h100-gpu.json").write_text("{bad json")

    with pytest.raises(ClusterStateError):
        load_node_group_state("cluster-a", "cluster-a-h100-gpu", base_dir=tmp_path)


def test_malformed_node_group_state_raises(tmp_path) -> None:
    path = tmp_path / "cluster-a" / "node-groups"
    path.mkdir(parents=True)
    (path / "cluster-a-h100-gpu.json").write_text(json.dumps({"name": "missing-required"}))

    with pytest.raises(ClusterStateError):
        load_node_group_state("cluster-a", "cluster-a-h100-gpu", base_dir=tmp_path)
