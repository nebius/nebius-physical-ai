"""Check topology rejection, single-publisher dispatch and rank-state integrity."""

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from npa.workbench.flex_pi.runtime import FlexPiError
from npa.workbench.flex_pi.training_topology import training_topology, rank_command
from npa.workbench.flex_pi import training_multinode as multinode


def _allocation(rank=0):
    return {
        "NPA_FLEX_PI_NODE_COUNT": "4",
        "SKYPILOT_NUM_NODES": "4",
        "SKYPILOT_NODE_RANK": str(rank),
        "SKYPILOT_NODE_IPS": "127.0.0.1\n127.0.0.2\n127.0.0.3\n127.0.0.4",
    }


@pytest.mark.parametrize("rank", range(4))
def test_four_hosts_launch_one_distinct_global_rank_each(monkeypatch, rank):
    for name, value in _allocation(rank).items():
        monkeypatch.setenv(name, value)
    argv = rank_command("--rank-request", "/private/request.json")
    assert "--nnodes=4" in argv
    assert "--nproc-per-node=1" in argv
    assert f"--node-rank={rank}" in argv
    assert "--standalone" not in argv


@pytest.mark.parametrize(
    "changed",
    [
        {"NPA_FLEX_PI_NODE_COUNT": "2"},
        {"NPA_FLEX_PI_NODE_COUNT": "1"},
        {"SKYPILOT_NUM_NODES": "8"},
        {"SKYPILOT_NODE_RANK": "4"},
        {"SKYPILOT_NODE_RANK": "-1"},
        {"SKYPILOT_NODE_RANK": "no"},
        {"SKYPILOT_NODE_IPS": ""},
        {"SKYPILOT_NODE_IPS": "127.0.0.1 " * 4},
        {"SKYPILOT_NODE_IPS": "127.0.0.1 127.0.0.2 127.0.0.3 bad-peer"},
    ],
)
def test_inconsistent_or_unsupported_topologies_fail_before_launch(changed):
    with pytest.raises(FlexPiError):
        training_topology({**_allocation(), **changed})


def test_local_layout_retains_four_gpu_contract():
    assert training_topology({}) == {
        "nodes": 1,
        "gpus_per_node": 4,
        "node_rank": 0,
        "master_address": "127.0.0.1",
    }


class _Store:
    def __init__(self):
        self.values = {}

    def set(self, key, value):
        self.values[key] = value.encode() if isinstance(value, str) else value

    def get(self, key):
        return self.values[key]

    def check(self, keys):
        return all(key in self.values for key in keys)


def test_follower_executes_phase_then_acknowledges_leader_result(monkeypatch):
    store = _Store()
    store.set("command/1", json.dumps({"sequence": 1, "plan": {"mode": "profile"}}))
    store.set("command/2", json.dumps({"result": {"verified": True}}))
    coordinator = multinode._Coordinator(store, 2)
    seen = []
    monkeypatch.setattr(coordinator, "_execute", seen.append)
    assert coordinator.follow() == {"verified": True}
    assert seen == [{"sequence": 1, "plan": {"mode": "profile"}}]
    assert store.get("finished/2") == b"acknowledged"


def test_peer_failure_interrupts_wait_before_any_next_phase():
    store = _Store()
    store.set("failure", "training node 3 failed")
    with pytest.raises(FlexPiError, match="node 3 failed"):
        multinode._wait_keys(store, ["command/2"])


def test_dead_peer_is_detected_without_limiting_training_duration(monkeypatch):
    store = _Store()
    watch = multinode._PeerWatch(store, 0)
    watch.last_seen = {rank: (b"old", 0) for rank in range(4)}
    for rank in range(4):
        store.set(f"heartbeat/{rank}", "old" if rank == 3 else "new")
    monkeypatch.setattr(multinode.time, "monotonic", lambda: 91)
    watch._check_peers()
    assert store.get("failure") == b"training node 3 lost its heartbeat"


def test_normalization_transport_preserves_original_bytes(tmp_path):
    original = b'{\r\n"statistics": [1, 2]}\r\n'
    path = tmp_path / "node/dataset_stats.json"
    plan = {
        "normalization_file": str(path),
        "execution": {"normalization_sha256": hashlib.sha256(original).hexdigest()},
    }
    coordinator = multinode._Coordinator(_Store(), 1)
    coordinator._prepare_inputs(plan, original.hex())
    assert path.read_bytes() == original
    with pytest.raises(FlexPiError, match="normalization differs"):
        coordinator._prepare_inputs(plan, b"tampered".hex())
    assert path.read_bytes() == original


def test_follower_restores_complete_checkpoint_once_per_manifest(tmp_path, monkeypatch):
    restored = []
    monkeypatch.setattr(
        "npa.clients.storage.StorageClient.from_environment", lambda: object()
    )
    monkeypatch.setattr(
        "npa.workbench.flex_pi.training_artifacts.restore_checkpoint",
        lambda *args: restored.append(args),
    )
    original = b"{}"
    checkpoint = {
        "source": "s3://example-bucket/run/checkpoint",
        "manifest": {"files": []},
        "directory": str(tmp_path / "restore"),
    }
    plan = {
        "distributed_checkpoint": checkpoint,
        "normalization_file": str(tmp_path / "dataset_stats.json"),
        "execution": {"normalization_sha256": hashlib.sha256(original).hexdigest()},
    }
    coordinator = multinode._Coordinator(_Store(), 1)
    coordinator._prepare_inputs(plan, original.hex())
    coordinator._prepare_inputs(plan, original.hex())
    assert len(restored) == 1
    assert restored[0][:3] == (
        checkpoint["manifest"],
        checkpoint["source"],
        Path(checkpoint["directory"]),
    )


@pytest.mark.parametrize("missing", [False, True])
def test_checkpoint_collects_all_saved_rng_files_and_rejects_missing_rank(
    tmp_path, monkeypatch, missing
):
    from npa.workbench.flex_pi.training_checkpoint import collect_rank_checkpoint

    peers = [
        {
            "rank": rank,
            "rng": f"rng-{rank}".encode(),
            "cursor": b'{"step":30}' if rank == 0 else None,
        }
        for rank in range(4)
    ]
    if missing:
        peers[3]["rng"] = None
    (tmp_path / "random_states_0.pkl").write_bytes(b"rng-0")
    (tmp_path / "trainer_state.json").write_bytes(b'{"step":30}')

    def gather(output, payload):
        assert payload == peers[0]
        output[:] = peers

    distributed = SimpleNamespace(get_rank=lambda: 0, all_gather_object=gather)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(distributed=distributed))
    monkeypatch.setitem(sys.modules, "torch.distributed", distributed)
    monkeypatch.setattr(
        "npa.workbench.flex_pi.training_checkpoint.training_topology",
        lambda: {"nodes": 4},
    )
    if missing:
        with pytest.raises(RuntimeError, match="rank RNG"):
            collect_rank_checkpoint(tmp_path)
        assert not (tmp_path / "random_states_1.pkl").exists()
    else:
        collect_rank_checkpoint(tmp_path)
        assert [
            (tmp_path / f"random_states_{rank}.pkl").read_bytes() for rank in range(4)
        ] == [row["rng"] for row in peers]
