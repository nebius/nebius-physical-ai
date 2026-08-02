"""Unit coverage for the multi-node probe stages (storage injected, no infrastructure)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from npa.workflows.multi_node_probe import (
    SCHEMA_NODE,
    SCHEMA_VERIFY,
    MultiNodeProbeError,
    node_report,
    report_node,
    summarize,
    verify_nodes,
)


def _env(rank: int, nodes: int = 2) -> dict[str, str]:
    return {
        "SKYPILOT_NODE_RANK": str(rank),
        "SKYPILOT_NUM_NODES": str(nodes),
        "SKYPILOT_NUM_GPUS_PER_NODE": "0",
        "SKYPILOT_NODE_IPS": "\n".join(f"10.0.0.{index + 1}" for index in range(nodes)),
    }


class _FakeStorage:
    def __init__(self, files: dict[str, bytes] | None = None) -> None:
        self.files = files or {}
        self.uploads: list[tuple[str, str]] = []

    def upload_file(self, local: str, uri: str) -> str:
        self.uploads.append((local, uri))
        self.files[uri] = Path(local).read_bytes()
        return uri

    def download_directory(self, uri: str, local_dir: str) -> str:
        root = Path(local_dir)
        root.mkdir(parents=True, exist_ok=True)
        prefix = uri.rstrip("/") + "/"
        for key, body in self.files.items():
            if key.startswith(prefix):
                dest = root / key[len(prefix) :]
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(body)
        return str(root)


def test_node_report_reads_the_skypilot_environment() -> None:
    report = node_report(env=_env(rank=1, nodes=3))

    assert report["schema"] == SCHEMA_NODE
    assert report["rank"] == 1
    assert report["num_nodes"] == 3
    assert report["node_ip_count"] == 3
    assert report["hostname"]


def test_node_report_without_a_rank_is_an_error() -> None:
    with pytest.raises(MultiNodeProbeError, match="SKYPILOT_NODE_RANK is unset"):
        node_report(env={})


def test_report_node_writes_a_rank_scoped_object(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in _env(rank=0).items():
        monkeypatch.setenv(key, value)
    storage = _FakeStorage()

    uri = report_node("s3://bucket/run/nodes/", client=storage)

    assert uri == "s3://bucket/run/nodes/rank-0.json"
    payload = json.loads(storage.files[uri].decode())
    assert payload["rank"] == 0 and payload["schema"] == SCHEMA_NODE


def test_summarize_accepts_a_complete_gang() -> None:
    reports = [
        {"rank": 0, "hostname": "pod-a"},
        {"rank": 1, "hostname": "pod-b"},
    ]

    summary = summarize(reports, expected_nodes=2)

    assert summary["schema"] == SCHEMA_VERIFY
    assert summary["ranks"] == [0, 1]
    assert summary["reported_nodes"] == 2


def test_summarize_rejects_a_missing_rank() -> None:
    reports = [{"rank": 0, "hostname": "pod-a"}]

    with pytest.raises(MultiNodeProbeError, match="one report per rank"):
        summarize(reports, expected_nodes=2)


def test_summarize_rejects_ranks_that_shared_a_host() -> None:
    """Two ranks on one host would mean the gang was not really multi-node."""

    reports = [
        {"rank": 0, "hostname": "pod-a"},
        {"rank": 1, "hostname": "pod-a"},
    ]

    with pytest.raises(MultiNodeProbeError, match="distinct hostnames"):
        summarize(reports, expected_nodes=2)


def test_summarize_rejects_a_bad_expectation() -> None:
    with pytest.raises(MultiNodeProbeError, match="expected_nodes must be >= 1"):
        summarize([{"rank": 0, "hostname": "a"}], expected_nodes=0)


def test_verify_nodes_round_trips_through_storage() -> None:
    storage = _FakeStorage(
        {
            "s3://bucket/run/nodes/rank-0.json": json.dumps(
                {"schema": SCHEMA_NODE, "rank": 0, "hostname": "pod-a"}
            ).encode(),
            "s3://bucket/run/nodes/rank-1.json": json.dumps(
                {"schema": SCHEMA_NODE, "rank": 1, "hostname": "pod-b"}
            ).encode(),
        }
    )

    summary = verify_nodes(
        "s3://bucket/run/nodes/", "s3://bucket/run/report/", 2, client=storage
    )

    assert summary["ranks"] == [0, 1]
    assert summary["written_uri"] == "s3://bucket/run/report/multi_node_report.json"
    assert "s3://bucket/run/report/multi_node_report.json" in storage.files


def test_verify_nodes_fails_when_nothing_was_written() -> None:
    with pytest.raises(MultiNodeProbeError, match="no node reports found"):
        verify_nodes("s3://bucket/run/nodes/", "s3://bucket/run/report/", 2, client=_FakeStorage())


def test_shipped_spec_node_count_matches_its_resource_profile() -> None:
    """`--var node_count=N` and `resources.gang.num_nodes` must not drift apart."""

    from npa.orchestration.npa_workflow.blueprints import resolve_npa_workflow_spec
    from npa.orchestration.npa_workflow.spec import load_spec

    path = resolve_npa_workflow_spec("multi-node-probe.yaml")
    assert path is not None
    spec = load_spec(path)

    assert int(spec.config["node_count"]) == int(spec.resources["gang"]["num_nodes"])
    assert int(spec.resources["gang"]["num_nodes"]) >= 2, (
        "a 1-node gang would prove nothing"
    )
