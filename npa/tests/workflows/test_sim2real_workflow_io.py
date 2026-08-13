from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from npa.workflows.sim2real import workflow_io
from npa.workflows.sim2real import workflow_stage


SOURCE_SHA = "1" * 40
IMAGE = "cr.example/npa/runtime@sha256:" + "2" * 64


def test_source_sha_requires_workflow_and_image_attestations_to_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NPA_IMAGE_SOURCE_SHA", SOURCE_SHA)
    monkeypatch.setenv("NPA_SIM2REAL_SOURCE_SHA", SOURCE_SHA)
    assert workflow_io.source_sha() == SOURCE_SHA

    monkeypatch.setenv("NPA_SIM2REAL_SOURCE_SHA", "3" * 40)
    with pytest.raises(RuntimeError, match="does not match"):
        workflow_io.source_sha()


def test_component_records_are_content_addressed_and_stage12_is_a_seam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uploads: dict[str, dict[str, object]] = {}

    class FakeStorage:
        def upload_file(self, local_file: str, uri: str) -> str:
            uploads[uri] = json.loads(Path(local_file).read_text())
            return uri

    monkeypatch.setattr(workflow_io, "storage", lambda: FakeStorage())
    monkeypatch.setenv("NPA_IMAGE_SOURCE_SHA", SOURCE_SHA)
    monkeypatch.setenv("NPA_SIM2REAL_SOURCE_SHA", SOURCE_SHA)
    monkeypatch.setenv("NPA_TASK_IMAGE", IMAGE)

    record = workflow_io.publish_component_record(
        root_uri="s3://bucket/run",
        stage=1,
        name="stage_01_trigger",
        tier="WORKS",
        evidence="validated",
        artifacts={"result": "s3://bucket/run/result.json"},
    )
    history_uri = (
        f"s3://bucket/run/components/history/stage_01/{record['content_sha256']}.json"
    )
    assert uploads[history_uri] == record
    assert uploads["s3://bucket/run/components/stage_01.json"] == record
    assert record["artifacts"]["image"] == IMAGE

    seam = workflow_io.publish_component_record(
        root_uri="s3://bucket/run",
        stage=12,
        name="stage_12_external_validation",
        tier="SEAM",
        evidence="external",
        artifacts={"seam": "s3://bucket/run/seam.json"},
    )
    assert seam["tier"] == "SEAM"
    assert "image" not in seam["artifacts"]

    with pytest.raises(ValueError, match="must remain"):
        workflow_io.publish_component_record(
            root_uri="s3://bucket/run",
            stage=12,
            name="bad",
            tier="WORKS",
            evidence="bad",
            artifacts={},
        )


def test_parallel_lane_records_preserve_distinct_execution_owners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uploads: dict[str, dict[str, object]] = {}

    class FakeStorage:
        def upload_file(self, local_file: str, uri: str) -> str:
            uploads[uri] = json.loads(Path(local_file).read_text())
            return uri

    provenances = [
        {
            "image": IMAGE,
            "source_sha": SOURCE_SHA,
            "workflow_job": f"managed-job-{index}",
            "gpu_products": ["NVIDIA RTX PRO 6000"],
        }
        for index in range(2)
    ]
    monkeypatch.setattr(workflow_io, "storage", lambda: FakeStorage())
    records = [
        workflow_io.publish_component_lane_record(
            root_uri="s3://bucket/run",
            stage=4,
            lane=f"shard-{index:05d}",
            evidence="generated declared shard",
            artifacts={"shard_index": index},
            execution_provenance=provenance,
        )
        for index, provenance in enumerate(provenances)
    ]
    assert [record["artifacts"]["workflow_job"] for record in records] == [
        "managed-job-0",
        "managed-job-1",
    ]
    assert all(
        f"s3://bucket/run/components/lanes/stage_04/shard-{index:05d}.json" in uploads
        for index in range(2)
    )

    joined = workflow_io.aggregate_parallel_provenance(provenances, stage=4)
    assert joined["workflow_jobs"] == ["managed-job-0", "managed-job-1"]
    assert joined["lane_count"] == 2

    conflicting = [dict(item) for item in provenances]
    conflicting[1]["workflow_job"] = "managed-job-0"
    with pytest.raises(ValueError, match="incomplete"):
        workflow_io.aggregate_parallel_provenance(conflicting, stage=4)


def test_reduced_proof_records_zero_policy_success_without_failing_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    written: dict[str, dict[str, object]] = {}
    published: list[dict[str, object]] = []
    monkeypatch.setattr(
        workflow_stage,
        "read_json",
        lambda *_args, **_kwargs: {
            "success_rate": 0.0,
            "policy_checkpoint_uri": "s3://bucket/run/checkpoint.pt",
        },
    )
    monkeypatch.setattr(
        workflow_stage,
        "write_json",
        lambda uri, payload, **_kwargs: written.setdefault(uri, payload),
    )
    monkeypatch.setattr(
        workflow_stage,
        "publish_component_record",
        lambda **kwargs: published.append(kwargs) or kwargs,
    )
    args = Namespace(
        root_uri="s3://bucket/run",
        run_id="run",
        outer_iteration=1,
        threshold=0.5,
        allow_early_exit=False,
    )
    workflow_stage._stage11(args)

    decision = written["s3://bucket/run/outer_loop/decision.json"]
    assert decision["decision"] == "loop_back_to_inner_loop"
    assert decision["success_rate"] == 0.0
    assert published[0]["tier"] == "WORKS"
    assert published[0]["next_action"] == "LOOP_OR_COMPLETE_BUDGET"
