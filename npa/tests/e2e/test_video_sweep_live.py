"""Opt-in live checks for hosted prompts, paired review, S3, and tracking services."""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from types import SimpleNamespace

import av
import numpy as np
import pytest

from npa.workflows.video_sweep import artifacts, execution, planning, tracking

pytestmark = pytest.mark.token_factory_e2e


def _video(path: Path) -> None:
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=8)
        stream.width, stream.height, stream.pix_fmt = 128, 128, "yuv420p"
        for step in range(24):
            pixels = np.full((128, 128, 3), 150, dtype=np.uint8)
            pixels[45:75, 15 + step : 45 + step] = (230, 20, 20)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _args(tmp_path):
    return SimpleNamespace(
        root_uri=str(tmp_path / "run"),
        run_id="video-sweep-live-" + uuid.uuid4().hex,
        sources_uri=str(tmp_path / "sources.json"),
        variants_uri=str(tmp_path / "variants.json"),
        workers=1,
        worker=0,
        samples=4,
        threshold=0.8,
        reasoner_model=os.environ["NPA_VIDEO_SWEEP_REASONER_MODEL"],
        merge_model=os.environ.get(
            "NPA_VIDEO_SWEEP_MERGE_MODEL", "nvidia/Nemotron-3_5-Lightning"
        ),
    )


@pytest.fixture
def live_review(tmp_path):
    if not os.environ.get("NPA_VIDEO_SWEEP_REASONER_MODEL"):
        pytest.skip("Select an exact hosted model with NPA_VIDEO_SWEEP_REASONER_MODEL")
    args = _args(tmp_path)
    source = tmp_path / "procedural.mp4"
    _video(source)
    artifacts.write_json(
        args.sources_uri,
        {"schema": "npa.video_sweep.sources.v1", "clips": [str(source)]},
    )
    variant = {
        "hint": "Keep the existing neutral lighting and red block unchanged.",
        "seed": 7,
        "control": "edge",
        "control_weight": 1.0,
        "guidance": 3.0,
    }
    artifacts.write_json(
        args.variants_uri,
        {"schema": "npa.video_sweep.variants.v1", "variants": [variant]},
    )
    planning.prepare(args)
    plan = artifacts.read_json(args.root_uri + "/plan.json")
    item = plan["items"][0]
    # A procedural identity pair tests the real judge; it is not Transfer evidence.
    candidate = {
        "id": item["id"],
        "uri": str(source),
        "sha256": artifacts.file_digest(source),
        "engine": "procedural-identity-pair",
    }
    artifacts.write_json(
        args.root_uri + "/workers/0.json",
        {"plan_sha256": artifacts.digest(plan), "worker": 0, "items": [candidate]},
    )
    execution.review(args)
    return args


def test_hosted_description_merge_and_paired_judge(live_review):
    plan, report = execution.reviewed(live_review)
    assert plan["items"][0]["prompt"].strip()
    assert report["items"][0]["judge_provenance"]["request_id"]
    assert report["items"][0]["judge_provenance"]["model"] == live_review.reasoner_model
    assert type(report["items"][0]["accepted"]) is bool


def test_postgres_and_mlflow_commit_and_replay(live_review):
    if not all(
        os.environ.get(name)
        for name in (
            "NPA_LINEAGE_POSTGRES_DSN",
            "MLFLOW_TRACKING_URI",
            "MLFLOW_EXPERIMENT_ID",
        )
    ):
        pytest.skip("Requires explicit private Postgres and MLflow services")
    tracking.lineage(live_review)
    first = artifacts.read_json(live_review.root_uri + "/lineage.json")
    tracking.lineage(live_review)
    assert artifacts.read_json(live_review.root_uri + "/lineage.json") == first
    assert len(first["items"]) == 1


def test_private_s3_artifact_roundtrip(tmp_path):
    prefix = os.environ.get("NPA_VIDEO_SWEEP_TEST_S3_URI", "")
    if not prefix:
        pytest.skip("Requires an explicit private test S3 prefix")
    uri = prefix.rstrip("/") + "/" + uuid.uuid4().hex + "/artifact.json"
    document = {"schema": "npa.video_sweep.test.v1", "procedural": True}
    try:
        artifacts.write_json(uri, document)
        assert artifacts.read_json(uri) == document
        artifacts.write_json(uri, document)
        with pytest.raises(ValueError, match="different stage result"):
            artifacts.write_json(uri, {**document, "procedural": False})
    finally:
        bucket, key = artifacts._object(uri)
        artifacts._client().delete_object(Bucket=bucket, Key=key)


def test_full_gpu_stages_require_real_transfer(tmp_path):
    if os.environ.get("NPA_VIDEO_SWEEP_FULL_GPU") != "1":
        pytest.skip("Run inside the Transfer runtime with explicit full GPU opt-in")
    args = _args(tmp_path)
    args.root_uri = (
        os.environ["NPA_VIDEO_SWEEP_TEST_S3_URI"].rstrip("/") + "/" + args.run_id
    )
    args.sources_uri = os.environ["NPA_VIDEO_SWEEP_SOURCES_URI"]
    args.variants_uri = os.environ["NPA_VIDEO_SWEEP_VARIANTS_URI"]
    from npa.workflows.video_sweep.publication import publish

    planning.prepare(args)
    execution.generate(args)
    execution.review(args)
    tracking.lineage(args)
    publish(args)
    dataset = artifacts.read_json(args.root_uri + "/dataset/manifest.json")
    assert dataset["clips"]
    report = artifacts.read_json(args.root_uri + "/review.json")
    assert all(row["engine"] == "cosmos-transfer2.5" for row in report["items"])
