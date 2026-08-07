"""Real-format GR00T telemetry artifact and publish-gate coverage."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest

from npa.workbench.foxglove.inspect import summarize_mcap
from npa.workflows import groot_visualization as viz


class _Body:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body


class FakeS3:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], tuple[bytes, str]] = {}

    def seed(
        self,
        uri: str,
        body: bytes,
        content_type: str = "application/octet-stream",
    ) -> None:
        ref = viz._split_s3(uri)
        self.objects[(ref.bucket, ref.key)] = (body, content_type)

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        body, _content_type = self.objects[(Bucket, Key)]
        return {"Body": _Body(body)}

    def put_object(
        self,
        *,
        Bucket: str,
        Key: str,
        Body: bytes,
        ContentType: str = "application/octet-stream",
    ) -> None:
        self.objects[(Bucket, Key)] = (bytes(Body), ContentType)

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        body, content_type = self.objects[(Bucket, Key)]
        return {
            "ContentLength": len(body),
            "ContentType": content_type,
            "ETag": '"fixture-etag"',
        }

    def list_objects_v2(
        self, *, Bucket: str, Prefix: str, **_kwargs: Any
    ) -> dict[str, Any]:
        contents = [
            {"Key": key, "Size": len(body), "ETag": '"fixture-etag"'}
            for (bucket, key), (body, _content_type) in sorted(self.objects.items())
            if bucket == Bucket and key.startswith(Prefix)
        ]
        return {"Contents": contents, "IsTruncated": False}

    def download_file(self, bucket: str, key: str, filename: str) -> None:
        body, _content_type = self.objects[(bucket, key)]
        target = Path(filename)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)


def _manifest(run_id: str) -> dict[str, Any]:
    return {
        "schema": "npa.groot.finetune.v1",
        "status": "completed",
        "run_id": run_id,
        "num_gpus": 2,
        "world_size": 2,
        "distinct_gpu_count": 2,
        "gpu_uuids": ["GPU-fixture-a", "GPU-fixture-b"],
        "collective_ok": True,
        "collective_sum": 3.0,
        "collective_expected": 3.0,
        "training_step": 1,
        "optimizer_step_ok": True,
        "loss": 1.03125,
        "loss_finite": True,
        "checkpoint_bytes": 4,
        "checkpoint_object_count": 1,
        "checkpoint_uri": "s3://bucket/run/checkpoints/",
        "training_log_uri": "s3://bucket/run/checkpoints/training.log",
    }


@pytest.fixture
def source_bundle(monkeypatch: pytest.MonkeyPatch) -> tuple[FakeS3, str]:
    pytest.importorskip("PIL")
    from PIL import Image

    run_id = "groot-real-fixture"
    client = FakeS3()
    client.seed(
        "s3://bucket/run/checkpoints/npa_groot_finetune_manifest.json",
        json.dumps(_manifest(run_id)).encode(),
        "application/json",
    )
    client.seed("s3://bucket/run/checkpoints/model.bin", b"real")
    client.seed(
        "s3://bucket/run/checkpoints/training.log",
        b"step=1 loss=1.03125 optimizer complete\n",
        "text/plain",
    )
    client.seed("s3://bucket/run/data/videos/episode_000000.mp4", b"real-video-fixture")

    def decode(_video: Path, frames_dir: Path, *, max_frames: int):
        frames_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for index in range(min(2, max_frames)):
            path = frames_dir / f"{index:06d}.png"
            buffer = io.BytesIO()
            Image.new("RGB", (8, 6), (index * 50, 100, 200)).save(buffer, "PNG")
            path.write_bytes(buffer.getvalue())
            paths.append(path)
        return paths, 30.0

    monkeypatch.setattr(viz, "_decode_dataset_video", decode)
    viz.validate_visualization_source(
        "s3://bucket/run/checkpoints/npa_groot_finetune_manifest.json",
        "s3://bucket/run/data/",
        "s3://bucket/run/reports/visualization-source/manifest.json",
        run_id,
        2,
        2,
        s3_client=client,
    )
    return client, run_id


def test_training_validation_fails_closed_on_missing_optimizer_evidence() -> None:
    payload = _manifest("run")
    payload["optimizer_step_ok"] = False
    with pytest.raises(viz.GrootVisualizationError, match="optimizer_step_ok"):
        viz._validate_training_manifest(payload, run_id="run", expected_gpu_count=2)


def test_source_bundle_records_factual_provenance_and_timestamp_label(
    source_bundle: tuple[FakeS3, str],
) -> None:
    client, run_id = source_bundle
    source = viz._read_s3_json(
        client, "s3://bucket/run/reports/visualization-source/manifest.json"
    )
    assert source["run_id"] == run_id
    assert source["training"]["loss"] == 1.03125
    assert source["training"]["world_size"] == 2
    assert source["dataset"]["source_object_uri"].endswith("episode_000000.mp4")
    assert source["dataset"]["timestamp_semantics"] == "dataset/synthetic-fps"
    assert source["dataset"]["is_robot_capture_time"] is False
    assert source["dataset"]["frame_count"] == 2


def test_real_mcap_rrd_and_publish_gate_round_trip(
    source_bundle: tuple[FakeS3, str], tmp_path: Path
) -> None:
    pytest.importorskip("mcap")
    pytest.importorskip("rerun")
    client, run_id = source_bundle
    mcap_uri = "s3://bucket/run/reports/groot-training.mcap"
    rrd_uri = "s3://bucket/run/reports/groot-training.rrd"
    manifest_uri = "s3://bucket/run/reports/visualization-manifest.json"

    written = viz.emit_mcap(
        "s3://bucket/run/reports/visualization-source/manifest.json",
        mcap_uri,
        run_id,
        30.0,
        s3_client=client,
    )
    mcap_path = tmp_path / "recording.mcap"
    mcap_path.write_bytes(
        client.objects[("bucket", "run/reports/groot-training.mcap")][0]
    )
    info = summarize_mcap(mcap_path)
    assert info.schemas["/camera"] == "foxglove.CompressedImage"
    assert info.schemas["/log"] == "foxglove.Log"
    assert {
        "/metrics/loss",
        "/metrics/gpu_count",
        "/metrics/world_size",
        "/metrics/checkpoint_bytes",
    } <= set(info.channels)
    assert info.metadata["npa"]["run_id"] == run_id
    assert info.metadata["npa"]["timestamps"] == "dataset/synthetic-fps"
    assert written["inspect"]["message_count"] > 0

    converted = viz.emit_rrd(mcap_uri, rrd_uri, run_id, s3_client=client)
    assert converted["inspect"]["parseable"] is True
    assert converted["inspect"]["application_id"] == "npa_groot_training"
    assert converted["inspect"]["recording_id"] == run_id
    assert converted["converter"]["provenance"]["run_id"] == run_id

    client.seed(
        "s3://bucket/run/workflow.yaml",
        b"apiVersion: npa.workflow/v0.0.1\nkind: Workflow\nmetadata:\n  name: groot-1-7-finetune\n",
        "application/yaml",
    )
    published = viz.publish_visualizations(
        "s3://bucket/run/reports/visualization-source/manifest.json",
        mcap_uri,
        rrd_uri,
        "s3://bucket/run/workflow.yaml",
        manifest_uri,
        run_id,
        s3_client=client,
    )
    assert published["status"] == "published"
    assert published["provenance_valid"] is True
    assert set(published["artifacts"]) == {
        "mcap",
        "rrd",
        "visualization_manifest",
        "workflow",
    }
    stored = viz._read_s3_json(client, manifest_uri)
    assert stored["artifacts"]["visualization_manifest"]["uri"] == manifest_uri


def test_publish_gate_fails_when_rrd_is_missing(
    source_bundle: tuple[FakeS3, str],
) -> None:
    client, run_id = source_bundle
    client.seed(
        "s3://bucket/run/workflow.yaml",
        b"apiVersion: npa.workflow/v0.0.1\nkind: Workflow\nmetadata:\n  name: groot\n",
    )
    with pytest.raises((viz.GrootVisualizationError, KeyError)):
        viz.publish_visualizations(
            "s3://bucket/run/reports/visualization-source/manifest.json",
            "s3://bucket/run/reports/missing.mcap",
            "s3://bucket/run/reports/missing.rrd",
            "s3://bucket/run/workflow.yaml",
            "s3://bucket/run/reports/visualization-manifest.json",
            run_id,
            s3_client=client,
        )
