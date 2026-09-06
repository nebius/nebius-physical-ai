"""Detector artifact metadata must use uploaded bytes without reading them back."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from npa.workbench.detection_training import evaluation, storage, training
from npa.workbench.detection_training.schemas import EvalRequest, TrainRequest


@pytest.fixture(params=["local", "s3"])
def destination(request, tmp_path, monkeypatch):
    client = Mock(spec=["put_object", "get_object", "download_file", "download_fileobj"])
    objects = {}

    def put_object(*, Bucket, Key, Body):
        objects[f"s3://{Bucket}/{Key}"] = Body

    client.put_object.side_effect = put_object
    for method in (client.get_object, client.download_file, client.download_fileobj):
        method.side_effect = AssertionError("unexpected artifact read-back")
    monkeypatch.setattr(storage, "_s3_client", lambda: client)
    base = "s3://synthetic/artifacts" if request.param == "s3" else str(tmp_path / "artifacts")
    writes = []
    original_write = storage.write_bytes_uri

    def capture_write(uri, payload):
        receipt = original_write(uri, payload)
        writes.append((uri, payload))
        return receipt

    monkeypatch.setattr(storage, "write_bytes_uri", capture_write)
    monkeypatch.setattr(training, "write_bytes_uri", capture_write)
    return SimpleNamespace(base=base, client=client, objects=objects, writes=writes, backend=request.param)


def _forbid_readback(monkeypatch, destination):
    monkeypatch.setattr(storage, "read_bytes_uri", Mock(side_effect=AssertionError("unexpected artifact read-back")))
    read_bytes = Path.read_bytes

    def check_local_read(path):
        if destination.backend == "local" and path.is_relative_to(destination.base):
            raise AssertionError("unexpected artifact read-back")
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", check_local_read)


def _assert_metadata(artifact, uri, payload):
    assert artifact.uri == uri
    assert artifact.size_bytes == len(payload)
    assert artifact.sha256 == hashlib.sha256(payload).hexdigest()
    assert artifact.exists and artifact.integrity_verified


def test_training_checkpoint_and_metrics_use_each_successful_write(monkeypatch, destination):
    import torch

    class Detector(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.calls = 0

        def forward(self, images, targets):
            self.calls += 1
            return {"loss": self.weight.square().sum()}

    model = Detector()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(training, "build_fasterrcnn_resnet50_fpn_v2", lambda **kwargs: model)
    batch = ([torch.zeros(1)], [{"labels": torch.ones(1, dtype=torch.int64)}])
    monkeypatch.setattr(training, "make_dataloader", lambda **kwargs: [batch])
    snapshots = []
    with monkeypatch.context() as no_reads:
        _forbid_readback(no_reads, destination)
        result = training.train_detector(
            TrainRequest(view="synthetic", output_uri=destination.base, epochs=2),
            run_id="synthetic-run", artifact_callback=snapshots.append,
        )

    assert result.status == "completed"
    assert model.calls == len(snapshots) == 2
    assert model.weight.item() < 1.0
    assert len(destination.writes) == 4
    for epoch, snapshot in enumerate(snapshots, start=1):
        assert len(snapshot) == epoch + 1
        for checkpoint_epoch, artifact in enumerate(snapshot[:-1], start=1):
            uri, payload = destination.writes[2 * (checkpoint_epoch - 1)]
            _assert_metadata(artifact, uri, payload)
            assert artifact.role == "checkpoint" and artifact.epoch == checkpoint_epoch
            assert torch.load(io.BytesIO(payload), weights_only=False)["epoch"] == checkpoint_epoch
        uri, payload = destination.writes[2 * epoch - 1]
        _assert_metadata(snapshot[-1], uri, payload)
        assert snapshot[-1].role == "training_metrics"
        assert len(json.loads(payload)["epochs"]) == epoch
    assert snapshots[0][-1].uri == snapshots[1][-1].uri
    assert snapshots[0][-1].sha256 != snapshots[1][-1].sha256
    assert result.artifacts == snapshots[-1]
    for artifact in result.artifacts:
        payload = destination.objects[artifact.uri] if destination.backend == "s3" else Path(artifact.uri).read_bytes()
        _assert_metadata(artifact, artifact.uri, payload)
    destination.client.get_object.assert_not_called()
    destination.client.download_file.assert_not_called()
    destination.client.download_fileobj.assert_not_called()


def test_evaluation_metrics_use_successful_write(monkeypatch, destination):
    monkeypatch.setattr(evaluation, "_evaluate_with_model", lambda request: {"mAP": 0.2, "mAP_50": 0.3, "mAP_75": 0.1})
    _forbid_readback(monkeypatch, destination)
    result = evaluation.evaluate_detector(EvalRequest(
        checkpoint_uri="synthetic.pt", eval_view="synthetic", output_uri=destination.base,
    ))
    assert len(destination.writes) == len(result.artifacts) == 1
    uri, payload = destination.writes[0]
    _assert_metadata(result.artifacts[0], uri, payload)
    assert result.artifacts[0].role == "evaluation_metrics"
    assert json.loads(payload)["mAP"] == 0.2
    destination.client.get_object.assert_not_called()


def test_describe_existing_artifact_reads_current_bytes_without_receipt(monkeypatch, destination):
    uri = storage.uri_join(destination.base, "existing.json")
    for payload in (b'{"old": true}', b'{"replacement": true}'):
        if destination.backend == "s3":
            destination.client.get_object.side_effect = lambda **kwargs: {"Body": io.BytesIO(payload)}
        else:
            Path(uri).parent.mkdir(parents=True, exist_ok=True)
            Path(uri).write_bytes(payload)
        artifact = storage.describe_artifact(
            uri, role="training_metrics", media_type="application/json", schema_version="synthetic.v1",
        )
        _assert_metadata(artifact, uri, payload)
    assert destination.writes == []
    if destination.backend == "s3":
        assert destination.client.get_object.call_count == 2


def test_write_receipt_is_bound_to_uri(monkeypatch, destination):
    uri = storage.uri_join(destination.base, "metrics.json")
    _forbid_readback(monkeypatch, destination)
    receipt = storage.write_json_uri(uri, {"label": "synthetic", "epoch": 1})
    _, payload = destination.writes[0]
    assert receipt.size_bytes == len(payload)
    assert receipt.sha256 == hashlib.sha256(payload).hexdigest()
    with pytest.raises(ValueError, match="artifact write receipt URI does not match artifact URI"):
        storage.describe_artifact(
            storage.uri_join(destination.base, "other.json"), role="training_metrics",
            media_type="application/json", schema_version="synthetic.v1", write_receipt=receipt,
        )


@pytest.mark.parametrize("use_receipt", [False, True])
def test_describe_empty_artifact_still_fails(monkeypatch, destination, use_receipt):
    uri = storage.uri_join(destination.base, "empty.json")
    receipt = storage.write_bytes_uri(uri, b"")
    if use_receipt:
        _forbid_readback(monkeypatch, destination)
    elif destination.backend == "s3":
        destination.client.get_object.side_effect = lambda **kwargs: {"Body": io.BytesIO(b"")}
    with pytest.raises(ValueError, match="produced artifact is empty"):
        storage.describe_artifact(
            uri, role="training_metrics", media_type="application/json", schema_version="synthetic.v1",
            write_receipt=receipt if use_receipt else None,
        )


def test_failed_write_never_publishes_artifact_metadata(monkeypatch, destination):
    monkeypatch.setattr(evaluation, "_evaluate_with_model", lambda request: {"mAP": 0.2, "mAP_50": 0.3, "mAP_75": 0.1})
    failure = Mock(side_effect=OSError("synthetic write failure"))
    if destination.backend == "s3":
        destination.client.put_object.side_effect = failure
    else:
        monkeypatch.setattr(storage.os, "replace", failure)
    describe = Mock(side_effect=AssertionError("failed write cannot publish metadata"))
    monkeypatch.setattr(evaluation, "describe_artifact", describe)
    with pytest.raises(OSError, match="synthetic write failure"):
        evaluation.evaluate_detector(EvalRequest(
            checkpoint_uri="synthetic.pt", eval_view="synthetic", output_uri=destination.base,
        ))
    describe.assert_not_called()
    assert destination.writes == []
    assert destination.objects == {}
    if destination.backend == "local":
        assert list(Path(destination.base).rglob(".artifact-*")) == []
