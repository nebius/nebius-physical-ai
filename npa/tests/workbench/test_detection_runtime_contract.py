"""Exercise detector adapters, authenticated probes, durable state and real artifact IO."""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from npa.cli.main import app as cli_app
from npa.workbench.detection_training import evaluation, service, training
from npa.workbench.detection_training.labels import category_id_map, detector_label_map
from npa.workbench.detection_training.run_store import RunStore, RunStoreError
from npa.workbench.detection_training.schemas import StatusResponse, TrainRequest, TrainResponse
from npa.workbench.detection_training.storage import describe_artifact


@pytest.mark.parametrize("label_map", [None, {"vehicle": 0, "person": 4}])
def test_actual_cli_adapter_calls_sdk_shared_training(monkeypatch, tmp_path, label_map):
    captured = []

    def train(request):
        captured.append(request)
        return TrainResponse(run_id="adapter-test", status="completed", checkpoint_uri_pattern="synthetic", metrics_uri="synthetic", total_epochs=2, manifest_sha256="synthetic")

    # Replace only expensive workload execution, leaving the actual CLI and SDK intact.
    monkeypatch.setattr(training, "train_detector", train)
    args = ["workbench", "detection-training", "train", "--view", "materialized", "--output-uri", str(tmp_path), "--epochs", "2"]
    if label_map is not None:
        args += ["--label-map", json.dumps(label_map)]
    result = CliRunner().invoke(cli_app, args)
    assert result.exit_code == 0, str(result.exception)
    assert len(captured) == 1
    assert captured[0].label_map == label_map
    assert captured[0].num_classes is None
    assert training.resolve_num_classes(captured[0]) == (6 if label_map else 10)


def test_label_identity_through_string_and_numeric_materialized_rows(monkeypatch):
    from npa.workbench.detection_training.dataloader import _coerce_mapped_targets

    source = {"vehicle": 0, "person": 4}
    mapped = detector_label_map(source)
    assert mapped == {"vehicle": 1, "person": 5}
    kwargs = {"label_map": mapped, "category_id_map": category_id_map(source)}
    boxes = [[0, 0, 10, 10], [1, 1, 8, 8]]
    assert _coerce_mapped_targets(boxes, ["vehicle", "person"], **kwargs) == _coerce_mapped_targets(boxes, [0, 4], **kwargs)
    assert _coerce_mapped_targets(boxes, [0, 4], **kwargs)[1] == [1, 5]
    assert detector_label_map({"vehicle": 1, "person": 4}) == {"vehicle": 1, "person": 4}
    with pytest.raises(ValueError, match="background"):
        TrainRequest(view="view", output_uri="out", label_map=source, num_classes=5)
    with pytest.raises(ValueError, match="distinct"):
        TrainRequest(view="view", output_uri="out", label_map={"vehicle": 0, "person": 0})


def test_checkpoint_category_metadata_is_authoritative():
    source = {"vehicle": 0, "person": 4}
    checkpoint = {"label_map": source, "detector_label_map": {"vehicle": 1, "person": 5}}
    assert evaluation.checkpoint_labels(checkpoint, requested=None) == ({"vehicle": 1, "person": 5}, {0: 1, 4: 5})
    assert evaluation.checkpoint_labels(checkpoint, requested=source) == ({"vehicle": 1, "person": 5}, {0: 1, 4: 5})
    with pytest.raises(evaluation.DetectionEvaluationError, match="differs"):
        evaluation.checkpoint_labels(checkpoint, requested={"vehicle": 4, "person": 0})
    # Older checkpoints were trained with unshifted IDs: never reinterpret their weights.
    assert evaluation.checkpoint_labels({"request": {"label_map": source}}, requested=None) == (source, None)


def test_missing_metric_dependency_is_failure_not_zero_score(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "torchmetrics.detection.mean_ap", None)
    with pytest.raises(evaluation.DetectionEvaluationError, match="required"):
        evaluation._compute_map_metrics(None, [], device="cpu")
    assert evaluation._finite_float(-1.0) == -1.0  # COCO undefined metric sentinel is honest.
    with pytest.raises(evaluation.DetectionEvaluationError, match="non-finite"):
        evaluation._finite_float(float("nan"))


def test_generated_authenticated_deployment_probe_calls_real_service(monkeypatch, tmp_path):
    from npa.cli.workbench import detection_training as cli
    monkeypatch.setattr(cli, "resolve_submit_credentials", lambda: SimpleNamespace(s3_access_key_id="", s3_secret_access_key="", s3_endpoint=""))
    monkeypatch.setenv("DETECTION_TRAINING_TOKEN", "synthetic-bearer")
    manifest = cli._kubernetes_manifest(image="example/detector:test", name="detector", namespace="default", port=8790, input_path="synthetic", output_path=str(tmp_path / "outputs"), node_selector_key="gpu", node_selector_value="synthetic", image_pull_secret="", auth_mode="token", token_env="DETECTION_TRAINING_TOKEN")
    pod = next(item for item in manifest["items"] if item["kind"] == "Deployment")["spec"]["template"]["spec"]
    container = pod["containers"][0]
    probe = container["readinessProbe"]["httpGet"]
    assert probe["port"] == container["ports"][0]["name"]
    assert "synthetic-bearer" not in json.dumps(manifest)
    assert "httpHeaders" not in probe
    assert pod["volumes"][0]["persistentVolumeClaim"]["claimName"] == "detector-state"
    with TestClient(service.create_app(auth_mode="token", token="synthetic-bearer", state_dir=tmp_path / "state")) as client:
        assert client.get(probe["path"]).json() == {"status": "ok"}
        for path in ("/health", "/system-info", "/runs", "/list", "/status?run_id=synthetic", "/artifacts?run_id=synthetic", "/artifacts/content?run_id=synthetic&sha256=synthetic"):
            assert client.get(path).status_code == 401
        assert client.post("/train", json={"view": "view", "output_uri": "out"}).status_code == 401
        assert client.post("/eval", json={"checkpoint_uri": "checkpoint", "eval_view": "view", "output_uri": "out"}).status_code == 401
        assert client.get("/health", headers={"Authorization": "Bearer synthetic-bearer"}).status_code == 200


def _record_artifacts(root, run_id="run", epochs=2):
    checkpoint = root / run_id / "actual-checkpoint.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"serialized checkpoint fixture")
    metrics = checkpoint.parent / "metrics.json"
    metrics.write_text('{"train_loss": 1.5}')
    return [
        describe_artifact(str(checkpoint), role="checkpoint", media_type="application/x-pytorch", schema_version="npa.detection.checkpoint.v1", epoch=epochs),
        describe_artifact(str(metrics), role="training_metrics", media_type="application/json", schema_version="npa.detection.training-metrics.v1"),
    ]


def test_completed_run_and_exact_artifacts_survive_new_service_client(monkeypatch, tmp_path):
    output = tmp_path / "outputs"
    state = tmp_path / "state"
    observed = []

    def execute(request, *, run_id, status_callback, artifact_callback):
        observed.append(request)
        artifacts = _record_artifacts(output, run_id)
        status_callback("running", 1, {"train_loss": 2.0}, None)
        artifact_callback(artifacts)
        status_callback("completed", 2, {"train_loss": 1.5}, None)
        return TrainResponse(run_id=run_id, status="completed", checkpoint_uri_pattern="unused", metrics_uri=artifacts[1].uri, total_epochs=2, manifest_sha256="synthetic", artifacts=artifacts)

    monkeypatch.setattr(service, "train_detector", execute)
    headers = {"Authorization": "Bearer synthetic-bearer"}
    with TestClient(service.create_app(auth_mode="token", token="synthetic-bearer", state_dir=state, output_scope=str(output))) as client:
        accepted = client.post("/train", headers=headers, json={"view": "view", "output_uri": str(output), "label_map": {"vehicle": 0}, "epochs": 2}).json()
        run_id = accepted["run_id"]
        assert accepted["status"] == "queued"
        before = client.get("/status", params={"run_id": run_id}, headers=headers).json()
        assert before["status"] == "completed"
        assert before["epochs_completed"] == 2
    with TestClient(service.create_app(auth_mode="token", token="synthetic-bearer", state_dir=state, output_scope=str(output))) as fresh:
        after = fresh.get("/status", params={"run_id": run_id}, headers=headers).json()
        assert after == before
        artifacts = fresh.get("/artifacts", params={"run_id": run_id}, headers=headers).json()["artifacts"]
        assert len(artifacts) == 2
        for artifact in artifacts:
            data = fresh.get("/artifacts/content", params={"run_id": run_id, "sha256": artifact["sha256"]}, headers=headers)
            assert data.status_code == 200
            assert len(data.content) == artifact["size_bytes"]
            assert hashlib.sha256(data.content).hexdigest() == artifact["sha256"]
        Path(artifacts[0]["uri"]).write_bytes(b"tampered")
        assert fresh.get("/artifacts/content", params={"run_id": run_id, "sha256": artifacts[0]["sha256"]}, headers=headers).status_code == 409
        assert fresh.get("/status", params={"run_id": run_id}).status_code == 401
    assert observed[0].label_map == {"vehicle": 0}


def test_failed_and_interrupted_runs_survive_restart_and_do_not_resume(monkeypatch, tmp_path):
    def fail_training(*args, **kwargs):
        raise training.DetectionTrainingError("dataset rejected")
    monkeypatch.setattr(service, "train_detector", fail_training)
    with TestClient(service.create_app(auth_mode="none", state_dir=tmp_path)) as client:
        run_id = client.post("/train", json={"view": "view", "output_uri": "out"}).json()["run_id"]
        assert client.get("/status", params={"run_id": run_id}).json()["error"] == "dataset rejected"
        client.app.state.store.create(StatusResponse(run_id="unfinished", status="running", total_epochs=3, epochs_completed=1))
    with TestClient(service.create_app(auth_mode="none", state_dir=tmp_path)) as fresh:
        failed = fresh.get("/status", params={"run_id": run_id}).json()
        assert failed["status"] == "failed"
        unfinished = fresh.get("/status", params={"run_id": "unfinished"}).json()
        assert unfinished["status"] == "interrupted"
        assert unfinished["epochs_completed"] == 1
        assert "automatic resume is unavailable" in unfinished["error"]


def test_store_serializes_concurrent_updates_and_rejects_false_completion(tmp_path):
    store = RunStore(tmp_path)
    store.start()
    try:
        store.create(StatusResponse(run_id="concurrent", status="running", total_epochs=2))
        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(lambda i: RunStore(tmp_path).update("concurrent", last_metrics={"writer": i}), range(40)))
        assert store.get("concurrent").revision == 41
        with pytest.raises(RunStoreError, match="verified"):
            store.update("concurrent", status="completed", epochs_completed=2)
        store.update("concurrent", status="failed", error="real failure")
        with pytest.raises(RunStoreError, match="terminal"):
            store.update("concurrent", status="running")
        with pytest.raises(RunStoreError, match="another"):
            RunStore(tmp_path).start()
    finally:
        store.close()


def test_store_and_service_cannot_cross_output_scope(tmp_path):
    first = RunStore(tmp_path / "state", scope="s3://synthetic/service-a")
    first.create(StatusResponse(run_id="scoped", status="queued"))
    with pytest.raises(RunStoreError, match="scope"):
        RunStore(tmp_path / "state", scope="s3://synthetic/service-b").get("scoped")
    with TestClient(service.create_app(auth_mode="none", state_dir=tmp_path / "other-state", output_scope="s3://synthetic/service-a")) as client:
        for uri in ("s3://synthetic/service-ab/output", "s3://synthetic/service-a/../other"):
            assert client.post("/train", json={"view": "view", "output_uri": uri}).status_code in {400, 403}
        assert client.get("/runs").json()["runs"] == []


def test_service_credentials_remain_request_local(monkeypatch):
    from npa.workbench.detection_training.storage import _S3_SETTINGS, storage_settings
    from npa.workbench.detection_training.schemas import CheckpointS3Settings
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "ambient-synthetic")

    def read_context(index):
        with storage_settings(CheckpointS3Settings(aws_access_key_id=f"synthetic-{index}")):
            assert os.environ["AWS_ACCESS_KEY_ID"] == "ambient-synthetic"
            return _S3_SETTINGS.get()["aws_access_key_id"]
    with ThreadPoolExecutor(max_workers=8) as executor:
        assert list(executor.map(read_context, range(40))) == [f"synthetic-{i}" for i in range(40)]
    assert _S3_SETTINGS.get() == {}


def test_real_materialized_view_and_checkpoint_preserve_category_identity(tmp_path):
    import io
    import lancedb
    import pyarrow as pa
    import torch
    from PIL import Image
    from npa.workbench.detection_training.dataloader import make_dataloader
    from npa.workbench.lancedb.views import create_mv, refresh_mv

    image = io.BytesIO()
    Image.new("RGB", (32, 32)).save(image, format="PNG")
    row = {"image_bytes": image.getvalue(), "ann_bboxes": [[1., 1., 10., 10.], [12., 12., 25., 25.]], "ann_categories": ["vehicle", "person"], "split": "train"}
    uri = str(tmp_path / "lance")
    db = lancedb.connect(uri)
    db.create_table("source", data=pa.Table.from_pylist([row, {**row, "split": "eval"}]))
    created = create_mv(name="training_slice", source_table="source", filter_sql="split = 'train'", lance_uri=uri)
    refreshed = refresh_mv(name="training_slice", lance_uri=uri)
    assert created.row_count == refreshed.row_count == 1
    assert db.open_table("training_slice").to_arrow().to_pylist()[0]["ann_categories"] == row["ann_categories"]
    source_map = {"vehicle": 0, "person": 4}
    batch = next(iter(make_dataloader(lance_uri=uri, view="training_slice", batch_size=1, label_map=detector_label_map(source_map), category_id_map=category_id_map(source_map))))
    assert batch[1][0]["labels"].tolist() == [1, 5]
    checkpoint = tmp_path / "checkpoint.pt"
    training.save_checkpoint(str(checkpoint), model=SimpleNamespace(state_dict=lambda: {"weight": torch.ones(1)}), optimizer=None, epoch=1, manifest_sha256=created.manifest_sha256, num_classes=6, request={"label_map": source_map})
    loaded = evaluation._load_checkpoint(str(checkpoint))
    assert loaded["label_map"] == source_map
    assert loaded["detector_label_map"] == {"vehicle": 1, "person": 5}
    assert loaded["num_classes"] == 6


def test_service_storage_scope_rejects_symlink_escape_and_foreign_input(tmp_path, monkeypatch):
    output = tmp_path / "outputs"
    output.mkdir()
    (output / "escape").symlink_to(tmp_path / "foreign", target_is_directory=True)
    monkeypatch.setenv("NPA_INPUT_PATH", "s3://synthetic/input")
    with TestClient(service.create_app(auth_mode="none", state_dir=tmp_path / "state", output_scope=str(output))) as client:
        assert client.post("/train", json={"view": "view", "output_uri": str(output / "escape" / "out")}).status_code == 403
        assert client.post("/train", json={"view": "view", "output_uri": str(output), "lance_uri": "s3://synthetic/other-input"}).status_code == 403
        assert client.get("/runs").json()["runs"] == []


def test_checkpoint_and_wandb_request_omit_storage_credentials():
    request = TrainRequest(view="view", output_uri="out", checkpoint_s3={"aws_access_key_id": "synthetic-access", "aws_secret_access_key": "synthetic-secret"})
    payload = training._checkpoint_request(request)
    assert "synthetic-access" not in json.dumps(payload)
    assert "synthetic-secret" not in json.dumps(payload)
    assert "aws_secret_access_key" not in payload["checkpoint_s3"]


def test_secret_provisioning_uses_private_files_and_suppresses_error_payload(monkeypatch, capsys):
    import subprocess
    import typer
    from npa.cli.workbench import detection_training as cli

    paths = []
    def execute(args, **kwargs):
        assert kwargs["capture"] and kwargs["redact_errors"]
        assert "synthetic-secret-value" not in " ".join(args)
        if args[0] == "create":
            for arg in args:
                if arg.startswith("--from-file="):
                    path = Path(arg.split("=", 2)[2])
                    assert path.stat().st_mode & 0o777 == 0o600
                    assert path.parent.stat().st_mode & 0o777 == 0o700
                    assert path.read_text() == "synthetic-secret-value"
                    paths.append(path)
            return '{"kind":"Secret","data":{}}'
        assert kwargs["stdin"] == '{"kind":"Secret","data":{}}'
        return ""
    with monkeypatch.context() as local:
        local.setattr(cli, "_kubectl", execute)
        cli._provision_service_secret(name="synthetic", namespace="default", kubeconfig="", env={"DETECTION_TRAINING_TOKEN": "synthetic-secret-value"})
    assert paths and all(not path.exists() for path in paths)
    def fail_process(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0], stderr="synthetic-secret-value")
    monkeypatch.setattr(cli.subprocess, "run", fail_process)
    with pytest.raises(typer.Exit):
        cli._kubectl(["apply", "-f", "-"], stdin="synthetic-secret-value", redact_errors=True)
    assert "synthetic-secret-value" not in capsys.readouterr().err


def test_evaluation_result_and_artifacts_are_discoverable_after_restart(monkeypatch, tmp_path):
    calls = []
    def score(request):
        calls.append(request)
        return {"mAP": 0.125, "mAP_50": 0.25, "mAP_75": 0.1, "per_category_AP": {"vehicle": 0.125}}
    monkeypatch.setattr(evaluation, "_evaluate_with_model", score)
    payload = {"checkpoint_uri": str(tmp_path / "outputs/checkpoint.pt"), "eval_view": "held_out", "output_uri": str(tmp_path / "outputs/eval")}
    headers = {"Authorization": "Bearer synthetic-bearer"}
    with TestClient(service.create_app(auth_mode="token", token="synthetic-bearer", state_dir=tmp_path / "state", output_scope=str(tmp_path / "outputs"))) as client:
        result = client.post("/eval", json=payload, headers=headers)
        assert result.status_code == 200
        result = result.json()
        run_id = result["eval_run_id"]
        before = client.get("/status", params={"run_id": run_id}, headers=headers).json()
        assert before["kind"] == "eval"
        assert before["status"] == "completed"
        assert before["evaluation"] == result
    with TestClient(service.create_app(auth_mode="token", token="synthetic-bearer", state_dir=tmp_path / "state", output_scope=str(tmp_path / "outputs"))) as fresh:
        assert fresh.get("/status", params={"run_id": run_id}, headers=headers).json() == before
        assert fresh.post("/eval", json=payload, headers=headers).json() == result
        assert len(calls) == 1
        artifact = fresh.get("/artifacts", params={"run_id": run_id}, headers=headers).json()["artifacts"][0]
        assert artifact["role"] == "evaluation_metrics"
        content = fresh.get("/artifacts/content", params={"run_id": run_id, "sha256": artifact["sha256"]}, headers=headers)
        assert content.status_code == 200
        assert hashlib.sha256(content.content).hexdigest() == artifact["sha256"]
        assert json.loads(content.content)["mAP"] == 0.125
        assert fresh.get("/artifacts", params={"run_id": run_id}).status_code == 401


def test_failed_evaluation_keeps_identity_without_repeating_work(monkeypatch, tmp_path):
    calls = []
    def fail(request):
        calls.append(request)
        raise evaluation.DetectionEvaluationError("checkpoint category identity mismatch")
    monkeypatch.setattr(evaluation, "_evaluate_with_model", fail)
    payload = {"checkpoint_uri": "synthetic.pt", "eval_view": "view", "output_uri": str(tmp_path / "outputs")}
    with TestClient(service.create_app(auth_mode="none", state_dir=tmp_path / "state")) as client:
        assert client.post("/eval", json=payload).status_code == 400
        records = client.get("/runs").json()["runs"]
        assert len(records) == 1
        assert records[0]["kind"] == "eval" and records[0]["status"] == "failed"
        assert "mismatch" in records[0]["error"]
    with TestClient(service.create_app(auth_mode="none", state_dir=tmp_path / "state")) as fresh:
        refused = fresh.post("/eval", json=payload)
        assert refused.status_code == 409
        assert refused.json()["detail"]["run_id"] == records[0]["run_id"]
        assert len(calls) == 1


def test_numeric_checkpoint_honors_explicit_eval_map_through_sdk_and_real_loader(monkeypatch, tmp_path):
    import io
    import lancedb
    import pyarrow as pa
    import torch
    from PIL import Image
    from npa.sdk.workbench import detection_training as sdk

    image = io.BytesIO()
    Image.new("RGB", (32, 32)).save(image, format="PNG")
    uri = str(tmp_path / "lance")
    lancedb.connect(uri).create_table("held_out", data=pa.Table.from_pylist([{
        "image_bytes": image.getvalue(), "ann_bboxes": [[1., 1., 10., 10.]], "ann_categories": ["person"],
    }]))
    checkpoint = tmp_path / "numeric.pt"
    torch.save({"num_classes": 2, "label_map": None, "detector_label_map": None, "model_state_dict": {"weight": torch.ones(1)}}, checkpoint)
    model = SimpleNamespace(load_state_dict=lambda state, **kwargs: None, to=lambda device: None, eval=lambda: None)
    monkeypatch.setattr(evaluation, "build_fasterrcnn_resnet50_fpn_v2", lambda **kwargs: model)
    observed = []
    def score(model, loader, **kwargs):
        observed.extend(int(label) for _, targets in loader for target in targets for label in target["labels"])
        return {"mAP": 0.2, "mAP_50": 0.4, "mAP_75": 0.1, "per_category_AP": {"class_1": 0.2}}
    monkeypatch.setattr(evaluation, "_compute_map_metrics", score)
    response = sdk.eval(checkpoint_uri=str(checkpoint), eval_view="held_out", lance_uri=uri, output_uri=str(tmp_path / "eval"), label_map={"person": 1})
    assert observed == [1]
    assert response.per_category_AP == {"person": 0.2}
    assert response.artifacts[0].integrity_verified
