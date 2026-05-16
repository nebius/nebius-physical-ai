from __future__ import annotations

import io
import json
import math
import os
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from typer.testing import CliRunner

from npa.cli.workbench.detection_training import app as detection_training_app
from npa.workbench.detection_training.schemas import EvalRequest, TrainRequest


runner = CliRunner()


class FakeTensor:
    def __init__(self, value: Any):
        import numpy as np

        self.array = np.asarray(value)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.array.shape

    def permute(self, *dims: int) -> "FakeTensor":
        import numpy as np

        return FakeTensor(np.transpose(self.array, dims))

    def __truediv__(self, value: float) -> "FakeTensor":
        return FakeTensor(self.array / value)

    def to(self, _device: Any) -> "FakeTensor":
        return self


class FakeLoss:
    def __init__(self, value: float):
        self.value = float(value)
        self.backward_called = False

    def __add__(self, other: Any) -> "FakeLoss":
        return FakeLoss(self.value + _number(other))

    def __radd__(self, other: Any) -> "FakeLoss":
        return FakeLoss(_number(other) + self.value)

    def detach(self) -> "FakeLoss":
        return self

    def cpu(self) -> "FakeLoss":
        return self

    def item(self) -> float:
        return self.value

    def backward(self) -> None:
        self.backward_called = True


def test_model_construction_replaces_predictor(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    class FakePredictor:
        def __init__(self, in_features: int, num_classes: int):
            self.in_features = in_features
            self.num_classes = num_classes

    class FakeBoxPredictor:
        cls_score = types.SimpleNamespace(in_features=256)

    class FakeModel:
        def __init__(self) -> None:
            self.roi_heads = types.SimpleNamespace(box_predictor=FakeBoxPredictor())

    class FakeWeights:
        COCO_V1 = "coco"

    def fake_factory(*, weights: Any):
        seen["weights"] = weights
        return FakeModel()

    torchvision = types.ModuleType("torchvision")
    models = types.ModuleType("torchvision.models")
    detection = types.ModuleType("torchvision.models.detection")
    faster_rcnn = types.ModuleType("torchvision.models.detection.faster_rcnn")
    detection.FasterRCNN_ResNet50_FPN_V2_Weights = FakeWeights
    detection.fasterrcnn_resnet50_fpn_v2 = fake_factory
    faster_rcnn.FastRCNNPredictor = FakePredictor
    monkeypatch.setitem(sys.modules, "torchvision", torchvision)
    monkeypatch.setitem(sys.modules, "torchvision.models", models)
    monkeypatch.setitem(sys.modules, "torchvision.models.detection", detection)
    monkeypatch.setitem(sys.modules, "torchvision.models.detection.faster_rcnn", faster_rcnn)

    from npa.workbench.detection_training.models import build_fasterrcnn_resnet50_fpn_v2

    model = build_fasterrcnn_resnet50_fpn_v2(num_classes=10)

    assert seen["weights"] == "coco"
    assert model.roi_heads.box_predictor.in_features == 256
    assert model.roi_heads.box_predictor.num_classes == 10


def test_lance_detection_dataset_yields_expected_shapes(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_torch(monkeypatch)
    from npa.workbench.detection_training.dataloader import LanceDetectionDataset

    dataset = LanceDetectionDataset(rows=[_synthetic_row()])
    image, target = dataset[0]

    assert image.shape == (3, 8, 8)
    assert target["boxes"].shape == (1, 4)
    assert target["labels"].shape == (1,)


def test_training_loop_with_mock_model_has_finite_loss() -> None:
    from npa.workbench.detection_training.training import train_one_epoch

    class Model:
        def __init__(self) -> None:
            self.train_called = False

        def train(self) -> None:
            self.train_called = True

        def __call__(self, images: list[Any], targets: list[dict[str, Any]]) -> dict[str, FakeLoss]:
            assert len(images) == 1
            assert targets[0]["labels"]
            return {"loss_classifier": FakeLoss(1.25), "loss_box_reg": FakeLoss(0.25)}

    class Optimizer:
        def __init__(self) -> None:
            self.steps = 0

        def zero_grad(self) -> None:
            pass

        def step(self) -> None:
            self.steps += 1

    model = Model()
    optimizer = Optimizer()

    loss = train_one_epoch(model, [(["image"], [{"labels": [1]}])], optimizer, device="cpu")

    assert model.train_called
    assert optimizer.steps == 1
    assert math.isfinite(loss)
    assert loss == 1.5


def test_checkpoint_writer_uses_expected_uri_pattern(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_torch(monkeypatch)
    captured: dict[str, bytes] = {}

    def fake_write(uri: str, payload: bytes) -> None:
        captured[uri] = payload

    monkeypatch.setattr("npa.workbench.detection_training.training.write_bytes_uri", fake_write)

    from npa.workbench.detection_training.training import checkpoint_uri_pattern, save_checkpoint

    pattern = checkpoint_uri_pattern("s3://bucket/out", "run-1")
    save_checkpoint(
        pattern.format(epoch=1),
        model=types.SimpleNamespace(state_dict=lambda: {"w": 1}),
        optimizer=types.SimpleNamespace(state_dict=lambda: {"lr": 0.1}),
        epoch=1,
        manifest_sha256="abc",
        num_classes=10,
        request={"view": "mv"},
    )

    assert "s3://bucket/out/run-1/checkpoints/epoch_1.pt" in captured
    assert captured["s3://bucket/out/run-1/checkpoints/epoch_1.pt"]


def test_evaluation_returns_expected_schema(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from npa.workbench.detection_training import evaluation

    monkeypatch.setattr(
        evaluation,
        "_evaluate_with_model",
        lambda _request: {"mAP": 0.1, "mAP_50": 0.2, "mAP_75": 0.05, "per_category_AP": {"class_1": 0.1}},
    )
    result = evaluation.evaluate_detector(
        EvalRequest(
            checkpoint_uri=str(tmp_path / "checkpoint.pt"),
            eval_view="mv",
            lance_uri=str(tmp_path / "db"),
            output_uri=str(tmp_path / "out"),
        )
    )

    assert result.mAP == 0.1
    assert result.mAP_50 == 0.2
    assert result.per_category_AP == {"class_1": 0.1}
    assert result.manifest_sha256


def test_request_response_validation() -> None:
    request = TrainRequest(view="bdd100k_rider_train", output_uri="s3://bucket/out")

    assert request.num_classes == 10
    with pytest.raises(ValueError):
        TrainRequest(view="", output_uri="s3://bucket/out")
    with pytest.raises(ValueError):
        TrainRequest(view="mv", output_uri="s3://bucket/out", batch_size=0)


def test_api_cli_sdk_service_mode_manifest_parity(monkeypatch: pytest.MonkeyPatch) -> None:
    import npa.cli.workbench.detection_training as cli_module
    import npa.sdk.workbench.detection_training as sdk_module
    import npa.workbench.detection_training.service as service_module
    from npa.workbench.detection_training.service import create_app

    def fake_train(request: TrainRequest, *, run_id: str | None = None, status_callback: Any = None):
        from npa.workbench.detection_training.training import checkpoint_uri_pattern, compute_manifest_sha256, metrics_uri
        from npa.workbench.detection_training.schemas import TrainResponse

        manifest = compute_manifest_sha256("train", request.model_dump(mode="json"))
        resolved_run = run_id or "run-test"
        if status_callback:
            status_callback("completed", request.epochs, {"train_loss": 1.0}, None)
        return TrainResponse(
            run_id=resolved_run,
            status="completed",
            checkpoint_uri_pattern=checkpoint_uri_pattern(request.output_uri, resolved_run),
            metrics_uri=metrics_uri(request.output_uri, resolved_run),
            total_epochs=request.epochs,
            manifest_sha256=manifest,
        )

    monkeypatch.setattr(service_module, "train_detector", fake_train)
    client = TestClient(create_app(auth_mode="none"))

    payload = {
        "view": "bdd100k_rider_train",
        "lance_uri": "s3://bucket/db/",
        "output_uri": "s3://bucket/out",
        "num_classes": 10,
        "epochs": 1,
        "batch_size": 2,
        "learning_rate": 0.005,
    }
    api_response = client.post("/train", json=payload)
    assert api_response.status_code == 200, api_response.text

    def fake_cli_request(method: str, endpoint: str, path: str, **kwargs: Any) -> dict[str, Any]:
        response = client.request(method, path, json=kwargs.get("payload"), params=kwargs.get("params"))
        assert response.status_code == 200, response.text
        return response.json()

    def fake_sdk_request(method: str, endpoint: str, path: str, **kwargs: Any) -> dict[str, Any]:
        response = client.request(method, path, json=kwargs.get("payload"), params=kwargs.get("params"))
        assert response.status_code == 200, response.text
        return response.json()

    monkeypatch.setattr(cli_module, "request_json", fake_cli_request)
    monkeypatch.setattr(sdk_module, "_request_json", fake_sdk_request)

    cli_response = runner.invoke(
        detection_training_app,
        [
            "train",
            "--service",
            "--endpoint",
            "http://dt.example",
            "--view",
            payload["view"],
            "--lance-uri",
            payload["lance_uri"],
            "--output-uri",
            payload["output_uri"],
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--output",
            "json",
        ],
    )
    assert cli_response.exit_code == 0, cli_response.output

    sdk_response = sdk_module.train(service=True, endpoint="http://dt.example", **payload)

    manifests = {
        api_response.json()["manifest_sha256"],
        json.loads(cli_response.output)["manifest_sha256"],
        sdk_response.manifest_sha256,
    }
    assert len(manifests) == 1


def test_cli_and_sdk_do_not_import_heavy_ml_dependencies_at_module_level() -> None:
    cli_source = Path("npa/src/npa/cli/workbench/detection_training.py").read_text()
    sdk_source = Path("npa/src/npa/sdk/workbench/detection_training.py").read_text()

    assert "import torch" not in cli_source
    assert "import torchvision" not in cli_source
    assert "import torch" not in sdk_source
    assert "import torchvision" not in sdk_source


def test_sdk_workbench_namespace_exports_sdk_module() -> None:
    from npa.sdk import workbench

    assert workbench.detection_training.__name__ == "npa.sdk.workbench.detection_training"
    assert hasattr(workbench.detection_training, "train")


def test_deploy_dry_run_contains_gpu_selector(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "npa.cli.workbench.detection_training.load_credentials",
        lambda: types.SimpleNamespace(s3_access_key_id="", s3_secret_access_key="", s3_endpoint="https://storage.example"),
    )

    result = runner.invoke(
        detection_training_app,
        [
            "deploy",
            "--image",
            "registry/npa-detection-training:test",
            "--output-path",
            "s3://bucket/out",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    deployment = [item for item in payload["items"] if item["kind"] == "Deployment"][0]
    assert deployment["spec"]["template"]["spec"]["nodeSelector"]["node.kubernetes.io/instance-type"] == "gpu-h100-sxm"
    assert deployment["spec"]["template"]["spec"]["containers"][0]["resources"]["limits"]["nvidia.com/gpu"] == "1"


def test_deploy_can_build_registry_pull_secret_from_docker_auth(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from npa.cli.workbench import detection_training

    docker_dir = tmp_path / ".docker"
    docker_dir.mkdir()
    (docker_dir / "config.json").write_text(
        json.dumps({"auths": {"cr.example.test": {"auth": "dXNlcjpwYXNz"}}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    assert detection_training._image_registry("cr.example.test/project/image:tag") == "cr.example.test"
    assert detection_training._image_registry("npa-detection-training:latest") == ""
    assert detection_training._docker_auth_config("cr.example.test") == {
        "auths": {"cr.example.test": {"auth": "dXNlcjpwYXNz"}}
    }


@pytest.mark.skipif(os.environ.get("NPA_INTEGRATION_E2E") != "1", reason="Set NPA_INTEGRATION_E2E=1 to run detection-training e2e")
def test_detection_training_e2e_placeholder() -> None:
    assert os.environ["NPA_INTEGRATION_E2E"] == "1"


def _install_fake_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = types.ModuleType("torch")
    fake_torch.float32 = "float32"
    fake_torch.int64 = "int64"
    fake_torch.as_tensor = lambda value, dtype=None: FakeTensor(value)
    fake_torch.empty = lambda shape, dtype=None: FakeTensor([[]] if shape == (0, 4) else [])
    fake_torch.save = lambda payload, buffer: buffer.write(json.dumps(payload, default=str).encode("utf-8"))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)


def _synthetic_row() -> dict[str, Any]:
    image = Image.new("RGB", (8, 8), color=(20, 40, 60))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return {
        "image_bytes": buffer.getvalue(),
        "ann_bboxes": [[1.0, 1.0, 5.0, 6.0]],
        "ann_categories": [1],
    }


def _number(value: Any) -> float:
    if isinstance(value, FakeLoss):
        return value.value
    return float(value)
