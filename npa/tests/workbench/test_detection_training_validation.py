"""Invalid detector overrides fail at every entry point before work begins."""

from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from typer.testing import CliRunner

from npa.cli.workbench.detection_training import app
from npa.sdk.workbench import detection_training as sdk
from npa.workbench.detection_training import service, training
from npa.workbench.detection_training.schemas import TrainRequest


@pytest.mark.parametrize("entry", ["training", "sdk", "cli", "service"])
def test_zero_epoch_override_rejected_before_training_work(monkeypatch, tmp_path, entry):
    output = tmp_path / "output"
    guards = []
    for name in (
        "compute_manifest_sha256", "storage_settings", "_start_wandb",
        "make_dataloader", "build_fasterrcnn_resnet50_fpn_v2",
        "train_one_epoch", "save_checkpoint", "write_bytes_uri", "write_json_uri",
    ):
        guard = Mock(side_effect=AssertionError(f"training work began: {name}"))
        monkeypatch.setattr(training, name, guard)
        guards.append(guard)
    status_callback = Mock()
    artifact_callback = Mock()
    payload = {"view": "materialized", "output_uri": str(output), "overrides": ["train.epochs=0"]}

    if entry == "training":
        # model_copy, like the old raw setattr override loop, bypasses validation.
        # Exercise the actual train_detector body, without replacing it with a mock.
        request = TrainRequest(view="materialized", output_uri=str(output)).model_copy(
            update={"epochs": 0, "overrides": ["train.epochs=0"]},
        )
        with pytest.raises(ValidationError) as caught:
            training.train_detector(request, status_callback=status_callback, artifact_callback=artifact_callback)
        errors = caught.value.errors()
    elif entry == "sdk":
        with pytest.raises(ValidationError) as caught:
            sdk.train(**payload)
        errors = caught.value.errors()
    elif entry == "cli":
        result = CliRunner().invoke(app, ["train", "--view", payload["view"], "--output-uri", str(output), "--override", "train.epochs=0"])
        assert result.exit_code != 0
        assert isinstance(result.exception, ValidationError)
        errors = result.exception.errors()
    else:
        with TestClient(service.create_app(auth_mode="none", state_dir=tmp_path / "state")) as client:
            response = client.post("/train", json=payload)
            assert response.status_code == 422
            errors = response.json()["detail"]
            assert client.get("/runs").json() == {"runs": []}

    assert len(errors) == 1
    assert errors[0]["loc"][-1] == "epochs"
    assert errors[0]["type"] == "greater_than_equal"
    assert errors[0]["msg"] == "Input should be greater than or equal to 1"
    for guard in guards:
        guard.assert_not_called()
    status_callback.assert_not_called()
    artifact_callback.assert_not_called()
    assert not output.exists()


@pytest.mark.parametrize(("override", "field", "error_type"), [
    ("epochs=-1", "epochs", "greater_than_equal"),
    ("train.epochs=invalid", "epochs", "int_parsing"),
    ("train.batch_size=0", "batch_size", "greater_than_equal"),
    ("optimizer.learning_rate=0", "learning_rate", "greater_than"),
    ("train.num_classes=1", "num_classes", "greater_than_equal"),
    ("view= ", "view", "string_too_short"),
    ("output_uri= ", "output_uri", "string_too_short"),
])
def test_all_overridden_fields_keep_schema_validation(override, field, error_type):
    with pytest.raises(ValidationError) as caught:
        TrainRequest(view="materialized", output_uri="output", overrides=[override])
    error = caught.value.errors()[0]
    assert error["loc"] == (field,)
    assert error["type"] == error_type


def test_valid_overrides_are_normalized_without_mutating_input():
    payload = {
        "view": "materialized", "output_uri": "output",
        "overrides": ["train.epochs=2", 'dataset.path=" data "', 'view=" slice "'],
    }
    request = TrainRequest.model_validate(payload)
    assert request.epochs == 2
    assert request.view == "slice"
    assert request.data_path == request.lance_uri == "data"
    assert payload["view"] == "materialized"
    assert "epochs" not in payload
