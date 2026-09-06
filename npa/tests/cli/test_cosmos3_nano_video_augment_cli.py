"""Augmentation CLI/SDK parity and non-generating recovery contracts."""

import json

import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.sdk.workbench import cosmos3 as sdk
from npa.workbench.cosmos import nano_video_augment_client as client


def test_augmentation_cli_sdk_parameter_parity(monkeypatch):
    observed = []

    def submit(**kwargs):
        observed.append(kwargs)
        return {"status": "succeeded", "publication_verified": True}

    monkeypatch.setattr(client, "submit_augmentation", submit)
    result = CliRunner().invoke(app, [
        "workbench", "cosmos3", "nano-video-augment", "--input-path", "s3://example-bucket/source.mp4",
        "--output-path", "s3://example-bucket/result", "--prompt", "Warm dim warehouse",
        "--control-guidance", "1.25", "--chunk-frames", "81", "--guidance-scale", "4",
    ])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["publication_verified"] is True
    sdk.nano_video_augment(input_path="s3://example-bucket/source.mp4", output_path="s3://example-bucket/result",
                           prompt="Warm dim warehouse", control_guidance=1.25, chunk_frames=81, guidance_scale=4)
    assert observed[0] == observed[1]
    assert "strength" not in observed[0] and "control_weight" not in observed[0]


def test_recovery_cli_sdk_only_call_recovery(monkeypatch):
    observed = []
    monkeypatch.setattr(client, "submit_augmentation", lambda **kwargs: pytest.fail("recovery generated"))

    def recover(**kwargs):
        observed.append(kwargs)
        return {"status": "pending", "generation_status": "unknown", "publication_verified": False}

    monkeypatch.setattr(client, "recover_augmentation", recover)
    result = CliRunner().invoke(app, ["workbench", "cosmos3", "nano-video-augment-recover",
                                     "--output-path", "s3://example-bucket/result"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["generation_status"] == "unknown"
    sdk.nano_video_augment_recover(output_path="s3://example-bucket/result")
    assert observed[0] == observed[1]


@pytest.mark.parametrize("path", ["/tmp/input.mp4", "file:///tmp/input.mp4", "https://example.com/video.mp4",
                                  "https://huggingface.co/datasets/example/video"])
def test_cli_rejects_non_s3_input_before_client(monkeypatch, path):
    monkeypatch.setattr(client, "submit_augmentation", lambda **kwargs: pytest.fail("invalid input reached client"))
    result = CliRunner().invoke(app, ["workbench", "cosmos3", "nano-video-augment", "--input-path", path,
                                     "--output-path", "s3://example-bucket/result", "--prompt", "test"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["status"] == "failed"


def test_exception_text_never_leaks_to_json(monkeypatch):
    def failed(**kwargs):
        raise RuntimeError("private endpoint and authorization data")

    monkeypatch.setattr(client, "recover_augmentation", failed)
    result = CliRunner().invoke(app, ["workbench", "cosmos3", "nano-video-augment-recover",
                                     "--output-path", "s3://example-bucket/result"])
    assert result.exit_code == 1
    assert json.loads(result.stdout) == {"status": "failed", "error_type": "RuntimeError"}
    assert "private endpoint" not in result.output
