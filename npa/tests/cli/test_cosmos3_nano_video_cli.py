"""Public S3 client contract and truthful fanout failures."""

import json

import pytest
from typer.testing import CliRunner

from npa.cli.workbench.cosmos3 import app
from npa.sdk.workbench import cosmos3 as sdk
from npa.workbench.cosmos import nano_video


@pytest.mark.parametrize("output_path", ["/tmp/output", "file:///tmp/output", "https://example.com/output"])
def test_local_or_http_handoff_is_rejected_before_generation(monkeypatch, output_path):
    def never(**kwargs):
        pytest.fail("invalid public path reached generation")

    monkeypatch.setattr(nano_video, "submit_batch", never)
    result = CliRunner().invoke(app, ["nano-video-batch", "--concurrency", "8", "--output-path", output_path])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["status"] == "failed"


def test_cli_and_sdk_call_same_client_and_failed_fanout_exits_nonzero(monkeypatch):
    seen = []

    def submit(**kwargs):
        seen.append(kwargs)
        return {"status": "failed", "completed": 7, "distinct_replicas": 7}

    monkeypatch.setattr(nano_video, "submit_batch", submit)
    result = CliRunner().invoke(app, ["nano-video-batch", "--concurrency", "8", "--output-path", "s3://example-bucket/batch"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["completed"] == 7
    response = sdk.nano_video_batch(output_path="s3://example-bucket/batch", concurrency=8)
    assert response["status"] == "failed"
    assert seen[0] == seen[1]


def test_provider_exception_does_not_disclose_endpoint_or_credentials(monkeypatch):
    def failed(**kwargs):
        raise RuntimeError("private deployment endpoint and secret-shaped detail")

    monkeypatch.setattr(nano_video, "submit_batch", failed)
    result = CliRunner().invoke(app, ["nano-video-batch", "--concurrency", "1", "--output-path", "s3://example-bucket/batch"])
    assert result.exit_code == 1
    assert json.loads(result.stdout) == {"status": "failed", "error_type": "RuntimeError"}
    assert "private deployment" not in result.output
