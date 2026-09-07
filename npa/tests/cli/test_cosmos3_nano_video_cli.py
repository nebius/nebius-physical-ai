"""Public S3 client contract and truthful fanout failures."""

import importlib.util
import json
import os
from pathlib import Path

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


def test_live_acceptance_parses_stdout_and_retains_stderr(monkeypatch, tmp_path):
    """Exercise the real acceptance harness with CPU-only workload boundaries."""
    path = Path(__file__).parents[1] / "e2e" / "test_cosmos3_nano_video_live.py"
    spec = importlib.util.spec_from_file_location("nano_video_live_harness", path)
    live = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(live)
    monkeypatch.setenv("NPA_COSMOS3_VIDEO_ENDPOINT", "http://127.0.0.1:8000")
    monkeypatch.setenv("NPA_COSMOS3_VIDEO_TOKEN", "test-token")
    monkeypatch.setenv("NPA_COSMOS3_VIDEO_EVIDENCE_DIR", str(tmp_path))
    monkeypatch.setenv("NPA_COSMOS3_SINGLE_OUTPUT_URI", "s3://example-bucket/single")
    monkeypatch.setenv("NPA_COSMOS3_FANOUT_OUTPUT_URI", "s3://example-bucket/fanout")
    seen = []

    def submit(*, concurrency, **kwargs):
        seen.append(concurrency)
        batch = Path(os.environ["NPA_COSMOS3_VIDEO_RECOVERY_DIR"]) / "npa-nano-video-cpu-fixture" / "batch"
        batch.mkdir(parents=True)
        (batch / "batch.json").write_text(json.dumps({
            "peak_overlapping_chunk_requests": concurrency, "fanout_verified": True,
        }))
        return {
            "status": "succeeded", "completed": concurrency,
            "distinct_replicas": concurrency, "peak_overlapping_rollouts": concurrency,
            "publication_verified": True,
        }

    monkeypatch.setattr(live, "nano_video_batch", submit)
    monkeypatch.setattr(nano_video, "submit_batch", submit)
    live.test_one_then_eight_complete_generation_requests(monkeypatch)
    assert seen == [1, 8]
    recovery = tmp_path / "concurrent-eight"
    stdout = (recovery / "cli-stdout.txt").read_text()
    stderr = (recovery / "cli-stderr.txt").read_text()
    assert json.loads(stdout)["completed"] == 8
    assert "command diagnostics were separated from JSON stdout" in stderr
    assert "command diagnostics" not in stdout
    with pytest.raises(json.JSONDecodeError):
        json.loads((recovery / "cli-output.txt").read_text())
