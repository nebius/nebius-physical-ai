"""Real B200 acceptance: one 30s generation, then eight complete concurrent runs.

Run against the ready 16-replica deployment with NPA_INTEGRATION_E2E=1,
NPA_COSMOS3_VIDEO_ENDPOINT, NPA_COSMOS3_VIDEO_TOKEN, distinct
NPA_COSMOS3_SINGLE_OUTPUT_URI / NPA_COSMOS3_FANOUT_OUTPUT_URI S3 prefixes,
normal NPA storage credentials, and an owner-only NPA_COSMOS3_VIDEO_EVIDENCE_DIR
outside the checkout. No mock/smoke fallback.
"""

import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.sdk.workbench.cosmos3 import nano_video_batch

pytestmark = pytest.mark.e2e


@pytest.mark.timeout(0)
def test_one_then_eight_complete_generation_requests(monkeypatch):
    required = (
        "NPA_COSMOS3_VIDEO_ENDPOINT", "NPA_COSMOS3_VIDEO_TOKEN",
        "NPA_COSMOS3_VIDEO_EVIDENCE_DIR", "NPA_COSMOS3_SINGLE_OUTPUT_URI",
        "NPA_COSMOS3_FANOUT_OUTPUT_URI",
    )
    if any(not os.environ.get(name) for name in required):
        pytest.skip("requires the explicit Nano B200 live deployment configuration")
    root = Path(os.environ["NPA_COSMOS3_VIDEO_EVIDENCE_DIR"])
    for concurrency, name, destination in (
        (1, "single", os.environ[required[3]]),
        (8, "concurrent-eight", os.environ[required[4]]),
    ):
        recovery = root / name
        recovery.mkdir(parents=True, exist_ok=False)
        monkeypatch.setenv("NPA_COSMOS3_VIDEO_RECOVERY_DIR", str(recovery))
        if concurrency == 1:
            result = nano_video_batch(output_path=destination, concurrency=concurrency)
        else:
            cli = CliRunner().invoke(app, [
                "workbench", "cosmos3", "nano-video-batch", "--concurrency", "8",
                "--output-path", destination,
            ])
            (recovery / "cli-output.txt").write_text(cli.output)
            assert cli.exit_code == 0, "Inspect retained CLI output and recovery artifacts"
            result = json.loads(cli.output)
        assert result["status"] == "succeeded"
        assert result["completed"] == concurrency
        assert result["distinct_replicas"] == concurrency
        assert result["peak_overlapping_rollouts"] == concurrency
        assert result["publication_verified"] is True
        manifests = list(recovery.glob("npa-nano-video-*/batch/batch.json"))
        assert len(manifests) == 1
        batch = json.loads(manifests[0].read_text())
        assert batch["peak_overlapping_chunk_requests"] == concurrency
        assert batch["fanout_verified"] is True
