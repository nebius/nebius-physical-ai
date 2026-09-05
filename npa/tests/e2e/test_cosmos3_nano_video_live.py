"""Real B200 acceptance: one 30s generation, then eight complete concurrent runs.

Run against the ready 16-replica deployment with NPA_INTEGRATION_E2E=1,
NPA_COSMOS3_VIDEO_ENDPOINT, NPA_COSMOS3_VIDEO_TOKEN and an owner-only
NPA_COSMOS3_VIDEO_EVIDENCE_DIR outside the checkout. No mock/smoke fallback.
"""

import os
from pathlib import Path

import pytest

from npa.workbench.cosmos.nano_video import run_batch

pytestmark = pytest.mark.e2e


@pytest.mark.timeout(0)
def test_one_then_eight_complete_generation_requests():
    required = ("NPA_COSMOS3_VIDEO_ENDPOINT", "NPA_COSMOS3_VIDEO_TOKEN", "NPA_COSMOS3_VIDEO_EVIDENCE_DIR")
    if any(not os.environ.get(name) for name in required):
        pytest.skip("requires the explicit Nano B200 live deployment configuration")
    root = Path(os.environ["NPA_COSMOS3_VIDEO_EVIDENCE_DIR"])
    for concurrency, name in ((1, "single"), (8, "concurrent-eight")):
        result = run_batch(endpoint=os.environ[required[0]], token=os.environ[required[1]],
                           output_dir=root / name, concurrency=concurrency)
        assert result["status"] == "succeeded"
        assert result["completed"] == concurrency
        assert result["distinct_replicas"] == concurrency
        assert result["peak_overlapping_rollouts"] == concurrency
