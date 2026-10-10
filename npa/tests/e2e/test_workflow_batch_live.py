"""Opt-in acceptance of batch admission and durable restart on operator capacity."""

import os
from pathlib import Path

import pytest

from npa.orchestration.npa_workflow.batch import run_batch
from npa.orchestration.npa_workflow.batch_plan import plan_batch

pytestmark = pytest.mark.e2e_pipeline


def test_workflow_batch_live():
    if os.environ.get("NPA_INTEGRATION_E2E") != "1":
        pytest.skip("set NPA_INTEGRATION_E2E=1 for real workflow execution")
    names = (
        "NPA_E2E_BATCH_MANIFEST",
        "NPA_E2E_BATCH_STATE_DIR",
        "NPA_E2E_BATCH_CONCURRENCY",
    )
    if any(not os.environ.get(name) for name in names):
        pytest.skip(
            "supply an operator-owned batch manifest, state directory, and concurrency"
        )
    manifest = Path(os.environ["NPA_E2E_BATCH_MANIFEST"])
    state_dir = Path(os.environ["NPA_E2E_BATCH_STATE_DIR"])
    concurrency = int(os.environ["NPA_E2E_BATCH_CONCURRENCY"])
    plan = plan_batch(manifest)
    assert len(plan["runs"]) >= 2, (
        "Use multiple real inputs to exercise batch admission"
    )
    result = run_batch(manifest, state_dir=state_dir, max_concurrent_runs=concurrency)
    assert len(result["runs"]) == len(plan["runs"])
    assert all(row["status"] == "succeeded" for row in result["runs"].values())
    replay = run_batch(
        manifest, state_dir=state_dir, max_concurrent_runs=concurrency, resume=True
    )
    assert all(row["status"] == "succeeded" for row in replay["runs"].values())
    assert all(
        replay["runs"][key]["attempt"] == row["attempt"] + 1
        for key, row in result["runs"].items()
    )
    assert all(
        replay["runs"][key]["stdout"] != row["stdout"]
        for key, row in result["runs"].items()
    )
