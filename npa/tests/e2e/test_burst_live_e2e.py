"""Live e2e coverage for the burst multi-node SkyPilot path.

This test is skip-by-default. Enable it with NPA_INTEGRATION_E2E=1,
NPA_E2E_BURST=1, NPA_BURST_E2E_IMAGE, and NPA_BURST_E2E_GPU_PER_NODE.
"""

from __future__ import annotations

import os
import time
import uuid

import pytest

from npa import burst


pytestmark = [pytest.mark.e2e, pytest.mark.e2e_skypilot, pytest.mark.gpu]

NONTERMINAL = {"PENDING", "STARTING", "RUNNING", "RECOVERING"}


@pytest.fixture(autouse=True)
def _require_live_burst() -> None:
    if os.environ.get("NPA_INTEGRATION_E2E") != "1":
        pytest.skip("NPA_INTEGRATION_E2E not set")
    if os.environ.get("NPA_E2E_BURST") != "1":
        pytest.skip("NPA_E2E_BURST not set")
    if not os.environ.get("NPA_BURST_E2E_IMAGE"):
        pytest.skip("NPA_BURST_E2E_IMAGE not set")
    if not os.environ.get("NPA_BURST_E2E_GPU_PER_NODE"):
        pytest.skip("NPA_BURST_E2E_GPU_PER_NODE not set")


def test_burst_two_node_job_reaches_running_and_reports_distributed_env() -> None:
    name = f"npa-burst-e2e-{uuid.uuid4().hex[:8]}"
    handle = burst.submit(
        image=os.environ["NPA_BURST_E2E_IMAGE"],
        num_nodes=int(os.environ.get("NPA_BURST_E2E_NODES", "2")),
        gpu_per_node=os.environ["NPA_BURST_E2E_GPU_PER_NODE"],
        entrypoint=(
            "python -c \"import os; "
            "print('BURST_E2E_PROCESS rank=%s world_size=%s master_addr=%s' % "
            "(os.environ.get('RANK'), os.environ.get('WORLD_SIZE'), os.environ.get('MASTER_ADDR')))\""
        ),
        name=name,
    )

    seen_status = ""
    deadline = time.monotonic() + int(os.environ.get("NPA_BURST_E2E_MAX_WAIT_SECONDS", "1800"))
    while time.monotonic() < deadline:
        current = burst.status(handle)
        seen_status = current.status
        if current.status == "RUNNING":
            break
        if current.status not in NONTERMINAL:
            pytest.fail(f"burst job reached terminal status before running: {current}")
        time.sleep(int(os.environ.get("NPA_BURST_E2E_POLL_SECONDS", "30")))
    else:
        pytest.fail(f"burst job did not reach RUNNING; last status={seen_status}")

    log_text = burst.logs(handle, follow=False, tail=200).text
    assert "NPA_BURST_DISTRIBUTED rank=0" in log_text
    assert "NPA_BURST_DISTRIBUTED rank=1" in log_text
    assert "world_size=" in log_text
    assert "master_addr=" in log_text
