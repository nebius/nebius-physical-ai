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
# Terminal statuses that mean the GPU family could not be scheduled (no
# capacity / not offered on this cloud), as opposed to a real job failure.
# These rotate to the next candidate and, if every candidate is unavailable,
# skip rather than fail.
CAPACITY_UNAVAILABLE = {"FAILED_PRECHECKS", "FAILED_NO_RESOURCE", "CANCELLED"}


@pytest.fixture(autouse=True)
def _require_live_burst() -> None:
    if os.environ.get("NPA_INTEGRATION_E2E") != "1":
        pytest.skip("NPA_INTEGRATION_E2E not set")
    if os.environ.get("NPA_E2E_BURST") != "1":
        pytest.skip("NPA_E2E_BURST not set")
    if not os.environ.get("NPA_BURST_E2E_IMAGE"):
        pytest.skip("NPA_BURST_E2E_IMAGE not set")
    if not _gpu_per_node_candidates():
        pytest.skip("NPA_BURST_E2E_GPU_PER_NODE not set")


def _remap_accelerator(accelerator: str) -> str:
    """Apply the optional live accelerator remap, e.g. ``L40S:1=RTXPRO...:1``.

    Regions/GPU types are never hard-coded here: the remap is supplied via
    ``NPA_E2E_ACCELERATOR_REMAP`` so a run can retarget onto whatever GPU family
    the live project actually has capacity for (RTX, L40S, H100, ...).
    """

    remap = os.environ.get("NPA_E2E_ACCELERATOR_REMAP", "").strip()
    for pair in remap.split(","):
        pair = pair.strip()
        if "=" not in pair:
            continue
        src, dst = (part.strip() for part in pair.split("=", 1))
        if src and dst and src == accelerator:
            return dst
    return accelerator


def _gpu_per_node_candidates() -> list[str]:
    """Ordered, de-duplicated GPU-per-node candidates to try in turn.

    ``NPA_BURST_E2E_GPU_PER_NODE`` may be a comma-separated fallback list (for
    example ``RTXPRO-6000-BLACKWELL-SERVER-EDITION:1,L40S:1``). Each entry is run
    through the accelerator remap so legacy specs retarget onto available GPUs.
    """

    raw = os.environ.get("NPA_BURST_E2E_GPU_PER_NODE", "")
    candidates: list[str] = []
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        # Try the remapped family first (e.g. RTX), then the original request
        # as a fallback so a VM-native GPU (e.g. L40S) still gets a chance.
        for candidate in (_remap_accelerator(entry), entry):
            if candidate and candidate not in candidates:
                candidates.append(candidate)
    return candidates


def test_burst_two_node_job_reaches_running_and_reports_distributed_env() -> None:
    num_nodes = int(os.environ.get("NPA_BURST_E2E_NODES", "2"))
    poll_seconds = int(os.environ.get("NPA_BURST_E2E_POLL_SECONDS", "30"))
    max_wait = int(os.environ.get("NPA_BURST_E2E_MAX_WAIT_SECONDS", "1800"))
    candidates = _gpu_per_node_candidates()

    per_candidate_wait = max(max_wait // max(len(candidates), 1), poll_seconds * 2)
    last_status = ""
    non_scheduling: list[str] = []
    entrypoint = (
        "python -c \"import os; "
        "print('BURST_E2E_PROCESS rank=%s world_size=%s master_addr=%s' % "
        "(os.environ.get('RANK'), os.environ.get('WORLD_SIZE'), os.environ.get('MASTER_ADDR')))\""
    )

    for gpu_per_node in candidates:
        name = f"npa-burst-e2e-{uuid.uuid4().hex[:8]}"
        handle = burst.submit(
            image=os.environ["NPA_BURST_E2E_IMAGE"],
            num_nodes=num_nodes,
            gpu_per_node=gpu_per_node,
            entrypoint=entrypoint,
            name=name,
        )
        reached_running = False
        terminal_failure = False
        deadline = time.monotonic() + per_candidate_wait
        while time.monotonic() < deadline:
            current = burst.status(handle)
            last_status = (current.status or "").upper()
            if last_status == "RUNNING":
                reached_running = True
                break
            if last_status not in NONTERMINAL:
                # Terminal for this GPU family; capacity-unavailable rotates,
                # a genuine failure is recorded.
                terminal_failure = last_status not in CAPACITY_UNAVAILABLE
                break
            time.sleep(poll_seconds)

        if reached_running:
            log_text = burst.logs(handle, follow=False, tail=200).text
            assert "NPA_BURST_DISTRIBUTED rank=0" in log_text
            assert "NPA_BURST_DISTRIBUTED rank=1" in log_text
            assert "world_size=" in log_text
            assert "master_addr=" in log_text
            return

        if terminal_failure:
            pytest.fail(
                f"burst job failed on {gpu_per_node}: status={last_status}"
            )
        # Never scheduled (still non-terminal at deadline) or capacity-unavailable.
        non_scheduling.append(f"{gpu_per_node}={last_status}")

    pytest.skip(
        "burst job could not be scheduled on any configured GPU family "
        "(no capacity / GPU not offered): " + ", ".join(non_scheduling)
    )
