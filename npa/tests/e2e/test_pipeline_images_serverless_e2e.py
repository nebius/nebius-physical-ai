"""Serverless e2e for renamed pipeline images (envgen, reference-policy, loop-eval).

These tests submit real Nebius Serverless Jobs in the published container
images and assert capability markers in job logs — not import/--help smokes.

Requires:
  NPA_INTEGRATION_E2E=1
  NPA_E2E_SERVERLESS_PROJECT=<nebius-project-id>
  ~/.npa/credentials.yaml with S3 + HF tokens
"""

from __future__ import annotations

import os
import re
import time
import uuid

import pytest

from npa.deploy.images import CONTAINER_IMAGE_NAMES, container_image_for_tool
from npa.smoke.capabilities import GOLDEN_EVAL_CAPABILITIES
from npa.smoke.manifest import container
from npa.smoke.serverless_runner import submit_golden_eval

PIPELINE_TOOLS = ("envgen", "reference-policy", "loop-eval")

# Log substrings that prove the functional golden eval ran inside the container.
CAPABILITY_LOG_MARKERS: dict[str, tuple[str, ...]] = {
    "envgen": (
        "[PASS] raw env generation",
        "[PASS] genesis cuda step",
    ),
    "reference-policy": (
        "policy_variant=reference",
        "[PASS] raw env generation",
        "[PASS] genesis cuda step",
    ),
    "loop-eval": (
        "[PASS] cuda available",
        "[PASS] franka pick-place rollout",
    ),
}

pytestmark = pytest.mark.e2e_serverless


@pytest.fixture(autouse=True)
def _require_pipeline_images_e2e() -> None:
    if os.environ.get("NPA_INTEGRATION_E2E") != "1":
        pytest.skip("NPA_INTEGRATION_E2E not set")
    if not os.environ.get("NPA_E2E_SERVERLESS_PROJECT"):
        pytest.skip("NPA_E2E_SERVERLESS_PROJECT not set")


@pytest.mark.parametrize("tool", PIPELINE_TOOLS)
def test_pipeline_image_uses_renamed_registry_ref(tool: str) -> None:
    image = container_image_for_tool(tool)
    expected_name = CONTAINER_IMAGE_NAMES[tool]
    assert f"/{expected_name}:" in image, image
    assert "sim2real" not in image.split("/")[-1]


@pytest.mark.parametrize("tool", PIPELINE_TOOLS)
def test_pipeline_golden_eval_is_capability_not_help(tool: str) -> None:
    spec = container(tool)
    command = spec.golden_eval.command
    assert "--help" not in command
    assert command.strip()
    assert tool in GOLDEN_EVAL_CAPABILITIES
    assert CAPABILITY_LOG_MARKERS[tool]


@pytest.mark.parametrize("tool", PIPELINE_TOOLS)
def test_pipeline_image_serverless_capability_e2e(tool: str) -> None:
    """Run the container functional golden eval on Nebius Serverless and read logs."""

    project_id = os.environ["NPA_E2E_SERVERLESS_PROJECT"]
    gpu = os.environ.get(f"NPA_E2E_{tool.upper().replace('-', '_')}_GPU") or os.environ.get(
        "NPA_E2E_PIPELINE_GPU", "h100"
    )
    timeout = os.environ.get("NPA_E2E_PIPELINE_TIMEOUT", "45m")
    poll_ceiling = float(os.environ.get("NPA_E2E_PIPELINE_POLL_CEILING_S", "3600"))

    image = container_image_for_tool(tool)
    assert "npa-sim2real" not in image

    run_id = f"pipeline-e2e-{tool}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid.uuid4().hex[:6]}"
    print(f"submit tool={tool} image={image} project={project_id} run={run_id}", flush=True)

    detail = submit_golden_eval(
        tool,
        gpu_type=gpu,
        project_id=project_id,
        timeout=timeout,
        poll_ceiling_s=poll_ceiling,
        on_state_change=lambda job: print(f"  {tool} status={getattr(job, 'status', '?')}", flush=True),
    )

    print(f"result tool={tool} ok={detail.get('ok')} status={detail.get('status')}", flush=True)
    log_tail = str(detail.get("log_tail") or "")
    if log_tail:
        print(f"log_tail[{tool}]:\n{log_tail[-4000:]}", flush=True)

    assert detail.get("image") == image
    assert detail.get("ok"), (
        f"{tool} serverless job failed status={detail.get('status')!r} "
        f"job_id={detail.get('job_id')!r} log_tail={log_tail[-500:]!r}"
    )

    combined = log_tail.lower()
    missing = [marker for marker in CAPABILITY_LOG_MARKERS[tool] if marker.lower() not in combined]
    assert not missing, (
        f"{tool} job succeeded but capability markers missing {missing}; "
        f"log_tail={log_tail[-1000:]!r}"
    )

    # Reject accidental help-only or import-only passes.
    assert not re.search(r"\busage:\s", combined, re.IGNORECASE)
