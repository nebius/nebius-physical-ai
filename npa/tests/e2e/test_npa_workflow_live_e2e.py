"""Live validation of NPA workflow specs against real infrastructure (optional)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from npa.orchestration.npa_workflow import build_plan, load_spec

pytestmark = pytest.mark.skipif(
    os.environ.get("NPA_INTEGRATION_E2E") != "1",
    reason="Set NPA_INTEGRATION_E2E=1 to run live NPA workflow spec checks.",
)

REPO_ROOT = Path(__file__).resolve().parents[4]
SPECS = REPO_ROOT / "npa" / "workflows" / "workbench" / "npa-workflows"


@pytest.mark.parametrize(
    "name",
    ["vlm-eval-single.yaml", "tokenfactory-rollout-judge.yaml", "sim2real-vlm-rl.yaml"],
)
def test_live_npa_workflow_specs_plan(name: str) -> None:
    """Ensure golden specs load and expand on the operator machine."""

    spec = load_spec(SPECS / name)
    plan = build_plan(spec, run_id="live-spec-check")
    assert plan.steps, name
