from __future__ import annotations

from pathlib import Path

from npa.orchestration.npa_workflow import build_plan, load_spec
from npa.orchestration.npa_workflow.scheduler import build_scheduler_plan


REPO_ROOT = Path(__file__).resolve().parents[4]
SPECS = REPO_ROOT / "npa" / "workflows" / "workbench" / "npa-workflows"


def test_scheduler_plan_includes_resources() -> None:
    spec = load_spec(SPECS / "vlm-eval-single.yaml")
    plan = build_plan(spec, run_id="sched-1")
    payload = build_scheduler_plan(spec, plan.steps, run_id="sched-1")
    assert payload["tasks"]
    task = payload["tasks"][0]
    assert task["resources"]["accelerators"] == "H100:1"
    assert task["command"]
