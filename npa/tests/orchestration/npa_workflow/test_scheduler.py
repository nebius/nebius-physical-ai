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


def test_scheduler_uses_interpreter_resolved_image_profile() -> None:
    spec = load_spec(SPECS / "sim2real.yaml")
    digest = "registry.example/npa/isaac@sha256:" + "a" * 64
    spec.config["isaac_image"] = digest
    spec.config["outer_iterations"] = "1"
    spec.config["inner_iterations"] = "1"
    plan = build_plan(spec, run_id="resolved-resource")
    step = next(item for item in plan.steps if item.state == "stage-07-rollouts")
    assert step.resources_profile["image"] == digest
    payload = build_scheduler_plan(spec, [step], run_id="resolved-resource")
    assert payload["tasks"][0]["image"] == digest
