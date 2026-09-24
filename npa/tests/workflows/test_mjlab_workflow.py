"""Bind native MJLab stage arguments, image routing and result contracts."""

from pathlib import Path

from npa.orchestration.npa_workflow.interpreter import build_plan
from npa.orchestration.npa_workflow.spec import load_spec
from npa.orchestration.npa_workflow.skypilot_render import (
    tool_image_key,
    tool_vendor_interpreters,
)
from npa.deploy.images import UNVALIDATED_PUBLICATION_TOOLS, publicly_publishable_tools

ROOT = Path(__file__).resolve().parents[3]


def test_complete_workflow_uses_native_container_and_independent_seed():
    spec = load_spec(ROOT / "workflows/testing/mjlab-train-eval.yaml")
    plan = build_plan(spec, run_id="test")
    assert [step.tool_ref for step in plan.steps] == [
        "workbench.mjlab.train",
        "workbench.mjlab.eval",
        "workbench.mjlab.export",
        "workbench.mjlab.eval",
    ]
    for step in plan.steps:
        assert tool_image_key(step.tool_ref) == "mjlab"
        assert tool_vendor_interpreters(step.tool_ref) == ("/usr/local/bin/python",)
        assert "--score" not in step.argv
    argv = plan.steps[-1].argv
    assert argv[argv.index("--seed") + 1] == "43"
    assert "eval-seed-43" in argv[argv.index("--output-path") + 1]
    assert plan.steps[1].argv != argv


def test_unbuilt_image_cannot_enter_public_release_plan():
    assert "mjlab" in UNVALIDATED_PUBLICATION_TOOLS
    assert "mjlab" not in publicly_publishable_tools()
