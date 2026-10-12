"""Bind native MJLab stage arguments, image routing and result contracts."""

from pathlib import Path
import json

import pytest
import yaml

from npa.orchestration.npa_workflow.interpreter import build_plan
from npa.orchestration.npa_workflow.spec import load_spec
from npa.orchestration.npa_workflow.skypilot_render import (
    SkypilotRenderOptions,
    render_skypilot_yaml,
    tool_image_key,
    tool_vendor_interpreters,
)
from npa.deploy.images import VALIDATION_CANDIDATE_TOOLS, publicly_publishable_tools

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("name", ("mjlab-eval", "mjlab-train-eval", "mjlab-render"))
def test_examples_render_the_qualified_rtx_image_without_an_override(name):
    qualification = json.loads(
        (
            ROOT / "docs/workbench/validation/mjlab-native-hd-gpu-20261010.json"
        ).read_text()
    )
    spec = load_spec(ROOT / f"workflows/testing/{name}.yaml")
    plan = build_plan(spec, run_id="qualified-mjlab")
    header, *documents = list(
        yaml.safe_load_all(
            render_skypilot_yaml(
                spec,
                plan,
                run_id="qualified-mjlab",
                options=SkypilotRenderOptions(
                    materialize_registry_secrets=False, include_aws_endpoint=False
                ),
            )
        )
    )
    assert header == {"name": name, "execution": "serial"}
    expected = (
        "docker:ghcr.io/nebius/nebius-physical-ai/npa-mjlab@"
        + qualification["image"]["digest"]
    )
    assert len(documents) == len(plan.steps)
    for document in documents:
        assert document["resources"]["image_id"] == expected
        assert document["resources"]["accelerators"] == "RTXPRO6000:1"


def test_complete_workflow_uses_native_container_and_independent_seed():
    spec = load_spec(ROOT / "workflows/testing/mjlab-train-eval.yaml")
    plan = build_plan(spec, run_id="test")
    assert [step.tool_ref for step in plan.steps] == [
        "workbench.mjlab.train",
        "workbench.mjlab.render",
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
    rendered = plan.steps[1]
    assert "--video" in rendered.argv
    output = rendered.argv[rendered.argv.index("--output-path") + 1]
    assert {artifact["uri"] for artifact in rendered.outputs} == {
        output + name
        for name in ("mjlab_eval.json", "episodes.json", "rollout.mp4", "rollout.html")
    }


def test_operator_candidate_cannot_enter_public_release_plan():
    assert "mjlab" in VALIDATION_CANDIDATE_TOOLS
    assert "mjlab" not in publicly_publishable_tools()


def test_render_workflow_publishes_playable_video_and_portable_html():
    spec = load_spec(ROOT / "workflows/testing/mjlab-render.yaml")
    plan = build_plan(spec, run_id="test-render")
    assert len(plan.steps) == 1
    step = plan.steps[0]
    assert step.tool_ref == "workbench.mjlab.render"
    assert "--video" in step.argv
    assert tool_image_key(step.tool_ref) == "mjlab"
    output = step.argv[step.argv.index("--output-path") + 1]
    assert {artifact["uri"] for artifact in step.outputs} == {
        output + name
        for name in ("mjlab_eval.json", "episodes.json", "rollout.mp4", "rollout.html")
    }
