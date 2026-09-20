"""SeedVR2 workflow reachability, handoff, image, and submit contracts."""

from __future__ import annotations

from pathlib import Path

import yaml

from npa.orchestration.npa_workflow import build_plan, load_spec, validate_spec
from npa.orchestration.npa_workflow.catalog import TOOL_CATALOG
from npa.orchestration.npa_workflow.skypilot_render import (
    TOOL_REF_IMAGE_TOOL,
    SkypilotRenderOptions,
    render_skypilot_yaml,
)
from npa.orchestration.npa_workflow.submit_matrix import SUBMIT_LIVE_MATRIX


ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / "workflows" / "testing" / "seedvr2-video-restoration.yaml"


def test_seedvr2_workflow_validates_and_reaches_every_real_operation() -> None:
    spec = load_spec(WORKFLOW)
    validate_spec(spec)
    plan = build_plan(spec, run_id="seedvr2-contract")
    assert [step.state for step in plan.steps] == [
        "probe",
        "restore",
        "verify",
        "review",
    ]
    assert [step.argv[3] for step in plan.steps] == [
        "probe",
        "restore",
        "verify",
        "review",
    ]
    assert spec.states["restore"].resources == "gpu"
    assert spec.states["verify"].resources == "gpu"
    assert spec.states["review"].terminal is True
    restore_argv = plan.steps[1].argv
    assert restore_argv[restore_argv.index("--probe-path") + 1].endswith("/probe.json")


def test_seedvr2_toolrefs_share_the_cli_and_path_contract() -> None:
    for verb in ("probe", "restore", "verify", "review"):
        entry = TOOL_CATALOG[f"workbench.seedvr2.{verb}"]
        assert entry.argv_template[:4] == ["npa", "workbench", "seedvr2", verb]
        assert "--input-path" in entry.argv_template
        assert "--output-path" in entry.argv_template
        assert "--run-id" in entry.argv_template
    assert TOOL_REF_IMAGE_TOOL["workbench.seedvr2"] == "seedvr2"


def test_seedvr2_workflow_is_live_submit_eligible() -> None:
    case = next(
        item
        for item in SUBMIT_LIVE_MATRIX
        if item.spec == "seedvr2-video-restoration.yaml"
    )
    assert case.image_tool == "seedvr2"
    assert case.plan_only is False
    assert case.secret_envs == ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")


def test_seedvr2_workflow_binds_every_stage_to_the_exact_image_digest() -> None:
    image = "registry.example/npa-seedvr2@sha256:" + "a" * 64
    spec = load_spec(WORKFLOW)
    rendered = render_skypilot_yaml(
        spec,
        build_plan(spec, run_id="seedvr2-digest"),
        run_id="seedvr2-digest",
        options=SkypilotRenderOptions(
            image_overrides={"*": image},
            materialize_registry_secrets=False,
        ),
    )
    tasks = [item for item in yaml.safe_load_all(rendered) if item and "envs" in item]

    assert len(tasks) == 4
    assert {task["resources"]["image_id"] for task in tasks} == {"docker:" + image}
    assert {task["envs"]["NPA_TASK_IMAGE"] for task in tasks} == {image}
