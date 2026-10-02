"""Verify resource snapshots share batch rendering and selector validation."""

from pathlib import Path

import pytest

from npa.orchestration.npa_workflow import build_plan, load_spec, skypilot_render
from npa.orchestration.npa_workflow.runtime import _resource_profiles_for_steps
from npa.orchestration.npa_workflow.skypilot_render import (
    NpaWorkflowRenderError,
    SkypilotRenderOptions,
    build_skypilot_task_doc,
)


SPEC_PATH = (
    Path(__file__).resolve().parents[4]
    / "workflows"
    / "testing"
    / "bdd100k-pipeline.yaml"
)


def test_resource_snapshot_validates_selectors_once_and_preserves_profiles(
    mocker, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = load_spec(SPEC_PATH)
    steps = build_plan(spec, run_id="resource-batch").steps
    options = SkypilotRenderOptions(materialize_registry_secrets=False)
    monkeypatch.setenv("NPA_WORKFLOW_GPU_ACCELERATOR", "H200:1")
    monkeypatch.setenv("NPA_WORKFLOW_GPU_MEMORY", "96Gi")
    expected = {}
    for step in steps:
        task = build_skypilot_task_doc(
            spec, step, run_id="resource-batch", options=options
        )
        expected[step.state] = {
            key: value
            for key, value in task["resources"].items()
            if key not in {"image_id", "image_login_config"}
        }
    validate = mocker.spy(skypilot_render, "validate_image_override_selectors")

    actual = _resource_profiles_for_steps(spec, steps, options, "resource-batch")

    assert len(steps) >= 10
    assert actual == expected
    assert any(profile.get("accelerators") == "H200:1" for profile in actual.values())
    validate.assert_called_once_with(spec, options)


@pytest.mark.parametrize("selector", ["workbench.foreign.tool", "workbench.*"])
def test_resource_snapshot_rejects_bad_selector_before_task_preparation(
    mocker, selector: str
) -> None:
    spec = load_spec(SPEC_PATH)
    steps = build_plan(spec, run_id="invalid-resource-selector").steps
    prepare = mocker.patch(
        "npa.orchestration.npa_workflow.skypilot_render.build_scheduler_task",
        side_effect=AssertionError("task preparation must follow selector validation"),
    )
    options = SkypilotRenderOptions(
        image_overrides={selector: "registry.example/custom:1"},
        materialize_registry_secrets=False,
    )

    with pytest.raises(NpaWorkflowRenderError, match="image override selector"):
        _resource_profiles_for_steps(spec, steps, options, "invalid-resource-selector")

    prepare.assert_not_called()
