"""Workflow contract for the GR00T LIBERO-X simulator evaluation path."""

from __future__ import annotations

from pathlib import Path

import yaml

from npa.orchestration.npa_workflow.interpreter import build_plan
from npa.orchestration.npa_workflow.scheduler import build_scheduler_plan
from npa.orchestration.npa_workflow.skypilot_render import SkypilotRenderOptions
from npa.orchestration.npa_workflow.spec import load_spec
from npa.orchestration.npa_workflow.submit import (
    merge_config_overrides,
    prepare_npa_workflow_for_submit,
)


ROOT = Path(__file__).resolve().parents[4]
SPEC_PATH = ROOT / "workflows/testing/groot-libero-x-closed-loop.yaml"
OBSERVED_SPEC_PATH = ROOT / "workflows/testing/groot-libero-x-observed-paired.yaml"
STATES = [
    "prepare_disjoint_evaluation",
    "run_nvidia_baseline_closed_loop",
    "run_derivative_closed_loop",
    "compare_closed_loop_metrics",
    "emit_verified_rollout_evidence",
]
TOOL_REFS = [
    "workflow.groot_libero_x.prepare_evaluation",
    "workbench.groot.libero_x_baseline",
    "workbench.groot.libero_x_derivative",
    "workflow.groot_libero_x.compare_closed_loop",
    "workflow.groot_libero_x.emit_evidence",
]


def test_closed_loop_workflow_has_five_connected_substantive_stages() -> None:
    spec = load_spec(SPEC_PATH)
    plan = build_plan(spec, run_id="groot-libero-x-contract")

    assert spec.api_version == "npa.workflow/v0.0.1"
    assert [step.state for step in plan.steps] == STATES
    assert [step.tool_ref for step in plan.steps] == TOOL_REFS
    assert spec.config["source_overlay"] is True
    assert spec.config["episodes_per_task"] == "10"
    assert spec.config["n_envs"] == "5"
    assert spec.config["max_episode_steps"] == "720"
    assert spec.config["n_action_steps"] == "8"
    assert spec.config["evaluation_accelerator"] == "H100:1"
    assert spec.states[STATES[-1]].terminal is True
    assert all(
        "closed-loop" not in step.description.lower() or step.tool_ref
        for step in spec.states.values()
    )

    baseline = plan.steps[1]
    derivative = plan.steps[2]
    assert (
        baseline.resources_profile["accelerators"]
        == spec.config["evaluation_accelerator"]
    )
    assert (
        derivative.resources_profile["accelerators"]
        == spec.config["evaluation_accelerator"]
    )
    configured_plan = build_plan(
        merge_config_overrides(spec, {"evaluation_accelerator": "RTXPRO6000:1"}),
        run_id="groot-libero-x-operator-gpu",
    )
    assert configured_plan.steps[1].resources_profile["accelerators"] == "RTXPRO6000:1"
    assert configured_plan.steps[2].resources_profile["accelerators"] == "RTXPRO6000:1"
    assert "--policy-name" in baseline.argv and "baseline" in baseline.argv
    assert "--policy-name" in derivative.argv and "derivative" in derivative.argv
    assert "--model-revision" in baseline.argv
    assert "--model-revision" in derivative.argv
    assert "--episodes-per-task" in baseline.argv
    assert "--n-action-steps" in derivative.argv

    previous_outputs: set[str] = set()
    for state in STATES:
        stage = spec.states[state]
        inputs = {artifact.uri for artifact in stage.inputs}
        if previous_outputs:
            assert inputs & previous_outputs, (
                f"{state} does not consume an earlier artifact"
            )
        previous_outputs.update(artifact.uri for artifact in stage.outputs)


def test_observed_paired_workflow_has_five_stages_without_held_out_input_claim() -> (
    None
):
    spec = load_spec(OBSERVED_SPEC_PATH)
    plan = build_plan(spec, run_id="groot-libero-x-observed-contract")

    assert spec.api_version == "npa.workflow/v0.0.1"
    assert len(plan.steps) == 5
    assert plan.steps[0].state == "prepare_observed_paired_evaluation"
    assert (
        plan.steps[0].tool_ref
        == "workflow.groot_libero_x.prepare_observed_paired_evaluation"
    )
    assert "training_task_manifest_uri" not in spec.config
    assert "observed_task_manifest_uri" in spec.config
    assert spec.config["evaluation_accelerator"] == "RTXPRO6000:1"
    assert "held-out" in str(spec.metadata["description"]).lower()
    assert "unknown" in spec.states[plan.steps[3].state].description.lower()


def test_closed_loop_workflow_renders_current_toolrefs_and_vendor_image(
    monkeypatch,
) -> None:
    monkeypatch.setenv("NPA_REGISTRY", "cr.ci.invalid/workbench")
    monkeypatch.setenv("NPA_PUBLIC_REGISTRY", "ghcr.io/nebius/nebius-physical-ai")
    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://example-bucket/source/npa")
    prepared = prepare_npa_workflow_for_submit(
        SPEC_PATH,
        run_id="groot-libero-x-render",
        render_options=SkypilotRenderOptions(
            registry="cr.ci.invalid/workbench", materialize_registry_secrets=False
        ),
    )
    try:
        scheduler = build_scheduler_plan(
            prepared.spec, prepared.plan.steps, run_id="groot-libero-x-render"
        )
        assert [task["name"] for task in scheduler["tasks"]] == STATES
        documents = [
            document
            for document in yaml.safe_load_all(prepared.skypilot_yaml_path.read_text())
            if document
        ]
        by_name = {document["name"]: document for document in documents[1:]}
        assert len(by_name) == len(STATES)
        for state in STATES:
            assert "groot_libero_x" in by_name[state]["run"]
        for state in STATES[1:3]:
            assert "/opt/groot/Isaac-GR00T/.venv/bin/python" in by_name[state]["setup"]
            assert "npa-groot" in by_name[state]["resources"]["image_id"]
        assert "av>=12,<17" in by_name[STATES[-1]]["setup"]
    finally:
        prepared.temp_dir.cleanup()
