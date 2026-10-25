"""SeedVR2 workflow reachability, handoff, image, and submit contracts."""

from __future__ import annotations

from pathlib import Path

import yaml
import pytest

from npa.orchestration.npa_workflow import build_plan, load_spec, validate_spec
from npa.orchestration.npa_workflow.catalog import TOOL_CATALOG
from npa.orchestration.npa_workflow.skypilot_render import (
    TOOL_REF_IMAGE_TOOL,
    NpaWorkflowRenderError,
    SkypilotRenderOptions,
    build_skypilot_task_doc,
    render_skypilot_yaml,
)
from npa.orchestration.npa_workflow.submit_matrix import SUBMIT_LIVE_MATRIX
from npa.orchestration.npa_workflow.submit import merge_config_overrides


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


@pytest.mark.parametrize("gpu,mode", [("H100", "sample"), ("B200", "posterior-mode")])
def test_resource_and_validation_contract_are_bound_to_same_override(gpu, mode):
    spec = merge_config_overrides(
        load_spec(WORKFLOW), {"seedvr2_gpu": gpu, "seedvr2_conditioning_mode": mode}
    )
    plan = build_plan(spec, run_id="seedvr2-contract")
    argv = plan.steps[1].argv
    assert argv[argv.index("--expected-gpu") + 1] == gpu
    assert argv[argv.index("--conditioning-mode") + 1] == mode
    rendered = render_skypilot_yaml(
        spec,
        plan,
        run_id="seedvr2-contract",
        options=SkypilotRenderOptions(
            image_overrides={"*": "registry.example/seedvr2@sha256:" + "a" * 64},
            materialize_registry_secrets=False,
        ),
    )
    tasks = [task for task in yaml.safe_load_all(rendered) if task and "envs" in task]
    assert [tasks[index]["resources"]["accelerators"] for index in (1, 2)] == [
        gpu + ":1"
    ] * 2


def test_workflow_defaults_remain_h100_sample():
    spec = load_spec(WORKFLOW)
    assert spec.config["seedvr2_gpu"] == "H100"
    assert spec.config["seedvr2_conditioning_mode"] == "sample"


@pytest.mark.parametrize("stage", range(4))
@pytest.mark.parametrize("default_setup", [True, False])
@pytest.mark.parametrize(
    "source",
    ["config", "ambient", "config-with-ambient-false", "baked-flag", "dependencies"],
)
def test_every_stage_rejects_overlays(monkeypatch, stage, default_setup, source):
    config = {}
    if source in {"config", "config-with-ambient-false", "baked-flag"}:
        config["source_overlay"] = True
    if source == "ambient":
        monkeypatch.setenv("NPA_SRC_OVERLAY", "1")
    if source == "config-with-ambient-false":
        monkeypatch.setenv("NPA_SRC_OVERLAY", "0")
    if source == "baked-flag":
        config.update(require_baked_npa=True, source_sha="b" * 40)
    if source == "dependencies":
        config["pip_extra"] = "unreviewed-package"
    spec = merge_config_overrides(load_spec(WORKFLOW), config)
    step = build_plan(spec, run_id="overlay-negative").steps[stage]
    with pytest.raises(NpaWorkflowRenderError, match="SeedVR2.*forbids"):
        build_skypilot_task_doc(
            spec,
            step,
            run_id="overlay-negative",
            options=SkypilotRenderOptions(
                image_overrides={"*": "registry.example/seedvr2@sha256:" + "a" * 64},
                materialize_registry_secrets=False,
                default_setup=default_setup,
            ),
        )


def test_seedvr2_does_not_stage_source_or_install_dependencies(monkeypatch):
    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://example-bucket/alternate-source")
    spec = load_spec(WORKFLOW)
    rendered = render_skypilot_yaml(
        spec,
        build_plan(spec, run_id="baked-only"),
        run_id="baked-only",
        options=SkypilotRenderOptions(
            image_overrides={"*": "registry.example/seedvr2@sha256:" + "a" * 64},
            materialize_registry_secrets=False,
        ),
    )
    tasks = [task for task in yaml.safe_load_all(rendered) if task and "envs" in task]
    assert len(tasks) == 4
    for task in tasks:
        assert "NPA_SRC_S3_URI" not in task["envs"]
        assert "NPA_SRC_OVERLAY" not in task["envs"]
        assert "pip install" not in task["setup"]
        assert "_require_baked_source()" in task["setup"]
        assert "_require_baked_source()" in task["run"]
        assert "/opt/npa-venv/bin/python" in task["setup"]


@pytest.mark.parametrize(
    "image",
    ["registry.example/seedvr2:mutable", "namespace/seedvr2@sha256:" + "a" * 64],
)
def test_seedvr2_requires_registry_qualified_immutable_image(image):
    spec = load_spec(WORKFLOW)
    with pytest.raises(NpaWorkflowRenderError, match="registry-qualified immutable"):
        render_skypilot_yaml(
            spec,
            build_plan(spec, run_id="mutable-negative"),
            run_id="mutable-negative",
            options=SkypilotRenderOptions(
                image_overrides={"*": image},
                materialize_registry_secrets=False,
            ),
        )
