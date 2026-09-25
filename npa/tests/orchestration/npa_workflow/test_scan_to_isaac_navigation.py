"""Verify scan-to-Isaac graph, image routing, source overlay, and live registration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from npa.orchestration.npa_workflow.interpreter import build_plan
from npa.orchestration.npa_workflow.skypilot_render import (
    SkypilotRenderOptions,
    render_skypilot_yaml,
)
from npa.orchestration.npa_workflow.spec import load_spec
from npa.orchestration.npa_workflow.submit_matrix import SUBMIT_LIVE_MATRIX

ROOT = Path(__file__).resolve().parents[4]
SPEC = ROOT / "workflows/testing/scan-to-isaac-navigation.yaml"


def test_scene_workflow_hands_exact_assembly_to_native_physics() -> None:
    """Check executable modules and durable handoffs without live infrastructure.

    Args:
        None.
    Returns:
        None.
    Raises:
        AssertionError: Graph, inputs, or expected runtime differs.
    """
    spec = load_spec(SPEC)
    plan = build_plan(spec, run_id="scene-contract")
    prepare, physics = plan.steps
    assert [step.state for step in plan.steps] == ["prepare", "physics"]
    assert prepare.argv[:3] == [
        "/opt/venv/bin/python",
        "-m",
        "npa.workbench.nurec.navigation_scene",
    ]
    assert physics.argv[:3] == [
        "/isaac-sim/python.sh",
        "-m",
        "npa.workbench.nurec.navigation_probe",
    ]
    assert prepare.argv[prepare.argv.index("--input-path") + 1] == ""
    assert physics.argv[physics.argv.index("--runtime-image") + 1] == "tool://isaac-lab"
    assembled = prepare.argv[prepare.argv.index("--output-path") + 1]
    assert physics.argv[physics.argv.index("--input-path") + 1] == assembled
    assert {item["uri"] for item in prepare.outputs} == {
        f"{assembled}/scene.usdz",
        f"{assembled}/provenance.json",
    }
    assert {item["uri"] for item in physics.inputs} == {
        item["uri"] for item in prepare.outputs
    }


def test_scene_render_preserves_source_and_separate_cpu_rtx_images(monkeypatch) -> None:
    """Render isolated CPU and Isaac stages with the staged source overlay.

    Args:
        monkeypatch: Fixture isolating the staged source URI.
    Returns:
        None.
    Raises:
        AssertionError: Image, accelerator, interpreter, or source route is wrong.
    """
    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://example-bucket/source-fixture")
    spec = load_spec(SPEC)
    rendered = render_skypilot_yaml(
        spec,
        build_plan(spec, run_id="scene-render"),
        run_id="scene-render",
        options=SkypilotRenderOptions(
            registry="registry.example.invalid", materialize_registry_secrets=False
        ),
    )
    tasks = [item for item in yaml.safe_load_all(rendered) if item and "run" in item]
    assert len(tasks) == 2
    assert "npa-content-agents" in tasks[0]["resources"]["image_id"]
    assert "accelerators" not in tasks[0]["resources"]
    assert "npa-isaac-lab" in tasks[1]["resources"]["image_id"]
    assert tasks[1]["resources"]["accelerators"] == "RTXPRO6000:1"
    assert (
        "/isaac-sim/python.sh -m npa.workbench.nurec.navigation_probe"
        in tasks[1]["run"]
    )
    for task in tasks:
        assert task["envs"]["NPA_SRC_OVERLAY"] == "1"
        assert "/tmp/npa-src-overlay/src" in task["run"]


def test_scene_live_case_requires_operator_inputs_and_readiness_matches_bytes() -> None:
    """Keep live execution explicit and readiness bound to this exact workflow.

    Args:
        None.
    Returns:
        None.
    Raises:
        AssertionError: Registration or readiness incorrectly claims qualification.
    """
    case = next(case for case in SUBMIT_LIVE_MATRIX if case.spec == SPEC.name)
    assert case.tier == "gpu" and case.runtime and not case.plan_only
    assert case.rotation_skip and "operator" in case.skip_reason
    assert case.max_wait_seconds == 0
    readiness = json.loads(SPEC.with_suffix(".readiness.json").read_text())
    assert readiness["workflow_sha256"] == hashlib.sha256(SPEC.read_bytes()).hexdigest()
    assert readiness["prerequisites"]["target_runtime"]["status"] == "unverified"
