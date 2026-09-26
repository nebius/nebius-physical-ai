"""Check the executable field-failure graph, render identities, and catalog registration."""

from pathlib import Path

import yaml

from npa.orchestration.npa_workflow.interpreter import build_plan
from npa.orchestration.npa_workflow.skypilot_render import (
    SkypilotRenderOptions,
    render_skypilot_yaml,
)
from npa.orchestration.npa_workflow.spec import load_spec
from npa.orchestration.npa_workflow.submit_matrix import SUBMIT_LIVE_MATRIX

_ROOT = Path(__file__).resolve().parents[4]
_SPEC = _ROOT / "workflows/testing/field-failure-policy-improvement.yaml"


def test_complete_graph_renders_each_real_stage_with_s3_handoffs(tmp_path, monkeypatch):
    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://fixture/source/npa")
    source = yaml.safe_load(_SPEC.read_text())
    image = "registry.example.invalid/navigation@sha256:" + "a" * 64
    for name in ["reconstruction", "training", "evaluation"]:
        source["config"][name + "_image"] = image
    for operation in ["reconstruct", "train", "evaluate"]:
        source["config"][operation + "_adapter"] = "navigation_adapter:" + operation
    path = tmp_path / _SPEC.name
    path.write_text(yaml.safe_dump(source))
    spec = load_spec(path)
    plan = build_plan(spec, run_id="contract-run")
    rendered = render_skypilot_yaml(
        spec,
        plan,
        run_id="contract-run",
        options=SkypilotRenderOptions(
            registry="registry.example.invalid", materialize_registry_secrets=False
        ),
    )
    tasks = [task for task in yaml.safe_load_all(rendered) if task and "run" in task]
    assert len(tasks) == 6
    for stage, task in zip(source["states"], tasks):
        assert "npa.workflows.field_failure " + stage in task["run"]
        assert "s3://example-bucket/field-failure/contract-run" in task["run"]
    for task in tasks[1:5]:
        assert task["resources"]["image_id"] == "docker:" + image
        assert "--runtime-image " + image in task["run"]
    assert tasks[3]["resources"] == tasks[4]["resources"]


def test_matrix_registration_is_executable_but_not_claimed_accepted():
    case = next(c for c in SUBMIT_LIVE_MATRIX if c.spec == _SPEC.name)
    assert case.runtime and not case.plan_only and case.tier == "gpu"
    assert case.rotation_skip and "operator" in case.skip_reason
    assert "no GPU acceptance" in case.notes


def test_no_training_stage_receives_heldout_input_or_report():
    source = yaml.safe_load(_SPEC.read_text())
    assert source["states"]["train"]["next"] == "baseline-evaluate"
    assert source["states"]["compare"]["terminal"]
    assert source["states"]["compare"]["writesDecision"]
    assert not any("deploy" in name or "promote" in name for name in source["states"])


def test_readiness_is_hash_bound_and_does_not_claim_live_acceptance():
    import hashlib
    import json

    record = json.loads(_SPEC.with_suffix(".readiness.json").read_text())
    assert record["workflow_sha256"] == hashlib.sha256(_SPEC.read_bytes()).hexdigest()
    assert record["planning"]["validation"]["status"] == "verified"
    assert record["prerequisites"]["source_image"]["status"] == "blocked"
    assert record["prerequisites"]["target_runtime"]["status"] == "unverified"
