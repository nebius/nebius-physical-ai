"""Verify that a public continuation anchor is frozen before any baseline learning."""

from types import SimpleNamespace

import pytest

from npa.workflows.field_failure import reference_demo, reference_demo_prepare


@pytest.mark.parametrize("coefficient", [0, -1.0, float("nan"), float("inf"), True])
def test_invalid_anchor_fails_before_fetching_public_inputs(monkeypatch, coefficient):
    monkeypatch.setattr(
        reference_demo_prepare,
        "materialize",
        lambda *args: pytest.fail("invalid learning settings fetched input data"),
    )
    with pytest.raises(ValueError):
        reference_demo_prepare.prepare_reference(
            SimpleNamespace(baseline_anchor_coefficient=coefficient)
        )


@pytest.mark.parametrize("coefficient", [None, 10.0])
def test_protocol_and_plan_freeze_same_anchor_without_changing_budget(
    tmp_path, monkeypatch, coefficient
):
    args = SimpleNamespace(
        baseline_anchor_coefficient=coefficient,
        sample_path="s3://example-bucket/sample",
        output_root="s3://example-bucket/run",
        navigation_image="registry.example/native@sha256:" + "a" * 64,
        reconstruction_image="registry.example/reconstruction@sha256:" + "b" * 64,
        baseline_iterations=1500,
        candidate_iterations=1500,
        num_envs=4000,
        episode_steps=300,
    )
    protocol = {"schema_version": "npa.field-failure.native-protocol.v1"}
    frozen = {}
    _stub_preparation(monkeypatch, tmp_path, protocol, frozen)
    plan = reference_demo_prepare.prepare_reference(args)
    expected = {} if coefficient is None else {"baseline_anchor_coefficient": 10.0}
    assert frozen["protocol"] == {**protocol, **expected}
    assert {key: plan[key] for key in expected} == expected
    assert ("baseline_anchor_coefficient" in plan) is (coefficient is not None)
    assert plan["baseline_iterations"] == plan["candidate_iterations"] == 1500
    assert plan["num_envs"] == 4000
    assert frozen["plan"] == plan


def _stub_preparation(monkeypatch, root, protocol, frozen):
    monkeypatch.setattr(reference_demo_prepare, "materialize", lambda *args: root)
    monkeypatch.setattr(
        reference_demo_prepare,
        "_prepare_inputs",
        lambda *args: (root, {"frozen": True}, dict(protocol)),
    )

    def publish_inputs(args, temporary, baseline, capture, actual):
        frozen["protocol"] = dict(actual)
        return {"protocol": {"sha256": "c" * 64}}

    monkeypatch.setattr(reference_demo_prepare, "_publish_inputs", publish_inputs)
    monkeypatch.setattr(
        reference_demo_prepare,
        "_publish",
        lambda uri, value: frozen.update(plan=dict(value)),
    )


def test_internal_cli_forwards_explicit_anchor_and_retains_optional_default(
    monkeypatch,
):
    parsed = []
    monkeypatch.setattr(reference_demo, "run_reference_stage", parsed.append)
    common = ["prepare", "--output-root", "s3://example-bucket/run", "--run-id", "test"]
    assert reference_demo.main(common) == 0
    assert reference_demo.main(common + ["--baseline-anchor-coefficient", "10.0"]) == 0
    assert parsed[0].baseline_anchor_coefficient is None
    assert parsed[1].baseline_anchor_coefficient == 10.0
