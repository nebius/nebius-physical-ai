"""Exact argv and artifact contracts for Encord workflow specs."""

from __future__ import annotations

from pathlib import Path

from npa.orchestration.npa_workflow import build_plan, load_spec, validate_spec

ROOT = Path(__file__).resolve().parents[3]
WORKFLOWS = ROOT / "npa" / "workflows" / "workbench" / "npa-workflows"


def _plan(name: str):
    spec = load_spec(WORKFLOWS / name)
    validate_spec(spec)
    return spec, build_plan(spec, run_id="contract")


def _value(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def test_push_spec_uses_register_and_exact_receipt_path() -> None:
    spec, plan = _plan("encord-push.yaml")
    step = plan.steps[0]
    assert [step.state for step in plan.steps] == ["push"]
    assert _value(step.argv, "--transfer") == "register"
    assert _value(step.argv, "--output-path") == step.outputs[0]["uri"]
    assert step.outputs[0]["schema"] == "npa.encord.push_receipt.v1"
    assert spec.states["push"].terminal


def test_pull_spec_defaults_to_no_label_initialization() -> None:
    spec, plan = _plan("encord-pull.yaml")
    step = plan.steps[0]
    assert _value(step.argv, "--label-export") == "none"
    assert _value(step.argv, "--output-path").rstrip("/") + "/manifest.json" == step.outputs[0]["uri"]
    assert spec.states["pull"].terminal


def test_roundtrip_consumes_artifacts_only_in_terminal_verifier() -> None:
    spec, plan = _plan("encord-roundtrip-smoke.yaml")
    assert [step.state for step in plan.steps] == ["push", "pull", "verify"]
    push, pull, verify = plan.steps
    assert not spec.states["pull"].inputs
    assert spec.states["pull"].needs == ["push"]
    assert not spec.states["push"].terminal
    assert not spec.states["pull"].terminal
    assert spec.states["verify"].terminal
    assert _value(verify.argv, "--receipt-uri") == push.outputs[0]["uri"]
    assert _value(verify.argv, "--manifest-uri") == pull.outputs[0]["uri"]
    assert _value(verify.argv, "--output-path") == verify.outputs[0]["uri"]
    assert {item.schema for item in spec.states["verify"].inputs} == {
        "npa.encord.push_receipt.v1",
        "npa.encord.pull_manifest.v1",
    }


def test_all_encord_stages_are_cpu_only() -> None:
    for name in ("encord-push.yaml", "encord-pull.yaml", "encord-roundtrip-smoke.yaml"):
        spec = load_spec(WORKFLOWS / name)
        assert "accelerators" not in spec.resources["cpu"]
        assert all(state.resources == "cpu" for state in spec.states.values())
