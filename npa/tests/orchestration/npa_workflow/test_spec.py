from __future__ import annotations

from pathlib import Path

import pytest

from npa.orchestration.npa_workflow import (
    NpaWorkflowError,
    build_plan,
    load_spec,
    validate_spec,
)
from npa.orchestration.npa_workflow.predicates import evaluate_predicate
from npa.orchestration.npa_workflow.tokens import TokenError, resolve_tokens

REPO_ROOT = Path(__file__).resolve().parents[4]
SPECS = REPO_ROOT / "npa" / "workflows" / "workbench" / "npa-workflows"


@pytest.mark.parametrize(
    "name",
    [
        "vlm-eval-single.yaml",
        "tokenfactory-rollout-judge.yaml",
        "sim2real-vlm-rl.yaml",
        "bdd100k-pipeline.yaml",
        "tokenfactory-cosmos-gate.yaml",
        "av-night-scene-hardening.yaml",
        "cosmos-synth-fanout-curation.yaml",
    ],
)
def test_example_specs_validate(name: str) -> None:
    spec = load_spec(SPECS / name)
    validate_spec(spec)
    assert spec.api_version == "npa.workflow/v0.0.1"


def test_token_resolution() -> None:
    text = resolve_tokens(
        "s3://{{config.bucket}}/{{run.id}}/out/",
        config={"bucket": "b"},
        run={"id": "run-1"},
    )
    assert text == "s3://b/run-1/out/"


def test_token_unknown_config_raises() -> None:
    with pytest.raises(TokenError):
        resolve_tokens("{{config.missing}}", config={}, run={"id": "x"})


def test_state_output_token() -> None:
    text = resolve_tokens(
        "{{state.decide.uri}}",
        config={},
        run={"id": "run-1"},
        state_outputs={"decide": {"uri": "s3://bucket/decision.json"}},
    )
    assert text == "s3://bucket/decision.json"


def test_sim2real_plan_expands_loops() -> None:
    spec = load_spec(SPECS / "sim2real-vlm-rl.yaml")
    plan = build_plan(spec, run_id="test-run", assume_decision="loop_back")
    states = [step.state for step in plan.steps]
    assert states.count("rollouts") == spec.config["inner_iterations"] * spec.config["outer_iterations"]
    assert "finalize" in states


def test_sim2real_plan_promote_early_exit() -> None:
    spec = load_spec(SPECS / "sim2real-vlm-rl.yaml")
    plan = build_plan(spec, run_id="test-run", assume_decision="promote_checkpoint")
    states = [step.state for step in plan.steps]
    assert states.count("rollouts") == spec.config["inner_iterations"]
    assert states.count("finalize") == 1
    assert states[-1] == "finalize"


def test_loop_max_accepts_braced_config_ref() -> None:
    from npa.orchestration.npa_workflow.spec import resolve_config_int

    assert resolve_config_int("{{config.outer_iterations}}", {"outer_iterations": 4}) == 4


def test_bdd100k_pipeline_plan_expands_eleven_stages() -> None:
    spec = load_spec(SPECS / "bdd100k-pipeline.yaml")
    plan = build_plan(spec, run_id="bdd100k-plan")
    states = [step.state for step in plan.steps]
    assert states == [
        "ingest",
        "backfill-cpu",
        "backfill-clip",
        "curate-views",
        "train-rider",
        "train-nighttime",
        "train-distant",
        "eval-rider",
        "eval-nighttime",
        "eval-distant",
        "review",
    ]


def test_build_plan_omits_assume_decision_for_loop_free_spec() -> None:
    # Loop-free specs (no state transitions) must not carry a spurious
    # loop_back_to_inner_loop assumption; the plan-spec CLI hides the line.
    spec = load_spec(SPECS / "vlm-eval-single.yaml")
    plan = build_plan(spec, run_id="loop-free")
    assert plan.assume_decision == ""


def test_build_plan_defaults_loop_back_for_specs_with_transitions() -> None:
    # Specs with dynamic transitions still default to loop_back so their loops
    # expand when no explicit assumption is passed.
    spec = load_spec(SPECS / "tokenfactory-cosmos-gate.yaml")
    plan = build_plan(spec, run_id="loop-default")
    assert plan.assume_decision == "loop_back_to_inner_loop"


def test_tokenfactory_cosmos_gate_plan_expands_refinement_loop() -> None:
    spec = load_spec(SPECS / "tokenfactory-cosmos-gate.yaml")
    plan = build_plan(spec, run_id="gate-plan", assume_decision="loop_back")
    states = [step.state for step in plan.steps]
    assert states[:2] == ["reason-scene", "augment-scene"]
    assert states.count("vlm-critique") == spec.config["refinement_iterations"]
    assert states.count("quality-gate") == spec.config["refinement_iterations"]
    assert states[-1] == "publish"


def test_invalid_api_version() -> None:
    path = SPECS / "vlm-eval-single.yaml"
    text = path.read_text().replace("v0.0.1", "v9.9.9")
    broken = SPECS.parent / "_tmp-broken.yaml"
    broken.write_text(text)
    try:
        with pytest.raises(NpaWorkflowError, match="apiVersion"):
            load_spec(broken)
    finally:
        broken.unlink(missing_ok=True)


def test_predicate_promote() -> None:
    assert evaluate_predicate("promote_checkpoint", {"last_decision": "promote_checkpoint"})
    assert not evaluate_predicate("promote_checkpoint", {"last_decision": "loop_back_to_inner_loop"})
