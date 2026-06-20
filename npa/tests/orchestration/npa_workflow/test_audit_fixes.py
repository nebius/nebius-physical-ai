from __future__ import annotations

from pathlib import Path

import pytest

from npa.orchestration.npa_workflow import NpaWorkflowError, build_plan, load_spec, run_workflow
from npa.orchestration.npa_workflow.predicates import evaluate_predicate
from npa.orchestration.npa_workflow.run_state import RunManifest, RunStateStore
from npa.orchestration.npa_workflow.spec import validate_spec

REPO_ROOT = Path(__file__).resolve().parents[4]
SPECS = REPO_ROOT / "npa" / "workflows" / "workbench" / "npa-workflows"


def test_promote_checkpoint_plans_finalize_once() -> None:
    spec = load_spec(SPECS / "sim2real-vlm-rl.yaml")
    plan = build_plan(spec, run_id="audit-promote", assume_decision="promote_checkpoint")
    states = [step.state for step in plan.steps]
    assert states.count("finalize") == 1
    assert len(plan.steps) == 11


def test_loop_back_assume_normalizes_for_plan_predicates() -> None:
    context = {"last_decision": "loop_back"}
    assert evaluate_predicate("loop_back", context) is True


def test_unbounded_cycle_rejected_at_validate() -> None:
    from npa.orchestration.npa_workflow.spec import StateSpec

    spec = load_spec(SPECS / "tokenfactory-rollout-judge.yaml")
    spec.states["reason-scene"].next = "judge-rollouts"
    spec.states["judge-rollouts"].terminal = False
    spec.states["judge-rollouts"].next = "reason-scene"
    spec.states["done"] = StateSpec(name="done", terminal=True)
    with pytest.raises(NpaWorkflowError, match="unbounded control-flow cycle"):
        validate_spec(spec)


def test_zero_loop_max_rejected_at_validate() -> None:
    spec = load_spec(SPECS / "sim2real-vlm-rl.yaml")
    spec.config["inner_iterations"] = 0
    with pytest.raises(NpaWorkflowError, match="loop.max must be >= 1"):
        validate_spec(spec)


def test_failed_execute_persists_failed_manifest(monkeypatch) -> None:
    spec = load_spec(SPECS / "vlm-eval-single.yaml")
    store: dict[tuple[str, str], bytes] = {}

    def writer(bucket: str, key: str, body: bytes) -> None:
        store[(bucket, key)] = body

    state_store = RunStateStore(
        bucket="bucket",
        prefix="runs/fail-demo",
        writer=writer,
        reader=lambda _b, _k: (_ for _ in ()).throw(FileNotFoundError()),
    )

    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.interpreter._execute_step",
        lambda step, execute=True: (_ for _ in ()).throw(
            NpaWorkflowError(f"boom at {step.state}")
        ),
    )

    with pytest.raises(NpaWorkflowError, match="boom"):
        run_workflow(
            spec,
            run_id="fail-demo",
            execute=True,
            state_store=state_store,
        )

    manifest = RunManifest.from_dict(__import__("json").loads(store[("bucket", "runs/fail-demo/npa-workflow/manifest.json")]))
    assert manifest.status == "failed"
    assert manifest.steps
    assert manifest.steps[0]["status"] == "failed"
