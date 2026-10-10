"""Prove two independent measured retry loops and runtime-only submission."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
import yaml

from npa.orchestration.npa_workflow import build_plan, load_spec, validate_spec
from npa.orchestration.npa_workflow.errors import NpaWorkflowError
from npa.orchestration.npa_workflow.interpreter import run_workflow
from npa.orchestration.npa_workflow.submit import spec_requires_runtime

SPEC = (
    Path(__file__).resolve().parents[4]
    / "npa/tests/fixtures/policy-training-slurm.json"
)


def test_pipeline_plan_contains_real_stages_and_requires_runtime():
    spec = load_spec(SPEC)
    validate_spec(spec)
    plan = build_plan(spec, run_id="test", assume_decision="promote_checkpoint")
    assert [s.state for s in plan.steps] == [
        "curate",
        "split",
        "pretrain",
        "evaluate-pretrain",
        "gate-pretrain",
        "finetune",
        "evaluate-finetune",
        "gate-finetune",
        "test-policy",
    ]
    assert spec_requires_runtime(spec)
    assert all(s.tool_ref.startswith("workflow.policy_training.") for s in plan.steps)
    assert plan.steps[0].resources_profile["cloud"] == "kubernetes"
    assert not any("accelerators" in s.resources_profile for s in plan.steps)
    assert spec.states["curate"].trigger.max_polls == 0
    for name in ("pretrain-loop", "finetune-loop"):
        assert spec.states[name].loop.max is None


def _retry_executor():
    counts, calls, decisions = Counter(), [], {}

    class Executor:
        def execute(self, step):
            counts[step.state] += 1
            calls.append(step)
            if step.state.startswith("gate-"):
                uri = step.outputs[0]["uri"]
                needed = 3 if step.state == "gate-pretrain" else 2
                decisions[uri] = (
                    "promote_checkpoint"
                    if counts[step.state] == needed
                    else "loop_back"
                )
            return {"state": step.state, "status": "ok"}

    def reader(bucket, key):
        return json.dumps({"decision": decisions[f"s3://{bucket}/{key}"]})

    return Executor(), reader, counts, calls


def test_each_failed_gate_repeats_only_its_training_phase():
    spec = load_spec(SPEC)
    executor, reader, counts, calls = _retry_executor()
    report = run_workflow(
        spec,
        run_id="test",
        execute=True,
        step_executor=executor,
        decision_reader=reader,
        assume_decision="promote_checkpoint",
    )
    assert report["status"] == "completed"
    assert counts == Counter(
        {
            "curate": 1,
            "split": 1,
            "pretrain": 3,
            "evaluate-pretrain": 3,
            "gate-pretrain": 3,
            "finetune": 2,
            "evaluate-finetune": 2,
            "gate-finetune": 2,
            "test-policy": 1,
        }
    )
    assert len({tuple(step.argv) for step in calls}) == len(calls)
    deployment = calls[-1]
    assert (
        "/finetune/2/decision.json"
        in deployment.argv[deployment.argv.index("--input-uri") + 1]
    )
    first_finetune = next(s for s in calls if s.state == "finetune")
    assert (
        "/pretrain/3/decision.json"
        in first_finetune.argv[first_finetune.argv.index("--input-uri") + 1]
    )


@pytest.mark.parametrize("change", ["no-runtime", "no-decision-writer"])
def test_until_only_loop_requires_runtime_and_real_decision_writer(tmp_path, change):
    data = yaml.safe_load(SPEC.read_text())
    if change == "no-runtime":
        data["metadata"].pop("executionMode")
    else:
        data["states"]["gate-pretrain"]["writesDecision"] = False
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(NpaWorkflowError, match="until-only"):
        validate_spec(load_spec(path))


@pytest.mark.parametrize("decision", ["", "unknown"])
def test_absent_gate_cannot_use_planning_assumption(decision):
    spec = load_spec(SPEC)

    class Executor:
        def execute(self, step):
            return {"state": step.state, "status": "ok"}

    with pytest.raises(NpaWorkflowError, match="measured decision"):
        run_workflow(
            spec,
            run_id="test",
            execute=True,
            step_executor=Executor(),
            decision_reader=lambda *args: json.dumps({"decision": decision}),
            assume_decision="promote_checkpoint",
        )
