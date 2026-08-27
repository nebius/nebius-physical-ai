from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from npa.orchestration.npa_workflow import load_spec, run_workflow
from npa.orchestration.npa_workflow.decisions import (
    DECISION_LOOP_BACK,
    DECISION_PROMOTE,
)
from npa.orchestration.npa_workflow.errors import NpaWorkflowError
from npa.orchestration.npa_workflow.interpreter import PlanStep, _execute_step
from npa.workflows.data_factory_stages import _persist_quality_disposition


REPO_ROOT = Path(__file__).resolve().parents[4]
SPECS = REPO_ROOT / "npa" / "workflows" / "workbench" / "npa-workflows"
SIM2REAL_DEMO = (
    REPO_ROOT
    / "npa"
    / "tests"
    / "fixtures"
    / "npa-workflows"
    / "sim2real-vlm-rl-demo.yaml"
)
PAIDF = SPECS / "physical-ai-data-factory.yaml"


def test_dynamic_execute_reads_decision_for_promote(monkeypatch) -> None:
    spec = load_spec(SIM2REAL_DEMO)

    def fake_reader(_bucket: str, _key: str) -> str:
        return json.dumps({"decision": DECISION_PROMOTE})

    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.interpreter._execute_step",
        lambda step, execute=True: {"state": step.state, "status": "ok"},
    )

    report = run_workflow(
        spec,
        run_id="dyn-promote",
        execute=True,
        decision_reader=fake_reader,
    )
    states = [step["state"] for step in report["steps"]]
    assert states.count("rollouts") == spec.config["inner_iterations"]
    assert states[-1] == "finalize"


def test_dynamic_execute_reads_decision_for_loop_back(monkeypatch) -> None:
    spec = load_spec(SIM2REAL_DEMO)
    decisions = iter([DECISION_LOOP_BACK, DECISION_PROMOTE])

    def fake_reader(_bucket: str, _key: str) -> str:
        return json.dumps({"decision": next(decisions, DECISION_PROMOTE)})

    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.interpreter._execute_step",
        lambda step, execute=True: {"state": step.state, "status": "ok"},
    )

    report = run_workflow(
        spec,
        run_id="dyn-loop",
        execute=True,
        decision_reader=fake_reader,
    )
    states = [step["state"] for step in report["steps"]]
    assert (
        states.count("rollouts")
        == spec.config["inner_iterations"] * spec.config["outer_iterations"]
    )


def test_paidf_promoted_runtime_takes_full_visualization_path(monkeypatch) -> None:
    spec = load_spec(PAIDF)
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.interpreter._execute_step",
        lambda step, execute=True: {"state": step.state, "status": "ok"},
    )

    report = run_workflow(
        spec,
        run_id="paidf-promoted",
        execute=True,
        decision_reader=lambda _bucket, _key: json.dumps(
            {"decision": DECISION_PROMOTE}
        ),
    )

    states = [step["state"] for step in report["steps"]]
    assert states[-2:] == ["visualize", "finalize"]
    assert states.index("quality-disposition") < states.index(
        "require-accepted-quality"
    ) < states.index("annotate-augmented")
    assert "visualize-rejected" not in states
    assert "reject-quality" not in states


def test_paidf_rejected_runtime_writes_rrd_before_rejecting(monkeypatch) -> None:
    spec = load_spec(PAIDF)
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.interpreter._execute_step",
        lambda step, execute=True: {"state": step.state, "status": "ok"},
    )

    report = run_workflow(
        spec,
        run_id="paidf-rejected",
        execute=True,
        decision_reader=lambda _bucket, _key: json.dumps(
            {"decision": DECISION_LOOP_BACK}
        ),
    )

    states = [step["state"] for step in report["steps"]]
    assert states[-2:] == ["visualize-rejected", "reject-quality"]
    assert "require-accepted-quality" not in states
    assert "annotate-augmented" not in states
    assert "cosmos-curate" not in states
    assert "curate" not in states
    assert "visualize" not in states
    assert "finalize" not in states
    assert states.index("quality-disposition") < states.index("visualize-rejected")


def test_paidf_runtime_executor_observes_terminal_nonzero_rejection(
    tmp_path: Path,
) -> None:
    """The dynamic runtime must propagate the real terminal rejection exit."""

    spec = load_spec(PAIDF)
    scores = tmp_path / "cosmos_evaluator.json"
    disposition = tmp_path / "quality_disposition.json"
    scores.write_text(
        json.dumps({"status": "completed", "score": 0.1, "passed": True})
    )
    _persist_quality_disposition(str(scores), str(disposition), 0.75)
    spec.config["scores_uri"] = str(scores)
    spec.config["quality_disposition_uri"] = str(disposition)
    spec.states["quality-disposition"].params["decision_uri"] = (
        "s3://example-bucket/paidf-test/final-decision.json"
    )
    spec.states["quality-gate"].params["decision_uri"] = (
        "s3://example-bucket/paidf-test/iteration-{{loop.grade}}/decision.json"
    )

    class _RuntimeExecutor:
        def __init__(self) -> None:
            self.states: list[str] = []

        def execute(self, step: PlanStep) -> dict:
            self.states.append(step.state)
            if step.state == "reject-quality":
                return _execute_step(step, execute=True)
            return {"state": step.state, "status": "ok"}

        def execute_parallel(self, steps, max_concurrency):  # noqa: ARG002
            return [self.execute(step) for step in steps]

    executor = _RuntimeExecutor()
    with pytest.raises(NpaWorkflowError, match="reject-quality failed"):
        run_workflow(
            spec,
            run_id="paidf-real-terminal-rejection",
            execute=True,
            decision_reader=lambda _bucket, _key: json.dumps(
                {"decision": DECISION_LOOP_BACK}
            ),
            step_executor=executor,
        )

    assert executor.states[-2:] == ["visualize-rejected", "reject-quality"]
    assert "annotate-augmented" not in executor.states
    assert "cosmos-curate" not in executor.states
    assert "curate" not in executor.states
    assert "finalize" not in executor.states


def test_paidf_non_runtime_serial_executor_stops_at_accepted_guard(
    tmp_path: Path,
) -> None:
    """An accepted preview cannot turn a real rejection into later stage work."""

    scores = tmp_path / "cosmos_evaluator.json"
    disposition = tmp_path / "quality_disposition.json"
    downstream_marker = tmp_path / "annotate-ran"
    scores.write_text(
        json.dumps({"status": "completed", "score": 0.1, "passed": True})
    )
    _persist_quality_disposition(str(scores), str(disposition), threshold=0.75)
    serial_steps = [
        PlanStep(
            state="require-accepted-quality",
            argv=[
                sys.executable,
                "-c",
                (
                    "import sys; from npa.workflows.data_factory_stages import "
                    "enforce_quality_disposition; "
                    "enforce_quality_disposition(*sys.argv[1:])"
                ),
                str(scores),
                str(disposition),
                "0.75",
            ],
        ),
        PlanStep(
            state="annotate-augmented",
            argv=[
                sys.executable,
                "-c",
                "from pathlib import Path; Path(__import__('sys').argv[1]).touch()",
                str(downstream_marker),
            ],
        ),
    ]

    with pytest.raises(NpaWorkflowError, match="require-accepted-quality failed"):
        for step in serial_steps:
            _execute_step(step, execute=True)

    assert not downstream_marker.exists()
