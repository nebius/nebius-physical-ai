"""Tests for the declarative DAG scheduler (sim2real run --dag).

The headline test is *parity*: the scheduler must drive the same stage methods,
in the same order, as ``runner.run_staged()`` — for both the full-iteration and
the promote early-exit cases. A recording fake workflow is exercised through
both code paths and the call logs are compared.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from npa.workflows.sim2real import scheduler
from npa.workflows.sim2real.runner import Sim2RealWorkflow
from npa.workflows.sim2real.scheduler import (
    DEFAULT_DAG_SPEC,
    DagSpec,
    DagSpecError,
    NodeSpec,
    load_spec,
    resolve_int,
    run_dag,
    topological_order,
    validate_spec,
)
from npa.workflows.sim2real.state import WorkflowState


# --------------------------------------------------------------------------- #
# Recording fake workflow — implements the surface run_staged()/run_dag() use.
# --------------------------------------------------------------------------- #
class _RecordingWorkflow(Sim2RealWorkflow):
    """Sim2RealWorkflow whose stage methods record calls instead of doing work."""

    def __init__(self, local_dir: Path, *, outer_iterations: int, promote_after: int | None):
        # Bypass the real __init__ (config.validate + output dir creation).
        self.config = SimpleNamespace(outer_iterations=outer_iterations, inner_iterations=1)
        self._local_dir = local_dir
        self.calls: list[str] = []
        self._promote_after = promote_after
        self._next_outer = 1

    def _make_state(self, *, status: str, decision: str | None) -> WorkflowState:
        state = WorkflowState(
            run_id="parity-test",
            local_artifact_dir=self._local_dir,
            current_quality=0.4,
            status=status,
            next_outer_iteration=self._next_outer,
        )
        if decision is not None:
            state.final_decision = {"decision": decision}
        return state

    def run_preamble(self) -> WorkflowState:  # type: ignore[override]
        self.calls.append("preamble")
        return self._make_state(status="preamble_completed", decision=None)

    def run_outer_iteration(self, *, outer_iteration: int, initial_quality=None) -> WorkflowState:  # type: ignore[override]
        self.calls.append(f"outer:{outer_iteration}")
        self._next_outer = outer_iteration + 1
        promote = self._promote_after is not None and outer_iteration >= self._promote_after
        decision = "promote_checkpoint" if promote else "loop_back_to_inner_loop"
        return self._make_state(status="outer_iteration_completed", decision=decision)

    def run_finalize(self, *, upload=None) -> dict:  # type: ignore[override]
        self.calls.append(f"finalize:upload={upload}")
        return {"report": "fake", "calls": list(self.calls)}


@pytest.fixture(autouse=True)
def _identity_config_reload(monkeypatch):
    """run_staged()/_exec_preamble import engine._config_from_workflow_state."""
    import npa.workflows.sim2real.engine as engine

    monkeypatch.setattr(
        engine, "_config_from_workflow_state", lambda config, _payload: config
    )


def _shipped_outer_loop_spec() -> DagSpec:
    return DagSpec(
        name="parity",
        nodes=[
            NodeSpec(id="preamble", executor="preamble"),
            NodeSpec(
                id="outer_loop",
                executor="outer_iteration",
                needs=["preamble"],
                loop=scheduler.LoopSpec(
                    max_iterations="config.outer_iterations", until="promote_checkpoint"
                ),
            ),
            NodeSpec(id="finalize", executor="finalize", needs=["outer_loop"]),
        ],
    )


@pytest.mark.parametrize(
    ("outer_iterations", "promote_after"),
    [
        (2, None),  # no promote → run all outer iterations
        (3, 1),     # promote on first iteration → early exit
        (3, 2),     # promote on second iteration
    ],
)
def test_dag_parity_with_run_staged(tmp_path, outer_iterations, promote_after):
    """run_dag drives the identical method sequence as run_staged()."""

    staged_wf = _RecordingWorkflow(
        tmp_path / "staged", outer_iterations=outer_iterations, promote_after=promote_after
    )
    (tmp_path / "staged").mkdir()
    staged_report = staged_wf.run_staged(upload=True)

    dag_wf = _RecordingWorkflow(
        tmp_path / "dag", outer_iterations=outer_iterations, promote_after=promote_after
    )
    (tmp_path / "dag").mkdir()
    dag_report = run_dag(dag_wf, _shipped_outer_loop_spec(), upload=True)

    assert dag_wf.calls == staged_wf.calls
    assert dag_report["calls"] == staged_report["calls"]


# --------------------------------------------------------------------------- #
# Spec loading + validation
# --------------------------------------------------------------------------- #
def test_shipped_spec_loads_and_validates():
    spec = load_spec(DEFAULT_DAG_SPEC)
    assert spec.node_ids() == ["preamble", "outer_loop", "finalize"]
    outer = next(n for n in spec.nodes if n.id == "outer_loop")
    assert outer.loop is not None
    assert outer.loop.until == "promote_checkpoint"
    assert outer.loop.max_iterations == "config.outer_iterations"
    # acyclic + executors known
    assert [n.id for n in topological_order(spec)] == ["preamble", "outer_loop", "finalize"]


def test_validate_rejects_unknown_executor():
    spec = DagSpec(name="x", nodes=[NodeSpec(id="a", executor="does-not-exist")])
    with pytest.raises(DagSpecError, match="unknown executor"):
        validate_spec(spec)


def test_validate_rejects_unknown_until_predicate():
    spec = DagSpec(
        name="x",
        nodes=[
            NodeSpec(id="preamble", executor="preamble"),
            NodeSpec(
                id="outer_loop",
                executor="outer_iteration",
                needs=["preamble"],
                loop=scheduler.LoopSpec(max_iterations=2, until="never_happens"),
            ),
        ],
    )
    with pytest.raises(DagSpecError, match="unknown until predicate"):
        validate_spec(spec)


def test_validate_rejects_missing_dependency():
    spec = DagSpec(
        name="x",
        nodes=[NodeSpec(id="finalize", executor="finalize", needs=["ghost"])],
    )
    with pytest.raises(DagSpecError, match="needs unknown node"):
        validate_spec(spec)


def test_topological_order_detects_cycle():
    spec = DagSpec(
        name="x",
        nodes=[
            NodeSpec(id="a", executor="preamble", needs=["b"]),
            NodeSpec(id="b", executor="finalize", needs=["a"]),
        ],
    )
    with pytest.raises(DagSpecError, match="cycle"):
        topological_order(spec)


def test_resolve_int_literal_and_config_ref():
    config = SimpleNamespace(outer_iterations=7)
    assert resolve_int(3, config) == 3
    assert resolve_int("5", config) == 5
    assert resolve_int("config.outer_iterations", config) == 7


def test_resolve_int_rejects_unknown_config_attr():
    with pytest.raises(DagSpecError, match="no attribute"):
        resolve_int("config.nope", SimpleNamespace())


def test_run_dag_requires_finalize_node(tmp_path):
    """A spec with no finalize node must error, not silently return None."""

    wf = _RecordingWorkflow(tmp_path, outer_iterations=1, promote_after=None)
    tmp_path.joinpath("state").mkdir(parents=True, exist_ok=True)
    spec = DagSpec(name="x", nodes=[NodeSpec(id="preamble", executor="preamble")])
    # preamble executor checks state dir; give it a clean local dir without state file
    with pytest.raises(DagSpecError, match="finalize"):
        run_dag(wf, spec)
