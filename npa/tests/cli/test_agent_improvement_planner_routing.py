"""Persist real planner failures under their configured action-loop owner."""

from __future__ import annotations

import json

import pytest

from npa.agent_backend.actions import run_action_loop
from npa.agent_backend.improvement_routes import ImprovementRuntime
from npa.agent_backend.improvements import ImprovementScope, ImprovementStore
from npa.agent_backend.sim2real_loop import drive_sim2real_loop


def _store(root, components):
    repository = root / "source"
    repository.mkdir(exist_ok=True)
    return ImprovementStore(
        root / "queue", repository=repository, evidence_directory=root / "evidence",
        scopes=[
            ImprovementScope(
                scope_id=component, component=component, files=(component + ".py",),
                base_revision="a" * 40, required_checks=("reproducer",),
            )
            for component in components
        ],
        reviewers=(),
    )


def _record(root, result, components):
    store = _store(root, components)
    feedback = ImprovementRuntime(lambda: store).record(
        result, {"request_id": "synthetic-routing-episode", "lessons": []},
    )
    assert feedback["status"] == "recorded"
    reopened = _store(root, components)
    items = reopened.list_items()
    assert feedback["item_ids"] == [item["id"] for item in items]
    return reopened, items


def _planner(*plans):
    responses = iter(plans)

    def call(*args, **kwargs):
        return {"choices": [{"message": {"content": json.dumps(next(responses))}}]}

    return call


@pytest.mark.parametrize("failure", ["unavailable", "malformed"])
@pytest.mark.parametrize("components", [
    ("action-loop",),
    ("action-loop", "sim2real-drive"),
])
def test_planner_failure_persists_only_for_action_loop(tmp_path, failure, components):
    planner_calls = []
    tool_calls = []

    def planner(*args, **kwargs):
        planner_calls.append(1)
        if failure == "unavailable":
            raise RuntimeError("synthetic planner unavailable")
        return {"choices": [{"message": {"content": "invalid action JSON"}}]}

    result = run_action_loop(
        "inspect available evidence",
        tools={"retrieval_search": lambda args: tool_calls.append(args)},
        model_call=planner,
    )
    assert result["ok"] is False
    assert result["stopped_reason"] == ("error" if failure == "unavailable" else "no_plan")
    assert len(planner_calls) == (1 if failure == "unavailable" else 2)
    assert tool_calls == []
    assert len(result["steps"]) == 1
    step = result["steps"][0]
    assert step["phase"] == "plan" and step["status"] == "error"
    assert "tool" not in step

    store, items = _record(tmp_path, result, components)
    assert [item["component"] for item in items] == ["action-loop"]
    assert items[0]["kind"] == "tool_error"
    assert items[0]["scope"]["files"] == ["action-loop.py"]
    history = store.history(items[0]["id"])
    assert len(history["occurrences"]) == 1
    assert history["occurrences"][0]["event_index"] == 0
    assert history["occurrences"][0]["evidence"] == step


def test_explicit_failed_tool_keeps_its_own_scope(tmp_path):
    result = run_action_loop(
        "inspect available evidence",
        tools={"retrieval_search": lambda args: {"error": "synthetic retrieval failure"}},
        model_call=_planner(
            {"tool": "retrieval_search", "args": {"query": "evidence"}},
            {"final": "The retrieval failed."},
        ),
    )
    store, items = _record(tmp_path, result, ("action-loop", "sim2real-drive", "retrieval_search"))
    assert [item["component"] for item in items] == ["retrieval_search"]
    assert store.history(items[0]["id"])["occurrences"][0]["evidence"]["args"] == {"query": "evidence"}


@pytest.mark.parametrize("stage,kind", [
    ("gate", "drive_error"),
    ("adjust", "drive_adjust_error"),
    ("diagnose", "drive_diagnosis_error"),
])
def test_real_drive_iterations_keep_sim2real_fallback(tmp_path, stage, kind):
    def failed(*args):
        raise RuntimeError("synthetic drive component failure")

    callbacks = {
        "gate": lambda *args: {"success_rate": 0.2, "threshold": 0.8},
        "diagnose": lambda *args: {"failure_mode": "synthetic"},
        "adjust": lambda config, diagnosis: config,
    }
    callbacks[stage] = failed
    result = drive_sim2real_loop(
        "evaluate a synthetic rollout", config={"run_id": "synthetic-run"},
        launch=lambda config: {"run_id": "synthetic-run"},
        status=lambda run: {
            "ok": True, "run": {"run_id": run},
            "sim_viz": {"run_id": run, "stage": "evaluation"},
        },
        confirm_token="synthetic-consent", session_token="synthetic-consent", **callbacks,
    )
    assert "tool" not in result["iterations"][0]
    _, items = _record(tmp_path, result, ("action-loop", "sim2real-drive"))
    assert [(item["component"], item["kind"]) for item in items] == [("sim2real-drive", kind)]


@pytest.mark.parametrize("case", ["success", "recovered", "terminal_empty", "confirmation"])
def test_successful_and_terminal_actions_do_not_create_findings(tmp_path, case):
    if case == "terminal_empty":
        result = run_action_loop(
            "list available runs", tools={"insights_query": lambda args: {"count": 0, "records": []}},
            model_call=_planner({"tool": "insights_query", "args": {}}),
        )
        assert any(step.get("terminal_observation") for step in result["steps"])
    elif case == "confirmation":
        result = run_action_loop(
            "launch a workflow", tools={"sim2real_submit": lambda args: pytest.fail("unconfirmed tool ran")},
            model_call=_planner({"tool": "sim2real_submit", "args": {}}),
        )
        assert result["needs_confirmation"] is True
    else:
        outputs = iter([
            {"error": "synthetic temporary retrieval failure"} if case == "recovered" else {"answer": "evidence"},
            {"answer": "recovered evidence"},
        ])
        plans = [{"tool": "retrieval_search", "args": {"query": "evidence"}}]
        if case == "recovered":
            plans.append({"tool": "retrieval_search", "args": {"query": "broader evidence"}})
        plans.append({"final": "Evidence retrieved."})
        result = run_action_loop(
            "inspect available evidence", tools={"retrieval_search": lambda args: next(outputs)},
            model_call=_planner(*plans),
        )
        assert result["ok"] is True
    _, items = _record(tmp_path, result, ("action-loop", "sim2real-drive", "retrieval_search", "insights_query", "sim2real_submit"))
    assert items == []
