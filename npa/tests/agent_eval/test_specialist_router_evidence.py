"""Keep durable classifier decisions and uncertain router spend in experiment evidence."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from npa.agent_backend.specialists.store import TaskStore


@pytest.fixture
def evidence():
    path = (
        Path(__file__).resolve().parents[3]
        / "npa/examples/specialists/workflows/evidence.py"
    )
    spec = importlib.util.spec_from_file_location("router_evidence", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _receipt(selection):
    return {"task": {"route": {"status": "explicit", "model_selection": selection}}}


def test_endpoint_route_retains_usage_and_workspace_assignment(evidence):
    selection = {
        "provider": "typesafe",
        "model": "jev-1.13.0",
        "status": "accepted",
        "api_call_attempted": True,
        "usage": {"input_tokens": 120, "output_tokens": 3},
        "selected_model": "model-a",
        "effective_model": "model-a",
        "attempt_id": "routing-task",
    }
    receipts = _receipt(selection)
    result = evidence._router_usage(receipts)
    assert result["usage_complete"] is True
    assert result["responses"] == [
        {**selection, "task_id": "task", "route_kind": "model"}
    ]
    assert "cached_tokens" not in result["responses"][0]["usage"]
    assert receipts["task"]["route"]["status"] == "explicit"


@pytest.mark.parametrize("status", ["accepted", "abstained", "unavailable"])
def test_router_response_without_counters_is_incomplete(evidence, status):
    result = evidence._router_usage(
        _receipt(
            {
                "provider": "typesafe",
                "status": status,
                "api_call_attempted": True,
                "usage": {},
            }
        )
    )
    assert result["usage_complete"] is False
    assert result["responses"][0]["usage"] == {}


def test_missing_key_preserves_unavailable_receipt_without_api_spend(evidence):
    result = evidence._router_usage(
        _receipt(
            {
                "provider": "typesafe",
                "status": "unavailable",
                "reason": "missing_credential",
                "api_call_attempted": False,
                "usage": {},
                "fallback": True,
            }
        )
    )
    assert result["usage_complete"] is True
    assert result["responses"][0]["api_call_attempted"] is False
    assert result["responses"][0]["reason"] == "missing_credential"


@pytest.mark.parametrize("attempted", ["unknown", None, 1])
def test_durable_pre_call_intent_cannot_hide_possible_api_effect(evidence, attempted):
    result = evidence._router_usage(
        _receipt(
            {
                "provider": "typesafe",
                "status": "started",
                "api_call_attempted": attempted,
                "usage": {},
                "attempt_id": "routing-task",
            }
        )
    )
    assert result["usage_complete"] is False
    assert result["responses"][0]["api_call_attempted"] == "unknown"


@pytest.mark.parametrize(
    "reason,attempted",
    [
        ("missing_credential", False),
        ("invalid_configuration", False),
        ("provider_transport_error", True),
        ("provider_http_error", True),
        ("invalid_response", True),
        ("unrecognized", "unknown"),
    ],
)
def test_historical_profile_router_records_keep_effect_uncertainty(
    evidence, reason, attempted
):
    result = evidence._router_usage(
        {
            "task": {
                "route": {
                    "provider": "typesafe",
                    "model": "jev-1.13.0",
                    "status": "unavailable",
                    "reason": reason,
                    "usage": {},
                }
            }
        }
    )
    assert result["responses"][0]["api_call_attempted"] == attempted
    assert result["usage_complete"] is (attempted is False)


def test_explicit_and_default_profile_choices_are_not_model_calls(evidence):
    assert evidence._router_usage(
        {
            "first": {"route": {"status": "explicit", "specialist": "planner"}},
            "second": {"route": {"status": "default", "specialist": "renderer"}},
        }
    ) == {"responses": [], "usage_complete": True}


def _accepted_route(input_tokens):
    return {
        "provider": "typesafe",
        "model": "jev-1.13.0",
        "status": "accepted",
        "usage": {"input_tokens": input_tokens, "output_tokens": 2},
    }


@pytest.fixture
def accounting():
    path = (
        Path(__file__).resolve().parents[3]
        / "npa/examples/specialists/workflow_usage.py"
    )
    spec = importlib.util.spec_from_file_location("router_accounting", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def dual_routing_snapshot(evidence, tmp_path):
    store = TaskStore(tmp_path / "team")
    store._submit("task", "planner", "policy", "request", "goal", _accepted_route(100))
    store._model_route("task", {**_accepted_route(200), "api_call_attempted": True})
    store._update("task", "completed")
    coordinator = TaskStore(tmp_path / "coordinator")
    (tmp_path / "codex.jsonl").write_text(
        json.dumps(
            {
                "type": "turn.completed",
                "usage": {"input_tokens": 1000, "output_tokens": 100},
            }
        )
        + "\n"
    )
    execution = {"arm": "astra-tofa", "exit_code": 0}
    evidence._snapshot(SimpleNamespace(store=store), coordinator, tmp_path, execution)
    return json.loads((tmp_path / "usage.json").read_text()), execution


@pytest.mark.parametrize("missing_scope", [None, "profile", "model"])
def test_both_routing_layers_are_frozen_and_priced_once(
    evidence, accounting, dual_routing_snapshot, missing_scope
):
    usage, execution = dual_routing_snapshot
    responses = usage["router"]["responses"]
    assert [response["route_kind"] for response in responses] == ["profile", "model"]
    assert all("model_selection" not in response for response in responses)
    if missing_scope:
        next(row for row in responses if row["route_kind"] == missing_scope)[
            "usage"
        ] = {}
    prices = {
        "models": {
            "gpt-6-astra": {
                "cache_policy": "full-input",
                "rate_options": [
                    {
                        "input_price_per_million_tokens": 2,
                        "output_price_per_million_tokens": 4,
                    }
                ],
            },
            "jev-1.13.0": {
                "cache_policy": "full-input",
                "rate_options": [
                    {
                        "input_price_per_million_tokens": 0.042,
                        "output_price_per_million_tokens": 0,
                    }
                ],
            },
        }
    }
    report = accounting.summarize_usage(usage, execution, prices)
    assert report["models"]["jev-1.13.0"]["recorded_api_calls"] == 2
    assert report["router"]["accepted_jev_routes"] == 2
    if missing_scope:
        assert report["estimated_cost_range_usd"] is None
        assert report["usage_complete"] is False
    else:
        assert report["estimated_cost_usd"] == pytest.approx(0.0024 + 300 * 0.042 / 1e6)


@pytest.mark.parametrize("missing_scope", ["profile", "model"])
def test_missing_counters_in_either_durable_route_mark_snapshot_incomplete(
    evidence, missing_scope
):
    root, nested = _accepted_route(100), _accepted_route(200)
    (root if missing_scope == "profile" else nested)["usage"] = {}
    root["model_selection"] = nested
    result = evidence._router_usage({"task": {"route": root}})
    assert len(result["responses"]) == 2
    assert result["usage_complete"] is False


def _coordinator_files(directory, invocations, *, completed=2):
    (directory / "coordinator-config.json").write_text(
        json.dumps({"turns": invocations})
    )
    (directory / "codex.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "type": "turn.completed",
                    "usage": {
                        "input_tokens": 100 * (index + 1),
                        "output_tokens": 10,
                    },
                }
            )
            + "\n"
            for index in range(completed)
        )
    )


def test_all_fresh_coordinator_turns_preserve_appended_usage(evidence, tmp_path):
    turns = [
        {"phase": "delegate", "exit_code": 0, "ended_epoch": 100},
        {"phase": "review", "exit_code": 0, "ended_epoch": 200},
    ]
    _coordinator_files(tmp_path, turns)
    result = evidence._astra_usage(tmp_path)
    assert result["usage_complete"] is True
    assert [turn["input_tokens"] for turn in result["turns"]] == [100, 200]
    assert result["invocations"]["expected_turns"] == 2


@pytest.mark.parametrize(
    "last_invocation,completed",
    [
        ({"exit_code": 9, "ended_epoch": 200}, 1),
        ({"exit_code": None, "ended_epoch": 200}, 1),
        ({"exit_code": 0}, 2),
        ({"exit_code": 0, "ended_epoch": 200}, 1),
        ({"exit_code": 0, "ended_epoch": 200}, 3),
    ],
)
def test_missing_or_failed_fresh_review_cannot_report_complete_cost(
    evidence, accounting, tmp_path, last_invocation, completed
):
    _coordinator_files(
        tmp_path,
        [
            {"exit_code": 0, "ended_epoch": 100},
            last_invocation,
        ],
        completed=completed,
    )
    astra = evidence._astra_usage(tmp_path)
    assert len(astra["turns"]) == completed
    assert astra["usage_complete"] is False
    usage = {
        "astra": astra,
        "specialists": {"usage_complete": True},
        "usage_complete": False,
    }
    prices = {
        "models": {
            "gpt-6-astra": {
                "cache_policy": "full-input",
                "rate_options": [
                    {
                        "input_price_per_million_tokens": 10,
                        "output_price_per_million_tokens": 50,
                    }
                ],
            }
        }
    }
    report = accounting.summarize_usage(usage, {"exit_code": 9}, prices)
    assert report["estimated_cost_range_usd"] is None
    assert report["totals"]["priced_recorded_cost_usd"] is not None
    assert "astra_usage_incomplete" in report["incomplete_reasons"]


def test_historical_single_invocation_config_still_requires_one_turn(
    evidence, tmp_path
):
    _coordinator_files(tmp_path, [], completed=1)
    (tmp_path / "coordinator-config.json").write_text(
        json.dumps({"argv": ["codex", "exec"]})
    )
    assert evidence._astra_usage(tmp_path)["usage_complete"] is True
    _coordinator_files(tmp_path, [], completed=2)
    (tmp_path / "coordinator-config.json").write_text(
        json.dumps({"argv": ["codex", "exec"]})
    )
    assert evidence._astra_usage(tmp_path)["usage_complete"] is False


def test_missing_modern_invocation_records_do_not_become_legacy_complete(
    evidence, tmp_path
):
    _coordinator_files(tmp_path, [], completed=1)
    (tmp_path / "coordinator-config.json").unlink()
    (tmp_path / "protocol.json").write_text(json.dumps({"coordination": "completion"}))
    result = evidence._astra_usage(tmp_path)
    assert result["usage_complete"] is False
    assert result["invocations"]["verification"] == "missing"


def test_provider_failure_remains_incomplete_after_successful_fallback(evidence):
    failure = {
        "type": "provider_failure",
        "model": "unavailable-model",
        "status_code": 404,
        "usage": {},
        "usage_missing": True,
        "request_metrics": {"attempts": 1},
    }
    response = {
        "type": "model",
        "model": "backup-model",
        "accepted": True,
        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
    }
    task = {
        "events": [failure, {"type": "model_fallback"}, response],
        "status": "completed",
        "calls": [],
    }
    result = evidence._specialist_usage({"task": task})
    assert result["usage_complete"] is False
    assert result["failures"] == [{"task_id": "task", **failure}]
    assert result["responses"] == [{"task_id": "task", **response}]
