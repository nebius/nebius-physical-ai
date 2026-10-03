"""Verify workflow accounting preserves missing usage, fallback spend and tariff uncertainty."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.fixture
def accounting():
    path = (
        Path(__file__).resolve().parents[3]
        / "npa/examples/specialists/workflow_usage.py"
    )
    spec = importlib.util.spec_from_file_location("workflow_usage", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rates(input_price=2, output_price=4, cached_price=1, write_price=3):
    return {
        "input_price_per_million_tokens": input_price,
        "output_price_per_million_tokens": output_price,
        "cached_input_price_per_million_tokens": cached_price,
        "cache_write_price_per_million_tokens": write_price,
    }


@pytest.fixture
def prices():
    return {
        "models": {
            "coordinator": {"cache_policy": "itemized", "rate_options": [_rates()]},
            "specialist": {"cache_policy": "full-input", "rate_options": [_rates()]},
        }
    }


@pytest.fixture
def evidence():
    return {
        "usage_complete": True,
        "astra": {
            "usage_complete": True,
            "matched_tool_scope": True,
            "turns": [
                {
                    "input_tokens": 1000,
                    "output_tokens": 100,
                    "cached_input_tokens": 200,
                    "cache_write_input_tokens": 50,
                    "reasoning_output_tokens": 70,
                }
            ],
        },
        "specialists": {"usage_complete": True, "responses": []},
    }


@pytest.fixture
def execution():
    return {
        "arm": "astra-tofa",
        "exit_code": 0,
        "snapshot_errors": {},
        "unfinished_tasks": [],
        "agent_tool_seconds": 10,
        "coordinator_seconds": 8,
    }


def _summarize(accounting, evidence, execution, prices):
    return accounting.summarize_usage(
        evidence, execution, prices, coordinator_model="coordinator"
    )


def _response(model, accepted, prompt=100, completion=20):
    return {
        "model": model,
        "accepted": accepted,
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion},
    }


def test_counts_rejected_fallback_and_all_models(
    accounting, evidence, execution, prices
):
    evidence["specialists"]["responses"] = [
        _response("specialist", False),
        _response("fallback", True),
        _response("specialist", True),
    ]
    prices["models"]["fallback"] = prices["models"]["specialist"]
    report = _summarize(accounting, evidence, execution, prices)
    assert report["usage_complete"] is True
    assert report["totals"]["usage_records"] == 4
    assert report["totals"]["tokens"]["input_tokens"] == 1300
    assert report["models"]["specialist"]["rejected_specialist_responses"] == 1
    assert report["models"]["specialist"]["accepted_specialist_responses"] == 1
    assert report["models"]["specialist"]["recorded_api_calls"] == 2
    assert report["models"]["coordinator"]["recorded_api_calls"] is None
    assert report["estimated_cost_usd"] == pytest.approx(0.00225 + 3 * 0.00028)


def test_reasoning_subset_and_cache_writes_charged_once(
    accounting, evidence, execution, prices
):
    report = _summarize(accounting, evidence, execution, prices)
    assert report["estimated_cost_usd"] == pytest.approx(
        (750 * 2 + 200 * 1 + 50 * 3 + 100 * 4) / 1e6
    )
    assert report["totals"]["tokens"]["reasoning_output_tokens"] == 70


def test_missing_cache_write_counter_produces_range(
    accounting, evidence, execution, prices
):
    del evidence["astra"]["turns"][0]["cache_write_input_tokens"]
    report = _summarize(accounting, evidence, execution, prices)
    assert report["usage_complete"] is True
    assert report["estimated_cost_usd"] is None
    assert report["estimated_cost_range_usd"] == pytest.approx(
        {"minimum": 0.0022, "maximum": 0.003}
    )
    assert report["totals"]["tokens"]["cache_write_input_tokens"] is None
    assert report["totals"]["missing_counter_records"]["cache_write_input_tokens"] == 1


def test_missing_cache_counters_use_all_feasible_partitions(
    accounting, evidence, execution, prices
):
    del evidence["astra"]["turns"][0]["cached_input_tokens"]
    del evidence["astra"]["turns"][0]["cache_write_input_tokens"]
    report = _summarize(accounting, evidence, execution, prices)
    assert report["estimated_cost_range_usd"] == pytest.approx(
        {"minimum": 0.0014, "maximum": 0.0034}
    )
    assert report["totals"]["tokens"]["cached_input_tokens"] is None


def test_aggregate_turn_cannot_select_context_tariff(
    accounting, evidence, execution, prices
):
    prices["models"]["coordinator"]["rate_options"].append(_rates(4, 8, 2, 6))
    report = _summarize(accounting, evidence, execution, prices)
    assert report["estimated_cost_usd"] is None
    assert report["estimated_cost_range_usd"] == pytest.approx(
        {"minimum": 0.00225, "maximum": 0.0045}
    )


def test_no_cache_discount_for_full_input_policy(
    accounting, evidence, execution, prices
):
    response = _response("specialist", True)
    response["usage"]["cached_tokens"] = 100
    evidence["specialists"]["responses"] = [response]
    report = _summarize(accounting, evidence, execution, prices)
    specialist = report["models"]["specialist"]
    assert specialist["tokens"]["cached_input_tokens"] == 100
    assert specialist["priced_recorded_cost_usd"]["minimum"] == pytest.approx(0.00028)
    assert specialist["tokens"]["cache_write_input_tokens"] is None


def test_missing_response_usage_never_becomes_zero_cost(
    accounting, evidence, execution, prices
):
    evidence["specialists"]["responses"] = [{"model": "specialist", "usage": {}}]
    report = _summarize(accounting, evidence, execution, prices)
    assert report["usage_complete"] is False
    assert report["estimated_cost_range_usd"] is None
    assert report["totals"]["tokens"]["input_tokens"] is None
    assert report["totals"]["known_tokens"]["input_tokens"] == 1000
    assert report["totals"]["unpriced_records"] == 1
    assert report["totals"]["priced_recorded_cost_usd"]["minimum"] > 0


@pytest.mark.parametrize(
    "source", ["astra", "specialists", "recorder", "snapshot", "interrupt"]
)
def test_incomplete_receipt_cannot_be_promoted(
    accounting, evidence, execution, prices, source
):
    if source in {"astra", "specialists"}:
        evidence[source]["usage_complete"] = False
    elif source == "recorder":
        evidence["usage_complete"] = False
    elif source == "snapshot":
        execution["snapshot_errors"] = {"task-receipts": "OSError"}
    else:
        execution["exit_code"] = None
    report = _summarize(accounting, evidence, execution, prices)
    assert report["usage_complete"] is False
    assert report["estimated_cost_range_usd"] is None
    assert report["totals"]["priced_records"] == 1


def test_missing_prices_leave_cost_unknown(accounting, evidence, execution, prices):
    del prices["models"]["coordinator"]["rate_options"][0][
        "cache_write_price_per_million_tokens"
    ]
    report = _summarize(accounting, evidence, execution, prices)
    assert report["usage_complete"] is True
    assert report["estimated_cost_range_usd"] is None
    assert report["totals"]["unpriced_records"] == 1
    assert report["totals"]["priced_recorded_cost_usd"] is None


def test_zero_cache_writes_do_not_require_write_price(
    accounting, evidence, execution, prices
):
    evidence["astra"]["turns"][0]["cache_write_input_tokens"] = 0
    del prices["models"]["coordinator"]["rate_options"][0][
        "cache_write_price_per_million_tokens"
    ]
    report = _summarize(accounting, evidence, execution, prices)
    assert report["estimated_cost_usd"] == pytest.approx(0.0022)


@pytest.mark.parametrize(
    "key,value",
    [
        ("input_tokens", -1),
        ("output_tokens", True),
        ("cached_input_tokens", 9999),
        ("reasoning_output_tokens", 101),
    ],
)
def test_invalid_token_counters_fail(
    accounting, evidence, execution, prices, key, value
):
    evidence["astra"]["turns"][0][key] = value
    with pytest.raises(ValueError):
        _summarize(accounting, evidence, execution, prices)


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True])
def test_invalid_rates_fail(accounting, evidence, execution, prices, value):
    prices["models"]["coordinator"]["rate_options"][0][
        "input_price_per_million_tokens"
    ] = value
    with pytest.raises(ValueError, match="finite nonnegative"):
        _summarize(accounting, evidence, execution, prices)


def test_tool_scope_and_pending_tasks_remain_visible(
    accounting, evidence, execution, prices
):
    evidence["astra"]["matched_tool_scope"] = False
    execution["unfinished_tasks"] = ["private-task-identity"]
    report = _summarize(accounting, evidence, execution, prices)
    assert report["matched_tool_scope"] is False
    assert report["unfinished_tasks_count"] == 1
    assert "private-task-identity" not in json.dumps(report)
    assert report["workflow_quality_assessed"] is False


def test_cli_reads_frozen_evidence_without_overwriting(
    accounting, evidence, execution, prices, tmp_path
):
    for name, value in (
        ("usage", evidence),
        ("execution", execution),
        ("prices", prices),
    ):
        (tmp_path / (name + ".json")).write_text(json.dumps(value))
    original = (tmp_path / "usage.json").read_bytes()
    argv = [
        sys.executable,
        accounting.__file__,
        "--evidence",
        str(tmp_path),
        "--prices",
        str(tmp_path / "prices.json"),
        "--coordinator-model",
        "coordinator",
        "--output",
        str(tmp_path / "report.json"),
    ]
    result = subprocess.run(argv, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert json.loads((tmp_path / "report.json").read_text())["usage_complete"] is True
    assert (tmp_path / "usage.json").read_bytes() == original
    repeated = subprocess.run(argv, capture_output=True, text=True, check=False)
    assert repeated.returncode != 0
    assert "FileExistsError" in repeated.stderr


def _router_response(status, *, attempted=True, usage=None, **extra):
    return {
        "task_id": "private-router-task",
        "provider": "typesafe",
        "model": "jev-1.13.0",
        "status": status,
        "api_call_attempted": attempted,
        "usage": {} if usage is None else usage,
        **extra,
    }


def test_router_acceptance_and_abstention_are_both_priced(
    accounting, evidence, execution, prices
):
    evidence["router"] = {
        "usage_complete": True,
        "responses": [
            _router_response(
                "accepted", usage={"input_tokens": 100, "output_tokens": 2}
            ),
            _router_response(
                "abstained",
                usage={"input_tokens": 200, "output_tokens": 3},
                fallback=True,
            ),
        ],
    }
    prices["models"]["jev-1.13.0"] = {
        "cache_policy": "full-input",
        "rate_options": [_rates(0.042, 0)],
    }
    report = _summarize(accounting, evidence, execution, prices)
    assert report["estimated_cost_usd"] == pytest.approx(0.00225 + 300 * 0.042 / 1e6)
    router = report["models"]["jev-1.13.0"]
    assert router["recorded_api_calls"] == 2
    assert router["accepted_specialist_responses"] == 0
    assert router["tokens"]["cached_input_tokens"] is None
    assert report["router"]["jev_responses_received"] == 2
    assert report["router"]["accepted_jev_routes"] == 1
    assert report["router"]["fallback_decisions"] == 1
    assert "private-router-task" not in json.dumps(report)


def test_unverified_router_model_keeps_tokens_without_inventing_price(
    accounting, evidence, execution, prices
):
    evidence["router"] = {
        "usage_complete": False,
        "responses": [
            _router_response(
                "unavailable",
                provider="token_factory",
                model="specialist",
                model_verified=False,
                usage={"input_tokens": 100, "output_tokens": 2},
            )
        ],
    }
    report = _summarize(accounting, evidence, execution, prices)
    assert report["estimated_cost_range_usd"] is None
    unknown = report["models"]["<unverified-router-model>"]
    assert unknown["tokens"]["input_tokens"] == 100
    assert unknown["tokens"]["output_tokens"] == 2
    assert report["totals"]["unpriced_records"] == 1


def test_missing_jev_key_is_no_call_not_a_free_jev_response(
    accounting, evidence, execution, prices
):
    evidence["router"] = {
        "usage_complete": True,
        "responses": [
            _router_response(
                "unavailable",
                attempted=False,
                reason="missing_credential",
                fallback=True,
            )
        ],
    }
    report = _summarize(accounting, evidence, execution, prices)
    assert report["usage_complete"] is True
    assert report["estimated_cost_usd"] == pytest.approx(0.00225)
    assert "jev-1.13.0" not in report["models"]
    assert report["router"]["api_calls_not_attempted"] == 1
    assert report["router"]["jev_responses_received"] == 0
    assert report["router"]["accepted_jev_routes"] == 0
    assert report["router"]["unavailable_routes"] == 1


@pytest.mark.parametrize("reason", ["provider_http_error", "provider_transport_error"])
def test_failed_router_call_keeps_unknown_spend(
    accounting, evidence, execution, prices, reason
):
    evidence["router"] = {
        "usage_complete": False,
        "responses": [_router_response("unavailable", reason=reason, fallback=True)],
    }
    report = _summarize(accounting, evidence, execution, prices)
    assert report["usage_complete"] is False
    assert report["estimated_cost_range_usd"] is None
    assert report["models"]["jev-1.13.0"]["tokens"]["input_tokens"] is None
    assert report["totals"]["unpriced_records"] == 1
    assert report["router"]["api_calls_attempted"] == 1
    assert report["router"]["jev_responses_received"] == 0


def test_interrupted_router_intent_is_not_reclassified_as_no_call(
    accounting, evidence, execution, prices
):
    evidence["router"] = {
        "usage_complete": False,
        "responses": [_router_response("started", attempted="unknown")],
    }
    report = _summarize(accounting, evidence, execution, prices)
    assert report["estimated_cost_range_usd"] is None
    assert report["models"]["jev-1.13.0"]["recorded_api_calls"] is None
    assert report["router"]["unknown_api_call_attempts"] == 1
    assert "router_call_outcome_unknown" in report["incomplete_reasons"]


def test_historical_receipts_do_not_acquire_router_claims(
    accounting, evidence, execution, prices
):
    report = _summarize(accounting, evidence, execution, prices)
    assert "router" not in report
    assert report["estimated_cost_usd"] == pytest.approx(0.00225)


def test_provider_failure_is_unpriced_even_when_fallback_completes(
    accounting, evidence, execution, prices
):
    evidence["specialists"]["responses"] = [_response("specialist", True)]
    evidence["specialists"]["failures"] = [
        {
            "type": "provider_failure",
            "model": "unavailable",
            "status_code": 404,
            "usage_missing": True,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            "request_metrics": {"attempts": 1},
        }
    ]
    prices["models"]["unavailable"] = prices["models"]["specialist"]
    report = _summarize(accounting, evidence, execution, prices)
    assert report["usage_complete"] is False
    assert report["estimated_cost_range_usd"] is None
    assert "provider_failure_usage_unknown" in report["incomplete_reasons"]
    failed = report["models"]["unavailable"]
    assert failed["failed_specialist_requests"] == 1
    assert failed["recorded_api_calls"] is None
    assert failed["tokens"]["input_tokens"] is None
    assert failed["priced_recorded_cost_usd"] is None
    assert failed["accepted_specialist_responses"] == 0
    assert failed["rejected_specialist_responses"] == 0
    assert report["models"]["specialist"]["accepted_specialist_responses"] == 1
    assert report["totals"]["priced_recorded_cost_usd"]["minimum"] == pytest.approx(
        0.00253
    )
