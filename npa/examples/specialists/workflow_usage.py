"""Summarize frozen workflow experiment usage with explicit operator pricing uncertainty."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


_COUNTERS = {
    "input_tokens": ("input_tokens", "prompt_tokens"),
    "output_tokens": ("output_tokens", "completion_tokens"),
    "cached_input_tokens": ("cached_input_tokens", "cached_tokens"),
    "cache_write_input_tokens": ("cache_write_input_tokens",),
    "reasoning_output_tokens": ("reasoning_output_tokens",),
}
_RATE_KEYS = {
    "input": "input_price_per_million_tokens",
    "output": "output_price_per_million_tokens",
    "cached": "cached_input_price_per_million_tokens",
    "written": "cache_write_price_per_million_tokens",
}


def _counter(usage, names):
    values = [usage[name] for name in names if name in usage]
    if not values:
        return None
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("token counters must be nonnegative integers")
    if len(set(values)) != 1:
        raise ValueError("conflicting token counter aliases")
    return values[0]


def _tokens(usage):
    if not isinstance(usage, dict):
        return dict.fromkeys(_COUNTERS)
    tokens = {name: _counter(usage, aliases) for name, aliases in _COUNTERS.items()}
    prompt, output = tokens["input_tokens"], tokens["output_tokens"]
    cached = tokens["cached_input_tokens"] or 0
    written = tokens["cache_write_input_tokens"] or 0
    reasoning = tokens["reasoning_output_tokens"] or 0
    if prompt is not None and cached + written > prompt:
        raise ValueError("cache token counts exceed input tokens")
    if output is not None and reasoning > output:
        raise ValueError("reasoning tokens exceed output tokens")
    return tokens


def _records(usage, coordinator_model):
    records = []
    for turn in usage.get("astra", {}).get("turns", []):
        records.append((coordinator_model, "coordinator-turn", None, _tokens(turn)))
    for response in usage.get("specialists", {}).get("responses", []):
        model = response.get("model") or "<unreported-model>"
        if not isinstance(model, str):
            raise ValueError("model identifiers must be strings")
        records.append(
            (
                model,
                "specialist-response",
                response.get("accepted"),
                _tokens(response.get("usage")),
            )
        )
    return records + _router_records(usage)


def _router_records(usage):
    records = []
    for response in usage.get("router", {}).get("responses", []):
        attempted = response.get("api_call_attempted", "unknown")
        if attempted is False:
            continue
        model = response.get("model") or "<unreported-router-model>"
        if not isinstance(model, str):
            raise ValueError("model identifiers must be strings")
        records.append(
            (
                model,
                "router-request",
                None if attempted is True else "unknown",
                _tokens(response.get("usage")),
            )
        )
    return records


def _validate_prices(prices):
    models = prices.get("models")
    if not isinstance(models, dict):
        raise ValueError("prices must contain a models object")
    for price in models.values():
        if not isinstance(price, dict):
            raise ValueError("each model price must be an object")
        if price.get("cache_policy") not in {"full-input", "itemized"}:
            raise ValueError("each model needs an explicit cache_policy")
        options = price.get("rate_options")
        if not isinstance(options, list) or not options:
            raise ValueError("each model needs a nonempty rate_options list")
        for rates in options:
            if not isinstance(rates, dict):
                raise ValueError("each rate option must be an object")
            for key in _RATE_KEYS.values():
                if key not in rates:
                    continue
                value = rates[key]
                if (
                    type(value) not in {int, float}
                    or not math.isfinite(value)
                    or value < 0
                ):
                    raise ValueError("token prices must be finite nonnegative numbers")
    return models


def _itemized_cost(tokens, rates):
    cached, written = tokens["cached_input_tokens"], tokens["cache_write_input_tokens"]
    remaining = tokens["input_tokens"] - (cached or 0) - (written or 0)
    fixed = tokens["output_tokens"] * rates["output"]
    possible_rates = [rates["input"]]
    for count, name in ((cached, "cached"), (written, "written")):
        if count == 0 or (count is None and remaining == 0):
            continue
        if rates[name] is None:
            return None
        if count is None:
            possible_rates.append(rates[name])
        else:
            fixed += count * rates[name]
    return (
        (fixed + remaining * min(possible_rates)) / 1e6,
        (fixed + remaining * max(possible_rates)) / 1e6,
    )


def _rate_cost(tokens, price, option):
    if tokens["input_tokens"] is None or tokens["output_tokens"] is None:
        return None
    rates = {name: option.get(key) for name, key in _RATE_KEYS.items()}
    if rates["input"] is None or rates["output"] is None:
        return None
    if price["cache_policy"] == "itemized":
        return _itemized_cost(tokens, rates)
    cost = (
        tokens["input_tokens"] * rates["input"]
        + tokens["output_tokens"] * rates["output"]
    ) / 1e6
    return cost, cost


def _record_cost(tokens, price):
    if price is None:
        return None
    costs = [_rate_cost(tokens, price, rates) for rates in price["rate_options"]]
    if any(cost is None for cost in costs):
        return None
    return min(cost[0] for cost in costs), max(cost[1] for cost in costs)


def _token_totals(records):
    known, missing = {}, {}
    for name in _COUNTERS:
        values = [record[3][name] for record in records]
        known[name] = sum(value for value in values if value is not None)
        missing[name] = values.count(None)
    totals = {name: known[name] if missing[name] == 0 else None for name in _COUNTERS}
    return {"tokens": totals, "known_tokens": known, "missing_counter_records": missing}


def _cost_totals(records, prices):
    costs = [_record_cost(record[3], prices.get(record[0])) for record in records]
    known = [cost for cost in costs if cost is not None]
    bounds = {
        "minimum": sum(cost[0] for cost in known),
        "maximum": sum(cost[1] for cost in known),
    }
    return {
        "priced_records": len(known),
        "unpriced_records": len(costs) - len(known),
        "priced_recorded_cost_usd": bounds if known else None,
    }


def _model_summary(records, price):
    kinds = sorted({record[1] for record in records})
    unknown_calls = "coordinator-turn" in kinds or any(
        record[1] == "router-request" and record[2] == "unknown" for record in records
    )
    return {
        "usage_records": len(records),
        "record_granularity": kinds,
        "recorded_api_calls": None if unknown_calls else len(records),
        "accepted_specialist_responses": sum(record[2] is True for record in records),
        "rejected_specialist_responses": sum(record[2] is False for record in records),
        "cache_policy": price.get("cache_policy") if price else None,
        "rate_options": len(price["rate_options"]) if price else 0,
        **_token_totals(records),
        **_cost_totals(records, {records[0][0]: price}),
    }


def _completeness(usage, execution, records):
    reasons = []
    sections = (
        ("astra", "specialists", "router")
        if "router" in usage
        else ("astra", "specialists")
    )
    for name in sections:
        if usage.get(name, {}).get("usage_complete") is not True:
            reasons.append(name + "_usage_incomplete")
    if any(
        record[1] == "router-request" and record[2] == "unknown" for record in records
    ):
        reasons.append("router_call_outcome_unknown")
    if usage.get("usage_complete") is not True:
        reasons.append("recorder_usage_incomplete")
    if not any(record[1] == "coordinator-turn" for record in records):
        reasons.append("no_coordinator_usage")
    if any(
        record[3][key] is None
        for record in records
        for key in ("input_tokens", "output_tokens")
    ):
        reasons.append("required_token_counters_missing")
    if execution.get("snapshot_errors"):
        reasons.append("snapshot_errors")
    if execution.get("error_type") or execution.get("exit_code") is None:
        reasons.append("execution_interrupted")
    return reasons


def _routing_summary(usage):
    responses = usage["router"].get("responses", [])
    observed = [
        response for response in responses if response.get("api_call_attempted") is True
    ]
    jev = [
        response
        for response in observed
        if response.get("provider") == "typesafe"
        and str(response.get("model", "")).startswith("jev-")
    ]
    statuses = Counter(response.get("status") for response in observed)
    jev_statuses = Counter(response.get("status") for response in jev)
    return {
        "decision_records": len(responses),
        "api_calls_attempted": len(observed),
        "api_calls_not_attempted": sum(
            response.get("api_call_attempted") is False for response in responses
        ),
        "unknown_api_call_attempts": sum(
            type(response.get("api_call_attempted")) is not bool
            for response in responses
        ),
        "accepted_routes": statuses["accepted"],
        "abstained_routes": statuses["abstained"],
        "unavailable_routes": sum(
            response.get("status") == "unavailable" for response in responses
        ),
        "fallback_decisions": sum(
            response.get("fallback") is True for response in responses
        ),
        "jev_responses_received": jev_statuses["accepted"] + jev_statuses["abstained"],
        "accepted_jev_routes": jev_statuses["accepted"],
    }


def _report(usage, execution, records, prices):
    reasons = _completeness(usage, execution, records)
    costs = _cost_totals(records, prices)
    bounds = costs["priced_recorded_cost_usd"]
    estimate = bounds if not reasons and costs["unpriced_records"] == 0 else None
    exact = estimate is not None and estimate["minimum"] == estimate["maximum"]
    return {
        "schema": "npa.specialists.workflow_usage.v1",
        "arm": execution.get("arm"),
        "usage_complete": not reasons,
        "incomplete_reasons": reasons,
        "matched_tool_scope": usage.get("astra", {}).get("matched_tool_scope") is True,
        "unfinished_tasks_count": len(execution.get("unfinished_tasks", [])),
        "agent_tool_seconds": execution.get("agent_tool_seconds"),
        "coordinator_seconds": execution.get("coordinator_seconds"),
        "estimated_cost_range_usd": estimate,
        "estimated_cost_usd": estimate["minimum"] if exact else None,
        "totals": {"usage_records": len(records), **_token_totals(records), **costs},
        "models": {
            model: _model_summary(
                [record for record in records if record[0] == model], prices.get(model)
            )
            for model in sorted({record[0] for record in records})
        },
        "cost_basis": "Operator rates applied to recorded usage; not an invoice.",
        "excluded_costs": [
            "GPU jobs",
            "host compute",
            "storage",
            "network",
            "human work",
        ],
        "workflow_quality_assessed": False,
    }


def summarize_usage(usage, execution, prices, *, coordinator_model="gpt-6-astra"):
    """Account for all recorded generations without inventing missing counters or rates.

    Args:
        usage: Frozen workflow experiment usage.json contents.
        execution: Corresponding finalized execution.json contents.
        prices: Operator table with models, cache_policy and possible rate_options.
        coordinator_model: Actual coordinator model identifier in the price table.

    Returns:
        Per-model counters, recorded cost bounds and complete-run bounds when known.

    Raises:
        ValueError: Token counters or operator prices are invalid.
    """
    models = _validate_prices(prices)
    records = _records(usage, coordinator_model)
    report = _report(usage, execution, records, models)
    if "router" in usage:
        report["router"] = _routing_summary(usage)
    serialized_prices = json.dumps(prices, sort_keys=True, allow_nan=False).encode()
    report["price_table_sha256"] = hashlib.sha256(serialized_prices).hexdigest()
    return report


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--prices", type=Path, required=True)
    parser.add_argument("--coordinator-model", default="gpt-6-astra")
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    usage = json.loads((options.evidence / "usage.json").read_text())
    execution = json.loads((options.evidence / "execution.json").read_text())
    prices = json.loads(options.prices.read_text())
    report = summarize_usage(
        usage, execution, prices, coordinator_model=options.coordinator_model
    )
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if options.output:
        with options.output.open("x") as stream:
            stream.write(payload)
    else:
        print(payload, end="")


if __name__ == "__main__":
    _main()
