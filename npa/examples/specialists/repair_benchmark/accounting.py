"""Reconcile Astra request telemetry and ToFa receipts without hiding missing or setup usage."""

from decimal import Decimal
import hashlib
import importlib.util
import json

COUNTERS = {
    "input_token_count": "input_tokens",
    "output_token_count": "output_tokens",
    "cached_token_count": "cached_input_tokens",
    "cache_write_token_count": "cache_write_input_tokens",
    "reasoning_token_count": "reasoning_output_tokens",
}


def _read(path):
    return json.loads(path.read_text())


def _jsonlines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _turns(directory):
    identity, result = None, {}
    for event in _jsonlines(directory / "run/codex.jsonl"):
        if event.get("type") == "thread.started":
            identity = hashlib.sha256(event["thread_id"].encode()).hexdigest()
        if event.get("type") == "turn.completed":
            if not identity or identity in result:
                raise ValueError("ambiguous coordinator invocation identity")
            result[identity] = event["usage"]
    if not result:
        raise ValueError("no measured Astra completion usage")
    return result


def _tokens(record):
    if any(type(record.get(key)) is not int or record[key] < 0 for key in COUNTERS):
        raise ValueError("missing or invalid per-request token counters")
    return {destination: record[source] for source, destination in COUNTERS.items()}


def _sum_tokens(records):
    return {name: sum(row[name] for row in records) for name in COUNTERS.values()}


def _request_cost(tokens, prices):
    ordinary = (
        tokens["input_tokens"]
        - tokens["cached_input_tokens"]
        - tokens["cache_write_input_tokens"]
    )
    if ordinary < 0 or tokens["reasoning_output_tokens"] > tokens["output_tokens"]:
        raise ValueError("overlapping or impossible token counters")
    long_context = tokens["input_tokens"] > 272000
    rates = prices["models"]["gpt-6-astra"]["rate_options"][int(long_context)]
    parts = (
        (ordinary, "input_price_per_million_tokens"),
        (tokens["cached_input_tokens"], "cached_input_price_per_million_tokens"),
        (tokens["cache_write_input_tokens"], "cache_write_price_per_million_tokens"),
        (tokens["output_tokens"], "output_price_per_million_tokens"),
    )
    return (
        sum(Decimal(count) * Decimal(str(rates[key])) for count, key in parts) / 1000000
    )


def _completion_tokens(identity, telemetry):
    rows = [
        item["attributes"]
        for item in telemetry
        if item["attributes"].get("conversation.id_sha256") == identity
    ]
    if any(
        row.get("success") is False
        or row.get("http.response.status_code", 200) >= 400
        or row.get("event.kind")
        in {"response.failed", "response.error", "response.incomplete"}
        for row in rows
    ):
        raise ValueError(
            "failed coordinator request requires separate usage reconciliation"
        )
    completed = [
        _tokens(row)
        for row in rows
        if row.get("event.kind") == "response.completed" and "input_token_count" in row
    ]
    if not completed:
        raise ValueError("missing per-request coordinator completion evidence")
    return completed


def _thread_cost(identity, expected, telemetry, prices):
    completed = _completion_tokens(identity, telemetry)
    setup = Decimal(0)
    charged = completed
    if _sum_tokens(charged) != expected:
        first, rest = completed[0], completed[1:]
        if (
            not rest
            or _sum_tokens(rest) != expected
            or first["input_tokens"] <= 0
            or any(first[key] != 0 for key in first if key != "input_tokens")
        ):
            raise ValueError("CLI and telemetry counters do not reconcile")
        setup, charged = _request_cost(first, prices), rest
    return {
        "charged": sum((_request_cost(row, prices) for row in charged), Decimal(0)),
        "setup_upper": setup,
        "requests": len(completed),
        "maximum_prompt_tokens": max(row["input_tokens"] for row in completed),
        "reconciled_tokens": expected,
    }


def _usage_report(root, directory, prices):
    spec = importlib.util.spec_from_file_location(
        "benchmark_workflow_usage", root / "workflow_usage.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.summarize_usage(
        _read(directory / "run/usage.json"),
        _read(directory / "run/execution.json"),
        prices,
    )


def _tofa_cost(report):
    total = Decimal(0)
    for model, row in report["models"].items():
        if model == "gpt-6-astra":
            continue
        bounds = row["priced_recorded_cost_usd"]
        if (
            row["unpriced_records"]
            or not bounds
            or bounds["minimum"] != bounds["maximum"]
        ):
            raise ValueError("missing or ambiguous ToFa request pricing")
        total += Decimal(str(bounds["minimum"]))
    return total


def _measured_turns(directory, prices):
    telemetry = _jsonlines(directory / "telemetry-snapshot.jsonl")
    return [
        _thread_cost(identity, expected, telemetry, prices)
        for identity, expected in _turns(directory).items()
    ]


def _account(root, directory, prices):
    result = {
        "complete": False,
        "total_usd": None,
        "actual_invoice_or_service_tier_proven": False,
        "basis": "Standard API equivalents; includes runtime Astra and ToFa, excludes benchmark development and hosting.",
    }
    try:
        report = _usage_report(root, directory, prices)
        result["usage_report"] = report
        if not report["usage_complete"] or not report["matched_tool_scope"]:
            raise ValueError("incomplete usage or tools outside the matched scope")
        turns = _measured_turns(directory, prices)
        astra = sum((turn["charged"] for turn in turns), Decimal(0))
        setup = sum((turn["setup_upper"] for turn in turns), Decimal(0))
        tofa = _tofa_cost(report)
        result.update(
            complete=True,
            astra_usd=float(astra),
            tofa_usd=float(tofa),
            setup_uncertainty_usd={"minimum": 0, "maximum": float(setup)},
            total_usd={
                "minimum": float(astra + tofa),
                "maximum": float(astra + tofa + setup),
            },
            coordinator_invocations=len(turns),
            coordinator_response_records=sum(turn["requests"] for turn in turns),
            maximum_coordinator_prompt_tokens=max(
                turn["maximum_prompt_tokens"] for turn in turns
            ),
        )
    except (KeyError, OSError, TypeError, ValueError) as error:
        result["incomplete_reason"] = (
            str(error) if type(error) is ValueError else type(error).__name__
        )
    return result
