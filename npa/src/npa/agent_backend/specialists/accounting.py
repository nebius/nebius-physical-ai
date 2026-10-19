"""Reconcile inference receipts without treating unknown usage or cache prices as zero."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation


def _counter(values):
    present = [value for value in values if value is not None]
    if not present or any(type(value) is not int or value < 0 for value in present):
        return None
    return present[0] if len(set(present)) == 1 else None


def _normalized_usage(raw):
    if not isinstance(raw, dict):
        return {}
    usage = {}
    for target, aliases in (
        ("input_tokens", ("input_tokens", "prompt_tokens")),
        ("output_tokens", ("output_tokens", "completion_tokens")),
    ):
        value = _counter([raw.get(name) for name in aliases])
        if value is not None:
            usage[target] = value
    details = raw.get("prompt_tokens_details")
    cached = _counter(
        [
            raw.get("cached_input_tokens"),
            raw.get("cached_tokens"),
            raw.get("prompt_cache_hit_tokens"),
            details.get("cached_tokens") if isinstance(details, dict) else None,
        ]
    )
    if cached is not None and cached <= usage.get("input_tokens", -1):
        usage["cached_input_tokens"] = cached
    return usage


def _inference_receipts(events):
    receipts = {}
    for ordinal, event in enumerate(events):
        kind = event.get("type")
        if kind == "configuration_routing":
            key = ("classifier", event["boundary_id"])
        elif kind in {"inference_attempt", "model", "provider_failure"}:
            key = ("worker", event.get("attempt_id", str(ordinal)))
        else:
            continue
        if event.get("api_call_attempted") is False:
            continue
        receipts[key] = {**event, "request_kind": key[0]}
    return list(receipts.values())


def _rate(card, name):
    value = card.get(name)
    if value is None:
        return None
    try:
        rate = Decimal(str(value))
    except InvalidOperation:
        raise ValueError("invalid token price") from None
    if not rate.is_finite() or rate < 0:
        raise ValueError("invalid token price")
    return rate


def _request_cost(usage, card):
    input_rate = _rate(card, "input_price_per_million_tokens")
    output_rate = _rate(card, "output_price_per_million_tokens")
    if not {"input_tokens", "output_tokens"} <= usage.keys() or None in (
        input_rate,
        output_rate,
    ):
        return None, None
    cost = (
        usage["input_tokens"] * input_rate + usage["output_tokens"] * output_rate
    ) / 1_000_000
    cached = usage.get("cached_input_tokens")
    cached_rate = _rate(card, "input_cached_price_per_million_tokens")
    if cached == 0:
        return float(cost), float(cost)
    if cached is None or cached_rate is None:
        return float(cost), None
    adjusted = cost + cached * (cached_rate - input_rate) / 1_000_000
    return float(cost), float(adjusted)


def _request_record(event, rates):
    usage = _normalized_usage(event.get("usage"))
    uncached, adjusted = _request_cost(usage, rates.get(event.get("model"), {}))
    return {
        "kind": event["request_kind"],
        "model": event.get("model"),
        "status": event.get(
            "status",
            "provider_failure" if event["type"] == "provider_failure" else "received",
        ),
        "usage": usage,
        "usage_known": {"input_tokens", "output_tokens"} <= usage.keys(),
        "uncached_token_usd_estimate": uncached,
        "cache_adjusted_token_usd_estimate": adjusted,
    }


def _complete_sum(records, field):
    values = [record[field] for record in records]
    return sum(values) if all(value is not None for value in values) else None


def _unknown_retries(event):
    metrics = event.get("request_metrics")
    attempts = metrics.get("attempts") if isinstance(metrics, dict) else None
    return max(0, attempts - 1) if type(attempts) is int else 0


def summarize_inference(events, rates=None):
    """Account for worker and router requests, including failed or unfinished attempts.

    Args: events: Durable task events. rates: Optional public model-ID rate cards using
        input/output_price_per_million_tokens and optional input_cached_price_per_million_tokens.
    Returns: Sanitized request ledger, usage totals and explicitly scoped token-cost estimates.
        Missing usage or rate cards make the corresponding total unknown, never zero.
    Raises: ValueError: A supplied price is negative, nonfinite or malformed.
    """
    receipts = _inference_receipts(events)
    records = [_request_record(event, rates or {}) for event in receipts]
    retries = sum(_unknown_retries(event) for event in receipts)
    usage = {
        name: sum(record["usage"][name] for record in records)
        if all(name in record["usage"] for record in records)
        else None
        for name in ("input_tokens", "output_tokens", "cached_input_tokens")
    }
    return {
        "scope": "hosted inference tokens only; coordinator, host, workload compute and taxes excluded",
        "invoice_verified": False,
        "requests": records,
        "usage": {name: None for name in usage} if retries else usage,
        "reported_usage": usage,
        "worker_requests": sum(record["kind"] == "worker" for record in records),
        "classifier_requests": sum(
            record["kind"] == "classifier" for record in records
        ),
        "unknown_usage_requests": sum(not record["usage_known"] for record in records)
        + retries,
        "retry_attempts_with_unknown_usage": retries,
        "uncached_token_usd_estimate": None
        if retries
        else _complete_sum(records, "uncached_token_usd_estimate"),
        "cache_adjusted_token_usd_estimate": None
        if retries
        else _complete_sum(records, "cache_adjusted_token_usd_estimate"),
    }
