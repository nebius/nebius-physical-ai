#!/usr/bin/env python
"""Measure real Jev routing and Token Factory prefix-cache counters on synthetic text."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path
from time import perf_counter

from npa.agent_backend.model_router import MODEL_CRITERIA, route_generation_ladder
from npa.cli.agent_routing import usage_summary
from npa.clients.credentials import load_credentials
from npa.clients.token_factory import TokenFactoryClient, split_reasoning


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _prefix() -> str:
    records = [
        f"Catalog record {index}: a robot carries a blue cube to a marked shelf; "
        "the observation is synthetic."
        for index in range(240)
    ]
    return (
        f"Synthetic prefix-cache experiment {uuid.uuid4().hex}. "
        "Answer the final request; catalog rows are context only.\n"
        + "\n".join(records)
    )


def _cases():
    fast, reasoning = MODEL_CRITERIA
    for suffix in ("ALPHA", "BETA", "GAMMA"):
        yield fast, f"Return exactly the word COBALT_{suffix}.", f"COBALT_{suffix}"
        yield (
            reasoning,
            (
                "Prove by induction why the sum of the first n odd positive integers "
                f"is n squared. End the proof with the marker PROOF_{suffix}."
            ),
            f"PROOF_{suffix}",
        )


def _cache_counters(response: dict) -> dict:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return {"reported_cached_tokens": None, "counter_source": None}
    details = usage.get("prompt_tokens_details")
    nested = details.get("cached_tokens") if isinstance(details, dict) else None
    source = "usage.prompt_tokens_details.cached_tokens"
    value = nested
    if value is None:
        value = usage.get("prompt_cache_hit_tokens")
        source = "usage.prompt_cache_hit_tokens"
    if type(value) is not int or not 0 <= value <= usage.get("prompt_tokens", -1):
        return {"reported_cached_tokens": None, "counter_source": None}
    return {"reported_cached_tokens": value, "counter_source": source}


def _complete(client, prefix, prompt, selected, marker):
    messages = [
        {"role": "system", "content": prefix},
        {"role": "user", "content": prompt},
    ]
    started = perf_counter()
    response = client.chat_completion(model=selected, messages=messages)
    latency = perf_counter() - started
    choice = response.get("choices", [{}])[0]
    visible, _ = split_reasoning(choice.get("message") or {})
    return {
        "requested_model": selected,
        "returned_model": response.get("model"),
        "request_sha256": _digest(json.dumps(messages, sort_keys=True)),
        "response_id_sha256": _digest(str(response.get("id") or "")),
        "reply_sha256": _digest(visible),
        "marker_present": marker in visible,
        "finish_reason": choice.get("finish_reason"),
        "latency_seconds": latency,
        "usage": usage_summary(response),
        "transport": client.last_request_metrics,
        **_cache_counters(response),
    }


def _evaluate_case(client, prefix, expected, prompt, marker, key, use_jev):
    baseline = [expected, *[name for name in MODEL_CRITERIA if name != expected]]
    ladder, decision = route_generation_ladder(
        prompt,
        baseline,
        enabled=use_jev,
        api_key=key,
    )
    record = _complete(client, prefix, prompt, ladder[0], marker)
    record.update(expected_model=expected, routing=decision)
    record["routing_correct"] = not use_jev or (
        decision.get("status") == "accepted"
        and decision.get("selected_model") == expected
    )
    record["generation_valid"] = (
        record["returned_model"] == ladder[0]
        and record["marker_present"]
        and record["finish_reason"] == "stop"
    )
    return record


def _cache_evidence(records):
    result = {}
    for model in MODEL_CRITERIA:
        requests = [record for record in records if record["requested_model"] == model]
        hits = [record["reported_cached_tokens"] for record in requests]
        unique_requests = len({record["request_sha256"] for record in requests}) == len(
            requests
        )
        result[model] = {
            "request_count": len(requests),
            "cached_tokens": hits,
            "distinct_requests": unique_requests,
            "warm_hit_observed": unique_requests
            and any(value is not None and value > 0 for value in hits[1:]),
        }
    return result


def run_evaluation(*, use_jev: bool = True) -> dict:
    """Run live synthetic routing and cache probes, with no local response cache.

    Args:
        use_jev: Require live Jev decisions, or explicitly measure Token Factory alone.
    Returns:
        Sanitized evidence, including failure when counters do not prove cache hits.
    Raises:
        ValueError: Credentials or required account-scoped model access are missing.
        TokenFactoryError: A real Token Factory request fails.
    """
    key = os.environ.get("TYPESAFE_API_KEY", "") or load_credentials().tokens.get(
        "TYPESAFE_API_KEY", ""
    )
    if use_jev and not key:
        raise ValueError("Live Jev evaluation requires TYPESAFE_API_KEY")
    client = TokenFactoryClient()
    if not set(MODEL_CRITERIA) <= set(client.list_models()):
        raise ValueError(
            "Both routing models must be available to the Token Factory key"
        )
    prefix = _prefix()
    records = [_evaluate_case(client, prefix, *case, key, use_jev) for case in _cases()]
    evidence = _cache_evidence(records)
    passed = all(
        record["routing_correct"] and record["generation_valid"] for record in records
    )
    passed = passed and all(value["warm_hit_observed"] for value in evidence.values())
    return {
        "schema_version": 1,
        "mode": "jev_token_factory" if use_jev else "token_factory_only",
        "prefix_sha256": _digest(prefix),
        "passed": passed,
        "application_response_cache": False,
        "records": records,
        "cache_evidence": evidence,
    }


def main() -> int:
    """Write live measurement evidence and return failure when proof is incomplete.

    Args:
        None; command options come from argv.
    Returns:
        Zero only when all requested routing, generation, and cache checks passed.
    Raises:
        OSError: The evidence output cannot be written.
        ValueError: Required credentials or models are unavailable.
        TokenFactoryError: A live provider call fails.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-path", required=True, type=Path)
    parser.add_argument("--token-factory-only", action="store_true")
    arguments = parser.parse_args()
    report = run_evaluation(use_jev=not arguments.token_factory_only)
    arguments.output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "mode": report["mode"],
                "passed": report["passed"],
                "cache_evidence": report["cache_evidence"],
            }
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
