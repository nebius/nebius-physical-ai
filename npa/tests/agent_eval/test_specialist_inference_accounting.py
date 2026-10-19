"""Reconcile actual paid attempts, partial usage and optional cached-token prices."""

from copy import deepcopy

import pytest

from npa.agent_backend.specialists.accounting import (
    _normalized_usage,
    summarize_inference,
)


_RATES = {
    "synthetic/worker": {
        "input_price_per_million_tokens": 1.4,
        "output_price_per_million_tokens": 4.4,
    },
    "synthetic/router": {
        "input_price_per_million_tokens": 0.06,
        "output_price_per_million_tokens": 0.24,
    },
}


def test_router_and_worker_usage_are_separate_without_double_counting_intents():
    events = [
        {
            "type": "configuration_routing",
            "boundary_id": "boundary",
            "status": "started",
        },
        {
            "type": "configuration_routing",
            "boundary_id": "boundary",
            "status": "accepted",
            "model": "synthetic/router",
            "usage": {"input_tokens": 100, "output_tokens": 10},
        },
        {
            "type": "inference_attempt",
            "attempt_id": "attempt",
            "model": "synthetic/worker",
            "status": "started",
        },
        {
            "type": "model",
            "attempt_id": "attempt",
            "model": "synthetic/worker",
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 20,
                "cached_tokens": 100,
            },
        },
    ]
    result = summarize_inference(events, _RATES)
    assert result["worker_requests"] == result["classifier_requests"] == 1
    assert result["unknown_usage_requests"] == 0
    assert result["usage"]["input_tokens"] == 1100
    assert result["uncached_token_usd_estimate"] == pytest.approx(0.0014964)
    assert result["cache_adjusted_token_usd_estimate"] is None
    assert not result["invoice_verified"]


@pytest.mark.parametrize(
    "cached",
    [
        {"cached_tokens": 60},
        {"prompt_cache_hit_tokens": 60},
        {"cached_input_tokens": 60},
        {"prompt_tokens_details": {"cached_tokens": 60}},
    ],
)
def test_cache_aliases_and_explicit_discount(cached):
    rates = deepcopy(_RATES)
    rates["synthetic/worker"]["input_cached_price_per_million_tokens"] = 0.14
    event = {
        "type": "model",
        "model": "synthetic/worker",
        "usage": {"prompt_tokens": 100, "completion_tokens": 10, **cached},
    }
    result = summarize_inference([event], rates)
    assert result["usage"]["cached_input_tokens"] == 60
    assert result["uncached_token_usd_estimate"] == pytest.approx(0.000184)
    assert result["cache_adjusted_token_usd_estimate"] == pytest.approx(0.0001084)


@pytest.mark.parametrize(
    "usage",
    [
        None,
        {},
        {"prompt_tokens": True, "completion_tokens": 3},
        {"prompt_tokens": -1, "completion_tokens": 3},
        {"input_tokens": 100, "prompt_tokens": 101, "completion_tokens": 3},
    ],
)
def test_missing_or_conflicting_counters_never_become_free_usage(usage):
    result = summarize_inference(
        [{"type": "model", "model": "synthetic/worker", "usage": usage}], _RATES
    )
    assert result["unknown_usage_requests"] == 1
    assert result["uncached_token_usd_estimate"] is None
    assert result["usage"]["input_tokens"] is None


@pytest.mark.parametrize(
    "usage",
    [
        {"prompt_tokens": 10, "completion_tokens": 2, "cached_tokens": 11},
        {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "cached_tokens": 1,
            "prompt_cache_hit_tokens": 2,
        },
    ],
)
def test_invalid_cache_count_keeps_uncached_estimate_but_no_discount(usage):
    result = summarize_inference(
        [{"type": "model", "model": "synthetic/worker", "usage": usage}], _RATES
    )
    assert result["unknown_usage_requests"] == 0
    assert result["usage"]["cached_input_tokens"] is None
    assert result["uncached_token_usd_estimate"] is not None
    assert result["cache_adjusted_token_usd_estimate"] is None


def test_unfinished_and_failed_requests_remain_unknown_and_are_not_counted_twice():
    events = [
        {
            "type": "inference_attempt",
            "attempt_id": "failed",
            "model": "synthetic/worker",
            "status": "started",
        },
        {
            "type": "provider_failure",
            "attempt_id": "failed",
            "model": "synthetic/worker",
            "usage": {},
        },
        {
            "type": "inference_attempt",
            "attempt_id": "lost",
            "model": "synthetic/worker",
            "status": "started",
        },
        {
            "type": "configuration_routing",
            "boundary_id": "lost-router",
            "status": "started",
            "model": "synthetic/router",
        },
    ]
    result = summarize_inference(events, _RATES)
    assert result["worker_requests"] == 2 and result["classifier_requests"] == 1
    assert result["unknown_usage_requests"] == 3
    assert result["uncached_token_usd_estimate"] is None


def test_rejected_generation_still_costs_tokens():
    event = {
        "type": "model",
        "model": "synthetic/worker",
        "accepted": False,
        "usage": {"prompt_tokens": 100, "completion_tokens": 10, "cached_tokens": 0},
    }
    result = summarize_inference([event], _RATES)
    assert result["uncached_token_usd_estimate"] == pytest.approx(0.000184)
    assert result["cache_adjusted_token_usd_estimate"] == pytest.approx(0.000184)


def test_missing_price_never_estimates_zero_cost():
    result = summarize_inference(
        [
            {
                "type": "model",
                "model": "synthetic/unknown",
                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
            }
        ],
        _RATES,
    )
    assert result["unknown_usage_requests"] == 0
    assert result["uncached_token_usd_estimate"] is None


@pytest.mark.parametrize("price", [-1, float("nan"), float("inf"), "bad"])
def test_invalid_price_is_rejected(price):
    rates = deepcopy(_RATES)
    rates["synthetic/worker"]["input_price_per_million_tokens"] = price
    with pytest.raises(ValueError, match="invalid token price"):
        summarize_inference(
            [
                {
                    "type": "model",
                    "model": "synthetic/worker",
                    "usage": {"prompt_tokens": 10, "completion_tokens": 2},
                }
            ],
            rates,
        )


def test_retry_usage_is_not_inferred_from_final_response():
    event = {
        "type": "model",
        "model": "synthetic/worker",
        "request_metrics": {"attempts": 2},
        "usage": {"prompt_tokens": 100, "completion_tokens": 10},
    }
    result = summarize_inference([event], _RATES)
    assert result["retry_attempts_with_unknown_usage"] == 1
    assert result["unknown_usage_requests"] == 1
    assert result["uncached_token_usd_estimate"] is None
    assert result["usage"]["input_tokens"] is None
    assert result["reported_usage"]["input_tokens"] == 100


def test_export_omits_provider_ids_and_request_content():
    event = {
        "type": "model",
        "model": "synthetic/worker",
        "response_id": "internal-response",
        "headers": {"authorization": "synthetic-secret"},
        "content": "private-source",
        "usage": {"prompt_tokens": 10, "completion_tokens": 2},
    }
    result = str(summarize_inference([event], _RATES))
    assert all(
        value not in result
        for value in ("internal-response", "synthetic-secret", "private-source")
    )


def test_no_requests_is_zero_but_partial_usage_remains_partial():
    assert summarize_inference([], _RATES)["uncached_token_usd_estimate"] == 0
    assert _normalized_usage({"prompt_tokens": -1, "completion_tokens": 8}) == {
        "output_tokens": 8
    }
