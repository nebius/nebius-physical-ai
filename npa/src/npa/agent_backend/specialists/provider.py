"""Record sanitized provider failures before a narrowly authorized endpoint handoff."""

from __future__ import annotations

import math

import httpx

from npa.clients.token_factory import TokenFactoryError

from .recovery import _RejectedGeneration


def _status(error):
    cause = error.__cause__
    if isinstance(cause, httpx.HTTPStatusError):
        return cause.response.status_code
    return None


def _metrics(client, status):
    raw = getattr(client, "last_request_metrics", {})
    metrics = {"status_code": status}
    if not isinstance(raw, dict):
        return metrics
    reported_status = raw.get("status_code")
    if (
        status is None
        and type(reported_status) is int
        and 100 <= reported_status <= 599
    ):
        metrics["status_code"] = reported_status
    for name in ("attempts", "retries"):
        if type(raw.get(name)) is int and raw[name] >= 0:
            metrics[name] = raw[name]
    latency = raw.get("latency_seconds")
    if type(latency) in (int, float) and math.isfinite(latency) and latency >= 0:
        metrics["latency_seconds"] = latency
    return metrics


def _failure_kind(error, status):
    if status == 404:
        return "endpoint_unavailable"
    if status is not None:
        return "provider_http_error"
    if isinstance(error.__cause__, httpx.TransportError):
        return "provider_transport_error"
    return "provider_error"


def _record_failure(client, endpoint, tools, error):
    status = _status(error)
    metrics = _metrics(client, status)
    tools.store._event(
        tools.task_id,
        {
            "type": "provider_failure",
            "model": endpoint.model,
            "status_code": metrics["status_code"],
            "failure_kind": _failure_kind(error, status),
            "request_metrics": metrics,
            "api_call_attempted": True if metrics.get("attempts", 0) > 0 else "unknown",
            "usage": {},
            "usage_missing": True,
        },
    )
    return status


def _request_model(client, endpoint, tools, **arguments):
    try:
        return client.chat_completion(model=endpoint.model, **arguments)
    except TokenFactoryError as error:
        status = _record_failure(client, endpoint, tools, error)
        if status == 404:
            raise _RejectedGeneration(
                "configured endpoint unavailable (HTTP 404)"
            ) from None
        # Provider bodies, request headers and URLs do not belong in durable errors.
        raise TokenFactoryError(
            "provider request failed; inspect provider_failure receipt"
        ) from None
