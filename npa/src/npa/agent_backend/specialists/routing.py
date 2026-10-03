"""Journal advisory endpoint selection inside an already authorized specialist profile."""

from __future__ import annotations

import os

from npa.agent_backend.model_router import JEV_MODEL, classify_generation_model
from npa.agent_backend.trajectory import redact
from npa.clients.credentials import load_credentials

from .store import UncertainOperation
from .token_router import _classify


def _endpoints(profile, state):
    endpoints = [profile, *profile.fallback_models]
    order = state.get("model_order", list(range(len(endpoints))))
    if sorted(order) != list(range(len(endpoints))):
        raise ValueError("model route is not a permutation of eligible endpoints")
    return [endpoints[index] for index in order]


def _model_order(profile, task):
    names = [profile.model, *(item.model for item in profile.fallback_models)]
    selection = task["route"].get("model_selection")
    if not selection:
        return list(range(len(names)))
    order = selection["endpoint_order"]
    if sorted(order) != sorted(names):
        raise ValueError("recorded model route changed eligible endpoints")
    return [names.index(name) for name in order]


def _route_model(profile, task, store):
    if profile.model_router == "explicit":
        return task
    previous = task["route"].get("model_selection")
    if previous:
        if previous["status"] == "started":
            raise UncertainOperation(
                "Model routing result was lost; do not repeat the paid request"
            )
        _model_order(profile, task)
        _require_selection(profile, previous)
        return task
    provider, model, key_env = _router_settings(profile)
    key = os.environ.get(key_env, "") or load_credentials().tokens.get(key_env, "")
    intent = {
        "provider": provider,
        "model": model,
        "status": "started",
        "api_call_attempted": "unknown",
        "usage": {},
        "attempt_id": task["id"] + ":model-selection",
    }
    if key:
        store._model_route(task["id"], intent)
    decision = _classify_task(profile, task["goal"], key)
    receipt = _route_receipt(profile, intent, decision, attempted=bool(key))
    store._model_route(task["id"], receipt)
    _require_selection(profile, receipt)
    return store._get(task["id"])


def _router_settings(profile):
    if profile.model_router == "jev":
        return "typesafe", JEV_MODEL, "TYPESAFE_API_KEY"
    endpoint = profile.routing_model
    return "token_factory", endpoint.model, endpoint.key_env


def _classify_task(profile, goal, key):
    if profile.model_router == "jev":
        return classify_generation_model(
            redact(goal), profile.model_criteria, api_key=key
        )
    return _classify(
        profile.routing_model, redact(goal), profile.model_criteria, api_key=key
    )


def _require_selection(profile, receipt):
    if profile.require_model_route and not (
        receipt.get("status") == "accepted"
        and receipt.get("api_call_attempted") is True
        and receipt.get("fallback") is False
        and receipt.get("selected_model")
        in {profile.model, *(item.model for item in profile.fallback_models)}
    ):
        name = "Jev" if profile.model_router == "jev" else "Token Factory"
        raise ValueError(
            f"Required {name} model selection was not accepted; generation blocked"
        )


def _route_receipt(profile, intent, decision, *, attempted):
    names = [profile.model, *(item.model for item in profile.fallback_models)]
    selected = decision.get("selected_model")
    if selected is not None and selected not in names:
        raise ValueError("router selected an endpoint outside the configured profile")
    effective = selected or profile.model
    return {
        **intent,
        **decision,
        "api_call_attempted": attempted,
        "effective_model": effective,
        "fallback": selected is None,
        "endpoint_order": [effective, *(name for name in names if name != effective)],
    }
