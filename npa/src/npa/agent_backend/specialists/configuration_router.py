"""Select capacity-feasible agent configurations with durable, history-aware decisions."""

from __future__ import annotations

import hashlib
import json
import os

from npa.agent_backend.trajectory import redact
from npa.clients.credentials import load_credentials

from .capacity import _admission, _context_view, _visible_messages
from .store import UncertainOperation
from .token_router import _classify, _format, _messages


def _candidates(state, profile, schemas):
    policy = profile.configuration_routing
    previous = next(
        (
            item
            for item in policy.candidates
            if item.name == state.get("configuration_route")
        ),
        None,
    )
    eligible = {}
    receipts = {}
    for candidate in policy.candidates:
        view = _context_view(state["messages"], candidate, profile, previous)
        capacity = _admission(candidate, view, tools=schemas, parallel_tool_calls=False)
        excluded = candidate.name in state.get("configuration_excluded", [])
        receipts[candidate.name] = {
            **capacity,
            "excluded": excluded,
            "tools_verified": candidate.tools_verified,
        }
        if capacity["admitted"] and candidate.tools_verified and not excluded:
            eligible[candidate.name] = (candidate, view)
    return eligible, receipts


def _routing_input(state, eligible, receipts):
    descriptions = {
        name: json.dumps(
            {
                "model": candidate.model,
                "criteria": candidate.description,
                "model_options": candidate.model_options,
                "input_price_per_million": candidate.input_price_per_million,
                "output_price_per_million": candidate.output_price_per_million,
                "compact_context": candidate.compact_context,
                "capacity": receipts[name],
            },
            sort_keys=True,
        )
        for name, (candidate, _) in eligible.items()
    }
    # Feed execution evidence through the already safe observation view, without hidden reasoning.
    smallest = min(eligible.values(), key=lambda item: len(json.dumps(item[1])))
    task = {
        "history": _visible_messages(smallest[1]),
        "previous_configuration": state.get("configuration_route"),
        "excluded_configurations": state.get("configuration_excluded", []),
        "policy": "Choose the least expensive capable configuration using criteria and execution history. Failures require diagnosis; preserve completed work. Switching can lose prompt-cache reuse; no cache discount is assumed.",
    }
    return redact(task), descriptions


def _saved_decision(tools, boundary):
    events = [
        event
        for event in tools.store._events(tools.task_id)
        if event.get("type") == "configuration_routing"
        and event.get("boundary_id") == boundary
    ]
    if not events:
        return None
    latest = events[-1]
    if latest["status"] == "started":
        raise UncertainOperation(
            "Configuration routing result was lost; do not repeat the paid request"
        )
    return latest


def _classify_configuration(state, profile, tools, eligible, receipts):
    endpoint = profile.configuration_routing.classifier
    task, descriptions = _routing_input(state, eligible, receipts)
    request = _messages(task, descriptions)
    capacity = _admission(endpoint, request, response_format=_format(descriptions))
    if not capacity["admitted"]:
        raise ValueError(
            "configuration classifier context capacity exhausted; no inference attempted"
        )
    boundary = hashlib.sha256(
        json.dumps([task, descriptions], sort_keys=True).encode()
    ).hexdigest()
    saved = _saved_decision(tools, boundary)
    if saved is not None:
        return saved
    key = os.environ.get(endpoint.key_env, "") or load_credentials().tokens.get(
        endpoint.key_env, ""
    )
    if not key:
        raise ValueError("configuration classifier credential is unavailable")
    intent = {
        "type": "configuration_routing",
        "boundary_id": boundary,
        "model": endpoint.model,
        "status": "started",
        "api_call_attempted": "unknown",
        "usage": {},
        "capacity": capacity,
        "eligible": receipts,
    }
    tools.store._event(tools.task_id, intent)
    decision = _classify(endpoint, task, descriptions, api_key=key)
    receipt = {
        **intent,
        **decision,
        "api_call_attempted": True,
        "selected_configuration": decision.get("selected_model"),
        "policy_revision": "npa.agent-configuration-router.v1",
    }
    tools.store._event(tools.task_id, receipt)
    return receipt


def _select_configuration(state, profile, tools):
    eligible, receipts = _candidates(state, profile, tools.schemas())
    current = state.get("configuration_route")
    if current in eligible:
        name, reason = current, "retain_feasible_configuration"
    elif len(eligible) == 1:
        name, reason = next(iter(eligible)), "only_feasible_configuration"
    elif eligible:
        decision = _classify_configuration(state, profile, tools, eligible, receipts)
        name, reason = decision.get("selected_configuration"), "history_classifier"
        if decision["status"] != "accepted" or name not in eligible:
            raise ValueError(
                "configuration classifier did not accept a feasible configuration; generation blocked"
            )
    else:
        tools.store._event(
            tools.task_id,
            {"type": "configuration_capacity_blocked", "eligible": receipts},
        )
        raise ValueError("no feasible configuration remains; generation blocked")
    candidate, view = eligible[name]
    tools.store._event(
        tools.task_id,
        {
            "type": "configuration_request",
            "configuration": name,
            "model": candidate.model,
            "reason": reason,
            "compact_context": candidate.compact_context,
            "capacity": receipts[name],
            "eligible": receipts,
        },
    )
    return candidate, view
