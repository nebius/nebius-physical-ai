"""Execute selected configurations and hand off failures without replaying tool effects."""

from __future__ import annotations

from npa.agent_backend.trajectory import redact

from .capacity import _visible_messages
from .configuration_router import _select_configuration
from .inference_journal import _generation_intent
from .host_verification import _completion_checks
from .progress import _failed_edit_cycle, _failed_verification_cycle
from .provider import _request_model, _metrics
from .recovery import _FailedOperation, _RejectedGeneration, _recovery_instruction


def _configuration_handoff(state, profile, tools, error):
    current = state.get("configuration_route")
    if not current:
        raise error
    excluded = [*state.get("configuration_excluded", []), current]
    if not any(
        item.name not in excluded for item in profile.configuration_routing.candidates
    ):
        if isinstance(error, _FailedOperation):
            return _self_recovery(state, tools)
        raise error
    tools.store._event(
        tools.task_id,
        {
            "type": "configuration_handoff",
            "from_configuration": current,
            "reason": str(error),
            "excluded_configurations": excluded,
        },
    )
    return {
        "messages": [
            *_visible_messages(state["messages"]),
            _recovery_instruction(error),
        ],
        "pending": [],
        "answer": "",
        "recovering": True,
        "operation_failure": "",
        "configuration_excluded": excluded,
        "configuration_route": current,
    }


def _self_recovery(state, tools):
    tools.store._event(
        tools.task_id,
        {
            "type": "configuration_self_recovery",
            "configuration": state["configuration_route"],
        },
    )
    return {
        "messages": [
            *state["messages"],
            {
                "role": "user",
                "content": "Runtime verification failed. No additional configuration remains. Diagnose the recorded failure and repair its inputs before checking again. Continue within the same grants; the failed operation was not replayed.",
            },
        ],
        "pending": [],
        "answer": "",
        "recovering": True,
        "operation_failure": "",
    }


def _active(tools, profile):
    task = tools.store._get(tools.task_id)
    return (
        task["status"] == "running"
        and not task["paused"]
        and profile.name not in tools.store._paused_profiles()
    )


def _configuration_model(state, profile, tools, client):
    from .graph import _client, _recorded_response

    if _failed_verification_cycle(state["messages"]):
        return _configuration_handoff(
            state,
            profile,
            tools,
            _RejectedGeneration(
                "deterministic verification repeated the same failed input hashes; inspect and change the inputs before rerunning"
            ),
        )
    if state.get("operation_failure"):
        return _configuration_handoff(
            state,
            profile,
            tools,
            _FailedOperation("Operation failed: " + state["operation_failure"]),
        )
    if _failed_edit_cycle(state["messages"]):
        return _configuration_handoff(
            state,
            profile,
            tools,
            _RejectedGeneration(
                "identical failed edit repeated without a successful edit; inspect the recorded tool error and current source"
            ),
        )
    candidate, messages = _select_configuration(state, profile, tools)
    active = {**state, "configuration_route": candidate.name}
    if not _active(tools, profile):
        return {"configuration_route": candidate.name, "recovering": True}
    extra = {
        **candidate.model_options,
        "tools": tools.schemas(),
        "parallel_tool_calls": False,
    }
    request_client = client or _client(candidate)
    attempt_id = _generation_intent(candidate, tools, messages, extra)
    try:
        response = _request_model(
            request_client,
            candidate,
            tools,
            attempt_id=attempt_id,
            messages=messages,
            extra=extra,
        )
        message, pending = _recorded_response(
            response,
            candidate.model,
            tools,
            state,
            attempt_id=attempt_id,
            request_metrics=_metrics(request_client, None),
            defer_verification=profile.verify_on_completion,
        )
    except _RejectedGeneration as error:
        return _configuration_handoff(active, profile, tools, error)
    messages, pending = _completion_checks(
        profile, tools, attempt_id, [*state["messages"], message], pending
    )
    return {
        "messages": messages,
        "pending": pending,
        "answer": "" if pending else redact(message["content"]),
        "recovering": False,
        "configuration_route": candidate.name,
    }
