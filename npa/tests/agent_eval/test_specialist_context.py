"""Keep usable source and failures while removing only superseded request content."""

import copy
import json
from types import SimpleNamespace

from npa.agent_backend.specialists.context import model_messages


def _observation(messages, name, **result):
    identity = str(len(messages))
    messages.extend(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": identity, "function": {"name": name, "arguments": "{}"}}
                ],
            },
            {
                "role": "tool",
                "tool_call_id": identity,
                "content": json.dumps({"ok": True, **result}),
            },
        ]
    )


def _read(messages, text, digest="old", start=1, end=20):
    _observation(
        messages,
        "read_file",
        path="source.py",
        sha256=digest,
        content=text,
        start_line=start,
        end_line=end,
    )


def test_edit_invalidates_old_source_without_mutating_durable_history():
    messages = [
        {"role": "system", "content": "Policy"},
        {"role": "user", "content": "Goal"},
    ]
    _read(messages, "old source " * 1000)
    _observation(messages, "edit_file", path="source.py", sha256="new")
    _read(messages, "current source", digest="new")
    original = copy.deepcopy(messages)
    result = model_messages(messages)
    assert messages == original
    assert "content" not in json.loads(result[3]["content"])
    assert json.loads(result[3]["content"])["sha256"] == "old"
    assert result[-1] == messages[-1]
    assert len(json.dumps(result)) < len(json.dumps(messages)) / 5
    assert [message.get("tool_call_id") for message in result] == [
        message.get("tool_call_id") for message in messages
    ]


def test_disjoint_and_partially_overlapping_current_ranges_are_preserved():
    messages = []
    _read(messages, "header", start=1, end=10)
    _read(messages, "function", start=50, end=80)
    _read(messages, "overlap", start=75, end=90)
    assert model_messages(messages) == messages
    _read(messages, "whole current source", start=1, end=100)
    result = model_messages(messages)
    assert all(
        "content" not in json.loads(result[index]["content"]) for index in (1, 3, 5)
    )
    assert result[-1] == messages[-1]


def test_failed_edit_and_read_do_not_invalidate_usable_source():
    messages = []
    _read(messages, "still current")
    _observation(
        messages, "edit_file", ok=False, path="source.py", error="hash mismatch"
    )
    _observation(messages, "read_file", ok=False, path="source.py", error="missing")
    assert model_messages(messages) == messages


def test_only_superseded_successful_operation_output_is_omitted():
    messages = []
    _observation(
        messages,
        "run_operation",
        operation="check",
        returncode=0,
        stdout="old pass",
        stderr="",
    )
    _observation(
        messages,
        "run_operation",
        ok=False,
        operation="check",
        returncode=1,
        stdout="failure detail",
        stderr="error",
    )
    _observation(
        messages,
        "run_operation",
        operation="check",
        returncode=0,
        stdout="latest pass",
        stderr="warning",
    )
    result = model_messages(messages, observation_operations={"check"})
    assert "stdout" not in json.loads(result[1]["content"])
    assert result[3:] == messages[3:]


def test_distinct_successful_submissions_keep_each_job_receipt():
    messages = []
    for job in ("first-job", "second-job"):
        _observation(
            messages, "run_operation", operation="submit", returncode=0, stdout=job
        )
    assert model_messages(messages, observation_operations={"status"}) == messages


def test_unknown_tool_or_unparseable_observation_is_not_reinterpreted():
    messages = []
    _observation(messages, "unknown", path="source.py", sha256="old", content="keep")
    _read(messages, "new", digest="new")
    messages.append({"role": "tool", "tool_call_id": "unknown", "content": "not json"})
    assert model_messages(messages) == messages


def test_graph_sends_compact_view_and_records_bytes_without_altering_history(tmp_path):
    from npa.agent_backend.specialists.config import Profile
    from npa.agent_backend.specialists.graph import _model

    profile = Profile(
        name="repair",
        description="Repair",
        model="synthetic",
        workspace=tmp_path,
        compact_context=True,
    )
    messages = []
    _read(messages, "stale source " * 1000)
    _observation(messages, "edit_file", path="source.py", sha256="new")
    original, sent, events = copy.deepcopy(messages), [], []

    def completion(**request):
        sent.append(request)
        return {
            "model": "synthetic",
            "choices": [{"finish_reason": "stop", "message": {"content": "Done"}}],
        }

    tools = SimpleNamespace(
        profile=profile,
        task_id="repair",
        schemas=lambda: [],
        store=SimpleNamespace(_event=lambda task, event: events.append(event)),
    )
    _model(
        {"messages": messages},
        profile,
        tools,
        SimpleNamespace(chat_completion=completion),
    )
    assert messages == original
    assert "content" not in json.loads(sent[0]["messages"][1]["content"])
    assert events[0]["type"] == "context_view"
    assert events[0]["request_message_bytes"] < events[0]["full_message_bytes"]
    assert "stale source" not in json.dumps(events)
