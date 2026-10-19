"""Escalate exact failed-edit cycles, preserving opportunities to use new evidence."""

import json

import pytest

from npa.agent_backend.specialists.progress import (
    _failed_edit_cycle,
    _failed_verification_cycle,
)


def _edit(identifier, *, old="missing", ok=False, tool="edit_file"):
    return [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": identifier,
                    "function": {
                        "name": tool,
                        "arguments": json.dumps(
                            {
                                "path": "config.json",
                                "expected_sha256": "same",
                                "old": old,
                                "new": "correct",
                            }
                        ),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": identifier,
            "content": json.dumps(
                {"ok": ok, "error": "old text must match exactly once"}
                if not ok
                else {"ok": True}
            ),
        },
    ]


def test_only_explicit_deterministic_failed_input_cycles_are_stopped():
    def receipt(digest, **extra):
        return {
            "role": "tool",
            "content": json.dumps(
                {
                    "ok": False,
                    "operation": "verify",
                    "deterministic_inputs": {"config.json": digest},
                    "inputs_unchanged": True,
                    **extra,
                }
            ),
        }

    assert _failed_verification_cycle([receipt("same"), receipt("same")])
    assert not _failed_verification_cycle([receipt("one"), receipt("two")])
    assert not _failed_verification_cycle(
        [receipt("same"), receipt("same", inputs_unchanged=False)]
    )
    assert not _failed_verification_cycle(
        [{"role": "tool", "content": '{"ok":false,"operation":"verify"}'}] * 2
    )


def test_identical_failed_edit_after_read_only_observation_is_a_cycle():
    messages = [
        {"role": "user", "content": "Repair"},
        *_edit("one"),
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "read", "function": {"name": "read_file", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "read", "content": '{"ok":true}'},
        *_edit("two"),
    ]
    assert _failed_edit_cycle(messages)


@pytest.mark.parametrize(
    "between",
    [_edit("changed", ok=True), [{"role": "user", "content": "Recovery boundary"}]],
)
def test_successful_edit_or_new_user_boundary_resets_cycle(between):
    assert not _failed_edit_cycle([*_edit("one"), *between, *_edit("two")])


def test_different_mutation_and_first_failure_are_not_cycles():
    assert not _failed_edit_cycle(_edit("one"))
    assert not _failed_edit_cycle([*_edit("one"), *_edit("two", old="different")])


def test_json_assignment_cycle_and_successful_assignment_boundary():
    assert _failed_edit_cycle([*_edit("one", tool="replace_json"), *_edit("two", tool="replace_json")])
    assert not _failed_edit_cycle([*_edit("one"), *_edit("assignment", tool="replace_json", ok=True), *_edit("two")])


def test_argument_formatting_cannot_disguise_an_identical_failed_mutation():
    messages = [*_edit("one"), *_edit("two")]
    function = messages[-2]["tool_calls"][0]["function"]
    arguments = json.loads(function["arguments"])
    function["arguments"] = json.dumps(dict(reversed(list(arguments.items()))), indent=2)
    assert _failed_edit_cycle(messages)


def test_success_and_non_mutation_failures_are_not_cycles():
    assert not _failed_edit_cycle([*_edit("one"), *_edit("two", ok=True)])
    assert not _failed_edit_cycle(
        [{"role": "tool", "tool_call_id": "read", "content": '{"ok":false}'}]
    )
