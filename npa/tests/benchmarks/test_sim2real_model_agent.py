from __future__ import annotations

import http.client
import json
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from npa.benchmarks.sim2real_model_agent import (
    CHECKPOINT_MARKER,
    CHECKPOINT_SUBMIT_ATTEMPT_MARKER,
    EFFECTIVE_CONTEXT_CHECKPOINT_PROMPT_TOKENS,
    EmptyStreamError,
    IndeterminateToolExecutionError,
    RequestTelemetry,
    StreamRecoveryError,
    _TRANSIENT_TRANSPORT_ERRORS,
    _bounded_tool_result,
    _context_checkpoint,
    _inside,
    _is_permanent_model_http_error,
    _last_request_index,
    _load_malformation_streak,
    _load_transcript,
    _malformation_recovery_action,
    _malformation_telemetry_record,
    _maybe_checkpoint,
    _next_malformation_streak,
    _run_tool,
    _sha,
    _request_active_token_estimate,
    _serialize_bounded_tool_result,
    _stream_chat,
    _submitted_workflow_state,
    _terminal_malformation_failure,
    _transport_telemetry_record,
    _validated_tool_calls,
    _workflow_submission_block_reason,
    _workflow_submit_command_kind,
    _workspace_preflight,
    _write_recovery_checkpoint,
)
from npa.benchmarks.sim2real_model_server import render_server_resources
from npa.benchmarks.sim2real_success import VerificationError, _lift_evidence


def _manifest(*, duration: float = 2.0, timestamped: bool = True) -> dict:
    rows = []
    for index, when in enumerate((0.0, 1.0, duration)):
        row = {
            "step": index,
            "sim_step": index,
            "simulator_ground_truth": {
                "stable_grasp": True,
                "object_lift_m": 0.051,
            },
        }
        if timestamped:
            row["sim_time_seconds"] = when
        rows.append(row)
    return {
        "schema": "npa.sim2real.action_rollout.v1",
        "source": "byo_isaac_policy_rollout",
        "sim_backend": "isaac",
        "policy_trained": True,
        "policy_checkpoint_sha256": "a" * 64,
        "policy_checkpoint_size_bytes": 10,
        "rollout_id": "rollout-1",
        "simulation_step_seconds": 1.0,
        "actions": rows,
    }


def test_lift_evidence_requires_physical_two_second_dwell(tmp_path: Path) -> None:
    found = _lift_evidence(
        tmp_path / "manifest.json",
        _manifest(),
        minimum_lift_m=0.05,
        minimum_hold_seconds=2.0,
    )
    assert found is not None
    assert found.duration_seconds == 2.0
    assert found.minimum_lift_m == pytest.approx(0.051)


def test_lift_evidence_rejects_short_or_broken_hold(tmp_path: Path) -> None:
    assert (
        _lift_evidence(
            tmp_path / "manifest.json",
            _manifest(duration=1.99),
            minimum_lift_m=0.05,
            minimum_hold_seconds=2.0,
        )
        is None
    )
    broken = _manifest()
    broken["actions"][1]["simulator_ground_truth"]["stable_grasp"] = False
    assert (
        _lift_evidence(
            tmp_path / "manifest.json",
            broken,
            minimum_lift_m=0.05,
            minimum_hold_seconds=2.0,
        )
        is None
    )


def test_lift_evidence_refuses_sample_count_as_time(tmp_path: Path) -> None:
    manifest = _manifest(timestamped=False)
    manifest.pop("simulation_step_seconds")
    with pytest.raises(VerificationError, match="simulation_step_seconds"):
        _lift_evidence(
            tmp_path / "manifest.json",
            manifest,
            minimum_lift_m=0.05,
            minimum_hold_seconds=2.0,
        )


def test_lift_evidence_rejects_gaps_in_temporal_coverage(tmp_path: Path) -> None:
    manifest = _manifest()
    manifest["actions"][-1]["sim_step"] = 3
    assert (
        _lift_evidence(
            tmp_path / "manifest.json",
            manifest,
            minimum_lift_m=0.05,
            minimum_hold_seconds=2.0,
        )
        is None
    )


def test_file_tools_cannot_escape_workspace(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="escapes"):
        _inside(tmp_path, "../outside")
    result = _run_tool(
        "write_file", {"path": "notes/result.txt", "content": "ok"}, tmp_path, {}
    )
    assert result["bytes"] == 2
    assert (tmp_path / "notes/result.txt").read_text() == "ok"


def test_tool_schema_round_trips_json_arguments(tmp_path: Path) -> None:
    path = tmp_path / "data.json"
    path.write_text(json.dumps({"ok": True}))
    result = _run_tool("read_file", {"path": "data.json"}, tmp_path, {})
    assert json.loads(result["content"]) == {"ok": True}


def test_large_tool_result_is_hash_bound_and_context_bounded() -> None:
    result = {"path": "large.txt", "content": "abcdef" * 2_000}

    bounded = _bounded_tool_result(result, max_characters=4_096)

    assert bounded["_npa_context_truncated"] is True
    assert bounded["full_result_characters"] > 10_000
    assert len(bounded["full_result_sha256"]) == 64
    assert "abcdef" in bounded["preview_head"]
    assert "targeted excerpt" in bounded["recovery_guidance"]
    assert len(_serialize_bounded_tool_result(result)) <= 4_096
    assert _bounded_tool_result({"ok": True}) == {"ok": True}


def test_bounded_submit_result_preserves_success_and_run_identifier() -> None:
    run_id = "submitted-large-result-1"
    assistant = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "submit",
                "type": "function",
                "function": {
                    "name": "run_command",
                    "arguments": json.dumps(
                        {
                            "command": "npa/.venv/bin/npa workbench workflow submit spec.yaml"
                        }
                    ),
                },
            }
        ],
    }
    content = _serialize_bounded_tool_result(
        {
            "exit_code": 0,
            "stdout": json.dumps({"run_id": run_id, "details": "x" * 12_000}),
        }
    )
    tool = {"role": "tool", "tool_call_id": "submit", "content": content}

    assert len(content) <= 4_096
    assert json.loads(content)["exit_code"] == 0
    assert _submitted_workflow_state([assistant, tool]) == (True, [run_id])


def test_complete_workflow_uses_authoritative_terminal_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, json.dumps({"status": "SUCCEEDED"}), ""
        ),
    )
    result = _run_tool(
        "complete_workflow",
        {"run_id": "sim2real-run-1"},
        tmp_path,
        {"NPA_PROJECT": "private-alias"},
    )
    assert result["terminal"] is True
    assert result["workflow_succeeded"] is True
    assert result["status"] == "SUCCEEDED"


def test_complete_workflow_rejects_untrusted_run_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="invalid format"):
        _run_tool(
            "complete_workflow",
            {"run_id": "run; touch escaped"},
            tmp_path,
            {"NPA_PROJECT": "private-alias"},
        )


def test_empty_stream_is_a_retryable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class EmptyResponse:
        def __enter__(self) -> EmptyResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def __iter__(self):
            return iter(())

    monkeypatch.setattr(
        urllib.request, "urlopen", lambda _request, **_kwargs: EmptyResponse()
    )
    moments = iter((10.0, 15.0))
    monkeypatch.setattr(
        "npa.benchmarks.sim2real_model_agent.time.monotonic", lambda: next(moments)
    )
    with pytest.raises(EmptyStreamError, match="empty event stream") as raised:
        _stream_chat("http://model.example/v1", "key", {"messages": []})
    assert raised.value.telemetry["elapsed_seconds"] == 5.0
    assert raised.value.telemetry["reason"] == "empty_event_stream"


class _StreamingResponse:
    def __init__(self, chunks: list[dict | str]) -> None:
        self.chunks = chunks
        self.closed = False

    def __enter__(self) -> _StreamingResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        self.closed = True

    def __iter__(self):
        for chunk in self.chunks:
            data = chunk if isinstance(chunk, str) else json.dumps(chunk)
            yield f"data: {data}\n".encode()


def test_valid_fragmented_glm_tool_call_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _StreamingResponse(
        [
            {
                "choices": [
                    {
                        "delta": {
                            "reasoning_content": "I should inspect first.",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "run_",
                                        "arguments": '{"command":"git ',
                                    },
                                }
                            ],
                        },
                        "finish_reason": None,
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {
                                        "name": "command",
                                        "arguments": 'status"}',
                                    },
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 6,
                    "total_tokens": 16,
                },
            },
            "[DONE]",
        ]
    )
    monkeypatch.setattr(urllib.request, "urlopen", lambda *_a, **_k: response)

    assistant, telemetry = _stream_chat(
        "http://model.example/v1", "key", {"messages": []}
    )
    calls = _validated_tool_calls(
        assistant, finish_reason=telemetry.finish_reason
    )

    assert calls[0][1] == {"command": "git status"}
    assert assistant["tool_calls"][0]["id"] == "call_1"
    assert telemetry.finish_reason == "tool_calls"
    assert telemetry.completion_tokens == 6
    assert telemetry.observed_tokens_lower_bound == 2
    assert response.closed is True


def test_reasoning_only_runaway_is_interrupted_with_progress_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _StreamingResponse(
        [
            {
                "choices": [
                    {
                        "delta": {"reasoning_content": "abcdef"},
                        "finish_reason": None,
                    }
                ]
            }
        ]
    )
    monkeypatch.setattr(urllib.request, "urlopen", lambda *_a, **_k: response)

    with pytest.raises(StreamRecoveryError) as raised:
        _stream_chat(
            "http://model.example/v1",
            "key",
            {"messages": []},
            safeguards={"no_tool_progress_characters": 6},
        )

    assert raised.value.reason == "no_usable_tool_call_progress"
    assert raised.value.telemetry["observed_characters_lower_bound"] == 6
    assert raised.value.telemetry["observed_tokens_lower_bound"] == 1
    assert raised.value.telemetry["elapsed_seconds"] >= 0
    assert response.closed is True


def test_usage_only_heartbeat_cannot_bypass_semantic_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _StreamingResponse([{"choices": [], "usage": {"total_tokens": 1}}])
    monkeypatch.setattr(urllib.request, "urlopen", lambda *_a, **_k: response)
    moments = iter((0.0, 2.0, 2.0))
    monkeypatch.setattr(
        "npa.benchmarks.sim2real_model_agent.time.monotonic", lambda: next(moments)
    )

    with pytest.raises(StreamRecoveryError) as raised:
        _stream_chat(
            "http://model.example/v1",
            "key",
            {"messages": []},
            safeguards={"no_tool_progress_seconds": 1},
        )

    assert raised.value.reason == "no_usable_tool_call_progress"
    assert raised.value.telemetry["observed_tokens_lower_bound"] == 0


def test_midstream_disconnect_preserves_observed_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DisconnectingResponse(_StreamingResponse):
        def __iter__(self):
            yield (
                b'data: {"choices":[{"delta":{"reasoning_content":"abc"},'
                b'"finish_reason":null}]}\n'
            )
            raise http.client.RemoteDisconnected("peer closed")

    response = DisconnectingResponse([])
    monkeypatch.setattr(urllib.request, "urlopen", lambda *_a, **_k: response)

    with pytest.raises(StreamRecoveryError) as raised:
        _stream_chat("http://model.example/v1", "key", {"messages": []})

    assert raised.value.reason == "stream_transport_interrupted"
    assert raised.value.telemetry["observed_tokens_lower_bound"] == 1
    assert raised.value.telemetry["observed_characters_lower_bound"] == 3
    assert response.closed is True


def test_non_object_sse_chunk_is_malformed_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _StreamingResponse(["[]"])
    monkeypatch.setattr(urllib.request, "urlopen", lambda *_a, **_k: response)

    with pytest.raises(StreamRecoveryError) as raised:
        _stream_chat("http://model.example/v1", "key", {"messages": []})

    assert raised.value.reason == "malformed_stream_shape"


@pytest.mark.parametrize(
    "chunk",
    [
        {
            "choices": [
                {
                    "delta": {"tool_calls": [{"index": 0, "function": "bad"}]},
                    "finish_reason": None,
                }
            ]
        },
        {
            "choices": [],
            "usage": {"prompt_tokens_details": "bad"},
        },
    ],
)
def test_nested_non_object_sse_shape_is_malformed_recovery(
    monkeypatch: pytest.MonkeyPatch, chunk: dict
) -> None:
    response = _StreamingResponse([chunk])
    monkeypatch.setattr(urllib.request, "urlopen", lambda *_a, **_k: response)

    with pytest.raises(StreamRecoveryError) as raised:
        _stream_chat("http://model.example/v1", "key", {"messages": []})

    assert raised.value.reason == "malformed_stream_shape"


def test_unvalidated_or_partial_tool_arguments_are_rejected() -> None:
    assistant = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "run_command", "arguments": '{"command":'},
            }
        ],
    }
    with pytest.raises(ValueError, match="incomplete or invalid JSON"):
        _validated_tool_calls(assistant, finish_reason="tool_calls")


def test_resume_loads_transcript_and_request_index(tmp_path: Path) -> None:
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(
        "\n".join(
            json.dumps(item)
            for item in (
                {"role": "assistant", "content": None, "tool_calls": []},
                {"role": "user", "content": "continue"},
            )
        )
        + "\n"
    )
    requests = tmp_path / "requests.jsonl"
    requests.write_text(
        json.dumps({"request_index": 3})
        + "\n"
        + json.dumps({"classification": "telemetry_correction"})
        + "\n"
        + json.dumps({"request_index": 8, "transport_error": "empty"})
        + "\n"
    )

    assert [item["role"] for item in _load_transcript(transcript)] == [
        "assistant",
        "user",
    ]
    assert _last_request_index(requests) == 8


def test_context_checkpoint_is_bounded_and_becomes_resume_boundary(
    tmp_path: Path,
) -> None:
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "old-" + "x" * 100},
        {"role": "user", "content": "recent-1"},
        {"role": "assistant", "content": "recent-2"},
    ]
    compacted, checkpoint = _context_checkpoint(messages, max_recent_chars=80)
    assert compacted[:2] == messages[:2]
    assert compacted[2] == checkpoint
    assert checkpoint["content"].startswith(CHECKPOINT_MARKER)
    assert "recent-2" in checkpoint["content"]
    assert "old-" not in checkpoint["content"]

    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(
        json.dumps(messages[2])
        + "\n"
        + json.dumps(checkpoint)
        + "\n"
        + json.dumps({"role": "assistant", "content": "after"})
        + "\n"
    )
    loaded = _load_transcript(transcript)
    assert loaded == [checkpoint, {"role": "assistant", "content": "after"}]


def test_effective_context_checkpoint_precedes_observed_sparse_prefill_oom(
    tmp_path: Path,
) -> None:
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        *(
            {"role": "assistant", "content": f"group-{index}-" + "x" * 4_000}
            for index in range(20)
        ),
    ]
    transcript = tmp_path / "transcript.jsonl"

    compacted = _maybe_checkpoint(
        messages,
        transcript,
        active_tokens_upper_bound=EFFECTIVE_CONTEXT_CHECKPOINT_PROMPT_TOKENS,
        context_limit=262_144,
        workspace_status="",
    )

    assert len(compacted) < len(messages)
    assert compacted[2]["content"].startswith(CHECKPOINT_MARKER)
    assert len(compacted[2]["content"]) < 20_000
    assert transcript.read_text().count(CHECKPOINT_MARKER) == 1

    unchanged = _maybe_checkpoint(
        messages,
        transcript,
        active_tokens_upper_bound=EFFECTIVE_CONTEXT_CHECKPOINT_PROMPT_TOKENS - 1,
        context_limit=262_144,
    )
    assert unchanged == messages

    repeated = _maybe_checkpoint(
        compacted,
        transcript,
        active_tokens_upper_bound=EFFECTIVE_CONTEXT_CHECKPOINT_PROMPT_TOKENS,
        context_limit=262_144,
    )
    assert repeated == compacted
    assert transcript.read_text().count(CHECKPOINT_MARKER) == 1

    advertised_transcript = tmp_path / "advertised.jsonl"
    advertised = _maybe_checkpoint(
        messages,
        advertised_transcript,
        active_tokens_upper_bound=8_500,
        context_limit=10_000,
    )
    assert advertised[2]["content"].startswith(CHECKPOINT_MARKER)

    crash_resume_transcript = tmp_path / "crash-resume.jsonl"
    crash_resume = _maybe_checkpoint(
        messages,
        crash_resume_transcript,
        context_limit=262_144,
    )
    assert crash_resume[2]["content"].startswith(CHECKPOINT_MARKER)


def test_startup_rebuilds_oversized_legacy_checkpoint_from_full_transcript(
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "transcript.jsonl"
    diagnostic = {"role": "assistant", "content": "safe diagnostic result"}
    oversized = {
        "role": "user",
        "content": CHECKPOINT_MARKER + "\n" + "x" * 153_520,
    }
    transcript.write_text(
        json.dumps(diagnostic) + "\n" + json.dumps(oversized) + "\n"
    )
    resumed = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        *_load_transcript(transcript),
    ]

    rebuilt = _maybe_checkpoint(
        resumed,
        transcript,
        context_limit=262_144,
        active_tokens_upper_bound=1,
    )

    assert len(rebuilt) == 3
    assert len(rebuilt[2]["content"]) < 20_000
    assert "safe diagnostic result" in rebuilt[2]["content"]
    assert transcript.read_text().count(CHECKPOINT_MARKER) == 2


def test_checkpoint_bounds_workspace_status_excerpt() -> None:
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "recent"},
    ]

    _, checkpoint = _context_checkpoint(
        messages,
        max_recent_chars=16_384,
        workspace_status=" M " + "x" * 100_000,
    )

    assert len(checkpoint["content"]) < 20_000
    assert "status lines: 1" in checkpoint["content"]
    assert "Workspace status SHA256:" in checkpoint["content"]


def test_checkpoint_bounds_many_historical_run_identifiers() -> None:
    history: list[dict] = []
    for index in range(1_200):
        call_id = f"call-{index}"
        history.extend(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": "run_command",
                                "arguments": json.dumps({"command": "true"}),
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": json.dumps(
                        {"run_id": f"run-{index:04d}-" + "x" * 110}
                    ),
                },
            ]
        )

    _, checkpoint = _context_checkpoint(
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
            *history,
        ],
        max_recent_chars=1,
    )

    assert len(checkpoint["content"]) < 20_000
    ids_line = next(
        line
        for line in checkpoint["content"].splitlines()
        if line.startswith("Durable workflow run identifiers: ")
    )
    assert len(json.loads(ids_line.split(": ", 1)[1])) == 16
    assert '"total_count":1200' in checkpoint["content"]


def test_repeated_checkpoint_preserves_full_run_identifier_summary(
    tmp_path: Path,
) -> None:
    def group(index: int) -> list[dict]:
        call_id = f"call-{index}"
        return [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "run_command",
                            "arguments": json.dumps({"command": "true"}),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": json.dumps({"run_id": f"run-{index:03d}"}),
            },
        ]

    transcript = tmp_path / "transcript.jsonl"
    history = [message for index in range(30) for message in group(index)]
    transcript.write_text(
        "".join(json.dumps(message) + "\n" for message in history)
    )
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        *history,
    ]
    first = _maybe_checkpoint(
        messages,
        transcript,
        context_limit=262_144,
        active_tokens_upper_bound=EFFECTIVE_CONTEXT_CHECKPOINT_PROMPT_TOKENS,
    )
    new_group = group(30)
    with transcript.open("a", encoding="utf-8") as handle:
        for message in new_group:
            handle.write(json.dumps(message) + "\n")

    second = _maybe_checkpoint(
        [*first, *new_group],
        transcript,
        context_limit=262_144,
        active_tokens_upper_bound=EFFECTIVE_CONTEXT_CHECKPOINT_PROMPT_TOKENS,
    )

    summary_line = next(
        line
        for line in second[2]["content"].splitlines()
        if line.startswith("Durable workflow run identifier summary: ")
    )
    summary = json.loads(summary_line.split(": ", 1)[1])
    all_ids = [f"run-{index:03d}" for index in range(31)]
    assert summary == {"total_count": 31, "all_sha256": _sha(all_ids)}


def test_checkpoint_preserves_durable_successful_submission_state(
    tmp_path: Path,
) -> None:
    assistant = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "submit",
                "type": "function",
                "function": {
                    "name": "run_command",
                    "arguments": json.dumps(
                        {"command": "npa workbench workflow submit spec.yaml"}
                    ),
                },
            }
        ],
    }
    result = {
        "role": "tool",
        "tool_call_id": "submit",
        "content": json.dumps({"exit_code": 0, "run_id": "run-123"}),
    }
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        assistant,
        result,
        {"role": "assistant", "content": "x" * 20_000},
    ]

    compacted, checkpoint = _context_checkpoint(
        messages, max_recent_chars=128, workspace_status=""
    )

    assert assistant not in compacted
    assert '"submitted":true' in checkpoint["content"]
    assert _submitted_workflow_state(compacted[2:]) == (True, ["run-123"])


def test_checkpoint_preserves_latest_failed_standalone_submit_attempt() -> None:
    assistant = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "submit",
                "type": "function",
                "function": {
                    "name": "run_command",
                    "arguments": json.dumps(
                        {"command": "npa workbench workflow submit spec.yaml"}
                    ),
                },
            }
        ],
    }
    result = {
        "role": "tool",
        "tool_call_id": "submit",
        "content": json.dumps(
            {"exit_code": 1, "stderr": "required credential preflight failed"}
        ),
    }
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        assistant,
        result,
        {"role": "assistant", "content": "x" * 20_000},
    ]

    compacted, checkpoint = _context_checkpoint(messages, max_recent_chars=128)

    assert assistant not in compacted
    assert CHECKPOINT_SUBMIT_ATTEMPT_MARKER in checkpoint["content"]
    assert "required credential preflight failed" in checkpoint["content"]
    assert '"exit_code":1' in checkpoint["content"]
    assert '"submitted":false' in checkpoint["content"]


def test_resume_upgrades_old_checkpoint_with_historical_submit_attempt(
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "transcript.jsonl"
    assistant = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "submit",
                "type": "function",
                "function": {
                    "name": "run_command",
                    "arguments": json.dumps(
                        {"command": "npa workbench workflow submit spec.yaml"}
                    ),
                },
            }
        ],
    }
    result = {
        "role": "tool",
        "tool_call_id": "submit",
        "content": json.dumps({"exit_code": 1, "stderr": "preflight failed"}),
    }
    old_checkpoint = {
        "role": "user",
        "content": CHECKPOINT_MARKER + "\nold checkpoint without submit attempt",
    }
    later = {"role": "assistant", "content": "continued diagnosis"}
    transcript.write_text(
        "".join(
            json.dumps(message, sort_keys=True) + "\n"
            for message in (assistant, result, old_checkpoint, later)
        )
    )
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        old_checkpoint,
        later,
    ]

    upgraded = _maybe_checkpoint(
        messages,
        transcript,
        context_limit=262_144,
        active_tokens_upper_bound=1,
    )

    assert len(upgraded) == 3
    assert CHECKPOINT_SUBMIT_ATTEMPT_MARKER in upgraded[2]["content"]
    assert "preflight failed" in upgraded[2]["content"]
    assert transcript.read_text().count(CHECKPOINT_SUBMIT_ATTEMPT_MARKER) == 1


def test_checkpoint_merges_historical_and_active_submit_attempts(
    tmp_path: Path,
) -> None:
    def attempt(index: int) -> tuple[dict, dict]:
        assistant = {
            "role": "assistant",
            "_npa_response_id": f"request-{index}-hash",
            "tool_calls": [
                {
                    "id": f"submit-{index}",
                    "type": "function",
                    "function": {
                        "name": "run_command",
                        "arguments": json.dumps(
                            {
                                "command": (
                                    "npa workbench workflow submit "
                                    f"spec-{index}.yaml"
                                )
                            }
                        ),
                    },
                }
            ],
        }
        result = {
            "role": "tool",
            "tool_call_id": f"submit-{index}",
            "content": json.dumps({"exit_code": index, "stderr": f"failure-{index}"}),
        }
        return assistant, result

    first = attempt(1)
    second = attempt(2)
    legacy_checkpoint = {
        "role": "user",
        "content": CHECKPOINT_MARKER + "\nlegacy checkpoint",
    }
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(
        "".join(
            json.dumps(message, sort_keys=True) + "\n"
            for message in (*first, legacy_checkpoint, *second)
        )
    )
    active = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        legacy_checkpoint,
        *second,
    ]

    upgraded = _maybe_checkpoint(
        active,
        transcript,
        context_limit=262_144,
        active_tokens_upper_bound=1,
    )
    marker_line = next(
        line
        for line in upgraded[2]["content"].splitlines()
        if line.startswith(CHECKPOINT_SUBMIT_ATTEMPT_MARKER)
    )
    preserved = json.loads(marker_line.removeprefix(CHECKPOINT_SUBMIT_ATTEMPT_MARKER))
    assert [item["result"]["exit_code"] for item in preserved] == [1, 2]

    third = attempt(3)
    compacted, checkpoint = _context_checkpoint(
        [*upgraded[:2], upgraded[2], *second, *third], max_recent_chars=1
    )
    marker_line = next(
        line
        for line in checkpoint["content"].splitlines()
        if line.startswith(CHECKPOINT_SUBMIT_ATTEMPT_MARKER)
    )
    preserved = json.loads(marker_line.removeprefix(CHECKPOINT_SUBMIT_ATTEMPT_MARKER))
    assert [item["result"]["exit_code"] for item in preserved] == [2, 3]
    assert len(compacted) == 3


def test_checkpoint_merges_journal_reconstructed_attempt_without_duplicate(
    tmp_path: Path,
) -> None:
    command = "npa workbench workflow submit spec.yaml"
    assistant = {
        "role": "assistant",
        "_npa_response_id": "request-2-hash",
        "tool_calls": [
            {
                "id": "submit-2",
                "type": "function",
                "function": {
                    "name": "run_command",
                    "arguments": json.dumps({"command": command}),
                },
            }
        ],
    }
    result = {
        "role": "tool",
        "tool_call_id": "submit-2",
        "content": json.dumps({"exit_code": 2}),
    }
    historical = {
        "command": "npa workbench workflow submit first.yaml",
        "result": {"exit_code": 1},
        "occurrence_id": "request-1-hash:submit-1",
    }
    checkpoint = {
        "role": "user",
        "content": (
            CHECKPOINT_MARKER
            + "\n"
            + CHECKPOINT_SUBMIT_ATTEMPT_MARKER
            + json.dumps([historical], separators=(",", ":"))
        ),
    }
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(json.dumps(checkpoint) + "\n")
    active = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        checkpoint,
        assistant,
        result,
    ]

    upgraded = _maybe_checkpoint(
        active,
        transcript,
        context_limit=262_144,
        active_tokens_upper_bound=EFFECTIVE_CONTEXT_CHECKPOINT_PROMPT_TOKENS,
    )
    marker_line = next(
        line
        for line in upgraded[2]["content"].splitlines()
        if line.startswith(CHECKPOINT_SUBMIT_ATTEMPT_MARKER)
    )
    preserved = json.loads(marker_line.removeprefix(CHECKPOINT_SUBMIT_ATTEMPT_MARKER))
    occurrence_ids = [item["occurrence_id"] for item in preserved]
    assert len(set(occurrence_ids)) == 2
    assert all(len(item) == 64 for item in occurrence_ids)
    repeated = _maybe_checkpoint(
        upgraded,
        transcript,
        context_limit=262_144,
        active_tokens_upper_bound=EFFECTIVE_CONTEXT_CHECKPOINT_PROMPT_TOKENS,
    )
    assert repeated == upgraded
    assert transcript.read_text().count(CHECKPOINT_SUBMIT_ATTEMPT_MARKER) == 2


def test_submit_attempt_checkpoint_bounds_model_supplied_tool_call_id() -> None:
    huge_call_id = "call-" + "x" * 100_000
    assistant = {
        "role": "assistant",
        "_npa_response_id": "request-1-hash",
        "tool_calls": [
            {
                "id": huge_call_id,
                "type": "function",
                "function": {
                    "name": "run_command",
                    "arguments": json.dumps(
                        {"command": "npa workbench workflow submit spec.yaml"}
                    ),
                },
            }
        ],
    }
    result = {
        "role": "tool",
        "tool_call_id": huge_call_id,
        "content": json.dumps({"exit_code": 1, "stderr": "preflight failed"}),
    }

    _, checkpoint = _context_checkpoint(
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
            assistant,
            result,
        ],
        max_recent_chars=1,
    )

    assert len(checkpoint["content"]) < 10_000
    assert huge_call_id not in checkpoint["content"]
    marker_line = next(
        line
        for line in checkpoint["content"].splitlines()
        if line.startswith(CHECKPOINT_SUBMIT_ATTEMPT_MARKER)
    )
    preserved = json.loads(marker_line.removeprefix(CHECKPOINT_SUBMIT_ATTEMPT_MARKER))
    assert len(preserved[0]["occurrence_id"]) == 64


def test_active_token_estimate_counts_completion_and_character_fallback() -> None:
    common = {
        "request_index": 1,
        "started_at": "2026-08-24T00:00:00Z",
        "latency_seconds": 1.0,
        "time_to_first_token_seconds": 0.1,
        "cached_tokens": None,
        "reasoning_tokens": None,
        "completion_tokens_per_second": None,
        "finish_reason": "tool_calls",
        "observed_tokens_lower_bound": 0,
    }
    with_usage = RequestTelemetry(
        **common,
        prompt_tokens=59_999,
        completion_tokens=10,
        total_tokens=60_009,
        observed_characters_lower_bound=40,
    )
    fallback = RequestTelemetry(
        **common,
        prompt_tokens=59_999,
        completion_tokens=None,
        total_tokens=None,
        observed_characters_lower_bound=40,
    )

    one_tool = [{"role": "tool", "content": "x" * 4_096}]
    two_tools = one_tool + [{"role": "tool", "content": "y" * 4_096}]

    with_usage_estimate = _request_active_token_estimate(with_usage, one_tool)
    fallback_estimate = _request_active_token_estimate(fallback, two_tools)

    assert with_usage_estimate is not None
    assert with_usage_estimate >= 60_009 + 4_096
    assert fallback_estimate is not None
    assert fallback_estimate >= 59_999 + 40 + 8_192


def test_resume_discards_incomplete_assistant_tool_group(tmp_path: Path) -> None:
    transcript = tmp_path / "transcript.jsonl"
    checkpoint = {"role": "user", "content": CHECKPOINT_MARKER + " safe"}
    incomplete = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "not-journaled",
                "type": "function",
                "function": {
                    "name": "run_command",
                    "arguments": json.dumps({"command": "git status --short"}),
                },
            }
        ],
    }
    transcript.write_text(
        "\n".join(json.dumps(message) for message in (checkpoint, incomplete)) + "\n"
    )

    assert _load_transcript(transcript) == [checkpoint]


def test_resume_reconstructs_journaled_tool_group_without_reexecution(
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text("")
    journal = tmp_path / "tool-results.jsonl"
    assistant = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "submitted",
                "type": "function",
                "function": {
                    "name": "run_command",
                    "arguments": json.dumps({"command": "submit-command"}),
                },
            },
            {
                "id": "not-started",
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": json.dumps({"path": "status.json"}),
                },
            },
        ],
    }
    result_message = {
        "role": "tool",
        "tool_call_id": "submitted",
        "content": _serialize_bounded_tool_result(
            {"exit_code": 0, "stdout": json.dumps({"run_id": "durable-run-1"})}
        ),
    }
    response_id = "response-1"
    records = [
        {
            "schema": "npa.sim2real.tool_execution.v2",
            "response_id": response_id,
            "assistant": assistant,
            "tool_call_id": "submitted",
            "phase": "intent",
        },
        {
            "schema": "npa.sim2real.tool_execution.v2",
            "response_id": response_id,
            "assistant": assistant,
            "tool_call_id": "submitted",
            "phase": "result",
            "tool_message": result_message,
        },
    ]
    journal.write_text("\n".join(json.dumps(record) for record in records) + "\n")

    loaded = _load_transcript(transcript, journal)

    assert loaded[0] == assistant
    assert loaded[1] == result_message
    assert json.loads(loaded[2]["content"])["error"] == "ControllerRecovery"


def test_resume_terminates_for_indeterminate_journaled_tool_execution(
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text("")
    journal = tmp_path / "tool-results.jsonl"
    assistant = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "possibly-executed",
                "type": "function",
                "function": {"name": "run_command", "arguments": "{}"},
            }
        ],
    }
    journal.write_text(
        json.dumps(
            {
                "schema": "npa.sim2real.tool_execution.v2",
                "response_id": "response-2",
                "assistant": assistant,
                "tool_call_id": "possibly-executed",
                "phase": "intent",
            }
        )
        + "\n"
    )

    with pytest.raises(IndeterminateToolExecutionError) as error:
        _load_transcript(transcript, journal)
    assert error.value.tool_call_ids == ["possibly-executed"]


def test_resume_does_not_merge_identical_response_occurrences(tmp_path: Path) -> None:
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text("")
    journal = tmp_path / "tool-results.jsonl"
    assistant = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "same-call",
                "type": "function",
                "function": {"name": "run_command", "arguments": "{}"},
            }
        ],
    }
    result_message = {
        "role": "tool",
        "tool_call_id": "same-call",
        "content": _serialize_bounded_tool_result({"exit_code": 0}),
    }
    records = [
        {
            "schema": "npa.sim2real.tool_execution.v2",
            "response_id": "request-1-same-hash",
            "assistant": assistant,
            "tool_call_id": "same-call",
            "phase": "intent",
        },
        {
            "schema": "npa.sim2real.tool_execution.v2",
            "response_id": "request-1-same-hash",
            "assistant": assistant,
            "tool_call_id": "same-call",
            "tool_message": result_message,
            "phase": "result",
        },
        {
            "schema": "npa.sim2real.tool_execution.v2",
            "response_id": "request-1-same-hash",
            "assistant": assistant,
            "phase": "transcript_committed",
        },
        {
            "schema": "npa.sim2real.tool_execution.v2",
            "response_id": "request-2-same-hash",
            "assistant": assistant,
            "tool_call_id": "same-call",
            "phase": "intent",
        },
    ]
    journal.write_text("\n".join(json.dumps(record) for record in records) + "\n")

    with pytest.raises(IndeterminateToolExecutionError) as error:
        _load_transcript(transcript, journal)
    assert error.value.response_id == "request-2-same-hash"


def test_resume_treats_torn_final_journal_record_as_uncommitted_intent(
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text('{"role":"user","content":"safe"}\n{"role":"assistant"')
    journal = tmp_path / "tool-results.jsonl"
    assistant = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "uncertain",
                "type": "function",
                "function": {"name": "run_command", "arguments": "{}"},
            }
        ],
    }
    journal.write_text(
        json.dumps(
            {
                "schema": "npa.sim2real.tool_execution.v2",
                "response_id": "request-3-hash",
                "assistant": assistant,
                "tool_call_id": "uncertain",
                "phase": "intent",
            }
        )
        + '\n{"schema":"npa.sim2real.tool_execution.v2","phase":"result"'
    )

    with pytest.raises(IndeterminateToolExecutionError) as error:
        _load_transcript(transcript, journal)
    assert error.value.tool_call_ids == ["uncertain"]


def test_recovery_checkpoint_keeps_only_complete_tool_boundaries_and_state() -> None:
    complete_assistant = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "complete-call",
                "type": "function",
                "function": {"name": "run_command", "arguments": "{}"},
            }
        ],
    }
    complete_result = {
        "role": "tool",
        "tool_call_id": "complete-call",
        "content": json.dumps({"run_id": "durable-run-1", "status": "RUNNING"}),
    }
    incomplete_assistant = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "partial-call",
                "type": "function",
                "function": {"name": "run_command", "arguments": '{"command":'},
            }
        ],
    }
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "original task"},
        complete_assistant,
        complete_result,
        incomplete_assistant,
    ]

    compacted, checkpoint = _context_checkpoint(
        messages,
        max_recent_chars=10_000,
        workspace_status=" M durable.txt\n",
        recovery_reason="tool_call_boundary_not_completed",
    )

    assert compacted[:2] == messages[:2]
    assert compacted[2] == checkpoint
    assert "durable-run-1" in checkpoint["content"]
    assert "durable.txt" in checkpoint["content"]
    assert "complete-call" in checkpoint["content"]
    assert "partial-call" not in checkpoint["content"]
    assert "incomplete assistant response was not added" in checkpoint["content"]


def test_live_malformed_recovery_uses_fixed_bounded_checkpoint_window(
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "transcript.jsonl"
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        *(
            {"role": "assistant", "content": f"old-{index}-" + "x" * 5_000}
            for index in range(100)
        ),
    ]

    compacted = _write_recovery_checkpoint(
        messages,
        transcript,
        workspace_status="",
        reason="no_usable_tool_call_progress",
    )

    assert len(compacted) == 3
    checkpoint = compacted[2]
    assert len(checkpoint["content"]) < 20_000
    assert "no_usable_tool_call_progress" in checkpoint["content"]
    assert transcript.read_text().count(CHECKPOINT_MARKER) == 1
    assert "old-99" in checkpoint["content"]

    transport_recovery = _write_recovery_checkpoint(
        compacted,
        transcript,
        workspace_status="",
        reason="transport_error:HTTPError",
    )
    repeated_malformed_recovery = _write_recovery_checkpoint(
        transport_recovery,
        transcript,
        workspace_status="",
        reason="no_usable_tool_call_progress",
    )

    assert transport_recovery == compacted
    assert repeated_malformed_recovery == compacted
    assert "old-99" in repeated_malformed_recovery[2]["content"]
    assert transcript.read_text().count(CHECKPOINT_MARKER) == 1


def test_repeated_identical_malformation_has_terminal_classification(
    tmp_path: Path,
) -> None:
    fingerprint: str | None = None
    count = 0
    for expected in (1, 2, 3):
        fingerprint, count = _next_malformation_streak(
            fingerprint, count, "same-fingerprint"
        )
        assert count == expected
    assert (
        _malformation_recovery_action(count, 3)
        == "terminate_repeated_identical_malformed_response"
    )

    telemetry = tmp_path / "requests.jsonl"
    telemetry.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "request_index": index,
                        "classification": "malformed_response",
                        "malformation_fingerprint": "same-fingerprint",
                        "elapsed_seconds": 1.0,
                        "observed_tokens_lower_bound": index,
                        "observed_characters_lower_bound": index * 4,
                        "reason": "no_usable_tool_call_progress",
                        "recovery_action": (
                            "discard_partial_response_rebuild_context_and_retry"
                        ),
                    }
                )
                for index in (1, 2)
            ]
        )
        + "\n"
    )
    assert _load_malformation_streak(telemetry) == ("same-fingerprint", 2)

    record = _malformation_telemetry_record(
        {
            "started_at": "2026-08-23T00:00:00Z",
            "elapsed_seconds": 12.5,
            "observed_tokens_lower_bound": 7,
            "observed_characters_lower_bound": 42,
            "reason": "no_usable_tool_call_progress",
        },
        request_index=3,
        response_shape={"has_reasoning": True},
        fingerprint="same-fingerprint",
        identical_count=3,
        action="terminate_repeated_identical_malformed_response",
    )
    assert {
        "request_index",
        "elapsed_seconds",
        "observed_tokens_lower_bound",
        "observed_characters_lower_bound",
        "reason",
        "recovery_action",
    } <= set(record)

    failure = _terminal_malformation_failure(
        request_index=3,
        fingerprint="same-fingerprint",
        identical_count=3,
        reason="no_usable_tool_call_progress",
        workflow_submitted=False,
        run_identifiers=[],
    )
    assert failure["classification"] == "repeated_identical_malformed_response"
    assert failure["workflow_submitted"] is False


def test_submission_state_excludes_plan_only_run_identifiers() -> None:
    assistant = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "plan",
                "type": "function",
                "function": {
                    "name": "run_command",
                    "arguments": json.dumps(
                        {
                            "command": "npa/.venv/bin/npa workbench workflow submit spec.yaml --plan-only"
                        }
                    ),
                },
            },
            {
                "id": "submit",
                "type": "function",
                "function": {
                    "name": "run_command",
                    "arguments": json.dumps(
                        {
                            "command": "npa/.venv/bin/npa workbench workflow submit spec.yaml"
                        }
                    ),
                },
            },
        ],
    }
    plan = {
        "role": "tool",
        "tool_call_id": "plan",
        "content": json.dumps(
            {"exit_code": 0, "stdout": json.dumps({"run_id": "plan-only-1"})}
        ),
    }
    failed_submit = {
        "role": "tool",
        "tool_call_id": "submit",
        "content": json.dumps(
            {"exit_code": 1, "stdout": json.dumps({"run_id": "not-submitted-1"})}
        ),
    }

    assert _submitted_workflow_state([assistant, plan, failed_submit]) == (False, [])

    succeeded_submit = {
        **failed_submit,
        "content": json.dumps(
            {"exit_code": 0, "stdout": json.dumps({"run_id": "submitted-1"})}
        ),
    }
    assert _submitted_workflow_state([assistant, plan, succeeded_submit]) == (
        True,
        ["submitted-1"],
    )


def test_submission_state_excludes_help_and_classifies_real_submit() -> None:
    help_assistant = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "help",
                "type": "function",
                "function": {
                    "name": "run_command",
                    "arguments": json.dumps(
                        {
                            "command": (
                                "npa/.venv/bin/npa workbench workflow submit --help"
                            )
                        }
                    ),
                },
            }
        ],
    }
    help_result = {
        "role": "tool",
        "tool_call_id": "help",
        "content": json.dumps({"exit_code": 0, "stdout": "Usage: submit"}),
    }

    assert _submitted_workflow_state([help_assistant, help_result]) == (False, [])
    assert (
        _workflow_submit_command_kind(
            "npa/.venv/bin/npa workbench workflow submit --help"
        )
        == "introspection"
    )
    assert (
        _workflow_submit_command_kind(
            "npa/.venv/bin/npa workbench workflow submit spec.yaml --plan-only"
        )
        == "introspection"
    )
    assert (
        _workflow_submit_command_kind(
            "npa/.venv/bin/npa workbench workflow submit spec.yaml"
        )
        == "standalone"
    )
    submitted_assistant = json.loads(json.dumps(help_assistant))
    submitted_assistant["tool_calls"][0]["function"]["arguments"] = json.dumps(
        {"command": "npa/.venv/bin/npa workbench workflow submit spec.yaml"}
    )
    submitted_result = {
        **help_result,
        "content": json.dumps({"exit_code": 0, "stdout": "submitted"}),
    }
    history = [submitted_assistant, submitted_result]
    assert _workflow_submission_block_reason(
        history,
        tool_name="run_command",
        arguments={
            "command": "npa/.venv/bin/npa workbench workflow submit spec.yaml"
        },
    ) == "DuplicateWorkflowSubmissionBlocked"
    assert _workflow_submission_block_reason(
        history,
        tool_name="run_command",
        arguments={"command": "npa/.venv/bin/npa workbench workflow submit --help"},
    ) is None


@pytest.mark.parametrize(
    "command",
    [
        "npa workbench workflow submit a.yaml; npa workbench workflow submit b.yaml",
        "bash -lc 'npa workbench workflow submit a.yaml'",
        "echo workbench workflow submit",
        "true || npa workbench workflow submit a.yaml",
        "npa workbench workflow submit a.yaml && npa workflow status run-1",
        "npa workbench workflow submit a.yaml > submit.log",
    ],
)
def test_compound_or_wrapped_workflow_submit_is_blocked(command: str) -> None:
    assert _workflow_submit_command_kind(command) == "unsafe"
    assert (
        _workflow_submission_block_reason(
            [], tool_name="run_command", arguments={"command": command}
        )
        == "UnsafeWorkflowSubmissionCommandBlocked"
    )


@pytest.mark.parametrize(
    "command",
    [
        "npa 'workbench' 'workflow' 'submit' spec.yaml",
        "npa workbench workflow 'submit' spec.yaml",
        r"npa workbench workflow submi\t spec.yaml",
    ],
)
def test_quoted_or_escaped_standalone_submit_is_still_classified(
    command: str,
) -> None:
    assert _workflow_submit_command_kind(command) == "standalone"


def test_plan_only_pipeline_remains_blocked_shell_composition() -> None:
    command = (
        "npa workbench workflow submit spec.yaml --plan-only 2>&1 | sed -n '1,20p'"
    )
    assert _workflow_submit_command_kind(command) == "unsafe"
    assert (
        _workflow_submission_block_reason(
            [], tool_name="run_command", arguments={"command": command}
        )
        == "UnsafeWorkflowSubmissionCommandBlocked"
    )


def test_remote_disconnect_telemetry_records_safe_recovery_action() -> None:
    disconnect = http.client.RemoteDisconnected("peer closed before response")
    assert isinstance(disconnect, _TRANSIENT_TRANSPORT_ERRORS)

    record = _transport_telemetry_record(
        disconnect,
        request_index=2,
        elapsed_seconds=35.5,
        recovery_action="discard_incomplete_response_rebuild_context_and_retry",
    )

    assert record["classification"] == "transport_error"
    assert record["request_index"] == 2
    assert record["elapsed_seconds"] == 35.5
    assert record["observed_tokens_lower_bound"] == 0
    assert record["observed_characters_lower_bound"] == 0
    assert record["reason"] == "RemoteDisconnected"
    assert record["recovery_action"].startswith("discard_incomplete_response")


def test_only_client_http_errors_are_permanent() -> None:
    client = urllib.error.HTTPError("http://model", 401, "unauthorized", {}, None)
    server = urllib.error.HTTPError("http://model", 503, "unavailable", {}, None)
    throttled = urllib.error.HTTPError("http://model", 429, "throttled", {}, None)

    assert _is_permanent_model_http_error(client) is True
    assert _is_permanent_model_http_error(server) is False
    assert _is_permanent_model_http_error(throttled) is False


def test_workspace_resume_preserves_dirty_detached_checkout(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("initial\n")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Benchmark Test",
            "-c",
            "user.email=benchmark@example.invalid",
            "commit",
            "-qm",
            "initial",
        ],
        cwd=tmp_path,
        check=True,
    )
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True
    ).strip()
    subprocess.run(["git", "checkout", "--detach", "-q"], cwd=tmp_path, check=True)
    tracked.write_text("model change\n")

    with pytest.raises(ValueError, match="and clean"):
        _workspace_preflight(tmp_path, commit, require_clean=True)
    status = _workspace_preflight(tmp_path, commit, require_clean=False)
    assert "tracked.txt" in status


def test_model_server_renderer_pins_image_and_isolates_multinode_endpoint() -> None:
    model = {
        "repository": "org/model",
        "revision": "a" * 40,
        "server": "sglang",
        "server_image": "registry/server@sha256:" + "b" * 64,
        "tool_call_parser": "parser",
        "reasoning_parser": "reasoner",
        "context_limit": 1000,
        "tensor_parallel_size": 16,
        "server_arguments": ["--random-seed=7"],
    }
    resources = render_server_resources(model, namespace="trial-a", service_name="m")
    endpoint, statefulset = resources[2], resources[3]
    assert endpoint["spec"]["selector"] == {"statefulset.kubernetes.io/pod-name": "m-0"}
    assert statefulset["spec"]["replicas"] == 2
    container = statefulset["spec"]["template"]["spec"]["containers"][0]
    assert container["image"].startswith("registry/server@sha256:")
    assert container["resources"]["limits"]["nvidia.com/gpu"] == 8
    assert "ORDINAL=${POD_NAME##*-}" in container["command"][-1]
    assert "--node-rank ${ORDINAL}" in container["command"][-1]
    cache_env = {
        item["name"]: item.get("value")
        for item in container["env"]
        if item["name"].endswith(("CACHE", "HOME"))
    }
    assert cache_env["HF_HUB_CACHE"].startswith("/mnt/data/model-cache/")
    assert cache_env["TRANSFORMERS_CACHE"] == cache_env["HF_HUB_CACHE"]
    network_env = {item["name"]: item.get("value") for item in container["env"]}
    assert network_env["GLOO_SOCKET_IFNAME"] == "eth0"
    assert network_env["NCCL_SOCKET_IFNAME"] == "eth0"
    pod_spec = statefulset["spec"]["template"]["spec"]
    assert pod_spec["hostNetwork"] is True
    assert any(volume["name"] == "infiniband" for volume in pod_spec["volumes"])


def test_model_server_renderer_rejects_unsupported_multinode_vllm() -> None:
    model = {
        "repository": "org/model",
        "revision": "a" * 40,
        "server": "vllm",
        "server_image": "registry/server@sha256:" + "b" * 64,
        "tool_call_parser": "parser",
        "context_limit": 1000,
        "tensor_parallel_size": 16,
    }

    with pytest.raises(ValueError, match="provider-neutral multi-node rendezvous"):
        render_server_resources(model, namespace="trial-a", service_name="m")


def test_glm_catalog_uses_release_with_blackwell_glm_support() -> None:
    npa_root = Path(__file__).resolve().parents[2]
    catalog = json.loads(
        (npa_root / "benchmarks/sim2real-three-model/models.json").read_text()
    )
    arguments = catalog["models"][0]["server_arguments"]
    assert catalog["models"][0]["server_image"].endswith(
        "@sha256:16aba8925507e631e1dc1e23d95d026533602591775f6a8db68b74ee99746155"
    )
    assert "--kv-cache-dtype=bfloat16" in arguments
    assert "--dsa-prefill-backend=flashmla_sparse" in arguments
    assert "--dsa-decode-backend=flashmla_kv" in arguments
    assert "--enforce-disable-flashinfer-allreduce-fusion" in arguments
    assert catalog["models"][0]["context_limit"] == 262144
