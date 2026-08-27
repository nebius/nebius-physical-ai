"""Standalone OpenAI-compatible tool agent for the Sim2Real benchmark.

This module deliberately has no dependency on ``npa.agent_backend`` or any
``npa cli agent`` surface. It drives one isolated detached-HEAD workspace and
records an append-only transcript suitable for private benchmark evidence.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import http.client
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from npa.benchmarks.sim2real_prepared_action import (
    PreparedActionContext,
    PreparedActionError,
    _append_private_jsonl,
    _prior_execution_state,
    create_receipt_from_request,
    execute_prepared_action,
    recover_occurrence,
    rejected_result,
    validate_receipt,
)
from npa.benchmarks.sim2real_success import VerificationError, verify_artifact_tree


TASK_TEXT = """From a clean checkout on the dev VM, first stage the canonical public-franka-lift preset through the NPA workflow trigger command, then validate and plan npa/workflows/workbench/npa-workflows/sim2real.yaml, submit it through the standard runtime, and monitor that run to terminal completion. Diagnose and make necessary fixes if it fails. Do not weaken the canonical workflow, its real components, or the strict requirement that the Franka arm grasp the cube, lift it at least 5 cm, and hold it for 2 seconds. Preserve unrelated changes. Finish with the run ID, commands, code changes, measured stage and grasp metrics, and artifact locations."""
CHECKPOINT_MARKER = "BENCHMARK_CONTEXT_CHECKPOINT_V1"
CHECKPOINT_SUBMIT_ATTEMPT_MARKER = "Recent standalone workflow submit attempts: "
RECOVERY_MARKER = "BENCHMARK_MALFORMED_RESPONSE_RECOVERY_V1"
PREPARED_ACTION_MARKER = "BENCHMARK_PREPARED_WORKFLOW_ACTION_V1"
MAX_CONTEXT_TOOL_RESULT_CHARACTERS = 4_096
# The pinned sparse-prefill deployment exhausted rank memory near 72k prompt
# tokens despite a larger advertised model context. This is an active-context
# compaction boundary, not a benchmark time, token, cost, or job budget.
EFFECTIVE_CONTEXT_CHECKPOINT_PROMPT_TOKENS = 60_000
MAX_CHECKPOINT_RECENT_CHARACTERS = 16_384
MAX_CHECKPOINT_WORKSPACE_STATUS_CHARACTERS = 8_192
MAX_CHECKPOINT_RUN_IDENTIFIERS = 16

DEFAULT_STREAM_SAFEGUARDS = {
    # These are per-response semantic-progress safeguards, not benchmark budgets.
    "idle_timeout_seconds": 180.0,
    "no_tool_progress_seconds": 900.0,
    "no_tool_progress_characters": 65_536,
    "tool_assembly_seconds": 300.0,
    "tool_assembly_characters": 65_536,
    "max_identical_malformed_responses": 3,
}

_TRANSIENT_TRANSPORT_ERRORS = (
    urllib.error.URLError,
    http.client.HTTPException,
    ConnectionError,
    TimeoutError,
)


BASE_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a shell command in the isolated trial workspace and return exact output.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 file beneath the isolated trial workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or replace a UTF-8 file beneath the isolated trial workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List workspace files matching a glob without reading their contents.",
            "parameters": {
                "type": "object",
                "properties": {"pattern": {"type": "string"}},
                "required": ["pattern"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "complete_workflow",
            "description": (
                "Ask the benchmark controller to independently verify that an NPA "
                "workflow run reached a terminal state. Use only after monitoring "
                "the submitted run to completion."
            ),
            "parameters": {
                "type": "object",
                "properties": {"run_id": {"type": "string"}},
                "required": ["run_id"],
                "additionalProperties": False,
            },
        },
    },
]

PREPARED_ACTION_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "submit_prepared_workflow",
        "description": (
            "Execute one operator-prepared, private, immutable NPA workflow action. "
            "Pass only the advertised action_id; do not reconstruct shell syntax or "
            "repeat private project, image, input, EULA, resume, or secret settings."
        ),
        "parameters": {
            "type": "object",
            "properties": {"action_id": {"type": "string"}},
            "required": ["action_id"],
            "additionalProperties": False,
        },
    },
}

TOOLS: list[dict[str, Any]] = [*BASE_TOOLS, PREPARED_ACTION_TOOL]


def _active_tools(prepared_receipt_path: Path | None) -> list[dict[str, Any]]:
    """Preserve legacy tool metadata unless a prepared action is configured."""

    return TOOLS if prepared_receipt_path is not None else BASE_TOOLS


_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@dataclass
class RequestTelemetry:
    request_index: int
    started_at: str
    latency_seconds: float
    time_to_first_token_seconds: float | None
    prompt_tokens: int | None
    cached_tokens: int | None
    completion_tokens: int | None
    reasoning_tokens: int | None
    total_tokens: int | None
    completion_tokens_per_second: float | None
    finish_reason: str | None
    observed_tokens_lower_bound: int
    observed_characters_lower_bound: int


class EmptyStreamError(RuntimeError):
    """The server closed a streaming response without a usable event."""

    def __init__(self, message: str, telemetry: dict[str, Any]) -> None:
        super().__init__(message)
        self.telemetry = telemetry


class StreamRecoveryError(RuntimeError):
    """A response made no usable tool-call progress and must be discarded."""

    def __init__(self, reason: str, telemetry: dict[str, Any]) -> None:
        super().__init__(reason)
        self.reason = reason
        self.telemetry = telemetry


class IndeterminateToolExecutionError(RuntimeError):
    """A write-ahead tool intent has no durable result after restart."""

    def __init__(self, response_id: str, tool_call_ids: list[str]) -> None:
        super().__init__("tool execution may have completed before its result was journaled")
        self.response_id = response_id
        self.tool_call_ids = tool_call_ids


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _elapsed_since(value: str) -> float:
    started = datetime.fromisoformat(value)
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    return max(0.0, (datetime.now(timezone.utc) - started).total_seconds())


def _sha(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _inside(workspace: Path, value: str) -> Path:
    candidate = (workspace / value).resolve()
    try:
        candidate.relative_to(workspace.resolve())
    except ValueError as exc:
        raise ValueError("path escapes the isolated trial workspace") from exc
    return candidate


def _run_tool(
    name: str,
    arguments: dict[str, Any],
    workspace: Path,
    env: dict[str, str],
    isolation: dict[str, Path] | None = None,
    *,
    prepared_receipt_path: Path | None = None,
    prepared_context: PreparedActionContext | None = None,
    occurrence_id: str = "",
) -> dict[str, Any]:
    started = time.monotonic()
    if name == "submit_prepared_workflow":
        action_id = str(arguments.get("action_id") or "").strip()
        if prepared_receipt_path is None or prepared_context is None:
            return rejected_result(
                action_id=action_id,
                classification="prepared_action_unavailable",
                message="no prepared workflow action is configured for this trial",
            )
        return execute_prepared_action(
            prepared_receipt_path,
            requested_action_id=action_id,
            occurrence_id=occurrence_id,
            context=prepared_context,
        )
    if name == "complete_workflow":
        run_id = str(arguments.get("run_id") or "").strip()
        project = str(env.get("NPA_PROJECT") or "").strip()
        if not _RUN_ID_RE.fullmatch(run_id):
            raise ValueError("run_id has an invalid format")
        if not project:
            raise ValueError("NPA_PROJECT is required for terminal verification")
        command = (
            "npa/.venv/bin/python -m npa workbench workflow status "
            f"{shlex.quote(run_id)} --project {shlex.quote(project)} --json"
        )
        observed = _run_tool(
            "run_command",
            {"command": command},
            workspace,
            env,
            isolation,
        )
        if int(observed.get("exit_code") or 0) != 0:
            return {
                "run_id": run_id,
                "terminal": False,
                "error": "authoritative workflow status lookup failed",
                "exit_code": observed.get("exit_code"),
                "stderr": str(observed.get("stderr") or "")[-1000:],
            }
        try:
            payload = json.loads(str(observed.get("stdout") or ""))
        except json.JSONDecodeError:
            return {
                "run_id": run_id,
                "terminal": False,
                "error": "authoritative workflow status was not valid JSON",
            }
        status = str(payload.get("status") or "").strip().upper()
        from npa.orchestration.npa_workflow.runtime import is_terminal

        return {
            "run_id": run_id,
            "status": status,
            "terminal": is_terminal(status),
            "workflow_succeeded": status
            in {"SUCCEEDED", "SUCCESS", "COMPLETED", "DONE"},
            "duration_seconds": time.monotonic() - started,
        }
    if name == "run_command":
        command = ["bash", "-lc", str(arguments["command"])]
        if isolation:
            namespace_script = r"""
set -eu
mount --make-rprivate /
mount -t tmpfs tmpfs /tmp
mkdir -p /tmp/npa-private-evidence
mkdir -p /tmp/npa-trial-workspace
mount --bind "$1" /tmp/npa-private-evidence
mount --bind "$2" /tmp/npa-trial-workspace
cd /tmp/npa-trial-workspace
mount -t tmpfs tmpfs "$3"
mount -t tmpfs tmpfs "$4"
exec bash -c "$5"
"""
            command = [
                "unshare",
                "--user",
                "--map-root-user",
                "--mount",
                "--pid",
                "--fork",
                "bash",
                "-c",
                namespace_script,
                "--",
                str(isolation["evidence"]),
                str(workspace),
                str(isolation["private_root"]),
                str(isolation["controller_repo"]),
                str(arguments["command"]),
            ]
        completed = subprocess.run(
            command,
            cwd=workspace,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        return {
            "exit_code": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "duration_seconds": time.monotonic() - started,
        }
    if name == "read_file":
        path = _inside(workspace, str(arguments["path"]))
        return {
            "path": str(path.relative_to(workspace)),
            "content": path.read_text(encoding="utf-8"),
        }
    if name == "write_file":
        path = _inside(workspace, str(arguments["path"]))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(arguments["content"]), encoding="utf-8")
        return {"path": str(path.relative_to(workspace)), "bytes": path.stat().st_size}
    if name == "list_files":
        pattern = str(arguments["pattern"])
        if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
            raise ValueError("glob must remain beneath the isolated trial workspace")
        return {
            "files": sorted(
                str(path.relative_to(workspace)) for path in workspace.glob(pattern)
            )
        }
    raise ValueError(f"unknown tool: {name}")


def _stream_chat(
    endpoint: str,
    api_key: str,
    payload: dict[str, Any],
    *,
    safeguards: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], RequestTelemetry]:
    policy = {**DEFAULT_STREAM_SAFEGUARDS, **(safeguards or {})}
    started_at = _utc()
    body = json.dumps(payload).encode()
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    started = time.monotonic()
    first: float | None = None
    tool_started: float | None = None
    tool_started_characters = 0
    content: list[str] = []
    reasoning: list[str] = []
    tool_calls: dict[int, dict[str, Any]] = {}
    usage: dict[str, Any] = {}
    finish_reason: str | None = None
    observed_tokens_lower_bound = 0
    observed_characters_lower_bound = 0

    def progress_record(reason: str) -> dict[str, Any]:
        return {
            "started_at": started_at,
            "elapsed_seconds": time.monotonic() - started,
            "time_to_first_token_seconds": first,
            "observed_tokens_lower_bound": observed_tokens_lower_bound,
            "observed_characters_lower_bound": observed_characters_lower_bound,
            "reason": reason,
            "finish_reason": finish_reason,
            "has_content": bool(content),
            "has_reasoning": bool(reasoning),
            "tool_call_fragments_observed": bool(tool_calls),
            "tool_call_indexes_observed": sorted(tool_calls),
        }

    def append_fragment(target: list[str], value: Any, field: str) -> None:
        nonlocal observed_characters_lower_bound
        if value in (None, ""):
            return
        if not isinstance(value, str):
            raise StreamRecoveryError(
                f"non_string_{field}_fragment", progress_record("malformed_stream_shape")
            )
        target.append(value)
        observed_characters_lower_bound += len(value)

    def enforce_semantic_progress() -> None:
        if finish_reason == "tool_calls":
            return
        elapsed = time.monotonic() - started
        if not tool_calls and (
            elapsed >= float(policy["no_tool_progress_seconds"])
            or observed_characters_lower_bound
            >= int(policy["no_tool_progress_characters"])
        ):
            raise StreamRecoveryError(
                "no_usable_tool_call_progress",
                progress_record("no_usable_tool_call_progress"),
            )
        if tool_started is not None and (
            time.monotonic() - tool_started
            >= float(policy["tool_assembly_seconds"])
            or observed_characters_lower_bound - tool_started_characters
            >= int(policy["tool_assembly_characters"])
        ):
            raise StreamRecoveryError(
                "tool_call_boundary_not_completed",
                progress_record("tool_call_boundary_not_completed"),
            )

    try:
        with urllib.request.urlopen(
            request, timeout=float(policy["idle_timeout_seconds"])
        ) as response:
            for raw in response:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    enforce_semantic_progress()
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if not isinstance(chunk, dict):
                    raise StreamRecoveryError(
                        "malformed_stream_shape",
                        progress_record("malformed_stream_shape"),
                    )
                chunk_usage = chunk.get("usage")
                if chunk_usage:
                    if not isinstance(chunk_usage, dict):
                        raise StreamRecoveryError(
                            "malformed_stream_shape",
                            progress_record("malformed_stream_shape"),
                        )
                    if any(
                        key in chunk_usage
                        and chunk_usage[key] is not None
                        and not isinstance(chunk_usage[key], dict)
                        for key in (
                            "prompt_tokens_details",
                            "completion_tokens_details",
                        )
                    ):
                        raise StreamRecoveryError(
                            "malformed_stream_shape",
                            progress_record("malformed_stream_shape"),
                        )
                    usage = chunk_usage
                choices = chunk.get("choices") or []
                if not isinstance(choices, list):
                    raise StreamRecoveryError(
                        "malformed_stream_shape",
                        progress_record("malformed_stream_shape"),
                    )
                if not choices:
                    enforce_semantic_progress()
                    continue
                choice = choices[0]
                if not isinstance(choice, dict):
                    raise StreamRecoveryError(
                        "malformed_stream_shape",
                        progress_record("malformed_stream_shape"),
                    )
                delta = choice.get("delta") or {}
                if not isinstance(delta, dict):
                    raise StreamRecoveryError(
                        "malformed_stream_shape",
                        progress_record("malformed_stream_shape"),
                    )
                if any(
                    delta.get(key)
                    for key in (
                        "content",
                        "reasoning",
                        "reasoning_content",
                        "tool_calls",
                    )
                ):
                    # A token-bearing streamed delta proves at least one generated
                    # token without assuming a tokenizer or chunks-per-token ratio.
                    observed_tokens_lower_bound += 1
                if first is None and any(
                    delta.get(key)
                    for key in (
                        "content",
                        "reasoning",
                        "reasoning_content",
                        "tool_calls",
                    )
                ):
                    first = time.monotonic() - started
                append_fragment(content, delta.get("content"), "content")
                append_fragment(
                    reasoning,
                    delta.get("reasoning") or delta.get("reasoning_content"),
                    "reasoning",
                )
                streamed_calls = delta.get("tool_calls") or []
                if not isinstance(streamed_calls, list) or any(
                    not isinstance(call, dict) for call in streamed_calls
                ):
                    raise StreamRecoveryError(
                        "malformed_stream_shape",
                        progress_record("malformed_stream_shape"),
                    )
                for call in streamed_calls:
                    if tool_started is None:
                        tool_started = time.monotonic()
                        tool_started_characters = observed_characters_lower_bound
                    try:
                        index = int(call.get("index", 0))
                    except (TypeError, ValueError) as exc:
                        raise StreamRecoveryError(
                            "invalid_tool_call_index",
                            progress_record("malformed_stream_shape"),
                        ) from exc
                    current = tool_calls.setdefault(
                        index,
                        {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        },
                    )
                    streamed_function = call.get("function") or {}
                    if not isinstance(streamed_function, dict):
                        raise StreamRecoveryError(
                            "malformed_stream_shape",
                            progress_record("malformed_stream_shape"),
                        )
                    for target, value, field in (
                        (current, call.get("id"), "tool_call_id"),
                        (
                            current["function"],
                            streamed_function.get("name"),
                            "tool_name",
                        ),
                        (
                            current["function"],
                            streamed_function.get("arguments"),
                            "tool_arguments",
                        ),
                    ):
                        key = "id" if field == "tool_call_id" else (
                            "name" if field == "tool_name" else "arguments"
                        )
                        fragments: list[str] = []
                        append_fragment(fragments, value, field)
                        if fragments:
                            target[key] += fragments[0]
                finish_reason = choice.get("finish_reason") or finish_reason
                if finish_reason == "tool_calls":
                    continue
                enforce_semantic_progress()
    except (TimeoutError, socket.timeout) as exc:
        if finish_reason != "tool_calls":
            raise StreamRecoveryError(
                "stream_idle_timeout", progress_record("stream_idle_timeout")
            ) from exc
    except json.JSONDecodeError as exc:
        raise StreamRecoveryError(
            "malformed_sse_json", progress_record("malformed_sse_json")
        ) from exc
    except _TRANSIENT_TRANSPORT_ERRORS as exc:
        if any((first, content, reasoning, tool_calls, usage, finish_reason)):
            raise StreamRecoveryError(
                "stream_transport_interrupted",
                progress_record("stream_transport_interrupted"),
            ) from exc
        raise
    if not any((content, reasoning, tool_calls, usage, finish_reason, first)):
        raise EmptyStreamError(
            "OpenAI-compatible endpoint returned an empty event stream",
            progress_record("empty_event_stream"),
        )
    details = usage.get("prompt_tokens_details") or {}
    completion_details = usage.get("completion_tokens_details") or {}
    message: dict[str, Any] = {"role": "assistant", "content": "".join(content) or None}
    if reasoning:
        message["reasoning_content"] = "".join(reasoning)
    if tool_calls:
        message["tool_calls"] = [tool_calls[index] for index in sorted(tool_calls)]
    latency = time.monotonic() - started
    completion_tokens = usage.get("completion_tokens")
    telemetry = RequestTelemetry(
        request_index=0,
        started_at=started_at,
        latency_seconds=latency,
        time_to_first_token_seconds=first,
        prompt_tokens=usage.get("prompt_tokens"),
        cached_tokens=details.get("cached_tokens"),
        completion_tokens=completion_tokens,
        reasoning_tokens=completion_details.get("reasoning_tokens"),
        total_tokens=usage.get("total_tokens"),
        completion_tokens_per_second=(
            float(completion_tokens) / latency
            if completion_tokens is not None and latency > 0
            else None
        ),
        finish_reason=finish_reason,
        observed_tokens_lower_bound=observed_tokens_lower_bound,
        observed_characters_lower_bound=observed_characters_lower_bound,
    )
    return message, telemetry


def _tool_argument_contract(name: str) -> tuple[set[str], set[str]]:
    contracts = {
        "run_command": ({"command"}, {"command"}),
        "read_file": ({"path"}, {"path"}),
        "write_file": ({"path", "content"}, {"path", "content"}),
        "list_files": ({"pattern"}, {"pattern"}),
        "complete_workflow": ({"run_id"}, {"run_id"}),
        "submit_prepared_workflow": ({"action_id"}, {"action_id"}),
    }
    try:
        return contracts[name]
    except KeyError as exc:
        raise ValueError(f"unknown tool name: {name}") from exc


def _validated_tool_calls(
    assistant: dict[str, Any], *, finish_reason: str | None
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    calls = assistant.get("tool_calls") or []
    if not isinstance(calls, list) or not calls:
        raise ValueError("response contains no tool calls")
    if finish_reason != "tool_calls":
        raise ValueError("response ended without a tool_calls finish boundary")
    validated: list[tuple[dict[str, Any], dict[str, Any]]] = []
    seen_ids: set[str] = set()
    for call in calls:
        if not isinstance(call, dict) or call.get("type") != "function":
            raise ValueError("tool call must be a function object")
        call_id = call.get("id")
        if not isinstance(call_id, str) or not call_id or call_id in seen_ids:
            raise ValueError("tool call id must be non-empty and unique")
        seen_ids.add(call_id)
        function = call.get("function")
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            raise ValueError("tool call function name is missing")
        name = function["name"]
        allowed, required = _tool_argument_contract(name)
        raw_arguments = function.get("arguments")
        if not isinstance(raw_arguments, str):
            raise ValueError("tool arguments must be a JSON string")
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            raise ValueError("tool arguments are incomplete or invalid JSON") from exc
        if not isinstance(arguments, dict):
            raise ValueError("tool arguments must decode to an object")
        keys = set(arguments)
        if not required <= keys or not keys <= allowed:
            raise ValueError("tool arguments do not match the declared schema")
        if any(not isinstance(arguments[key], str) for key in keys):
            raise ValueError("tool argument values must be strings")
        validated.append((call, arguments))
    return validated


def _bounded_tool_result(
    result: dict[str, Any],
    *,
    max_characters: int = MAX_CONTEXT_TOOL_RESULT_CHARACTERS,
) -> dict[str, Any]:
    """Keep full evidence off-context while preserving a hash-bound useful preview."""

    serialized = json.dumps(result, sort_keys=True, separators=(",", ":"))
    if len(serialized) <= max_characters:
        return result
    run_identifiers: set[str] = set()

    def collect_run_identifiers(value: Any, key: str | None = None) -> None:
        if isinstance(value, dict):
            for child_key, child in value.items():
                collect_run_identifiers(child, str(child_key))
        elif isinstance(value, list):
            for child in value:
                collect_run_identifiers(child, key)
        elif isinstance(value, str):
            if key in {"run_id", "workflow_run_id"} and _RUN_ID_RE.fullmatch(value):
                run_identifiers.add(value)
                return
            if value[:1] in {"{", "["}:
                try:
                    collect_run_identifiers(json.loads(value), key)
                except json.JSONDecodeError:
                    pass

    collect_run_identifiers(result)
    preserved = {
        key: result[key]
        for key in ("exit_code", "status", "terminal", "workflow_succeeded")
        if key in result
        and isinstance(result[key], (bool, int, float, str))
        and (not isinstance(result[key], str) or len(result[key]) <= 256)
    }
    if run_identifiers:
        preserved["run_identifiers"] = [
            {"run_id": run_id} for run_id in sorted(run_identifiers)[:8]
        ]

    head_characters = min(2_048, max(256, max_characters // 2))
    tail_characters = min(512, max(128, max_characters // 8))
    bounded = {
        "_npa_context_truncated": True,
        "full_result_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
        "full_result_characters": len(serialized),
        **preserved,
        "preview_head": serialized[:head_characters],
        "preview_tail": serialized[-tail_characters:],
        "recovery_guidance": (
            "The full result is retained in private append-only evidence. "
            "Use run_command with rg or sed for a targeted excerpt; do not repeat "
            "the same broad read_file or list_files call."
        ),
    }
    while (
        len(json.dumps(bounded, sort_keys=True, separators=(",", ":")))
        > max_characters
    ):
        if len(bounded["preview_head"]) > 256:
            bounded["preview_head"] = bounded["preview_head"][:-256]
        elif len(bounded["preview_tail"]) > 128:
            bounded["preview_tail"] = bounded["preview_tail"][:-128]
        else:
            break
    return bounded


def _serialize_bounded_tool_result(result: dict[str, Any]) -> str:
    return json.dumps(
        _bounded_tool_result(result), sort_keys=True, separators=(",", ":")
    )


def _response_shape(assistant: dict[str, Any]) -> dict[str, Any]:
    calls = assistant.get("tool_calls") or []
    return {
        "has_content": bool(assistant.get("content")),
        "has_reasoning": bool(assistant.get("reasoning_content")),
        "tool_call_count": len(calls) if isinstance(calls, list) else None,
        "tool_names": [
            str((call.get("function") or {}).get("name") or "")
            for call in calls
            if isinstance(call, dict)
        ],
    }


def _malformation_fingerprint(reason: str, response_shape: dict[str, Any]) -> str:
    return _sha({"reason": reason, "response_shape": response_shape})


def _load_malformation_streak(path: Path) -> tuple[str | None, int]:
    fingerprint: str | None = None
    count = 0
    if not path.exists():
        return fingerprint, count
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("classification") == "response_completed":
            fingerprint, count = None, 0
        elif record.get("classification") == "malformed_response":
            observed = str(record.get("malformation_fingerprint") or "")
            if observed and observed == fingerprint:
                count += 1
            else:
                fingerprint, count = observed or None, 1
    return fingerprint, count


def _next_malformation_streak(
    previous_fingerprint: str | None,
    previous_count: int,
    fingerprint: str,
) -> tuple[str, int]:
    return (
        (fingerprint, previous_count + 1)
        if fingerprint == previous_fingerprint
        else (fingerprint, 1)
    )


def _malformation_recovery_action(count: int, maximum: int) -> str:
    return (
        "terminate_repeated_identical_malformed_response"
        if count >= maximum
        else "discard_partial_response_rebuild_context_and_retry"
    )


def _malformation_telemetry_record(
    recovery: dict[str, Any],
    *,
    request_index: int,
    response_shape: dict[str, Any],
    fingerprint: str,
    identical_count: int,
    action: str,
) -> dict[str, Any]:
    return {
        **recovery,
        "request_index": request_index,
        "classification": "malformed_response",
        "response_shape": response_shape,
        "malformation_fingerprint": fingerprint,
        "identical_malformation_count": identical_count,
        "recovery_action": action,
        "at": _utc(),
    }


def _terminal_malformation_failure(
    *,
    request_index: int,
    fingerprint: str,
    identical_count: int,
    reason: str,
    workflow_submitted: bool,
    run_identifiers: list[str],
) -> dict[str, Any]:
    return {
        "schema": "npa.sim2real.model_agent_benchmark.failure.v2",
        "classification": "repeated_identical_malformed_response",
        "request_index": request_index,
        "malformation_fingerprint": fingerprint,
        "identical_malformation_count": identical_count,
        "last_reason": reason,
        "workflow_submitted": workflow_submitted,
        "workflow_run_identifiers": run_identifiers,
        "completed_at": _utc(),
    }


def _transport_telemetry_record(
    exc: BaseException,
    *,
    request_index: int,
    elapsed_seconds: float,
    recovery_action: str,
) -> dict[str, Any]:
    return {
        "request_index": request_index,
        "classification": "transport_error",
        "elapsed_seconds": elapsed_seconds,
        "observed_tokens_lower_bound": 0,
        "observed_characters_lower_bound": 0,
        "reason": type(exc).__name__,
        "transport_error": repr(exc),
        "recovery_action": recovery_action,
        "at": _utc(),
    }


def _is_permanent_model_http_error(exc: BaseException) -> bool:
    return (
        isinstance(exc, urllib.error.HTTPError)
        and 400 <= exc.code < 500
        and exc.code not in {408, 409, 425, 429}
    )


def _read_transcript_messages(path: Path) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if not path.exists():
        transcript_lines: list[str] = []
    else:
        transcript_lines = path.read_text(encoding="utf-8").splitlines()
    last_transcript_line = max(
        (index for index, line in enumerate(transcript_lines, 1) if line.strip()),
        default=0,
    )
    for line_number, line in enumerate(transcript_lines, 1):
        if not line.strip():
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            if line_number == last_transcript_line:
                break
            raise
        if not isinstance(message, dict) or message.get("role") not in {
            "assistant",
            "tool",
            "user",
        }:
            raise ValueError(f"invalid transcript entry at line {line_number}")
        messages.append(message)
    return messages


def _load_transcript(
    path: Path,
    tool_results_path: Path | None = None,
    prepared_state_path: Path | None = None,
) -> list[dict[str, Any]]:
    messages = _read_transcript_messages(path)
    complete_response_ids = {
        str(group[0].get("_npa_response_id") or "")
        for group in _safe_message_groups(messages)
        if group and group[0].get("role") == "assistant" and group[0].get("tool_calls")
    }
    complete_response_ids.discard("")
    checkpoint_indexes = [
        index
        for index, message in enumerate(messages)
        if message.get("role") == "user"
        and str(message.get("content") or "").startswith(CHECKPOINT_MARKER)
    ]
    if checkpoint_indexes:
        messages = messages[checkpoint_indexes[-1] :]
    messages = [message for group in _safe_message_groups(messages) for message in group]
    if tool_results_path is None or not tool_results_path.exists():
        return messages

    journal_groups: dict[str, dict[str, Any]] = {}
    journal_lines = tool_results_path.read_text(encoding="utf-8").splitlines()
    last_journal_line = max(
        (index for index, line in enumerate(journal_lines, 1) if line.strip()),
        default=0,
    )
    for line_number, line in enumerate(journal_lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if line_number == last_journal_line:
                break
            raise
        if record.get("schema") != "npa.sim2real.tool_execution.v2":
            continue
        response_id = str(record.get("response_id") or "")
        if not response_id:
            raise ValueError(f"invalid tool journal entry at line {line_number}")
        group = journal_groups.setdefault(
            response_id,
            {"assistant": record.get("assistant"), "intents": {}, "results": {}},
        )
        phase = record.get("phase")
        call_id = str(record.get("tool_call_id") or "")
        if phase == "intent" and call_id:
            group["intents"][call_id] = record
        elif phase == "result" and call_id:
            group["results"][call_id] = record.get("tool_message")
        elif phase == "transcript_committed":
            group["committed"] = True

    for response_id, group in journal_groups.items():
        if group.get("committed") or response_id in complete_response_ids:
            continue
        assistant = group.get("assistant")
        if not isinstance(assistant, dict):
            raise ValueError(f"tool journal response {response_id} has no assistant")
        unresolved = sorted(set(group["intents"]) - set(group["results"]))
        recovered_results: dict[str, dict[str, Any]] = {}
        still_indeterminate: list[str] = []
        for call_id in unresolved:
            intent = group["intents"][call_id]
            if (
                intent.get("tool_name") != "submit_prepared_workflow"
                or prepared_state_path is None
            ):
                still_indeterminate.append(call_id)
                continue
            occurrence_id = str(intent.get("occurrence_id") or "")
            action_id = str(intent.get("prepared_action_id") or "")
            recovery, recovered = recover_occurrence(
                prepared_state_path,
                action_id=action_id,
                occurrence_id=occurrence_id,
            )
            if recovery == "indeterminate":
                still_indeterminate.append(call_id)
            elif recovery == "finished" and recovered is not None:
                recovered_results[call_id] = {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": _serialize_bounded_tool_result(recovered),
                }
            else:
                recovered_results[call_id] = {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": _serialize_bounded_tool_result(
                        {
                            "error": "ControllerRecovery",
                            "classification": "prepared_action_not_started",
                            "message": (
                                "The controller stopped before the prepared action "
                                "crossed its durable execution boundary; it is safe "
                                "to invoke the same typed action again."
                            ),
                        }
                    ),
                }
        if still_indeterminate:
            raise IndeterminateToolExecutionError(response_id, still_indeterminate)
        tool_messages: list[dict[str, Any]] = []
        for call in assistant.get("tool_calls") or []:
            call_id = str(call.get("id") or "") if isinstance(call, dict) else ""
            has_result = call_id in group["results"] or call_id in recovered_results
            tool_message = group["results"].get(call_id) or recovered_results.get(call_id)
            if has_result and not isinstance(tool_message, dict):
                raise IndeterminateToolExecutionError(response_id, [call_id])
            if not has_result:
                tool_message = {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": _serialize_bounded_tool_result(
                        {
                            "error": "ControllerRecovery",
                            "message": (
                                "This tool call was not executed before the controller "
                                "stopped. Inspect durable state before retrying it."
                            ),
                        }
                    ),
                }
            tool_messages.append(tool_message)
        messages.extend([assistant, *tool_messages])
    return messages


def _last_request_index(path: Path) -> int:
    last = 0
    if not path.exists():
        return last
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        record = json.loads(line)
        if (
            record.get("classification") == "telemetry_correction"
            and "request_index" not in record
        ):
            continue
        try:
            last = max(last, int(record["request_index"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid request telemetry entry at line {line_number}"
            ) from exc
    return last


def _safe_message_groups(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        if message.get("role") == "tool":
            index += 1
            continue
        if message.get("role") != "assistant" or not message.get("tool_calls"):
            groups.append([message])
            index += 1
            continue
        expected = {
            str(call.get("id"))
            for call in message.get("tool_calls") or []
            if isinstance(call, dict) and call.get("id")
        }
        group = [message]
        observed: set[str] = set()
        cursor = index + 1
        while cursor < len(messages) and messages[cursor].get("role") == "tool":
            group.append(messages[cursor])
            observed.add(str(messages[cursor].get("tool_call_id") or ""))
            cursor += 1
        if expected and observed == expected:
            groups.append(group)
        index = cursor
    return groups


def _collect_run_identifiers(messages: list[dict[str, Any]]) -> list[str]:
    found: set[str] = set()

    def visit(value: Any, key: str | None = None) -> None:
        if isinstance(value, dict):
            for child_key, child in value.items():
                visit(child, str(child_key))
        elif isinstance(value, list):
            for child in value:
                visit(child, key)
        elif key in {"run_id", "workflow_run_id", "safe_run_reference"} and isinstance(value, str):
            if _RUN_ID_RE.fullmatch(value):
                found.add(value)

    for message in messages:
        content = message.get("content")
        if message.get("role") == "tool" and isinstance(content, str):
            try:
                visit(json.loads(content))
            except json.JSONDecodeError:
                continue
        elif message.get("role") == "user" and isinstance(content, str):
            for line in content.splitlines():
                if not line.startswith("Durable workflow run identifiers: "):
                    continue
                try:
                    visit(json.loads(line.split(": ", 1)[1]), "run_id")
                except json.JSONDecodeError:
                    continue
    return sorted(found)


def _submitted_workflow_state(
    messages: list[dict[str, Any]],
) -> tuple[bool, list[str]]:
    """Return only run IDs backed by a successful non-plan workflow submit."""

    tool_results = {
        str(message.get("tool_call_id") or ""): message
        for message in messages
        if message.get("role") == "tool" and message.get("tool_call_id")
    }
    submitted = False
    run_ids: set[str] = set()

    for message in messages:
        content = str(message.get("content") or "")
        if message.get("role") != "user" or not content.startswith(
            CHECKPOINT_MARKER
        ):
            continue
        for line in content.splitlines():
            if not line.startswith("Durable workflow submission state: "):
                continue
            try:
                state = json.loads(line.split(": ", 1)[1])
            except json.JSONDecodeError:
                continue
            if state.get("submitted") is True:
                submitted = True
            for run_id in state.get("run_ids") or []:
                if isinstance(run_id, str) and _RUN_ID_RE.fullmatch(run_id):
                    run_ids.add(run_id)

    def collect(value: Any, key: str | None = None) -> None:
        if isinstance(value, dict):
            for child_key, child in value.items():
                collect(child, str(child_key))
        elif isinstance(value, list):
            for child in value:
                collect(child, key)
        elif isinstance(value, str):
            if key in {"run_id", "workflow_run_id", "safe_run_reference"} and _RUN_ID_RE.fullmatch(value):
                run_ids.add(value)
                return
            if value[:1] in {"{", "["}:
                try:
                    collect(json.loads(value), key)
                except json.JSONDecodeError:
                    pass

    for message in messages:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            function = call.get("function")
            if not isinstance(function, dict):
                continue
            result_message = tool_results.get(str(call.get("id") or ""))
            if function.get("name") == "submit_prepared_workflow":
                if result_message is None:
                    continue
                try:
                    prepared_result = json.loads(
                        str(result_message.get("content") or "")
                    )
                except json.JSONDecodeError:
                    continue
                if prepared_result.get("submission_accepted") is True:
                    submitted = True
                    collect(prepared_result)
                continue
            if function.get("name") != "run_command":
                continue
            try:
                arguments = json.loads(str(function.get("arguments") or ""))
                command = str(arguments.get("command") or "")
            except json.JSONDecodeError:
                continue
            if _workflow_submit_command_kind(command) != "standalone":
                continue
            if result_message is None:
                continue
            try:
                result = json.loads(str(result_message.get("content") or ""))
            except json.JSONDecodeError:
                continue
            try:
                if "exit_code" not in result:
                    continue
                exit_code = int(result["exit_code"])
            except (AttributeError, TypeError, ValueError):
                continue
            if exit_code != 0:
                continue
            submitted = True
            collect(result)
    return submitted, sorted(run_ids)


def _prepared_action_consumed_state(
    messages: list[dict[str, Any]], action_id: str = ""
) -> bool:
    current_action_id = ""
    for message in messages:
        content = str(message.get("content") or "")
        if message.get("role") == "user":
            if content.startswith(PREPARED_ACTION_MARKER):
                match = re.search(r"Action ID: ([A-Za-z0-9._-]+)", content)
                if match:
                    current_action_id = match.group(1).rstrip(".")
            for line in content.splitlines():
                if not line.startswith("Durable prepared action state: "):
                    continue
                try:
                    state = json.loads(line.split(": ", 1)[1])
                except json.JSONDecodeError:
                    continue
                state_action_id = str(state.get("action_id") or "")
                if state.get("consumed") is True and (
                    not action_id
                    or state_action_id == action_id
                    or (not state_action_id and current_action_id == action_id)
                ):
                    return True
        if message.get("role") != "tool":
            continue
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            continue
        if (
            result.get("schema")
            == "npa.sim2real.prepared_workflow_action.result.v1"
            and result.get("action_consumed") is True
            and (
                not action_id
                or str(result.get("action_id") or "") == action_id
                or current_action_id == action_id
            )
        ):
            return True
    return False


def _workflow_submit_command_kind(command: str) -> str:
    """Classify submit-like shell text without permitting compound side effects."""

    try:
        tokens = shlex.split(command)
    except ValueError:
        return (
            "unsafe"
            if re.search(r"\bworkbench\s+workflow\s+submit\b", command)
            else "none"
        )
    sequences = [
        index
        for index in range(max(0, len(tokens) - 2))
        if tokens[index : index + 3] == ["workbench", "workflow", "submit"]
    ]
    raw_submit_like = bool(
        re.search(r"\bworkbench\s+workflow\s+submit\b", command)
    )
    if not sequences and not raw_submit_like:
        return "none"
    if len(sequences) != 1:
        return "unsafe"
    index = sequences[0]
    if index != 1 or Path(tokens[0]).name != "npa":
        return "unsafe"
    if re.search(r"[;&|`$<>()\n\r#]", command):
        return "unsafe"
    suffix = tokens[index + 3 :]
    if "--help" in suffix or "-h" in suffix or "--plan-only" in suffix:
        return "introspection"
    if not suffix or suffix[0].startswith("-"):
        return "unsafe"
    return "standalone"


def _workflow_submission_block_reason(
    messages: list[dict[str, Any]],
    *,
    tool_name: str,
    arguments: dict[str, Any],
    durable_prepared_state: str = "unused",
) -> str | None:
    prepared_consumed = durable_prepared_state != "unused"
    if tool_name == "submit_prepared_workflow":
        action_id = str(arguments.get("action_id") or "").strip()
        if _prepared_action_consumed_state(messages, action_id) or prepared_consumed:
            return "DuplicateWorkflowSubmissionBlocked"
        return None
    if tool_name != "run_command":
        return None
    kind = _workflow_submit_command_kind(str(arguments.get("command") or ""))
    if kind == "unsafe":
        return "UnsafeWorkflowSubmissionCommandBlocked"
    if kind == "standalone" and (
        _submitted_workflow_state(messages)[0]
        or _prepared_action_consumed_state(messages)
        or prepared_consumed
    ):
        return "DuplicateWorkflowSubmissionBlocked"
    return None


def _recent_standalone_submit_attempts(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return the two latest complete direct submit attempts, successful or not."""

    recent: list[dict[str, Any]] = []
    for group in _safe_message_groups(messages):
        message = group[0]
        content = str(message.get("content") or "")
        if message.get("role") == "user" and content.startswith(CHECKPOINT_MARKER):
            for line in content.splitlines():
                if not line.startswith(CHECKPOINT_SUBMIT_ATTEMPT_MARKER):
                    continue
                try:
                    candidates = json.loads(
                        line.removeprefix(CHECKPOINT_SUBMIT_ATTEMPT_MARKER)
                    )
                except json.JSONDecodeError:
                    continue
                if isinstance(candidates, dict):
                    candidates = [candidates]
                if isinstance(candidates, list):
                    recent = [
                        _bounded_submit_attempt(
                            candidate.get("command"),
                            candidate.get("result"),
                            occurrence_id=candidate.get("occurrence_id"),
                        )
                        for candidate in candidates[-2:]
                        if isinstance(candidate, dict)
                    ]
            continue
        assistant = message
        if assistant.get("role") != "assistant":
            continue
        results = {
            str(item.get("tool_call_id") or ""): item
            for item in group[1:]
            if item.get("role") == "tool"
        }
        for call in assistant.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            function = call.get("function")
            if not isinstance(function, dict) or function.get("name") != "run_command":
                continue
            try:
                arguments = json.loads(str(function.get("arguments") or ""))
                command = str(arguments.get("command") or "")
            except (AttributeError, json.JSONDecodeError):
                continue
            if _workflow_submit_command_kind(command) != "standalone":
                continue
            result_message = results.get(str(call.get("id") or ""))
            if result_message is None:
                continue
            result_content = str(result_message.get("content") or "")
            try:
                result: Any = json.loads(result_content)
            except json.JSONDecodeError:
                result = result_content
            response_id = str(assistant.get("_npa_response_id") or "")
            tool_call_id = str(call.get("id") or "")
            occurrence_id = _sha(
                {
                    "response": response_id or assistant,
                    "tool_call_id": tool_call_id,
                }
            )
            recent.append(
                _bounded_submit_attempt(command, result, occurrence_id=occurrence_id)
            )
            recent = recent[-2:]
    return recent


def _bounded_submit_attempt(
    command: Any, result: Any, *, occurrence_id: Any = None
) -> dict[str, Any]:
    command_text = (
        command
        if isinstance(command, str)
        else json.dumps(command, sort_keys=True, separators=(",", ":"))
    )
    if len(command_text) > 2_048:
        command_value: Any = {
            "sha256": hashlib.sha256(command_text.encode()).hexdigest(),
            "characters": len(command_text),
            "preview_head": command_text[:1_536],
            "preview_tail": command_text[-256:],
        }
    else:
        command_value = command
    if isinstance(result, dict):
        result_value = _bounded_tool_result(result)
    elif len(str(result)) > MAX_CONTEXT_TOOL_RESULT_CHARACTERS:
        result_value = _bounded_tool_result({"content": str(result)})
    else:
        result_value = result
    attempt = {"command": command_value, "result": result_value}
    if isinstance(occurrence_id, str) and occurrence_id:
        attempt["occurrence_id"] = (
            occurrence_id
            if re.fullmatch(r"[0-9a-f]{64}", occurrence_id)
            else _sha({"legacy_occurrence_id": occurrence_id})
        )
    return attempt


def _merge_submit_attempts(
    preserved: list[dict[str, Any]], active: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for attempt in (*preserved, *active):
        occurrence_id = str(attempt.get("occurrence_id") or "")
        key = f"occurrence:{occurrence_id}" if occurrence_id else f"content:{_sha(attempt)}"
        if key in seen:
            continue
        seen.add(key)
        merged.append(attempt)
    return merged[-2:]


def _context_checkpoint(
    messages: list[dict[str, Any]],
    *,
    max_recent_chars: int,
    workspace_status: str = "",
    recovery_reason: str | None = None,
    preserved_submit_attempts: list[dict[str, Any]] | None = None,
    preserved_run_identifiers: list[str] | None = None,
    durable_prepared_state: str = "unused",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    recent_candidates = [
        message
        for message in messages[2:]
        if not (
            message.get("role") == "user"
            and str(message.get("content") or "").startswith(CHECKPOINT_MARKER)
        )
    ]
    groups = _safe_message_groups(recent_candidates)
    recent_groups: list[list[dict[str, Any]]] = []
    used = 0
    for group in reversed(groups):
        size = len(json.dumps(group, sort_keys=True, separators=(",", ":")))
        if used + size > max_recent_chars:
            break
        recent_groups.append(group)
        used += size
    recent_groups.reverse()
    recent = [message for group in recent_groups for message in group]
    workflow_submitted, submitted_run_ids = _submitted_workflow_state(messages[2:])
    prepared_checkpoints = [
        str(message.get("content") or "")
        for message in messages[2:]
        if message.get("role") == "user"
        and str(message.get("content") or "").startswith(PREPARED_ACTION_MARKER)
    ]
    prepared_action_id = ""
    if prepared_checkpoints:
        match = re.search(r"Action ID: ([A-Za-z0-9._-]+)", prepared_checkpoints[-1])
        prepared_action_id = match.group(1).rstrip(".") if match else ""
    prepared_action_consumed = (
        durable_prepared_state != "unused"
        or _prepared_action_consumed_state(messages[2:], prepared_action_id)
    )
    active_submit_attempts = _recent_standalone_submit_attempts(messages[2:])
    submit_attempts = _merge_submit_attempts(
        preserved_submit_attempts or [], active_submit_attempts
    )
    all_run_ids = sorted(
        set(_collect_run_identifiers(messages[2:]))
        | set(submitted_run_ids)
        | set(preserved_run_identifiers or [])
    )
    checkpoint_run_ids: list[str] = []
    for run_id in (*submitted_run_ids, *all_run_ids):
        if run_id not in checkpoint_run_ids:
            checkpoint_run_ids.append(run_id)
        if len(checkpoint_run_ids) >= MAX_CHECKPOINT_RUN_IDENTIFIERS:
            break
    checkpoint_submitted_run_ids = [
        run_id for run_id in checkpoint_run_ids if run_id in submitted_run_ids
    ]
    workspace_lines = workspace_status.splitlines()
    workspace_excerpt_lines: list[str] = []
    workspace_excerpt_characters = 0
    for line in workspace_lines[:200]:
        added = len(line) + 1
        if (
            workspace_excerpt_characters + added
            > MAX_CHECKPOINT_WORKSPACE_STATUS_CHARACTERS
        ):
            break
        workspace_excerpt_lines.append(line)
        workspace_excerpt_characters += added
    recovery = (
        f"{RECOVERY_MARKER}\nDiscarded response reason: {recovery_reason}. "
        "The incomplete assistant response was not added to history and none of "
        "its tool arguments were executed.\n"
        if recovery_reason
        else ""
    )
    checkpoint = {
        "role": "user",
        "content": (
            f"{CHECKPOINT_MARKER}\n"
            + recovery
            + "The standalone benchmark controller deterministically compacted "
            "earlier complete message groups at a safe boundary. The full "
            "append-only transcript remains private evidence. "
            "Do not infer success from this checkpoint. Re-read durable workspace "
            "and runtime state with tools as needed, then continue the original "
            "task.\n"
            f"Prior active-context SHA256: {_sha(messages[2:])}\n"
            f"Prior messages: {len(messages) - 2}; verbatim recent messages: "
            f"{len(recent)}\n"
            f"Durable workflow run identifiers: {json.dumps(checkpoint_run_ids)}\n"
            "Durable workflow run identifier summary: "
            + json.dumps(
                {
                    "total_count": len(all_run_ids),
                    "all_sha256": _sha(all_run_ids),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            "Durable workflow submission state: "
            + json.dumps(
                {
                    "submitted": workflow_submitted,
                    "run_ids": checkpoint_submitted_run_ids,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            + "Durable prepared action state: "
            + json.dumps(
                {
                    "action_id": prepared_action_id,
                    "consumed": prepared_action_consumed,
                    "available": bool(prepared_action_id) and not prepared_action_consumed,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            + (
                CHECKPOINT_SUBMIT_ATTEMPT_MARKER
                + json.dumps(submit_attempts, sort_keys=True, separators=(",", ":"))
                + "\n"
                if submit_attempts
                else ""
            )
            + (
                f"{PREPARED_ACTION_MARKER}\n"
                + (
                    "Typed action available: none; the prepared action is already consumed.\n"
                    if prepared_action_consumed
                    else "Typed action available: submit_prepared_workflow. "
                    f"Action ID: {prepared_action_id}.\n"
                )
                if prepared_action_id
                else ""
            )
            + f"Workspace status SHA256: {hashlib.sha256(workspace_status.encode()).hexdigest()}; "
            f"status lines: {len(workspace_lines)}\n"
            "Workspace status follows:\n"
            + "\n".join(workspace_excerpt_lines)
            + "\n"
            "Verbatim recent transcript JSON follows:\n"
            + json.dumps(recent, sort_keys=True, separators=(",", ":"))
        ),
    }
    return messages[:2] + [checkpoint], checkpoint


def _maybe_checkpoint(
    messages: list[dict[str, Any]],
    transcript_path: Path,
    *,
    context_limit: int,
    workspace_status: str = "",
    active_tokens_upper_bound: int | None = None,
    durable_prepared_state: str = "unused",
) -> list[dict[str, Any]]:
    checkpoint_tokens = min(
        int(context_limit * 0.85), EFFECTIVE_CONTEXT_CHECKPOINT_PROMPT_TOKENS
    )
    active = messages[2:]
    transcript_messages = _read_transcript_messages(transcript_path)
    preserved_submit_attempts = _recent_standalone_submit_attempts(
        transcript_messages
    )
    active_has_submit_attempt = any(
        message.get("role") == "user"
        and CHECKPOINT_SUBMIT_ATTEMPT_MARKER in str(message.get("content") or "")
        for message in active
    )
    needs_submit_attempt_upgrade = (
        bool(preserved_submit_attempts)
        and any(
            message.get("role") == "user"
            and str(message.get("content") or "").startswith(CHECKPOINT_MARKER)
            for message in active
        )
        and not active_has_submit_attempt
    )
    needs_prepared_state_upgrade = durable_prepared_state != "unused" and any(
        "Typed action available: submit_prepared_workflow"
        in str(message.get("content") or "")
        for message in active
        if message.get("role") == "user"
    )
    checkpoint_only = (
        len(active) == 1
        and active[0].get("role") == "user"
        and str(active[0].get("content") or "").startswith(CHECKPOINT_MARKER)
    )
    if (
        checkpoint_only
        and not needs_submit_attempt_upgrade
        and not needs_prepared_state_upgrade
        and _message_token_upper_bound(messages) < checkpoint_tokens
    ):
        return messages
    if active_tokens_upper_bound is None:
        active_tokens_upper_bound = _message_token_upper_bound(messages)
    elif checkpoint_only:
        active_tokens_upper_bound = max(
            active_tokens_upper_bound, _message_token_upper_bound(messages)
        )
    if (
        active_tokens_upper_bound < checkpoint_tokens
        and not needs_submit_attempt_upgrade
        and not needs_prepared_state_upgrade
    ):
        return messages
    checkpoint_source = (
        messages[:2] + transcript_messages
        if checkpoint_only
        else messages
    )
    compacted, checkpoint = _context_checkpoint(
        checkpoint_source,
        max_recent_chars=MAX_CHECKPOINT_RECENT_CHARACTERS,
        workspace_status=workspace_status,
        preserved_submit_attempts=preserved_submit_attempts,
        preserved_run_identifiers=_collect_run_identifiers(transcript_messages),
        durable_prepared_state=durable_prepared_state,
    )
    with transcript_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(checkpoint, sort_keys=True) + "\n")
    return compacted


def _write_recovery_checkpoint(
    messages: list[dict[str, Any]],
    transcript_path: Path,
    *,
    workspace_status: str,
    reason: str,
    durable_prepared_state: str = "unused",
) -> list[dict[str, Any]]:
    active = messages[2:]
    transcript_messages = _read_transcript_messages(transcript_path)
    checkpoint_only = (
        len(active) == 1
        and active[0].get("role") == "user"
        and str(active[0].get("content") or "").startswith(CHECKPOINT_MARKER)
    )
    needs_prepared_state_upgrade = durable_prepared_state != "unused" and any(
        "Typed action available: submit_prepared_workflow"
        in str(message.get("content") or "")
        for message in active
        if message.get("role") == "user"
    )
    if (
        checkpoint_only
        and not needs_prepared_state_upgrade
        and _message_token_upper_bound(messages)
        < EFFECTIVE_CONTEXT_CHECKPOINT_PROMPT_TOKENS
    ):
        # The new recovery reason is already durable in request telemetry.
        # Recompacting a checkpoint-only history would erase its safe suffix.
        return messages
    checkpoint_source = (
        messages[:2] + transcript_messages
        if checkpoint_only
        else messages
    )
    compacted, checkpoint = _context_checkpoint(
        checkpoint_source,
        max_recent_chars=MAX_CHECKPOINT_RECENT_CHARACTERS,
        workspace_status=workspace_status,
        recovery_reason=reason,
        preserved_submit_attempts=_recent_standalone_submit_attempts(
            transcript_messages
        ),
        preserved_run_identifiers=_collect_run_identifiers(transcript_messages),
        durable_prepared_state=durable_prepared_state,
    )
    with transcript_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(checkpoint, sort_keys=True) + "\n")
    return compacted


def _message_token_upper_bound(messages: list[dict[str, Any]]) -> int:
    # json.dumps escapes non-ASCII by default. One token per serialized ASCII
    # character is deliberately conservative for crash-resume context safety.
    return len(json.dumps(messages, sort_keys=True, separators=(",", ":")))


def _request_active_token_estimate(
    telemetry: RequestTelemetry,
    appended_messages: list[dict[str, Any]],
) -> int | None:
    if telemetry.total_tokens is not None:
        base = telemetry.total_tokens
    elif telemetry.prompt_tokens is not None:
        completion = telemetry.completion_tokens
        if completion is None:
            completion = telemetry.observed_characters_lower_bound
        base = telemetry.prompt_tokens + completion
    else:
        return None
    return base + _message_token_upper_bound(appended_messages)


def _workspace_preflight(
    workspace: Path,
    expected_commit: str,
    *,
    require_clean: bool,
    allow_descendant: bool = False,
) -> str:
    if not workspace.is_dir():
        raise ValueError(f"workspace is missing: {workspace}")
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=workspace, text=True
    ).strip()
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=workspace, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=workspace, text=True
    )
    expected_matches = head == expected_commit
    if allow_descendant and not expected_matches:
        expected_matches = (
            subprocess.run(
                ["git", "merge-base", "--is-ancestor", expected_commit, head],
                cwd=workspace,
                check=False,
            ).returncode
            == 0
        )
    if not expected_matches or branch or (require_clean and status):
        raise ValueError(
            "trial workspace must be detached at the recorded origin/main commit"
            + (" or a descendant commit" if allow_descendant else "")
            + (" and clean" if require_clean else "")
        )
    return status


def _prepared_action_settings(config: dict[str, Any]) -> tuple[Path | None, str]:
    value = config.get("prepared_action")
    if value in (None, {}):
        return None, ""
    if not isinstance(value, dict) or set(value) != {"receipt", "intervention_reason"}:
        raise ValueError("prepared_action must contain receipt and intervention_reason")
    receipt = Path(str(value["receipt"])).resolve()
    reason = str(value["intervention_reason"] or "").strip()
    if reason != "typed_prepared_workflow_action":
        raise ValueError("prepared_action intervention_reason is invalid")
    return receipt, reason


def _record_tool_schema_intervention(
    *,
    evidence: Path,
    request_index: int,
    existing_hash: str,
    current_hash: str,
    receipt_path: Path,
) -> None:
    interventions = evidence / "interventions.jsonl"
    if interventions.exists():
        for line in interventions.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if (
                record.get("classification") == "typed_prepared_workflow_action"
                and record.get("new_tool_schema_sha256") == current_hash
            ):
                return
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    _append_private_jsonl(
        interventions,
        {
            "schema": "npa.sim2real.model_agent_benchmark.intervention.v1",
            "classification": "typed_prepared_workflow_action",
            "attribution": "benchmark_intervention",
            "request_index": request_index,
            "prior_tool_schema_sha256": existing_hash,
            "new_tool_schema_sha256": current_hash,
            "receipt_sha256": receipt.get("receipt_sha256"),
            "at": _utc(),
        },
    )


def _inject_prepared_action_checkpoint(
    messages: list[dict[str, Any]],
    transcript_path: Path,
    *,
    receipt: dict[str, Any],
    durable_prepared_state: str = "unused",
) -> list[dict[str, Any]]:
    action_id = str(receipt["action_id"])
    consumed = durable_prepared_state != "unused" or _prepared_action_consumed_state(
        messages[2:], action_id
    )
    expected_availability = "Typed action available: none" if consumed else "Typed action available: submit_prepared_workflow"
    latest_marker_index = -1
    latest_attempt_index = -1
    for index, message in enumerate(messages):
        content = str(message.get("content") or "")
        if message.get("role") == "user" and content.startswith(
            PREPARED_ACTION_MARKER
        ):
            match = re.search(r"Action ID: ([A-Za-z0-9._-]+)", content)
            marker_action_id = match.group(1).rstrip(".") if match else ""
            if marker_action_id == action_id and expected_availability in content:
                latest_marker_index = index
        if message.get("role") != "tool":
            continue
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            continue
        result_action_id = str(result.get("action_id") or "").rstrip(".")
        if (
            result.get("schema")
            == "npa.sim2real.prepared_workflow_action.result.v1"
            and result_action_id == action_id
        ):
            latest_attempt_index = index
    if latest_marker_index >= 0 and latest_marker_index > latest_attempt_index:
        return messages
    submitted, run_ids = _submitted_workflow_state(messages[2:])
    checkpoint = {
        "role": "user",
        "content": (
            f"{PREPARED_ACTION_MARKER}\n"
            "Completed preflights (receipt-bound): "
            + ", ".join(item["name"] for item in receipt["preflights"])
            + ".\n"
            + (
                "Current blocker: none; the workflow has a durable submission.\n"
                if submitted
                else "Current blocker: the prepared workflow has not crossed its real submission boundary.\n"
            )
            + "Durable submitted state: "
            + json.dumps(
                {"submitted": submitted, "run_id_count": len(run_ids)},
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            + (
                "Typed action available: none; the prepared action is already consumed."
                if consumed
                else f"Typed action available: submit_prepared_workflow. Action ID: {action_id}. "
                "Invoke it directly with only this action ID. Do not reconstruct or repeat "
                "the private command, project, image, input, EULA, resume, or secret settings."
            )
        ),
    }
    messages.append(checkpoint)
    with transcript_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(checkpoint, sort_keys=True) + "\n")
    return messages


def _require_descendant(path: Path, parent: Path, label: str) -> None:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} must be beneath the private benchmark root") from exc


def run(config_path: Path) -> int:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    workspace = Path(config["workspace"]).resolve()
    evidence = Path(config["evidence_dir"]).resolve()
    private_root = Path(config["private_root"]).resolve()
    controller_repo = Path(config["controller_repo_root"]).resolve()
    _require_descendant(workspace, private_root, "workspace")
    _require_descendant(evidence, private_root, "evidence directory")
    prepared_receipt_path, _intervention_reason = _prepared_action_settings(config)
    active_tools = _active_tools(prepared_receipt_path)
    prepared_control_dir = (
        prepared_receipt_path.parent if prepared_receipt_path is not None else None
    )
    if prepared_receipt_path is not None:
        _require_descendant(
            prepared_receipt_path, private_root, "prepared action receipt"
        )
    evidence.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(evidence, 0o700)
    meta_path = evidence / "run.json"
    is_resume = meta_path.exists()
    workspace_status = _workspace_preflight(
        workspace,
        str(config["origin_main_commit"]),
        require_clean=not is_resume,
        allow_descendant=is_resume,
    )
    system_prompt_path = Path(config["system_prompt_file"]).resolve()
    system = system_prompt_path.read_text(encoding="utf-8")
    task_text = str(config.get("task_text") or TASK_TEXT).strip()
    completion_mode = str(config.get("completion_mode") or "strict_grasp").strip()
    if completion_mode not in {"strict_grasp", "workflow_terminal"}:
        raise ValueError("completion_mode must be strict_grasp or workflow_terminal")
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": task_text},
    ]
    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in (config.get("environment") or {}).items()})
    endpoint = str(config["endpoint"])
    api_key_env = str(config.get("api_key_env") or "").strip()
    if api_key_env:
        api_key = str(os.environ.get(api_key_env) or "").strip()
        if not api_key:
            raise ValueError(
                "configured API key environment variable is unset: " + api_key_env
            )
    else:
        api_key = str(config.get("api_key") or "benchmark-local")
    transcript_path = evidence / "transcript.jsonl"
    telemetry_path = evidence / "requests.jsonl"
    tool_results_path = evidence / "tool-results.jsonl"
    try:
        messages.extend(
            _load_transcript(
                transcript_path,
                tool_results_path,
                (
                    prepared_control_dir / "prepared-action-state.jsonl"
                    if prepared_control_dir is not None
                    else None
                ),
            )
        )
    except IndeterminateToolExecutionError as exc:
        failure = {
            "schema": "npa.sim2real.model_agent_benchmark.failure.v2",
            "classification": "indeterminate_tool_execution_after_restart",
            "response_id": exc.response_id,
            "tool_call_ids": exc.tool_call_ids,
            "recovery_action": "terminate_without_reexecuting_possible_side_effect",
            "completed_at": _utc(),
        }
        (evidence / "failure.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True), encoding="utf-8"
        )
        return 3
    request_index = _last_request_index(telemetry_path)
    context_limit = int(config["serving"]["context_limit"])
    stream_safeguards = {
        **DEFAULT_STREAM_SAFEGUARDS,
        **(config.get("stream_safeguards") or {}),
    }
    unknown_safeguards = set(stream_safeguards) - set(DEFAULT_STREAM_SAFEGUARDS)
    if unknown_safeguards:
        raise ValueError(
            f"unknown stream safeguards: {sorted(unknown_safeguards)}"
        )
    for key, value in stream_safeguards.items():
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"stream safeguard {key} must be positive")
    if not isinstance(stream_safeguards["max_identical_malformed_responses"], int):
        raise ValueError("max_identical_malformed_responses must be an integer")
    stream_safeguards["max_identical_malformed_responses"] = int(
        stream_safeguards["max_identical_malformed_responses"]
    )
    meta = {
        "schema": "npa.sim2real.model_agent_benchmark.run.v1",
        "model": config["model"],
        "revision": config["revision"],
        "origin_main_commit": config["origin_main_commit"],
        "seed": config["seed"],
        "system_prompt_sha256": hashlib.sha256(system.encode()).hexdigest(),
        "task_sha256": hashlib.sha256(task_text.encode()).hexdigest(),
        "tool_schema_sha256": _sha(active_tools),
        "completion_mode": completion_mode,
        "serving": config["serving"],
        "stream_safeguards": stream_safeguards,
        "started_at": _utc(),
    }
    if meta_path.exists():
        existing_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        for key in (
            "model",
            "revision",
            "origin_main_commit",
            "seed",
            "system_prompt_sha256",
            "task_sha256",
            "tool_schema_sha256",
            "completion_mode",
            "serving",
            "stream_safeguards",
        ):
            if existing_meta.get(key) == meta.get(key):
                continue
            if (
                key == "tool_schema_sha256"
                and prepared_receipt_path is not None
                and existing_meta.get(key) == _sha(BASE_TOOLS)
                and meta.get(key) == _sha(active_tools)
            ):
                _record_tool_schema_intervention(
                    evidence=evidence,
                    request_index=request_index,
                    existing_hash=str(existing_meta[key]),
                    current_hash=str(meta[key]),
                    receipt_path=prepared_receipt_path,
                )
                continue
            if key == "tool_schema_sha256" and prepared_receipt_path is not None:
                interventions_path = evidence / "interventions.jsonl"
                if interventions_path.exists() and any(
                    json.loads(line).get("new_tool_schema_sha256") == meta.get(key)
                    for line in interventions_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ):
                    continue
            raise ValueError(f"resume metadata mismatch: {key}")
        meta = existing_meta
    else:
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if is_resume:
        with (evidence / "resumes.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "at": _utc(),
                        "request_index": request_index,
                        "workspace_status_sha256": hashlib.sha256(
                            workspace_status.encode()
                        ).hexdigest(),
                        "workspace_status_lines": len(workspace_status.splitlines()),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    isolation = {
        "evidence": evidence,
        "private_root": private_root,
        "controller_repo": controller_repo,
    }
    env["NPA_PRIVATE_EVIDENCE"] = "/tmp/npa-private-evidence"
    env["HOME"] = "/tmp/npa-private-evidence/home"
    prepared_context = PreparedActionContext(
        workspace=workspace,
        evidence=evidence,
        control_dir=prepared_control_dir or evidence,
        private_root=private_root,
        environment=env,
        isolation=isolation,
    )
    isolation_check = _run_tool(
        "run_command", {"command": "true"}, workspace, env, isolation
    )
    if isolation_check["exit_code"] != 0:
        raise RuntimeError(
            "trial mount-namespace isolation preflight failed: "
            + isolation_check["stderr"].strip()
        )
    def current_prepared_state() -> str:
        if prepared_control_dir is None or prepared_receipt_path is None:
            return "unused"
        return _prior_execution_state(
            prepared_control_dir / "prepared-action-state.jsonl"
        )

    if prepared_receipt_path is not None:
        try:
            receipt = validate_receipt(
                prepared_receipt_path,
                requested_action_id=str(
                    json.loads(
                        prepared_receipt_path.read_text(encoding="utf-8")
                    ).get("action_id")
                    or ""
                ),
                context=prepared_context,
                require_secrets=False,
            )
        except PreparedActionError as exc:
            raise ValueError(
                f"prepared action preflight failed: {exc.classification}"
            ) from exc
        durable_prepared_state = current_prepared_state()
        messages = _inject_prepared_action_checkpoint(
            messages,
            transcript_path,
            receipt=receipt,
            durable_prepared_state=durable_prepared_state,
        )
    messages = _maybe_checkpoint(
        messages,
        transcript_path,
        context_limit=context_limit,
        workspace_status=workspace_status,
        durable_prepared_state=current_prepared_state(),
    )
    malformed_fingerprint, identical_malformed_count = _load_malformation_streak(
        telemetry_path
    )
    consecutive_stream_failures = 0
    while True:
        request_index += 1
        request_started = time.monotonic()
        payload = {
            "model": config["served_model_name"],
            "messages": [
                {
                    key: value
                    for key, value in message.items()
                    if not key.startswith("_npa_")
                }
                for message in messages
            ],
            "tools": active_tools,
            "tool_choice": "auto",
            "temperature": 0,
            "seed": config["seed"],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        recovery: dict[str, Any] | None = None
        validated_calls: list[tuple[dict[str, Any], dict[str, Any]]] = []
        try:
            assistant, telemetry = _stream_chat(
                endpoint,
                api_key,
                payload,
                safeguards=stream_safeguards,
            )
        except StreamRecoveryError as exc:
            recovery = {
                **exc.telemetry,
                "reason": exc.reason,
                "response_shape": {
                    "has_content": bool(exc.telemetry.get("has_content")),
                    "has_reasoning": bool(exc.telemetry.get("has_reasoning")),
                    "tool_call_fragments_observed": bool(
                        exc.telemetry.get("tool_call_fragments_observed")
                    ),
                    "tool_call_indexes_observed": exc.telemetry.get(
                        "tool_call_indexes_observed"
                    ),
                },
            }
        except EmptyStreamError as exc:
            recovery = {
                **exc.telemetry,
                "response_shape": {
                    "has_content": False,
                    "has_reasoning": False,
                    "tool_call_fragments_observed": False,
                    "tool_call_indexes_observed": [],
                },
            }
        except _TRANSIENT_TRANSPORT_ERRORS as exc:
            if _is_permanent_model_http_error(exc):
                record = _transport_telemetry_record(
                    exc,
                    request_index=request_index,
                    elapsed_seconds=time.monotonic() - request_started,
                    recovery_action="terminate_permanent_model_endpoint_http_error",
                )
                with telemetry_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                failure = {
                    "schema": "npa.sim2real.model_agent_benchmark.failure.v2",
                    "classification": "permanent_model_endpoint_http_error",
                    "request_index": request_index,
                    "http_status": exc.code,
                    "workflow_submitted": _submitted_workflow_state(messages[2:])[0],
                    "workflow_run_identifiers": _submitted_workflow_state(
                        messages[2:]
                    )[1],
                    "completed_at": _utc(),
                }
                (evidence / "failure.json").write_text(
                    json.dumps(failure, indent=2, sort_keys=True), encoding="utf-8"
                )
                return 3
            consecutive_stream_failures += 1
            recovery_action = (
                "discard_incomplete_response_rebuild_context_and_retry"
                if consecutive_stream_failures == 1
                else "retry_from_last_safe_checkpoint"
            )
            with telemetry_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        _transport_telemetry_record(
                            exc,
                            request_index=request_index,
                            elapsed_seconds=time.monotonic() - request_started,
                            recovery_action=recovery_action,
                        ),
                        sort_keys=True,
                    )
                    + "\n"
                )
            if consecutive_stream_failures == 1:
                current_workspace_status = subprocess.check_output(
                    ["git", "status", "--porcelain"], cwd=workspace, text=True
                )
                messages = _write_recovery_checkpoint(
                    messages,
                    transcript_path,
                    workspace_status=current_workspace_status,
                    reason=f"transport_error:{type(exc).__name__}",
                    durable_prepared_state=current_prepared_state(),
                )
            time.sleep(min(30.0, float(2 ** min(consecutive_stream_failures, 5))))
            continue
        if recovery is None:
            calls = assistant.get("tool_calls") or []
            if calls:
                try:
                    validated_calls = _validated_tool_calls(
                        assistant, finish_reason=telemetry.finish_reason
                    )
                except ValueError as exc:
                    recovery = {
                        **asdict(telemetry),
                        "elapsed_seconds": telemetry.latency_seconds,
                        "reason": str(exc),
                        "response_shape": _response_shape(assistant),
                    }
            elif (
                not assistant.get("content")
                or telemetry.finish_reason in {"length", "content_filter"}
            ):
                recovery = {
                    **asdict(telemetry),
                    "elapsed_seconds": telemetry.latency_seconds,
                    "reason": (
                        "reasoning_only_response_without_tool_call"
                        if assistant.get("reasoning_content")
                        else "response_without_usable_content_or_tool_call"
                    ),
                    "response_shape": _response_shape(assistant),
                }
        if recovery is not None:
            response_shape = dict(recovery.pop("response_shape"))
            reason = str(recovery.get("reason") or "malformed_model_response")
            fingerprint = _malformation_fingerprint(reason, response_shape)
            malformed_fingerprint, identical_malformed_count = (
                _next_malformation_streak(
                    malformed_fingerprint,
                    identical_malformed_count,
                    fingerprint,
                )
            )
            action = _malformation_recovery_action(
                identical_malformed_count,
                int(stream_safeguards["max_identical_malformed_responses"]),
            )
            terminal = action == "terminate_repeated_identical_malformed_response"
            record = _malformation_telemetry_record(
                recovery,
                request_index=request_index,
                response_shape=response_shape,
                fingerprint=fingerprint,
                identical_count=identical_malformed_count,
                action=action,
            )
            with telemetry_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            if terminal:
                workflow_submitted, submitted_run_ids = _submitted_workflow_state(
                    messages[2:]
                )
                failure = _terminal_malformation_failure(
                    request_index=request_index,
                    fingerprint=fingerprint,
                    identical_count=identical_malformed_count,
                    reason=reason,
                    workflow_submitted=workflow_submitted,
                    run_identifiers=submitted_run_ids,
                )
                (evidence / "failure.json").write_text(
                    json.dumps(failure, indent=2, sort_keys=True), encoding="utf-8"
                )
                return 2
            current_workspace_status = subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=workspace, text=True
            )
            messages = _write_recovery_checkpoint(
                messages,
                transcript_path,
                workspace_status=current_workspace_status,
                reason=reason,
                durable_prepared_state=current_prepared_state(),
            )
            time.sleep(min(30.0, float(2**identical_malformed_count)))
            continue
        consecutive_stream_failures = 0
        malformed_fingerprint, identical_malformed_count = None, 0
        telemetry.request_index = request_index
        with telemetry_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {**asdict(telemetry), "classification": "response_completed"},
                    sort_keys=True,
                )
                + "\n"
            )
        if validated_calls:
            terminal_completion: dict[str, Any] | None = None
            tool_messages: list[dict[str, Any]] = []
            assistant_hash = _sha(assistant)
            response_id = f"request-{request_index}-{assistant_hash}"
            assistant["_npa_response_id"] = response_id
            for call, arguments in validated_calls:
                occurrence_id = _sha(
                    {"response_id": response_id, "tool_call_id": call["id"]}
                )
                journal_base = {
                    "schema": "npa.sim2real.tool_execution.v2",
                    "response_id": response_id,
                    "assistant_sha256": assistant_hash,
                    "assistant": assistant,
                    "tool_call_id": call["id"],
                    "tool_name": call["function"]["name"],
                    "occurrence_id": occurrence_id,
                }
                if call["function"]["name"] == "submit_prepared_workflow":
                    journal_base["prepared_action_id"] = arguments["action_id"]
                _append_private_jsonl(
                    tool_results_path,
                    {**journal_base, "at": _utc(), "phase": "intent"},
                )
                tool_name = call["function"]["name"]
                generic_real_submit = (
                    tool_name == "run_command"
                    and _workflow_submit_command_kind(
                        str(arguments.get("command") or "")
                    )
                    == "standalone"
                    and prepared_control_dir is not None
                    and prepared_receipt_path is not None
                )
                submission_lock = None
                try:
                    if generic_real_submit:
                        lock_path = prepared_control_dir / "prepared-action.lock"
                        submission_lock = lock_path.open("a+", encoding="utf-8")
                        os.chmod(lock_path, 0o600)
                        fcntl.flock(submission_lock.fileno(), fcntl.LOCK_EX)
                    durable_prepared_state = (
                        _prior_execution_state(
                            prepared_control_dir / "prepared-action-state.jsonl"
                        )
                        if prepared_control_dir is not None
                        and prepared_receipt_path is not None
                        else "unused"
                    )
                    submission_block_reason = _workflow_submission_block_reason(
                        [*messages, assistant, *tool_messages],
                        tool_name=tool_name,
                        arguments=arguments,
                        durable_prepared_state=durable_prepared_state,
                    )
                    if submission_block_reason:
                        if tool_name == "submit_prepared_workflow":
                            result = rejected_result(
                                action_id=arguments["action_id"],
                                classification="duplicate_submission_prevented",
                                message="one durable workflow submission already exists",
                            )
                        else:
                            result = {
                                "error": submission_block_reason,
                                "message": (
                                    "Workflow submission must be one direct standalone NPA "
                                    "command, and only one successful non-plan submission is "
                                    "allowed. Rerun help, --plan-only, or the real submit with "
                                    "no pipes, redirects, wrappers, shell interpolation, or "
                                    "compound diagnostics; use separate read-only tool calls "
                                    "for output inspection and monitoring."
                                ),
                            }
                    else:
                        if generic_real_submit:
                            _append_private_jsonl(
                                prepared_control_dir
                                / "prepared-action-state.jsonl",
                                {
                                    "schema": "npa.sim2real.prepared_workflow_action.state.v1",
                                    "phase": "execution_started",
                                    "action_id": str(receipt["action_id"]),
                                    "occurrence_id": occurrence_id,
                                    "transition": "generic_workflow_submit",
                                    "at": _utc(),
                                },
                            )
                        try:
                            result = _run_tool(
                                tool_name,
                                arguments,
                                workspace,
                                env,
                                isolation,
                                prepared_receipt_path=prepared_receipt_path,
                                prepared_context=prepared_context,
                                occurrence_id=occurrence_id,
                            )
                        except Exception as exc:
                            result = {"error": type(exc).__name__, "message": str(exc)}
                        if generic_real_submit:
                            _append_private_jsonl(
                                prepared_control_dir
                                / "prepared-action-state.jsonl",
                                {
                                    "schema": "npa.sim2real.prepared_workflow_action.state.v1",
                                    "phase": "execution_finished",
                                    "action_id": str(receipt["action_id"]),
                                    "occurrence_id": occurrence_id,
                                    "transition": "generic_workflow_submit",
                                    "result_sha256": _sha(result),
                                    "at": _utc(),
                                },
                            )
                finally:
                    if submission_lock is not None:
                        submission_lock.close()
                tool_message = {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": _serialize_bounded_tool_result(result),
                }
                _append_private_jsonl(
                    tool_results_path,
                    {
                        **journal_base,
                        "schema": "npa.sim2real.tool_execution.v2",
                        "at": _utc(),
                        "phase": "result",
                        "result": result,
                        "tool_message": tool_message,
                    },
                )
                tool_messages.append(tool_message)
                if (
                    completion_mode == "workflow_terminal"
                    and call["function"]["name"] == "complete_workflow"
                    and result.get("terminal") is True
                ):
                    terminal_completion = result
            messages.extend([assistant, *tool_messages])
            with transcript_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    "".join(
                        json.dumps(message, sort_keys=True) + "\n"
                        for message in (assistant, *tool_messages)
                    )
                )
            _append_private_jsonl(
                tool_results_path,
                {
                    "schema": "npa.sim2real.tool_execution.v2",
                    "response_id": response_id,
                    "assistant": assistant,
                    "at": _utc(),
                    "phase": "transcript_committed",
                },
            )
            if terminal_completion is not None:
                verification = {
                    "schema": "npa.sim2real.model_agent_benchmark.workflow_terminal.v1",
                    "completion_mode": completion_mode,
                    **terminal_completion,
                    "end_to_end_wall_seconds": _elapsed_since(meta["started_at"]),
                    "completed_at": _utc(),
                }
                (evidence / "success.json").write_text(
                    json.dumps(verification, indent=2), encoding="utf-8"
                )
                return 0
            messages = _maybe_checkpoint(
                messages,
                transcript_path,
                context_limit=context_limit,
                workspace_status=subprocess.check_output(
                    ["git", "status", "--porcelain"], cwd=workspace, text=True
                ),
                active_tokens_upper_bound=_request_active_token_estimate(
                    telemetry, tool_messages
                ),
                durable_prepared_state=current_prepared_state(),
            )
            continue

        messages.append(assistant)
        with transcript_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(assistant, sort_keys=True) + "\n")

        artifact_root = workspace / str(
            config.get("artifact_root", "benchmark-artifacts")
        )
        try:
            verification = verify_artifact_tree(
                artifact_root,
                minimum_lift_m=float(config.get("minimum_lift_m", 0.05)),
                minimum_hold_seconds=float(config.get("minimum_hold_seconds", 2.0)),
                rerun_bin=str(config.get("rerun_bin", "rerun")),
            )
        except VerificationError as exc:
            feedback = {
                "role": "user",
                "content": "Independent machine verification has not passed. Continue the same task; do not claim success until this exact verifier passes.\n\n"
                + str(exc),
            }
            messages.append(feedback)
            with transcript_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(feedback, sort_keys=True) + "\n")
            messages = _maybe_checkpoint(
                messages,
                transcript_path,
                context_limit=context_limit,
                workspace_status=subprocess.check_output(
                    ["git", "status", "--porcelain"], cwd=workspace, text=True
                ),
                active_tokens_upper_bound=_request_active_token_estimate(
                    telemetry, [feedback]
                ),
                durable_prepared_state=current_prepared_state(),
            )
            continue
        verification["end_to_end_wall_seconds"] = _elapsed_since(meta["started_at"])
        verification["completed_at"] = _utc()
        (evidence / "success.json").write_text(
            json.dumps(verification, indent=2), encoding="utf-8"
        )
        return 0


def verify_command(args: argparse.Namespace) -> int:
    try:
        result = verify_artifact_tree(
            Path(args.artifact_root),
            minimum_lift_m=args.minimum_lift_m,
            minimum_hold_seconds=args.minimum_hold_seconds,
            rerun_bin=args.rerun_bin,
        )
    except VerificationError as exc:
        print(json.dumps({"passed": False, "error": str(exc)}, indent=2))
        return 1
    print(json.dumps(result, indent=2))
    return 0


def prepare_action_command(args: argparse.Namespace) -> int:
    try:
        receipt = create_receipt_from_request(args.request, args.output)
    except (OSError, ValueError, PreparedActionError) as exc:
        classification = getattr(exc, "classification", type(exc).__name__)
        print(
            json.dumps(
                {"created": False, "classification": classification},
                sort_keys=True,
            )
        )
        return 1
    print(
        json.dumps(
            {
                "created": True,
                "action_id": receipt["action_id"],
                "receipt_sha256": receipt["receipt_sha256"],
                "argv_sha256": receipt["argv_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--config", type=Path, required=True)
    prepare_parser = sub.add_parser("prepare-action")
    prepare_parser.add_argument("--request", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--artifact-root", type=Path, required=True)
    verify_parser.add_argument("--minimum-lift-m", type=float, default=0.05)
    verify_parser.add_argument("--minimum-hold-seconds", type=float, default=2.0)
    verify_parser.add_argument("--rerun-bin", default="rerun")
    args = parser.parse_args()
    if args.command == "run":
        return run(args.config)
    if args.command == "prepare-action":
        return prepare_action_command(args)
    return verify_command(args)


if __name__ == "__main__":
    sys.exit(main())
