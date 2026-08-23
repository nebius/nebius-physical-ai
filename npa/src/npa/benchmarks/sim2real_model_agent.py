"""Standalone OpenAI-compatible tool agent for the Sim2Real benchmark.

This module deliberately has no dependency on ``npa.agent_backend`` or any
``npa cli agent`` surface. It drives one isolated detached-HEAD workspace and
records an append-only transcript suitable for private benchmark evidence.
"""

from __future__ import annotations

import argparse
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

from npa.benchmarks.sim2real_success import VerificationError, verify_artifact_tree


TASK_TEXT = """From a clean checkout on the dev VM, first validate and plan the canonical public-franka-lift preset for npa/workflows/workbench/npa-workflows/sim2real.yaml, submit it through the standard runtime, and monitor that run to terminal completion. Diagnose and make necessary fixes if it fails. Do not weaken the canonical workflow, its real components, or the strict requirement that the Franka arm grasp the cube, lift it at least 5 cm, and hold it for 2 seconds. Preserve unrelated changes. Finish with the run ID, commands, code changes, measured stage and grasp metrics, and artifact locations."""
CHECKPOINT_MARKER = "BENCHMARK_CONTEXT_CHECKPOINT_V1"
RECOVERY_MARKER = "BENCHMARK_MALFORMED_RESPONSE_RECOVERY_V1"
MAX_CONTEXT_TOOL_RESULT_CHARACTERS = 4_096

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


TOOLS: list[dict[str, Any]] = [
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
) -> dict[str, Any]:
    started = time.monotonic()
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


def _load_transcript(
    path: Path, tool_results_path: Path | None = None
) -> list[dict[str, Any]]:
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
            {"assistant": record.get("assistant"), "intents": set(), "results": {}},
        )
        phase = record.get("phase")
        call_id = str(record.get("tool_call_id") or "")
        if phase == "intent" and call_id:
            group["intents"].add(call_id)
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
        unresolved = sorted(group["intents"] - set(group["results"]))
        if unresolved:
            raise IndeterminateToolExecutionError(response_id, unresolved)
        tool_messages: list[dict[str, Any]] = []
        for call in assistant.get("tool_calls") or []:
            call_id = str(call.get("id") or "") if isinstance(call, dict) else ""
            has_result = call_id in group["results"]
            tool_message = group["results"].get(call_id)
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


def _latest_prompt_tokens(path: Path) -> int | None:
    latest: int | None = None
    if not path.exists():
        return latest
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line).get("prompt_tokens")
        if isinstance(value, int):
            latest = value
    return latest


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
        elif key in {"run_id", "workflow_run_id"} and isinstance(value, str):
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

    def collect(value: Any, key: str | None = None) -> None:
        if isinstance(value, dict):
            for child_key, child in value.items():
                collect(child, str(child_key))
        elif isinstance(value, list):
            for child in value:
                collect(child, key)
        elif isinstance(value, str):
            if key in {"run_id", "workflow_run_id"} and _RUN_ID_RE.fullmatch(value):
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
            if not isinstance(function, dict) or function.get("name") != "run_command":
                continue
            try:
                arguments = json.loads(str(function.get("arguments") or ""))
                command = shlex.split(str(arguments.get("command") or ""))
            except (json.JSONDecodeError, ValueError):
                continue
            is_submit = any(
                command[index : index + 3] == ["workbench", "workflow", "submit"]
                for index in range(max(0, len(command) - 2))
            )
            if not is_submit or "--plan-only" in command:
                continue
            result_message = tool_results.get(str(call.get("id") or ""))
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


def _context_checkpoint(
    messages: list[dict[str, Any]],
    *,
    max_recent_chars: int,
    workspace_status: str = "",
    recovery_reason: str | None = None,
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
    run_ids = _collect_run_identifiers(messages[2:])
    workspace_lines = workspace_status.splitlines()
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
            +
            "The standalone benchmark controller deterministically compacted "
            "earlier complete message groups at a safe boundary. The full "
            "append-only transcript remains private evidence. "
            "Do not infer success from this checkpoint. Re-read durable workspace "
            "and runtime state with tools as needed, then continue the original "
            "task.\n"
            f"Prior active-context SHA256: {_sha(messages[2:])}\n"
            f"Prior messages: {len(messages) - 2}; verbatim recent messages: "
            f"{len(recent)}\n"
            f"Durable workflow run identifiers: {json.dumps(run_ids)}\n"
            f"Workspace status SHA256: {hashlib.sha256(workspace_status.encode()).hexdigest()}; "
            f"status lines: {len(workspace_lines)}\n"
            "Workspace status follows:\n"
            + "\n".join(workspace_lines[:200])
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
    prompt_tokens: int | None,
    context_limit: int,
    workspace_status: str = "",
) -> list[dict[str, Any]]:
    if prompt_tokens is None or prompt_tokens < int(context_limit * 0.85):
        return messages
    compacted, checkpoint = _context_checkpoint(
        messages,
        max_recent_chars=max(16_384, int(context_limit * 1.5)),
        workspace_status=workspace_status,
    )
    with transcript_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(checkpoint, sort_keys=True) + "\n")
    return compacted


def _workspace_preflight(
    workspace: Path, expected_commit: str, *, require_clean: bool
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
    if head != expected_commit or branch or (require_clean and status):
        raise ValueError(
            "trial workspace must be detached at the recorded origin/main commit"
            + (" and clean" if require_clean else "")
        )
    return status


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
    evidence.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(evidence, 0o700)
    meta_path = evidence / "run.json"
    is_resume = meta_path.exists()
    workspace_status = _workspace_preflight(
        workspace,
        str(config["origin_main_commit"]),
        require_clean=not is_resume,
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
    api_key = str(config.get("api_key") or "benchmark-local")
    transcript_path = evidence / "transcript.jsonl"
    telemetry_path = evidence / "requests.jsonl"
    tool_results_path = evidence / "tool-results.jsonl"
    try:
        messages.extend(_load_transcript(transcript_path, tool_results_path))
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
        "tool_schema_sha256": _sha(TOOLS),
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
            if existing_meta.get(key) != meta.get(key):
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
    isolation_check = _run_tool(
        "run_command", {"command": "true"}, workspace, env, isolation
    )
    if isolation_check["exit_code"] != 0:
        raise RuntimeError(
            "trial mount-namespace isolation preflight failed: "
            + isolation_check["stderr"].strip()
        )
    messages = _maybe_checkpoint(
        messages,
        transcript_path,
        prompt_tokens=_latest_prompt_tokens(telemetry_path),
        context_limit=context_limit,
        workspace_status=workspace_status,
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
            "tools": TOOLS,
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
                messages, checkpoint = _context_checkpoint(
                    messages,
                    max_recent_chars=16_384,
                    workspace_status=current_workspace_status,
                    recovery_reason=f"transport_error:{type(exc).__name__}",
                )
                with transcript_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(checkpoint, sort_keys=True) + "\n")
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
            messages, checkpoint = _context_checkpoint(
                messages,
                max_recent_chars=max(16_384, int(context_limit * 1.5)),
                workspace_status=current_workspace_status,
                recovery_reason=reason,
            )
            with transcript_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(checkpoint, sort_keys=True) + "\n")
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
                journal_base = {
                    "schema": "npa.sim2real.tool_execution.v2",
                    "response_id": response_id,
                    "assistant_sha256": assistant_hash,
                    "assistant": assistant,
                    "tool_call_id": call["id"],
                    "tool_name": call["function"]["name"],
                }
                with tool_results_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {**journal_base, "at": _utc(), "phase": "intent"},
                            sort_keys=True,
                        )
                        + "\n"
                    )
                try:
                    result = _run_tool(
                        call["function"]["name"],
                        arguments,
                        workspace,
                        env,
                        isolation,
                    )
                except (
                    Exception
                ) as exc:  # tool errors are observations, not controller failures
                    result = {"error": type(exc).__name__, "message": str(exc)}
                tool_message = {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": _serialize_bounded_tool_result(result),
                }
                with tool_results_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                **journal_base,
                                "schema": "npa.sim2real.tool_execution.v2",
                                "at": _utc(),
                                "phase": "result",
                                "result": result,
                                "tool_message": tool_message,
                            },
                            sort_keys=True,
                        )
                        + "\n"
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
            with tool_results_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "schema": "npa.sim2real.tool_execution.v2",
                            "response_id": response_id,
                            "assistant": assistant,
                            "at": _utc(),
                            "phase": "transcript_committed",
                        },
                        sort_keys=True,
                    )
                    + "\n"
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
                prompt_tokens=telemetry.prompt_tokens,
                context_limit=context_limit,
                workspace_status=subprocess.check_output(
                    ["git", "status", "--porcelain"], cwd=workspace, text=True
                ),
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
                prompt_tokens=telemetry.prompt_tokens,
                context_limit=context_limit,
                workspace_status=subprocess.check_output(
                    ["git", "status", "--porcelain"], cwd=workspace, text=True
                ),
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--config", type=Path, required=True)
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--artifact-root", type=Path, required=True)
    verify_parser.add_argument("--minimum-lift-m", type=float, default=0.05)
    verify_parser.add_argument("--minimum-hold-seconds", type=float, default=2.0)
    verify_parser.add_argument("--rerun-bin", default="rerun")
    args = parser.parse_args()
    if args.command == "run":
        return run(args.config)
    return verify_command(args)


if __name__ == "__main__":
    sys.exit(main())
