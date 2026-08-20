"""Standalone OpenAI-compatible tool agent for the Sim2Real benchmark.

This module deliberately has no dependency on ``npa.agent_backend`` or any
``npa cli agent`` surface. It drives one isolated detached-HEAD workspace and
records an append-only transcript suitable for private benchmark evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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


TASK_TEXT = """From a clean checkout on the dev VM, operate the canonical npa/workflows/workbench/npa-workflows/sim2real.yaml through the standard runtime. Validate, plan, submit, monitor, diagnose, and make necessary fixes until the Franka arm grasps the cube, lifts it at least 5 cm, and holds it for 2 seconds. Do not weaken evaluation thresholds, replace real components, or fabricate success. Preserve unrelated changes. Finish with run IDs, commands, code changes, measured success metrics, and artifact locations."""
CHECKPOINT_MARKER = "BENCHMARK_CONTEXT_CHECKPOINT_V1"


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
]


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


class EmptyStreamError(RuntimeError):
    """The server closed a streaming response without a usable event."""


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
    endpoint: str, api_key: str, payload: dict[str, Any]
) -> tuple[dict[str, Any], RequestTelemetry]:
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
    content: list[str] = []
    reasoning: list[str] = []
    tool_calls: dict[int, dict[str, Any]] = {}
    usage: dict[str, Any] = {}
    finish_reason: str | None = None
    with urllib.request.urlopen(request) as response:
        for raw in response:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage"):
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta") or {}
            if first is None and any(
                delta.get(key)
                for key in ("content", "reasoning", "reasoning_content", "tool_calls")
            ):
                first = time.monotonic() - started
            if delta.get("content"):
                content.append(delta["content"])
            if delta.get("reasoning") or delta.get("reasoning_content"):
                reasoning.append(
                    delta.get("reasoning") or delta.get("reasoning_content")
                )
            for call in delta.get("tool_calls") or []:
                index = int(call.get("index", 0))
                current = tool_calls.setdefault(
                    index,
                    {
                        "id": "",
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    },
                )
                current["id"] += call.get("id") or ""
                function = call.get("function") or {}
                current["function"]["name"] += function.get("name") or ""
                current["function"]["arguments"] += function.get("arguments") or ""
            finish_reason = choice.get("finish_reason") or finish_reason
    if not any((content, reasoning, tool_calls, usage, finish_reason, first)):
        raise EmptyStreamError(
            "OpenAI-compatible endpoint returned an empty event stream"
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
    )
    return message, telemetry


def _load_transcript(path: Path) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if not path.exists():
        return messages
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        message = json.loads(line)
        if not isinstance(message, dict) or message.get("role") not in {
            "assistant",
            "tool",
            "user",
        }:
            raise ValueError(f"invalid transcript entry at line {line_number}")
        messages.append(message)
    checkpoint_indexes = [
        index
        for index, message in enumerate(messages)
        if message.get("role") == "user"
        and str(message.get("content") or "").startswith(CHECKPOINT_MARKER)
    ]
    if checkpoint_indexes:
        messages = messages[checkpoint_indexes[-1] :]
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


def _context_checkpoint(
    messages: list[dict[str, Any]], *, max_recent_chars: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    recent_candidates = [
        message
        for message in messages[2:]
        if not (
            message.get("role") == "user"
            and str(message.get("content") or "").startswith(CHECKPOINT_MARKER)
        )
    ]
    recent: list[dict[str, Any]] = []
    used = 0
    for message in reversed(recent_candidates):
        size = len(json.dumps(message, sort_keys=True, separators=(",", ":")))
        if recent and used + size > max_recent_chars:
            break
        recent.append(message)
        used += size
    recent.reverse()
    checkpoint = {
        "role": "user",
        "content": (
            f"{CHECKPOINT_MARKER}\n"
            "The standalone benchmark controller deterministically compacted "
            "earlier context because the configured model context was nearly "
            "full. The full append-only transcript remains private evidence. "
            "Do not infer success from this checkpoint. Re-read durable workspace "
            "and runtime state with tools as needed, then continue the original "
            "task.\n"
            f"Prior active-context SHA256: {_sha(messages[2:])}\n"
            f"Prior messages: {len(messages) - 2}; verbatim recent messages: "
            f"{len(recent)}\n"
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
) -> list[dict[str, Any]]:
    if prompt_tokens is None or prompt_tokens < int(context_limit * 0.85):
        return messages
    compacted, checkpoint = _context_checkpoint(
        messages, max_recent_chars=max(16_384, int(context_limit * 1.5))
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
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": TASK_TEXT},
    ]
    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in (config.get("environment") or {}).items()})
    endpoint = str(config["endpoint"])
    api_key = str(config.get("api_key") or "benchmark-local")
    transcript_path = evidence / "transcript.jsonl"
    telemetry_path = evidence / "requests.jsonl"
    messages.extend(_load_transcript(transcript_path))
    request_index = _last_request_index(telemetry_path)
    context_limit = int(config["serving"]["context_limit"])
    meta = {
        "schema": "npa.sim2real.model_agent_benchmark.run.v1",
        "model": config["model"],
        "revision": config["revision"],
        "origin_main_commit": config["origin_main_commit"],
        "seed": config["seed"],
        "system_prompt_sha256": hashlib.sha256(system.encode()).hexdigest(),
        "task_sha256": hashlib.sha256(TASK_TEXT.encode()).hexdigest(),
        "tool_schema_sha256": _sha(TOOLS),
        "serving": config["serving"],
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
            "serving",
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
    )
    consecutive_stream_failures = 0
    while True:
        request_index += 1
        payload = {
            "model": config["served_model_name"],
            "messages": messages,
            "tools": TOOLS,
            "tool_choice": "auto",
            "temperature": 0,
            "seed": config["seed"],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        try:
            assistant, telemetry = _stream_chat(endpoint, api_key, payload)
        except (
            urllib.error.URLError,
            json.JSONDecodeError,
            EmptyStreamError,
        ) as exc:
            with telemetry_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "request_index": request_index,
                            "transport_error": repr(exc),
                            "at": _utc(),
                        }
                    )
                    + "\n"
                )
            consecutive_stream_failures += 1
            time.sleep(min(30.0, float(2 ** min(consecutive_stream_failures, 5))))
            continue
        consecutive_stream_failures = 0
        telemetry.request_index = request_index
        with telemetry_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(telemetry), sort_keys=True) + "\n")
        messages.append(assistant)
        with transcript_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(assistant, sort_keys=True) + "\n")
        calls = assistant.get("tool_calls") or []
        if calls:
            for call in calls:
                try:
                    arguments = json.loads(call["function"]["arguments"] or "{}")
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
                    "content": json.dumps(result),
                }
                messages.append(tool_message)
                with transcript_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(tool_message, sort_keys=True) + "\n")
            messages = _maybe_checkpoint(
                messages,
                transcript_path,
                prompt_tokens=telemetry.prompt_tokens,
                context_limit=context_limit,
            )
            continue

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
