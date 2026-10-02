"""Bind a failed workflow operation to its retained original command trace."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shlex

from npa.cluster.absent_evidence import pinned_bytes, require


def _expand(value: str, bindings: dict[str, str]) -> str:
    for key, replacement in bindings.items():
        value = value.replace("${" + key + "}", replacement)
        value = re.sub(r"\$" + re.escape(key) + r"\b", lambda _: replacement, value)
    require(not any(character in value for character in "$`\n"), "Dynamic selector")
    return value


def _selectors(command: str) -> tuple[list[str], dict[str, str]]:
    lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|<>")
    lexer.whitespace_split = True
    tokens = list(lexer)
    bindings = {}
    for token in tokens:
        match = re.fullmatch(
            r"(L|ISO|RUN|K|NPA_CONFIG_DIR|KUBECONFIG)=(.*)", token, re.DOTALL
        )
        if match:
            require(match[1] not in bindings, "Repeated command selector")
            bindings[match[1]] = _expand(match[2], bindings)
    marker = ["-m", "npa.cli.main", "workbench", "workflow", "submit"]
    starts = [i for i in range(len(tokens)) if tokens[i : i + 5] == marker]
    require(len(starts) == 1, "Original trace must contain one supported submit")
    arguments = tokens[starts[0] + 5 :]
    end = next((i for i, value in enumerate(arguments) if value == ">"), None)
    require(end is not None, "Original producer output path is missing")
    require(end + 1 < len(arguments), "Original producer output path is empty")
    bindings["output_path"] = _expand(arguments[end + 1], bindings)
    return arguments[:end], bindings


def _option(arguments: list[str], bindings: dict[str, str], *names: str) -> str:
    indices = [i for i, value in enumerate(arguments) if value in names]
    require(len(indices) == 1, "Original command selector is missing or ambiguous")
    position = indices[0] + 1
    require(position < len(arguments), "Original command selector lacks a value")
    return _expand(arguments[position], bindings)


def _record(manifest: dict) -> dict:
    lines = pinned_bytes(manifest["producer_transcript"]).splitlines()
    line = manifest["producer_line"]
    require(type(line) is int and 1 <= line <= len(lines), "Missing producer line")
    record = json.loads(lines[line - 1])
    require(
        record.get("type") == "tool_call" and record.get("subtype") == "completed",
        "Original completed producer trace required",
    )
    return record


def validate_trace(manifest: dict, journal: dict, response: dict) -> dict:
    """Read original tool bytes without executing the retained shell command.

    Args:
        manifest: Pinned original trace and isolated-root selectors.
        journal: Original failed operation generation.
        response: Original NPA submit response, retained separately.
    Returns:
        Exact root, context, and command source identity.
    Raises:
        ValueError: Required original selectors are absent or inconsistent.
    """
    record = _record(manifest)
    call = record["tool_call"]["shellToolCall"]
    command = call["args"]["command"]
    require(call["result"]["success"]["command"] == command, "Producer trace changed")
    arguments, bindings = _selectors(command)
    require(
        bindings["output_path"] == manifest["producer_response"]["path"],
        "Response is not the original producer output",
    )
    root = _option(arguments, bindings, "--isolated-config-dir")
    require(
        Path(root).is_absolute() and str(Path(root).resolve()) == root, "Invalid root"
    )
    require(root == manifest["isolated_root"], "Original producer root mismatch")
    return _bound_command(arguments, bindings, record, journal, response, root)


def _bound_command(arguments, bindings, record, journal, response, root) -> dict:
    require(
        _option(arguments, bindings, "--project") == journal["project_alias"]
        and _option(arguments, bindings, "--run-id", "--resume-run")
        == journal["requested_name"],
        "Original producing operation does not match",
    )
    context = _option(arguments, bindings, "--infra")
    require(context.startswith("k8s/") and len(context) > 4, "Non-Kubernetes producer")
    require(
        _option(arguments, bindings, "--controller-backend") == "kubernetes"
        and "--pool" not in arguments,
        "Unsupported controller or pool producer",
    )
    observed = datetime.fromtimestamp(record["timestamp_ms"] / 1000, timezone.utc)
    updated = datetime.fromisoformat(journal["updated_at"].replace("Z", "+00:00"))
    require(
        observed.replace(microsecond=0) == updated.replace(microsecond=0),
        "Wrong attempt",
    )
    require(response.get("status") == "failed", "Original failed response required")
    controller = response["launch_transaction"]["controller"]
    require(controller["selected_context"] == context[4:], "Original context changed")
    ledger = Path(bindings["NPA_CONFIG_DIR"]) / "workflow-submissions"
    ledger /= journal["project_alias"]
    ledger /= journal["requested_name"] + ".json"
    return {
        "root": root,
        "context": context[4:],
        "controller": controller["name"],
        "ledger_path": str(ledger),
        "kubeconfig_path": bindings["KUBECONFIG"],
    }
