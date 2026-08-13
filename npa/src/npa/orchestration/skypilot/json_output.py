"""Strict parsing for JSON emitted after harmless SkyPilot CLI diagnostics."""

from __future__ import annotations

import re
import subprocess
import json
from typing import Any

from npa.clients.json_output import parse_single_json_document


_EMPTY_QUEUE_MESSAGES = {
    "no in-progress managed jobs.",
    "no in-progress managed jobs found.",
    "no in-progress managed jobs found",
    "sky.exceptions.clusternotuperror: no in-progress managed jobs.",
}


def queue_rows_from_output(output: str) -> list[dict[str, Any]] | None:
    """Parse a verified SkyPilot queue list from one unambiguous JSON payload."""

    payload = parse_single_json_document(output)
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict) and isinstance(payload.get("jobs"), list):
        rows = payload["jobs"]
    else:
        return None
    if not all(isinstance(row, dict) for row in rows):
        return None
    return rows


def is_verified_empty_queue_result(
    result: subprocess.CompletedProcess[str],
) -> bool:
    """Recognize only SkyPilot 0.12.2's complete benign empty-queue result."""

    rows = verified_structured_queue_rows(result)
    if rows is not None:
        return not rows
    if result.returncode not in {0, 1}:
        return False
    stdout_lines = _semantic_queue_lines(result.stdout, structured=False)
    stderr_lines = _semantic_queue_lines(result.stderr, structured=False)
    if stdout_lines is None or stderr_lines is None:
        return False
    # SkyPilot can echo the same benign marker on both streams. Evidence is the
    # semantic marker, not the number of times its presentation layer printed it.
    markers = {
        line
        for line in (*stdout_lines, *stderr_lines)
        if line in _EMPTY_QUEUE_MESSAGES
    }
    return bool(markers)


def verified_structured_queue_rows(
    result: subprocess.CompletedProcess[str],
) -> list[dict[str, Any]] | None:
    """Return authoritative structured rows with only non-contradictory diagnostics."""

    if result.returncode != 0:
        return None
    rows = queue_rows_from_output(result.stdout)
    if rows is None:
        return None
    stdout_without_json = _without_single_json_document(result.stdout)
    if stdout_without_json is None:
        return None
    if _semantic_queue_lines(stdout_without_json, structured=True) is None:
        return None
    if _semantic_queue_lines(result.stderr, structured=True) is None:
        return None
    return rows


def _semantic_queue_lines(value: str, *, structured: bool) -> list[str] | None:
    """Classify diagnostics by meaning; unknown/job-like/error text fails closed."""

    ansi = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
    normalized: list[str] = []
    for raw in str(value or "").splitlines():
        line = " ".join(ansi.sub("", raw).strip().lower().split())
        if not line:
            continue
        if line in _EMPTY_QUEUE_MESSAGES:
            normalized.append(line)
            continue
        if re.fullmatch(r"[-=─━]{3,}", line):
            continue
        words = set(re.findall(r"[a-z]+", line))
        if "managed" in words and ({"job", "jobs"} & words) and words <= {
            "managed",
            "job",
            "jobs",
            "queue",
            "fetching",
            "checking",
            "statuses",
            "status",
            "done",
        }:
            continue
        contradictory = re.search(
            r"(?i)(?:unauth|forbidden|permission denied|access denied|traceback|"
            r"\b(?:fatal|error|exception)\b|job[_ -]?id|"
            r"\b(?:pending|starting|running|recovering|cancelling)\b)",
            line,
        )
        if line.startswith("warning:") and not contradictory:
            continue
        if structured and line.startswith(("note:", "notice:", "info:")) and not contradictory:
            continue
        return None
    return normalized


def _without_single_json_document(value: str) -> str | None:
    """Remove the one JSON document while retaining surrounding diagnostics."""

    text = str(value or "")
    decoder = json.JSONDecoder()
    matches: list[tuple[int, int]] = []
    for index, character in enumerate(text):
        if character not in "[{":
            continue
        try:
            _payload, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        matches.append((index, index + end))
    unique = []
    for span in matches:
        if span not in unique:
            unique.append(span)
    outer = [
        span
        for span in unique
        if all(
            other == span or (span[0] <= other[0] and other[1] <= span[1])
            for other in unique
        )
    ]
    if len(outer) != 1:
        return None
    start, end = outer[0]
    return text[:start] + text[end:]
