"""Strict parsing for JSON emitted after harmless SkyPilot CLI diagnostics."""

from __future__ import annotations

import re
import subprocess
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

    if result.returncode not in {0, 1}:
        return False
    stdout_lines = _known_empty_queue_lines(result.stdout, stream="stdout")
    stderr_lines = _known_empty_queue_lines(result.stderr, stream="stderr")
    if stdout_lines is None or stderr_lines is None:
        return False
    markers = [
        line
        for line in (*stdout_lines, *stderr_lines)
        if line in _EMPTY_QUEUE_MESSAGES
    ]
    return len(markers) == 1


def _known_empty_queue_lines(value: str, *, stream: str) -> list[str] | None:
    """Remove only pinned SkyPilot presentation lines; reject all other text."""

    ansi = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
    normalized: list[str] = []
    for raw in str(value or "").splitlines():
        line = " ".join(ansi.sub("", raw).strip().lower().split())
        if not line:
            continue
        if line in _EMPTY_QUEUE_MESSAGES:
            normalized.append(line)
            continue
        if stream == "stdout" and (
            line
            in {
                "managed jobs",
                "managed jobs queue",
                "fetching managed job statuses...",
                "checking managed jobs",
                "checking managed jobs...",
                "checking managed jobs... done",
            }
            or re.fullmatch(r"[-=─━]{3,}", line)
        ):
            continue
        if stream == "stderr" and (
            line.startswith("warning: skypilot telemetry ")
            or line.startswith("warning: skypilot update check ")
            or line.startswith("warning: managed jobs output format ")
        ):
            continue
        return None
    return normalized
