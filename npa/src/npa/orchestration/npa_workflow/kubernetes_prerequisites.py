"""Shared Kubernetes placement parsing for workflow submit preflights."""

from __future__ import annotations

import json
import re


def cpu_millicores(value: object) -> int:
    raw = str(value or "").strip()
    try:
        return int(raw[:-1]) if raw.endswith("m") else int(float(raw) * 1000)
    except ValueError:
        return 0


def memory_bytes(value: object) -> int:
    raw = str(value or "").strip()
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([KMGTPE]i?|)", raw)
    if not match:
        return 0
    number = float(match.group(1))
    suffix = match.group(2)
    if not suffix:
        return int(number)
    powers = {letter: index for index, letter in enumerate("KMGTPE", 1)}
    return int(number * ((1024 if suffix.endswith("i") else 1000) ** powers[suffix[0]]))


def ready_schedulable_cpu_nodes(
    nodes_json: str,
    *,
    minimum_cpu_millicores: int,
    minimum_memory_bytes: int,
) -> list[str]:
    """Return Ready, untainted non-GPU nodes with the requested capacity."""

    try:
        items = (json.loads(nodes_json) or {}).get("items") or []
    except (TypeError, json.JSONDecodeError):
        return []
    ready: list[str] = []
    for node in items:
        spec = node.get("spec") or {}
        status = node.get("status") or {}
        allocatable = status.get("allocatable") or {}
        conditions = status.get("conditions") or []
        is_ready = any(
            item.get("type") == "Ready" and str(item.get("status")).lower() == "true"
            for item in conditions
        )
        blocking_taint = any(
            str(item.get("effect") or "") in {"NoSchedule", "NoExecute"}
            for item in (spec.get("taints") or [])
        )
        if (
            is_ready
            and not spec.get("unschedulable", False)
            and not blocking_taint
            and cpu_millicores(allocatable.get("cpu")) >= minimum_cpu_millicores
            and memory_bytes(allocatable.get("memory")) >= minimum_memory_bytes
            and cpu_millicores(allocatable.get("nvidia.com/gpu")) == 0
        ):
            ready.append(str((node.get("metadata") or {}).get("name") or "<unnamed>"))
    return ready
