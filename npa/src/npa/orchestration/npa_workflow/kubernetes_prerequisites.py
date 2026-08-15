"""Shared Kubernetes placement parsing for workflow submit preflights."""

from __future__ import annotations

import json
import re
from decimal import Decimal, InvalidOperation


_QUANTITY_RE = re.compile(
    r"([+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+))"
    r"(?:(Ki|Mi|Gi|Ti|Pi|Ei)|([numkKMGTPE])|([eE][+-]?[0-9]+))?"
)


def _decimal_quantity(value: object) -> Decimal | None:
    """Parse the numeric value of a Kubernetes resource quantity."""

    raw = str(value or "").strip()
    match = _QUANTITY_RE.fullmatch(raw)
    if not match:
        return None
    try:
        number = Decimal(match.group(1))
        binary_suffix, decimal_suffix, exponent_suffix = match.groups()[1:]
        if binary_suffix:
            power = "KMGTPE".index(binary_suffix[0]) + 1
            return number * (Decimal(1024) ** power)
        if decimal_suffix:
            powers = {
                "n": -9,
                "u": -6,
                "m": -3,
                "k": 3,
                "K": 3,
                "M": 6,
                "G": 9,
                "T": 12,
                "P": 15,
                "E": 18,
            }
            return number * (Decimal(10) ** powers[decimal_suffix])
        if exponent_suffix:
            return number * (Decimal(10) ** int(exponent_suffix[1:]))
        return number
    except (InvalidOperation, ValueError):
        return None


def cpu_millicores(value: object) -> int:
    parsed = _decimal_quantity(value)
    return max(0, int(parsed * 1000)) if parsed is not None else 0


def memory_bytes(value: object) -> int:
    parsed = _decimal_quantity(value)
    return max(0, int(parsed)) if parsed is not None else 0


def integer_resource(value: object) -> int:
    """Parse an integer-valued Kubernetes extended resource such as a GPU."""

    parsed = _decimal_quantity(value)
    if parsed is None or parsed < 0 or parsed != parsed.to_integral_value():
        return 0
    return int(parsed)


def format_cpu_memory_requirement(
    cpu_millicores_required: int, memory_bytes_required: int
) -> str:
    """Render the exact threshold constants with Kubernetes resource names."""

    cpu = Decimal(cpu_millicores_required) / 1000
    memory_gib = Decimal(memory_bytes_required) / (1024**3)
    return f"{cpu:g} CPU / {memory_gib:g} GiB allocatable"


def ready_schedulable_cpu_nodes(
    nodes_json: str,
    *,
    minimum_cpu_millicores: int,
    minimum_memory_bytes: int,
) -> list[str]:
    """Return Ready, schedulable nodes with the requested CPU and memory."""

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
        ):
            ready.append(str((node.get("metadata") or {}).get("name") or "<unnamed>"))
    return ready
