"""Translate SkyPilot GPU shapes into conservative Kubernetes requests."""

from collections.abc import Mapping
from decimal import Decimal, DecimalException, ROUND_CEILING
import math
import re

from npa.orchestration.skypilot.gpu_catalog import parse_accelerator_request


_NUMBER = r"[+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
_CPU = re.compile(rf"({_NUMBER})\+?")
_MEMORY = re.compile(rf"({_NUMBER})(KB|Ki|MB|Mi|GB|Gi|TB|Ti|PB|Pi)?\+?", re.I)
_MEMORY_UNITS = {
    "": Decimal(1), "gb": Decimal(1), "gi": Decimal(1),
    "kb": Decimal(1) / 1024**2, "ki": Decimal(1) / 1024**2,
    "mb": Decimal(1) / 1024, "mi": Decimal(1) / 1024,
    "tb": Decimal(1024), "ti": Decimal(1024),
    "pb": Decimal(1024**2), "pi": Decimal(1024**2),
}


def _quantity(raw: object, *, memory: bool = False) -> Decimal:
    if isinstance(raw, bool) or not isinstance(raw, (str, int, float)):
        raise ValueError("SkyPilot GPU CPU/memory quantities must be positive finite numbers")
    match = (_MEMORY if memory else _CPU).fullmatch(str(raw).strip())
    if match is None:
        raise ValueError("Unsupported SkyPilot GPU quantity; use absolute CPU and memory requests")
    value = Decimal(match.group(1))
    if memory:
        value *= _MEMORY_UNITS[(match.group(2) or "").lower()]
    if not value.is_finite() or value <= 0 or not math.isfinite(float(value)):
        raise ValueError("SkyPilot GPU CPU/memory quantities must be positive finite numbers")
    return value


def kubernetes_gpu_quantities(
    resources: Mapping[str, object], *, accelerator: str,
) -> tuple[str, str]:
    """Match SkyPilot's Kubernetes GPU defaults and pod-template units.

    SkyPilot 0.12.2 normalizes memory suffixes with binary ratios, but its
    Kubernetes pod template emits the resulting number with decimal ``G``.
    Its instance names round shapes to one decimal place. Check at least both
    the requested value and that rendered value, rounding upward to Kubernetes
    millicores/bytes. Native Kubernetes quantity parsing remains separate.
    """
    if resources.get("instance_type"):
        raise ValueError("GPU capacity preflight requires explicit CPU/memory quantities instead of instance_type")
    try:
        cpu_input = resources.get("cpus")
        cpu = (
            Decimal(4 * parse_accelerator_request(accelerator).quantity)
            if cpu_input in (None, "") else _quantity(cpu_input)
        )
        cpu = max(cpu, Decimal(format(float(cpu), ".1f")))
        memory_input = resources.get("memory")
        memory = 4 * cpu if memory_input in (None, "") else _quantity(memory_input, memory=True)
        memory = max(memory, Decimal(format(float(memory), ".1f")))
        cpu_millis = int((cpu * 1000).to_integral_value(rounding=ROUND_CEILING))
        memory_bytes = int((memory * 10**9).to_integral_value(rounding=ROUND_CEILING))
    except (DecimalException, OverflowError) as exc:
        raise ValueError("SkyPilot GPU CPU/memory quantity cannot be represented") from exc
    return f"{cpu_millis}m", str(memory_bytes)
