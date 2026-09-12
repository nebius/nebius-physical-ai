"""Translate SkyPilot GPU shapes into conservative Kubernetes requests."""

from collections.abc import Mapping
from decimal import Decimal, DecimalException
from fractions import Fraction
import math
import re
import sys

from npa.orchestration.skypilot.gpu_catalog import parse_accelerator_request


_NUMBER = r"[+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
_CPU = re.compile(rf"({_NUMBER})\+?")
_MEMORY = re.compile(rf"({_NUMBER})(KB|Ki|MB|Mi|GB|Gi|TB|Ti|PB|Pi)?\+?", re.I)
_MEMORY_UNITS = {
    "": Fraction(1), "gb": Fraction(1), "gi": Fraction(1),
    "kb": Fraction(1, 1024**2), "ki": Fraction(1, 1024**2),
    "mb": Fraction(1, 1024), "mi": Fraction(1, 1024),
    "tb": Fraction(1024), "ti": Fraction(1024),
    "pb": Fraction(1024**2), "pi": Fraction(1024**2),
}
# These bounds follow from float eligibility and the smallest output quantum,
# not an input-size limit. They avoid expanding arbitrarily large exponents.
_MAX_RAW_EXPONENT = Decimal.from_float(sys.float_info.max).adjusted() + len(
    str(max(unit.denominator for unit in _MEMORY_UNITS.values()))
)
_MIN_CAPACITY_QUANTUM = Fraction(1, 4 * 10**9)
_MIN_RAW_EXPONENT = -len(str(
    _MIN_CAPACITY_QUANTUM.denominator * max(unit.numerator for unit in _MEMORY_UNITS.values())
))


def _quantity(raw: object, *, memory: bool = False) -> Fraction:
    if isinstance(raw, bool) or not isinstance(raw, (str, int, float)):
        raise ValueError("SkyPilot GPU CPU/memory quantities must be positive finite numbers")
    match = (_MEMORY if memory else _CPU).fullmatch(str(raw).strip())
    if match is None:
        raise ValueError("Unsupported SkyPilot GPU quantity; use absolute CPU and memory requests")
    value = Decimal(match.group(1))
    if not value.is_finite() or value <= 0:
        raise ValueError("SkyPilot GPU CPU/memory quantities must be positive finite numbers")
    if value.adjusted() > _MAX_RAW_EXPONENT:
        # Even the largest suffix denominator cannot normalize this to a float.
        raise ValueError("SkyPilot GPU CPU/memory quantities must be positive finite numbers")
    if value.adjusted() < _MIN_RAW_EXPONENT:
        # Even the largest suffix numerator leaves this below 1/(4e9).
        # This representative has exactly the same positive output ceilings:
        # one millicore, one byte, and one byte for the default 4x CPU memory.
        return _MIN_CAPACITY_QUANTUM
    exact = Fraction(value)
    if memory:
        exact *= _MEMORY_UNITS[(match.group(2) or "").lower()]
    if not math.isfinite(float(exact)):
        raise ValueError("SkyPilot GPU CPU/memory quantities must be positive finite numbers")
    return exact


def kubernetes_gpu_quantities(
    resources: Mapping[str, object], *, accelerator: str,
) -> tuple[str, str]:
    """Match SkyPilot's Kubernetes GPU defaults and pod-template units.

    SkyPilot 0.12.2 normalizes memory suffixes with binary ratios, but its
    Kubernetes pod template emits the resulting number with decimal ``G``.
    Its instance names round shapes to one decimal place. Check at least both
    the requested value and that rendered value, rounding upward to Kubernetes
    millicores/bytes. Exact rational arithmetic retains positive fractions beyond
    Decimal's context precision. Native Kubernetes quantity parsing remains separate.

    Args:
        resources: SkyPilot resource settings with absolute CPU and memory values.
        accelerator: Selected GPU name and count.
    Returns:
        Conservative Kubernetes CPU millicores and memory bytes as strings.
    Raises:
        ValueError: A resource shape or quantity is unsupported or unrepresentable.
    """
    if resources.get("instance_type"):
        raise ValueError("GPU capacity preflight requires explicit CPU/memory quantities instead of instance_type")
    try:
        cpu_input = resources.get("cpus")
        cpu = (
            Fraction(4 * parse_accelerator_request(accelerator).quantity)
            if cpu_input in (None, "") else _quantity(cpu_input)
        )
        cpu = max(cpu, Fraction(format(float(cpu), ".1f")))
        memory_input = resources.get("memory")
        memory = 4 * cpu if memory_input in (None, "") else _quantity(memory_input, memory=True)
        memory = max(memory, Fraction(format(float(memory), ".1f")))
        cpu_millis = math.ceil(cpu * 1000)
        memory_bytes = math.ceil(memory * 10**9)
    except (DecimalException, OverflowError) as exc:
        raise ValueError("SkyPilot GPU CPU/memory quantity cannot be represented") from exc
    return f"{cpu_millis}m", str(memory_bytes)
