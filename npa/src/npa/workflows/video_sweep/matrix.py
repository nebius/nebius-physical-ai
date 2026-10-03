"""Expand native Cosmos3 parameter axes into a deterministic candidate matrix."""

from __future__ import annotations

from itertools import product

from npa.workflows.video_sweep.artifacts import digest
from npa.workflows.video_sweep.cosmos3 import validate_variant

_FIELDS = {
    "hint",
    "prompt",
    "seed",
    "control_guidance",
    "edge_threshold",
    "guidance",
    "num_steps",
    "cfg_normalization",
    "first_chunk_conditional_frames",
}
_TEXT = {"hint", "prompt"}


def expand(sweep: dict) -> list[dict]:
    """Expand every axis combination and validate its native generation controls.

    Args:
        sweep: Fixed base fields and nonempty lists of axis values.
    Returns:
        Variants ordered by sorted axis names and declared value order.
    Raises:
        ValueError: Fields, axes, duplicate values or native controls are invalid.
    """
    _validate_axes(sweep)
    axes = sorted(sweep["axes"])
    variants = []
    for values in product(*(sweep["axes"][key] for key in axes)):
        variant = {**sweep["base"], **dict(zip(axes, values, strict=True))}
        validate_variant(variant)
        variants.append(variant)
    return variants


def _validate_axes(sweep):
    if not isinstance(sweep, dict) or set(sweep) != {"base", "axes"}:
        raise ValueError("A sweep requires exactly base and axes")
    base, axes = sweep["base"], sweep["axes"]
    if not isinstance(base, dict) or not isinstance(axes, dict) or not axes:
        raise ValueError("Sweep base must be an object and axes must be nonempty")
    if (set(base) | set(axes)) - _FIELDS:
        raise ValueError("Unknown native sweep parameter")
    if set(base) & set(axes):
        raise ValueError("A parameter must occur in either base or axes, not both")
    for values in axes.values():
        if not isinstance(values, list) or not values:
            raise ValueError("Each sweep axis requires a nonempty list")
        if any(value in values[:index] for index, value in enumerate(values)):
            raise ValueError("Duplicate values in a sweep axis")


def validate_plan(plan: dict) -> None:
    """Verify that a stored matrix plan covers every source and combination.

    Args:
        plan: Stored plan, optionally containing a native sweep definition.
    Returns:
        None; historical explicit-row plans remain supported.
    Raises:
        ValueError: Matrix coverage, source grouping or candidate identity differs.
    """
    if "sweep" not in plan:
        return
    if plan.get("generator") != "cosmos3-nano":
        raise ValueError("Parameter matrices require the native Cosmos3 backend")
    variants = expand(plan["sweep"])
    items = plan["items"]
    if not items or len(items) % len(variants):
        raise ValueError("Plan does not contain complete parameter combinations")
    sources = []
    for offset in range(0, len(items), len(variants)):
        source = items[offset]["source"]
        sources.append(source["sha256"])
        for item, variant in zip(
            items[offset : offset + len(variants)], variants, strict=True
        ):
            identity = digest({"source": source["sha256"], "variant": variant})
            if (
                item["source"] != source
                or item["variant"] != variant
                or item["id"] != identity
            ):
                raise ValueError("Plan candidate differs from the configured matrix")
    if len(set(sources)) != len(sources):
        raise ValueError("Matrix plan contains duplicate sources")


def describe(sweep: dict, sources: int, workers: int) -> dict:
    """Describe the complete fanout without exposing prompts or source locations.

    Args:
        sweep: Validated native base and axes configuration.
        sources: Number of input sources.
        workers: Number of declared GPU worker partitions.
    Returns:
        Public-safe axis values, fixed controls and candidate assignments.
    Raises:
        ValueError: The sweep or partition counts are invalid.
    """
    variants = expand(sweep)
    if any(type(value) is not int or value < 1 for value in (sources, workers)):
        raise ValueError("Source and worker counts must be positive integers")
    axes = {
        key: _display_values(key, sweep["axes"][key]) for key in sorted(sweep["axes"])
    }
    return {
        "mode": "cartesian",
        "axes": axes,
        "base": {
            key: value for key, value in sweep["base"].items() if key not in _TEXT
        },
        "sources": sources,
        "combinations": len(variants),
        "workers": workers,
        "jobs": [
            _job(index, source, variant, variants, sweep, workers)
            for index, (source, variant) in enumerate(
                product(range(sources), range(len(variants)))
            )
        ],
    }


def _display_values(key, values):
    if key in _TEXT:
        return [f"{key.title()} {index + 1}" for index in range(len(values))]
    return values


def _job(index, source, variant, variants, sweep, workers):
    coordinates = {}
    for key, values in sorted(sweep["axes"].items()):
        position = values.index(variants[variant][key])
        coordinates[key] = _display_values(key, values)[position]
    return {
        "candidate": index + 1,
        "source": source + 1,
        "combination": variant + 1,
        "worker": index % workers,
        "parameters": coordinates,
    }
