"""Closed, trusted postprocess registry for successful BYOF solution runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class PostprocessContext:
    run_prefix_uri: str
    project: str | None


Postprocessor = Callable[[PostprocessContext], dict[str, Any]]


def _wan_single(context: PostprocessContext) -> dict[str, Any]:
    from npa.solutions.wan2_2.rerun import publish_wan_rrd_from_s3

    return publish_wan_rrd_from_s3(
        context.run_prefix_uri, variant="single", project=context.project
    )


def _wan_multigpu(context: PostprocessContext) -> dict[str, Any]:
    from npa.solutions.wan2_2.rerun import publish_wan_rrd_from_s3

    return publish_wan_rrd_from_s3(
        context.run_prefix_uri, variant="multigpu", project=context.project
    )


POSTPROCESSORS: dict[str, Postprocessor] = {
    "wan2.2": _wan_single,
    "wan2.2-multigpu": _wan_multigpu,
}


def run_registered_postprocess(
    key: str, context: PostprocessContext
) -> dict[str, Any] | None:
    """Run a registered solution hook, if one exists; never import from config."""

    normalized = str(key or "").strip()
    if not normalized:
        return None
    processor = POSTPROCESSORS.get(normalized)
    if processor is None:
        return None
    return processor(context)
