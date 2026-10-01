"""eval_harness workbench SDK; mirrors the npa eval-harness CLI stages.

Thin, real wrappers over the pipeline stage functions in
``npa.workflows.byof.eval_harness_pipeline``: no ``NotImplementedError``,
no CLI indirection.
"""

from __future__ import annotations

from typing import Any, Mapping


def run(
    *,
    task: str,
    policy: str,
    output_uri: str,
    episodes: int = 10,
    seed: int = 0,
    judge: str = "heuristic",
    max_steps: int = 500,
    vlm_endpoint_url: str = "",
) -> Mapping[str, Any]:
    """Evaluate one policy on a task; write JSON + Markdown reports."""
    from npa.workflows.byof.eval_harness_pipeline import run as _run

    return _run(
        task=task,
        policy=policy,
        output_uri=output_uri,
        episodes=episodes,
        seed=seed,
        judge=judge,
        max_steps=max_steps,
        vlm_endpoint_url=vlm_endpoint_url,
    )


def compare(
    *,
    task: str,
    policy_a: str,
    policy_b: str,
    output_uri: str,
    episodes: int = 10,
    seed: int = 0,
    judge: str = "heuristic",
    max_steps: int = 500,
    vlm_endpoint_url: str = "",
) -> Mapping[str, Any]:
    """A/B compare two policies with paired seeds; write comparison reports."""
    from npa.workflows.byof.eval_harness_pipeline import compare as _compare

    return _compare(
        task=task,
        policy_a=policy_a,
        policy_b=policy_b,
        output_uri=output_uri,
        episodes=episodes,
        seed=seed,
        judge=judge,
        max_steps=max_steps,
        vlm_endpoint_url=vlm_endpoint_url,
    )


__all__ = ["compare", "run"]
