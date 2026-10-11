"""eval_harness workbench SDK; mirrors the npa eval-harness CLI stages.

Thin, real wrappers over the pipeline stage functions in
``npa.workflows.byof.eval_harness_pipeline``: no ``NotImplementedError``,
no CLI indirection.
"""

from __future__ import annotations

from typing import Any, Mapping

from npa.cli.path_contract import PathContractError, validate_write_path


def run(
    *,
    task: str,
    policy: str,
    output_path: str = "",
    output_uri: str | None = None,
    episodes: int = 10,
    seed: int = 0,
    judge: str = "heuristic",
    max_steps: int = 500,
    vlm_endpoint_url: str = "",
) -> Mapping[str, Any]:
    """Evaluate one policy on a task; write JSON + Markdown reports."""
    from npa.workflows.byof.eval_harness_pipeline import run as _run

    if output_path and output_uri and output_path != output_uri:
        raise ValueError("output_path and compatibility output_uri must match")
    try:
        output_path = validate_write_path(
            output_path or output_uri or "",
            tool="eval-harness SDK run",
            option="output_path",
            required=True,
        )
    except PathContractError as exc:
        raise ValueError(str(exc)) from exc
    return _run(
        task=task,
        policy=policy,
        output_path=output_path,
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
    output_path: str = "",
    output_uri: str | None = None,
    episodes: int = 10,
    seed: int = 0,
    judge: str = "heuristic",
    max_steps: int = 500,
    vlm_endpoint_url: str = "",
) -> Mapping[str, Any]:
    """A/B compare two policies with paired seeds; write comparison reports."""
    from npa.workflows.byof.eval_harness_pipeline import compare as _compare

    if output_path and output_uri and output_path != output_uri:
        raise ValueError("output_path and compatibility output_uri must match")
    try:
        output_path = validate_write_path(
            output_path or output_uri or "",
            tool="eval-harness SDK compare",
            option="output_path",
            required=True,
        )
    except PathContractError as exc:
        raise ValueError(str(exc)) from exc
    return _compare(
        task=task,
        policy_a=policy_a,
        policy_b=policy_b,
        output_path=output_path,
        episodes=episodes,
        seed=seed,
        judge=judge,
        max_steps=max_steps,
        vlm_endpoint_url=vlm_endpoint_url,
    )


__all__ = ["compare", "run"]
