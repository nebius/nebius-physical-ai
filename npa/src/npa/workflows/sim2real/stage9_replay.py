"""Durable replay helper for compositional Sim2Real Stage 9."""

from __future__ import annotations

from typing import Any

from npa.workflows.sim2real.checkpoint_selection import select_best_checkpoint
from npa.workflows.sim2real.stage9_checkpoints import validate_replay_checkpoint_pool


def existing_replay(
    *,
    prior: dict[str, Any],
    outer_iteration: int,
    inner_iteration: int,
    actions_uri: str,
    evaluation_uri: str,
    signal_uri: str,
    sample_vlm_eval: dict[str, Any],
    sample_signal: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None:
    """Adopt exact committed PPO and checkpoint-sweep evidence on a Job retry.

    Args:
        prior: Durable evidence from the same outer loop.
        outer_iteration: Requested outer-loop coordinate.
        inner_iteration: Requested completed training-pass coordinate.
        actions_uri, evaluation_uri, signal_uri: Current authoritative S3 prefixes.
        sample_vlm_eval: Verified first evaluator result.
        sample_signal: Verified first training signal.
    Returns:
        The current candidate, global selection, and trainer output, or None.
    Raises:
        RuntimeError: Evidence is partial, divergent, or out of order.
    """
    matches = _same_iteration(prior, outer_iteration, inner_iteration)
    if matches is None:
        return None
    iteration, candidates = matches
    _validate_iteration_inputs(
        iteration,
        inner_iteration,
        actions_uri,
        evaluation_uri,
        signal_uri,
        sample_vlm_eval,
        sample_signal,
    )
    return _select_replay_checkpoint(prior, iteration, candidates)


def _same_iteration(prior, outer_iteration, inner_iteration):
    iterations = list(prior.get("iterations") or [])
    candidates = list(prior.get("checkpoint_candidates") or [])
    iteration_matches = [
        item
        for item in iterations
        if int(item.get("iteration") or 0) == inner_iteration
    ]
    candidate_matches = [
        item
        for item in candidates
        if int(item.get("outer_iteration") or 0) == outer_iteration
        and int(item.get("inner_iteration") or 0) == inner_iteration
    ]
    if not iteration_matches and not candidate_matches:
        if any(
            int(item.get("iteration") or 0) > inner_iteration for item in iterations
        ):
            raise RuntimeError(
                "Stage 9 cannot insert evidence before a later iteration"
            )
        return None
    if len(iteration_matches) != 1 or not candidate_matches:
        raise RuntimeError("Stage 9 found partial or duplicate same-iteration evidence")
    if int(prior.get("outer_iteration") or 0) != outer_iteration:
        raise RuntimeError(
            "Stage 9 replay evidence belongs to a different outer iteration"
        )
    if max(int(item.get("iteration") or 0) for item in iterations) != inner_iteration:
        raise RuntimeError("Stage 9 may replay only the latest committed iteration")
    return iteration_matches[0], candidate_matches


def _validate_iteration_inputs(
    iteration,
    inner_iteration,
    actions_uri,
    evaluation_uri,
    signal_uri,
    sample_vlm_eval,
    sample_signal,
):
    expected = {
        "iteration": inner_iteration,
        "actions_uri": actions_uri,
        "vlm_eval_uri": evaluation_uri,
        "signal_uri": signal_uri,
        "sample_vlm_eval": sample_vlm_eval,
        "sample_signal": sample_signal,
    }
    if any(iteration.get(key) != value for key, value in expected.items()):
        raise RuntimeError(
            "Stage 9 same-iteration replay conflicts with durable evidence"
        )


def _select_replay_checkpoint(prior, iteration, candidates):
    update = dict(iteration.get("update") or {})
    if any(row.get("evaluation_split") != "validation" for row in candidates):
        raise RuntimeError("Stage 9 replay checkpoint lineage is inconsistent")
    validate_replay_checkpoint_pool(update, candidates)
    current_selection = select_best_checkpoint(candidates)
    candidate = next(
        row
        for row in candidates
        if row["checkpoint_uri"] == current_selection["checkpoint_uri"]
    )
    selection = select_best_checkpoint(list(prior.get("checkpoint_candidates") or []))
    if (
        prior.get("selected_checkpoint_uri") != selection["checkpoint_uri"]
        or prior.get("final_checkpoint_uri") != selection["checkpoint_uri"]
        or prior.get("checkpoint_selection") != selection
        or prior.get("selected_validation_report") != selection.get("validation_report")
    ):
        raise RuntimeError("Stage 9 replay selection conflicts with its candidates")
    return candidate, selection, update
