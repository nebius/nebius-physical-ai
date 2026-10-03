"""Durable replay helper for compositional Sim2Real Stage 9."""

from __future__ import annotations

from typing import Any

from npa.workflows.sim2real.checkpoint_selection import select_best_checkpoint


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
    """Validate and adopt evidence from an exact Stage 9 retry.

    Stage 9 writes its durable evidence before the Kubernetes Job becomes terminal.
    A replacement Job therefore treats an exact record at the same loop coordinate
    as committed output. Any partial, divergent, or out-of-order record fails closed.
    """

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
    if len(iteration_matches) != 1 or len(candidate_matches) != 1:
        raise RuntimeError("Stage 9 found partial or duplicate same-iteration evidence")
    if int(prior.get("outer_iteration") or 0) != outer_iteration:
        raise RuntimeError(
            "Stage 9 replay evidence belongs to a different outer iteration"
        )
    if max(int(item.get("iteration") or 0) for item in iterations) != inner_iteration:
        raise RuntimeError("Stage 9 may replay only the latest committed iteration")

    iteration = iteration_matches[0]
    candidate = candidate_matches[0]
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
    update = dict(iteration.get("update") or {})
    if candidate.get("evaluation_split") != "validation" or candidate.get(
        "checkpoint_uri"
    ) != update.get("checkpoint_path"):
        raise RuntimeError("Stage 9 replay checkpoint lineage is inconsistent")
    selection = select_best_checkpoint(candidates)
    if (
        prior.get("selected_checkpoint_uri") != selection["checkpoint_uri"]
        or prior.get("final_checkpoint_uri") != selection["checkpoint_uri"]
        or prior.get("checkpoint_selection") != selection
        or prior.get("selected_validation_report") != selection.get("validation_report")
    ):
        raise RuntimeError("Stage 9 replay selection conflicts with its candidates")
    return candidate, selection, update
