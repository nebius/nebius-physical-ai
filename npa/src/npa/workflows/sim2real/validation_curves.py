"""Join each completed native pass to its own final checkpoint validation."""

from __future__ import annotations

from typing import Any

from npa.workflows.sim2real.stage9_checkpoints import validate_replay_checkpoint_pool


def native_pass_validation_candidates(
    payload: dict[str, Any], outer: int
) -> dict[int, dict[str, Any]]:
    """Validate the complete checkpoint sweep before choosing each pass's final row.

    Args:
        payload: Canonical evidence with complete native passes and checkpoint sweeps.
        outer: Exact outer iteration represented by the evidence document.
    Returns:
        Each inner pass's final checkpoint candidate, retaining its own score.
    Raises:
        RuntimeError: Pass identities, checkpoint pools, or final identities disagree.
    """
    records = payload.get("iterations")
    population = payload.get("checkpoint_candidates")
    if not isinstance(records, list) or not isinstance(population, list):
        raise RuntimeError("Validation passes and candidate population must be lists")
    groups: dict[int, list[dict[str, Any]]] = {}
    for candidate in population:
        if not isinstance(candidate, dict):
            raise RuntimeError("Validation candidate must be an object")
        inner = candidate.get("inner_iteration")
        if (
            type(inner) is not int
            or inner < 1
            or type(candidate.get("outer_iteration")) is not int
            or candidate["outer_iteration"] != outer
            or candidate.get("evaluation_split") != "validation"
        ):
            raise RuntimeError("Validation candidate iteration or split is invalid")
        groups.setdefault(inner, []).append(candidate)
    final = {}
    for record in records:
        if not isinstance(record, dict):
            raise RuntimeError("Validation pass must be an object")
        inner = record.get("iteration")
        if type(inner) is not int or inner < 1 or inner in final or inner not in groups:
            raise RuntimeError("Validation pass identity is missing or ambiguous")
        update = record.get("update")
        if not isinstance(update, dict):
            raise RuntimeError("Validation pass has no native trainer result")
        validate_replay_checkpoint_pool(update, groups[inner])
        final_uri = update.get("final_checkpoint_uri", update.get("checkpoint_path"))
        if "periodic_checkpoints" in update and (
            update.get("final_checkpoint_alias_verified") is not True
            or not isinstance(update.get("final_checkpoint_sha256"), str)
            or update["final_checkpoint_sha256"] != update.get("checkpoint_sha256")
        ):
            raise RuntimeError(
                "Validation sweep has no verified final checkpoint alias"
            )
        matches = [
            candidate
            for candidate in groups[inner]
            if candidate.get("checkpoint_uri") == final_uri
        ]
        if len(matches) != 1:
            raise RuntimeError("Validation sweep lacks an exact final checkpoint")
        final[inner] = matches[0]
    if set(final) != set(groups):
        raise RuntimeError("Validation candidates do not cover the exact passes")
    return final
