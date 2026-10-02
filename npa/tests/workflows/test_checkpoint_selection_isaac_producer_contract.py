"""Wire the real held-out-report producer into the real checkpoint selector.

CPU-only. Unlike ``test_sim2real_efficacy_contract.py`` (hand-built
``validation_report`` dicts), this calls the real producer
(``byo_isaac_eval.per_env_from_distances`` + ``build_heldout_report``) and
feeds its real output into the real ``select_best_checkpoint``.

The zero/missing-distance fixtures are explicitly constructed producer
inputs, not captured live samples (an exact 0.0m mean across every held-out
env is not typical), used to prove the contract distinguishes a genuine zero
from genuinely absent evidence.
"""

from __future__ import annotations

from typing import Any

from npa.workflows.sim2real.byo_isaac_eval import (
    build_heldout_report,
    per_env_from_distances,
)
from npa.workflows.sim2real.checkpoint_selection import select_best_checkpoint

ISAAC_TASK = "Isaac-Lift-Cube-Franka-v0"


def _validation_candidate(
    *, checkpoint_uri: str, training_iteration: int, report: dict[str, Any]
) -> dict[str, Any]:
    return {
        "evaluation_split": "validation",
        "training_iteration": training_iteration,
        "checkpoint_uri": checkpoint_uri,
        "checkpoint_sha256": f"sha256-{training_iteration:04d}",
        "validation_report": report,
    }


def _real_zero_distance_report() -> dict[str, Any]:
    """Real producer report from held-out envs that all land on 0.0m."""

    per_env = per_env_from_distances(
        [0.0, 0.0, 0.0, 0.0],
        success_dist_m=0.05,
        env_ids=["env-0", "env-1", "env-2", "env-3"],
    )
    return build_heldout_report(
        per_env,
        isaac_task=ISAAC_TASK,
        checkpoint_uri="s3://bucket/perfect/model_latest.pt",
        source="byo_isaac_eval",
    )


def _real_missing_distance_report() -> dict[str, Any]:
    """Real producer report from envs that never populated a distance.

    ``build_heldout_report`` only populates ``mean_object_goal_distance_m``
    when at least one row's ``details`` has ``object_goal_distance_m``;
    omitting it here is the real absence path, not a hand-deleted key.
    """

    per_env = [
        {"env_id": f"env-{i}", "success": False, "score": 0.0, "details": {}}
        for i in range(4)
    ]
    return build_heldout_report(
        per_env,
        isaac_task=ISAAC_TASK,
        checkpoint_uri="s3://bucket/no-evidence/model_latest.pt",
        source="byo_isaac_eval",
    )


def _real_nonzero_distance_report(
    distance_m: float, *, checkpoint_uri: str
) -> dict[str, Any]:
    per_env = per_env_from_distances(
        [distance_m] * 4,
        success_dist_m=0.05,
        env_ids=["env-0", "env-1", "env-2", "env-3"],
    )
    return build_heldout_report(
        per_env,
        isaac_task=ISAAC_TASK,
        checkpoint_uri=checkpoint_uri,
        source="byo_isaac_eval",
    )


def test_real_zero_distance_report_has_no_success_evidence_gap() -> None:
    """Sanity: the real-zero fixture reaches the selector as 0.0, not missing."""

    report = _real_zero_distance_report()
    assert report["success_summary"]["mean_object_goal_distance_m"] == 0.0
    assert (
        "mean_object_goal_distance_m"
        not in _real_missing_distance_report()["success_summary"]
    )


def test_producer_to_selector_prefers_real_zero_distance_over_missing_evidence() -> (
    None
):
    """A real 0.0m producer report must beat a real missing-evidence report."""

    perfect = _validation_candidate(
        checkpoint_uri="s3://bucket/perfect/model_latest.pt",
        training_iteration=200,
        report=_real_zero_distance_report(),
    )
    no_evidence = _validation_candidate(
        checkpoint_uri="s3://bucket/no-evidence/model_latest.pt",
        training_iteration=100,
        report=_real_missing_distance_report(),
    )

    selected = select_best_checkpoint([perfect, no_evidence])
    assert selected["checkpoint_uri"] == "s3://bucket/perfect/model_latest.pt"
    # Index 6 of the rank key is ``-mean_distance``; a genuine 0.0m distance
    # must be a finite -0.0, never the -1e9 missing-evidence sentinel.
    assert selected["rank_key"][6] == 0.0

    # Order independence: swapping input order must not change the winner.
    selected_swapped = select_best_checkpoint([no_evidence, perfect])
    assert selected_swapped["checkpoint_uri"] == selected["checkpoint_uri"]


def test_producer_to_selector_missing_evidence_never_outranks_any_real_distance() -> (
    None
):
    """Missing distance evidence must lose to a real, even poor, distance."""

    no_evidence = _validation_candidate(
        checkpoint_uri="s3://bucket/no-evidence/model_latest.pt",
        training_iteration=100,
        report=_real_missing_distance_report(),
    )
    poor_but_measured = _validation_candidate(
        checkpoint_uri="s3://bucket/poor/model_latest.pt",
        training_iteration=200,
        report=_real_nonzero_distance_report(
            0.45, checkpoint_uri="s3://bucket/poor/model_latest.pt"
        ),
    )

    selected = select_best_checkpoint([no_evidence, poor_but_measured])
    assert selected["checkpoint_uri"] == "s3://bucket/poor/model_latest.pt"


def _independent_rank_from_geometry(
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Recompute the winner directly from per-env geometry.

    Deliberately does not call ``checkpoint_rank_key`` or any
    ``checkpoint_selection.py`` helper, so a bug shared between the producer
    and the selector can't make both this and the selector agree by
    coincidence.
    """

    def _score(candidate: dict[str, Any]) -> tuple[float, float]:
        per_env = candidate["validation_report"]["per_env"]
        strict_successes = sum(1 for row in per_env if row.get("success"))
        strict_rate = strict_successes / len(per_env)
        distances = [
            row["details"]["object_goal_distance_m"]
            for row in per_env
            if "object_goal_distance_m" in row.get("details", {})
        ]
        mean_distance = sum(distances) / len(distances) if distances else None
        return (strict_rate, mean_distance)

    def _sort_key(candidate: dict[str, Any]) -> tuple[float, float]:
        strict_rate, mean_distance = _score(candidate)
        # Missing distance evidence must rank worse than any real distance,
        # mirroring the sentinel semantics under independent recomputation.
        distance_component = -mean_distance if mean_distance is not None else -1.0e9
        return (strict_rate, distance_component)

    return max(candidates, key=_sort_key)


def test_independent_geometry_recompute_agrees_with_selector() -> None:
    """Cross-check select_best_checkpoint against a from-scratch recompute."""

    candidates = [
        _validation_candidate(
            checkpoint_uri="s3://bucket/best/model_latest.pt",
            training_iteration=300,
            report=_real_zero_distance_report(),
        ),
        _validation_candidate(
            checkpoint_uri="s3://bucket/mid/model_latest.pt",
            training_iteration=200,
            report=_real_nonzero_distance_report(
                0.12, checkpoint_uri="s3://bucket/mid/model_latest.pt"
            ),
        ),
        _validation_candidate(
            checkpoint_uri="s3://bucket/no-evidence/model_latest.pt",
            training_iteration=100,
            report=_real_missing_distance_report(),
        ),
    ]

    selected = select_best_checkpoint(candidates)
    independent = _independent_rank_from_geometry(candidates)
    assert selected["checkpoint_uri"] == independent["checkpoint_uri"]
    assert selected["checkpoint_uri"] == "s3://bucket/best/model_latest.pt"


def _pre_fix_mean_distance(summary: dict[str, Any]) -> float:
    """Pre-fix ``float(x or 1e9)`` expression, for regression proof only."""

    return float(summary.get("mean_object_goal_distance_m") or 1.0e9)


def _pre_fix_distance_component(candidate: dict[str, Any]) -> float:
    report = candidate["validation_report"]
    return -_pre_fix_mean_distance(report.get("success_summary") or {})


def test_pre_fix_rank_key_would_have_misranked_real_producer_output() -> None:
    """Both candidates tie on strict success rate (both under 0.05m) and on
    decomposed metrics; only the mean distance (0.0 vs 0.03) differs. The
    fix must prefer 0.0; the pre-fix sentinel coercion would have picked the
    worse checkpoint instead.
    """

    perfect = _validation_candidate(
        checkpoint_uri="s3://bucket/perfect/model_latest.pt",
        training_iteration=200,
        report=_real_zero_distance_report(),
    )
    good = _validation_candidate(
        checkpoint_uri="s3://bucket/good/model_latest.pt",
        training_iteration=100,
        report=_real_nonzero_distance_report(
            0.03, checkpoint_uri="s3://bucket/good/model_latest.pt"
        ),
    )

    selected = select_best_checkpoint([perfect, good])
    assert selected["checkpoint_uri"] == "s3://bucket/perfect/model_latest.pt"

    pre_fix_winner = max([perfect, good], key=_pre_fix_distance_component)
    assert pre_fix_winner["checkpoint_uri"] == "s3://bucket/good/model_latest.pt", (
        "expected the historical bug to flip the outcome on this real "
        "producer-built pair; if it no longer does, this test no longer "
        "documents a real regression"
    )
