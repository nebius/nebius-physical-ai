"""Stage 9 evaluator-coverage boundary for compositional Sim2Real."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def validate_stage7_cosmos3_coverage(
    stage7: dict[str, Any], cosmos3: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Return evaluations only when one Cosmos3 result covers Stage 7 exactly."""

    rollout_dirs = stage7.get("rollout_dirs")
    stage7_ids = (
        [Path(str(item).rstrip("/")).name for item in rollout_dirs]
        if isinstance(rollout_dirs, list)
        else []
    )
    evaluations = cosmos3.get("evaluations")
    evaluation_rows = evaluations if isinstance(evaluations, list) else []
    evaluation_ids = [
        str(item.get("rollout_id") or "")
        for item in evaluation_rows
        if isinstance(item, dict)
    ]
    source_ids_value = cosmos3.get("source_rollout_ids")
    source_ids = (
        [str(item) for item in source_ids_value]
        if isinstance(source_ids_value, list)
        else []
    )
    if (
        stage7.get("schema") != "npa.sim2real.policy_rollouts.v1"
        or not stage7_ids
        or any(not item for item in stage7_ids)
        or len(set(stage7_ids)) != len(stage7_ids)
        or not evaluation_rows
        or len(evaluation_ids) != len(evaluation_rows)
        or any(not item for item in evaluation_ids)
        or len(set(evaluation_ids)) != len(evaluation_ids)
        or len(source_ids) != len(evaluation_ids)
        or len(set(source_ids)) != len(source_ids)
        or set(source_ids) != set(evaluation_ids)
        or len(stage7_ids) != len(evaluation_ids)
        or set(stage7_ids) != set(evaluation_ids)
    ):
        raise RuntimeError(
            "Stage 8 Cosmos3 evaluations do not exactly cover Stage 7 rollouts"
        )
    return {
        rollout_id: item for rollout_id, item in zip(evaluation_ids, evaluation_rows)
    }
