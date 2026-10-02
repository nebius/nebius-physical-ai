"""Deterministically aggregate holdout losses and select an intermediate checkpoint."""

from __future__ import annotations

import dataclasses
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


@dataclasses.dataclass(frozen=True)
class LossRow:
    """Describe one deterministic holdout example loss."""

    step: int
    task_id: int
    episode_index: int
    episode_relative_frame: int
    action_loss: float
    stage_cross_entropy: float
    action_dimension_losses: tuple[float, ...]


def aggregate(rows: list[LossRow]) -> list[dict[str, object]]:
    """Aggregate finite losses per checkpoint with equal task weighting."""

    if not rows:
        raise ValueError("holdout rows are empty")
    by_step: dict[int, list[LossRow]] = defaultdict(list)
    for row in rows:
        values = (
            row.action_loss,
            row.stage_cross_entropy,
            *row.action_dimension_losses,
        )
        if row.task_id not in {0, 1, 22} or len(row.action_dimension_losses) != 23:
            raise ValueError(
                "holdout row has an invalid task or action dimension count"
            )
        if not np.isfinite(values).all():
            raise ValueError("holdout loss contains non-finite values")
        by_step[row.step].append(row)
    results = []
    for step, step_rows in sorted(by_step.items()):
        per_task = _task_metrics(step, step_rows)
        score = float(
            np.mean([per_task[str(task)]["action_loss"] for task in (0, 1, 22)])
        )
        results.append({"step": step, "selection_score": score, "per_task": per_task})
    return results


def _task_metrics(step: int, rows: list[LossRow]) -> dict[str, object]:
    metrics = {}
    for task_id in (0, 1, 22):
        task_rows = [row for row in rows if row.task_id == task_id]
        if not task_rows:
            raise ValueError(f"checkpoint {step} lacks task {task_id} holdout rows")
        metrics[str(task_id)] = {
            "examples": len(task_rows),
            "action_loss": float(np.mean([row.action_loss for row in task_rows])),
            "stage_cross_entropy": float(
                np.mean([row.stage_cross_entropy for row in task_rows])
            ),
            "action_dimension_losses": np.mean(
                [row.action_dimension_losses for row in task_rows], axis=0
            ).tolist(),
        }
    return metrics


def select_checkpoint(rows: list[LossRow]) -> dict[str, object]:
    """Select the lowest equal-task action loss, breaking exact ties by earlier step."""

    metrics = aggregate(rows)
    selected = min(metrics, key=lambda row: (row["selection_score"], row["step"]))
    return {
        "schema": "npa.behavior.rlc-holdout-selection.v1",
        "rule": "minimum_equal_task_mean_action_loss_then_earliest_step",
        "selected_step": selected["step"],
        "selected_score": selected["selection_score"],
        "checkpoints": metrics,
    }


def write_selection(path: Path, rows: list[LossRow]) -> None:
    """Write the deterministic checkpoint-selection receipt."""

    path.write_text(
        json.dumps(select_checkpoint(rows), indent=2, sort_keys=True) + "\n"
    )
