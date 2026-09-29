"""Apply the complete-panel stock-anchored checkpoint selection rule."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SelectionRule:
    """Declare the immutable inputs to offline checkpoint selection."""

    task_stages: tuple[tuple[int, int], ...] = ((0, 5), (1, 6), (22, 9))
    candidate_steps: tuple[int, ...] = (400, 800, 1200, 1600, 2000, 2399)
    stock_step: int = -1
    uniform_flow_ratio: float = 1.02
    non_late_proxy_ratio: float = 1.005
    action_dimensions: int = 23


REQUIRED_FIELDS = frozenset(
    {
        "step",
        "task_id",
        "episode_index",
        "episode_relative_frame",
        "teacher_stage",
        "action_loss",
        "stage_cross_entropy",
        "action_dimension_losses",
    }
)
DEFAULT_RULE = SelectionRule()


def _mean(values: Iterable[float]) -> float:
    rows = list(values)
    if not rows or not all(math.isfinite(value) for value in rows):
        raise ValueError("selection metric is empty or non-finite")
    return math.fsum(rows) / len(rows)


def _integer(value: Any) -> bool:
    return type(value) is int


def _nonnegative_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _validate_rule(rule: SelectionRule) -> dict[int, int]:
    tasks = dict(rule.task_stages)
    if (
        not tasks
        or len(tasks) != len(rule.task_stages)
        or any(
            not _integer(task) or not _integer(stages) or task < 0 or stages <= 0
            for task, stages in tasks.items()
        )
    ):
        raise ValueError("task-stage registry differs")
    expected_steps = {rule.stock_step, *rule.candidate_steps}
    if (
        not _integer(rule.stock_step)
        or any(not _integer(step) for step in rule.candidate_steps)
        or len(expected_steps) != len(rule.candidate_steps) + 1
    ):
        raise ValueError("checkpoint steps must be unique")
    if (
        not _integer(rule.action_dimensions)
        or rule.action_dimensions <= 0
        or type(rule.uniform_flow_ratio) not in (int, float)
        or type(rule.non_late_proxy_ratio) not in (int, float)
        or not math.isfinite(rule.uniform_flow_ratio)
        or not math.isfinite(rule.non_late_proxy_ratio)
        or rule.uniform_flow_ratio < 1
        or rule.non_late_proxy_ratio < 1
    ):
        raise ValueError("selection thresholds differ")
    return tasks


def _validate_row(row, rule, tasks, expected_steps):
    if set(row) != REQUIRED_FIELDS:
        raise ValueError("holdout loss row fields differ")
    step, task = row["step"], row["task_id"]
    if (
        not _integer(step)
        or not _integer(task)
        or step not in expected_steps
        or task not in tasks
    ):
        raise ValueError("holdout step or task differs")
    values = row["action_dimension_losses"]
    if (
        not isinstance(values, list)
        or len(values) != rule.action_dimensions
        or not all(_nonnegative_number(value) for value in values)
    ):
        raise ValueError("holdout action-dimension losses differ")
    if not _nonnegative_number(row["action_loss"]) or not _nonnegative_number(
        row["stage_cross_entropy"]
    ):
        raise ValueError("holdout scalar loss differs")
    if any(
        not _integer(row[field]) or row[field] < 0
        for field in ("episode_index", "episode_relative_frame")
    ):
        raise ValueError("holdout sample identity differs")
    if (
        not _integer(row["teacher_stage"])
        or not 0 <= row["teacher_stage"] < tasks[task]
    ):
        raise ValueError("holdout teacher stage differs")
    return step, task


def _aligned_panels(rows, rule, tasks):
    expected_steps = {rule.stock_step, *rule.candidate_steps}
    by_step: dict[int, dict[tuple[int, int, int], dict]] = defaultdict(dict)
    for row in rows:
        step, task = _validate_row(row, rule, tasks, expected_steps)
        key = (task, row["episode_index"], row["episode_relative_frame"])
        if key in by_step[step]:
            raise ValueError("holdout sample is duplicated")
        by_step[step][key] = row
    if set(by_step) != expected_steps:
        raise ValueError("holdout checkpoint set differs")
    stock = by_step[rule.stock_step]
    if not stock:
        raise ValueError("stock holdout panel is empty")
    for step, samples in by_step.items():
        if set(samples) != set(stock):
            raise ValueError(f"holdout sample panel differs at step {step}")
        if any(
            row["teacher_stage"] != stock[key]["teacher_stage"]
            for key, row in samples.items()
        ):
            raise ValueError(f"holdout teacher stage differs at step {step}")
    return by_step


def _task_metrics(samples, task, stages):
    task_rows = [row for key, row in samples.items() if key[0] == task]
    late_start = math.ceil(2 * stages / 3)
    late = [row for row in task_rows if row["teacher_stage"] >= late_start]
    non_late = [row for row in task_rows if row["teacher_stage"] < late_start]
    return {
        "examples": len(task_rows),
        "action_loss": _mean(row["action_loss"] for row in task_rows),
        "non_late_proxy_loss": _mean(row["action_loss"] for row in non_late),
        "late_proxy_loss": _mean(row["action_loss"] for row in late),
        "late_teacher_stage_start": late_start,
    }


def _checkpoint_metrics(step, samples, tasks, rule):
    per_task = {
        str(task): _task_metrics(samples, task, stages)
        for task, stages in tasks.items()
    }
    return {
        "step": step,
        "label": "stock" if step == rule.stock_step else f"step-{step}",
        "per_task": per_task,
        "equal_task_action_loss": _mean(
            per_task[str(task)]["action_loss"] for task in tasks
        ),
        "equal_task_late_proxy_loss": _mean(
            per_task[str(task)]["late_proxy_loss"] for task in tasks
        ),
    }


def aggregate(rows: list[dict[str, Any]], rule: SelectionRule = DEFAULT_RULE) -> dict:
    """Aggregate one complete and sample-aligned checkpoint panel.

    Args:
        rows: Typed per-example losses for stock and every candidate.
        rule: Frozen tasks, stages, checkpoints, thresholds, and action width.

    Returns:
        Per-checkpoint, per-task, and equal-task metrics.

    Raises:
        ValueError: Rows are missing, duplicated, non-finite, or misaligned.
    """
    tasks = _validate_rule(rule)
    panels = _aligned_panels(rows, rule, tasks)
    return {
        step: _checkpoint_metrics(step, samples, tasks, rule)
        for step, samples in sorted(panels.items())
    }


def _candidate_eligibility(candidate, stock, tasks, rule):
    reasons = []
    for task in tasks:
        baseline = stock["per_task"][str(task)]
        observed = candidate["per_task"][str(task)]
        if observed["action_loss"] > rule.uniform_flow_ratio * baseline["action_loss"]:
            reasons.append(f"task{task}_uniform_flow_drift")
        if (
            observed["non_late_proxy_loss"]
            > rule.non_late_proxy_ratio * baseline["non_late_proxy_loss"]
        ):
            reasons.append(f"task{task}_non_late_drift")
    candidate["eligible"] = not reasons
    candidate["ineligibility_reasons"] = reasons
    candidate["late_proxy_improvement_over_stock"] = (
        stock["equal_task_late_proxy_loss"] - candidate["equal_task_late_proxy_loss"]
    )
    return not reasons


def _selection_receipt(chosen, eligible, metrics, rule):
    return {
        "schema": "npa.behavior.stock-anchored-offline-selection.v1",
        "status": "offline_checkpoint_selected_not_rollout_evaluated",
        "rule": "stock_no_drift_then_equal_task_late_proxy_improvement_then_earliest_step",
        "positive_late_improvement_required": False,
        "thresholds": {
            "uniform_flow_ratio": rule.uniform_flow_ratio,
            "non_late_proxy_ratio": rule.non_late_proxy_ratio,
        },
        "late_proxy_semantics": "last third of equal-time teacher-stage bins; descriptive, not contact labels",
        "selected_step": chosen["step"],
        "selected_label": chosen["label"],
        "fallback_used": not eligible,
        "checkpoints": [
            metrics[step] for step in (rule.stock_step, *rule.candidate_steps)
        ],
    }


def select(rows: list[dict[str, Any]], rule: SelectionRule = DEFAULT_RULE) -> dict:
    """Select an eligible checkpoint without using rollout outcomes.

    Args:
        rows: Complete typed stock and candidate loss rows.
        rule: Frozen selection registry and thresholds.

    Returns:
        Selection receipt with metrics and eligibility for every checkpoint.

    Raises:
        ValueError: The rule or any complete-panel input differs.
    """
    tasks = _validate_rule(rule)
    metrics = aggregate(rows, rule)
    stock = metrics[rule.stock_step]
    eligible = []
    for step in rule.candidate_steps:
        candidate = metrics[step]
        if _candidate_eligibility(candidate, stock, tasks, rule):
            eligible.append(candidate)
    chosen = (
        min(
            eligible,
            key=lambda row: (-row["late_proxy_improvement_over_stock"], row["step"]),
        )
        if eligible
        else stock
    )
    return _selection_receipt(chosen, eligible, metrics, rule)
