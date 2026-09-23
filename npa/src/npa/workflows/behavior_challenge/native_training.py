"""Run restartable native optimizer updates with durable milestone publication."""

from __future__ import annotations

import dataclasses
import math
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol


@dataclasses.dataclass(frozen=True)
class NativeTrainingPlan:
    """Describe one fixed native-update schedule.

    Args: final_step: Last logical update. milestones: Ordered durable steps.
    Returns: Immutable validated schedule.
    Raises: ValueError when chronology is invalid.
    """

    final_step: int
    milestones: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.final_step) is not int or self.final_step <= 0:
            raise ValueError("final_step must be positive")
        if any(type(value) is not int or value < 0 for value in self.milestones):
            raise ValueError("milestones must be nonnegative integers")
        if self.milestones != tuple(sorted(set(self.milestones))):
            raise ValueError("milestones must be unique and ordered")
        if not self.milestones or self.milestones[0] != 0:
            raise ValueError("milestones must start at released-parent step zero")
        if self.milestones[-1] != self.final_step:
            raise ValueError("the final milestone must equal final_step")


class NativeTrainingRuntime(Protocol):
    """Provide native loader, update, and checkpoint operations.

    Args: Implemented by a concrete scientific runtime.
    Returns: A structural protocol; it is not instantiated directly.
    Raises: Runtime implementations fail closed on contract drift.
    """

    state: Any
    cursor: Any

    def next_delivered_batch(self) -> tuple[Any, Mapping[str, Any]]:
        """Return the next batch and delivery identity.

        Args: None. Returns: Native batch and delivery record.
        Raises: ValueError when delivered order differs.
        """
        ...

    def train_step(self, state: Any, batch: Any) -> tuple[Any, Mapping[str, Any]]:
        """Execute one native update.

        Args: state: Current TrainState. batch: Native batch.
        Returns: Updated state and metrics. Raises: ValueError on drift.
        """
        ...

    def synchronize(self, value: Any) -> None:
        """Synchronize one native value.

        Args: value: Device value. Returns: None. Raises: Runtime backend errors.
        """
        ...

    def commit_cursor(self, delivery: Mapping[str, Any]) -> None:
        """Commit one delivered update.

        Args: delivery: Exact pending identity. Returns: None.
        Raises: ValueError when delivery differs.
        """
        ...

    def state_step(self) -> int:
        """Return TrainState step. Args: None. Returns: Step. Raises: None."""
        ...

    def cursor_step(self) -> int:
        """Return cursor step. Args: None. Returns: Step. Raises: None."""
        ...

    def durable_milestones(self) -> dict[str, Any]:
        """Return durable history. Args: None. Returns: History. Raises: None."""
        ...

    def released_parent_milestone(self) -> dict[str, Any]:
        """Return parent identity. Args: None. Returns: Identity. Raises: None."""
        ...

    def save_and_publish(self, root: Path, step: int) -> dict[str, Any]:
        """Publish a milestone.

        Args: root: Checkpoint root. step: Logical step.
        Returns: Provider result. Raises: ValueError on contract drift.
        """
        ...

    def retire_prior(self, step: int) -> dict[str, Any] | None:
        """Retire prior local state.

        Args: step: Current durable step. Returns: Retirement record or None.
        Raises: ValueError when prior state differs.
        """
        ...

    def runtime_record(self) -> dict[str, Any]:
        """Return runtime provenance. Args: None. Returns: Record. Raises: None."""
        ...


def _finite_metrics(values: Mapping[str, Any]) -> dict[str, float]:
    required = ("loss", "grad_norm", "param_norm", "learning_rate")
    metrics = {name: float(values[name]) for name in required}
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError("native training metric is non-finite")
    return metrics


def _verify_start(runtime: NativeTrainingRuntime, plan: NativeTrainingPlan) -> int:
    step = runtime.state_step()
    if step != runtime.cursor_step() or step not in plan.milestones:
        raise ValueError("training must start at a complete durable milestone")
    durable = runtime.durable_milestones()
    expected = [str(value) for value in plan.milestones if value <= step]
    if sorted(durable, key=int) != expected:
        raise ValueError("durable milestone history differs")
    if durable["0"] != runtime.released_parent_milestone():
        raise ValueError("released parent step-zero identity differs")
    return step


def _update(runtime: NativeTrainingRuntime, step: int, clock: Any) -> dict[str, Any]:
    loader_started = clock()
    batch, delivery = runtime.next_delivered_batch()
    loader_seconds = clock() - loader_started
    update_started = clock()
    new_state, raw_metrics = runtime.train_step(runtime.state, batch)
    runtime.synchronize((new_state, raw_metrics))
    metrics = _finite_metrics(raw_metrics)
    update_seconds = clock() - update_started
    runtime.state = new_state
    if runtime.state_step() != step:
        raise ValueError("native TrainState step differs after update")
    runtime.commit_cursor(delivery)
    if runtime.cursor_step() != step:
        raise ValueError("committed cursor differs after update")
    return {
        "step": step,
        **metrics,
        "loader_consumer_wait_seconds": loader_seconds,
        "synchronized_native_update_seconds": update_seconds,
    }


def _milestone(runtime: NativeTrainingRuntime, root: Path, step: int) -> dict[str, Any]:
    if runtime.state_step() != step or runtime.cursor_step() != step:
        raise ValueError("checkpoint state and cursor chronology differ")
    durable = runtime.save_and_publish(root, step)
    if durable.get("provider_manifest", {}).get("provider_readback") is not True:
        raise ValueError("milestone lacks complete provider readback")
    retirement = runtime.retire_prior(step)
    if retirement and retirement.get("status") == "cleanup_required":
        raise RuntimeError("milestone is durable but prior checkpoint cleanup failed")
    return {"durable": durable, "retirement": retirement}


def run_native_training(
    runtime: NativeTrainingRuntime,
    plan: NativeTrainingPlan,
    checkpoint_root: Path,
    *,
    clock: Any = time.monotonic,
) -> dict[str, Any]:
    """Run direct native updates and publish each declared milestone.

    Args: runtime: Concrete OpenPI adapter. plan: Fixed update schedule.
        checkpoint_root: Durable local root. clock: Monotonic timing function.
    Returns: Complete training receipt with durable milestones and metrics.
    Raises: ValueError for chronology/metric drift; RuntimeError for cleanup failure.
    """
    start = _verify_start(runtime, plan)
    records = runtime.durable_milestones()
    metrics: list[dict[str, Any]] = []
    started = clock()
    for step in range(start + 1, plan.final_step + 1):
        metrics.append(_update(runtime, step, clock))
        if step in plan.milestones[1:]:
            records[str(step)] = _milestone(runtime, checkpoint_root, step)
    return {
        "schema": "npa.behavior.comet-native-full-training.v1",
        "status": "native_updates_and_declared_milestones_durable",
        "start_logical_update": start,
        "logical_updates": plan.final_step,
        "milestones": list(plan.milestones),
        "manager_index_semantics": "manager_step_equals_logical_update_minus_one",
        "checkpoints": records,
        "hot_path_metrics": metrics,
        "runtime": runtime.runtime_record(),
        "elapsed_seconds": clock() - started,
        "selection_or_scoring_executed": False,
        "serving_export_qualified": False,
    }
