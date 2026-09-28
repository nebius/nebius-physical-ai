"""Read exact official TRAIN Q progress without changing evaluation state."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import inspect
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

SCHEMA = "npa.behavior.train-official-q-row.v1"
OFFICIAL_Q_SOURCE = {
    "bddl/condition_evaluation.py": {
        "bytes": 31046,
        "sha256": "e4e7c236f49a871b576666ccada4bc6e31b18c50f25361364de56bd7de013cb8",
    },
    "omnigibson/eval/evaluator.py": {
        "bytes": 34580,
        "sha256": "7c66958717945b7c2f618cb6d779828ebcd945c4c4968193360d787f576cb4e5",
    },
    "omnigibson/metrics/task_metric.py": {
        "bytes": 3414,
        "sha256": "9f9c20fa5c3b8e57c46d81e04e71eb2b63c007809a43631a6b4fe3c791ceef8e",
    },
    "omnigibson/tasks/behavior_task.py": {
        "bytes": 43560,
        "sha256": "0daeda0263295b84ac703fdd7561571c0fd65661c090c9ebae9f9f802aca58c7",
    },
}
_ROW_FIELDS = {
    "schema",
    "frame_index",
    "initial_satisfied_options",
    "current_satisfied_options",
    "success",
    "q_score",
    "source",
}


def _file_identity(path: Path) -> dict[str, int | str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Official Q source member differs")
    payload = path.read_bytes()
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _source_identity(objects: Sequence[object]) -> dict[str, dict[str, int | str]]:
    observed = {}
    for item, suffix in zip(objects, OFFICIAL_Q_SOURCE, strict=True):
        path = Path(inspect.getsourcefile(item) or "")
        if not path.as_posix().endswith(suffix):
            raise ValueError("Official Q source path differs")
        observed[suffix] = _file_identity(path)
    if observed != OFFICIAL_Q_SOURCE:
        raise ValueError("Official Q source bytes differ")
    return deepcopy(OFFICIAL_Q_SOURCE)


def official_q_source() -> dict[str, dict[str, int | str]]:
    """Validate and return the exact official v3.9.3 Q implementation identity."""
    from bddl.condition_evaluation import evaluate_state
    from omnigibson.eval.evaluator import BatchedEvaluator
    from omnigibson.metrics.task_metric import compute_q_score
    from omnigibson.tasks.behavior_task import BehaviorTask

    return _source_identity(
        (evaluate_state, BatchedEvaluator, compute_q_score, BehaviorTask)
    )


def _masks(value: object, name: str) -> list[list[bool]]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"Official Q {name} options differ")
    result = []
    for option in value:
        if not isinstance(option, (list, tuple)) or not option:
            raise ValueError(f"Official Q {name} option differs")
        if any(type(item) is not bool for item in option):
            raise ValueError(f"Official Q {name} predicate differs")
        result.append(list(option))
    return result


def _same_shape(
    initial: Sequence[Sequence[bool]], current: Sequence[Sequence[bool]]
) -> None:
    if len(initial) != len(current) or any(
        len(left) != len(right) for left, right in zip(initial, current, strict=True)
    ):
        raise ValueError("Official Q option shape changed within the rollout")


def _score(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Official Q score type differs")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("Official Q score value differs")
    return result


def recompute_official_q(
    *, success, now_satisfied_options, initial_satisfied_options
) -> float:
    """Recompute the pinned TaskMetric formula without importing the simulator."""
    if success:
        return 1.0
    if not now_satisfied_options:
        return 0.0
    scores = []
    for current, initial in zip(
        now_satisfied_options, initial_satisfied_options, strict=True
    ):
        scores.append(
            sum(not before and after for before, after in zip(initial, current))
            / len(current)
            if current
            else 0.0
        )
    return max(scores) if scores else 0.0


class OfficialQObserver:
    """Snapshot one bound TaskMetric after each unchanged official apply call."""

    def __init__(
        self, metric: Any, compute: Callable[..., float], source: Mapping[str, Any]
    ):
        self.metric = metric
        self.compute = compute
        self.source = deepcopy(dict(source))

    def observe(self, frame_index: int) -> dict[str, Any]:
        if type(frame_index) is not int or frame_index < 0:
            raise ValueError("Official Q frame index differs")
        initial = _masks(
            getattr(self.metric, "initial_predicate_states", None), "initial"
        )
        accessor = getattr(self.metric, "env_accessor", None)
        if accessor is None or not callable(
            getattr(accessor, "get_goal_option_satisfaction", None)
        ):
            raise ValueError("Official Q environment accessor differs")
        current = _masks(accessor.get_goal_option_satisfaction(), "current")
        _same_shape(initial, current)
        success = accessor.success
        if type(success) is not bool:
            raise ValueError("Official Q success value differs")
        q_score = _score(
            self.compute(
                success=success,
                now_satisfied_options=current,
                initial_satisfied_options=initial,
            )
        )
        return {
            "schema": SCHEMA,
            "frame_index": frame_index,
            "initial_satisfied_options": initial,
            "current_satisfied_options": current,
            "success": success,
            "q_score": q_score,
            "source": deepcopy(self.source),
        }


def bind_official_q_observer(evaluator: Any) -> OfficialQObserver:
    """Bind the one official TaskMetric owned by a one-environment TRAIN evaluator."""
    from omnigibson.metrics.task_metric import TaskMetric, compute_q_score

    states = getattr(evaluator, "instance_eval_states", None)
    if not isinstance(states, list) or len(states) != 1:
        raise ValueError("Official Q observer requires one evaluator environment")
    metrics = [metric for metric in states[0].metrics if type(metric) is TaskMetric]
    if len(metrics) != 1 or metrics[0].env_accessor is not states[0].env_accessor:
        raise ValueError("Official Q TaskMetric binding differs")
    return OfficialQObserver(metrics[0], compute_q_score, official_q_source())


def validate_official_q_rows(
    rows: Sequence[Mapping[str, Any]],
    compute: Callable[..., float],
    source: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Recompute every immutable row with the exact admitted Q function."""
    validated = []
    initial = None
    for frame, candidate in enumerate(rows):
        row, first = validate_official_q_row(
            candidate, frame=frame, initial=initial, compute=compute, source=source
        )
        if initial is None:
            initial = first
        validated.append(row)
    if not validated:
        raise ValueError("Official Q rows are empty")
    return validated


def validate_official_q_row(candidate, *, frame, initial, compute, source):
    """Validate one Q row against admitted source and rollout chronology."""
    if not isinstance(candidate, Mapping) or set(candidate) != _ROW_FIELDS:
        raise ValueError("Official Q row fields differ")
    first = _masks(candidate["initial_satisfied_options"], "initial")
    current = _masks(candidate["current_satisfied_options"], "current")
    _same_shape(first, current)
    if (
        (initial is not None and first != initial)
        or candidate["schema"] != SCHEMA
        or candidate["frame_index"] != frame
    ):
        raise ValueError("Official Q row chronology differs")
    if candidate["source"] != source or type(candidate["success"]) is not bool:
        raise ValueError("Official Q row authority differs")
    expected = _score(
        compute(
            success=candidate["success"],
            now_satisfied_options=current,
            initial_satisfied_options=first,
        )
    )
    if _score(candidate["q_score"]) != expected:
        raise ValueError("Official Q row score differs")
    return deepcopy(dict(candidate)), first


def validate_official_q_terminal(
    rows: Sequence[Mapping[str, Any]], result: Mapping[Any, Any], instance_id: int
) -> None:
    """Require the last observed Q to equal the unchanged official result."""
    if set(result) != {instance_id} or not rows:
        raise ValueError("Official Q terminal result differs")
    official = result[instance_id]
    q_score = (
        official.get("q_score", {}).get("final")
        if isinstance(official, Mapping)
        else None
    )
    success = official.get("success") if isinstance(official, Mapping) else None
    if type(success) is not bool or rows[-1].get("success") is not success:
        raise ValueError("Official Q terminal success differs")
    if _score(rows[-1].get("q_score")) != _score(q_score):
        raise ValueError("Official Q terminal score differs")


__all__ = [
    "OFFICIAL_Q_SOURCE",
    "OfficialQObserver",
    "bind_official_q_observer",
    "official_q_source",
    "recompute_official_q",
    "validate_official_q_row",
    "validate_official_q_rows",
    "validate_official_q_terminal",
]
