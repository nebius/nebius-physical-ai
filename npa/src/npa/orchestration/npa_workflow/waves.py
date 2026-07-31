"""Group a planned ``npa.workflow`` into execution *waves*.

A **wave** is the unit the runtime tier submits and waits on:

* ``serial`` — exactly one planned step (today's behaviour for every spec).
* ``parallel`` — the members of a ``parallel:`` group, launched concurrently as a
  SkyPilot JobGroup (chunked into batches when ``maxConcurrency`` is smaller than
  the group).

The static plan produced by :func:`npa.orchestration.npa_workflow.interpreter.build_plan`
stays flat and serial (so ``--plan-only`` output is unchanged); this module is the
lens that recovers the concurrent shape from ``PlanStep.group``. The runtime tier
consumes waves, and ``plan-spec --waves`` prints them for offline inspection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from npa.orchestration.npa_workflow.interpreter import ExecutionPlan, PlanStep, build_plan
from npa.orchestration.npa_workflow.spec import NpaWorkflowSpec, resolve_config_int

WAVE_SERIAL = "serial"
WAVE_PARALLEL = "parallel"


def split_into_batches(
    steps: Sequence[PlanStep], max_concurrency: int, *, cap: int = 0
) -> list[list[PlanStep]]:
    """Chunk steps into concurrency-bounded batches.

    Single source of truth for batching so the offline ``--waves`` preview and the
    runtime executor can never disagree about how a group is split. ``cap`` is the
    operator's optional ceiling (``--max-concurrency``); it can only *lower* the
    group's declared bound. A non-positive bound means "the whole group at once".
    """

    items = list(steps)
    if not items:
        return []
    limit = int(max_concurrency) if max_concurrency and int(max_concurrency) > 0 else len(items)
    if cap and int(cap) > 0:
        limit = min(limit, int(cap))
    limit = max(1, limit)
    return [items[start : start + limit] for start in range(0, len(items), limit)]


@dataclass(frozen=True)
class Wave:
    """One submit-and-wait unit."""

    index: int
    kind: str
    steps: tuple[PlanStep, ...]
    group: str = ""
    max_concurrency: int = 1

    @property
    def name(self) -> str:
        return self.group or (self.steps[0].state if self.steps else f"wave-{self.index}")

    def batches(self) -> list[list[PlanStep]]:
        """Split the wave into concurrency-bounded batches (submitted in order)."""

        return split_into_batches(self.steps, self.max_concurrency)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "kind": self.kind,
            "group": self.group,
            "name": self.name,
            "max_concurrency": self.max_concurrency,
            "batches": len(self.batches()),
            "steps": [
                {
                    "state": step.state,
                    "iteration": step.iteration,
                    "tool_ref": step.tool_ref,
                    "resources": step.resources,
                    "outputs": [item["uri"] for item in step.outputs],
                }
                for step in self.steps
            ],
        }


@dataclass
class WavePlan:
    workflow: str
    run_id: str
    assume_decision: str = ""
    waves: list[Wave] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow": self.workflow,
            "run_id": self.run_id,
            "assume_decision": self.assume_decision,
            "wave_count": len(self.waves),
            "parallel_waves": sum(1 for wave in self.waves if wave.kind == WAVE_PARALLEL),
            "waves": [wave.to_dict() for wave in self.waves],
        }


def group_max_concurrency(spec: NpaWorkflowSpec, group: str, member_count: int) -> int:
    """Resolve ``maxConcurrency`` for a group (defaults to the whole group)."""

    state = spec.states.get(group)
    if state is None or state.max_concurrency is None:
        return max(1, member_count)
    return max(1, resolve_config_int(state.max_concurrency, spec.config))


def waves_from_steps(spec: NpaWorkflowSpec, steps: Sequence[PlanStep]) -> list[Wave]:
    """Fold a flat step list into waves using ``PlanStep.group`` runs.

    A repeated member state inside the same group run starts a new wave: that only
    happens when a loop re-enters the group, and each loop iteration is its own
    submit-and-wait unit.
    """

    waves: list[Wave] = []
    buffer: list[PlanStep] = []
    buffer_group = ""
    seen_states: set[str] = set()

    def flush() -> None:
        nonlocal buffer, buffer_group, seen_states
        if not buffer:
            return
        if buffer_group:
            waves.append(
                Wave(
                    index=len(waves),
                    kind=WAVE_PARALLEL,
                    steps=tuple(buffer),
                    group=buffer_group,
                    max_concurrency=group_max_concurrency(spec, buffer_group, len(buffer)),
                )
            )
        else:
            for step in buffer:
                waves.append(
                    Wave(index=len(waves), kind=WAVE_SERIAL, steps=(step,), max_concurrency=1)
                )
        buffer = []
        buffer_group = ""
        seen_states = set()

    for step in steps:
        group = step.group or ""
        if group != buffer_group or (group and step.state in seen_states):
            flush()
            buffer_group = group
        buffer.append(step)
        if group:
            seen_states.add(step.state)
        else:
            flush()
    flush()
    return waves


def wave_plan_from_plan(spec: NpaWorkflowSpec, plan: ExecutionPlan, *, run_id: str) -> WavePlan:
    return WavePlan(
        workflow=plan.workflow,
        run_id=run_id,
        assume_decision=plan.assume_decision,
        waves=waves_from_steps(spec, plan.steps),
    )


def build_wave_plan(
    spec: NpaWorkflowSpec,
    *,
    run_id: str = "plan-run",
    assume_decision: str = "",
) -> WavePlan:
    """Plan a spec and return its wave shape (offline preview of runtime execution)."""

    plan = build_plan(spec, run_id=run_id, assume_decision=assume_decision)
    return wave_plan_from_plan(spec, plan, run_id=run_id)


__all__ = [
    "WAVE_PARALLEL",
    "WAVE_SERIAL",
    "Wave",
    "WavePlan",
    "build_wave_plan",
    "group_max_concurrency",
    "wave_plan_from_plan",
    "waves_from_steps",
]
