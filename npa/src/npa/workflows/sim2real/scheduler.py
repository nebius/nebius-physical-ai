"""Declarative DAG scheduler for the Sim2Real staged workflow.

This is the "true YAML" execution path. The workflow's *shape* — stage
dependencies, the outer loop with its conditional ``promote_checkpoint``
early-exit, and the fixed-count inner loop — lives as declarative data in
``sim2real.dag.yaml`` instead of as hard-coded Python control flow in
``runner.run_staged()``.

Stage *implementations are unchanged*: each executable node maps to an existing
``Sim2RealWorkflow`` entrypoint (``run_preamble`` → preamble stages 1-6,
``run_outer_iteration`` → inner loop 7-9 ×N + heldout 10 + decision 11,
``run_finalize`` → finalize 12-14). Because the scheduler drives the same
methods in the same order as ``run_staged()``, the two paths produce identical
stage records — see ``tests/workflows/test_sim2real_scheduler.py``.

The scheduler is opt-in behind ``sim2real run --dag <spec>``; the default path
remains ``run_staged()``.

Design rules honoured here:

* No ``eval``. Loop bounds resolve from named config fields (``config.<attr>``)
  or integer literals; the ``until`` condition resolves from a small registry of
  named predicates (currently ``promote_checkpoint``).
* Per-node failure aggregates into the process exit code: ``run_dag`` raises, the
  same way ``run_staged`` raises, so the CLI returns non-zero (a failure recorded
  in the JSON report is never a substitute for a non-zero exit).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from npa.workflows.sim2real.models import Sim2RealLoopConfig, Sim2RealLoopError
from npa.workflows.sim2real.runner import Sim2RealWorkflow
from npa.workflows.sim2real.state import WorkflowState

DAG_SPEC_SCHEMA = "npa.workflow/v1"

# Default spec shipped with the repo, resolved relative to the workbench tree.
# __file__ = <repo>/npa/src/npa/workflows/sim2real/scheduler.py; parents[4] = <repo>/npa,
# which is where the workbench workflow YAMLs (runbook.yaml, sim2real.dag.yaml) live.
DEFAULT_DAG_SPEC = (
    Path(__file__).resolve().parents[4]
    / "workflows"
    / "workbench"
    / "sim2real"
    / "sim2real.dag.yaml"
)


class DagSpecError(Sim2RealLoopError):
    """Raised when the DAG spec is malformed (unknown executor, cycle, ...)."""


@dataclass
class LoopSpec:
    max_iterations: Any = None  # int literal or "config.<attr>"
    until: str | None = None  # named predicate, e.g. "promote_checkpoint"


@dataclass
class NodeSpec:
    id: str
    executor: str
    needs: list[str] = field(default_factory=list)
    loop: LoopSpec | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class DagSpec:
    name: str
    nodes: list[NodeSpec]
    api_version: str = DAG_SPEC_SCHEMA

    def node_ids(self) -> list[str]:
        return [n.id for n in self.nodes]


# --------------------------------------------------------------------------- #
# Spec loading + validation
# --------------------------------------------------------------------------- #
def load_spec(path: str | Path) -> DagSpec:
    """Load and validate a DAG spec from YAML."""

    import yaml

    spec_path = Path(path)
    if not spec_path.is_file():
        raise DagSpecError(f"DAG spec not found: {spec_path}")
    data = yaml.safe_load(spec_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise DagSpecError(f"DAG spec must be a mapping, got {type(data).__name__}")

    raw_nodes = data.get("nodes") or data.get("stages") or []
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise DagSpecError("DAG spec must declare a non-empty 'nodes' list")

    nodes: list[NodeSpec] = []
    for entry in raw_nodes:
        if not isinstance(entry, dict) or "id" not in entry:
            raise DagSpecError(f"each node needs an 'id': {entry!r}")
        loop_raw = entry.get("loop")
        loop = None
        if loop_raw is not None:
            if not isinstance(loop_raw, dict):
                raise DagSpecError(f"node {entry['id']}: 'loop' must be a mapping")
            loop = LoopSpec(
                max_iterations=loop_raw.get("max_iterations", loop_raw.get("repeat")),
                until=loop_raw.get("until"),
            )
        nodes.append(
            NodeSpec(
                id=str(entry["id"]),
                executor=str(entry.get("executor", entry["id"])),
                needs=list(entry.get("needs") or []),
                loop=loop,
                raw=entry,
            )
        )

    spec = DagSpec(
        name=str(data.get("name", "sim2real-staged-loop")),
        nodes=nodes,
        api_version=str(data.get("apiVersion", DAG_SPEC_SCHEMA)),
    )
    validate_spec(spec)
    return spec


def validate_spec(spec: DagSpec) -> None:
    """Validate node ids are unique, executors known, needs resolve, acyclic."""

    seen: set[str] = set()
    for node in spec.nodes:
        if node.id in seen:
            raise DagSpecError(f"duplicate node id: {node.id}")
        seen.add(node.id)
        if node.executor not in NODE_EXECUTORS:
            raise DagSpecError(
                f"node {node.id}: unknown executor {node.executor!r} "
                f"(known: {sorted(NODE_EXECUTORS)})"
            )
        if node.loop and node.loop.until and node.loop.until not in UNTIL_PREDICATES:
            raise DagSpecError(
                f"node {node.id}: unknown until predicate {node.loop.until!r} "
                f"(known: {sorted(UNTIL_PREDICATES)})"
            )
    for node in spec.nodes:
        for dep in node.needs:
            if dep not in seen:
                raise DagSpecError(f"node {node.id}: needs unknown node {dep!r}")
    topological_order(spec)  # raises on cycle


def topological_order(spec: DagSpec) -> list[NodeSpec]:
    """Return nodes in dependency order (Kahn's algorithm); raise on a cycle."""

    by_id = {n.id: n for n in spec.nodes}
    indegree = {n.id: 0 for n in spec.nodes}
    dependents: dict[str, list[str]] = {n.id: [] for n in spec.nodes}
    for node in spec.nodes:
        for dep in node.needs:
            indegree[node.id] += 1
            dependents[dep].append(node.id)

    # Preserve declaration order among ready nodes for deterministic execution.
    ready = [n.id for n in spec.nodes if indegree[n.id] == 0]
    ordered: list[NodeSpec] = []
    while ready:
        current = ready.pop(0)
        ordered.append(by_id[current])
        for nxt in dependents[current]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                ready.append(nxt)
    if len(ordered) != len(spec.nodes):
        remaining = sorted(set(by_id) - {n.id for n in ordered})
        raise DagSpecError(f"DAG spec has a cycle among nodes: {remaining}")
    return ordered


# --------------------------------------------------------------------------- #
# Resolution helpers (no eval)
# --------------------------------------------------------------------------- #
def resolve_int(value: Any, config: Sim2RealLoopConfig) -> int:
    """Resolve a loop bound: integer literal or ``config.<attr>`` reference."""

    if isinstance(value, bool):  # guard: bool is an int subclass
        raise DagSpecError(f"loop bound must be int or config ref, got {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("config."):
            attr = text[len("config.") :]
            if not hasattr(config, attr):
                raise DagSpecError(f"config has no attribute {attr!r}")
            return int(getattr(config, attr))
        if text.isdigit():
            return int(text)
    raise DagSpecError(f"cannot resolve loop bound {value!r}")


UNTIL_PREDICATES: dict[str, Callable[[WorkflowState], bool]] = {
    "promote_checkpoint": lambda state: state.should_promote(),
}


# --------------------------------------------------------------------------- #
# Execution context + node executors
# --------------------------------------------------------------------------- #
@dataclass
class SchedulerContext:
    workflow: Sim2RealWorkflow
    upload: bool | None = None
    initial_quality: float | None = None
    state: WorkflowState | None = None
    report: dict[str, Any] | None = None


def _exec_preamble(ctx: SchedulerContext, node: NodeSpec) -> None:
    """Run preamble (stages 1-6) or resume from persisted state.

    Mirrors the head of ``runner.run_staged()`` exactly.
    """

    from npa.workflows.sim2real.engine import _config_from_workflow_state

    workflow = ctx.workflow
    state_path = WorkflowState.path_for(workflow.local_dir)
    if not state_path.exists():
        state = workflow.run_preamble()
        workflow.config = _config_from_workflow_state(
            workflow.config, state.to_payload()
        )
    else:
        state = WorkflowState.load(workflow.local_dir)
    if ctx.initial_quality is not None:
        state.current_quality = float(ctx.initial_quality)
        state.save()
    ctx.state = state


def _exec_outer_iteration(ctx: SchedulerContext, node: NodeSpec) -> None:
    """Run the outer loop (inner loop ×N + heldout + decision per pass).

    Loop bounds and the early-exit predicate come from the spec, mirroring the
    ``for outer_iteration in range(start, outer_iterations + 1)`` loop in
    ``run_staged()``.
    """

    if ctx.state is None:
        raise DagSpecError(f"node {node.id}: outer loop requires a preamble node first")
    workflow = ctx.workflow
    loop = node.loop or LoopSpec(max_iterations="config.outer_iterations")
    max_iterations = resolve_int(loop.max_iterations, workflow.config)
    until = UNTIL_PREDICATES[loop.until] if loop.until else (lambda _s: False)

    start = ctx.state.next_outer_iteration
    for outer_iteration in range(start, max_iterations + 1):
        ctx.state = workflow.run_outer_iteration(outer_iteration=outer_iteration)
        if until(ctx.state):
            break


def _exec_finalize(ctx: SchedulerContext, node: NodeSpec) -> None:
    """Run finalize (stages 12-14). Mirrors the tail of ``run_staged()``."""

    if ctx.state is None:
        raise DagSpecError(f"node {node.id}: finalize requires earlier nodes to run")
    workflow = ctx.workflow
    if ctx.state.status != "completed":
        ctx.report = workflow.run_finalize(upload=ctx.upload)
        return
    from npa.workflows.sim2real.engine import run_finalize

    ctx.report = run_finalize(
        workflow.config,
        local_dir=workflow.local_dir,
        stage_records=list(ctx.state.stage_records),
        components=list(ctx.state.components),
        outer_history=list(ctx.state.outer_history),
        final_inner=dict(ctx.state.final_inner or {}),
        final_eval=dict(ctx.state.final_eval or {}),
        final_decision=dict(ctx.state.final_decision or {}),
        upload=ctx.upload,
    )


NODE_EXECUTORS: dict[str, Callable[[SchedulerContext, NodeSpec], None]] = {
    "preamble": _exec_preamble,
    "outer_iteration": _exec_outer_iteration,
    "finalize": _exec_finalize,
}


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def run_dag(
    workflow: Sim2RealWorkflow,
    spec: DagSpec,
    *,
    upload: bool | None = None,
    initial_quality: float | None = None,
) -> dict[str, Any]:
    """Execute the DAG spec by driving ``workflow``'s stage methods.

    Returns the finalize report (same shape as ``run_staged``). Raises if any
    node fails, so the caller's exit code reflects the real outcome.
    """

    ordered = topological_order(spec)
    ctx = SchedulerContext(
        workflow=workflow, upload=upload, initial_quality=initial_quality
    )
    for node in ordered:
        executor = NODE_EXECUTORS[node.executor]
        executor(ctx, node)
    if ctx.report is None:
        raise DagSpecError(
            "DAG completed without a finalize node producing a report; "
            "spec must include a node with executor 'finalize'"
        )
    return ctx.report
