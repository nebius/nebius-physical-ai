"""Build execution plans and run NPA workflow state machines."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from typing import Any, Callable

from npa.orchestration.npa_workflow.artifacts import require_input_artifacts
from npa.orchestration.npa_workflow.catalog import (
    argv_for_tool,
    drop_empty_optional_flags,
)
from npa.orchestration.npa_workflow.decisions import (
    load_decision,
    normalize_decision,
    refresh_context_decision,
)
from npa.orchestration.npa_workflow.errors import NpaWorkflowError
from npa.orchestration.npa_workflow.predicates import evaluate_predicate
from npa.orchestration.npa_workflow.run_state import (
    RunManifest,
    RunStateStore,
    store_for_config,
)
from npa.orchestration.npa_workflow.spec import (
    NpaWorkflowSpec,
    StateSpec,
    config_truthy,
    resolve_config_int,
    resolve_resource_profile,
)
from npa.orchestration.npa_workflow.tokens import resolve_tokens


@dataclass
class PlanStep:
    state: str
    iteration: int | None = None
    loop_label: str = ""
    argv: list[str] = field(default_factory=list)
    shell: str = ""
    tool_ref: str = ""
    resources: str = "default"
    resources_profile: dict[str, Any] = field(default_factory=dict)
    outputs: list[dict[str, str]] = field(default_factory=list)
    inputs: list[dict[str, str]] = field(default_factory=list)
    #: Name of the ``parallel:`` group this step belongs to ("" for serial steps).
    group: str = ""


@dataclass
class ExecutionPlan:
    workflow: str
    api_version: str
    initial: str
    assume_decision: str = ""
    steps: list[PlanStep] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow": self.workflow,
            "api_version": self.api_version,
            "initial": self.initial,
            "assume_decision": self.assume_decision,
            "steps": [
                {
                    "state": step.state,
                    "iteration": step.iteration,
                    "loop_label": step.loop_label,
                    "argv": step.argv,
                    "shell": step.shell,
                    "tool_ref": step.tool_ref,
                    "resources": step.resources,
                    "resources_profile": step.resources_profile,
                    "outputs": step.outputs,
                    "inputs": step.inputs,
                    "group": step.group,
                }
                for step in self.steps
            ],
        }


@dataclass
class RunContext:
    config: dict[str, Any]
    run: dict[str, Any]
    last_decision: str = ""
    state_outputs: dict[str, dict[str, str]] = field(default_factory=dict)
    loop_iterations: dict[str, int] = field(default_factory=dict)
    outer_iteration: int = 0
    inner_iteration: int = 0

    def as_predicate_context(self) -> dict[str, Any]:
        return {
            "last_decision": self.last_decision,
            "outer_iteration": self.outer_iteration,
            "inner_iteration": self.inner_iteration,
            "config": self.config,
            "run": self.run,
        }


def _resolve_assume(spec: NpaWorkflowSpec, assume_decision: str | None) -> str:
    """Normalize the planning branch assumption.

    Only loop-free specs get an empty assumption. Specs with dynamic transitions
    still default to ``loop_back`` so their loops expand; loop-free specs no
    longer report a spurious ``loop_back_to_inner_loop``.
    """

    raw_assume = (assume_decision or "").strip() or str(
        spec.config.get("plan_assume_decision") or ""
    )
    if not raw_assume and any(state.transitions for state in spec.states.values()):
        raw_assume = "loop_back"
    return normalize_decision(raw_assume)


def build_plan(
    spec: NpaWorkflowSpec,
    *,
    run_id: str = "plan-run",
    assume_decision: str = "",
) -> ExecutionPlan:
    assume = _resolve_assume(spec, assume_decision)
    ctx = _make_context(spec, run_id=run_id)
    plan = ExecutionPlan(
        workflow=spec.name,
        api_version=spec.api_version,
        initial=spec.initial,
        assume_decision=assume,
    )
    _expand_state(spec, spec.initial, ctx, plan, assume_decision=assume)
    return plan


def run_workflow(
    spec: NpaWorkflowSpec,
    *,
    run_id: str,
    execute: bool = False,
    assume_decision: str = "",
    persist_state: bool = False,
    require_inputs: bool = False,
    on_step: Callable[[PlanStep], None] | None = None,
    decision_reader: Any | None = None,
    artifact_checker: Any | None = None,
    state_store: RunStateStore | None = None,
    step_executor: Any | None = None,
    trigger_waiter: Callable[[StateSpec, str, "RunContext"], dict[str, Any]]
    | None = None,
) -> dict[str, Any]:
    """Plan or execute a workflow.

    ``step_executor`` swaps out *where* a step runs without changing the traversal:
    the default runs the step locally with ``subprocess``; the runtime tier
    (``runtime.SkyPilotWaveExecutor``) submits it to SkyPilot and waits for a
    terminal status. ``trigger_waiter`` blocks before a state whose ``trigger:``
    prefix has not produced data yet.
    """
    assume = _resolve_assume(spec, assume_decision)
    ctx = _make_context(spec, run_id=run_id)
    store = state_store
    if store is None and persist_state:
        store = store_for_config(ctx.config, run_id=run_id)
        if store is None:
            raise NpaWorkflowError(
                "persist_state requires config.bucket to be set in the workflow spec"
            )
    from npa.orchestration.npa_workflow.run_state import input_source_from_config

    manifest = RunManifest(
        workflow=spec.name,
        run_id=run_id,
        api_version=spec.api_version,
        status="running" if execute else "planned",
        input_source=input_source_from_config(ctx.config),
    )
    if store is not None:
        try:
            store.write_manifest(manifest)
        except Exception as exc:
            raise NpaWorkflowError(
                f"failed to persist workflow manifest: {exc}"
            ) from exc

    results: list[dict[str, Any]] = []
    status = "planned"
    error: NpaWorkflowError | None = None
    try:
        if execute:
            _execute_state_machine(
                spec,
                spec.initial,
                ctx,
                assume_decision=assume,
                require_inputs=require_inputs,
                on_step=on_step,
                decision_reader=decision_reader,
                artifact_checker=artifact_checker,
                step_executor=step_executor,
                trigger_waiter=trigger_waiter,
                results_out=results,
            )
            status = "completed"
        else:
            plan = build_plan(spec, run_id=run_id, assume_decision=assume)
            for step in plan.steps:
                if on_step is not None:
                    on_step(step)
                results.append(_execute_step(step, execute=False))
            status = "planned"
    except NpaWorkflowError as exc:
        status = "failed"
        error = exc
    finally:
        if store is not None:
            manifest.status = status
            manifest.steps = results
            try:
                store.write_manifest(manifest)
            except Exception as exc:
                persist_error = NpaWorkflowError(
                    f"failed to persist workflow manifest: {exc}"
                )
                if error is None:
                    error = persist_error
                    status = "failed"

    if error is None:
        plan = build_plan(spec, run_id=run_id, assume_decision=assume)
    else:
        plan = ExecutionPlan(
            workflow=spec.name,
            api_version=spec.api_version,
            initial=spec.initial,
            assume_decision=assume,
        )
    report = {
        "workflow": spec.name,
        "run_id": run_id,
        "status": status,
        "steps": results,
        "plan": plan.to_dict(),
        "run_prefix_uri": store.run_prefix_uri if store is not None else "",
    }
    if error is not None:
        raise error
    return report


def _make_context(spec: NpaWorkflowSpec, *, run_id: str) -> RunContext:
    run = {"id": run_id, "prefix": f"{spec.name}/{run_id}", **dict(spec.run_defaults)}
    run["id"] = run_id
    config_with_tool_defaults = dict(spec.config)
    from npa.orchestration.npa_workflow.catalog import config_defaults_for_tool

    for state in spec.states.values():
        if not state.tool_ref:
            continue
        for key, value in config_defaults_for_tool(state.tool_ref).items():
            config_with_tool_defaults.setdefault(key, value)
    config = _resolve_config_strings(config_with_tool_defaults, run=run)
    if config.get("prefix"):
        run["prefix"] = resolve_tokens(str(config["prefix"]), config=config, run=run)
    return RunContext(config=config, run=run)


def _resolve_config_strings(
    config: dict[str, Any], *, run: dict[str, Any]
) -> dict[str, Any]:
    resolved: dict[str, Any] = dict(config)
    for _ in range(4):
        changed = False
        for key, value in list(resolved.items()):
            if isinstance(value, str) and "{{" in value:
                new_value = resolve_tokens(value, config=resolved, run=run)
                if new_value != value:
                    resolved[key] = new_value
                    changed = True
        if not changed:
            break
    return resolved


def _expand_state(
    spec: NpaWorkflowSpec,
    state_name: str,
    ctx: RunContext,
    plan: ExecutionPlan,
    *,
    assume_decision: str,
    loop_label: str = "",
    follow_transitions: bool = True,
) -> None:
    state = spec.states[state_name]
    _guard_plan_size(spec, plan)

    if state.parallel:
        # Plan-time preview flattens a parallel group into declared order, exactly
        # like `sequence:`. The concurrent shape lives in the wave plan
        # (`waves.build_wave_plan`) and is executed by the runtime tier; keeping the
        # static plan serial means `--plan-only` output (and every existing
        # plan-only guardrail) is unchanged for serial and parallel specs alike.
        for member in state.parallel:
            _append_state_step(
                spec,
                spec.states[member],
                ctx,
                plan,
                loop_label=state.name,
                group=state.name,
            )
            _guard_plan_size(spec, plan)
        if state.next:
            _expand_state(spec, state.next, ctx, plan, assume_decision=assume_decision)
        return

    if state.sequence:
        if state.loop:
            max_iter = resolve_config_int(state.loop.max or 1, ctx.config)
            for iteration in range(1, max_iter + 1):
                ctx.outer_iteration = iteration
                ctx.loop_iterations[state.name] = iteration
                ctx.last_decision = ""
                for child in state.sequence:
                    _expand_state(
                        spec,
                        child,
                        ctx,
                        plan,
                        assume_decision=assume_decision,
                        loop_label=state.name,
                        follow_transitions=False,
                    )
                ctx.last_decision = assume_decision
                if state.loop.until and evaluate_predicate(
                    state.loop.until, ctx.as_predicate_context()
                ):
                    break
            if state.next:
                _expand_state(
                    spec,
                    state.next,
                    ctx,
                    plan,
                    assume_decision=assume_decision,
                )
            return

        for child in state.sequence:
            _expand_state(
                spec,
                child,
                ctx,
                plan,
                assume_decision=assume_decision,
                loop_label=state.name,
                follow_transitions=False,
            )
        if state.next:
            _expand_state(spec, state.next, ctx, plan, assume_decision=assume_decision)
        return

    if state.loop:
        max_iter = resolve_config_int(state.loop.max or 1, ctx.config)
        for iteration in range(1, max_iter + 1):
            ctx.inner_iteration = iteration
            ctx.loop_iterations[state.name] = iteration
            _append_state_step(
                spec, state, ctx, plan, iteration=iteration, loop_label=loop_label
            )
            ctx.last_decision = assume_decision
            if state.loop.until and evaluate_predicate(
                state.loop.until, ctx.as_predicate_context()
            ):
                break
        next_name = _resolve_transition(state, ctx) or state.next
        if next_name:
            _expand_state(
                spec,
                next_name,
                ctx,
                plan,
                assume_decision=assume_decision,
            )
        return

    _append_state_step(spec, state, ctx, plan, loop_label=loop_label)
    if state.terminal:
        return
    ctx.last_decision = assume_decision if state.transitions else ctx.last_decision
    next_name = ""
    if follow_transitions:
        next_name = _resolve_transition(state, ctx) or state.next
    elif not state.transitions:
        next_name = state.next
    if next_name:
        _expand_state(
            spec,
            next_name,
            ctx,
            plan,
            assume_decision=assume_decision,
        )


def state_config(state: StateSpec, ctx: RunContext) -> dict[str, Any]:
    """Return the config mapping used to resolve one state's tokens.

    ``params`` on a state is a *config overlay* scoped to that state, so N members
    of a ``parallel:`` sweep can share one ``toolRef`` argv template and still
    differ (learning rate, output prefix, ...). Overlay values may themselves use
    ``{{config.*}}`` / ``{{run.*}}`` tokens, resolved against the base config.
    """

    if not state.params:
        return ctx.config
    overlay = dict(ctx.config)
    for key, value in state.params.items():
        if isinstance(value, str):
            overlay[key] = resolve_tokens(
                value,
                config=ctx.config,
                run=ctx.run,
                state_outputs=ctx.state_outputs,
                loop_iterations=ctx.loop_iterations,
            )
        else:
            overlay[key] = value
    return overlay


def _guard_plan_size(spec: NpaWorkflowSpec, plan: ExecutionPlan) -> None:
    limit = _execution_step_limit(spec)
    if len(plan.steps) >= limit:
        raise NpaWorkflowError(
            "plan exceeded step limit; check for unbounded control-flow cycles"
        )


def _execution_step_limit(spec: NpaWorkflowSpec) -> int:
    return max(256, len(spec.states) * 64)


def _guard_execution_depth(spec: NpaWorkflowSpec, depth: int) -> None:
    if depth >= _execution_step_limit(spec):
        raise NpaWorkflowError(
            "execution exceeded step limit; check for unbounded control-flow cycles"
        )


def _sequence_decision_writer(
    spec: NpaWorkflowSpec, state: StateSpec
) -> StateSpec | None:
    for child in reversed(state.sequence):
        candidate = spec.states.get(child)
        if candidate is not None and candidate.writes_decision:
            return candidate
    return None


def _append_state_step(
    spec: NpaWorkflowSpec,
    state: StateSpec,
    ctx: RunContext,
    plan: ExecutionPlan,
    *,
    iteration: int | None = None,
    loop_label: str = "",
    group: str = "",
) -> None:
    plan.steps.append(
        build_step(
            spec,
            state,
            ctx,
            iteration=iteration,
            loop_label=loop_label,
            group=group,
        )
    )
    _record_state_outputs(state, ctx, plan.steps[-1])


def build_step(
    spec: NpaWorkflowSpec,
    state: StateSpec,
    ctx: RunContext,
    *,
    iteration: int | None = None,
    loop_label: str = "",
    group: str = "",
) -> PlanStep:
    """Materialize one planned step (tokens resolved, params overlay applied)."""

    argv, shell, tool_ref = _resolved_run(state, ctx)
    config = state_config(state, ctx)
    outputs = [
        {
            "uri": resolve_tokens(
                artifact.uri,
                config=config,
                run=ctx.run,
                state_outputs=ctx.state_outputs,
                loop_iterations=ctx.loop_iterations,
            ),
            "schema": artifact.schema,
        }
        for artifact in state.outputs
        if artifact.uri
    ]
    return PlanStep(
        state=state.name,
        iteration=iteration,
        loop_label=loop_label,
        argv=argv,
        shell=shell,
        tool_ref=tool_ref,
        resources=state.resources,
        resources_profile=_resources_profile(
            spec,
            state.resources,
            config=config,
            run=ctx.run,
            state_outputs=ctx.state_outputs,
            loop_iterations=ctx.loop_iterations,
        ),
        outputs=outputs,
        inputs=_resolved_inputs(state, ctx),
        group=group,
    )


def _resources_profile(
    spec: NpaWorkflowSpec,
    profile: str,
    *,
    config: dict[str, Any],
    run: dict[str, Any],
    state_outputs: dict[str, dict[str, str]],
    loop_iterations: dict[str, int],
) -> dict[str, Any]:
    raw = spec.resources.get(profile) or spec.resources.get("default") or {}
    if not isinstance(raw, dict):
        return {}
    return resolve_resource_profile(
        profile,
        raw,
        config=config,
        run=run,
        state_outputs=state_outputs,
        loop_iterations=loop_iterations,
    )


def _resolved_inputs(state: StateSpec, ctx: RunContext) -> list[dict[str, str]]:
    config = state_config(state, ctx)
    return [
        {
            "uri": resolve_tokens(
                artifact.uri,
                config=config,
                run=ctx.run,
                state_outputs=ctx.state_outputs,
                loop_iterations=ctx.loop_iterations,
            ),
            "schema": artifact.schema,
        }
        for artifact in state.inputs
        if artifact.uri
    ]


def _record_state_outputs(state: StateSpec, ctx: RunContext, step: PlanStep) -> None:
    if not step.outputs:
        return
    ctx.state_outputs[state.name] = {
        f"output_{index}": output["uri"]
        for index, output in enumerate(step.outputs, start=1)
    }
    primary = step.outputs[0]["uri"]
    ctx.state_outputs[state.name]["uri"] = primary


def _refresh_decision(
    ctx: RunContext,
    *,
    state: StateSpec | None = None,
    reader: Any | None = None,
    read_s3: bool = False,
) -> None:
    if read_s3:
        uri = _resolved_state_decision_uri(state, ctx) if state is not None else ""
        if uri:
            ctx.last_decision = load_decision(uri, reader=reader)
        else:
            ctx.last_decision = refresh_context_decision(
                ctx.as_predicate_context(), reader=reader
            )


def _resolved_state_decision_uri(state: StateSpec, ctx: RunContext) -> str:
    """Return the exact decision artifact resolved for one executed state.

    A dynamic loop may override ``decision_uri`` with ``{{loop.*}}`` in the
    decision writer's params. Reading only the workflow-wide config after the
    state completes silently points at a legacy or previous-iteration object.
    Prefer the already-resolved declared decision output, including when it is
    not the state's first output, then fall back to the state-scoped config.
    """

    outputs = ctx.state_outputs.get(state.name) or {}
    for index, artifact in enumerate(state.outputs, start=1):
        if "decision" not in str(artifact.schema).lower():
            continue
        uri = str(outputs.get(f"output_{index}") or "").strip()
        if uri:
            return uri
    config = state_config(state, ctx)
    raw = str(config.get("decision_uri") or "").strip()
    if not raw:
        return ""
    return resolve_tokens(
        raw,
        config=config,
        run=ctx.run,
        state_outputs=ctx.state_outputs,
        loop_iterations=ctx.loop_iterations,
    )


def _execute_state_machine(
    spec: NpaWorkflowSpec,
    state_name: str,
    ctx: RunContext,
    *,
    assume_decision: str,
    require_inputs: bool,
    on_step: Callable[[PlanStep], None] | None,
    decision_reader: Any | None,
    artifact_checker: Any | None,
    step_executor: Any | None = None,
    trigger_waiter: Callable[[StateSpec, str, "RunContext"], dict[str, Any]]
    | None = None,
    loop_label: str = "",
    follow_transitions: bool = True,
    results_out: list[dict[str, Any]] | None = None,
    depth: int = 0,
) -> list[dict[str, Any]]:
    _guard_execution_depth(spec, depth)
    state = spec.states[state_name]
    results: list[dict[str, Any]] = results_out if results_out is not None else []

    if state.parallel:
        group_records = _run_parallel_group(
            spec,
            state,
            ctx,
            require_inputs=require_inputs,
            on_step=on_step,
            artifact_checker=artifact_checker,
            step_executor=step_executor,
            trigger_waiter=trigger_waiter,
        )
        results.extend(group_records)
        failed = [item for item in group_records if item.get("status") == "failed"]
        if failed:
            # Surface the ROOT cause (first failure), not the cascade of members
            # that were skipped once the barrier could no longer be satisfied.
            raise NpaWorkflowError(
                str(failed[0].get("error") or f"parallel group {state.name} failed")
            )
        if state.next:
            _execute_state_machine(
                spec,
                state.next,
                ctx,
                assume_decision=assume_decision,
                require_inputs=require_inputs,
                on_step=on_step,
                decision_reader=decision_reader,
                artifact_checker=artifact_checker,
                step_executor=step_executor,
                trigger_waiter=trigger_waiter,
                results_out=results,
                depth=depth + 1,
            )
        return results

    if state.sequence:
        if state.loop:
            max_iter = resolve_config_int(state.loop.max or 1, ctx.config)
            for iteration in range(1, max_iter + 1):
                ctx.outer_iteration = iteration
                ctx.loop_iterations[state.name] = iteration
                ctx.last_decision = ""
                for child in state.sequence:
                    _execute_state_machine(
                        spec,
                        child,
                        ctx,
                        assume_decision=assume_decision,
                        require_inputs=require_inputs,
                        on_step=on_step,
                        decision_reader=decision_reader,
                        artifact_checker=artifact_checker,
                        step_executor=step_executor,
                        trigger_waiter=trigger_waiter,
                        loop_label=state.name,
                        follow_transitions=False,
                        results_out=results,
                        depth=depth + 1,
                    )
                if not ctx.last_decision:
                    decision_writer = _sequence_decision_writer(spec, state)
                    _refresh_decision(
                        ctx,
                        state=decision_writer,
                        reader=decision_reader,
                        read_s3=decision_writer is not None,
                    )
                if not ctx.last_decision:
                    ctx.last_decision = assume_decision
                if state.loop.until and evaluate_predicate(
                    state.loop.until, ctx.as_predicate_context()
                ):
                    break
            if state.next:
                _execute_state_machine(
                    spec,
                    state.next,
                    ctx,
                    assume_decision=assume_decision,
                    require_inputs=require_inputs,
                    on_step=on_step,
                    decision_reader=decision_reader,
                    artifact_checker=artifact_checker,
                    step_executor=step_executor,
                    trigger_waiter=trigger_waiter,
                    results_out=results,
                    depth=depth + 1,
                )
            return results

        for child in state.sequence:
            _execute_state_machine(
                spec,
                child,
                ctx,
                assume_decision=assume_decision,
                require_inputs=require_inputs,
                on_step=on_step,
                decision_reader=decision_reader,
                artifact_checker=artifact_checker,
                step_executor=step_executor,
                trigger_waiter=trigger_waiter,
                loop_label=state.name,
                follow_transitions=False,
                results_out=results,
                depth=depth + 1,
            )
        if state.next:
            _execute_state_machine(
                spec,
                state.next,
                ctx,
                assume_decision=assume_decision,
                require_inputs=require_inputs,
                on_step=on_step,
                decision_reader=decision_reader,
                artifact_checker=artifact_checker,
                step_executor=step_executor,
                trigger_waiter=trigger_waiter,
                results_out=results,
                depth=depth + 1,
            )
        return results

    if state.loop:
        max_iter = resolve_config_int(state.loop.max or 1, ctx.config)
        for iteration in range(1, max_iter + 1):
            ctx.inner_iteration = iteration
            ctx.loop_iterations[state.name] = iteration
            record = _run_single_state(
                spec,
                state,
                ctx,
                iteration=iteration,
                loop_label=loop_label,
                require_inputs=require_inputs,
                on_step=on_step,
                artifact_checker=artifact_checker,
                step_executor=step_executor,
                trigger_waiter=trigger_waiter,
            )
            results.append(record)
            if record.get("status") == "failed":
                raise NpaWorkflowError(
                    str(record.get("error") or f"state {state.name} failed")
                )
            _refresh_decision(ctx, reader=decision_reader, read_s3=False)
            if not ctx.last_decision:
                ctx.last_decision = assume_decision
            if state.loop.until and evaluate_predicate(
                state.loop.until, ctx.as_predicate_context()
            ):
                break
        next_name = _resolve_transition(state, ctx) or state.next
        if next_name:
            _execute_state_machine(
                spec,
                next_name,
                ctx,
                assume_decision=assume_decision,
                require_inputs=require_inputs,
                on_step=on_step,
                decision_reader=decision_reader,
                artifact_checker=artifact_checker,
                step_executor=step_executor,
                trigger_waiter=trigger_waiter,
                results_out=results,
                depth=depth + 1,
            )
        return results

    record = _run_single_state(
        spec,
        state,
        ctx,
        loop_label=loop_label,
        require_inputs=require_inputs,
        on_step=on_step,
        artifact_checker=artifact_checker,
        step_executor=step_executor,
        trigger_waiter=trigger_waiter,
    )
    results.append(record)
    if record.get("status") == "failed":
        raise NpaWorkflowError(str(record.get("error") or f"state {state.name} failed"))
    if state.terminal:
        return results
    if state.transitions:
        _refresh_decision(ctx, state=state, reader=decision_reader, read_s3=True)
        if not ctx.last_decision:
            ctx.last_decision = assume_decision
    next_name = ""
    if follow_transitions:
        next_name = _resolve_transition(state, ctx) or state.next
    elif not state.transitions:
        next_name = state.next
    if next_name:
        _execute_state_machine(
            spec,
            next_name,
            ctx,
            assume_decision=assume_decision,
            require_inputs=require_inputs,
            on_step=on_step,
            decision_reader=decision_reader,
            artifact_checker=artifact_checker,
            step_executor=step_executor,
            trigger_waiter=trigger_waiter,
            results_out=results,
            depth=depth + 1,
        )
    return results


def _run_parallel_group(
    spec: NpaWorkflowSpec,
    state: StateSpec,
    ctx: RunContext,
    *,
    require_inputs: bool,
    on_step: Callable[[PlanStep], None] | None,
    artifact_checker: Any | None,
    step_executor: Any | None,
    trigger_waiter: Callable[[StateSpec, str, RunContext], dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Run every member of a ``parallel:`` group and barrier on all of them.

    With an executor the whole group is handed over in one call so it can be
    launched concurrently (SkyPilot JobGroup, bounded by ``maxConcurrency``).
    Without one (local ``--execute``) members run in declared order, which keeps
    the local interpreter dependency-free.
    """

    members = [spec.states[name] for name in state.parallel]
    steps: list[PlanStep] = [
        build_step(spec, member, ctx, loop_label=state.name, group=state.name)
        for member in members
    ]
    for step in steps:
        if require_inputs and step.inputs:
            require_input_artifacts(
                [item["uri"] for item in step.inputs], checker=artifact_checker
            )
        if on_step is not None:
            on_step(step)

    if step_executor is None:
        records: list[dict[str, Any]] = []
        for member, step in zip(members, steps):
            try:
                wait_for_trigger(member, ctx, waiter=trigger_waiter)
                record = _dispatch_step(step, None)
            except NpaWorkflowError as exc:
                records.append(
                    {
                        "state": step.state,
                        "iteration": step.iteration,
                        "group": state.name,
                        "status": "failed",
                        "error": str(exc),
                    }
                )
                continue
            record.setdefault("group", state.name)
            _record_state_outputs(member, ctx, step)
            records.append(record)
        return records

    for member in members:
        wait_for_trigger(member, ctx, waiter=trigger_waiter)
    max_concurrency = len(steps)
    if state.max_concurrency is not None:
        max_concurrency = max(1, resolve_config_int(state.max_concurrency, ctx.config))
    records = list(
        step_executor.execute_parallel(
            steps,
            group=state.name,
            max_concurrency=max_concurrency,
        )
    )
    for member, step, record in zip(members, steps, records):
        record.setdefault("group", state.name)
        if record.get("status") != "failed":
            _record_state_outputs(member, ctx, step)
    return records


def _run_single_state(
    spec: NpaWorkflowSpec,
    state: StateSpec,
    ctx: RunContext,
    *,
    iteration: int | None = None,
    loop_label: str = "",
    require_inputs: bool,
    on_step: Callable[[PlanStep], None] | None,
    artifact_checker: Any | None,
    step_executor: Any | None = None,
    trigger_waiter: Callable[[StateSpec, str, "RunContext"], dict[str, Any]]
    | None = None,
) -> dict[str, Any]:
    plan = ExecutionPlan(
        workflow=spec.name,
        api_version=spec.api_version,
        initial=spec.initial,
    )
    _append_state_step(
        spec,
        state,
        ctx,
        plan,
        iteration=iteration,
        loop_label=loop_label,
    )
    step = plan.steps[-1]
    if require_inputs and step.inputs:
        require_input_artifacts(
            [item["uri"] for item in step.inputs],
            checker=artifact_checker,
        )
    if on_step is not None:
        on_step(step)
    try:
        wait_for_trigger(state, ctx, waiter=trigger_waiter)
        record = _dispatch_step(step, step_executor)
    except NpaWorkflowError as exc:
        record = _with_resources(
            {
                "state": step.state,
                "iteration": step.iteration,
                "status": "failed",
                "error": str(exc),
            },
            step,
        )
    else:
        _record_state_outputs(state, ctx, step)
    return record


def _with_resources(record: dict[str, Any], step: PlanStep) -> dict[str, Any]:
    """Ensure a step record carries the resolved resource profile.

    Executors describe *what happened*, and the resources a step ran with are part
    of that record no matter which tier produced it (local subprocess or the runtime
    tier's SkyPilot wave executor). Manifest consumers depend on it: the insights
    backbone reads ``resources_profile.accelerators`` to report a run's GPU count.
    """
    if not isinstance(record, dict):
        return record
    record.setdefault("resources", step.resources)
    if not record.get("resources_profile"):
        record["resources_profile"] = dict(step.resources_profile)
    record.setdefault("inputs", [dict(item) for item in step.inputs])
    record.setdefault("outputs", [dict(item) for item in step.outputs])
    return record


def _dispatch_step(step: PlanStep, step_executor: Any | None) -> dict[str, Any]:
    """Run one step locally, or hand it to an injected executor (runtime tier)."""

    if step_executor is None:
        # Module-level lookup keeps monkeypatching `_execute_step` working.
        return _with_resources(_execute_step(step, execute=True), step)
    return _with_resources(step_executor.execute(step), step)


def wait_for_trigger(
    state: StateSpec,
    ctx: RunContext,
    *,
    waiter: Callable[[StateSpec, str, RunContext], dict[str, Any]] | None,
) -> dict[str, Any] | None:
    """Block on a state's ``trigger:`` prefix before its work runs."""

    if state.trigger is None or waiter is None:
        return None
    uri = resolve_tokens(
        state.trigger.uri,
        config=state_config(state, ctx),
        run=ctx.run,
        state_outputs=ctx.state_outputs,
        loop_iterations=ctx.loop_iterations,
    )
    return waiter(state, uri, ctx)


def _resolved_run(state: StateSpec, ctx: RunContext) -> tuple[list[str], str, str]:
    config = state_config(state, ctx)
    if state.tool_ref:
        argv = [
            resolve_tokens(
                token,
                config=config,
                run=ctx.run,
                state_outputs=ctx.state_outputs,
                loop_iterations=ctx.loop_iterations,
            )
            for token in argv_for_tool(state.tool_ref)
        ]
        return drop_empty_optional_flags(state.tool_ref, argv), "", state.tool_ref
    if state.run is None:
        return [], "", ""
    shell = resolve_tokens(
        state.run.shell,
        config=config,
        run=ctx.run,
        state_outputs=ctx.state_outputs,
        loop_iterations=ctx.loop_iterations,
    )
    argv = [
        resolve_tokens(
            token,
            config=config,
            run=ctx.run,
            state_outputs=ctx.state_outputs,
            loop_iterations=ctx.loop_iterations,
        )
        for token in state.run.argv
    ]
    return argv, shell, ""


def _resolve_transition(state: StateSpec, ctx: RunContext) -> str:
    for tr in state.transitions:
        if tr.if_config and not config_truthy(tr.if_config, ctx.config):
            continue
        if tr.when is None:
            return tr.goto
        if evaluate_predicate(tr.when, ctx.as_predicate_context()):
            return tr.goto
    return ""


def _execute_step(step: PlanStep, *, execute: bool) -> dict[str, Any]:
    record: dict[str, Any] = {
        "state": step.state,
        "iteration": step.iteration,
        "status": "planned",
        # The durable run manifest is the record of *how* a step ran, so it must
        # carry the resolved resource profile (accelerators, cpus, memory). Without
        # it, consumers of the manifest -- notably the insights backbone, which
        # derives a run's GPU count from ``resources_profile.accelerators`` -- have
        # no way to know a step used a GPU.
        "resources": step.resources,
        "resources_profile": dict(step.resources_profile),
    }
    if not execute:
        if step.argv:
            record["argv"] = step.argv
        if step.shell:
            record["shell"] = step.shell
        if step.tool_ref:
            record["tool_ref"] = step.tool_ref
        return record

    if step.argv:
        proc = subprocess.run(step.argv, capture_output=True, text=True, check=False)
        record["argv"] = step.argv
        record["returncode"] = proc.returncode
        record["status"] = "ok" if proc.returncode == 0 else "failed"
        if proc.returncode != 0:
            raise NpaWorkflowError(
                f"state {step.state} failed (exit {proc.returncode}): "
                f"{proc.stderr or proc.stdout}"
            )
        return record

    if step.shell.strip():
        # shell=True interpolates resolved config tokens into a bash string.
        # Spec authors are trusted today; untrusted config values would be an injection surface.
        proc = subprocess.run(
            step.shell,
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True,
            check=False,
        )
        record["shell"] = step.shell
        record["returncode"] = proc.returncode
        record["status"] = "ok" if proc.returncode == 0 else "failed"
        if proc.returncode != 0:
            raise NpaWorkflowError(
                f"state {step.state} failed (exit {proc.returncode}): "
                f"{proc.stderr or proc.stdout}"
            )
        return record

    record["status"] = "skipped"
    return record
