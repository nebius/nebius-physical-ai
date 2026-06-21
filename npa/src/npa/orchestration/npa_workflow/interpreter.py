"""Build execution plans and run NPA workflow state machines."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from typing import Any, Callable

from npa.orchestration.npa_workflow.artifacts import require_input_artifacts
from npa.orchestration.npa_workflow.catalog import argv_for_tool
from npa.orchestration.npa_workflow.decisions import normalize_decision, refresh_context_decision
from npa.orchestration.npa_workflow.errors import NpaWorkflowError
from npa.orchestration.npa_workflow.predicates import evaluate_predicate
from npa.orchestration.npa_workflow.run_state import RunManifest, RunStateStore, store_for_config
from npa.orchestration.npa_workflow.spec import (
    NpaWorkflowSpec,
    StateSpec,
    config_truthy,
    resolve_config_int,
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


def build_plan(
    spec: NpaWorkflowSpec,
    *,
    run_id: str = "plan-run",
    assume_decision: str = "",
) -> ExecutionPlan:
    raw_assume = assume_decision or str(spec.config.get("plan_assume_decision") or "loop_back")
    assume = normalize_decision(raw_assume)
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
) -> dict[str, Any]:
    raw_assume = assume_decision or str(spec.config.get("plan_assume_decision") or "loop_back")
    assume = normalize_decision(raw_assume)
    ctx = _make_context(spec, run_id=run_id)
    store = state_store or (store_for_config(ctx.config, run_id=run_id) if persist_state else None)
    manifest = RunManifest(
        workflow=spec.name,
        run_id=run_id,
        api_version=spec.api_version,
        status="running" if execute else "planned",
    )
    if store is not None:
        try:
            store.write_manifest(manifest)
        except Exception as exc:
            raise NpaWorkflowError(f"failed to persist workflow manifest: {exc}") from exc

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
    config = _resolve_config_strings(dict(spec.config), run=run)
    if config.get("prefix"):
        run["prefix"] = resolve_tokens(str(config["prefix"]), config=config, run=run)
    return RunContext(config=config, run=run)


def _resolve_config_strings(config: dict[str, Any], *, run: dict[str, Any]) -> dict[str, Any]:
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

    if state.sequence:
        if state.loop:
            max_iter = resolve_config_int(state.loop.max or 1, ctx.config)
            for iteration in range(1, max_iter + 1):
                ctx.outer_iteration = iteration
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
            _append_state_step(
                spec, state, ctx, plan, iteration=iteration, loop_label=loop_label
            )
            ctx.last_decision = assume_decision
            if state.loop.until and evaluate_predicate(state.loop.until, ctx.as_predicate_context()):
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


def _sequence_refreshes_decision(spec: NpaWorkflowSpec, state: StateSpec) -> bool:
    return any(
        spec.states[child].writes_decision
        for child in state.sequence
        if child in spec.states
    )


def _append_state_step(
    spec: NpaWorkflowSpec,
    state: StateSpec,
    ctx: RunContext,
    plan: ExecutionPlan,
    *,
    iteration: int | None = None,
    loop_label: str = "",
) -> None:
    argv, shell, tool_ref = _resolved_run(state, ctx)
    outputs = [
        {
            "uri": resolve_tokens(
                artifact.uri,
                config=ctx.config,
                run=ctx.run,
                state_outputs=ctx.state_outputs,
            ),
            "schema": artifact.schema,
        }
        for artifact in state.outputs
        if artifact.uri
    ]
    plan.steps.append(
        PlanStep(
            state=state.name,
            iteration=iteration,
            loop_label=loop_label,
            argv=argv,
            shell=shell,
            tool_ref=tool_ref,
            resources=state.resources,
            resources_profile=_resources_profile(spec, state.resources),
            outputs=outputs,
            inputs=_resolved_inputs(state, ctx),
        )
    )
    _record_state_outputs(state, ctx, plan.steps[-1])


def _resources_profile(spec: NpaWorkflowSpec, profile: str) -> dict[str, Any]:
    raw = spec.resources.get(profile) or spec.resources.get("default") or {}
    return dict(raw) if isinstance(raw, dict) else {}


def _resolved_inputs(state: StateSpec, ctx: RunContext) -> list[dict[str, str]]:
    return [
        {
            "uri": resolve_tokens(
                artifact.uri,
                config=ctx.config,
                run=ctx.run,
                state_outputs=ctx.state_outputs,
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
        f"output_{index}": output["uri"] for index, output in enumerate(step.outputs, start=1)
    }
    primary = step.outputs[0]["uri"]
    ctx.state_outputs[state.name]["uri"] = primary


def _refresh_decision(
    ctx: RunContext,
    *,
    reader: Any | None = None,
    read_s3: bool = False,
) -> None:
    if read_s3:
        ctx.last_decision = refresh_context_decision(ctx.as_predicate_context(), reader=reader)


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
    loop_label: str = "",
    follow_transitions: bool = True,
    results_out: list[dict[str, Any]] | None = None,
    depth: int = 0,
) -> list[dict[str, Any]]:
    _guard_execution_depth(spec, depth)
    state = spec.states[state_name]
    results: list[dict[str, Any]] = results_out if results_out is not None else []

    if state.sequence:
        if state.loop:
            max_iter = resolve_config_int(state.loop.max or 1, ctx.config)
            for iteration in range(1, max_iter + 1):
                ctx.outer_iteration = iteration
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
                        loop_label=state.name,
                        follow_transitions=False,
                        results_out=results,
                        depth=depth + 1,
                    )
                if not ctx.last_decision:
                    _refresh_decision(
                        ctx,
                        reader=decision_reader,
                        read_s3=_sequence_refreshes_decision(spec, state),
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
                results_out=results,
                depth=depth + 1,
            )
        return results

    if state.loop:
        max_iter = resolve_config_int(state.loop.max or 1, ctx.config)
        for iteration in range(1, max_iter + 1):
            ctx.inner_iteration = iteration
            record = _run_single_state(
                spec,
                state,
                ctx,
                iteration=iteration,
                loop_label=loop_label,
                require_inputs=require_inputs,
                on_step=on_step,
                artifact_checker=artifact_checker,
            )
            results.append(record)
            if record.get("status") == "failed":
                raise NpaWorkflowError(str(record.get("error") or f"state {state.name} failed"))
            _refresh_decision(ctx, reader=decision_reader, read_s3=False)
            if not ctx.last_decision:
                ctx.last_decision = assume_decision
            if state.loop.until and evaluate_predicate(state.loop.until, ctx.as_predicate_context()):
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
    )
    results.append(record)
    if record.get("status") == "failed":
        raise NpaWorkflowError(str(record.get("error") or f"state {state.name} failed"))
    if state.terminal:
        return results
    if state.transitions:
        _refresh_decision(ctx, reader=decision_reader, read_s3=True)
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
            results_out=results,
            depth=depth + 1,
        )
    return results


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
        record = _execute_step(step, execute=True)
    except NpaWorkflowError as exc:
        record = {
            "state": step.state,
            "iteration": step.iteration,
            "status": "failed",
            "error": str(exc),
        }
    else:
        _record_state_outputs(state, ctx, step)
    return record


def _resolved_run(state: StateSpec, ctx: RunContext) -> tuple[list[str], str, str]:
    if state.tool_ref:
        argv = [
            resolve_tokens(
                token,
                config=ctx.config,
                run=ctx.run,
                state_outputs=ctx.state_outputs,
            )
            for token in argv_for_tool(state.tool_ref)
        ]
        return argv, "", state.tool_ref
    if state.run is None:
        return [], "", ""
    shell = resolve_tokens(
        state.run.shell,
        config=ctx.config,
        run=ctx.run,
        state_outputs=ctx.state_outputs,
    )
    argv = [
        resolve_tokens(token, config=ctx.config, run=ctx.run, state_outputs=ctx.state_outputs)
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
