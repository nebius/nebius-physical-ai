"""Runtime orchestration tier for ``npa.workflow/v0.0.1``.

This sits **above** :func:`npa.orchestration.npa_workflow.scheduler.build_scheduler_task`
and turns the plan-time engine into a real runtime engine:

    plan the next wave -> render it -> submit it to SkyPilot -> poll to a terminal
    status -> read the *actual* decision artifact from S3 -> decide (iterate,
    branch, early-exit, fan out) -> replan

Design notes
------------
* **No second engine.** The traversal is the existing dynamic walker in
  ``interpreter._execute_state_machine`` (which already re-reads
  ``config.decision_uri`` after decision states). This module supplies the
  *executor* for that walker, so loops, ``transitions``/``goto`` and early-exit
  semantics cannot drift between the plan-time and runtime paths.
* **Existing decision contract.** Decisions are read with
  ``decisions.load_decision`` / ``refresh_context_decision`` from the artifacts
  that ``write_*_decision`` toolRefs and ``data_factory_stages.grade_gate``
  already write. Nothing new is invented.
* **Seam respected.** Waves are rendered through
  ``skypilot_render.render_skypilot_steps_yaml``, i.e. through
  ``build_skypilot_task_doc`` -> ``build_scheduler_task``. The orchestrator never
  reaches into rendering internals.
* **Durable + resumable.** Every wave attempt is written to
  ``<config.prefix>/npa-workflow/runtime.json`` (``npa.workflow.runtime.v1``).
  Re-running the same ``run_id`` with ``resume=True`` replays succeeded waves from
  the ledger instead of resubmitting them.
* **Plan-only is untouched.** ``--assume-decision`` + the flattened serial render
  remain the offline path; this tier is opt-in (``submit --runtime``).
"""

from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from npa.orchestration.npa_workflow.decisions import normalize_decision
from npa.orchestration.npa_workflow.errors import NpaWorkflowError
from npa.orchestration.npa_workflow.interpreter import (
    ExecutionPlan,
    PlanStep,
    RunContext,
    build_plan,
    run_workflow,
)
from npa.orchestration.npa_workflow.run_state import (
    RunStateStore,
    RuntimeRunState,
    store_for_config,
    utc_now,
)
from npa.orchestration.npa_workflow.skypilot_render import (
    SkypilotRenderOptions,
    assert_no_unresolved_placeholders,
    render_skypilot_steps_yaml,
)
from npa.orchestration.npa_workflow.spec import NpaWorkflowSpec, StateSpec

TERMINAL_OK = frozenset({"SUCCEEDED", "SUCCESS", "COMPLETED", "DONE"})
TERMINAL_FAIL = frozenset(
    {
        "FAILED",
        "FAIL",
        "FAILED_PRECHECKS",
        "FAILED_SETUP",
        "FAILED_RUNTIME",
        "FAILED_CONTROLLER",
        "FAILED_NO_RESOURCE",
        "CANCELLED",
        "CANCELED",
        "STOPPED",
    }
)

DEFAULT_POLL_SECONDS = 30
DEFAULT_MAX_WAIT_SECONDS = 3600


def is_terminal_ok(status: str) -> bool:
    return status.upper() in TERMINAL_OK


def is_terminal_fail(status: str) -> bool:
    upper = status.upper()
    return upper in TERMINAL_FAIL or upper.startswith("FAILED")


def is_terminal(status: str) -> bool:
    return is_terminal_ok(status) or is_terminal_fail(status)


@dataclass
class RuntimeOptions:
    """Knobs for one runtime-orchestrated run."""

    poll_seconds: int = DEFAULT_POLL_SECONDS
    max_wait_seconds: int = DEFAULT_MAX_WAIT_SECONDS
    retries: int = 0
    retry_backoff_seconds: int = 30
    cancel_on_timeout: bool = True
    max_concurrency: int = 0  # 0 = honour each group's own maxConcurrency
    secret_envs: tuple[str, ...] = ()
    submit_timeout: int = 1800
    infra: str = ""
    controller_backend: str = "kubernetes"
    isolated_config_dir: Path | None = None
    resume: bool = False


@dataclass
class WaveAttempt:
    """One submit-and-wait attempt, as recorded in the durable ledger."""

    key: str
    states: list[str]
    kind: str
    group: str = ""
    attempt: int = 1
    job_id: str = ""
    job_name: str = ""
    status: str = "pending"
    sky_status: str = ""
    started_at: str = ""
    ended_at: str = ""
    tasks: list[dict[str, Any]] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    error: str = ""
    replayed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "replayed": self.replayed,
            "states": list(self.states),
            "kind": self.kind,
            "group": self.group,
            "attempt": self.attempt,
            "job_id": self.job_id,
            "job_name": self.job_name,
            "status": self.status,
            "sky_status": self.sky_status,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "tasks": list(self.tasks),
            "outputs": list(self.outputs),
            "error": self.error,
        }


def wave_key(steps: Sequence[PlanStep], *, group: str, sequence_number: int) -> str:
    """Stable identity of a wave inside one run (used for resume/idempotency).

    Includes the loop labels and iteration numbers so a loop body that runs three
    times produces three distinct keys, and a monotonic sequence number so two
    structurally identical waves in different parts of the graph never collide.
    """

    parts = [
        f"{step.loop_label}:{step.state}:{step.iteration if step.iteration is not None else '-'}"
        for step in steps
    ]
    return f"{sequence_number:03d}|{group or 'serial'}|" + ",".join(parts)


class SkyPilotWaveExecutor:
    """Execute planned steps as SkyPilot managed jobs, one wave at a time.

    A wave is either a single step (serial pipeline document) or the members of a
    ``parallel:`` group (a JobGroup, chunked to respect ``maxConcurrency``). Every
    dependency is injected so unit tests never touch SkyPilot, S3 or a cluster.
    """

    def __init__(
        self,
        spec: NpaWorkflowSpec,
        *,
        run_id: str,
        render_options: SkypilotRenderOptions | None = None,
        options: RuntimeOptions | None = None,
        ledger: "RuntimeLedger | None" = None,
        submitter: Callable[..., Any] | None = None,
        status_fn: Callable[..., Any] | None = None,
        timeline_fn: Callable[..., list[dict[str, Any]]] | None = None,
        canceller: Callable[..., Any] | None = None,
        sleeper: Callable[[float], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        logger: Callable[[str], None] | None = None,
    ) -> None:
        self.spec = spec
        self.run_id = run_id
        self.render_options = render_options or SkypilotRenderOptions()
        self.options = options or RuntimeOptions()
        self.ledger = ledger or RuntimeLedger(None, workflow=spec.name, run_id=run_id)
        self._submitter = submitter
        self._status_fn = status_fn
        self._timeline_fn = timeline_fn
        self._canceller = canceller
        self._sleep = sleeper or time.sleep
        self._clock = clock
        self._log = logger or (lambda message: None)
        self._sequence = 0
        self.attempts: list[WaveAttempt] = []

    # ------------------------------------------------------------------ public

    def execute(self, step: PlanStep) -> dict[str, Any]:
        """StepExecutor protocol: run a single planned step as one managed job."""

        attempt = self._run_wave([step], kind="serial", group="")
        record: dict[str, Any] = {
            "state": step.state,
            "iteration": step.iteration,
            "status": "ok" if attempt.status == "succeeded" else "failed",
            "job_id": attempt.job_id,
            "job_name": attempt.job_name,
            "sky_status": attempt.sky_status,
            "wave_key": attempt.key,
        }
        if step.tool_ref:
            record["tool_ref"] = step.tool_ref
        if attempt.status != "succeeded":
            record["error"] = attempt.error or f"state {step.state} failed"
            raise NpaWorkflowError(record["error"])
        return record

    def execute_parallel(
        self,
        steps: Sequence[PlanStep],
        *,
        group: str,
        max_concurrency: int,
    ) -> list[dict[str, Any]]:
        """StepExecutor protocol: run a fan-out group concurrently, then barrier."""

        limit = max(1, self.options.max_concurrency or max_concurrency or len(steps))
        batches = [
            list(steps)[start : start + limit] for start in range(0, len(steps), limit)
        ]
        records: list[dict[str, Any]] = []
        for batch_index, batch in enumerate(batches, start=1):
            kind = "parallel" if len(batch) > 1 else "serial"
            self._log(
                f"wave {group} batch {batch_index}/{len(batches)} "
                f"({kind}, {len(batch)} task(s)): {[s.state for s in batch]}"
            )
            attempt = self._run_wave(batch, kind=kind, group=group)
            failed = attempt.status != "succeeded"
            for step in batch:
                record: dict[str, Any] = {
                    "state": step.state,
                    "iteration": step.iteration,
                    "group": group,
                    "status": "failed" if failed else "ok",
                    "job_id": attempt.job_id,
                    "job_name": attempt.job_name,
                    "sky_status": attempt.sky_status,
                    "wave_key": attempt.key,
                }
                if step.tool_ref:
                    record["tool_ref"] = step.tool_ref
                if failed:
                    record["error"] = attempt.error or f"parallel group {group} failed"
                records.append(record)
            if failed:
                # Barrier semantics: a failed batch stops the group; remaining
                # members are reported as not-run so the ledger stays honest.
                for step in [s for later in batches[batch_index:] for s in later]:
                    records.append(
                        {
                            "state": step.state,
                            "iteration": step.iteration,
                            "group": group,
                            "status": "failed",
                            "error": f"skipped after {group} batch {batch_index} failed",
                        }
                    )
                break
        return records

    # ----------------------------------------------------------------- private

    def _run_wave(self, steps: Sequence[PlanStep], *, kind: str, group: str) -> WaveAttempt:
        self._sequence += 1
        key = wave_key(steps, group=group, sequence_number=self._sequence)

        replayed = self.ledger.completed(key) if self.options.resume else None
        if replayed is not None:
            attempt = WaveAttempt(
                key=key,
                states=[step.state for step in steps],
                kind=kind,
                group=group,
                attempt=int(replayed.get("attempt") or 1),
                job_id=str(replayed.get("job_id") or ""),
                job_name=str(replayed.get("job_name") or ""),
                status="succeeded",
                sky_status=str(replayed.get("sky_status") or "SUCCEEDED"),
                started_at=str(replayed.get("started_at") or ""),
                ended_at=str(replayed.get("ended_at") or ""),
                tasks=list(replayed.get("tasks") or []),
                outputs=list(replayed.get("outputs") or []),
                replayed=True,
            )
            self._log(f"wave {key}: replayed from ledger (job {attempt.job_id})")
            self.attempts.append(attempt)
            return attempt

        last_error = ""
        for attempt_number in range(1, max(1, self.options.retries + 1) + 1):
            attempt = WaveAttempt(
                key=key,
                states=[step.state for step in steps],
                kind=kind,
                group=group,
                attempt=attempt_number,
                started_at=utc_now(),
                outputs=[item["uri"] for step in steps for item in step.outputs],
            )
            self.attempts.append(attempt)
            try:
                self._submit_and_wait(steps, kind=kind, group=group, attempt=attempt)
            except NpaWorkflowError as exc:
                attempt.status = "failed"
                attempt.error = str(exc)
                attempt.ended_at = utc_now()
                self.ledger.record(attempt)
                last_error = str(exc)
            else:
                attempt.status = "succeeded"
                attempt.ended_at = utc_now()
                self.ledger.record(attempt)
                return attempt
            if attempt_number <= self.options.retries:
                self._log(
                    f"wave {key}: attempt {attempt_number} failed ({last_error}); retrying"
                )
                self._sleep(self.options.retry_backoff_seconds)
        failed = self.attempts[-1]
        failed.error = last_error or "wave failed"
        return failed

    def _submit_and_wait(
        self,
        steps: Sequence[PlanStep],
        *,
        kind: str,
        group: str,
        attempt: WaveAttempt,
    ) -> None:
        job_name = self._job_name(steps, group=group, attempt=attempt)
        attempt.job_name = job_name
        yaml_text = render_skypilot_steps_yaml(
            self.spec,
            steps,
            run_id=self.run_id,
            options=self.render_options,
            execution="parallel" if kind == "parallel" else "serial",
            name=job_name,
        )
        assert_no_unresolved_placeholders(yaml_text)

        with tempfile.TemporaryDirectory(prefix="npa-workflow-wave-") as tmp:
            path = Path(tmp) / f"{job_name}.skypilot.yaml"
            path.write_text(yaml_text, encoding="utf-8")
            result = self._submit(path, job_name)
        job_id = str(getattr(result, "job_id", "") or job_name)
        attempt.job_id = job_id
        status = str(getattr(result, "status", "SUBMITTED") or "SUBMITTED").upper()
        attempt.sky_status = status
        self._log(f"wave {attempt.key}: submitted job_id={job_id} name={job_name}")

        final_status = self._poll(job_id, attempt)
        attempt.sky_status = final_status
        attempt.tasks = self._timeline(job_id)
        if not is_terminal_ok(final_status):
            raise NpaWorkflowError(
                f"wave {attempt.key} reached terminal status {final_status} "
                f"(job_id={job_id}, name={job_name})"
            )

    def _poll(self, job_id: str, attempt: WaveAttempt) -> str:
        deadline = self._clock() + max(1, self.options.max_wait_seconds)
        last = "UNKNOWN"
        while True:
            current = self._status(job_id)
            last = str(getattr(current, "status", "") or "UNKNOWN").upper()
            if is_terminal(last):
                return last
            if self._clock() >= deadline:
                if self.options.cancel_on_timeout:
                    self._cancel(job_id, attempt.job_name)
                raise NpaWorkflowError(
                    f"wave {attempt.key} did not reach a terminal status within "
                    f"{self.options.max_wait_seconds}s (last={last}, job_id={job_id})"
                )
            self._sleep(self.options.poll_seconds)

    def _job_name(self, steps: Sequence[PlanStep], *, group: str, attempt: WaveAttempt) -> str:
        label = group or steps[0].state
        suffix = f"-a{attempt.attempt}" if attempt.attempt > 1 else ""
        iteration = steps[0].iteration
        if iteration is not None:
            label = f"{label}-{iteration}"
        name = f"{self.run_id}-{self._sequence:02d}-{label}{suffix}"
        return _sanitize_job_name(name)

    def _submit(self, path: Path, job_name: str) -> Any:
        submitter = self._submitter
        if submitter is None:
            from npa.orchestration.skypilot.workflow import submit_workflow as submitter

        return submitter(
            path,
            job_name,
            isolated_config_dir=self.options.isolated_config_dir,
            controller_backend=self.options.controller_backend,
            infra=self.options.infra,
            secret_envs=list(self.options.secret_envs),
            timeout=self.options.submit_timeout,
        )

    def _status(self, job_id: str) -> Any:
        status_fn = self._status_fn
        if status_fn is None:
            from npa.orchestration.skypilot.workflow import workflow_status as status_fn

        return status_fn(job_id)

    def _timeline(self, job_id: str) -> list[dict[str, Any]]:
        timeline_fn = self._timeline_fn
        if timeline_fn is None:
            from npa.orchestration.skypilot.workflow import (
                workflow_task_statuses as timeline_fn,
            )

        try:
            return list(timeline_fn(job_id))
        except Exception as exc:  # noqa: BLE001 - evidence collection is best-effort
            self._log(f"timeline unavailable for job {job_id}: {exc}")
            return []

    def _cancel(self, job_id: str, job_name: str) -> None:
        canceller = self._canceller
        if canceller is None:
            try:
                from npa.orchestration.skypilot._bin import resolve_config
                from npa.orchestration.skypilot.workflow_state import cancel_workflow_job
            except Exception:  # noqa: BLE001
                return

            def canceller(**kwargs: Any) -> Any:  # type: ignore[misc]
                runtime = resolve_config()
                return cancel_workflow_job(sky_bin=str(runtime.sky_bin), **kwargs)

        try:
            canceller(job_id=str(job_id), run_id=job_name, cluster=job_name)
            self._log(f"cancelled job {job_id} ({job_name}) after timeout")
        except Exception as exc:  # noqa: BLE001 - never mask the timeout error
            self._log(f"cancel failed for job {job_id}: {exc}")


def _sanitize_job_name(name: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "-_" else "-" for char in name)
    cleaned = cleaned.strip("-_").lower() or "npa-workflow"
    return cleaned[:60].rstrip("-_")


class RuntimeLedger:
    """Durable wave ledger (``npa.workflow.runtime.v1``) with in-memory fallback."""

    def __init__(
        self,
        store: RunStateStore | None,
        *,
        workflow: str,
        run_id: str,
        api_version: str = "",
        resume: bool = False,
    ) -> None:
        self.store = store
        self.state = RuntimeRunState(workflow=workflow, run_id=run_id, api_version=api_version)
        if store is not None and resume:
            existing = store.read_runtime_state()
            if existing is not None and existing.run_id == run_id:
                self.state = existing
                self.state.status = "running"

    @property
    def run_prefix_uri(self) -> str:
        return self.store.run_prefix_uri if self.store is not None else ""

    def completed(self, key: str) -> dict[str, Any] | None:
        return self.state.completed_wave(key)

    def record(self, attempt: WaveAttempt) -> None:
        self.state.record_wave(attempt.to_dict())
        self.flush()

    def record_decision(self, payload: Mapping[str, Any]) -> None:
        self.state.decisions.append(dict(payload))
        self.flush()

    def record_watermark(self, key: str, value: Any) -> None:
        self.state.watermarks[key] = value
        self.flush()

    def set_status(self, status: str) -> None:
        self.state.status = status
        self.flush()

    def flush(self) -> None:
        if self.store is None:
            return
        try:
            self.store.write_runtime_state(self.state)
        except Exception as exc:  # noqa: BLE001 - never lose a run over a state write
            raise NpaWorkflowError(f"failed to persist runtime state: {exc}") from exc


class RecordingDecisionReader:
    """Wrap a decision reader so every runtime gate read lands in the ledger."""

    def __init__(self, reader: Callable[[str, str], str] | None, ledger: RuntimeLedger) -> None:
        self._reader = reader
        self._ledger = ledger
        self.reads: list[dict[str, Any]] = []

    def __call__(self, bucket: str, key: str) -> str:
        if self._reader is not None:
            body = self._reader(bucket, key)
        else:
            from npa.orchestration.npa_workflow.decisions import _read_object

            body = _read_object(bucket, key)
        payload = {
            "uri": f"s3://{bucket}/{key}",
            "body": str(body)[:2000],
            "read_at": utc_now(),
        }
        try:
            import json as _json

            decoded = _json.loads(body)
            if isinstance(decoded, dict):
                payload["decision"] = normalize_decision(
                    str(decoded.get("decision") or decoded.get("last_decision") or "")
                )
        except Exception:  # noqa: BLE001 - the caller surfaces malformed payloads
            pass
        self.reads.append(payload)
        self._ledger.record_decision(payload)
        return body


def s3_trigger_waiter(
    *,
    ledger: RuntimeLedger | None = None,
    lister: Callable[[str, str], Iterable[str]] | None = None,
    sleeper: Callable[[float], None] = time.sleep,
    logger: Callable[[str], None] | None = None,
) -> Callable[[StateSpec, str, RunContext], dict[str, Any]]:
    """Build a driver-side watcher for ``trigger:`` states.

    Polls an object-storage prefix until at least ``minObjects`` keys are present,
    recording the observed watermark in the ledger so a resumed run does not wait
    again for data it already saw.
    """

    log = logger or (lambda message: None)

    def _list(uri: str) -> list[str]:
        from npa.orchestration.npa_workflow.decisions import parse_s3_uri

        bucket, prefix = parse_s3_uri(uri)
        if lister is not None:
            return list(lister(bucket, prefix))
        from npa.clients.storage import StorageClient

        client = StorageClient.from_environment()
        response = client._s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        return [str(item["Key"]) for item in response.get("Contents") or ()]

    def waiter(state: StateSpec, uri: str, ctx: RunContext) -> dict[str, Any]:
        trigger = state.trigger
        assert trigger is not None
        if ledger is not None:
            seen = ledger.state.watermarks.get(state.name)
            if seen and int(seen.get("objects") or 0) >= trigger.min_objects:
                return dict(seen)
        polls = 0
        while True:
            keys = _list(uri)
            polls += 1
            if len(keys) >= trigger.min_objects:
                watermark = {
                    "uri": uri,
                    "objects": len(keys),
                    "polls": polls,
                    "observed_at": utc_now(),
                    "sample": keys[:5],
                }
                if ledger is not None:
                    ledger.record_watermark(state.name, watermark)
                log(f"trigger {state.name}: {len(keys)} object(s) at {uri} after {polls} poll(s)")
                return watermark
            if trigger.max_polls and polls >= trigger.max_polls:
                raise NpaWorkflowError(
                    f"state {state.name}: trigger {uri} still has {len(keys)} object(s) "
                    f"after {polls} poll(s) (need {trigger.min_objects})"
                )
            log(f"trigger {state.name}: waiting for data at {uri} (poll {polls})")
            sleeper(trigger.poll_seconds)

    return waiter


@dataclass
class RuntimeReport:
    workflow: str
    run_id: str
    status: str
    waves: list[dict[str, Any]] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    steps: list[dict[str, Any]] = field(default_factory=list)
    run_prefix_uri: str = ""
    runtime_state_uri: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow": self.workflow,
            "run_id": self.run_id,
            "status": self.status,
            "wave_count": len(self.waves),
            "waves": list(self.waves),
            "decisions": list(self.decisions),
            "steps": list(self.steps),
            "run_prefix_uri": self.run_prefix_uri,
            "runtime_state_uri": self.runtime_state_uri,
            "error": self.error,
        }


def run_workflow_runtime(
    spec: NpaWorkflowSpec,
    *,
    run_id: str,
    render_options: SkypilotRenderOptions | None = None,
    options: RuntimeOptions | None = None,
    assume_decision: str = "",
    require_inputs: bool = False,
    state_store: RunStateStore | None = None,
    decision_reader: Callable[[str, str], str] | None = None,
    executor: SkyPilotWaveExecutor | None = None,
    trigger_waiter: Callable[[StateSpec, str, RunContext], dict[str, Any]] | None = None,
    logger: Callable[[str], None] | None = None,
) -> RuntimeReport:
    """Drive a spec to completion through the runtime tier.

    ``assume_decision`` is only a *fallback* here: when a gate artifact cannot be
    read the traversal keeps the plan-time behaviour instead of hanging. The real
    decisions come from S3.
    """

    opts = options or RuntimeOptions()
    log = logger or (lambda message: None)

    if executor is not None:
        ledger = executor.ledger
    else:
        store = state_store
        if store is None:
            store = store_for_config(_resolved_config(spec, run_id), run_id=run_id)
        ledger = RuntimeLedger(
            store,
            workflow=spec.name,
            run_id=run_id,
            api_version=spec.api_version,
            resume=opts.resume,
        )
    ledger.set_status("running")

    recording_reader = RecordingDecisionReader(decision_reader, ledger)
    wave_executor = executor or SkyPilotWaveExecutor(
        spec,
        run_id=run_id,
        render_options=render_options,
        options=opts,
        ledger=ledger,
        logger=log,
    )
    waiter = trigger_waiter
    if waiter is None and any(state.trigger for state in spec.states.values()):
        waiter = s3_trigger_waiter(ledger=ledger, logger=log)

    status = "succeeded"
    error = ""
    steps: list[dict[str, Any]] = []
    try:
        report = run_workflow(
            spec,
            run_id=run_id,
            execute=True,
            assume_decision=assume_decision,
            require_inputs=require_inputs,
            decision_reader=recording_reader,
            step_executor=wave_executor,
            trigger_waiter=waiter,
        )
        steps = list(report.get("steps") or [])
    except NpaWorkflowError as exc:
        status = "failed"
        error = str(exc)
    ledger.set_status(status)

    return RuntimeReport(
        workflow=spec.name,
        run_id=run_id,
        status=status,
        waves=[attempt.to_dict() for attempt in wave_executor.attempts],
        decisions=list(ledger.state.decisions),
        steps=steps,
        run_prefix_uri=ledger.run_prefix_uri,
        runtime_state_uri=(
            f"{ledger.run_prefix_uri.rstrip('/')}/npa-workflow/runtime.json"
            if ledger.run_prefix_uri
            else ""
        ),
        error=error,
    )


def _resolved_config(spec: NpaWorkflowSpec, run_id: str) -> dict[str, Any]:
    """Resolve ``config`` tokens the same way the interpreter does (bucket/prefix)."""

    from npa.orchestration.npa_workflow.interpreter import _make_context

    ctx = _make_context(spec, run_id=run_id)
    return dict(ctx.config)


def plan_preview(spec: NpaWorkflowSpec, *, run_id: str, assume_decision: str = "") -> ExecutionPlan:
    """Convenience for callers that want the flattened plan next to a runtime run."""

    return build_plan(spec, run_id=run_id, assume_decision=assume_decision)


def secret_env_names(extra: Sequence[str] = ()) -> tuple[str, ...]:
    """Environment variable names worth forwarding to every wave."""

    names: list[str] = []
    for name in [*extra, "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]:
        if name and name not in names and os.environ.get(name):
            names.append(name)
    return tuple(names)


__all__ = [
    "RecordingDecisionReader",
    "RuntimeLedger",
    "RuntimeOptions",
    "RuntimeReport",
    "SkyPilotWaveExecutor",
    "WaveAttempt",
    "is_terminal",
    "is_terminal_fail",
    "is_terminal_ok",
    "run_workflow_runtime",
    "s3_trigger_waiter",
    "secret_env_names",
    "wave_key",
]
