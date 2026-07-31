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
from npa.orchestration.npa_workflow.waves import split_into_batches

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
#: How many consecutive `sky jobs queue` failures to tolerate before giving up on a
#: wave. A busy API server or a query timeout says nothing about the job itself, so
#: transient errors must not orphan a running GPU job.
MAX_CONSECUTIVE_STATUS_ERRORS = 5


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
    #: True when this wave attached to a managed job a previous driver had left in
    #: flight (``--resume``) instead of submitting a new one.
    adopted: bool = False
    #: Highest number of member tasks observed RUNNING at the same time while
    #: polling. For a parallel wave this is the direct, unambiguous evidence that
    #: the group really ran concurrently rather than as a serialized chain.
    max_concurrent_observed: int = 0
    observations: list[dict[str, Any]] = field(default_factory=list)
    #: Transient status-query failures tolerated while polling (evidence that the
    #: wave survived a flaky control plane rather than silently leaking a job).
    status_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "replayed": self.replayed,
            "adopted": self.adopted,
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
            "max_concurrent_observed": self.max_concurrent_observed,
            "observations": list(self.observations),
            "status_errors": list(self.status_errors),
        }


def plan_fingerprint(spec: NpaWorkflowSpec, *, run_id: str, assume_decision: str = "") -> str:
    """Fingerprint the plan a ledger belongs to.

    Wave keys carry a monotonic sequence number, so replaying them is only sound when
    the traversal is identical. If the spec or its config changed between runs the path
    can diverge and every wave after the divergence would be resubmitted — the exact
    double-spend ``--resume`` exists to prevent. Hashing the flattened plan (states,
    iterations, commands, resources) makes that detectable instead of silent.
    """

    import hashlib
    import json as _json

    plan = build_plan(spec, run_id=run_id, assume_decision=assume_decision)
    payload = _json.dumps(
        {
            "workflow": spec.name,
            "api_version": spec.api_version,
            "steps": [
                {
                    "state": step.state,
                    "iteration": step.iteration,
                    "group": step.group,
                    "argv": step.argv,
                    "shell": step.shell,
                    "resources": step.resources,
                }
                for step in plan.steps
            ],
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


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
        name_lookup_fn: Callable[[str], list[str]] | None = None,
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
        self._name_lookup_fn = name_lookup_fn
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

        # Shared with the offline `--waves` preview so the two can never disagree.
        # The CLI knob is a *cap*: it only lowers a group's declared maxConcurrency.
        batches = split_into_batches(
            steps, max_concurrency, cap=self.options.max_concurrency
        )
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

        if self.options.resume:
            adopted = self._reconcile_in_flight(key, steps, kind=kind, group=group)
            if adopted is not None:
                return adopted

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
            except BaseException as exc:  # noqa: BLE001 - see _abort_wave
                # ANY abort (workflow error, transient tooling error, KeyboardInterrupt)
                # must cancel the managed job we just launched: leaving it running
                # bills GPUs for a driver that is no longer watching it.
                self._abort_wave(attempt, exc)
                last_error = attempt.error
                if not isinstance(exc, Exception):
                    raise  # BaseException (Ctrl-C, SystemExit): cancel, then propagate
                if not isinstance(exc, NpaWorkflowError):
                    # Unexpected tooling failure: do not silently retry into more
                    # spend — surface it as a workflow failure with the cause.
                    return self.attempts[-1]
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

    def _reconcile_in_flight(
        self,
        key: str,
        steps: Sequence[PlanStep],
        *,
        kind: str,
        group: str,
    ) -> WaveAttempt | None:
        """Adopt (or clear) a wave the ledger left in flight before resubmitting it.

        A driver that died mid-poll leaves a ``running`` record whose managed job may
        still be burning GPUs. Blindly resubmitting on ``--resume`` would double the
        spend, so the recorded job is queried first:

        * already succeeded → adopt it, no resubmission;
        * still non-terminal → keep polling **that** job (never a second copy);
        * terminal failure → clear it and fall through to a fresh attempt.
        """

        record = self.ledger.in_flight_wave(key)
        if record is None:
            return None
        job_id = str(record.get("job_id") or "")
        job_name = str(record.get("job_name") or "")
        if not job_id:
            self._log(
                f"wave {key}: ledger has an in-flight record without a job id; "
                "resubmitting"
            )
            return None

        attempt = WaveAttempt(
            key=key,
            states=[step.state for step in steps],
            kind=kind,
            group=group,
            attempt=int(record.get("attempt") or 1),
            job_id=job_id,
            job_name=job_name,
            started_at=str(record.get("started_at") or utc_now()),
            outputs=[item["uri"] for step in steps for item in step.outputs],
            adopted=True,
        )
        self.attempts.append(attempt)

        try:
            status = str(getattr(self._status(job_id), "status", "") or "UNKNOWN").upper()
        except Exception as exc:  # noqa: BLE001 - cannot reconcile blind
            self._abort_wave(
                attempt,
                NpaWorkflowError(
                    f"wave {key}: cannot determine the state of in-flight job {job_id} "
                    f"recorded by a previous run ({exc}); refusing to submit a second "
                    "copy. Cancel it manually or drop --resume."
                ),
            )
            return attempt

        if is_terminal_ok(status):
            self._log(f"wave {key}: in-flight job {job_id} already succeeded; adopting it")
            attempt.status = "succeeded"
            attempt.sky_status = status
            attempt.ended_at = utc_now()
            attempt.tasks = self._timeline(job_id)
            self.ledger.record(attempt)
            return attempt

        if is_terminal_fail(status):
            self._log(
                f"wave {key}: in-flight job {job_id} ended {status}; retrying with a "
                "fresh submission"
            )
            self.attempts.pop()
            return None

        self._log(
            f"wave {key}: job {job_id} from a previous driver is still {status}; "
            "attaching to it instead of submitting a second copy"
        )
        attempt.status = "running"
        attempt.sky_status = status
        self.ledger.record(attempt)
        try:
            final_status = self._poll(job_id, attempt, observe_tasks=len(steps) > 1)
            attempt.sky_status = final_status
            attempt.tasks = self._timeline(job_id)
            if not is_terminal_ok(final_status):
                raise NpaWorkflowError(
                    f"wave {key} (adopted job {job_id}) reached terminal status "
                    f"{final_status}"
                )
        except BaseException as exc:  # noqa: BLE001 - same abort contract as a fresh wave
            self._abort_wave(attempt, exc)
            if not isinstance(exc, Exception):
                raise
            return attempt
        attempt.status = "succeeded"
        attempt.ended_at = utc_now()
        self.ledger.record(attempt)
        return attempt

    def _abort_wave(self, attempt: WaveAttempt, exc: BaseException) -> None:
        """Record a failed attempt and cancel its managed job if one is in flight."""

        attempt.status = "failed"
        if isinstance(exc, NpaWorkflowError):
            attempt.error = str(exc)
        else:
            attempt.error = f"{type(exc).__name__}: {exc}"
        attempt.ended_at = utc_now()
        # Cancel whenever something may still be running. With a job id we cancel the
        # managed job; with only a name (e.g. the submit reported no id) the canceller
        # still tears the cluster down, which is what stops the billing.
        if attempt.job_name and not is_terminal(attempt.sky_status):
            self._log(
                f"wave {attempt.key}: aborting with job "
                f"{attempt.job_id or attempt.job_name} possibly in flight "
                f"({attempt.error}); cancelling it"
            )
            self._cancel(attempt.job_id, attempt.job_name)
            attempt.sky_status = "CANCELLED"
        self.ledger.record(attempt)

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
        job_id = self._resolve_job_id(
            job_name, str(getattr(result, "job_id", "") or "").strip(), attempt
        )
        attempt.job_id = job_id
        status = str(getattr(result, "status", "SUBMITTED") or "SUBMITTED").upper()
        attempt.sky_status = status
        attempt.status = "running"
        # Persist the in-flight wave immediately: if the driver dies mid-wave the
        # ledger still names the managed job, so an operator can find (and cancel)
        # it instead of leaking a cluster.
        self.ledger.record(attempt)
        self._log(f"wave {attempt.key}: submitted job_id={job_id} name={job_name}")

        final_status = self._poll(job_id, attempt, observe_tasks=len(steps) > 1)
        attempt.sky_status = final_status
        attempt.tasks = self._timeline(job_id)
        if not is_terminal_ok(final_status):
            raise NpaWorkflowError(
                f"wave {attempt.key} reached terminal status {final_status} "
                f"(job_id={job_id}, name={job_name})"
            )

    def _poll(self, job_id: str, attempt: WaveAttempt, *, observe_tasks: bool = False) -> str:
        deadline = self._clock() + max(1, self.options.max_wait_seconds)
        last = "UNKNOWN"
        consecutive_status_errors = 0
        while True:
            try:
                current = self._status(job_id)
            except Exception as exc:  # noqa: BLE001 - a status hiccup must not abort a job
                # `sky jobs queue` can time out or trip over a busy API server. The
                # job itself is unaffected, so keep polling (bounded) instead of
                # bubbling out of the retry logic and orphaning a running GPU job.
                consecutive_status_errors += 1
                attempt.status_errors.append(f"{type(exc).__name__}: {exc}"[:200])
                if consecutive_status_errors > MAX_CONSECUTIVE_STATUS_ERRORS:
                    raise NpaWorkflowError(
                        f"wave {attempt.key}: {consecutive_status_errors} consecutive "
                        f"status queries failed for job {job_id}; last error: {exc}"
                    ) from exc
                self._log(
                    f"wave {attempt.key}: status query {consecutive_status_errors} failed "
                    f"({exc}); job {job_id} still running, retrying"
                )
                if self._clock() >= deadline:
                    if self.options.cancel_on_timeout:
                        self._cancel(job_id, attempt.job_name)
                    raise NpaWorkflowError(
                        f"wave {attempt.key} did not reach a terminal status within "
                        f"{self.options.max_wait_seconds}s (last={last}, job_id={job_id})"
                    ) from exc
                self._sleep(self.options.poll_seconds)
                continue
            consecutive_status_errors = 0
            last = str(getattr(current, "status", "") or "UNKNOWN").upper()
            if observe_tasks:
                self._observe_concurrency(job_id, attempt)
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

    def _resolve_job_id(self, job_name: str, parsed: str, attempt: WaveAttempt) -> str:
        """Trust the launched job NAME, not the id scraped from launch output.

        A flaky API server can leave a stale ``Job submitted, ID: N`` in the stream:
        live, the driver parsed a cancelled job's id, declared the wave CANCELLED, and
        walked away from the real job — four GPUs kept running. So the parsed id is
        cross-checked against the job name, and recovered from it when it disagrees.
        """

        names = self._job_ids_by_name(job_name)
        if names:
            if parsed and parsed in names:
                return parsed
            resolved = names[0]
            if parsed:
                self._log(
                    f"wave {attempt.key}: launch output reported job_id={parsed}, but "
                    f"{job_name!r} is job_id={resolved}; using the name lookup"
                )
                attempt.status_errors.append(
                    f"stale job id from launch output: {parsed} != {resolved}"
                )
            return resolved
        if parsed:
            # Name lookup unavailable (e.g. the queue query failed); the parsed id is
            # all we have, and polling it is better than abandoning the job.
            return parsed
        raise NpaWorkflowError(
            f"wave {attempt.key}: SkyPilot reported no job id for {job_name!r} and the "
            "job could not be found by name; refusing to poll a job we cannot identify"
        )

    def _job_ids_by_name(self, job_name: str) -> list[str]:
        lookup = self._name_lookup_fn
        if lookup is None:
            from npa.orchestration.skypilot.workflow import (
                find_job_ids_by_name as lookup,
            )

        try:
            return [str(item) for item in lookup(job_name)]
        except Exception as exc:  # noqa: BLE001 - fall back to the parsed id
            self._log(f"job-id lookup by name failed for {job_name}: {exc}")
            return []

    def _observe_concurrency(self, job_id: str, attempt: WaveAttempt) -> None:
        """Record how many member tasks are RUNNING at this instant."""

        tasks = self._timeline(job_id)
        if not tasks:
            return
        running = [
            str(task.get("task_name") or task.get("task_id"))
            for task in tasks
            if str(task.get("status") or "").upper() in {"RUNNING", "RECOVERING"}
        ]
        observation = {
            "observed_at": utc_now(),
            "running": sorted(running),
            "running_count": len(running),
            "statuses": {
                str(task.get("task_name") or task.get("task_id")): task.get("status")
                for task in tasks
            },
        }
        attempt.observations.append(observation)
        attempt.max_concurrent_observed = max(attempt.max_concurrent_observed, len(running))
        if len(running) > 1:
            self._log(
                f"wave {attempt.key}: {len(running)} tasks running concurrently "
                f"({', '.join(sorted(running))})"
            )

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
    """Durable wave ledger (``npa.workflow.runtime.v1``) with in-memory fallback.

    **Single writer by design.** ``flush`` rewrites the whole document, so two drivers
    on the same ``run_id`` would clobber each other's records, and ``--resume``
    reconciliation assumes exactly one prior driver. Run one driver per run id; use a
    fresh run id (or a different ``config.prefix``) for a concurrent run.
    """

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

    def in_flight_wave(self, key: str) -> dict[str, Any] | None:
        return self.state.in_flight_wave(key)

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
    """Wrap a decision reader so every runtime gate read lands in the ledger.

    Also implements the *documented* fallback: when the gate artifact does not exist
    yet, the reader synthesizes the plan-time assumption instead of hard-failing the
    run. A corrupt artifact is still an error — silently looping on unreadable JSON
    would be worse than stopping.
    """

    def __init__(
        self,
        reader: Callable[[str, str], str] | None,
        ledger: RuntimeLedger,
        *,
        assume_decision: str = "",
        logger: Callable[[str], None] | None = None,
    ) -> None:
        self._reader = reader
        self._ledger = ledger
        self._assume = normalize_decision(assume_decision or "")
        self._log = logger or (lambda message: None)
        self.reads: list[dict[str, Any]] = []
        self.missing: list[str] = []

    def __call__(self, bucket: str, key: str) -> str:
        uri = f"s3://{bucket}/{key}"
        try:
            body = self._read(bucket, key)
        except FileNotFoundError:
            if not self._assume:
                raise
            # No gate artifact yet (a stage that did not write one, or an eval that
            # produced nothing): fall back to the plan-time assumption so the run
            # keeps the documented offline behaviour instead of dying.
            self._log(
                f"decision artifact {uri} not found; falling back to the plan-time "
                f"assumption {self._assume!r}"
            )
            self.missing.append(uri)
            payload = {
                "uri": uri,
                "decision": self._assume,
                "source": "assume_decision_fallback",
                "read_at": utc_now(),
            }
            self._ledger.record_decision(payload)
            self.reads.append(payload)
            import json as _json

            return _json.dumps({"decision": self._assume})
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
        except (ValueError, TypeError) as exc:
            # Record why the ledger entry has no normalized decision; the caller
            # (decisions.load_decision) still raises on a malformed payload.
            payload["decode_error"] = str(exc)[:200]
        self.reads.append(payload)
        self._ledger.record_decision(payload)
        return body

    def _read(self, bucket: str, key: str) -> str:
        if self._reader is not None:
            return self._reader(bucket, key)
        from npa.orchestration.npa_workflow.decisions import _read_object

        return _read_object(bucket, key)


def s3_trigger_waiter(
    *,
    ledger: RuntimeLedger | None = None,
    lister: Callable[[str, str], Iterable[str]] | None = None,
    sleeper: Callable[[float], None] = time.sleep,
    logger: Callable[[str], None] | None = None,
    max_wait_seconds: int = DEFAULT_MAX_WAIT_SECONDS,
    clock: Callable[[], float] = time.monotonic,
) -> Callable[[StateSpec, str, RunContext], dict[str, Any]]:
    """Build a driver-side watcher for ``trigger:`` states.

    Polls an object-storage prefix until at least ``minObjects`` keys are present,
    recording the observed watermark in the ledger so a resumed run does not wait
    again for data it already saw.

    The wait is bounded twice over: by ``trigger.maxPolls`` when the spec sets one,
    and **always** by ``max_wait_seconds`` (the run's per-wave deadline). Without the
    second bound a spec that leaves ``maxPolls`` at its default of 0 would wait for
    data forever instead of failing the run.
    """

    log = logger or (lambda message: None)

    def _list(uri: str) -> list[str]:
        from npa.orchestration.npa_workflow.decisions import parse_s3_uri

        bucket, prefix = parse_s3_uri(uri)
        if lister is not None:
            return list(lister(bucket, prefix))
        from npa.clients.storage import StorageClient

        # Paginate: a single list_objects_v2 caps at 1000 keys, so a trigger with
        # minObjects > 1000 (or a busy prefix) could never be satisfied and would spin
        # until the deadline. Same pattern as rl_sweep/_list_keys.
        s3 = StorageClient.from_environment().s3
        keys: list[str] = []
        token: str | None = None
        while True:
            kwargs: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
            if token:
                kwargs["ContinuationToken"] = token
            page = s3.list_objects_v2(**kwargs)
            keys.extend(str(item["Key"]) for item in page.get("Contents") or ())
            if not page.get("IsTruncated"):
                return keys
            token = page.get("NextContinuationToken")

    def waiter(state: StateSpec, uri: str, ctx: RunContext) -> dict[str, Any]:
        trigger = state.trigger
        assert trigger is not None
        if ledger is not None:
            seen = ledger.state.watermarks.get(state.name)
            if seen and int(seen.get("objects") or 0) >= trigger.min_objects:
                return dict(seen)
        polls = 0
        deadline = clock() + max(1, max_wait_seconds)
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
            if clock() >= deadline:
                raise NpaWorkflowError(
                    f"state {state.name}: trigger {uri} still has {len(keys)} object(s) "
                    f"(need {trigger.min_objects}) after waiting {max_wait_seconds}s"
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
    #: Trigger watermarks observed during the run, keyed by state name.
    watermarks: dict[str, Any] = field(default_factory=dict)
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
            "watermarks": dict(self.watermarks),
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

    ``assume_decision`` is only a *fallback* here, and a narrow one: when a gate
    artifact does not exist yet the reader synthesizes the assumption so the run keeps
    the documented plan-time behaviour. A gate artifact that exists but is unreadable
    or malformed still fails the run — silently looping on corrupt JSON would be worse
    than stopping. Every real decision comes from S3 and is recorded in the ledger.
    """

    opts = options or RuntimeOptions()
    log = logger or (lambda message: None)

    if executor is not None:
        ledger = executor.ledger
    else:
        store = state_store
        if store is None:
            store = store_for_config(_resolved_config(spec, run_id), run_id=run_id)
        if store is None:
            message = (
                "config.bucket is not set, so no runtime ledger can be written: "
                "--resume will have nothing to replay and every wave would be "
                "resubmitted"
            )
            if opts.resume:
                raise NpaWorkflowError(
                    f"{message}. Set config.bucket (or --var bucket=...) before "
                    "resuming."
                )
            log(f"warning: {message}")
        ledger = RuntimeLedger(
            store,
            workflow=spec.name,
            run_id=run_id,
            api_version=spec.api_version,
            resume=opts.resume,
        )
    fingerprint = plan_fingerprint(spec, run_id=run_id, assume_decision=assume_decision)
    recorded = ledger.state.plan_fingerprint
    if opts.resume and recorded and recorded != fingerprint:
        raise NpaWorkflowError(
            f"refusing to resume run {run_id!r}: the recorded ledger describes a "
            f"different plan (fingerprint {recorded} != {fingerprint}). Resuming would "
            "replay wave keys that no longer describe the same work and could submit "
            "duplicate jobs. Re-run without --resume under a NEW run id, or restore the "
            "spec/--var values the run started with."
        )
    if recorded != fingerprint:
        ledger.state.plan_fingerprint = fingerprint
    ledger.set_status("running")

    recording_reader = RecordingDecisionReader(
        decision_reader, ledger, assume_decision=assume_decision, logger=log
    )
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
        waiter = s3_trigger_waiter(
            ledger=ledger, logger=log, max_wait_seconds=opts.max_wait_seconds
        )

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
    except Exception as exc:  # noqa: BLE001 - a long-running driver must always
        # reach a terminal ledger status; the error type is kept in the report so
        # an unexpected crash is not mistaken for a workflow-level failure.
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
        log(f"runtime driver crashed: {error}")
    ledger.set_status(status)

    return RuntimeReport(
        workflow=spec.name,
        run_id=run_id,
        status=status,
        waves=[attempt.to_dict() for attempt in wave_executor.attempts],
        decisions=list(ledger.state.decisions),
        steps=steps,
        watermarks=dict(ledger.state.watermarks),
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
