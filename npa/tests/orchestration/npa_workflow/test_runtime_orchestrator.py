"""Unit coverage for the npa.workflow runtime orchestrator tier.

Every dependency is faked: no SkyPilot, no S3, no cluster, no sleeping. The tests
pin the behaviours the runtime tier exists for — real early-exit, data-dependent
branching, concurrent fan-out with bounded concurrency and a barrier, retry,
resume/idempotency and the trigger/watch pattern — plus the guarantee that the
runtime traversal matches the plan-time unroll for the same decisions.
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml

from npa.orchestration.npa_workflow import build_plan, load_spec
from npa.orchestration.npa_workflow.run_state import RunStateStore, RuntimeRunState
from npa.orchestration.npa_workflow.runtime import (
    RuntimeLedger,
    RuntimeOptions,
    SkyPilotWaveExecutor,
    run_workflow_runtime,
    s3_trigger_waiter,
)
from npa.orchestration.npa_workflow.skypilot_render import SkypilotRenderOptions

GATE_LOOP_SPEC = """
apiVersion: npa.workflow/v0.0.1
kind: Workflow

metadata:
  name: gate-loop-demo

config:
  bucket: example-bucket
  prefix: "gate-loop/{{run.id}}"
  max_iterations: 3
  decision_uri: "s3://{{config.bucket}}/{{config.prefix}}/gate/decision.json"

resources:
  cpu:
    cloud: kubernetes
    cpus: 4
    memory: 16Gi

initial: refine

states:
  refine:
    description: Bounded refine loop with a runtime gate.
    loop:
      max: "{{config.max_iterations}}"
      until: promote_checkpoint
    sequence:
      - work
      - gate
    next: publish

  work:
    description: Do the work.
    run:
      shell: "echo work {{run.id}}"
    resources: cpu

  gate:
    description: Write the decision artifact.
    writesDecision: true
    needs: [work]
    run:
      shell: "echo gate {{config.decision_uri}}"
    resources: cpu
    outputs:
      - uri: "{{config.decision_uri}}"
        schema: npa.sim2real.threshold_decision.v1
    transitions:
      - when: promote_checkpoint
        goto: publish
      - when: loop_back
        goto: refine

  publish:
    description: Publish once promoted.
    run:
      shell: "echo publish"
    resources: cpu
    terminal: true
"""

FANOUT_SPEC = """
apiVersion: npa.workflow/v0.0.1
kind: Workflow

metadata:
  name: fanout-runtime-demo

config:
  bucket: example-bucket
  prefix: "fanout/{{run.id}}"

resources:
  cpu:
    cloud: kubernetes
    cpus: 4
    memory: 16Gi

initial: shards

states:
  shards:
    description: Three concurrent shards, two at a time.
    parallel: [shard-a, shard-b, shard-c]
    maxConcurrency: 2
    next: join

  shard-a:
    description: Shard A.
    run:
      shell: "echo a"
    resources: cpu

  shard-b:
    description: Shard B.
    run:
      shell: "echo b"
    resources: cpu

  shard-c:
    description: Shard C.
    run:
      shell: "echo c"
    resources: cpu

  join:
    description: Barrier.
    needs: [shards]
    run:
      shell: "echo join"
    resources: cpu
    terminal: true
"""

TRIGGER_SPEC = """
apiVersion: npa.workflow/v0.0.1
kind: Workflow

metadata:
  name: trigger-demo

config:
  bucket: example-bucket
  prefix: "trigger/{{run.id}}"

resources:
  cpu:
    cloud: kubernetes
    cpus: 4
    memory: 16Gi

initial: ingest

states:
  ingest:
    description: Wait for fresh data, then ingest it.
    trigger:
      uri: "s3://{{config.bucket}}/{{config.prefix}}/inbox/"
      pollSeconds: 1
      maxPolls: 5
    run:
      shell: "echo ingest"
    resources: cpu
    terminal: true
"""


# --------------------------------------------------------------------------- fakes


@dataclass
class FakeResult:
    status: str = "SUBMITTED"
    job_id: str = "1"


class FakeSubmitter:
    """Captures every wave submission (rendered YAML + job name)."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, path: Path, job_name: str, **kwargs: Any) -> FakeResult:
        text = Path(path).read_text(encoding="utf-8")
        docs = [doc for doc in yaml.safe_load_all(text) if doc is not None]
        self.calls.append(
            {
                "job_name": job_name,
                "yaml": text,
                "header": docs[0],
                "tasks": [doc["name"] for doc in docs[1:]],
                "kwargs": kwargs,
            }
        )
        return FakeResult(job_id=str(len(self.calls)))


class FakeStatus:
    """Scripted managed-job statuses; defaults to immediate success."""

    def __init__(self, statuses: list[str] | None = None) -> None:
        self.statuses = list(statuses or [])
        self.calls: list[str] = []

    def __call__(self, job_id: str, **_: Any) -> FakeResult:
        self.calls.append(job_id)
        status = self.statuses.pop(0) if self.statuses else "SUCCEEDED"
        return FakeResult(status=status, job_id=job_id)


class MemoryStore(RunStateStore):
    """RunStateStore backed by a dict (mirrors the injected reader/writer seam)."""

    def __init__(self, objects: dict[str, bytes] | None = None) -> None:
        self.objects: dict[str, bytes] = objects if objects is not None else {}
        super().__init__(
            bucket="unit-bucket",
            prefix="unit-prefix",
            reader=self._read_obj,
            writer=self._write_obj,
        )

    def _read_obj(self, bucket: str, key: str) -> str:
        if key not in self.objects:
            raise FileNotFoundError(f"s3://{bucket}/{key}")
        return self.objects[key].decode("utf-8")

    def _write_obj(self, bucket: str, key: str, body: bytes) -> None:
        self.objects[key] = body


def _write_spec(tmp_path: Path, text: str, name: str = "spec.yaml") -> Path:
    path = tmp_path / name
    path.write_text(textwrap.dedent(text).lstrip(), encoding="utf-8")
    return path


def _decision_reader(sequence: list[str]):
    remaining = list(sequence)

    def reader(_bucket: str, _key: str) -> str:
        decision = remaining.pop(0) if remaining else sequence[-1]
        return json.dumps({"decision": decision})

    return reader


def _executor(
    spec,
    *,
    run_id: str = "rt-1",
    submitter: FakeSubmitter | None = None,
    status_fn: FakeStatus | None = None,
    options: RuntimeOptions | None = None,
    store: MemoryStore | None = None,
    sleeps: list[float] | None = None,
    cancels: list[dict[str, Any]] | None = None,
) -> SkyPilotWaveExecutor:
    opts = options or RuntimeOptions(poll_seconds=0, max_wait_seconds=60)
    ledger = RuntimeLedger(
        store,
        workflow=spec.name,
        run_id=run_id,
        api_version=spec.api_version,
        resume=opts.resume,
    )
    return SkyPilotWaveExecutor(
        spec,
        run_id=run_id,
        render_options=SkypilotRenderOptions(image_overrides={"*": "cr.example/x:1"}),
        options=opts,
        ledger=ledger,
        submitter=submitter or FakeSubmitter(),
        status_fn=status_fn or FakeStatus(),
        timeline_fn=lambda job_id: [
            {"task_id": 0, "task_name": "t", "status": "SUCCEEDED", "job_id": job_id}
        ],
        canceller=(lambda **kwargs: cancels.append(kwargs)) if cancels is not None else None,
        sleeper=(sleeps.append if sleeps is not None else (lambda _seconds: None)),
        clock=_fake_clock(),
    )


def _fake_clock():
    ticks = {"now": 0.0}

    def clock() -> float:
        ticks["now"] += 1.0
        return ticks["now"]

    return clock


# ------------------------------------------------------------------- early exit


def test_runtime_early_exits_when_gate_promotes_on_first_iteration(tmp_path: Path) -> None:
    spec = load_spec(_write_spec(tmp_path, GATE_LOOP_SPEC))
    submitter = FakeSubmitter()
    executor = _executor(spec, submitter=submitter)

    report = run_workflow_runtime(
        spec,
        run_id="rt-early",
        executor=executor,
        options=executor.options,
        decision_reader=_decision_reader(["promote_checkpoint"]),
    )

    assert report.status == "succeeded"
    submitted_states = [call["tasks"] for call in submitter.calls]
    assert submitted_states == [["work"], ["gate"], ["publish"]]
    # Real early-exit: the loop budget is 3 but only one iteration ran.
    assert sum(1 for call in submitter.calls if call["tasks"] == ["work"]) == 1
    assert report.decisions and report.decisions[-1]["decision"] == "promote_checkpoint"


def test_runtime_runs_full_budget_when_gate_keeps_looping(tmp_path: Path) -> None:
    spec = load_spec(_write_spec(tmp_path, GATE_LOOP_SPEC))
    submitter = FakeSubmitter()
    executor = _executor(spec, submitter=submitter)

    report = run_workflow_runtime(
        spec,
        run_id="rt-full",
        executor=executor,
        options=executor.options,
        decision_reader=_decision_reader(["loop_back", "loop_back", "loop_back"]),
    )

    assert report.status == "succeeded"
    assert sum(1 for call in submitter.calls if call["tasks"] == ["work"]) == 3
    assert sum(1 for call in submitter.calls if call["tasks"] == ["gate"]) == 3
    assert submitter.calls[-1]["tasks"] == ["publish"]
    assert len(report.decisions) == 3


def test_runtime_branch_follows_transition_goto(tmp_path: Path) -> None:
    """A promote decision routes straight to the transition target (data-dependent goto)."""

    spec = load_spec(_write_spec(tmp_path, GATE_LOOP_SPEC))
    submitter = FakeSubmitter()
    executor = _executor(spec, submitter=submitter)

    run_workflow_runtime(
        spec,
        run_id="rt-goto",
        executor=executor,
        options=executor.options,
        decision_reader=_decision_reader(["loop_back", "promote_checkpoint"]),
    )

    ordered = [call["tasks"][0] for call in submitter.calls]
    assert ordered == ["work", "gate", "work", "gate", "publish"]


@pytest.mark.parametrize(
    ("decisions", "assume"),
    [
        (["promote_checkpoint"], "promote_checkpoint"),
        (["loop_back", "loop_back", "loop_back"], "loop_back"),
    ],
)
def test_runtime_sequence_matches_plan_time_unroll(
    tmp_path: Path, decisions: list[str], assume: str
) -> None:
    """Anti-divergence guard: same decisions -> same step sequence as --assume-decision."""

    spec = load_spec(_write_spec(tmp_path, GATE_LOOP_SPEC))
    submitter = FakeSubmitter()
    executor = _executor(spec, submitter=submitter)
    run_workflow_runtime(
        spec,
        run_id="rt-eq",
        executor=executor,
        options=executor.options,
        decision_reader=_decision_reader(decisions),
    )
    runtime_states = [call["tasks"][0] for call in submitter.calls]
    plan_states = [step.state for step in build_plan(spec, run_id="rt-eq", assume_decision=assume).steps]
    assert runtime_states == plan_states


# ---------------------------------------------------------------------- fan-out


def test_runtime_launches_parallel_group_as_job_group_with_barrier(tmp_path: Path) -> None:
    spec = load_spec(_write_spec(tmp_path, FANOUT_SPEC))
    submitter = FakeSubmitter()
    executor = _executor(spec, submitter=submitter)

    report = run_workflow_runtime(
        spec, run_id="rt-fanout", executor=executor, options=executor.options
    )

    assert report.status == "succeeded"
    # maxConcurrency: 2 over three members -> [a,b] as a JobGroup, then [c], then join.
    assert [call["tasks"] for call in submitter.calls] == [
        ["shard-a", "shard-b"],
        ["shard-c"],
        ["join"],
    ]
    assert submitter.calls[0]["header"]["execution"] == "parallel"
    assert submitter.calls[1]["header"]["execution"] == "serial"
    assert submitter.calls[2]["header"]["execution"] == "serial"
    # Barrier: the downstream state is submitted only after the group's waves.
    assert submitter.calls[-1]["tasks"] == ["join"]
    assert [wave["kind"] for wave in report.waves] == ["parallel", "serial", "serial"]


def test_runtime_max_concurrency_option_is_a_cap_not_an_override(tmp_path: Path) -> None:
    """--max-concurrency can only lower a group's declared bound (cost control)."""

    spec = load_spec(_write_spec(tmp_path, FANOUT_SPEC))  # group declares 2

    tighter = FakeSubmitter()
    tight_options = RuntimeOptions(poll_seconds=0, max_wait_seconds=60, max_concurrency=1)
    run_workflow_runtime(
        spec,
        run_id="rt-tight",
        executor=_executor(spec, submitter=tighter, options=tight_options),
        options=tight_options,
    )
    assert [call["tasks"] for call in tighter.calls] == [
        ["shard-a"],
        ["shard-b"],
        ["shard-c"],
        ["join"],
    ]

    wider = FakeSubmitter()
    wide_options = RuntimeOptions(poll_seconds=0, max_wait_seconds=60, max_concurrency=8)
    run_workflow_runtime(
        spec,
        run_id="rt-wide",
        executor=_executor(spec, submitter=wider, options=wide_options),
        options=wide_options,
    )
    # Still two batches: the spec's maxConcurrency: 2 is respected.
    assert [call["tasks"] for call in wider.calls] == [
        ["shard-a", "shard-b"],
        ["shard-c"],
        ["join"],
    ]


def test_parallel_group_failure_stops_the_barrier(tmp_path: Path) -> None:
    spec = load_spec(_write_spec(tmp_path, FANOUT_SPEC))
    submitter = FakeSubmitter()
    status_fn = FakeStatus(["FAILED"])
    executor = _executor(spec, submitter=submitter, status_fn=status_fn)

    report = run_workflow_runtime(
        spec, run_id="rt-fanout-fail", executor=executor, options=executor.options
    )

    assert report.status == "failed"
    assert [call["tasks"] for call in submitter.calls] == [["shard-a", "shard-b"]]
    assert "join" not in [state for wave in report.waves for state in wave["states"]]


# ------------------------------------------------------------ retry / resume / timeout


def test_wave_retry_recovers_from_a_transient_failure(tmp_path: Path) -> None:
    spec = load_spec(_write_spec(tmp_path, FANOUT_SPEC))
    submitter = FakeSubmitter()
    options = RuntimeOptions(
        poll_seconds=0, max_wait_seconds=60, retries=1, retry_backoff_seconds=0
    )
    executor = _executor(
        spec, submitter=submitter, status_fn=FakeStatus(["FAILED"]), options=options
    )

    report = run_workflow_runtime(
        spec, run_id="rt-retry", executor=executor, options=options
    )

    assert report.status == "succeeded"
    attempts = [wave for wave in report.waves if wave["states"] == ["shard-a", "shard-b"]]
    assert [wave["attempt"] for wave in attempts] == [1, 2]
    assert attempts[0]["status"] == "failed"
    assert attempts[1]["status"] == "succeeded"


def test_wave_retry_exhausted_fails_the_run(tmp_path: Path) -> None:
    spec = load_spec(_write_spec(tmp_path, FANOUT_SPEC))
    options = RuntimeOptions(
        poll_seconds=0, max_wait_seconds=60, retries=1, retry_backoff_seconds=0
    )
    executor = _executor(
        spec, status_fn=FakeStatus(["FAILED", "FAILED"]), options=options
    )

    report = run_workflow_runtime(spec, run_id="rt-retry-fail", executor=executor, options=options)

    assert report.status == "failed"
    assert "FAILED" in report.error


def test_timeout_cancels_the_managed_job(tmp_path: Path) -> None:
    spec = load_spec(_write_spec(tmp_path, FANOUT_SPEC))
    cancels: list[dict[str, Any]] = []
    options = RuntimeOptions(poll_seconds=0, max_wait_seconds=2, cancel_on_timeout=True)
    executor = _executor(
        spec,
        status_fn=FakeStatus(["RUNNING"] * 20),
        options=options,
        cancels=cancels,
    )

    report = run_workflow_runtime(spec, run_id="rt-timeout", executor=executor, options=options)

    assert report.status == "failed"
    assert "did not reach a terminal status" in report.error
    assert cancels and cancels[0]["job_id"]


def test_resume_replays_completed_waves_instead_of_resubmitting(tmp_path: Path) -> None:
    spec = load_spec(_write_spec(tmp_path, FANOUT_SPEC))
    store = MemoryStore()

    first_submitter = FakeSubmitter()
    first = _executor(
        spec,
        run_id="rt-resume",
        submitter=first_submitter,
        status_fn=FakeStatus(["SUCCEEDED", "SUCCEEDED", "FAILED"]),
        store=store,
    )
    first_report = run_workflow_runtime(
        spec, run_id="rt-resume", executor=first, options=first.options
    )
    assert first_report.status == "failed"
    assert len(first_submitter.calls) == 3

    # Ledger survived on the (in-memory) object store.
    persisted = store.read_runtime_state()
    assert isinstance(persisted, RuntimeRunState)
    assert [wave["status"] for wave in persisted.waves] == ["succeeded", "succeeded", "failed"]

    second_submitter = FakeSubmitter()
    resume_options = RuntimeOptions(poll_seconds=0, max_wait_seconds=60, resume=True)
    second = _executor(
        spec,
        run_id="rt-resume",
        submitter=second_submitter,
        options=resume_options,
        store=store,
    )
    second_report = run_workflow_runtime(
        spec, run_id="rt-resume", executor=second, options=resume_options
    )

    assert second_report.status == "succeeded"
    # Only the previously failed wave is resubmitted; the first two are replayed.
    assert [call["tasks"] for call in second_submitter.calls] == [["join"]]
    assert [wave["replayed"] for wave in second_report.waves] == [True, True, False]


# --------------------------------------------------------------------- trigger


def test_trigger_waits_for_objects_then_runs(tmp_path: Path) -> None:
    spec = load_spec(_write_spec(tmp_path, TRIGGER_SPEC))
    submitter = FakeSubmitter()
    store = MemoryStore()
    executor = _executor(spec, run_id="rt-trigger", submitter=submitter, store=store)

    listings = [[], [], ["inbox/a.json"]]
    sleeps: list[float] = []

    def lister(_bucket: str, _prefix: str) -> list[str]:
        return listings.pop(0) if listings else ["inbox/a.json"]

    waiter = s3_trigger_waiter(
        ledger=executor.ledger, lister=lister, sleeper=sleeps.append
    )
    report = run_workflow_runtime(
        spec,
        run_id="rt-trigger",
        executor=executor,
        options=executor.options,
        trigger_waiter=waiter,
    )

    assert report.status == "succeeded"
    assert sleeps == [1, 1]  # two empty polls before data arrived
    assert executor.ledger.state.watermarks["ingest"]["objects"] == 1
    assert [call["tasks"] for call in submitter.calls] == [["ingest"]]


def test_trigger_gives_up_after_max_polls(tmp_path: Path) -> None:
    spec = load_spec(_write_spec(tmp_path, TRIGGER_SPEC))
    executor = _executor(spec, run_id="rt-trigger-timeout")
    waiter = s3_trigger_waiter(
        ledger=executor.ledger, lister=lambda *_: [], sleeper=lambda _s: None
    )

    report = run_workflow_runtime(
        spec,
        run_id="rt-trigger-timeout",
        executor=executor,
        options=executor.options,
        trigger_waiter=waiter,
    )

    assert report.status == "failed"
    assert "trigger" in report.error


# ----------------------------------------------------------------------- ledger


def test_ledger_persists_waves_and_decisions(tmp_path: Path) -> None:
    spec = load_spec(_write_spec(tmp_path, GATE_LOOP_SPEC))
    store = MemoryStore()
    executor = _executor(spec, run_id="rt-ledger", store=store)

    report = run_workflow_runtime(
        spec,
        run_id="rt-ledger",
        executor=executor,
        options=executor.options,
        decision_reader=_decision_reader(["promote_checkpoint"]),
    )

    assert report.runtime_state_uri.endswith("/npa-workflow/runtime.json")
    persisted = store.read_runtime_state()
    assert persisted is not None
    assert persisted.schema_version == "npa.workflow.runtime.v1"
    assert persisted.status == "succeeded"
    assert [wave["states"] for wave in persisted.waves] == [["work"], ["gate"], ["publish"]]
    assert persisted.decisions[-1]["decision"] == "promote_checkpoint"
    assert persisted.decisions[-1]["uri"].endswith("/gate/decision.json")
