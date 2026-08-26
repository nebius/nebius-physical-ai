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
from npa.orchestration.npa_workflow.errors import NpaWorkflowError
from npa.orchestration.npa_workflow.run_state import RunStateStore, RuntimeRunState
from npa.orchestration.npa_workflow.runtime import (
    RuntimeLedger,
    RuntimeOptions,
    SkyPilotWaveExecutor,
    WaveAttempt,
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

SCOPED_GATE_LOOP_SPEC = """
apiVersion: npa.workflow/v0.0.1
kind: Workflow

metadata:
  name: scoped-gate-loop-demo

config:
  bucket: example-bucket
  prefix: "scoped-gate/{{run.id}}"
  max_iterations: 2
  decision_uri: "s3://{{config.bucket}}/{{config.prefix}}/legacy/decision.json"

resources:
  cpu:
    cloud: kubernetes
    cpus: 4
    memory: 16Gi

initial: refine

states:
  refine:
    description: Bounded loop whose gate has an iteration-scoped decision.
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
      shell: "echo work"
    resources: cpu

  gate:
    description: Write this iteration's decision artifact.
    writesDecision: true
    needs: [work]
    params:
      decision_uri: >-
        s3://{{config.bucket}}/{{config.prefix}}/iteration-{{loop.refine}}/decision.json
    run:
      shell: "echo gate {{config.decision_uri}}"
    resources: cpu
    outputs:
      - uri: "{{config.prefix}}/gate-report.json"
        schema: npa.example.report.v1
      - uri: "{{config.decision_uri}}"
        schema: npa.sim2real.threshold_decision.v1

  publish:
    description: Publish after promotion or loop exhaustion.
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


def test_stage_ledger_heartbeat_requires_real_progress_and_poll_only_updates_observation() -> (
    None
):
    state = RuntimeRunState(
        workflow="synthetic",
        run_id="ledger-heartbeat",
        api_version="npa.workflow/v0.0.1",
    )
    base = {
        "key": "wave-augment",
        "states": ["augment"],
        "attempt": 1,
        "status": "running",
        "sky_status": "RUNNING",
        "job_id": "101",
        "started_at": "2026-08-04T00:00:00Z",
    }
    state.record_wave(
        {
            **base,
            "tasks": [
                {
                    "task_name": "augment",
                    "status": "RUNNING",
                    "last_progress_at": "2026-08-04T00:01:00Z",
                }
            ],
            "observations": [{"observed_at": "2026-08-04T00:02:00Z"}],
        }
    )
    # A later scheduler poll has no progress/heartbeat timestamp.
    state.record_wave(
        {
            **base,
            "tasks": [{"task_name": "augment", "status": "RUNNING"}],
            "observations": [{"observed_at": "2026-08-04T00:07:00Z"}],
        }
    )

    assert len(state.stages) == 1
    stage = state.stages[0]
    assert stage["managed_job_id"] == "101"
    assert stage["last_heartbeat_at"] == "2026-08-04T00:01:00Z"
    assert stage["heartbeat_source"] == "scheduler_task_progress"
    assert stage["last_observed_at"] == "2026-08-04T00:07:00Z"


def test_stage_ledger_keeps_attempts_and_parallel_job_attribution_separate() -> None:
    state = RuntimeRunState(
        workflow="synthetic",
        run_id="ledger-attribution",
        api_version="npa.workflow/v0.0.1",
    )
    state.record_wave(
        {
            "key": "parallel-1",
            "states": ["shard-a", "shard-b"],
            "attempt": 1,
            "status": "failed",
            "sky_status": "FAILED",
            "job_id": "201",
            "ended_at": "2026-08-04T00:03:00Z",
        }
    )
    state.record_wave(
        {
            "key": "parallel-1",
            "states": ["shard-a", "shard-b"],
            "attempt": 2,
            "status": "succeeded",
            "sky_status": "SUCCEEDED",
            "job_id": "202",
            "ended_at": "2026-08-04T00:08:00Z",
        }
    )
    state.record_wave(
        {
            "key": "finalize-1",
            "states": ["finalize"],
            "attempt": 1,
            "status": "succeeded",
            "sky_status": "SUCCEEDED",
            "job_id": "203",
        }
    )

    by_stage_attempt = {(item["stage"], item["attempt"]): item for item in state.stages}
    assert by_stage_attempt[("shard-a", 1)]["managed_job_id"] == "201"
    assert by_stage_attempt[("shard-b", 1)]["managed_job_id"] == "201"
    assert by_stage_attempt[("shard-a", 2)]["managed_job_id"] == "202"
    assert by_stage_attempt[("shard-b", 2)]["terminal_outcome"] == "succeeded"
    assert by_stage_attempt[("finalize", 1)]["managed_job_id"] == "203"


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
    name_lookup_fn: Any | None = None,
    output_checker: Any | None = None,
    reconcile_fn: Any | None = None,
) -> SkyPilotWaveExecutor:
    opts = options or RuntimeOptions(poll_seconds=0, max_wait_seconds=60)
    ledger = RuntimeLedger(
        store,
        workflow=spec.name,
        run_id=run_id,
        api_version=spec.api_version,
        resume=opts.resume,
    )
    effective_lookup = name_lookup_fn or (lambda name: [])

    def default_reconcile(name: str, *, job_id: str = ""):
        from npa.orchestration.skypilot.workflow import ManagedJobEvidence

        ids = [str(item) for item in effective_lookup(name)]
        selected = job_id or (ids[0] if ids else "")
        return (
            ManagedJobEvidence("found", job_id=selected, status="UNKNOWN")
            if selected
            else ManagedJobEvidence("absent")
        )

    return SkyPilotWaveExecutor(
        spec,
        run_id=run_id,
        render_options=SkypilotRenderOptions(
            image_overrides={"*": "cr.example/x@sha256:" + "c" * 64}
        ),
        options=opts,
        ledger=ledger,
        submitter=submitter or FakeSubmitter(),
        status_fn=status_fn or FakeStatus(),
        timeline_fn=lambda job_id: [
            {"task_id": 0, "task_name": "t", "status": "SUCCEEDED", "job_id": job_id}
        ],
        canceller=(lambda **kwargs: cancels.append(kwargs))
        if cancels is not None
        else None,
        # Default: the launched name resolves to the id the fake submitter reported.
        name_lookup_fn=effective_lookup,
        output_checker=output_checker or (lambda _uri: True),
        reconcile_fn=reconcile_fn or default_reconcile,
        sleeper=(sleeps.append if sleeps is not None else (lambda _seconds: None)),
        clock=_fake_clock(),
    )


def _fake_clock():
    ticks = {"now": 0.0}

    def clock() -> float:
        ticks["now"] += 1.0
        return ticks["now"]

    return clock


def test_default_skypilot_calls_preserve_explicit_runtime_isolation(
    tmp_path: Path, mocker
) -> None:
    """Every live control call must use the same run-local Sky state/config."""

    spec = load_spec(_write_spec(tmp_path, TRIGGER_SPEC))
    isolated = tmp_path / "isolated-sky"
    config_path = tmp_path / "sky.yaml"
    sky_bin = tmp_path / "sky"
    options = RuntimeOptions(
        isolated_config_dir=isolated,
        config_path=config_path,
        sky_bin=str(sky_bin),
        infra="k8s/test-context",
    )
    executor = SkyPilotWaveExecutor(
        spec,
        run_id="rt-isolated",
        options=options,
        output_checker=lambda _uri: True,
    )
    submit = mocker.patch(
        "npa.orchestration.skypilot.workflow.submit_workflow",
        return_value=FakeResult(),
    )
    status = mocker.patch(
        "npa.orchestration.skypilot.workflow.workflow_status",
        return_value=FakeResult(status="SUCCEEDED"),
    )
    timeline = mocker.patch(
        "npa.orchestration.skypilot.workflow.workflow_task_statuses",
        return_value=[],
    )
    lookup = mocker.patch(
        "npa.orchestration.skypilot.workflow.find_job_ids_by_name",
        return_value=["1"],
    )
    reconcile = mocker.patch(
        "npa.orchestration.skypilot.workflow.lookup_managed_job",
    )

    rendered = tmp_path / "rendered.yaml"
    rendered.write_text("name: isolated\n", encoding="utf-8")
    attempt = WaveAttempt(
        key="isolated-wave",
        states=["trigger"],
        kind="serial",
        group="",
        attempt=1,
    )
    executor._submit(rendered, "isolated-job", attempt)
    executor._status("1")
    executor._timeline("1")
    assert executor._job_ids_by_name("isolated-job") == ["1"]
    executor._reconcile_exact("isolated-job", "1")

    expected = {
        "isolated_config_dir": isolated,
        "config_path": config_path,
        "sky_bin": str(sky_bin),
    }
    call = submit.call_args
    assert call.args == (rendered, "isolated-job")
    assert callable(call.kwargs["transaction_recorder"])
    assert {
        key: value
        for key, value in call.kwargs.items()
        if key != "transaction_recorder"
    } == {
        **expected,
        "controller_backend": "kubernetes",
        "infra": "k8s/test-context",
        "secret_envs": [],
        "extra_env": {},
        "timeout": 1800,
        "logical_launch_id": "",
    }
    status.assert_called_once_with("1", **expected)
    timeline.assert_called_once_with("1", **expected)
    lookup.assert_called_once_with("isolated-job", **expected)
    reconcile.assert_called_once_with("isolated-job", job_id="1", **expected)


def test_successful_job_without_declared_output_fails_closed(tmp_path: Path) -> None:
    spec = load_spec(_write_spec(tmp_path, GATE_LOOP_SPEC))
    executor = _executor(spec, output_checker=lambda _uri: False)

    report = run_workflow_runtime(
        spec,
        run_id="rt-missing-output",
        executor=executor,
        options=executor.options,
        decision_reader=_decision_reader(["promote_checkpoint"]),
    )

    assert report.status == "failed"
    assert "completed without declared durable output" in report.error
    assert any(wave["status"] == "failed" for wave in report.waves)


def test_declared_output_checker_receives_uri_and_ledger_keeps_schema(
    tmp_path: Path,
) -> None:
    spec = load_spec(_write_spec(tmp_path, GATE_LOOP_SPEC))
    checked: list[str] = []

    def checker(uri: str) -> bool:
        assert isinstance(uri, str)
        checked.append(uri)
        return True

    executor = _executor(spec, output_checker=checker)
    report = run_workflow_runtime(
        spec,
        run_id="rt-output-declaration",
        executor=executor,
        options=executor.options,
        decision_reader=_decision_reader(["promote_checkpoint"]),
    )

    assert report.status == "succeeded"
    assert checked == [
        "s3://example-bucket/gate-loop/rt-output-declaration/gate/decision.json"
    ]
    gate = next(wave for wave in report.waves if wave["states"] == ["gate"])
    assert gate["outputs"] == [
        {
            "uri": checked[0],
            "schema": "npa.sim2real.threshold_decision.v1",
        }
    ]


# ------------------------------------------------------------------- early exit


def test_runtime_early_exits_when_gate_promotes_on_first_iteration(
    tmp_path: Path,
) -> None:
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


def test_runtime_persists_exact_submitted_workflow_yaml(tmp_path: Path) -> None:
    spec = load_spec(_write_spec(tmp_path, GATE_LOOP_SPEC))
    store = MemoryStore()
    executor = _executor(spec, store=store)
    workflow_yaml = GATE_LOOP_SPEC.encode("utf-8")

    report = run_workflow_runtime(
        spec,
        run_id="rt-workflow-artifact",
        executor=executor,
        state_store=store,
        options=executor.options,
        decision_reader=_decision_reader(["promote_checkpoint"]),
        workflow_yaml=workflow_yaml,
    )

    assert report.status == "succeeded"
    assert store.objects["unit-prefix/workflow.yaml"] == workflow_yaml


def test_runtime_uses_executor_ledger_store_for_exact_workflow_yaml(
    tmp_path: Path,
) -> None:
    spec = load_spec(_write_spec(tmp_path, GATE_LOOP_SPEC))
    store = MemoryStore()
    executor = _executor(spec, store=store)
    workflow_yaml = GATE_LOOP_SPEC.encode("utf-8")

    report = run_workflow_runtime(
        spec,
        run_id="rt-executor-workflow-artifact",
        executor=executor,
        options=executor.options,
        decision_reader=_decision_reader(["promote_checkpoint"]),
        workflow_yaml=workflow_yaml,
    )

    assert report.status == "succeeded"
    assert store.objects["unit-prefix/workflow.yaml"] == workflow_yaml


def test_runtime_rejects_workflow_yaml_without_any_durable_store(
    tmp_path: Path,
) -> None:
    spec = load_spec(_write_spec(tmp_path, GATE_LOOP_SPEC))
    executor = _executor(spec)

    with pytest.raises(NpaWorkflowError, match="workflow_yaml requires a durable"):
        run_workflow_runtime(
            spec,
            run_id="rt-no-workflow-store",
            executor=executor,
            options=executor.options,
            workflow_yaml=GATE_LOOP_SPEC.encode("utf-8"),
        )


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
    # This traverses the actual dynamic loop, rather than manually rendering two
    # states. Each repeated work wave gets the runtime's next durable ordinal.
    work_fences = []
    for call in submitter.calls:
        if call["tasks"] != ["work"]:
            continue
        task = [doc for doc in yaml.safe_load_all(call["yaml"]) if doc][1]
        work_fences.append(int(task["envs"]["NPA_WORKFLOW_FENCE_SEQUENCE"]))
        assert task["envs"]["NPA_WORKFLOW_FENCE_ATTEMPT"] == "1"
    assert work_fences == [1, 3, 5]


def test_runtime_refreshes_launch_dependencies_for_every_wave(tmp_path: Path) -> None:
    """A loop can outlive a short-lived registry credential minted at submit."""

    spec = load_spec(_write_spec(tmp_path, GATE_LOOP_SPEC))
    submitter = FakeSubmitter()
    refreshed: list[list[str]] = []

    def refresh(path: Path) -> None:
        docs = [doc for doc in yaml.safe_load_all(path.read_text()) if doc]
        refreshed.append([doc["name"] for doc in docs[1:]])

    options = RuntimeOptions(
        poll_seconds=0,
        max_wait_seconds=60,
        pre_submit_hook=refresh,
    )
    executor = _executor(spec, submitter=submitter, options=options)

    report = run_workflow_runtime(
        spec,
        run_id="rt-refresh-every-wave",
        executor=executor,
        options=options,
        decision_reader=_decision_reader(["loop_back", "promote_checkpoint"]),
    )

    assert report.status == "succeeded"
    assert refreshed == [["work"], ["gate"], ["work"], ["gate"], ["publish"]]
    assert refreshed == [call["tasks"] for call in submitter.calls]


def test_runtime_refreshes_launch_dependencies_again_before_retry(
    tmp_path: Path,
) -> None:
    spec = load_spec(_write_spec(tmp_path, FANOUT_SPEC))
    submitter = FakeSubmitter()
    refreshed: list[str] = []
    options = RuntimeOptions(
        poll_seconds=0,
        max_wait_seconds=60,
        retries=1,
        retry_backoff_seconds=0,
        pre_submit_hook=lambda path: refreshed.append(path.read_text()),
    )
    executor = _executor(
        spec,
        submitter=submitter,
        status_fn=FakeStatus(["FAILED", "SUCCEEDED"]),
        options=options,
    )

    report = run_workflow_runtime(
        spec, run_id="rt-refresh-retry", executor=executor, options=options
    )

    assert report.status == "succeeded"
    assert len(refreshed) == len(submitter.calls) == 4
    assert submitter.calls[0]["tasks"] == submitter.calls[1]["tasks"] == [
        "shard-a",
        "shard-b",
    ]


def test_runtime_launch_dependency_refresh_failure_prevents_submit(
    tmp_path: Path,
) -> None:
    spec = load_spec(_write_spec(tmp_path, FANOUT_SPEC))
    submitter = FakeSubmitter()

    def fail_refresh(_path: Path) -> None:
        raise RuntimeError("registry credential refresh failed")

    options = RuntimeOptions(
        poll_seconds=0,
        max_wait_seconds=60,
        pre_submit_hook=fail_refresh,
    )
    executor = _executor(spec, submitter=submitter, options=options)

    report = run_workflow_runtime(
        spec, run_id="rt-refresh-fail-closed", executor=executor, options=options
    )

    assert report.status == "failed"
    assert "registry credential refresh failed" in report.error
    assert submitter.calls == []


def test_runtime_reads_exact_iteration_scoped_decision_output(tmp_path: Path) -> None:
    spec = load_spec(_write_spec(tmp_path, SCOPED_GATE_LOOP_SPEC))
    submitter = FakeSubmitter()
    executor = _executor(spec, submitter=submitter)
    reads: list[tuple[str, str]] = []
    decisions = iter(("loop_back", "promote_checkpoint"))

    def reader(bucket: str, key: str) -> str:
        reads.append((bucket, key))
        return json.dumps({"decision": next(decisions)})

    report = run_workflow_runtime(
        spec,
        run_id="rt-scoped-decision",
        executor=executor,
        options=executor.options,
        decision_reader=reader,
        assume_decision="loop_back",
    )

    assert report.status == "succeeded"
    assert reads == [
        (
            "example-bucket",
            "scoped-gate/rt-scoped-decision/iteration-1/decision.json",
        ),
        (
            "example-bucket",
            "scoped-gate/rt-scoped-decision/iteration-2/decision.json",
        ),
    ]
    assert all(
        decision.get("source") != "assume_decision_fallback"
        for decision in report.decisions
    )
    assert [call["tasks"][0] for call in submitter.calls] == [
        "work",
        "gate",
        "work",
        "gate",
        "publish",
    ]


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
    plan_states = [
        step.state
        for step in build_plan(spec, run_id="rt-eq", assume_decision=assume).steps
    ]
    assert runtime_states == plan_states


# ---------------------------------------------------------------------- fan-out


def test_runtime_launches_parallel_group_as_job_group_with_barrier(
    tmp_path: Path,
) -> None:
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


def test_runtime_max_concurrency_option_is_a_cap_not_an_override(
    tmp_path: Path,
) -> None:
    """--max-concurrency can only lower a group's declared bound (cost control)."""

    spec = load_spec(_write_spec(tmp_path, FANOUT_SPEC))  # group declares 2

    tighter = FakeSubmitter()
    tight_options = RuntimeOptions(
        poll_seconds=0, max_wait_seconds=60, max_concurrency=1
    )
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
    wide_options = RuntimeOptions(
        poll_seconds=0, max_wait_seconds=60, max_concurrency=8
    )
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


def test_runtime_forwards_skypilot_config_path(tmp_path: Path) -> None:
    spec = load_spec(_write_spec(tmp_path, FANOUT_SPEC))
    submitter = FakeSubmitter()
    config_path = tmp_path / "sky-config.yaml"
    options = RuntimeOptions(
        poll_seconds=0, max_wait_seconds=60, config_path=config_path
    )
    executor = _executor(spec, submitter=submitter, options=options)

    report = run_workflow_runtime(
        spec, run_id="rt-config-path", executor=executor, options=options
    )

    assert report.status == "succeeded"
    assert submitter.calls
    assert all(call["kwargs"]["config_path"] == config_path for call in submitter.calls)


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
    attempts = [
        wave for wave in report.waves if wave["states"] == ["shard-a", "shard-b"]
    ]
    assert [wave["attempt"] for wave in attempts] == [1, 2]
    assert attempts[0]["status"] == "failed"
    assert attempts[1]["status"] == "succeeded"
    first_docs = [doc for doc in yaml.safe_load_all(submitter.calls[0]["yaml"]) if doc]
    second_docs = [doc for doc in yaml.safe_load_all(submitter.calls[1]["yaml"]) if doc]
    assert {doc["envs"]["NPA_WORKFLOW_FENCE_ATTEMPT"] for doc in first_docs[1:]} == {
        "1"
    }
    assert {doc["envs"]["NPA_WORKFLOW_FENCE_ATTEMPT"] for doc in second_docs[1:]} == {
        "2"
    }
    first_attempt_ids = {
        doc["envs"]["NPA_WORKFLOW_ATTEMPT_ID"] for doc in first_docs[1:]
    }
    second_attempt_ids = {
        doc["envs"]["NPA_WORKFLOW_ATTEMPT_ID"] for doc in second_docs[1:]
    }
    assert len(first_attempt_ids) == len(second_attempt_ids) == 1
    assert first_attempt_ids.isdisjoint(second_attempt_ids)
    assert {
        doc["envs"]["NPA_WORKFLOW_FENCE_SEQUENCE"] for doc in first_docs[1:]
    } == {doc["envs"]["NPA_WORKFLOW_FENCE_SEQUENCE"] for doc in second_docs[1:]}


def test_launch_transaction_cannot_overwrite_the_scheduler_publication_fence() -> None:
    attempt = WaveAttempt(
        key="wave",
        states=["augment"],
        kind="serial",
        scheduler_fence_sequence=7,
    )

    SkyPilotWaveExecutor._apply_launch_transaction(
        attempt,
        {
            "launch_sequence": 3,
            "logical_launch_id": "durable-launch",
            "recovery_decision": "interrupted_verified_absent",
        },
    )

    assert attempt.scheduler_fence_sequence == 7
    assert attempt.launch_sequence == 3
    assert attempt.to_dict()["scheduler_fence_sequence"] == 7


def test_wave_retry_exhausted_fails_the_run(tmp_path: Path) -> None:
    spec = load_spec(_write_spec(tmp_path, FANOUT_SPEC))
    options = RuntimeOptions(
        poll_seconds=0, max_wait_seconds=60, retries=1, retry_backoff_seconds=0
    )
    executor = _executor(
        spec, status_fn=FakeStatus(["FAILED", "FAILED"]), options=options
    )

    report = run_workflow_runtime(
        spec, run_id="rt-retry-fail", executor=executor, options=options
    )

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

    report = run_workflow_runtime(
        spec, run_id="rt-timeout", executor=executor, options=options
    )

    assert report.status == "failed"
    assert "did not reach a terminal status" in report.error
    assert cancels and cancels[0]["job_id"]


def test_timeout_without_cancel_preserves_the_in_flight_job(tmp_path: Path) -> None:
    spec = load_spec(_write_spec(tmp_path, FANOUT_SPEC))
    cancels: list[dict[str, Any]] = []
    options = RuntimeOptions(
        poll_seconds=0, max_wait_seconds=2, cancel_on_timeout=False
    )
    executor = _executor(
        spec,
        status_fn=FakeStatus(["RUNNING"] * 20),
        options=options,
        cancels=cancels,
    )

    report = run_workflow_runtime(
        spec, run_id="rt-timeout-preserve", executor=executor, options=options
    )

    assert report.status == "failed"
    assert "did not reach a terminal status" in report.error
    assert not cancels
    assert report.waves[0]["status"] == "running"
    assert report.waves[0]["sky_status"] == "SUBMITTED"


def test_timeout_without_cancel_never_retries_a_preserved_job(tmp_path: Path) -> None:
    """Scratch regression: retries must not turn one preserved job into three."""

    spec = load_spec(_write_spec(tmp_path, FANOUT_SPEC))
    submitter = FakeSubmitter()
    cancels: list[dict[str, Any]] = []
    options = RuntimeOptions(
        poll_seconds=0,
        max_wait_seconds=2,
        retries=2,
        retry_backoff_seconds=0,
        cancel_on_timeout=False,
    )
    executor = _executor(
        spec,
        submitter=submitter,
        status_fn=FakeStatus(["RUNNING"] * 20),
        options=options,
        cancels=cancels,
    )

    report = run_workflow_runtime(
        spec, run_id="rt-timeout-preserve-retries", executor=executor, options=options
    )

    assert report.status == "failed"
    assert len(submitter.calls) == 1
    assert cancels == []
    assert len(report.waves) == 1
    assert report.waves[0]["status"] == "running"
    assert executor.ledger.state.in_flight_wave(report.waves[0]["key"]) is not None


def test_zero_max_wait_is_unbounded(tmp_path: Path) -> None:
    spec = load_spec(_write_spec(tmp_path, FANOUT_SPEC))
    options = RuntimeOptions(poll_seconds=0, max_wait_seconds=0)
    executor = _executor(
        spec,
        status_fn=FakeStatus(["PENDING", "RUNNING", "SUCCEEDED"] * 3),
        options=options,
    )

    report = run_workflow_runtime(
        spec, run_id="rt-unbounded-wait", executor=executor, options=options
    )

    assert report.status == "succeeded"


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
    assert [wave["status"] for wave in persisted.waves] == [
        "succeeded",
        "succeeded",
        "failed",
    ]

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

    assert second_report.status == "failed"
    # Successful waves replay, while a terminal workload failure remains terminal.
    assert second_submitter.calls == []
    assert [wave["replayed"] for wave in second_report.waves] == [True, True, True]


def test_resume_adopts_terminal_success_after_output_check_driver_failure(
    tmp_path: Path,
) -> None:
    spec = load_spec(_write_spec(tmp_path, GATE_LOOP_SPEC))
    store = MemoryStore()
    first_submitter = FakeSubmitter()

    def interrupted_checker(uri: str) -> bool:
        assert uri.endswith("/gate/decision.json")
        raise RuntimeError("driver interrupted after scheduler success")

    first = _executor(
        spec,
        run_id="rt-post-success-adopt",
        submitter=first_submitter,
        store=store,
        output_checker=interrupted_checker,
    )
    first_report = run_workflow_runtime(
        spec,
        run_id="rt-post-success-adopt",
        executor=first,
        options=first.options,
        decision_reader=_decision_reader(["promote_checkpoint"]),
    )

    assert first_report.status == "failed"
    assert [call["tasks"] for call in first_submitter.calls] == [["work"], ["gate"]]
    failed_gate = first_report.waves[-1]
    assert failed_gate["status"] == "failed"
    assert failed_gate["sky_status"] == "SUCCEEDED"

    resumed_submitter = FakeSubmitter()
    options = RuntimeOptions(poll_seconds=0, max_wait_seconds=60, resume=True)
    resumed = _executor(
        spec,
        run_id="rt-post-success-adopt",
        submitter=resumed_submitter,
        options=options,
        store=store,
        output_checker=lambda uri: uri.endswith("/gate/decision.json"),
    )
    report = run_workflow_runtime(
        spec,
        run_id="rt-post-success-adopt",
        executor=resumed,
        options=options,
        decision_reader=_decision_reader(["promote_checkpoint"]),
    )

    assert report.status == "succeeded"
    assert [call["tasks"] for call in resumed_submitter.calls] == [["publish"]]
    adopted_gate = next(wave for wave in report.waves if wave["states"] == ["gate"])
    assert adopted_gate["job_id"] == failed_gate["job_id"]
    assert adopted_gate["status"] == "succeeded"
    assert adopted_gate["adopted"] is True
    assert adopted_gate["replayed"] is True
    assert (
        adopted_gate["recovery_decision"]
        == "adopted_terminal_success_after_driver_failure"
    )


def test_resume_with_explicit_retry_replays_success_and_retries_terminal_wave(
    tmp_path: Path,
) -> None:
    spec = load_spec(_write_spec(tmp_path, FANOUT_SPEC))
    store = MemoryStore()
    first = _executor(
        spec,
        run_id="rt-explicit-retry",
        submitter=FakeSubmitter(),
        status_fn=FakeStatus(["SUCCEEDED", "SUCCEEDED", "FAILED"]),
        store=store,
    )
    first_report = run_workflow_runtime(
        spec, run_id="rt-explicit-retry", executor=first, options=first.options
    )
    assert first_report.status == "failed"
    persisted = store.read_runtime_state()
    assert persisted is not None
    failed_join = next(wave for wave in persisted.waves if wave["states"] == ["join"])
    assert failed_join["scheduler_fence_sequence"] == 3
    # Simulate a launch transaction phase that differs from the scheduler wave
    # ordinal. Resume must never render this transaction field as publication
    # authority.
    failed_join["launch_sequence"] = 99
    store.write_runtime_state(persisted)

    submitter = FakeSubmitter()
    options = RuntimeOptions(
        poll_seconds=0,
        max_wait_seconds=60,
        resume=True,
        retries=1,
        retry_backoff_seconds=0,
    )
    resumed = _executor(
        spec,
        run_id="rt-explicit-retry",
        submitter=submitter,
        options=options,
        store=store,
    )
    report = run_workflow_runtime(
        spec, run_id="rt-explicit-retry", executor=resumed, options=options
    )

    assert report.status == "succeeded"
    assert [call["tasks"] for call in submitter.calls] == [["join"]]
    assert submitter.calls[0]["job_name"].endswith("-a2")
    assert [wave["replayed"] for wave in report.waves[:2]] == [True, True]
    retry_doc = [doc for doc in yaml.safe_load_all(submitter.calls[0]["yaml"]) if doc][1]
    assert retry_doc["envs"]["NPA_WORKFLOW_FENCE_SEQUENCE"] == "3"
    assert retry_doc["envs"]["NPA_WORKFLOW_FENCE_ATTEMPT"] == "2"
    retried = next(wave for wave in report.waves if wave["states"] == ["join"])
    assert retried["scheduler_fence_sequence"] == 3
    assert retried["launch_sequence"] == 0


def test_long_run_id_preserves_retry_attempt_suffix(tmp_path: Path) -> None:
    spec = load_spec(_write_spec(tmp_path, FANOUT_SPEC))
    run_id = "paidf-" + "x" * 80
    executor = _executor(spec, run_id=run_id)
    executor._sequence = 2
    step = next(iter(build_plan(spec, run_id=run_id).steps))

    name = executor._job_name(
        [step],
        group="",
        attempt=WaveAttempt(key="wave", states=[step.state], kind="serial", attempt=3),
    )

    assert len(name) <= 60
    assert name.endswith("-a3")


def _canonical_sim2real_1x1():
    root = Path(__file__).resolve().parents[4]
    spec = load_spec(
        root / "npa" / "workflows" / "workbench" / "npa-workflows" / "sim2real.yaml"
    )
    image = "cr.example/npa/runtime@sha256:" + "b" * 64
    spec.config.update(
        {
            "source_sha": "a" * 40,
            "outer_iterations": "1",
            "inner_iterations": "1",
            "env_count": "12",
            "rollout_count": "1",
            "ppo_iterations": "2",
            "validation_count": "3",
            "gold_count": "3",
            "controller_image": image,
            "transfer_image": image,
            "envgen_image": image,
            "isaac_image": image,
            "viewer_image": image,
            "isaac_cache_pvc": "isaac-cache",
        }
    )
    return spec


def _trigger_ready(_state, _run_id, _context):
    return {"uri": "s3://unit/trigger/", "objects": 1}


def test_canonical_three_iteration_budget_promotes_and_finalizes_outer_one() -> None:
    spec = _canonical_sim2real_1x1()
    spec.config.update({"outer_iterations": "3", "allow_early_exit": "1"})
    submitter = FakeSubmitter()
    executor = _executor(spec, run_id="canonical-early-one", submitter=submitter)

    report = run_workflow_runtime(
        spec,
        run_id="canonical-early-one",
        executor=executor,
        options=executor.options,
        decision_reader=_decision_reader(["promote_checkpoint"]),
        trigger_waiter=_trigger_ready,
    )

    assert report.status == "succeeded"
    assert sum("stage-07-rollouts" in call["tasks"] for call in submitter.calls) == 1
    decision_yaml = next(
        call["yaml"]
        for call in submitter.calls
        if call["tasks"] == ["stage-11-decision"]
    )
    assert "--allow-early-exit" in decision_yaml
    assert "--allow-early-exit 1" in decision_yaml
    for state in ("stage-13-retrigger", "stage-14-visualize"):
        rendered = next(
            call["yaml"] for call in submitter.calls if call["tasks"] == [state]
        )
        assert "--outer-iteration 1" in rendered
        assert "--outer-iteration 3" not in rendered


def test_canonical_resume_replays_single_stage8_then_restarts_at_stage9() -> None:
    spec = _canonical_sim2real_1x1()
    store = MemoryStore()
    first_submitter = FakeSubmitter()
    first = _executor(
        spec,
        run_id="canonical-stage8-resume",
        submitter=first_submitter,
        status_fn=FakeStatus(["SUCCEEDED"] * 8 + ["FAILED"]),
        store=store,
    )
    first_report = run_workflow_runtime(
        spec,
        run_id="canonical-stage8-resume",
        executor=first,
        options=first.options,
        decision_reader=_decision_reader(["loop_back"]),
        trigger_waiter=_trigger_ready,
    )
    assert first_report.status == "failed"
    assert first_submitter.calls[-2]["tasks"] == ["stage-08-cosmos3"]
    assert first_submitter.calls[-1]["tasks"] == ["stage-09-ppo"]

    options = RuntimeOptions(
        poll_seconds=0,
        max_wait_seconds=60,
        resume=True,
        retries=1,
        retry_backoff_seconds=0,
    )
    resumed_submitter = FakeSubmitter()
    resumed = _executor(
        spec,
        run_id="canonical-stage8-resume",
        submitter=resumed_submitter,
        options=options,
        store=store,
    )
    report = run_workflow_runtime(
        spec,
        run_id="canonical-stage8-resume",
        executor=resumed,
        options=options,
        decision_reader=_decision_reader(["loop_back"]),
        trigger_waiter=_trigger_ready,
    )
    assert report.status == "succeeded"
    assert [call["tasks"] for call in resumed_submitter.calls] == [
        ["stage-09-ppo"],
        ["stage-10-gold"],
        ["stage-11-decision"],
        ["stage-12-external-seam"],
        ["stage-13-retrigger"],
        ["stage-14-visualize"],
    ]
    stage8 = next(wave for wave in report.waves if "stage-08-cosmos3" in wave["states"])
    assert stage8["replayed"] is True


def test_canonical_finalization_resume_submits_only_stage14() -> None:
    spec = _canonical_sim2real_1x1()
    store = MemoryStore()
    first = _executor(
        spec,
        run_id="canonical-finalize-resume",
        status_fn=FakeStatus(["SUCCEEDED"] * 13 + ["FAILED"]),
        store=store,
    )
    first_report = run_workflow_runtime(
        spec,
        run_id="canonical-finalize-resume",
        executor=first,
        options=first.options,
        decision_reader=_decision_reader(["loop_back"]),
        trigger_waiter=_trigger_ready,
    )
    assert first_report.status == "failed"
    assert first_report.waves[-1]["states"] == ["stage-14-visualize"]

    options = RuntimeOptions(
        poll_seconds=0,
        max_wait_seconds=60,
        resume=True,
        retries=1,
        retry_backoff_seconds=0,
    )
    submitter = FakeSubmitter()
    resumed = _executor(
        spec,
        run_id="canonical-finalize-resume",
        submitter=submitter,
        options=options,
        store=store,
    )
    report = run_workflow_runtime(
        spec,
        run_id="canonical-finalize-resume",
        executor=resumed,
        options=options,
        decision_reader=_decision_reader(["loop_back"]),
        trigger_waiter=_trigger_ready,
    )
    assert report.status == "succeeded"
    assert [call["tasks"] for call in submitter.calls] == [["stage-14-visualize"]]


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
    assert [wave["states"] for wave in persisted.waves] == [
        ["work"],
        ["gate"],
        ["publish"],
    ]
    assert persisted.decisions[-1]["decision"] == "promote_checkpoint"
    assert persisted.decisions[-1]["uri"].endswith("/gate/decision.json")


# ------------------------------------------------- cost safety / leak protection
#
# These pin the review findings: a wave must never end with a managed job still
# running, and --resume must never submit a second copy of work already in flight.


class BoomStatus:
    """Status function that raises for the first ``failures`` calls."""

    def __init__(self, failures: int, exc: Exception | None = None) -> None:
        self.failures = failures
        self.calls = 0
        self.exc = exc or TimeoutError("sky jobs queue timed out")

    def __call__(self, job_id: str, **_: Any) -> FakeResult:
        self.calls += 1
        if self.calls <= self.failures:
            raise self.exc
        return FakeResult(status="SUCCEEDED", job_id=job_id)


def test_transient_status_errors_do_not_orphan_the_job(tmp_path: Path) -> None:
    spec = load_spec(_write_spec(tmp_path, FANOUT_SPEC))
    cancels: list[dict[str, Any]] = []
    executor = _executor(spec, status_fn=BoomStatus(failures=3), cancels=cancels)

    report = run_workflow_runtime(
        spec, run_id="rt-flaky-status", executor=executor, options=executor.options
    )

    assert report.status == "succeeded"
    assert not cancels, "a recoverable status hiccup must not cancel a healthy job"
    first = report.waves[0]
    assert len(first["status_errors"]) == 3
    assert "TimeoutError" in first["status_errors"][0]


def test_persistent_status_errors_cancel_the_job_and_fail(tmp_path: Path) -> None:
    spec = load_spec(_write_spec(tmp_path, FANOUT_SPEC))
    cancels: list[dict[str, Any]] = []
    executor = _executor(spec, status_fn=BoomStatus(failures=99), cancels=cancels)

    report = run_workflow_runtime(
        spec, run_id="rt-dead-status", executor=executor, options=executor.options
    )

    assert report.status == "failed"
    assert "consecutive" in report.error
    # The whole point: the job we launched is not left running.
    assert cancels and cancels[0]["job_id"] == "1"
    assert report.waves[0]["sky_status"] == "SUBMITTED"
    assert report.waves[0]["cancellation"]["state"] == "requested"


def test_exact_cancellation_is_verified_without_masking_primary_error(
    tmp_path: Path,
) -> None:
    spec = load_spec(_write_spec(tmp_path, FANOUT_SPEC))
    calls = {"count": 0}

    def status(job_id: str, **_: Any) -> FakeResult:
        calls["count"] += 1
        if calls["count"] <= 6:
            raise TimeoutError("queue transport failed")
        return FakeResult(status="CANCELLED", job_id=job_id)

    cancels: list[dict[str, Any]] = []
    executor = _executor(spec, status_fn=status, cancels=cancels)
    report = run_workflow_runtime(
        spec, run_id="rt-cancel-verified", executor=executor, options=executor.options
    )
    wave = report.waves[0]
    assert report.status == "failed"
    assert "consecutive" in wave["primary_error"]
    assert wave["cancellation"]["state"] == "verified"
    assert wave["sky_status"] == "CANCELLED"
    assert cancels[0]["job_id"] == "1"


def test_exact_cancellation_failure_is_truthful_and_primary_error_survives(
    tmp_path: Path,
) -> None:
    spec = load_spec(_write_spec(tmp_path, FANOUT_SPEC))

    def cancel_failure(**_kwargs: Any) -> None:
        raise PermissionError("cancel forbidden")

    executor = _executor(spec, status_fn=BoomStatus(failures=99))
    executor._canceller = cancel_failure
    report = run_workflow_runtime(
        spec, run_id="rt-cancel-failed", executor=executor, options=executor.options
    )
    wave = report.waves[0]
    assert "consecutive" in wave["primary_error"]
    assert wave["cancellation"]["state"] == "failed"
    assert "forbidden" in wave["cancellation"]["error"]
    assert wave["sky_status"] != "CANCELLED"


def test_unexpected_submit_error_tears_down_defensively_and_fails_fast(
    tmp_path: Path,
) -> None:
    """A failed submit may still have provisioned a cluster, so tear it down.

    ``sky jobs launch`` can raise *after* provisioning starts (e.g. a submit
    timeout), so an abort during submission attempts a teardown by cluster name
    rather than assuming nothing was created. The run then fails immediately
    instead of retrying into more spend.
    """

    spec = load_spec(_write_spec(tmp_path, FANOUT_SPEC))
    cancels: list[dict[str, Any]] = []

    def boom_submitter(path: Path, job_name: str, **kwargs: Any):
        raise RuntimeError("submit timed out")

    executor = _executor(spec, submitter=boom_submitter, cancels=cancels)
    report = run_workflow_runtime(
        spec, run_id="rt-submit-boom", executor=executor, options=executor.options
    )

    assert report.status == "failed"
    assert "RuntimeError: submit timed out" in report.waves[0]["error"]
    assert not cancels, "an ID-less uncertain launch must never cancel by name"
    assert report.waves[0]["cancellation"]["state"] == "not_applicable"
    # One attempt only: an unexpected tooling failure is not retried.
    assert len(report.waves) == 1


def test_unidentifiable_job_is_rejected_instead_of_polling_unknown(
    tmp_path: Path,
) -> None:
    """An unidentifiable job used to burn max_wait_seconds and then leak.

    With no id from the launch output AND no match by name there is nothing safe to
    poll (`sky jobs queue` matches numeric ids), so the wave fails immediately after a
    defensive teardown instead of sitting on UNKNOWN for the whole deadline.
    """

    spec = load_spec(_write_spec(tmp_path, FANOUT_SPEC))
    cancels: list[dict[str, Any]] = []
    status_fn = FakeStatus()

    class NoIdSubmitter(FakeSubmitter):
        def __call__(self, path: Path, job_name: str, **kwargs: Any) -> FakeResult:
            super().__call__(path, job_name, **kwargs)
            return FakeResult(job_id="")

    executor = _executor(
        spec, submitter=NoIdSubmitter(), status_fn=status_fn, cancels=cancels
    )
    report = run_workflow_runtime(
        spec, run_id="rt-no-jobid", executor=executor, options=executor.options
    )

    assert report.status == "failed"
    assert "could not be found by name" in report.error
    assert status_fn.calls == [], "must not poll a job it cannot identify"
    assert not cancels, "no fuzzy/name-only cancellation is permitted"


def test_resume_attaches_to_an_in_flight_job_instead_of_resubmitting(
    tmp_path: Path,
) -> None:
    spec = load_spec(_write_spec(tmp_path, FANOUT_SPEC))
    store = MemoryStore()

    # First driver: submits the group, then "dies" while polling.
    first_submitter = FakeSubmitter()
    first = _executor(
        spec,
        run_id="rt-adopt",
        submitter=first_submitter,
        status_fn=BoomStatus(failures=99),
        store=store,
    )
    first_report = run_workflow_runtime(
        spec, run_id="rt-adopt", executor=first, options=first.options
    )
    assert first_report.status == "failed"
    assert len(first_submitter.calls) == 1

    # Simulate "the job actually kept running": rewrite the ledger record to running.
    persisted = store.read_runtime_state()
    assert persisted is not None
    key = persisted.waves[0]["key"]
    persisted.record_wave(
        {**persisted.waves[0], "status": "running", "sky_status": "RUNNING"}
    )
    store.write_runtime_state(persisted)

    # Second driver resumes: it must poll job 1, not submit a second copy.
    second_submitter = FakeSubmitter()
    resume_options = RuntimeOptions(poll_seconds=0, max_wait_seconds=60, resume=True)
    second = _executor(
        spec,
        run_id="rt-adopt",
        submitter=second_submitter,
        options=resume_options,
        store=store,
    )
    second_report = run_workflow_runtime(
        spec, run_id="rt-adopt", executor=second, options=resume_options
    )

    assert second_report.status == "succeeded"
    adopted = [wave for wave in second_report.waves if wave.get("adopted")]
    assert adopted and adopted[0]["job_id"] == "1"
    assert adopted[0]["key"] == key
    # Only the *remaining* waves were submitted; the in-flight one was adopted.
    assert [call["tasks"] for call in second_submitter.calls] == [["shard-c"], ["join"]]


def test_resume_preserves_an_in_flight_job_that_actually_failed(tmp_path: Path) -> None:
    spec = load_spec(_write_spec(tmp_path, FANOUT_SPEC))
    store = MemoryStore()
    state = RuntimeRunState(workflow=spec.name, run_id="rt-adopt-failed")
    state.record_wave(
        {
            "key": "001|shards|shards:shard-a:-,shards:shard-b:-",
            "status": "running",
            "job_id": "77",
            "job_name": "rt-adopt-failed-01-shards",
            "attempt": 1,
        }
    )
    store.write_runtime_state(state)

    resume_options = RuntimeOptions(poll_seconds=0, max_wait_seconds=60, resume=True)
    submitter = FakeSubmitter()
    executor = _executor(
        spec,
        run_id="rt-adopt-failed",
        submitter=submitter,
        status_fn=FakeStatus(["FAILED"]),  # the adopted job had died
        options=resume_options,
        store=store,
    )
    report = run_workflow_runtime(
        spec, run_id="rt-adopt-failed", executor=executor, options=resume_options
    )

    assert report.status == "failed"
    assert submitter.calls == []


def test_resume_relaunches_only_authoritatively_absent_transient_wave(
    tmp_path: Path,
) -> None:
    from npa.orchestration.skypilot.workflow import ManagedJobEvidence

    spec = load_spec(_write_spec(tmp_path, FANOUT_SPEC))
    store = MemoryStore()
    state = RuntimeRunState(workflow=spec.name, run_id="rt-absent-retry")
    state.record_wave(
        {
            "key": "001|shards|shards:shard-a:-,shards:shard-b:-",
            "status": "failed",
            "job_name": "rt-absent-retry-01-shards",
            "attempt": 1,
            "error_category": "kubernetes_transport",
            "recovery_decision": "recovery_deadline_exhausted_verified_absent",
            "cancellation": {"state": "not_applicable", "error": ""},
        }
    )
    store.write_runtime_state(state)
    options = RuntimeOptions(poll_seconds=0, max_wait_seconds=60, resume=True)
    submitter = FakeSubmitter()
    executor = _executor(
        spec,
        run_id="rt-absent-retry",
        submitter=submitter,
        options=options,
        store=store,
        reconcile_fn=lambda *_args, **_kwargs: ManagedJobEvidence("absent"),
    )
    report = run_workflow_runtime(
        spec, run_id="rt-absent-retry", executor=executor, options=options
    )
    assert report.status == "succeeded"
    assert [call["tasks"] for call in submitter.calls][0] == ["shard-a", "shard-b"]
    attempts = [
        item
        for item in store.read_runtime_state().waves
        if item["key"] == "001|shards|shards:shard-a:-,shards:shard-b:-"
    ]
    assert [item["attempt"] for item in attempts] == [1, 2]


def test_resume_absent_submitted_wave_stays_blocked_across_repeated_resume(
    tmp_path: Path,
) -> None:
    from npa.orchestration.skypilot.workflow import ManagedJobEvidence

    spec = load_spec(_write_spec(tmp_path, FANOUT_SPEC))
    store = MemoryStore()
    state = RuntimeRunState(workflow=spec.name, run_id="rt-sticky-absence")
    state.record_wave(
        {
            "key": "001|shards|shards:shard-a:-,shards:shard-b:-",
            "status": "running",
            "job_id": "77",
            "job_name": "rt-sticky-absence-01-shards",
            "attempt": 1,
            "sky_status": "SUBMITTED",
            "logical_launch_id": "logical-sticky-absence",
            "launch_sequence": 1,
            "recovery_decision": "submitted_and_reconciled",
        }
    )
    store.write_runtime_state(state)
    options = RuntimeOptions(poll_seconds=0, max_wait_seconds=60, resume=True)

    for _ in range(2):
        submitter = FakeSubmitter()
        executor = _executor(
            spec,
            run_id="rt-sticky-absence",
            submitter=submitter,
            options=options,
            store=store,
            reconcile_fn=lambda *_args, **_kwargs: ManagedJobEvidence("absent"),
        )
        report = run_workflow_runtime(
            spec, run_id="rt-sticky-absence", executor=executor, options=options
        )
        assert report.status == "failed"
        assert submitter.calls == []
        assert (
            store.read_runtime_state().waves[0]["recovery_decision"]
            == "resume_block_terminal_or_legacy_absence"
        )


def test_explicit_resume_relaunches_absent_submitted_wave_as_new_attempt(
    tmp_path: Path,
) -> None:
    from npa.orchestration.skypilot.workflow import ManagedJobEvidence

    output_spec = FANOUT_SPEC
    for shard, next_state in (
        ("a", "shard-b"),
        ("b", "shard-c"),
        ("c", "join"),
    ):
        output_spec = output_spec.replace(
            f"    resources: cpu\n\n  {next_state}:",
            "    resources: cpu\n"
            "    outputs:\n"
            f'      - uri: "s3://{{{{config.bucket}}}}/'
            f'{{{{config.prefix}}}}/shard-{shard}.json"\n\n'
            f"  {next_state}:",
            1,
        )
    spec = load_spec(_write_spec(tmp_path, output_spec))
    store = MemoryStore()
    state = RuntimeRunState(workflow=spec.name, run_id="rt-explicit-absence")
    state.record_wave(
        {
            "key": "001|shards|shards:shard-a:-,shards:shard-b:-",
            "status": "running",
            "job_id": "77",
            "job_name": "rt-explicit-absence-01-shards",
            "attempt": 1,
            "sky_status": "SUBMITTED",
            "logical_launch_id": "logical-explicit-absence",
            "launch_sequence": 1,
            "recovery_decision": "submitted_and_reconciled",
        }
    )
    store.write_runtime_state(state)
    options = RuntimeOptions(
        poll_seconds=0,
        max_wait_seconds=60,
        resume=True,
        retry_absent_in_flight=True,
    )
    submitter = FakeSubmitter()
    executor = _executor(
        spec,
        run_id="rt-explicit-absence",
        submitter=submitter,
        options=options,
        store=store,
        output_checker=lambda _uri: bool(submitter.calls),
        reconcile_fn=lambda *_args, **_kwargs: ManagedJobEvidence("absent"),
    )

    report = run_workflow_runtime(
        spec, run_id="rt-explicit-absence", executor=executor, options=options
    )

    assert report.status == "succeeded"
    assert submitter.calls[0]["job_name"].endswith("-a2")
    attempts = [
        item
        for item in store.read_runtime_state().waves
        if item["key"] == "001|shards|shards:shard-a:-,shards:shard-b:-"
    ]
    assert [item["attempt"] for item in attempts] == [1, 2]
    assert (
        attempts[0]["recovery_decision"]
        == "operator_authorized_verified_absent_relaunch"
    )
    assert attempts[1]["status"] == "succeeded"


def test_explicit_absent_resume_refuses_existing_declared_output(
    tmp_path: Path,
) -> None:
    from npa.orchestration.skypilot.workflow import ManagedJobEvidence

    output_spec = FANOUT_SPEC.replace(
        "    resources: cpu\n\n  shard-b:",
        "    resources: cpu\n"
        "    outputs:\n"
        '      - uri: "s3://{{config.bucket}}/{{config.prefix}}/shard-a.json"\n\n'
        "  shard-b:",
        1,
    )
    spec = load_spec(_write_spec(tmp_path, output_spec))
    store = MemoryStore()
    state = RuntimeRunState(workflow=spec.name, run_id="rt-output-present")
    state.record_wave(
        {
            "key": "001|shards|shards:shard-a:-,shards:shard-b:-",
            "status": "running",
            "job_id": "77",
            "job_name": "rt-output-present-01-shards",
            "attempt": 1,
            "sky_status": "RUNNING",
            "logical_launch_id": "logical-output-present",
            "launch_sequence": 1,
            "recovery_decision": "submitted_and_reconciled",
        }
    )
    store.write_runtime_state(state)
    options = RuntimeOptions(
        poll_seconds=0,
        max_wait_seconds=60,
        resume=True,
        retry_absent_in_flight=True,
    )
    submitter = FakeSubmitter()
    executor = _executor(
        spec,
        run_id="rt-output-present",
        submitter=submitter,
        options=options,
        store=store,
        output_checker=lambda _uri: True,
        reconcile_fn=lambda *_args, **_kwargs: ManagedJobEvidence("absent"),
    )

    report = run_workflow_runtime(
        spec, run_id="rt-output-present", executor=executor, options=options
    )

    assert report.status == "failed"
    assert submitter.calls == []
    assert (
        store.read_runtime_state().waves[0]["recovery_decision"]
        == "resume_block_output_present"
    )


def test_resume_blocks_indeterminate_incomplete_wave_without_submit(
    tmp_path: Path,
) -> None:
    from npa.orchestration.skypilot.workflow import ManagedJobEvidence

    spec = load_spec(_write_spec(tmp_path, FANOUT_SPEC))
    store = MemoryStore()
    state = RuntimeRunState(workflow=spec.name, run_id="rt-indeterminate")
    state.record_wave(
        {
            "key": "001|shards|shards:shard-a:-,shards:shard-b:-",
            "status": "running",
            "job_name": "rt-indeterminate-01-shards",
            "attempt": 1,
            "recovery_decision": "block_indeterminate",
        }
    )
    store.write_runtime_state(state)
    options = RuntimeOptions(poll_seconds=0, max_wait_seconds=60, resume=True)
    submitter = FakeSubmitter()
    executor = _executor(
        spec,
        run_id="rt-indeterminate",
        submitter=submitter,
        options=options,
        store=store,
        reconcile_fn=lambda *_args, **_kwargs: ManagedJobEvidence(
            "unavailable", error="controller queue refused"
        ),
    )
    report = run_workflow_runtime(
        spec, run_id="rt-indeterminate", executor=executor, options=options
    )
    assert report.status == "failed"
    assert submitter.calls == []
    assert report.waves[0]["recovery_decision"] == "block_indeterminate"
    assert "refusing a duplicate" in report.error


def test_resume_without_a_ledger_bucket_fails_fast(tmp_path: Path) -> None:
    """Silently resubmitting everything is worse than refusing to resume."""

    text = FANOUT_SPEC.replace("  bucket: example-bucket\n", "")
    text = text.replace(
        '  prefix: "fanout/{{run.id}}"', '  prefix: "fanout/{{run.id}}"'
    )
    spec = load_spec(_write_spec(tmp_path, text))
    with pytest.raises(NpaWorkflowError, match="config.bucket is not set"):
        run_workflow_runtime(
            spec,
            run_id="rt-no-bucket",
            options=RuntimeOptions(poll_seconds=0, max_wait_seconds=10, resume=True),
        )


# ------------------------------------------------------ decision-contract edges


def test_missing_decision_artifact_falls_back_to_the_assumption(tmp_path: Path) -> None:
    spec = load_spec(_write_spec(tmp_path, GATE_LOOP_SPEC))
    submitter = FakeSubmitter()
    executor = _executor(spec, submitter=submitter)

    def missing_reader(bucket: str, key: str) -> str:
        raise FileNotFoundError(f"s3://{bucket}/{key}")

    report = run_workflow_runtime(
        spec,
        run_id="rt-missing-decision",
        executor=executor,
        options=executor.options,
        decision_reader=missing_reader,
        assume_decision="promote_checkpoint",
    )

    assert report.status == "succeeded"
    assert report.decisions[-1]["source"] == "assume_decision_fallback"
    assert report.decisions[-1]["decision"] == "promote_checkpoint"
    assert [call["tasks"][0] for call in submitter.calls] == [
        "work",
        "gate",
        "publish",
    ]


def test_missing_decision_without_an_assumption_fails(tmp_path: Path) -> None:
    spec = load_spec(_write_spec(tmp_path, GATE_LOOP_SPEC))
    executor = _executor(spec)

    def missing_reader(bucket: str, key: str) -> str:
        raise FileNotFoundError(f"s3://{bucket}/{key}")

    report = run_workflow_runtime(
        spec,
        run_id="rt-missing-strict",
        executor=executor,
        options=executor.options,
        decision_reader=missing_reader,
    )
    assert report.status == "failed"


def test_corrupt_decision_artifact_fails_the_run(tmp_path: Path) -> None:
    """A malformed gate artifact must stop the run, not silently loop."""

    spec = load_spec(_write_spec(tmp_path, GATE_LOOP_SPEC))
    executor = _executor(spec)

    report = run_workflow_runtime(
        spec,
        run_id="rt-corrupt-decision",
        executor=executor,
        options=executor.options,
        decision_reader=lambda bucket, key: "{not json",
        assume_decision="promote_checkpoint",
    )

    assert report.status == "failed"
    decisions = [d for d in report.decisions if d.get("decode_error")]
    assert decisions, "the unreadable payload should be recorded in the ledger"


def test_paidf_cosmos3_runtime_rejection_visualizes_and_skips_downstream() -> None:
    spec_path = (
        Path(__file__).resolve().parents[4]
        / "npa"
        / "workflows"
        / "workbench"
        / "npa-workflows"
        / "paidf-cosmos3.yaml"
    )
    spec = load_spec(spec_path)
    submitter = FakeSubmitter()
    executor = _executor(spec, run_id="paidf-reject", submitter=submitter)

    report = run_workflow_runtime(
        spec,
        run_id="paidf-reject",
        executor=executor,
        options=executor.options,
        decision_reader=_decision_reader(["loop_back"]),
        assume_decision="promote_checkpoint",
    )

    task_names = [name for call in submitter.calls for name in call["tasks"]]
    assert report.status == "succeeded"
    assert any("visualize-quality-evidence" in name for name in task_names)
    assert any("reject-quality" in name for name in task_names)
    for forbidden in ("annotate-augmented", "cosmos-curate", "curate", "finalize"):
        assert not any(forbidden in name for name in task_names)


# -------------------------------------------------------------- trigger bounds


def test_trigger_is_bounded_by_the_run_deadline(tmp_path: Path) -> None:
    """maxPolls: 0 must not mean 'wait forever'."""

    text = TRIGGER_SPEC.replace("      maxPolls: 5\n", "")
    spec = load_spec(_write_spec(tmp_path, text))
    assert spec.states["ingest"].trigger is not None
    assert spec.states["ingest"].trigger.max_polls == 0

    executor = _executor(spec, run_id="rt-trigger-deadline")
    waiter = s3_trigger_waiter(
        ledger=executor.ledger,
        lister=lambda *_: [],
        sleeper=lambda _s: None,
        max_wait_seconds=3,
        clock=_fake_clock(),
    )
    report = run_workflow_runtime(
        spec,
        run_id="rt-trigger-deadline",
        executor=executor,
        options=executor.options,
        trigger_waiter=waiter,
    )

    assert report.status == "failed"
    assert "after waiting 3s" in report.error


# --------------------------------------------------- job-id trust (live bug fix)


def test_stale_job_id_from_launch_output_is_corrected_by_name(tmp_path: Path) -> None:
    """Live bug: a stale parsed id made the driver abandon a running 4-GPU job.

    The launch output reported a cancelled job's id, so the driver polled that id,
    saw CANCELLED, declared the wave failed and walked away while the real job kept
    running. The job NAME is authoritative.
    """

    spec = load_spec(_write_spec(tmp_path, FANOUT_SPEC))
    submitter = FakeSubmitter()

    class StaleIdSubmitter(FakeSubmitter):
        def __call__(self, path: Path, job_name: str, **kwargs: Any) -> FakeResult:
            super().__call__(path, job_name, **kwargs)
            return FakeResult(job_id="140")  # stale id from a cancelled job

    polled: list[str] = []

    def status_fn(job_id: str, **_: Any) -> FakeResult:
        polled.append(job_id)
        # The stale id is a cancelled job; the real one succeeds.
        return FakeResult(status="CANCELLED" if job_id == "140" else "SUCCEEDED")

    executor = _executor(
        spec,
        submitter=StaleIdSubmitter(),
        status_fn=status_fn,
        # The old id still matches the deterministic wave name; newest-first
        # lookup must win even when the stale parsed id appears later in the list.
        name_lookup_fn=lambda name: ["141", "140"],
    )
    report = run_workflow_runtime(
        spec, run_id="rt-stale-id", executor=executor, options=executor.options
    )

    assert report.status == "succeeded"
    assert "140" not in polled, "must not poll the stale id"
    assert report.waves[0]["job_id"] == "141"
    assert any("stale job id" in err for err in report.waves[0]["status_errors"])
    del submitter


def test_empty_job_id_is_recovered_from_the_job_name(tmp_path: Path) -> None:
    spec = load_spec(_write_spec(tmp_path, FANOUT_SPEC))

    class NoIdSubmitter(FakeSubmitter):
        def __call__(self, path: Path, job_name: str, **kwargs: Any) -> FakeResult:
            super().__call__(path, job_name, **kwargs)
            return FakeResult(job_id="")

    executor = _executor(
        spec, submitter=NoIdSubmitter(), name_lookup_fn=lambda name: ["77"]
    )
    report = run_workflow_runtime(
        spec, run_id="rt-recover-id", executor=executor, options=executor.options
    )
    assert report.status == "succeeded"
    assert report.waves[0]["job_id"] == "77"


def test_no_job_id_and_no_name_match_fails_the_wave(tmp_path: Path) -> None:
    spec = load_spec(_write_spec(tmp_path, FANOUT_SPEC))
    cancels: list[dict[str, Any]] = []

    class NoIdSubmitter(FakeSubmitter):
        def __call__(self, path: Path, job_name: str, **kwargs: Any) -> FakeResult:
            super().__call__(path, job_name, **kwargs)
            return FakeResult(job_id="")

    executor = _executor(
        spec,
        submitter=NoIdSubmitter(),
        cancels=cancels,
        name_lookup_fn=lambda name: [],
    )
    report = run_workflow_runtime(
        spec, run_id="rt-no-id", executor=executor, options=executor.options
    )
    assert report.status == "failed"
    assert "could not be found by name" in report.error
    assert not cancels, "no exact job ID means cancellation is not applicable"


# ------------------------------------------------------- review follow-ups (#225)


def test_trigger_listing_is_paginated(tmp_path: Path, mocker) -> None:
    """A single list_objects_v2 caps at 1000 keys; minObjects above that must work."""

    spec = load_spec(_write_spec(tmp_path, TRIGGER_SPEC))
    pages = [
        {
            "Contents": [{"Key": f"inbox/{i}.json"} for i in range(1000)],
            "IsTruncated": True,
            "NextContinuationToken": "tok",
        },
        {"Contents": [{"Key": "inbox/1000.json"}], "IsTruncated": False},
    ]
    calls: list[dict[str, Any]] = []

    class FakeS3:
        def list_objects_v2(self, **kwargs: Any) -> dict[str, Any]:
            calls.append(kwargs)
            return pages[len(calls) - 1]

    mocker.patch(
        "npa.clients.storage.StorageClient.from_environment",
        return_value=mocker.Mock(s3=FakeS3()),
    )
    executor = _executor(spec, run_id="rt-trigger-page")
    waiter = s3_trigger_waiter(ledger=executor.ledger, sleeper=lambda _s: None)
    state = spec.states["ingest"]
    state.trigger.min_objects = 1001  # type: ignore[union-attr]

    watermark = waiter(state, "s3://bucket/inbox/", None)  # type: ignore[arg-type]

    assert watermark["objects"] == 1001
    assert len(calls) == 2, "must follow the continuation token"
    assert calls[1]["ContinuationToken"] == "tok"


def test_resume_refuses_a_ledger_recorded_for_a_different_plan(tmp_path: Path) -> None:
    """Resuming a diverged plan would replay keys that no longer mean the same work."""

    spec = load_spec(_write_spec(tmp_path, FANOUT_SPEC))
    store = MemoryStore()

    first = _executor(spec, run_id="rt-fp", store=store)
    assert (
        run_workflow_runtime(
            spec, run_id="rt-fp", executor=first, options=first.options
        ).status
        == "succeeded"
    )
    recorded = store.read_runtime_state()
    assert recorded is not None and recorded.plan_fingerprint

    # A spec change (extra shard) makes the recorded wave keys meaningless.
    changed = FANOUT_SPEC.replace(
        "    parallel: [shard-a, shard-b, shard-c]",
        "    parallel: [shard-a, shard-b]",
    ).replace(
        """  shard-c:
    description: Shard C.
    run:
      shell: "echo c"
    resources: cpu

""",
        "",
    )
    changed_spec = load_spec(_write_spec(tmp_path, changed, name="changed.yaml"))
    resume_options = RuntimeOptions(poll_seconds=0, max_wait_seconds=60, resume=True)
    resumed = _executor(
        changed_spec, run_id="rt-fp", options=resume_options, store=store
    )

    with pytest.raises(NpaWorkflowError, match="different plan"):
        run_workflow_runtime(
            changed_spec, run_id="rt-fp", executor=resumed, options=resume_options
        )


def test_resume_accepts_an_unchanged_plan(tmp_path: Path) -> None:
    spec = load_spec(_write_spec(tmp_path, FANOUT_SPEC))
    store = MemoryStore()
    first = _executor(spec, run_id="rt-fp-ok", store=store)
    run_workflow_runtime(spec, run_id="rt-fp-ok", executor=first, options=first.options)

    resume_options = RuntimeOptions(poll_seconds=0, max_wait_seconds=60, resume=True)
    submitter = FakeSubmitter()
    second = _executor(
        spec,
        run_id="rt-fp-ok",
        submitter=submitter,
        options=resume_options,
        store=store,
    )
    report = run_workflow_runtime(
        spec, run_id="rt-fp-ok", executor=second, options=resume_options
    )
    assert report.status == "succeeded"
    assert not submitter.calls, "an unchanged plan must replay, not resubmit"
