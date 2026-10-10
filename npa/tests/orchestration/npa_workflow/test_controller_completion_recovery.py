"""Prove exact successful jobs recover controller failures without duplicate work."""

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
import yaml

from npa.orchestration.npa_workflow import build_plan, load_spec
from npa.orchestration.npa_workflow.runtime import RuntimeOptions, run_workflow_runtime
from test_runtime_orchestrator import (
    GATE_LOOP_SPEC,
    FANOUT_SPEC,
    FakeStatus,
    FakeSubmitter,
    MemoryStore,
    _decision_reader,
    _executor,
    _write_spec,
)


@pytest.fixture
def failed_controller(tmp_path):
    spec = load_spec(_write_spec(tmp_path, GATE_LOOP_SPEC))
    store = MemoryStore()
    first = _executor(
        spec,
        run_id="controller-completion",
        store=store,
        status_fn=FakeStatus(["SUCCEEDED", "FAILED_CONTROLLER"]),
    )
    report = run_workflow_runtime(
        spec,
        run_id=first.run_id,
        executor=first,
        options=first.options,
        decision_reader=_decision_reader(["promote_checkpoint"]),
    )
    assert report.status == "failed"
    record = copy.deepcopy(report.waves[-1])
    assert record["sky_status"] == "FAILED_CONTROLLER"
    # The fake submitter omits the native transaction receipt; model its closed
    # launch evidence while retaining the real runtime's controller-failure fields.
    record["launch_sequence"] = 1
    record["error_category"] = "none"
    state = store.read_runtime_state()
    state.record_wave(record)
    store.write_runtime_state(state)
    return SimpleNamespace(spec=spec, store=store, record=record, run_id=first.run_id)


def _evidence(case, **changes):
    record = case.record
    evidence = SimpleNamespace(
        outcome="found",
        job_id=record["job_id"],
        job_name=record["job_name"],
        status="SUCCEEDED",
        workload_observable=True,
        task_rows=(
            {
                "job_id": record["job_id"],
                "task_id": 0,
                "task_name": record["job_name"],
                "status": "SUCCEEDED",
            },
        ),
    )
    return SimpleNamespace(**{**vars(evidence), **changes})


def _resume(case, *, evidence=None, output_checker=None, retries=0):
    submits, observations, reads, cancels = FakeSubmitter(), [], [], []

    def reconcile(name, *, job_id=""):
        observations.append((name, job_id))
        return evidence if evidence is not None else _evidence(case)

    def check_output(uri):
        reads.append(uri)
        return output_checker(uri) if output_checker is not None else True

    options = RuntimeOptions(
        poll_seconds=0, max_wait_seconds=60, resume=True, retries=retries
    )
    executor = _executor(
        case.spec,
        run_id=case.run_id,
        store=case.store,
        options=options,
        submitter=submits,
        reconcile_fn=reconcile,
        output_checker=check_output,
        cancels=cancels,
    )
    report = run_workflow_runtime(
        case.spec,
        run_id=case.run_id,
        executor=executor,
        options=options,
        decision_reader=_decision_reader(["promote_checkpoint"]),
    )
    return SimpleNamespace(
        report=report,
        submits=submits.calls,
        observations=observations,
        reads=reads,
        cancels=cancels,
    )


def test_verified_controller_completion_advances_without_repeating_job(
    failed_controller,
):
    case = failed_controller
    before = copy.deepcopy(case.record)
    terminal_events = {
        key: body
        for key, body in case.store.objects.items()
        if "/attempt_terminal-" in key
    }
    assert terminal_events
    outcome = _resume(case)
    assert outcome.report.status == "succeeded"
    assert [call["tasks"] for call in outcome.submits] == [["publish"]]
    assert outcome.observations == [(before["job_name"], before["job_id"])]
    assert outcome.cancels == []
    assert all(case.store.objects[key] == body for key, body in terminal_events.items())
    adopted = next(wave for wave in outcome.report.waves if wave["states"] == ["gate"])
    for name in (
        "job_id",
        "job_name",
        "logical_launch_id",
        "attempt",
        "launch_sequence",
        "immutable_identity",
    ):
        assert adopted[name] == before[name]
    assert adopted["status"] == "succeeded"
    assert adopted["replayed"] is True and adopted["adopted"] is True
    assert adopted["primary_error"] == before["error"]
    assert adopted["error"] == ""
    proof = adopted["reconciliation"][-1]
    assert adopted["tasks"] == list(_evidence(case).task_rows)
    assert proof["task_rows"] == adopted["tasks"]
    assert proof["job_name"] == before["job_name"]
    assert proof["workload_observable"] is True
    assert proof["prior_sky_status"] == "FAILED_CONTROLLER"
    assert (
        proof["prior_record_sha256"]
        == hashlib.sha256(
            json.dumps(before, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )

    replay = _resume(case)
    assert replay.report.status == "succeeded"
    assert replay.submits == [] and replay.observations == [] and replay.cancels == []


@pytest.mark.parametrize("retries", [0, 3])
@pytest.mark.parametrize(
    "change",
    [
        {"outcome": "absent"},
        {"outcome": "unavailable"},
        {"job_id": "999"},
        {"job_name": "another-job"},
        {"job_name": ""},
        {"workload_observable": False},
        {"status": "RUNNING"},
        {"status": "FAILED"},
        {"status": "UNKNOWN"},
        {"task_rows": ()},
    ],
)
def test_unverified_controller_completion_never_retries(
    failed_controller, change, retries
):
    case = failed_controller
    before = copy.deepcopy(case.store.read_runtime_state().waves)
    outcome = _resume(case, evidence=_evidence(case, **change), retries=retries)
    assert outcome.report.status == "failed"
    assert "controller completion blocked" in outcome.report.error
    assert outcome.submits == [] and outcome.cancels == []
    assert case.store.read_runtime_state().waves == before


@pytest.mark.parametrize(
    "change",
    [
        {"status": "RUNNING"},
        {"status": ""},
        {"job_id": "999"},
        {"task_id": True},
        {"task_id": "0"},
        {"task_id": 1},
        {"task_name": "gate"},
        {"task_name": "another-job"},
    ],
)
def test_controller_completion_requires_native_task_membership(
    failed_controller, change
):
    case = failed_controller
    good = _evidence(case)
    evidence = SimpleNamespace(
        **{
            **vars(good),
            "task_rows": ({**good.task_rows[0], **change},),
        }
    )
    outcome = _resume(case, evidence=evidence, retries=3)
    assert outcome.report.status == "failed"
    assert outcome.submits == [] and outcome.cancels == []


@pytest.mark.parametrize(
    "name,value",
    [
        ("attempt", True),
        ("launch_sequence", False),
        ("launch_sequence", 0),
        ("job_id", "01"),
        ("job_id", "0"),
        ("job_id", "١"),
        ("job_name", "another-job"),
        ("logical_launch_id", "other-attempt"),
        ("states", ["another-state"]),
        ("outputs", []),
        ("partial_launch", {"launch_sequence": 1}),
        ("recovery_reservation", {"attempt": 2}),
    ],
)
def test_controller_completion_rejects_changed_identity_before_provider(
    failed_controller, name, value
):
    case = failed_controller
    state = case.store.read_runtime_state()
    state.waves[-1][name] = value
    case.store.write_runtime_state(state)
    outcome = _resume(case, retries=3)
    assert outcome.report.status == "failed"
    assert outcome.submits == [] and outcome.cancels == []
    assert outcome.observations == []


@pytest.mark.parametrize("name", ["workflow_sha256", "source_sha256", "image_digest"])
def test_controller_completion_rejects_immutable_identity_change(
    failed_controller, name
):
    case = failed_controller
    state = case.store.read_runtime_state()
    state.waves[-1]["immutable_identity"][name] = "changed"
    case.store.write_runtime_state(state)
    outcome = _resume(case, retries=3)
    assert outcome.report.status == "failed"
    assert outcome.observations == [] and outcome.submits == []


def test_controller_completion_requires_real_declared_outputs(failed_controller):
    outcome = _resume(failed_controller, output_checker=lambda _uri: False, retries=3)
    assert outcome.report.status == "failed"
    assert "declared outputs are not verified" in outcome.report.error
    assert outcome.submits == [] and outcome.cancels == []


def test_controller_completion_storage_failure_keeps_original_wave(failed_controller):
    def unavailable(_uri):
        raise PermissionError("storage unavailable")

    before = copy.deepcopy(failed_controller.store.read_runtime_state().waves)
    outcome = _resume(failed_controller, output_checker=unavailable, retries=3)
    assert outcome.report.status == "failed"
    assert outcome.submits == [] and outcome.cancels == []
    assert failed_controller.store.read_runtime_state().waves == before


@pytest.mark.parametrize("member_status", ["SUCCEEDED", "RUNNING"])
def test_parallel_controller_recovery_reuses_whole_wave(tmp_path, member_status):
    document = yaml.safe_load(FANOUT_SPEC)
    for name in ("shard-a", "shard-b"):
        document["states"][name]["outputs"] = [
            {
                "uri": "s3://example-bucket/{{config.prefix}}/" + name + ".json",
            }
        ]
    spec = load_spec(_write_spec(tmp_path, yaml.safe_dump(document)))
    store = MemoryStore()
    first = _executor(spec, store=store, status_fn=FakeStatus(["FAILED_CONTROLLER"]))
    report = run_workflow_runtime(
        spec, run_id=first.run_id, executor=first, options=first.options
    )
    assert report.status == "failed"
    record = copy.deepcopy(report.waves[-1])
    record["launch_sequence"] = 1
    state = store.read_runtime_state()
    state.record_wave(record)
    store.write_runtime_state(state)
    case = SimpleNamespace(spec=spec, store=store, record=record, run_id=first.run_id)
    rows = tuple(
        {
            "job_id": record["job_id"],
            "task_id": index,
            "task_name": name,
            "status": member_status if index == 1 else "SUCCEEDED",
        }
        for index, name in enumerate(record["states"])
    )
    outcome = _resume(case, evidence=_evidence(case, task_rows=rows[::-1]), retries=3)
    if member_status == "RUNNING":
        assert outcome.report.status == "failed"
        assert outcome.submits == []
        return
    assert outcome.report.status == "succeeded"
    assert [call["tasks"] for call in outcome.submits] == [["shard-c"], ["join"]]
    assert outcome.report.waves[0]["job_id"] == record["job_id"]
    assert outcome.report.waves[0]["tasks"] == list(rows[::-1])


@pytest.mark.parametrize(
    "mutation,valid",
    [
        ("unchanged", True),
        ("reordered", True),
        ("duplicate-id", False),
        ("wrong-name", False),
        ("missing", False),
        ("bool-id", False),
    ],
)
def test_controller_completion_binds_parallel_native_members(tmp_path, mutation, valid):
    spec = load_spec(_write_spec(tmp_path, FANOUT_SPEC))
    executor = _executor(spec)
    steps = build_plan(spec, run_id=executor.run_id).steps[:2]
    assert [step.state for step in steps] == ["shard-a", "shard-b"]
    rows = [
        {"task_id": index, "task_name": step.state} for index, step in enumerate(steps)
    ]
    if mutation == "reordered":
        rows.reverse()
    elif mutation == "duplicate-id":
        rows[1]["task_id"] = 0
    elif mutation == "wrong-name":
        rows[1]["task_name"] = "other-task"
    elif mutation == "missing":
        rows.pop()
    elif mutation == "bool-id":
        rows[1]["task_id"] = True
    assert (
        executor._controller_task_membership_matches(
            {"job_name": "parallel-job"}, steps, rows
        )
        is valid
    )
