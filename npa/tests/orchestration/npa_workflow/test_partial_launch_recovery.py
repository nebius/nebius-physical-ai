"""Exercise partial managed-job recovery through the default SDK/runtime boundary."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from npa.orchestration.npa_workflow import build_plan, load_spec
from npa.orchestration.npa_workflow.runtime import NpaWorkflowError, RuntimeOptions
from npa.orchestration.skypilot.launch_transaction import (
    FailureCategory, LaunchState, LaunchTransactionError, LaunchTransactionResult,
)
from npa.orchestration.skypilot.workflow import ManagedJobEvidence
from test_runtime_orchestrator import (
    GATE_LOOP_SPEC, MemoryStore, _executor, _supervisor_preflight, _write_spec,
    runtime_sdk_submission,
)

__all__ = ["runtime_sdk_submission"]


def _partial(kwargs, category=FailureCategory.KUBERNETES_TRANSPORT, *, job_id="41"):
    result = LaunchTransactionResult(
        LaunchState.INDETERMINATE, kwargs["logical_id"], job_id=job_id, launch_sequence=1,
        category=category, existence="found", primary_error="credential RPC stream closed before file sync",
        recovery_decision="reject_unobservable_queue_record_after_launch_failure",
        reconciliations=[dict(state="found", job_id=job_id, status="PENDING", workload_observable=False)],
    )
    kwargs["record"](result.to_dict())
    raise LaunchTransactionError(result.primary_error, result)


@pytest.fixture()
def partial_runtime(tmp_path, runtime_sdk_submission):
    spec = load_spec(_write_spec(tmp_path, GATE_LOOP_SPEC))
    gate = next(step for step in build_plan(spec, run_id="partial-test").steps if step.state == "gate")
    case = SimpleNamespace(spec=spec, gate=gate, sdk=runtime_sdk_submission, launches=[], cancels=[],
                           terminal="CANCELLED", output=False, store=MemoryStore())

    def launch(**kwargs):
        case.launches.append(kwargs["logical_id"])
        if len(case.launches) == 1:
            _partial(kwargs)
        case.output = True
        return LaunchTransactionResult(LaunchState.SUBMITTED, kwargs["logical_id"], job_id="42", launch_sequence=1)

    case.sdk.job.side_effect = launch
    case.lookup = lambda name, job_id="": ManagedJobEvidence(
        "found", job_id=job_id, status=case.terminal if case.cancels else "PENDING",
        workload_observable=bool(case.cancels),
    )
    case.status = lambda job_id: SimpleNamespace(status=case.terminal if job_id == "41" else "SUCCEEDED")
    case.options = RuntimeOptions(poll_seconds=0, preflight_evidence=_supervisor_preflight())
    case.check = lambda uri: case.output
    return case


def _driver(case, **kwargs):
    executor = _executor(
        case.spec, run_id="partial-test", store=case.store, cancels=case.cancels,
        options=kwargs.get("options", case.options), status_fn=case.status,
        reconcile_fn=lambda *args, **kwargs: case.lookup(*args, **kwargs),
        output_checker=lambda uri: case.check(uri),
    )
    executor._submitter = None
    return executor


def test_default_runtime_replaces_verified_partial_launch_once(partial_runtime):
    case = partial_runtime
    executor = _driver(case)
    assert executor.execute(case.gate)["status"] == "ok"
    assert len(case.launches) == 2 and case.launches[0] != case.launches[1]
    assert len(case.cancels) == 1 and case.cancels[0]["job_id"] == "41"
    assert case.sdk.preflight.call_count == 3  # original, fresh recovery, successor
    assert case.sdk.api.call_count == case.sdk.controller.call_count == 2
    parent, successor = executor.attempts
    assert parent.partial_launch["cause"] == "kubernetes_transport"
    assert parent.sky_status == "CANCELLED"
    assert [parent.infrastructure_recovery_count, successor.infrastructure_recovery_count] == [0, 1]
    assert parent.partial_launch["recovery_reservation"] == successor.recovery_reservation
    assert successor.logical_launch_id == parent.partial_launch["recovery_reservation"]["successor"]["logical_attempt_id"]
    proof = executor._attempt_preflight(parent)
    assert proof.scope["source"] == "default_sdk_recovery_preflight" and proof.relaunch_ready


@pytest.mark.parametrize("category", [FailureCategory.AUTH, FailureCategory.RBAC, FailureCategory.CONFIG, FailureCategory.UNKNOWN])
def test_nontransport_partial_row_never_enters_recovery(partial_runtime, category):
    case = partial_runtime
    case.sdk.job.side_effect = lambda **kwargs: _partial(kwargs, category)
    executor = _driver(case)
    with pytest.raises(NpaWorkflowError):
        executor.execute(case.gate)
    assert case.sdk.job.call_count == case.sdk.preflight.call_count == 1
    assert not executor.attempts[0].partial_launch


@pytest.mark.parametrize("mode", ["present", "query_error", "after_cancel"])
def test_outputs_prevent_partial_relaunch(partial_runtime, mode):
    case = partial_runtime

    def check(uri):
        if mode == "query_error":
            raise OSError("synthetic unavailable object store")
        return mode == "present" or bool(case.cancels)

    case.check = check
    executor = _driver(case)
    with pytest.raises(NpaWorkflowError, match="outputs"):
        executor.execute(case.gate)
    assert len(case.launches) == 1
    assert len(case.cancels) == (1 if mode == "after_cancel" else 0)
    assert not executor.attempts[0].recovery_reservation


@pytest.mark.parametrize("valid", [True, False])
def test_cancellation_race_retains_actual_success(partial_runtime, valid):
    case = partial_runtime
    case.terminal = "SUCCEEDED"
    case.check = lambda uri: bool(case.cancels) and valid
    executor = _driver(case)
    if valid:
        assert executor.execute(case.gate)["status"] == "ok"
    else:
        with pytest.raises(NpaWorkflowError, match="terminal success"):
            executor.execute(case.gate)
    assert len(case.launches) == len(case.cancels) == 1
    assert executor.attempts[0].sky_status == "SUCCEEDED"
    assert not executor.attempts[0].recovery_reservation


@pytest.mark.parametrize("outcome", ["absent", "unavailable", "wrong_id", "wrong_name", "running", "unknown_terminal"])
def test_uncertain_exact_identity_or_terminal_blocks(partial_runtime, outcome):
    case = partial_runtime

    def lookup(name, *, job_id=""):
        if outcome == "unknown_terminal" and not case.cancels:
            return ManagedJobEvidence("found", job_id=job_id, status="PENDING", workload_observable=False)
        return SimpleNamespace(
            outcome=outcome if outcome in {"absent", "unavailable"} else "found",
            job_id="99" if outcome == "wrong_id" else job_id,
            job_name="different-name" if outcome == "wrong_name" else name,
            status="RUNNING" if outcome == "running" else "UNKNOWN", workload_observable=True,
        )

    case.lookup = lookup
    executor = _driver(case)
    with pytest.raises(NpaWorkflowError, match="partial-launch recovery blocked"):
        executor.execute(case.gate)
    assert len(case.launches) == 1 and not executor.attempts[0].recovery_reservation


def test_failed_cancellation_never_reserves_recovery(partial_runtime):
    case = partial_runtime
    executor = _driver(case)
    executor._canceller = lambda **kwargs: {"cancel_returncode": 1, "cancel_stderr": "denied"}
    with pytest.raises(NpaWorkflowError, match="cancellation"):
        executor.execute(case.gate)
    assert len(case.launches) == 1 and case.sdk.preflight.call_count == 1


def test_fresh_real_preflight_failure_blocks_replacement(partial_runtime):
    case = partial_runtime
    case.sdk.preflight.side_effect = [(None, {}, {}), ValueError("exact image access denied; token=synthetic-secret")]
    executor = _driver(case)
    with pytest.raises(NpaWorkflowError, match="fresh shared SDK execution preflight"):
        executor.execute(case.gate)
    assert len(case.launches) == len(case.cancels) == 1
    assert not executor._attempt_preflight(executor.attempts[0]).relaunch_ready
    error = executor.attempts[0].error
    assert "exact image access denied" in error and "synthetic-secret" not in error
    assert executor.attempts[0].partial_launch["recovery_preflight_error"] == "exact image access denied; token=<redacted>"


def test_existing_infrastructure_policy_is_not_reset(partial_runtime):
    case = partial_runtime
    case.options = replace(case.options, max_infrastructure_recoveries=0)
    executor = _driver(case)
    with pytest.raises(NpaWorkflowError, match="INFRASTRUCTURE_RECOVERY_EXHAUSTED"):
        executor.execute(case.gate)
    assert len(case.launches) == 1
    assert executor.attempts[0].infrastructure_recovery_exhausted


def _crash_before_successor(case, monkeypatch, *, before_parent_commit=False):
    executor = _driver(case)
    if before_parent_commit:
        original = executor.ledger.record

        def record(attempt):
            if attempt.partial_launch.get("recovery_reservation"):
                raise SystemExit("after immutable reservation, before mutable receipt")
            return original(attempt)

        monkeypatch.setattr(executor.ledger, "record", record)
    else:
        monkeypatch.setattr(executor, "_sleep", lambda seconds: (_ for _ in ()).throw(SystemExit("before successor")))
    with pytest.raises(SystemExit):
        executor.execute(case.gate)
    return executor


def _reservation_events(case):
    from npa.orchestration.npa_workflow.supervisor import SupervisorLedger

    return [event for event in SupervisorLedger(case.store).events() if event["phase"] == "recovery_reserved"]


@pytest.mark.parametrize("before_parent_commit", [False, True])
def test_resume_consumes_existing_reservation_once(partial_runtime, monkeypatch, before_parent_commit):
    case = partial_runtime
    _crash_before_successor(case, monkeypatch, before_parent_commit=before_parent_commit)
    reservation = _reservation_events(case)
    assert len(reservation) == 1
    executor = _driver(case, options=replace(case.options, resume=True))
    assert executor.execute(case.gate)["status"] == "ok"
    assert len(case.launches) == 2 and len(case.cancels) == 1
    assert len(_reservation_events(case)) == 1
    assert executor.attempts[-1].logical_launch_id == reservation[0]["new_attempt_identity"]["logical_attempt_id"]
    assert executor.attempts[-1].infrastructure_recovery_count == 1


def _crash_with_successor_intent(case, monkeypatch, *, posted=False):
    executor = _driver(case)
    original = executor._submit

    def submit(path, name, attempt):
        if attempt.attempt == 2:
            attempt.launch_sequence = int(posted)
            attempt.error_category = "kubernetes_transport"
            attempt.recovery_decision = "block_indeterminate"
            executor.ledger.record(attempt)
            raise SystemExit("reserved successor driver interrupted")
        return original(path, name, attempt)

    monkeypatch.setattr(executor, "_submit", submit)
    with pytest.raises(SystemExit):
        executor.execute(case.gate)
    assert len(case.launches) == 1
    return executor


def test_resume_pre_post_reservation_reuses_exact_identity(partial_runtime, monkeypatch):
    case = partial_runtime
    original = _crash_with_successor_intent(case, monkeypatch)
    successor = original.attempts[-1]
    assert successor.status == "failed" and successor.launch_sequence == 0
    case.lookup = lambda name, job_id="": ManagedJobEvidence("absent")
    executor = _driver(case, options=replace(case.options, resume=True))
    assert executor.execute(case.gate)["status"] == "ok"
    assert len(case.launches) == 2 and case.launches[-1] == successor.logical_launch_id
    assert executor.attempts[-1].recovery_resumed and not executor.attempts[-1].replayed
    assert executor.attempts[-1].infrastructure_recovery_count == 1
    assert len(_reservation_events(case)) == 1


@pytest.mark.parametrize("state", ["RUNNING", "SUCCEEDED"])
def test_failed_reserved_successor_is_adopted_when_observable(partial_runtime, monkeypatch, state):
    case = partial_runtime
    original = _crash_with_successor_intent(case, monkeypatch, posted=True)
    successor = original.attempts[-1]
    assert successor.status == "failed" and not successor.job_id
    assert successor.recovery_decision == "block_indeterminate"
    case.output = True
    case.lookup = lambda name, job_id="": ManagedJobEvidence("found", job_id="42", status=state)
    executor = _driver(case, options=replace(case.options, resume=True))
    assert executor.execute(case.gate)["status"] == "ok"
    adopted = executor.attempts[-1]
    assert adopted.adopted and adopted.job_id == "42"
    assert adopted.logical_launch_id == successor.logical_launch_id and adopted.attempt == 2
    assert adopted.infrastructure_recovery_count == 1
    assert len(case.launches) == 1 and len(case.cancels) == 1
    assert len(_reservation_events(case)) == 1


@pytest.mark.parametrize("outcome", ["absent", "unavailable", "partial"])
def test_posted_reserved_successor_with_uncertain_evidence_never_relaunches(partial_runtime, monkeypatch, outcome):
    case = partial_runtime
    _crash_with_successor_intent(case, monkeypatch, posted=True)
    case.lookup = lambda name, job_id="": ManagedJobEvidence(
        "found" if outcome == "partial" else outcome, job_id="42", status="PENDING", workload_observable=False,
    )
    executor = _driver(case, options=replace(case.options, resume=True, retries=2))
    with pytest.raises(NpaWorkflowError, match="unverified POST or partial-row"):
        executor.execute(case.gate)
    assert len(case.launches) == len(case.cancels) == 1
    assert executor.attempts[-1].infrastructure_recovery_count == 1


@pytest.mark.parametrize("mutation", ["parent_run", "parent_logical", "parent_job", "parent_source", "count", "missing_event", "unreadable_event", "conflict", "no_store"])
def test_successor_consumption_requires_exact_durable_parent_event(partial_runtime, monkeypatch, mutation):
    from npa.orchestration.npa_workflow.supervisor import SupervisorLedger

    case = partial_runtime
    original = _crash_with_successor_intent(case, monkeypatch)
    attempt = original.attempts[-1]
    if mutation.startswith("parent_"):
        field = {"parent_run": "run_id", "parent_logical": "logical_attempt_id",
                 "parent_job": "provider_job_id", "parent_source": "source_sha256"}[mutation]
        attempt.recovery_reservation["parent"][field] = "different"
        original.ledger.record(attempt)
    elif mutation == "count":
        attempt.infrastructure_recovery_count = 0
        original.ledger.record(attempt)
    elif mutation in {"missing_event", "unreadable_event"}:
        for key in list(case.store.objects):
            if "/recovery_reserved-" in key:
                if mutation == "missing_event":
                    del case.store.objects[key]
                else:
                    case.store.objects[key] = b"invalid-json"
    elif mutation == "conflict":
        event = _reservation_events(case)[0]
        event["new_attempt_identity"]["logical_attempt_id"] = "conflicting"
        SupervisorLedger(case.store).record(event)
    executor = _driver(case, options=replace(case.options, resume=True))
    if mutation == "no_store":
        executor.ledger.store = None
    case.lookup = lambda name, job_id="": ManagedJobEvidence("absent")
    with pytest.raises(NpaWorkflowError, match="partial-launch recovery blocked"):
        executor.execute(case.gate)
    assert len(case.launches) == 1 and len(case.cancels) == 1


def test_other_wave_reservation_cannot_replace_exact_parent_event(partial_runtime, monkeypatch):
    from npa.orchestration.npa_workflow.supervisor import SupervisorLedger

    case = partial_runtime
    _crash_before_successor(case, monkeypatch)
    event = _reservation_events(case)[0]
    expected = event["new_attempt_identity"]["logical_attempt_id"]
    event["attempt_identity"]["logical_attempt_id"] = "other-wave-parent"
    event["new_attempt_identity"]["logical_attempt_id"] = "other-wave-successor"
    SupervisorLedger(case.store).record(event)
    executor = _driver(case, options=replace(case.options, resume=True))
    assert executor.execute(case.gate)["status"] == "ok"
    assert executor.attempts[-1].logical_launch_id == expected
    assert len(case.launches) == 2 and len(_reservation_events(case)) == 2


@pytest.mark.parametrize("identity", ["workflow_sha256", "source_sha256", "image_digest", "job_name", "logical_launch_id"])
def test_recorded_partial_identity_drift_blocks_before_cancel(partial_runtime, monkeypatch, identity):
    from npa.orchestration.npa_workflow import launch_recovery

    case = partial_runtime
    original = launch_recovery._capture_partial_launch

    def capture(attempt, payload, rendered_hash, run_id):
        original(attempt, payload, rendered_hash, run_id)
        if attempt.partial_launch:
            setattr(attempt, identity, "different")
            if identity == "logical_launch_id":
                attempt.partial_launch["parent"]["logical_attempt_id"] = "different"

    monkeypatch.setattr(launch_recovery, "_capture_partial_launch", capture)
    executor = _driver(case)
    with pytest.raises(NpaWorkflowError, match="immutable attempt identity"):
        executor.execute(case.gate)
    assert len(case.launches) == 1 and not case.cancels


def test_custom_submitter_exception_cannot_assert_real_sdk_proof(partial_runtime):
    from npa.orchestration.skypilot.workflow import SkyPilotSubmitError

    case = partial_runtime
    executor = _driver(case)

    def custom(path, name, **kwargs):
        try:
            _partial({"logical_id": "arbitrary", "record": lambda payload: None})
        except LaunchTransactionError as exc:
            raise SkyPilotSubmitError("synthetic failure", transaction=exc.result) from exc

    executor._submitter = custom
    with pytest.raises(NpaWorkflowError, match="synthetic failure"):
        executor.execute(case.gate)
    assert not executor.attempts[0].partial_launch
    assert case.sdk.preflight.call_count == case.sdk.job.call_count == 0


@pytest.mark.parametrize("mutation", ["mixed_observable", "unknown_previous", "wrong_job", "wrong_logical", "unknown_state"])
def test_sdk_partial_proof_rejects_mixed_or_mismatched_evidence(partial_runtime, mutation):
    case = partial_runtime

    def launch(**kwargs):
        recorder = kwargs["record"]

        def record(payload):
            if mutation in {"mixed_observable", "unknown_previous"}:
                previous = dict(state="found", status="RUNNING", job_id="41", workload_observable=True)
                if mutation == "unknown_previous":
                    previous = dict(state="unavailable")
                payload["reconciliations"].insert(0, previous)
            elif mutation == "wrong_job":
                payload["reconciliations"][-1]["job_id"] = "99"
            elif mutation == "wrong_logical":
                payload["logical_launch_id"] = "other-attempt"
            else:
                payload["reconciliations"][-1]["status"] = "UNKNOWN"
            recorder(payload)

        _partial({**kwargs, "record": record})

    case.sdk.job.side_effect = launch
    executor = _driver(case)
    with pytest.raises(NpaWorkflowError):
        executor.execute(case.gate)
    assert not executor.attempts[0].partial_launch
    assert case.sdk.job.call_count == case.sdk.preflight.call_count == 1


def test_output_appearing_during_recovery_preflight_blocks(partial_runtime):
    case = partial_runtime

    def preflight(docs, **kwargs):
        if case.cancels:
            case.output = True
        return None, {}, {}

    case.sdk.preflight.side_effect = preflight
    executor = _driver(case)
    with pytest.raises(NpaWorkflowError, match="outputs"):
        executor.execute(case.gate)
    assert len(case.launches) == 1 and len(case.cancels) == 1
    assert case.sdk.preflight.call_count == 2 and not _reservation_events(case)


def test_declared_output_substitution_blocks_resumed_recovery(partial_runtime, monkeypatch):
    case = partial_runtime
    original = _crash_before_successor(case, monkeypatch)
    parent = original.attempts[-1]
    parent.outputs[0]["uri"] = "s3://unit-bucket/different/report.json"
    original.ledger.record(parent)
    executor = _driver(case, options=replace(case.options, resume=True))
    with pytest.raises(NpaWorkflowError, match="immutable attempt identity"):
        executor.execute(case.gate)
    assert len(case.launches) == len(case.cancels) == 1


def test_unavailable_reservation_store_cannot_consume_successor(partial_runtime, monkeypatch):
    case = partial_runtime
    _crash_with_successor_intent(case, monkeypatch)
    monkeypatch.setattr(case.store, "list_artifacts", lambda prefix: (_ for _ in ()).throw(OSError("unavailable")))
    executor = _driver(case, options=replace(case.options, resume=True))
    with pytest.raises(NpaWorkflowError, match="durable reservation evidence"):
        executor.execute(case.gate)
    assert len(case.launches) == 1 and len(case.cancels) == 1


def test_fresh_preflight_keeps_failed_parent_preparation_bytes(partial_runtime):
    case = partial_runtime
    prepared = {}

    def launch(**kwargs):
        prepared["directory"] = Path(case.sdk.preflight.call_args.kwargs["extra_env"]["SKYPILOT_GLOBAL_CONFIG"]).parent
        prepared["files"] = {path.name: path.read_bytes() for path in prepared["directory"].iterdir() if path.is_file()}
        _partial(kwargs)

    from pathlib import Path

    case.sdk.job.side_effect = launch
    case.sdk.preflight.side_effect = [(None, {}, {}), (None, {}, {"token": "refreshed-value"})]
    executor = _driver(case)
    # Stop after the real refresh; no third SDK launch is needed for this check.
    executor.options = replace(case.options, max_infrastructure_recoveries=0)
    with pytest.raises(NpaWorkflowError, match="INFRASTRUCTURE_RECOVERY_EXHAUSTED"):
        executor.execute(case.gate)
    assert case.sdk.preflight.call_count == 2
    assert {path.name: path.read_bytes() for path in prepared["directory"].iterdir() if path.is_file()} == prepared["files"]
    refreshed = Path(case.sdk.preflight.call_args.kwargs["extra_env"]["SKYPILOT_GLOBAL_CONFIG"]).parent
    assert refreshed != prepared["directory"] and refreshed.parent == prepared["directory"].parent
    assert case.sdk.api.call_count == case.sdk.controller.call_count == 1


def _configure_two_recoveries(case):
    case.options = replace(case.options, max_infrastructure_recoveries=2)

    def launch(**kwargs):
        case.launches.append(kwargs["logical_id"])
        if len(case.launches) < 3:
            _partial(kwargs, job_id=str(40 + len(case.launches)))
        case.output = True
        return LaunchTransactionResult(LaunchState.SUBMITTED, kwargs["logical_id"], job_id="43", launch_sequence=1)

    def lookup(name, *, job_id=""):
        cancelled = any(item["job_id"] == job_id for item in case.cancels)
        return ManagedJobEvidence("found", job_id=job_id, status="CANCELLED" if cancelled else "PENDING",
                                  workload_observable=cancelled)

    case.sdk.job.side_effect = launch
    case.lookup = lookup
    case.status = lambda job_id: SimpleNamespace(status="SUCCEEDED" if job_id == "43" else "CANCELLED")


def test_multiple_authorized_recoveries_keep_both_chain_links(partial_runtime):
    case = partial_runtime
    _configure_two_recoveries(case)
    executor = _driver(case)
    assert executor.execute(case.gate)["status"] == "ok"
    first, second, final = executor.attempts
    assert [value.infrastructure_recovery_count for value in executor.attempts] == [0, 1, 2]
    assert second.recovery_reservation == first.partial_launch["recovery_reservation"]
    assert final.recovery_reservation == second.partial_launch["recovery_reservation"]
    assert second.recovery_reservation != second.partial_launch["recovery_reservation"]
    assert len(case.launches) == 3 and len(case.cancels) == 2
    assert len(_reservation_events(case)) == 2 and case.sdk.preflight.call_count == 5


@pytest.fixture()
def audit_runtime(monkeypatch):
    import importlib
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]))
    return importlib.import_module("tests.e2e.paidf_runtime_audit").assert_completed_fresh_runtime


def _audit_case(case, executor):
    import io

    prefix = "paidf-cosmos3/partial-test/"
    objects = {prefix + key[key.index("npa-workflow/"):]: body for key, body in case.store.objects.items()
               if "npa-workflow/" in key}
    objects.update(getattr(case, "published", {}))
    client = SimpleNamespace(
        head_object=lambda **kwargs: {"ContentLength": len(objects[kwargs["Key"]])},
        get_paginator=lambda name: SimpleNamespace(paginate=lambda **kwargs: [
            {"Contents": [{"Key": key, "Size": len(objects[key])} for key in objects if key.startswith(kwargs["Prefix"])]}
        ]),
        get_object=lambda **kwargs: {"Body": io.BytesIO(objects[kwargs["Key"]])},
    )
    runtime = executor.ledger.state.to_dict()
    runtime["status"] = "succeeded"  # The outer driver writes this after all logical waves finish.
    return client, prefix, objects, runtime


@pytest.mark.parametrize("recoveries", [1, 2])
def test_completed_paidf_audit_validates_full_automatic_chain(partial_runtime, audit_runtime, recoveries):
    case = partial_runtime
    if recoveries == 2:
        _configure_two_recoveries(case)
    executor = _driver(case)
    assert executor.execute(case.gate)["status"] == "ok"
    client, prefix, objects, runtime = _audit_case(case, executor)
    audit_runtime(client, "unit-bucket", prefix, runtime)
    assert len(runtime["waves"]) == recoveries + 1


def _change_event(objects, change, *, phase="recovery_reserved"):
    import hashlib
    import json

    key = next(key for key in objects if "/" + phase + "-" in key)
    event = json.loads(objects.pop(key))
    change(event)
    body = (json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n").encode()
    key = key.rsplit("/", 1)[0] + "/" + phase + "-" + hashlib.sha256(body).hexdigest() + ".json"
    objects[key] = body


@pytest.mark.parametrize("mutation", ["auth", "uncancelled", "parent", "successor", "manual_retry", "extra_failure",
                                     "replayed", "adopted", "preflight", "outputs", "event_identity", "event_count",
                                     "event_hash", "missing_event"])
def test_paidf_audit_rejects_unproven_recovery(partial_runtime, audit_runtime, mutation):
    import copy

    case = partial_runtime
    executor = _driver(case)
    executor.execute(case.gate)
    client, prefix, objects, runtime = _audit_case(case, executor)
    parent, successor = runtime["waves"]
    if mutation == "auth":
        parent["partial_launch"]["cause"] = "auth"
    elif mutation == "uncancelled":
        parent["cancellation"]["state"] = "requested"
    elif mutation == "parent":
        parent["partial_launch"]["parent"]["provider_job_id"] = "different"
    elif mutation == "successor":
        successor["logical_launch_id"] = "different"
    elif mutation == "manual_retry":
        parent["partial_launch"] = {}
    elif mutation == "extra_failure":
        runtime["waves"].insert(1, copy.deepcopy(parent))
    elif mutation in {"replayed", "adopted"}:
        successor[mutation] = True
    elif mutation == "event_hash":
        key = next(key for key in objects if "/recovery_reserved-" in key)
        objects[key] += b" "
    elif mutation == "missing_event":
        for key in list(objects):
            if "/recovery_reserved-" in key:
                del objects[key]
    else:
        changes = {"preflight": lambda event: event["preflight"]["checks"].update(gang_capacity="unknown"),
                   "outputs": lambda event: event["outputs"].update(status="partial"),
                   "event_identity": lambda event: event["attempt_identity"].update(provider_job_id="different"),
                   "event_count": lambda event: event["infrastructure_recovery_policy"].update(used=1)}
        _change_event(objects, changes[mutation])
    with pytest.raises((AssertionError, KeyError)):
        audit_runtime(client, "unit-bucket", prefix, runtime)


def test_paidf_audit_rejects_operator_resumed_recovery(partial_runtime, monkeypatch, audit_runtime):
    case = partial_runtime
    _crash_before_successor(case, monkeypatch)
    executor = _driver(case, options=replace(case.options, resume=True))
    executor.execute(case.gate)
    client, prefix, objects, runtime = _audit_case(case, executor)
    with pytest.raises(AssertionError):
        audit_runtime(client, "unit-bucket", prefix, runtime)


@pytest.mark.parametrize("mutation", ["false_ready", "missing_ready", "missing_marker", "unexpected_marker", "scope", "hash"])
def test_paidf_audit_rejects_invalid_redacted_preflight_proof(partial_runtime, audit_runtime, mutation):
    case = partial_runtime
    executor = _driver(case)
    executor.execute(case.gate)
    client, prefix, objects, runtime = _audit_case(case, executor)

    def change(event):
        preflight = event["preflight"]
        if mutation == "false_ready":
            preflight["relaunch_ready"] = False
        elif mutation == "missing_ready":
            del preflight["relaunch_ready"]
        elif mutation == "missing_marker":
            del preflight["checks"]["credentials_access"]
        elif mutation == "unexpected_marker":
            preflight["checks"]["credentials_access"] = "pass"
        elif mutation == "scope":
            preflight["scope"]["source"] = "synthetic_or_stale"
        else:
            preflight["scope"]["rendered_wave_sha256"] = "d" * 64

    _change_event(objects, change)
    with pytest.raises((AssertionError, KeyError)):
        audit_runtime(client, "unit-bucket", prefix, runtime)


def test_arbitrary_cancelled_status_is_not_exact_cancellation_proof(partial_runtime):
    case = partial_runtime
    case.lookup = lambda name, job_id="": ManagedJobEvidence("found", job_id=job_id, status="CANCELLED")
    executor = _driver(case)
    with pytest.raises(NpaWorkflowError, match="recorded exact cancellation"):
        executor.execute(case.gate)
    assert len(case.launches) == 1 and not case.cancels
    assert not _reservation_events(case)


def test_reserved_successor_refuses_changed_event_bytes(partial_runtime, monkeypatch):
    case = partial_runtime
    _crash_with_successor_intent(case, monkeypatch)
    key = next(key for key in case.store.objects if "/recovery_reserved-" in key)
    case.store.objects[key] += b" "
    executor = _driver(case, options=replace(case.options, resume=True))
    with pytest.raises(NpaWorkflowError, match="durable reservation evidence"):
        executor.execute(case.gate)
    assert len(case.launches) == 1 and len(case.cancels) == 1



def _success_race(case, tmp_path, *, kind="file", before_cancel=False):
    from urllib.parse import urlsplit

    text = GATE_LOOP_SPEC.replace("example-bucket", "unit-bucket").replace("gate-loop/", "paidf-cosmos3/")
    if kind == "directory":
        text = text.replace("schema: npa.sim2real.threshold_decision.v1", "schema: npa.sim2real.threshold_decision.v1\n        kind: directory")
    case.spec = load_spec(_write_spec(tmp_path, text))
    case.gate = next(step for step in build_plan(case.spec, run_id="partial-test").steps if step.state == "gate")
    output = case.gate.outputs[0]
    key = urlsplit(output["uri"]).path.lstrip("/")
    published_key = key.rstrip("/") + "/decision.json" if kind == "directory" else key
    case.published = {}
    case.terminal = "SUCCEEDED"
    case.check = lambda uri: bool(case.published)

    def status(job_id):
        case.published[published_key] = b'{"run_id":"partial-test","decision":"promote_checkpoint"}\n'
        return SimpleNamespace(status="SUCCEEDED")

    if before_cancel:
        def lookup(name, *, job_id=""):
            status(job_id)
            return ManagedJobEvidence("found", job_id=job_id, status="SUCCEEDED")
        case.lookup = lookup
    case.status = status
    executor = _driver(case)
    assert executor.execute(case.gate)["status"] == "ok"
    return executor, _audit_case(case, executor)


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_paidf_audit_accepts_actual_success_racing_exact_cancellation(partial_runtime, tmp_path, audit_runtime, kind):
    case = partial_runtime
    executor, (client, prefix, objects, runtime) = _success_race(case, tmp_path, kind=kind)
    actual = executor.attempts[0]
    assert len(case.launches) == len(case.cancels) == len(executor.attempts) == 1
    assert actual.sky_status == "SUCCEEDED" and actual.status == "succeeded"
    assert actual.partial_launch and actual.cancellation_state == "verified"
    assert not actual.replayed and not actual.adopted and not actual.recovery_resumed
    audit_runtime(client, "unit-bucket", prefix, runtime)
    assert case.sdk.preflight.call_count == 1 and not _reservation_events(case)


@pytest.mark.parametrize("category", ["auth", "rbac", "config"])
def test_paidf_chain_audit_rejects_category_inconsistent_with_transport_proof(partial_runtime, audit_runtime, category):
    case = partial_runtime
    executor = _driver(case)
    executor.execute(case.gate)
    client, prefix, objects, runtime = _audit_case(case, executor)
    runtime["waves"][0]["error_category"] = category
    with pytest.raises(AssertionError):
        audit_runtime(client, "unit-bucket", prefix, runtime)


@pytest.mark.parametrize("mutation", ["auth", "rbac", "config", "proof_identity", "requested_cancel", "false_success",
                                     "outgoing_reservation", "manual_resume", "replayed", "adopted", "wrong_decision"])
def test_paidf_success_race_audit_rejects_inconsistent_completion(partial_runtime, tmp_path, audit_runtime, mutation):
    case = partial_runtime
    executor, (client, prefix, objects, runtime) = _success_race(case, tmp_path)
    wave = runtime["waves"][0]
    if mutation in {"auth", "rbac", "config"}:
        wave["error_category"] = mutation
    elif mutation == "proof_identity":
        wave["partial_launch"]["parent"]["provider_job_id"] = "another-job"
    elif mutation == "requested_cancel":
        wave["cancellation"]["state"] = "requested"
    elif mutation == "false_success":
        wave["sky_status"] = "CANCELLED"
    elif mutation == "outgoing_reservation":
        wave["partial_launch"]["recovery_reservation"] = {"used": 1}
    elif mutation == "manual_resume":
        wave["recovery_resumed"] = True
    elif mutation in {"replayed", "adopted"}:
        wave[mutation] = True
    else:
        wave["recovery_decision"] = "relaunch_incomplete_wave"
    with pytest.raises(AssertionError):
        audit_runtime(client, "unit-bucket", prefix, runtime)


@pytest.mark.parametrize("mutation", ["missing", "hash", "identity", "status", "category", "outputs"])
def test_paidf_success_race_requires_exact_terminal_receipt(partial_runtime, tmp_path, audit_runtime, mutation):
    case = partial_runtime
    executor, (client, prefix, objects, runtime) = _success_race(case, tmp_path)
    key = next(key for key in objects if "/attempt_terminal-" in key)
    if mutation == "missing":
        del objects[key]
    elif mutation == "hash":
        objects[key] += b" "
    else:
        changes = {"identity": lambda event: event["attempt_identity"].update(provider_job_id="different"),
                   "status": lambda event: event["attempt"].update(sky_status="CANCELLED"),
                   "category": lambda event: event.update(classification="auth"),
                   "outputs": lambda event: event["attempt"].update(outputs=[])}
        _change_event(objects, changes[mutation], phase="attempt_terminal")
    with pytest.raises(AssertionError):
        audit_runtime(client, "unit-bucket", prefix, runtime)


@pytest.mark.parametrize("kind", ["file", "directory"])
@pytest.mark.parametrize("mutation", ["empty", "missing", "unavailable"])
def test_paidf_success_race_requires_published_declared_outputs(partial_runtime, tmp_path, audit_runtime, kind, mutation):
    case = partial_runtime
    executor, (client, prefix, objects, runtime) = _success_race(case, tmp_path, kind=kind)
    key = next(iter(case.published))
    if mutation == "empty":
        objects[key] = b""
    elif mutation == "missing":
        del objects[key]
    else:
        def unavailable(**kwargs):
            raise OSError("synthetic storage query unavailable")
        client.head_object = unavailable
        if kind == "directory":
            original = client.get_paginator
            def paginator(name):
                delegate = original(name)
                def paginate(**kwargs):
                    if kwargs["Prefix"].startswith(prefix + "gate/"):
                        raise OSError("synthetic storage query unavailable")
                    return delegate.paginate(**kwargs)
                return SimpleNamespace(paginate=paginate)
            client.get_paginator = paginator
    with pytest.raises((AssertionError, KeyError, OSError)):
        audit_runtime(client, "unit-bucket", prefix, runtime)


@pytest.mark.parametrize("uri", ["s3://another-bucket/paidf-cosmos3/partial-test/output.json",
    "s3://unit-bucket/paidf-cosmos3/partial-test-collision/output.json",
    "s3://unit-bucket/paidf-cosmos3/partial-test/../other/output.json",
    "s3://unit-bucket/paidf-cosmos3/partial-test/output.json?signature=synthetic",
    "s3://unit-bucket/paidf-cosmos3/partial-test/output.json#fragment"])
def test_paidf_success_race_rejects_unowned_output_locations(partial_runtime, tmp_path, audit_runtime, uri):
    case = partial_runtime
    executor, (client, prefix, objects, runtime) = _success_race(case, tmp_path)
    wave = runtime["waves"][0]
    wave["outputs"][0]["uri"] = uri
    _change_event(objects, lambda event: event["attempt"].update(outputs=wave["outputs"]), phase="attempt_terminal")
    client.head_object = lambda **kwargs: pytest.fail("unowned output must not be queried")
    with pytest.raises(AssertionError):
        audit_runtime(client, "unit-bucket", prefix, runtime)



@pytest.mark.parametrize("kind", ["file", "directory"])
def test_paidf_audit_accepts_verified_success_before_cancellation_is_needed(partial_runtime, tmp_path, audit_runtime, kind):
    case = partial_runtime
    executor, (client, prefix, objects, runtime) = _success_race(case, tmp_path, kind=kind, before_cancel=True)
    actual = executor.attempts[0]
    assert len(case.launches) == 1 and not case.cancels
    assert actual.cancellation_state == "not_applicable" and actual.sky_status == "SUCCEEDED"
    assert not actual.partial_launch.get("outputs_absent_before_cancel")
    audit_runtime(client, "unit-bucket", prefix, runtime)
