"""Recover a proven pre-payload transport failure through durable wave supervision."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any

from npa.orchestration.npa_workflow.run_state import utc_now
from npa.orchestration.npa_workflow.supervisor import (
    AttemptIdentity,
    BackendObservation,
    BackendState,
    RecoveryContext,
    SupervisorLedger,
    WorkflowRunSupervisor,
    validate_declared_outputs,
)
from npa.orchestration.skypilot.launch_transaction import logical_launch_identity
from npa.verification import sanitize_reason

_TRANSIENT_CAUSES = frozenset(
    {
        "kubernetes_transport",
        "kubernetes_rate_limit",
        "kubernetes_server",
    }
)
_PARTIAL_DECISION = "reject_unobservable_queue_record_after_launch_failure"


def _identity(attempt: Any, run_id: str) -> AttemptIdentity:
    return AttemptIdentity(
        runtime="skypilot",
        run_id=run_id,
        attempt=attempt.attempt,
        logical_attempt_id=attempt.logical_launch_id,
        provider_job_id=attempt.job_id,
        provider_job_name=attempt.job_name,
        workflow_sha256=attempt.workflow_sha256,
        source_sha256=attempt.source_sha256,
        image_digest=attempt.image_digest,
    )


def _capture_partial_launch(
    attempt: Any, payload: Any, rendered_hash: str, run_id: str
) -> None:
    # Only the default SDK's post-preflight recorder calls this function. Neither
    # a custom submitter's exception nor an arbitrary cancelled row is proof.
    rows = payload.get("reconciliations") or []
    row = rows[-1] if rows else {}
    job_id = str(payload.get("job_id") or "")
    if not (
        payload.get("category") in _TRANSIENT_CAUSES
        and payload.get("recovery_decision") == _PARTIAL_DECISION
        and payload.get("logical_launch_id") == attempt.logical_launch_id
        and int(payload.get("launch_sequence") or 0) > 0
        and job_id
        and row.get("job_id") == job_id
        and row.get("state") == "found"
        and row.get("status") == "PENDING"
        and row.get("workload_observable") is False
        and not attempt.observations
        and all(previous.get("state") == "absent" for previous in rows[:-1])
    ):
        return
    parent = replace(_identity(attempt, run_id), provider_job_id=job_id)
    attempt.partial_launch = {
        "schema": 1,
        "wave_key": attempt.key,
        "parent": parent.to_dict(),
        "rendered_wave_sha256": rendered_hash,
        "cause": payload["category"],
        "workload_observable_at_failure": False,
    }


def _blocked(attempt: Any, reason: str, *, cause: BaseException | None = None) -> None:
    from npa.orchestration.npa_workflow.runtime import SupervisedWaveFailure

    attempt.recovery_decision = "block_relaunch"
    attempt.supervisor_blocks_cancellation = True
    attempt.operator_remedy = sanitize_reason(reason)
    raise SupervisedWaveFailure(
        f"wave {attempt.key}: partial-launch recovery blocked: {reason}. "
        "Preserve the exact attempt and restore authoritative evidence before resuming."
    ) from cause


def _expected_identity(executor: Any, steps: Any, attempt: Any) -> AttemptIdentity:
    from npa.orchestration.npa_workflow.runtime import (
        _image_identity,
        _source_identity,
        _workflow_identity,
    )

    return AttemptIdentity(
        runtime="skypilot",
        run_id=executor.run_id,
        attempt=attempt.attempt,
        logical_attempt_id=logical_launch_identity(
            executor.options.project or "default",
            executor.run_id,
            attempt.key,
            str(attempt.attempt),
            ",".join(attempt.states),
        ),
        provider_job_id=attempt.job_id,
        provider_job_name=executor._job_name(
            steps, group=attempt.group, attempt=attempt
        ),
        workflow_sha256=_workflow_identity(executor.spec),
        source_sha256=_source_identity(),
        image_digest=_image_identity(executor.render_options),
    )


def _require_proof(executor: Any, steps: Any, attempt: Any) -> AttemptIdentity:
    proof = attempt.partial_launch
    identity = _identity(attempt, executor.run_id)
    expected = _expected_identity(executor, steps, attempt)
    if not (
        executor._submitter is None
        and executor.options.supervise_pending
        and executor.ledger.store is not None
        and proof.get("schema") == 1
        and proof.get("wave_key") == attempt.key
        and proof.get("parent") == identity.to_dict() == expected.to_dict()
        and proof.get("cause") in _TRANSIENT_CAUSES
        and attempt.error_category == proof.get("cause")
        and proof.get("workload_observable_at_failure") is False
        and all(
            (identity.workflow_sha256, identity.source_sha256, identity.image_digest)
        )
        and proof.get("rendered_wave_sha256")
        and not attempt.observations
        and attempt.outputs == [dict(item) for step in steps for item in step.outputs]
    ):
        _blocked(
            attempt, "recorded SDK proof or immutable attempt identity does not match"
        )
    return identity


def _exact_evidence(executor: Any, attempt: Any) -> Any:
    try:
        evidence = executor._reconcile_exact(attempt.job_name, attempt.job_id)
    except Exception as exc:
        _blocked(attempt, f"exact job query failed: {sanitize_reason(exc)}", cause=exc)
    if not (
        evidence.outcome == "found"
        and evidence.job_id == attempt.job_id
        and getattr(evidence, "job_name", attempt.job_name) == attempt.job_name
    ):
        _blocked(attempt, "exact job ID/name cannot be authoritatively reconciled")
    return evidence


def _outputs(executor: Any, attempt: Any) -> Any:
    from npa.orchestration.npa_workflow.runtime import _declared_output_uri

    declared = tuple(_declared_output_uri(item) for item in attempt.outputs)
    if not declared or not all(declared):
        _blocked(attempt, "declared output identity is incomplete")
    return validate_declared_outputs(declared, executor._output_checker)


def _require_absent(executor: Any, attempt: Any) -> Any:
    outputs = _outputs(executor, attempt)
    if not outputs.all_absent:
        _blocked(attempt, "declared outputs are present, partial, or unavailable")
    return outputs


def _terminal_evidence(executor: Any, attempt: Any) -> Any:
    from npa.orchestration.npa_workflow.runtime import is_terminal

    evidence = _exact_evidence(executor, attempt)
    status = str(evidence.status).upper()
    if not is_terminal(status):
        if status != "PENDING" or evidence.workload_observable:
            _blocked(
                attempt,
                "the reserved row gained workload evidence or has unknown state",
            )
        _require_absent(executor, attempt)
        attempt.partial_launch["outputs_absent_before_cancel"] = utc_now()
        executor.ledger.record(attempt)
        state, error = executor._cancel(attempt.job_id, attempt.job_name)
        attempt.cancellation_state, attempt.cancellation_error = state, error
        executor.ledger.record(attempt)
        if state != "verified":
            _blocked(attempt, "exact cancellation did not verify a terminal state")
        evidence = _exact_evidence(executor, attempt)
    # Cancellation can race with success. Retain the provider's actual outcome,
    # never the abort helper's normalized CANCELLED label.
    attempt.sky_status = str(evidence.status).upper()
    if (
        attempt.sky_status not in {"CANCELLED", "SUCCEEDED"}
        or not evidence.workload_observable
    ):
        _blocked(
            attempt, "actual terminal outcome is not verified cancellation or success"
        )
    if attempt.sky_status == "CANCELLED" and not (
        attempt.cancellation_state == "verified"
        and attempt.partial_launch.get("outputs_absent_before_cancel")
    ):
        _blocked(
            attempt,
            "recorded exact cancellation and its pre-cancel output check are missing",
        )
    executor.ledger.record(attempt)
    return evidence


def _render_original(executor: Any, steps: Any, attempt: Any) -> str:
    from npa.orchestration.npa_workflow.runtime import render_skypilot_steps_yaml

    rendered = render_skypilot_steps_yaml(
        executor.spec,
        steps,
        run_id=executor.run_id,
        options=replace(
            executor.render_options,
            execution_attempt_id=attempt.logical_launch_id,
            execution_fence_sequence=attempt.scheduler_fence_sequence,
            execution_fence_attempt=attempt.attempt,
        ),
        execution="parallel" if attempt.kind == "parallel" else "serial",
        name=attempt.job_name,
    )
    if (
        hashlib.sha256(rendered.encode()).hexdigest()
        != attempt.partial_launch["rendered_wave_sha256"]
    ):
        _blocked(attempt, "rendered task differs from the failed SDK-gated attempt")
    return rendered


def _refresh_preflight(executor: Any, steps: Any, attempt: Any) -> None:
    from npa.orchestration.skypilot.workflow import _refresh_workflow_preflight

    executor._preflight_by_attempt.pop((attempt.key, attempt.attempt), None)
    rendered = _render_original(executor, steps, attempt)
    with tempfile.TemporaryDirectory(
        prefix="npa-wave-recovery-preflight-"
    ) as directory:
        path = Path(directory) / "workflow.yaml"
        path.write_text(rendered, encoding="utf-8")
        path.chmod(0o600)
        if executor.options.pre_submit_hook is not None:
            executor.options.pre_submit_hook(path)
        if (
            hashlib.sha256(path.read_bytes()).hexdigest()
            != attempt.partial_launch["rendered_wave_sha256"]
        ):
            _blocked(attempt, "preflight hook changed the immutable rendered task")
        digest = _refresh_workflow_preflight(
            path,
            attempt.job_name,
            config_path=executor.options.config_path,
            isolated_config_dir=executor.options.isolated_config_dir,
            controller_backend=executor.options.controller_backend,
            infra=executor.options.infra,
            sky_bin=executor.options.sky_bin or None,
            project=executor.options.project,
            extra_env=executor._wave_credentials(attempt),
        )
    executor._record_submit_preflight(
        attempt, digest, source="default_sdk_recovery_preflight"
    )


def _successor(executor: Any, steps: Any, attempt: Any) -> AttemptIdentity:
    next_attempt = replace(attempt, attempt=attempt.attempt + 1, job_id="")
    return _expected_identity(executor, steps, next_attempt)


def _reservation_events(executor: Any, attempt: Any) -> list[dict[str, Any]]:
    root = f"npa-workflow/supervisor/attempts/{attempt.logical_launch_id}/"
    events = []
    try:
        for key in executor.ledger.store.list_artifacts(root):
            if not key.startswith(root + "recovery_reserved-"):
                continue
            raw = executor.ledger.store.read_artifact(key)
            digest = hashlib.sha256(raw).hexdigest()
            if key != root + f"recovery_reserved-{digest}.json":
                _blocked(attempt, "durable reservation content hash changed")
            event = json.loads(raw)
            if (
                event.get("phase") != "recovery_reserved"
                or event.get("schema_version") != "npa.workflow.supervisor.v1"
            ):
                _blocked(attempt, "durable reservation schema changed")
            events.append(event)
    except Exception as exc:
        _blocked(
            attempt, "durable reservation evidence is unavailable or invalid", cause=exc
        )
    return events


def _matching_reservation(
    executor: Any, attempt: Any, successor: AttemptIdentity
) -> dict[str, Any]:
    parent = _identity(attempt, executor.run_id).to_dict()
    matches = []
    for event in _reservation_events(executor, attempt):
        identity = event.get("attempt_identity") or {}
        if not (
            identity == parent
            and event.get("new_attempt_identity") == successor.to_dict()
            and (event.get("recovery") or {}).get("relaunch_allowed") is True
            and (event.get("observation") or {}).get("state") == "cancelled"
            and (event.get("observation") or {}).get("reason_code")
            == attempt.partial_launch["cause"].upper()
            and (event.get("infrastructure_recovery_policy") or {}).get("used")
            == attempt.infrastructure_recovery_count
        ):
            _blocked(attempt, "conflicting durable recovery reservation")
        _require_event_preflight(attempt, executor.run_id, event)
        matches.append(event)
    return matches[-1] if matches else {}


def _require_event_preflight(attempt: Any, run_id: str, event: Any) -> None:
    preflight = event.get("preflight") or {}
    checks = preflight.get("checks") or {}
    required = (
        "exact_image_pull",
        "accelerator_resolution",
        "per_node_gpu_shape",
        "gang_capacity",
    )
    if not (
        preflight.get("relaunch_ready") is True
        and checks.get("credentials_access") == "<redacted>"
        and all(checks.get(name) == "pass" for name in required)
        and preflight.get("scope")
        == {
            "source": "default_sdk_recovery_preflight",
            "run_id": run_id,
            "wave_key": attempt.key,
            "attempt": attempt.attempt,
            "rendered_wave_sha256": attempt.partial_launch["rendered_wave_sha256"],
        }
    ):
        _blocked(attempt, "durable reservation lacks the exact SDK preflight proof")


class _TerminalPartialAdapter:
    runtime = "skypilot"
    deferred_launch = True

    def __init__(
        self, observation: BackendObservation, successor: AttemptIdentity
    ) -> None:
        self.observation, self.successor = observation, successor

    def observe(self, _identity: AttemptIdentity) -> BackendObservation:
        return self.observation

    def launch_recovery(
        self, _identity: AttemptIdentity, *, checkpoint: Any
    ) -> AttemptIdentity:
        return self.successor

    def cancel_exact(self, _identity: AttemptIdentity) -> Any:
        raise RuntimeError("terminal partial-launch supervision cannot cancel again")


def _reserve(
    executor: Any, steps: Any, attempt: Any, identity: AttemptIdentity
) -> None:
    from npa.orchestration.npa_workflow.runtime import SupervisedWaveFailure

    successor = _successor(executor, steps, attempt)
    existing = _matching_reservation(executor, attempt, successor)
    context = RecoveryContext(
        expected_workflow_sha256=identity.workflow_sha256,
        expected_source_sha256=identity.source_sha256,
        expected_image_digest=identity.image_digest,
        outputs=_require_absent(executor, attempt),
        preflight=executor._attempt_preflight(attempt),
        infrastructure_recoveries=attempt.infrastructure_recovery_count,
        max_infrastructure_recoveries=executor.options.max_infrastructure_recoveries,
    )
    observation = BackendObservation(
        BackendState.CANCELLED,
        reason_code=attempt.partial_launch["cause"].upper(),
        evidence={
            "source": "default_sdk_partial_launch",
            "terminal_status": attempt.sky_status,
        },
    )
    result = (
        WorkflowRunSupervisor(
            adapter=_TerminalPartialAdapter(observation, successor),
            ledger=SupervisorLedger(executor.ledger.store),
        ).reconcile(identity, context)
        if not existing
        else _recheck_reserved(existing, identity, observation, context)
    )
    recovery = result["recovery"]
    attempt.recovery_decision = recovery["action"]
    attempt.infrastructure_recovery_exhausted = (
        recovery["reason_code"] == "INFRASTRUCTURE_RECOVERY_EXHAUSTED"
    )
    if not recovery.get("relaunch_allowed"):
        _blocked(attempt, recovery["reason_code"])
    attempt.partial_launch["recovery_reservation"] = {
        "parent": identity.to_dict(),
        "successor": successor.to_dict(),
        "used": attempt.infrastructure_recovery_count + 1,
        "wave_key": attempt.key,
    }
    attempt.status, attempt.ended_at = "failed", utc_now()
    executor.ledger.record(attempt)
    raise SupervisedWaveFailure(
        "verified partial transport launch reserved one successor",
        relaunch_allowed=True,
    )


def _recheck_reserved(
    existing: Any, identity: Any, observation: Any, context: Any
) -> Any:
    from npa.orchestration.npa_workflow.supervisor import decide_recovery

    # A durable reservation is not a cached capacity/output decision. Revalidate
    # it in this driver while preserving its one deterministic successor identity.
    return {
        **existing,
        "recovery": decide_recovery(identity, observation, context).to_dict(),
    }


def _recover_partial_launch(executor: Any, steps: Any, attempt: Any) -> bool:
    if not attempt.partial_launch:
        return False
    identity = _require_proof(executor, steps, attempt)
    _terminal_evidence(executor, attempt)
    if attempt.sky_status == "SUCCEEDED":
        if not _outputs(executor, attempt).all_valid:
            _blocked(attempt, "terminal success has incomplete output evidence")
        attempt.status, attempt.ended_at = "succeeded", utc_now()
        attempt.recovery_decision = "reuse_completed_wave"
        executor.ledger.record(attempt)
        return True
    _require_absent(executor, attempt)
    try:
        _refresh_preflight(executor, steps, attempt)
    except Exception as exc:
        from npa.orchestration.npa_workflow.runtime import SupervisedWaveFailure

        if isinstance(exc, SupervisedWaveFailure):
            raise
        detail = sanitize_reason(exc)
        attempt.partial_launch["recovery_preflight_error"] = detail
        _blocked(
            attempt,
            f"fresh shared SDK execution preflight did not pass: {detail}",
            cause=exc,
        )
    _reserve(executor, steps, attempt, identity)
    return False


@dataclass
class _ReservedResume:
    attempt_number: int
    used: int
    reservation: dict[str, Any]
    existing_attempt: Any = None


def _check_consumption(executor: Any, steps: Any, attempt: Any) -> None:
    reservation = attempt.recovery_reservation
    if not reservation:
        return
    if executor.ledger.store is None or executor._submitter is not None:
        _blocked(
            attempt, "reserved recovery requires the durable store and default SDK"
        )
    successor = reservation.get("successor") or {}
    expected = replace(_expected_identity(executor, steps, attempt), provider_job_id="")
    if not (
        reservation.get("wave_key") == attempt.key
        and successor == expected.to_dict()
        and reservation.get("used") == attempt.infrastructure_recovery_count
        and successor.get("attempt")
        == (reservation.get("parent") or {}).get("attempt", 0) + 1
        and attempt.outputs == [dict(item) for step in steps for item in step.outputs]
    ):
        _blocked(attempt, "reserved successor identity or recovery accounting changed")
    _require_reservation_event(executor, steps, attempt, expected)


def _require_reservation_event(
    executor: Any, steps: Any, attempt: Any, successor: AttemptIdentity
) -> None:
    parent_identity = attempt.recovery_reservation.get("parent") or {}
    records = [
        record
        for record in executor.ledger.state.waves
        if record.get("key") == attempt.key
        and record.get("logical_launch_id") == parent_identity.get("logical_attempt_id")
    ]
    if len(records) != 1:
        _blocked(attempt, "the exact reserved parent attempt is missing or ambiguous")
    parent = executor._attempt_from_record(
        records[0], steps=steps, kind=attempt.kind, group=attempt.group
    )
    try:
        if _require_proof(executor, steps, parent).to_dict() != parent_identity:
            _blocked(attempt, "reserved parent identity changed")
        event = _matching_reservation(executor, parent, successor)
    except Exception as exc:
        _blocked(
            attempt,
            "durable reservation evidence is unavailable or conflicting",
            cause=exc,
        )
    if (
        not event
        or attempt.recovery_reservation["used"]
        != parent.infrastructure_recovery_count + 1
    ):
        _blocked(
            attempt, "durable reservation is missing or recovery accounting changed"
        )


def _adopt_reserved(executor: Any, steps: Any, attempt: Any, evidence: Any) -> Any:
    from npa.orchestration.npa_workflow.runtime import (
        NpaWorkflowError,
        is_terminal,
        is_terminal_ok,
    )

    attempt.job_id, attempt.sky_status = evidence.job_id, str(evidence.status).upper()
    attempt.status, attempt.adopted = "running", True
    executor.attempts.append(attempt)
    executor.ledger.record(attempt)
    try:
        if not is_terminal(attempt.sky_status):
            attempt.sky_status = executor._poll(
                attempt.job_id, attempt, observe_tasks=len(steps) > 1
            )
        if not is_terminal_ok(attempt.sky_status):
            raise NpaWorkflowError(
                f"reserved successor reached terminal status {attempt.sky_status}"
            )
        executor._require_outputs(attempt.outputs, key=attempt.key)
    except BaseException as exc:
        executor._abort_wave(attempt, exc)
        if not isinstance(exc, Exception):
            raise
        return attempt
    attempt.status, attempt.ended_at = "succeeded", utc_now()
    attempt.tasks = executor._timeline(attempt.job_id)
    executor.ledger.record(attempt)
    return attempt


def _resume_reserved_successor(executor: Any, steps: Any, attempt: Any) -> Any:
    _check_consumption(executor, steps, attempt)
    evidence = executor._reconcile_exact(attempt.job_name, attempt.job_id)
    if (
        evidence.outcome == "absent"
        and attempt.launch_sequence == 0
        and not attempt.job_id
    ):
        _require_absent(executor, attempt)
        return _ReservedResume(
            attempt.attempt,
            attempt.infrastructure_recovery_count,
            attempt.recovery_reservation,
            attempt,
        )
    if evidence.outcome == "found" and evidence.workload_observable:
        if (
            not evidence.job_id
            or (attempt.job_id and evidence.job_id != attempt.job_id)
            or getattr(evidence, "job_name", attempt.job_name) != attempt.job_name
        ):
            _blocked(attempt, "reserved successor exact job ID/name changed")
        return _adopt_reserved(executor, steps, attempt, evidence)
    _blocked(attempt, "reserved successor has unverified POST or partial-row evidence")


def _resume_launch_recovery(
    executor: Any, steps: Any, key: str, kind: str, group: str
) -> Any:
    from npa.orchestration.npa_workflow.runtime import SupervisedWaveFailure

    record = executor.ledger.latest_wave(key)
    if not record or record.get("status") == "succeeded":
        return None
    if not record.get("partial_launch") and not record.get("recovery_reservation"):
        return None
    attempt = executor._attempt_from_record(record, steps=steps, kind=kind, group=group)
    attempt.recovery_resumed = True
    try:
        if attempt.partial_launch:
            if _recover_partial_launch(executor, steps, attempt):
                executor.attempts.append(attempt)
                return attempt
        else:
            return _resume_reserved_successor(executor, steps, attempt)
    except SupervisedWaveFailure as exc:
        if exc.relaunch_allowed:
            reservation = attempt.partial_launch["recovery_reservation"]
            return _ReservedResume(
                reservation["successor"]["attempt"], reservation["used"], reservation
            )
        executor._abort_wave(attempt, exc)
        executor.attempts.append(attempt)
        return attempt
    return None
