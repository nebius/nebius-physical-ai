from __future__ import annotations

from dataclasses import replace
import json
from typing import Any

import pytest

from npa.clients.serverless import EndpointNotFoundError, JobInfo
from npa.orchestration.npa_workflow.run_state import RunStateStore
from npa.orchestration.npa_workflow.supervisor import (
    ArtifactValidation,
    AttemptIdentity,
    BackendObservation,
    BackendState,
    CheckpointValidation,
    FailureClass,
    PreflightEvidence,
    RecoveryAction,
    RecoveryContext,
    ServerlessRecoverySpec,
    ServerlessSupervisorAdapter,
    SkyPilotSupervisorAdapter,
    SupervisorLedger,
    WorkflowRunSupervisor,
    classify_observation,
    decide_recovery,
    validate_declared_outputs,
)


class MemoryStore(RunStateStore):
    def __init__(self, objects: dict[str, bytes] | None = None) -> None:
        self.objects = objects if objects is not None else {}
        super().__init__(
            bucket="unit-bucket",
            prefix="runs/run-1",
            reader=self._read_object,
            writer=self._write_object,
            artifact_lister=self._list_objects,
        )

    def _read_object(self, _bucket: str, key: str) -> bytes:
        if key not in self.objects:
            raise FileNotFoundError(key)
        return self.objects[key]

    def _write_object(self, _bucket: str, key: str, body: bytes) -> None:
        self.objects[key] = body

    def _list_objects(self, _bucket: str, prefix: str) -> list[str]:
        return [key for key in self.objects if key.startswith(prefix)]


def identity(**overrides: Any) -> AttemptIdentity:
    values: dict[str, Any] = {
        "runtime": "skypilot",
        "run_id": "run-1",
        "attempt": 1,
        "logical_attempt_id": "wave:1",
        "provider_job_id": "job-1",
        "provider_job_name": "run-1-wave-a1",
        "workflow_sha256": "w" * 64,
        "source_sha256": "s" * 64,
        "image_digest": "i" * 64,
        "checkpoint_prefix": "s3://unit-bucket/runs/run-1/checkpoints/",
    }
    values.update(overrides)
    return AttemptIdentity(**values)


def ready_preflight() -> PreflightEvidence:
    return PreflightEvidence(
        checks={name: "pass" for name in PreflightEvidence.REQUIRED_RELAUNCH_CHECKS},
        observed_at="2026-08-30T00:00:00Z",
    )


def context(
    *,
    outputs: ArtifactValidation | None = None,
    checkpoint: CheckpointValidation | None = None,
    preflight: PreflightEvidence | None = None,
) -> RecoveryContext:
    return RecoveryContext(
        expected_workflow_sha256="w" * 64,
        expected_source_sha256="s" * 64,
        expected_image_digest="i" * 64,
        outputs=outputs
        or ArtifactValidation(
            "absent",
            declared=("s3://unit-bucket/runs/run-1/result.json",),
            missing=("s3://unit-bucket/runs/run-1/result.json",),
        ),
        preflight=preflight or ready_preflight(),
        checkpoint=checkpoint or CheckpointValidation(),
    )


class RecordingAdapter:
    runtime = "skypilot"

    def __init__(self, observation: BackendObservation) -> None:
        self.observation = observation
        self.cancelled: list[str] = []
        self.launched: list[str] = []

    def observe(self, _identity: AttemptIdentity) -> BackendObservation:
        return self.observation

    def cancel_exact(self, attempt: AttemptIdentity) -> dict[str, Any]:
        self.cancelled.append(attempt.provider_job_id)
        return {
            "provider_job_id": attempt.provider_job_id,
            "status": "cancelled",
            "exact": True,
        }

    def launch_recovery(
        self,
        attempt: AttemptIdentity,
        *,
        checkpoint: CheckpointValidation,
    ) -> AttemptIdentity:
        self.launched.append(attempt.logical_attempt_id)
        return replace(
            attempt,
            attempt=attempt.attempt + 1,
            logical_attempt_id="wave:2",
            provider_job_id="job-2",
            provider_job_name="run-1-wave-a2",
        )


@pytest.mark.parametrize(
    ("reason", "failure_class"),
    [
        ("IMAGE_PULL_AUTH", FailureClass.ACTIONABLE_CONFIGURATION),
        ("MISSING_SECRET", FailureClass.ACTIONABLE_CONFIGURATION),
        ("ACCELERATOR_MISMATCH", FailureClass.ACTIONABLE_CONFIGURATION),
        ("NODE_NOT_READY", FailureClass.TRANSIENT_INFRASTRUCTURE),
        ("CONTAINER_CRASH", FailureClass.PAYLOAD),
        ("SOMETHING_NEW", FailureClass.UNKNOWN),
    ],
)
def test_failure_taxonomy_is_stable(reason: str, failure_class: FailureClass) -> None:
    assert (
        classify_observation(
            BackendObservation(BackendState.FAILED, reason_code=reason)
        )
        is failure_class
    )


def test_configuration_stall_cancels_only_exact_attempt_and_terminalizes() -> None:
    adapter = RecordingAdapter(
        BackendObservation(
            BackendState.QUEUED,
            reason_code="MISSING_CONFIGMAP",
            message="ConfigMap reference is missing",
        )
    )
    ledger = SupervisorLedger(MemoryStore())

    result = WorkflowRunSupervisor(adapter=adapter, ledger=ledger).reconcile(
        identity(), context()
    )

    assert result["recovery"]["action"] == "cancel_and_terminalize"
    assert result["classification"] == "actionable_configuration"
    assert adapter.cancelled == ["job-1"]
    assert adapter.launched == []
    assert [event["phase"] for event in ledger.events()] == [
        "decision",
        "cancellation",
    ]


def test_transient_live_attempt_is_cancelled_verified_then_relaunched() -> None:
    adapter = RecordingAdapter(
        BackendObservation(
            BackendState.QUEUED,
            reason_code="NODE_NOT_READY",
        )
    )
    ledger = SupervisorLedger(MemoryStore())

    result = WorkflowRunSupervisor(adapter=adapter, ledger=ledger).reconcile(
        identity(), context()
    )

    assert result["phase"] == "launch"
    assert result["new_attempt_identity"]["run_id"] == "run-1"
    assert result["new_attempt_identity"]["attempt"] == 2
    assert adapter.cancelled == ["job-1"]
    assert adapter.launched == ["wave:1"]


def test_unverified_cancellation_blocks_duplicate_launch() -> None:
    adapter = RecordingAdapter(
        BackendObservation(
            BackendState.QUEUED,
            reason_code="NODE_NOT_READY",
        )
    )
    adapter.cancel_exact = lambda _attempt: {  # type: ignore[method-assign]
        "provider_job_id": "job-1",
        "status": "cancelling",
        "exact": True,
    }

    result = WorkflowRunSupervisor(
        adapter=adapter, ledger=SupervisorLedger(MemoryStore())
    ).reconcile(identity(), context())

    assert result["recovery"]["action"] == "block_relaunch"
    assert result["recovery"]["reason_code"] == "CANCELLATION_UNVERIFIED"
    assert adapter.launched == []


@pytest.mark.parametrize(
    "observation",
    [
        BackendObservation(
            BackendState.AMBIGUOUS,
            reason_code="CONTROLLER_UNAVAILABLE",
            exact_identity=False,
        ),
        BackendObservation(
            BackendState.RUNNING,
            reason_code="",
            exact_identity=False,
        ),
    ],
)
def test_ambiguous_identity_blocks_relaunch_and_cancellation(
    observation: BackendObservation,
) -> None:
    adapter = RecordingAdapter(observation)
    result = WorkflowRunSupervisor(
        adapter=adapter, ledger=SupervisorLedger(MemoryStore())
    ).reconcile(identity(), context())

    assert result["recovery"]["action"] == "block_relaunch"
    assert adapter.cancelled == []
    assert adapter.launched == []


def test_valid_completed_outputs_reuse_wave_without_launch() -> None:
    decision = decide_recovery(
        identity(),
        BackendObservation(BackendState.SUCCEEDED),
        context(
            outputs=ArtifactValidation(
                "valid",
                declared=("s3://unit-bucket/result",),
                valid=("s3://unit-bucket/result",),
            )
        ),
    )
    assert decision.action is RecoveryAction.REUSE_COMPLETED_WAVE


def test_partial_output_evidence_blocks_transient_relaunch() -> None:
    decision = decide_recovery(
        identity(),
        BackendObservation(
            BackendState.ABSENT, reason_code="PROVIDER_INTERRUPTION"
        ),
        context(
            outputs=ArtifactValidation(
                "partial",
                declared=("s3://unit-bucket/a", "s3://unit-bucket/b"),
                valid=("s3://unit-bucket/a",),
                missing=("s3://unit-bucket/b",),
            )
        ),
    )
    assert decision.action is RecoveryAction.BLOCK_RELAUNCH
    assert decision.reason_code == "OUTPUT_EVIDENCE_AMBIGUOUS"


def test_infrastructure_recovery_policy_exhaustion_is_terminal() -> None:
    decision = decide_recovery(
        identity(),
        BackendObservation(BackendState.QUEUED, reason_code="NODE_NOT_READY"),
        replace(
            context(),
            infrastructure_recoveries=2,
            max_infrastructure_recoveries=2,
        ),
    )

    assert decision.action is RecoveryAction.CANCEL_AND_TERMINALIZE
    assert decision.reason_code == "INFRASTRUCTURE_RECOVERY_EXHAUSTED"
    assert not decision.relaunch_allowed


def test_exhaustion_requires_verified_exact_cancellation() -> None:
    adapter = RecordingAdapter(
        BackendObservation(BackendState.QUEUED, reason_code="NODE_NOT_READY")
    )
    adapter.cancel_exact = lambda _attempt: {  # type: ignore[method-assign]
        "provider_job_id": "job-1",
        "status": "cancelling",
        "exact": True,
    }

    result = WorkflowRunSupervisor(
        adapter=adapter, ledger=SupervisorLedger(MemoryStore())
    ).reconcile(
        identity(),
        replace(
            context(),
            infrastructure_recoveries=1,
            max_infrastructure_recoveries=1,
        ),
    )

    assert result["recovery"]["action"] == "block_relaunch"
    assert result["recovery"]["reason_code"] == "CANCELLATION_UNVERIFIED"
    assert adapter.launched == []


def test_checkpoint_recovery_requires_real_loader_and_valid_checkpoint() -> None:
    unsupported = decide_recovery(
        identity(),
        BackendObservation(
            BackendState.ABSENT, reason_code="PROVIDER_INTERRUPTION"
        ),
        context(checkpoint=CheckpointValidation(requested=True)),
    )
    supported = decide_recovery(
        identity(),
        BackendObservation(
            BackendState.ABSENT, reason_code="PROVIDER_INTERRUPTION"
        ),
        context(
            checkpoint=CheckpointValidation(
                requested=True,
                supported=True,
                valid=True,
                uri="s3://unit-bucket/checkpoints/step-10/",
                loader="tool.load_checkpoint.v1",
            )
        ),
    )

    assert unsupported.reason_code == "CHECKPOINT_RECOVERY_UNSUPPORTED"
    assert not unsupported.relaunch_allowed
    assert supported.action is RecoveryAction.RESUME_APPLICATION_CHECKPOINT
    assert supported.checkpoint_mode == "application_checkpoint"


def test_skypilot_adapter_prefers_typed_event_over_unknown_pod_diagnostic() -> None:
    from npa.orchestration.skypilot.job_blockers import JobBlockerReport, PodBlocker
    from npa.orchestration.skypilot.workflow import ManagedJobEvidence

    adapter = SkyPilotSupervisorAdapter(
        lookup=lambda _name, *, job_id="": ManagedJobEvidence(
            "found", job_id=job_id, status="PENDING"
        ),
        blocker_inspector=lambda **_kwargs: JobBlockerReport(
            blockers=[
                PodBlocker(
                    pod="exact-pod",
                    phase="Pending",
                    reason="SchedulingGated",
                    reason_code="PENDING_UNKNOWN",
                ),
                PodBlocker(
                    pod="exact-pod",
                    phase="Pending",
                    reason="FailedScheduling",
                    message="quota exhausted",
                    reason_code="CAPACITY_OR_QUOTA",
                ),
            ]
        ),
    )

    observation = adapter.observe(identity())

    assert observation.reason_code == "CAPACITY_OR_QUOTA"
    assert classify_observation(observation) is FailureClass.TRANSIENT_INFRASTRUCTURE


def test_process_restart_reads_content_addressed_immutable_history() -> None:
    objects: dict[str, bytes] = {}
    first = SupervisorLedger(MemoryStore(objects))
    uri = first.record(
        {
            "recorded_at": "2026-08-30T00:00:00Z",
            "phase": "decision",
            "attempt_identity": identity().to_dict(),
            "classification": "unknown",
        }
    )
    restarted = SupervisorLedger(MemoryStore(objects))

    assert restarted.latest()["classification"] == "unknown"  # type: ignore[index]
    assert uri.endswith(".json")
    assert restarted.record(restarted.latest()) == uri  # type: ignore[arg-type]
    assert len(objects) == 1


def test_immutable_artifact_rejects_conflicting_bytes() -> None:
    store = MemoryStore()
    store.write_immutable_artifact("evidence/event.json", b"one")
    with pytest.raises(ValueError, match="already differs"):
        store.write_immutable_artifact("evidence/event.json", b"two")


def test_output_validation_is_fail_closed_on_storage_error() -> None:
    def checker(uri: str) -> bool:
        if uri.endswith("b"):
            raise PermissionError("access denied for credential=do-not-store")
        return True

    result = validate_declared_outputs(
        ["s3://unit-bucket/a", "s3://unit-bucket/b"], checker
    )
    assert result.status == "indeterminate"
    assert result.valid == ("s3://unit-bucket/a",)


class FakeServerlessClient:
    def __init__(self) -> None:
        self.jobs: dict[str, JobInfo] = {}
        self.create_calls: list[str] = []

    def get_job(self, job_id_or_name: str, project_id: str) -> JobInfo:
        for job in self.jobs.values():
            if job_id_or_name in {job.id, job.name}:
                return job
        raise EndpointNotFoundError(
            "not found", project_id=project_id, endpoint_id=job_id_or_name
        )

    def cancel_job(self, job_id: str, project_id: str) -> JobInfo:
        job = self.get_job(job_id, project_id)
        cancelled = replace(job, status="cancelled")
        self.jobs[job.id] = cancelled
        return cancelled

    def create_job(self, **kwargs: Any) -> JobInfo:
        self.create_calls.append(kwargs["name"])
        job = JobInfo(
            id=f"provider-{len(self.create_calls) + 1}",
            name=kwargs["name"],
            project_id=kwargs["project_id"],
            status="queued",
        )
        self.jobs[job.id] = job
        return job


def serverless_adapter(client: FakeServerlessClient) -> ServerlessSupervisorAdapter:
    return ServerlessSupervisorAdapter(
        ServerlessRecoverySpec(
            project_id="project-role",
            image="registry.example/tool@sha256:" + "a" * 64,
            command="run",
            gpu_type="gpu-platform",
            output_path="s3://unit-bucket/runs/run-1/",
        ),
        client=client,
    )


def test_serverless_recovery_creates_new_attempt_under_same_run() -> None:
    client = FakeServerlessClient()
    adapter = serverless_adapter(client)
    old = identity(
        runtime="serverless",
        provider_job_id="missing-provider-job",
        provider_job_name="run-1-wave-a1",
    )

    result = WorkflowRunSupervisor(
        adapter=adapter, ledger=SupervisorLedger(MemoryStore())
    ).reconcile(old, context())

    assert result["new_attempt_identity"]["run_id"] == "run-1"
    assert result["new_attempt_identity"]["attempt"] == 2
    assert client.create_calls == ["run-1-wave-a2"]


def test_serverless_restart_adopts_deterministic_attempt_without_duplicate() -> None:
    client = FakeServerlessClient()
    client.jobs["provider-2"] = JobInfo(
        id="provider-2",
        name="run-1-wave-a2",
        project_id="project-role",
        status="queued",
    )
    adapter = serverless_adapter(client)
    old = identity(
        runtime="serverless",
        provider_job_id="missing-provider-job",
        provider_job_name="run-1-wave-a1",
    )

    recovered = adapter.launch_recovery(old, checkpoint=CheckpointValidation())

    assert recovered.provider_job_id == "provider-2"
    assert client.create_calls == []


def test_serverless_image_auth_failure_is_configuration_not_payload() -> None:
    client = FakeServerlessClient()
    client.jobs["provider-1"] = JobInfo(
        id="provider-1",
        name="run-1-wave-a1",
        project_id="project-role",
        status="failed",
        log_tail="container image pull denied: unauthorized",
    )
    observation = serverless_adapter(client).observe(
        identity(
            runtime="serverless",
            provider_job_id="provider-1",
            provider_job_name="run-1-wave-a1",
        )
    )

    assert observation.reason_code == "IMAGE_PULL_AUTH"
    assert classify_observation(observation) is FailureClass.ACTIONABLE_CONFIGURATION


def test_supervisor_event_redacts_secret_shaped_fields() -> None:
    store = MemoryStore()
    ledger = SupervisorLedger(store)
    ledger.record(
        {
            "recorded_at": "2026-08-30T00:00:00Z",
            "phase": "decision",
            "attempt_identity": identity().to_dict(),
            "evidence": {
                "access_token": "sensitive-value",
                "message": "authorization failed",
            },
        }
    )
    serialized = json.dumps(ledger.latest())
    assert "sensitive-value" not in serialized
    assert "<redacted>" in serialized


def test_failure_record_has_machine_readable_fields() -> None:
    decision = decide_recovery(
        identity(),
        BackendObservation(
            BackendState.QUEUED, reason_code="IMAGE_REFERENCE_INVALID"
        ),
        context(),
    )
    assert decision.to_dict() == {
        "action": "cancel_and_terminalize",
        "failure_class": "actionable_configuration",
        "reason_code": "IMAGE_REFERENCE_INVALID",
        "remediation": decision.remediation,
        "relaunch_allowed": False,
        "checkpoint_mode": "wave_restart",
    }
