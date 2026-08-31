"""Durable, runtime-neutral supervision for workflow execution attempts.

SkyPilot remains the Kubernetes workflow engine.  This module observes and
reconciles the exact provider identity that the runtime already recorded; it is
not a scheduler and never runs inside a payload pod.  Every decision is written
as a content-addressed S3 event so a new supervisor process can resume from
object-storage evidence without private controller memory.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from typing import Any, Protocol

from npa.orchestration.npa_workflow.run_state import RunStateStore, utc_now
from npa.verification import sanitize_reason

SUPERVISOR_SCHEMA_VERSION = "npa.workflow.supervisor.v1"


class FailureClass(str, Enum):
    NONE = "none"
    ACTIONABLE_CONFIGURATION = "actionable_configuration"
    TRANSIENT_INFRASTRUCTURE = "transient_infrastructure"
    PAYLOAD = "payload"
    UNKNOWN = "unknown"


class BackendState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ABSENT = "absent"
    AMBIGUOUS = "ambiguous"
    UNKNOWN = "unknown"


class RecoveryAction(str, Enum):
    CONTINUE = "continue"
    ADOPT_EXACT_ATTEMPT = "adopt_exact_attempt"
    REUSE_COMPLETED_WAVE = "reuse_completed_wave"
    RELAUNCH_INCOMPLETE_WAVE = "relaunch_incomplete_wave"
    RESUME_APPLICATION_CHECKPOINT = "resume_application_checkpoint"
    CANCEL_AND_TERMINALIZE = "cancel_and_terminalize"
    TERMINALIZE = "terminalize"
    BLOCK_RELAUNCH = "block_relaunch"


CONFIGURATION_REASON_CODES = frozenset(
    {
        "IMAGE_PULL_AUTH",
        "IMAGE_NOT_FOUND",
        "IMAGE_PULL_FAILED",
        "IMAGE_REFERENCE_INVALID",
        "MISSING_SECRET",
        "MISSING_CONFIGMAP",
        "CREATE_CONTAINER_CONFIG_ERROR",
        "MALFORMED_POD_CONFIG",
        "ACCELERATOR_MISMATCH",
        "IMPOSSIBLE_GPU_SHAPE",
        "INVALID_ARGUMENT",
        "AUTHENTICATION",
        "AUTHORIZATION",
        "CHECKPOINT_INCOMPATIBLE",
    }
)
TRANSIENT_REASON_CODES = frozenset(
    {
        "CAPACITY_OR_QUOTA",
        "GANG_CAPACITY_UNAVAILABLE",
        "NODE_NOT_READY",
        "PREEMPTED",
        "PROVIDER_INTERRUPTION",
        "KUBERNETES_TRANSPORT",
        "KUBERNETES_RATE_LIMIT",
        "KUBERNETES_SERVER",
        "SERVERLESS_TRANSPORT",
        "SERVERLESS_CAPACITY",
        "CONTROLLER_UNAVAILABLE",
    }
)
PAYLOAD_REASON_CODES = frozenset(
    {
        "CONTAINER_CRASH",
        "INIT_CONTAINER_FAILED",
        "PAYLOAD_EXIT_NONZERO",
        "FAILED_RUNTIME",
        "DECLARED_OUTPUT_MISSING",
    }
)


@dataclass(frozen=True)
class AttemptIdentity:
    runtime: str
    run_id: str
    attempt: int
    logical_attempt_id: str
    provider_job_id: str = ""
    provider_job_name: str = ""
    workflow_sha256: str = ""
    source_sha256: str = ""
    image_digest: str = ""
    checkpoint_prefix: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtime": self.runtime,
            "run_id": self.run_id,
            "attempt": self.attempt,
            "logical_attempt_id": self.logical_attempt_id,
            "provider_job_id": self.provider_job_id,
            "provider_job_name": self.provider_job_name,
            "workflow_sha256": self.workflow_sha256,
            "source_sha256": self.source_sha256,
            "image_digest": self.image_digest,
            "checkpoint_prefix": self.checkpoint_prefix,
        }


@dataclass(frozen=True)
class BackendObservation:
    state: BackendState
    reason_code: str = ""
    message: str = ""
    exact_identity: bool = True
    workload_observable: bool = True
    observed_at: str = field(default_factory=utc_now)
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "reason_code": self.reason_code,
            "message": sanitize_reason(self.message),
            "exact_identity": self.exact_identity,
            "workload_observable": self.workload_observable,
            "observed_at": self.observed_at,
            "evidence": _sanitized_mapping(self.evidence),
        }


@dataclass(frozen=True)
class ArtifactValidation:
    status: str
    declared: tuple[str, ...] = ()
    valid: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    error: str = ""

    @property
    def all_valid(self) -> bool:
        return bool(self.declared) and self.status == "valid" and not self.missing

    @property
    def all_absent(self) -> bool:
        return bool(self.declared) and self.status == "absent" and not self.valid

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "declared": list(self.declared),
            "valid": list(self.valid),
            "missing": list(self.missing),
            "error": sanitize_reason(self.error),
        }


@dataclass(frozen=True)
class CheckpointValidation:
    requested: bool = False
    supported: bool = False
    valid: bool = False
    uri: str = ""
    loader: str = ""
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "supported": self.supported,
            "valid": self.valid,
            "uri": self.uri,
            "loader": self.loader,
            "detail": sanitize_reason(self.detail),
        }


@dataclass(frozen=True)
class PreflightEvidence:
    checks: Mapping[str, str] = field(default_factory=dict)
    observed_at: str = ""

    REQUIRED_RELAUNCH_CHECKS = frozenset(
        {
            "exact_image_pull",
            "credentials_access",
            "accelerator_resolution",
            "per_node_gpu_shape",
            "gang_capacity",
        }
    )

    @property
    def relaunch_ready(self) -> bool:
        return all(
            str(self.checks.get(name) or "").lower() in {"pass", "not_required"}
            for name in self.REQUIRED_RELAUNCH_CHECKS
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "checks": dict(sorted(self.checks.items())),
            "observed_at": self.observed_at,
            "relaunch_ready": self.relaunch_ready,
        }


@dataclass(frozen=True)
class RecoveryContext:
    expected_workflow_sha256: str
    expected_source_sha256: str
    expected_image_digest: str
    outputs: ArtifactValidation
    preflight: PreflightEvidence = field(default_factory=PreflightEvidence)
    checkpoint: CheckpointValidation = field(default_factory=CheckpointValidation)
    infrastructure_recoveries: int = 0
    max_infrastructure_recoveries: int = 1


@dataclass(frozen=True)
class RecoveryDecision:
    action: RecoveryAction
    failure_class: FailureClass
    reason_code: str
    remediation: str
    relaunch_allowed: bool = False
    checkpoint_mode: str = "wave_restart"

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "failure_class": self.failure_class.value,
            "reason_code": self.reason_code,
            "remediation": self.remediation,
            "relaunch_allowed": self.relaunch_allowed,
            "checkpoint_mode": self.checkpoint_mode,
        }


class RuntimeAdapter(Protocol):
    runtime: str

    def observe(self, identity: AttemptIdentity) -> BackendObservation: ...

    def cancel_exact(self, identity: AttemptIdentity) -> Mapping[str, Any]: ...

    def launch_recovery(
        self,
        identity: AttemptIdentity,
        *,
        checkpoint: CheckpointValidation,
    ) -> AttemptIdentity: ...


def classify_observation(observation: BackendObservation) -> FailureClass:
    code = observation.reason_code.strip().upper()
    if observation.state in {BackendState.AMBIGUOUS, BackendState.UNKNOWN}:
        return FailureClass.UNKNOWN
    if code in CONFIGURATION_REASON_CODES:
        return FailureClass.ACTIONABLE_CONFIGURATION
    if code in TRANSIENT_REASON_CODES:
        return FailureClass.TRANSIENT_INFRASTRUCTURE
    if code in PAYLOAD_REASON_CODES:
        return FailureClass.PAYLOAD
    if observation.state is BackendState.FAILED:
        return FailureClass.PAYLOAD if not code else FailureClass.UNKNOWN
    return FailureClass.NONE


def decide_recovery(
    identity: AttemptIdentity,
    observation: BackendObservation,
    context: RecoveryContext,
) -> RecoveryDecision:
    failure_class = classify_observation(observation)
    code = observation.reason_code.strip().upper() or "UNCLASSIFIED"
    if not observation.exact_identity or not observation.workload_observable:
        return RecoveryDecision(
            RecoveryAction.BLOCK_RELAUNCH,
            FailureClass.UNKNOWN,
            "AMBIGUOUS_ATTEMPT_IDENTITY",
            "Restore authoritative provider evidence for the exact recorded attempt; do not relaunch or cancel by name.",
        )
    if failure_class is FailureClass.ACTIONABLE_CONFIGURATION:
        action = (
            RecoveryAction.CANCEL_AND_TERMINALIZE
            if observation.state in {BackendState.QUEUED, BackendState.RUNNING}
            and identity.provider_job_id
            else RecoveryAction.TERMINALIZE
        )
        return RecoveryDecision(
            action,
            failure_class,
            code,
            _configuration_remediation(code),
        )
    if failure_class is FailureClass.PAYLOAD:
        return RecoveryDecision(
            RecoveryAction.TERMINALIZE,
            failure_class,
            code,
            "Inspect the exact attempt logs and fix the payload; infrastructure retry is disabled.",
        )
    if observation.state is BackendState.SUCCEEDED:
        if context.outputs.all_valid:
            return RecoveryDecision(
                RecoveryAction.REUSE_COMPLETED_WAVE,
                FailureClass.NONE,
                "DECLARED_OUTPUTS_VALID",
                "Every declared S3 output was validated; reuse this completed wave.",
            )
        return RecoveryDecision(
            RecoveryAction.TERMINALIZE,
            FailureClass.PAYLOAD,
            "DECLARED_OUTPUT_MISSING",
            "The provider reports success but declared S3 outputs are incomplete; inspect the payload publication contract.",
        )
    if (
        observation.state in {BackendState.QUEUED, BackendState.RUNNING}
        and failure_class is FailureClass.NONE
    ):
        return RecoveryDecision(
            RecoveryAction.ADOPT_EXACT_ATTEMPT,
            failure_class,
            code if failure_class is not FailureClass.NONE else "ATTEMPT_LIVE",
            "Continue observing the exact recorded provider job.",
        )
    if observation.state in {BackendState.AMBIGUOUS, BackendState.UNKNOWN}:
        return RecoveryDecision(
            RecoveryAction.BLOCK_RELAUNCH,
            FailureClass.UNKNOWN,
            code,
            "Backend evidence is unknown or ambiguous; restore authoritative access before relaunching.",
        )
    if failure_class is not FailureClass.TRANSIENT_INFRASTRUCTURE:
        return RecoveryDecision(
            RecoveryAction.BLOCK_RELAUNCH,
            FailureClass.UNKNOWN,
            code,
            "The failure is not proven transient; inspect exact backend evidence before recovery.",
        )
    if not _immutable_identity_matches(identity, context):
        return RecoveryDecision(
            RecoveryAction.BLOCK_RELAUNCH,
            FailureClass.UNKNOWN,
            "IMMUTABLE_IDENTITY_MISMATCH",
            "Restore the recorded workflow, source, and image identities or start a new NPA run ID.",
        )
    if context.infrastructure_recoveries >= context.max_infrastructure_recoveries:
        action = (
            RecoveryAction.CANCEL_AND_TERMINALIZE
            if observation.state in {BackendState.QUEUED, BackendState.RUNNING}
            and identity.provider_job_id
            else RecoveryAction.TERMINALIZE
        )
        return RecoveryDecision(
            action,
            failure_class,
            "INFRASTRUCTURE_RECOVERY_EXHAUSTED",
            "The finite infrastructure recovery policy is exhausted; cancel the exact live attempt when present, then inspect the durable attempt history before explicitly starting or resuming a run.",
        )
    if context.outputs.all_valid:
        return RecoveryDecision(
            RecoveryAction.REUSE_COMPLETED_WAVE,
            failure_class,
            "DECLARED_OUTPUTS_VALID",
            "Every declared S3 output is valid; no provider relaunch is needed.",
        )
    if not context.outputs.all_absent:
        return RecoveryDecision(
            RecoveryAction.BLOCK_RELAUNCH,
            FailureClass.UNKNOWN,
            "OUTPUT_EVIDENCE_AMBIGUOUS",
            "Declared output evidence is partial or unavailable; preserve it and restore authoritative S3 access before recovery.",
        )
    if not context.preflight.relaunch_ready:
        return RecoveryDecision(
            RecoveryAction.BLOCK_RELAUNCH,
            FailureClass.ACTIONABLE_CONFIGURATION,
            "PREFLIGHT_NOT_READY",
            "Re-run exact image, credential/access, accelerator, per-node shape, and gang-capacity preflight before recovery.",
        )
    checkpoint = context.checkpoint
    if checkpoint.requested:
        if not checkpoint.supported or not checkpoint.valid or not checkpoint.loader:
            return RecoveryDecision(
                RecoveryAction.BLOCK_RELAUNCH,
                FailureClass.ACTIONABLE_CONFIGURATION,
                "CHECKPOINT_RECOVERY_UNSUPPORTED",
                "This tool has no verified compatible checkpoint loader; recover only at a completed-wave boundary.",
            )
        return RecoveryDecision(
            RecoveryAction.RESUME_APPLICATION_CHECKPOINT,
            failure_class,
            code,
            "Launch a new provider attempt under the same NPA run ID using the verified application checkpoint.",
            relaunch_allowed=True,
            checkpoint_mode="application_checkpoint",
        )
    return RecoveryDecision(
        RecoveryAction.RELAUNCH_INCOMPLETE_WAVE,
        failure_class,
        code,
        "Launch only the incomplete wave as a new provider attempt under the same NPA run ID.",
        relaunch_allowed=True,
        checkpoint_mode="wave_restart",
    )


class SupervisorLedger:
    """Content-addressed supervisor events stored under the run prefix."""

    def __init__(self, store: RunStateStore) -> None:
        self.store = store

    def record(self, payload: Mapping[str, Any]) -> str:
        document = {
            "schema_version": SUPERVISOR_SCHEMA_VERSION,
            **_sanitized_mapping(payload),
        }
        body = (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()
        digest = hashlib.sha256(body).hexdigest()
        identity = document.get("attempt_identity") or {}
        logical_id = _safe_component(str(identity.get("logical_attempt_id") or "attempt"))
        phase = _safe_component(str(document.get("phase") or "observation"))
        key = f"npa-workflow/supervisor/attempts/{logical_id}/{phase}-{digest}.json"
        return self.store.write_immutable_artifact(
            key, body, content_type="application/json"
        )

    def events(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for key in self.store.list_artifacts("npa-workflow/supervisor/attempts"):
            try:
                payload = json.loads(self.store.read_artifact(key))
            except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
                continue
            if isinstance(payload, dict) and payload.get("schema_version") == SUPERVISOR_SCHEMA_VERSION:
                result.append(payload)
        phase_order = {
            "decision": 0,
            "cancellation": 1,
            "recovery_reserved": 2,
            "launch": 2,
        }
        return sorted(
            result,
            key=lambda item: (
                str(item.get("recorded_at") or ""),
                int((item.get("attempt_identity") or {}).get("attempt") or 0),
                phase_order.get(str(item.get("phase") or ""), 99),
            ),
        )

    def latest(self) -> dict[str, Any] | None:
        events = self.events()
        return events[-1] if events else None


class WorkflowRunSupervisor:
    """Perform one fail-closed reconciliation pass for an exact attempt."""

    def __init__(self, *, adapter: RuntimeAdapter, ledger: SupervisorLedger) -> None:
        self.adapter = adapter
        self.ledger = ledger

    def reconcile(
        self,
        identity: AttemptIdentity,
        context: RecoveryContext,
    ) -> dict[str, Any]:
        if identity.runtime != self.adapter.runtime:
            raise ValueError("attempt runtime does not match supervisor adapter")
        observation = self.adapter.observe(identity)
        decision = decide_recovery(identity, observation, context)
        base: dict[str, Any] = {
            "recorded_at": utc_now(),
            "phase": "decision",
            "attempt_identity": identity.to_dict(),
            "observation": observation.to_dict(),
            "classification": decision.failure_class.value,
            "recovery": decision.to_dict(),
            "outputs": context.outputs.to_dict(),
            "checkpoint": context.checkpoint.to_dict(),
            "preflight": context.preflight.to_dict(),
            "infrastructure_recovery_policy": {
                "used": context.infrastructure_recoveries,
                "limit": context.max_infrastructure_recoveries,
                "exhausted": (
                    context.infrastructure_recoveries
                    >= context.max_infrastructure_recoveries
                ),
            },
        }
        base["event_uri"] = self.ledger.record(base)
        if decision.action is RecoveryAction.CANCEL_AND_TERMINALIZE:
            cancellation = _sanitized_mapping(self.adapter.cancel_exact(identity))
            result = {
                **base,
                "recorded_at": utc_now(),
                "phase": "cancellation",
                "cancellation": cancellation,
            }
            cancel_status = str(cancellation.get("status") or "").lower()
            if not bool(cancellation.get("exact")) or cancel_status not in {
                "cancelled",
                "canceled",
                "failed",
                "succeeded",
            }:
                blocked = RecoveryDecision(
                    RecoveryAction.BLOCK_RELAUNCH,
                    FailureClass.UNKNOWN,
                    "CANCELLATION_UNVERIFIED",
                    "Verify terminal state for the exact provider attempt before terminalizing recovery.",
                )
                result["classification"] = blocked.failure_class.value
                result["recovery"] = blocked.to_dict()
            result["event_uri"] = self.ledger.record(result)
            return result
        if decision.relaunch_allowed:
            if observation.state in {BackendState.QUEUED, BackendState.RUNNING}:
                cancellation = _sanitized_mapping(
                    self.adapter.cancel_exact(identity)
                )
                cancel_status = str(cancellation.get("status") or "").lower()
                cancel_exact = bool(cancellation.get("exact"))
                cancellation_event = {
                    **base,
                    "recorded_at": utc_now(),
                    "phase": "cancellation",
                    "cancellation": cancellation,
                }
                cancellation_event["event_uri"] = self.ledger.record(
                    cancellation_event
                )
                if not cancel_exact or cancel_status not in {
                    "cancelled",
                    "canceled",
                    "failed",
                    "succeeded",
                }:
                    blocked = RecoveryDecision(
                        RecoveryAction.BLOCK_RELAUNCH,
                        FailureClass.UNKNOWN,
                        "CANCELLATION_UNVERIFIED",
                        "Verify terminal state for the exact provider attempt before relaunching.",
                    )
                    cancellation_event["classification"] = blocked.failure_class.value
                    cancellation_event["recovery"] = blocked.to_dict()
                    return cancellation_event
            launched = self.adapter.launch_recovery(
                identity, checkpoint=context.checkpoint
            )
            deferred = bool(getattr(self.adapter, "deferred_launch", False))
            result = {
                **base,
                "recorded_at": utc_now(),
                "phase": "recovery_reserved" if deferred else "launch",
                "new_attempt_identity": launched.to_dict(),
            }
            result["event_uri"] = self.ledger.record(result)
            return result
        return base


class SkyPilotSupervisorAdapter:
    runtime = "skypilot"
    # SkyPilot provider creation remains inside the runtime's existing
    # crash-safe launch transaction. The adapter reserves the next immutable
    # identity; the outer wave loop crosses the provider boundary.
    deferred_launch = True

    def __init__(
        self,
        *,
        lookup: Callable[..., Any] | None = None,
        blocker_inspector: Callable[..., Any] | None = None,
        canceller: Callable[[AttemptIdentity], Mapping[str, Any]] | None = None,
        launcher: Callable[[AttemptIdentity, CheckpointValidation], AttemptIdentity] | None = None,
        context: str = "",
    ) -> None:
        self._lookup = lookup
        self._blockers = blocker_inspector
        self._canceller = canceller
        self._launcher = launcher
        self._context = context

    def observe(self, identity: AttemptIdentity) -> BackendObservation:
        from npa.orchestration.skypilot.workflow import lookup_managed_job

        lookup = self._lookup or lookup_managed_job
        try:
            evidence = lookup(
                identity.provider_job_name,
                job_id=identity.provider_job_id,
            )
        except Exception as exc:  # noqa: BLE001 - uncertainty blocks relaunch
            return BackendObservation(
                BackendState.AMBIGUOUS,
                reason_code="CONTROLLER_UNAVAILABLE",
                message=sanitize_reason(exc),
                exact_identity=False,
            )
        outcome = str(getattr(evidence, "outcome", "") or "").lower()
        if outcome == "absent":
            return BackendObservation(
                BackendState.ABSENT,
                reason_code="KUBERNETES_TRANSPORT",
                evidence={"lookup": "exact_absence"},
            )
        if outcome != "found":
            return BackendObservation(
                BackendState.AMBIGUOUS,
                reason_code="CONTROLLER_UNAVAILABLE",
                message=str(getattr(evidence, "error", "") or outcome),
                exact_identity=False,
            )
        observed_id = str(getattr(evidence, "job_id", "") or "")
        exact = bool(identity.provider_job_id and observed_id == identity.provider_job_id)
        observable = bool(getattr(evidence, "workload_observable", True))
        status = str(getattr(evidence, "status", "") or "UNKNOWN").upper()
        state = _skypilot_state(status)
        reason = ""
        message = ""
        blocker_payload: list[dict[str, Any]] = []
        if state is BackendState.QUEUED:
            from npa.orchestration.skypilot.job_blockers import inspect_job_blockers

            inspect = self._blockers or inspect_job_blockers
            report = inspect(job_id=observed_id, context=self._context)
            for blocker in getattr(report, "blockers", []) or []:
                blocker_payload.append(
                    {
                        "reason_code": str(getattr(blocker, "reason_code", "") or ""),
                        "reason": str(getattr(blocker, "reason", "") or ""),
                        "message": sanitize_reason(getattr(blocker, "message", "")),
                    }
                )
            if blocker_payload:
                typed_codes = (
                    CONFIGURATION_REASON_CODES
                    | TRANSIENT_REASON_CODES
                    | PAYLOAD_REASON_CODES
                )
                selected = next(
                    (
                        blocker
                        for blocker in blocker_payload
                        if blocker["reason_code"] in typed_codes
                    ),
                    blocker_payload[0],
                )
                reason = selected["reason_code"]
                message = selected["message"]
            elif getattr(report, "unready_nodes", None):
                reason = "NODE_NOT_READY"
            elif getattr(report, "error", ""):
                # The exact managed-job queue observation above is authoritative.
                # Failure of the optional pod diagnostic must not manufacture a
                # transient failure and trigger cancellation/relaunch.
                message = str(getattr(report, "error", ""))
        return BackendObservation(
            state,
            reason_code=reason,
            message=message,
            exact_identity=exact,
            workload_observable=observable,
            evidence={"status": status, "blockers": blocker_payload},
        )

    def cancel_exact(self, identity: AttemptIdentity) -> Mapping[str, Any]:
        if not identity.provider_job_id:
            raise ValueError("exact SkyPilot cancellation requires a recorded job ID")
        if self._canceller is None:
            raise RuntimeError("SkyPilot cancellation callback is required")
        return self._canceller(identity)

    def launch_recovery(
        self,
        identity: AttemptIdentity,
        *,
        checkpoint: CheckpointValidation,
    ) -> AttemptIdentity:
        if self._launcher is None:
            raise RuntimeError("SkyPilot recovery must use the runtime launch transaction")
        return self._launcher(identity, checkpoint)


@dataclass(frozen=True)
class ServerlessRecoverySpec:
    project_id: str = field(repr=False)
    image: str = ""
    command: str = ""
    gpu_type: str = ""
    gpu_count: int = 1
    output_path: str = ""
    preset: str = ""
    subnet_id: str = field(default="", repr=False)
    timeout: str = "1h"
    env: Mapping[str, str] = field(default_factory=dict, repr=False)
    secret_env: Mapping[str, str] = field(default_factory=dict, repr=False)


class ServerlessSupervisorAdapter:
    runtime = "serverless"

    def __init__(self, spec: ServerlessRecoverySpec, *, client: Any | None = None) -> None:
        if client is None:
            from npa.clients.serverless import ServerlessClient

            client = ServerlessClient()
        self.spec = spec
        self.client = client

    def observe(self, identity: AttemptIdentity) -> BackendObservation:
        from npa.clients.serverless import EndpointNotFoundError, ServerlessClientError

        if not identity.provider_job_id:
            return BackendObservation(
                BackendState.AMBIGUOUS,
                reason_code="AMBIGUOUS_ATTEMPT_IDENTITY",
                exact_identity=False,
            )
        try:
            job = self.client.get_job(identity.provider_job_id, self.spec.project_id)
        except EndpointNotFoundError:
            return BackendObservation(
                BackendState.ABSENT,
                reason_code="PROVIDER_INTERRUPTION",
                evidence={"lookup": "exact_absence"},
            )
        except ServerlessClientError as exc:
            return BackendObservation(
                BackendState.AMBIGUOUS,
                reason_code="SERVERLESS_TRANSPORT",
                message=sanitize_reason(exc),
                exact_identity=False,
            )
        exact = str(job.id or "") == identity.provider_job_id
        state = _serverless_state(str(job.status or ""))
        reason = ""
        detail = (
            f"{job.scheduling_state} {job.pending_reason} {job.log_tail}"
        ).lower()
        if state is BackendState.QUEUED and any(
            marker in detail for marker in ("capacity", "quota", "resource", "no gpu")
        ):
            reason = "SERVERLESS_CAPACITY"
        elif state is BackendState.FAILED:
            if "image" in detail and any(
                marker in detail
                for marker in ("unauthorized", "authentication", "denied", "403")
            ):
                reason = "IMAGE_PULL_AUTH"
            elif "image" in detail and any(
                marker in detail for marker in ("not found", "manifest unknown", "404")
            ):
                reason = "IMAGE_NOT_FOUND"
            elif "invalid image" in detail or "invalid reference" in detail:
                reason = "IMAGE_REFERENCE_INVALID"
            elif "preempt" in detail or "interrupted" in detail:
                reason = "PROVIDER_INTERRUPTION"
            else:
                reason = "PAYLOAD_EXIT_NONZERO"
        return BackendObservation(
            state,
            reason_code=reason,
            message=str(job.pending_reason or ""),
            exact_identity=exact,
            evidence={
                "status": job.status,
                "scheduling_state": job.scheduling_state,
                "queued_for_seconds": job.queued_for_seconds,
            },
        )

    def cancel_exact(self, identity: AttemptIdentity) -> Mapping[str, Any]:
        job = self.client.cancel_job(identity.provider_job_id, self.spec.project_id)
        return {"provider_job_id": job.id, "status": job.status, "exact": job.id == identity.provider_job_id}

    def launch_recovery(
        self,
        identity: AttemptIdentity,
        *,
        checkpoint: CheckpointValidation,
    ) -> AttemptIdentity:
        from npa.clients.serverless import EndpointNotFoundError

        attempt = identity.attempt + 1
        name = f"{identity.provider_job_name.rsplit('-a', 1)[0]}-a{attempt}"
        # Deterministic name lookup makes restart after a create timeout/crash an
        # adoption, not a duplicate provider submission.
        try:
            job = self.client.get_job(name, self.spec.project_id)
        except EndpointNotFoundError:
            env = dict(self.spec.env)
            if checkpoint.requested:
                env["NPA_CHECKPOINT_URI"] = checkpoint.uri
                env["NPA_CHECKPOINT_LOADER"] = checkpoint.loader
            job = self.client.create_job(
                project_id=self.spec.project_id,
                name=name,
                image=self.spec.image,
                command=self.spec.command,
                gpu_type=self.spec.gpu_type,
                gpu_count=self.spec.gpu_count,
                output_path=self.spec.output_path,
                extra_env=self.spec.secret_env,
                env=env,
                preset=self.spec.preset,
                timeout=self.spec.timeout,
                subnet_id=self.spec.subnet_id,
            )
        return AttemptIdentity(
            runtime=self.runtime,
            run_id=identity.run_id,
            attempt=attempt,
            logical_attempt_id=f"{identity.logical_attempt_id.rsplit(':', 1)[0]}:{attempt}",
            provider_job_id=str(job.id or ""),
            provider_job_name=name,
            workflow_sha256=identity.workflow_sha256,
            source_sha256=identity.source_sha256,
            image_digest=identity.image_digest,
            checkpoint_prefix=identity.checkpoint_prefix,
        )


def validate_declared_outputs(
    uris: Sequence[str], checker: Callable[[str], bool]
) -> ArtifactValidation:
    declared = tuple(str(uri or "").strip() for uri in uris if str(uri or "").strip())
    if not declared:
        return ArtifactValidation("indeterminate", error="no declared S3 outputs")
    valid: list[str] = []
    missing: list[str] = []
    try:
        for uri in declared:
            (valid if checker(uri) else missing).append(uri)
    except Exception as exc:  # noqa: BLE001 - storage uncertainty blocks relaunch
        return ArtifactValidation(
            "indeterminate",
            declared=declared,
            valid=tuple(valid),
            missing=tuple(missing),
            error=sanitize_reason(exc),
        )
    status = "valid" if len(valid) == len(declared) else "absent" if not valid else "partial"
    return ArtifactValidation(status, declared, tuple(valid), tuple(missing))


def _immutable_identity_matches(identity: AttemptIdentity, context: RecoveryContext) -> bool:
    pairs = (
        (identity.workflow_sha256, context.expected_workflow_sha256),
        (identity.source_sha256, context.expected_source_sha256),
        (identity.image_digest, context.expected_image_digest),
    )
    return all(recorded and expected and recorded == expected for recorded, expected in pairs)


def _configuration_remediation(code: str) -> str:
    if code.startswith("IMAGE_"):
        return "Fix the exact image reference or exact-registry pull credentials, then start a new run or explicitly resume after preflight."
    if code in {"MISSING_SECRET", "MISSING_CONFIGMAP", "CREATE_CONTAINER_CONFIG_ERROR"}:
        return "Create the referenced Secret or ConfigMap and verify the rendered pod config before resuming."
    if code in {"ACCELERATOR_MISMATCH", "IMPOSSIBLE_GPU_SHAPE"}:
        return "Resolve the advertised accelerator and request a per-node GPU shape the target can satisfy."
    return "Correct the recorded configuration and re-run preflight; automatic retry is disabled."


def _skypilot_state(status: str) -> BackendState:
    if status in {"SUCCEEDED", "SUCCESS", "COMPLETED", "DONE"}:
        return BackendState.SUCCEEDED
    if status in {"CANCELLED", "CANCELED", "STOPPED"}:
        return BackendState.CANCELLED
    if status.startswith("FAILED"):
        return BackendState.FAILED
    if status in {"RUNNING", "RECOVERING"}:
        return BackendState.RUNNING
    if status in {"PENDING", "STARTING", "SUBMITTED", "RETRYING"}:
        return BackendState.QUEUED
    return BackendState.UNKNOWN


def _serverless_state(status: str) -> BackendState:
    normalized = status.strip().lower()
    if normalized == "succeeded":
        return BackendState.SUCCEEDED
    if normalized == "failed":
        return BackendState.FAILED
    if normalized == "cancelled":
        return BackendState.CANCELLED
    if normalized == "running":
        return BackendState.RUNNING
    if normalized == "queued":
        return BackendState.QUEUED
    return BackendState.UNKNOWN


def _safe_component(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "-_" else "-" for char in value)
    return cleaned.strip("-_")[:120] or "event"


def _sanitized_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    def clean(item: Any, key: str = "") -> Any:
        lowered = key.lower()
        if any(marker in lowered for marker in ("secret", "password", "token", "credential")):
            return "<redacted>"
        if isinstance(item, Mapping):
            return {str(k): clean(v, str(k)) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(v, key) for v in item]
        if isinstance(item, str):
            return sanitize_reason(item)
        return item

    return clean(value)


__all__ = [
    "ArtifactValidation",
    "AttemptIdentity",
    "BackendObservation",
    "BackendState",
    "CheckpointValidation",
    "FailureClass",
    "PreflightEvidence",
    "RecoveryAction",
    "RecoveryContext",
    "RecoveryDecision",
    "ServerlessRecoverySpec",
    "ServerlessSupervisorAdapter",
    "SkyPilotSupervisorAdapter",
    "SupervisorLedger",
    "WorkflowRunSupervisor",
    "classify_observation",
    "decide_recovery",
    "validate_declared_outputs",
]
