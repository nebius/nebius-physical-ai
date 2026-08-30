"""Shared durable supervision loop for production Serverless Jobs call sites."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import time
from typing import Any

from npa.orchestration.npa_workflow.supervisor import (
    AttemptIdentity,
    CheckpointValidation,
    PreflightEvidence,
    RecoveryAction,
    RecoveryContext,
    ServerlessSupervisorAdapter,
    SupervisorLedger,
    WorkflowRunSupervisor,
    validate_declared_outputs,
)


class ServerlessSupervisionError(RuntimeError):
    """A fail-closed terminal decision from the shared supervisor."""


@dataclass(frozen=True)
class ServerlessSupervisionConfig:
    expected_workflow_sha256: str
    expected_source_sha256: str
    expected_image_digest: str
    declared_outputs: tuple[str, ...]
    max_infrastructure_recoveries: int = 1
    checkpoint: CheckpointValidation = field(default_factory=CheckpointValidation)
    preflight: PreflightEvidence = field(default_factory=PreflightEvidence)
    poll_interval_seconds: float = 30.0
    wait_ceiling_seconds: float = 3600.0


def supervise_serverless_job(
    *,
    adapter: ServerlessSupervisorAdapter,
    ledger: SupervisorLedger,
    identity: AttemptIdentity,
    config: ServerlessSupervisionConfig,
    output_checker: Callable[[str], bool],
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[AttemptIdentity, Any]:
    """Observe, recover, and validate one exact Serverless Job until terminal.

    Provider re-attempts are created only by ``ServerlessSupervisorAdapter``.
    Payload failures never consume or inherit the infrastructure recovery policy.
    """

    if config.max_infrastructure_recoveries < 0:
        raise ValueError("max_infrastructure_recoveries must be non-negative")
    started = clock()
    recoveries = max(0, identity.attempt - 1)
    supervisor = WorkflowRunSupervisor(adapter=adapter, ledger=ledger)
    current = identity
    while True:
        outputs = validate_declared_outputs(config.declared_outputs, output_checker)
        result = supervisor.reconcile(
            current,
            RecoveryContext(
                expected_workflow_sha256=config.expected_workflow_sha256,
                expected_source_sha256=config.expected_source_sha256,
                expected_image_digest=config.expected_image_digest,
                outputs=outputs,
                preflight=config.preflight,
                checkpoint=config.checkpoint,
                infrastructure_recoveries=recoveries,
                max_infrastructure_recoveries=(config.max_infrastructure_recoveries),
            ),
        )
        recovery = result.get("recovery") or {}
        action = RecoveryAction(str(recovery.get("action") or "block_relaunch"))
        if action is RecoveryAction.ADOPT_EXACT_ATTEMPT:
            if (
                config.wait_ceiling_seconds > 0
                and clock() - started >= config.wait_ceiling_seconds
            ):
                raise TimeoutError(
                    "Serverless Job did not reach a terminal state within the wait ceiling"
                )
            sleeper(max(0.0, config.poll_interval_seconds))
            continue
        if action is RecoveryAction.REUSE_COMPLETED_WAVE:
            return current, adapter.client.get_job(
                current.provider_job_id, adapter.spec.project_id
            )
        next_identity = result.get("new_attempt_identity")
        if isinstance(next_identity, dict):
            current = AttemptIdentity(**next_identity)
            recoveries += 1
            continue
        reason = str(recovery.get("reason_code") or "UNCLASSIFIED")
        remediation = str(recovery.get("remediation") or "")
        raise ServerlessSupervisionError(f"{reason}: {remediation}".strip())


__all__ = [
    "ServerlessSupervisionConfig",
    "ServerlessSupervisionError",
    "supervise_serverless_job",
]
