"""Fail-closed Kubernetes managed-job launch transactions.

SkyPilot 0.12.2 does not make ``jobs launch --name`` idempotent.  This module
therefore owns the boundary around that call: prove the selected Kubernetes API
is stable, serialize one logical launch locally, and reconcile structured queue
evidence before deciding whether another POST is safe.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
import fcntl
import hashlib
import os
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
import time
from typing import Any

from npa.orchestration.skypilot.workflow_state import redact_text


READINESS_SCHEMA_VERSION = "npa.skypilot.kubernetes-readiness.v1"
LAUNCH_SCHEMA_VERSION = "npa.skypilot.launch-transaction.v1"

# Product defaults live here so one-shot and runtime submissions cannot drift.
DEFAULT_REQUIRED_SUCCESSES = 3
DEFAULT_STABILITY_WINDOW_SECONDS = 10.0
DEFAULT_SAMPLE_INTERVAL_SECONDS = 5.0
DEFAULT_READINESS_DEADLINE_SECONDS = 120.0
DEFAULT_RECOVERY_DEADLINE_SECONDS = 180.0
DEFAULT_BACKOFF_INITIAL_SECONDS = 2.0
DEFAULT_BACKOFF_CAP_SECONDS = 20.0
DEFAULT_BACKOFF_MULTIPLIER = 2.0
DEFAULT_BACKOFF_JITTER_RATIO = 0.20


class FailureCategory(str, Enum):
    """Stable machine-readable failure taxonomy."""

    NONE = "none"
    KUBERNETES_TRANSPORT = "kubernetes_transport"
    KUBERNETES_RATE_LIMIT = "kubernetes_rate_limit"
    KUBERNETES_SERVER = "kubernetes_server"
    AUTH = "auth"
    RBAC = "rbac"
    CONTEXT = "context"
    IDENTITY = "identity"
    CERTIFICATE = "certificate"
    CONFIG = "config"
    CAPACITY = "capacity"
    WORKLOAD = "workload"
    SCHEMA = "schema"
    INTERRUPTED = "interrupted"
    UNKNOWN = "unknown"


class EvidenceState(str, Enum):
    READY = "ready"
    TRANSIENT_UNAVAILABLE = "transient_unavailable"
    TERMINAL = "terminal"
    AMBIGUOUS = "ambiguous"


class ControllerState(str, Enum):
    UP = "up"
    STOPPED = "stopped"
    ABSENT = "absent"
    UNHEALTHY = "unhealthy"
    AMBIGUOUS = "ambiguous"


class ReconciliationState(str, Enum):
    FOUND = "found"
    ABSENT = "absent"
    UNAVAILABLE = "unavailable"
    AMBIGUOUS = "ambiguous"


class LaunchState(str, Enum):
    SUBMITTED = "submitted"
    ADOPTED = "adopted"
    TERMINAL_FAILURE = "terminal_failure"
    TRANSIENT_API_FAILURE = "transient_api_failure"
    INDETERMINATE = "indeterminate"
    INTERRUPTED = "interrupted"


@dataclass(frozen=True)
class CommandEvidence:
    """Redacted command evidence; never contains kubeconfig contents or secrets."""

    argv: tuple[str, ...]
    returncode: int
    stdout: str = ""
    stderr: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "argv": list(self.argv),
            "returncode": self.returncode,
            "stdout": redact_text(self.stdout)[:1000],
            "stderr": redact_text(self.stderr)[:1000],
        }


@dataclass(frozen=True)
class ProbeObservation:
    state: EvidenceState
    category: FailureCategory = FailureCategory.NONE
    observed_at: str = ""
    monotonic_at: float = 0.0
    message: str = ""
    evidence: CommandEvidence | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "state": self.state.value,
            "category": self.category.value,
            "observed_at": self.observed_at,
            "monotonic_at": self.monotonic_at,
            "message": redact_text(self.message)[:1000],
        }
        if self.evidence is not None:
            payload["evidence"] = self.evidence.to_dict()
        return payload


@dataclass(frozen=True)
class StabilityPolicy:
    required_successes: int = DEFAULT_REQUIRED_SUCCESSES
    window_seconds: float = DEFAULT_STABILITY_WINDOW_SECONDS
    sample_interval_seconds: float = DEFAULT_SAMPLE_INTERVAL_SECONDS
    deadline_seconds: float = DEFAULT_READINESS_DEADLINE_SECONDS

    def __post_init__(self) -> None:
        if self.required_successes < 2:
            raise ValueError("Kubernetes API stability requires at least two successes")
        if min(
            self.window_seconds,
            self.sample_interval_seconds,
            self.deadline_seconds,
        ) < 0:
            raise ValueError("Kubernetes API stability durations must be non-negative")


@dataclass
class StabilityResult:
    state: EvidenceState
    category: FailureCategory
    samples: list[ProbeObservation] = field(default_factory=list)
    consecutive_successes: int = 0
    stable_for_seconds: float = 0.0
    error: str = ""

    @property
    def ready(self) -> bool:
        return self.state is EvidenceState.READY

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": READINESS_SCHEMA_VERSION,
            "state": self.state.value,
            "category": self.category.value,
            "sample_count": len(self.samples),
            "consecutive_successes": self.consecutive_successes,
            "stable_for_seconds": self.stable_for_seconds,
            "error": redact_text(self.error)[:1000],
            "samples": [sample.to_dict() for sample in self.samples],
        }


@dataclass(frozen=True)
class ReconciliationEvidence:
    state: ReconciliationState
    job_id: str = ""
    status: str = ""
    workload_observable: bool = False
    workload_evidence: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "job_id": self.job_id,
            "status": self.status,
            "workload_observable": self.workload_observable,
            "workload_evidence": redact_text(self.workload_evidence)[:1000],
            "error": redact_text(self.error)[:1000],
        }


@dataclass(frozen=True)
class RecoveryPolicy:
    deadline_seconds: float = DEFAULT_RECOVERY_DEADLINE_SECONDS
    initial_backoff_seconds: float = DEFAULT_BACKOFF_INITIAL_SECONDS
    cap_backoff_seconds: float = DEFAULT_BACKOFF_CAP_SECONDS
    multiplier: float = DEFAULT_BACKOFF_MULTIPLIER
    jitter_ratio: float = DEFAULT_BACKOFF_JITTER_RATIO

    def delay(self, sequence: int, *, random_value: float) -> float:
        raw = min(
            self.cap_backoff_seconds,
            self.initial_backoff_seconds * self.multiplier ** max(sequence - 1, 0),
        )
        centered = max(0.0, min(1.0, random_value)) * 2.0 - 1.0
        return max(0.0, min(self.cap_backoff_seconds, raw * (1 + centered * self.jitter_ratio)))


@dataclass
class LaunchTransactionResult:
    state: LaunchState
    logical_launch_id: str
    job_id: str = ""
    launch_sequence: int = 0
    category: FailureCategory = FailureCategory.NONE
    readiness: list[dict[str, Any]] = field(default_factory=list)
    reconciliations: list[dict[str, Any]] = field(default_factory=list)
    primary_error: str = ""
    reconciliation_error: str = ""
    recovery_decision: str = ""
    operator_remedy: str = ""
    existence: str = "indeterminate"
    controller: dict[str, str] = field(default_factory=dict)
    launch_result: Any = field(default=None, repr=False)

    @property
    def ok(self) -> bool:
        return self.state in {LaunchState.SUBMITTED, LaunchState.ADOPTED}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": LAUNCH_SCHEMA_VERSION,
            "state": self.state.value,
            "logical_launch_id": self.logical_launch_id,
            "job_id": self.job_id,
            "launch_sequence": self.launch_sequence,
            "category": self.category.value,
            "readiness": list(self.readiness),
            "reconciliations": list(self.reconciliations),
            "primary_error": redact_text(self.primary_error)[:2000],
            "reconciliation_error": redact_text(self.reconciliation_error)[:2000],
            "recovery_decision": self.recovery_decision,
            "operator_remedy": self.operator_remedy,
            "existence": self.existence,
            "controller": dict(self.controller),
        }


class LaunchTransactionError(RuntimeError):
    def __init__(self, message: str, result: LaunchTransactionResult) -> None:
        super().__init__(message)
        self.result = result


_TERMINAL_PATTERNS: tuple[tuple[FailureCategory, tuple[str, ...]], ...] = (
    (FailureCategory.IDENTITY, ("identity mismatch", "wrong context", "belongs to")),
    (FailureCategory.CONTEXT, ("context does not exist", "context not found", "not found in kubeconfig", "invalid context", "invalid kube-context", "no current-context")),
    (FailureCategory.RBAC, ("forbidden", "permission denied", "cannot list", "cannot get resource")),
    (FailureCategory.AUTH, ("unauthorized", "authentication required", "authentication failed", "credentials expired", "invalid bearer token", "exec plugin")),
    (FailureCategory.CERTIFICATE, ("certificate signed by unknown authority", "certificate has expired", "x509:")),
    (FailureCategory.CONFIG, ("invalid kubeconfig", "failed to load kubeconfig", "no such file or directory", "executable file not found")),
    (FailureCategory.SCHEMA, ("invalid pod_config", "validation error", "invalid yaml", "schema")),
    (FailureCategory.CAPACITY, ("insufficient", "quota", "failed_prechecks", "no resource", "unschedulable")),
    (FailureCategory.WORKLOAD, ("imagepull", "errimagepull", "failed_setup", "workload failed")),
)
_TRANSIENT_PATTERNS: tuple[tuple[FailureCategory, tuple[str, ...]], ...] = (
    (FailureCategory.KUBERNETES_RATE_LIMIT, ("too many requests", "status code 429", "http 429")),
    (FailureCategory.KUBERNETES_SERVER, ("status code 500", "status code 502", "status code 503", "status code 504", "internal server error", "service unavailable", "bad gateway", "gateway timeout")),
    (FailureCategory.KUBERNETES_TRANSPORT, ("connection refused", "connection reset", "connection aborted", "unexpected eof", "eof", "tls handshake timeout", "temporary failure in name resolution", "no route to host", "network is unreachable", "i/o timeout", "dial tcp", "server closed idle connection")),
)


def classify_failure(
    *, phase: str, stdout: str = "", stderr: str = "", exception: BaseException | None = None
) -> tuple[EvidenceState, FailureCategory]:
    """Classify only Kubernetes controller-bound transport errors as transient."""

    detail = "\n".join((stdout, stderr, str(exception or ""))).lower()
    for category, patterns in _TERMINAL_PATTERNS:
        if any(pattern in detail for pattern in patterns):
            return EvidenceState.TERMINAL, category
    transient_phase = phase in {"readiness", "controller", "launch", "reconciliation"}
    if transient_phase:
        for category, patterns in _TRANSIENT_PATTERNS:
            if any(pattern in detail for pattern in patterns):
                return EvidenceState.TRANSIENT_UNAVAILABLE, category
    return EvidenceState.AMBIGUOUS, FailureCategory.UNKNOWN


def _utc_timestamp() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _redacted_argv(argv: Sequence[str]) -> tuple[str, ...]:
    redacted: list[str] = []
    hide_next = False
    for value in argv:
        if hide_next:
            redacted.append("<redacted-path>")
            hide_next = False
            continue
        redacted.append(str(value))
        if value in {"--kubeconfig", "--token", "--certificate-authority", "--client-key"}:
            hide_next = True
    return tuple(redacted)


class KubectlApiProbe:
    """Probe ``/readyz`` using SkyPilot's exact environment and selected context."""

    def __init__(
        self,
        *,
        env: Mapping[str, str],
        context: str,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        clock: Callable[[], float] = time.monotonic,
        timestamp: Callable[[], str] = _utc_timestamp,
        kubectl: str | None = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        self.env = dict(env)
        self.context = context.strip()
        self.runner = runner
        self.clock = clock
        self.timestamp = timestamp
        self.kubectl = kubectl or shutil.which("kubectl") or ""
        self.timeout_seconds = timeout_seconds

    def __call__(self) -> ProbeObservation:
        now = self.clock()
        observed = self.timestamp()
        if not self.kubectl:
            return ProbeObservation(
                EvidenceState.TERMINAL,
                FailureCategory.CONFIG,
                observed,
                now,
                "kubectl is required to verify Kubernetes API stability",
            )
        if not self.context:
            return ProbeObservation(
                EvidenceState.TERMINAL,
                FailureCategory.CONTEXT,
                observed,
                now,
                "the Kubernetes controller launch has no exact selected context",
            )
        argv = [self.kubectl, "--context", self.context, "get", "--raw=/readyz"]
        try:
            result = self.runner(
                argv,
                env=self.env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout_seconds,
                check=False,
            )
        except (KeyboardInterrupt, InterruptedError):
            raise
        except BaseException as exc:  # command boundary evidence
            state, category = classify_failure(phase="readiness", exception=exc)
            return ProbeObservation(state, category, observed, now, redact_text(str(exc)))
        evidence = CommandEvidence(
            _redacted_argv(argv), result.returncode, result.stdout, result.stderr
        )
        if result.returncode == 0 and str(result.stdout or "").strip().lower().startswith("ok"):
            return ProbeObservation(
                EvidenceState.READY,
                FailureCategory.NONE,
                observed,
                now,
                "Kubernetes API /readyz succeeded",
                evidence,
            )
        state, category = classify_failure(
            phase="readiness", stdout=result.stdout, stderr=result.stderr
        )
        return ProbeObservation(
            state,
            category,
            observed,
            now,
            "Kubernetes API /readyz did not provide ready evidence",
            evidence,
        )


def wait_for_api_stability(
    probe: Callable[[], ProbeObservation],
    *,
    policy: StabilityPolicy = StabilityPolicy(),
    clock: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
    progress: Callable[[str], None] | None = None,
) -> StabilityResult:
    """Require a consecutive streak spanning both a count and a time window."""

    started = clock()
    deadline = started + policy.deadline_seconds
    samples: list[ProbeObservation] = []
    streak = 0
    streak_started = 0.0
    while True:
        try:
            sample = probe()
        except (KeyboardInterrupt, InterruptedError) as exc:
            return StabilityResult(
                EvidenceState.TERMINAL,
                FailureCategory.INTERRUPTED,
                samples,
                streak,
                max(0.0, clock() - streak_started) if streak else 0.0,
                redact_text(str(exc) or "interrupted"),
            )
        samples.append(sample)
        if sample.state is EvidenceState.READY:
            if streak == 0:
                streak_started = sample.monotonic_at
            streak += 1
            stable_for = max(0.0, sample.monotonic_at - streak_started)
            if progress is not None:
                progress(
                    f"Kubernetes API stability {streak}/{policy.required_successes}; "
                    f"stable {stable_for:.1f}/{policy.window_seconds:.1f}s"
                )
            if streak >= policy.required_successes and stable_for >= policy.window_seconds:
                return StabilityResult(
                    EvidenceState.READY,
                    FailureCategory.NONE,
                    samples,
                    streak,
                    stable_for,
                )
        elif sample.state is EvidenceState.TRANSIENT_UNAVAILABLE:
            streak = 0
            streak_started = 0.0
            if progress is not None:
                progress(
                    "Kubernetes API stability streak reset after transient "
                    f"{sample.category.value}"
                )
        else:
            return StabilityResult(
                sample.state,
                sample.category,
                samples,
                streak,
                0.0,
                sample.message,
            )
        if clock() >= deadline:
            category = samples[-1].category if samples else FailureCategory.UNKNOWN
            return StabilityResult(
                EvidenceState.TRANSIENT_UNAVAILABLE,
                category,
                samples,
                streak,
                max(0.0, clock() - streak_started) if streak else 0.0,
                f"Kubernetes API did not remain stable within {policy.deadline_seconds:g}s",
            )
        try:
            sleeper(policy.sample_interval_seconds)
        except (KeyboardInterrupt, InterruptedError) as exc:
            return StabilityResult(
                EvidenceState.TERMINAL,
                FailureCategory.INTERRUPTED,
                samples,
                streak,
                0.0,
                redact_text(str(exc) or "interrupted"),
            )


def logical_launch_identity(*parts: str) -> str:
    """Return an opaque identity derived from the exact launch contract."""

    encoded = "\0".join(str(part) for part in parts).encode("utf-8")
    return "npa-launch-" + hashlib.sha256(encoded).hexdigest()[:32]


def _lock_root(root: Path | None) -> Path:
    if root is not None:
        return root
    runtime = os.environ.get("XDG_RUNTIME_DIR", "").strip()
    if runtime:
        return Path(runtime) / "npa" / "launch-locks"
    return Path(tempfile.gettempdir()) / f"npa-{os.getuid()}" / "launch-locks"


@contextmanager
def launch_identity_lock(identity: str, *, root: Path | None = None) -> Iterator[Path]:
    """Crash-safe owner-only advisory lock for one logical launch identity."""

    directory = _lock_root(root)
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    directory.chmod(0o700)
    path = directory / f"{identity}.lock"
    if path.is_symlink():
        raise LaunchTransactionError(
            "refusing symlinked launch lock",
            LaunchTransactionResult(
                LaunchState.INDETERMINATE,
                identity,
                category=FailureCategory.CONFIG,
                primary_error="launch lock is a symlink",
            ),
        )
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield path
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _raise_result(result: LaunchTransactionResult) -> None:
    detail = result.primary_error or result.reconciliation_error or result.state.value
    raise LaunchTransactionError(
        f"managed-job launch {result.state.value} ({result.category.value}): {detail}. "
        f"{result.operator_remedy}".strip(),
        result,
    )


def run_launch_transaction(
    *,
    logical_id: str,
    readiness: Callable[[], StabilityResult],
    launch: Callable[[], Any],
    reconcile: Callable[[], ReconciliationEvidence],
    classify_launch_error: Callable[[BaseException], tuple[EvidenceState, FailureCategory]],
    recovery_policy: RecoveryPolicy = RecoveryPolicy(),
    lock_root: Path | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
    random_source: Callable[[], float] = random.random,
    record: Callable[[dict[str, Any]], None] | None = None,
    progress: Callable[[str], None] | None = None,
) -> LaunchTransactionResult:
    """Launch exactly once unless structured evidence proves a retry safe."""

    transaction = LaunchTransactionResult(LaunchState.INDETERMINATE, logical_id)

    def checkpoint() -> None:
        if record is not None:
            record(transaction.to_dict())

    with launch_identity_lock(logical_id, root=lock_root):
        initial = reconcile()
        transaction.reconciliations.append(initial.to_dict())
        if initial.state is ReconciliationState.FOUND:
            transaction.existence = "found"
            transaction.job_id = initial.job_id
            if not initial.workload_observable:
                transaction.state = LaunchState.INDETERMINATE
                transaction.reconciliation_error = (
                    "the exact managed-job queue record has no scheduler or workload "
                    "observability evidence"
                )
                transaction.recovery_decision = "block_unobservable_existing_record"
                transaction.operator_remedy = (
                    "Treat this as a possible pre-submit phantom, not a healthy job. "
                    "Inspect the run with `npa workbench workflow status`, cancel the "
                    "exact stale run with `npa workbench workflow cancel`, repair the "
                    "shared controller with `npa skypilot cleanup-controller`, and "
                    "resume the same run ID."
                )
                checkpoint()
                _raise_result(transaction)
            transaction.state = LaunchState.ADOPTED
            transaction.job_id = initial.job_id
            transaction.recovery_decision = "adopt_existing"
            if progress is not None:
                progress(f"reconciliation adopted exact managed job {initial.job_id}")
            checkpoint()
            return transaction
        if initial.state is not ReconciliationState.ABSENT:
            transaction.existence = "indeterminate"
            transaction.reconciliation_error = initial.error
            transaction.recovery_decision = "block_indeterminate"
            transaction.operator_remedy = (
                "Restore exact `sky jobs queue --all --output json` access, then use "
                "`--resume-run` with the same run ID. Do not launch or cancel by name."
            )
            checkpoint()
            _raise_result(transaction)
        transaction.existence = "absent"

        started = clock()
        deadline = started + recovery_policy.deadline_seconds
        transient_sequence = 0
        while True:
            stable = readiness()
            transaction.readiness.append(stable.to_dict())
            if not stable.ready:
                transaction.category = stable.category
                transaction.primary_error = stable.error
                transaction.state = (
                    LaunchState.INTERRUPTED
                    if stable.category is FailureCategory.INTERRUPTED
                    else LaunchState.TERMINAL_FAILURE
                    if stable.state is EvidenceState.TERMINAL
                    else LaunchState.TRANSIENT_API_FAILURE
                    if stable.state is EvidenceState.TRANSIENT_UNAVAILABLE
                    else LaunchState.INDETERMINATE
                )
                transaction.recovery_decision = "readiness_blocked"
                transaction.operator_remedy = (
                    "Fix the exact kube context/auth/config evidence and resume the same run."
                    if stable.state is EvidenceState.TERMINAL
                    else "Wait for the selected Kubernetes API to stabilize, then resume the same run."
                )
                checkpoint()
                _raise_result(transaction)
            if transient_sequence and clock() >= deadline:
                transaction.state = LaunchState.TRANSIENT_API_FAILURE
                transaction.recovery_decision = (
                    "recovery_deadline_exhausted_verified_absent"
                )
                transaction.operator_remedy = (
                    "The exact job is verified absent. Resume the same run after "
                    "the control plane recovers."
                )
                checkpoint()
                _raise_result(transaction)
            checkpoint()

            transaction.launch_sequence += 1
            try:
                launch_result = launch()
            except (KeyboardInterrupt, InterruptedError) as exc:
                state, category = EvidenceState.TERMINAL, FailureCategory.INTERRUPTED
                primary = redact_text(str(exc) or "interrupted")
            except BaseException as exc:
                state, category = classify_launch_error(exc)
                primary = redact_text(str(exc))
            else:
                after_success = reconcile()
                transaction.reconciliations.append(after_success.to_dict())
                # `sky jobs launch --async` returns after the API request is
                # accepted, before the managed-job row is necessarily visible.
                # Reconcile the exact logical name under the existing finite
                # launch-transaction deadline; never interpret temporary
                # absence as permission for a second provider submission.
                reconciliation_sequence = 0
                while (
                    after_success.state is ReconciliationState.ABSENT
                    and clock() < deadline
                ):
                    reconciliation_sequence += 1
                    delay = recovery_policy.delay(
                        reconciliation_sequence,
                        random_value=random_source(),
                    )
                    if progress is not None:
                        progress(
                            "launch accepted; waiting for exact managed-job observability"
                        )
                    sleeper(delay)
                    after_success = reconcile()
                    transaction.reconciliations.append(after_success.to_dict())
                if after_success.state is ReconciliationState.FOUND:
                    transaction.existence = "found"
                    transaction.state = LaunchState.SUBMITTED
                    transaction.job_id = after_success.job_id
                    transaction.launch_result = launch_result
                    transaction.recovery_decision = "submitted_and_reconciled"
                    if progress is not None:
                        progress(
                            f"launch sequence {transaction.launch_sequence} reconciled "
                            f"managed job {after_success.job_id}"
                        )
                    checkpoint()
                    return transaction
                transaction.state = LaunchState.INDETERMINATE
                transaction.existence = "indeterminate"
                transaction.reconciliation_error = after_success.error or (
                    "launch returned success but the exact managed job was not visible"
                )
                transaction.recovery_decision = "block_after_uncertain_success"
                transaction.operator_remedy = (
                    "Do not retry or cancel by name. Restore queue access and resume the same run "
                    "so the exact job can be reconciled."
                )
                checkpoint()
                _raise_result(transaction)

            transaction.primary_error = primary
            transaction.category = category
            after_failure = reconcile()
            transaction.reconciliations.append(after_failure.to_dict())
            if after_failure.state is ReconciliationState.FOUND:
                transaction.existence = "found"
                transaction.job_id = after_failure.job_id
                if not after_failure.workload_observable:
                    transaction.state = (
                        LaunchState.INTERRUPTED
                        if category is FailureCategory.INTERRUPTED
                        else LaunchState.TERMINAL_FAILURE
                        if state is EvidenceState.TERMINAL
                        else LaunchState.INDETERMINATE
                    )
                    transaction.reconciliation_error = (
                        "launch failed and the exact queue record never became "
                        "observable to the scheduler or workload runtime"
                    )
                    transaction.recovery_decision = (
                        "reject_unobservable_queue_record_after_launch_failure"
                    )
                    transaction.operator_remedy = (
                        "The queue row may have been allocated before controller file "
                        "sync. Do not poll it as a submitted workload. Inspect and cancel "
                        "the exact run with NPA commands, repair the shared controller, "
                        "then resume the same run ID."
                    )
                    checkpoint()
                    _raise_result(transaction)
                transaction.state = LaunchState.ADOPTED
                transaction.recovery_decision = "adopt_after_uncertain_launch"
                if progress is not None:
                    progress(
                        f"launch client failed, but reconciliation adopted exact managed job "
                        f"{after_failure.job_id}"
                    )
                checkpoint()
                return transaction
            if after_failure.state is not ReconciliationState.ABSENT:
                transaction.state = LaunchState.INDETERMINATE
                transaction.existence = "indeterminate"
                transaction.reconciliation_error = after_failure.error
                transaction.recovery_decision = "block_indeterminate"
                transaction.operator_remedy = (
                    "Do not retry or cancel by name. Restore queue access and resume the same run."
                )
                checkpoint()
                _raise_result(transaction)
            transaction.existence = "absent"
            if state is not EvidenceState.TRANSIENT_UNAVAILABLE:
                transaction.state = (
                    LaunchState.INTERRUPTED
                    if category is FailureCategory.INTERRUPTED
                    else LaunchState.TERMINAL_FAILURE
                )
                transaction.recovery_decision = "verified_absent_no_retry"
                transaction.operator_remedy = (
                    "The exact job is verified absent; fix the terminal launch error before resuming."
                )
                checkpoint()
                _raise_result(transaction)
            transient_sequence += 1
            now = clock()
            delay = recovery_policy.delay(
                transient_sequence, random_value=random_source()
            )
            if now + delay > deadline:
                transaction.state = LaunchState.TRANSIENT_API_FAILURE
                transaction.recovery_decision = "recovery_deadline_exhausted_verified_absent"
                transaction.operator_remedy = (
                    "The exact job is verified absent. Resume the same run after the control plane recovers."
                )
                checkpoint()
                _raise_result(transaction)
            if progress is not None:
                progress(
                    f"transient {category.value}; exact job absence verified; "
                    f"rechecking API stability before launch sequence {transaction.launch_sequence + 1}"
                )
            checkpoint()
            try:
                sleeper(delay)
            except (KeyboardInterrupt, InterruptedError) as exc:
                transaction.state = LaunchState.INTERRUPTED
                transaction.category = FailureCategory.INTERRUPTED
                transaction.primary_error = redact_text(str(exc) or "interrupted")
                transaction.recovery_decision = "interrupted_verified_absent"
                checkpoint()
                _raise_result(transaction)
