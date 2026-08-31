"""Hermetic controller-launch transaction and Kubernetes stability tests."""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path
from typing import Any

import pytest

from npa.orchestration.skypilot.launch_transaction import (
    EvidenceState,
    FailureCategory,
    KubectlApiProbe,
    LaunchState,
    LaunchTransactionError,
    ProbeObservation,
    ReconciliationEvidence,
    ReconciliationState,
    RecoveryPolicy,
    StabilityPolicy,
    StabilityResult,
    classify_failure,
    logical_launch_identity,
    run_launch_transaction,
    wait_for_api_stability,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class SequenceProbe:
    def __init__(self, clock: FakeClock, states: list[EvidenceState]) -> None:
        self.clock = clock
        self.states = list(states)
        self.calls = 0

    def __call__(self) -> ProbeObservation:
        state = self.states[min(self.calls, len(self.states) - 1)]
        self.calls += 1
        category = (
            FailureCategory.NONE
            if state is EvidenceState.READY
            else FailureCategory.KUBERNETES_TRANSPORT
        )
        return ProbeObservation(
            state,
            category,
            f"t{self.clock.now:g}",
            self.clock.now,
            state.value,
        )


def _stable() -> StabilityResult:
    return StabilityResult(EvidenceState.READY, FailureCategory.NONE)


def _transient(exc: BaseException) -> tuple[EvidenceState, FailureCategory]:
    return classify_failure(phase="launch", exception=exc)


def test_consecutive_readiness_requires_count_and_full_window() -> None:
    clock = FakeClock()
    probe = SequenceProbe(clock, [EvidenceState.READY])
    result = wait_for_api_stability(
        probe,
        policy=StabilityPolicy(
            required_successes=3,
            window_seconds=10,
            sample_interval_seconds=5,
            deadline_seconds=20,
        ),
        clock=clock,
        sleeper=clock.sleep,
    )
    assert result.ready
    assert result.consecutive_successes == 3
    assert result.stable_for_seconds == 10
    assert len(result.samples) == 3


def test_readiness_streak_resets_on_transient_failure() -> None:
    clock = FakeClock()
    probe = SequenceProbe(
        clock,
        [
            EvidenceState.READY,
            EvidenceState.READY,
            EvidenceState.TRANSIENT_UNAVAILABLE,
            EvidenceState.READY,
            EvidenceState.READY,
            EvidenceState.READY,
        ],
    )
    result = wait_for_api_stability(
        probe,
        policy=StabilityPolicy(3, 4, 2, 20),
        clock=clock,
        sleeper=clock.sleep,
    )
    assert result.ready
    assert len(result.samples) == 6
    assert result.stable_for_seconds == 4


def test_readiness_exact_threshold_does_not_round_window() -> None:
    clock = FakeClock()
    result = wait_for_api_stability(
        SequenceProbe(clock, [EvidenceState.READY]),
        policy=StabilityPolicy(3, 2.5, 1.25, 5),
        clock=clock,
        sleeper=clock.sleep,
    )
    assert result.ready
    assert result.stable_for_seconds == pytest.approx(2.5)


def test_readiness_deadline_is_bounded() -> None:
    clock = FakeClock()
    result = wait_for_api_stability(
        SequenceProbe(clock, [EvidenceState.TRANSIENT_UNAVAILABLE]),
        policy=StabilityPolicy(3, 2, 1, 3),
        clock=clock,
        sleeper=clock.sleep,
    )
    assert result.state is EvidenceState.TRANSIENT_UNAVAILABLE
    assert clock.now == 3
    assert "within 3s" in result.error


def test_terminal_readiness_fails_immediately_without_sleep() -> None:
    clock = FakeClock()
    probe = SequenceProbe(clock, [EvidenceState.TERMINAL])
    result = wait_for_api_stability(
        probe,
        policy=StabilityPolicy(3, 2, 1, 30),
        clock=clock,
        sleeper=clock.sleep,
    )
    assert result.state is EvidenceState.TERMINAL
    assert clock.sleeps == []


def test_readiness_interruption_is_prompt_and_typed() -> None:
    clock = FakeClock()

    def interrupt(_seconds: float) -> None:
        raise KeyboardInterrupt

    result = wait_for_api_stability(
        SequenceProbe(clock, [EvidenceState.READY]),
        policy=StabilityPolicy(3, 2, 1, 30),
        clock=clock,
        sleeper=interrupt,
    )
    assert result.category is FailureCategory.INTERRUPTED
    assert len(result.samples) == 1


@pytest.mark.parametrize(
    ("phase", "detail", "state", "category"),
    [
        ("launch", "connect: connection refused", EvidenceState.TRANSIENT_UNAVAILABLE, FailureCategory.KUBERNETES_TRANSPORT),
        ("launch", "HTTP 429 Too Many Requests", EvidenceState.TRANSIENT_UNAVAILABLE, FailureCategory.KUBERNETES_RATE_LIMIT),
        ("launch", "503 Service Unavailable", EvidenceState.TRANSIENT_UNAVAILABLE, FailureCategory.KUBERNETES_SERVER),
        ("launch", "TLS handshake timeout", EvidenceState.TRANSIENT_UNAVAILABLE, FailureCategory.KUBERNETES_TRANSPORT),
        ("launch", "forbidden: cannot list pods", EvidenceState.TERMINAL, FailureCategory.RBAC),
        ("launch", "x509: certificate expired", EvidenceState.TERMINAL, FailureCategory.CERTIFICATE),
        ("launch", "context not found", EvidenceState.TERMINAL, FailureCategory.CONTEXT),
        ("launch", "task timed out running user code", EvidenceState.AMBIGUOUS, FailureCategory.UNKNOWN),
        ("workload", "connection refused", EvidenceState.AMBIGUOUS, FailureCategory.UNKNOWN),
    ],
)
def test_failure_taxonomy_is_phase_aware(
    phase: str,
    detail: str,
    state: EvidenceState,
    category: FailureCategory,
) -> None:
    assert classify_failure(phase=phase, stderr=detail) == (state, category)


def test_backoff_jitter_is_capped_and_bounded() -> None:
    policy = RecoveryPolicy(
        initial_backoff_seconds=4,
        cap_backoff_seconds=10,
        multiplier=2,
        jitter_ratio=0.25,
    )
    assert policy.delay(1, random_value=0) == 3
    assert policy.delay(1, random_value=1) == 5
    assert policy.delay(9, random_value=1) == 10
    assert 0 <= policy.delay(9, random_value=-100) <= 10


def test_kubectl_probe_uses_exact_context_and_redacts_sensitive_argv() -> None:
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def runner(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, stdout="ok\n", stderr="")

    probe = KubectlApiProbe(
        env={"KUBECONFIG": "/secret/location/config", "TOKEN": "do-not-print"},
        context="exact-context",
        runner=runner,
        clock=lambda: 7.0,
        timestamp=lambda: "now",
        kubectl="/usr/bin/kubectl",
    )
    sample = probe()
    payload = sample.to_dict()
    assert sample.state is EvidenceState.READY
    assert calls[0][0] == [
        "/usr/bin/kubectl",
        "--context",
        "exact-context",
        "get",
        "--raw=/readyz",
    ]
    rendered = repr(payload)
    assert "/secret/location/config" not in rendered
    assert "do-not-print" not in rendered


def test_launch_accepted_but_client_failed_is_adopted() -> None:
    exists = False

    def launch() -> None:
        nonlocal exists
        exists = True
        raise RuntimeError("connection reset by peer")

    result = run_launch_transaction(
        logical_id="accepted",
        readiness=_stable,
        launch=launch,
        reconcile=lambda: ReconciliationEvidence(
            ReconciliationState.FOUND,
            "41",
            "PENDING",
            workload_observable=True,
            workload_evidence="scheduler_state",
        )
        if exists
        else ReconciliationEvidence(ReconciliationState.ABSENT),
        classify_launch_error=_transient,
    )
    assert result.state is LaunchState.ADOPTED
    assert result.job_id == "41"
    assert result.launch_sequence == 1
    assert result.reconciliations[-1]["workload_observable"] is True


def test_getcwd_rsync_failure_rejects_phantom_pending_queue_record() -> None:
    """A row allocated before controller file sync is not a submitted workload."""

    exists = False

    def launch() -> None:
        nonlocal exists
        exists = True
        raise RuntimeError(
            "getcwd() failed: No such file or directory; "
            "rsync failed with return code 3"
        )

    with pytest.raises(LaunchTransactionError) as caught:
        run_launch_transaction(
            logical_id="getcwd-rsync-phantom",
            readiness=_stable,
            launch=launch,
            reconcile=lambda: ReconciliationEvidence(
                ReconciliationState.FOUND,
                "125",
                "PENDING",
                workload_observable=False,
                workload_evidence="",
            )
            if exists
            else ReconciliationEvidence(ReconciliationState.ABSENT),
            classify_launch_error=_transient,
        )

    result = caught.value.result
    assert result.state is LaunchState.TERMINAL_FAILURE
    assert result.category is FailureCategory.CONFIG
    assert (
        result.recovery_decision
        == "reject_unobservable_queue_record_after_launch_failure"
    )
    assert result.job_id == "125"
    assert result.reconciliations[-1] == {
        "state": "found",
        "job_id": "125",
        "status": "PENDING",
        "workload_observable": False,
        "workload_evidence": "",
        "error": "",
    }


def test_resume_rejects_existing_unobservable_phantom_record() -> None:
    with pytest.raises(LaunchTransactionError) as caught:
        run_launch_transaction(
            logical_id="resume-phantom",
            readiness=_stable,
            launch=lambda: pytest.fail("phantom record must block relaunch"),
            reconcile=lambda: ReconciliationEvidence(
                ReconciliationState.FOUND, "125", "PENDING"
            ),
            classify_launch_error=_transient,
        )

    assert caught.value.result.state is LaunchState.INDETERMINATE
    assert (
        caught.value.result.recovery_decision
        == "block_unobservable_existing_record"
    )


def test_resume_relaunches_instead_of_adopting_cancelled_job() -> None:
    reconciliations = iter(
        [
            ReconciliationEvidence(
                ReconciliationState.FOUND,
                "125",
                "CANCELLED",
                workload_observable=True,
                workload_evidence="scheduler_state",
            ),
            ReconciliationEvidence(
                ReconciliationState.FOUND,
                "126",
                "PENDING",
                workload_observable=True,
                workload_evidence="scheduler_state",
            ),
        ]
    )
    launches = 0

    def launch() -> None:
        nonlocal launches
        launches += 1

    result = run_launch_transaction(
        logical_id="resume-after-cancel",
        readiness=_stable,
        launch=launch,
        reconcile=lambda: next(reconciliations),
        classify_launch_error=_transient,
    )

    assert launches == 1
    assert result.state is LaunchState.SUBMITTED
    assert result.job_id == "126"
    assert result.recovery_decision == "submitted_and_reconciled"


def test_authoritative_absence_allows_one_safe_retry() -> None:
    clock = FakeClock()
    launches = 0
    exists = False

    def launch() -> object:
        nonlocal launches, exists
        launches += 1
        if launches == 1:
            raise RuntimeError("connection refused")
        exists = True
        return object()

    result = run_launch_transaction(
        logical_id="safe-retry",
        readiness=_stable,
        launch=launch,
        reconcile=lambda: ReconciliationEvidence(
            ReconciliationState.FOUND, "9", "PENDING"
        )
        if exists
        else ReconciliationEvidence(ReconciliationState.ABSENT),
        classify_launch_error=_transient,
        recovery_policy=RecoveryPolicy(30, 1, 1, 2, 0),
        clock=clock,
        sleeper=clock.sleep,
        random_source=lambda: 0.5,
    )
    assert result.state is LaunchState.SUBMITTED
    assert result.job_id == "9"
    assert launches == 2
    assert len(result.readiness) == 2


def test_repeated_transient_failure_stops_at_recovery_deadline() -> None:
    clock = FakeClock()
    launches = 0

    def launch() -> None:
        nonlocal launches
        launches += 1
        raise RuntimeError("connection refused")

    with pytest.raises(LaunchTransactionError) as caught:
        run_launch_transaction(
            logical_id="deadline",
            readiness=_stable,
            launch=launch,
            reconcile=lambda: ReconciliationEvidence(ReconciliationState.ABSENT),
            classify_launch_error=_transient,
            recovery_policy=RecoveryPolicy(3, 2, 2, 2, 0),
            clock=clock,
            sleeper=clock.sleep,
            random_source=lambda: 0.5,
        )
    assert caught.value.result.recovery_decision == "recovery_deadline_exhausted_verified_absent"
    assert caught.value.result.state is LaunchState.TRANSIENT_API_FAILURE
    assert caught.value.result.existence == "absent"
    assert launches == 2


@pytest.mark.parametrize("message", ["unauthorized", "forbidden", "context not found"])
def test_terminal_auth_context_rbac_never_retries(message: str) -> None:
    launches = 0

    def launch() -> None:
        nonlocal launches
        launches += 1
        raise RuntimeError(message)

    with pytest.raises(LaunchTransactionError) as caught:
        run_launch_transaction(
            logical_id=f"terminal-{message}",
            readiness=_stable,
            launch=launch,
            reconcile=lambda: ReconciliationEvidence(ReconciliationState.ABSENT),
            classify_launch_error=_transient,
        )
    assert launches == 1
    assert caught.value.result.state is LaunchState.TERMINAL_FAILURE
    assert caught.value.result.recovery_decision == "verified_absent_no_retry"


def test_indeterminate_reconciliation_never_launches_or_retries() -> None:
    launches = 0

    def launch() -> None:
        nonlocal launches
        launches += 1

    with pytest.raises(LaunchTransactionError) as caught:
        run_launch_transaction(
            logical_id="indeterminate",
            readiness=_stable,
            launch=launch,
            reconcile=lambda: ReconciliationEvidence(
                ReconciliationState.UNAVAILABLE, error="queue API unavailable"
            ),
            classify_launch_error=_transient,
        )
    assert launches == 0
    assert caught.value.result.state is LaunchState.INDETERMINATE


def test_async_launch_waits_for_exact_job_observability_without_relaunch() -> None:
    clock = FakeClock()
    launches = 0
    observations = iter(
        [
            ReconciliationEvidence(ReconciliationState.ABSENT),
            ReconciliationEvidence(ReconciliationState.ABSENT),
            ReconciliationEvidence(ReconciliationState.ABSENT),
            ReconciliationEvidence(ReconciliationState.FOUND, "88", "PENDING"),
        ]
    )

    def launch() -> object:
        nonlocal launches
        launches += 1
        return object()

    result = run_launch_transaction(
        logical_id="async-observability",
        readiness=_stable,
        launch=launch,
        reconcile=lambda: next(observations),
        classify_launch_error=_transient,
        recovery_policy=RecoveryPolicy(30, 1, 1, 2, 0),
        clock=clock,
        sleeper=clock.sleep,
        random_source=lambda: 0.5,
    )

    assert result.state is LaunchState.SUBMITTED
    assert result.job_id == "88"
    assert launches == 1
    assert len(result.reconciliations) == 4


def test_two_local_callers_produce_at_most_one_launch_and_second_adopts(
    tmp_path: Path,
) -> None:
    identity = logical_launch_identity("project", "run", "wave", "attempt-1")
    guard = threading.Lock()
    job_id = ""
    launches = 0
    barrier = threading.Barrier(2)
    results: list[Any] = []

    def reconcile() -> ReconciliationEvidence:
        with guard:
            return (
                ReconciliationEvidence(
                    ReconciliationState.FOUND,
                    job_id,
                    "PENDING",
                    workload_observable=True,
                    workload_evidence="scheduler_state",
                )
                if job_id
                else ReconciliationEvidence(ReconciliationState.ABSENT)
            )

    def launch() -> object:
        nonlocal launches, job_id
        with guard:
            launches += 1
            job_id = "88"
        return object()

    def caller() -> None:
        barrier.wait()
        results.append(
            run_launch_transaction(
                logical_id=identity,
                readiness=_stable,
                launch=launch,
                reconcile=reconcile,
                classify_launch_error=_transient,
                lock_root=tmp_path,
            )
        )

    threads = [threading.Thread(target=caller), threading.Thread(target=caller)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    assert all(not thread.is_alive() for thread in threads)
    assert launches == 1
    assert sorted(result.state.value for result in results) == ["adopted", "submitted"]
    lock = tmp_path / f"{identity}.lock"
    assert lock.stat().st_mode & 0o777 == 0o600


def test_hermetic_incident_boundary_recovers_without_cancellation() -> None:
    """Cluster/GPU snapshot passed, controller POST refused, then safely recovered."""

    clock = FakeClock()
    api_states = [
        EvidenceState.READY,
        EvidenceState.READY,
        EvidenceState.READY,
        EvidenceState.TRANSIENT_UNAVAILABLE,
        EvidenceState.READY,
        EvidenceState.READY,
        EvidenceState.READY,
    ]
    probe = SequenceProbe(clock, api_states)
    launches = 0
    exact_job = ""
    cancellations: list[str] = []

    def readiness() -> StabilityResult:
        return wait_for_api_stability(
            probe,
            policy=StabilityPolicy(3, 2, 1, 10),
            clock=clock,
            sleeper=clock.sleep,
        )

    def launch() -> object:
        nonlocal launches, exact_job
        launches += 1
        if launches == 1:
            raise RuntimeError("Kubernetes API endpoint: connection refused")
        exact_job = "101"
        return object()

    result = run_launch_transaction(
        logical_id=logical_launch_identity("project", "paidf-run", "wave-001", "1"),
        readiness=readiness,
        launch=launch,
        reconcile=lambda: ReconciliationEvidence(
            ReconciliationState.FOUND, exact_job, "PENDING"
        )
        if exact_job
        else ReconciliationEvidence(ReconciliationState.ABSENT),
        classify_launch_error=_transient,
        recovery_policy=RecoveryPolicy(30, 1, 1, 2, 0),
        clock=clock,
        sleeper=clock.sleep,
        random_source=lambda: 0.5,
    )
    assert result.state is LaunchState.SUBMITTED
    assert result.job_id == "101"
    assert launches == 2
    assert cancellations == []
    assert result.recovery_decision == "submitted_and_reconciled"
