from __future__ import annotations

import pytest

from npa.agent_backend.gpu_allocation_fallback import (
    ON_DEMAND,
    PREEMPTIBLE,
    candidate_is_compatible,
    classify_failure,
    record_attempt,
    record_consent,
)


def _request(**overrides):
    request = {
        "gpu_family": "rtx-pro",
        "gpu_product": "RTXPRO6000",
        "gpu_count": 2,
        "image": "registry.example/npa@sha256:synthetic",
        "image_digest": "sha256:synthetic",
        "sm": "sm_120",
        "rt_cores_required": True,
        "backend": "kubernetes",
        "model": "policy-a",
        "workload_tier": "render",
        "execution_mode": "train",
        "boot_disk_count": 2,
        "boot_disk_size_bytes": 2 * 1023 * 1024**3,
        "pool": ON_DEMAND,
    }
    request.update(overrides)
    return request


def _candidate(**overrides):
    candidate = _request(pool=PREEMPTIBLE)
    candidate.update(overrides)
    return candidate


@pytest.mark.parametrize(
    ("code", "message", "category"),
    [
        ("quota_exhausted", "", "quota_exhausted"),
        ("capacity_exhausted", "", "capacity_exhausted"),
        ("unschedulable_gpu", "", "unschedulable_gpu"),
        ("no_compatible_product", "", "no_compatible_product"),
        (
            "",
            "0/3 nodes are Unschedulable: insufficient nvidia.com/gpu",
            "unschedulable_gpu",
        ),
    ],
)
def test_qualifying_classifications(code: str, message: str, category: str) -> None:
    assert classify_failure(code, message) == {"category": category, "qualifying": True}


@pytest.mark.parametrize(
    "code",
    [
        "auth",
        "rbac",
        "network",
        "image_pull",
        "checkpoint",
        "application",
        "runtime",
        "timeout",
    ],
)
def test_non_placement_failures_never_count(code: str) -> None:
    state, decision = record_attempt(
        None,
        logical_allocation="run-a",
        request=_request(),
        failure_code=code,
        failure_message="quota text nested in an authentication failure",
        candidate=_candidate(),
    )
    assert state["qualifying_attempts"] == 0
    assert decision["prompt"] is False


def test_prompts_once_at_default_third_qualifying_attempt() -> None:
    state = None
    decisions = []
    for _ in range(4):
        state, decision = record_attempt(
            state,
            logical_allocation="run-a",
            request=_request(),
            failure_code="capacity_exhausted",
            evidence={"source": "scheduler"},
            candidate=_candidate(),
        )
        decisions.append(decision)
    assert [item["prompt"] for item in decisions] == [False, False, True, False]
    assert decisions[2]["reason"] == "failed_attempt_threshold"


def test_deterministic_preflight_prompts_immediately() -> None:
    state, decision = record_attempt(
        None,
        logical_allocation="run-a",
        request=_request(),
        failure_code="quota_exhausted",
        evidence={
            "source": "provider-preflight",
            "on_demand_impossible": True,
            "preemptible_available": True,
        },
        candidate=_candidate(),
    )
    assert state["qualifying_attempts"] == 1
    assert decision["prompt"] is True
    assert decision["reason"] == "deterministic_preflight"


def test_deterministic_preflight_does_not_need_a_prior_failed_apply() -> None:
    _state, decision = record_attempt(
        None,
        logical_allocation="run-a",
        request=_request(),
        evidence={
            "source": "provider-preflight",
            "on_demand_impossible": True,
            "preemptible_available": True,
        },
        candidate=_candidate(),
    )
    assert decision["prompt"] is True


def test_missing_invariant_or_preemptible_source_never_prompts() -> None:
    request = _request()
    del request["image_digest"]
    assert candidate_is_compatible(request, _candidate()) is False
    state, decision = record_attempt(
        None,
        logical_allocation="run-a",
        request=_request(pool=PREEMPTIBLE),
        failure_code="capacity_exhausted",
        candidate=_candidate(),
    )
    assert state["qualifying_attempts"] == 0
    assert decision == {"prompt": False, "reason": "not_on_demand"}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("gpu_family", "h100"),
        ("gpu_count", 1),
        ("image_digest", "sha256:other"),
        ("sm", "sm_90"),
        ("rt_cores_required", False),
        ("backend", "vm"),
        ("model", "policy-b"),
        ("workload_tier", "headless"),
        ("execution_mode", "infer"),
        ("boot_disk_count", 1),
        ("boot_disk_size_bytes", 1023 * 1024**3),
    ],
)
def test_compatibility_and_disk_invariants_block_fallback(
    field: str, value: object
) -> None:
    assert candidate_is_compatible(_request(), _candidate(**{field: value})) is False


def test_consent_yes_requires_bound_digest_and_only_changes_pool() -> None:
    request = _request()
    state, decision = record_attempt(
        None,
        logical_allocation="run-a",
        request=request,
        failure_code="quota_exhausted",
        evidence={
            "source": "provider-preflight",
            "on_demand_impossible": True,
            "preemptible_available": True,
        },
        candidate=_candidate(),
    )
    with pytest.raises(ValueError, match="not bound"):
        record_consent(state, accepted=True, confirmed_action_digest="wrong")
    accepted = record_consent(
        state,
        accepted=True,
        confirmed_action_digest=decision["proposed_action"]["digest"],
    )
    assert accepted["selected_pool"] == PREEMPTIBLE
    assert accepted["invariants"] == state["invariants"]
    with pytest.raises(ValueError, match="no GPU"):
        record_consent(accepted, accepted=True, confirmed_action_digest="anything")


def test_accepted_preemptible_pool_suppresses_later_on_demand_evidence() -> None:
    state, decision = record_attempt(
        None,
        logical_allocation="run-a",
        request=_request(),
        failure_code="quota_exhausted",
        evidence={
            "source": "provider-preflight",
            "on_demand_impossible": True,
            "preemptible_available": True,
            "fingerprint": "first",
        },
        candidate=_candidate(),
    )
    accepted = record_consent(
        state,
        accepted=True,
        confirmed_action_digest=decision["proposed_action"]["digest"],
    )
    later, later_decision = record_attempt(
        accepted,
        logical_allocation="run-a",
        request=_request(),
        failure_code="capacity_exhausted",
        evidence={
            "source": "provider-preflight",
            "on_demand_impossible": True,
            "preemptible_available": True,
            "fingerprint": "materially-new-evidence",
        },
        candidate=_candidate(),
    )
    assert later_decision == {"prompt": False, "reason": "preemptible_already_selected"}
    assert later["selected_pool"] == PREEMPTIBLE
    assert later["consent"] == "accepted"


def test_decline_preserves_on_demand_and_new_evidence_reprompts() -> None:
    state, _ = record_attempt(
        None,
        logical_allocation="run-a",
        request=_request(),
        failure_code="quota_exhausted",
        evidence={
            "source": "provider-preflight",
            "on_demand_impossible": True,
            "preemptible_available": True,
        },
        candidate=_candidate(),
    )
    declined = record_consent(state, accepted=False)
    assert declined["selected_pool"] == ON_DEMAND
    same, same_decision = record_attempt(
        declined,
        logical_allocation="run-a",
        request=_request(),
        failure_code="quota_exhausted",
        evidence={
            "source": "provider-preflight",
            "on_demand_impossible": True,
            "preemptible_available": True,
        },
        candidate=_candidate(),
    )
    assert same_decision["prompt"] is False
    changed, changed_decision = record_attempt(
        same,
        logical_allocation="run-a",
        request=_request(),
        failure_code="capacity_exhausted",
        evidence={
            "source": "scheduler",
            "on_demand_impossible": True,
            "preemptible_available": True,
        },
        candidate=_candidate(),
    )
    assert changed_decision["prompt"] is True
    assert changed["status"] == "awaiting-consent"


def test_success_and_changed_logical_allocation_reset_attempts() -> None:
    state, _ = record_attempt(
        None,
        logical_allocation="run-a",
        request=_request(),
        failure_code="capacity_exhausted",
        candidate=_candidate(),
    )
    succeeded, _ = record_attempt(
        state,
        logical_allocation="run-a",
        request=_request(),
        success=True,
    )
    assert succeeded["qualifying_attempts"] == 0
    assert succeeded["status"] == "succeeded"
    changed, _ = record_attempt(
        state,
        logical_allocation="run-b",
        request=_request(),
        failure_code="capacity_exhausted",
        candidate=_candidate(),
    )
    assert changed["qualifying_attempts"] == 1
    assert changed["logical_allocation_ref"] != state["logical_allocation_ref"]


def test_provenance_is_redacted_and_bounded() -> None:
    state, _ = record_attempt(
        None,
        logical_allocation="customer-run-name",
        request=_request(),
        failure_code="capacity_exhausted",
        failure_message="provider raw response with customer-id",
        evidence={"source": "terraform", "raw": "secret provider response"},
        candidate=_candidate(),
    )
    rendered = repr(state)
    assert "customer-run-name" not in rendered
    assert "customer-id" not in rendered
    assert "secret provider response" not in rendered
