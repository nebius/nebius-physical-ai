from __future__ import annotations

import pytest

from npa.verification import (
    CACHED,
    VERIFICATION_UNAVAILABLE,
    apply_verification,
    classify_verification_failure,
    sanitize_reason,
)


@pytest.mark.parametrize(
    ("reason", "code", "category"),
    [
        (
            "dial tcp: lookup api.example.invalid: no such host",
            "DNS_RESOLUTION_FAILED",
            "DNS",
        ),
        ("request deadline exceeded", "LIVE_QUERY_TIMEOUT", "TIMEOUT"),
        ("pods is forbidden by RBAC", "LIVE_QUERY_FORBIDDEN", "RBAC"),
        ("401 Unauthorized", "LIVE_QUERY_AUTHENTICATION_FAILED", "AUTHENTICATION"),
        ("stale Kubernetes context not found", "LIVE_CONTEXT_MISMATCH", "CONTEXT"),
        ("malformed JSON response", "LIVE_RESPONSE_UNPARSEABLE", "RESPONSE"),
        ("jobs controller is unreachable", "LIVE_CONTROLLER_UNREACHABLE", "CONTROLLER"),
    ],
)
def test_live_failure_categories_are_stable(
    reason: str, code: str, category: str
) -> None:
    assert classify_verification_failure(reason) == (code, category)


def test_unavailable_state_is_not_healthy_and_preserves_last_known() -> None:
    payload = apply_verification(
        {"status": "RUNNING"},
        status=VERIFICATION_UNAVAILABLE,
        target="managed-job-8",
        last_known_state="RUNNING",
        last_known_at="2026-08-04T01:02:03Z",
        last_known_source="stage_ledger",
        reason="lookup controller.example.invalid: no such host; token=top-secret",
        retry_command="npa workbench workflow status synthetic-run",
        attempted_at="2026-08-06T01:02:03Z",
    )

    assert payload["status"] == "VERIFICATION_UNAVAILABLE"
    assert payload["live_verified"] is False
    assert payload["automation_may_trust_state"] is False
    assert payload["last_known"] == {
        "state": "RUNNING",
        "observed_at": "2026-08-04T01:02:03Z",
        "source": "stage_ledger",
    }
    assert payload["live_verification"]["error_code"] == "DNS_RESOLUTION_FAILED"
    assert payload["live_verification"]["attempted_at"] == "2026-08-06T01:02:03Z"
    assert "top-secret" not in str(payload)


def test_cached_state_is_explicit_and_untrusted() -> None:
    payload = apply_verification(
        {"state": "READY"},
        status=CACHED,
        target="cluster-a",
        last_known_state="READY",
        state_key="state",
    )

    assert payload["state"] == "CACHED"
    assert payload["verification_status"] == "CACHED"
    assert payload["last_known"]["state"] == "READY"
    assert payload["live_verified"] is False


def test_sanitizer_removes_secret_assignments_and_presigned_queries() -> None:
    sanitized = sanitize_reason(
        "authorization=Bearer-secret AWS_SECRET_ACCESS_KEY=hidden "
        "Bearer jwt.hidden.value https://user:password@api.example.invalid/path "
        "https://storage.example/item?X-Amz-Signature=secret"
    )

    assert "Bearer-secret" not in sanitized
    assert "hidden" not in sanitized
    assert "password" not in sanitized
    assert "X-Amz-Signature" not in sanitized
    assert "<redacted>" in sanitized
