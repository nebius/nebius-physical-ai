from __future__ import annotations

import pytest

from npa.verification import (
    CACHED,
    VERIFICATION_UNAVAILABLE,
    apply_verification,
    classify_verification_failure,
    sanitize_failure_reason,
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


def test_sanitizer_removes_quoted_assignments_and_non_http_queries() -> None:
    sanitized = sanitize_reason(
        '{"aws_secret_access_key":"quoted-secret"} '
        "s3://bucket/key?X-Amz-Signature=object-secret"
    )

    assert "quoted-secret" not in sanitized
    assert "X-Amz-Signature" not in sanitized
    assert "<redacted>" in sanitized


def test_failure_sanitizer_requires_and_redacts_explicit_secrets() -> None:
    assert "hunter2" not in sanitize_failure_reason(
        "login failed for hunter2",
        secrets=("hunter2",),
    )
    with pytest.raises(TypeError):
        sanitize_failure_reason("login failed")  # type: ignore[call-arg]


def test_display_failure_redactor_preserves_multiline_recovery_text() -> None:
    from npa.verification import redact_failure_text

    opaque_secret = "synthetic-opaque-display-credential"
    message = (
        "provider rejected HF_TOKEN=hf_synthetic_display_token\n"
        "retry: npa workbench workflow status synthetic-run\n"
        "details: s3://bucket/key?X-Amz-Signature=synthetic-query\n"
        'quoted: {"api_key": "synthetic quoted credential"}\n'
        "transport: custom+s3://synthetic-user:synthetic-password@bucket/key\n"
        f"opaque: {opaque_secret}"
    )

    sanitized = redact_failure_text(message, secrets=(opaque_secret,))

    assert sanitized.count("\n") == message.count("\n")
    assert "retry: npa workbench workflow status synthetic-run" in sanitized
    assert "hf_synthetic_display_token" not in sanitized
    assert "X-Amz-Signature" not in sanitized
    assert "synthetic quoted credential" not in sanitized
    assert "synthetic-user" not in sanitized
    assert "synthetic-password" not in sanitized
    assert opaque_secret not in sanitized
    assert '{"api_key": "<redacted>"}' in sanitized
    assert sanitized.count("<redacted>") >= 5


@pytest.mark.parametrize(
    ("message", "secret"),
    [
        (
            'provider rejected api_key: "SYNTH-UNTERMINATED-DOUBLE',
            "SYNTH-UNTERMINATED-DOUBLE",
        ),
        (
            "provider rejected api_key: 'SYNTH-UNTERMINATED-SINGLE",
            "SYNTH-UNTERMINATED-SINGLE",
        ),
        (
            '{"api_key": "SYNTH-DOUBLE operator\'s credential"}',
            "SYNTH-DOUBLE",
        ),
        (
            "{'api_key': 'SYNTH-SINGLE operator \"quoted\" credential'}",
            "SYNTH-SINGLE",
        ),
        (
            'provider response truncated at "authorization": "SYNTH-TRUNCATED',
            "SYNTH-TRUNCATED",
        ),
        ("Authorization: Basic SYNTHBASE64VALUE", "SYNTHBASE64VALUE"),
    ],
)
def test_display_failure_redactor_fails_closed_for_malformed_quoted_assignments(
    message: str,
    secret: str,
) -> None:
    from npa.verification import redact_failure_text

    sanitized = redact_failure_text(message, secrets=())

    assert secret not in sanitized
    assert "<redacted>" in sanitized


def test_display_failure_redactor_never_consumes_later_recovery_lines() -> None:
    from npa.verification import redact_failure_text

    message = (
        'provider rejected api_key: "SYNTH-LINE-SECRET\n'
        "retry: npa workbench workflow status synthetic-run\n"
        '"docs"'
    )

    sanitized = redact_failure_text(message, secrets=())

    assert "SYNTH-LINE-SECRET" not in sanitized
    assert "retry: npa workbench workflow status synthetic-run" in sanitized
    assert sanitized.count("\n") == message.count("\n")


def test_display_failure_redactor_consumes_ambiguous_quoted_value_tail() -> None:
    from npa.verification import redact_failure_text

    message = (
        'provider rejected api_key: "SYNTH-MALFORMED "quoted" credential"\n'
        "retry: npa workbench workflow status synthetic-run"
    )

    sanitized = redact_failure_text(message, secrets=())

    assert "SYNTH-MALFORMED" not in sanitized
    assert "quoted" not in sanitized
    assert "credential" not in sanitized
    assert "retry: npa workbench workflow status synthetic-run" in sanitized


def test_display_failure_redactor_preserves_structured_fields_after_secret() -> None:
    from npa.verification import redact_failure_text

    message = '{"api_key": "SYNTH-SECRET", "retry": "keep-this"}'

    assert redact_failure_text(message, secrets=()) == (
        '{"api_key": "<redacted>", "retry": "keep-this"}'
    )


@pytest.mark.parametrize("scope", ["config", "run", "state", "loop"])
def test_display_failure_redactor_preserves_unknown_workflow_token_name(
    scope: str,
) -> None:
    from npa.verification import redact_failure_text

    message = f"state score-rollouts: unknown {scope} token: {scope}.does_not_exist"

    assert redact_failure_text(message, secrets=()) == message


def test_display_failure_redactor_does_not_allow_shapeless_unknown_token() -> None:
    from npa.verification import redact_failure_text

    message = "state score-rollouts: unknown config token: SYNTH-OPAQUE-SECRET"

    assert redact_failure_text(message, secrets=()).endswith("token: <redacted>")


@pytest.mark.parametrize("prefix", [".", "-", "+", "..."])
def test_display_failure_redactor_covers_punctuation_prefixed_urls(
    prefix: str,
) -> None:
    from npa.verification import redact_failure_text

    message = (
        f"{prefix}https://synth-user:synth-pass@example.invalid/path "
        f"{prefix}s3://bucket/key?X-Amz-Signature=synth-signature"
    )

    sanitized = redact_failure_text(message, secrets=())

    assert "synth-user" not in sanitized
    assert "synth-pass" not in sanitized
    assert "synth-signature" not in sanitized
    assert f"{prefix}https://<redacted>@example.invalid/path" in sanitized
    assert f"{prefix}s3://bucket/key?<redacted>" in sanitized


def test_display_failure_redactor_handles_long_secret_assignment_key() -> None:
    from npa.verification import redact_failure_text

    secret = "SYNTH-LONG-KEY-SECRET"
    message = f"{'x' * 256}_token={secret}"

    sanitized = redact_failure_text(message, secrets=())

    assert secret not in sanitized
    assert sanitized.endswith("=<redacted>")


def test_display_failure_redactor_handles_dense_same_line_assignments() -> None:
    from npa.verification import redact_failure_text

    message = ", ".join(f"token=SYNTH-DENSE-{index}" for index in range(5_000))

    sanitized = redact_failure_text(message, secrets=())

    assert "SYNTH-DENSE" not in sanitized
    assert sanitized.count("token=<redacted>") == 5_000


def test_assignment_redactor_never_scans_back_to_line_start_per_match() -> None:
    from npa.diagnostic_redaction import _redact_secret_assignments

    class NoBackwardScan(str):
        def rfind(self, *_args: object, **_kwargs: object) -> int:
            raise AssertionError("assignment context lookup must stay bounded")

    message = NoBackwardScan(", ".join(f"token=value-{index}" for index in range(50)))

    assert _redact_secret_assignments(message).count("token=<redacted>") == 50


@pytest.mark.parametrize(
    "separator",
    [",", ";", "|", "><", '","next":"'],
)
def test_display_failure_redactor_covers_compact_multiple_urls(
    separator: str,
) -> None:
    from npa.verification import redact_failure_text, sanitize_failure_reason

    message = (
        f"https://synth-user-1:synth-pass-1@one.invalid{separator}"
        "https://synth-user-2:synth-pass-2@two.invalid"
    )

    for sanitized in (
        redact_failure_text(message, secrets=()),
        sanitize_failure_reason(message, secrets=()),
    ):
        for secret in (
            "synth-user-1",
            "synth-pass-1",
            "synth-user-2",
            "synth-pass-2",
        ):
            assert secret not in sanitized
        assert sanitized.count("<redacted>@") == 2


def test_display_failure_redactor_covers_query_before_compact_second_url() -> None:
    from npa.verification import redact_failure_text

    message = (
        "https://one.invalid/path?X-Amz-Signature=synth-signature,"
        "https://synth-user:synth-pass@two.invalid"
    )

    sanitized = redact_failure_text(message, secrets=())

    assert "synth-signature" not in sanitized
    assert "synth-user" not in sanitized
    assert "synth-pass" not in sanitized
    assert "<redacted>" in sanitized


@pytest.mark.parametrize("nested_prefix", ["target=", "", "x"])
def test_display_failure_redactor_covers_signature_after_nested_url(
    nested_prefix: str,
) -> None:
    from npa.verification import redact_failure_text, sanitize_failure_reason

    signature = "synth-nested-signature"
    message = (
        "storage rejected s3://bucket/key?"
        f"{nested_prefix}https://storage.invalid/path"
        f"&X-Amz-Signature={signature}"
    )

    for sanitized in (
        redact_failure_text(message, secrets=()),
        sanitize_failure_reason(message, secrets=()),
    ):
        assert signature not in sanitized
        assert "X-Amz-Signature" not in sanitized
        assert "s3://bucket/key?<redacted>" in sanitized


def test_display_failure_redactor_preserves_line_after_bearer_label() -> None:
    from npa.verification import redact_failure_text

    message = "provider rejected bearer\nretry: npa workbench health preflight"

    assert redact_failure_text(message, secrets=()) == message


def test_display_failure_redactor_does_not_consume_url_scheme_after_bearer() -> None:
    from npa.verification import redact_failure_text

    message = (
        "missing bearer "
        "https://synth-user:synth-password@storage.invalid/path?"
        "X-Amz-Signature=synth-signature"
    )

    assert redact_failure_text(message, secrets=()) == (
        "missing bearer https://<redacted>@storage.invalid/path?<redacted>"
    )


def test_display_failure_redactor_accepts_nonbreaking_assignment_space() -> None:
    from npa.verification import redact_failure_text

    secret = "SYNTH-NBSP-SECRET"

    assert redact_failure_text(f"api_key:\u00a0{secret}", secrets=()) == (
        "api_key:\u00a0<redacted>"
    )


@pytest.mark.parametrize("separator", ["\u00a0", "\u202f", "\u2007"])
def test_display_failure_redactor_accepts_horizontal_unicode_space(
    separator: str,
) -> None:
    from npa.verification import redact_failure_text

    secret = "SYNTH-HORIZONTAL-SPACE-SECRET"

    assert redact_failure_text(f"Bearer{separator}{secret}", secrets=()) == (
        "Bearer <redacted>"
    )
    assert (
        redact_failure_text(
            f"Authorization:{separator}Basic{separator}{secret}",
            secrets=(),
        )
        == f"Authorization:{separator}<redacted>"
    )


@pytest.mark.parametrize(
    "separator",
    ["\r", "\n", "\v", "\f", "\x1c", "\x1d", "\x1e", "\x85", "\u2028", "\u2029"],
)
def test_display_failure_redactor_preserves_every_line_boundary(
    separator: str,
) -> None:
    from npa.verification import redact_failure_text

    message = f"missing api_key:{separator}retry: npa workbench health preflight"

    assert redact_failure_text(message, secrets=()) == message


def test_display_failure_redactor_does_not_treat_next_line_as_secret_value() -> None:
    from npa.verification import redact_failure_text

    message = "missing token:\n  run npa workbench health preflight"

    assert redact_failure_text(message, secrets=()) == message


@pytest.mark.parametrize(
    ("message", "secret"),
    [
        ("Bearer SYNTHETIC-OPAQUE-CREDENTIAL", "SYNTHETIC-OPAQUE-CREDENTIAL"),
        (
            "https://synthetic-user:synthetic-password@provider.invalid/path",
            "synthetic-password",
        ),
        (
            "s3://bucket/key?X-Amz-Signature=synthetic-signature",
            "synthetic-signature",
        ),
        (
            "-----BEGIN PRIVATE KEY-----\nsynthetic-key\n-----END PRIVATE KEY-----",
            "synthetic-key",
        ),
        ("AKIAABCDEFGHIJKLMNOP", "AKIAABCDEFGHIJKLMNOP"),
    ],
)
def test_workflow_state_redactor_covers_unstructured_credential_formats(
    message: str,
    secret: str,
) -> None:
    from npa.orchestration.skypilot.workflow_state import redact_text

    assert secret not in redact_text(message)


def test_workflow_state_redactor_preserves_nonsecret_diagnostics() -> None:
    from npa.orchestration.skypilot.workflow_state import redact_text

    message = (
        "retry: npa workbench workflow status synthetic-run\n"
        "endpoint=https://provider.invalid/path\n"
        "state score-rollouts: unknown config token: config.does_not_exist"
    )

    assert redact_text(message) == message


def test_workflow_state_redactor_is_idempotent_and_preserves_lines() -> None:
    from npa.orchestration.skypilot.workflow_state import redact_text

    opaque = "synthetic-exact-opaque-secret"
    message = (
        "provider rejected Bearer SYNTHETIC-BEARER\n"
        "-----BEGIN PRIVATE KEY-----\n"
        "synthetic-key\n"
        "-----END PRIVATE KEY-----\n"
        "retry: npa workbench health preflight\n"
        f"opaque={opaque}"
    )

    sanitized = redact_text(message, secrets=(opaque,))

    assert sanitized.count("\n") == message.count("\n")
    assert "retry: npa workbench health preflight" in sanitized
    assert redact_text(sanitized, secrets=(opaque,)) == sanitized
    assert all(
        secret not in sanitized
        for secret in ("SYNTHETIC-BEARER", "synthetic-key", opaque)
    )


def test_workflow_state_redactor_preserves_same_line_recovery_context() -> None:
    from npa.orchestration.skypilot.workflow_state import redact_text

    message = 'ERROR auth failed: token="synthetic-secret" (http 401) retry in 30s'

    assert redact_text(message) == (
        'ERROR auth failed: token="<redacted>" (http 401) retry in 30s'
    )


def test_workflow_state_redactor_preserves_token_counters() -> None:
    from npa.orchestration.skypilot.workflow_state import redact_text

    message = "prompt_tokens=812 completion_tokens=133 total_tokens=945"

    assert redact_text(message) == message


def test_workflow_state_redactor_covers_unencoded_at_in_url_userinfo() -> None:
    from npa.orchestration.skypilot.workflow_state import redact_text

    sanitized = redact_text(
        "https://synthetic-user:p@ssword-synthetic@provider.invalid/path"
    )

    assert "synthetic-user" not in sanitized
    assert "ssword-synthetic" not in sanitized
    assert sanitized == "https://<redacted>@provider.invalid/path"


def test_private_key_redactor_fails_closed_linearly_for_truncated_blocks(
    monkeypatch,
) -> None:
    import npa.diagnostic_redaction as redaction

    original = redaction._PRIVATE_KEY_END

    class CountingPattern:
        def __init__(self) -> None:
            self.calls = 0

        def search(self, text: str, start: int = 0):
            self.calls += 1
            return original.search(text, start)

    end_pattern = CountingPattern()
    monkeypatch.setattr(redaction, "_PRIVATE_KEY_END", end_pattern)
    message = ("-----BEGIN PRIVATE KEY-----\nsynthetic-key\n" * 5_000).rstrip()

    sanitized = redaction.redact_diagnostic_text(message)

    assert end_pattern.calls == 1
    assert "synthetic-key" not in sanitized
    assert "BEGIN PRIVATE KEY" not in sanitized
    assert sanitized.count("\n") == message.count("\n")
