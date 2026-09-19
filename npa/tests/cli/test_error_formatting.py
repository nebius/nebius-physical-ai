from __future__ import annotations

import json

from npa.cli._error_formatting import format_error_for_user
from npa.clients.serverless import (
    AuthError,
    EndpointNotFoundError,
    JobIdentityError,
    JobSubmissionIndeterminateError,
    QuotaError,
    ServerlessClientError,
    TransientServerlessError,
)


def test_indeterminate_submission_surfaces_identity_fields_in_text_when_durable() -> (
    None
):
    exc = JobSubmissionIndeterminateError(
        "create_job lookup-by-name recovery failed after a response timeout",
        project_id="project-1",
        job_name="train-1",
        provider_job_id="",
        durable=True,
    )

    rendered = format_error_for_user(exc, output_format="text")

    assert "submission outcome is unknown" in rendered
    assert "project-1" in rendered
    assert "train-1" in rendered
    assert "Do not submit a new job under a different name" in rendered
    assert "a durable submission record will adopt the existing job" in rendered


def test_indeterminate_submission_surfaces_identity_fields_in_json() -> None:
    exc = JobSubmissionIndeterminateError(
        "create_job lookup-by-name recovery failed after a response timeout",
        project_id="project-1",
        job_name="train-1",
        provider_job_id="provider-9",
        durable=True,
    )

    payload = json.loads(format_error_for_user(exc, output_format="json"))

    assert payload["error"] == "JobSubmissionIndeterminate"
    assert payload["project_id"] == "project-1"
    assert payload["job_name"] == "train-1"
    assert payload["provider_job_id"] == "provider-9"
    assert payload["durable"] is True


def test_indeterminate_submission_omits_blank_provider_job_id_line_in_text() -> None:
    exc = JobSubmissionIndeterminateError(
        "message",
        project_id="project-1",
        job_name="train-1",
        provider_job_id="",
        durable=True,
    )

    rendered = format_error_for_user(exc, output_format="text")

    assert "Provider job ID" not in rendered


def test_indeterminate_submission_does_not_promise_durable_reconnect_when_not_durable() -> (
    None
):
    """A non-durable call site has no journal; the guidance must not overclaim one."""
    exc = JobSubmissionIndeterminateError(
        "create_job lookup-by-name recovery failed after a response timeout",
        project_id="project-1",
        job_name="train-1",
        durable=False,
    )

    rendered = format_error_for_user(exc, output_format="text")
    payload = json.loads(format_error_for_user(exc, output_format="json"))

    assert "durable submission record will adopt" not in rendered
    assert "no durable submission record" in rendered
    assert "will still attempt to create a new job" in rendered
    assert payload["durable"] is False


def test_not_enough_resources_still_uses_its_own_formatter() -> None:
    exc = QuotaError(
        "quota exceeded",
        project_id="project-1",
        platform="gpu-h200-sxm",
        preset="1gpu-16vcpu-200gb",
        gpu_count=1,
        suggested_alternatives=["Retry later"],
    )

    rendered = format_error_for_user(exc, output_format="text")

    assert "Quota limit reached" in rendered
    assert "Retry later" in rendered


def test_auth_error_still_uses_its_own_formatter() -> None:
    exc = AuthError("401 unauthenticated")

    rendered = format_error_for_user(exc, output_format="text")

    assert "Nebius authentication failed" in rendered
    assert "npa configure" in rendered


def test_endpoint_not_found_still_uses_its_own_formatter() -> None:
    exc = EndpointNotFoundError(
        "not found",
        project_id="project-1",
        endpoint_name="my-endpoint",
        endpoint_id="endpoint-1",
    )

    rendered = format_error_for_user(exc, output_format="text")

    assert "was not found" in rendered
    assert "my-endpoint" in rendered


def test_generic_serverless_error_message_is_redacted() -> None:
    exc = ServerlessClientError("failed for token=hf_abcdefghijklmnop")

    rendered = format_error_for_user(exc, output_format="text")
    payload = json.loads(format_error_for_user(exc, output_format="json"))

    assert "hf_abcdefghijklmnop" not in rendered
    assert "hf_abcdefghijklmnop" not in payload["message"]


def test_generic_serverless_error_message_is_bounded() -> None:
    exc = ServerlessClientError("x" * 5000)

    rendered = format_error_for_user(exc, output_format="text")

    assert len(rendered) < 5000
    assert "truncated" in rendered


def test_generic_serverless_error_falls_back_to_message_only() -> None:
    exc = ServerlessClientError("some other serverless failure")

    rendered = format_error_for_user(exc, output_format="text")

    assert "some other serverless failure" in rendered


def test_job_identity_error_falls_back_to_generic_serverless_formatter() -> None:
    """JobIdentityError has no extra identity fields; the base message must survive."""
    exc = JobIdentityError("create_job returned an unexpected job identity or project")

    rendered = format_error_for_user(exc, output_format="text")
    payload = json.loads(format_error_for_user(exc, output_format="json"))

    assert "unexpected job identity" in rendered
    assert (
        payload["message"]
        == "create_job returned an unexpected job identity or project"
    )


def test_transient_serverless_error_falls_back_to_generic_serverless_formatter() -> (
    None
):
    exc = TransientServerlessError("503 unavailable")

    rendered = format_error_for_user(exc, output_format="text")

    assert "503 unavailable" in rendered


def test_unrelated_exception_uses_generic_unexpected_formatter() -> None:
    rendered = format_error_for_user(ValueError("boom"), output_format="text")
    payload = json.loads(
        format_error_for_user(ValueError("boom"), output_format="json")
    )

    assert "Unexpected error" in rendered
    assert payload["error"] == "UnexpectedError"
    assert payload["message"] == "boom"


def test_output_format_is_case_insensitive_and_defaults_to_text() -> None:
    exc = AuthError("401")

    assert "{" not in format_error_for_user(exc, output_format="TEXT")
    assert (
        json.loads(format_error_for_user(exc, output_format="JSON"))["error"] == "Auth"
    )
    assert "{" not in format_error_for_user(exc, output_format="not-a-real-format")
