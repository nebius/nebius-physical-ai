"""Unit tests for `npa.cli._error_formatting`."""

from __future__ import annotations

import json

from npa.cli._error_formatting import format_error_for_user
from npa.clients.serverless import (
    AuthError,
    EndpointNotFoundError,
    NotEnoughResourcesError,
    QuotaError,
    ServerlessClientError,
)


# ── NotEnoughResources / Quota ────────────────────────────────────────────


def test_format_ner_text_includes_resource_details() -> None:
    exc = NotEnoughResourcesError(
        message="no gpu",
        project_id="proj-1",
        platform="gpu-h200-sxm",
        preset="1gpu-16vcpu-200gb",
        gpu_count=1,
        suggested_alternatives=["8gpu-h200-sxm", "1gpu-h100"],
    )
    out = format_error_for_user(exc)
    assert "Not enough resources" in out
    assert "Project: proj-1" in out
    assert "Platform: gpu-h200-sxm" in out
    assert "GPU count: 1" in out
    assert "8gpu-h200-sxm" in out
    assert "docs/cli-errors.md" in out


def test_format_ner_json() -> None:
    exc = NotEnoughResourcesError(message="full", project_id="p")
    data = json.loads(format_error_for_user(exc, output_format="JSON"))
    assert data["error"] == "NotEnoughResources"
    assert data["error_class"] == "capacity"
    assert data["project_id"] == "p"


def test_format_quota_uses_quota_heading() -> None:
    exc = QuotaError(message="limit", project_id="p", platform="x")
    text = format_error_for_user(exc)
    assert "Quota limit reached" in text
    data = json.loads(format_error_for_user(exc, output_format="json"))
    assert data["error"] == "Quota"
    assert data["error_class"] == "quota"


def test_format_ner_no_optional_fields_omits_blank_line() -> None:
    exc = NotEnoughResourcesError(message="boom")
    out = format_error_for_user(exc)
    assert "Cause: capacity (boom)" in out
    # No platform/project/preset lines
    assert "Project:" not in out


# ── Auth ──────────────────────────────────────────────────────────────────


def test_format_auth_text_includes_hint() -> None:
    exc = AuthError(message="401", hint="refresh token")
    out = format_error_for_user(exc)
    assert "authentication failed" in out
    assert "Hint: refresh token" in out


def test_format_auth_json() -> None:
    exc = AuthError(message="403")
    data = json.loads(format_error_for_user(exc, output_format="json"))
    assert data["error"] == "Auth"
    assert data["message"] == "403"


# ── EndpointNotFound ──────────────────────────────────────────────────────


def test_format_endpoint_not_found_text_with_all_fields() -> None:
    exc = EndpointNotFoundError(
        message="missing",
        project_id="proj",
        endpoint_name="sonic-wb",
        endpoint_id="endpoint-abc",
    )
    out = format_error_for_user(exc)
    assert "Serverless resource was not found" in out
    assert "Project: proj" in out
    assert "Name: sonic-wb" in out
    assert "ID: endpoint-abc" in out


def test_format_endpoint_not_found_json() -> None:
    exc = EndpointNotFoundError(message="missing", endpoint_id="x")
    data = json.loads(format_error_for_user(exc, output_format="json"))
    assert data["error"] == "EndpointNotFound"
    assert data["endpoint_id"] == "x"


# ── Generic serverless / generic ─────────────────────────────────────────


def test_format_generic_serverless_text() -> None:
    exc = ServerlessClientError(message="boom")
    assert "Nebius serverless request failed" in format_error_for_user(exc)


def test_format_generic_serverless_json() -> None:
    exc = ServerlessClientError(message="x")
    data = json.loads(format_error_for_user(exc, output_format="json"))
    assert data == {"error": "ServerlessClientError", "message": "x"}


def test_format_generic_unexpected_exception_text() -> None:
    out = format_error_for_user(ValueError("nope"))
    assert "Unexpected error" in out
    assert "nope" in out


def test_format_generic_unexpected_exception_json() -> None:
    data = json.loads(format_error_for_user(RuntimeError("oops"), output_format="json"))
    assert data == {"error": "UnexpectedError", "message": "oops"}
