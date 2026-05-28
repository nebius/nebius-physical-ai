"""Coverage tests for `npa.clients.serverless` pure helpers."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from npa.clients import serverless as sv
from npa.clients.serverless import EndpointStatus


# ── EndpointStatus.from_value ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("running", EndpointStatus.RUNNING),
        ("Active", EndpointStatus.RUNNING),
        ("ready", EndpointStatus.RUNNING),
        ("creating", EndpointStatus.CREATING),
        ("PROVISIONING", EndpointStatus.CREATING),
        ("starting", EndpointStatus.CREATING),
        ("pending", EndpointStatus.CREATING),
        ("stopped", EndpointStatus.STOPPED),
        ("stopping", EndpointStatus.STOPPED),
        ("inactive", EndpointStatus.STOPPED),
        ("failed", EndpointStatus.FAILED),
        ("error", EndpointStatus.FAILED),
        ("crashed", EndpointStatus.FAILED),
        ("deleting", EndpointStatus.DELETING),
        ("terminating", EndpointStatus.DELETING),
        ("deleted", EndpointStatus.DELETED),
        ("", EndpointStatus.UNKNOWN),
        (None, EndpointStatus.UNKNOWN),
        ("mystery", EndpointStatus.UNKNOWN),
    ],
)
def test_endpoint_status_from_value(raw, expected) -> None:
    assert EndpointStatus.from_value(raw) is expected


# ── _job_status / _int_value ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("queued", "queued"),
        ("PENDING", "queued"),
        ("provisioning", "queued"),
        ("running", "running"),
        ("active", "running"),
        ("succeeded", "succeeded"),
        ("complete", "succeeded"),
        ("done", "succeeded"),
        ("failed", "failed"),
        ("error", "failed"),
        ("cancelling", "cancelling"),
        ("canceling", "cancelling"),
        ("stopping", "cancelling"),
        ("cancelled", "cancelled"),
        ("canceled", "cancelled"),
        ("stopped", "cancelled"),
        ("", "unknown"),
        (None, "unknown"),
        ("weird", "unknown"),
    ],
)
def test_job_status(raw, expected) -> None:
    assert sv._job_status(raw) == expected


def test_int_value_falls_back_on_bad_input() -> None:
    assert sv._int_value("3") == 3
    assert sv._int_value(7) == 7
    assert sv._int_value(None) == 0
    assert sv._int_value("not-a-number") == 0


# ── _queued_for_seconds ───────────────────────────────────────────────────


def test_queued_for_seconds_zulu() -> None:
    now = datetime(2026, 5, 14, 12, 0, 0, tzinfo=timezone.utc)
    assert sv._queued_for_seconds("2026-05-14T11:55:00Z", now=now) == 300


def test_queued_for_seconds_iso_with_offset() -> None:
    now = datetime(2026, 5, 14, 12, 0, 0, tzinfo=timezone.utc)
    assert sv._queued_for_seconds("2026-05-14T11:55:00+00:00", now=now) == 300


def test_queued_for_seconds_naive_assumed_utc() -> None:
    now = datetime(2026, 5, 14, 12, 0, 0, tzinfo=timezone.utc)
    assert sv._queued_for_seconds("2026-05-14T11:55:00", now=now) == 300


def test_queued_for_seconds_empty_and_bad_inputs() -> None:
    assert sv._queued_for_seconds("") == 0
    assert sv._queued_for_seconds("not-a-date") == 0


def test_queued_for_seconds_future_is_clamped_to_zero() -> None:
    now = datetime(2026, 5, 14, 11, 0, 0, tzinfo=timezone.utc)
    assert sv._queued_for_seconds("2026-05-14T11:55:00Z", now=now) == 0


# ── _map_scheduling_state ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "value,expected",
    [
        ("", ""),
        ("WAITING-FOR-CAPACITY", "waiting_for_capacity"),
        ("no_gpu_available", "waiting_for_capacity"),
        ("quota exceeded", "waiting_for_capacity"),
        ("resource limit", "waiting_for_capacity"),
        ("scheduled", "scheduled"),
        ("Accepted", "scheduled"),
        ("queued by scheduler", "scheduled"),
        ("pending review", "scheduled"),
        ("running", "running"),
        ("unknown state", ""),
    ],
)
def test_map_scheduling_state(value, expected) -> None:
    assert sv._map_scheduling_state(value) == expected


# ── _is_secret_env_key / _is_sensitive_log_key ───────────────────────────


def test_is_secret_env_key() -> None:
    assert sv._is_secret_env_key("AWS_SECRET_ACCESS_KEY")
    assert sv._is_secret_env_key("API_TOKEN")
    assert sv._is_secret_env_key("MY_PASSWORD")
    assert sv._is_secret_env_key("HF_API_KEY")
    assert sv._is_secret_env_key("PRIVATE_KEY")
    assert not sv._is_secret_env_key("AWS_REGION")
    assert not sv._is_secret_env_key("NPA_OUTPUT_PATH")


def test_is_sensitive_log_key() -> None:
    assert sv._is_sensitive_log_key("HF_TOKEN")
    assert sv._is_sensitive_log_key("API_KEY")
    assert sv._is_sensitive_log_key("SUPER_SECRET")
    assert sv._is_sensitive_log_key("MY_PASSWORD")
    assert not sv._is_sensitive_log_key("NPA_OUTPUT_PATH")


# ── _redact_env_arg / _redact_cli_args ───────────────────────────────────


def test_redact_env_arg_redacts_sensitive() -> None:
    assert sv._redact_env_arg("TOKEN=abcdef") == "TOKEN=<redacted>"
    assert sv._redact_env_arg("API_KEY=xyz") == "API_KEY=<redacted>"
    assert sv._redact_env_arg("PASSWORD=hunter2") == "PASSWORD=<redacted>"


def test_redact_env_arg_leaves_non_sensitive_alone() -> None:
    assert sv._redact_env_arg("AWS_REGION=eu-north1") == "AWS_REGION=eu-north1"
    assert sv._redact_env_arg("no-equals-here") == "no-equals-here"


def test_redact_cli_args_redacts_password_and_token_flags() -> None:
    args = [
        "nebius",
        "ai",
        "endpoint",
        "create",
        "--registry-password",
        "super-secret",
        "--token",
        "tk-123",
        "--env",
        "HF_TOKEN=abcdef",
        "--env=PATH=/usr/local/bin",
        "--name",
        "ep",
    ]
    redacted = sv._redact_cli_args(args)
    assert "super-secret" not in redacted
    assert "tk-123" not in redacted
    assert "abcdef" not in redacted
    # Indices of redactions
    assert redacted[redacted.index("--registry-password") + 1] == "<redacted>"
    assert redacted[redacted.index("--token") + 1] == "<redacted>"
    assert "HF_TOKEN=<redacted>" in redacted
    # Non-sensitive env passes through (in --env= form)
    assert "--env=PATH=/usr/local/bin" in redacted
    # Non-sensitive flag passes through
    assert "--name" in redacted
    assert "ep" in redacted


# ── _endpoint_url ─────────────────────────────────────────────────────────


def test_endpoint_url_from_status_url() -> None:
    assert sv._endpoint_url({"status": {"url": "https://x"}}) == "https://x"


def test_endpoint_url_auto_prepends_scheme() -> None:
    assert sv._endpoint_url({"status": {"url": "host.example"}}) == "http://host.example"


def test_endpoint_url_from_public_endpoints_list() -> None:
    data = {"status": {"public_endpoints": ["https://e1"]}}
    assert sv._endpoint_url(data) == "https://e1"


def test_endpoint_url_from_endpoints_list_dict() -> None:
    data = {"status": {"endpoints": [{"url": "https://e2"}]}}
    assert sv._endpoint_url(data) == "https://e2"


def test_endpoint_url_from_endpoints_list_string() -> None:
    data = {"status": {"endpoints": ["https://e3"]}}
    assert sv._endpoint_url(data) == "https://e3"


def test_endpoint_url_missing_returns_empty() -> None:
    assert sv._endpoint_url({}) == ""
    assert sv._endpoint_url({"status": {}}) == ""
