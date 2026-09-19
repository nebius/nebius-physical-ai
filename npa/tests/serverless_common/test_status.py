from __future__ import annotations

import pytest

from npa.clients.serverless import AuthError, JobInfo, TransientServerlessError
from npa.serverless_common import job_status_payload


def _job(status: str, **overrides) -> JobInfo:
    defaults = dict(id="job-1", name="train-1", project_id="project-1", status=status)
    return JobInfo(**{**defaults, **overrides})


class _StubClient:
    """Minimal double: only the two methods `job_status_payload` calls."""

    def __init__(self, *, queue_state: str = "", logs: str | Exception = ""):
        self._queue_state = queue_state
        self._logs = logs
        self.log_calls: list[tuple[str, str]] = []

    def classify_queue_state(self, info: JobInfo) -> str:
        return self._queue_state or info.status

    def get_job_logs(self, job_id: str, project_id: str, *, tail: int | None = None) -> str:
        self.log_calls.append((job_id, project_id))
        if isinstance(self._logs, Exception):
            raise self._logs
        return self._logs


def test_queued_job_keeps_existing_capacity_hint_shape() -> None:
    client = _StubClient(queue_state="waiting_for_capacity")
    payload = job_status_payload(client, _job("queued", queued_for_seconds=492), gpu_count=8)

    assert payload["status"] == "waiting_for_capacity"
    assert payload["queue_state_classification"] == "capacity"
    assert payload["queued_for_seconds"] == 492
    assert payload["gpu_count"] == 8
    assert not client.log_calls


def test_succeeded_job_never_fetches_logs() -> None:
    client = _StubClient(logs=AssertionError("must not be called for a healthy job"))
    payload = job_status_payload(client, _job("succeeded"))

    assert payload["status"] == "succeeded"
    assert "log_tail" not in payload
    assert not client.log_calls


def test_failed_job_reports_pending_reason_and_fresh_log_tail() -> None:
    client = _StubClient(logs="Traceback: OOMKilled\nexit 137")
    info = _job(
        "failed",
        pending_reason="OOM_KILLED",
        scheduling_state="RUNNING",
        log_tail="cached provider message",
    )

    payload = job_status_payload(client, info)

    assert payload["status"] == "failed"
    assert payload["pending_reason"] == "OOM_KILLED"
    assert payload["scheduling_state"] == "RUNNING"
    assert payload["log_tail"] == "Traceback: OOMKilled\nexit 137"
    assert payload["log_tail_source"] == "job_logs"
    assert "log_fetch_error" not in payload
    assert payload["hint"].startswith("Job failed.")
    assert client.log_calls == [("job-1", "project-1")]


def test_cancelled_job_gets_cancelled_hint_not_failed_hint() -> None:
    client = _StubClient(logs="cancelled cleanly")
    payload = job_status_payload(client, _job("cancelled"))

    assert payload["hint"] == "Job was cancelled."


def test_failed_log_fetch_falls_back_to_cached_message_without_hiding_the_error() -> None:
    client = _StubClient(logs=AuthError("403 forbidden"))
    info = _job("failed", log_tail="cached provider message")

    payload = job_status_payload(client, info)

    # The fetch failure must not be silently swallowed...
    assert payload["log_fetch_error"] == {"error_type": "AuthError", "message": "403 forbidden"}
    # ...nor mistaken for "the job produced no logs" (cached message is kept)...
    assert payload["log_tail"] == "cached provider message"
    assert payload["log_tail_source"] == "cached_provider_message"
    # ...nor allowed to overwrite the true, already-observed job status.
    assert payload["status"] == "failed"
    assert payload["raw_status"] == "failed"


def test_transient_log_fetch_error_is_distinguishable_from_auth_error() -> None:
    client = _StubClient(logs=TransientServerlessError("503 unavailable"))
    payload = job_status_payload(client, _job("failed"))

    assert payload["log_fetch_error"]["error_type"] == "TransientServerlessError"


def test_empty_log_fetch_falls_back_to_cached_message() -> None:
    client = _StubClient(logs="   \n  ")
    info = _job("failed", log_tail="cached provider message")

    payload = job_status_payload(client, info)

    assert payload["log_tail"] == "cached provider message"
    assert payload["log_tail_source"] == "cached_provider_message"
    assert "log_fetch_error" not in payload


def test_log_tail_is_redacted_before_display() -> None:
    client = _StubClient(logs="starting job\nAWS_SECRET_ACCESS_KEY=super-secret-value\ndone")
    payload = job_status_payload(client, _job("failed"))

    assert "super-secret-value" not in payload["log_tail"]
    assert "<redacted>" in payload["log_tail"]


def test_cached_fallback_log_tail_is_redacted_too() -> None:
    client = _StubClient(logs=AuthError("403 forbidden"))
    info = _job("failed", log_tail="HF_TOKEN=hf_abcdefghijklmnop still cached")

    payload = job_status_payload(client, info)

    assert "hf_abcdefghijklmnop" not in payload["log_tail"]
    assert "<redacted>" in payload["log_tail"]


def test_pending_reason_is_redacted() -> None:
    client = _StubClient(logs="")
    info = _job("failed", pending_reason="leaked NGC_API_KEY=nvapi-abcdefghijklmnop in reason")

    payload = job_status_payload(client, info)

    assert "nvapi-abcdefghijklmnop" not in payload["pending_reason"]


def test_huge_single_line_log_tail_is_bounded_by_characters_not_just_lines() -> None:
    client = _StubClient(logs="x" * 10_000)
    payload = job_status_payload(client, _job("failed"))

    assert len(payload["log_tail"]) < 10_000
    assert "truncated" in payload["log_tail"]


def test_huge_cached_fallback_log_tail_is_bounded() -> None:
    client = _StubClient(logs=AuthError("403 forbidden"))
    info = _job("failed", log_tail="y" * 10_000)

    payload = job_status_payload(client, info)

    assert len(payload["log_tail"]) < 10_000
    assert "truncated" in payload["log_tail"]


def test_huge_pending_reason_is_bounded() -> None:
    client = _StubClient(logs="")
    info = _job("failed", pending_reason="z" * 5_000)

    payload = job_status_payload(client, info)

    assert len(payload["pending_reason"]) < 5_000


def test_log_fetch_error_message_is_redacted() -> None:
    client = _StubClient(logs=AuthError("failed for token=hf_abcdefghijklmnop"))
    payload = job_status_payload(client, _job("failed"))

    assert "hf_abcdefghijklmnop" not in payload["log_fetch_error"]["message"]


@pytest.mark.parametrize("status", ["running", "unknown"])
def test_non_terminal_non_queued_status_gets_bare_payload(status: str) -> None:
    client = _StubClient(logs=AssertionError("must not be called"))
    payload = job_status_payload(client, _job(status))

    assert payload["status"] == status
    assert "log_tail" not in payload
    assert "hint" not in payload
    assert not client.log_calls
