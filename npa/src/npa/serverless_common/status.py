"""Shared operator-facing status payload for one Nebius Serverless Job.

A CLI `status` command is often the operator's first move after a job stops
progressing, so the payload it prints must carry the provider's actual reason,
not just a normalized status word. `JobInfo` already parses that reason
(`pending_reason`, `scheduling_state`, `log_tail`) out of the provider
response at no extra API cost; this module is the one place that turns it
into the dict every workbench tool's `status` command renders.
"""

from __future__ import annotations

from typing import Any

from npa.clients.serverless import JobInfo, ServerlessClient, ServerlessClientError
from npa.orchestration.skypilot.workflow_state import redact_text

_FAILURE_LOG_TAIL_LINES = 40
_LOG_TAIL_CHAR_LIMIT = 4000
_REASON_CHAR_LIMIT = 1000
_TERMINAL_FAILURE_STATUSES = frozenset({"failed", "cancelled"})


def _bounded(text: str, limit: int) -> str:
    """Cap displayed text length; a line-count bound alone does not cap bytes."""
    if len(text) <= limit:
        return text
    return text[:limit] + f"... [truncated, {len(text)} chars total]"


def job_status_payload(
    client: ServerlessClient,
    info: JobInfo,
    *,
    platform: str = "",
    gpu_count: int = 0,
) -> dict[str, Any]:
    """Build the status dict a workbench tool's `status` command renders.

    Args:
        client: Client used to classify the queue state and, for a
            failed or cancelled job, fetch a fresh log tail. Never called
            again for a queued, running, or succeeded job.
        info: The job to describe.
        platform: GPU platform to report for a queued job, if the caller
            knows the originally requested platform (falls back to
            ``info.platform``).
        gpu_count: GPU count to report for a queued job, if the caller knows
            the originally requested count (falls back to ``info.gpu_count``).

    Returns:
        A JSON-serializable dict. `status`/`raw_status` always reflect
        ``info`` as observed; a failed attempt to fetch a fresh log tail
        never changes them. Queued jobs get a queue classification and a
        capacity hint. Failed and cancelled jobs get the provider's pending
        reason, scheduling state, and a log tail, so the operator does not
        need a separate command to see why the job stopped. If that log
        fetch itself fails, the cached ``info.log_tail`` is used instead and
        the failure is reported in `log_fetch_error`, not hidden.

    Raises:
        None.
    """
    status = client.classify_queue_state(info)
    payload: dict[str, Any] = {
        "job_id": info.id,
        "job_name": info.name,
        "status": status,
        "raw_status": info.status,
        "output_uris": list(info.output_uris),
    }
    if info.status == "queued":
        payload["queue_state_classification"] = (
            "capacity" if status == "waiting_for_capacity" else "scheduled"
        )
        payload["queued_for_seconds"] = info.queued_for_seconds
        payload["platform"] = platform or info.platform
        payload["gpu_count"] = gpu_count or info.gpu_count
        payload["hint"] = (
            "Platform may be at capacity. Retry status in a few minutes."
            if status == "waiting_for_capacity"
            else "Job is scheduled and waiting to start."
        )
    elif info.status in _TERMINAL_FAILURE_STATUSES:
        _add_failure_diagnostics(payload, client, info)
    return payload


def _add_failure_diagnostics(
    payload: dict[str, Any], client: ServerlessClient, info: JobInfo
) -> None:
    """Attach pending reason, scheduling state, and a redacted log tail.

    A failed or empty log-tail fetch is recorded in `log_fetch_error` with
    its exact exception type rather than silently substituted, so a
    transient lookup failure is never mistaken for "the job had no logs."
    Every text field rendered here — the cached fallback log tail, the
    pending reason, and the fetch-error message — is redacted the same way
    as a freshly fetched log tail, since all three can originate from
    provider-echoed or job-printed text, and bounded in length: a line-count
    limit on the fetch alone does not cap a single very long line.
    """
    cached_log_tail = _bounded(redact_text(info.log_tail).strip(), _LOG_TAIL_CHAR_LIMIT)
    if info.pending_reason:
        payload["pending_reason"] = _bounded(
            redact_text(info.pending_reason), _REASON_CHAR_LIMIT
        )
    if info.scheduling_state:
        payload["scheduling_state"] = info.scheduling_state
    try:
        fetched = client.get_job_logs(
            info.id, info.project_id, tail=_FAILURE_LOG_TAIL_LINES
        )
    except ServerlessClientError as exc:
        payload["log_tail"] = cached_log_tail
        payload["log_tail_source"] = "cached_provider_message"
        payload["log_fetch_error"] = {
            "error_type": type(exc).__name__,
            "message": _bounded(redact_text(str(exc)), _REASON_CHAR_LIMIT),
        }
    else:
        redacted = _bounded(redact_text(fetched).strip(), _LOG_TAIL_CHAR_LIMIT)
        payload["log_tail"] = redacted or cached_log_tail
        payload["log_tail_source"] = (
            "job_logs" if redacted else "cached_provider_message"
        )
    payload["hint"] = (
        "Job was cancelled."
        if info.status == "cancelled"
        else "Job failed. See log_tail for the provider's reported reason."
    )


__all__ = ["job_status_payload"]
