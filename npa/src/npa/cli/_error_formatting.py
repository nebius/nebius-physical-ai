"""Customer-facing error formatting for the npa CLI."""

from __future__ import annotations

import json

from npa.clients.serverless import (
    AuthError,
    EndpointNotFoundError,
    JobSubmissionIndeterminateError,
    NotEnoughResourcesError,
    QuotaError,
    ServerlessClientError,
)
from npa.orchestration.skypilot.workflow_state import redact_text

_MESSAGE_LIMIT = 2000


def _safe_message(text: str) -> str:
    """Redact secret-shaped substrings and bound length before display.

    Exception text can echo provider stderr or a subprocess command line, so
    it gets the same treatment as any other operator-facing diagnostic text
    rather than being trusted as already-safe.
    """
    redacted = redact_text(str(text))
    if len(redacted) <= _MESSAGE_LIMIT:
        return redacted
    return redacted[:_MESSAGE_LIMIT] + f"... [truncated, {len(redacted)} chars total]"


def format_error_for_user(exc: Exception, *, output_format: str = "text") -> str:
    """Format an exception for human users or JSON-consuming agents."""
    fmt = "json" if str(output_format).lower() == "json" else "text"
    if isinstance(exc, NotEnoughResourcesError):
        return _format_ner(exc, fmt)
    if isinstance(exc, AuthError):
        return _format_auth(exc, fmt)
    if isinstance(exc, EndpointNotFoundError):
        return _format_not_found(exc, fmt)
    if isinstance(exc, JobSubmissionIndeterminateError):
        return _format_indeterminate(exc, fmt)
    if isinstance(exc, ServerlessClientError):
        return _format_generic_serverless(exc, fmt)
    return _format_generic(exc, fmt)


def _json(data: dict[str, object]) -> str:
    return json.dumps(data, indent=2)


def _format_ner(exc: NotEnoughResourcesError, output_format: str) -> str:
    name = "Quota" if isinstance(exc, QuotaError) else "NotEnoughResources"
    message = _safe_message(exc.message)
    data = {
        "error": name,
        "error_class": exc.error_class,
        "message": message,
        "project_id": exc.project_id,
        "platform": exc.platform,
        "preset": exc.preset,
        "gpu_count": exc.gpu_count,
        "suggested_alternatives": exc.suggested_alternatives,
    }
    if output_format == "json":
        return _json(data)

    heading = (
        "Quota limit reached."
        if isinstance(exc, QuotaError)
        else "Not enough resources to schedule this request."
    )
    lines = [f"Error: {heading}", ""]
    for label, value in (
        ("Project", exc.project_id),
        ("Platform", exc.platform),
        ("Preset", exc.preset),
        ("GPU count", str(exc.gpu_count) if exc.gpu_count else ""),
    ):
        if value:
            lines.append(f"  {label}: {value}")
    if any((exc.project_id, exc.platform, exc.preset, exc.gpu_count)):
        lines.append("")
    lines.append(f"  Cause: {exc.error_class} ({message})")
    if exc.suggested_alternatives:
        lines.extend(["", "  Try one of:"])
        lines.extend(f"    - {item}" for item in exc.suggested_alternatives)
    lines.extend(["", "  See: docs/cli-errors.md"])
    return "\n".join(lines)


def _format_auth(exc: AuthError, output_format: str) -> str:
    message = _safe_message(exc.message)
    if output_format == "json":
        return _json({"error": "Auth", "message": message, "hint": exc.hint})
    return "\n".join(
        [
            "Error: Nebius authentication failed.",
            "",
            f"  Cause: {message}",
            f"  Hint: {exc.hint}",
        ]
    )


def _format_not_found(exc: EndpointNotFoundError, output_format: str) -> str:
    message = _safe_message(exc.message)
    data = {
        "error": "EndpointNotFound",
        "message": message,
        "project_id": exc.project_id,
        "endpoint_name": exc.endpoint_name,
        "endpoint_id": exc.endpoint_id,
    }
    if output_format == "json":
        return _json(data)
    lines = ["Error: Serverless resource was not found.", "", f"  Cause: {message}"]
    if exc.project_id:
        lines.append(f"  Project: {exc.project_id}")
    if exc.endpoint_name:
        lines.append(f"  Name: {exc.endpoint_name}")
    if exc.endpoint_id:
        lines.append(f"  ID: {exc.endpoint_id}")
    return "\n".join(lines)


def _format_indeterminate(
    exc: JobSubmissionIndeterminateError, output_format: str
) -> str:
    message = _safe_message(exc.message)
    data = {
        "error": "JobSubmissionIndeterminate",
        "message": message,
        "project_id": exc.project_id,
        "job_name": exc.job_name,
        "provider_job_id": exc.provider_job_id,
        "durable": exc.durable,
    }
    if output_format == "json":
        return _json(data)
    lines = [
        "Error: Serverless Job submission outcome is unknown.",
        "",
        f"  Cause: {message}",
    ]
    for label, value in (
        ("Project", exc.project_id),
        ("Job name", exc.job_name),
        ("Provider job ID", exc.provider_job_id),
    ):
        if value:
            lines.append(f"  {label}: {value}")
    lines.append("")
    if exc.durable:
        lines.append(
            "  Do not submit a new job under a different name. Re-run the same "
            "command with the same job name to reconnect; a durable submission "
            "record will adopt the existing job instead of creating a new one."
        )
    else:
        lines.append(
            "  This call path has no durable submission record, so a retry "
            "with the same job name will still attempt to create a new job. "
            "Check whether a job with this name already exists before "
            "resubmitting."
        )
    return "\n".join(lines)


def _format_generic_serverless(exc: ServerlessClientError, output_format: str) -> str:
    message = _safe_message(exc.message)
    if output_format == "json":
        return _json({"error": "ServerlessClientError", "message": message})
    return f"Error: Nebius serverless request failed.\n\n  Cause: {message}"


def _format_generic(exc: Exception, output_format: str) -> str:
    message = _safe_message(str(exc))
    if output_format == "json":
        return _json({"error": "UnexpectedError", "message": message})
    return f"Error: Unexpected error: {message}"
