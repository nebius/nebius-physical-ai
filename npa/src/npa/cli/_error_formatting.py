"""Customer-facing error formatting for the npa CLI."""

from __future__ import annotations

import json

from npa.clients.serverless import (
    AuthError,
    EndpointNotFoundError,
    NotEnoughResourcesError,
    QuotaError,
    ServerlessClientError,
)


def format_error_for_user(exc: Exception, *, output_format: str = "text") -> str:
    """Format an exception for human users or JSON-consuming agents."""
    fmt = "json" if str(output_format).lower() == "json" else "text"
    if isinstance(exc, NotEnoughResourcesError):
        return _format_ner(exc, fmt)
    if isinstance(exc, AuthError):
        return _format_auth(exc, fmt)
    if isinstance(exc, EndpointNotFoundError):
        return _format_not_found(exc, fmt)
    if isinstance(exc, ServerlessClientError):
        return _format_generic_serverless(exc, fmt)
    return _format_generic(exc, fmt)


def _json(data: dict[str, object]) -> str:
    return json.dumps(data, indent=2)


def _format_ner(exc: NotEnoughResourcesError, output_format: str) -> str:
    name = "Quota" if isinstance(exc, QuotaError) else "NotEnoughResources"
    data = {
        "error": name,
        "error_class": exc.error_class,
        "message": exc.message,
        "project_id": exc.project_id,
        "platform": exc.platform,
        "preset": exc.preset,
        "gpu_count": exc.gpu_count,
        "suggested_alternatives": exc.suggested_alternatives,
    }
    if output_format == "json":
        return _json(data)

    heading = "Quota limit reached." if isinstance(exc, QuotaError) else "Not enough resources to schedule this request."
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
    lines.append(f"  Cause: {exc.error_class} ({exc.message})")
    if exc.suggested_alternatives:
        lines.extend(["", "  Try one of:"])
        lines.extend(f"    - {item}" for item in exc.suggested_alternatives)
    lines.extend(["", "  See: docs/cli-errors.md"])
    return "\n".join(lines)


def _format_auth(exc: AuthError, output_format: str) -> str:
    if output_format == "json":
        return _json({"error": "Auth", "message": exc.message, "hint": exc.hint})
    return "\n".join(["Error: Nebius authentication failed.", "", f"  Cause: {exc.message}", f"  Hint: {exc.hint}"])


def _format_not_found(exc: EndpointNotFoundError, output_format: str) -> str:
    data = {
        "error": "EndpointNotFound",
        "message": exc.message,
        "project_id": exc.project_id,
        "endpoint_name": exc.endpoint_name,
        "endpoint_id": exc.endpoint_id,
    }
    if output_format == "json":
        return _json(data)
    lines = ["Error: Serverless resource was not found.", "", f"  Cause: {exc.message}"]
    if exc.project_id:
        lines.append(f"  Project: {exc.project_id}")
    if exc.endpoint_name:
        lines.append(f"  Name: {exc.endpoint_name}")
    if exc.endpoint_id:
        lines.append(f"  ID: {exc.endpoint_id}")
    return "\n".join(lines)


def _format_generic_serverless(exc: ServerlessClientError, output_format: str) -> str:
    if output_format == "json":
        return _json({"error": "ServerlessClientError", "message": exc.message})
    return f"Error: Nebius serverless request failed.\n\n  Cause: {exc.message}"


def _format_generic(exc: Exception, output_format: str) -> str:
    if output_format == "json":
        return _json({"error": "UnexpectedError", "message": str(exc)})
    return f"Error: Unexpected error: {exc}"
