"""npa exception hierarchy."""

from __future__ import annotations


class NpaError(Exception):
    """Base for all npa-raised exceptions. Subclass this for new error types."""


class ScopedCredentialError(NpaError):
    """Raised when scoped credentials fail and --allow-host-creds was not set."""

    def __init__(
        self,
        bucket: str,
        operation: str,
        remediation: str | None = None,
        *,
        source_project: str | None = None,
        target_project: str | None = None,
        failed_project: str | None = None,
    ):
        self.bucket = bucket
        self.operation = operation
        self.source_project = source_project
        self.target_project = target_project
        self.failed_project = failed_project
        self.remediation = remediation or (
            "Pass --allow-host-creds to fall back to host credentials, "
            "or grant the scoped principal access to the bucket."
        )
        project_context = ""
        if failed_project:
            project_context = f" in project '{failed_project}'"
        if source_project or target_project:
            boundaries = []
            if source_project:
                boundaries.append(f"source_project='{source_project}'")
            if target_project:
                boundaries.append(f"target_project='{target_project}'")
            project_context += f" ({', '.join(boundaries)})"
        super().__init__(
            f"Scoped credentials failed for {operation} on bucket '{bucket}'"
            f"{project_context}. {self.remediation}"
        )


__all__ = ["NpaError", "ScopedCredentialError"]
