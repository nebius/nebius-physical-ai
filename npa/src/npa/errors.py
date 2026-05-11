"""npa exception hierarchy."""

from __future__ import annotations


class NpaError(Exception):
    """Base for all npa-raised exceptions. Subclass this for new error types."""


class ScopedCredentialError(NpaError):
    """Raised when scoped credentials fail and --allow-host-creds was not set."""

    def __init__(self, bucket: str, operation: str, remediation: str | None = None):
        self.bucket = bucket
        self.operation = operation
        self.remediation = remediation or (
            "Pass --allow-host-creds to fall back to host credentials, "
            "or grant the scoped principal access to the bucket."
        )
        super().__init__(
            f"Scoped credentials failed for {operation} on bucket '{bucket}'. "
            f"{self.remediation}"
        )
