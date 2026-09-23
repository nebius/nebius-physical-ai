"""Report SQLite failure classes and phases without leaking provider or file diagnostics."""

from __future__ import annotations

import re


class StorageFailure(RuntimeError):
    """A durable journal or checkpoint operation failed; effects must not be retried.

    Args: phase: Controller-owned operation label. error: Original SQLite exception.
    Returns: Exception with safe structured diagnostics, without raw error text.
    Raises: None.
    """

    def __init__(self, phase, error):
        code = getattr(error, "sqlite_errorcode", None)
        name = getattr(error, "sqlite_errorname", None)
        self.diagnostic = {
            "phase": phase,
            "sqlite_errorcode": code if type(code) is int else None,
            "sqlite_errorname": name
            if isinstance(name, str) and re.fullmatch(r"SQLITE_[A-Z0-9_]+", name)
            else None,
        }
        kind = self.diagnostic["sqlite_errorname"] or "SQLITE_ERROR_UNKNOWN"
        super().__init__(
            f"Durable storage failed during {phase} ({kind}); inspect storage before recovery"
        )
