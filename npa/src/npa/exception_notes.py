"""Retain secondary failure diagnostics without replacing the original exception."""


def add_exception_note(error: BaseException, note: str) -> None:
    """Append a diagnostic using the same notes contract on supported Python versions.

    Args:
        error: Original failure that must remain authoritative.
        note: Secondary diagnostic to retain with that failure.
    Returns:
        None.
    Raises:
        None.
    """
    error.add_note(note)
