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
    add_note = getattr(error, "add_note", None)
    if callable(add_note):
        add_note(note)
        return
    # Python 3.10 has no add_note; retain notes for programmatic consumers.
    notes = list(getattr(error, "__notes__", ()))
    notes.append(note)
    error.__notes__ = notes
