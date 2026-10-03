"""Domain errors shared by Arena evaluation and evidence validation."""


class IsaacArenaError(RuntimeError):
    """Reject an invalid Arena request, runtime result, or evidence artifact.

    Args:
        message: The failed contract and its diagnostic context.

    Returns:
        None.

    Raises:
        None.
    """
