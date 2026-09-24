"""Define the official BEHAVIOR evaluator revisions supported by NPA."""

from __future__ import annotations

UPSTREAM_COMMITS = {
    "3.9.2": "b1979916ec1549b10a4e65e630bc6504a9af1b00",
    "3.9.3": "6cbf70b075816096e9be53958780769f3264d25d",
}
UPSTREAM_COMMIT = UPSTREAM_COMMITS["3.9.3"]


def require_supported_upstream(commit: object) -> str:
    """Return one explicitly supported official evaluator revision.

    Args:
        commit: Full Git commit declared by a recipe or immutable panel.
    Returns:
        The commit as a validated string.
    Raises:
        ValueError: The value is not one of the supported official revisions.
    """
    if not isinstance(commit, str) or commit not in UPSTREAM_COMMITS.values():
        raise ValueError("Use a supported official BEHAVIOR evaluator commit")
    return commit


def evaluator_version(commit: object) -> str:
    """Return the official release name for a supported evaluator commit.

    Args:
        commit: Full supported evaluator Git commit.
    Returns:
        Official version string such as ``3.9.3``.
    Raises:
        ValueError: The commit is unsupported.
    """
    revision = require_supported_upstream(commit)
    return next(
        version for version, value in UPSTREAM_COMMITS.items() if value == revision
    )
