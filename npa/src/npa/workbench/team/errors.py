"""Define public, credential-free failures at the team service boundary."""

from npa.errors import NpaError


class TeamError(NpaError):
    """Represent an invalid team operation without exposing backend credentials.

    Args:
        *args: Credential-free error details.
    Returns:
        A TeamError instance.
    Raises:
        None.
    """

    status_code = 400


class AuthenticationError(TeamError):
    """Reject missing or unverifiable external identities.

    Args:
        *args: Credential-free error details.
    Returns:
        A AuthenticationError instance.
    Raises:
        None.
    """

    status_code = 401


class AuthorizationError(TeamError):
    """Reject operations outside the authenticated actor's grants.

    Args:
        *args: Credential-free error details.
    Returns:
        A AuthorizationError instance.
    Raises:
        None.
    """

    status_code = 403


class RunNotFoundError(TeamError):
    """Hide run existence from callers without access.

    Args:
        *args: Credential-free error details.
    Returns:
        A RunNotFoundError instance.
    Raises:
        None.
    """

    status_code = 404


class ConflictError(TeamError):
    """Reject conflicting run identities or concurrent lifecycle operations.

    Args:
        *args: Credential-free error details.
    Returns:
        A ConflictError instance.
    Raises:
        None.
    """

    status_code = 409


class BackendError(TeamError):
    """Report an unavailable backend without returning its private output.

    Args:
        *args: Credential-free error details.
    Returns:
        A BackendError instance.
    Raises:
        None.
    """

    status_code = 503
