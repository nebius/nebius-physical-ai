"""Accept local keys or explicitly linked external JWTs as the same stable person."""

from .accounts import KEY_PREFIX, Accounts
from .authentication import TokenVerifier
from .errors import AuthenticationError
from .nebius_authentication import NebiusProfileVerifier


class AccountAuthentication:
    """Resolve both supported credentials through the permanent account registry.

    Args:
        config: Local-account configuration with an optional external issuer.
        external: Injectable external token verifier for acceptance tests.
    Returns:
        Shared API authentication boundary.
    Raises:
        TeamError: Account storage or issuer settings are invalid.
    """

    def __init__(self, config, *, external=None):
        """Prepare local accounts and the optional external JWT verifier.

        Args:
            config, external: Installation and optional verified-token adapter.
        Returns:
            None.
        Raises:
            TeamError: Local state cannot be initialized.
        """
        self.accounts = Accounts(config)
        self.external = external
        if self.external is None and config.identity is not None:
            self.external = TokenVerifier(config.identity)
        if self.external is None and config.nebius_identity is not None:
            self.external = NebiusProfileVerifier(config.nebius_identity)

    def verify(self, authorization):
        """Authenticate a key or explicitly linked signed external bearer token.

        Args:
            authorization: Untrusted Authorization header.
        Returns:
            Stable Workbench identity with current local groups.
        Raises:
            AuthenticationError: Credential is invalid, disabled or unlinked.
        """
        scheme, separator, token = authorization.partition(" ")
        if separator and scheme.lower() == "bearer" and token.startswith(KEY_PREFIX):
            return self.accounts.authenticate(token)[0]
        if self.external is None:
            raise AuthenticationError("a valid Workbench access key is required")
        return self.accounts.resolve(self.external.verify(authorization))


def current_actor(config, actor):
    """Recheck local account state while preserving legacy issuer-bound installations.

    Args:
        config, actor: Current policy and previously authenticated actor.
    Returns:
        Current local membership, or the original legacy identity.
    Raises:
        AuthenticationError: The local account was disabled or removed.
    """
    return Accounts(config).refresh(actor) if config.account_namespace else actor
