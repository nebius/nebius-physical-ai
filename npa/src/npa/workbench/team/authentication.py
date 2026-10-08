"""Verify external JWTs against one configured issuer without trusting proxy headers."""

from __future__ import annotations

import threading

from .errors import AuthenticationError
from .models import Actor, IdentityProvider


class TokenVerifier:
    """Verify issuer-bound JWT signatures and claims using the configured JWKS URL.

    Args:
        provider: Trusted issuer, audience, algorithms, and JWKS URL.
        jwks_client: Optional signing-key transport for verification tests.
    Returns:
        A TokenVerifier instance.
    Raises:
        ImportError: JWT verification dependencies are unavailable.
    """

    def __init__(self, provider: IdentityProvider, *, jwks_client=None):
        """Prepare verification without contacting the identity provider.

        Args:
            provider: Administrator-selected issuer, audience, and signing keys.
            jwks_client: Injectable signing-key client for offline tests.
        Returns:
            None.
        Raises:
            ImportError: The team authentication dependency is absent.
        """
        import jwt

        self.provider = provider
        self.keys = jwks_client or jwt.PyJWKClient(provider.jwks_url)
        self.lock = threading.Lock()

    def verify(self, authorization: str) -> Actor:
        """Authenticate a bearer token without accepting caller-supplied identity fields.

        Args:
            authorization: HTTP Authorization header.
        Returns:
            Verified external actor.
        Raises:
            AuthenticationError: Token, signature, issuer, audience, or claims are invalid.
        """
        import jwt

        scheme, separator, token = authorization.partition(" ")
        if not separator or scheme.lower() != "bearer" or not token.strip():
            raise AuthenticationError("a valid bearer token is required")
        try:
            with self.lock:
                key = self.keys.get_signing_key_from_jwt(token).key
            claims = jwt.decode(
                token,
                key,
                algorithms=list(self.provider.algorithms),
                audience=self.provider.audience,
                issuer=self.provider.issuer,
                options={"require": ["exp", "iat", "iss", "aud", "sub"]},
            )
            return self._actor(claims)
        except (jwt.PyJWTError, ValueError, TypeError) as exc:
            raise AuthenticationError("bearer token could not be verified") from exc

    def _actor(self, claims):
        groups = claims.get(self.provider.groups_claim, [])
        if not isinstance(groups, list) or any(not isinstance(g, str) for g in groups):
            raise ValueError("invalid group claim")
        if not isinstance(claims["sub"], str) or not claims["sub"]:
            raise ValueError("invalid subject claim")
        return Actor(
            issuer=self.provider.issuer, subject=claims["sub"], groups=frozenset(groups)
        )
